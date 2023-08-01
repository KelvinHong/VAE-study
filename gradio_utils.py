import os
import gradio as gr
import json
import datetime
import torch
import csv
import shutil 
import warnings
import pandas as pd
from numpy.random import default_rng
from sklearn.manifold import TSNE
from math import ceil
from utils import *

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

pref = None
if os.path.isfile(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        pref = json.load(f)


def toggle_lr_scheduler(flag):
    return gr.update(visible=flag, value="")

def validate_and_create_config(model_type: str, dataset: str, batch_size: int, 
            optimizer: str, model_name: str, 
            epochs: int, learning_rate: float, latent_dimension: int, 
            lr_scheduler_flag: bool, lr_scheduler_formula: str,
            additional_hp: str, resume_flag: bool):
    model_dir = os.path.join(OUTPUT_ROOT, model_name)
    model_dir = os.path.abspath(model_dir)
    model_dir = model_dir.replace(os.sep, '/')
    lr_scheduler_formula = lr_scheduler_formula.strip()
    

        
    if " " in model_name:
        raise gr.Error("Model name cannot contains whitespace. ")
    if os.path.isdir(model_dir) and not resume_flag:
        message = f"""Model with the same name [{model_name}] already exists. 
        Please choose a different name! If you want to resume from checkpoint, 
        enable the Resume checkbox. """
        raise gr.Error(message)
    if resume_flag:
        if not os.path.isdir(model_dir):
            raise gr.Error(f"""There is no model to resume from because the model directory 
                {model_dir} doesn't exists or is empty.""")

    if learning_rate == 0:
        raise gr.Error("Learning rate cannot be zero.") 
    os.makedirs(model_dir, exist_ok=True)
    # Store model definitions in the model, for future references
    target_encoder = os.path.join(model_dir, "encoder.py")
    target_decoder = os.path.join(model_dir, "decoder.py")
    if not os.path.isfile(target_encoder):
        shutil.copyfile("./encoder.py", target_encoder)
    if not os.path.isfile(target_decoder):
        shutil.copyfile("./decoder.py", target_decoder)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    model_code = MODEL_TYPE_MAPPING[model_type] 
    # Validate additional hyperparameters
    if not additional_hp.strip():
        json_dict = {}
    else:
        try:
            json_dict = json.loads(additional_hp)
        except Exception as e:
            raise ValueError("JSON format fault: ", e)
    
    # Verify parameters values
    if "perceptual" in json_dict:
        # Check it is in 0-1
        if abs(json_dict["perceptual"]-0.5) > 0.5:
            raise ValueError("Perceptual coefficient should be in the [0,1] interval.")
    # Check missing keys
    if model_code == "beta-vae" and "beta" not in json_dict:
        raise ValueError("A betaVAE model should have a specified 'beta' value.")
    if model_code == "mmd-vae" and not {"alpha", "lambda"}.issubset(set(json_dict.keys())):
        raise ValueError("A MMD InfoVAE model should have specified 'alpha' and 'lambda' values.")
    if model_code in ["lagging-vae", "reslag-vae"] and not {"aggresive-steps", "aggresive-cutoff-epoch"}.issubset(set(json_dict.keys())):
        raise ValueError("A Lagging VAE model should have specified 'aggresive-steps' and 'aggresive-cutoff-epoch' values.")
    # Check abundant keys
    if (model_code == "beta-vae" and not set(json_dict.keys()).issubset({"beta", "capacity", "perceptual"})) \
        or (model_code == "mmd-vae" and not set(json_dict.keys()).issubset({"alpha", "lambda", "perceptual"})) \
        or (model_code in ["lagging-vae", "reslag-vae"]  and not set(json_dict.keys()).issubset({"aggresive-steps", "aggresive-cutoff-epoch", "perceptual"})) \
        or (model_code in ["ae", "vae"] and json_dict):
        warnings.warn("Your hyperparameters contains unwanted keys which will be ignored. ")
    args = {
        "timestamp": timestamp,
        "model_type": model_type, 
        "model_code": model_code,
        "batch_size": batch_size,
        "dataset": dataset,
        "optimizer": optimizer,
        "model_name": model_name,
        "additional_hp": json.dumps(additional_hp),
        "model_dir": model_dir,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "latent_dimension": latent_dimension,
        "lr_scheduler": lr_scheduler_formula,
        "resume_flag": resume_flag,
    }
    parse_additional_hp(args)
    config_path = os.path.join(model_dir, "config.json")
    # Save config after validate resume learning has no issue.
    pretty_dump(args, config_path)
    pretty_dump(args, CACHE_FILE)
    return args

def accept_and_train(model_type: str, dataset: str, batch_size: int, optimizer: str, model_name: str, 
            epochs: int, learning_rate: float, latent_dimension: int, 
            lr_scheduler_flag: bool, lr_scheduler_formula: str,
            additional_hp: str, resume_flag: bool,  progress = gr.Progress(track_tqdm=True)):
    args = validate_and_create_config(model_type, dataset, batch_size, optimizer, model_name, 
                epochs, learning_rate, latent_dimension, 
                lr_scheduler_flag, lr_scheduler_formula,
                additional_hp, resume_flag)
    print("Training Info:")
    for key, value in args.items():
        print(f"\t{key}:\t{value}")
    model_code = args["model_code"]
    
    checkpoint_path = os.path.join(args["model_dir"], "checkpoint.pth")
    checkpoint_earlystop_path = os.path.join(args["model_dir"], "checkpoint_earlystop.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
    print(f"PyTorch Training using device: {device}.")

    # Load dataset
    train_dataset, valid_dataset, train_dataloader, valid_dataloader = \
        dataset_loader(args["dataset"], batch_size=batch_size)
    # Train Mode
    print(f"Training {model_type}")
    model = init_model(latent_dim = args["latent_dimension"], dataset = args["dataset"], 
                            architecture=args["model_code"], device=device, legacy_model_name=args["model_name"])
    if model_code == "beta-vae":
        print(f"Using Beta VAE, beta value is {model.beta}")
    elif model_code == "mmd-vae":
        print(f"Using MMD InfoVAE, alpha value is {model.alpha}, lambda value is {model.l}")
    config_path = os.path.join(args["model_dir"], "config.json")
    # If model is VAE with lagging training regime, use two optimizers
    if model_code not in ["lagging-vae", "reslag-vae"]:
        optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=args["learning_rate"])
    else: 
        optimizer = MultipleOptimizer(
            getattr(torch.optim, optimizer)(model.vencoder.parameters(), lr=args["learning_rate"]),
            getattr(torch.optim, optimizer)(model.decoder.parameters(), lr=args["learning_rate"])
        )
    if (not lr_scheduler_flag) or (not lr_scheduler_formula): 
        scheduler_type = None
        scheduler = None
    elif lr_scheduler_formula == "ReduceLROnPlateau":
        scheduler_type = lr_scheduler_formula
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Save model based on best validation score so far.
    best_valid_loss = float("inf")
    train_losses = {}
    valid_losses = {}
    start_epoch = 1
    # Resume training if enabled
    if resume_flag and os.path.isfile(checkpoint_path):
        print(f"Resume training by loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        try:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            # Possibly caused by changing model or optimizer during resume training
            raise gr.Error(f"Detected changes in model or optimizer for resume training. "\
                        + f"We suggest reload the preferences and train again without edit. "\
                        + f"It could also caused by changes in model definitions.")
        if "scheduler" in checkpoint:
            # Legacy (load entire scheduler which doesn't work, but still is here to make the code works.)
            print("Currently loading scheduler as a whole which should be avoided in the future.")
            scheduler = checkpoint['scheduler']
        elif "scheduler_state_dict" in checkpoint and scheduler is not None:
            # Valid way (Only load state dict if scheduler is defined)
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except:
                raise gr.Error(f"Detected changes in learning rate scheduler for resume training. "\
                        + f"We suggest reload the preferences and train again without edit. "\
                        + f"It could also caused by changes in model definitions.")
            print("Scheduler loaded with state_dict, which is correct. ")

        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        if model_code in ["ae", "vae"]:
            best_valid_loss = sum([stuff[-1] for stuff in valid_losses.values()])
        else:
            best_valid_loss = valid_losses["total"][-1]
        if start_epoch <= args["epochs"]:
            print(f"Resume info:" \
                    + f"\tStarting from epoch {start_epoch}"\
                    + f"\tBest validation loss {best_valid_loss:.6f}")
        else:
            print("Couldn't resume training because the specify end epoch already reached.")
    # Detect any additional parameters:
    if "perceptual" in args and model_code not in ["ae", "vae"]:
        print(f"Using perceptual loss from Alexnet, ratio = {args['perceptual']}.")
    # Add "aggresive" into args when model uses lagging training regime.
    if model_code in ["lagging-vae", "reslag-vae"]:
        args["aggresive"] = True
    for epoch in range(start_epoch, args["epochs"]+1):
        if model_code in ["lagging-vae", "reslag-vae"] and epoch >= args["aggresive-cutoff-epoch"]:
            args["aggresive"] = False
        train_loss_dict, carrying_metrics = train(model, model_code, train_dataloader, optimizer, 
                                device, epoch=epoch, total_epoch=args["epochs"], others = args)
        
        valid_loss_dict = evaluate(model, model_code, valid_dataloader, 
                                device, epoch=epoch, total_epoch=args["epochs"], others = args,
                                carrying_metrics=carrying_metrics)
        
        if model_code in ["lagging-vae", "reslag-vae"] and args["dataset"] in ["MNIST", "FashionMNIST"]:
            # Calculate Mutual Information for MNIST only , other datasets not having enough memory to calculate this.
            model.eval()
            MI = model.mutual_info(valid_dataloader, valid_dataset, valid_loss_dict["latent"])
            print("[Lagging VAE] Mutual Information is", round(MI.item(), 4))
        for key in carrying_metrics:
            if carrying_metrics[key]:
                metric_list = carrying_metrics[key]
                print(f"----Carrying '{key}'")
                print(f"\t\tMean {np.mean(metric_list)}")
                print(f"\t\tStd {np.std(metric_list)}")
                print(f"\t\tMinimum {np.min(metric_list)}")
                print(f"\t\tMaximum {np.max(metric_list)}")
        print(f"Epoch {epoch} - Train vs Valid: ")
        for key in train_loss_dict:
            if valid_loss_dict[key] != 0:
                print(f"\t-{key.title()}: {train_loss_dict[key]:.6f} || {valid_loss_dict[key]:.6f}")
        if not train_losses: train_losses = {key:[] for key in train_loss_dict}
        if not valid_losses: valid_losses = {key:[] for key in valid_loss_dict}
        # Save figures
        for key in train_loss_dict:
            # Do not save perceptual loss if it is zeros
            train_losses[key].append(train_loss_dict[key])
            valid_losses[key].append(valid_loss_dict[key])
            if key == "perceptual" and train_loss_dict[key] == 0:
                continue
            plt.clf()
            plt.plot(list(range(1, epoch+1)), train_losses[key], label = "Train Loss", color="r")
            plt.plot(list(range(1, epoch+1)), valid_losses[key], label = "Valid Loss", color="g")
            plt.legend()
            plt.title(f"{MODEL_TYPE_INV_MAPPING[model_code]} Training Progress ({key.title()})")
            plt.savefig(os.path.join(args["model_dir"], f"training_result_{key}.png"))
            
        # Update learning rate scheduler
        if "total" not in train_loss_dict:
            current_train_loss = sum(train_loss_dict.values())
            current_valid_loss = sum(valid_loss_dict.values())
        else:
            current_train_loss = train_loss_dict["total"]
            current_valid_loss = valid_loss_dict["total"]
        if scheduler is not None:
            scheduler.step(current_valid_loss)
        # Print current learning rate
        if model_code not in ["lagging-vae", "reslag-vae"]:
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.optimizers[0].param_groups[0]['lr']

        print(f"Current learning rate is {current_lr}.")
        # Update metrics to tensorboard
        # writer.add_scalar("Loss/train", current_train_loss, epoch)
        # writer.add_scalar("Loss/valid", current_valid_loss, epoch)
        # Save model based on improvements on validation loss.
        if current_valid_loss < best_valid_loss:
            best_valid_loss = current_valid_loss
            # Save the current epoch if model is better.
            args["current_epoch"] = epoch
            pretty_dump(args, config_path)
            checkpoint = { 
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Model and results stored in {args['model_dir']} based on validation loss improvement.")
            if current_train_loss >= current_valid_loss:
                # On top of valid improvement, save the checkpoint when it is not overfitting. 
                torch.save(checkpoint, checkpoint_earlystop_path)
                print(f"Early stop model is stored in {args['model_dir']} once more because it is not overfitted yet.")

        # Save training results to csv for future references
        train_results_path = os.path.join(args["model_dir"], "metrics.csv")
        with open(train_results_path, "w", newline='') as f:
            w = csv.writer(f)
            w.writerow([""] + [f"{section}_{part}" for section in ["train", "validate"] for part in train_loss_dict])
            for r in range(epoch):
                w.writerow([f"epoch_{r+1}"] + [f"{losses[part][r]:.6f}" for losses in [train_losses, valid_losses] for part in train_loss_dict])
        if current_lr < CUTOFF_LR:
            print(f"Since learning rate {current_lr} is smaller than the preset cutoff {CUTOFF_LR}, "\
                + f"the model terminates on Epoch {epoch} because the learning rate is too small to make a difference.")
            break
    print(f"Model and results stored in {args['model_dir']}. ")
    return "Done"


def headless_accept(model_type: str, dataset: str, batch_size: int, optimizer: str, model_name: str, 
            epochs: int, learning_rate: float, latent_dimension: int, 
            lr_scheduler_flag: bool, lr_scheduler_formula: str,
            additional_hp: str, resume_flag: bool):
    global HEADLESS_COMMAND
    if not model_name:
        # reject empty model name
        raise gr.Error("Model name cannot be empty. ")
    if " " in model_name:
        raise gr.Error("Model name cannot contain whitespace. ")
    # Create config file
    validate_and_create_config(model_type, dataset, batch_size, optimizer, model_name, 
                epochs, learning_rate, latent_dimension, 
                lr_scheduler_flag, lr_scheduler_formula,
                additional_hp, resume_flag)
    if not HEADLESS_COMMAND:
        # Initialize command
        HEADLESS_COMMAND = f"python headless_experiments.py -m {model_name}"
    else:
        HEADLESS_COMMAND += f" {model_name}"
    return gr.update(value=HEADLESS_COMMAND)

def headless_execute(command: str):
    if not command:
        return
    print("Running headless experiments remotely...")
    os.system(command)



def headless_erase():
    global HEADLESS_COMMAND
    HEADLESS_COMMAND = ""
    return gr.update(value="")

def show_graphs(model_name):
    ret = []
    model_dir = os.path.join(OUTPUT_ROOT, model_name)
    empty_model_message = f"The model {model_name} doesn't have metrics to view yet. Please train the model first. "
    if not os.path.isdir(model_dir):
        raise gr.Error(empty_model_message)
    for file in os.listdir(model_dir)[:]:
        if file.endswith(".png"):
            # filepath = model_dir + "/" + file
            filepath = os.path.join(model_dir, file)
            ret.append(filepath)
    if len(ret) == 0:
        raise gr.Error(empty_model_message)
    return gr.update(value=ret)

def backend_load_config(model_name=None):
    if model_name is None:
        global pref
        with open(CACHE_FILE, "r") as f:
            config = json.load(f)
    else:
        model_dir = os.path.join(OUTPUT_ROOT, model_name)
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
    return [
        gr.update(value=config["model_type"]),
        gr.update(value=config["dataset"]),
        gr.update(value=config["batch_size"]),
        gr.update(value=config["optimizer"]),
        gr.update(value=config["model_name"]),
        gr.update(value=config["epochs"]),
        gr.update(value=config["learning_rate"]),
        gr.update(value=config["latent_dimension"]),
        gr.update(value=config["lr_scheduler"]!=""),
        gr.update(value=config["lr_scheduler"], visible=config["lr_scheduler"]!=""),
        gr.update(value=config["additional_hp"]),
    ]

def load_pref():
    # Load from previous session
    try:
        ret = backend_load_config()
    except:
        return [gr.update()]*11
    return ret

def load_from_model_name(resume_flag, model_name):
    # For resume learning
    if not resume_flag:
        return [gr.update()]*11
    try:
        ret = backend_load_config(model_name)
    except:
        return [gr.update()]*11
    return ret

def find_existing_models() -> list:
    ret = []
    for f in os.listdir(OUTPUT_ROOT):
        full_f = os.path.join(OUTPUT_ROOT, f)
        if os.path.isdir(full_f) and os.path.isfile(os.path.join(full_f, "checkpoint.pth")):
            ret.append(f)
    # Sort base on dataset used [MNIST, Fashion, CIFAR10, CIFAR100, CelebA]
    ret.sort(key = get_dataset_from_name)
    return ret

def find_existing_models_ui():
    ret = find_existing_models()
    return gr.update(choices=ret)


def load_config_to_inference(model_name):
    config_path = os.path.join(OUTPUT_ROOT, model_name, "config.json")
    if not os.path.isfile(config_path):
        raise gr.Error(f"The model {model_name} doesn't exist yet.")
    with open(config_path, "r") as f:
        content = f.read().strip()
        data = json.loads(content)
    dataset_for_training = data["dataset"]
    compatible_datasets = COMPATIBLE_DATAMAP[dataset_for_training]
    if len(compatible_datasets) == 1:
        prompt = f"The model is trained on the {dataset_for_training} dataset."
    else:
        other_datasets = [dataset for dataset in compatible_datasets if dataset != dataset_for_training]
        prompt = f"The model was trained on the {dataset_for_training} dataset, but can " +\
                f"be used on other datasets including [{', '.join(other_datasets)}] "+\
                "while not guarantee accurate results. "
    return [
        gr.update(choices=compatible_datasets, value=dataset_for_training), 
        gr.update(value=prompt),
        gr.update(value=content),
    ]

def load_dataset(dataset_name, train_or_valid):
    global CURRENT_DATASET_NAME, CURRENT_TRAIN_OR_VALID, CURRENT_DATASET
    # If dataset already in used. 
    if CURRENT_DATASET_NAME == dataset_name and CURRENT_TRAIN_OR_VALID == train_or_valid:
        pass 
    else:
        # If loading different dataset 
        CURRENT_DATASET_NAME = dataset_name
        CURRENT_TRAIN_OR_VALID = train_or_valid
        try:
            CURRENT_DATASET = dataset_loader_atomic(CURRENT_DATASET_NAME, CURRENT_TRAIN_OR_VALID)
        except:
            raise gr.Error(f"Model could not be loaded, something is wrong. Info: [{dataset_name, train_or_valid}].")
    return gr.update(maximum=len(CURRENT_DATASET), label=f"Choose image from [{CURRENT_DATASET_NAME}] [{CURRENT_TRAIN_OR_VALID.title()}]")

def load_image_to_ui(index):
    global CURRENT_DATASET
    # The index is 1-based, so have to minus 1.
    index_0based = index-1
    tensor_image = CURRENT_DATASET[index_0based][0]
    # print(tensor_image.shape)
    numpy_image = dump_tensor_to_numpy(tensor_image)
    # w, h = numpy_image.shape[:2]
    return gr.update(value=numpy_image)

def load_model(model_name):
    global CURRENT_MODEL, CURRENT_MODEL_NAME, CURRENT_GRAYSCALE
    if CURRENT_MODEL_NAME != model_name:
        # Load model
        if not model_name.strip():
            raise gr.Error("Model name cannot be empty.")
        model_name = model_name.strip()
        model_dir = os.path.join(OUTPUT_ROOT, model_name)
        if not os.path.isdir(model_dir):
            raise gr.Error("Unable to load model because the model directory doesn't exist.")  
        json_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(json_path):
            raise gr.Error("Model directory exists but config not found. ")
        with open(json_path, "r") as f:
            config = json.load(f)
        CURRENT_MODEL = init_model(latent_dim=config["latent_dimension"], dataset=config["dataset"], 
                            architecture=config["model_code"], device=DEVICE, legacy_model_name=model_name)
        checkpoint_path = os.path.join(model_dir, "checkpoint.pth")
        if not os.path.isfile(checkpoint_path):
            raise gr.Error("Model file not found, the training process is likely corrupted.")
        CURRENT_MODEL.load_state_dict(torch.load(checkpoint_path)["model"])
        CURRENT_GRAYSCALE = config["dataset"] in ["MNIST", "FashionMNIST"]
        CURRENT_MODEL_NAME = model_name

def inference(image):
    global CURRENT_GRAYSCALE
    # image is a numpy array, shape (W, H, C) 
    # grayscale will be converted to (W, H, 3) with repeated values
    t = torch.from_numpy(image).to(DEVICE)
    t = torch.permute(t, dims=(2,0,1))
    t = t/255 # Transform to 0-1 (float)
    # Make batch
    t = torch.unsqueeze(t, dim=0)
    # Check grayscale or not
    if CURRENT_GRAYSCALE:
        t = t[:, :1, :, :]
    # Inferencing
    output = CURRENT_MODEL(t) # [B, C, W, H]
    output = output[0]
    numpy_output = dump_tensor_to_numpy(output)
    return gr.update(value=numpy_output)


def delete_model(model_name):
    if not model_name.strip():
        # Prevent empty string as it will delete entire models/ directory.
        raise gr.Error("Please select a model to be deleted. ")
    model_dir = os.path.join(OUTPUT_ROOT, model_name)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
        return f"Directory {model_dir} removed successfully"
    return "Directory not found, nothing change."

def view_last_nrows(num_rows: int): 
    existing_models = find_existing_models()
    dataset_order = {key: i for i, key in enumerate(COMPATIBLE_DATAMAP.keys())}
    extra_arguments = ["beta", "alpha", "lambda", "capacity"]
    ret = []
    for ind, model_name in enumerate(existing_models):
        if ind >= MAX_NUM_DATAFRAMES:
            warnings.warn(f"You have more than {MAX_NUM_DATAFRAMES} hence some model's result won't be shown. ")
            break
        model_dir = os.path.join(OUTPUT_ROOT, model_name)
        csv_result = os.path.join(model_dir, "metrics.csv")
        config_path = os.path.join(model_dir, "config.json")

        if not os.path.isfile(csv_result):
            warnings.warn(f"This shouldn't happen: the model {model_name} exists"\
                          + f"but do not have a checkpoint model. ")
            continue
        # Load csv
        df = pd.read_csv(csv_result)
        # Get best epoch
        with open(config_path, "r") as f:
            data = json.load(f)
        if "current_epoch" not in data:
            continue
        best_epoch = data["current_epoch"]
        dataset = data["dataset"]
        # Get all "valid" data
        headers = df.columns
        needed = [header for header in headers if header.startswith("validate")]
        best_row = df[df.iloc[:, 0] == f"epoch_{best_epoch}"]
        # Select first row then columns with validate.
        best_row = best_row.iloc[0].loc[needed]
        best_row_str = f"Best Valid: Epoch {best_epoch}, " \
            +", ".join([f"{name.replace('validate_', '')}={best_row[name]}" for name in needed])\
            +f", latent dimension is {data['latent_dimension']}, "\
            +"Extra: " + ", ".join(f"{arg} = {data[arg]}" for arg in extra_arguments if arg in data)
        ret.append([gr.update(value=df.tail(n=num_rows), visible=True, label="# " + model_name + "\n"+best_row_str), dataset_order[dataset]])
    # Reorder based on dataset
    ret.sort(key=lambda x: x[1])
    ret = [item[0] for item in ret]
    model_count = len(ret)
    ret += [gr.update(visible=False)] * (MAX_NUM_DATAFRAMES - model_count) 
    return ret

def verify_bulk_seed(seed):
    if seed != -1 and seed < 0:
        raise gr.Error("Negative seeds are not allowed except -1.")
    return "Seed Validated"

def bulk_inference(seed):
    global CURRENT_DATASET, CURRENT_MODEL
    n = len(CURRENT_DATASET)
    # Candidate selections
    if seed == -1:
        rng = default_rng()
    else:
        rng = default_rng(int(seed))
    n_sample = 8
    indices = rng.choice(list(range(n)), size=n_sample, replace=False)
    minibatch = 8 # With limited GPU memory, smaller minibatch is better.
    # Create batch
    batch_list =  [CURRENT_DATASET[ind][0] for ind in indices]
    # Inference
    CURRENT_MODEL.eval()
    minibatches = []
    for i in range(0, n_sample, minibatch):
        batch = batch_list[i: i+minibatch]
        batch = torch.stack(batch).to(DEVICE) # Batch size is 50
        output_batch = CURRENT_MODEL(batch)
        minibatches.append(output_batch)
    full_batch = torch.cat(minibatches)
    # Make both into collage. Only works for 50 images
    np_batch = tensor_to_collage(batch_list, nrow=8)
    np_output_batch = tensor_to_collage(full_batch, nrow=8)
    return [gr.update(value=np_batch), gr.update(value=np_output_batch)]

def get_latent(model, input):
    if hasattr(model, "encoder"):
        prediction = model.encoder(input)
    elif hasattr(model, "vencoder"):
        prediction = model.vencoder(input)
    return prediction

def bulk_inference_all(dataloader):
    global CURRENT_MODEL
    CURRENT_MODEL.eval()
    it = iter(dataloader)
    num_batches = len(dataloader)
    ret = []
    for idx in tqdm(range(0, num_batches)):
        # Extract input from dataloader
        input, label = next(it)
        input = input.to(DEVICE)
        # Compute latent from the image.
        if hasattr(CURRENT_MODEL, "encoder"):
            prediction = CURRENT_MODEL.encoder(input)
        else:
            prediction = CURRENT_MODEL.vencoder(input)

        ret.append(prediction.cpu())
    ret = torch.cat(ret)
    return ret
        
def tsne_broadcast(model_name):
    config_path = os.path.join(OUTPUT_ROOT, model_name, "config.json")
    with open(config_path, "r") as f:
        data = json.load(f)
        dataset_name = data["dataset"]
    return gr.update(choices=COMPATIBLE_DATAMAP[dataset_name], value=dataset_name)

def generate_tsne(model_name, dataset_name, split):
    dataset = dataset_loader_atomic(dataset_name, split)
    labels = [datapoint[1] for datapoint in dataset]
    dl = DataLoader(dataset, batch_size=32, shuffle=False)
    load_model(model_name)
    model_dir = os.path.join(OUTPUT_ROOT, model_name)
    config_path = os.path.join(model_dir, "config.json")

    latents = bulk_inference_all(dl)
    np_latents = latents.cpu().detach().numpy()
    # Use different hyperparameters depends on dataset.
    if dataset_name in ["MNIST", "FashionMNIST"]:
        perplexity=30
        ee = 12
    elif dataset_name in ["CIFAR10", "CIFAR100"]:
        perplexity= 30
        ee = 20

    else:
        raise NotImplementedError(f"T-SNE analysis is not implemented for {dataset_name} yet.")
    print(f"T-SNE for {dataset_name} using perplexity {perplexity}.")
    embedded = TSNE(n_components=2, early_exaggeration=ee,
                    init='pca', perplexity=perplexity, verbose=1).fit_transform(np_latents)
    trun_n = embedded.shape[0]
    labels = labels[:trun_n]
    fig, ax = plt.subplots()

    groups = pd.DataFrame(embedded, columns=['x', 'y']).assign(category=labels).groupby('category')
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name)

    ax.legend()
    ax.set_title(f'Model {model_name}: t-SNE plot on {dataset_name}-{split}.')
    tsne_path = os.path.join(OUTPUT_ROOT, model_name, f"tsne_{split}.png")
    fig.savefig(tsne_path)
    return tsne_path

def latent_traversal(image_id: int, bound: float = 2):
    global CURRENT_DATASET, CURRENT_MODEL
    CURRENT_MODEL.eval()
    try:
        CURRENT_MODEL.decoder.eval()
    except AttributeError:
        pass # Sometimes decoder might be a function instead of a submodel.
    index_0based = image_id-1 # UI is 1-based, change to 0-base
    tensor_image = CURRENT_DATASET[index_0based][0].to(DEVICE)
    tensor_image = torch.unsqueeze(tensor_image, dim=0)
    # Evaluate latent
    latent = get_latent(CURRENT_MODEL, tensor_image)
    latent_dim = latent.shape[1]
    # Create 11 images for a single row
    # 10 rows for a grid. 
    # So there is max 110 images in a grid
    images_per_row = 11
    sample_points = torch.linspace(-bound, bound, images_per_row).to(DEVICE)
    np_images = []
    for row_batch in range(ceil(latent_dim/10)):
        start = row_batch*10
        all_tensors = []
        for row_num in range(min(10, latent_dim-start)):
            row_ind = start + row_num
            pivot = latent[0, row_ind]
            pivot_nearest_ind = torch.argmin(torch.abs(sample_points - pivot))
            repeated_latent = latent.repeat(images_per_row, 1)
            repeated_latent[:, row_ind] = torch.clone(sample_points)
            tensor_images = CURRENT_MODEL.decoder(repeated_latent)
            tensor_images = F.pad(tensor_images, (2,2,2,2), value=0)
            # Highlight the nearest image
            nearest = tensor_images[pivot_nearest_ind]
            nearest_shape = nearest.shape
            inds = [0, 1, nearest_shape[2]-2, nearest_shape[2]-1]
            nearest[:, inds, :] = 1
            nearest[:, :, inds] = 1
            grids = torchvision.utils.make_grid(tensor_images, nrow=images_per_row, padding=0)
            all_tensors.append(grids)
        grand_tensor = torch.cat(all_tensors, dim=1)
        grand_np = dump_tensor_to_numpy(grand_tensor)
        np_images.append(grand_np)
    return latent_dim, gr.update(value=np_images)

def splitted_generation(z, batch=10):
    # Generate multiple images using minibatch, suitable for low-spec GPU.
    global CURRENT_MODEL
    CURRENT_MODEL.eval()
    recs = []
    n = z.shape[0]
    for ind in range(0, n, batch):
        recs.append(CURRENT_MODEL.decoder(z[ind: ind+batch]))
    recs = torch.cat(recs)
    return recs

def sample_from_prior():
    # Sample 16 images and make a 4x4 grid.
    n, nrow = 16, 4
    global CURRENT_MODEL_NAME, CURRENT_MODEL
    if CURRENT_MODEL_NAME is None:
        raise gr.Error("Please select a model before sampling.")
    model_dir = f"./{OUTPUT_ROOT}/{CURRENT_MODEL_NAME}/"
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        data = json.load(f)
    CURRENT_MODEL.eval()
    latent_dimension = data["latent_dimension"]
    z = torch.randn((n, latent_dimension)).to(DEVICE)
    recs = splitted_generation(z)
    collage = torchvision.utils.make_grid(recs, nrow=nrow)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    image_path = os.path.join(model_dir, f"prior_sampling_{timestamp}.png")
    torchvision.utils.save_image(collage, image_path)
    return image_path

def interpolate_two(seed):
    global CURRENT_MODEL, CURRENT_MODEL_NAME, CURRENT_DATASET
    model_dir = os.path.join(OUTPUT_ROOT, CURRENT_MODEL_NAME)
    n = len(CURRENT_DATASET)
    # Candidate selections
    if seed == -1:
        rng = default_rng()
    else:
        rng = default_rng(int(seed))
    n_sample = 2 * 10 # Take 10 pairs of images
    indices = rng.choice(list(range(n)), size=n_sample, replace=False)
    # Create batch
    batch_list =  [CURRENT_DATASET[ind][0] for ind in indices]
    batch = torch.stack(batch_list).to(DEVICE)

    # Inference
    CURRENT_MODEL.eval()
    zs = get_latent(CURRENT_MODEL, batch)
    # Create interpolated latents
    zs = torch.reshape(zs, (10, 2, -1))
    z1 = zs[:, :1, :]
    z2 = zs[:, 1:, :]
    coef1 = torch.linspace(1, 0, steps=11).view(1, -1, 1).to(DEVICE)
    coef2 = torch.linspace(0, 1, steps=11).view(1, -1, 1).to(DEVICE)
    interpolated_latents = z1 * coef1 + z2 * coef2
    latents_linearshape = torch.reshape(interpolated_latents, (110, -1))

    # Generate data from interpolated latents
    recs = splitted_generation(latents_linearshape)

    collage = torchvision.utils.make_grid(recs, nrow=11)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    image_path = os.path.join(model_dir, f"interpolate_two_{timestamp}.png")
    torchvision.utils.save_image(collage, image_path)

    return image_path

def interpolate_four(seed):
    global CURRENT_MODEL, CURRENT_MODEL_NAME, CURRENT_DATASET
    model_dir = os.path.join(OUTPUT_ROOT, CURRENT_MODEL_NAME)
    n = len(CURRENT_DATASET)
    # Candidate selections
    if seed == -1:
        rng = default_rng()
    else:
        rng = default_rng(int(seed))
    n_sample = 4 # Take 4 images
    indices = rng.choice(list(range(n)), size=n_sample, replace=False)
    # Create batch
    batch_list =  [CURRENT_DATASET[ind][0] for ind in indices]
    batch = torch.stack(batch_list).to(DEVICE) # (4, C, H, W)

    # Inference
    CURRENT_MODEL.eval()
    zs = get_latent(CURRENT_MODEL, batch) # (4, latent_dims)
    coef1 = torch.linspace(1, 0, steps=11).view(-1, 1).to(DEVICE)
    coef2 = torch.linspace(0, 1, steps=11).view(-1, 1).to(DEVICE)
    # Interpolate (create z1-z3 and z2-z4 pairs)
    left_vertical = zs[0:1, :] * coef1 + zs[2:3, :] * coef2
    right_vertical = zs[1:2, :] * coef1 + zs[3:4, :] * coef2
    # Interpolate again (create [z1-z3]-[z2-z4])
    coef1 = torch.linspace(1, 0, steps=11).view(1, -1, 1).to(DEVICE)
    coef2 = torch.linspace(0, 1, steps=11).view(1, -1, 1).to(DEVICE)
    full_latents = left_vertical.view(11, 1, -1) * coef1 + right_vertical.view(11, 1, -1) * coef2
    latents_linearshape = torch.reshape(full_latents, (121, -1))

    # Generate data from interpolated latents
    recs = splitted_generation(latents_linearshape)

    collage = torchvision.utils.make_grid(recs, nrow=11)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    image_path = os.path.join(model_dir, f"interpolate_four_{timestamp}.png")
    torchvision.utils.save_image(collage, image_path)

    return image_path


