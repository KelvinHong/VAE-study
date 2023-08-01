import argparse
import os
import json
from gradio_utils import OUTPUT_ROOT, accept_and_train, MODEL_TYPE_MAPPING

parser = argparse.ArgumentParser("Headless training for VAE-GAN Project - You can train multiple models with this script.")
parser.add_argument("-m", "--models", nargs="+", help="Model names")
arg = parser.parse_args()

print("==========Headless Training Initializing...==========")
for i, model_name in enumerate(arg.models):

    print(f"==========Training Model {i+1}: {model_name}==========")
    config_path = os.path.join(OUTPUT_ROOT, model_name, "config.json")
    with open(config_path, "r") as f:
        args = json.load(f)
    accept_and_train(model_type=args["model_type"], dataset=args["dataset"],
                batch_size=args["batch_size"],
                optimizer=args["optimizer"], model_name=args["model_name"],
                epochs=args["epochs"], learning_rate=args["learning_rate"],
                latent_dimension=args["latent_dimension"], lr_scheduler_flag=args["lr_scheduler"]!="",
                lr_scheduler_formula=args["lr_scheduler"], 
                additional_hp=args["additional_hp"],
                resume_flag=True,
            )
    