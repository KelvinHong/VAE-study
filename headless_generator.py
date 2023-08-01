# To generate command for headless_experiments without using the WebUI
from itertools import product
import datetime
import os
import gradio_utils
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--direct', action=argparse.BooleanOptionalAction)
ARGS = parser.parse_args()
base_command = "python headless_experiments.py -m "
now = datetime.datetime.now()
datestamp = now.strftime("%d%m%Y-%H%M%S")
timestamp = now.strftime("%d-%m-%Y %H:%M:%S")
if not os.path.isfile("model_meta.json"): 
    raise ValueError("Please create a json file 'model_meta.json' in the root directory," +\
                     " then supply a list of model config so we can read your training configs.")
with open("model_meta.json", "r") as f:
    grid = json.load(f)
num_models = len(grid)
for id, instance in enumerate(grid):
    model_name =  f"{datestamp}-{instance['model_code']}-{instance['dataset']}-id{id+1}"
    model_code = instance['model_code']
    args = {
        "timestamp": timestamp,
        "model_type": gradio_utils.MODEL_TYPE_INV_MAPPING[model_code],
        "model_code": model_code,
        "batch_size": 32,
        "dataset": instance['dataset'],
        "optimizer": "Adam",
        "model_name": model_name,
        "additional_hp": instance['additional_hp'],
        "model_dir": None,
        "epochs": instance['epochs'],
        "learning_rate": 0.002,
        "latent_dimension": instance['latent_dimension'],
        "lr_scheduler": instance['lr_scheduler'],
        "resume_flag": os.path.isdir(os.path.join(gradio_utils.OUTPUT_ROOT, model_name))
    }
    gradio_utils.validate_and_create_config(model_type=args["model_type"], 
                    dataset=args["dataset"], batch_size=args["batch_size"], 
                    optimizer=args["optimizer"], model_name=args["model_name"], 
                    epochs=args["epochs"], learning_rate=args["learning_rate"], 
                    latent_dimension=args["latent_dimension"], 
                    lr_scheduler_flag=args["lr_scheduler"] != "", 
                    lr_scheduler_formula=args["lr_scheduler"],
                    additional_hp=args["additional_hp"],resume_flag=args["resume_flag"])
    base_command += args["model_name"] + " "

print(f"Initialized {num_models} models. ")
print(f"Run this command to train the models: {base_command}")
if ARGS.direct:
    os.system(base_command)