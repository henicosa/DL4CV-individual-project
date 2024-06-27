import yaml
import dlcv.train as train
import os

def create_folders(output_path):
    folders = ["performance", "models", "predictions", "visualisations", "configurations"]
    
    for folder in folders:
        folder_path = os.path.join(output_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")

default_config_str = """# config.yaml
DATA:
  ROOT: "data"
  BATCH_SIZE: 2

MODEL:
  PRETRAINED_WEIGHTS: false
  FREEZE_LAYERS: "0"

TRAINING:
  EPOCHS: 3
  BASE_LR: 0.001
  STRATIFICATION_RATES: false
  MOMENTUM: 0.09
  WEIGHT_DECAY: 0.001
  OPTIMIZER: "SGD"
  BACKBONE: "mobilenet_v2"

AUGMENTATION:
  HORIZONTAL_FLIP_PROB: 0.0
  ROTATION_DEGREES: 0.0

OUTPUT:
  OUTPUT_PATH: "output"

SYSTEM:
  NO_CUDA: false
  DO_EARLY_STOPPING: false"""

#
# Execution Wrapper with storing configuration in a file
#
def execute_training_from_config_file(run_name, filepath):
    with open(filepath, 'r') as file:
        options = yaml.safe_load(file)
    execute_training(run_name, options)
    
def collapse_options(config):
    args = {}
    if "DATA" in config:
        args.update(config["DATA"])
    if "MODEL" in config:
        args.update(config["MODEL"])
    if "AUGMENTATION" in config:
        args.update(config["AUGMENTATION"])
    if "TRAINING" in config:
        args.update(config["TRAINING"])
    if "OUTPUT" in config:
        args.update(config["OUTPUT"])
    if "SYSTEM" in config:
        args.update(config["SYSTEM"])
    
    return args

def merge_options(options):

    # Parse the YAML content
    default_config = yaml.safe_load(default_config_str)
    
    def recursive_update(default_config, update):
        for key, value in update.items():             
            if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                recursive_update(default_config[key], value)
            else:
                default_config[key] = value
    
    recursive_update(default_config, options)
    
    return default_config
    

import time

class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.task_name = None

    def start(self, name=None):
        self.start_time = time.time()
        self.end_time = None
        self.task_name = name
        if name:
            print(f"Stopwatch started for task: {name}")
        else:
            print("Stopwatch started.")

    def stop(self):
        if self.start_time is None:
            print("Stopwatch has not been started.")
            return
        
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        hours, minutes, seconds = self._format_time(elapsed_time)
        if self.task_name:
            print(f"Time taken for {self.task_name}: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")
        else:
            print(f"Time taken: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")
    
    def _format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return int(hours), int(minutes), seconds


def execute_training(run_name, options):

    merged_options = merge_options(options)
    
    args = collapse_options(merged_options)
    create_folders(args["OUTPUT_PATH"])
    with open(os.path.join(args["OUTPUT_PATH"] , "configurations/" + run_name + '.yaml'), 'w') as file:
        yaml.dump(options, file)
    yaml_string = yaml.dump(merged_options, default_flow_style=False)

    stopwatch = Stopwatch()
    stopwatch.start("Training for " + run_name )
    train.train_notebook(run_name, args)
    stopwatch.stop()

    