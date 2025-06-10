import tensorflow as tf # type: ignore
import tqdm
import json
import os
import numpy as np

class LoadWeights:
    def __init__(self):
        print("TensorFlow version:", tf.__version__)
        print("tqdm version:", tqdm.__version__)

    def __load_gpt2_params_from_tf_ckpt(self, ckpt_path, settings):
        # Initialize parameters dictionary with empty blocks for each layer
        params = {"blocks": [{} for _ in range(settings["n_layer"])]}
        # Iterate over each variable in the checkpoint
        for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
            variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
            # Process the variable name to extract relevant parts
            print("name", name)
            variable_name_parts = name.split("/")[1:] # Skip the 'model/' prefix
            # Identify the target dictionary for the variable
            target_dict = params
            if variable_name_parts[0].startswith("h"):
                layer_number = int(variable_name_parts[0][1:])
                target_dict = params['blocks'][layer_number]
                # Recursively access or create nested dictionaries
            for key in variable_name_parts [1:-1]:
                target_dict = target_dict.setdefault(key, {})
                # Assign the variable array to the last key
            last_key = variable_name_parts [-1]
            target_dict [last_key] = variable_array
        return params

    def load_gpt2(self, model_dir):
        # Load settings and params
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        print(tf_ckpt_path)
        settings = json.load(open(os.path.join(model_dir, "hparams.json")))
        params = self.__load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

        return settings, params
    
loadWeights = LoadWeights()
settings, params = loadWeights.load_gpt2("src/main/resources/gpt2/124M")
print(params['wte'].shape)