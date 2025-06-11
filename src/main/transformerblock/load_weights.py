from asyncio.log import logger
import tensorflow as tf # type: ignore
import tqdm
import json
import os
import torch
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
    
    def __assign(self, left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))
    
    def load_weights_into_gpt2(self, gpt, params):
        gpt.pos_emb.weight = self.__assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = self.__assign(gpt.tok_emb.weight, params['wte'])
    
        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = self.__assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = self.__assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = self.__assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = self.__assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = self.__assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = self.__assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = self.__assign(
                gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = self.__assign(
                gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = self.__assign(
                gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = self.__assign(
                gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = self.__assign(
                gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = self.__assign(
                gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = self.__assign(
                gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = self.__assign(
                gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = self.__assign(
                gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = self.__assign(
                gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = self.__assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = self.__assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = self.__assign(gpt.out_head.weight, params["wte"])

    def load_gpt2(self, model_dir):
        # Load settings and params
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        logger.log(1, f"The Check Points Path {tf_ckpt_path}","", exc_info=1)
        settings = json.load(open(os.path.join(model_dir, "hparams.json")))
        params = self.__load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
        logger.log(1, f"Params and Setting Extracted Successfully...Params Keys{params.keys()} and Model Setting Values {settings}", "",  exc_info=1)
        return settings, params
    
# loadWeights = LoadWeights()
# settings, params = loadWeights.load_gpt2("src/main/resources/gpt2/124M")
# print(params['wte'].shape)