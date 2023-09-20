# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# Copyright 2023 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Models used in Stable diffusion.
Modified from stable_diffusion_tensorrt_txt2img.py in diffusers and TensorRT demo diffusion.
"""
import logging
import os
import tempfile

import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from ort_optimizer import OrtStableDiffusionOptimizer
from polygraphy.backend.onnx.loader import fold_constants
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel

logger = logging.getLogger(__name__)


class TrtOptimizer:
    def __init__(self, onnx_graph):
        self.graph = gs.import_onnx(onnx_graph)

    def cleanup(self):
        self.graph.cleanup().toposort()

    def get_optimized_onnx_graph(self):
        return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)

    def infer_shapes(self):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_onnx_path = os.path.join(temp_dir, 'model.onnx')
                onnx.save_model(onnx_graph,
                    input_onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False)
                output_onnx_path = os.path.join(temp_dir, 'model_with_shape.onnx')
                onnx.shape_inference.infer_shapes_path(input_onnx_path, output_onnx_path)
                onnx_graph = onnx.load(output_onnx_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)

class StableDiffusionModelHelper:
    @staticmethod
    def get_name(version, pipeline):
        if version == "1.4":
            if pipeline.is_inpaint():
                return "runwayml/stable-diffusion-inpainting"
            else:
                return "CompVis/stable-diffusion-v1-4"
        elif version == "1.5":
            if pipeline.is_inpaint():
                return "runwayml/stable-diffusion-inpainting"
            else:
                return "runwayml/stable-diffusion-v1-5"
        elif version == "2.0-base":
            if pipeline.is_inpaint():
                return "stabilityai/stable-diffusion-2-inpainting"
            else:
                return "stabilityai/stable-diffusion-2-base"
        elif version == "2.0":
            if pipeline.is_inpaint():
                return "stabilityai/stable-diffusion-2-inpainting"
            else:
                return "stabilityai/stable-diffusion-2"
        elif version == "2.1":
            return "stabilityai/stable-diffusion-2-1"
        elif version == "2.1-base":
            return "stabilityai/stable-diffusion-2-1-base"
        elif version == 'xl-1.0':
            if pipeline.is_sd_xl_base():
                return "stabilityai/stable-diffusion-xl-base-1.0"
            elif pipeline.is_sd_xl_refiner():
                return "stabilityai/stable-diffusion-xl-refiner-1.0"
            else:
                raise ValueError(f"Unsupported SDXL 1.0 pipeline {pipeline.name}")
        else:
            raise ValueError(f"Incorrect version {version}")
    
    @staticmethod
    def get_clip_embedding_dim(version, pipeline):
        # TODO: can we read from config instead
        if version in ("1.4", "1.5"):
            return 768
        elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
            return 1024
        elif version in ("xl-1.0") and pipeline.is_sd_xl_base():
            return 768
        else:
            raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

    @staticmethod
    def get_clipwithproj_embedding_dim(version, pipeline):
        if version in ("xl-1.0"):
            return 1280
        else:
            raise ValueError(f"Invalid version {version} + pipeline {pipeline}")
    
    @staticmethod
    def get_unet_embedding_dim(version, pipeline):
        if version in ("1.4", "1.5"):
            return 768
        elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
            return 1024
        elif version in ("xl-1.0") and pipeline.is_sd_xl_base():
            return 2048
        elif version in ("xl-1.0") and pipeline.is_sd_xl_refiner():
            return 1280
        else:
            raise ValueError(f"Invalid version {version} + pipeline {pipeline}")    
            
class BaseModel:
    def __init__(self, version:str, pipeline, model, device="cuda", fp16:bool=False, max_batch_size:int=16, embedding_dim:int=768, text_maxlen:int=77):
        self.name = self.__class__.__name__
        
        self.version = version
        self.pipeline = pipeline.name()
        self.use_safetensor = pipeline.is_sd_xl()
        self.path = StableDiffusionModelHelper.get_name(version, pipeline)
        
        self.model = model
        self.fp16 = fp16
        self.device = device

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

        MODEL_TYPE_DICT = {
            "CLIP" : "clip",
            "UNet" : "unet",
            "VAE": "vae",            
            "UNetXL" : "unet",
            "CLIPWithProj": "clip",
        }
        self.model_type = MODEL_TYPE_DICT[self.name]
        self.ort_optimizer = OrtStableDiffusionOptimizer(self.model_type)

    def get_model(self):
        return self.model

    def load_model(self, framework_model_dir:str, hf_token:str, subfolder:str):
        pass
    
    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_profile_id(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        """For TensorRT EP"""
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)

        profile_id = f"_b_{batch_size}" if static_batch else f"_b_{min_batch}_{max_batch}"

        if self.name != "CLIP":
            if static_image_shape:
                profile_id += f"_h_{image_height}_w_{image_width}"
            else:
                profile_id += f"_h_{min_image_height}_{max_image_height}_w_{min_image_width}_{max_image_width}"

        return profile_id

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        """For TensorRT"""
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize_ort(self, input_onnx_path, optimized_onnx_path, to_fp16=True):
        self.ort_optimizer.optimize(input_onnx_path, optimized_onnx_path, to_fp16)

    def optimize_trt(self, input_onnx_path, optimized_onnx_path):
        onnx_graph = onnx.load(input_onnx_path)
        opt = TrtOptimizer(onnx_graph)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.cleanup()
        onnx_opt_graph = opt.get_optimized_onnx_graph()
        onnx.save(onnx_opt_graph, optimized_onnx_path)

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_image_shape else self.min_image_shape
        max_image_height = image_height if static_image_shape else self.max_image_shape
        min_image_width = image_width if static_image_shape else self.min_image_shape
        max_image_width = image_width if static_image_shape else self.max_image_shape
        min_latent_height = latent_height if static_image_shape else self.min_latent_shape
        max_latent_height = latent_height if static_image_shape else self.max_latent_shape
        min_latent_width = latent_width if static_image_shape else self.min_latent_shape
        max_latent_width = latent_width if static_image_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )

class CLIP(BaseModel):
    def __init__(self, version, pipeline, model, device, max_batch_size, embedding_dim, output_hidden_state=False, clip_skip=0):
        super().__init__(
            version,
            pipeline,
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim
        )
        self.output_hidden_state = output_hidden_state
        
        # see https://github.com/huggingface/diffusers/pull/5057 for more information of clip_skip.
        # Clip_skip=1 means that the output of the pre-final layer will be used for computing the prompt embeddings.
        self.clip_skip = clip_skip

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_image_shape
        )
        return {
            "input_ids": [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }
        
        if self.output_hidden_state:
            output["hidden_state"] = (batch_size, self.text_maxlen, self.embedding_dim)
            
        return output

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def clip_add_hidden_states(self):
        
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
                    
        assert self.clip_skip >= 0 and self.clip_skip < hidden_layers

        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers - 1 - self.clip_skip):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers - 1 - self.clip_skip):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        return onnx_graph
        
    def optimize_trt(self, input_onnx_path, optimized_onnx_path):
        onnx_graph = onnx.load(input_onnx_path)
        opt = TrtOptimizer(onnx_graph)
        opt.select_outputs([0])  # delete graph output#1
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.select_outputs([0], names=["text_embeddings"])  # rename network output
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        if self.output_hidden_state:
            onnx_opt_graph = opt.clip_add_hidden_states()
        onnx_opt_graph = opt.get_optimized_onnx_graph()
        onnx.save(onnx_opt_graph, optimized_onnx_path)

    #TODO: move this out since we pass the model object in constructor.
    def load_model(self, framework_model_dir, hf_token, subfolder="text_encoder"):
        clip_model_dir = os.path.join(framework_model_dir, self.version, self.pipeline, subfolder)
        if not os.path.exists(clip_model_dir):
            model = CLIPTextModel.from_pretrained(self.path,
                subfolder=subfolder,
                use_safetensors=self.use_safetensor,
                use_auth_token=hf_token).to(self.device)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"Load CLIP pytorch model from: {clip_model_dir}")
            model = CLIPTextModel.from_pretrained(clip_model_dir).to(self.device)
        return model
    
class CLIPWithProj(CLIP):
    def __init__(self,
        version,
        pipeline,
        model,
        device='cuda',
        max_batch_size=16,
        output_hidden_states=False,
        clip_skip=0):

        super().__init__(
            version, pipeline, model, device=device, max_batch_size=max_batch_size, 
            embedding_dim=StableDiffusionModelHelper.get_clipwithproj_embedding_dim(version, pipeline),
            output_hidden_states=output_hidden_states,
            clip_skip=clip_skip)

    def load_model(self, framework_model_dir, hf_token, subfolder="text_encoder_2"):
        clip_model_dir = os.path.join(framework_model_dir, self.version, self.pipeline, subfolder)
        if not os.path.exists(clip_model_dir):
            model = CLIPTextModelWithProjection.from_pretrained(self.path,
                subfolder=subfolder,
                use_safetensors=self.use_safetensor,
                use_auth_token=hf_token).to(self.device)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"Load CLIP pytorch model from: {clip_model_dir}")
            model = CLIPTextModelWithProjection.from_pretrained(clip_model_dir).to(self.device)
        return model

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

        return output
    
class UNet(BaseModel):
    def __init__(
        self,
        version,
        pipeline,
        model,
        device="cuda",
        fp16=False,  # used by TRT
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
    ):
        super().__init__(
            version,
            pipeline,
            model=model,
            device=device,
            fp16=fp16,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim

    def load_model(self, framework_model_dir, hf_token, subfolder="unet"):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        unet_model_dir = os.path.join(framework_model_dir, self.version, self.pipeline, subfolder)
        if not os.path.exists(unet_model_dir):
            model = UNet2DConditionModel.from_pretrained(self.path,
                subfolder=subfolder,
                use_safetensors=self.use_safetensor,
                use_auth_token=hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(unet_model_dir)
        else:
            print(f"Load UNet pytorch model from: {unet_model_dir}")
            model = UNet2DConditionModel.from_pretrained(unet_model_dir).to(self.device)
        return model
    
    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)
        return {
            "sample": [
                (2 * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (2 * batch_size, self.unet_dim, latent_height, latent_width),
                (2 * max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (2 * min_batch, self.text_maxlen, self.embedding_dim),
                (2 * batch_size, self.text_maxlen, self.embedding_dim),
                (2 * max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": [1],
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(
                2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.tensor([1.0], dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        )


class UNetXL(BaseModel):
    def __init__(self,
        version,
        pipeline,
        model,
        device="cuda",
        fp16=False,  # used by TRT
        max_batch_size=16,
        text_maxlen=77,
        unet_dim=4,
        time_dim=6
    ):
        super().__init__(version, pipeline, model, device=device, fp16=fp16, max_batch_size=max_batch_size,
                         embedding_dim=StableDiffusionModelHelper.get_unet_embedding_dim(version, pipeline),
                         text_maxlen=text_maxlen)
        self.unet_dim = unet_dim
        self.time_dim = time_dim

    def load_model(self, framework_model_dir, hf_token, subfolder="unet"):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        unet_model_dir = os.path.join(framework_model_dir, self.version, self.pipeline, subfolder)
        if not os.path.exists(unet_model_dir):
            model = UNet2DConditionModel.from_pretrained(self.path,
                subfolder=subfolder,
                use_safetensors=self.use_safetensor,
                use_auth_token=hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(unet_model_dir)
        else:
            print(f"Load UNet pytorch model from: {unet_model_dir}")
            model = UNet2DConditionModel.from_pretrained(unet_model_dir).to(self.device)
        return model

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states', 'text_embeds', 'time_ids']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '2B'},
            'latent': {0: '2B', 2: 'H', 3: 'W'},
            'text_embeds': {0: '2B'},
            'time_ids': {0: '2B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(2*min_batch, self.unet_dim, min_latent_height, min_latent_width), (2*batch_size, self.unet_dim, latent_height, latent_width), (2*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(2*min_batch, self.text_maxlen, self.embedding_dim), (2*batch_size, self.text_maxlen, self.embedding_dim), (2*max_batch, self.text_maxlen, self.embedding_dim)],
            'text_embeds': [(2*min_batch, 1280), (2*batch_size, 1280), (2*max_batch, 1280)],
            'time_ids': [(2*min_batch, self.time_dim), (2*batch_size, self.time_dim), (2*max_batch, self.time_dim)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (2*batch_size, 4, latent_height, latent_width),
            'text_embeds': (2*batch_size, 1280),
            'time_ids': (2*batch_size, self.time_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(2*batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(2*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            {
                'added_cond_kwargs': {
                    'text_embeds': torch.randn(2*batch_size, 1280, dtype=dtype, device=self.device),
                    'time_ids' : torch.randn(2*batch_size, self.time_dim, dtype=dtype, device=self.device)
                }
            }
        )

# VAE Decoder
class VAE(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super().__init__(
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        )

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {"latent": {0: "B", 2: "H", 3: "W"}, "images": {0: "B", 2: "8H", 3: "8W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)
