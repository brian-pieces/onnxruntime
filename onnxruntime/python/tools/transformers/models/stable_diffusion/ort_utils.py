# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import os
import shutil
from collections import OrderedDict
from typing import Any, Dict

import torch

import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import TypeHelper, CudaSession

logger = logging.getLogger(__name__)

class Engine(CudaSession):
    def __init__(self, engine_path, provider: str, device_id: int = 0, enable_cuda_graph=False):
        self.engine_path = engine_path
        self.provider = provider
        self.provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)

        device = torch.device("cuda", device_id)
        ort_session = ort.InferenceSession(
            self.engine_path,
            providers=[
                (provider, self.provider_options),
                "CPUExecutionProvider",
            ],
        )

        super().__init__(ort_session, device, enable_cuda_graph)

class Engines:
    def __init__(self, provider, onnx_opset: int = 14):
        self.provider = provider
        self.engines = {}
        self.onnx_opset = onnx_opset

    @staticmethod
    def get_onnx_path(onnx_dir, model_name):
        return os.path.join(onnx_dir, model_name + ".onnx")

    @staticmethod
    def get_engine_path(engine_dir, model_name, profile_id):
        return os.path.join(engine_dir, model_name + profile_id + ".onnx")

    def build(
        self,
        models,
        engine_dir: str,
        onnx_dir: str,
        force_engine_rebuild: bool = False,
        fp16: bool = True,
        device_id: int = 0,
        enable_cuda_graph: bool = False,
    ):
        profile_id = "_fp16" if fp16 else "_fp32"

        if force_engine_rebuild:
            if os.path.isdir(onnx_dir):
                logger.info("Remove existing directory %s since force_engine_rebuild is enabled", onnx_dir)
                shutil.rmtree(onnx_dir)
            if os.path.isdir(engine_dir):
                logger.info("Remove existing directory %s since force_engine_rebuild is enabled", engine_dir)
                shutil.rmtree(engine_dir)

        if not os.path.isdir(engine_dir):
            os.makedirs(engine_dir)

        if not os.path.isdir(onnx_dir):
            os.makedirs(onnx_dir)

        # Export models to ONNX
        for model_name, model_obj in models.items():
            onnx_path = Engines.get_onnx_path(onnx_dir, model_name)
            onnx_opt_path = Engines.get_engine_path(engine_dir, model_name, profile_id)
            if os.path.exists(onnx_opt_path):
                logger.info("Found cached optimized model: %s", onnx_opt_path)
            else:
                if os.path.exists(onnx_path):
                    logger.info("Found cached model: %s", onnx_path)
                else:
                    logger.info("Exporting model: %s", onnx_path)
                    model = model_obj.get_model().to(model_obj.device)
                    with torch.inference_mode():
                        inputs = model_obj.get_sample_input(1, 512, 512)
                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=self.onnx_opset,
                            do_constant_folding=True,
                            input_names=model_obj.get_input_names(),
                            output_names=model_obj.get_output_names(),
                            dynamic_axes=model_obj.get_dynamic_axes(),
                        )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

                # Optimize onnx
                logger.info("Generating optimized model: %s", onnx_opt_path)
                model_obj.optimize_ort(onnx_path, onnx_opt_path, to_fp16=fp16)

        for model_name in models:
            engine_path = Engines.get_engine_path(engine_dir, model_name, profile_id)
            engine = Engine(engine_path, self.provider, device_id=device_id, enable_cuda_graph=enable_cuda_graph)
            logger.info("%s options for %s: %s", self.provider, model_name, engine.provider_options)
            self.engines[model_name] = engine

    def get_engine(self, model_name):
        return self.engines[model_name]
