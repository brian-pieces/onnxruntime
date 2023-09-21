import os

os.environ["ORT_ENABLE_FUSED_CAUSAL_ATTENTION"] = "1"

from optimum.onnxruntime import ORTStableDiffusionXLPipeline

inputs_list = [
    {
        "prompt" : "anime artwork a girl looking at the sea, dramatic, anime style, key visual, vibrant, studio anime, highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
        "height" : 896,
        "width": 1152,
        "guidance_scale": 9
    },
    {
        "prompt" : "starry night vivid painting by van gogh",
        "height" : 768,
        "width": 1344,
        "guidance_scale": 8
    },
    {
        "prompt" : "breathtaking selfie photograph of astronaut floating in space, earth in the background. award-winning, professional, highly detailed",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry",
        "height" : 1024,
        "width": 1024,
        "guidance_scale": 10,
    }
]

for dir in ["sd_xl_rc2"]:
    pipeline = ORTStableDiffusionXLPipeline.from_pretrained(dir, provider="CUDAExecutionProvider")
    for inputs in inputs_list:
        steps = 30
        batch_size = 2
        images = pipeline(
            **inputs,
            num_inference_steps=steps,
            num_images_per_prompt=batch_size,
        ).images

        prompt = inputs["prompt"]
        p = "_".join(prompt.split(' ')[:6])
        for k, image in enumerate(images):
            image.save(f"{dir}__{p}__{k}.jpg")
