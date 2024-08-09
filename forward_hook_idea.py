from transformers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
import torch

# Function to set hooks on the layers of interest in the UNet model
def set_hooks(unet):
    attn_weights = {}

    def hook_fn_forward(module, input, output):
        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            # Capture the weights as needed (example for BatchNorm2d layer)
            attn_weights[module] = output

    for name, module in unet.named_modules():
        # Register forward hook for your target layers (example here using BatchNorm2d)
        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            module.register_forward_hook(hook_fn_forward)
    
    return attn_weights

# Load the stable diffusion pipeline with the tiny model
pipeline = StableDiffusionPipeline.from_pretrained("bk-sdm-tiny")

# Access the UNet model from the pipeline and set hooks
unet_model = pipeline.unet
attn_weights = set_hooks(unet_model)

# Generate image while capturing attention weights
prompt = "A horse running through a field"
generator = torch.Generator(device="cuda").manual_seed(123)
output = pipeline(prompt, num_inference_steps=50, generator=generator)

# Retrieve the generated image
generated_image = output.images[0]

# Example of accessing weights captured during inferenc
for layer, weight in attn_weights.items():
    print(f"Layer {layer}: Weights shape -> {weight.shape}")

# Save or use the generated image
generated_image.save("horse.png")
print("Saved at horse.png")

"""
def get_intermediate_layers(model, inputs):
    intermediate_outputs = {}

    def hook(module, input, output):
        intermediate_outputs[module.name] = output
    
    for name, layer in model.named_modules():
        if 'conv' in name or 'down' in name or 'up' in name:
            layer.register_forward_hook(hook)

    _ = model(inputs)
    return intermediate_outputs

inputs = ...  # your input tensor
intermediate_outputs = get_intermediate_layers(model, inputs)
"""