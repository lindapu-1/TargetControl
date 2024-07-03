# Controllable Text-to-Image Generation with Customized Guidance on Appearance and Position (Stable diffusion)

This implementation is designed to control a specific target that appears in the prompt, focusing on its appearance and position. It allows you to transfer the appearance from one generated image to a new generated image and specify the target position using box coordinates. By extracting features from Cross Attention layers as guidance, this method operates effectively **without the need for any model training or fine-tuning, which is different from LoRA and ControlNet.** This approach leverages the inherent capabilities of the Cross Attention mechanism to ensure accurate and efficient feature transfer and positioning.
<img width="800" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/345a5255-f1e8-416b-9ec8-4ff9b0288a9b">


## Experiments & Results
Given one **ONE** reference image, the method can almost retain the details of the target concept.
<img width="780" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/06f95ae7-efe2-4d02-9e0c-e27aa18ed3af">


The method can also be applied to features other than the concept appearance, such as **position, size, and so on**. The only difference is the design of loss function (energy function).

<img width="790" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/8df5ef4d-30d7-4cd3-a79d-47c5af3c555b">


It is also easy to learn **multiple reference image on seperate targets**, and combine them into one generated image.

<img width="770" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/b25fc7a5-5c28-473e-8425-40d7fd7d0f49">



## How to use
The .ipynb contains the pipeline and the edited unet is in my_model. Please load the Unet from my_model directory, rather than the diffuser package. 
Please see the details in Poster.pdf


## Related paper
The code is based on the implementation of the following papers: 

Training-free layout control with cross-attention guidance. 

Diffusion self-guidance for controllable image generation.

Prompt-to-prompt image editing with cross-attention control.



