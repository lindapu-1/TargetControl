# Controllable Text-to-Image Generation with Customized Guidance on Appearance and Position (Stable diffusion)

This implementation is designed to control a specific target that appears in the prompt, focusing on its appearance and position. It allows you to transfer the appearance from one generated image to a new generated image and specify the target position using box coordinates. By extracting features from Cross Attention layers as guidance, this method operates effectively **without the need for any model training or fine-tuning, which is different from LoRA and ControlNet.** This approach leverages the inherent capabilities of the Cross Attention mechanism to ensure accurate and efficient feature transfer and positioning.
<img width="776" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/345a5255-f1e8-416b-9ec8-4ff9b0288a9b">


## Experiments & Results
Given one **ONE** reference image, the method can almost retain the details of the target concept.
<img width="776" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/c7987741-ca17-445e-b0e5-e968f7725308">

The method can also be applied to features other than the concept appearance, such as position, size, and so on. The only difference is the design of loss function (energy function).
<img width="776" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/fffdcce6-061e-4ade-b0d7-4c8c1d5a91a2">

It is also easy to learn** multiple reference image on seperate targets**, and combine them into one generated image.
<img width="776" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/96e93e04-aac9-40f0-bd69-46685077d1fb">

The method is compatible with other customization methods, such as Textual Inversion, to control the output from different levels. 
<img width="776" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/7dd70d5d-dbea-4884-a825-0f9a49c7d21a">


## How to use
The .ipynb contains the pipeline and the edited unet is in my_model. Please load the Unet from my_model directory, rather than the diffuser package. 

Please see the details in Poster.pdf

Some of the results are on the right side:
<img width="776" alt="image" src="https://github.com/lindapu-1/TargetControl/assets/97086254/374e24be-e02f-4787-9fd0-62b076c1b148">







## Related paper
The code is based on the implementation of the following papers: 

Training-free layout control with cross-attention guidance. 

Diffusion self-guidance for controllable image generation.

Prompt-to-prompt image editing with cross-attention control.



