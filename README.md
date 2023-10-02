# Controllable Text-to-Image Generation with Customized Guidance on Appearance and Position (Stable diffusion)

This implementation aims to control one target that appeared in the prompt with appearance and position. You can copy appearance from one generated image to a new generated image, and specify the target position with box coordinates.

By extract features from Cross Attention layers as guidance, it works withhout any training or finetuning. 


