# Controllable Text-to-Image Generation with Customized Guidance on Appearance and Position (Stable diffusion)

This implementation aims to control one target that appeared in the prompt with appearance and position. You can copy appearance from one generated image to a new generated image, and specify the target position with box coordinates.

By extract features from Cross Attention layers as guidance, it works withhout any training or finetuning. 

Some of the results: 
![image](https://github.com/lindapu-1/TargetControl/assets/97086254/125972c1-f46c-40bb-86a9-98fa69c4d6a9)



## Related work
Training-free layout control with cross-attention guidance. 
Diffusion self-guidance for controllable image generation. arXiv preprint arXiv:2306.00986, 2023.
Prompt-to-prompt image editing with cross-attention control. In ICLR, 2023.



