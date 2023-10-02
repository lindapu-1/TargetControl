import torch
import math
from PIL import Image, ImageDraw, ImageFont ##
import os ##
#bboxes 为什么要搞三个dim
#attn map chunk 2，因为empirically发现只提取一半的head做guidance效果比较好
#为什么要+1 index，是因为tokenizer会加入【startoftext】吗，对，加入了SOT，EOT



def compute_ca_loss(attn_maps_mid, attn_maps_up, bboxes, object_positions):##################
    loss=0
    object_number=len(bboxes)
    if object_number==0:#那么引导的loss直接=0
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    #开始mani

    #attn_mid
    for attn_map_integrated in attn_maps_mid:
        attn_map=attn_map_integrated.chunk(2)[1]##############################
    
        b, i, j=attn_map.shape#(batch*n_heads, sl1, sl2) 
        H=W=int(math.sqrt(i))#i is the length of no. of pixels in total, h, w = squere root of i
        for obj_idx in range(object_number):#第i个物品
            obj_loss=0
            mask=torch.zeros(size=(H,W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H,W))
            for obj_box in bboxes[obj_idx]:#bboxes eg: [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]]
            #给第i个物体对应的box设置mask    
                x_min, y_min, x_max, y_max = int(obj_box[0]*W), int(obj_box[1]*H), \
                int(obj_box[2]*W), int(obj_box[3]*H)
                mask[y_min:y_max, x_min:x_max]=1 #box里的mask=1
            
            #object_positions = Pharse2idx(prompt, phrases), eg [[1,2],[5]]
            for obj_position in object_positions[obj_idx]:#for 1 in [1,2]
                ca_map_obj=attn_map[:,:,obj_position].reshape(b,H,W)#selecting the 1st colume in each batch, 
                #each colume conatins i values refer to how each pixel attend to this word 
                #shape (b,i) is then changed to (b,h,w) for a attention map
                activation_value=(ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                #vector of shape (b) / (b)
                #box里的weight/全部的weight
                obj_loss += torch.mean((1-activation_value)**2)#(1-百分比)**2 for batch=1
            
            loss+=(obj_loss/len(object_positions[obj_idx]))

    #attn_up
    for attn_map_integrated in attn_maps_up:
        attn_map=attn_maps_up.chunk(2)[1] 

        b, i, j=attn_map.shape 
        H=W=int(math.sqrt(i))
        for obj_idx in range(object_number):
            obj_loss=0
            mask=torch.zeros(size=(H,W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H,W))
            for obj_box in bboxes[obj_idx]:#bboxes eg: [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]]
            #给第i个物体对应的box设置mask    
                x_min, y_min, x_max, y_max = int(obj_box[0]*W), int(obj_box[1]*H), \
                int(obj_box[2]*W), int(obj_box[3]*H)
                mask[y_min:y_max, x_min:x_max]=1 
            
            #object_positions = Pharse2idx(prompt, phrases), eg [[1,2],[5]]
            for obj_position in object_positions[obj_idx]:#for 1 in [1,2]
                ca_map_obj=attn_map[:,:,obj_position].reshape(b,H,W)#selecting the 1st colume in each batch, 
                #each colume conatins i values refer to how each pixel attend to this word 
                #shape (b,i) is then changed to (b,h,w) for a attention map
                activation_value=(ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
                #vector of shape (b) / (b)
                #box里的weight/全部的weight
                obj_loss += torch.mean((1-activation_value)**2)#(1-%)**2 for batch=1
            
            loss+=(obj_loss/len(object_positions[obj_idx]))#如果一个obj对应两个word，那么分别占1/2，保证每个obj的weight都一样（不会因为word多就占比多

        
    loss=loss/(object_number*len((attn_maps_up[0])+len(attn_maps_mid)))
    return loss

def Pharse2idx(prompt, phrases):#"phrases"= "hello kitty; ball"
    phrases=[x.strip() for x in phrases.split(';')] #return a list [hello kitty,ball]
    prompt_list=prompt.strip('.').split(' ')#return a list of words in prompt
    object_positions=[]
    for obj in phrases:
        obj_position=[]
        for word in obj.split(' '):
            obj_first_index=prompt_list.index(word)+1#########################################
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)
        #[[1,2],[5]] refer to the position of hello kitty and ball
        #then hello kitty is the first and 2nd word and ball is the 5th word

    return object_positions


def draw_box(pil_img, bboxes, phrases, save_path):
    draw=ImageDraw.Draw(pil_img)#create a draw object
    font=ImageFont.truetype('./FreeMono.ttf', 25)#load 字体
    phrases=[x.strip() for x in phrases.split(';')]#return list of obj name
    for obj_boxes, phrase in zip(bboxes, phrases):
        for obj_box in obj_bboxes:
            x_0, y_0, x_1, y_1=obj_box[0], obj_box[1], obj_box[2], obj_box[3]
            draw.rectangle([int(x_0*512), int(y_0*512), int(x_1*512), int(y_1*512)], outline='red', width=5)
            draw.text((int(x_0*512)+5, int(y_0*512)+5), phrase, font=font, fill=(255,0,0))
    pil_img.save(save_path)

def save_image(pil_img, save_path):
    pil_img.save(save_path)



#def setup_logger

