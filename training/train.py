import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm 

from diffusers import StableDiffusionPipeline,DDPMScheduler
from transformers import CLIPTokenizer

from training.dataset import LayoutDataset
from training.controlnet import ControlNet
from models.layout_encoder import LayoutEncoder
from models.fusion import ConditioningFusion

class Trainer:
    def __init__(self,config):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dataset
        self.dataset=LayoutDataset(config['train_json'],config['tensor_dir'],config['image_dir'])
        
        self.loader=DataLoader(self.dataset,batch_size=config['batch_size'],shuffle=True)
        
        
        # Load Stable Diffusion Model 
        self.pipe=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipe.to(self.device)
        
        self.unet=self.pipe.unet
        self.vae=self.pipe.vae
        self.text_encoder=self.pipe.text_encoder
        self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14") 
        
        # Additional custom modules 
        self.controlnet=ControlNet().to(self.device)
        self.layout_encoder=LayoutEncoder().to(self.device)
        self.fusion=ConditioningFusion().to(self.device)
        
        self.scheduler=DDPMScheduler(num_train_timesteps=1000)
        
        # Optimizer
        
        self.optimizer=torch.optim.AdamW(
            list(self.unet.parameters()) + list(self.controlnet.parameters()) + list(self.layout_encoder.parameters()),lr=config['lr']
        )
        
    def encode_text(self,captions):
        tokens=self.tokenizer(captions,padding="max_length",max_length=77,return_tensors="pt")
        input_ids=tokens.input_ids.to(self.device)
        
        with torch.no_grad():
            return self.text_encoder(input_ids)[0]
        
    def encode_images(self,images):
        images=images.to(self.device)
        
        with torch.no_grad():
            latents=self.vae.encode(images).latest_dist.sample()
            latents=latents*0.18215
        return latents
    
    def train_step(self,batch):
        layouts=batch['layout'].to(self.device)
        control=batch['control'].to(self.device)
        images=batch['image'].to(self.device)
        captions=batch['caption']
        
        # Encode all the inputs
        latents=self.encode_images(images)
        noise=torch.randn_like(latents)
        timesteps=torch.randint(0,1000,(latents.shape[0],),device=self.device).long()
        
        noisy_latents=self.scheduler.add_noise(latents,noise,timesteps)
        
        text_emb=self.encode_text(captions)
        layout_emb=self.layout_encoder(layouts)
        cong=self.fusion(text_emb,layout_emb)
        
        # Controlnet features
        control_feats=self.controlnet(control) 
        
        # Inject control features into UNet
        for feat in control_feats:
            resized=torch.nn.functional.interpolate(feat,size=noisy_latents.shape[2:],mode='bilinear',align_corners=False)
            noisy_latents=noisy_latents+resized
            
        # UNet forward pass
        pred_noise=self.unet(noisy_latents,timesteps,encoder_hidden_states=cond).sample
        
        loss=torch.nn.functional.mse_loss(pred_noise,noise)
        return loss
    
    def train(self,epochs):
        for epoch in range(epochs):
            loop=tqdm(self.loader)
            
            for batch in loop:
                self.optimizer.zero_grad()
                
                loss=self.train_step(batch)
                loss.backward()
                self.optimizer.step()
                
                loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(loss=loss.item())
            
            torch.save({
                'unet':self.unet.state_dict(),
                'controlnet':self.controlnet.state_dict(),
                'layout_encoder':self.layout_encoder.state_dict(),
                'fusion':self.fusion.state_dict(),
                'optimizer':self.optimizer.state_dict()
            },f"checkpoint_epoch_{epoch+1}.pt")
            
            
            
        