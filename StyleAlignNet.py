#!/usr/bin/env python
# coding: utf-8




import numpy as np
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import Custom_loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm




dir = '' #using Main dir





a_list = [dir+'/CT/masks/A/'+items for items in os.listdir(dir+'/CT/masks/A/')]
b_list = [dir+'/CT/masks/B/'+items for items in os.listdir(dir+'/CT/masks/B/')]
c_list = [dir+'/CT/masks/C/'+items for items in os.listdir(dir+'/CT/masks/C/')]



A_inout = a_list[:100]
B_inout = b_list[:100]
C_inout = c_list[:100]
A_val = a_list[100:125]
B_val = b_list[100:125]
C_val = c_list[100:125]




class ImageMaskDataset(Dataset):
    def __init__(self, true_dir, A_dir, transform=None):
        self.true_dir = true_dir
        self.A_dir = A_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.true_dir)

    def __getitem__(self, idx):
        # Load image and convert to grayscale
        true_path = self.true_dir[idx]
        A_path = self.A_dir[idx]
        
        true_img = Image.open(true_path).convert('L')  # Convert to grayscale
        A_img = Image.open(A_path).convert('L')  # Grayscale mask
        
        if self.transform:
            true_img = self.transform(true_img)
            A_img = self.transform(A_img)
        
        return true_img, A_img


transform = transforms.Compose([transforms.ToTensor()])
Train_dataset = ImageMaskDataset(A_inout, C_inout, transform=transform)
Val_dataset = ImageMaskDataset(A_val, C_val, transform=transform)
Train_dataloader = torch.utils.data.DataLoader(Train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
Val_dataloader = torch.utils.data.DataLoader(Val_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)


class Style_model(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(Style_model, self).__init__()
        
        # Encoder path
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        # Final output layer to get a single channel output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path with max pooling
        enc1 = self.encoder1(x)  # (batch, 64, 512, 512)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))  # (batch, 128, 256, 256)
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))  # (batch, 256, 128, 128)
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, kernel_size=2))  # (batch, 512, 32, 32)

        # Decoder path with upsampling and skip connections
        dec3 = self.upconv3(bottleneck)  # (batch, 256, 128, 128)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        dec3 = self.decoder3(dec3)  # (batch, 256, 128, 128)
        
        dec2 = self.upconv2(dec3)  # (batch, 128, 256, 256)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)  # (batch, 128, 256, 256)
        
        dec1 = self.upconv1(dec2)  # (batch, 64, 512, 512)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        dec1 = self.decoder1(dec1)  # (batch, 64, 512, 512)
        
        # Final output layer
        output = self.final_conv(dec1)  # (batch, 1, 512, 512)
        
        return output

    def generate_mask(self, x, threshold=0.5):
        logits = self.forward(x)  # Get logits from the model
        probabilities = torch.sigmoid(logits)  # Use sigmoid for binary mask generation

        # Generate binary mask
        mask = (probabilities > threshold).float()  # Thresholding to create binary mask
        return mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Style_model(in_channels=1, num_classes=1)  # Assuming single-channel output for the mask
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)




optimizer = optim.Adam(model.parameters(), lr=0.00005)



num_epochs = 1000
patience=30
best_val_loss = float('inf')
check_path = ''#set the name
patience_counter=0
accumulation_steps = 1

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    train_bar = tqdm(Train_dataloader, desc=f"Training Epoch {epoch+1}", ncols=100, leave=False)

    optimizer.zero_grad()

    for batch_idx, (true_img, A_img) in enumerate(train_bar):
        true_img = true_img.to(device)
        A_img =  A_img.to(device)
        
        # Forward pass
        outputs = model(true_img)  # Get predictions

        if torch.any(torch.isnan(outputs)):
            break
            
        outputs = torch.sigmoid(outputs)
            
        length_loss = Custom_loss.length_loss(outputs, A_img)
        kl_loss = Custom_loss.KL_Loss(outputs, A_img)
        direct_loss = Custom_loss.direction_loss(outputs,A_img)
        polar_loss = Custom_loss.polar_loss(outputs, A_img)
        iou_loss = 1-Custom_loss.mean_iou(outputs, A_img)

        loss = (length_loss+6*direct_loss+2*kl_loss+0.001*polar_loss+iou_loss) / accumulation_steps

        # Backward pass and optimization
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx +1) == len(train_bar):
            optimizer.step()
            optimizer.zero_grad()

        train_loss += (length_loss+6*direct_loss+2*kl_loss+0.001*polar_loss+iou_loss).item()

        train_bar.set_postfix(loss=(length_loss+6*direct_loss+2*kl_loss+0.001*polar_loss+iou_loss).item())

    train_loss /= len(Train_dataloader)

    # Validation loop (optional)
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    val_bar = tqdm(Val_dataloader, desc=f"Validation Epoch {epoch+1}", ncols=100, leave=False)

    with torch.no_grad():
        for batch_idx, (true_img, A_img) in enumerate(val_bar):
            true_img = true_img.to(device)
            A_img =  A_img.to(device)

            outputs = model(true_img)
            outputs = torch.sigmoid(outputs)
            
            length_loss = Custom_loss.length_loss(outputs, A_img)
            kl_loss = Custom_loss.KL_Loss(outputs, A_img)
            direct_loss = Custom_loss.direction_loss(outputs,A_img)
            polar_loss = Custom_loss.polar_loss(outputs, A_img)
            iou_loss = 1-Custom_loss.mean_iou(outputs, A_img)

            loss = length_loss+6*direct_loss+2*kl_loss+0.001*polar_loss+iou_loss
            
            val_loss += loss.item()

            val_bar.set_postfix(loss=loss.item())

    val_loss /= len(Val_dataloader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), check_path)
        patience_counter = 0
    else:
        patience_counter +=1

    if patience_counter == patience:
        break


# In[22]:


Mymodel = Style_model()  # Assuming single-channel output for the mask
Mymodel = nn.DataParallel(Mymodel)
Mymodel.load_state_dict(torch.load(check_path))
Mymodel.to(device)


# In[23]:


Test_true = a_list[125:]


# In[24]:


class ImageMaskDataset(Dataset):
    def __init__(self, true_dir, transform=None):
        self.true_dir = true_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.true_dir)

    def __getitem__(self, idx):
        # Load image and convert to grayscale
        true_path = self.true_dir[idx]
        
        true_img = Image.open(true_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            true_img = self.transform(true_img)
        
        return true_img


# In[25]:


test_dataset = ImageMaskDataset(Test_true, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


# In[26]:


def show_images(images, masks, outputs, num_samples=4):
    plt.figure(figsize=(10,10))
    for i in range(num_samples):
        # Model Output
        plt.subplot(1, num_samples, i+1)
        plt.imshow(outputs[i][0], cmap='gray')
        plt.title("Model Output")
        plt.axis('off')




os.mkdir(dir+'')#save dir



with torch.no_grad():  # Disable gradient calculation for inference
    for i, inputs in enumerate(test_dataloader):
        inputs = inputs.to(device)  # Move input to the appropriate device (CPU/GPU)
        
        # Generate mask using the trained model
        mask = Mymodel.module.generate_mask(inputs, threshold=0.5)  # Adjust the threshold if needed
        
        # Convert mask to NumPy and save it as an image
        mask_np = mask.squeeze().cpu().numpy()  # Remove extra dimensions and move to CPU
#         mask_np = 1-mask_np
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))  # Convert to image format
        
        # Save the mask image
        mask_image.save(dir+'/save_dir/'+ Test_true[i][30:]) # using your dir in save_dir
