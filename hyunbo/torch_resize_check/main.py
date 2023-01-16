import torch
import glob
import imageio
import os

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from resize import imresize

BICUBIC_LR_x4_folder = '/workspace/LIIF/trainsets/trainH/DIV2K/DIV2K_train_LR_bicubic/X4'
ORIGINAL =  '/workspace/LIIF/trainsets/trainH/DIV2K/DIV2K_train_HR/*'




def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))



def resize_fn2(img, size):
    return transforms.ToTensor()(
            transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))


def main():
    item = 0
    for idx, original_path in enumerate(glob.glob(ORIGINAL)):
        # get image 
        hr_img = imageio.imread(original_path) 
        hr_img = torch.from_numpy(hr_img).permute(2, 0, 1).float()/255.
        
        basename = os.path.basename(original_path).split('.')[0]

        lr_path = os.path.join(BICUBIC_LR_x4_folder, basename + 'x4.png')
        lr_img = imageio.imread(lr_path)
        lr_img = torch.from_numpy(lr_img).permute(2, 0, 1).float()/255.

        # resize
        resized_img_from_pytorch = resize_fn(hr_img, size=(lr_img.shape[1], lr_img.shape[2]))
        resized_img_from_sanghyunbo = imresize(hr_img, 1/4)

        # check diff
        diff_from_pytorch = (resized_img_from_pytorch- lr_img).abs()
        diff_from_sanghyun = (resized_img_from_sanghyunbo - lr_img).abs()
        

        print("resized_img_from_pytorch size: ", resized_img_from_pytorch.max())
        print("resized_img_from_sanghyunbo size: ", resized_img_from_sanghyunbo.max())
        print("diff of pytorch: ", diff_from_pytorch.mean())
        print("diff of sanghyun: ", diff_from_sanghyun.mean())
        print("====================")
        if diff_from_sanghyun.mean() < diff_from_pytorch.mean():
            item += 1
        
        if idx % 50 == 0:
            print(f"current idx: {idx+1}, item: {item}")

    print("item: ", item)



if __name__ == "__main__":
    main()

