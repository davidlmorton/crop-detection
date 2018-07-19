#! /usr/bin/env python

import fire
from PIL import Image
from crop_detection.y_model import YModel
from torchvision import transforms
from torchvision.models import resnet18
import sconce
import sys
import torch

class Runner:
    def predict(self, first, second):
        base_model = resnet18(pretrained=True)
        model = YModel(base_model=base_model)
        model.load_state_dict(torch.load('saved_weights.h5'))


        first_image = Image.open(first)
        second_image = Image.open(second)

        def square_pad(image):
            size = image.size
            large_side = max(size)
            width_pad = large_side - size[0]
            height_pad = large_side - size[1]
            return transforms.Pad((width_pad//2, height_pad//2))(image)

        transform = transforms.Compose((
            square_pad,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ))
        first_tensor = transform(first_image)
        second_tensor = transform(second_image)

        inputs = torch.cat((first_tensor, second_tensor))[None, :]

        model.train(False)
        outputs = model(inputs)
        is_paired = torch.exp(outputs['is_paired']).squeeze().data.numpy().argmax()
        is_swapped = torch.exp(outputs['is_swapped']).squeeze().data.numpy().argmax()

        if is_paired:
            if is_swapped:
                print(f"{first} is a cropped version of {second}")
                x = int(outputs['outputs'][0, 0] * second_image.size[0])
                y = int(outputs['outputs'][0, 1] * second_image.size[1])
                print(f"The top left corner of {first} can be found in {second} at about ({x}, {y})")
            else:
                print(f"{second} is a cropped version of {first}")
                x = int(outputs['outputs'][0, 0] * first_image.size[0])
                y = int(outputs['outputs'][0, 1] * first_image.size[1])
                print(f"The top left corner of {second} can be found in {first} at about ({x}, {y})")
        else:
            print("The images are not cropped versions of one another")

if __name__ == '__main__':
    fire.Fire(Runner)
