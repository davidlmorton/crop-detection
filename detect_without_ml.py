#! /usr/bin/env python

from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

import fire
import time
import torch


def get_original_and_crop_tensors(first_image, second_image, scale=1.0):
    original, crop = get_original_and_crop(first_image, second_image)

    original = original.resize((int(original.size[0]*scale), int(original.size[1]*scale)))
    crop = crop.resize((int(crop.size[0]*scale), int(crop.size[1]*scale)))

    original_tensor = transforms.ToTensor()(original)
    crop_tensor = transforms.ToTensor()(crop)

    return original_tensor, crop_tensor


def get_original_and_crop(first_image, second_image):
    if is_swapped(first_image, second_image):
        original = second_image
        crop = first_image
    else:
        original = first_image
        crop = second_image

    return original, crop


def is_swapped(first_image, second_image):
    first_image_size = first_image.size[0] * first_image.size[1]
    second_image_size = second_image.size[0] * second_image.size[1]

    return first_image_size < second_image_size


def predict_crop_coords(original_tensor, crop_tensor, scale=1.0):
    normalized_conv = get_normalized_conv(original_tensor, crop_tensor)
    return get_max_coords(normalized_conv, scale=scale)


def get_normalized_conv(target, kernel, lognorm=True):
    t = target[None]
    k = kernel[None]
    nk = torch.ones(k.shape)
    b = torch.Tensor([1.0])

    t.requires_grad = False
    k.requires_grad = False
    nk.requires_grad = False
    b.requires_grad = False

    conv = F.conv2d(t, k, bias=b, stride=1, padding=0, dilation=(1,1) , groups=1)
    norm = F.conv2d(t, nk, bias=b, stride=1, padding=0, dilation=(1,1) , groups=1)

    return torch.log(conv) / torch.log(norm)


def get_max_coords(tensor, scale=1.0):
    pt = tensor.squeeze().data.numpy().argmax()
    loc = int((pt % tensor.shape[-1])/scale), int((pt // tensor.shape[-1])/scale)
    return loc


def fine_tune_prediction(t1, t2, loc):
    w, h = loc

    best_mad = get_mad(t1, t2, w, h)
    best_w = w
    best_h = h
    for dw in range(-5, 6):
        for dh in range(-5, 6):
            this_w = w + dw
            this_h = h + dw
            this_mad = get_mad(t1, t2, this_w, this_h)

            if this_mad < best_mad:
                best_mad = this_mad
                best_w = this_w
                best_h = this_h
    return best_w, best_h


def get_mad(t1, t2, w, h):
    idx = get_idx(t2, w, h)
    return _get_mad(t1, t2, idx)


def get_idx(t, w, h):
    idx = (slice(None), slice(h, h+t.size(1)), slice(w, w+t.size(2)))
    return idx


def _get_mad(t1, t2, idx):
    x = (t1[idx] - t2)
    return (t1[idx] - t2).abs().mean()


def is_paired(t1, t2, w, h):
    # threshold found through experimentation
    return get_mad(t1, t2, w, h) < 0.1


class Runner:
    def predict(self, first, second, scale=0.4):
        t1 = time.time()
        first_image = Image.open(first)
        second_image = Image.open(second)

        original_tensor, crop_tensor = get_original_and_crop_tensors(first_image, second_image, scale=scale)

        predicted_loc = predict_crop_coords(original_tensor, crop_tensor, scale=scale)

        full_original_tensor, full_crop_tensor = get_original_and_crop_tensors(
                first_image, second_image, scale=1.0)

        loc = fine_tune_prediction(full_original_tensor, full_crop_tensor, predicted_loc)
        if is_paired(full_original_tensor, full_crop_tensor, *loc):
            if is_swapped(first_image, second_image):
                print(f"{first} is a cropped version of {second}")
                print(f"The top left corner of {first} can be found in {second} at about ({loc})")
            else:
                print(f"{second} is a cropped version of {first}")
                print(f"The top left corner of {second} can be found in {first} at about ({loc})")
        else:
            print("The images are not cropped versions of one another")

        t2 = time.time()
        print(f"Took: {(t2 - t1)*1000:3.1f}ms")


if __name__ == '__main__':
    fire.Fire(Runner)
