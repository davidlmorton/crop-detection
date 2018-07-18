# Crop Detection via Deep Learning

This is a little project to show how you can detect if an image is a cropped version of another using Deep Learning.

## The approach

The basic approach is to fine tune a resnet that has been pretrained on imagenet data to do the task for us.
  * First we'll need to construct our training and validation datasets.
    - The inputs will be 6xNxN tensors where the 6 channels are the RGB of each of the two images.
      * regardless of the input image sizes, we'll transform them to NxN
    - The targets will be:
      1) is_cropped_pair (two element tensor of floats) - this will be [1, 0] if the two images are a pair, [0, 1]
      otherwise.
      2) is_swapped (two element tensor of floats) - this will be [1, 0] if the first image is the original (and
      the second one is the cropped version) and [0, 1] otherwise.
      3) top_left_coords (two element tensor of floats) - this will be [x, y] where 0 <= x <= 1 and similarly for y.
      It represents the x, y coordinates of the cropped image within the original image.

  * Then we'll fine-tune a resnet
    - We will load in the weights from pretraining on imagenet.
    - We will remove the last two layers (the pooling and fully connected layers) and replace them with our own
      classifiers and regressors.
    - We will train just our newly added layers first, then unfreeze the rest of the model and train the entire net.

  * With a trained network, we'll build a console application that loads the images, runs them through our network and
    displays the predictions.

## Benefits of this approach

* It has the potential to scale very nicely as it is a single feed foward model of reasonable size.
* The training data can be as big as you want, you just need images, of any variety.  No hand labeling is required.
* It should be very robust to differences in jpg encoding qualities, in fact we can make sure it is by construction of
  our training data.

## Concerns with this approach

* It may not be easy to train to high accuracy
* It could start considering similar (but distinct) images as cropped versions of each other if we aren't careful with
  our training/validation dataset.
* It could be considerd overkill for the simplicity of the toy problem.
