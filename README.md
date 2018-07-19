# Crop Detection via Deep Learning

The task is to write an application that takes two images as input, and responds with:

1. If one of the images is a cropped version of the other.
2. If yes, what are the coordinates of the top left corner of the cropped image in the uncropped image.

The application should work well on JPEGs with some lossy compression enabled.

## Summary of results

I attempted to train a neural network to do this task, but wasn't able to achieve anything close
to good accuracy.  I'll first describe how to operate the (admitedly poor) system.  Then I'll outline the approach.
Finally I'll go into why I think the attempt failed and what could be done better.

## Running the console application

To run the application, have python 3.6+ installed as well as tox and then run something like this:

```
$ tox -e python detect.py predict snowy-uncropped.jpg snowy-small.jpg
```

Output will look something like this (ignoring the stuff from tox):

```
snowy-small.jpg is a cropped version of snowy-uncropped.jpg
The top left corner of snowy-small.jpg can be found in snowy-uncropped.jpg at about (428, 334)
```

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
    - We will train just our newly added layers first.
    - We can then unfreeze the rest of the model and train the entire net (though I saw no improvement doing this).

  * With a trained network, we'll build a console application that loads the images, runs them through our network and
    displays the predictions.

#### Benefits of this approach

* It has the potential to scale very nicely as it is a single feed foward model of reasonable size.
* The training data can be as big as you want, you just need images, of any variety.  No hand labeling is required.
* It should be very robust to differences in jpg encoding qualities, in fact we can make sure it is by construction of
  our training data.

#### Concerns with this approach

* It may not be easy to train to high accuracy
* It could start considering similar (but distinct) images as cropped versions of each other if we aren't careful with
  our training/validation dataset.
* It could be considered overkill for the simplicity of the problem.

## Results and Retrospective

Before we get into the discussion, a little vocabulary so we are on the same page.

  * paired - Two images are `paired` if one is a cropped version of the other.
  * swapped - Two images are `swapped` if they are `paired`, but the first image is not the uncropped version.

#### Issues with the Dataset

Just building the dataset was full of interesting challenges.
There is a lot of ambiguity in the original task statement.
For example, how similar/different can the two images be before they're not considered `paired`?
Would a small rotation or contrast enhancement disqualify a pairing?
When creating the dataset, I chose to assume that the two images would be identical, except for the cropping.
Therefore, for training non-pairing training samples, the two images are completely distinct.
This turned out to make the model poor at correctly identifying `paired` images if they happened to be somewhat similar (i.e. same subject but different angle).

So I also created a 'hard_mode' flag for the dataset where the images in non-pairing training samples were just the horizontal flip of one another.
This ensures that they share the same statistics, however this 'hard_mode' was too hard for any network architecture I tried to train.

Another dataset related ambiguity is, what aspect ratios will we allow in the cropped versions?
I didn't restrict the aspect ratio and I think that was a mistake.
The models were able to very easily able to predict if images were `swapped`, but couldn't predict `paired` very well.
I believe the network simply learned that extreme aspect ratios were always the cropped version of the images.

#### Issues with the Model

I tried a number of variations of the model committed here.
All of them were built off of the resnet backbone that was pretrained as an image classifier on the imagenet dataset.
This might be a bad idea, for a number of reasons.
For one thing, the training process included data augmentation such as cropping, rotating, and flipping the images.
That means that the network has learned internal representations that are robust to those kinds of augmentations... the exact kind of augmentation we are hoping to detect here.
Another reason imagenet pretrained networks might not be great, is that they're all trained on low resolution 224x224px images.
If the original image is large, and cropping is small, it's likely that there isn't enough information for the network to determine they're `paired`, due to excessive downsampling.
