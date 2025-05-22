# Cats vs Dogs Image Classification with Scikit-Learn

Image classification has been a problem for the Deep Learning tools to solve.

I recently decided to try to solve this problem with Scikit-Learn.  I have used scikit-learn with image processing in the past but never for image classification.

I came upon the following resources:

## YouTube

https://youtu.be/lzXKsY3bANw?si=OX2WVuTblQTJ84sO

## Github

https://github.com/probabl-ai/youtube-appendix/blob/main/01-sklearn-image/notebook.ipynb

## PyPI

https://pypi.org/project/embetter/

which has some invaluable embeddings specifically designed for the scikit-learn ecosystem.

After walking through the resources, I decided to try it on the `Cats vs Dogs` ( or is it `Dogs vs Cats`) dataset which I downloaded years ago.  I had worked through this dataset using Tensorflow/Keras back in the day.

To my surprise, scikit-learn with the embetter image embeddings did surprisingly well.

Using a naive LogisticRegression classifier, the model had an average cross validation accuracy score of `0.9956338874424192`.

Testing this on a holdout dataset of 20 cat images and 20 dog images, it was able to classify all of the holdouts correctly.

### CLIP ( Contrastive Language–Image Pretraining ) Background Information

**What is a CLIP encoder?**

CLIP stands for Contrastive Language–Image Pretraining. It’s a model made by OpenAI that can understand both images and text, and match them to each other. So, if you show CLIP a picture of a dog and the word “dog,” it will know they go together.

CLIP uses two main parts:

An image encoder – that looks at an image and turns it into numbers (called an embedding).
A text encoder – that does the same thing for text.

**What does it mean to "encode" an image?**

When we encode an image, we’re turning it into a list of numbers that represents the important stuff about the image—kind of like its fingerprint.

This list of numbers is called a vector or embedding. It's like a summary of the image that CLIP can use to compare it with other images or text.

Think of it like this:

Imagine you take a picture of a cat.
The CLIP image encoder looks at that picture and gives you a list of, say, 512 numbers.
These numbers don't look like much to us, but to the model, they capture key features like shapes, colors, and what objects are in the image.

**What does the output represent?**

The output is a list of numbers (a vector)—for example:

[0.12, -0.58, 0.33, ..., 0.05]  ← 512 numbers
Each number in that list represents a different feature or pattern in the image. Alone, they don’t mean much to humans, but together, they help the computer know what’s in the image.

**For example:**

Similar images (like two pictures of dogs) will have similar vectors.
Different images (like a dog vs. a car) will have different vectors.

**Why is this useful?**

Because once an image is a vector:

You can compare it with text vectors (like the word "cat" or "dog").
You can search for images that are similar.
You can do things like captioning, clustering, or even generating images based on text.

**Summary**

A CLIP encoder turns images into numbers (vectors).
These numbers summarize what’s in the image.
They help computers understand and compare images and text—even if the computer has never seen the exact image before.


**Here’s how it works:**

When you pass an image to CLIP:

The image is loaded (usually as pixels).
It’s resized and preprocessed (to match what CLIP expects—like 224×224 pixels).
Then it's passed through the image encoder (like a modified ResNet or Vision Transformer).
The encoder outputs a vector of numbers (the embedding) that represents only the visual content of the image.



