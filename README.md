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

Using a naive LogisticRegression classifier, the model had an average cross validation accuracy score of `0.99515612489992`.

Testing this on a holdout dataset of 10 cat images and 10 dog images, it was able to classify all of the holdouts correctly.




