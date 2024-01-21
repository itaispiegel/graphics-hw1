## Introduction
This repo contains our work on the first project in the course about graphics, computer vision and image processing at TAU during the spring semester of 2023.
In this project, we implemented in Python two algorithms: [GrabCut](https://en.wikipedia.org/wiki/GrabCut) and [Poisson Blending](https://en.wikipedia.org/wiki/Gradient-domain_image_processing).

### GrabCut
This is a computer vision algorithm, which performs image segmentation. Its input is an image, and a rectangle marked on it, and the algorithm separates the target object of the image from its background.
The algorithm uses Gaussian Mixture Models to estimate the color distribution of the target object and the background and runs a graph cut optimization on the image until convergence.

### Poisson Blending
This is an image-processing algorithm that blends two images. It uses the Poisson equation to blend the images smoothly and seamlessly.
