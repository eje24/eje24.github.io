---
layout: post
title: CNN Garden
author: Ezra
katex: True
---

Convolutional neural networks (CNN's) are ubiquitous in machine learning today. This popularity stems from one remarkable feature of CNN's: translational equivariance. Basically, this means that if I first translate an image and then pass it through a CNN, I get the same result as if I had first passed it through the CNN, and then translated it. It would be a mistake to conclude that this is the whole picture. Concealed by the familiar scenery of flat, two-dimensional, pixelated images lies a siginficantly more general framework, of which "conventional" CNN's are only a special case. Consider:

- The key word in in *translational equivariance* is *translational*, and the set of translations is but one of many subgroups of the group of symmetries of the plane.
- **Convolutional** neural networks are commonly implemented not as convolutions, but as *cross-correlations*. Cross-correlations are taken with respect to some symmetry group, and produce functions defined on this same symmetry group. The fact that conventional CNN's take in functions on the plane and produce functions on the plane is just a coincidence: the translation group has the "same structure" as the plane (think about this!). 
- We can achieve more exotic equivariance properties by taking cross-correlations with respect to more complex symmetry groups (think: rotations). With this perspective, we can think of a conventional CNN as lifting a function on $\mathbb{Z}^2$ to a function on $\mathbb{T}(2)$ (the translation group of $\mathbb{Z}^2$). In general, cross-correlation lifts a *signal* from a base space to a desired symmetry group.
- Why limit ourselves to the plane? Just as conventional CNN's are based on translational cross-correlation on the plane, we may similarly consider translational cross-correlation in higher dimensional space, rotational cross-correlation on the plane, and even rotational cross-correlation on the sphere (not flat), to name a few examples. Such spaces on which groups act are known as *homogeneous spaces*.
- Groups can act not just on a "base space", but also on channels (think: RGB in images) for more expressive CNN's. The combination of a base space (the plane) and channel space can be realized as a single space (the associated bundle) on which a symmetry group might act. Group action on channel space can then be extended naturally to a group action on the combined space (the induced representation).

My idea was to follow the literature and present successively more general views of CNN's, discussing each big idea in turn. The goal is for it to read like walking through a garden of ideas, hence the title. Until the full draft is written out in this blog, the garden will take the form of this [linked PDF](/assets/cnn_garden.pdf). 

### Notes

These notes were compiled as part of my final project for [Symmetry for Machine Learning](https://symm4ml.mit.edu/symm4ml/info), taught by Professor Tess Smidt. Professor Smidt is also one of the co-founders of the popular [e3nn](https://docs.e3nn.org/en/latest/index.html) library for E(3)-equivariant neural networks. 
