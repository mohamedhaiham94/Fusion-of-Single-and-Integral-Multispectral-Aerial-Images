Source Code and Data: Fusion of Single and Integral Multispectral Aerial Images
====================================================================

This is a pytorch implementation of Youssef, M. and Bimber, O., 2023. Fusion of Single and Integral Multispectral Aerial Images. Remote Sens. 2024, 16, 673. (https://doi.org/10.3390/rs16040673)

**Abstract:**
An adequate fusion of the most significant salient information from multiple input channels
is essential for many aerial imaging tasks. While multispectral recordings reveal features in various
spectral ranges, synthetic aperture sensing makes occluded features visible. We present a first and
hybrid (model- and learning-based) architecture for fusing the most significant features from conventional aerial images with the ones from integral aerial images that are the result of synthetic aperture
sensing for removing occlusion. It combines the environment’s spatial references with features of
unoccluded targets that would normally be hidden by dense vegetation. Our method outperforms
state-of-the-art two-channel and multi-channel fusion approaches visually and quantitatively in
common metrics, such as mutual information, visual information fidelity, and peak signal-to-noise
ratio. The proposed model does not require manually tuned parameters, can be extended to an
arbitrary number and arbitrary combinations of spectral channels, and is reconfigurable for addressing different use cases. We demonstrate examples for search and rescue, wildfire detection, and
wildlife observation.

**Authors:** Mohamed Youssef and Oliver Bimber


# Installation
* Download and extract.
* Install Python (>3) and the required packages (common data science packages such as numpy, pandas, opencv2, ...). 
* Install PyTorch. 
* Run 'test.py' and the generated fused image will be saved to results folder, similar if you want to run alhpa-blending or colormap fusion.

# License:
* Data: Creative Commons Attribution 4.0 International
* Code: MIT License (license text below)
    
        Copyright 2020 Johannes Kepler University Linz

        Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.