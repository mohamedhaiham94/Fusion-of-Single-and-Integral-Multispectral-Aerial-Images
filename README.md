Source Code and Data: Fusion of Single and Integral Multispectral Aerial Images
====================================================================

This is a pytorch implementation of Youssef, M. and Bimber, O., 2023. Fusion of Single and Integral Multispectral Aerial Images. arXiv preprint arXiv:2311.17515. (https://arxiv.org/abs/2311.17515)

**Abstract:**
We present a novel hybrid (model- and learning-based) architecture for fusing the most significant features from conventional aerial images and integral aerial images that result from synthetic aperture sensing for removing occlusion caused by dense vegetation. It combines the environment’s spatial references with features of unoccuded targets. Our method out-beats the state-of-the-art, does not require manually tuned parameters, can be extended to an arbitrary number and combinations of spectral channels, and is reconfigurable to address different use-cases. 

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