https://research.nvidia.com/publication/2020-07_FLIP

I am not the author of this algorithm. I only optimized it slightly (it can be further improved)
This repository uses the same base file and modifies only minimally the function calls.

Python bindings are available and expose a single function: `computeFlip`.
A Python implementation is also available.

Please see testpy.py for an example.
For performance reasons, the images to compare have to be in one large vector, and width and height have to be supplied.
This allows a very straightforward and fast `memcpy`.

Many loops have been parallelized and some merged.
This implementation heavily relies on `#pragma omp single` to allow for concurrent tasks

## Python Bindings and Python implementation:
The NVidia team will present new code for HDR and LDR soon, I don't know if my implementation will become obsolete. In any case, it fits my use case well enough.

There was both a Python and a C++ implementation. Surprisingly, they were both as bad in terms of performance (~6s per image in 1200x700).

I made both an optimized python and an optimized C++ version. The optimized C++ is faster but its python bindings are slower. 

--- 
# INSTALLATION for C++:
You will need C++ 17, openMP support, `apt-get install pybind11`, or build pybind yourself
```
mkdir build
cd build
cmake ..
make -j
```

There will be a warning in `commandline.h`. This is not my code, and as long as it works, I won't fix it.

The output for C++ binary is `./build/flipo`.<br>
The output for C++ library is `./build/bin/flip.so`.<br>
The output for Python library is `./build/bin/pyflip.cpython[...]`

After building, you can run `testpy.py` in a notebook or anything that allows you to see matplotlib figures.

---
## Performance:

I tried it on my pc with an i7-7820HK.
My tests yielded ~6seconds for a 720x1200 image for the original C++ implementation.

My C++ implementation runs in around 0.3s, making it 20 times faster. The optimized python implementation runs in 0.5s. Sadly, the python bindings are actually slower. At the very end of this document, several further improvements are suggested. I expect that it should be possible to optimize it down to a few dozens of milliseconds.

---
## Details:

FLIP relies on several 2D convolutions:
- Edge detection and point detection on greyscale image once regular and once transposed, uses the channels to do only one call per image
- preprocess requires one full channel convolution per image

There are many loops scanning the whole array, sometimes for no other reason than clarity.

Because clarity is great, but speed is better, I removed the HuntAdjustment call and integrated it with other computations.

My main contribution here is the use of 1D convolutions.
Indeed, FLIP uses separable kernels, all computed from gaussians.

A 2D gaussian filter is trivially converted into the generating vector by simply setting $y=0$ and normalizing, and this is done for the detection filters. This is not so easy for the other filter they use:

The spatial filter generation uses:
$$
    f(x,y)=a_1e^{-a_2(x^2+y^2)}+b_1e^{-b_2(x^2+y^2)}
$$

As a sum of gaussians, we need to find a way to create two corresponding vectors.
$f(x,y)$ is used to create an $NxN$ matrix $F$ that will be the convolution kernel:
$$
    F_{i,j}=f(i-r,j-r),
$$

with  $N=2r+1$ and $i,j$ spanning $[-r,r]$.
Note that $F$ is then normalized into $\tilde{F}$:
$$
\tilde{F}=F/S_F\\
S_F=\sum_i\sum_j F_{i,j}
$$
This notation means every element of the matrix $F$ is divided by $S_F$.

We construct the correct vectors $g_1$ and $g_2$:
$$
g_1(x)=\sqrt{a_1}e^{-a_2x^2}\\
g_2(x)=\sqrt{b_1}e^{-b_2x^2}.
$$

They have with the following property:
$$
F=g_1\otimes g_1+g_2\otimes g_2\\
F_{i,j}=g_1(j) * g_1(i) + g_2(j) * g_2(i),
$$
with the crossed circle denoting the outer product.

Because $F$ is normalized, we need to find the normalization for our vectors too, characterized by:

$$
\tilde{F}=\tilde{G}= \frac{g_1\otimes g_1+g_2\otimes g_2}{S_G}.
$$

By identification, we trivially have:
$$
S_G=S_F
$$

We write the normalization:
$$
\tilde{g_1}\otimes \tilde{g_1} =\frac{g_1\otimes g_1}{S(g_1)^2+S(g_2)^2}\\
\texttt{with:} \tilde{g_1}=\alpha_1 g_1\\
\alpha g_1\otimes \alpha g_1 =\frac{g_1\otimes g_1}{S(g_1)^2+S(g_2)^2}\\
\alpha^2 g_1\otimes g_1 =\frac{g_1\otimes g_1}{S(g_1)^2+S(g_2)^2}\\
\texttt{by identification: } \alpha=\frac{1}{\sqrt{S(g_1)^2+S(g_2)^2}}\\
$$

So, we compute our vectors this way to create a separable convolution of the sum of gaussians.

Here in pseudocode for the convolution:
```
c1=convolve1D(image, alpha*g_1, alpha*g_1)
c2=convolve1D(image, alpha*g_2, alpha*g_2)
result=c1+c2.
```
--- 
## Further improvement:
We actually waste one convolution 4 times due to the use of the color3 structure in the `computeFeatureDifference` function. Indeed, the third layer is set to zero and is not actually used, but still computed.

The biggest problem of this implementation is the array of structure (AoS) which is quite inefficient, and hard to optimize outside of simple parallelism. Vectorization should be done on SoA, and the improvement should be quite huge.

Breaking the current layout and merging all the loops and operations together when possible should have a big impact and would avoid to compute twice the same thing. Creating custom vectorized functions for all these loops is also interesting obviously.