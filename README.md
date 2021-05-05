https://research.nvidia.com/publication/2020-07_FLIP

! This is not the new HDR variant !

I am not the author of this algorithm. I only optimized it slightly (it can be further improved)
This repository uses the same base file and modifies only minimally the function calls.

Python bindings are available and expose a single function: `computeFlip`.
A Python implementation is also available.

Please see testpy.py for an example.

Many loops have been parallelized and some merged.
This implementation heavily relies on `#pragma omp single` to allow for concurrent tasks

## Python Bindings and Python implementation:
The NVidia team will present new code for HDR and LDR soon, I don't know if my implementation will become obsolete. In any case, it fits my use case well enough.

There was both a Python and a C++ implementation. Surprisingly, they were both as bad in terms of performance (~6s per image in 1200x700 on my machine).

I made both an optimized python and an optimized C++ version. The optimized C++ is faster but its python bindings are slower. 

## Using the Python version:
Simply copy the flipInPython directory somewhere you want to use it, then use the following import.

``̀ 
from flipInPython.flip_opti import compute_flip_opti
```

Please see in the pytest.py the example for each solution. Note that the original python flip implementation is present too, for comparison.
--- 
# INSTALLATION for C++:
You will need C++ 17 (i.e. a recent enough compiler version), openMP support, `apt-get install pybind11`, or build pybind yourself
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

The mathematic details are in the Math.pdf file.

--- 
## Further improvement:
We actually waste one convolution 4 times due to the use of the color3 structure in the `computeFeatureDifference` function. Indeed, the third layer is set to zero and is not actually used, but still computed.

The biggest problem of this implementation is the array of structure (AoS) which is quite inefficient, and hard to optimize outside of simple parallelism. Vectorization should be done on SoA, and the improvement should be quite huge.

Breaking the current layout and merging all the loops and operations together when possible should have a big impact and would avoid to compute twice the same thing. Creating custom vectorized functions for all these loops is also interesting obviously.
