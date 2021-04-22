#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>    

#define _USE_MATH_DEFINES
#include <cmath>

///////////////////////////////////////////////////////////////////////////////
// single precision float version
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSeparable(
    float* in, float* out, int dataSizeX, int dataSizeY, float* kernelX, int kSizeX, float* kernelY, int kSizeY)
{
    int i, j, k, m, n;
    float *tmp, *sum;                // intermediate data buffer
    float *inPtr, *outPtr;           // working pointers
    float *tmpPtr, *tmpPtr2;         // working pointers
    int kCenter, kOffset, endIndex;  // kernel indice

    // check validity of params
    if (!in || !out || !kernelX || !kernelY)
        return false;
    if (dataSizeX <= 0 || kSizeX <= 0)
        return false;

    // allocate temp storage to keep intermediate result
    tmp = new float[dataSizeX * dataSizeY];
    if (!tmp)
        return false;  // memory allocation error

    // store accumulated sum
    sum = new float[dataSizeX];
    if (!sum)
        return false;  // memory allocation error

    // covolve horizontal direction ///////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeX >> 1;           // center index of kernel array
    endIndex = dataSizeX - kCenter;  // index for full kernel convolution

    // init working pointers
    inPtr = in;
    tmpPtr = tmp;  // store intermediate results from 1D horizontal convolution

    // start horizontal convolution (x-direction)
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        kOffset = 0;  // starting index of partial kernel varies for each sample

        // COLUMN FROM index=0 TO index=kCenter-1
        for (j = 0; j < kCenter; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kCenter + kOffset, m = 0; k >= 0; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++tmpPtr;   // next output
            ++kOffset;  // increase starting index of kernel
        }

        // COLUMN FROM index=kCenter TO index=(dataSizeX-kCenter-1)
        for (j = kCenter; j < endIndex; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulate

            for (k = kSizeX - 1, m = 0; k >= 0; --k, ++m)  // full kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;   // next input
            ++tmpPtr;  // next output
        }

        kOffset = 1;  // ending index of partial kernel varies for each sample

        // COLUMN FROM index=(dataSizeX-kCenter) TO index=(dataSizeX-1)
        for (j = endIndex; j < dataSizeX; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kSizeX - 1, m = 0; k >= kOffset; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;    // next input
            ++tmpPtr;   // next output
            ++kOffset;  // increase ending index of partial kernel
        }

        inPtr += kCenter;  // next row
    }
    // END OF HORIZONTAL CONVOLUTION //////////////////////

    // start vertical direction ///////////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeY >> 1;           // center index of vertical kernel
    endIndex = dataSizeY - kCenter;  // index where full kernel convolution should stop

    // set working pointers
    tmpPtr = tmpPtr2 = tmp;
    outPtr = out;

    // clear out array before accumulation
    for (i = 0; i < dataSizeX; ++i)
        sum[i] = 0;

    // start to convolve vertical direction (y-direction)

    // ROW FROM index=0 TO index=(kCenter-1)
    kOffset = 0;  // starting index of partial kernel varies for each sample
    for (i = 0; i < kCenter; ++i)
    {
        for (k = kCenter + kOffset; k >= 0; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output array
            sum[n] = 0;        // reset to zero for next summing
            ++outPtr;          // next element of output
        }

        tmpPtr = tmpPtr2;  // reset input pointer
        ++kOffset;         // increase starting index of kernel
    }

    // ROW FROM index=kCenter TO index=(dataSizeY-kCenter-1)
    for (i = kCenter; i < endIndex; ++i)
    {
        for (k = kSizeY - 1; k >= 0; --k)  // convolve with full kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output buffer
            sum[n] = 0;        // reset before next summing
            ++outPtr;          // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;
    }

    // ROW FROM index=(dataSizeY-kCenter) TO index=(dataSizeY-1)
    kOffset = 1;  // ending index of partial kernel varies for each sample
    for (i = endIndex; i < dataSizeY; ++i)
    {
        for (k = kSizeY - 1; k >= kOffset; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output array
            sum[n] = 0;        // reset to 0 for next sum
            ++outPtr;          // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;  // next input
        ++kOffset;         // increase ending index of kernel
    }
    // END OF VERTICAL CONVOLUTION ////////////////////////

    // deallocate temp buffers
    delete[] tmp;
    delete[] sum;
    return true;
}
class color3
{
public:
    float x,y,z;
    color3() { x = 0.0f; y = 0.0f; z = 0.0f; }
    color3(const float v) { x = v; y = v; z = v; }
    color3(const float _x, const float _y, const float _z) { x = _x; y = _y; z = _z; }
    color3 operator+(const color3& c){ return {x+c.x,y+c.y,z+c.z}; }
    color3 operator+=(const color3& c){ return *this={x+c.x,y+c.y,z+c.z}; }
    color3 operator*(const color3& c){ return {x*c.x,y*c.y,z*c.z}; }
    color3 operator/(const color3& c){ return {x/c.x,y/c.y,z/c.z}; }
    color3 operator/=(const color3& c){ return *this={x/c.x,y/c.y,z/c.z}; }
    bool operator==(const color3& c){ return (x == c.x && y == c.y && z==c.z); }
    color3& abs() { x = fabsf(x); y = fabsf(y); z = fabsf(z); return *this; }
    color3& max(const float value) { x = std::max(x, value); y = std::max(y, value); z = std::max(z, value); return *this; }
    color3& min(const float value) { x = std::min(x, value); y = std::min(y, value); z = std::min(z, value); return *this; }
    color3& min(const color3& value) { x = std::min(x, value.x); y = std::min(y, value.y);  z = std::min(z, value.z); return *this; }
};
///////////////////////////////////////////////////////////////////////////////
// single precision float version
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSeparable3Channels(
    color3* in, color3* out, int dataSizeX, int dataSizeY, color3* kernelX, int kSizeX, color3* kernelY, int kSizeY)
{
    int i, j, k, m, n;
    color3 *tmp, *sum;                // intermediate data buffer
    color3 *inPtr, *outPtr;           // working pointers
    color3 *tmpPtr, *tmpPtr2;         // working pointers
    int kCenter, kOffset, endIndex;  // kernel indice

    // check validity of params
    if (!in || !out || !kernelX || !kernelY)
        return false;
    if (dataSizeX <= 0 || kSizeX <= 0)
        return false;

    // allocate temp storage to keep intermediate result
    tmp = new color3[dataSizeX * dataSizeY];
    if (!tmp)
        return false;  // memory allocation error

    // store accumulated sum
    sum = new color3[dataSizeX];
    if (!sum)
        return false;  // memory allocation error

    // covolve horizontal direction ///////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeX >> 1;           // center index of kernel array
    endIndex = dataSizeX - kCenter;  // index for full kernel convolution

    // init working pointers
    inPtr = in;
    tmpPtr = tmp;  // store intermediate results from 1D horizontal convolution

    // start horizontal convolution (x-direction)
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        kOffset = 0;  // starting index of partial kernel varies for each sample

        // COLUMN FROM index=0 TO index=kCenter-1
        for (j = 0; j < kCenter; ++j)
        {
            *tmpPtr = .0f;  // init to 0 before accumulation

            for (k = kCenter + kOffset, m = 0; k >= 0; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++tmpPtr;   // next output
            ++kOffset;  // increase starting index of kernel
        }

        // COLUMN FROM index=kCenter TO index=(dataSizeX-kCenter-1)
        for (j = kCenter; j < endIndex; ++j)
        {
            *tmpPtr = 0.f;  // init to 0 before accumulate

            for (k = kSizeX - 1, m = 0; k >= 0; --k, ++m)  // full kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;   // next input
            ++tmpPtr;  // next output
        }

        kOffset = 1;  // ending index of partial kernel varies for each sample

        // COLUMN FROM index=(dataSizeX-kCenter) TO index=(dataSizeX-1)
        for (j = endIndex; j < dataSizeX; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kSizeX - 1, m = 0; k >= kOffset; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;    // next input
            ++tmpPtr;   // next output
            ++kOffset;  // increase ending index of partial kernel
        }

        inPtr += kCenter;  // next row
    }
    // END OF HORIZONTAL CONVOLUTION //////////////////////

    // start vertical direction ///////////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeY >> 1;           // center index of vertical kernel
    endIndex = dataSizeY - kCenter;  // index where full kernel convolution should stop

    // set working pointers
    tmpPtr = tmpPtr2 = tmp;
    outPtr = out;

    // clear out array before accumulation
    for (i = 0; i < dataSizeX; ++i)
        sum[i] = 0.;

    // start to convolve vertical direction (y-direction)

    // ROW FROM index=0 TO index=(kCenter-1)
    kOffset = 0;  // starting index of partial kernel varies for each sample
    for (i = 0; i < kCenter; ++i)
    {
        for (k = kCenter + kOffset; k >= 0; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output array
            sum[n] = 0;        // reset to zero for next summing
            ++outPtr;          // next element of output
        }

        tmpPtr = tmpPtr2;  // reset input pointer
        ++kOffset;         // increase starting index of kernel
    }

    // ROW FROM index=kCenter TO index=(dataSizeY-kCenter-1)
    for (i = kCenter; i < endIndex; ++i)
    {
        for (k = kSizeY - 1; k >= 0; --k)  // convolve with full kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output buffer
            sum[n] = 0;        // reset before next summing
            ++outPtr;          // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;
    }

    // ROW FROM index=(dataSizeY-kCenter) TO index=(dataSizeY-1)
    kOffset = 1;  // ending index of partial kernel varies for each sample
    for (i = endIndex; i < dataSizeY; ++i)
    {
        for (k = kSizeY - 1; k >= kOffset; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output array
            sum[n] = 0;        // reset to 0 for next sum
            ++outPtr;          // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;  // next input
        ++kOffset;         // increase ending index of kernel
    }
    // END OF VERTICAL CONVOLUTION ////////////////////////

    // deallocate temp buffers
    delete[] tmp;
    delete[] sum;
    return true;
}