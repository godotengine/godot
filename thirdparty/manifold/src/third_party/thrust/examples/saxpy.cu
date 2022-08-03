#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <algorithm>

// This example illustrates how to implement the SAXPY
// operation (Y[i] = a * X[i] + Y[i]) using Thrust. 
// The saxpy_slow function demonstrates the most
// straightforward implementation using a temporary
// array and two separate transformations, one with
// multiplies and one with plus.  The saxpy_fast function
// implements the operation with a single transformation
// and represents "best practice".

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::device_vector<float> temp(X.size());
   
    // temp <- A
    thrust::fill(temp.begin(), temp.end(), A);
    
    // temp <- A * X
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

    // Y <- A * X + Y
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

int main(void)
{
    // initialize host arrays
    float x[4] = {1.0, 1.0, 1.0, 1.0};
    float y[4] = {1.0, 2.0, 3.0, 4.0};

    {
        // transfer to device
        thrust::device_vector<float> X(x, x + 4);
        thrust::device_vector<float> Y(y, y + 4);

        // slow method
        saxpy_slow(2.0, X, Y);
    }

    {
        // transfer to device
        thrust::device_vector<float> X(x, x + 4);
        thrust::device_vector<float> Y(y, y + 4);

        // fast method
        saxpy_fast(2.0, X, Y);
    }
    
    return 0;
}

