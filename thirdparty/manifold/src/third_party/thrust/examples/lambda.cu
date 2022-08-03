#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <iostream>

// This example demonstrates the use of placeholders to implement
// the SAXPY operation (i.e. Y[i] = a * X[i] + Y[i]).
//
// Placeholders enable developers to write concise inline expressions
// instead of full functors for many simple operations.  For example,
// the placeholder expression "_1 + _2" means to add the first argument,
// represented by _1, to the second argument, represented by _2.
// The names _1, _2, _3, _4 ... _10 represent the first ten arguments
// to the function.
// 
// In this example, the placeholder expression "a * _1 + _2" is used
// to implement the SAXPY operation.  Note that the placeholder 
// implementation is considerably shorter and written inline. 


// allows us to use "_1" instead of "thrust::placeholders::_1"
using namespace thrust::placeholders;


// implementing SAXPY with a functor is cumbersome and verbose
struct saxpy_functor
  : public thrust::binary_function<float, float, float>
{
  float a;
  
  saxpy_functor(float a) : a(a) {}

  __host__ __device__
  float operator()(float x, float y)
  {
    return a * x + y;
  }
};



int main(void)
{
  // input data
  float a = 2.0f;
  float x[4] = {1, 2, 3, 4};
  float y[4] = {1, 1, 1, 1};
   
  // SAXPY implemented with a functor (function object)
  {
    thrust::device_vector<float> X(x, x + 4);
    thrust::device_vector<float> Y(y, y + 4);

    thrust::transform(X.begin(), X.end(),  // input range #1
                      Y.begin(),           // input range #2
                      Y.begin(),           // output range
                      saxpy_functor(a));   // functor
    
    std::cout << "SAXPY (functor method)" << std::endl;
    for (size_t i = 0; i < 4; i++)
      std::cout << a << " * " << x[i] << " + " << y[i] << " = " << Y[i] << std::endl;
  }

  // SAXPY implemented with a placeholders
  {
    thrust::device_vector<float> X(x, x + 4);
    thrust::device_vector<float> Y(y, y + 4);

    thrust::transform(X.begin(), X.end(),  // input range #1
                      Y.begin(),           // input range #2
                      Y.begin(),           // output range
                      a * _1 + _2);        // placeholder expression

    std::cout << "SAXPY (placeholder method)" << std::endl;
    for (size_t i = 0; i < 4; i++)
      std::cout << a << " * " << x[i] << " + " << y[i] << " = " << Y[i] << std::endl;
  }

  return 0;
}

