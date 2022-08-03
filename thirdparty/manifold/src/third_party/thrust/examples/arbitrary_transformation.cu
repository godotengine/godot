#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)
#include <thrust/zip_function.h>
#endif // >= C++11

// This example shows how to implement an arbitrary transformation of
// the form output[i] = F(first[i], second[i], third[i], ... ).
// In this example, we use a function with 3 inputs and 1 output.
//
// Iterators for all four vectors (3 inputs + 1 output) are "zipped"
// into a single sequence of tuples with the zip_iterator.
//  
// The arbitrary_functor receives a tuple that contains four elements,
// which are references to values in each of the four sequences. When we
// access the tuple 't' with the get() function,
//      get<0>(t) returns a reference to A[i],
//      get<1>(t) returns a reference to B[i],
//      get<2>(t) returns a reference to C[i],
//      get<3>(t) returns a reference to D[i].
//
// In this example, we can implement the transformation,
//      D[i] = A[i] + B[i] * C[i];
// by invoking arbitrary_functor() on each of the tuples using for_each.
//
// If we are using a functor that is not designed for zip iterators by taking a
// tuple instead of individual arguments we can adapt this function using the
// zip_function adaptor (C++11 only).
//
// Note that we could extend this example to implement functions with an
// arbitrary number of input arguments by zipping more sequence together.
// With the same approach we can have multiple *output* sequences, if we 
// wanted to implement something like
//      D[i] = A[i] + B[i] * C[i];
//      E[i] = A[i] + B[i] + C[i];
//
// The possibilities are endless! :)

struct arbitrary_functor1
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
    }
};

#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)
struct arbitrary_functor2
{
    __host__ __device__
    void operator()(const float& a, const float& b, const float& c, float& d)
    {
        // D[i] = A[i] + B[i] * C[i];
        d = a + b * c;
    }
};
#endif // >= C++11

int main(void)
{
    // allocate storage
    thrust::device_vector<float> A(5);
    thrust::device_vector<float> B(5);
    thrust::device_vector<float> C(5);
    thrust::device_vector<float> D1(5);

    // initialize input vectors
    A[0] = 3;  B[0] = 6;  C[0] = 2; 
    A[1] = 4;  B[1] = 7;  C[1] = 5; 
    A[2] = 0;  B[2] = 2;  C[2] = 7; 
    A[3] = 8;  B[3] = 1;  C[3] = 4; 
    A[4] = 2;  B[4] = 8;  C[4] = 3; 

    // apply the transformation
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D1.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end(),   C.end(),   D1.end())),
                     arbitrary_functor1());

    // print the output
    std::cout << "Tuple functor" << std::endl;
    for(int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D1[i] << std::endl;

    // apply the transformation using zip_function
#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)
    thrust::device_vector<float> D2(5);
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D2.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end(),   C.end(),   D2.end())),
                     thrust::make_zip_function(arbitrary_functor2()));

    // print the output
    std::cout << "N-ary functor" << std::endl;
    for(int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D2[i] << std::endl;
#endif // >= C++11
}

