#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

#include <iostream>
#include <iterator>

// This example computes a run-length code [1] for an array of characters.
//
// [1] http://en.wikipedia.org/wiki/Run-length_encoding


int main(void)
{
    // input data on the host
    const char data[] = "aaabbbbbcddeeeeeeeeeff";

    const size_t N = (sizeof(data) / sizeof(char)) - 1;

    // copy input data to the device
    thrust::device_vector<char> input(data, data + N);

    // allocate storage for output data and run lengths
    thrust::device_vector<char> output(N);
    thrust::device_vector<int>  lengths(N);
    
    // print the initial data
    std::cout << "input data:" << std::endl;
    thrust::copy(input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));
    std::cout << std::endl << std::endl;

    // compute run lengths
    size_t num_runs = thrust::reduce_by_key
                                    (input.begin(), input.end(),          // input key sequence
                                     thrust::constant_iterator<int>(1),   // input value sequence
                                     output.begin(),                      // output key sequence
                                     lengths.begin()                      // output value sequence
                                     ).first - output.begin();            // compute the output size
    
    // print the output
    std::cout << "run-length encoded output:" << std::endl;
    for(size_t i = 0; i < num_runs; i++)
        std::cout << "(" << output[i] << "," << lengths[i] << ")";
    std::cout << std::endl;
    
    return 0;
}

