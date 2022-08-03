#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iterator>

// This example decodes a run-length code [1] for an array of characters.
//
// [1] http://en.wikipedia.org/wiki/Run-length_encoding


int main(void)
{
    // allocate storage for compressed input and run lengths
    thrust::device_vector<char> input(6);
    thrust::device_vector<int>  lengths(6);
    input[0] = 'a';  lengths[0] = 3;
    input[1] = 'b';  lengths[1] = 5;
    input[2] = 'c';  lengths[2] = 1;
    input[3] = 'd';  lengths[3] = 2;
    input[4] = 'e';  lengths[4] = 9;
    input[5] = 'f';  lengths[5] = 2;
    
    // print the initial data
    std::cout << "run-length encoded input:" << std::endl;
    for(size_t i = 0; i < 6; i++)
        std::cout << "(" << input[i] << "," << lengths[i] << ")";
    std::cout << std::endl << std::endl;

    // scan the lengths
    thrust::inclusive_scan(lengths.begin(), lengths.end(), lengths.begin());
    
    // output size is sum of the run lengths
    int N = lengths.back();

    // compute input index for each output element
    thrust::device_vector<int> indices(N);
    thrust::lower_bound(lengths.begin(), lengths.end(),
                        thrust::counting_iterator<int>(1),
                        thrust::counting_iterator<int>(N + 1),
                        indices.begin());

    // gather input elements
    thrust::device_vector<char> output(N);
    thrust::gather(indices.begin(), indices.end(),
                   input.begin(),
                   output.begin());

    // print the initial data
    std::cout << "decoded output:" << std::endl;
    thrust::copy(output.begin(), output.end(), std::ostream_iterator<char>(std::cout, ""));
    std::cout << std::endl;

    return 0;
}

