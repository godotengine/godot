#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <iostream>

// This example shows how to perform a lexicographical sort on multiple keys.
//
// http://en.wikipedia.org/wiki/Lexicographical_order

template <typename KeyVector, typename PermutationVector>
void update_permutation(KeyVector& keys, PermutationVector& permutation)
{
    // temporary storage for keys
    KeyVector temp(keys.size());

    // permute the keys with the current reordering
    thrust::gather(permutation.begin(), permutation.end(), keys.begin(), temp.begin());

    // stable_sort the permuted keys and update the permutation
    thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}


template <typename KeyVector, typename PermutationVector>
void apply_permutation(KeyVector& keys, PermutationVector& permutation)
{
    // copy keys to temporary vector
    KeyVector temp(keys.begin(), keys.end());

    // permute the keys
    thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys.begin());
}


thrust::host_vector<int> random_vector(size_t N)
{
    thrust::host_vector<int> vec(N);
    static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<int> dist(0, 9);

    for (size_t i = 0; i < N; i++)
        vec[i] = dist(rng);

    return vec;
}


int main(void)
{
    size_t N = 20;

    // generate three arrays of random values
    thrust::device_vector<int> upper  = random_vector(N);
    thrust::device_vector<int> middle = random_vector(N);
    thrust::device_vector<int> lower  = random_vector(N);
    
    std::cout << "Unsorted Keys" << std::endl;
    for(size_t i = 0; i < N; i++)
    {
        std::cout << "(" << upper[i] << "," << middle[i] << "," << lower[i] << ")" << std::endl;
    }

    // initialize permutation to [0, 1, 2, ... ,N-1]
    thrust::device_vector<int> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    // sort from least significant key to most significant keys
    update_permutation(lower,  permutation);
    update_permutation(middle, permutation);
    update_permutation(upper,  permutation);

    // Note: keys have not been modified
    // Note: permutation now maps unsorted keys to sorted order
  
    // permute the key arrays by the final permuation
    apply_permutation(lower,  permutation);
    apply_permutation(middle, permutation);
    apply_permutation(upper,  permutation);

    std::cout << "Sorted Keys" << std::endl;
    for(size_t i = 0; i < N; i++)
    {
        std::cout << "(" << upper[i] << "," << middle[i] << "," << lower[i] << ")" << std::endl;
    }

    return 0;
}

