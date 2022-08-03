#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>


// This example shows how thrust::zip_iterator can be used to create a
// 'virtual' array of structures.  In this case the structure is a 3d
// vector type (Float3) whose (x,y,z) components will be stored in
// three separate float arrays.  The zip_iterator "zips" these arrays
// into a single virtual Float3 array.



// We'll use a 3-tuple to store our 3d vector type
typedef thrust::tuple<float,float,float> Float3;


// This functor implements the dot product between 3d vectors
struct DotProduct : public thrust::binary_function<Float3,Float3,float>
{
    __host__ __device__
        float operator()(const Float3& a, const Float3& b) const
        {
            return thrust::get<0>(a) * thrust::get<0>(b) +    // x components
                   thrust::get<1>(a) * thrust::get<1>(b) +    // y components
                   thrust::get<2>(a) * thrust::get<2>(b);     // z components
        }
};



// Return a host vector with random values in the range [0,1)
thrust::host_vector<float> random_vector(const size_t N,
                                         unsigned int seed = thrust::default_random_engine::default_seed)
{
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    thrust::host_vector<float> temp(N);
    for(size_t i = 0; i < N; i++) {
        temp[i] = u01(rng);
    }
    return temp;
}


int main(void)
{
    // number of vectors
    const size_t N = 1000;

    // We'll store the components of the 3d vectors in separate arrays. One set of
    // arrays will store the 'A' vectors and another set will store the 'B' vectors.

    // This 'structure of arrays' (SoA) approach is usually more efficient than the
    // 'array of structures' (AoS) approach.  The primary reason is that structures,
    // like Float3, don't always obey the memory coalescing rules, so they are not
    // efficiently transferred to and from memory.  Another reason to prefer SoA to
    // AoS is that we don't aways want to process all members of the structure.  For
    // example, if we only need to look at first element of the structure then it
    // is wasteful to load the entire structure from memory.  With the SoA approach,
    // we can chose which elements of the structure we wish to read.

    thrust::device_vector<float> A0 = random_vector(N);  // x components of the 'A' vectors
    thrust::device_vector<float> A1 = random_vector(N);  // y components of the 'A' vectors
    thrust::device_vector<float> A2 = random_vector(N);  // z components of the 'A' vectors

    thrust::device_vector<float> B0 = random_vector(N);  // x components of the 'B' vectors
    thrust::device_vector<float> B1 = random_vector(N);  // y components of the 'B' vectors
    thrust::device_vector<float> B2 = random_vector(N);  // z components of the 'B' vectors

    // Storage for result of each dot product
    thrust::device_vector<float> result(N);


    // We'll now illustrate two ways to use zip_iterator to compute the dot
    // products.  The first method is verbose but shows how the parts fit together.
    // The second method hides these details and is more concise.


    // METHOD #1
    // Defining a zip_iterator type can be a little cumbersome ...
    typedef thrust::device_vector<float>::iterator                     FloatIterator;
    typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator> FloatIteratorTuple;
    typedef thrust::zip_iterator<FloatIteratorTuple>                   Float3Iterator;

    // Now we'll create some zip_iterators for A and B
    Float3Iterator A_first = thrust::make_zip_iterator(thrust::make_tuple(A0.begin(), A1.begin(), A2.begin()));
    Float3Iterator A_last  = thrust::make_zip_iterator(thrust::make_tuple(A0.end(),   A1.end(),   A2.end()));
    Float3Iterator B_first = thrust::make_zip_iterator(thrust::make_tuple(B0.begin(), B1.begin(), B2.begin()));

    // Finally, we pass the zip_iterators into transform() as if they
    // were 'normal' iterators for a device_vector<Float3>.
    thrust::transform(A_first, A_last, B_first, result.begin(), DotProduct());


    // METHOD #2
    // Alternatively, we can avoid creating variables for X_first, X_last,
    // and Y_first and invoke transform() directly.
    thrust::transform( thrust::make_zip_iterator(thrust::make_tuple(A0.begin(), A1.begin(), A2.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(A0.end(),   A1.end(),   A2.end())),
                       thrust::make_zip_iterator(thrust::make_tuple(B0.begin(), B1.begin(), B2.begin())),
                       result.begin(),
                       DotProduct() );



    // Finally, we'll print a few results

    // Example output
    // (0.840188,0.45724,0.0860517) * (0.0587587,0.456151,0.322409) = 0.285683
    // (0.394383,0.640368,0.180886) * (0.0138811,0.24875,0.0221609) = 0.168775
    // (0.783099,0.717092,0.426423) * (0.622212,0.0699601,0.234811) = 0.63755
    // (0.79844,0.460067,0.0470658) * (0.0391351,0.742097,0.354747) = 0.389358
    std::cout << std::fixed;
    for(size_t i = 0; i < 4; i++)
    {
        Float3 a = A_first[i];
        Float3 b = B_first[i];
        float dot = result[i];

        std::cout << "(" << thrust::get<0>(a) << "," << thrust::get<1>(a) << "," << thrust::get<2>(a) << ")";
        std::cout << " * ";
        std::cout << "(" << thrust::get<0>(b) << "," << thrust::get<1>(b) << "," << thrust::get<2>(b) << ")";
        std::cout << " = ";
        std::cout << dot << std::endl;
    }

    return 0;
}

