#include <unittest/unittest.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

template <class Vector>
void TestTransformOutputIterator(void)
{
    typedef typename Vector::value_type T;

    typedef thrust::square<T> UnaryFunction;
    typedef typename Vector::iterator Iterator;

    Vector input(4);
    Vector output(4);
    
    // initialize input
    thrust::sequence(input.begin(), input.end(), T{1});
   
    // construct transform_iterator
    thrust::transform_output_iterator<UnaryFunction, Iterator> output_iter(output.begin(), UnaryFunction());

    thrust::copy(input.begin(), input.end(), output_iter);

    Vector gold_output(4);
    gold_output[0] = 1;
    gold_output[1] = 4;
    gold_output[2] = 9;
    gold_output[3] = 16;

    ASSERT_EQUAL(output, gold_output);

}
DECLARE_VECTOR_UNITTEST(TestTransformOutputIterator);

template <class Vector>
void TestMakeTransformOutputIterator(void)
{
    typedef typename Vector::value_type T;

    typedef thrust::square<T> UnaryFunction;

    Vector input(4);
    Vector output(4);
    
    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);
   
    thrust::copy(input.begin(), input.end(),
                 thrust::make_transform_output_iterator(output.begin(), UnaryFunction()));

    Vector gold_output(4);
    gold_output[0] = 1;
    gold_output[1] = 4;
    gold_output[2] = 9;
    gold_output[3] = 16;
    ASSERT_EQUAL(output, gold_output);

}
DECLARE_VECTOR_UNITTEST(TestMakeTransformOutputIterator);

template <typename T>
struct TestTransformOutputIteratorScan
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_result(n);
        thrust::device_vector<T> d_result(n);

        // run on host
        thrust::inclusive_scan(thrust::make_transform_iterator(h_data.begin(), thrust::negate<T>()),
                               thrust::make_transform_iterator(h_data.end(),   thrust::negate<T>()),
                               h_result.begin());
        // run on device
        thrust::inclusive_scan(d_data.begin(), d_data.end(),
                               thrust::make_transform_output_iterator(
                                   d_result.begin(), thrust::negate<T>()));


        ASSERT_EQUAL(h_result, d_result);
    }
};
VariableUnitTest<TestTransformOutputIteratorScan, SignedIntegralTypes> TestTransformOutputIteratorScanInstance;

