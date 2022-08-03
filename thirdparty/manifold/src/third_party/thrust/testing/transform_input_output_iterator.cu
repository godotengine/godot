#include <unittest/unittest.h>
#include <thrust/iterator/transform_input_output_iterator.h>

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

template <class Vector>
void TestTransformInputOutputIterator(void)
{
    typedef typename Vector::value_type T;

    typedef thrust::negate<T> InputFunction;
    typedef thrust::square<T> OutputFunction;
    typedef typename Vector::iterator Iterator;

    Vector input(4);
    Vector squared(4);
    Vector negated(4);
    
    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);
   
    // construct transform_iterator
    thrust::transform_input_output_iterator<InputFunction, OutputFunction, Iterator>
        transform_iter(squared.begin(), InputFunction(), OutputFunction());

    // transform_iter writes squared value
    thrust::copy(input.begin(), input.end(), transform_iter);

    Vector gold_squared(4);
    gold_squared[0] = 1;
    gold_squared[1] = 4;
    gold_squared[2] = 9;
    gold_squared[3] = 16;

    ASSERT_EQUAL(squared, gold_squared);

    // negated value read from transform_iter
    thrust::copy_n(transform_iter, squared.size(), negated.begin());

    Vector gold_negated(4);
    gold_negated[0] = -1;
    gold_negated[1] = -4;
    gold_negated[2] = -9;
    gold_negated[3] = -16;

    ASSERT_EQUAL(negated, gold_negated);

}
DECLARE_VECTOR_UNITTEST(TestTransformInputOutputIterator);

template <class Vector>
void TestMakeTransformInputOutputIterator(void)
{
    typedef typename Vector::value_type T;

    typedef thrust::negate<T> InputFunction;
    typedef thrust::square<T> OutputFunction;

    Vector input(4);
    Vector negated(4);
    Vector squared(4);
    
    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);

    // negated value read from transform iterator
    thrust::copy_n(thrust::make_transform_input_output_iterator(input.begin(), InputFunction(), OutputFunction()),
                   input.size(), negated.begin());

    Vector gold_negated(4);
    gold_negated[0] = -1;
    gold_negated[1] = -2;
    gold_negated[2] = -3;
    gold_negated[3] = -4;

    ASSERT_EQUAL(negated, gold_negated);

    // squared value writen by transform iterator
    thrust::copy(negated.begin(), negated.end(),
                 thrust::make_transform_input_output_iterator(squared.begin(), InputFunction(), OutputFunction()));

    Vector gold_squared(4);
    gold_squared[0] = 1;
    gold_squared[1] = 4;
    gold_squared[2] = 9;
    gold_squared[3] = 16;

    ASSERT_EQUAL(squared, gold_squared);

}
DECLARE_VECTOR_UNITTEST(TestMakeTransformInputOutputIterator);

template <typename T>
struct TestTransformInputOutputIteratorScan
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_result(n);
        thrust::device_vector<T> d_result(n);

        // run on host (uses forward iterator negate)
        thrust::inclusive_scan(thrust::make_transform_input_output_iterator(h_data.begin(), thrust::negate<T>(), thrust::identity<T>()),
                               thrust::make_transform_input_output_iterator(h_data.end(),   thrust::negate<T>(), thrust::identity<T>()),
                               h_result.begin());
        // run on device (uses reverse iterator negate)
        thrust::inclusive_scan(d_data.begin(), d_data.end(),
                               thrust::make_transform_input_output_iterator(
                                   d_result.begin(), thrust::square<T>(), thrust::negate<T>()));


        ASSERT_EQUAL(h_result, d_result);
    }
};
VariableUnitTest<TestTransformInputOutputIteratorScan, IntegralTypes> TestTransformInputOutputIteratorScanInstance;

