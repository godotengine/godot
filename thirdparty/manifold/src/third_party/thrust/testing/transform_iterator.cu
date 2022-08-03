#include <unittest/unittest.h>
#include <thrust/iterator/transform_iterator.h>

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

#include <memory>

template <class Vector>
void TestTransformIterator(void)
{
    typedef typename Vector::value_type T;

    typedef thrust::negate<T> UnaryFunction;
    typedef typename Vector::iterator Iterator;

    Vector input(4);
    Vector output(4);
    
    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);
   
    // construct transform_iterator
    thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

    thrust::copy(iter, iter + 4, output.begin());

    ASSERT_EQUAL(output[0], -1);
    ASSERT_EQUAL(output[1], -2);
    ASSERT_EQUAL(output[2], -3);
    ASSERT_EQUAL(output[3], -4);

}
DECLARE_VECTOR_UNITTEST(TestTransformIterator);

template <class Vector>
void TestMakeTransformIterator(void)
{
    typedef typename Vector::value_type T;

    typedef thrust::negate<T> UnaryFunction;
    typedef typename Vector::iterator Iterator;

    Vector input(4);
    Vector output(4);
    
    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);
   
    // construct transform_iterator
    thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

    thrust::copy(thrust::make_transform_iterator(input.begin(), UnaryFunction()), 
                 thrust::make_transform_iterator(input.end(), UnaryFunction()), 
                 output.begin());

    ASSERT_EQUAL(output[0], -1);
    ASSERT_EQUAL(output[1], -2);
    ASSERT_EQUAL(output[2], -3);
    ASSERT_EQUAL(output[3], -4);

}
DECLARE_VECTOR_UNITTEST(TestMakeTransformIterator);

template <typename T>
struct TestTransformIteratorReduce
{
    void operator()(const size_t n)
    {
        thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
        thrust::device_vector<T> d_data = h_data;

        // run on host
        T h_result = thrust::reduce( thrust::make_transform_iterator(h_data.begin(), thrust::negate<T>()),
                                     thrust::make_transform_iterator(h_data.end(),   thrust::negate<T>()) );

        // run on device
        T d_result = thrust::reduce( thrust::make_transform_iterator(d_data.begin(), thrust::negate<T>()),
                                     thrust::make_transform_iterator(d_data.end(),   thrust::negate<T>()) );

        ASSERT_EQUAL(h_result, d_result);
    }
};
VariableUnitTest<TestTransformIteratorReduce, IntegralTypes> TestTransformIteratorReduceInstance;


struct ExtractValue{
    int operator()(std::unique_ptr<int> const& n){
        return *n;
    }
};

void TestTransformIteratorNonCopyable(){

    thrust::host_vector<std::unique_ptr<int>> hv(4);
    hv[0].reset(new int{1});
    hv[1].reset(new int{2});
    hv[2].reset(new int{3});
    hv[3].reset(new int{4});

    auto transformed = thrust::make_transform_iterator(hv.begin(), ExtractValue{});
    ASSERT_EQUAL(transformed[0], 1);
    ASSERT_EQUAL(transformed[1], 2);
    ASSERT_EQUAL(transformed[2], 3);
    ASSERT_EQUAL(transformed[3], 4);

}

DECLARE_UNITTEST(TestTransformIteratorNonCopyable);

