#include <unittest/unittest.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>


THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN


template <typename Iterator1, typename Iterator2>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
__global__
#endif 
void simple_copy_on_device(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
    while(first1 != last1)
        *(first2++) = *(first1++);
}

template <typename Iterator1, typename Iterator2>
void simple_copy(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    simple_copy_on_device<<<1,1>>>(first1, last1, first2);
#else
    simple_copy_on_device(first1, last1, first2);
#endif
}


void TestDeviceDereferenceDeviceVectorIterator(void)
{
    thrust::device_vector<int> input = unittest::random_integers<int>(100); 
    thrust::device_vector<int> output(input.size(), 0);

    simple_copy(input.begin(), input.end(), output.begin());

    ASSERT_EQUAL(input, output);
}
DECLARE_UNITTEST(TestDeviceDereferenceDeviceVectorIterator);

void TestDeviceDereferenceDevicePtr(void)
{
    thrust::device_vector<int> input = unittest::random_integers<int>(100); 
    thrust::device_vector<int> output(input.size(), 0);

    thrust::device_ptr<int> _first1 = &input[0];
    thrust::device_ptr<int> _last1  = _first1 + input.size();
    thrust::device_ptr<int> _first2 = &output[0];

    simple_copy(_first1, _last1, _first2);

    ASSERT_EQUAL(input, output);
}
DECLARE_UNITTEST(TestDeviceDereferenceDevicePtr);

void TestDeviceDereferenceTransformIterator(void)
{
    thrust::device_vector<int> input = unittest::random_integers<int>(100); 
    thrust::device_vector<int> output(input.size(), 0);

    simple_copy(thrust::make_transform_iterator(input.begin(), thrust::identity<int>()),
                thrust::make_transform_iterator(input.end (),  thrust::identity<int>()),
                output.begin());

    ASSERT_EQUAL(input, output);
}
DECLARE_UNITTEST(TestDeviceDereferenceTransformIterator);

void TestDeviceDereferenceCountingIterator(void)
{
    thrust::counting_iterator<int> first(1);
    thrust::counting_iterator<int> last(6);

    thrust::device_vector<int> output(5);

    simple_copy(first, last, output.begin());

    ASSERT_EQUAL(output[0], 1);
    ASSERT_EQUAL(output[1], 2);
    ASSERT_EQUAL(output[2], 3);
    ASSERT_EQUAL(output[3], 4);
    ASSERT_EQUAL(output[4], 5);
}
DECLARE_UNITTEST(TestDeviceDereferenceCountingIterator);

void TestDeviceDereferenceTransformedCountingIterator(void)
{
    thrust::counting_iterator<int> first(1);
    thrust::counting_iterator<int> last(6);

    thrust::device_vector<int> output(5);

    simple_copy(thrust::make_transform_iterator(first, thrust::negate<int>()),
                thrust::make_transform_iterator(last,  thrust::negate<int>()),
                output.begin());

    ASSERT_EQUAL(output[0], -1);
    ASSERT_EQUAL(output[1], -2);
    ASSERT_EQUAL(output[2], -3);
    ASSERT_EQUAL(output[3], -4);
    ASSERT_EQUAL(output[4], -5);
}
DECLARE_UNITTEST(TestDeviceDereferenceTransformedCountingIterator);

THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END
