#include <unittest/unittest.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

void TestDevicePointerManipulation(void)
{
    thrust::device_vector<int> data(5);

    thrust::device_ptr<int> begin(&data[0]);
    thrust::device_ptr<int> end(&data[0] + 5);

    ASSERT_EQUAL(end - begin, 5);

    begin++;
    begin--;
    
    ASSERT_EQUAL(end - begin, 5);

    begin += 1;
    begin -= 1;
    
    ASSERT_EQUAL(end - begin, 5);

    begin = begin + (int) 1;
    begin = begin - (int) 1;

    ASSERT_EQUAL(end - begin, 5);

    begin = begin + (unsigned int) 1;
    begin = begin - (unsigned int) 1;
    
    ASSERT_EQUAL(end - begin, 5);
    
    begin = begin + (size_t) 1;
    begin = begin - (size_t) 1;

    ASSERT_EQUAL(end - begin, 5);

    begin = begin + (ptrdiff_t) 1;
    begin = begin - (ptrdiff_t) 1;

    ASSERT_EQUAL(end - begin, 5);

    begin = begin + (thrust::device_ptr<int>::difference_type) 1;
    begin = begin - (thrust::device_ptr<int>::difference_type) 1;

    ASSERT_EQUAL(end - begin, 5);
}
DECLARE_UNITTEST(TestDevicePointerManipulation);


void TestMakeDevicePointer(void)
{
    typedef int T;

    T *raw_ptr = 0;

    thrust::device_ptr<T> p0 = thrust::device_pointer_cast(raw_ptr);

    ASSERT_EQUAL(thrust::raw_pointer_cast(p0), raw_ptr);

    thrust::device_ptr<T> p1 = thrust::device_pointer_cast(p0);

    ASSERT_EQUAL(p0, p1);
}
DECLARE_UNITTEST(TestMakeDevicePointer);


template<typename Vector>
void TestRawPointerCast(void)
{
    typedef typename Vector::value_type T;

    Vector vec(3);

    T * first;
    T * last;
    
    first = thrust::raw_pointer_cast(&vec[0]);
    last  = thrust::raw_pointer_cast(&vec[3]);
    ASSERT_EQUAL(last - first, 3);

    first = thrust::raw_pointer_cast(&vec.front());
    last  = thrust::raw_pointer_cast(&vec.back());
    ASSERT_EQUAL(last - first, 2);

    // Do we want these to work?
    //first = thrust::raw_pointer_cast(vec.begin());
    //last  = thrust::raw_pointer_cast(vec.end());
    //ASSERT_EQUAL(last - first, 3);
}
DECLARE_VECTOR_UNITTEST(TestRawPointerCast);


#if THRUST_CPP_DIALECT >= 2011
template<typename T>
void TestDevicePointerNullptrCompatibility()
{
    thrust::device_ptr<T> p0(nullptr);

    ASSERT_EQUAL_QUIET(nullptr, p0);
    ASSERT_EQUAL_QUIET(p0, nullptr);

    p0 = nullptr;

    ASSERT_EQUAL_QUIET(nullptr, p0);
    ASSERT_EQUAL_QUIET(p0, nullptr);
}
DECLARE_GENERIC_UNITTEST(TestDevicePointerNullptrCompatibility);

template<typename T>
void TestDevicePointerBoolConversion()
{
    thrust::device_ptr<T> p0(nullptr);
    auto const b = bool(p0);

    ASSERT_EQUAL_QUIET(false, b);
}
DECLARE_GENERIC_UNITTEST(TestDevicePointerBoolConversion);
#endif

