#include <unittest/unittest.h>
#include <thrust/sequence.h>
#include <thrust/device_malloc_allocator.h>

template <class Vector>
struct TestVectorRangeInsertSimple
{
    void operator()(size_t)
    {
        typedef typename Vector::value_type T;

        Vector v1(5);
        thrust::sequence(v1.begin(), v1.end());

        // test when insertion range fits inside capacity
        // and the size of the insertion is greater than the number
        // of displaced elements
        Vector v2(3);
        v2.reserve(10);
        thrust::sequence(v2.begin(), v2.end());

        size_t new_size = v2.size() + v1.size();
        size_t insertion_size = v1.end() - v1.begin();
        size_t num_displaced = v2.end() - (v2.begin() + 1);

        ASSERT_EQUAL(true, v2.capacity()   >= new_size);
        ASSERT_EQUAL(true, insertion_size  >  num_displaced);

        v2.insert(v2.begin() + 1,
                  v1.begin(), v1.end());

        ASSERT_EQUAL(T(0), v2[0]);

        ASSERT_EQUAL(T(0), v2[1]);
        ASSERT_EQUAL(T(1), v2[2]);
        ASSERT_EQUAL(T(2), v2[3]);
        ASSERT_EQUAL(T(3), v2[4]);
        ASSERT_EQUAL(T(4), v2[5]);

        ASSERT_EQUAL(T(1), v2[6]);
        ASSERT_EQUAL(T(2), v2[7]);
        
        ASSERT_EQUAL(8lu,  v2.size());
        ASSERT_EQUAL(10lu, v2.capacity());

        // test when insertion range fits inside capacity
        // and the size of the insertion is equal to the number
        // of displaced elements
        Vector v3(5);
        v3.reserve(10);
        thrust::sequence(v3.begin(), v3.end());

        new_size = v3.size() + v1.size();
        insertion_size = v1.end() - v1.begin();
        num_displaced = v3.end() - v3.begin();

        ASSERT_EQUAL(true, v3.capacity()   >=  new_size);
        ASSERT_EQUAL(true, insertion_size  ==  num_displaced);

        v3.insert(v3.begin(),
                  v1.begin(), v1.end());

        ASSERT_EQUAL(T(0), v3[0]);
        ASSERT_EQUAL(T(1), v3[1]);
        ASSERT_EQUAL(T(2), v3[2]);
        ASSERT_EQUAL(T(3), v3[3]);
        ASSERT_EQUAL(T(4), v3[4]);

        ASSERT_EQUAL(T(0), v3[5]);
        ASSERT_EQUAL(T(1), v3[6]);
        ASSERT_EQUAL(T(2), v3[7]);
        ASSERT_EQUAL(T(3), v3[8]);
        ASSERT_EQUAL(T(4), v3[9]);

        ASSERT_EQUAL(10lu, v3.size());
        ASSERT_EQUAL(10lu, v3.capacity());

        // test when insertion range fits inside capacity
        // and the size of the insertion is less than the
        // number of displaced elements
        Vector v4(5);
        v4.reserve(10);
        thrust::sequence(v4.begin(), v4.end());

        new_size = v4.size() + v1.size();
        insertion_size = (v1.begin() + 3) - v1.begin();
        num_displaced = v4.end() - (v4.begin() + 1);

        ASSERT_EQUAL(true, v4.capacity()   >=  new_size);
        ASSERT_EQUAL(true, insertion_size  <   num_displaced);

        v4.insert(v4.begin() + 1,
                  v1.begin(), v1.begin() + 3);

        ASSERT_EQUAL(T(0), v4[0]);

        ASSERT_EQUAL(T(0), v4[1]);
        ASSERT_EQUAL(T(1), v4[2]);
        ASSERT_EQUAL(T(2), v4[3]);

        ASSERT_EQUAL(T(1), v4[4]);
        ASSERT_EQUAL(T(2), v4[5]);
        ASSERT_EQUAL(T(3), v4[6]);
        ASSERT_EQUAL(T(4), v4[7]);

        ASSERT_EQUAL(8lu, v4.size());
        ASSERT_EQUAL(10lu, v4.capacity());

        // test when insertion range does not fit inside capacity
        Vector v5(5);
        thrust::sequence(v5.begin(), v5.end());

        new_size = v5.size() + v1.size();

        ASSERT_EQUAL(true, v5.capacity() < new_size);

        v5.insert(v5.begin() + 1,
                  v1.begin(), v1.end());

        ASSERT_EQUAL(T(0), v5[0]);

        ASSERT_EQUAL(T(0), v5[1]);
        ASSERT_EQUAL(T(1), v5[2]);
        ASSERT_EQUAL(T(2), v5[3]);
        ASSERT_EQUAL(T(3), v5[4]);
        ASSERT_EQUAL(T(4), v5[5]);

        ASSERT_EQUAL(T(1), v5[6]);
        ASSERT_EQUAL(T(2), v5[7]);
        ASSERT_EQUAL(T(3), v5[8]);
        ASSERT_EQUAL(T(4), v5[9]);

        ASSERT_EQUAL(10lu, v5.size());
    }
}; // end TestVectorRangeInsertSimple
VectorUnitTest<TestVectorRangeInsertSimple, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator> TestVectorRangeInsertSimpleDeviceInstance;
VectorUnitTest<TestVectorRangeInsertSimple, NumericTypes, thrust::host_vector,   std::allocator>                  TestVectorRangeInsertSimpleHostInstance;

template <class T>
struct TestVectorRangeInsert
{
    void operator()(size_t n)
    {
        thrust::host_vector<T>   h_src = unittest::random_samples<T>(n + 3);
        thrust::host_vector<T>   h_dst = unittest::random_samples<T>(n);

        thrust::device_vector<T> d_src = h_src;
        thrust::device_vector<T> d_dst = h_dst;

        // choose insertion range at random
        size_t begin = n > 0 ? (size_t)h_src[n]   % n : 0;
        size_t end   = n > 0 ? (size_t)h_src[n+1] % n : 0;
        if(end < begin) thrust::swap(begin,end);

        // choose insertion position at random
        size_t position = n > 0 ? (size_t)h_src[n+2] % n : 0;

        // insert on host
        h_dst.insert(h_dst.begin() + position,
                     h_src.begin() + begin,
                     h_src.begin() + end);

        // insert on device
        d_dst.insert(d_dst.begin() + position,
                     d_src.begin() + begin,
                     d_src.begin() + end);

        ASSERT_EQUAL(h_dst, d_dst);
    }
}; // end TestVectorRangeInsert
VariableUnitTest<TestVectorRangeInsert, IntegralTypes> TestVectorRangeInsertInstance;

template <class Vector>
struct TestVectorFillInsertSimple
{
    void operator()(size_t)
    {
        typedef typename Vector::value_type T;

        // test when insertion range fits inside capacity
        // and the size of the insertion is greater than the number
        // of displaced elements
        Vector v1(3);
        v1.reserve(10);
        thrust::sequence(v1.begin(), v1.end());

        size_t insertion_size = 5;
        size_t new_size = v1.size() + insertion_size;
        size_t num_displaced = v1.end() - (v1.begin() + 1);

        ASSERT_EQUAL(true, v1.capacity()   >= new_size);
        ASSERT_EQUAL(true, insertion_size  >  num_displaced);

        v1.insert(v1.begin() + 1, insertion_size, 13);

        ASSERT_EQUAL(T(0), v1[0]);

        ASSERT_EQUAL(T(13), v1[1]);
        ASSERT_EQUAL(T(13), v1[2]);
        ASSERT_EQUAL(T(13), v1[3]);
        ASSERT_EQUAL(T(13), v1[4]);
        ASSERT_EQUAL(T(13), v1[5]);

        ASSERT_EQUAL(T(1), v1[6]);
        ASSERT_EQUAL(T(2), v1[7]);
        
        ASSERT_EQUAL(8lu,  v1.size());
        ASSERT_EQUAL(10lu, v1.capacity());

        // test when insertion range fits inside capacity
        // and the size of the insertion is equal to the number
        // of displaced elements
        Vector v2(5);
        v2.reserve(10);
        thrust::sequence(v2.begin(), v2.end());

        insertion_size = 5;
        new_size = v2.size() + insertion_size;
        num_displaced = v2.end() - v2.begin();

        ASSERT_EQUAL(true, v2.capacity()   >=  new_size);
        ASSERT_EQUAL(true, insertion_size  ==  num_displaced);

        v2.insert(v2.begin(), insertion_size, 13);

        ASSERT_EQUAL(T(13), v2[0]);
        ASSERT_EQUAL(T(13), v2[1]);
        ASSERT_EQUAL(T(13), v2[2]);
        ASSERT_EQUAL(T(13), v2[3]);
        ASSERT_EQUAL(T(13), v2[4]);

        ASSERT_EQUAL(T(0), v2[5]);
        ASSERT_EQUAL(T(1), v2[6]);
        ASSERT_EQUAL(T(2), v2[7]);
        ASSERT_EQUAL(T(3), v2[8]);
        ASSERT_EQUAL(T(4), v2[9]);

        ASSERT_EQUAL(10lu, v2.size());
        ASSERT_EQUAL(10lu, v2.capacity());

        // test when insertion range fits inside capacity
        // and the size of the insertion is less than the
        // number of displaced elements
        Vector v3(5);
        v3.reserve(10);
        thrust::sequence(v3.begin(), v3.end());

        insertion_size = 3;
        new_size = v3.size() + insertion_size;
        num_displaced = v3.end() - (v3.begin() + 1);

        ASSERT_EQUAL(true, v3.capacity()   >=  new_size);
        ASSERT_EQUAL(true, insertion_size  <   num_displaced);

        v3.insert(v3.begin() + 1, insertion_size, 13);

        ASSERT_EQUAL(T(0), v3[0]);

        ASSERT_EQUAL(T(13), v3[1]);
        ASSERT_EQUAL(T(13), v3[2]);
        ASSERT_EQUAL(T(13), v3[3]);

        ASSERT_EQUAL(T(1), v3[4]);
        ASSERT_EQUAL(T(2), v3[5]);
        ASSERT_EQUAL(T(3), v3[6]);
        ASSERT_EQUAL(T(4), v3[7]);

        ASSERT_EQUAL(8lu, v3.size());
        ASSERT_EQUAL(10lu, v3.capacity());

        // test when insertion range does not fit inside capacity
        Vector v4(5);
        thrust::sequence(v4.begin(), v4.end());

        insertion_size = 5;
        new_size = v4.size() + insertion_size;

        ASSERT_EQUAL(true, v4.capacity() < new_size);

        v4.insert(v4.begin() + 1, insertion_size, 13);

        ASSERT_EQUAL(T(0), v4[0]);

        ASSERT_EQUAL(T(13), v4[1]);
        ASSERT_EQUAL(T(13), v4[2]);
        ASSERT_EQUAL(T(13), v4[3]);
        ASSERT_EQUAL(T(13), v4[4]);
        ASSERT_EQUAL(T(13), v4[5]);

        ASSERT_EQUAL(T(1), v4[6]);
        ASSERT_EQUAL(T(2), v4[7]);
        ASSERT_EQUAL(T(3), v4[8]);
        ASSERT_EQUAL(T(4), v4[9]);

        ASSERT_EQUAL(10lu, v4.size());
    }
}; // end TestVectorFillInsertSimple
VectorUnitTest<TestVectorFillInsertSimple, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator> TestVectorFillInsertSimpleDeviceInstance;
VectorUnitTest<TestVectorFillInsertSimple, NumericTypes, thrust::host_vector,   std::allocator>                  TestVectorFillInsertSimpleHostInstance;

template <class T>
struct TestVectorFillInsert
{
    void operator()(size_t n)
    {
        thrust::host_vector<T>   h_dst = unittest::random_samples<T>(n + 2);

        thrust::device_vector<T> d_dst = h_dst;

        // choose insertion position at random
        size_t position = n > 0 ? (size_t)h_dst[n] % n : 0;

        // choose insertion size at random
        size_t insertion_size = n > 0 ? (size_t)h_dst[n] % n : 13;

        // insert on host
        h_dst.insert(h_dst.begin() + position,
                     insertion_size,
                     13);

        // insert on device
        d_dst.insert(d_dst.begin() + position,
                     insertion_size,
                     13);

        ASSERT_EQUAL(h_dst, d_dst);
    }
}; // end TestVectorFillInsert
VariableUnitTest<TestVectorFillInsert, IntegralTypes> TestVectorFillInsertInstance;

