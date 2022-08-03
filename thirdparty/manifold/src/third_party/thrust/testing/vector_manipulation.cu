#include <unittest/unittest.h>
#include <thrust/device_malloc_allocator.h>
#include <vector>

template <class Vector>
void TestVectorManipulation(size_t n)
{
    typedef typename Vector::iterator   Iterator;
    typedef typename Vector::value_type T;

    thrust::host_vector<T> src = unittest::random_samples<T>(n);
    ASSERT_EQUAL(src.size(), n);

    // basic initialization
    Vector test0(n);
    Vector test1(n, T(3));
    ASSERT_EQUAL(test0.size(), n);
    ASSERT_EQUAL(test1.size(), n);
    ASSERT_EQUAL((test1 == std::vector<T>(n, T(3))), true);

#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && (_MSC_VER <= 1400)
    // XXX MSVC 2005's STL unintentionally uses adl to dispatch advance which
    //     produces an ambiguity between std::advance & thrust::advance
    //     don't produce a KNOWN_FAILURE, just ignore the issue
#else
    // initializing from other vector
    std::vector<T> stl_vector(src.begin(), src.end());
    Vector cpy0 = src;
    Vector cpy1(stl_vector);
    Vector cpy2(stl_vector.begin(), stl_vector.end());
    ASSERT_EQUAL(cpy0, src);
    ASSERT_EQUAL(cpy1, src);
    ASSERT_EQUAL(cpy2, src);
#endif

    // resizing
    Vector vec1(src);
    vec1.resize(n + 3);
    ASSERT_EQUAL(vec1.size(), n + 3);
    vec1.resize(n);
    ASSERT_EQUAL(vec1.size(), n);
    ASSERT_EQUAL(vec1, src); 
    
    vec1.resize(n + 20, T(11));
    Vector tail(vec1.begin() + n, vec1.end());
    ASSERT_EQUAL((tail == std::vector<T>(20, T(11))), true);

    // shrinking a vector should not invalidate iterators
    Iterator first = vec1.begin();
    vec1.resize(10);
    ASSERT_EQUAL_QUIET(first, vec1.begin());

    vec1.resize(0);
    ASSERT_EQUAL(vec1.size(), 0lu);
    ASSERT_EQUAL(vec1.empty(), true);
    vec1.resize(10);
    ASSERT_EQUAL(vec1.size(), 10lu);
    vec1.clear();
    ASSERT_EQUAL(vec1.size(), 0lu);
    vec1.resize(5);
    ASSERT_EQUAL(vec1.size(), 5lu);

    // push_back
    Vector vec2;
    for(size_t i = 0; i < 10; ++i)
    {
        ASSERT_EQUAL(vec2.size(), i);
        vec2.push_back(T(i));
        ASSERT_EQUAL(vec2.size(), i + 1);
        for(size_t j = 0; j <= i; j++)
            ASSERT_EQUAL(vec2[j], T(j));
        ASSERT_EQUAL(vec2.back(), T(i));
    }

    // pop_back
    for(size_t i = 10; i > 0; --i)
    {
        ASSERT_EQUAL(vec2.size(), i);
        ASSERT_EQUAL(vec2.back(), T(i - 1));
        vec2.pop_back();
        ASSERT_EQUAL(vec2.size(), i - 1);
        for(size_t j = 0; j < i; j++)
            ASSERT_EQUAL(vec2[j], T(j));
    }

    //TODO test swap, erase(pos), erase(begin, end)
}

template <typename T>
void TestVectorManipulationHost(size_t n)
{
    TestVectorManipulation< thrust::host_vector<T> >(n);
}
DECLARE_VARIABLE_UNITTEST(TestVectorManipulationHost);

template <typename T>
void TestVectorManipulationDevice(size_t n)
{
    TestVectorManipulation< thrust::device_vector<T> >(n);
}
DECLARE_VARIABLE_UNITTEST(TestVectorManipulationDevice);

