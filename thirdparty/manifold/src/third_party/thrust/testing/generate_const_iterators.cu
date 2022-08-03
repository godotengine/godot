#include <unittest/runtime_static_assert.h>
#include <unittest/unittest.h>
#include <thrust/generate.h>

struct generator
{
    __host__ __device__
    int operator()() const
    {
        return 1;
    }
};

void TestGenerateConstIteratorCompilationError()
{
    thrust::host_vector<int> test1(10);

    ASSERT_STATIC_ASSERT(thrust::generate(test1.cbegin(), test1.cend(), generator()));
    ASSERT_STATIC_ASSERT(thrust::generate_n(test1.cbegin(), 10, generator()));
}
DECLARE_UNITTEST(TestGenerateConstIteratorCompilationError);

void TestFillConstIteratorCompilationError()
{
    thrust::host_vector<int> test1(10);
    ASSERT_STATIC_ASSERT(thrust::fill(test1.cbegin(), test1.cend(), 1));
}
DECLARE_UNITTEST(TestFillConstIteratorCompilationError);

