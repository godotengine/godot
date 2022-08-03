#include <unittest/unittest.h>

#include <thrust/detail/seq.h>
#include <thrust/system/cpp/detail/par.h>
#include <thrust/system/omp/detail/par.h>
#include <thrust/system/tbb/detail/par.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <thrust/system/cuda/detail/par.h>
#endif

template<typename T>
struct test_allocator_t
{
};

test_allocator_t<int> test_allocator = test_allocator_t<int>();
const test_allocator_t<int> const_test_allocator = test_allocator_t<int>();

struct test_memory_resource_t final : thrust::mr::memory_resource<>
{
    void * do_allocate(std::size_t size, std::size_t) override
    {
        return reinterpret_cast<void *>(size);
    }

    void do_deallocate(void * ptr, std::size_t size, std::size_t) override
    {
        ASSERT_EQUAL(ptr, reinterpret_cast<void *>(size));
    }
} test_memory_resource;

template<typename Policy, template <typename> class CRTPBase>
struct policy_info
{
    typedef Policy policy;

    template<template <typename, template <typename> class> class Template, typename Argument>
    struct apply_base_second
    {
        typedef Template<Argument, CRTPBase> type;
    };
};

template<typename PolicyInfo>
struct TestAllocatorAttachment
{
    template<typename Expected, typename T>
    static void assert_correct(T)
    {
        ASSERT_EQUAL(
            (thrust::detail::is_same<
                T,
                typename PolicyInfo::template apply_base_second<
                    thrust::detail::execute_with_allocator,
                    Expected
                >::type
            >::value), true);
    }

    template<typename ExpectedResource, typename T>
    static void assert_npa_correct(T)
    {
        ASSERT_EQUAL(
            (thrust::detail::is_same<
                T,
                typename PolicyInfo::template apply_base_second<
                    thrust::detail::execute_with_allocator,
                    thrust::mr::allocator<
                        thrust::detail::max_align_t,
                        ExpectedResource
                    >
                >::type
            >::value), true);
    }

    template<typename Policy>
    void test_temporary_allocation_valid(Policy policy)
    {
        using thrust::detail::get_temporary_buffer;

        return_temporary_buffer(
            policy,
            get_temporary_buffer<int>(
                policy,
                123
            ).first,
            123
        );
    }

    void operator()()
    {
        typename PolicyInfo::policy policy;

        // test correctness of attachment
        assert_correct<test_allocator_t<int> >(policy(test_allocator_t<int>()));
        assert_correct<test_allocator_t<int>&>(policy(test_allocator));
        assert_correct<test_allocator_t<int> >(policy(const_test_allocator));

        assert_npa_correct<test_memory_resource_t>(policy(&test_memory_resource));

        // test whether the resulting policy is actually usable
        // a real allocator is necessary here, unlike above
        std::allocator<int> alloc;
        const std::allocator<int> const_alloc;

        test_temporary_allocation_valid(policy(std::allocator<int>()));
        test_temporary_allocation_valid(policy(alloc));
        test_temporary_allocation_valid(policy(const_alloc));
        test_temporary_allocation_valid(policy(&test_memory_resource));

        #if THRUST_CPP_DIALECT >= 2011
        test_temporary_allocation_valid(policy(std::allocator<int>()).after(1));
        test_temporary_allocation_valid(policy(alloc).after(1));
        test_temporary_allocation_valid(policy(const_alloc).after(1));
        #endif
    }
};

typedef policy_info<
    thrust::detail::seq_t,
    thrust::system::detail::sequential::execution_policy
> sequential_info;
typedef policy_info<
    thrust::system::cpp::detail::par_t,
    thrust::system::cpp::detail::execution_policy
> cpp_par_info;
typedef policy_info<
    thrust::system::omp::detail::par_t,
    thrust::system::omp::detail::execution_policy
> omp_par_info;
typedef policy_info<
    thrust::system::tbb::detail::par_t,
    thrust::system::tbb::detail::execution_policy
> tbb_par_info;

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
typedef policy_info<
    thrust::system::cuda::detail::par_t,
    thrust::cuda_cub::execute_on_stream_base
> cuda_par_info;
#endif

SimpleUnitTest<
    TestAllocatorAttachment,
    unittest::type_list<
        sequential_info,
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cuda_par_info,
#endif
        cpp_par_info,
        omp_par_info,
        tbb_par_info
    >
> TestAllocatorAttachmentInstance;
