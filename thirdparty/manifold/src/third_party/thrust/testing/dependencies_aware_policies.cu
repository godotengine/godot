#include <unittest/unittest.h>

#include <thrust/detail/config.h>
#include <thrust/detail/seq.h>
#include <thrust/system/cpp/detail/par.h>
#include <thrust/system/omp/detail/par.h>
#include <thrust/system/tbb/detail/par.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <thrust/system/cuda/detail/par.h>
#endif

#if THRUST_CPP_DIALECT >= 2011

template<typename T>
struct test_allocator_t
{
};

test_allocator_t<int> test_allocator = test_allocator_t<int>();

template<int I>
struct test_dependency_t
{
};

template<int I>
test_dependency_t<I> test_dependency()
{
    return {};
}

template<typename Policy, template<typename> class CRTPBase>
struct policy_info
{
    using policy = Policy;

    template<template<template<typename> class, typename...> class Template, typename ...Arguments>
    using apply_base_first = Template<CRTPBase, Arguments...>;

    template<template<typename, template<typename> class, typename...> class Template, typename First, typename ...Arguments>
    using apply_base_second = Template<First, CRTPBase, Arguments...>;
};

template<typename PolicyInfo>
struct TestDependencyAttachment
{
    template<typename ...Expected, typename T>
    static void assert_correct(T)
    {
        ASSERT_EQUAL(
            (thrust::detail::is_same<
                T,
                typename PolicyInfo::template apply_base_first<
                    thrust::detail::execute_with_dependencies,
                    Expected...
                >
            >::value), true);
    }

    template<typename Allocator, typename ...Expected, typename T>
    static void assert_correct_with_allocator(T)
    {
        ASSERT_EQUAL(
            (thrust::detail::is_same<
                T,
                typename PolicyInfo::template apply_base_second<
                    thrust::detail::execute_with_allocator_and_dependencies,
                    Allocator,
                    Expected...
                >
            >::value), true);
    }

    void operator()()
    {
        typename PolicyInfo::policy policy;

        assert_correct<
            test_dependency_t<1>
        >(policy
            .after(
                test_dependency<1>()
            )
        );

        assert_correct<
            test_dependency_t<1>,
            test_dependency_t<2>
        >(policy
            .after(
                test_dependency<1>(),
                test_dependency<2>()
            )
        );

        assert_correct<
            test_dependency_t<1>,
            test_dependency_t<2>,
            test_dependency_t<3>
        >(policy
            .after(
                test_dependency<1>(),
                test_dependency<2>(),
                test_dependency<3>()
            )
        );

        assert_correct_with_allocator<
            test_allocator_t<int> &,
            test_dependency_t<1>
        >(policy(test_allocator)
            .after(
                test_dependency<1>()
            )
        );

        assert_correct_with_allocator<
            test_allocator_t<int> &,
            test_dependency_t<1>,
            test_dependency_t<2>
        >(policy(test_allocator)
            .after(
                test_dependency<1>(),
                test_dependency<2>()
            )
        );

        assert_correct_with_allocator<
            test_allocator_t<int> &,
            test_dependency_t<1>,
            test_dependency_t<2>,
            test_dependency_t<3>
        >(policy(test_allocator)
            .after(
                test_dependency<1>(),
                test_dependency<2>(),
                test_dependency<3>()
            )
        );
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
    TestDependencyAttachment,
    unittest::type_list<
        // TODO: uncomment when dependencies are generalized to all backends
        // sequential_info,
        // cpp_par_info,
        // omp_par_info,
        // tbb_par_info,
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cuda_par_info
#endif
    >
> TestDependencyAttachmentInstance;

#else // C++11

void TestDummy()
{
}
DECLARE_UNITTEST(TestDummy);

#endif // C++11
