#include <thrust/detail/config.h>

// Disabled on MSVC && NVCC < 11.1 for GH issue #1098.
#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && defined(__CUDACC__)
#if (__CUDACC_VER_MAJOR__ < 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 1)
#define THRUST_BUG_1098_ACTIVE
#endif // NVCC version check
#endif // MSVC + NVCC check

#if THRUST_CPP_DIALECT >= 2014 && !defined(THRUST_BUG_1098_ACTIVE)

#include <unittest/unittest.h>

#include <thrust/async/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

enum wait_policy
{
  wait_for_futures
, do_not_wait_for_futures
};

template <typename T>
struct custom_greater
{
  __host__ __device__
  bool operator()(T rhs, T lhs) const
  {
    return lhs > rhs;
  }
};

#define DEFINE_SORT_INVOKER(name, ...)                                        \
  template <typename T>                                                       \
  struct name                                                                 \
  {                                                                           \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static void sync(                                                         \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    {                                                                         \
      ::thrust::sort(                                                         \
        THRUST_FWD(first), THRUST_FWD(last)                                   \
      );                                                                      \
    }                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto async(                                                        \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_RETURNS(                                                           \
      ::thrust::async::sort(                                                  \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last)                                   \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_SORT_INVOKER(
  sort_invoker
);
DEFINE_SORT_INVOKER(
  sort_invoker_device, thrust::device
);

#define DEFINE_SORT_OP_INVOKER(name, op, ...)                                 \
  template <typename T>                                                       \
  struct name                                                                 \
  {                                                                           \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static void sync(                                                         \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    {                                                                         \
      ::thrust::sort(                                                         \
        THRUST_FWD(first), THRUST_FWD(last), op<T>{}                          \
      );                                                                      \
    }                                                                         \
                                                                              \
    template <                                                                \
      typename ForwardIt, typename Sentinel                                   \
    >                                                                         \
    __host__                                                                  \
    static auto async(                                                        \
      ForwardIt&& first, Sentinel&& last                                      \
    )                                                                         \
    THRUST_RETURNS(                                                           \
      ::thrust::async::sort(                                                  \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last), op<T>{}                          \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_SORT_OP_INVOKER(
  sort_invoker_less,        thrust::less
);
DEFINE_SORT_OP_INVOKER(
  sort_invoker_less_device, thrust::less, thrust::device 
);

DEFINE_SORT_OP_INVOKER(
  sort_invoker_greater,        thrust::greater
);
DEFINE_SORT_OP_INVOKER(
  sort_invoker_greater_device, thrust::greater, thrust::device 
);

DEFINE_SORT_OP_INVOKER(
  sort_invoker_custom_greater,        custom_greater
);
DEFINE_SORT_OP_INVOKER(
  sort_invoker_custom_greater_device, custom_greater, thrust::device 
);

#undef DEFINE_SORT_INVOKER
#undef DEFINE_SORT_OP_INVOKER

///////////////////////////////////////////////////////////////////////////////

template <template <typename> class SortInvoker, wait_policy WaitPolicy>
struct test_async_sort
{
  template <typename T>
  struct tester
  {
    __host__
    void operator()(std::size_t n)
    {
      thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
      thrust::device_vector<T> d0_data(h0_data);

      ASSERT_EQUAL(h0_data, d0_data);

      SortInvoker<T>::sync(
        h0_data.begin(), h0_data.end()
      );

      auto f0 = SortInvoker<T>::async(
        d0_data.begin(), d0_data.end()
      );

      THRUST_IF_CONSTEXPR(wait_for_futures == WaitPolicy)
      {
        f0.wait();

        ASSERT_EQUAL(h0_data, d0_data);
      }
    }
  };
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker
    , wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker
    , do_not_wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_device
    , wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_policy
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_device
    , do_not_wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_policy_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_less
    , wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_less
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_less
    , do_not_wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_less_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_less_device
    , wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_policy_less
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_less_device
    , do_not_wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_policy_less_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_greater
    , wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_greater
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_greater
    , do_not_wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_greater_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_greater_device
    , wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_policy_greater
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_greater_device
    , do_not_wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_policy_greater_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_custom_greater
    , wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_custom_greater
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_custom_greater
    , do_not_wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_custom_greater_no_wait
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_custom_greater_device
    , wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_policy_custom_greater
);
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(
  THRUST_PP_EXPAND_ARGS(
    test_async_sort<
      sort_invoker_custom_greater_device
    , do_not_wait_for_futures
    >::tester
  )
, NumericTypes
, test_async_sort_policy_custom_greater_no_wait
);

///////////////////////////////////////////////////////////////////////////////

// TODO: Async copy then sort.

// TODO: Test future return type.

#endif

