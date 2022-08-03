#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <async/test_policy_overloads.h>

#include <async/inclusive_scan/mixin.h>

template <typename input_value_type,
          typename output_value_type   = input_value_type,
          typename alternate_binary_op = thrust::maximum<>>
struct simple_invoker
    : testing::async::mixin::input::device_vector<input_value_type>
    , testing::async::mixin::output::device_vector<output_value_type>
    , testing::async::inclusive_scan::mixin::postfix_args::
        all_overloads<alternate_binary_op>
    , testing::async::inclusive_scan::mixin::invoke_reference::
        host_synchronous<input_value_type, output_value_type>
    , testing::async::inclusive_scan::mixin::invoke_async::simple
    , testing::async::mixin::compare_outputs::assert_almost_equal_if_fp_quiet
{
  static std::string description()
  {
    return "simple invocation with device vectors";
  }
};

template <typename T>
struct test_simple
{
  void operator()(std::size_t num_values) const
  {
    testing::async::test_policy_overloads<simple_invoker<T>>::run(num_values);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(test_simple, NumericTypes);

// Testing the in-place algorithm uses the exact same instantiations of the
// underlying scan implementation as above. Test them here to avoid compiling
// them twice.
template <typename input_value_type,
          typename alternate_binary_op = thrust::maximum<>>
struct simple_inplace_invoker
    : testing::async::mixin::input::device_vector<input_value_type>
    , testing::async::mixin::output::device_vector_reuse_input<input_value_type>
    , testing::async::inclusive_scan::mixin::postfix_args::
        all_overloads<alternate_binary_op>
    , testing::async::inclusive_scan::mixin::invoke_reference::host_synchronous<
        input_value_type>
    , testing::async::inclusive_scan::mixin::invoke_async::simple
    , testing::async::mixin::compare_outputs::assert_almost_equal_if_fp_quiet
{
  static std::string description()
  {
    return "simple in-place invocation with device vectors";
  }
};

template <typename T>
struct test_simple_in_place
{
  void operator()(std::size_t num_values) const
  {
    using invoker = simple_inplace_invoker<T>;
    testing::async::test_policy_overloads<invoker>::run(num_values);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(test_simple_in_place, NumericTypes);

#endif // C++14
