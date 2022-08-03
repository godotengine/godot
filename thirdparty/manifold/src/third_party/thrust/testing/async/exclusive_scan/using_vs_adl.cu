#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <async/test_policy_overloads.h>

#include <async/exclusive_scan/mixin.h>

// Verify what happens when calling the algorithm without any namespace
// qualifiers:
// - If the async entry point is available in the global namespace due to a
//   using statement, the async algorithm should be called.
// - Otherwise, ADL should resolve the call to the synchronous algo in the
//   thrust:: namespace.

namespace invoke_reference
{

template <typename input_value_type,
          typename output_value_type = input_value_type>
struct adl_host_synchronous
{
  template <typename InputType,
            typename OutputType,
            typename PostfixArgTuple,
            std::size_t... PostfixArgIndices>
  static void invoke_reference(InputType const& input,
                               OutputType& output,
                               PostfixArgTuple&& postfix_tuple,
                               std::index_sequence<PostfixArgIndices...>)
  {
    // Create host versions of the input/output:
    thrust::host_vector<input_value_type> host_input(input.cbegin(),
                                                     input.cend());
    thrust::host_vector<output_value_type> host_output(host_input.size());

    using OutIter = thrust::remove_cvref_t<decltype(host_output.begin())>;

    // ADL should resolve this to the synchronous `thrust::` algorithm.
    // This is checked by ensuring that the call returns an output iterator.
    OutIter result =
      exclusive_scan(host_input.cbegin(),
                     host_input.cend(),
                     host_output.begin(),
                     std::get<PostfixArgIndices>(THRUST_FWD(postfix_tuple))...);
    (void)result;

    // Copy back to device.
    output = host_output;
  }
};

} // namespace invoke_reference

namespace invoke_async
{

struct using_namespace
{
  template <typename PrefixArgTuple,
            std::size_t... PrefixArgIndices,
            typename InputType,
            typename OutputType,
            typename PostfixArgTuple,
            std::size_t... PostfixArgIndices>
  static auto invoke_async(PrefixArgTuple&& prefix_tuple,
                           std::index_sequence<PrefixArgIndices...>,
                           InputType const& input,
                           OutputType& output,
                           PostfixArgTuple&& postfix_tuple,
                           std::index_sequence<PostfixArgIndices...>)
  {
    // Importing the CPO into the current namespace should unambiguously resolve
    // this call to the CPO, as opposed to resolving to the thrust:: algorithm
    // via ADL. This is verified by checking that an event is returned.
    using namespace thrust::async;
    thrust::device_event e =
      exclusive_scan(std::get<PrefixArgIndices>(THRUST_FWD(prefix_tuple))...,
                     input.cbegin(),
                     input.cend(),
                     output.begin(),
                     std::get<PostfixArgIndices>(THRUST_FWD(postfix_tuple))...);
    return e;
  }
};

struct using_cpo
{
  template <typename PrefixArgTuple,
            std::size_t... PrefixArgIndices,
            typename InputType,
            typename OutputType,
            typename PostfixArgTuple,
            std::size_t... PostfixArgIndices>
  static auto invoke_async(PrefixArgTuple&& prefix_tuple,
                           std::index_sequence<PrefixArgIndices...>,
                           InputType const& input,
                           OutputType& output,
                           PostfixArgTuple&& postfix_tuple,
                           std::index_sequence<PostfixArgIndices...>)
  {
    // Importing the CPO into the current namespace should unambiguously resolve
    // this call to the CPO, as opposed to resolving to the thrust:: algorithm
    // via ADL. This is verified by checking that an event is returned.
    using thrust::async::exclusive_scan;
    thrust::device_event e =
      exclusive_scan(std::get<PrefixArgIndices>(THRUST_FWD(prefix_tuple))...,
                     input.cbegin(),
                     input.cend(),
                     output.begin(),
                     std::get<PostfixArgIndices>(THRUST_FWD(postfix_tuple))...);
    return e;
  }
};

} // namespace invoke_async

template <typename input_value_type,
          typename output_value_type   = input_value_type,
          typename initial_value_type  = input_value_type,
          typename alternate_binary_op = thrust::maximum<>>
struct using_namespace_invoker
    : testing::async::mixin::input::device_vector<input_value_type>
    , testing::async::mixin::output::device_vector<output_value_type>
    , testing::async::exclusive_scan::mixin::postfix_args::
        all_overloads<initial_value_type, alternate_binary_op>
    , invoke_reference::adl_host_synchronous<input_value_type, output_value_type>
    , invoke_async::using_namespace
    , testing::async::mixin::compare_outputs::assert_almost_equal_if_fp_quiet
{
  static std::string description()
  {
    return "importing async CPO with `using namespace thrust::async`";
  }
};

void test_using_namespace()
{
  using invoker = using_namespace_invoker<int>;
  testing::async::test_policy_overloads<invoker>::run(128);
}
DECLARE_UNITTEST(test_using_namespace);

template <typename input_value_type,
          typename output_value_type   = input_value_type,
          typename initial_value_type  = input_value_type,
          typename alternate_binary_op = thrust::maximum<>>
struct using_cpo_invoker
    : testing::async::mixin::input::device_vector<input_value_type>
    , testing::async::mixin::output::device_vector<output_value_type>
    , testing::async::exclusive_scan::mixin::postfix_args::
        all_overloads<initial_value_type, alternate_binary_op>
    , invoke_reference::adl_host_synchronous<input_value_type, output_value_type>
    , invoke_async::using_cpo
    , testing::async::mixin::compare_outputs::assert_almost_equal_if_fp_quiet
{
  static std::string description()
  {
    return "importing async CPO with "
           "`using namespace thrust::async::exclusive_scan`";
  }
};

void test_using_cpo()
{
  using invoker = using_cpo_invoker<int>;
  testing::async::test_policy_overloads<invoker>::run(128);
}
DECLARE_UNITTEST(test_using_cpo);

#endif // C++14
