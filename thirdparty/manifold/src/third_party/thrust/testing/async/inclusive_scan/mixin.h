#pragma once

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/async/scan.h>

#include <thrust/scan.h>

#include <async/mixin.h>

namespace testing
{
namespace async
{
namespace inclusive_scan
{

namespace mixin
{

//------------------------------------------------------------------------------
namespace postfix_args
{

template <typename alternate_binary_op = thrust::maximum<>>
struct all_overloads
{
  using postfix_args_type = std::tuple< // List any extra arg overloads:
    std::tuple<>,                       // - no extra args
    std::tuple<alternate_binary_op>     // - binary_op
    >;

  static postfix_args_type generate_postfix_args()
  {
    return postfix_args_type{std::tuple<>{}, std::make_tuple(alternate_binary_op{})};
  }
};

} // namespace postfix_args

//------------------------------------------------------------------------------
namespace invoke_reference
{

template <typename input_value_type,
          typename output_value_type = input_value_type>
struct host_synchronous
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

    // Run host synchronous algorithm to generate reference.
    thrust::inclusive_scan(host_input.cbegin(),
                           host_input.cend(),
                           host_output.begin(),
                           std::get<PostfixArgIndices>(
                             THRUST_FWD(postfix_tuple))...);

    // Copy back to device.
    output = host_output;
  }
};

} // namespace invoke_reference

//------------------------------------------------------------------------------
namespace invoke_async
{

struct simple
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
    auto e = thrust::async::inclusive_scan(
      std::get<PrefixArgIndices>(THRUST_FWD(prefix_tuple))...,
      input.cbegin(),
      input.cend(),
      output.begin(),
      std::get<PostfixArgIndices>(THRUST_FWD(postfix_tuple))...);
    return e;
  }
};

} // namespace invoke_async

} // namespace mixin
} // namespace inclusive_scan
} // namespace async
} // namespace testing

#endif // C++14
