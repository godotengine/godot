#pragma once

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/device_allocator.h>
#include <thrust/future.h>

#include <unittest/unittest.h>

#include <string>

// TODO Cover these cases from testing/async_reduce.cu:
//   - [x] test_async_reduce_after ("after_future" in test_policy_overloads)
//   - [ ] test_async_reduce_on_then_after (KNOWN_FAILURE, see #1195)
//     - [ ] all the child variants (e.g. with allocator) too
//   - [ ] test_async_copy_then_reduce (Need to figure out how to fit this in)
//   - [ ] test_async_reduce_caching (only useful when returning future)

namespace testing
{

namespace async
{

// Tests that policies are handled correctly for all overloads of an async
// algorithm.
//
// The AlgoDef parameter type defines an async algorithm, its overloads, and
// abstracts its invocation. See the async/mixins.h for a documented example of
// this interface and some convenience mixins that can be used to construct a
// definition quickly.
//
// The AlgoDef interface is used to run several tests of the algorithm,
// exhaustively testing all overloads for algorithm correctness and proper
// policy handling.
//
// ## Basic tests
//
// In the basic tests, each overload is called repeatedly with:
// 1) No policy
// 2) thrust::device
// 3) thrust::device(thrust::device_allocator<void>)
// 4) thrust::device.on(stream)
// 5) thrust::device(thrust::device_allocator<void>).on(stream)
//
// The output of the async algorithm is compared against a reference output,
// and the returned event/future is tested to make sure it holds a reference to
// the expected stream.
//
// ## After Future tests
//
// The after_future tests check that the future/event returned from an algorithm
// behaves properly when consumed by a policy's `.after` method.
template <typename AlgoDef>
struct test_policy_overloads
{
  using algo_def          = AlgoDef;
  using input_type        = typename algo_def::input_type;
  using output_type       = typename algo_def::output_type;
  using postfix_args_type = typename algo_def::postfix_args_type;

  static constexpr std::size_t num_postfix_arg_sets =
    std::tuple_size<postfix_args_type>::value;

  // Main entry point; call this from a unit test function.
  static void run(std::size_t num_values)
  {
    test_postfix_overloads(num_values);
  }

private:
  template <std::size_t Size>
  using size_const = std::integral_constant<std::size_t, Size>;

  //----------------------------------------------------------------------------
  // Recursively call sub tests for each overload set in postfix_args:
  template <std::size_t PostfixIdx = 0>
  static void test_postfix_overloads(std::size_t const num_values,
                                     size_const<PostfixIdx> = {})
  {
    static_assert(PostfixIdx < num_postfix_arg_sets, "Internal error.");

    run_basic_policy_tests<PostfixIdx>(num_values);
    run_after_future_tests<PostfixIdx>(num_values);

    // Recurse to test next round of overloads:
    test_postfix_overloads(num_values, size_const<PostfixIdx + 1>{});
  }

  static void test_postfix_overloads(std::size_t const,
                                     size_const<num_postfix_arg_sets>)
  {
    // terminal case, no-op
  }

  //----------------------------------------------------------------------------
  // For the specified postfix overload set, test the algorithm with several
  // different policy configurations.
  template <std::size_t PostfixIdx>
  static void run_basic_policy_tests(std::size_t const num_values)
  {
    // When a policy uses the default stream, the algorithm implementation
    // should spawn a new stream in the returned event:
    auto using_default_stream = [](auto& e) {
      ASSERT_NOT_EQUAL(thrust::cuda_cub::default_stream(),
                       e.stream().native_handle());
    };

    // When a policy uses a non-default stream, the implementation should pass
    // the stream through to the output:
    thrust::system::cuda::detail::unique_stream test_stream{};
    auto using_test_stream = [&test_stream](auto& e) {
      ASSERT_EQUAL(test_stream.native_handle(), e.stream().native_handle());
    };

    // Test the different types of policies:
    basic_policy_test<PostfixIdx>("(no policy)",
                                   std::make_tuple(),
                                   using_default_stream,
                                   num_values);

    basic_policy_test<PostfixIdx>("thrust::device",
                                   std::make_tuple(thrust::device),
                                   using_default_stream,
                                   num_values);

    basic_policy_test<PostfixIdx>(
      "thrust::device(thrust::device_allocator<void>{})",
      std::make_tuple(thrust::device(thrust::device_allocator<void>{})),
      using_default_stream,
      num_values);

    basic_policy_test<PostfixIdx>("thrust::device.on(test_stream.get())",
                                   std::make_tuple(
                                     thrust::device.on(test_stream.get())),
                                   using_test_stream,
                                   num_values);

    basic_policy_test<PostfixIdx>(
      "thrust::device(thrust::device_allocator<void>{}).on(test_stream.get())",
      std::make_tuple(
        thrust::device(thrust::device_allocator<void>{}).on(test_stream.get())),
      using_test_stream,
      num_values);
  }

  // Invoke the algorithm multiple times with the provided policy and validate
  // the results.
  template <std::size_t PostfixIdx,
            typename PrefixArgTuple,
            typename ValidateEvent>
  static void basic_policy_test(std::string const &policy_desc,
                                PrefixArgTuple &&prefix_tuple_ref,
                                ValidateEvent const &validate,
                                std::size_t num_values)
  try
  {
    // Sink the prefix tuple into a const local so it can be safely passed to
    // multiple invocations without worrying about potential modifications.
    using prefix_tuple_type = thrust::remove_cvref_t<PrefixArgTuple>;
    prefix_tuple_type const prefix_tuple = THRUST_FWD(prefix_tuple_ref);

    using postfix_tuple_type =
      std::tuple_element_t<PostfixIdx, postfix_args_type>;
    postfix_tuple_type const postfix_tuple = get_postfix_tuple<PostfixIdx>();

    // Generate index sequences for the tuples:
    constexpr auto prefix_tuple_size  = std::tuple_size<prefix_tuple_type>{};
    constexpr auto postfix_tuple_size = std::tuple_size<postfix_tuple_type>{};
    using prefix_index_seq  = std::make_index_sequence<prefix_tuple_size>;
    using postfix_index_seq = std::make_index_sequence<postfix_tuple_size>;

    // Use unique, non-const inputs for each invocation to support in-place
    // algo_def configurations.
    input_type input_a   = algo_def::generate_input(num_values);
    input_type input_b   = algo_def::generate_input(num_values);
    input_type input_c   = algo_def::generate_input(num_values);
    input_type input_d   = algo_def::generate_input(num_values);
    input_type input_ref = algo_def::generate_input(num_values);

    output_type output_a   = algo_def::generate_output(num_values, input_a);
    output_type output_b   = algo_def::generate_output(num_values, input_b);
    output_type output_c   = algo_def::generate_output(num_values, input_c);
    output_type output_d   = algo_def::generate_output(num_values, input_d);
    output_type output_ref = algo_def::generate_output(num_values, input_ref);

    // Invoke multiple overlapping async algorithms, capturing their outputs
    // and events/futures:
    auto e_a = algo_def::invoke_async(prefix_tuple,
                                      prefix_index_seq{},
                                      input_a,
                                      output_a,
                                      postfix_tuple,
                                      postfix_index_seq{});
    auto e_b = algo_def::invoke_async(prefix_tuple,
                                      prefix_index_seq{},
                                      input_b,
                                      output_b,
                                      postfix_tuple,
                                      postfix_index_seq{});
    auto e_c = algo_def::invoke_async(prefix_tuple,
                                      prefix_index_seq{},
                                      input_c,
                                      output_c,
                                      postfix_tuple,
                                      postfix_index_seq{});
    auto e_d = algo_def::invoke_async(prefix_tuple,
                                      prefix_index_seq{},
                                      input_d,
                                      output_d,
                                      postfix_tuple,
                                      postfix_index_seq{});

    // Let reference calc overlap with async testing:
    algo_def::invoke_reference(input_ref,
                               output_ref,
                               postfix_tuple,
                               postfix_index_seq{});

    // These wait on the e_X events:
    algo_def::compare_outputs(e_a, output_ref, output_a);
    algo_def::compare_outputs(e_b, output_ref, output_b);
    algo_def::compare_outputs(e_c, output_ref, output_c);
    algo_def::compare_outputs(e_d, output_ref, output_d);

    validate(e_a);
    validate(e_b);
    validate(e_c);
    validate(e_d);
  }
  catch (unittest::UnitTestException &exc)
  {
    // Append some identifying information to the exception to help with
    // debugging:
    using overload_t = std::tuple_element_t<PostfixIdx, postfix_args_type>;

    std::string const overload_desc =
      unittest::demangle(typeid(overload_t).name());
    std::string const input_desc =
      unittest::demangle(typeid(input_type).name());
    std::string const output_desc =
      unittest::demangle(typeid(output_type).name());

    exc << "\n"
        << " - algo_def::description = " << algo_def::description() << "\n"
        << " - test = basic_policy\n"
        << " - policy = " << policy_desc << "\n"
        << " - input_type = " << input_desc << "\n"
        << " - output_type = " << output_desc << "\n"
        << " - tuple of trailing arguments = " << overload_desc << "\n"
        << " - num_values = " << num_values;
    throw;
  }

  //----------------------------------------------------------------------------
  // Test .after(event/future) handling:
  template <std::size_t PostfixIdx>
  static void run_after_future_tests(std::size_t const num_values)
  try
  {
    using postfix_tuple_type =
    std::tuple_element_t<PostfixIdx, postfix_args_type>;
    postfix_tuple_type const postfix_tuple = get_postfix_tuple<PostfixIdx>();

    // Generate index sequences for the tuples. Prefix size always = 1 here,
    // since the async algorithms are always invoked with a single prefix
    // arg (the execution policy) here.
    constexpr auto postfix_tuple_size = std::tuple_size<postfix_tuple_type>{};
    using prefix_index_seq  = std::make_index_sequence<1>;
    using postfix_index_seq = std::make_index_sequence<postfix_tuple_size>;

    // Use unique, non-const inputs for each invocation to support in-place
    // algo_def configurations.
    input_type input_a   = algo_def::generate_input(num_values);
    input_type input_b   = algo_def::generate_input(num_values);
    input_type input_c   = algo_def::generate_input(num_values);
    input_type input_tmp = algo_def::generate_input(num_values);
    input_type input_ref = algo_def::generate_input(num_values);

    output_type output_a   = algo_def::generate_output(num_values, input_a);
    output_type output_b   = algo_def::generate_output(num_values, input_b);
    output_type output_c   = algo_def::generate_output(num_values, input_c);
    output_type output_tmp = algo_def::generate_output(num_values, input_tmp);
    output_type output_ref = algo_def::generate_output(num_values, input_ref);

    auto e_a = algo_def::invoke_async(std::make_tuple(thrust::device),
                                      prefix_index_seq{},
                                      input_a,
                                      output_a,
                                      postfix_tuple,
                                      postfix_index_seq{});
    ASSERT_EQUAL(true, e_a.valid_stream());
    auto const stream_a = e_a.stream().native_handle();

    // Execution on default stream should create a new stream in the result:
    ASSERT_NOT_EQUAL_QUIET(thrust::cuda_cub::default_stream(), stream_a);

    //--------------------------------------------------------------------------
    // Test event consumption when the event is an rvalue.
    //--------------------------------------------------------------------------
    // Using `forward_as_tuple` instead of `make_tuple` to explicitly control
    // value categories.
    // Explicitly order this invocation after e_a:
    auto e_b =
      algo_def::invoke_async(std::forward_as_tuple(thrust::device.after(e_a)),
                             prefix_index_seq{},
                             input_b,
                             output_b,
                             postfix_tuple,
                             postfix_index_seq{});
    ASSERT_EQUAL(true, e_b.valid_stream());
    auto const stream_b = e_b.stream().native_handle();

    // Second invocation should use same stream as before:
    ASSERT_EQUAL_QUIET(stream_a, stream_b);

    // Verify that double consumption of e_a produces an exception:
    ASSERT_THROWS_EQUAL(auto x = algo_def::invoke_async(
                          std::forward_as_tuple(thrust::device.after(e_a)),
                          prefix_index_seq{},
                          input_tmp,
                          output_tmp,
                          postfix_tuple,
                          postfix_index_seq{});
                        THRUST_UNUSED_VAR(x),
                        thrust::event_error,
                        thrust::event_error(thrust::event_errc::no_state));

    //--------------------------------------------------------------------------
    // Test event consumption when the event is an lvalue
    //--------------------------------------------------------------------------
    // Explicitly order this invocation after e_b:
    auto policy_after_e_b = thrust::device.after(e_b);
    auto policy_after_e_b_tuple = std::forward_as_tuple(policy_after_e_b);
    auto e_c =
      algo_def::invoke_async(policy_after_e_b_tuple,
                             prefix_index_seq{},
                             input_c,
                             output_c,
                             postfix_tuple,
                             postfix_index_seq{});
    ASSERT_EQUAL(true, e_c.valid_stream());
    auto const stream_c = e_c.stream().native_handle();

    // Should use same stream as e_b:
    ASSERT_EQUAL_QUIET(stream_b, stream_c);

    // Verify that double consumption of e_b produces an exception:
    ASSERT_THROWS_EQUAL(
      auto x = algo_def::invoke_async(policy_after_e_b_tuple,
                                      prefix_index_seq{},
                                      input_tmp,
                                      output_tmp,
                                      postfix_tuple,
                                      postfix_index_seq{});
      THRUST_UNUSED_VAR(x),
      thrust::event_error,
      thrust::event_error(thrust::event_errc::no_state));

    // Let reference calc overlap with async testing:
    algo_def::invoke_reference(input_ref,
                               output_ref,
                               postfix_tuple,
                               postfix_index_seq{});

    // Validate results
    // Use e_c for all three checks -- e_a and e_b will not pass the event
    // checks since their streams were stolen by dependencies.
    algo_def::compare_outputs(e_c, output_ref, output_a);
    algo_def::compare_outputs(e_c, output_ref, output_b);
    algo_def::compare_outputs(e_c, output_ref, output_c);
  }
  catch (unittest::UnitTestException &exc)
  {
    // Append some identifying information to the exception to help with
    // debugging:
    using postfix_t = std::tuple_element_t<PostfixIdx, postfix_args_type>;

    std::string const postfix_desc =
      unittest::demangle(typeid(postfix_t).name());
    std::string const input_desc =
      unittest::demangle(typeid(input_type).name());
    std::string const output_desc =
      unittest::demangle(typeid(output_type).name());

    exc << "\n"
        << " - algo_def::description = " << algo_def::description() << "\n"
        << " - test = after_future\n"
        << " - input_type = " << input_desc << "\n"
        << " - output_type = " << output_desc << "\n"
        << " - tuple of trailing arguments = " << postfix_desc << "\n"
        << " - num_values = " << num_values;
    throw;
  }

  //----------------------------------------------------------------------------
  // Various helper functions:
  template <std::size_t PostfixIdx>
  static auto get_postfix_tuple()
  {
    return std::get<PostfixIdx>(algo_def::generate_postfix_args());
  }
};

} // namespace async
} // namespace testing

#endif // C++14
