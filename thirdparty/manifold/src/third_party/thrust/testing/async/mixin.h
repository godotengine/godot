#pragma once

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <thrust/type_traits/logical_metafunctions.h>

#include <unittest/unittest.h>
#include <unittest/util_async.h>

#include <tuple>
#include <type_traits>

// clang-format off

// This file contains a set of mix-in classes that define an algorithm
// definition for use with test_policy_overloads<algo_def>. The algorithm
// definition describes the details of a thrust::async algorithm invocation:
//
// - Input type and initialization
// - Output type and initialization (supports in-place, too)
// - Postfix arguments that define the algorithm's overload set
// - Abstracted invocation of the async algorithm
// - Abstracted invocation of a reference algorithm
// - Validation of async vs. reference output
// - A description string.
//
// This definition is used by test_policy_overloads to test each overload
// against a reference while injecting a variety of execution policies. This
// validates that each overload behaves correctly according to some reference.
//
// Since much of the algorithm definition is generic and may be reused in
// multiple tests with slight changes, a mix-in system is used to simplify
// the creation of algorithm definitions. The following namespace hierarchy is
// used to organize these generic components:
//
// * testing::async::mixin::
// ** ::input - Input types/values (device vectors, counting iterators, etc)
// ** ::output - Output types/values (device vectors, inplace device vectors,
//                                    discard iterators, etc)
// ** ::postfix_args - Algorithm specific overload sets
// ** ::invoke_reference - Algorithm specific reference invocation
// ** ::invoke_async - Algorithm specific async algo invocation
// ** ::compare_outputs - Compare output values.
//
// Each algorithm should define its own `mixins.h` header to declare algorithm
// specific mixins (e.g. postfix_args, invoke_reference, and invoke_async)
// in a testing::async::<algorithm_name>::mixins namespace structure.
//
// For example, the test.async.exclusive_scan.basic test uses the following
// algorithm definition from mix-ins:
//
// ```
//   #include <async/test_policy_overloads.h>
//   #include <async/mixin.h>
//   #include <async/exclusive_scan/mixin.h>
//   template <typename input_value_type,
//            typename output_value_type   = input_value_type,
//            typename initial_value_type  = input_value_type,
//            typename alternate_binary_op = thrust::maximum<>>
//   struct basic_invoker
//      : testing::async::mixin::input::device_vector<input_value_type>
//      , testing::async::mixin::output::device_vector<output_value_type>
//      , testing::async::exclusive_scan::mixin::postfix_args::
//          all_overloads<initial_value_type, alternate_binary_op>
//      , testing::async::exclusive_scan::mixin::invoke_reference::
//          host_synchronous<input_value_type, output_value_type>
//      , testing::async::exclusive_scan::mixin::invoke_async::basic
//      , testing::async::mixin::compare_outputs::assert_equal_quiet
//   {
//     static std::string description()
//     {
//       return "basic invocation with device vectors";
//     }
//   };
//
//   ...
//
//   testing::async::test_policy_overloads<basic_invoker<T>>::run(num_values);
// ```
//
// The basic_invoker class expands to something similar to the following:
//
// ```
//  template <typename input_value_type,
//            typename output_value_type   = input_value_type,
//            typename initial_value_type  = input_value_type,
//            typename alternate_binary_op = thrust::maximum<>>
//  struct basic_invoker
//  {
//  public:
//
//    static std::string description()
//    {
//      return "basic invocation with device vectors";
//    }
//
//    //-------------------------------------------------------------------------
//    // testing::async::mixin::input::device_vector
//    //
//    // input_type must provide idiomatic definitions of:
//    // - `using iterator = ...;`
//    // - `iterator begin() const { ... }`
//    // - `iterator end() const { ... }`
//    // - `size_t size() const { ... }`
//    using input_type = thrust::device_vector<input_value_type>;
//
//    // Generate an instance of the input:
//    static input_type generate_input(std::size_t num_values)
//    {
//      input_type input(num_values);
//      thrust::sequence(input.begin(), input.end(), 25, 3);
//      return input;
//    }
//
//    //-------------------------------------------------------------------------
//    // testing::async::mixin::output::device_vector
//    //
//    // output_type must provide idiomatic definitions of:
//    // - `using iterator = ...;`
//    // - `iterator begin() { ... }`
//    using output_type = thrust::device_vector<output_value_type>;
//
//    // Generate an instance of the output:
//    // Might be more complicated, eg. fancy iterators, etc
//    static output_type generate_output(std::size_t num_values)
//    {
//      return output_type(num_values);
//    }
//
//    //-------------------------------------------------------------------------
//    // testing::async::exclusive_scan::mixin::postfix_args::all_overloads
//    using postfix_args_type = std::tuple<   // List any extra arg overloads:
//      std::tuple<>,                                       // - no extra args
//      std::tuple<initial_value_type>,                     // - initial_value
//      std::tuple<initial_value_type, alternate_binary_op> // - initial_value, binary_op
//      >;
//
//    // Create instances of the extra arguments to use when invoking the
//    // algorithm:
//    static postfix_args_type generate_postfix_args()
//    {
//      return postfix_args_type{
//        std::tuple<>{},                            // no extra args
//        std::make_tuple(initial_value_type{42}),   // initial_value
//        // initial_value, binary_op:
//        std::make_tuple(initial_value_Type{57}, alternate_binary_op{})
//      };
//    }
//
//    //-------------------------------------------------------------------------
//    //
//    testing::async::exclusive_scan::mixin::invoke_reference::host_synchronous
//    //
//    // Invoke a reference implementation for a single overload as described by
//    // postfix_tuple. This tuple contains instances of any trailing arguments
//    // to pass to the algorithm. The tuple/index_sequence pattern is used to
//    // support a "no extra args" overload, since the parameter pack expansion
//    // will do exactly what we want in all cases.
//    template <typename PostfixArgTuple, std::size_t... PostfixArgIndices>
//    static void invoke_reference(input_type const &input,
//                                 output_type &output,
//                                 PostfixArgTuple &&postfix_tuple,
//                                 std::index_sequence<PostfixArgIndices...>)
//    {
//      // Create host versions of the input/output:
//      thrust::host_vector<input_value_type> host_input(input.cbegin(),
//                                                       input.cend());
//      thrust::host_vector<output_value_type> host_output(host_input.size());
//
//      // Run host synchronous algorithm to generate reference.
//      thrust::exclusive_scan(host_input.cbegin(),
//                             host_input.cend(),
//                             host_output.begin(),
//                             std::get<PostfixArgIndices>(
//                               THRUST_FWD(postfix_tuple))...);
//
//      // Copy back to device.
//      output = host_output;
//    }
//
//    //-------------------------------------------------------------------------
//    // testing::async::mixin::exclusive_scan::mixin::invoke_async::basic
//    //
//    // Invoke the async algorithm for a single overload as described by
//    // the prefix and postfix tuples. These tuples contains instances of any
//    // additional arguments to pass to the algorithm. The tuple/index_sequence
//    // pattern is used to support the "no extra args" overload, since the
//    // parameter pack expansion will do exactly what we want in all cases.
//    // Prefix args are included here (but not for invoke_reference) to allow
//    // the test framework to change the execution policy.
//    // This method must return an event or future.
//    template <typename PrefixArgTuple,
//              std::size_t... PrefixArgIndices,
//              typename PostfixArgTuple,
//              std::size_t... PostfixArgIndices>
//    static auto invoke_async(PrefixArgTuple &&prefix_tuple,
//                             std::index_sequence<PrefixArgIndices...>,
//                             input_type const &input,
//                             output_type &output,
//                             PostfixArgTuple &&postfix_tuple,
//                             std::index_sequence<PostfixArgIndices...>)
//    {
//      output.resize(input.size());
//      auto e = thrust::async::exclusive_scan(
//        std::get<PrefixArgIndices>(THRUST_FWD(prefix_tuple))...,
//        input.cbegin(),
//        input.cend(),
//        output.begin(),
//        std::get<PostfixArgIndices>(THRUST_FWD(postfix_tuple))...);
//      return e;
//    }
//
//    //-------------------------------------------------------------------------
//    // testing::async::mixin::compare_outputs::assert_equal_quiet
//    //
//    // Wait on and validate the event/future (usually with TEST_EVENT_WAIT /
//    // TEST_FUTURE_VALUE_RETRIEVAL), then check that the reference output
//    // matches the testing output.
//    template <typename EventType>
//    static void compare_outputs(EventType &e,
//                                output_type const &ref,
//                                output_type const &test)
//    {
//      TEST_EVENT_WAIT(e);
//      ASSERT_EQUAL_QUIET(ref, test);
//    }
// };
// ```
//
// Similar invokers with slight tweaks are used in other
// async/exclusive_scan/*.cu tests.

// clang-format on

namespace testing
{
namespace async
{
namespace mixin
{

//------------------------------------------------------------------------------
namespace input
{

template <typename value_type>
struct device_vector
{
  using input_type = thrust::device_vector<value_type>;

  static input_type generate_input(std::size_t num_values)
  {
    input_type input(num_values);
    thrust::sequence(input.begin(),
                     input.end(),
                     static_cast<value_type>(1),
                     static_cast<value_type>(1));
    return input;
  }
};

template <typename value_type>
struct counting_iterator_from_0
{
  struct input_type
  {
    using iterator = thrust::counting_iterator<value_type>;

    std::size_t num_values;

    iterator begin() const { return iterator{static_cast<value_type>(0)}; }
    iterator cbegin() const { return iterator{static_cast<value_type>(0)}; }

    iterator end() const { return iterator{static_cast<value_type>(num_values)}; }
    iterator cend() const { return iterator{static_cast<value_type>(num_values)}; }

    std::size_t size() const { return num_values; }
  };

  static input_type generate_input(std::size_t num_values)
  {
    return {num_values};
  }
};

template <typename value_type>
struct counting_iterator_from_1
{
  struct input_type
  {
    using iterator = thrust::counting_iterator<value_type>;

    std::size_t num_values;

    iterator begin() const { return iterator{static_cast<value_type>(1)}; }
    iterator cbegin() const { return iterator{static_cast<value_type>(1)}; }

    iterator end() const { return iterator{static_cast<value_type>(1 + num_values)}; }
    iterator cend() const { return iterator{static_cast<value_type>(1 + num_values)}; }

    std::size_t size() const { return num_values; }
  };

  static input_type generate_input(std::size_t num_values)
  {
    return {num_values};
  }
};

template <typename value_type>
struct constant_iterator_1
{
  struct input_type
  {
    using iterator = thrust::constant_iterator<value_type>;

    std::size_t num_values;

    iterator begin() const { return iterator{static_cast<value_type>(1)}; }
    iterator cbegin() const { return iterator{static_cast<value_type>(1)}; }

    iterator end() const
    {
      return iterator{static_cast<value_type>(1)} + num_values;
    }
    iterator cend() const
    {
      return iterator{static_cast<value_type>(1)} + num_values;
    }

    std::size_t size() const { return num_values; }
  };

  static input_type generate_input(std::size_t num_values)
  {
    return {num_values};
  }
};

} // namespace input

//------------------------------------------------------------------------------
namespace output
{

template <typename value_type>
struct device_vector
{
  using output_type = thrust::device_vector<value_type>;

  template <typename InputType>
  static output_type generate_output(std::size_t num_values,
                                     InputType& /* unused */)
  {
    return output_type(num_values);
  }
};

template <typename value_type>
struct device_vector_reuse_input
{
  using output_type = thrust::device_vector<value_type>&;

  template <typename InputType>
  static output_type generate_output(std::size_t /*num_values*/,
                                     InputType& input)
  {
    return input;
  }
};

struct discard_iterator
{
  struct output_type
  {
    using iterator = thrust::discard_iterator<>;

    iterator begin() const { return thrust::make_discard_iterator(); }
    iterator cbegin() const { return thrust::make_discard_iterator(); }
  };

  template <typename InputType>
  static output_type generate_output(std::size_t /* num_values */,
                                     InputType& /* input */)
  {
    return output_type{};
  }
};

} // namespace output

//------------------------------------------------------------------------------
namespace postfix_args
{
/* Defined per algorithm. Example:
 *
 * // Defines several overloads:
 * // algorithm([policy,] input, output) // no postfix args
 * // algorithm([policy,] input, output, initial_value)
 * // algorithm([policy,] input, output, initial_value, binary_op)
 * template <typename value_type,
 *           typename alternate_binary_op = thrust::maximum<>>
 * struct all_overloads
 * {
 *   using postfix_args_type = std::tuple<     // List any extra arg overloads:
 *     std::tuple<>,                               // - no extra args
 *     std::tuple<value_type>,                     // - initial_value
 *     std::tuple<value_type, alternate_binary_op> // - initial_value, binary_op
 *     >;
 *
 *   static postfix_args_type generate_postfix_args()
 *   {
 *     return postfix_args_type{
 *       std::tuple<>{},                            // no extra args
 *       std::make_tuple(initial_value_type{42}),   // initial_value
 *       // initial_value, binary_op:
 *       std::make_tuple(initial_value_Type{57}, alternate_binary_op{})
 *   }
 * };
 *
 */
}

//------------------------------------------------------------------------------
namespace invoke_reference
{

/* Defined per algorithm. Example:
 *
 * template <typename input_value_type,
 *           typename output_value_type = input_value_type>
 * struct host_synchronous
 * {
 *   template <typename InputType,
 *             typename OutputType,
 *             typename PostfixArgTuple,
 *             std::size_t... PostfixArgIndices>
 *   static void invoke_reference(InputType const& input,
 *                                OutputType& output,
 *                                PostfixArgTuple&& postfix_tuple,
 *                                std::index_sequence<PostfixArgIndices...>)
 *   {
 *     // Create host versions of the input/output:
 *     thrust::host_vector<input_value_type> host_input(input.cbegin(),
 *                                                      input.cend());
 *     thrust::host_vector<output_value_type> host_output(host_input.size());
 *
 *     // Run host synchronous algorithm to generate reference.
 *     // Be sure to call a backend that doesn't use the same underlying
 *     // implementation.
 *     thrust::exclusive_scan(host_input.cbegin(),
 *                            host_input.cend(),
 *                            host_output.begin(),
 *                            std::get<PostfixArgIndices>(
 *                              THRUST_FWD(postfix_tuple))...);
 *
 *     // Copy back to device.
 *     output = host_output;
 *   }
 * };
 *
 */

// Used to save time when testing unverifiable invocations (discard_iterators)
struct noop
{
  template <typename... Ts>
  static void invoke_reference(Ts&&...)
  {}
};

} // namespace invoke_reference

//------------------------------------------------------------------------------
namespace invoke_async
{

/* Defined per algorithm. Example:
 *
 * struct basic
 * {
 *   template <typename PrefixArgTuple,
 *             std::size_t... PrefixArgIndices,
 *             typename InputType,
 *             typename OutputType,
 *             typename PostfixArgTuple,
 *             std::size_t... PostfixArgIndices>
 *   static auto invoke_async(PrefixArgTuple&& prefix_tuple,
 *                            std::index_sequence<PrefixArgIndices...>,
 *                            InputType const& input,
 *                            OutputType& output,
 *                            PostfixArgTuple&& postfix_tuple,
 *                            std::index_sequence<PostfixArgIndices...>)
 *   {
 *     auto e = thrust::async::exclusive_scan(
 *       std::get<PrefixArgIndices>(THRUST_FWD(prefix_tuple))...,
 *       input.cbegin(),
 *       input.cend(),
 *       output.begin(),
 *       std::get<PostfixArgIndices>(THRUST_FWD(postfix_tuple))...);
 *     return e;
 *   }
 * };
 */

} // namespace invoke_async

//------------------------------------------------------------------------------
namespace compare_outputs
{

namespace detail
{

void basic_event_validation(thrust::device_event& e)
{
  TEST_EVENT_WAIT(e);
}

template <typename T>
void basic_event_validation(thrust::device_future<T>& f)
{
  TEST_FUTURE_VALUE_RETRIEVAL(f);
}

} // namespace detail

struct assert_equal
{
  template <typename EventType, typename OutputType>
  static void compare_outputs(EventType& e,
                              OutputType const& ref,
                              OutputType const& test)
  {
    detail::basic_event_validation(e);
    ASSERT_EQUAL(ref, test);
  }
};

struct assert_almost_equal
{
  template <typename EventType, typename OutputType>
  static void compare_outputs(EventType& e,
                              OutputType const& ref,
                              OutputType const& test)
  {
    detail::basic_event_validation(e);
    ASSERT_ALMOST_EQUAL(ref, test);
  }
};

// Does an 'almost_equal' comparison for floating point types. Since fp
// addition is non-associative, this is sometimes necessary.
struct assert_almost_equal_if_fp
{
private:
  template <typename EventType, typename OutputType>
  static void compare_outputs_impl(EventType& e,
                                   OutputType const& ref,
                                   OutputType const& test,
                                   std::false_type /* is_floating_point */)
  {
    detail::basic_event_validation(e);
    ASSERT_EQUAL(ref, test);
  }

  template <typename EventType, typename OutputType>
  static void compare_outputs_impl(EventType& e,
                                   OutputType const& ref,
                                   OutputType const& test,
                                   std::true_type /* is_floating_point */)
  {
    detail::basic_event_validation(e);
    ASSERT_ALMOST_EQUAL(ref, test);
  }

public:
  template <typename EventType, typename OutputType>
  static void compare_outputs(EventType& e,
                              OutputType const& ref,
                              OutputType const& test)
  {
    using value_type = typename OutputType::value_type;
    compare_outputs_impl(e, ref, test, std::is_floating_point<value_type>{});
  }
};

struct assert_equal_quiet
{
  template <typename EventType, typename OutputType>
  static void compare_outputs(EventType& e,
                              OutputType const& ref,
                              OutputType const& test)
  {
    detail::basic_event_validation(e);
    ASSERT_EQUAL_QUIET(ref, test);
  }
};

// Does an 'almost_equal' comparison for floating point types, since fp
// addition is non-associative
struct assert_almost_equal_if_fp_quiet
{
private:
  template <typename EventType, typename OutputType>
  static void compare_outputs_impl(EventType& e,
                                   OutputType const& ref,
                                   OutputType const& test,
                                   std::false_type /* is_floating_point */)
  {
    detail::basic_event_validation(e);
    ASSERT_EQUAL_QUIET(ref, test);
  }

  template <typename EventType, typename OutputType>
  static void compare_outputs_impl(EventType& e,
                                   OutputType const& ref,
                                   OutputType const& test,
                                   std::true_type /* is_floating_point */)
  {
    detail::basic_event_validation(e);
    ASSERT_ALMOST_EQUAL(ref, test);
  }

public:
  template <typename EventType, typename OutputType>
  static void compare_outputs(EventType& e,
                              OutputType const& ref,
                              OutputType const& test)
  {
    using value_type = typename OutputType::value_type;
    compare_outputs_impl(e, ref, test, std::is_floating_point<value_type>{});
  }
};

// Used to save time when testing unverifiable invocations (discard_iterators).
// Just does basic validation of the future/event.
struct noop
{
  template <typename EventType, typename... Ts>
  static void compare_outputs(EventType &e, Ts&&...)
  {
    detail::basic_event_validation(e);
  }
};

} // namespace compare_outputs

} // namespace mixin
} // namespace async
} // namespace testing

#endif // C++14
