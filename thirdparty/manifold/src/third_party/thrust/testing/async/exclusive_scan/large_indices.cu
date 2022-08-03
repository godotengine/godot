#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2014

#include <async/test_policy_overloads.h>

#include <async/exclusive_scan/mixin.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/optional.h>

#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/iterator_facade_category.h>

#include <cstdint>

// This test is an adaptation of TestInclusiveScanWithBigIndices from scan.cu.

namespace
{

// Fake iterator that asserts
// (a) it is written with a sequence and
// (b) a defined maximum value is written at some point
//
// This allows us to test very large problem sizes without actually allocating
// large amounts of memory that would exceed most devices' capacity.
struct assert_sequence_iterator
{
  using value_type      = std::int64_t;
  using difference_type = std::int64_t;

  // Defined for thrust::iterator_traits:
  using pointer           = value_type*;
  using reference         = assert_sequence_iterator; // weird but convenient
  using iterator_category =
    typename thrust::detail::iterator_facade_category<
      thrust::device_system_tag,
      thrust::random_access_traversal_tag,
      value_type,
      reference>::type;

  std::int64_t expected{0};
  std::int64_t max{0};
  mutable thrust::device_ptr<bool> found_max{nullptr};
  mutable thrust::device_ptr<bool> unexpected_value{nullptr};

  // Should be called on the first iterator generated. This needs to be
  // done explicitly from the host.
  void initialize_shared_state()
  {
    found_max        = thrust::device_malloc<bool>(1);
    unexpected_value = thrust::device_malloc<bool>(1);
    *found_max        = false;
    *unexpected_value = false;
  }

  // Should be called only once on the initialized iterator. This needs to be
  // done explicitly from the host.
  void free_shared_state() const
  {
    thrust::device_free(found_max);
    thrust::device_free(unexpected_value);
    found_max        = nullptr;
    unexpected_value = nullptr;
  }

  __host__ __device__ assert_sequence_iterator operator+(difference_type i) const
  {
    return clone(expected + i);
  }

  __host__ __device__ reference operator[](difference_type i) const
  {
    return clone(expected + i);
  }

  // Some weirdness, this iterator acts like its own reference
  __device__ assert_sequence_iterator operator=(value_type val)
  {
    if (val != expected)
    {
      printf("Error: expected %lld, got %lld\n", expected, val);
      *unexpected_value = true;
    }
    else if (val == max)
    {
      *found_max = true;
    }

    return *this;
  }

private:
  __host__ __device__
  assert_sequence_iterator clone(value_type new_expected) const
  {
    return {new_expected, max, found_max, unexpected_value};
  }
};

// output mixin that generates assert_sequence_iterators.
// Must be paired with validate_assert_sequence_iterators mixin to free
// shared state.
struct assert_sequence_output
{
  struct output_type
  {
    using iterator = assert_sequence_iterator;

    iterator iter;

    explicit output_type(iterator&& it)
        : iter{std::move(it)}
    {
      iter.initialize_shared_state();
    }

    ~output_type()
    {
      iter.free_shared_state();
    }

    iterator begin() { return iter; }
  };

  template <typename InputType>
  static output_type generate_output(std::size_t num_values, InputType&)
  {
    using value_type = typename assert_sequence_iterator::value_type;
    assert_sequence_iterator it{0,
                                // minus one bc exclusive scan:
                                static_cast<value_type>(num_values - 1),
                                nullptr,
                                nullptr};
    return output_type{std::move(it)};
  }
};

struct validate_assert_sequence_iterators
{
  using output_t = assert_sequence_output::output_type;

  template <typename EventType>
  static void compare_outputs(EventType& e,
                              output_t const&,
                              output_t const& test)
  {
    testing::async::mixin::compare_outputs::detail::basic_event_validation(e);

    ASSERT_EQUAL(*test.iter.unexpected_value, false);
    ASSERT_EQUAL(*test.iter.found_max, true);
  }
};

//------------------------------------------------------------------------------
// Overloads without custom binary operators use thrust::plus<>, so use
// constant input iterator to generate the output sequence:
struct default_bin_op_overloads
{
  using postfix_args_type = std::tuple< // List any extra arg overloads:
    std::tuple<>,                       // - no extra args
    std::tuple<uint64_t>                // - initial_value
    >;

  static postfix_args_type generate_postfix_args()
  {
    return postfix_args_type{std::tuple<>{}, std::tuple<uint64_t>{0}};
  }
};

struct default_bin_op_invoker
    : testing::async::mixin::input::constant_iterator_1<std::int64_t>
    , assert_sequence_output
    , default_bin_op_overloads
    , testing::async::mixin::invoke_reference::noop
    , testing::async::exclusive_scan::mixin::invoke_async::simple
    , validate_assert_sequence_iterators
{
  static std::string description()
  {
    return "test large array indices with default binary operator";
  }
};

} // anon namespace

void test_large_indices_default_scan_op()
{
  // Test problem sizes around signed/unsigned int max:
  testing::async::test_policy_overloads<default_bin_op_invoker>::run(1ll << 30);
  testing::async::test_policy_overloads<default_bin_op_invoker>::run(1ll << 31);
  testing::async::test_policy_overloads<default_bin_op_invoker>::run(1ll << 32);
  testing::async::test_policy_overloads<default_bin_op_invoker>::run(1ll << 33);
}
DECLARE_UNITTEST(test_large_indices_default_scan_op);

namespace
{

//------------------------------------------------------------------------------
// Generate the output sequence using counting iterators and thrust::max<> for
// custom operator overloads.
struct custom_bin_op_overloads
{
  using postfix_args_type = std::tuple<     // List any extra arg overloads:
    std::tuple<uint64_t, thrust::maximum<>> // - initial_value, binop
  >;

  static postfix_args_type generate_postfix_args()
  {
    return postfix_args_type{std::make_tuple(0, thrust::maximum<>{})};
  }
};

struct custom_bin_op_invoker
  : testing::async::mixin::input::counting_iterator_from_1<std::int64_t>
    , assert_sequence_output
    , custom_bin_op_overloads
    , testing::async::mixin::invoke_reference::noop
    , testing::async::exclusive_scan::mixin::invoke_async::simple
    , validate_assert_sequence_iterators
{
  static std::string description()
  {
    return "test large array indices with custom binary operator";
  }
};

} // namespace

void test_large_indices_custom_scan_op()
{
  // Test problem sizes around signed/unsigned int max:
  testing::async::test_policy_overloads<custom_bin_op_invoker>::run(1ll << 30);
  testing::async::test_policy_overloads<custom_bin_op_invoker>::run(1ll << 31);
  testing::async::test_policy_overloads<custom_bin_op_invoker>::run(1ll << 32);
  testing::async::test_policy_overloads<custom_bin_op_invoker>::run(1ll << 33);
}
DECLARE_UNITTEST(test_large_indices_custom_scan_op);

#endif // C++14
