// Wrap thrust and cub in different enclosing namespaces
// (In practice, you probably want these to be the same, in which case just
// set THRUST_CUB_WRAPPED_NAMESPACE to set both).
#define THRUST_WRAPPED_NAMESPACE wrap_thrust
#define CUB_WRAPPED_NAMESPACE    wrap_cub

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <unittest/unittest.h>

// Test that we can use a few common utilities and algorithms from a wrapped
// namespace at runtime. More extensive testing is performed by the header
// tests and the check_namespace.cmake test.
void TestWrappedNamespace()
{
  const std::size_t n = 2048;

  const auto in_1_begin =
    ::wrap_thrust::thrust::make_constant_iterator<int>(12);
  const auto in_2_begin =
    ::wrap_thrust::thrust::make_counting_iterator<int>(1024);

  // Check that the qualifier resolves properly:
  THRUST_NS_QUALIFIER::device_vector<int> d_out(n);

  ::wrap_thrust::thrust::transform(in_1_begin,
                                   in_1_begin + n,
                                   in_2_begin,
                                   d_out.begin(),
                                   ::wrap_thrust::thrust::plus<>{});

  ::wrap_thrust::thrust::host_vector<int> h_out(d_out);

  for (std::size_t i = 0; i < n; ++i)
  {
    ASSERT_EQUAL(h_out[i], static_cast<int>(i) + 1024 + 12);
  }
}
DECLARE_UNITTEST(TestWrappedNamespace);
