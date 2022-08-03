#include <unittest/unittest.h>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>

void TestEqualRangeOnStream()
{ // Regression test for GH issue #921 (nvbug 2173437)
  typedef typename thrust::device_vector<int> vector_t;
  typedef typename vector_t::iterator iterator_t;
  typedef thrust::pair<iterator_t, iterator_t> result_t;

  vector_t input(10);
  thrust::sequence(thrust::device, input.begin(), input.end(), 0);
  cudaStream_t stream = 0;
  result_t result = thrust::equal_range(thrust::cuda::par.on(stream),
                                        input.begin(), input.end(),
                                        5);

  ASSERT_EQUAL(5, thrust::distance(input.begin(), result.first));
  ASSERT_EQUAL(6, thrust::distance(input.begin(), result.second));
}
DECLARE_UNITTEST(TestEqualRangeOnStream);
