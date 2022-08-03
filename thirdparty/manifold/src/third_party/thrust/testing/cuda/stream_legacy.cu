#include <unittest/unittest.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/util.h>

#include <thread>

void verify_stream()
{
  auto exec = thrust::device;
  auto stream = thrust::cuda_cub::stream(exec);
  ASSERT_EQUAL(stream, cudaStreamLegacy);
}

void TestLegacyDefaultStream()
{
  verify_stream();

  std::thread t(verify_stream);
  t.join();
}
DECLARE_UNITTEST(TestLegacyDefaultStream);
