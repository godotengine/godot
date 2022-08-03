#include <unittest/unittest.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/memory.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>


template<typename T1, typename T2>
bool are_same_type(const T1 &, const T2 &)
{
  return false;
}


template<typename T>
bool are_same_type(const T &, const T &)
{
  return true;
}


void TestSelectSystemCudaToCpp()
{
  using thrust::system::detail::generic::select_system;

  thrust::cuda::tag cuda_tag;
  thrust::cpp::tag cpp_tag;
  thrust::cuda_cub::cross_system<thrust::cuda::tag,thrust::cpp::tag> cuda_to_cpp(cuda_tag, cpp_tag);

  // select_system(cuda::tag, thrust::host_system_tag) should return cuda_to_cpp
  bool is_cuda_to_cpp = are_same_type(cuda_to_cpp, select_system(cuda_tag, cpp_tag));
  ASSERT_EQUAL(true, is_cuda_to_cpp);
}
DECLARE_UNITTEST(TestSelectSystemCudaToCpp);


template<typename Iterator>
__global__ void get_temporary_buffer_kernel(size_t n, Iterator result)
{
  *result = thrust::get_temporary_buffer<int>(thrust::seq, n);
}


template<typename Pointer>
__global__ void return_temporary_buffer_kernel(Pointer ptr, std::ptrdiff_t n)
{
  thrust::return_temporary_buffer(thrust::seq, ptr, n);
}


void TestGetTemporaryBufferDeviceSeq()
{
  const std::ptrdiff_t n = 9001;

  typedef thrust::pointer<int, thrust::detail::seq_t> pointer;
  typedef thrust::pair<pointer, std::ptrdiff_t> ptr_and_sz_type;
  thrust::device_vector<ptr_and_sz_type> d_result(1);
  
  get_temporary_buffer_kernel<<<1,1>>>(n, d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ptr_and_sz_type ptr_and_sz = d_result[0];

  if(ptr_and_sz.second > 0)
  {
    ASSERT_EQUAL(ptr_and_sz.second, n);

    const int ref_val = 13;
    thrust::device_vector<int> ref(n, ref_val);

    thrust::fill_n(thrust::device, ptr_and_sz.first, n, ref_val);

    ASSERT_EQUAL(true, thrust::all_of(thrust::device, ptr_and_sz.first, ptr_and_sz.first + n, thrust::placeholders::_1 == ref_val));

    return_temporary_buffer_kernel<<<1,1>>>(ptr_and_sz.first, ptr_and_sz.second);
    {
      cudaError_t const err = cudaDeviceSynchronize();
      ASSERT_EQUAL(cudaSuccess, err);
    }
  }
}
DECLARE_UNITTEST(TestGetTemporaryBufferDeviceSeq);


template<typename Iterator>
__global__ void malloc_kernel(size_t n, Iterator result)
{
  *result = static_cast<int*>(thrust::malloc(thrust::seq, sizeof(int) * n).get());
}


template<typename Pointer>
__global__ void free_kernel(Pointer ptr)
{
  thrust::free(thrust::seq, ptr);
}


void TestMallocDeviceSeq()
{
  const std::ptrdiff_t n = 9001;

  typedef thrust::pointer<int, thrust::detail::seq_t> pointer;
  thrust::device_vector<pointer> d_result(1);
  
  malloc_kernel<<<1,1>>>(n, d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  pointer ptr = d_result[0];

  if(ptr.get() != 0)
  {
    const int ref_val = 13;
    thrust::device_vector<int> ref(n, ref_val);

    thrust::fill_n(thrust::device, ptr, n, ref_val);

    ASSERT_EQUAL(true, thrust::all_of(thrust::device, ptr, ptr + n, thrust::placeholders::_1 == ref_val));

    free_kernel<<<1,1>>>(ptr);
    {
      cudaError_t const err = cudaDeviceSynchronize();
      ASSERT_EQUAL(cudaSuccess, err);
    }
  }
}
DECLARE_UNITTEST(TestMallocDeviceSeq);

