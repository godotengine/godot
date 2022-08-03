#include <unittest/unittest.h>

#include <thrust/complex.h>
#include <thrust/detail/preprocessor.h>
#include <thrust/detail/alignment.h>

#include <cuda_fp16.h>

template <typename T, typename VectorT>
void TestComplexAlignment()
{
  THRUST_STATIC_ASSERT(
    sizeof(thrust::complex<T>) == sizeof(VectorT)
  );
  THRUST_STATIC_ASSERT(
    THRUST_ALIGNOF(thrust::complex<T>) == THRUST_ALIGNOF(VectorT)
  );

  THRUST_STATIC_ASSERT(
    sizeof(thrust::complex<T const>) == sizeof(VectorT)
  );
  THRUST_STATIC_ASSERT(
    THRUST_ALIGNOF(thrust::complex<T const>) == THRUST_ALIGNOF(VectorT)
  );
}
DECLARE_UNITTEST_WITH_NAME(
  THRUST_PP_EXPAND_ARGS(TestComplexAlignment<char, char2>)
, TestComplexCharAlignment
);
DECLARE_UNITTEST_WITH_NAME(
  THRUST_PP_EXPAND_ARGS(TestComplexAlignment<short, short2>)
, TestComplexShortAlignment
);
DECLARE_UNITTEST_WITH_NAME(
  THRUST_PP_EXPAND_ARGS(TestComplexAlignment<int, int2>)
, TestComplexIntAlignment
);
DECLARE_UNITTEST_WITH_NAME(
  THRUST_PP_EXPAND_ARGS(TestComplexAlignment<long, long2>)
, TestComplexLongAlignment
);
DECLARE_UNITTEST_WITH_NAME(
  THRUST_PP_EXPAND_ARGS(TestComplexAlignment<__half, __half2>)
, TestComplexHalfAlignment
);
DECLARE_UNITTEST_WITH_NAME(
  THRUST_PP_EXPAND_ARGS(TestComplexAlignment<float, float2>)
, TestComplexFloatAlignment
);
DECLARE_UNITTEST_WITH_NAME(
  THRUST_PP_EXPAND_ARGS(TestComplexAlignment<double, double2>)
, TestComplexDoubleAlignment
);
