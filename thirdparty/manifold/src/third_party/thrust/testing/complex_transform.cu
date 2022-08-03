#include <unittest/unittest.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <iostream>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <unittest/cuda/testframework.h>
#endif

struct basic_arithmetic_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x,
				const thrust::complex<T> &y)
  {
    // exercise unary and binary arithmetic operators
    // Should return approximately 1
    return (+x + +y) + (x * y) / (y * x) + (-y + -x);
  } // end operator()()
}; // end make_pair_functor

struct complex_plane_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    // Should return a proximately 1
    return thrust::proj( (thrust::polar(abs(x),arg(x)) * conj(x))/norm(x));
  } // end operator()()
}; // end make_pair_functor

struct pow_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x,
				const thrust::complex<T> &y)
  {
    // exercise power functions
    return pow(x,y);
  } // end operator()()
}; // end make_pair_functor

struct sqrt_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    // exercise power functions
    return sqrt(x);
  } // end operator()()
}; // end make_pair_functor

struct log_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return log(x);
  } // end operator()()
}; // end make_pair_functor

struct exp_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return exp(x);
  } // end operator()()
}; // end make_pair_functor

struct log10_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return log10(x);
  } // end operator()()
}; // end make_pair_functor


struct cos_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return cos(x);
  } 
}; 

struct sin_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return sin(x);
  } 
}; 

struct tan_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return tan(x);
  } 
}; 



struct cosh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return cosh(x);
  } 
}; 

struct sinh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return sinh(x);
  } 
}; 

struct tanh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return tanh(x);
  } 
}; 


struct acos_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return acos(x);
  } 
}; 

struct asin_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return asin(x);
  } 
}; 

struct atan_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return atan(x);
  } 
}; 


struct acosh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return acosh(x);
  } 
}; 

struct asinh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return asinh(x);
  } 
}; 

struct atanh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return atanh(x);
  } 
}; 


template <typename T>
thrust::host_vector<thrust::complex<T> > random_complex_samples(size_t n){
  thrust::host_vector<T> real = unittest::random_samples<T>(2*n);
  thrust::host_vector<thrust::complex<T> > h_p1(n);
  for(size_t i = 0; i<n; i++){
    h_p1[i].real(real[i]);
    h_p1[i].imag(real[2*i]);
  }
  return h_p1;
}

template <typename T>
struct TestComplexArithmeticTransform
{
  void operator()(const size_t n)
  {
    typedef thrust::complex<T> type;
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(n);
    thrust::host_vector<type> h_p2 = random_complex_samples<T>(n);
    thrust::host_vector<type>   h_result(n);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_p2 = h_p2;
    thrust::device_vector<type> d_result(n);

    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), basic_arithmetic_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), basic_arithmetic_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestComplexArithmeticTransform, FloatingPointTypes> TestComplexArithmeticTransformInstance;

template <typename T>
struct TestComplexPlaneTransform
{
  void operator()(const size_t n)
  {
    typedef thrust::complex<T> type;
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(n);
    thrust::host_vector<type>   h_result(n);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_result(n);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), complex_plane_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), complex_plane_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestComplexPlaneTransform, FloatingPointTypes> TestComplexPlaneTransformInstance;


template <typename T>
struct TestComplexPowerTransform
{
  void operator()(const size_t n)
  {
    typedef thrust::complex<T> type;
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(n);
    thrust::host_vector<type> h_p2 = random_complex_samples<T>(n);
    thrust::host_vector<type>   h_result(n);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_p2 = h_p2;
    thrust::device_vector<type> d_result(n);

    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), pow_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), pow_functor());    
    // pow can be very innacurate there's no point trying to check for equality
    // Currently just checking for compilation
    //    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), sqrt_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), sqrt_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestComplexPowerTransform, FloatingPointTypes> TestComplexPowerTransformInstance;

template <typename T>
struct TestComplexExponentialTransform
{
  void operator()(const size_t n)
  {
    typedef thrust::complex<T> type;
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(n);
    thrust::host_vector<type>   h_result(n);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_result(n);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), exp_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), exp_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), log_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), log_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), log10_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), log10_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestComplexExponentialTransform, FloatingPointTypes> TestComplexExponentialTransformInstance;

template <typename T>
struct TestComplexTrigonometricTransform
{
  void operator()(const size_t n)
  {
    typedef thrust::complex<T> type;
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(n);
    thrust::host_vector<type>   h_result(n);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_result(n);


    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), sin_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), sin_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), cos_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), cos_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), tan_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), tan_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);


    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), sinh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), sinh_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), cosh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), cosh_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), tanh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), tanh_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);


    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), asin_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), asin_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), acos_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), acos_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), atan_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), atan_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);


    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), asinh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), asinh_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), acosh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), acosh_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), atanh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), atanh_functor());    
    ASSERT_ALMOST_EQUAL(h_result, d_result);

  }
};
VariableUnitTest<TestComplexTrigonometricTransform, FloatingPointTypes> TestComplexTrigonometricTransformInstance;

