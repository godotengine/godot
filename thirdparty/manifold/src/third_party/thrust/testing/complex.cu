#include <unittest/unittest.h>

#include <thrust/complex.h>
#include <thrust/detail/config.h>

#include <complex>
#include <iostream>
#include <sstream>

/* 
   The following tests do not check for the numerical accuracy of the operations.
   That is tested in a separate program (complex_accuracy.cpp) which requires mpfr, 
   and takes a lot of time to run.   
 */

template<typename T>
struct TestComplexSizeAndAlignment
{
  void operator()()
  {
    THRUST_STATIC_ASSERT(
      sizeof(thrust::complex<T>) == sizeof(T) * 2
    );
    THRUST_STATIC_ASSERT(
      THRUST_ALIGNOF(thrust::complex<T>) == THRUST_ALIGNOF(T) * 2
    );

    THRUST_STATIC_ASSERT(
      sizeof(thrust::complex<T const>) == sizeof(T) * 2
    );
    THRUST_STATIC_ASSERT(
      THRUST_ALIGNOF(thrust::complex<T const>) == THRUST_ALIGNOF(T) * 2
    );
  }
};
SimpleUnitTest<TestComplexSizeAndAlignment, FloatingPointTypes> TestComplexSizeAndAlignmentInstance;

template<typename T>
struct TestComplexConstructors
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);
    
    thrust::complex<T> a(data[0],data[1]);
    thrust::complex<T> b(a);
    a = thrust::complex<T>(data[0],data[1]);
    ASSERT_ALMOST_EQUAL(a,b);
    
    a = thrust::complex<T>(data[0]);
    ASSERT_EQUAL(data[0], a.real());
    ASSERT_EQUAL(T(0), a.imag());
    
    a = thrust::complex<T>();
    ASSERT_ALMOST_EQUAL(a,std::complex<T>(0));
    
    a = thrust::complex<T>(thrust::complex<float>(static_cast<float>(data[0]),static_cast<float>(data[1])));
    ASSERT_ALMOST_EQUAL(a,b);
    
    a = thrust::complex<T>(thrust::complex<double>(static_cast<double>(data[0]),static_cast<double>(data[1])));
    ASSERT_ALMOST_EQUAL(a,b);
    
    a = thrust::complex<T>(std::complex<float>(static_cast<float>(data[0]),static_cast<float>(data[1])));
    ASSERT_ALMOST_EQUAL(a,b);
    
    a = thrust::complex<T>(std::complex<double>(static_cast<double>(data[0]),static_cast<double>(data[1])));
    ASSERT_ALMOST_EQUAL(a,b);
  }
};
SimpleUnitTest<TestComplexConstructors, FloatingPointTypes> TestComplexConstructorsInstance;


template<typename T>
struct TestComplexGetters
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    thrust::complex<T> z(data[0], data[1]);

    ASSERT_EQUAL(data[0], z.real());
    ASSERT_EQUAL(data[1], z.imag());

    z.real(data[1]);
    z.imag(data[0]);
    ASSERT_EQUAL(data[1], z.real());
    ASSERT_EQUAL(data[0], z.imag());

    volatile thrust::complex<T> v(data[0], data[1]);

    ASSERT_EQUAL(data[0], v.real());
    ASSERT_EQUAL(data[1], v.imag());

    v.real(data[1]);
    v.imag(data[0]);
    ASSERT_EQUAL(data[1], v.real());
    ASSERT_EQUAL(data[0], v.imag());
  }
};
SimpleUnitTest<TestComplexGetters, FloatingPointTypes> TestComplexGettersInstance;

template<typename T>
struct TestComplexMemberOperators
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);
    thrust::host_vector<T> data_b = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);

    std::complex<T> c(a);
    std::complex<T> d(b);

    a += b;
    c += d;
    ASSERT_ALMOST_EQUAL(a,c);

    a -= b;
    c -= d;
    ASSERT_ALMOST_EQUAL(a,c);

    a *= b;
    c *= d;
    ASSERT_ALMOST_EQUAL(a,c);

    a /= b;
    c /= d;
    ASSERT_ALMOST_EQUAL(a,c);

    // casting operator
    c = (std::complex<T>)a;
  }
};
SimpleUnitTest<TestComplexMemberOperators, FloatingPointTypes> TestComplexMemberOperatorsInstance;


template<typename T>
struct TestComplexBasicArithmetic
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    thrust::complex<T> a(data[0], data[1]);
    std::complex<T> b(a);

    // Test the basic arithmetic functions against std
    
    ASSERT_ALMOST_EQUAL(abs(a),abs(b));

    ASSERT_ALMOST_EQUAL(arg(a),arg(b));

    ASSERT_ALMOST_EQUAL(norm(a),norm(b));

    ASSERT_EQUAL(conj(a),conj(b));

    ASSERT_ALMOST_EQUAL(thrust::polar(data[0],data[1]),std::polar(data[0],data[1]));

    // random_samples does not seem to produce infinities so proj(z) == z
    ASSERT_EQUAL(proj(a),a);
    
  }
};
SimpleUnitTest<TestComplexBasicArithmetic, FloatingPointTypes> TestComplexBasicArithmeticInstance;


template<typename T>
struct TestComplexBinaryArithmetic
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);
    thrust::host_vector<T> data_b = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);

    ASSERT_ALMOST_EQUAL(a*b,std::complex<T>(a) * std::complex<T>(b));
    ASSERT_ALMOST_EQUAL(a*data_b[0],std::complex<T>(a) * data_b[0]);
    ASSERT_ALMOST_EQUAL(data_a[0]*b,data_b[0] * std::complex<T>(b));

    ASSERT_ALMOST_EQUAL(a / b, std::complex<T>(a) / std::complex<T>(b));
    ASSERT_ALMOST_EQUAL(a / data_b[0], std::complex<T>(a) / data_b[0]);
    ASSERT_ALMOST_EQUAL(data_a[0] / b, data_b[0] / std::complex<T>(b));

    ASSERT_EQUAL(a + b, std::complex<T>(a) + std::complex<T>(b));
    ASSERT_EQUAL(a + data_b[0], std::complex<T>(a) + data_b[0]);
    ASSERT_EQUAL(data_a[0] + b, data_b[0] + std::complex<T>(b));

    ASSERT_EQUAL(a - b, std::complex<T>(a) - std::complex<T>(b));
    ASSERT_EQUAL(a - data_b[0], std::complex<T>(a) - data_b[0]);
    ASSERT_EQUAL(data_a[0] - b, data_b[0] - std::complex<T>(b));
    
  }
};
SimpleUnitTest<TestComplexBinaryArithmetic, FloatingPointTypes> TestComplexBinaryArithmeticInstance;

template<typename T>
struct TestComplexUnaryArithmetic
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);

    ASSERT_EQUAL(+a,+std::complex<T>(a));
    ASSERT_EQUAL(-a,-std::complex<T>(a));
    
  }
};
SimpleUnitTest<TestComplexUnaryArithmetic, FloatingPointTypes> TestComplexUnaryArithmeticInstance;


template<typename T>
struct TestComplexExponentialFunctions
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::complex<T> b(a);

    ASSERT_ALMOST_EQUAL(exp(a),exp(b));
    ASSERT_ALMOST_EQUAL(log(a),log(b));
    ASSERT_ALMOST_EQUAL(log10(a),log10(b));
    
  }
};
SimpleUnitTest<TestComplexExponentialFunctions, FloatingPointTypes> TestComplexExponentialFunctionsInstance;


template<typename T>
struct TestComplexPowerFunctions
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);
    thrust::host_vector<T> data_b = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);
    std::complex<T> c(a);
    std::complex<T> d(b);

    ASSERT_ALMOST_EQUAL(pow(a,b),pow(c,d));
    ASSERT_ALMOST_EQUAL(pow(a,b.real()),pow(c,d.real()));
    ASSERT_ALMOST_EQUAL(pow(a.real(),b),pow(c.real(),d));

    ASSERT_ALMOST_EQUAL(sqrt(a),sqrt(c));

  }
};
SimpleUnitTest<TestComplexPowerFunctions, FloatingPointTypes> TestComplexPowerFunctionsInstance;

template<typename T>
struct TestComplexTrigonometricFunctions
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::complex<T> c(a);

    ASSERT_ALMOST_EQUAL(cos(a),cos(c));
    ASSERT_ALMOST_EQUAL(sin(a),sin(c));
    ASSERT_ALMOST_EQUAL(tan(a),tan(c));

    ASSERT_ALMOST_EQUAL(cosh(a),cosh(c));
    ASSERT_ALMOST_EQUAL(sinh(a),sinh(c));
    ASSERT_ALMOST_EQUAL(tanh(a),tanh(c));

#if THRUST_CPP_DIALECT >= 2011

    ASSERT_ALMOST_EQUAL(acos(a),acos(c));
    ASSERT_ALMOST_EQUAL(asin(a),asin(c));
    ASSERT_ALMOST_EQUAL(atan(a),atan(c));

    ASSERT_ALMOST_EQUAL(acosh(a),acosh(c));
    ASSERT_ALMOST_EQUAL(asinh(a),asinh(c));
    ASSERT_ALMOST_EQUAL(atanh(a),atanh(c));

#endif


  }
};
SimpleUnitTest<TestComplexTrigonometricFunctions, FloatingPointTypes> TestComplexTrigonometricFunctionsInstance;

template<typename T>
struct TestComplexStreamOperators
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);
    thrust::complex<T> a(data_a[0], data_a[1]);
    std::stringstream out;
    out << a;
    thrust::complex<T> b;
    out >> b;
    ASSERT_ALMOST_EQUAL(a,b);
  }
};
SimpleUnitTest<TestComplexStreamOperators, FloatingPointTypes> TestComplexStreamOperatorsInstance;

#if THRUST_CPP_DIALECT >= 2011
template<typename T>
struct TestComplexStdComplexDeviceInterop
{
  void operator()()
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(6);
    std::vector<std::complex<T> > vec(10);
    vec[0] = std::complex<T>(data[0], data[1]);
    vec[1] = std::complex<T>(data[2], data[3]);
    vec[2] = std::complex<T>(data[4], data[5]);

    thrust::device_vector<thrust::complex<T> > device_vec = vec;
    ASSERT_ALMOST_EQUAL(vec[0].real(), thrust::complex<T>(device_vec[0]).real());
    ASSERT_ALMOST_EQUAL(vec[0].imag(), thrust::complex<T>(device_vec[0]).imag());
    ASSERT_ALMOST_EQUAL(vec[1].real(), thrust::complex<T>(device_vec[1]).real());
    ASSERT_ALMOST_EQUAL(vec[1].imag(), thrust::complex<T>(device_vec[1]).imag());
    ASSERT_ALMOST_EQUAL(vec[2].real(), thrust::complex<T>(device_vec[2]).real());
    ASSERT_ALMOST_EQUAL(vec[2].imag(), thrust::complex<T>(device_vec[2]).imag());
  }
};
SimpleUnitTest<TestComplexStdComplexDeviceInterop, FloatingPointTypes> TestComplexStdComplexDeviceInteropInstance;
#endif

