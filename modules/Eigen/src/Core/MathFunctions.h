// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATHFUNCTIONS_H
#define EIGEN_MATHFUNCTIONS_H

// source: http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
// TODO this should better be moved to NumTraits
#define EIGEN_PI 3.141592653589793238462643383279502884197169399375105820974944592307816406L

namespace Eigen {

// On WINCE, std::abs is defined for int only, so let's defined our own overloads:
// This issue has been confirmed with MSVC 2008 only, but the issue might exist for more recent versions too.
#if EIGEN_OS_WINCE && EIGEN_COMP_MSVC && EIGEN_COMP_MSVC<=1500
long        abs(long        x) { return (labs(x));  }
double      abs(double      x) { return (fabs(x));  }
float       abs(float       x) { return (fabsf(x)); }
long double abs(long double x) { return (fabsl(x)); }
#endif

namespace internal {

/** \internal \class global_math_functions_filtering_base
  *
  * What it does:
  * Defines a typedef 'type' as follows:
  * - if type T has a member typedef Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl, then
  *   global_math_functions_filtering_base<T>::type is a typedef for it.
  * - otherwise, global_math_functions_filtering_base<T>::type is a typedef for T.
  *
  * How it's used:
  * To allow to defined the global math functions (like sin...) in certain cases, like the Array expressions.
  * When you do sin(array1+array2), the object array1+array2 has a complicated expression type, all what you want to know
  * is that it inherits ArrayBase. So we implement a partial specialization of sin_impl for ArrayBase<Derived>.
  * So we must make sure to use sin_impl<ArrayBase<Derived> > and not sin_impl<Derived>, otherwise our partial specialization
  * won't be used. How does sin know that? That's exactly what global_math_functions_filtering_base tells it.
  *
  * How it's implemented:
  * SFINAE in the style of enable_if. Highly susceptible of breaking compilers. With GCC, it sure does work, but if you replace
  * the typename dummy by an integer template parameter, it doesn't work anymore!
  */

template<typename T, typename dummy = void>
struct global_math_functions_filtering_base
{
  typedef T type;
};

template<typename T> struct always_void { typedef void type; };

template<typename T>
struct global_math_functions_filtering_base
  <T,
   typename always_void<typename T::Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl>::type
  >
{
  typedef typename T::Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl type;
};

#define EIGEN_MATHFUNC_IMPL(func, scalar) Eigen::internal::func##_impl<typename Eigen::internal::global_math_functions_filtering_base<scalar>::type>
#define EIGEN_MATHFUNC_RETVAL(func, scalar) typename Eigen::internal::func##_retval<typename Eigen::internal::global_math_functions_filtering_base<scalar>::type>::type

/****************************************************************************
* Implementation of real                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct real_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return x;
  }
};

template<typename Scalar>
struct real_default_impl<Scalar,true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    using std::real;
    return real(x);
  }
};

template<typename Scalar> struct real_impl : real_default_impl<Scalar> {};

#ifdef EIGEN_CUDA_ARCH
template<typename T>
struct real_impl<std::complex<T> >
{
  typedef T RealScalar;
  EIGEN_DEVICE_FUNC
  static inline T run(const std::complex<T>& x)
  {
    return x.real();
  }
};
#endif

template<typename Scalar>
struct real_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of imag                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct imag_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar&)
  {
    return RealScalar(0);
  }
};

template<typename Scalar>
struct imag_default_impl<Scalar,true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    using std::imag;
    return imag(x);
  }
};

template<typename Scalar> struct imag_impl : imag_default_impl<Scalar> {};

#ifdef EIGEN_CUDA_ARCH
template<typename T>
struct imag_impl<std::complex<T> >
{
  typedef T RealScalar;
  EIGEN_DEVICE_FUNC
  static inline T run(const std::complex<T>& x)
  {
    return x.imag();
  }
};
#endif

template<typename Scalar>
struct imag_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of real_ref                                             *
****************************************************************************/

template<typename Scalar>
struct real_ref_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar& run(Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[0];
  }
  EIGEN_DEVICE_FUNC
  static inline const RealScalar& run(const Scalar& x)
  {
    return reinterpret_cast<const RealScalar*>(&x)[0];
  }
};

template<typename Scalar>
struct real_ref_retval
{
  typedef typename NumTraits<Scalar>::Real & type;
};

/****************************************************************************
* Implementation of imag_ref                                             *
****************************************************************************/

template<typename Scalar, bool IsComplex>
struct imag_ref_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar& run(Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[1];
  }
  EIGEN_DEVICE_FUNC
  static inline const RealScalar& run(const Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[1];
  }
};

template<typename Scalar>
struct imag_ref_default_impl<Scalar, false>
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(Scalar&)
  {
    return Scalar(0);
  }
  EIGEN_DEVICE_FUNC
  static inline const Scalar run(const Scalar&)
  {
    return Scalar(0);
  }
};

template<typename Scalar>
struct imag_ref_impl : imag_ref_default_impl<Scalar, NumTraits<Scalar>::IsComplex> {};

template<typename Scalar>
struct imag_ref_retval
{
  typedef typename NumTraits<Scalar>::Real & type;
};

/****************************************************************************
* Implementation of conj                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct conj_impl
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(const Scalar& x)
  {
    return x;
  }
};

template<typename Scalar>
struct conj_impl<Scalar,true>
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(const Scalar& x)
  {
    using std::conj;
    return conj(x);
  }
};

template<typename Scalar>
struct conj_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of abs2                                                 *
****************************************************************************/

template<typename Scalar,bool IsComplex>
struct abs2_impl_default
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return x*x;
  }
};

template<typename Scalar>
struct abs2_impl_default<Scalar, true> // IsComplex
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return real(x)*real(x) + imag(x)*imag(x);
  }
};

template<typename Scalar>
struct abs2_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return abs2_impl_default<Scalar,NumTraits<Scalar>::IsComplex>::run(x);
  }
};

template<typename Scalar>
struct abs2_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of norm1                                                *
****************************************************************************/

template<typename Scalar, bool IsComplex>
struct norm1_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    EIGEN_USING_STD_MATH(abs);
    return abs(real(x)) + abs(imag(x));
  }
};

template<typename Scalar>
struct norm1_default_impl<Scalar, false>
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(const Scalar& x)
  {
    EIGEN_USING_STD_MATH(abs);
    return abs(x);
  }
};

template<typename Scalar>
struct norm1_impl : norm1_default_impl<Scalar, NumTraits<Scalar>::IsComplex> {};

template<typename Scalar>
struct norm1_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of hypot                                                *
****************************************************************************/

template<typename Scalar>
struct hypot_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar run(const Scalar& x, const Scalar& y)
  {
    EIGEN_USING_STD_MATH(abs);
    EIGEN_USING_STD_MATH(sqrt);
    RealScalar _x = abs(x);
    RealScalar _y = abs(y);
    Scalar p, qp;
    if(_x>_y)
    {
      p = _x;
      qp = _y / p;
    }
    else
    {
      p = _y;
      qp = _x / p;
    }
    if(p==RealScalar(0)) return RealScalar(0);
    return p * sqrt(RealScalar(1) + qp*qp);
  }
};

template<typename Scalar>
struct hypot_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of cast                                                 *
****************************************************************************/

template<typename OldType, typename NewType>
struct cast_impl
{
  EIGEN_DEVICE_FUNC
  static inline NewType run(const OldType& x)
  {
    return static_cast<NewType>(x);
  }
};

// here, for once, we're plainly returning NewType: we don't want cast to do weird things.

template<typename OldType, typename NewType>
EIGEN_DEVICE_FUNC
inline NewType cast(const OldType& x)
{
  return cast_impl<OldType, NewType>::run(x);
}

/****************************************************************************
* Implementation of round                                                   *
****************************************************************************/

#if EIGEN_HAS_CXX11_MATH
  template<typename Scalar>
  struct round_impl {
    static inline Scalar run(const Scalar& x)
    {
      EIGEN_STATIC_ASSERT((!NumTraits<Scalar>::IsComplex), NUMERIC_TYPE_MUST_BE_REAL)
      EIGEN_USING_STD_MATH(round);
      return round(x);
    }
  };
#else
  template<typename Scalar>
  struct round_impl
  {
    static inline Scalar run(const Scalar& x)
    {
      EIGEN_STATIC_ASSERT((!NumTraits<Scalar>::IsComplex), NUMERIC_TYPE_MUST_BE_REAL)
      EIGEN_USING_STD_MATH(floor);
      EIGEN_USING_STD_MATH(ceil);
      return (x > Scalar(0)) ? floor(x + Scalar(0.5)) : ceil(x - Scalar(0.5));
    }
  };
#endif

template<typename Scalar>
struct round_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of arg                                                     *
****************************************************************************/

#if EIGEN_HAS_CXX11_MATH
  template<typename Scalar>
  struct arg_impl {
    static inline Scalar run(const Scalar& x)
    {
      EIGEN_USING_STD_MATH(arg);
      return arg(x);
    }
  };
#else
  template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
  struct arg_default_impl
  {
    typedef typename NumTraits<Scalar>::Real RealScalar;
    EIGEN_DEVICE_FUNC
    static inline RealScalar run(const Scalar& x)
    {
      return (x < Scalar(0)) ? Scalar(EIGEN_PI) : Scalar(0); }
  };

  template<typename Scalar>
  struct arg_default_impl<Scalar,true>
  {
    typedef typename NumTraits<Scalar>::Real RealScalar;
    EIGEN_DEVICE_FUNC
    static inline RealScalar run(const Scalar& x)
    {
      EIGEN_USING_STD_MATH(arg);
      return arg(x);
    }
  };

  template<typename Scalar> struct arg_impl : arg_default_impl<Scalar> {};
#endif

template<typename Scalar>
struct arg_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of expm1                                                   *
****************************************************************************/

// This implementation is based on GSL Math's expm1.
namespace std_fallback {
  // fallback expm1 implementation in case there is no expm1(Scalar) function in namespace of Scalar,
  // or that there is no suitable std::expm1 function available. Implementation
  // attributed to Kahan. See: http://www.plunk.org/~hatch/rightway.php.
  template<typename Scalar>
  EIGEN_DEVICE_FUNC inline Scalar expm1(const Scalar& x) {
    EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
    typedef typename NumTraits<Scalar>::Real RealScalar;

    EIGEN_USING_STD_MATH(exp);
    Scalar u = exp(x);
    if (u == Scalar(1)) {
      return x;
    }
    Scalar um1 = u - RealScalar(1);
    if (um1 == Scalar(-1)) {
      return RealScalar(-1);
    }

    EIGEN_USING_STD_MATH(log);
    return (u - RealScalar(1)) * x / log(u);
  }
}

template<typename Scalar>
struct expm1_impl {
  EIGEN_DEVICE_FUNC static inline Scalar run(const Scalar& x)
  {
    EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
    #if EIGEN_HAS_CXX11_MATH
    using std::expm1;
    #endif
    using std_fallback::expm1;
    return expm1(x);
  }
};


template<typename Scalar>
struct expm1_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of log1p                                                   *
****************************************************************************/

namespace std_fallback {
  // fallback log1p implementation in case there is no log1p(Scalar) function in namespace of Scalar,
  // or that there is no suitable std::log1p function available
  template<typename Scalar>
  EIGEN_DEVICE_FUNC inline Scalar log1p(const Scalar& x) {
    EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
    typedef typename NumTraits<Scalar>::Real RealScalar;
    EIGEN_USING_STD_MATH(log);
    Scalar x1p = RealScalar(1) + x;
    return ( x1p == Scalar(1) ) ? x : x * ( log(x1p) / (x1p - RealScalar(1)) );
  }
}

template<typename Scalar>
struct log1p_impl {
  EIGEN_DEVICE_FUNC static inline Scalar run(const Scalar& x)
  {
    EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
    #if EIGEN_HAS_CXX11_MATH
    using std::log1p;
    #endif
    using std_fallback::log1p;
    return log1p(x);
  }
};


template<typename Scalar>
struct log1p_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of pow                                                  *
****************************************************************************/

template<typename ScalarX,typename ScalarY, bool IsInteger = NumTraits<ScalarX>::IsInteger&&NumTraits<ScalarY>::IsInteger>
struct pow_impl
{
  //typedef Scalar retval;
  typedef typename ScalarBinaryOpTraits<ScalarX,ScalarY,internal::scalar_pow_op<ScalarX,ScalarY> >::ReturnType result_type;
  static EIGEN_DEVICE_FUNC inline result_type run(const ScalarX& x, const ScalarY& y)
  {
    EIGEN_USING_STD_MATH(pow);
    return pow(x, y);
  }
};

template<typename ScalarX,typename ScalarY>
struct pow_impl<ScalarX,ScalarY, true>
{
  typedef ScalarX result_type;
  static EIGEN_DEVICE_FUNC inline ScalarX run(ScalarX x, ScalarY y)
  {
    ScalarX res(1);
    eigen_assert(!NumTraits<ScalarY>::IsSigned || y >= 0);
    if(y & 1) res *= x;
    y >>= 1;
    while(y)
    {
      x *= x;
      if(y&1) res *= x;
      y >>= 1;
    }
    return res;
  }
};

/****************************************************************************
* Implementation of random                                               *
****************************************************************************/

template<typename Scalar,
         bool IsComplex,
         bool IsInteger>
struct random_default_impl {};

template<typename Scalar>
struct random_impl : random_default_impl<Scalar, NumTraits<Scalar>::IsComplex, NumTraits<Scalar>::IsInteger> {};

template<typename Scalar>
struct random_retval
{
  typedef Scalar type;
};

template<typename Scalar> inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random(const Scalar& x, const Scalar& y);
template<typename Scalar> inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random();

template<typename Scalar>
struct random_default_impl<Scalar, false, false>
{
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    return x + (y-x) * Scalar(std::rand()) / Scalar(RAND_MAX);
  }
  static inline Scalar run()
  {
    return run(Scalar(NumTraits<Scalar>::IsSigned ? -1 : 0), Scalar(1));
  }
};

enum {
  meta_floor_log2_terminate,
  meta_floor_log2_move_up,
  meta_floor_log2_move_down,
  meta_floor_log2_bogus
};

template<unsigned int n, int lower, int upper> struct meta_floor_log2_selector
{
  enum { middle = (lower + upper) / 2,
         value = (upper <= lower + 1) ? int(meta_floor_log2_terminate)
               : (n < (1 << middle)) ? int(meta_floor_log2_move_down)
               : (n==0) ? int(meta_floor_log2_bogus)
               : int(meta_floor_log2_move_up)
  };
};

template<unsigned int n,
         int lower = 0,
         int upper = sizeof(unsigned int) * CHAR_BIT - 1,
         int selector = meta_floor_log2_selector<n, lower, upper>::value>
struct meta_floor_log2 {};

template<unsigned int n, int lower, int upper>
struct meta_floor_log2<n, lower, upper, meta_floor_log2_move_down>
{
  enum { value = meta_floor_log2<n, lower, meta_floor_log2_selector<n, lower, upper>::middle>::value };
};

template<unsigned int n, int lower, int upper>
struct meta_floor_log2<n, lower, upper, meta_floor_log2_move_up>
{
  enum { value = meta_floor_log2<n, meta_floor_log2_selector<n, lower, upper>::middle, upper>::value };
};

template<unsigned int n, int lower, int upper>
struct meta_floor_log2<n, lower, upper, meta_floor_log2_terminate>
{
  enum { value = (n >= ((unsigned int)(1) << (lower+1))) ? lower+1 : lower };
};

template<unsigned int n, int lower, int upper>
struct meta_floor_log2<n, lower, upper, meta_floor_log2_bogus>
{
  // no value, error at compile time
};

template<typename Scalar>
struct random_default_impl<Scalar, false, true>
{
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    typedef typename conditional<NumTraits<Scalar>::IsSigned,std::ptrdiff_t,std::size_t>::type ScalarX;
    if(y<x)
      return x;
    // the following difference might overflow on a 32 bits system,
    // but since y>=x the result converted to an unsigned long is still correct.
    std::size_t range = ScalarX(y)-ScalarX(x);
    std::size_t offset = 0;
    // rejection sampling
    std::size_t divisor = 1;
    std::size_t multiplier = 1;
    if(range<RAND_MAX) divisor = (std::size_t(RAND_MAX)+1)/(range+1);
    else               multiplier = 1 + range/(std::size_t(RAND_MAX)+1);
    do {
      offset = (std::size_t(std::rand()) * multiplier) / divisor;
    } while (offset > range);
    return Scalar(ScalarX(x) + offset);
  }

  static inline Scalar run()
  {
#ifdef EIGEN_MAKING_DOCS
    return run(Scalar(NumTraits<Scalar>::IsSigned ? -10 : 0), Scalar(10));
#else
    enum { rand_bits = meta_floor_log2<(unsigned int)(RAND_MAX)+1>::value,
           scalar_bits = sizeof(Scalar) * CHAR_BIT,
           shift = EIGEN_PLAIN_ENUM_MAX(0, int(rand_bits) - int(scalar_bits)),
           offset = NumTraits<Scalar>::IsSigned ? (1 << (EIGEN_PLAIN_ENUM_MIN(rand_bits,scalar_bits)-1)) : 0
    };
    return Scalar((std::rand() >> shift) - offset);
#endif
  }
};

template<typename Scalar>
struct random_default_impl<Scalar, true, false>
{
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    return Scalar(random(real(x), real(y)),
                  random(imag(x), imag(y)));
  }
  static inline Scalar run()
  {
    typedef typename NumTraits<Scalar>::Real RealScalar;
    return Scalar(random<RealScalar>(), random<RealScalar>());
  }
};

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run(x, y);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random()
{
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run();
}

// Implementatin of is* functions

// std::is* do not work with fast-math and gcc, std::is* are available on MSVC 2013 and newer, as well as in clang.
#if (EIGEN_HAS_CXX11_MATH && !(EIGEN_COMP_GNUC_STRICT && __FINITE_MATH_ONLY__)) || (EIGEN_COMP_MSVC>=1800) || (EIGEN_COMP_CLANG)
#define EIGEN_USE_STD_FPCLASSIFY 1
#else
#define EIGEN_USE_STD_FPCLASSIFY 0
#endif

template<typename T>
EIGEN_DEVICE_FUNC
typename internal::enable_if<internal::is_integral<T>::value,bool>::type
isnan_impl(const T&) { return false; }

template<typename T>
EIGEN_DEVICE_FUNC
typename internal::enable_if<internal::is_integral<T>::value,bool>::type
isinf_impl(const T&) { return false; }

template<typename T>
EIGEN_DEVICE_FUNC
typename internal::enable_if<internal::is_integral<T>::value,bool>::type
isfinite_impl(const T&) { return true; }

template<typename T>
EIGEN_DEVICE_FUNC
typename internal::enable_if<(!internal::is_integral<T>::value)&&(!NumTraits<T>::IsComplex),bool>::type
isfinite_impl(const T& x)
{
  #ifdef EIGEN_CUDA_ARCH
    return (::isfinite)(x);
  #elif EIGEN_USE_STD_FPCLASSIFY
    using std::isfinite;
    return isfinite EIGEN_NOT_A_MACRO (x);
  #else
    return x<=NumTraits<T>::highest() && x>=NumTraits<T>::lowest();
  #endif
}

template<typename T>
EIGEN_DEVICE_FUNC
typename internal::enable_if<(!internal::is_integral<T>::value)&&(!NumTraits<T>::IsComplex),bool>::type
isinf_impl(const T& x)
{
  #ifdef EIGEN_CUDA_ARCH
    return (::isinf)(x);
  #elif EIGEN_USE_STD_FPCLASSIFY
    using std::isinf;
    return isinf EIGEN_NOT_A_MACRO (x);
  #else
    return x>NumTraits<T>::highest() || x<NumTraits<T>::lowest();
  #endif
}

template<typename T>
EIGEN_DEVICE_FUNC
typename internal::enable_if<(!internal::is_integral<T>::value)&&(!NumTraits<T>::IsComplex),bool>::type
isnan_impl(const T& x)
{
  #ifdef EIGEN_CUDA_ARCH
    return (::isnan)(x);
  #elif EIGEN_USE_STD_FPCLASSIFY
    using std::isnan;
    return isnan EIGEN_NOT_A_MACRO (x);
  #else
    return x != x;
  #endif
}

#if (!EIGEN_USE_STD_FPCLASSIFY)

#if EIGEN_COMP_MSVC

template<typename T> EIGEN_DEVICE_FUNC bool isinf_msvc_helper(T x)
{
  return _fpclass(x)==_FPCLASS_NINF || _fpclass(x)==_FPCLASS_PINF;
}

//MSVC defines a _isnan builtin function, but for double only
EIGEN_DEVICE_FUNC inline bool isnan_impl(const long double& x) { return _isnan(x)!=0; }
EIGEN_DEVICE_FUNC inline bool isnan_impl(const double& x)      { return _isnan(x)!=0; }
EIGEN_DEVICE_FUNC inline bool isnan_impl(const float& x)       { return _isnan(x)!=0; }

EIGEN_DEVICE_FUNC inline bool isinf_impl(const long double& x) { return isinf_msvc_helper(x); }
EIGEN_DEVICE_FUNC inline bool isinf_impl(const double& x)      { return isinf_msvc_helper(x); }
EIGEN_DEVICE_FUNC inline bool isinf_impl(const float& x)       { return isinf_msvc_helper(x); }

#elif (defined __FINITE_MATH_ONLY__ && __FINITE_MATH_ONLY__ && EIGEN_COMP_GNUC)

#if EIGEN_GNUC_AT_LEAST(5,0)
  #define EIGEN_TMP_NOOPT_ATTRIB EIGEN_DEVICE_FUNC inline __attribute__((optimize("no-finite-math-only")))
#else
  // NOTE the inline qualifier and noinline attribute are both needed: the former is to avoid linking issue (duplicate symbol),
  //      while the second prevent too aggressive optimizations in fast-math mode:
  #define EIGEN_TMP_NOOPT_ATTRIB EIGEN_DEVICE_FUNC inline __attribute__((noinline,optimize("no-finite-math-only")))
#endif

template<> EIGEN_TMP_NOOPT_ATTRIB bool isnan_impl(const long double& x) { return __builtin_isnan(x); }
template<> EIGEN_TMP_NOOPT_ATTRIB bool isnan_impl(const double& x)      { return __builtin_isnan(x); }
template<> EIGEN_TMP_NOOPT_ATTRIB bool isnan_impl(const float& x)       { return __builtin_isnan(x); }
template<> EIGEN_TMP_NOOPT_ATTRIB bool isinf_impl(const double& x)      { return __builtin_isinf(x); }
template<> EIGEN_TMP_NOOPT_ATTRIB bool isinf_impl(const float& x)       { return __builtin_isinf(x); }
template<> EIGEN_TMP_NOOPT_ATTRIB bool isinf_impl(const long double& x) { return __builtin_isinf(x); }

#undef EIGEN_TMP_NOOPT_ATTRIB

#endif

#endif

// The following overload are defined at the end of this file
template<typename T> EIGEN_DEVICE_FUNC bool isfinite_impl(const std::complex<T>& x);
template<typename T> EIGEN_DEVICE_FUNC bool isnan_impl(const std::complex<T>& x);
template<typename T> EIGEN_DEVICE_FUNC bool isinf_impl(const std::complex<T>& x);

template<typename T> T generic_fast_tanh_float(const T& a_x);

} // end namespace internal

/****************************************************************************
* Generic math functions                                                    *
****************************************************************************/

namespace numext {

#if !defined(EIGEN_CUDA_ARCH) && !defined(__SYCL_DEVICE_ONLY__)
template<typename T>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE T mini(const T& x, const T& y)
{
  EIGEN_USING_STD_MATH(min);
  return min EIGEN_NOT_A_MACRO (x,y);
}

template<typename T>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE T maxi(const T& x, const T& y)
{
  EIGEN_USING_STD_MATH(max);
  return max EIGEN_NOT_A_MACRO (x,y);
}


#elif defined(__SYCL_DEVICE_ONLY__)
template<typename T>
EIGEN_ALWAYS_INLINE T mini(const T& x, const T& y)
{

  return y < x ? y : x;
}

template<typename T>
EIGEN_ALWAYS_INLINE T maxi(const T& x, const T& y)
{

  return x < y ? y : x;
}

EIGEN_ALWAYS_INLINE int mini(const int& x, const int& y)
{
  return cl::sycl::min(x,y);
}

EIGEN_ALWAYS_INLINE int maxi(const int& x, const int& y)
{
  return cl::sycl::max(x,y);
}

EIGEN_ALWAYS_INLINE unsigned int mini(const unsigned int& x, const unsigned int& y)
{
  return cl::sycl::min(x,y);
}

EIGEN_ALWAYS_INLINE unsigned int maxi(const unsigned int& x, const unsigned int& y)
{
  return cl::sycl::max(x,y);
}

EIGEN_ALWAYS_INLINE  long mini(const long & x, const long & y)
{
  return cl::sycl::min(x,y);
}

EIGEN_ALWAYS_INLINE  long maxi(const long & x, const long & y)
{
  return cl::sycl::max(x,y);
}

EIGEN_ALWAYS_INLINE unsigned long mini(const unsigned long& x, const unsigned long& y)
{
  return cl::sycl::min(x,y);
}

EIGEN_ALWAYS_INLINE unsigned long maxi(const unsigned long& x, const unsigned long& y)
{
  return cl::sycl::max(x,y);
}


EIGEN_ALWAYS_INLINE float mini(const float& x, const float& y)
{
  return cl::sycl::fmin(x,y);
}

EIGEN_ALWAYS_INLINE float maxi(const float& x, const float& y)
{
  return cl::sycl::fmax(x,y);
}

EIGEN_ALWAYS_INLINE double mini(const double& x, const double& y)
{
  return cl::sycl::fmin(x,y);
}

EIGEN_ALWAYS_INLINE double maxi(const double& x, const double& y)
{
  return cl::sycl::fmax(x,y);
}

#else
template<typename T>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE T mini(const T& x, const T& y)
{
  return y < x ? y : x;
}
template<>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE float mini(const float& x, const float& y)
{
  return fminf(x, y);
}
template<typename T>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE T maxi(const T& x, const T& y)
{
  return x < y ? y : x;
}
template<>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE float maxi(const float& x, const float& y)
{
  return fmaxf(x, y);
}
#endif


template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(real, Scalar) real(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(real, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline typename internal::add_const_on_value_type< EIGEN_MATHFUNC_RETVAL(real_ref, Scalar) >::type real_ref(const Scalar& x)
{
  return internal::real_ref_impl<Scalar>::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(real_ref, Scalar) real_ref(Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(real_ref, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(imag, Scalar) imag(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(imag, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(arg, Scalar) arg(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(arg, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline typename internal::add_const_on_value_type< EIGEN_MATHFUNC_RETVAL(imag_ref, Scalar) >::type imag_ref(const Scalar& x)
{
  return internal::imag_ref_impl<Scalar>::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(imag_ref, Scalar) imag_ref(Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(imag_ref, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(conj, Scalar) conj(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(conj, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(abs2, Scalar) abs2(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(abs2, Scalar)::run(x);
}

EIGEN_DEVICE_FUNC
inline bool abs2(bool x) { return x; }

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(norm1, Scalar) norm1(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(norm1, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(hypot, Scalar) hypot(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(hypot, Scalar)::run(x, y);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(log1p, Scalar) log1p(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(log1p, Scalar)::run(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   log1p(float x) { return cl::sycl::log1p(x); }
EIGEN_ALWAYS_INLINE double  log1p(double x) { return cl::sycl::log1p(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float log1p(const float &x) { return ::log1pf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double log1p(const double &x) { return ::log1p(x); }
#endif

template<typename ScalarX,typename ScalarY>
EIGEN_DEVICE_FUNC
inline typename internal::pow_impl<ScalarX,ScalarY>::result_type pow(const ScalarX& x, const ScalarY& y)
{
  return internal::pow_impl<ScalarX,ScalarY>::run(x, y);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   pow(float x, float y) { return cl::sycl::pow(x, y); }
EIGEN_ALWAYS_INLINE double  pow(double x, double y) { return cl::sycl::pow(x, y); }
#endif // defined(__SYCL_DEVICE_ONLY__)

template<typename T> EIGEN_DEVICE_FUNC bool (isnan)   (const T &x) { return internal::isnan_impl(x); }
template<typename T> EIGEN_DEVICE_FUNC bool (isinf)   (const T &x) { return internal::isinf_impl(x); }
template<typename T> EIGEN_DEVICE_FUNC bool (isfinite)(const T &x) { return internal::isfinite_impl(x); }

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   isnan(float x) { return cl::sycl::isnan(x); }
EIGEN_ALWAYS_INLINE double  isnan(double x) { return cl::sycl::isnan(x); }
EIGEN_ALWAYS_INLINE float   isinf(float x) { return cl::sycl::isinf(x); }
EIGEN_ALWAYS_INLINE double  isinf(double x) { return cl::sycl::isinf(x); }
EIGEN_ALWAYS_INLINE float   isfinite(float x) { return cl::sycl::isfinite(x); }
EIGEN_ALWAYS_INLINE double  isfinite(double x) { return cl::sycl::isfinite(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(round, Scalar) round(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(round, Scalar)::run(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   round(float x) { return cl::sycl::round(x); }
EIGEN_ALWAYS_INLINE double  round(double x) { return cl::sycl::round(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

template<typename T>
EIGEN_DEVICE_FUNC
T (floor)(const T& x)
{
  EIGEN_USING_STD_MATH(floor);
  return floor(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   floor(float x) { return cl::sycl::floor(x); }
EIGEN_ALWAYS_INLINE double  floor(double x) { return cl::sycl::floor(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float floor(const float &x) { return ::floorf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double floor(const double &x) { return ::floor(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC
T (ceil)(const T& x)
{
  EIGEN_USING_STD_MATH(ceil);
  return ceil(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   ceil(float x) { return cl::sycl::ceil(x); }
EIGEN_ALWAYS_INLINE double  ceil(double x) { return cl::sycl::ceil(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float ceil(const float &x) { return ::ceilf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double ceil(const double &x) { return ::ceil(x); }
#endif


/** Log base 2 for 32 bits positive integers.
  * Conveniently returns 0 for x==0. */
inline int log2(int x)
{
  eigen_assert(x>=0);
  unsigned int v(x);
  static const int table[32] = { 0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31 };
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return table[(v * 0x07C4ACDDU) >> 27];
}

/** \returns the square root of \a x.
  *
  * It is essentially equivalent to \code using std::sqrt; return sqrt(x); \endcode,
  * but slightly faster for float/double and some compilers (e.g., gcc), thanks to
  * specializations when SSE is enabled.
  *
  * It's usage is justified in performance critical functions, like norm/normalize.
  */
template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T sqrt(const T &x)
{
  EIGEN_USING_STD_MATH(sqrt);
  return sqrt(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   sqrt(float x) { return cl::sycl::sqrt(x); }
EIGEN_ALWAYS_INLINE double  sqrt(double x) { return cl::sycl::sqrt(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T log(const T &x) {
  EIGEN_USING_STD_MATH(log);
  return log(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   log(float x) { return cl::sycl::log(x); }
EIGEN_ALWAYS_INLINE double  log(double x) { return cl::sycl::log(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)


#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float log(const float &x) { return ::logf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double log(const double &x) { return ::log(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
typename internal::enable_if<NumTraits<T>::IsSigned || NumTraits<T>::IsComplex,typename NumTraits<T>::Real>::type
abs(const T &x) {
  EIGEN_USING_STD_MATH(abs);
  return abs(x);
}

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
typename internal::enable_if<!(NumTraits<T>::IsSigned || NumTraits<T>::IsComplex),typename NumTraits<T>::Real>::type
abs(const T &x) {
  return x;
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   abs(float x) { return cl::sycl::fabs(x); }
EIGEN_ALWAYS_INLINE double  abs(double x) { return cl::sycl::fabs(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float abs(const float &x) { return ::fabsf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double abs(const double &x) { return ::fabs(x); }

template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float abs(const std::complex<float>& x) {
  return ::hypotf(x.real(), x.imag());
}

template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double abs(const std::complex<double>& x) {
  return ::hypot(x.real(), x.imag());
}
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T exp(const T &x) {
  EIGEN_USING_STD_MATH(exp);
  return exp(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   exp(float x) { return cl::sycl::exp(x); }
EIGEN_ALWAYS_INLINE double  exp(double x) { return cl::sycl::exp(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float exp(const float &x) { return ::expf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double exp(const double &x) { return ::exp(x); }
#endif

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(expm1, Scalar) expm1(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(expm1, Scalar)::run(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   expm1(float x) { return cl::sycl::expm1(x); }
EIGEN_ALWAYS_INLINE double  expm1(double x) { return cl::sycl::expm1(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float expm1(const float &x) { return ::expm1f(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double expm1(const double &x) { return ::expm1(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T cos(const T &x) {
  EIGEN_USING_STD_MATH(cos);
  return cos(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   cos(float x) { return cl::sycl::cos(x); }
EIGEN_ALWAYS_INLINE double  cos(double x) { return cl::sycl::cos(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float cos(const float &x) { return ::cosf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double cos(const double &x) { return ::cos(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T sin(const T &x) {
  EIGEN_USING_STD_MATH(sin);
  return sin(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   sin(float x) { return cl::sycl::sin(x); }
EIGEN_ALWAYS_INLINE double  sin(double x) { return cl::sycl::sin(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float sin(const float &x) { return ::sinf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double sin(const double &x) { return ::sin(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T tan(const T &x) {
  EIGEN_USING_STD_MATH(tan);
  return tan(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   tan(float x) { return cl::sycl::tan(x); }
EIGEN_ALWAYS_INLINE double  tan(double x) { return cl::sycl::tan(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float tan(const float &x) { return ::tanf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double tan(const double &x) { return ::tan(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T acos(const T &x) {
  EIGEN_USING_STD_MATH(acos);
  return acos(x);
}

#if EIGEN_HAS_CXX11_MATH
template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T acosh(const T &x) {
  EIGEN_USING_STD_MATH(acosh);
  return acosh(x);
}
#endif

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   acos(float x) { return cl::sycl::acos(x); }
EIGEN_ALWAYS_INLINE double  acos(double x) { return cl::sycl::acos(x); }
EIGEN_ALWAYS_INLINE float   acosh(float x) { return cl::sycl::acosh(x); }
EIGEN_ALWAYS_INLINE double  acosh(double x) { return cl::sycl::acosh(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float acos(const float &x) { return ::acosf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double acos(const double &x) { return ::acos(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T asin(const T &x) {
  EIGEN_USING_STD_MATH(asin);
  return asin(x);
}

#if EIGEN_HAS_CXX11_MATH
template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T asinh(const T &x) {
  EIGEN_USING_STD_MATH(asinh);
  return asinh(x);
}
#endif

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   asin(float x) { return cl::sycl::asin(x); }
EIGEN_ALWAYS_INLINE double  asin(double x) { return cl::sycl::asin(x); }
EIGEN_ALWAYS_INLINE float   asinh(float x) { return cl::sycl::asinh(x); }
EIGEN_ALWAYS_INLINE double  asinh(double x) { return cl::sycl::asinh(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float asin(const float &x) { return ::asinf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double asin(const double &x) { return ::asin(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T atan(const T &x) {
  EIGEN_USING_STD_MATH(atan);
  return atan(x);
}

#if EIGEN_HAS_CXX11_MATH
template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T atanh(const T &x) {
  EIGEN_USING_STD_MATH(atanh);
  return atanh(x);
}
#endif

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   atan(float x) { return cl::sycl::atan(x); }
EIGEN_ALWAYS_INLINE double  atan(double x) { return cl::sycl::atan(x); }
EIGEN_ALWAYS_INLINE float   atanh(float x) { return cl::sycl::atanh(x); }
EIGEN_ALWAYS_INLINE double  atanh(double x) { return cl::sycl::atanh(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float atan(const float &x) { return ::atanf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double atan(const double &x) { return ::atan(x); }
#endif


template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T cosh(const T &x) {
  EIGEN_USING_STD_MATH(cosh);
  return cosh(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   cosh(float x) { return cl::sycl::cosh(x); }
EIGEN_ALWAYS_INLINE double  cosh(double x) { return cl::sycl::cosh(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float cosh(const float &x) { return ::coshf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double cosh(const double &x) { return ::cosh(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T sinh(const T &x) {
  EIGEN_USING_STD_MATH(sinh);
  return sinh(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   sinh(float x) { return cl::sycl::sinh(x); }
EIGEN_ALWAYS_INLINE double  sinh(double x) { return cl::sycl::sinh(x); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float sinh(const float &x) { return ::sinhf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double sinh(const double &x) { return ::sinh(x); }
#endif

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T tanh(const T &x) {
  EIGEN_USING_STD_MATH(tanh);
  return tanh(x);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   tanh(float x) { return cl::sycl::tanh(x); }
EIGEN_ALWAYS_INLINE double  tanh(double x) { return cl::sycl::tanh(x); }
#elif (!defined(EIGEN_CUDACC)) && EIGEN_FAST_MATH
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float tanh(float x) { return internal::generic_fast_tanh_float(x); }
#endif

#ifdef EIGEN_CUDACC
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float tanh(const float &x) { return ::tanhf(x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double tanh(const double &x) { return ::tanh(x); }
#endif

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T fmod(const T& a, const T& b) {
  EIGEN_USING_STD_MATH(fmod);
  return fmod(a, b);
}

#if defined(__SYCL_DEVICE_ONLY__)
EIGEN_ALWAYS_INLINE float   fmod(float x, float y) { return cl::sycl::fmod(x, y); }
EIGEN_ALWAYS_INLINE double  fmod(double x, double y) { return cl::sycl::fmod(x, y); }
#endif // defined(__SYCL_DEVICE_ONLY__)

#ifdef EIGEN_CUDACC
template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float fmod(const float& a, const float& b) {
  return ::fmodf(a, b);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double fmod(const double& a, const double& b) {
  return ::fmod(a, b);
}
#endif

} // end namespace numext

namespace internal {

template<typename T>
EIGEN_DEVICE_FUNC bool isfinite_impl(const std::complex<T>& x)
{
  return (numext::isfinite)(numext::real(x)) && (numext::isfinite)(numext::imag(x));
}

template<typename T>
EIGEN_DEVICE_FUNC bool isnan_impl(const std::complex<T>& x)
{
  return (numext::isnan)(numext::real(x)) || (numext::isnan)(numext::imag(x));
}

template<typename T>
EIGEN_DEVICE_FUNC bool isinf_impl(const std::complex<T>& x)
{
  return ((numext::isinf)(numext::real(x)) || (numext::isinf)(numext::imag(x))) && (!(numext::isnan)(x));
}

/****************************************************************************
* Implementation of fuzzy comparisons                                       *
****************************************************************************/

template<typename Scalar,
         bool IsComplex,
         bool IsInteger>
struct scalar_fuzzy_default_impl {};

template<typename Scalar>
struct scalar_fuzzy_default_impl<Scalar, false, false>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  template<typename OtherScalar> EIGEN_DEVICE_FUNC
  static inline bool isMuchSmallerThan(const Scalar& x, const OtherScalar& y, const RealScalar& prec)
  {
    return numext::abs(x) <= numext::abs(y) * prec;
  }
  EIGEN_DEVICE_FUNC
  static inline bool isApprox(const Scalar& x, const Scalar& y, const RealScalar& prec)
  {
    return numext::abs(x - y) <= numext::mini(numext::abs(x), numext::abs(y)) * prec;
  }
  EIGEN_DEVICE_FUNC
  static inline bool isApproxOrLessThan(const Scalar& x, const Scalar& y, const RealScalar& prec)
  {
    return x <= y || isApprox(x, y, prec);
  }
};

template<typename Scalar>
struct scalar_fuzzy_default_impl<Scalar, false, true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  template<typename OtherScalar> EIGEN_DEVICE_FUNC
  static inline bool isMuchSmallerThan(const Scalar& x, const Scalar&, const RealScalar&)
  {
    return x == Scalar(0);
  }
  EIGEN_DEVICE_FUNC
  static inline bool isApprox(const Scalar& x, const Scalar& y, const RealScalar&)
  {
    return x == y;
  }
  EIGEN_DEVICE_FUNC
  static inline bool isApproxOrLessThan(const Scalar& x, const Scalar& y, const RealScalar&)
  {
    return x <= y;
  }
};

template<typename Scalar>
struct scalar_fuzzy_default_impl<Scalar, true, false>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  template<typename OtherScalar> EIGEN_DEVICE_FUNC
  static inline bool isMuchSmallerThan(const Scalar& x, const OtherScalar& y, const RealScalar& prec)
  {
    return numext::abs2(x) <= numext::abs2(y) * prec * prec;
  }
  EIGEN_DEVICE_FUNC
  static inline bool isApprox(const Scalar& x, const Scalar& y, const RealScalar& prec)
  {
    return numext::abs2(x - y) <= numext::mini(numext::abs2(x), numext::abs2(y)) * prec * prec;
  }
};

template<typename Scalar>
struct scalar_fuzzy_impl : scalar_fuzzy_default_impl<Scalar, NumTraits<Scalar>::IsComplex, NumTraits<Scalar>::IsInteger> {};

template<typename Scalar, typename OtherScalar> EIGEN_DEVICE_FUNC
inline bool isMuchSmallerThan(const Scalar& x, const OtherScalar& y,
                              const typename NumTraits<Scalar>::Real &precision = NumTraits<Scalar>::dummy_precision())
{
  return scalar_fuzzy_impl<Scalar>::template isMuchSmallerThan<OtherScalar>(x, y, precision);
}

template<typename Scalar> EIGEN_DEVICE_FUNC
inline bool isApprox(const Scalar& x, const Scalar& y,
                     const typename NumTraits<Scalar>::Real &precision = NumTraits<Scalar>::dummy_precision())
{
  return scalar_fuzzy_impl<Scalar>::isApprox(x, y, precision);
}

template<typename Scalar> EIGEN_DEVICE_FUNC
inline bool isApproxOrLessThan(const Scalar& x, const Scalar& y,
                               const typename NumTraits<Scalar>::Real &precision = NumTraits<Scalar>::dummy_precision())
{
  return scalar_fuzzy_impl<Scalar>::isApproxOrLessThan(x, y, precision);
}

/******************************************
***  The special case of the  bool type ***
******************************************/

template<> struct random_impl<bool>
{
  static inline bool run()
  {
    return random<int>(0,1)==0 ? false : true;
  }
};

template<> struct scalar_fuzzy_impl<bool>
{
  typedef bool RealScalar;

  template<typename OtherScalar> EIGEN_DEVICE_FUNC
  static inline bool isMuchSmallerThan(const bool& x, const bool&, const bool&)
  {
    return !x;
  }

  EIGEN_DEVICE_FUNC
  static inline bool isApprox(bool x, bool y, bool)
  {
    return x == y;
  }

  EIGEN_DEVICE_FUNC
  static inline bool isApproxOrLessThan(const bool& x, const bool& y, const bool&)
  {
    return (!x) || y;
  }

};


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_MATHFUNCTIONS_H
