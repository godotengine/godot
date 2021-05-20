// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Transcendental functions from "ispc": https://github.com/ispc/ispc/
// Most of the transcendental implementations in ispc code come from
// Solomon Boulos's "syrah": https://github.com/boulos/syrah/

#include "../simd/simd.h"

namespace embree
{

namespace fastapprox
{

template <typename T>
__forceinline T sin(const T &v)
{
  static const float piOverTwoVec = 1.57079637050628662109375;
  static const float twoOverPiVec = 0.636619746685028076171875;
  auto scaled = v * twoOverPiVec;
  auto kReal = floor(scaled);
  auto k = toInt(kReal);

  // Reduced range version of x
  auto x = v - kReal * piOverTwoVec;
  auto kMod4 = k & 3;
  auto sinUseCos = (kMod4 == 1 | kMod4 == 3);
  auto flipSign = (kMod4 > 1);

  // These coefficients are from sollya with fpminimax(sin(x)/x, [|0, 2,
  // 4, 6, 8, 10|], [|single...|], [0;Pi/2]);
  static const float sinC2  = -0.16666667163372039794921875;
  static const float sinC4  = +8.333347737789154052734375e-3;
  static const float sinC6  = -1.9842604524455964565277099609375e-4;
  static const float sinC8  = +2.760012648650445044040679931640625e-6;
  static const float sinC10 = -2.50293279435709337121807038784027099609375e-8;

  static const float cosC2  = -0.5;
  static const float cosC4  = +4.166664183139801025390625e-2;
  static const float cosC6  = -1.388833043165504932403564453125e-3;
  static const float cosC8  = +2.47562347794882953166961669921875e-5;
  static const float cosC10 = -2.59630184018533327616751194000244140625e-7;

  auto outside = select(sinUseCos, 1., x);
  auto c2  = select(sinUseCos, T(cosC2),  T(sinC2));
  auto c4  = select(sinUseCos, T(cosC4),  T(sinC4));
  auto c6  = select(sinUseCos, T(cosC6),  T(sinC6));
  auto c8  = select(sinUseCos, T(cosC8),  T(sinC8));
  auto c10 = select(sinUseCos, T(cosC10), T(sinC10));

  auto x2 = x * x;
  auto formula = x2 * c10 + c8;
  formula = x2 * formula + c6;
  formula = x2 * formula + c4;
  formula = x2 * formula + c2;
  formula = x2 * formula + 1.;
  formula *= outside;

  formula = select(flipSign, -formula, formula);
  return formula;
}

template <typename T>
__forceinline T cos(const T &v)
{
  static const float piOverTwoVec = 1.57079637050628662109375;
  static const float twoOverPiVec = 0.636619746685028076171875;
  auto scaled = v * twoOverPiVec;
  auto kReal = floor(scaled);
  auto k = toInt(kReal);

  // Reduced range version of x
  auto x = v - kReal * piOverTwoVec;

  auto kMod4 = k & 3;
  auto cosUseCos = (kMod4 == 0 | kMod4 == 2);
  auto flipSign = (kMod4 == 1 | kMod4 == 2);

  const float sinC2  = -0.16666667163372039794921875;
  const float sinC4  = +8.333347737789154052734375e-3;
  const float sinC6  = -1.9842604524455964565277099609375e-4;
  const float sinC8  = +2.760012648650445044040679931640625e-6;
  const float sinC10 = -2.50293279435709337121807038784027099609375e-8;

  const float cosC2  = -0.5;
  const float cosC4  = +4.166664183139801025390625e-2;
  const float cosC6  = -1.388833043165504932403564453125e-3;
  const float cosC8  = +2.47562347794882953166961669921875e-5;
  const float cosC10 = -2.59630184018533327616751194000244140625e-7;

  auto outside = select(cosUseCos, 1., x);
  auto c2  = select(cosUseCos, T(cosC2),  T(sinC2));
  auto c4  = select(cosUseCos, T(cosC4),  T(sinC4));
  auto c6  = select(cosUseCos, T(cosC6),  T(sinC6));
  auto c8  = select(cosUseCos, T(cosC8),  T(sinC8));
  auto c10 = select(cosUseCos, T(cosC10), T(sinC10));

  auto x2 = x * x;
  auto formula = x2 * c10 + c8;
  formula = x2 * formula + c6;
  formula = x2 * formula + c4;
  formula = x2 * formula + c2;
  formula = x2 * formula + 1.;
  formula *= outside;

  formula = select(flipSign, -formula, formula);
  return formula;
}

template <typename T>
__forceinline void sincos(const T &v, T &sinResult, T &cosResult)
{
  const float piOverTwoVec = 1.57079637050628662109375;
  const float twoOverPiVec = 0.636619746685028076171875;
  auto scaled = v * twoOverPiVec;
  auto kReal = floor(scaled);
  auto k = toInt(kReal);

  // Reduced range version of x
  auto x = v - kReal * piOverTwoVec;
  auto kMod4 = k & 3;
  auto cosUseCos = ((kMod4 == 0) | (kMod4 == 2));
  auto sinUseCos = ((kMod4 == 1) | (kMod4 == 3));
  auto sinFlipSign = (kMod4 > 1);
  auto cosFlipSign = ((kMod4 == 1) | (kMod4 == 2));

  const float oneVec = +1.;
  const float sinC2  = -0.16666667163372039794921875;
  const float sinC4  = +8.333347737789154052734375e-3;
  const float sinC6  = -1.9842604524455964565277099609375e-4;
  const float sinC8  = +2.760012648650445044040679931640625e-6;
  const float sinC10 = -2.50293279435709337121807038784027099609375e-8;

  const float cosC2  = -0.5;
  const float cosC4  = +4.166664183139801025390625e-2;
  const float cosC6  = -1.388833043165504932403564453125e-3;
  const float cosC8  = +2.47562347794882953166961669921875e-5;
  const float cosC10 = -2.59630184018533327616751194000244140625e-7;

  auto x2 = x * x;

  auto sinFormula = x2 * sinC10 + sinC8;
  auto cosFormula = x2 * cosC10 + cosC8;
  sinFormula = x2 * sinFormula + sinC6;
  cosFormula = x2 * cosFormula + cosC6;

  sinFormula = x2 * sinFormula + sinC4;
  cosFormula = x2 * cosFormula + cosC4;

  sinFormula = x2 * sinFormula + sinC2;
  cosFormula = x2 * cosFormula + cosC2;

  sinFormula = x2 * sinFormula + oneVec;
  cosFormula = x2 * cosFormula + oneVec;

  sinFormula *= x;

  sinResult = select(sinUseCos, cosFormula, sinFormula);
  cosResult = select(cosUseCos, cosFormula, sinFormula);

  sinResult = select(sinFlipSign, -sinResult, sinResult);
  cosResult = select(cosFlipSign, -cosResult, cosResult);
}

template <typename T>
__forceinline T tan(const T &v)
{
  const float piOverFourVec = 0.785398185253143310546875;
  const float fourOverPiVec = 1.27323949337005615234375;

  auto xLt0 = v < 0.;
  auto y = select(xLt0, -v, v);
  auto scaled = y * fourOverPiVec;

  auto kReal = floor(scaled);
  auto k = toInt(kReal);

  auto x = y - kReal * piOverFourVec;

  // If k & 1, x -= Pi/4
  auto needOffset = (k & 1) != 0;
  x = select(needOffset, x - piOverFourVec, x);

  // If k & 3 == (0 or 3) let z = tan_In...(y) otherwise z = -cot_In0To...
  auto kMod4 = k & 3;
  auto useCotan = (kMod4 == 1) | (kMod4 == 2);

  const float oneVec = 1.0;

  const float tanC2  = +0.33333075046539306640625;
  const float tanC4  = +0.13339905440807342529296875;
  const float tanC6  = +5.3348250687122344970703125e-2;
  const float tanC8  = +2.46033705770969390869140625e-2;
  const float tanC10 = +2.892402000725269317626953125e-3;
  const float tanC12 = +9.500005282461643218994140625e-3;

  const float cotC2  = -0.3333333432674407958984375;
  const float cotC4  = -2.222204394638538360595703125e-2;
  const float cotC6  = -2.11752182804048061370849609375e-3;
  const float cotC8  = -2.0846328698098659515380859375e-4;
  const float cotC10 = -2.548247357481159269809722900390625e-5;
  const float cotC12 = -3.5257363606433500535786151885986328125e-7;

  auto x2 = x * x;
  T z;
  if (any(useCotan))
  {
    auto cotVal = x2 * cotC12 + cotC10;
    cotVal = x2 * cotVal + cotC8;
    cotVal = x2 * cotVal + cotC6;
    cotVal = x2 * cotVal + cotC4;
    cotVal = x2 * cotVal + cotC2;
    cotVal = x2 * cotVal + oneVec;
    // The equation is for x * cot(x) but we need -x * cot(x) for the tan part.
    cotVal /= -x;
    z = cotVal;
  }
  auto useTan = !useCotan;
  if (any(useTan))
  {
    auto tanVal = x2 * tanC12 + tanC10;
    tanVal = x2 * tanVal + tanC8;
    tanVal = x2 * tanVal + tanC6;
    tanVal = x2 * tanVal + tanC4;
    tanVal = x2 * tanVal + tanC2;
    tanVal = x2 * tanVal + oneVec;
    // Equation was for tan(x)/x
    tanVal *= x;
    z = select(useTan, tanVal, z);
  }
  return select(xLt0, -z, z);
}

template <typename T>
__forceinline T asin(const T &x0)
{
  auto isneg = (x0 < 0.f);
  auto x = abs(x0);
  auto isnan = (x > 1.f);

  // sollya
  // fpminimax(((asin(x)-pi/2)/-sqrt(1-x)), [|0,1,2,3,4,5|],[|single...|],
  //           [1e-20;.9999999999999999]);
  // avg error: 1.1105439e-06, max error 1.3187528e-06
  auto v = 1.57079517841339111328125f +
           x * (-0.21450997889041900634765625f +
                x * (8.78556668758392333984375e-2f +
                     x * (-4.489909112453460693359375e-2f +
                          x * (1.928029954433441162109375e-2f +
                               x * (-4.3095736764371395111083984375e-3f)))));

  v *= -sqrt(1.f - x);
  v = v + 1.57079637050628662109375f;

  v = select(v < 0.f, T(0.f), v);
  v = select(isneg, -v, v);
  v = select(isnan, T(cast_i2f(0x7fc00000)), v);

  return v;
}

template <typename T>
__forceinline T acos(const T &v)
{
  return 1.57079637050628662109375f - asin(v);
}

template <typename T>
__forceinline T atan(const T &v)
{
  const float piOverTwoVec = 1.57079637050628662109375;
  // atan(-x) = -atan(x) (so flip from negative to positive first)
  // If x > 1 -> atan(x) = Pi/2 - atan(1/x)
  auto xNeg = v < 0.f;
  auto xFlipped = select(xNeg, -v, v);

  auto xGt1 = xFlipped > 1.;
  auto x = select(xGt1, rcpSafe(xFlipped), xFlipped);

  // These coefficients approximate atan(x)/x
  const float atanC0  = +0.99999988079071044921875;
  const float atanC2  = -0.3333191573619842529296875;
  const float atanC4  = +0.199689209461212158203125;
  const float atanC6  = -0.14015688002109527587890625;
  const float atanC8  = +9.905083477497100830078125e-2;
  const float atanC10 = -5.93664981424808502197265625e-2;
  const float atanC12 = +2.417283318936824798583984375e-2;
  const float atanC14 = -4.6721356920897960662841796875e-3;

  auto x2 = x * x;
  auto result = x2 * atanC14 + atanC12;
  result = x2 * result + atanC10;
  result = x2 * result + atanC8;
  result = x2 * result + atanC6;
  result = x2 * result + atanC4;
  result = x2 * result + atanC2;
  result = x2 * result + atanC0;
  result *= x;

  result = select(xGt1, piOverTwoVec - result, result);
  result = select(xNeg, -result, result);
  return result;
}

template <typename T>
__forceinline T atan2(const T &y, const T &x)
{
  const float piVec = 3.1415926536;
  // atan2(y, x) =
  //
  // atan2(y > 0, x = +-0) ->  Pi/2
  // atan2(y < 0, x = +-0) -> -Pi/2
  // atan2(y = +-0, x < +0) -> +-Pi
  // atan2(y = +-0, x >= +0) -> +-0
  //
  // atan2(y >= 0, x < 0) ->  Pi + atan(y/x)
  // atan2(y <  0, x < 0) -> -Pi + atan(y/x)
  // atan2(y, x > 0) -> atan(y/x)
  //
  // and then a bunch of code for dealing with infinities.
  auto yOverX = y * rcpSafe(x);
  auto atanArg = atan(yOverX);
  auto xLt0 = x < 0.f;
  auto yLt0 = y < 0.f;
  auto offset = select(xLt0,
                select(yLt0, T(-piVec), T(piVec)), 0.f);
  return offset + atanArg;
}

template <typename T>
__forceinline T exp(const T &v)
{
  const float ln2Part1 = 0.6931457519;
  const float ln2Part2 = 1.4286067653e-6;
  const float oneOverLn2 = 1.44269502162933349609375;

  auto scaled = v * oneOverLn2;
  auto kReal = floor(scaled);
  auto k = toInt(kReal);

  // Reduced range version of x
  auto x = v - kReal * ln2Part1;
  x -= kReal * ln2Part2;

  // These coefficients are for e^x in [0, ln(2)]
  const float one = 1.;
  const float c2 = 0.4999999105930328369140625;
  const float c3 = 0.166668415069580078125;
  const float c4 = 4.16539050638675689697265625e-2;
  const float c5 = 8.378830738365650177001953125e-3;
  const float c6 = 1.304379315115511417388916015625e-3;
  const float c7 = 2.7555381529964506626129150390625e-4;

  auto result = x * c7 + c6;
  result = x * result + c5;
  result = x * result + c4;
  result = x * result + c3;
  result = x * result + c2;
  result = x * result + one;
  result = x * result + one;

  // Compute 2^k (should differ for float and double, but I'll avoid
  // it for now and just do floats)
  const int fpbias = 127;
  auto biasedN = k + fpbias;
  auto overflow = kReal > fpbias;
  // Minimum exponent is -126, so if k is <= -127 (k + 127 <= 0)
  // we've got underflow. -127 * ln(2) -> -88.02. So the most
  // negative float input that doesn't result in zero is like -88.
  auto underflow = kReal <= -fpbias;
  const int infBits = 0x7f800000;
  biasedN <<= 23;
  // Reinterpret this thing as float
  auto twoToTheN = asFloat(biasedN);
  // Handle both doubles and floats (hopefully eliding the copy for float)
  auto elemtype2n = twoToTheN;
  result *= elemtype2n;
  result = select(overflow, cast_i2f(infBits), result);
  result = select(underflow, 0., result);
  return result;
}

// Range reduction for logarithms takes log(x) -> log(2^n * y) -> n
// * log(2) + log(y) where y is the reduced range (usually in [1/2, 1)).
template <typename T, typename R>
__forceinline void __rangeReduceLog(const T &input,
                                    T &reduced,
                                    R &exponent)
{
  auto intVersion = asInt(input);
  // single precision = SEEE EEEE EMMM MMMM MMMM MMMM MMMM MMMM
  // exponent mask    = 0111 1111 1000 0000 0000 0000 0000 0000
  //                    0x7  0xF  0x8  0x0  0x0  0x0  0x0  0x0
  // non-exponent     = 1000 0000 0111 1111 1111 1111 1111 1111
  //                  = 0x8  0x0  0x7  0xF  0xF  0xF  0xF  0xF

  //const int exponentMask(0x7F800000)
  static const int nonexponentMask = 0x807FFFFF;

  // We want the reduced version to have an exponent of -1 which is
  // -1 + 127 after biasing or 126
  static const int exponentNeg1 = (126l << 23);
  // NOTE(boulos): We don't need to mask anything out since we know
  // the sign bit has to be 0. If it's 1, we need to return infinity/nan
  // anyway (log(x), x = +-0 -> infinity, x < 0 -> NaN).
  auto biasedExponent = intVersion >> 23; // This number is [0, 255] but it means [-127, 128]

  auto offsetExponent = biasedExponent + 1; // Treat the number as if it were 2^{e+1} * (1.m)/2
  exponent = offsetExponent - 127;          // get the real value

  // Blend the offset_exponent with the original input (do this in
  // int for now, until I decide if float can have & and &not)
  auto blended = (intVersion & nonexponentMask) | (exponentNeg1);
  reduced = asFloat(blended);
}

template <typename T> struct ExponentType            { };
template <int N>      struct ExponentType<vfloat_impl<N>> { typedef vint<N> Ty; };
template <>           struct ExponentType<float>     { typedef int     Ty; };

template <typename T>
__forceinline T log(const T &v)
{
  T reduced;
  typename ExponentType<T>::Ty exponent;

  const int nanBits = 0x7fc00000;
  const int negInfBits = 0xFF800000;
  const float nan = cast_i2f(nanBits);
  const float negInf = cast_i2f(negInfBits);
  auto useNan = v < 0.;
  auto useInf = v == 0.;
  auto exceptional = useNan | useInf;
  const float one = 1.0;

  auto patched = select(exceptional, one, v);
  __rangeReduceLog(patched, reduced, exponent);

  const float ln2 = 0.693147182464599609375;

  auto x1 = one - reduced;
  const float c1 = +0.50000095367431640625;
  const float c2 = +0.33326041698455810546875;
  const float c3 = +0.2519190013408660888671875;
  const float c4 = +0.17541764676570892333984375;
  const float c5 = +0.3424419462680816650390625;
  const float c6 = -0.599632322788238525390625;
  const float c7 = +1.98442304134368896484375;
  const float c8 = -2.4899270534515380859375;
  const float c9 = +1.7491014003753662109375;

  auto result = x1 * c9 + c8;
  result = x1 * result + c7;
  result = x1 * result + c6;
  result = x1 * result + c5;
  result = x1 * result + c4;
  result = x1 * result + c3;
  result = x1 * result + c2;
  result = x1 * result + c1;
  result = x1 * result + one;

  // Equation was for -(ln(red)/(1-red))
  result *= -x1;
  result += toFloat(exponent) * ln2;

  return select(exceptional,
                select(useNan, T(nan), T(negInf)),
                result);
}

template <typename T>
__forceinline T pow(const T &x, const T &y)
{
  auto x1 = abs(x);
  auto z = exp(y * log(x1));

  // Handle special cases
  const float twoOver23 = 8388608.0f;
  auto yInt = y == round(y);
  auto yOddInt = select(yInt, asInt(abs(y) + twoOver23) << 31, 0); // set sign bit

  // x == 0
  z = select(x == 0.0f,
      select(y < 0.0f, T(inf) | signmsk(x),
      select(y == 0.0f, T(1.0f), asFloat(yOddInt) & x)), z);

  // x < 0
  auto xNegative = x < 0.0f;
  if (any(xNegative))
  {
    auto z1 = z | asFloat(yOddInt);
    z1 = select(yInt, z1, std::numeric_limits<float>::quiet_NaN());
    z = select(xNegative, z1, z);
  }

  auto xFinite = isfinite(x);
  auto yFinite = isfinite(y);
  if (all(xFinite & yFinite))
    return z;

  // x finite and y infinite
  z = select(andn(xFinite, yFinite),
      select(x1 == 1.0f, 1.0f,
      select((x1 > 1.0f) ^ (y < 0.0f), inf, T(0.0f))), z);

  // x infinite
  z = select(xFinite, z,
      select(y == 0.0f, 1.0f,
      select(y < 0.0f, T(0.0f), inf) | (asFloat(yOddInt) & x)));

  return z;
}

template <typename T>
__forceinline T pow(const T &x, float y)
{
  return pow(x, T(y));
}

} // namespace fastapprox

} // namespace embree
