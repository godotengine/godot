//===-- APFloat.cpp - Implement APFloat class -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision floating
// point values and provide a variety of arithmetic operations on them.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cstring>
#include <limits.h>

using namespace llvm;

/// A macro used to combine two fcCategory enums into one key which can be used
/// in a switch statement to classify how the interaction of two APFloat's
/// categories affects an operation.
///
/// TODO: If clang source code is ever allowed to use constexpr in its own
/// codebase, change this into a static inline function.
#define PackCategoriesIntoKey(_lhs, _rhs) ((_lhs) * 4 + (_rhs))

/* Assumed in hexadecimal significand parsing, and conversion to
   hexadecimal strings.  */
static_assert(integerPartWidth % 4 == 0, "Part width must be divisible by 4!");

namespace llvm {

  /* Represents floating point arithmetic semantics.  */
  struct fltSemantics {
    /* The largest E such that 2^E is representable; this matches the
       definition of IEEE 754.  */
    APFloat::ExponentType maxExponent;

    /* The smallest E such that 2^E is a normalized number; this
       matches the definition of IEEE 754.  */
    APFloat::ExponentType minExponent;

    /* Number of bits in the significand.  This includes the integer
       bit.  */
    unsigned int precision;

    /* Number of bits actually used in the semantics. */
    unsigned int sizeInBits;
  };

  const fltSemantics APFloat::IEEEhalf = { 15, -14, 11, 16 };
  const fltSemantics APFloat::IEEEsingle = { 127, -126, 24, 32 };
  const fltSemantics APFloat::IEEEdouble = { 1023, -1022, 53, 64 };
  const fltSemantics APFloat::IEEEquad = { 16383, -16382, 113, 128 };
  const fltSemantics APFloat::x87DoubleExtended = { 16383, -16382, 64, 80 };
  const fltSemantics APFloat::Bogus = { 0, 0, 0, 0 };

  /* The PowerPC format consists of two doubles.  It does not map cleanly
     onto the usual format above.  It is approximated using twice the
     mantissa bits.  Note that for exponents near the double minimum,
     we no longer can represent the full 106 mantissa bits, so those
     will be treated as denormal numbers.

     FIXME: While this approximation is equivalent to what GCC uses for
     compile-time arithmetic on PPC double-double numbers, it is not able
     to represent all possible values held by a PPC double-double number,
     for example: (long double) 1.0 + (long double) 0x1p-106
     Should this be replaced by a full emulation of PPC double-double?  */
  const fltSemantics APFloat::PPCDoubleDouble = { 1023, -1022 + 53, 53 + 53, 128 };

  /* A tight upper bound on number of parts required to hold the value
     pow(5, power) is

       power * 815 / (351 * integerPartWidth) + 1

     However, whilst the result may require only this many parts,
     because we are multiplying two values to get it, the
     multiplication may require an extra part with the excess part
     being zero (consider the trivial case of 1 * 1, tcFullMultiply
     requires two parts to hold the single-part result).  So we add an
     extra one to guarantee enough space whilst multiplying.  */
  const unsigned int maxExponent = 16383;
  const unsigned int maxPrecision = 113;
  const unsigned int maxPowerOfFiveExponent = maxExponent + maxPrecision - 1;
  const unsigned int maxPowerOfFiveParts = 2 + ((maxPowerOfFiveExponent * 815)
                                                / (351 * integerPartWidth));
}

/* A bunch of private, handy routines.  */

static inline unsigned int
partCountForBits(unsigned int bits)
{
  return ((bits) + integerPartWidth - 1) / integerPartWidth;
}

/* Returns 0U-9U.  Return values >= 10U are not digits.  */
static inline unsigned int
decDigitValue(unsigned int c)
{
  return c - '0';
}

/* Return the value of a decimal exponent of the form
   [+-]ddddddd.

   If the exponent overflows, returns a large exponent with the
   appropriate sign.  */
static int
readExponent(StringRef::iterator begin, StringRef::iterator end)
{
  bool isNegative;
  unsigned int absExponent;
  const unsigned int overlargeExponent = 24000;  /* FIXME.  */
  StringRef::iterator p = begin;

  assert(p != end && "Exponent has no digits");

  isNegative = (*p == '-');
  if (*p == '-' || *p == '+') {
    p++;
    assert(p != end && "Exponent has no digits");
  }

  absExponent = decDigitValue(*p++);
  assert(absExponent < 10U && "Invalid character in exponent");

  for (; p != end; ++p) {
    unsigned int value;

    value = decDigitValue(*p);
    assert(value < 10U && "Invalid character in exponent");

    value += absExponent * 10;
    if (absExponent >= overlargeExponent) {
      absExponent = overlargeExponent;
      p = end;  /* outwit assert below */
      break;
    }
    absExponent = value;
  }

  assert(p == end && "Invalid exponent in exponent");

  if (isNegative)
    return -(int) absExponent;
  else
    return (int) absExponent;
}

/* This is ugly and needs cleaning up, but I don't immediately see
   how whilst remaining safe.  */
static int
totalExponent(StringRef::iterator p, StringRef::iterator end,
              int exponentAdjustment)
{
  int unsignedExponent;
  bool negative, overflow;
  int exponent = 0;

  assert(p != end && "Exponent has no digits");

  negative = *p == '-';
  if (*p == '-' || *p == '+') {
    p++;
    assert(p != end && "Exponent has no digits");
  }

  unsignedExponent = 0;
  overflow = false;
  for (; p != end; ++p) {
    unsigned int value;

    value = decDigitValue(*p);
    assert(value < 10U && "Invalid character in exponent");

    unsignedExponent = unsignedExponent * 10 + value;
    if (unsignedExponent > 32767) {
      overflow = true;
      break;
    }
  }

  if (exponentAdjustment > 32767 || exponentAdjustment < -32768)
    overflow = true;

  if (!overflow) {
    exponent = unsignedExponent;
    if (negative)
      exponent = -exponent;
    exponent += exponentAdjustment;
    if (exponent > 32767 || exponent < -32768)
      overflow = true;
  }

  if (overflow)
    exponent = negative ? -32768: 32767;

  return exponent;
}

static StringRef::iterator
skipLeadingZeroesAndAnyDot(StringRef::iterator begin, StringRef::iterator end,
                           StringRef::iterator *dot)
{
  StringRef::iterator p = begin;
  *dot = end;
  while (p != end && *p == '0')
    p++;

  if (p != end && *p == '.') {
    *dot = p++;

    assert(end - begin != 1 && "Significand has no digits");

    while (p != end && *p == '0')
      p++;
  }

  return p;
}

/* Given a normal decimal floating point number of the form

     dddd.dddd[eE][+-]ddd

   where the decimal point and exponent are optional, fill out the
   structure D.  Exponent is appropriate if the significand is
   treated as an integer, and normalizedExponent if the significand
   is taken to have the decimal point after a single leading
   non-zero digit.

   If the value is zero, V->firstSigDigit points to a non-digit, and
   the return exponent is zero.
*/
struct decimalInfo {
  const char *firstSigDigit;
  const char *lastSigDigit;
  int exponent;
  int normalizedExponent;
};

static void
interpretDecimal(StringRef::iterator begin, StringRef::iterator end,
                 decimalInfo *D)
{
  StringRef::iterator dot = end;
  StringRef::iterator p = skipLeadingZeroesAndAnyDot (begin, end, &dot);

  D->firstSigDigit = p;
  D->exponent = 0;
  D->normalizedExponent = 0;

  for (; p != end; ++p) {
    if (*p == '.') {
      assert(dot == end && "String contains multiple dots");
      dot = p++;
      if (p == end)
        break;
    }
    if (decDigitValue(*p) >= 10U)
      break;
  }

  if (p != end) {
    assert((*p == 'e' || *p == 'E') && "Invalid character in significand");
    assert(p != begin && "Significand has no digits");
    assert((dot == end || p - begin != 1) && "Significand has no digits");

    /* p points to the first non-digit in the string */
    D->exponent = readExponent(p + 1, end);

    /* Implied decimal point?  */
    if (dot == end)
      dot = p;
  }

  /* If number is all zeroes accept any exponent.  */
  if (p != D->firstSigDigit) {
    /* Drop insignificant trailing zeroes.  */
    if (p != begin) {
      do
        do
          p--;
        while (p != begin && *p == '0');
      while (p != begin && *p == '.');
    }

    /* Adjust the exponents for any decimal point.  */
    D->exponent += static_cast<APFloat::ExponentType>((dot - p) - (dot > p));
    D->normalizedExponent = (D->exponent +
              static_cast<APFloat::ExponentType>((p - D->firstSigDigit)
                                      - (dot > D->firstSigDigit && dot < p)));
  }

  D->lastSigDigit = p;
}

/* Return the trailing fraction of a hexadecimal number.
   DIGITVALUE is the first hex digit of the fraction, P points to
   the next digit.  */
static lostFraction
trailingHexadecimalFraction(StringRef::iterator p, StringRef::iterator end,
                            unsigned int digitValue)
{
  unsigned int hexDigit;

  /* If the first trailing digit isn't 0 or 8 we can work out the
     fraction immediately.  */
  if (digitValue > 8)
    return lfMoreThanHalf;
  else if (digitValue < 8 && digitValue > 0)
    return lfLessThanHalf;

  // Otherwise we need to find the first non-zero digit.
  while (p != end && (*p == '0' || *p == '.'))
    p++;

  assert(p != end && "Invalid trailing hexadecimal fraction!");

  hexDigit = hexDigitValue(*p);

  /* If we ran off the end it is exactly zero or one-half, otherwise
     a little more.  */
  if (hexDigit == -1U)
    return digitValue == 0 ? lfExactlyZero: lfExactlyHalf;
  else
    return digitValue == 0 ? lfLessThanHalf: lfMoreThanHalf;
}

/* Return the fraction lost were a bignum truncated losing the least
   significant BITS bits.  */
static lostFraction
lostFractionThroughTruncation(const integerPart *parts,
                              unsigned int partCount,
                              unsigned int bits)
{
  unsigned int lsb;

  lsb = APInt::tcLSB(parts, partCount);

  /* Note this is guaranteed true if bits == 0, or LSB == -1U.  */
  if (bits <= lsb)
    return lfExactlyZero;
  if (bits == lsb + 1)
    return lfExactlyHalf;
  if (bits <= partCount * integerPartWidth &&
      APInt::tcExtractBit(parts, bits - 1))
    return lfMoreThanHalf;

  return lfLessThanHalf;
}

/* Shift DST right BITS bits noting lost fraction.  */
static lostFraction
shiftRight(integerPart *dst, unsigned int parts, unsigned int bits)
{
  lostFraction lost_fraction;

  lost_fraction = lostFractionThroughTruncation(dst, parts, bits);

  APInt::tcShiftRight(dst, parts, bits);

  return lost_fraction;
}

/* Combine the effect of two lost fractions.  */
static lostFraction
combineLostFractions(lostFraction moreSignificant,
                     lostFraction lessSignificant)
{
  if (lessSignificant != lfExactlyZero) {
    if (moreSignificant == lfExactlyZero)
      moreSignificant = lfLessThanHalf;
    else if (moreSignificant == lfExactlyHalf)
      moreSignificant = lfMoreThanHalf;
  }

  return moreSignificant;
}

/* The error from the true value, in half-ulps, on multiplying two
   floating point numbers, which differ from the value they
   approximate by at most HUE1 and HUE2 half-ulps, is strictly less
   than the returned value.

   See "How to Read Floating Point Numbers Accurately" by William D
   Clinger.  */
static unsigned int
HUerrBound(bool inexactMultiply, unsigned int HUerr1, unsigned int HUerr2)
{
  assert(HUerr1 < 2 || HUerr2 < 2 || (HUerr1 + HUerr2 < 8));

  if (HUerr1 + HUerr2 == 0)
    return inexactMultiply * 2;  /* <= inexactMultiply half-ulps.  */
  else
    return inexactMultiply + 2 * (HUerr1 + HUerr2);
}

/* The number of ulps from the boundary (zero, or half if ISNEAREST)
   when the least significant BITS are truncated.  BITS cannot be
   zero.  */
static integerPart
ulpsFromBoundary(const integerPart *parts, unsigned int bits, bool isNearest)
{
  unsigned int count, partBits;
  integerPart part, boundary;

  assert(bits != 0);

  bits--;
  count = bits / integerPartWidth;
  partBits = bits % integerPartWidth + 1;

  part = parts[count] & (~(integerPart) 0 >> (integerPartWidth - partBits));

  if (isNearest)
    boundary = (integerPart) 1 << (partBits - 1);
  else
    boundary = 0;

  if (count == 0) {
    if (part - boundary <= boundary - part)
      return part - boundary;
    else
      return boundary - part;
  }

  if (part == boundary) {
    while (--count)
      if (parts[count])
        return ~(integerPart) 0; /* A lot.  */

    return parts[0];
  } else if (part == boundary - 1) {
    while (--count)
      if (~parts[count])
        return ~(integerPart) 0; /* A lot.  */

    return -parts[0];
  }

  return ~(integerPart) 0; /* A lot.  */
}

/* Place pow(5, power) in DST, and return the number of parts used.
   DST must be at least one part larger than size of the answer.  */
static unsigned int
powerOf5(integerPart *dst, unsigned int power)
{
  static const integerPart firstEightPowers[] = { 1, 5, 25, 125, 625, 3125,
                                                  15625, 78125 };
  integerPart pow5s[maxPowerOfFiveParts * 2 + 5];
  pow5s[0] = 78125 * 5;

  unsigned int partsCount[16] = { 1 };
  integerPart scratch[maxPowerOfFiveParts], *p1, *p2, *pow5;
  unsigned int result;
  assert(power <= maxExponent);

  p1 = dst;
  p2 = scratch;

  *p1 = firstEightPowers[power & 7];
  power >>= 3;

  result = 1;
  pow5 = pow5s;

  for (unsigned int n = 0; power; power >>= 1, n++) {
    unsigned int pc;

    pc = partsCount[n];

    /* Calculate pow(5,pow(2,n+3)) if we haven't yet.  */
    if (pc == 0) {
      pc = partsCount[n - 1];
      APInt::tcFullMultiply(pow5, pow5 - pc, pow5 - pc, pc, pc);
      pc *= 2;
      if (pow5[pc - 1] == 0)
        pc--;
      partsCount[n] = pc;
    }

    if (power & 1) {
      integerPart *tmp;

      APInt::tcFullMultiply(p2, p1, pow5, result, pc);
      result += pc;
      if (p2[result - 1] == 0)
        result--;

      /* Now result is in p1 with partsCount parts and p2 is scratch
         space.  */
      tmp = p1, p1 = p2, p2 = tmp;
    }

    pow5 += pc;
  }

  if (p1 != dst)
    APInt::tcAssign(dst, p1, result);

  return result;
}

#if 0 // HLSL Change
/* Zero at the end to avoid modular arithmetic when adding one; used
   when rounding up during hexadecimal output.  */
static const char hexDigitsLower[] = "0123456789abcdef0";
static const char hexDigitsUpper[] = "0123456789ABCDEF0";
static const char infinityL[] = "infinity";
static const char infinityU[] = "INFINITY";
static const char NaNL[] = "nan";
static const char NaNU[] = "NAN";

/* Write out an integerPart in hexadecimal, starting with the most
   significant nibble.  Write out exactly COUNT hexdigits, return
   COUNT.  */
static unsigned int
partAsHex (_Out_writes_(count) char *dst, integerPart part, unsigned int count,
           _In_count_(16) const char *hexDigitChars)
{
  unsigned int result = count;

  assert(count != 0 && count <= integerPartWidth / 4);

  part >>= (integerPartWidth - 4 * count);
  while (count--) {
    dst[count] = hexDigitChars[part & 0xf];
    part >>= 4;
  }

  return result;
}

/* Write out an unsigned decimal integer.  */
static char *
writeUnsignedDecimal (_Out_writes_(10) char *dst, unsigned int n) // HLSL Change: '4294967295' is ten characters
{
  char buff[40], *p;

  p = buff;
  do
    *p++ = '0' + n % 10;
  while (n /= 10);

  do
    *dst++ = *--p;
  while (p != buff);

  return dst;
}

/* Write out a signed decimal integer.  */
static char *
writeSignedDecimal(_Out_writes_(11) char *dst, int value) // HLSL Change: '-2147483648' is eleven characters
{
  if (value < 0) {
    *dst++ = '-';
    dst = writeUnsignedDecimal(dst, -(unsigned) value);
  } else
    dst = writeUnsignedDecimal(dst, value);

  return dst;
}
#endif // HLSL Change

/* Constructors.  */
void
APFloat::initialize(const fltSemantics *ourSemantics)
{
  unsigned int count;

  semantics = ourSemantics;
  count = partCount();
  if (count > 1)
    significand.parts = new integerPart[count];
}

void
APFloat::freeSignificand()
{
  if (needsCleanup())
    delete [] significand.parts;
}

void
APFloat::assign(const APFloat &rhs)
{
  assert(semantics == rhs.semantics);

  sign = rhs.sign;
  category = rhs.category;
  exponent = rhs.exponent;
  if (isFiniteNonZero() || category == fcNaN)
    copySignificand(rhs);
}

void
APFloat::copySignificand(const APFloat &rhs)
{
  assert(isFiniteNonZero() || category == fcNaN);
  assert(rhs.partCount() >= partCount());

  APInt::tcAssign(significandParts(), rhs.significandParts(),
                  partCount());
}

/* Make this number a NaN, with an arbitrary but deterministic value
   for the significand.  If double or longer, this is a signalling NaN,
   which may not be ideal.  If float, this is QNaN(0).  */
void APFloat::makeNaN(bool SNaN, bool Negative, const APInt *fill)
{
  category = fcNaN;
  sign = Negative;

  integerPart *significand = significandParts();
  unsigned numParts = partCount();

  // Set the significand bits to the fill.
  if (!fill || fill->getNumWords() < numParts)
    APInt::tcSet(significand, 0, numParts);
  if (fill) {
    APInt::tcAssign(significand, fill->getRawData(),
                    std::min(fill->getNumWords(), numParts));

    // Zero out the excess bits of the significand.
    unsigned bitsToPreserve = semantics->precision - 1;
    unsigned part = bitsToPreserve / 64;
    bitsToPreserve %= 64;
    significand[part] &= ((1ULL << bitsToPreserve) - 1);
    for (part++; part != numParts; ++part)
      significand[part] = 0;
  }

  unsigned QNaNBit = semantics->precision - 2;

  if (SNaN) {
    // We always have to clear the QNaN bit to make it an SNaN.
    APInt::tcClearBit(significand, QNaNBit);

    // If there are no bits set in the payload, we have to set
    // *something* to make it a NaN instead of an infinity;
    // conventionally, this is the next bit down from the QNaN bit.
    if (APInt::tcIsZero(significand, numParts))
      APInt::tcSetBit(significand, QNaNBit - 1);
  } else {
    // We always have to set the QNaN bit to make it a QNaN.
    APInt::tcSetBit(significand, QNaNBit);
  }

  // For x87 extended precision, we want to make a NaN, not a
  // pseudo-NaN.  Maybe we should expose the ability to make
  // pseudo-NaNs?
  if (semantics == &APFloat::x87DoubleExtended)
    APInt::tcSetBit(significand, QNaNBit + 1);
}

APFloat APFloat::makeNaN(const fltSemantics &Sem, bool SNaN, bool Negative,
                         const APInt *fill) {
  APFloat value(Sem, uninitialized);
  value.makeNaN(SNaN, Negative, fill);
  return value;
}

APFloat &
APFloat::operator=(const APFloat &rhs)
{
  if (this != &rhs) {
    if (semantics != rhs.semantics) {
      freeSignificand();
      initialize(rhs.semantics);
    }
    assign(rhs);
  }

  return *this;
}

APFloat &
APFloat::operator=(APFloat &&rhs) {
  freeSignificand();

  semantics = rhs.semantics;
  significand = rhs.significand;
  exponent = rhs.exponent;
  category = rhs.category;
  sign = rhs.sign;

  rhs.semantics = &Bogus;
  return *this;
}

bool
APFloat::isDenormal() const {
  return isFiniteNonZero() && (exponent == semantics->minExponent) &&
         (APInt::tcExtractBit(significandParts(), 
                              semantics->precision - 1) == 0);
}

bool
APFloat::isSmallest() const {
  // The smallest number by magnitude in our format will be the smallest
  // denormal, i.e. the floating point number with exponent being minimum
  // exponent and significand bitwise equal to 1 (i.e. with MSB equal to 0).
  return isFiniteNonZero() && exponent == semantics->minExponent &&
    significandMSB() == 0;
}

bool APFloat::isSignificandAllOnes() const {
  // Test if the significand excluding the integral bit is all ones. This allows
  // us to test for binade boundaries.
  const integerPart *Parts = significandParts();
  const unsigned PartCount = partCount();
  for (unsigned i = 0; i < PartCount - 1; i++)
    if (~Parts[i])
      return false;

  // Set the unused high bits to all ones when we compare.
  const unsigned NumHighBits =
    PartCount*integerPartWidth - semantics->precision + 1;
  assert(NumHighBits <= integerPartWidth && "Can not have more high bits to "
         "fill than integerPartWidth");
  const integerPart HighBitFill =
    ~integerPart(0) << (integerPartWidth - NumHighBits);
  if (~(Parts[PartCount - 1] | HighBitFill))
    return false;

  return true;
}

bool APFloat::isSignificandAllZeros() const {
  // Test if the significand excluding the integral bit is all zeros. This
  // allows us to test for binade boundaries.
  const integerPart *Parts = significandParts();
  const unsigned PartCount = partCount();

  for (unsigned i = 0; i < PartCount - 1; i++)
    if (Parts[i])
      return false;

  const unsigned NumHighBits =
    PartCount*integerPartWidth - semantics->precision + 1;
  assert(NumHighBits <= integerPartWidth && "Can not have more high bits to "
         "clear than integerPartWidth");
  const integerPart HighBitMask = ~integerPart(0) >> NumHighBits;

  if (Parts[PartCount - 1] & HighBitMask)
    return false;

  return true;
}

bool
APFloat::isLargest() const {
  // The largest number by magnitude in our format will be the floating point
  // number with maximum exponent and with significand that is all ones.
  return isFiniteNonZero() && exponent == semantics->maxExponent
    && isSignificandAllOnes();
}

bool
APFloat::bitwiseIsEqual(const APFloat &rhs) const {
  if (this == &rhs)
    return true;
  if (semantics != rhs.semantics ||
      category != rhs.category ||
      sign != rhs.sign)
    return false;
  if (category==fcZero || category==fcInfinity)
    return true;
  else if (isFiniteNonZero() && exponent!=rhs.exponent)
    return false;
  else {
    int i= partCount();
    const integerPart* p=significandParts();
    const integerPart* q=rhs.significandParts();
    for (; i>0; i--, p++, q++) {
      if (*p != *q)
        return false;
    }
    return true;
  }
}

APFloat::APFloat(const fltSemantics &ourSemantics, integerPart value) {
  initialize(&ourSemantics);
  sign = 0;
  category = fcNormal;
  zeroSignificand();
  exponent = ourSemantics.precision - 1;
  significandParts()[0] = value;
  normalize(rmNearestTiesToEven, lfExactlyZero);
}

APFloat::APFloat(const fltSemantics &ourSemantics) {
  initialize(&ourSemantics);
  category = fcZero;
  sign = false;
}

APFloat::APFloat(const fltSemantics &ourSemantics, uninitializedTag tag) {
  // Allocates storage if necessary but does not initialize it.
  initialize(&ourSemantics);
}

APFloat::APFloat(const fltSemantics &ourSemantics, StringRef text) {
  initialize(&ourSemantics);
  convertFromString(text, rmNearestTiesToEven);
}

APFloat::APFloat(const APFloat &rhs) {
  initialize(rhs.semantics);
  assign(rhs);
}

APFloat::APFloat(APFloat &&rhs) : semantics(&Bogus) {
  *this = std::move(rhs);
}

APFloat::~APFloat()
{
  freeSignificand();
}

// Profile - This method 'profiles' an APFloat for use with FoldingSet.
void APFloat::Profile(FoldingSetNodeID& ID) const {
  ID.Add(bitcastToAPInt());
}

unsigned int
APFloat::partCount() const
{
  return partCountForBits(semantics->precision + 1);
}

unsigned int
APFloat::semanticsPrecision(const fltSemantics &semantics)
{
  return semantics.precision;
}

const integerPart *
APFloat::significandParts() const
{
  return const_cast<APFloat *>(this)->significandParts();
}

integerPart *
APFloat::significandParts()
{
  if (partCount() > 1)
    return significand.parts;
  else
    return &significand.part;
}

void
APFloat::zeroSignificand()
{
  APInt::tcSet(significandParts(), 0, partCount());
}

/* Increment an fcNormal floating point number's significand.  */
void
APFloat::incrementSignificand()
{
  integerPart carry;

  carry = APInt::tcIncrement(significandParts(), partCount());

  /* Our callers should never cause us to overflow.  */
  assert(carry == 0);
  (void)carry;
}

/* Add the significand of the RHS.  Returns the carry flag.  */
integerPart
APFloat::addSignificand(const APFloat &rhs)
{
  integerPart *parts;

  parts = significandParts();

  assert(semantics == rhs.semantics);
  assert(exponent == rhs.exponent);

  return APInt::tcAdd(parts, rhs.significandParts(), 0, partCount());
}

/* Subtract the significand of the RHS with a borrow flag.  Returns
   the borrow flag.  */
integerPart
APFloat::subtractSignificand(const APFloat &rhs, integerPart borrow)
{
  integerPart *parts;

  parts = significandParts();

  assert(semantics == rhs.semantics);
  assert(exponent == rhs.exponent);

  return APInt::tcSubtract(parts, rhs.significandParts(), borrow,
                           partCount());
}

/* Multiply the significand of the RHS.  If ADDEND is non-NULL, add it
   on to the full-precision result of the multiplication.  Returns the
   lost fraction.  */
lostFraction
APFloat::multiplySignificand(const APFloat &rhs, const APFloat *addend)
{
  unsigned int omsb;        // One, not zero, based MSB.
  unsigned int partsCount, newPartsCount, precision;
  integerPart *lhsSignificand;
  integerPart scratch[4];
  integerPart *fullSignificand;
  lostFraction lost_fraction;
  bool ignored;

  assert(semantics == rhs.semantics);

  precision = semantics->precision;

  // Allocate space for twice as many bits as the original significand, plus one
  // extra bit for the addition to overflow into.
  newPartsCount = partCountForBits(precision * 2 + 1);

  if (newPartsCount > 4)
    fullSignificand = new integerPart[newPartsCount];
  else
    fullSignificand = scratch;

  lhsSignificand = significandParts();
  partsCount = partCount();

  APInt::tcFullMultiply(fullSignificand, lhsSignificand,
                        rhs.significandParts(), partsCount, partsCount);

  lost_fraction = lfExactlyZero;
  omsb = APInt::tcMSB(fullSignificand, newPartsCount) + 1;
  exponent += rhs.exponent;

  // Assume the operands involved in the multiplication are single-precision
  // FP, and the two multiplicants are:
  //   *this = a23 . a22 ... a0 * 2^e1
  //     rhs = b23 . b22 ... b0 * 2^e2
  // the result of multiplication is:
  //   *this = c48 c47 c46 . c45 ... c0 * 2^(e1+e2)
  // Note that there are three significant bits at the left-hand side of the 
  // radix point: two for the multiplication, and an overflow bit for the
  // addition (that will always be zero at this point). Move the radix point
  // toward left by two bits, and adjust exponent accordingly.
  exponent += 2;

  if (addend && addend->isNonZero()) {
    // The intermediate result of the multiplication has "2 * precision" 
    // signicant bit; adjust the addend to be consistent with mul result.
    //
    Significand savedSignificand = significand;
    const fltSemantics *savedSemantics = semantics;
    fltSemantics extendedSemantics;
    opStatus status;
    unsigned int extendedPrecision;

    // Normalize our MSB to one below the top bit to allow for overflow.
    extendedPrecision = 2 * precision + 1;
    if (omsb != extendedPrecision - 1) {
      assert(extendedPrecision > omsb);
      APInt::tcShiftLeft(fullSignificand, newPartsCount,
                         (extendedPrecision - 1) - omsb);
      exponent -= (extendedPrecision - 1) - omsb;
    }

    /* Create new semantics.  */
    extendedSemantics = *semantics;
    extendedSemantics.precision = extendedPrecision;

    if (newPartsCount == 1)
      significand.part = fullSignificand[0];
    else
      significand.parts = fullSignificand;
    semantics = &extendedSemantics;

    APFloat extendedAddend(*addend);
    status = extendedAddend.convert(extendedSemantics, rmTowardZero, &ignored);
    assert(status == opOK);
    (void)status;

    // Shift the significand of the addend right by one bit. This guarantees
    // that the high bit of the significand is zero (same as fullSignificand),
    // so the addition will overflow (if it does overflow at all) into the top bit.
    lost_fraction = extendedAddend.shiftSignificandRight(1);
    assert(lost_fraction == lfExactlyZero &&
           "Lost precision while shifting addend for fused-multiply-add.");

    lost_fraction = addOrSubtractSignificand(extendedAddend, false);

    /* Restore our state.  */
    if (newPartsCount == 1)
      fullSignificand[0] = significand.part;
    significand = savedSignificand;
    semantics = savedSemantics;

    omsb = APInt::tcMSB(fullSignificand, newPartsCount) + 1;
  }

  // Convert the result having "2 * precision" significant-bits back to the one
  // having "precision" significant-bits. First, move the radix point from 
  // poision "2*precision - 1" to "precision - 1". The exponent need to be
  // adjusted by "2*precision - 1" - "precision - 1" = "precision".
  exponent -= precision + 1;

  // In case MSB resides at the left-hand side of radix point, shift the
  // mantissa right by some amount to make sure the MSB reside right before
  // the radix point (i.e. "MSB . rest-significant-bits").
  //
  // Note that the result is not normalized when "omsb < precision". So, the
  // caller needs to call APFloat::normalize() if normalized value is expected.
  if (omsb > precision) {
    unsigned int bits, significantParts;
    lostFraction lf;

    bits = omsb - precision;
    significantParts = partCountForBits(omsb);
    lf = shiftRight(fullSignificand, significantParts, bits);
    lost_fraction = combineLostFractions(lf, lost_fraction);
    exponent += bits;
  }

  APInt::tcAssign(lhsSignificand, fullSignificand, partsCount);

  if (newPartsCount > 4)
    delete [] fullSignificand;

  return lost_fraction;
}

/* Multiply the significands of LHS and RHS to DST.  */
lostFraction
APFloat::divideSignificand(const APFloat &rhs)
{
  unsigned int bit, i, partsCount;
  const integerPart *rhsSignificand;
  integerPart *lhsSignificand, *dividend, *divisor;
  integerPart scratch[4];
  lostFraction lost_fraction;

  assert(semantics == rhs.semantics);

  lhsSignificand = significandParts();
  rhsSignificand = rhs.significandParts();
  partsCount = partCount();

  if (partsCount > 2)
    dividend = new integerPart[partsCount * 2];
  else
    dividend = scratch;

  divisor = dividend + partsCount;

  /* Copy the dividend and divisor as they will be modified in-place.  */
  for (i = 0; i < partsCount; i++) {
    dividend[i] = lhsSignificand[i];
    divisor[i] = rhsSignificand[i];
    lhsSignificand[i] = 0;
  }

  exponent -= rhs.exponent;

  unsigned int precision = semantics->precision;

  /* Normalize the divisor.  */
  bit = precision - APInt::tcMSB(divisor, partsCount) - 1;
  if (bit) {
    exponent += bit;
    APInt::tcShiftLeft(divisor, partsCount, bit);
  }

  /* Normalize the dividend.  */
  bit = precision - APInt::tcMSB(dividend, partsCount) - 1;
  if (bit) {
    exponent -= bit;
    APInt::tcShiftLeft(dividend, partsCount, bit);
  }

  /* Ensure the dividend >= divisor initially for the loop below.
     Incidentally, this means that the division loop below is
     guaranteed to set the integer bit to one.  */
  if (APInt::tcCompare(dividend, divisor, partsCount) < 0) {
    exponent--;
    APInt::tcShiftLeft(dividend, partsCount, 1);
    assert(APInt::tcCompare(dividend, divisor, partsCount) >= 0);
  }

  /* Long division.  */
  for (bit = precision; bit; bit -= 1) {
    if (APInt::tcCompare(dividend, divisor, partsCount) >= 0) {
      APInt::tcSubtract(dividend, divisor, 0, partsCount);
      APInt::tcSetBit(lhsSignificand, bit - 1);
    }

    APInt::tcShiftLeft(dividend, partsCount, 1);
  }

  /* Figure out the lost fraction.  */
  int cmp = APInt::tcCompare(dividend, divisor, partsCount);

  if (cmp > 0)
    lost_fraction = lfMoreThanHalf;
  else if (cmp == 0)
    lost_fraction = lfExactlyHalf;
  else if (APInt::tcIsZero(dividend, partsCount))
    lost_fraction = lfExactlyZero;
  else
    lost_fraction = lfLessThanHalf;

  if (partsCount > 2)
    delete [] dividend;

  return lost_fraction;
}

unsigned int
APFloat::significandMSB() const
{
  return APInt::tcMSB(significandParts(), partCount());
}

unsigned int
APFloat::significandLSB() const
{
  return APInt::tcLSB(significandParts(), partCount());
}

/* Note that a zero result is NOT normalized to fcZero.  */
lostFraction
APFloat::shiftSignificandRight(unsigned int bits)
{
  /* Our exponent should not overflow.  */
  assert((ExponentType) (exponent + bits) >= exponent);

  exponent += bits;

  return shiftRight(significandParts(), partCount(), bits);
}

/* Shift the significand left BITS bits, subtract BITS from its exponent.  */
void
APFloat::shiftSignificandLeft(unsigned int bits)
{
  assert(bits < semantics->precision);

  if (bits) {
    unsigned int partsCount = partCount();

    APInt::tcShiftLeft(significandParts(), partsCount, bits);
    exponent -= bits;

    assert(!APInt::tcIsZero(significandParts(), partsCount));
  }
}

APFloat::cmpResult
APFloat::compareAbsoluteValue(const APFloat &rhs) const
{
  int compare;

  assert(semantics == rhs.semantics);
  assert(isFiniteNonZero());
  assert(rhs.isFiniteNonZero());

  compare = exponent - rhs.exponent;

  /* If exponents are equal, do an unsigned bignum comparison of the
     significands.  */
  if (compare == 0)
    compare = APInt::tcCompare(significandParts(), rhs.significandParts(),
                               partCount());

  if (compare > 0)
    return cmpGreaterThan;
  else if (compare < 0)
    return cmpLessThan;
  else
    return cmpEqual;
}

/* Handle overflow.  Sign is preserved.  We either become infinity or
   the largest finite number.  */
APFloat::opStatus
APFloat::handleOverflow(roundingMode rounding_mode)
{
  /* Infinity?  */
  if (rounding_mode == rmNearestTiesToEven ||
      rounding_mode == rmNearestTiesToAway ||
      (rounding_mode == rmTowardPositive && !sign) ||
      (rounding_mode == rmTowardNegative && sign)) {
    category = fcInfinity;
    return (opStatus) (opOverflow | opInexact);
  }

  /* Otherwise we become the largest finite number.  */
  category = fcNormal;
  exponent = semantics->maxExponent;
  APInt::tcSetLeastSignificantBits(significandParts(), partCount(),
                                   semantics->precision);

  return opInexact;
}

/* Returns TRUE if, when truncating the current number, with BIT the
   new LSB, with the given lost fraction and rounding mode, the result
   would need to be rounded away from zero (i.e., by increasing the
   signficand).  This routine must work for fcZero of both signs, and
   fcNormal numbers.  */
bool
APFloat::roundAwayFromZero(roundingMode rounding_mode,
                           lostFraction lost_fraction,
                           unsigned int bit) const
{
  /* NaNs and infinities should not have lost fractions.  */
  assert(isFiniteNonZero() || category == fcZero);

  /* Current callers never pass this so we don't handle it.  */
  assert(lost_fraction != lfExactlyZero);

  switch (rounding_mode) {
  case rmNearestTiesToAway:
    return lost_fraction == lfExactlyHalf || lost_fraction == lfMoreThanHalf;

  case rmNearestTiesToEven:
    if (lost_fraction == lfMoreThanHalf)
      return true;

    /* Our zeroes don't have a significand to test.  */
    if (lost_fraction == lfExactlyHalf && category != fcZero)
      return APInt::tcExtractBit(significandParts(), bit);

    return false;

  case rmTowardZero:
    return false;

  case rmTowardPositive:
    return !sign;

  case rmTowardNegative:
    return sign;
  }
  llvm_unreachable("Invalid rounding mode found");
}

APFloat::opStatus
APFloat::normalize(roundingMode rounding_mode,
                   lostFraction lost_fraction)
{
  unsigned int omsb;                /* One, not zero, based MSB.  */
  int exponentChange;

  if (!isFiniteNonZero())
    return opOK;

  /* Before rounding normalize the exponent of fcNormal numbers.  */
  omsb = significandMSB() + 1;

  if (omsb) {
    /* OMSB is numbered from 1.  We want to place it in the integer
       bit numbered PRECISION if possible, with a compensating change in
       the exponent.  */
    exponentChange = omsb - semantics->precision;

    /* If the resulting exponent is too high, overflow according to
       the rounding mode.  */
    if (exponent + exponentChange > semantics->maxExponent)
      return handleOverflow(rounding_mode);

    /* Subnormal numbers have exponent minExponent, and their MSB
       is forced based on that.  */
    if (exponent + exponentChange < semantics->minExponent)
      exponentChange = semantics->minExponent - exponent;

    /* Shifting left is easy as we don't lose precision.  */
    if (exponentChange < 0) {
      assert(lost_fraction == lfExactlyZero);

      shiftSignificandLeft(-exponentChange);

      return opOK;
    }

    if (exponentChange > 0) {
      lostFraction lf;

      /* Shift right and capture any new lost fraction.  */
      lf = shiftSignificandRight(exponentChange);

      lost_fraction = combineLostFractions(lf, lost_fraction);

      /* Keep OMSB up-to-date.  */
      if (omsb > (unsigned) exponentChange)
        omsb -= exponentChange;
      else
        omsb = 0;
    }
  }

  /* Now round the number according to rounding_mode given the lost
     fraction.  */

  /* As specified in IEEE 754, since we do not trap we do not report
     underflow for exact results.  */
  if (lost_fraction == lfExactlyZero) {
    /* Canonicalize zeroes.  */
    if (omsb == 0)
      category = fcZero;

    return opOK;
  }

  /* Increment the significand if we're rounding away from zero.  */
  if (roundAwayFromZero(rounding_mode, lost_fraction, 0)) {
    if (omsb == 0)
      exponent = semantics->minExponent;

    incrementSignificand();
    omsb = significandMSB() + 1;

    /* Did the significand increment overflow?  */
    if (omsb == (unsigned) semantics->precision + 1) {
      /* Renormalize by incrementing the exponent and shifting our
         significand right one.  However if we already have the
         maximum exponent we overflow to infinity.  */
      if (exponent == semantics->maxExponent) {
        category = fcInfinity;

        return (opStatus) (opOverflow | opInexact);
      }

      shiftSignificandRight(1);

      return opInexact;
    }
  }

  /* The normal case - we were and are not denormal, and any
     significand increment above didn't overflow.  */
  if (omsb == semantics->precision)
    return opInexact;

  /* We have a non-zero denormal.  */
  assert(omsb < semantics->precision);

  /* Canonicalize zeroes.  */
  if (omsb == 0)
    category = fcZero;

  /* The fcZero case is a denormal that underflowed to zero.  */
  return (opStatus) (opUnderflow | opInexact);
}

APFloat::opStatus
APFloat::addOrSubtractSpecials(const APFloat &rhs, bool subtract)
{
  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcInfinity, fcZero):
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    // We need to be sure to flip the sign here for subtraction because we
    // don't have a separate negate operation so -NaN becomes 0 - NaN here.
    sign = rhs.sign ^ subtract;
    category = fcNaN;
    copySignificand(rhs);
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcInfinity):
    category = fcInfinity;
    sign = rhs.sign ^ subtract;
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcNormal):
    assign(rhs);
    sign = rhs.sign ^ subtract;
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcZero):
    /* Sign depends on rounding mode; handled by caller.  */
    return opOK;

  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
    /* Differently signed infinities can only be validly
       subtracted.  */
    if (((sign ^ rhs.sign)!=0) != subtract) {
      makeNaN();
      return opInvalidOp;
    }

    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    return opDivByZero;
  }
}

/* Add or subtract two normal numbers.  */
lostFraction
APFloat::addOrSubtractSignificand(const APFloat &rhs, bool subtract)
{
  integerPart carry;
  lostFraction lost_fraction;
  int bits;

  /* Determine if the operation on the absolute values is effectively
     an addition or subtraction.  */
  subtract ^= static_cast<bool>(sign ^ rhs.sign);

  /* Are we bigger exponent-wise than the RHS?  */
  bits = exponent - rhs.exponent;

  /* Subtraction is more subtle than one might naively expect.  */
  if (subtract) {
    APFloat temp_rhs(rhs);
    bool reverse;

    if (bits == 0) {
      reverse = compareAbsoluteValue(temp_rhs) == cmpLessThan;
      lost_fraction = lfExactlyZero;
    } else if (bits > 0) {
      lost_fraction = temp_rhs.shiftSignificandRight(bits - 1);
      shiftSignificandLeft(1);
      reverse = false;
    } else {
      lost_fraction = shiftSignificandRight(-bits - 1);
      temp_rhs.shiftSignificandLeft(1);
      reverse = true;
    }

    if (reverse) {
      carry = temp_rhs.subtractSignificand
        (*this, lost_fraction != lfExactlyZero);
      copySignificand(temp_rhs);
      sign = !sign;
    } else {
      carry = subtractSignificand
        (temp_rhs, lost_fraction != lfExactlyZero);
    }

    /* Invert the lost fraction - it was on the RHS and
       subtracted.  */
    if (lost_fraction == lfLessThanHalf)
      lost_fraction = lfMoreThanHalf;
    else if (lost_fraction == lfMoreThanHalf)
      lost_fraction = lfLessThanHalf;

    /* The code above is intended to ensure that no borrow is
       necessary.  */
    assert(!carry);
    (void)carry;
  } else {
    if (bits > 0) {
      APFloat temp_rhs(rhs);

      lost_fraction = temp_rhs.shiftSignificandRight(bits);
      carry = addSignificand(temp_rhs);
    } else {
      lost_fraction = shiftSignificandRight(-bits);
      carry = addSignificand(rhs);
    }

    /* We have a guard bit; generating a carry cannot happen.  */
    assert(!carry);
    (void)carry;
  }

  return lost_fraction;
}

APFloat::opStatus
APFloat::multiplySpecials(const APFloat &rhs)
{
  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
    sign = false;
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    sign = false;
    category = fcNaN;
    copySignificand(rhs);
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcInfinity):
  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
    category = fcInfinity;
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcNormal):
  case PackCategoriesIntoKey(fcNormal, fcZero):
  case PackCategoriesIntoKey(fcZero, fcZero):
    category = fcZero;
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcInfinity):
  case PackCategoriesIntoKey(fcInfinity, fcZero):
    makeNaN();
    return opInvalidOp;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    return opOK;
  }
}

APFloat::opStatus
APFloat::divideSpecials(const APFloat &rhs)
{
  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    category = fcNaN;
    copySignificand(rhs);
  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
    sign = false;
  case PackCategoriesIntoKey(fcInfinity, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcZero, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcNormal):
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcInfinity):
    category = fcZero;
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcZero):
    category = fcInfinity;
    return opDivByZero;

  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcZero):
    makeNaN();
    return opInvalidOp;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    return opOK;
  }
}

APFloat::opStatus
APFloat::modSpecials(const APFloat &rhs)
{
  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
  case PackCategoriesIntoKey(fcZero, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcNormal):
  case PackCategoriesIntoKey(fcNormal, fcInfinity):
    return opOK;

  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    sign = false;
    category = fcNaN;
    copySignificand(rhs);
    return opOK;

  case PackCategoriesIntoKey(fcNormal, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcZero):
  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcZero):
    makeNaN();
    return opInvalidOp;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    return opOK;
  }
}

/* Change sign.  */
void
APFloat::changeSign()
{
  /* Look mummy, this one's easy.  */
  sign = !sign;
}

void
APFloat::clearSign()
{
  /* So is this one. */
  sign = 0;
}

void
APFloat::copySign(const APFloat &rhs)
{
  /* And this one. */
  sign = rhs.sign;
}

/* Normalized addition or subtraction.  */
APFloat::opStatus
APFloat::addOrSubtract(const APFloat &rhs, roundingMode rounding_mode,
                       bool subtract)
{
  opStatus fs;

  fs = addOrSubtractSpecials(rhs, subtract);

  /* This return code means it was not a simple case.  */
  if (fs == opDivByZero) {
    lostFraction lost_fraction;

    lost_fraction = addOrSubtractSignificand(rhs, subtract);
    fs = normalize(rounding_mode, lost_fraction);

    /* Can only be zero if we lost no fraction.  */
    assert(category != fcZero || lost_fraction == lfExactlyZero);
  }

  /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
     positive zero unless rounding to minus infinity, except that
     adding two like-signed zeroes gives that zero.  */
  if (category == fcZero) {
    if (rhs.category != fcZero || (sign == rhs.sign) == subtract)
      sign = (rounding_mode == rmTowardNegative);
  }

  return fs;
}

/* Normalized addition.  */
APFloat::opStatus
APFloat::add(const APFloat &rhs, roundingMode rounding_mode)
{
  return addOrSubtract(rhs, rounding_mode, false);
}

/* Normalized subtraction.  */
APFloat::opStatus
APFloat::subtract(const APFloat &rhs, roundingMode rounding_mode)
{
  return addOrSubtract(rhs, rounding_mode, true);
}

/* Normalized multiply.  */
APFloat::opStatus
APFloat::multiply(const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;

  sign ^= rhs.sign;
  fs = multiplySpecials(rhs);

  if (isFiniteNonZero()) {
    lostFraction lost_fraction = multiplySignificand(rhs, nullptr);
    fs = normalize(rounding_mode, lost_fraction);
    if (lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);
  }

  return fs;
}

/* Normalized divide.  */
APFloat::opStatus
APFloat::divide(const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;

  sign ^= rhs.sign;
  fs = divideSpecials(rhs);

  if (isFiniteNonZero()) {
    lostFraction lost_fraction = divideSignificand(rhs);
    fs = normalize(rounding_mode, lost_fraction);
    if (lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);
  }

  return fs;
}

/* Normalized remainder.  This is not currently correct in all cases.  */
APFloat::opStatus
APFloat::remainder(const APFloat &rhs)
{
  opStatus fs;
  APFloat V = *this;
  unsigned int origSign = sign;

  fs = V.divide(rhs, rmNearestTiesToEven);
  if (fs == opDivByZero)
    return fs;

  int parts = partCount();
  integerPart *x = new integerPart[parts];
  bool ignored;
  fs = V.convertToInteger(x, parts * integerPartWidth, true,
                          rmNearestTiesToEven, &ignored);
  if (fs==opInvalidOp)
    return fs;

  fs = V.convertFromZeroExtendedInteger(x, parts * integerPartWidth, true,
                                        rmNearestTiesToEven);
  assert(fs==opOK);   // should always work

  fs = V.multiply(rhs, rmNearestTiesToEven);
  assert(fs==opOK || fs==opInexact);   // should not overflow or underflow

  fs = subtract(V, rmNearestTiesToEven);
  assert(fs==opOK || fs==opInexact);   // likewise

  if (isZero())
    sign = origSign;    // IEEE754 requires this
  delete[] x;
  return fs;
}

/* Normalized llvm frem (C fmod).
   This is not currently correct in all cases.  */
APFloat::opStatus
APFloat::mod(const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;
  fs = modSpecials(rhs);

  if (isFiniteNonZero() && rhs.isFiniteNonZero()) {
    APFloat V = *this;
    unsigned int origSign = sign;

    fs = V.divide(rhs, rmNearestTiesToEven);
    if (fs == opDivByZero)
      return fs;

    int parts = partCount();
    integerPart *x = new integerPart[parts];
    bool ignored;
    fs = V.convertToInteger(x, parts * integerPartWidth, true,
                            rmTowardZero, &ignored);
    if (fs==opInvalidOp)
      return fs;

    fs = V.convertFromZeroExtendedInteger(x, parts * integerPartWidth, true,
                                          rmNearestTiesToEven);
    assert(fs==opOK);   // should always work

    fs = V.multiply(rhs, rounding_mode);
    assert(fs==opOK || fs==opInexact);   // should not overflow or underflow

    fs = subtract(V, rounding_mode);
    assert(fs==opOK || fs==opInexact);   // likewise

    if (isZero())
      sign = origSign;    // IEEE754 requires this
    delete[] x;
  }
  return fs;
}

/* Normalized fused-multiply-add.  */
APFloat::opStatus
APFloat::fusedMultiplyAdd(const APFloat &multiplicand,
                          const APFloat &addend,
                          roundingMode rounding_mode)
{
  opStatus fs;

  /* Post-multiplication sign, before addition.  */
  sign ^= multiplicand.sign;

  /* If and only if all arguments are normal do we need to do an
     extended-precision calculation.  */
  if (isFiniteNonZero() &&
      multiplicand.isFiniteNonZero() &&
      addend.isFinite()) {
    lostFraction lost_fraction;

    lost_fraction = multiplySignificand(multiplicand, &addend);
    fs = normalize(rounding_mode, lost_fraction);
    if (lost_fraction != lfExactlyZero)
      fs = (opStatus) (fs | opInexact);

    /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
       positive zero unless rounding to minus infinity, except that
       adding two like-signed zeroes gives that zero.  */
    if (category == fcZero && !(fs & opUnderflow) && sign != addend.sign)
      sign = (rounding_mode == rmTowardNegative);
  } else {
    fs = multiplySpecials(multiplicand);

    /* FS can only be opOK or opInvalidOp.  There is no more work
       to do in the latter case.  The IEEE-754R standard says it is
       implementation-defined in this case whether, if ADDEND is a
       quiet NaN, we raise invalid op; this implementation does so.

       If we need to do the addition we can do so with normal
       precision.  */
    if (fs == opOK)
      fs = addOrSubtract(addend, rounding_mode, false);
  }

  return fs;
}

/* Rounding-mode corrrect round to integral value.  */
APFloat::opStatus APFloat::roundToIntegral(roundingMode rounding_mode) {
  opStatus fs;

  // If the exponent is large enough, we know that this value is already
  // integral, and the arithmetic below would potentially cause it to saturate
  // to +/-Inf.  Bail out early instead.
  if (isFiniteNonZero() && exponent+1 >= (int)semanticsPrecision(*semantics))
    return opOK;

  // The algorithm here is quite simple: we add 2^(p-1), where p is the
  // precision of our format, and then subtract it back off again.  The choice
  // of rounding modes for the addition/subtraction determines the rounding mode
  // for our integral rounding as well.
  // NOTE: When the input value is negative, we do subtraction followed by
  // addition instead.
  APInt IntegerConstant(NextPowerOf2(semanticsPrecision(*semantics)), 1);
  IntegerConstant <<= semanticsPrecision(*semantics)-1;
  APFloat MagicConstant(*semantics);
  fs = MagicConstant.convertFromAPInt(IntegerConstant, false,
                                      rmNearestTiesToEven);
  MagicConstant.copySign(*this);

  if (fs != opOK)
    return fs;

  // Preserve the input sign so that we can handle 0.0/-0.0 cases correctly.
  bool inputSign = isNegative();

  fs = add(MagicConstant, rounding_mode);
  if (fs != opOK && fs != opInexact)
    return fs;

  fs = subtract(MagicConstant, rounding_mode);

  // Restore the input sign.
  if (inputSign != isNegative())
    changeSign();

  return fs;
}


/* Comparison requires normalized numbers.  */
APFloat::cmpResult
APFloat::compare(const APFloat &rhs) const
{
  cmpResult result;

  assert(semantics == rhs.semantics);

  switch (PackCategoriesIntoKey(category, rhs.category)) {
  default:
    llvm_unreachable(nullptr);

  case PackCategoriesIntoKey(fcNaN, fcZero):
  case PackCategoriesIntoKey(fcNaN, fcNormal):
  case PackCategoriesIntoKey(fcNaN, fcInfinity):
  case PackCategoriesIntoKey(fcNaN, fcNaN):
  case PackCategoriesIntoKey(fcZero, fcNaN):
  case PackCategoriesIntoKey(fcNormal, fcNaN):
  case PackCategoriesIntoKey(fcInfinity, fcNaN):
    return cmpUnordered;

  case PackCategoriesIntoKey(fcInfinity, fcNormal):
  case PackCategoriesIntoKey(fcInfinity, fcZero):
  case PackCategoriesIntoKey(fcNormal, fcZero):
    if (sign)
      return cmpLessThan;
    else
      return cmpGreaterThan;

  case PackCategoriesIntoKey(fcNormal, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcInfinity):
  case PackCategoriesIntoKey(fcZero, fcNormal):
    if (rhs.sign)
      return cmpGreaterThan;
    else
      return cmpLessThan;

  case PackCategoriesIntoKey(fcInfinity, fcInfinity):
    if (sign == rhs.sign)
      return cmpEqual;
    else if (sign)
      return cmpLessThan;
    else
      return cmpGreaterThan;

  case PackCategoriesIntoKey(fcZero, fcZero):
    return cmpEqual;

  case PackCategoriesIntoKey(fcNormal, fcNormal):
    break;
  }

  /* Two normal numbers.  Do they have the same sign?  */
  if (sign != rhs.sign) {
    if (sign)
      result = cmpLessThan;
    else
      result = cmpGreaterThan;
  } else {
    /* Compare absolute values; invert result if negative.  */
    result = compareAbsoluteValue(rhs);

    if (sign) {
      if (result == cmpLessThan)
        result = cmpGreaterThan;
      else if (result == cmpGreaterThan)
        result = cmpLessThan;
    }
  }

  return result;
}

/// APFloat::convert - convert a value of one floating point type to another.
/// The return value corresponds to the IEEE754 exceptions.  *losesInfo
/// records whether the transformation lost information, i.e. whether
/// converting the result back to the original type will produce the
/// original value (this is almost the same as return value==fsOK, but there
/// are edge cases where this is not so).

APFloat::opStatus
APFloat::convert(const fltSemantics &toSemantics,
                 roundingMode rounding_mode, bool *losesInfo)
{
  lostFraction lostFraction;
  unsigned int newPartCount, oldPartCount;
  opStatus fs;
  int shift;
  const fltSemantics &fromSemantics = *semantics;

  lostFraction = lfExactlyZero;
  newPartCount = partCountForBits(toSemantics.precision + 1);
  oldPartCount = partCount();
  shift = toSemantics.precision - fromSemantics.precision;

  bool X86SpecialNan = false;
  if (&fromSemantics == &APFloat::x87DoubleExtended &&
      &toSemantics != &APFloat::x87DoubleExtended && category == fcNaN &&
      (!(*significandParts() & 0x8000000000000000ULL) ||
       !(*significandParts() & 0x4000000000000000ULL))) {
    // x86 has some unusual NaNs which cannot be represented in any other
    // format; note them here.
    X86SpecialNan = true;
  }

  // If this is a truncation of a denormal number, and the target semantics
  // has larger exponent range than the source semantics (this can happen
  // when truncating from PowerPC double-double to double format), the
  // right shift could lose result mantissa bits.  Adjust exponent instead
  // of performing excessive shift.
  if (shift < 0 && isFiniteNonZero()) {
    int exponentChange = significandMSB() + 1 - fromSemantics.precision;
    if (exponent + exponentChange < toSemantics.minExponent)
      exponentChange = toSemantics.minExponent - exponent;
    if (exponentChange < shift)
      exponentChange = shift;
    if (exponentChange < 0) {
      shift -= exponentChange;
      exponent += exponentChange;
    }
  }

  // If this is a truncation, perform the shift before we narrow the storage.
  if (shift < 0 && (isFiniteNonZero() || category==fcNaN))
    lostFraction = shiftRight(significandParts(), oldPartCount, -shift);

  // Fix the storage so it can hold to new value.
  if (newPartCount > oldPartCount) {
    // The new type requires more storage; make it available.
    integerPart *newParts;
    newParts = new integerPart[newPartCount];
    APInt::tcSet(newParts, 0, newPartCount);
    if (isFiniteNonZero() || category==fcNaN)
      APInt::tcAssign(newParts, significandParts(), oldPartCount);
    freeSignificand();
    significand.parts = newParts;
  } else if (newPartCount == 1 && oldPartCount != 1) {
    // Switch to built-in storage for a single part.
    integerPart newPart = 0;
    if (isFiniteNonZero() || category==fcNaN)
      newPart = significandParts()[0];
    freeSignificand();
    significand.part = newPart;
  }

  // Now that we have the right storage, switch the semantics.
  semantics = &toSemantics;

  // If this is an extension, perform the shift now that the storage is
  // available.
  if (shift > 0 && (isFiniteNonZero() || category==fcNaN))
    APInt::tcShiftLeft(significandParts(), newPartCount, shift);

  if (isFiniteNonZero()) {
    fs = normalize(rounding_mode, lostFraction);
    *losesInfo = (fs != opOK);
  } else if (category == fcNaN) {
    *losesInfo = lostFraction != lfExactlyZero || X86SpecialNan;

    // For x87 extended precision, we want to make a NaN, not a special NaN if
    // the input wasn't special either.
    if (!X86SpecialNan && semantics == &APFloat::x87DoubleExtended)
      APInt::tcSetBit(significandParts(), semantics->precision - 1);

    // gcc forces the Quiet bit on, which means (float)(double)(float_sNan)
    // does not give you back the same bits.  This is dubious, and we
    // don't currently do it.  You're really supposed to get
    // an invalid operation signal at runtime, but nobody does that.
    fs = opOK;
  } else {
    *losesInfo = false;
    fs = opOK;
  }

  return fs;
}

/* Convert a floating point number to an integer according to the
   rounding mode.  If the rounded integer value is out of range this
   returns an invalid operation exception and the contents of the
   destination parts are unspecified.  If the rounded value is in
   range but the floating point number is not the exact integer, the C
   standard doesn't require an inexact exception to be raised.  IEEE
   854 does require it so we do that.

   Note that for conversions to integer type the C standard requires
   round-to-zero to always be used.  */
APFloat::opStatus
APFloat::convertToSignExtendedInteger(integerPart *parts, unsigned int width,
                                      bool isSigned,
                                      roundingMode rounding_mode,
                                      bool *isExact) const
{
  lostFraction lost_fraction;
  const integerPart *src;
  unsigned int dstPartsCount, truncatedBits;

  *isExact = false;

  /* Handle the three special cases first.  */
  if (category == fcInfinity || category == fcNaN)
    return opInvalidOp;

  dstPartsCount = partCountForBits(width);

  if (category == fcZero) {
    APInt::tcSet(parts, 0, dstPartsCount);
    // Negative zero can't be represented as an int.
    *isExact = !sign;
    return opOK;
  }

  src = significandParts();

  /* Step 1: place our absolute value, with any fraction truncated, in
     the destination.  */
  if (exponent < 0) {
    /* Our absolute value is less than one; truncate everything.  */
    APInt::tcSet(parts, 0, dstPartsCount);
    /* For exponent -1 the integer bit represents .5, look at that.
       For smaller exponents leftmost truncated bit is 0. */
    truncatedBits = semantics->precision -1U - exponent;
  } else {
    /* We want the most significant (exponent + 1) bits; the rest are
       truncated.  */
    unsigned int bits = exponent + 1U;

    /* Hopelessly large in magnitude?  */
    if (bits > width)
      return opInvalidOp;

    if (bits < semantics->precision) {
      /* We truncate (semantics->precision - bits) bits.  */
      truncatedBits = semantics->precision - bits;
      APInt::tcExtract(parts, dstPartsCount, src, bits, truncatedBits);
    } else {
      /* We want at least as many bits as are available.  */
      APInt::tcExtract(parts, dstPartsCount, src, semantics->precision, 0);
      APInt::tcShiftLeft(parts, dstPartsCount, bits - semantics->precision);
      truncatedBits = 0;
    }
  }

  /* Step 2: work out any lost fraction, and increment the absolute
     value if we would round away from zero.  */
  if (truncatedBits) {
    lost_fraction = lostFractionThroughTruncation(src, partCount(),
                                                  truncatedBits);
    if (lost_fraction != lfExactlyZero &&
        roundAwayFromZero(rounding_mode, lost_fraction, truncatedBits)) {
      if (APInt::tcIncrement(parts, dstPartsCount))
        return opInvalidOp;     /* Overflow.  */
    }
  } else {
    lost_fraction = lfExactlyZero;
  }

  /* Step 3: check if we fit in the destination.  */
  unsigned int omsb = APInt::tcMSB(parts, dstPartsCount) + 1;

  if (sign) {
    if (!isSigned) {
      /* Negative numbers cannot be represented as unsigned.  */
      if (omsb != 0)
        return opInvalidOp;
    } else {
      /* It takes omsb bits to represent the unsigned integer value.
         We lose a bit for the sign, but care is needed as the
         maximally negative integer is a special case.  */
      if (omsb == width && APInt::tcLSB(parts, dstPartsCount) + 1 != omsb)
        return opInvalidOp;

      /* This case can happen because of rounding.  */
      if (omsb > width)
        return opInvalidOp;
    }

    APInt::tcNegate (parts, dstPartsCount);
  } else {
    if (omsb >= width + !isSigned)
      return opInvalidOp;
  }

  if (lost_fraction == lfExactlyZero) {
    *isExact = true;
    return opOK;
  } else
    return opInexact;
}

/* Same as convertToSignExtendedInteger, except we provide
   deterministic values in case of an invalid operation exception,
   namely zero for NaNs and the minimal or maximal value respectively
   for underflow or overflow.
   The *isExact output tells whether the result is exact, in the sense
   that converting it back to the original floating point type produces
   the original value.  This is almost equivalent to result==opOK,
   except for negative zeroes.
*/
APFloat::opStatus
APFloat::convertToInteger(integerPart *parts, unsigned int width,
                          bool isSigned,
                          roundingMode rounding_mode, bool *isExact) const
{
  opStatus fs;

  fs = convertToSignExtendedInteger(parts, width, isSigned, rounding_mode,
                                    isExact);

  if (fs == opInvalidOp) {
    unsigned int bits, dstPartsCount;

    dstPartsCount = partCountForBits(width);

    if (category == fcNaN)
      bits = 0;
    else if (sign)
      bits = isSigned;
    else
      bits = width - isSigned;

    APInt::tcSetLeastSignificantBits(parts, dstPartsCount, bits);
    if (sign && isSigned)
      APInt::tcShiftLeft(parts, dstPartsCount, width - 1);
  }

  return fs;
}

/* Same as convertToInteger(integerPart*, ...), except the result is returned in
   an APSInt, whose initial bit-width and signed-ness are used to determine the
   precision of the conversion.
 */
APFloat::opStatus
APFloat::convertToInteger(APSInt &result,
                          roundingMode rounding_mode, bool *isExact) const
{
  unsigned bitWidth = result.getBitWidth();
  SmallVector<uint64_t, 4> parts(result.getNumWords());
  opStatus status = convertToInteger(
    parts.data(), bitWidth, result.isSigned(), rounding_mode, isExact);
  // Keeps the original signed-ness.
  result = APInt(bitWidth, parts);
  return status;
}

/* Convert an unsigned integer SRC to a floating point number,
   rounding according to ROUNDING_MODE.  The sign of the floating
   point number is not modified.  */
APFloat::opStatus
APFloat::convertFromUnsignedParts(const integerPart *src,
                                  unsigned int srcCount,
                                  roundingMode rounding_mode)
{
  unsigned int omsb, precision, dstCount;
  integerPart *dst;
  lostFraction lost_fraction;

  category = fcNormal;
  omsb = APInt::tcMSB(src, srcCount) + 1;
  dst = significandParts();
  dstCount = partCount();
  precision = semantics->precision;

  /* We want the most significant PRECISION bits of SRC.  There may not
     be that many; extract what we can.  */
  if (precision <= omsb) {
    exponent = omsb - 1;
    lost_fraction = lostFractionThroughTruncation(src, srcCount,
                                                  omsb - precision);
    APInt::tcExtract(dst, dstCount, src, precision, omsb - precision);
  } else {
    exponent = precision - 1;
    lost_fraction = lfExactlyZero;
    APInt::tcExtract(dst, dstCount, src, omsb, 0);
  }

  return normalize(rounding_mode, lost_fraction);
}

APFloat::opStatus
APFloat::convertFromAPInt(const APInt &Val,
                          bool isSigned,
                          roundingMode rounding_mode)
{
  unsigned int partCount = Val.getNumWords();
  APInt api = Val;

  sign = false;
  if (isSigned && api.isNegative()) {
    sign = true;
    api = -api;
  }

  return convertFromUnsignedParts(api.getRawData(), partCount, rounding_mode);
}

/* Convert a two's complement integer SRC to a floating point number,
   rounding according to ROUNDING_MODE.  ISSIGNED is true if the
   integer is signed, in which case it must be sign-extended.  */
APFloat::opStatus
APFloat::convertFromSignExtendedInteger(const integerPart *src,
                                        unsigned int srcCount,
                                        bool isSigned,
                                        roundingMode rounding_mode)
{
  opStatus status;

  if (isSigned &&
      APInt::tcExtractBit(src, srcCount * integerPartWidth - 1)) {
    integerPart *copy;

    /* If we're signed and negative negate a copy.  */
    sign = true;
    copy = new integerPart[srcCount];
    APInt::tcAssign(copy, src, srcCount);
    APInt::tcNegate(copy, srcCount);
    status = convertFromUnsignedParts(copy, srcCount, rounding_mode);
    delete [] copy;
  } else {
    sign = false;
    status = convertFromUnsignedParts(src, srcCount, rounding_mode);
  }

  return status;
}

/* FIXME: should this just take a const APInt reference?  */
APFloat::opStatus
APFloat::convertFromZeroExtendedInteger(const integerPart *parts,
                                        unsigned int width, bool isSigned,
                                        roundingMode rounding_mode)
{
  unsigned int partCount = partCountForBits(width);
  APInt api = APInt(width, makeArrayRef(parts, partCount));

  sign = false;
  if (isSigned && APInt::tcExtractBit(parts, width - 1)) {
    sign = true;
    api = -api;
  }

  return convertFromUnsignedParts(api.getRawData(), partCount, rounding_mode);
}

APFloat::opStatus
APFloat::convertFromHexadecimalString(StringRef s, roundingMode rounding_mode)
{
  lostFraction lost_fraction = lfExactlyZero;

  category = fcNormal;
  zeroSignificand();
  exponent = 0;

  integerPart *significand = significandParts();
  unsigned partsCount = partCount();
  unsigned bitPos = partsCount * integerPartWidth;
  bool computedTrailingFraction = false;

  // Skip leading zeroes and any (hexa)decimal point.
  StringRef::iterator begin = s.begin();
  StringRef::iterator end = s.end();
  StringRef::iterator dot;
  StringRef::iterator p = skipLeadingZeroesAndAnyDot(begin, end, &dot);
  StringRef::iterator firstSignificantDigit = p;

  while (p != end) {
    integerPart hex_value;

    if (*p == '.') {
      assert(dot == end && "String contains multiple dots");
      dot = p++;
      continue;
    }

    hex_value = hexDigitValue(*p);
    if (hex_value == -1U)
      break;

    p++;

    // Store the number while we have space.
    if (bitPos) {
      bitPos -= 4;
      hex_value <<= bitPos % integerPartWidth;
      significand[bitPos / integerPartWidth] |= hex_value;
    } else if (!computedTrailingFraction) {
      lost_fraction = trailingHexadecimalFraction(p, end, hex_value);
      computedTrailingFraction = true;
    }
  }

  /* Hex floats require an exponent but not a hexadecimal point.  */
  assert(p != end && "Hex strings require an exponent");
  assert((*p == 'p' || *p == 'P') && "Invalid character in significand");
  assert(p != begin && "Significand has no digits");
  assert((dot == end || p - begin != 1) && "Significand has no digits");

  /* Ignore the exponent if we are zero.  */
  if (p != firstSignificantDigit) {
    int expAdjustment;

    /* Implicit hexadecimal point?  */
    if (dot == end)
      dot = p;

    /* Calculate the exponent adjustment implicit in the number of
       significant digits.  */
    expAdjustment = static_cast<int>(dot - firstSignificantDigit);
    if (expAdjustment < 0)
      expAdjustment++;
    expAdjustment = expAdjustment * 4 - 1;

    /* Adjust for writing the significand starting at the most
       significant nibble.  */
    expAdjustment += semantics->precision;
    expAdjustment -= partsCount * integerPartWidth;

    /* Adjust for the given exponent.  */
    exponent = totalExponent(p + 1, end, expAdjustment);
  }

  return normalize(rounding_mode, lost_fraction);
}

APFloat::opStatus
APFloat::roundSignificandWithExponent(const integerPart *decSigParts,
                                      unsigned sigPartCount, int exp,
                                      roundingMode rounding_mode)
{
  unsigned int parts, pow5PartCount;
  fltSemantics calcSemantics = { 32767, -32767, 0, 0 };
  integerPart pow5Parts[maxPowerOfFiveParts];
  bool isNearest;

  isNearest = (rounding_mode == rmNearestTiesToEven ||
               rounding_mode == rmNearestTiesToAway);

  parts = partCountForBits(semantics->precision + 11);

  /* Calculate pow(5, abs(exp)).  */
  pow5PartCount = powerOf5(pow5Parts, exp >= 0 ? exp: -exp);

  for (;; parts *= 2) {
    opStatus sigStatus, powStatus;
    unsigned int excessPrecision, truncatedBits;

    calcSemantics.precision = parts * integerPartWidth - 1;
    excessPrecision = calcSemantics.precision - semantics->precision;
    truncatedBits = excessPrecision;

    APFloat decSig = APFloat::getZero(calcSemantics, sign);
    APFloat pow5(calcSemantics);

    sigStatus = decSig.convertFromUnsignedParts(decSigParts, sigPartCount,
                                                rmNearestTiesToEven);
    powStatus = pow5.convertFromUnsignedParts(pow5Parts, pow5PartCount,
                                              rmNearestTiesToEven);
    /* Add exp, as 10^n = 5^n * 2^n.  */
    decSig.exponent += exp;

    lostFraction calcLostFraction;
    integerPart HUerr, HUdistance;
    unsigned int powHUerr;

    if (exp >= 0) {
      /* multiplySignificand leaves the precision-th bit set to 1.  */
      calcLostFraction = decSig.multiplySignificand(pow5, nullptr);
      powHUerr = powStatus != opOK;
    } else {
      calcLostFraction = decSig.divideSignificand(pow5);
      /* Denormal numbers have less precision.  */
      if (decSig.exponent < semantics->minExponent) {
        excessPrecision += (semantics->minExponent - decSig.exponent);
        truncatedBits = excessPrecision;
        if (excessPrecision > calcSemantics.precision)
          excessPrecision = calcSemantics.precision;
      }
      /* Extra half-ulp lost in reciprocal of exponent.  */
      powHUerr = (powStatus == opOK && calcLostFraction == lfExactlyZero) ? 0:2;
    }

    /* Both multiplySignificand and divideSignificand return the
       result with the integer bit set.  */
    assert(APInt::tcExtractBit
           (decSig.significandParts(), calcSemantics.precision - 1) == 1);

    HUerr = HUerrBound(calcLostFraction != lfExactlyZero, sigStatus != opOK,
                       powHUerr);
    HUdistance = 2 * ulpsFromBoundary(decSig.significandParts(),
                                      excessPrecision, isNearest);

    /* Are we guaranteed to round correctly if we truncate?  */
    if (HUdistance >= HUerr) {
      APInt::tcExtract(significandParts(), partCount(), decSig.significandParts(),
                       calcSemantics.precision - excessPrecision,
                       excessPrecision);
      /* Take the exponent of decSig.  If we tcExtract-ed less bits
         above we must adjust our exponent to compensate for the
         implicit right shift.  */
      exponent = (decSig.exponent + semantics->precision
                  - (calcSemantics.precision - excessPrecision));
      calcLostFraction = lostFractionThroughTruncation(decSig.significandParts(),
                                                       decSig.partCount(),
                                                       truncatedBits);
      return normalize(rounding_mode, calcLostFraction);
    }
  }
}

APFloat::opStatus
APFloat::convertFromDecimalString(StringRef str, roundingMode rounding_mode)
{
  decimalInfo D;
  opStatus fs;

  /* Scan the text.  */
  StringRef::iterator p = str.begin();
  interpretDecimal(p, str.end(), &D);

  /* Handle the quick cases.  First the case of no significant digits,
     i.e. zero, and then exponents that are obviously too large or too
     small.  Writing L for log 10 / log 2, a number d.ddddd*10^exp
     definitely overflows if

           (exp - 1) * L >= maxExponent

     and definitely underflows to zero where

           (exp + 1) * L <= minExponent - precision

     With integer arithmetic the tightest bounds for L are

           93/28 < L < 196/59            [ numerator <= 256 ]
           42039/12655 < L < 28738/8651  [ numerator <= 65536 ]
  */

  // Test if we have a zero number allowing for strings with no null terminators
  // and zero decimals with non-zero exponents.
  // 
  // We computed firstSigDigit by ignoring all zeros and dots. Thus if
  // D->firstSigDigit equals str.end(), every digit must be a zero and there can
  // be at most one dot. On the other hand, if we have a zero with a non-zero
  // exponent, then we know that D.firstSigDigit will be non-numeric.
  if (D.firstSigDigit == str.end() || decDigitValue(*D.firstSigDigit) >= 10U) {
    category = fcZero;
    fs = opOK;

  /* Check whether the normalized exponent is high enough to overflow
     max during the log-rebasing in the max-exponent check below. */
  } else if (D.normalizedExponent - 1 > INT_MAX / 42039) {
    fs = handleOverflow(rounding_mode);

  /* If it wasn't, then it also wasn't high enough to overflow max
     during the log-rebasing in the min-exponent check.  Check that it
     won't overflow min in either check, then perform the min-exponent
     check. */
  } else if (D.normalizedExponent - 1 < INT_MIN / 42039 ||
             (D.normalizedExponent + 1) * 28738 <=
               8651 * (semantics->minExponent - (int) semantics->precision)) {
    /* Underflow to zero and round.  */
    category = fcNormal;
    zeroSignificand();
    fs = normalize(rounding_mode, lfLessThanHalf);

  /* We can finally safely perform the max-exponent check. */
  } else if ((D.normalizedExponent - 1) * 42039
             >= 12655 * semantics->maxExponent) {
    /* Overflow and round.  */
    fs = handleOverflow(rounding_mode);
  } else {
    integerPart *decSignificand;
    unsigned int partCount;

    /* A tight upper bound on number of bits required to hold an
       N-digit decimal integer is N * 196 / 59.  Allocate enough space
       to hold the full significand, and an extra part required by
       tcMultiplyPart.  */
    partCount = static_cast<unsigned int>(D.lastSigDigit - D.firstSigDigit) + 1;
    partCount = partCountForBits(1 + 196 * partCount / 59);
    decSignificand = new integerPart[partCount + 1];
    partCount = 0;

    /* Convert to binary efficiently - we do almost all multiplication
       in an integerPart.  When this would overflow do we do a single
       bignum multiplication, and then revert again to multiplication
       in an integerPart.  */
    do {
      integerPart decValue, val, multiplier;

      val = 0;
      multiplier = 1;

      do {
        if (*p == '.') {
          p++;
          if (p == str.end()) {
            break;
          }
        }
        decValue = decDigitValue(*p++);
        assert(decValue < 10U && "Invalid character in significand");
        multiplier *= 10;
        val = val * 10 + decValue;
        /* The maximum number that can be multiplied by ten with any
           digit added without overflowing an integerPart.  */
      } while (p <= D.lastSigDigit && multiplier <= (~ (integerPart) 0 - 9) / 10);

      /* Multiply out the current part.  */
      APInt::tcMultiplyPart(decSignificand, decSignificand, multiplier, val,
                            partCount, partCount + 1, false);

      /* If we used another part (likely but not guaranteed), increase
         the count.  */
      if (decSignificand[partCount])
        partCount++;
    } while (p <= D.lastSigDigit);

    category = fcNormal;
    fs = roundSignificandWithExponent(decSignificand, partCount,
                                      D.exponent, rounding_mode);

    delete [] decSignificand;
  }

  return fs;
}

bool
APFloat::convertFromStringSpecials(StringRef str) {
  if (str.equals("inf") || str.equals("INFINITY") || str.equals("1.#INF")) { // HLSL Change - support 1.#INF
    makeInf(false);
    return true;
  }

  if (str.equals("-inf") || str.equals("-INFINITY") || str.equals("-1.#INF")) { // HLSL Change - support 1.#INF
    makeInf(true);
    return true;
  }

  if (str.equals("nan") || str.equals("NaN")) {
    makeNaN(false, false);
    return true;
  }

  if (str.equals("-nan") || str.equals("-NaN")) {
    makeNaN(false, true);
    return true;
  }

  return false;
}

APFloat::opStatus
APFloat::convertFromString(StringRef str, roundingMode rounding_mode)
{
  assert(!str.empty() && "Invalid string length");

  // Handle special cases.
  if (convertFromStringSpecials(str))
    return opOK;

  /* Handle a leading minus sign.  */
  StringRef::iterator p = str.begin();
  size_t slen = str.size();
  sign = *p == '-' ? 1 : 0;
  if (*p == '-' || *p == '+') {
    p++;
    slen--;
    assert(slen && "String has no digits");
  }

  if (slen >= 2 && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
    assert(slen - 2 && "Invalid string");
    return convertFromHexadecimalString(StringRef(p + 2, slen - 2),
                                        rounding_mode);
  }

  return convertFromDecimalString(StringRef(p, slen), rounding_mode);
}

#if 0 // HLSL Change
/* Write out a hexadecimal representation of the floating point value
   to DST, which must be of sufficient size, in the C99 form
   [-]0xh.hhhhp[+-]d.  Return the number of characters written,
   excluding the terminating NUL.

   If UPPERCASE, the output is in upper case, otherwise in lower case.

   HEXDIGITS digits appear altogether, rounding the value if
   necessary.  If HEXDIGITS is 0, the minimal precision to display the
   number precisely is used instead.  If nothing would appear after
   the decimal point it is suppressed.

   The decimal exponent is always printed and has at least one digit.
   Zero values display an exponent of zero.  Infinities and NaNs
   appear as "infinity" or "nan" respectively.

   The above rules are as specified by C99.  There is ambiguity about
   what the leading hexadecimal digit should be.  This implementation
   uses whatever is necessary so that the exponent is displayed as
   stored.  This implies the exponent will fall within the IEEE format
   range, and the leading hexadecimal digit will be 0 (for denormals),
   1 (normal numbers) or 2 (normal numbers rounded-away-from-zero with
   any other digits zero).
*/
unsigned int
APFloat::convertToHexString(char *dst, unsigned int hexDigits,
                            bool upperCase, roundingMode rounding_mode) const
{
  char *p;

  p = dst;
  if (sign)
    *dst++ = '-';

  switch (category) {
  case fcInfinity:
    memcpy (dst, upperCase ? infinityU: infinityL, sizeof infinityU - 1);
    dst += sizeof infinityL - 1;
    break;

  case fcNaN:
    memcpy (dst, upperCase ? NaNU: NaNL, sizeof NaNU - 1);
    dst += sizeof NaNU - 1;
    break;

  case fcZero:
    *dst++ = '0';
    *dst++ = upperCase ? 'X': 'x';
    *dst++ = '0';
    if (hexDigits > 1) {
      *dst++ = '.';
      memset (dst, '0', hexDigits - 1);
      dst += hexDigits - 1;
    }
    *dst++ = upperCase ? 'P': 'p';
    *dst++ = '0';
    break;

  case fcNormal:
    dst = convertNormalToHexString (dst, hexDigits, upperCase, rounding_mode);
    break;
  }

  *dst = 0;

  return static_cast<unsigned int>(dst - p);
}

/* Does the hard work of outputting the correctly rounded hexadecimal
   form of a normal floating point number with the specified number of
   hexadecimal digits.  If HEXDIGITS is zero the minimum number of
   digits necessary to print the value precisely is output.  */
char *
APFloat::convertNormalToHexString(char *dst, unsigned int hexDigits,
                                  bool upperCase,
                                  roundingMode rounding_mode) const
{
  unsigned int count, valueBits, shift, partsCount, outputDigits;
  const char *hexDigitChars;
  const integerPart *significand;
  char *p;
  bool roundUp;

  *dst++ = '0';
  *dst++ = upperCase ? 'X': 'x';

  roundUp = false;
  hexDigitChars = upperCase ? hexDigitsUpper: hexDigitsLower;

  significand = significandParts();
  partsCount = partCount();

  /* +3 because the first digit only uses the single integer bit, so
     we have 3 virtual zero most-significant-bits.  */
  valueBits = semantics->precision + 3;
  shift = integerPartWidth - valueBits % integerPartWidth;

  /* The natural number of digits required ignoring trailing
     insignificant zeroes.  */
  outputDigits = (valueBits - significandLSB () + 3) / 4;

  /* hexDigits of zero means use the required number for the
     precision.  Otherwise, see if we are truncating.  If we are,
     find out if we need to round away from zero.  */
  if (hexDigits) {
    if (hexDigits < outputDigits) {
      /* We are dropping non-zero bits, so need to check how to round.
         "bits" is the number of dropped bits.  */
      unsigned int bits;
      lostFraction fraction;

      bits = valueBits - hexDigits * 4;
      fraction = lostFractionThroughTruncation (significand, partsCount, bits);
      roundUp = roundAwayFromZero(rounding_mode, fraction, bits);
    }
    outputDigits = hexDigits;
  }

  /* Write the digits consecutively, and start writing in the location
     of the hexadecimal point.  We move the most significant digit
     left and add the hexadecimal point later.  */
  p = ++dst;

  count = (valueBits + integerPartWidth - 1) / integerPartWidth;

  while (outputDigits && count) {
    integerPart part;

    /* Put the most significant integerPartWidth bits in "part".  */
    if (--count == partsCount)
      part = 0;  /* An imaginary higher zero part.  */
    else
      part = significand[count] << shift;

    if (count && shift)
      part |= significand[count - 1] >> (integerPartWidth - shift);

    /* Convert as much of "part" to hexdigits as we can.  */
    unsigned int curDigits = integerPartWidth / 4;

    if (curDigits > outputDigits)
      curDigits = outputDigits;
    dst += partAsHex (dst, part, curDigits, hexDigitChars);
    outputDigits -= curDigits;
  }

  if (roundUp) {
    char *q = dst;

    /* Note that hexDigitChars has a trailing '0'.  */
    do {
      q--;
      *q = hexDigitChars[hexDigitValue (*q) + 1];
    } while (*q == '0');
    assert(q >= p);
  } else {
    /* Add trailing zeroes.  */
    memset (dst, '0', outputDigits);
    dst += outputDigits;
  }

  /* Move the most significant digit to before the point, and if there
     is something after the decimal point add it.  This must come
     after rounding above.  */
  p[-1] = p[0];
  if (dst -1 == p)
    dst--;
  else
    p[0] = '.';

  /* Finally output the exponent.  */
  *dst++ = upperCase ? 'P': 'p';

  return writeSignedDecimal (dst, exponent);
}

#endif

hash_code llvm::hash_value(const APFloat &Arg) {
  if (!Arg.isFiniteNonZero())
    return hash_combine((uint8_t)Arg.category,
                        // NaN has no sign, fix it at zero.
                        Arg.isNaN() ? (uint8_t)0 : (uint8_t)Arg.sign,
                        Arg.semantics->precision);

  // Normal floats need their exponent and significand hashed.
  return hash_combine((uint8_t)Arg.category, (uint8_t)Arg.sign,
                      Arg.semantics->precision, Arg.exponent,
                      hash_combine_range(
                        Arg.significandParts(),
                        Arg.significandParts() + Arg.partCount()));
}

// Conversion from APFloat to/from host float/double.  It may eventually be
// possible to eliminate these and have everybody deal with APFloats, but that
// will take a while.  This approach will not easily extend to long double.
// Current implementation requires integerPartWidth==64, which is correct at
// the moment but could be made more general.

// Denormals have exponent minExponent in APFloat, but minExponent-1 in
// the actual IEEE respresentations.  We compensate for that here.

APInt
APFloat::convertF80LongDoubleAPFloatToAPInt() const
{
  assert(semantics == (const llvm::fltSemantics*)&x87DoubleExtended);
  assert(partCount()==2);

  uint64_t myexponent, mysignificand;

  if (isFiniteNonZero()) {
    myexponent = exponent+16383; //bias
    mysignificand = significandParts()[0];
    if (myexponent==1 && !(mysignificand & 0x8000000000000000ULL))
      myexponent = 0;   // denormal
  } else if (category==fcZero) {
    myexponent = 0;
    mysignificand = 0;
  } else if (category==fcInfinity) {
    myexponent = 0x7fff;
    mysignificand = 0x8000000000000000ULL;
  } else {
    assert(category == fcNaN && "Unknown category");
    myexponent = 0x7fff;
    mysignificand = significandParts()[0];
  }

  uint64_t words[2];
  words[0] = mysignificand;
  words[1] =  ((uint64_t)(sign & 1) << 15) |
              (myexponent & 0x7fffLL);
  return APInt(80, words);
}

APInt
APFloat::convertPPCDoubleDoubleAPFloatToAPInt() const
{
  assert(semantics == (const llvm::fltSemantics*)&PPCDoubleDouble);
  assert(partCount()==2);

  uint64_t words[2];
  opStatus fs;
  bool losesInfo;

  // Convert number to double.  To avoid spurious underflows, we re-
  // normalize against the "double" minExponent first, and only *then*
  // truncate the mantissa.  The result of that second conversion
  // may be inexact, but should never underflow.
  // Declare fltSemantics before APFloat that uses it (and
  // saves pointer to it) to ensure correct destruction order.
  fltSemantics extendedSemantics = *semantics;
  extendedSemantics.minExponent = IEEEdouble.minExponent;
  APFloat extended(*this);
  fs = extended.convert(extendedSemantics, rmNearestTiesToEven, &losesInfo);
  assert(fs == opOK && !losesInfo);
  (void)fs;

  APFloat u(extended);
  fs = u.convert(IEEEdouble, rmNearestTiesToEven, &losesInfo);
  assert(fs == opOK || fs == opInexact);
  (void)fs;
  words[0] = *u.convertDoubleAPFloatToAPInt().getRawData();

  // If conversion was exact or resulted in a special case, we're done;
  // just set the second double to zero.  Otherwise, re-convert back to
  // the extended format and compute the difference.  This now should
  // convert exactly to double.
  if (u.isFiniteNonZero() && losesInfo) {
    fs = u.convert(extendedSemantics, rmNearestTiesToEven, &losesInfo);
    assert(fs == opOK && !losesInfo);
    (void)fs;

    APFloat v(extended);
    v.subtract(u, rmNearestTiesToEven);
    fs = v.convert(IEEEdouble, rmNearestTiesToEven, &losesInfo);
    assert(fs == opOK && !losesInfo);
    (void)fs;
    words[1] = *v.convertDoubleAPFloatToAPInt().getRawData();
  } else {
    words[1] = 0;
  }

  return APInt(128, words);
}

APInt
APFloat::convertQuadrupleAPFloatToAPInt() const
{
  assert(semantics == (const llvm::fltSemantics*)&IEEEquad);
  assert(partCount()==2);

  uint64_t myexponent, mysignificand, mysignificand2;

  if (isFiniteNonZero()) {
    myexponent = exponent+16383; //bias
    mysignificand = significandParts()[0];
    mysignificand2 = significandParts()[1];
    if (myexponent==1 && !(mysignificand2 & 0x1000000000000LL))
      myexponent = 0;   // denormal
  } else if (category==fcZero) {
    myexponent = 0;
    mysignificand = mysignificand2 = 0;
  } else if (category==fcInfinity) {
    myexponent = 0x7fff;
    mysignificand = mysignificand2 = 0;
  } else {
    assert(category == fcNaN && "Unknown category!");
    myexponent = 0x7fff;
    mysignificand = significandParts()[0];
    mysignificand2 = significandParts()[1];
  }

  uint64_t words[2];
  words[0] = mysignificand;
  words[1] = ((uint64_t)(sign & 1) << 63) |
             ((myexponent & 0x7fff) << 48) |
             (mysignificand2 & 0xffffffffffffLL);

  return APInt(128, words);
}

APInt
APFloat::convertDoubleAPFloatToAPInt() const
{
  assert(semantics == (const llvm::fltSemantics*)&IEEEdouble);
  assert(partCount()==1);

  uint64_t myexponent, mysignificand;

  if (isFiniteNonZero()) {
    myexponent = exponent+1023; //bias
    mysignificand = *significandParts();
    if (myexponent==1 && !(mysignificand & 0x10000000000000LL))
      myexponent = 0;   // denormal
  } else if (category==fcZero) {
    myexponent = 0;
    mysignificand = 0;
  } else if (category==fcInfinity) {
    myexponent = 0x7ff;
    mysignificand = 0;
  } else {
    assert(category == fcNaN && "Unknown category!");
    myexponent = 0x7ff;
    mysignificand = *significandParts();
  }

  return APInt(64, ((((uint64_t)(sign & 1) << 63) |
                     ((myexponent & 0x7ff) <<  52) |
                     (mysignificand & 0xfffffffffffffLL))));
}

APInt
APFloat::convertFloatAPFloatToAPInt() const
{
  assert(semantics == (const llvm::fltSemantics*)&IEEEsingle);
  assert(partCount()==1);

  uint32_t myexponent, mysignificand;

  if (isFiniteNonZero()) {
    myexponent = exponent+127; //bias
    mysignificand = (uint32_t)*significandParts();
    if (myexponent == 1 && !(mysignificand & 0x800000))
      myexponent = 0;   // denormal
  } else if (category==fcZero) {
    myexponent = 0;
    mysignificand = 0;
  } else if (category==fcInfinity) {
    myexponent = 0xff;
    mysignificand = 0;
  } else {
    assert(category == fcNaN && "Unknown category!");
    myexponent = 0xff;
    mysignificand = (uint32_t)*significandParts();
  }

  return APInt(32, (((sign&1) << 31) | ((myexponent&0xff) << 23) |
                    (mysignificand & 0x7fffff)));
}

APInt
APFloat::convertHalfAPFloatToAPInt() const
{
  assert(semantics == (const llvm::fltSemantics*)&IEEEhalf);
  assert(partCount()==1);

  uint32_t myexponent, mysignificand;

  if (isFiniteNonZero()) {
    myexponent = exponent+15; //bias
    mysignificand = (uint32_t)*significandParts();
    if (myexponent == 1 && !(mysignificand & 0x400))
      myexponent = 0;   // denormal
  } else if (category==fcZero) {
    myexponent = 0;
    mysignificand = 0;
  } else if (category==fcInfinity) {
    myexponent = 0x1f;
    mysignificand = 0;
  } else {
    assert(category == fcNaN && "Unknown category!");
    myexponent = 0x1f;
    mysignificand = (uint32_t)*significandParts();
  }

  return APInt(16, (((sign&1) << 15) | ((myexponent&0x1f) << 10) |
                    (mysignificand & 0x3ff)));
}

// This function creates an APInt that is just a bit map of the floating
// point constant as it would appear in memory.  It is not a conversion,
// and treating the result as a normal integer is unlikely to be useful.

APInt
APFloat::bitcastToAPInt() const
{
  if (semantics == (const llvm::fltSemantics*)&IEEEhalf)
    return convertHalfAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics*)&IEEEsingle)
    return convertFloatAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics*)&IEEEdouble)
    return convertDoubleAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics*)&IEEEquad)
    return convertQuadrupleAPFloatToAPInt();

  if (semantics == (const llvm::fltSemantics*)&PPCDoubleDouble)
    return convertPPCDoubleDoubleAPFloatToAPInt();

  assert(semantics == (const llvm::fltSemantics*)&x87DoubleExtended &&
         "unknown format!");
  return convertF80LongDoubleAPFloatToAPInt();
}

float
APFloat::convertToFloat() const
{
  assert(semantics == (const llvm::fltSemantics*)&IEEEsingle &&
         "Float semantics are not IEEEsingle");
  APInt api = bitcastToAPInt();
  return api.bitsToFloat();
}

double
APFloat::convertToDouble() const
{
  assert(semantics == (const llvm::fltSemantics*)&IEEEdouble &&
         "Float semantics are not IEEEdouble");
  APInt api = bitcastToAPInt();
  return api.bitsToDouble();
}

/// Integer bit is explicit in this format.  Intel hardware (387 and later)
/// does not support these bit patterns:
///  exponent = all 1's, integer bit 0, significand 0 ("pseudoinfinity")
///  exponent = all 1's, integer bit 0, significand nonzero ("pseudoNaN")
///  exponent = 0, integer bit 1 ("pseudodenormal")
///  exponent!=0 nor all 1's, integer bit 0 ("unnormal")
/// At the moment, the first two are treated as NaNs, the second two as Normal.
void
APFloat::initFromF80LongDoubleAPInt(const APInt &api)
{
  assert(api.getBitWidth()==80);
  uint64_t i1 = api.getRawData()[0];
  uint64_t i2 = api.getRawData()[1];
  uint64_t myexponent = (i2 & 0x7fff);
  uint64_t mysignificand = i1;

  initialize(&APFloat::x87DoubleExtended);
  assert(partCount()==2);

  sign = static_cast<unsigned int>(i2>>15);
  if (myexponent==0 && mysignificand==0) {
    // exponent, significand meaningless
    category = fcZero;
  } else if (myexponent==0x7fff && mysignificand==0x8000000000000000ULL) {
    // exponent, significand meaningless
    category = fcInfinity;
  } else if (myexponent==0x7fff && mysignificand!=0x8000000000000000ULL) {
    // exponent meaningless
    category = fcNaN;
    significandParts()[0] = mysignificand;
    significandParts()[1] = 0;
  } else {
    category = fcNormal;
    exponent = myexponent - 16383;
    significandParts()[0] = mysignificand;
    significandParts()[1] = 0;
    if (myexponent==0)          // denormal
      exponent = -16382;
  }
}

void
APFloat::initFromPPCDoubleDoubleAPInt(const APInt &api)
{
  assert(api.getBitWidth()==128);
  uint64_t i1 = api.getRawData()[0];
  uint64_t i2 = api.getRawData()[1];
  opStatus fs;
  bool losesInfo;

  // Get the first double and convert to our format.
  initFromDoubleAPInt(APInt(64, i1));
  fs = convert(PPCDoubleDouble, rmNearestTiesToEven, &losesInfo);
  assert(fs == opOK && !losesInfo);
  (void)fs;

  // Unless we have a special case, add in second double.
  if (isFiniteNonZero()) {
    APFloat v(IEEEdouble, APInt(64, i2));
    fs = v.convert(PPCDoubleDouble, rmNearestTiesToEven, &losesInfo);
    assert(fs == opOK && !losesInfo);
    (void)fs;

    add(v, rmNearestTiesToEven);
  }
}

void
APFloat::initFromQuadrupleAPInt(const APInt &api)
{
  assert(api.getBitWidth()==128);
  uint64_t i1 = api.getRawData()[0];
  uint64_t i2 = api.getRawData()[1];
  uint64_t myexponent = (i2 >> 48) & 0x7fff;
  uint64_t mysignificand  = i1;
  uint64_t mysignificand2 = i2 & 0xffffffffffffLL;

  initialize(&APFloat::IEEEquad);
  assert(partCount()==2);

  sign = static_cast<unsigned int>(i2>>63);
  if (myexponent==0 &&
      (mysignificand==0 && mysignificand2==0)) {
    // exponent, significand meaningless
    category = fcZero;
  } else if (myexponent==0x7fff &&
             (mysignificand==0 && mysignificand2==0)) {
    // exponent, significand meaningless
    category = fcInfinity;
  } else if (myexponent==0x7fff &&
             (mysignificand!=0 || mysignificand2 !=0)) {
    // exponent meaningless
    category = fcNaN;
    significandParts()[0] = mysignificand;
    significandParts()[1] = mysignificand2;
  } else {
    category = fcNormal;
    exponent = myexponent - 16383;
    significandParts()[0] = mysignificand;
    significandParts()[1] = mysignificand2;
    if (myexponent==0)          // denormal
      exponent = -16382;
    else
      significandParts()[1] |= 0x1000000000000LL;  // integer bit
  }
}

void
APFloat::initFromDoubleAPInt(const APInt &api)
{
  assert(api.getBitWidth()==64);
  uint64_t i = *api.getRawData();
  uint64_t myexponent = (i >> 52) & 0x7ff;
  uint64_t mysignificand = i & 0xfffffffffffffLL;

  initialize(&APFloat::IEEEdouble);
  assert(partCount()==1);

  sign = static_cast<unsigned int>(i>>63);
  if (myexponent==0 && mysignificand==0) {
    // exponent, significand meaningless
    category = fcZero;
  } else if (myexponent==0x7ff && mysignificand==0) {
    // exponent, significand meaningless
    category = fcInfinity;
  } else if (myexponent==0x7ff && mysignificand!=0) {
    // exponent meaningless
    category = fcNaN;
    *significandParts() = mysignificand;
  } else {
    category = fcNormal;
    exponent = myexponent - 1023;
    *significandParts() = mysignificand;
    if (myexponent==0)          // denormal
      exponent = -1022;
    else
      *significandParts() |= 0x10000000000000LL;  // integer bit
  }
}

void
APFloat::initFromFloatAPInt(const APInt & api)
{
  assert(api.getBitWidth()==32);
  uint32_t i = (uint32_t)*api.getRawData();
  uint32_t myexponent = (i >> 23) & 0xff;
  uint32_t mysignificand = i & 0x7fffff;

  initialize(&APFloat::IEEEsingle);
  assert(partCount()==1);

  sign = i >> 31;
  if (myexponent==0 && mysignificand==0) {
    // exponent, significand meaningless
    category = fcZero;
  } else if (myexponent==0xff && mysignificand==0) {
    // exponent, significand meaningless
    category = fcInfinity;
  } else if (myexponent==0xff && mysignificand!=0) {
    // sign, exponent, significand meaningless
    category = fcNaN;
    *significandParts() = mysignificand;
  } else {
    category = fcNormal;
    exponent = myexponent - 127;  //bias
    *significandParts() = mysignificand;
    if (myexponent==0)    // denormal
      exponent = -126;
    else
      *significandParts() |= 0x800000; // integer bit
  }
}

void
APFloat::initFromHalfAPInt(const APInt & api)
{
  assert(api.getBitWidth()==16);
  uint32_t i = (uint32_t)*api.getRawData();
  uint32_t myexponent = (i >> 10) & 0x1f;
  uint32_t mysignificand = i & 0x3ff;

  initialize(&APFloat::IEEEhalf);
  assert(partCount()==1);

  sign = i >> 15;
  if (myexponent==0 && mysignificand==0) {
    // exponent, significand meaningless
    category = fcZero;
  } else if (myexponent==0x1f && mysignificand==0) {
    // exponent, significand meaningless
    category = fcInfinity;
  } else if (myexponent==0x1f && mysignificand!=0) {
    // sign, exponent, significand meaningless
    category = fcNaN;
    *significandParts() = mysignificand;
  } else {
    category = fcNormal;
    exponent = myexponent - 15;  //bias
    *significandParts() = mysignificand;
    if (myexponent==0)    // denormal
      exponent = -14;
    else
      *significandParts() |= 0x400; // integer bit
  }
}

/// Treat api as containing the bits of a floating point number.  Currently
/// we infer the floating point type from the size of the APInt.  The
/// isIEEE argument distinguishes between PPC128 and IEEE128 (not meaningful
/// when the size is anything else).
void
APFloat::initFromAPInt(const fltSemantics* Sem, const APInt& api)
{
  if (Sem == &IEEEhalf)
    return initFromHalfAPInt(api);
  if (Sem == &IEEEsingle)
    return initFromFloatAPInt(api);
  if (Sem == &IEEEdouble)
    return initFromDoubleAPInt(api);
  if (Sem == &x87DoubleExtended)
    return initFromF80LongDoubleAPInt(api);
  if (Sem == &IEEEquad)
    return initFromQuadrupleAPInt(api);
  if (Sem == &PPCDoubleDouble)
    return initFromPPCDoubleDoubleAPInt(api);

  llvm_unreachable(nullptr);
}

APFloat
APFloat::getAllOnesValue(unsigned BitWidth, bool isIEEE)
{
  switch (BitWidth) {
  case 16:
    return APFloat(IEEEhalf, APInt::getAllOnesValue(BitWidth));
  case 32:
    return APFloat(IEEEsingle, APInt::getAllOnesValue(BitWidth));
  case 64:
    return APFloat(IEEEdouble, APInt::getAllOnesValue(BitWidth));
  case 80:
    return APFloat(x87DoubleExtended, APInt::getAllOnesValue(BitWidth));
  case 128:
    if (isIEEE)
      return APFloat(IEEEquad, APInt::getAllOnesValue(BitWidth));
    return APFloat(PPCDoubleDouble, APInt::getAllOnesValue(BitWidth));
  default:
    llvm_unreachable("Unknown floating bit width");
  }
}

unsigned APFloat::getSizeInBits(const fltSemantics &Sem) {
  return Sem.sizeInBits;
}

/// Make this number the largest magnitude normal number in the given
/// semantics.
void APFloat::makeLargest(bool Negative) {
  // We want (in interchange format):
  //   sign = {Negative}
  //   exponent = 1..10
  //   significand = 1..1
  category = fcNormal;
  sign = Negative;
  exponent = semantics->maxExponent;

  // Use memset to set all but the highest integerPart to all ones.
  integerPart *significand = significandParts();
  unsigned PartCount = partCount();
  memset(significand, 0xFF, sizeof(integerPart)*(PartCount - 1));

  // Set the high integerPart especially setting all unused top bits for
  // internal consistency.
  const unsigned NumUnusedHighBits =
    PartCount*integerPartWidth - semantics->precision;
  significand[PartCount - 1] = (NumUnusedHighBits < integerPartWidth)
                                   ? (~integerPart(0) >> NumUnusedHighBits)
                                   : 0;
}

/// Make this number the smallest magnitude denormal number in the given
/// semantics.
void APFloat::makeSmallest(bool Negative) {
  // We want (in interchange format):
  //   sign = {Negative}
  //   exponent = 0..0
  //   significand = 0..01
  category = fcNormal;
  sign = Negative;
  exponent = semantics->minExponent;
  APInt::tcSet(significandParts(), 1, partCount());
}


APFloat APFloat::getLargest(const fltSemantics &Sem, bool Negative) {
  // We want (in interchange format):
  //   sign = {Negative}
  //   exponent = 1..10
  //   significand = 1..1
  APFloat Val(Sem, uninitialized);
  Val.makeLargest(Negative);
  return Val;
}

APFloat APFloat::getSmallest(const fltSemantics &Sem, bool Negative) {
  // We want (in interchange format):
  //   sign = {Negative}
  //   exponent = 0..0
  //   significand = 0..01
  APFloat Val(Sem, uninitialized);
  Val.makeSmallest(Negative);
  return Val;
}

APFloat APFloat::getSmallestNormalized(const fltSemantics &Sem, bool Negative) {
  APFloat Val(Sem, uninitialized);

  // We want (in interchange format):
  //   sign = {Negative}
  //   exponent = 0..0
  //   significand = 10..0

  Val.category = fcNormal;
  Val.zeroSignificand();
  Val.sign = Negative;
  Val.exponent = Sem.minExponent;
  Val.significandParts()[partCountForBits(Sem.precision)-1] |=
    (((integerPart) 1) << ((Sem.precision - 1) % integerPartWidth));

  return Val;
}

APFloat::APFloat(const fltSemantics &Sem, const APInt &API) {
  initFromAPInt(&Sem, API);
}

APFloat::APFloat(float f) {
  initFromAPInt(&IEEEsingle, APInt::floatToBits(f));
}

APFloat::APFloat(double d) {
  initFromAPInt(&IEEEdouble, APInt::doubleToBits(d));
}

namespace {
  void append(SmallVectorImpl<char> &Buffer, StringRef Str) {
    Buffer.append(Str.begin(), Str.end());
  }

  /// Removes data from the given significand until it is no more
  /// precise than is required for the desired precision.
  void AdjustToPrecision(APInt &significand,
                         int &exp, unsigned FormatPrecision) {
    unsigned bits = significand.getActiveBits();

    // 196/59 is a very slight overestimate of lg_2(10).
    unsigned bitsRequired = (FormatPrecision * 196 + 58) / 59;

    if (bits <= bitsRequired) return;

    unsigned tensRemovable = (bits - bitsRequired) * 59 / 196;
    if (!tensRemovable) return;

    exp += tensRemovable;

    APInt divisor(significand.getBitWidth(), 1);
    APInt powten(significand.getBitWidth(), 10);
    while (true) {
      if (tensRemovable & 1)
        divisor *= powten;
      tensRemovable >>= 1;
      if (!tensRemovable) break;
      powten *= powten;
    }

    significand = significand.udiv(divisor);

    // Truncate the significand down to its active bit count.
    significand = significand.trunc(significand.getActiveBits());
  }


  void AdjustToPrecision(SmallVectorImpl<char> &buffer,
                         int &exp, unsigned FormatPrecision) {
    unsigned N = buffer.size();
    if (N <= FormatPrecision) return;

    // The most significant figures are the last ones in the buffer.
    unsigned FirstSignificant = N - FormatPrecision;

    // Round.
    // FIXME: this probably shouldn't use 'round half up'.

    // Rounding down is just a truncation, except we also want to drop
    // trailing zeros from the new result.
    if (buffer[FirstSignificant - 1] < '5') {
      while (FirstSignificant < N && buffer[FirstSignificant] == '0')
        FirstSignificant++;

      exp += FirstSignificant;
      buffer.erase(&buffer[0], &buffer[FirstSignificant]);
      return;
    }

    // Rounding up requires a decimal add-with-carry.  If we continue
    // the carry, the newly-introduced zeros will just be truncated.
    for (unsigned I = FirstSignificant; I != N; ++I) {
      if (buffer[I] == '9') {
        FirstSignificant++;
      } else {
        buffer[I]++;
        break;
      }
    }

    // If we carried through, we have exactly one digit of precision.
    if (FirstSignificant == N) {
      exp += FirstSignificant;
      buffer.clear();
      buffer.push_back('1');
      return;
    }

    exp += FirstSignificant;
    buffer.erase(&buffer[0], &buffer[FirstSignificant]);
  }
}

void APFloat::toString(SmallVectorImpl<char> &Str,
                       unsigned FormatPrecision,
                       unsigned FormatMaxPadding) const {
  switch (category) {
  case fcInfinity:
    if (isNegative())
      return append(Str, "-Inf");
    else
      return append(Str, "+Inf");

  case fcNaN: return append(Str, "NaN");

  case fcZero:
    if (isNegative())
      Str.push_back('-');

    if (!FormatMaxPadding)
      append(Str, "0.0E+0");
    else
      Str.push_back('0');
    return;

  case fcNormal:
    break;
  }

  if (isNegative())
    Str.push_back('-');

  // Decompose the number into an APInt and an exponent.
  int exp = exponent - ((int) semantics->precision - 1);
  APInt significand(semantics->precision,
                    makeArrayRef(significandParts(),
                                 partCountForBits(semantics->precision)));

  // Set FormatPrecision if zero.  We want to do this before we
  // truncate trailing zeros, as those are part of the precision.
  if (!FormatPrecision) {
    // We use enough digits so the number can be round-tripped back to an
    // APFloat. The formula comes from "How to Print Floating-Point Numbers
    // Accurately" by Steele and White.
    // FIXME: Using a formula based purely on the precision is conservative;
    // we can print fewer digits depending on the actual value being printed.

    // FormatPrecision = 2 + floor(significandBits / lg_2(10))
    FormatPrecision = 2 + semantics->precision * 59 / 196;
  }

  // Ignore trailing binary zeros.
  int trailingZeros = significand.countTrailingZeros();
  exp += trailingZeros;
  significand = significand.lshr(trailingZeros);

  // Change the exponent from 2^e to 10^e.
  if (exp == 0) {
    // Nothing to do.
  } else if (exp > 0) {
    // Just shift left.
    significand = significand.zext(semantics->precision + exp);
    significand <<= exp;
    exp = 0;
  } else { /* exp < 0 */
    int texp = -exp;

    // We transform this using the identity:
    //   (N)(2^-e) == (N)(5^e)(10^-e)
    // This means we have to multiply N (the significand) by 5^e.
    // To avoid overflow, we have to operate on numbers large
    // enough to store N * 5^e:
    //   log2(N * 5^e) == log2(N) + e * log2(5)
    //                 <= semantics->precision + e * 137 / 59
    //   (log_2(5) ~ 2.321928 < 2.322034 ~ 137/59)

    unsigned precision = semantics->precision + (137 * texp + 136) / 59;

    // Multiply significand by 5^e.
    //   N * 5^0101 == N * 5^(1*1) * 5^(0*2) * 5^(1*4) * 5^(0*8)
    significand = significand.zext(precision);
    APInt five_to_the_i(precision, 5);
    while (true) {
      if (texp & 1) significand *= five_to_the_i;

      texp >>= 1;
      if (!texp) break;
      five_to_the_i *= five_to_the_i;
    }
  }

  AdjustToPrecision(significand, exp, FormatPrecision);

  SmallVector<char, 256> buffer;

  // Fill the buffer.
  unsigned precision = significand.getBitWidth();
  APInt ten(precision, 10);
  APInt digit(precision, 0);

  bool inTrail = true;
  while (significand != 0) {
    // digit <- significand % 10
    // significand <- significand / 10
    APInt::udivrem(significand, ten, significand, digit);

    unsigned d = digit.getZExtValue();

    // Drop trailing zeros.
    if (inTrail && !d) exp++;
    else {
      buffer.push_back((char) ('0' + d));
      inTrail = false;
    }
  }

  assert(!buffer.empty() && "no characters in buffer!");

  // Drop down to FormatPrecision.
  // TODO: don't do more precise calculations above than are required.
  AdjustToPrecision(buffer, exp, FormatPrecision);

  unsigned NDigits = buffer.size();

  // Check whether we should use scientific notation.
  bool FormatScientific;
  if (!FormatMaxPadding)
    FormatScientific = true;
  else {
    if (exp >= 0) {
      // 765e3 --> 765000
      //              ^^^
      // But we shouldn't make the number look more precise than it is.
      FormatScientific = ((unsigned) exp > FormatMaxPadding ||
                          NDigits + (unsigned) exp > FormatPrecision);
    } else {
      // Power of the most significant digit.
      int MSD = exp + (int) (NDigits - 1);
      if (MSD >= 0) {
        // 765e-2 == 7.65
        FormatScientific = false;
      } else {
        // 765e-5 == 0.00765
        //           ^ ^^
        FormatScientific = ((unsigned) -MSD) > FormatMaxPadding;
      }
    }
  }

  // Scientific formatting is pretty straightforward.
  if (FormatScientific) {
    exp += (NDigits - 1);

    Str.push_back(buffer[NDigits-1]);
    Str.push_back('.');
    if (NDigits == 1)
      Str.push_back('0');
    else
      for (unsigned I = 1; I != NDigits; ++I)
        Str.push_back(buffer[NDigits-1-I]);
    Str.push_back('E');

    Str.push_back(exp >= 0 ? '+' : '-');
    if (exp < 0) exp = -exp;
    SmallVector<char, 6> expbuf;
    do {
      expbuf.push_back((char) ('0' + (exp % 10)));
      exp /= 10;
    } while (exp);
    for (unsigned I = 0, E = expbuf.size(); I != E; ++I)
      Str.push_back(expbuf[E-1-I]);
    return;
  }

  // Non-scientific, positive exponents.
  if (exp >= 0) {
    for (unsigned I = 0; I != NDigits; ++I)
      Str.push_back(buffer[NDigits-1-I]);
    for (unsigned I = 0; I != (unsigned) exp; ++I)
      Str.push_back('0');
    return;
  }

  // Non-scientific, negative exponents.

  // The number of digits to the left of the decimal point.
  int NWholeDigits = exp + (int) NDigits;

  unsigned I = 0;
  if (NWholeDigits > 0) {
    for (; I != (unsigned) NWholeDigits; ++I)
      Str.push_back(buffer[NDigits-I-1]);
    Str.push_back('.');
  } else {
    unsigned NZeros = 1 + (unsigned) -NWholeDigits;

    Str.push_back('0');
    Str.push_back('.');
    for (unsigned Z = 1; Z != NZeros; ++Z)
      Str.push_back('0');
  }

  for (; I != NDigits; ++I)
    Str.push_back(buffer[NDigits-I-1]);
}

bool APFloat::getExactInverse(APFloat *inv) const {
  // Special floats and denormals have no exact inverse.
  if (!isFiniteNonZero())
    return false;

  // Check that the number is a power of two by making sure that only the
  // integer bit is set in the significand.
  if (significandLSB() != semantics->precision - 1)
    return false;

  // Get the inverse.
  APFloat reciprocal(*semantics, 1ULL);
  if (reciprocal.divide(*this, rmNearestTiesToEven) != opOK)
    return false;

  // Avoid multiplication with a denormal, it is not safe on all platforms and
  // may be slower than a normal division.
  if (reciprocal.isDenormal())
    return false;

  assert(reciprocal.isFiniteNonZero() &&
         reciprocal.significandLSB() == reciprocal.semantics->precision - 1);

  if (inv)
    *inv = reciprocal;

  return true;
}

bool APFloat::isSignaling() const {
  if (!isNaN())
    return false;

  // IEEE-754R 2008 6.2.1: A signaling NaN bit string should be encoded with the
  // first bit of the trailing significand being 0.
  return !APInt::tcExtractBit(significandParts(), semantics->precision - 2);
}

/// IEEE-754R 2008 5.3.1: nextUp/nextDown.
///
/// *NOTE* since nextDown(x) = -nextUp(-x), we only implement nextUp with
/// appropriate sign switching before/after the computation.
APFloat::opStatus APFloat::next(bool nextDown) {
  // If we are performing nextDown, swap sign so we have -x.
  if (nextDown)
    changeSign();

  // Compute nextUp(x)
  opStatus result = opOK;

  // Handle each float category separately.
  switch (category) {
  case fcInfinity:
    // nextUp(+inf) = +inf
    if (!isNegative())
      break;
    // nextUp(-inf) = -getLargest()
    makeLargest(true);
    break;
  case fcNaN:
    // IEEE-754R 2008 6.2 Par 2: nextUp(sNaN) = qNaN. Set Invalid flag.
    // IEEE-754R 2008 6.2: nextUp(qNaN) = qNaN. Must be identity so we do not
    //                     change the payload.
    if (isSignaling()) {
      result = opInvalidOp;
      // For consistency, propagate the sign of the sNaN to the qNaN.
      makeNaN(false, isNegative(), nullptr);
    }
    break;
  case fcZero:
    // nextUp(pm 0) = +getSmallest()
    makeSmallest(false);
    break;
  case fcNormal:
    // nextUp(-getSmallest()) = -0
    if (isSmallest() && isNegative()) {
      APInt::tcSet(significandParts(), 0, partCount());
      category = fcZero;
      exponent = 0;
      break;
    }

    // nextUp(getLargest()) == INFINITY
    if (isLargest() && !isNegative()) {
      APInt::tcSet(significandParts(), 0, partCount());
      category = fcInfinity;
      exponent = semantics->maxExponent + 1;
      break;
    }

    // nextUp(normal) == normal + inc.
    if (isNegative()) {
      // If we are negative, we need to decrement the significand.

      // We only cross a binade boundary that requires adjusting the exponent
      // if:
      //   1. exponent != semantics->minExponent. This implies we are not in the
      //   smallest binade or are dealing with denormals.
      //   2. Our significand excluding the integral bit is all zeros.
      bool WillCrossBinadeBoundary =
        exponent != semantics->minExponent && isSignificandAllZeros();

      // Decrement the significand.
      //
      // We always do this since:
      //   1. If we are dealing with a non-binade decrement, by definition we
      //   just decrement the significand.
      //   2. If we are dealing with a normal -> normal binade decrement, since
      //   we have an explicit integral bit the fact that all bits but the
      //   integral bit are zero implies that subtracting one will yield a
      //   significand with 0 integral bit and 1 in all other spots. Thus we
      //   must just adjust the exponent and set the integral bit to 1.
      //   3. If we are dealing with a normal -> denormal binade decrement,
      //   since we set the integral bit to 0 when we represent denormals, we
      //   just decrement the significand.
      integerPart *Parts = significandParts();
      APInt::tcDecrement(Parts, partCount());

      if (WillCrossBinadeBoundary) {
        // Our result is a normal number. Do the following:
        // 1. Set the integral bit to 1.
        // 2. Decrement the exponent.
        APInt::tcSetBit(Parts, semantics->precision - 1);
        exponent--;
      }
    } else {
      // If we are positive, we need to increment the significand.

      // We only cross a binade boundary that requires adjusting the exponent if
      // the input is not a denormal and all of said input's significand bits
      // are set. If all of said conditions are true: clear the significand, set
      // the integral bit to 1, and increment the exponent. If we have a
      // denormal always increment since moving denormals and the numbers in the
      // smallest normal binade have the same exponent in our representation.
      bool WillCrossBinadeBoundary = !isDenormal() && isSignificandAllOnes();

      if (WillCrossBinadeBoundary) {
        integerPart *Parts = significandParts();
        APInt::tcSet(Parts, 0, partCount());
        APInt::tcSetBit(Parts, semantics->precision - 1);
        assert(exponent != semantics->maxExponent &&
               "We can not increment an exponent beyond the maxExponent allowed"
               " by the given floating point semantics.");
        exponent++;
      } else {
        incrementSignificand();
      }
    }
    break;
  }

  // If we are performing nextDown, swap sign so we have -nextUp(-x)
  if (nextDown)
    changeSign();

  return result;
}

void
APFloat::makeInf(bool Negative) {
  category = fcInfinity;
  sign = Negative;
  exponent = semantics->maxExponent + 1;
  APInt::tcSet(significandParts(), 0, partCount());
}

void
APFloat::makeZero(bool Negative) {
  category = fcZero;
  sign = Negative;
  exponent = semantics->minExponent-1;
  APInt::tcSet(significandParts(), 0, partCount());  
}

APFloat llvm::scalbn(APFloat X, int Exp) {
  if (X.isInfinity() || X.isZero() || X.isNaN())
    return X;

  auto MaxExp = X.getSemantics().maxExponent;
  auto MinExp = X.getSemantics().minExponent;
  if (Exp > (MaxExp - X.exponent))
    // Overflow saturates to infinity.
    return APFloat::getInf(X.getSemantics(), X.isNegative());
  if (Exp < (MinExp - X.exponent))
    // Underflow saturates to zero.
    return APFloat::getZero(X.getSemantics(), X.isNegative());

  X.exponent += Exp;
  return X;
}
