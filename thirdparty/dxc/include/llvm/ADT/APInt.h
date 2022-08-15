//===-- llvm/ADT/APInt.h - For Arbitrary Precision Integer -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a class to represent arbitrary precision
/// integral constant values and operations on them.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_APINT_H
#define LLVM_ADT_APINT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <climits>
#include <cstring>
#include <string>

namespace llvm {
class FoldingSetNodeID;
class StringRef;
class hash_code;
class raw_ostream;

template <typename T> class SmallVectorImpl;

// An unsigned host type used as a single part of a multi-part
// bignum.
typedef uint64_t integerPart;

const unsigned int host_char_bit = 8;
const unsigned int integerPartWidth =
    host_char_bit * static_cast<unsigned int>(sizeof(integerPart));

//===----------------------------------------------------------------------===//
//                              APInt Class
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/// \brief Class for arbitrary precision integers.
///
/// APInt is a functional replacement for common case unsigned integer type like
/// "unsigned", "unsigned long" or "uint64_t", but also allows non-byte-width
/// integer sizes and large integer value types such as 3-bits, 15-bits, or more
/// than 64-bits of precision. APInt provides a variety of arithmetic operators
/// and methods to manipulate integer values of any bit-width. It supports both
/// the typical integer arithmetic and comparison operations as well as bitwise
/// manipulation.
///
/// The class has several invariants worth noting:
///   * All bit, byte, and word positions are zero-based.
///   * Once the bit width is set, it doesn't change except by the Truncate,
///     SignExtend, or ZeroExtend operations.
///   * All binary operators must be on APInt instances of the same bit width.
///     Attempting to use these operators on instances with different bit
///     widths will yield an assertion.
///   * The value is stored canonically as an unsigned value. For operations
///     where it makes a difference, there are both signed and unsigned variants
///     of the operation. For example, sdiv and udiv. However, because the bit
///     widths must be the same, operations such as Mul and Add produce the same
///     results regardless of whether the values are interpreted as signed or
///     not.
///   * In general, the class tries to follow the style of computation that LLVM
///     uses in its IR. This simplifies its use for LLVM.
///
class APInt {
  unsigned BitWidth; ///< The number of bits in this APInt.

  /// This union is used to store the integer value. When the
  /// integer bit-width <= 64, it uses VAL, otherwise it uses pVal.
  union {
    uint64_t VAL;   ///< Used to store the <= 64 bits integer value.
    uint64_t *pVal; ///< Used to store the >64 bits integer value.
  };

  /// This enum is used to hold the constants we needed for APInt.
  enum {
    /// Bits in a word
    APINT_BITS_PER_WORD =
        static_cast<unsigned int>(sizeof(uint64_t)) * CHAR_BIT,
    /// Byte size of a word
    APINT_WORD_SIZE = static_cast<unsigned int>(sizeof(uint64_t))
  };

  friend struct DenseMapAPIntKeyInfo;

  /// \brief Fast internal constructor
  ///
  /// This constructor is used only internally for speed of construction of
  /// temporaries. It is unsafe for general use so it is not public.
  APInt(uint64_t *val, unsigned bits) : BitWidth(bits), pVal(val) {}

  /// \brief Determine if this APInt just has one word to store value.
  ///
  /// \returns true if the number of bits <= 64, false otherwise.
  bool isSingleWord() const { return BitWidth <= APINT_BITS_PER_WORD; }

  /// \brief Determine which word a bit is in.
  ///
  /// \returns the word position for the specified bit position.
  static unsigned whichWord(unsigned bitPosition) {
    return bitPosition / APINT_BITS_PER_WORD;
  }

  /// \brief Determine which bit in a word a bit is in.
  ///
  /// \returns the bit position in a word for the specified bit position
  /// in the APInt.
  static unsigned whichBit(unsigned bitPosition) {
    return bitPosition % APINT_BITS_PER_WORD;
  }

  /// \brief Get a single bit mask.
  ///
  /// \returns a uint64_t with only bit at "whichBit(bitPosition)" set
  /// This method generates and returns a uint64_t (word) mask for a single
  /// bit at a specific bit position. This is used to mask the bit in the
  /// corresponding word.
  static uint64_t maskBit(unsigned bitPosition) {
    return 1ULL << whichBit(bitPosition);
  }

  /// \brief Clear unused high order bits
  ///
  /// This method is used internally to clear the top "N" bits in the high order
  /// word that are not used by the APInt. This is needed after the most
  /// significant word is assigned a value to ensure that those bits are
  /// zero'd out.
  APInt &clearUnusedBits() {
    // Compute how many bits are used in the final word
    unsigned wordBits = BitWidth % APINT_BITS_PER_WORD;
    if (wordBits == 0)
      // If all bits are used, we want to leave the value alone. This also
      // avoids the undefined behavior of >> when the shift is the same size as
      // the word size (64).
      return *this;

    // Mask out the high bits.
    uint64_t mask = ~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - wordBits);
    if (isSingleWord())
      VAL &= mask;
    else
      pVal[getNumWords() - 1] &= mask;
    return *this;
  }

  /// \brief Get the word corresponding to a bit position
  /// \returns the corresponding word for the specified bit position.
  uint64_t getWord(unsigned bitPosition) const {
    return isSingleWord() ? VAL : pVal[whichWord(bitPosition)];
  }

  /// \brief Convert a char array into an APInt
  ///
  /// \param radix 2, 8, 10, 16, or 36
  /// Converts a string into a number.  The string must be non-empty
  /// and well-formed as a number of the given base. The bit-width
  /// must be sufficient to hold the result.
  ///
  /// This is used by the constructors that take string arguments.
  ///
  /// StringRef::getAsInteger is superficially similar but (1) does
  /// not assume that the string is well-formed and (2) grows the
  /// result to hold the input.
  void fromString(unsigned numBits, StringRef str, uint8_t radix);

  /// \brief An internal division function for dividing APInts.
  ///
  /// This is used by the toString method to divide by the radix. It simply
  /// provides a more convenient form of divide for internal use since KnuthDiv
  /// has specific constraints on its inputs. If those constraints are not met
  /// then it provides a simpler form of divide.
  static void divide(const APInt LHS, unsigned lhsWords, const APInt &RHS,
                     unsigned rhsWords, APInt *Quotient, APInt *Remainder);

  /// out-of-line slow case for inline constructor
  void initSlowCase(unsigned numBits, uint64_t val, bool isSigned);

  /// shared code between two array constructors
  void initFromArray(ArrayRef<uint64_t> array);

  /// out-of-line slow case for inline copy constructor
  void initSlowCase(const APInt &that);

  /// out-of-line slow case for shl
  APInt shlSlowCase(unsigned shiftAmt) const;

  /// out-of-line slow case for operator&
  APInt AndSlowCase(const APInt &RHS) const;

  /// out-of-line slow case for operator|
  APInt OrSlowCase(const APInt &RHS) const;

  /// out-of-line slow case for operator^
  APInt XorSlowCase(const APInt &RHS) const;

  /// out-of-line slow case for operator=
  APInt &AssignSlowCase(const APInt &RHS);

  /// out-of-line slow case for operator==
  bool EqualSlowCase(const APInt &RHS) const;

  /// out-of-line slow case for operator==
  bool EqualSlowCase(uint64_t Val) const;

  /// out-of-line slow case for countLeadingZeros
  unsigned countLeadingZerosSlowCase() const;

  /// out-of-line slow case for countTrailingOnes
  unsigned countTrailingOnesSlowCase() const;

  /// out-of-line slow case for countPopulation
  unsigned countPopulationSlowCase() const;

public:
  /// \name Constructors
  /// @{

  /// \brief Create a new APInt of numBits width, initialized as val.
  ///
  /// If isSigned is true then val is treated as if it were a signed value
  /// (i.e. as an int64_t) and the appropriate sign extension to the bit width
  /// will be done. Otherwise, no sign extension occurs (high order bits beyond
  /// the range of val are zero filled).
  ///
  /// \param numBits the bit width of the constructed APInt
  /// \param val the initial value of the APInt
  /// \param isSigned how to treat signedness of val
  APInt(unsigned numBits, uint64_t val, bool isSigned = false)
      : BitWidth(numBits), VAL(0) {
    assert(BitWidth && "bitwidth too small");
    if (isSingleWord())
      VAL = val;
    else
      initSlowCase(numBits, val, isSigned);
    clearUnusedBits();
  }

  /// \brief Construct an APInt of numBits width, initialized as bigVal[].
  ///
  /// Note that bigVal.size() can be smaller or larger than the corresponding
  /// bit width but any extraneous bits will be dropped.
  ///
  /// \param numBits the bit width of the constructed APInt
  /// \param bigVal a sequence of words to form the initial value of the APInt
  APInt(unsigned numBits, ArrayRef<uint64_t> bigVal);

  /// Equivalent to APInt(numBits, ArrayRef<uint64_t>(bigVal, numWords)), but
  /// deprecated because this constructor is prone to ambiguity with the
  /// APInt(unsigned, uint64_t, bool) constructor.
  ///
  /// If this overload is ever deleted, care should be taken to prevent calls
  /// from being incorrectly captured by the APInt(unsigned, uint64_t, bool)
  /// constructor.
  APInt(unsigned numBits, unsigned numWords, const uint64_t bigVal[]);

  /// \brief Construct an APInt from a string representation.
  ///
  /// This constructor interprets the string \p str in the given radix. The
  /// interpretation stops when the first character that is not suitable for the
  /// radix is encountered, or the end of the string. Acceptable radix values
  /// are 2, 8, 10, 16, and 36. It is an error for the value implied by the
  /// string to require more bits than numBits.
  ///
  /// \param numBits the bit width of the constructed APInt
  /// \param str the string to be interpreted
  /// \param radix the radix to use for the conversion
  APInt(unsigned numBits, StringRef str, uint8_t radix);

  /// Simply makes *this a copy of that.
  /// @brief Copy Constructor.
  APInt(const APInt &that) : BitWidth(that.BitWidth), VAL(0) {
    if (isSingleWord())
      VAL = that.VAL;
    else
      initSlowCase(that);
  }

  /// \brief Move Constructor.
  APInt(APInt &&that) : BitWidth(that.BitWidth), VAL(that.VAL) {
    that.BitWidth = 0;
  }

  /// \brief Destructor.
  ~APInt() {
    if (needsCleanup())
      delete[] pVal;
  }

  /// \brief Default constructor that creates an uninitialized APInt.
  ///
  /// This is useful for object deserialization (pair this with the static
  ///  method Read).
  explicit APInt() : BitWidth(1) {}

  /// \brief Returns whether this instance allocated memory.
  bool needsCleanup() const { return !isSingleWord(); }

  /// Used to insert APInt objects, or objects that contain APInt objects, into
  ///  FoldingSets.
  void Profile(FoldingSetNodeID &id) const;

  /// @}
  /// \name Value Tests
  /// @{

  /// \brief Determine sign of this APInt.
  ///
  /// This tests the high bit of this APInt to determine if it is set.
  ///
  /// \returns true if this APInt is negative, false otherwise
  bool isNegative() const { return (*this)[BitWidth - 1]; }

  /// \brief Determine if this APInt Value is non-negative (>= 0)
  ///
  /// This tests the high bit of the APInt to determine if it is unset.
  bool isNonNegative() const { return !isNegative(); }

  /// \brief Determine if this APInt Value is positive.
  ///
  /// This tests if the value of this APInt is positive (> 0). Note
  /// that 0 is not a positive value.
  ///
  /// \returns true if this APInt is positive.
  bool isStrictlyPositive() const { return isNonNegative() && !!*this; }

  /// \brief Determine if all bits are set
  ///
  /// This checks to see if the value has all bits of the APInt are set or not.
  bool isAllOnesValue() const {
    if (isSingleWord())
      return VAL == ~integerPart(0) >> (APINT_BITS_PER_WORD - BitWidth);
    return countPopulationSlowCase() == BitWidth;
  }

  /// \brief Determine if this is the largest unsigned value.
  ///
  /// This checks to see if the value of this APInt is the maximum unsigned
  /// value for the APInt's bit width.
  bool isMaxValue() const { return isAllOnesValue(); }

  /// \brief Determine if this is the largest signed value.
  ///
  /// This checks to see if the value of this APInt is the maximum signed
  /// value for the APInt's bit width.
  bool isMaxSignedValue() const {
    return !isNegative() && countPopulation() == BitWidth - 1;
  }

  /// \brief Determine if this is the smallest unsigned value.
  ///
  /// This checks to see if the value of this APInt is the minimum unsigned
  /// value for the APInt's bit width.
  bool isMinValue() const { return !*this; }

  /// \brief Determine if this is the smallest signed value.
  ///
  /// This checks to see if the value of this APInt is the minimum signed
  /// value for the APInt's bit width.
  bool isMinSignedValue() const {
    return isNegative() && isPowerOf2();
  }

  /// \brief Check if this APInt has an N-bits unsigned integer value.
  bool isIntN(unsigned N) const {
    assert(N && "N == 0 ???");
    return getActiveBits() <= N;
  }

  /// \brief Check if this APInt has an N-bits signed integer value.
  bool isSignedIntN(unsigned N) const {
    assert(N && "N == 0 ???");
    return getMinSignedBits() <= N;
  }

  /// \brief Check if this APInt's value is a power of two greater than zero.
  ///
  /// \returns true if the argument APInt value is a power of two > 0.
  bool isPowerOf2() const {
    if (isSingleWord())
      return isPowerOf2_64(VAL);
    return countPopulationSlowCase() == 1;
  }

  /// \brief Check if the APInt's value is returned by getSignBit.
  ///
  /// \returns true if this is the value returned by getSignBit.
  bool isSignBit() const { return isMinSignedValue(); }

  /// \brief Convert APInt to a boolean value.
  ///
  /// This converts the APInt to a boolean value as a test against zero.
  bool getBoolValue() const { return !!*this; }

  /// If this value is smaller than the specified limit, return it, otherwise
  /// return the limit value.  This causes the value to saturate to the limit.
  uint64_t getLimitedValue(uint64_t Limit = ~0ULL) const {
    return (getActiveBits() > 64 || getZExtValue() > Limit) ? Limit
                                                            : getZExtValue();
  }

  /// \brief Check if the APInt consists of a repeated bit pattern.
  ///
  /// e.g. 0x01010101 satisfies isSplat(8).
  /// \param SplatSizeInBits The size of the pattern in bits. Must divide bit
  /// width without remainder.
  bool isSplat(unsigned SplatSizeInBits) const;

  /// @}
  /// \name Value Generators
  /// @{

  /// \brief Gets maximum unsigned value of APInt for specific bit width.
  static APInt getMaxValue(unsigned numBits) {
    return getAllOnesValue(numBits);
  }

  /// \brief Gets maximum signed value of APInt for a specific bit width.
  static APInt getSignedMaxValue(unsigned numBits) {
    APInt API = getAllOnesValue(numBits);
    API.clearBit(numBits - 1);
    return API;
  }

  /// \brief Gets minimum unsigned value of APInt for a specific bit width.
  static APInt getMinValue(unsigned numBits) { return APInt(numBits, 0); }

  /// \brief Gets minimum signed value of APInt for a specific bit width.
  static APInt getSignedMinValue(unsigned numBits) {
    APInt API(numBits, 0);
    API.setBit(numBits - 1);
    return API;
  }

  /// \brief Get the SignBit for a specific bit width.
  ///
  /// This is just a wrapper function of getSignedMinValue(), and it helps code
  /// readability when we want to get a SignBit.
  static APInt getSignBit(unsigned BitWidth) {
    return getSignedMinValue(BitWidth);
  }

  /// \brief Get the all-ones value.
  ///
  /// \returns the all-ones value for an APInt of the specified bit-width.
  static APInt getAllOnesValue(unsigned numBits) {
    return APInt(numBits, UINT64_MAX, true);
  }

  /// \brief Get the '0' value.
  ///
  /// \returns the '0' value for an APInt of the specified bit-width.
  static APInt getNullValue(unsigned numBits) { return APInt(numBits, 0); }

  /// \brief Compute an APInt containing numBits highbits from this APInt.
  ///
  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the low bits and right shift to the least significant bit.
  ///
  /// \returns the high "numBits" bits of this APInt.
  APInt getHiBits(unsigned numBits) const;

  /// \brief Compute an APInt containing numBits lowbits from this APInt.
  ///
  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the high bits.
  ///
  /// \returns the low "numBits" bits of this APInt.
  APInt getLoBits(unsigned numBits) const;

  /// \brief Return an APInt with exactly one bit set in the result.
  static APInt getOneBitSet(unsigned numBits, unsigned BitNo) {
    APInt Res(numBits, 0);
    Res.setBit(BitNo);
    return Res;
  }

  /// \brief Get a value with a block of bits set.
  ///
  /// Constructs an APInt value that has a contiguous range of bits set. The
  /// bits from loBit (inclusive) to hiBit (exclusive) will be set. All other
  /// bits will be zero. For example, with parameters(32, 0, 16) you would get
  /// 0x0000FFFF. If hiBit is less than loBit then the set bits "wrap". For
  /// example, with parameters (32, 28, 4), you would get 0xF000000F.
  ///
  /// \param numBits the intended bit width of the result
  /// \param loBit the index of the lowest bit set.
  /// \param hiBit the index of the highest bit set.
  ///
  /// \returns An APInt value with the requested bits set.
  static APInt getBitsSet(unsigned numBits, unsigned loBit, unsigned hiBit) {
    assert(hiBit <= numBits && "hiBit out of range");
    assert(loBit < numBits && "loBit out of range");
    if (hiBit < loBit)
      return getLowBitsSet(numBits, hiBit) |
             getHighBitsSet(numBits, numBits - loBit);
    return getLowBitsSet(numBits, hiBit - loBit).shl(loBit);
  }

  /// \brief Get a value with high bits set
  ///
  /// Constructs an APInt value that has the top hiBitsSet bits set.
  ///
  /// \param numBits the bitwidth of the result
  /// \param hiBitsSet the number of high-order bits set in the result.
  static APInt getHighBitsSet(unsigned numBits, unsigned hiBitsSet) {
    assert(hiBitsSet <= numBits && "Too many bits to set!");
    // Handle a degenerate case, to avoid shifting by word size
    if (hiBitsSet == 0)
      return APInt(numBits, 0);
    unsigned shiftAmt = numBits - hiBitsSet;
    // For small values, return quickly
    if (numBits <= APINT_BITS_PER_WORD)
      return APInt(numBits, ~0ULL << shiftAmt);
    return getAllOnesValue(numBits).shl(shiftAmt);
  }

  /// \brief Get a value with low bits set
  ///
  /// Constructs an APInt value that has the bottom loBitsSet bits set.
  ///
  /// \param numBits the bitwidth of the result
  /// \param loBitsSet the number of low-order bits set in the result.
  static APInt getLowBitsSet(unsigned numBits, unsigned loBitsSet) {
    assert(loBitsSet <= numBits && "Too many bits to set!");
    // Handle a degenerate case, to avoid shifting by word size
    if (loBitsSet == 0)
      return APInt(numBits, 0);
    if (loBitsSet == APINT_BITS_PER_WORD)
      return APInt(numBits, UINT64_MAX);
    // For small values, return quickly.
    if (loBitsSet <= APINT_BITS_PER_WORD)
      return APInt(numBits, UINT64_MAX >> (APINT_BITS_PER_WORD - loBitsSet));
    return getAllOnesValue(numBits).lshr(numBits - loBitsSet);
  }

  /// \brief Return a value containing V broadcasted over NewLen bits.
  static APInt getSplat(unsigned NewLen, const APInt &V) {
    assert(NewLen >= V.getBitWidth() && "Can't splat to smaller bit width!");

    APInt Val = V.zextOrSelf(NewLen);
    for (unsigned I = V.getBitWidth(); I < NewLen; I <<= 1)
      Val |= Val << I;

    return Val;
  }

  /// \brief Determine if two APInts have the same value, after zero-extending
  /// one of them (if needed!) to ensure that the bit-widths match.
  static bool isSameValue(const APInt &I1, const APInt &I2) {
    if (I1.getBitWidth() == I2.getBitWidth())
      return I1 == I2;

    if (I1.getBitWidth() > I2.getBitWidth())
      return I1 == I2.zext(I1.getBitWidth());

    return I1.zext(I2.getBitWidth()) == I2;
  }

  /// \brief Overload to compute a hash_code for an APInt value.
  friend hash_code hash_value(const APInt &Arg);

  /// This function returns a pointer to the internal storage of the APInt.
  /// This is useful for writing out the APInt in binary form without any
  /// conversions.
  const uint64_t *getRawData() const {
    if (isSingleWord())
      return &VAL;
    return &pVal[0];
  }

  /// @}
  /// \name Unary Operators
  /// @{

  /// \brief Postfix increment operator.
  ///
  /// \returns a new APInt value representing *this incremented by one
  const APInt operator++(int) {
    APInt API(*this);
    ++(*this);
    return API;
  }

  /// \brief Prefix increment operator.
  ///
  /// \returns *this incremented by one
  APInt &operator++();

  /// \brief Postfix decrement operator.
  ///
  /// \returns a new APInt representing *this decremented by one.
  const APInt operator--(int) {
    APInt API(*this);
    --(*this);
    return API;
  }

  /// \brief Prefix decrement operator.
  ///
  /// \returns *this decremented by one.
  APInt &operator--();

  /// \brief Unary bitwise complement operator.
  ///
  /// Performs a bitwise complement operation on this APInt.
  ///
  /// \returns an APInt that is the bitwise complement of *this
  APInt operator~() const {
    APInt Result(*this);
    Result.flipAllBits();
    return Result;
  }

  /// \brief Unary negation operator
  ///
  /// Negates *this using two's complement logic.
  ///
  /// \returns An APInt value representing the negation of *this.
  APInt operator-() const { return APInt(BitWidth, 0) - (*this); }

  /// \brief Logical negation operator.
  ///
  /// Performs logical negation operation on this APInt.
  ///
  /// \returns true if *this is zero, false otherwise.
  bool operator!() const {
    if (isSingleWord())
      return !VAL;

    for (unsigned i = 0; i != getNumWords(); ++i)
      if (pVal[i])
        return false;
    return true;
  }

  /// @}
  /// \name Assignment Operators
  /// @{

  /// \brief Copy assignment operator.
  ///
  /// \returns *this after assignment of RHS.
  APInt &operator=(const APInt &RHS) {
    // If the bitwidths are the same, we can avoid mucking with memory
    if (isSingleWord() && RHS.isSingleWord()) {
      VAL = RHS.VAL;
      BitWidth = RHS.BitWidth;
      return clearUnusedBits();
    }

    return AssignSlowCase(RHS);
  }

  /// @brief Move assignment operator.
  APInt &operator=(APInt &&that) {
    if (!isSingleWord()) {
      // The MSVC STL shipped in 2013 requires that self move assignment be a
      // no-op.  Otherwise algorithms like stable_sort will produce answers
      // where half of the output is left in a moved-from state.
      if (this == &that)
        return *this;
      delete[] pVal;
    }

    // Use memcpy so that type based alias analysis sees both VAL and pVal
    // as modified.
    memcpy(&VAL, &that.VAL, sizeof(uint64_t));

    // If 'this == &that', avoid zeroing our own bitwidth by storing to 'that'
    // first.
    unsigned ThatBitWidth = that.BitWidth;
    that.BitWidth = 0;
    BitWidth = ThatBitWidth;

    return *this;
  }

  /// \brief Assignment operator.
  ///
  /// The RHS value is assigned to *this. If the significant bits in RHS exceed
  /// the bit width, the excess bits are truncated. If the bit width is larger
  /// than 64, the value is zero filled in the unspecified high order bits.
  ///
  /// \returns *this after assignment of RHS value.
  APInt &operator=(uint64_t RHS);

  /// \brief Bitwise AND assignment operator.
  ///
  /// Performs a bitwise AND operation on this APInt and RHS. The result is
  /// assigned to *this.
  ///
  /// \returns *this after ANDing with RHS.
  APInt &operator&=(const APInt &RHS);

  /// \brief Bitwise OR assignment operator.
  ///
  /// Performs a bitwise OR operation on this APInt and RHS. The result is
  /// assigned *this;
  ///
  /// \returns *this after ORing with RHS.
  APInt &operator|=(const APInt &RHS);

  /// \brief Bitwise OR assignment operator.
  ///
  /// Performs a bitwise OR operation on this APInt and RHS. RHS is
  /// logically zero-extended or truncated to match the bit-width of
  /// the LHS.
  APInt &operator|=(uint64_t RHS) {
    if (isSingleWord()) {
      VAL |= RHS;
      clearUnusedBits();
    } else {
      pVal[0] |= RHS;
    }
    return *this;
  }

  /// \brief Bitwise XOR assignment operator.
  ///
  /// Performs a bitwise XOR operation on this APInt and RHS. The result is
  /// assigned to *this.
  ///
  /// \returns *this after XORing with RHS.
  APInt &operator^=(const APInt &RHS);

  /// \brief Multiplication assignment operator.
  ///
  /// Multiplies this APInt by RHS and assigns the result to *this.
  ///
  /// \returns *this
  APInt &operator*=(const APInt &RHS);

  /// \brief Addition assignment operator.
  ///
  /// Adds RHS to *this and assigns the result to *this.
  ///
  /// \returns *this
  APInt &operator+=(const APInt &RHS);

  /// \brief Subtraction assignment operator.
  ///
  /// Subtracts RHS from *this and assigns the result to *this.
  ///
  /// \returns *this
  APInt &operator-=(const APInt &RHS);

  /// \brief Left-shift assignment function.
  ///
  /// Shifts *this left by shiftAmt and assigns the result to *this.
  ///
  /// \returns *this after shifting left by shiftAmt
  APInt &operator<<=(unsigned shiftAmt) {
    *this = shl(shiftAmt);
    return *this;
  }

  /// @}
  /// \name Binary Operators
  /// @{

  /// \brief Bitwise AND operator.
  ///
  /// Performs a bitwise AND operation on *this and RHS.
  ///
  /// \returns An APInt value representing the bitwise AND of *this and RHS.
  APInt operator&(const APInt &RHS) const {
    assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
    if (isSingleWord())
      return APInt(getBitWidth(), VAL & RHS.VAL);
    return AndSlowCase(RHS);
  }
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT And(const APInt &RHS) const {
    return this->operator&(RHS);
  }

  /// \brief Bitwise OR operator.
  ///
  /// Performs a bitwise OR operation on *this and RHS.
  ///
  /// \returns An APInt value representing the bitwise OR of *this and RHS.
  APInt operator|(const APInt &RHS) const {
    assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
    if (isSingleWord())
      return APInt(getBitWidth(), VAL | RHS.VAL);
    return OrSlowCase(RHS);
  }

  /// \brief Bitwise OR function.
  ///
  /// Performs a bitwise or on *this and RHS. This is implemented by simply
  /// calling operator|.
  ///
  /// \returns An APInt value representing the bitwise OR of *this and RHS.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT Or(const APInt &RHS) const {
    return this->operator|(RHS);
  }

  /// \brief Bitwise XOR operator.
  ///
  /// Performs a bitwise XOR operation on *this and RHS.
  ///
  /// \returns An APInt value representing the bitwise XOR of *this and RHS.
  APInt operator^(const APInt &RHS) const {
    assert(BitWidth == RHS.BitWidth && "Bit widths must be the same");
    if (isSingleWord())
      return APInt(BitWidth, VAL ^ RHS.VAL);
    return XorSlowCase(RHS);
  }

  /// \brief Bitwise XOR function.
  ///
  /// Performs a bitwise XOR operation on *this and RHS. This is implemented
  /// through the usage of operator^.
  ///
  /// \returns An APInt value representing the bitwise XOR of *this and RHS.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT Xor(const APInt &RHS) const {
    return this->operator^(RHS);
  }

  /// \brief Multiplication operator.
  ///
  /// Multiplies this APInt by RHS and returns the result.
  APInt operator*(const APInt &RHS) const;

  /// \brief Addition operator.
  ///
  /// Adds RHS to this APInt and returns the result.
  APInt operator+(const APInt &RHS) const;
  APInt operator+(uint64_t RHS) const { return (*this) + APInt(BitWidth, RHS); }

  /// \brief Subtraction operator.
  ///
  /// Subtracts RHS from this APInt and returns the result.
  APInt operator-(const APInt &RHS) const;
  APInt operator-(uint64_t RHS) const { return (*this) - APInt(BitWidth, RHS); }

  /// \brief Left logical shift operator.
  ///
  /// Shifts this APInt left by \p Bits and returns the result.
  APInt operator<<(unsigned Bits) const { return shl(Bits); }

  /// \brief Left logical shift operator.
  ///
  /// Shifts this APInt left by \p Bits and returns the result.
  APInt operator<<(const APInt &Bits) const { return shl(Bits); }

  /// \brief Arithmetic right-shift function.
  ///
  /// Arithmetic right-shift this APInt by shiftAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT ashr(unsigned shiftAmt) const;

  /// \brief Logical right-shift function.
  ///
  /// Logical right-shift this APInt by shiftAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT lshr(unsigned shiftAmt) const;

  /// \brief Left-shift function.
  ///
  /// Left-shift this APInt by shiftAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT shl(unsigned shiftAmt) const {
    assert(shiftAmt <= BitWidth && "Invalid shift amount");
    if (isSingleWord()) {
      if (shiftAmt >= BitWidth)
        return APInt(BitWidth, 0); // avoid undefined shift results
      return APInt(BitWidth, VAL << shiftAmt);
    }
    return shlSlowCase(shiftAmt);
  }

  /// \brief Rotate left by rotateAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT rotl(unsigned rotateAmt) const;

  /// \brief Rotate right by rotateAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT rotr(unsigned rotateAmt) const;

  /// \brief Arithmetic right-shift function.
  ///
  /// Arithmetic right-shift this APInt by shiftAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT ashr(const APInt &shiftAmt) const;

  /// \brief Logical right-shift function.
  ///
  /// Logical right-shift this APInt by shiftAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT lshr(const APInt &shiftAmt) const;

  /// \brief Left-shift function.
  ///
  /// Left-shift this APInt by shiftAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT shl(const APInt &shiftAmt) const;

  /// \brief Rotate left by rotateAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT rotl(const APInt &rotateAmt) const;

  /// \brief Rotate right by rotateAmt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT rotr(const APInt &rotateAmt) const;

  /// \brief Unsigned division operation.
  ///
  /// Perform an unsigned divide operation on this APInt by RHS. Both this and
  /// RHS are treated as unsigned quantities for purposes of this division.
  ///
  /// \returns a new APInt value containing the division result
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT udiv(const APInt &RHS) const;

  /// \brief Signed division function for APInt.
  ///
  /// Signed divide this APInt by APInt RHS.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT sdiv(const APInt &RHS) const;

  /// \brief Unsigned remainder operation.
  ///
  /// Perform an unsigned remainder operation on this APInt with RHS being the
  /// divisor. Both this and RHS are treated as unsigned quantities for purposes
  /// of this operation. Note that this is a true remainder operation and not a
  /// modulo operation because the sign follows the sign of the dividend which
  /// is *this.
  ///
  /// \returns a new APInt value containing the remainder result
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT urem(const APInt &RHS) const;

  /// \brief Function for signed remainder operation.
  ///
  /// Signed remainder operation on APInt.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT srem(const APInt &RHS) const;

  /// \brief Dual division/remainder interface.
  ///
  /// Sometimes it is convenient to divide two APInt values and obtain both the
  /// quotient and remainder. This function does both operations in the same
  /// computation making it a little more efficient. The pair of input arguments
  /// may overlap with the pair of output arguments. It is safe to call
  /// udivrem(X, Y, X, Y), for example.
  static void udivrem(const APInt &LHS, const APInt &RHS, APInt &Quotient,
                      APInt &Remainder);

  static void sdivrem(const APInt &LHS, const APInt &RHS, APInt &Quotient,
                      APInt &Remainder);

  // Operations that return overflow indicators.
  APInt sadd_ov(const APInt &RHS, bool &Overflow) const;
  APInt uadd_ov(const APInt &RHS, bool &Overflow) const;
  APInt ssub_ov(const APInt &RHS, bool &Overflow) const;
  APInt usub_ov(const APInt &RHS, bool &Overflow) const;
  APInt sdiv_ov(const APInt &RHS, bool &Overflow) const;
  APInt smul_ov(const APInt &RHS, bool &Overflow) const;
  APInt umul_ov(const APInt &RHS, bool &Overflow) const;
  APInt sshl_ov(const APInt &Amt, bool &Overflow) const;
  APInt ushl_ov(const APInt &Amt, bool &Overflow) const;

  /// \brief Array-indexing support.
  ///
  /// \returns the bit value at bitPosition
  bool operator[](unsigned bitPosition) const {
    assert(bitPosition < getBitWidth() && "Bit position out of bounds!");
    return (maskBit(bitPosition) &
            (isSingleWord() ? VAL : pVal[whichWord(bitPosition)])) !=
           0;
  }

  /// @}
  /// \name Comparison Operators
  /// @{

  /// \brief Equality operator.
  ///
  /// Compares this APInt with RHS for the validity of the equality
  /// relationship.
  bool operator==(const APInt &RHS) const {
    assert(BitWidth == RHS.BitWidth && "Comparison requires equal bit widths");
    if (isSingleWord())
      return VAL == RHS.VAL;
    return EqualSlowCase(RHS);
  }

  /// \brief Equality operator.
  ///
  /// Compares this APInt with a uint64_t for the validity of the equality
  /// relationship.
  ///
  /// \returns true if *this == Val
  bool operator==(uint64_t Val) const {
    if (isSingleWord())
      return VAL == Val;
    return EqualSlowCase(Val);
  }

  /// \brief Equality comparison.
  ///
  /// Compares this APInt with RHS for the validity of the equality
  /// relationship.
  ///
  /// \returns true if *this == Val
  bool eq(const APInt &RHS) const { return (*this) == RHS; }

  /// \brief Inequality operator.
  ///
  /// Compares this APInt with RHS for the validity of the inequality
  /// relationship.
  ///
  /// \returns true if *this != Val
  bool operator!=(const APInt &RHS) const { return !((*this) == RHS); }

  /// \brief Inequality operator.
  ///
  /// Compares this APInt with a uint64_t for the validity of the inequality
  /// relationship.
  ///
  /// \returns true if *this != Val
  bool operator!=(uint64_t Val) const { return !((*this) == Val); }

  /// \brief Inequality comparison
  ///
  /// Compares this APInt with RHS for the validity of the inequality
  /// relationship.
  ///
  /// \returns true if *this != Val
  bool ne(const APInt &RHS) const { return !((*this) == RHS); }

  /// \brief Unsigned less than comparison
  ///
  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the less-than relationship.
  ///
  /// \returns true if *this < RHS when both are considered unsigned.
  bool ult(const APInt &RHS) const;

  /// \brief Unsigned less than comparison
  ///
  /// Regards both *this as an unsigned quantity and compares it with RHS for
  /// the validity of the less-than relationship.
  ///
  /// \returns true if *this < RHS when considered unsigned.
  bool ult(uint64_t RHS) const {
    return getActiveBits() > 64 ? false : getZExtValue() < RHS;
  }

  /// \brief Signed less than comparison
  ///
  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the less-than relationship.
  ///
  /// \returns true if *this < RHS when both are considered signed.
  bool slt(const APInt &RHS) const;

  /// \brief Signed less than comparison
  ///
  /// Regards both *this as a signed quantity and compares it with RHS for
  /// the validity of the less-than relationship.
  ///
  /// \returns true if *this < RHS when considered signed.
  bool slt(int64_t RHS) const {
    return getMinSignedBits() > 64 ? isNegative() : getSExtValue() < RHS;
  }

  /// \brief Unsigned less or equal comparison
  ///
  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the less-or-equal relationship.
  ///
  /// \returns true if *this <= RHS when both are considered unsigned.
  bool ule(const APInt &RHS) const { return ult(RHS) || eq(RHS); }

  /// \brief Unsigned less or equal comparison
  ///
  /// Regards both *this as an unsigned quantity and compares it with RHS for
  /// the validity of the less-or-equal relationship.
  ///
  /// \returns true if *this <= RHS when considered unsigned.
  bool ule(uint64_t RHS) const { return !ugt(RHS); }

  /// \brief Signed less or equal comparison
  ///
  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the less-or-equal relationship.
  ///
  /// \returns true if *this <= RHS when both are considered signed.
  bool sle(const APInt &RHS) const { return slt(RHS) || eq(RHS); }

  /// \brief Signed less or equal comparison
  ///
  /// Regards both *this as a signed quantity and compares it with RHS for the
  /// validity of the less-or-equal relationship.
  ///
  /// \returns true if *this <= RHS when considered signed.
  bool sle(uint64_t RHS) const { return !sgt(RHS); }

  /// \brief Unsigned greather than comparison
  ///
  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the greater-than relationship.
  ///
  /// \returns true if *this > RHS when both are considered unsigned.
  bool ugt(const APInt &RHS) const { return !ult(RHS) && !eq(RHS); }

  /// \brief Unsigned greater than comparison
  ///
  /// Regards both *this as an unsigned quantity and compares it with RHS for
  /// the validity of the greater-than relationship.
  ///
  /// \returns true if *this > RHS when considered unsigned.
  bool ugt(uint64_t RHS) const {
    return getActiveBits() > 64 ? true : getZExtValue() > RHS;
  }

  /// \brief Signed greather than comparison
  ///
  /// Regards both *this and RHS as signed quantities and compares them for the
  /// validity of the greater-than relationship.
  ///
  /// \returns true if *this > RHS when both are considered signed.
  bool sgt(const APInt &RHS) const { return !slt(RHS) && !eq(RHS); }

  /// \brief Signed greater than comparison
  ///
  /// Regards both *this as a signed quantity and compares it with RHS for
  /// the validity of the greater-than relationship.
  ///
  /// \returns true if *this > RHS when considered signed.
  bool sgt(int64_t RHS) const {
    return getMinSignedBits() > 64 ? !isNegative() : getSExtValue() > RHS;
  }

  /// \brief Unsigned greater or equal comparison
  ///
  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the greater-or-equal relationship.
  ///
  /// \returns true if *this >= RHS when both are considered unsigned.
  bool uge(const APInt &RHS) const { return !ult(RHS); }

  /// \brief Unsigned greater or equal comparison
  ///
  /// Regards both *this as an unsigned quantity and compares it with RHS for
  /// the validity of the greater-or-equal relationship.
  ///
  /// \returns true if *this >= RHS when considered unsigned.
  bool uge(uint64_t RHS) const { return !ult(RHS); }

  /// \brief Signed greather or equal comparison
  ///
  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the greater-or-equal relationship.
  ///
  /// \returns true if *this >= RHS when both are considered signed.
  bool sge(const APInt &RHS) const { return !slt(RHS); }

  /// \brief Signed greater or equal comparison
  ///
  /// Regards both *this as a signed quantity and compares it with RHS for
  /// the validity of the greater-or-equal relationship.
  ///
  /// \returns true if *this >= RHS when considered signed.
  bool sge(int64_t RHS) const { return !slt(RHS); }

  /// This operation tests if there are any pairs of corresponding bits
  /// between this APInt and RHS that are both set.
  bool intersects(const APInt &RHS) const { return (*this & RHS) != 0; }

  /// @}
  /// \name Resizing Operators
  /// @{

  /// \brief Truncate to new width.
  ///
  /// Truncate the APInt to a specified width. It is an error to specify a width
  /// that is greater than or equal to the current width.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT trunc(unsigned width) const;

  /// \brief Sign extend to a new width.
  ///
  /// This operation sign extends the APInt to a new width. If the high order
  /// bit is set, the fill on the left will be done with 1 bits, otherwise zero.
  /// It is an error to specify a width that is less than or equal to the
  /// current width.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT sext(unsigned width) const;

  /// \brief Zero extend to a new width.
  ///
  /// This operation zero extends the APInt to a new width. The high order bits
  /// are filled with 0 bits.  It is an error to specify a width that is less
  /// than or equal to the current width.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT zext(unsigned width) const;

  /// \brief Sign extend or truncate to width
  ///
  /// Make this APInt have the bit width given by \p width. The value is sign
  /// extended, truncated, or left alone to make it that width.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT sextOrTrunc(unsigned width) const;

  /// \brief Zero extend or truncate to width
  ///
  /// Make this APInt have the bit width given by \p width. The value is zero
  /// extended, truncated, or left alone to make it that width.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT zextOrTrunc(unsigned width) const;

  /// \brief Sign extend or truncate to width
  ///
  /// Make this APInt have the bit width given by \p width. The value is sign
  /// extended, or left alone to make it that width.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT sextOrSelf(unsigned width) const;

  /// \brief Zero extend or truncate to width
  ///
  /// Make this APInt have the bit width given by \p width. The value is zero
  /// extended, or left alone to make it that width.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT zextOrSelf(unsigned width) const;

  /// @}
  /// \name Bit Manipulation Operators
  /// @{

  /// \brief Set every bit to 1.
  void setAllBits() {
    if (isSingleWord())
      VAL = UINT64_MAX;
    else {
      // Set all the bits in all the words.
      for (unsigned i = 0; i < getNumWords(); ++i)
        pVal[i] = UINT64_MAX;
    }
    // Clear the unused ones
    clearUnusedBits();
  }

  /// \brief Set a given bit to 1.
  ///
  /// Set the given bit to 1 whose position is given as "bitPosition".
  void setBit(unsigned bitPosition);

  /// \brief Set every bit to 0.
  void clearAllBits() {
    if (isSingleWord())
      VAL = 0;
    else
      memset(pVal, 0, getNumWords() * APINT_WORD_SIZE);
  }

  /// \brief Set a given bit to 0.
  ///
  /// Set the given bit to 0 whose position is given as "bitPosition".
  void clearBit(unsigned bitPosition);

  /// \brief Toggle every bit to its opposite value.
  void flipAllBits() {
    if (isSingleWord())
      VAL ^= UINT64_MAX;
    else {
      for (unsigned i = 0; i < getNumWords(); ++i)
        pVal[i] ^= UINT64_MAX;
    }
    clearUnusedBits();
  }

  /// \brief Toggles a given bit to its opposite value.
  ///
  /// Toggle a given bit to its opposite value whose position is given
  /// as "bitPosition".
  void flipBit(unsigned bitPosition);

  /// @}
  /// \name Value Characterization Functions
  /// @{

  /// \brief Return the number of bits in the APInt.
  unsigned getBitWidth() const { return BitWidth; }

  /// \brief Get the number of words.
  ///
  /// Here one word's bitwidth equals to that of uint64_t.
  ///
  /// \returns the number of words to hold the integer value of this APInt.
  unsigned getNumWords() const { return getNumWords(BitWidth); }

  /// \brief Get the number of words.
  ///
  /// *NOTE* Here one word's bitwidth equals to that of uint64_t.
  ///
  /// \returns the number of words to hold the integer value with a given bit
  /// width.
  static unsigned getNumWords(unsigned BitWidth) {
    return ((uint64_t)BitWidth + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD;
  }

  /// \brief Compute the number of active bits in the value
  ///
  /// This function returns the number of active bits which is defined as the
  /// bit width minus the number of leading zeros. This is used in several
  /// computations to see how "wide" the value is.
  unsigned getActiveBits() const { return BitWidth - countLeadingZeros(); }

  /// \brief Compute the number of active words in the value of this APInt.
  ///
  /// This is used in conjunction with getActiveData to extract the raw value of
  /// the APInt.
  unsigned getActiveWords() const {
    unsigned numActiveBits = getActiveBits();
    return numActiveBits ? whichWord(numActiveBits - 1) + 1 : 1;
  }

  /// \brief Get the minimum bit size for this signed APInt
  ///
  /// Computes the minimum bit width for this APInt while considering it to be a
  /// signed (and probably negative) value. If the value is not negative, this
  /// function returns the same value as getActiveBits()+1. Otherwise, it
  /// returns the smallest bit width that will retain the negative value. For
  /// example, -1 can be written as 0b1 or 0xFFFFFFFFFF. 0b1 is shorter and so
  /// for -1, this function will always return 1.
  unsigned getMinSignedBits() const {
    if (isNegative())
      return BitWidth - countLeadingOnes() + 1;
    return getActiveBits() + 1;
  }

  /// \brief Get zero extended value
  ///
  /// This method attempts to return the value of this APInt as a zero extended
  /// uint64_t. The bitwidth must be <= 64 or the value must fit within a
  /// uint64_t. Otherwise an assertion will result.
  uint64_t getZExtValue() const {
    if (isSingleWord())
      return VAL;
    assert(getActiveBits() <= 64 && "Too many bits for uint64_t");
    return pVal[0];
  }

  /// \brief Get sign extended value
  ///
  /// This method attempts to return the value of this APInt as a sign extended
  /// int64_t. The bit width must be <= 64 or the value must fit within an
  /// int64_t. Otherwise an assertion will result.
  int64_t getSExtValue() const {
    if (isSingleWord())
      return int64_t(VAL << (APINT_BITS_PER_WORD - BitWidth)) >>
             (APINT_BITS_PER_WORD - BitWidth);
    assert(getMinSignedBits() <= 64 && "Too many bits for int64_t");
    return int64_t(pVal[0]);
  }

  /// \brief Get bits required for string value.
  ///
  /// This method determines how many bits are required to hold the APInt
  /// equivalent of the string given by \p str.
  static unsigned getBitsNeeded(StringRef str, uint8_t radix);

  /// \brief The APInt version of the countLeadingZeros functions in
  ///   MathExtras.h.
  ///
  /// It counts the number of zeros from the most significant bit to the first
  /// one bit.
  ///
  /// \returns BitWidth if the value is zero, otherwise returns the number of
  ///   zeros from the most significant bit to the first one bits.
  unsigned countLeadingZeros() const {
    if (isSingleWord()) {
      unsigned unusedBits = APINT_BITS_PER_WORD - BitWidth;
      return llvm::countLeadingZeros(VAL) - unusedBits;
    }
    return countLeadingZerosSlowCase();
  }

  /// \brief Count the number of leading one bits.
  ///
  /// This function is an APInt version of the countLeadingOnes
  /// functions in MathExtras.h. It counts the number of ones from the most
  /// significant bit to the first zero bit.
  ///
  /// \returns 0 if the high order bit is not set, otherwise returns the number
  /// of 1 bits from the most significant to the least
  unsigned countLeadingOnes() const;

  /// Computes the number of leading bits of this APInt that are equal to its
  /// sign bit.
  unsigned getNumSignBits() const {
    return isNegative() ? countLeadingOnes() : countLeadingZeros();
  }

  /// \brief Count the number of trailing zero bits.
  ///
  /// This function is an APInt version of the countTrailingZeros
  /// functions in MathExtras.h. It counts the number of zeros from the least
  /// significant bit to the first set bit.
  ///
  /// \returns BitWidth if the value is zero, otherwise returns the number of
  /// zeros from the least significant bit to the first one bit.
  unsigned countTrailingZeros() const;

  /// \brief Count the number of trailing one bits.
  ///
  /// This function is an APInt version of the countTrailingOnes
  /// functions in MathExtras.h. It counts the number of ones from the least
  /// significant bit to the first zero bit.
  ///
  /// \returns BitWidth if the value is all ones, otherwise returns the number
  /// of ones from the least significant bit to the first zero bit.
  unsigned countTrailingOnes() const {
    if (isSingleWord())
      return llvm::countTrailingOnes(VAL);
    return countTrailingOnesSlowCase();
  }

  /// \brief Count the number of bits set.
  ///
  /// This function is an APInt version of the countPopulation functions
  /// in MathExtras.h. It counts the number of 1 bits in the APInt value.
  ///
  /// \returns 0 if the value is zero, otherwise returns the number of set bits.
  unsigned countPopulation() const {
    if (isSingleWord())
      return llvm::countPopulation(VAL);
    return countPopulationSlowCase();
  }

  /// @}
  /// \name Conversion Functions
  /// @{
  void print(raw_ostream &OS, bool isSigned) const;

  /// Converts an APInt to a string and append it to Str.  Str is commonly a
  /// SmallString.
  void toString(SmallVectorImpl<char> &Str, unsigned Radix, bool Signed,
                bool formatAsCLiteral = false) const;

  /// Considers the APInt to be unsigned and converts it into a string in the
  /// radix given. The radix can be 2, 8, 10 16, or 36.
  void toStringUnsigned(SmallVectorImpl<char> &Str, unsigned Radix = 10) const {
    toString(Str, Radix, false, false);
  }

  /// Considers the APInt to be signed and converts it into a string in the
  /// radix given. The radix can be 2, 8, 10, 16, or 36.
  void toStringSigned(SmallVectorImpl<char> &Str, unsigned Radix = 10) const {
    toString(Str, Radix, true, false);
  }

  /// \brief Return the APInt as a std::string.
  ///
  /// Note that this is an inefficient method.  It is better to pass in a
  /// SmallVector/SmallString to the methods above to avoid thrashing the heap
  /// for the string.
  std::string toString(unsigned Radix, bool Signed) const;

  /// \returns a byte-swapped representation of this APInt Value.
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT byteSwap() const;

  /// \brief Converts this APInt to a double value.
  double roundToDouble(bool isSigned) const;

  /// \brief Converts this unsigned APInt to a double value.
  double roundToDouble() const { return roundToDouble(false); }

  /// \brief Converts this signed APInt to a double value.
  double signedRoundToDouble() const { return roundToDouble(true); }

  /// \brief Converts APInt bits to a double
  ///
  /// The conversion does not do a translation from integer to double, it just
  /// re-interprets the bits as a double. Note that it is valid to do this on
  /// any bit width. Exactly 64 bits will be translated.
  double bitsToDouble() const {
    union {
      uint64_t I;
      double D;
    } T;
    T.I = (isSingleWord() ? VAL : pVal[0]);
    return T.D;
  }

  /// \brief Converts APInt bits to a double
  ///
  /// The conversion does not do a translation from integer to float, it just
  /// re-interprets the bits as a float. Note that it is valid to do this on
  /// any bit width. Exactly 32 bits will be translated.
  float bitsToFloat() const {
    union {
      unsigned I;
      float F;
    } T;
    T.I = unsigned((isSingleWord() ? VAL : pVal[0]));
    return T.F;
  }

  /// \brief Converts a double to APInt bits.
  ///
  /// The conversion does not do a translation from double to integer, it just
  /// re-interprets the bits of the double.
  static APInt LLVM_ATTRIBUTE_UNUSED_RESULT doubleToBits(double V) {
    union {
      uint64_t I;
      double D;
    } T;
    T.D = V;
    return APInt(sizeof T * CHAR_BIT, T.I);
  }

  /// \brief Converts a float to APInt bits.
  ///
  /// The conversion does not do a translation from float to integer, it just
  /// re-interprets the bits of the float.
  static APInt LLVM_ATTRIBUTE_UNUSED_RESULT floatToBits(float V) {
    union {
      unsigned I;
      float F;
    } T;
    T.F = V;
    return APInt(sizeof T * CHAR_BIT, T.I);
  }

  /// @}
  /// \name Mathematics Operations
  /// @{

  /// \returns the floor log base 2 of this APInt.
  unsigned logBase2() const { return BitWidth - 1 - countLeadingZeros(); }

  /// \returns the ceil log base 2 of this APInt.
  unsigned ceilLogBase2() const {
    return BitWidth - (*this - 1).countLeadingZeros();
  }

  /// \returns the nearest log base 2 of this APInt. Ties round up.
  ///
  /// NOTE: When we have a BitWidth of 1, we define:
  /// 
  ///   log2(0) = UINT32_MAX
  ///   log2(1) = 0
  ///
  /// to get around any mathematical concerns resulting from
  /// referencing 2 in a space where 2 does no exist.
  unsigned nearestLogBase2() const {
    // Special case when we have a bitwidth of 1. If VAL is 1, then we
    // get 0. If VAL is 0, we get UINT64_MAX which gets truncated to
    // UINT32_MAX.
    if (BitWidth == 1)
      return VAL - 1;

    // Handle the zero case.
    if (!getBoolValue())
      return UINT32_MAX;

    // The non-zero case is handled by computing:
    //
    //   nearestLogBase2(x) = logBase2(x) + x[logBase2(x)-1].
    //
    // where x[i] is referring to the value of the ith bit of x.
    unsigned lg = logBase2();
    return lg + unsigned((*this)[lg - 1]);
  }

  /// \returns the log base 2 of this APInt if its an exact power of two, -1
  /// otherwise
  int32_t exactLogBase2() const {
    if (!isPowerOf2())
      return -1;
    return logBase2();
  }

  /// \brief Compute the square root
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT sqrt() const;

  /// \brief Get the absolute value;
  ///
  /// If *this is < 0 then return -(*this), otherwise *this;
  APInt LLVM_ATTRIBUTE_UNUSED_RESULT abs() const {
    if (isNegative())
      return -(*this);
    return *this;
  }

  /// \returns the multiplicative inverse for a given modulo.
  APInt multiplicativeInverse(const APInt &modulo) const;

  /// @}
  /// \name Support for division by constant
  /// @{

  /// Calculate the magic number for signed division by a constant.
  struct ms;
  ms magic() const;

  /// Calculate the magic number for unsigned division by a constant.
  struct mu;
  mu magicu(unsigned LeadingZeros = 0) const;

  /// @}
  /// \name Building-block Operations for APInt and APFloat
  /// @{

  // These building block operations operate on a representation of arbitrary
  // precision, two's-complement, bignum integer values. They should be
  // sufficient to implement APInt and APFloat bignum requirements. Inputs are
  // generally a pointer to the base of an array of integer parts, representing
  // an unsigned bignum, and a count of how many parts there are.

  /// Sets the least significant part of a bignum to the input value, and zeroes
  /// out higher parts.
  static void tcSet(integerPart *, integerPart, unsigned int);

  /// Assign one bignum to another.
  static void tcAssign(integerPart *, const integerPart *, unsigned int);

  /// Returns true if a bignum is zero, false otherwise.
  static bool tcIsZero(const integerPart *, unsigned int);

  /// Extract the given bit of a bignum; returns 0 or 1.  Zero-based.
  static int tcExtractBit(const integerPart *, unsigned int bit);

  /// Copy the bit vector of width srcBITS from SRC, starting at bit srcLSB, to
  /// DST, of dstCOUNT parts, such that the bit srcLSB becomes the least
  /// significant bit of DST.  All high bits above srcBITS in DST are
  /// zero-filled.
  static void tcExtract(integerPart *, unsigned int dstCount,
                        const integerPart *, unsigned int srcBits,
                        unsigned int srcLSB);

  /// Set the given bit of a bignum.  Zero-based.
  static void tcSetBit(integerPart *, unsigned int bit);

  /// Clear the given bit of a bignum.  Zero-based.
  static void tcClearBit(integerPart *, unsigned int bit);

  /// Returns the bit number of the least or most significant set bit of a
  /// number.  If the input number has no bits set -1U is returned.
  static unsigned int tcLSB(const integerPart *, unsigned int);
  static unsigned int tcMSB(const integerPart *parts, unsigned int n);

  /// Negate a bignum in-place.
  static void tcNegate(integerPart *, unsigned int);

  /// DST += RHS + CARRY where CARRY is zero or one.  Returns the carry flag.
  static integerPart tcAdd(integerPart *, const integerPart *,
                           integerPart carry, unsigned);

  /// DST -= RHS + CARRY where CARRY is zero or one. Returns the carry flag.
  static integerPart tcSubtract(integerPart *, const integerPart *,
                                integerPart carry, unsigned);

  /// DST += SRC * MULTIPLIER + PART   if add is true
  /// DST  = SRC * MULTIPLIER + PART   if add is false
  ///
  /// Requires 0 <= DSTPARTS <= SRCPARTS + 1.  If DST overlaps SRC they must
  /// start at the same point, i.e. DST == SRC.
  ///
  /// If DSTPARTS == SRC_PARTS + 1 no overflow occurs and zero is returned.
  /// Otherwise DST is filled with the least significant DSTPARTS parts of the
  /// result, and if all of the omitted higher parts were zero return zero,
  /// otherwise overflow occurred and return one.
  static int tcMultiplyPart(integerPart *dst, const integerPart *src,
                            integerPart multiplier, integerPart carry,
                            unsigned int srcParts, unsigned int dstParts,
                            bool add);

  /// DST = LHS * RHS, where DST has the same width as the operands and is
  /// filled with the least significant parts of the result.  Returns one if
  /// overflow occurred, otherwise zero.  DST must be disjoint from both
  /// operands.
  static int tcMultiply(integerPart *, const integerPart *, const integerPart *,
                        unsigned);

  /// DST = LHS * RHS, where DST has width the sum of the widths of the
  /// operands.  No overflow occurs.  DST must be disjoint from both
  /// operands. Returns the number of parts required to hold the result.
  static unsigned int tcFullMultiply(integerPart *, const integerPart *,
                                     const integerPart *, unsigned, unsigned);

  /// If RHS is zero LHS and REMAINDER are left unchanged, return one.
  /// Otherwise set LHS to LHS / RHS with the fractional part discarded, set
  /// REMAINDER to the remainder, return zero.  i.e.
  ///
  ///  OLD_LHS = RHS * LHS + REMAINDER
  ///
  /// SCRATCH is a bignum of the same size as the operands and result for use by
  /// the routine; its contents need not be initialized and are destroyed.  LHS,
  /// REMAINDER and SCRATCH must be distinct.
  static int tcDivide(integerPart *lhs, const integerPart *rhs,
                      integerPart *remainder, integerPart *scratch,
                      unsigned int parts);

  /// Shift a bignum left COUNT bits.  Shifted in bits are zero.  There are no
  /// restrictions on COUNT.
  static void tcShiftLeft(integerPart *, unsigned int parts,
                          unsigned int count);

  /// Shift a bignum right COUNT bits.  Shifted in bits are zero.  There are no
  /// restrictions on COUNT.
  static void tcShiftRight(integerPart *, unsigned int parts,
                           unsigned int count);

  /// The obvious AND, OR and XOR and complement operations.
  static void tcAnd(integerPart *, const integerPart *, unsigned int);
  static void tcOr(integerPart *, const integerPart *, unsigned int);
  static void tcXor(integerPart *, const integerPart *, unsigned int);
  static void tcComplement(integerPart *, unsigned int);

  /// Comparison (unsigned) of two bignums.
  static int tcCompare(const integerPart *, const integerPart *, unsigned int);

  /// Increment a bignum in-place.  Return the carry flag.
  static integerPart tcIncrement(integerPart *, unsigned int);

  /// Decrement a bignum in-place.  Return the borrow flag.
  static integerPart tcDecrement(integerPart *, unsigned int);

  /// Set the least significant BITS and clear the rest.
  static void tcSetLeastSignificantBits(integerPart *, unsigned int,
                                        unsigned int bits);

  /// \brief debug method
  void dump() const;

  /// @}
};

/// Magic data for optimising signed division by a constant.
struct APInt::ms {
  APInt m;    ///< magic number
  unsigned s; ///< shift amount
};

/// Magic data for optimising unsigned division by a constant.
struct APInt::mu {
  APInt m;    ///< magic number
  bool a;     ///< add indicator
  unsigned s; ///< shift amount
};

inline bool operator==(uint64_t V1, const APInt &V2) { return V2 == V1; }

inline bool operator!=(uint64_t V1, const APInt &V2) { return V2 != V1; }

inline raw_ostream &operator<<(raw_ostream &OS, const APInt &I) {
  I.print(OS, true);
  return OS;
}

namespace APIntOps {

/// \brief Determine the smaller of two APInts considered to be signed.
inline APInt smin(const APInt &A, const APInt &B) { return A.slt(B) ? A : B; }

/// \brief Determine the larger of two APInts considered to be signed.
inline APInt smax(const APInt &A, const APInt &B) { return A.sgt(B) ? A : B; }

/// \brief Determine the smaller of two APInts considered to be signed.
inline APInt umin(const APInt &A, const APInt &B) { return A.ult(B) ? A : B; }

/// \brief Determine the larger of two APInts considered to be unsigned.
inline APInt umax(const APInt &A, const APInt &B) { return A.ugt(B) ? A : B; }

/// \brief Check if the specified APInt has a N-bits unsigned integer value.
inline bool isIntN(unsigned N, const APInt &APIVal) { return APIVal.isIntN(N); }

/// \brief Check if the specified APInt has a N-bits signed integer value.
inline bool isSignedIntN(unsigned N, const APInt &APIVal) {
  return APIVal.isSignedIntN(N);
}

/// \returns true if the argument APInt value is a sequence of ones starting at
/// the least significant bit with the remainder zero.
inline bool isMask(unsigned numBits, const APInt &APIVal) {
  return numBits <= APIVal.getBitWidth() &&
         APIVal == APInt::getLowBitsSet(APIVal.getBitWidth(), numBits);
}

/// \brief Return true if the argument APInt value contains a sequence of ones
/// with the remainder zero.
inline bool isShiftedMask(unsigned numBits, const APInt &APIVal) {
  return isMask(numBits, (APIVal - APInt(numBits, 1)) | APIVal);
}

/// \brief Returns a byte-swapped representation of the specified APInt Value.
inline APInt byteSwap(const APInt &APIVal) { return APIVal.byteSwap(); }

/// \brief Returns the floor log base 2 of the specified APInt value.
inline unsigned logBase2(const APInt &APIVal) { return APIVal.logBase2(); }

/// \brief Compute GCD of two APInt values.
///
/// This function returns the greatest common divisor of the two APInt values
/// using Euclid's algorithm.
///
/// \returns the greatest common divisor of Val1 and Val2
APInt GreatestCommonDivisor(const APInt &Val1, const APInt &Val2);

/// \brief Converts the given APInt to a double value.
///
/// Treats the APInt as an unsigned value for conversion purposes.
inline double RoundAPIntToDouble(const APInt &APIVal) {
  return APIVal.roundToDouble();
}

/// \brief Converts the given APInt to a double value.
///
/// Treats the APInt as a signed value for conversion purposes.
inline double RoundSignedAPIntToDouble(const APInt &APIVal) {
  return APIVal.signedRoundToDouble();
}

/// \brief Converts the given APInt to a float vlalue.
inline float RoundAPIntToFloat(const APInt &APIVal) {
  return float(RoundAPIntToDouble(APIVal));
}

/// \brief Converts the given APInt to a float value.
///
/// Treast the APInt as a signed value for conversion purposes.
inline float RoundSignedAPIntToFloat(const APInt &APIVal) {
  return float(APIVal.signedRoundToDouble());
}

/// \brief Converts the given double value into a APInt.
///
/// This function convert a double value to an APInt value.
APInt RoundDoubleToAPInt(double Double, unsigned width);

/// \brief Converts a float value into a APInt.
///
/// Converts a float value into an APInt value.
inline APInt RoundFloatToAPInt(float Float, unsigned width) {
  return RoundDoubleToAPInt(double(Float), width);
}

/// \brief Arithmetic right-shift function.
///
/// Arithmetic right-shift the APInt by shiftAmt.
inline APInt ashr(const APInt &LHS, unsigned shiftAmt) {
  return LHS.ashr(shiftAmt);
}

/// \brief Logical right-shift function.
///
/// Logical right-shift the APInt by shiftAmt.
inline APInt lshr(const APInt &LHS, unsigned shiftAmt) {
  return LHS.lshr(shiftAmt);
}

/// \brief Left-shift function.
///
/// Left-shift the APInt by shiftAmt.
inline APInt shl(const APInt &LHS, unsigned shiftAmt) {
  return LHS.shl(shiftAmt);
}

/// \brief Signed division function for APInt.
///
/// Signed divide APInt LHS by APInt RHS.
inline APInt sdiv(const APInt &LHS, const APInt &RHS) { return LHS.sdiv(RHS); }

/// \brief Unsigned division function for APInt.
///
/// Unsigned divide APInt LHS by APInt RHS.
inline APInt udiv(const APInt &LHS, const APInt &RHS) { return LHS.udiv(RHS); }

/// \brief Function for signed remainder operation.
///
/// Signed remainder operation on APInt.
inline APInt srem(const APInt &LHS, const APInt &RHS) { return LHS.srem(RHS); }

/// \brief Function for unsigned remainder operation.
///
/// Unsigned remainder operation on APInt.
inline APInt urem(const APInt &LHS, const APInt &RHS) { return LHS.urem(RHS); }

/// \brief Function for multiplication operation.
///
/// Performs multiplication on APInt values.
inline APInt mul(const APInt &LHS, const APInt &RHS) { return LHS * RHS; }

/// \brief Function for addition operation.
///
/// Performs addition on APInt values.
inline APInt add(const APInt &LHS, const APInt &RHS) { return LHS + RHS; }

/// \brief Function for subtraction operation.
///
/// Performs subtraction on APInt values.
inline APInt sub(const APInt &LHS, const APInt &RHS) { return LHS - RHS; }

/// \brief Bitwise AND function for APInt.
///
/// Performs bitwise AND operation on APInt LHS and
/// APInt RHS.
inline APInt And(const APInt &LHS, const APInt &RHS) { return LHS & RHS; }

/// \brief Bitwise OR function for APInt.
///
/// Performs bitwise OR operation on APInt LHS and APInt RHS.
inline APInt Or(const APInt &LHS, const APInt &RHS) { return LHS | RHS; }

/// \brief Bitwise XOR function for APInt.
///
/// Performs bitwise XOR operation on APInt.
inline APInt Xor(const APInt &LHS, const APInt &RHS) { return LHS ^ RHS; }

/// \brief Bitwise complement function.
///
/// Performs a bitwise complement operation on APInt.
inline APInt Not(const APInt &APIVal) { return ~APIVal; }

} // End of APIntOps namespace

// See friend declaration above. This additional declaration is required in
// order to compile LLVM with IBM xlC compiler.
hash_code hash_value(const APInt &Arg);
} // End of llvm namespace

#endif
