//===-- TimeValue.h - Declare OS TimeValue Concept --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file declares the operating system TimeValue concept.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TIMEVALUE_H
#define LLVM_SUPPORT_TIMEVALUE_H

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
namespace sys {
  /// This class is used where a precise fixed point in time is required. The
  /// range of TimeValue spans many hundreds of billions of years both past and
  /// present.  The precision of TimeValue is to the nanosecond. However, the
  /// actual precision of its values will be determined by the resolution of
  /// the system clock. The TimeValue class is used in conjunction with several
  /// other lib/System interfaces to specify the time at which a call should
  /// timeout, etc.
  /// @since 1.4
  /// @brief Provides an abstraction for a fixed point in time.
  class TimeValue {

  /// @name Constants
  /// @{
  public:

    /// A constant TimeValue representing the smallest time
    /// value permissible by the class. MinTime is some point
    /// in the distant past, about 300 billion years BCE.
    /// @brief The smallest possible time value.
    static TimeValue MinTime() {
      return TimeValue ( INT64_MIN,0 );
    }

    /// A constant TimeValue representing the largest time
    /// value permissible by the class. MaxTime is some point
    /// in the distant future, about 300 billion years AD.
    /// @brief The largest possible time value.
    static TimeValue MaxTime() {
      return TimeValue ( INT64_MAX,0 );
    }

    /// A constant TimeValue representing the base time,
    /// or zero time of 00:00:00 (midnight) January 1st, 2000.
    /// @brief 00:00:00 Jan 1, 2000 UTC.
    static TimeValue ZeroTime() {
      return TimeValue ( 0,0 );
    }

    /// A constant TimeValue for the Posix base time which is
    /// 00:00:00 (midnight) January 1st, 1970.
    /// @brief 00:00:00 Jan 1, 1970 UTC.
    static TimeValue PosixZeroTime() {
      return TimeValue ( PosixZeroTimeSeconds,0 );
    }

    /// A constant TimeValue for the Win32 base time which is
    /// 00:00:00 (midnight) January 1st, 1601.
    /// @brief 00:00:00 Jan 1, 1601 UTC.
    static TimeValue Win32ZeroTime() {
      return TimeValue ( Win32ZeroTimeSeconds,0 );
    }

  /// @}
  /// @name Types
  /// @{
  public:
    typedef int64_t SecondsType;    ///< Type used for representing seconds.
    typedef int32_t NanoSecondsType;///< Type used for representing nanoseconds.

    enum TimeConversions {
      NANOSECONDS_PER_SECOND = 1000000000,  ///< One Billion
      MICROSECONDS_PER_SECOND = 1000000,    ///< One Million
      MILLISECONDS_PER_SECOND = 1000,       ///< One Thousand
      NANOSECONDS_PER_MICROSECOND = 1000,   ///< One Thousand
      NANOSECONDS_PER_MILLISECOND = 1000000,///< One Million
      NANOSECONDS_PER_WIN32_TICK = 100      ///< Win32 tick is 10^7 Hz (10ns)
    };

  /// @}
  /// @name Constructors
  /// @{
  public:
    /// \brief Default construct a time value, initializing to ZeroTime.
    TimeValue() : seconds_(0), nanos_(0) {}

    /// Caller provides the exact value in seconds and nanoseconds. The
    /// \p nanos argument defaults to zero for convenience.
    /// @brief Explicit constructor
    explicit TimeValue (SecondsType seconds, NanoSecondsType nanos = 0)
      : seconds_( seconds ), nanos_( nanos ) { this->normalize(); }

    /// Caller provides the exact value as a double in seconds with the
    /// fractional part representing nanoseconds.
    /// @brief Double Constructor.
    explicit TimeValue( double new_time )
      : seconds_( 0 ) , nanos_ ( 0 ) {
      SecondsType integer_part = static_cast<SecondsType>( new_time );
      seconds_ = integer_part;
      nanos_ = static_cast<NanoSecondsType>( (new_time -
               static_cast<double>(integer_part)) * NANOSECONDS_PER_SECOND );
      this->normalize();
    }

    /// This is a static constructor that returns a TimeValue that represents
    /// the current time.
    /// @brief Creates a TimeValue with the current time (UTC).
    static TimeValue now();

  /// @}
  /// @name Operators
  /// @{
  public:
    /// Add \p that to \p this.
    /// @returns this
    /// @brief Incrementing assignment operator.
    TimeValue& operator += (const TimeValue& that ) {
      this->seconds_ += that.seconds_  ;
      this->nanos_ += that.nanos_ ;
      this->normalize();
      return *this;
    }

    /// Subtract \p that from \p this.
    /// @returns this
    /// @brief Decrementing assignment operator.
    TimeValue& operator -= (const TimeValue &that ) {
      this->seconds_ -= that.seconds_ ;
      this->nanos_ -= that.nanos_ ;
      this->normalize();
      return *this;
    }

    /// Determine if \p this is less than \p that.
    /// @returns True iff *this < that.
    /// @brief True if this < that.
    int operator < (const TimeValue &that) const { return that > *this; }

    /// Determine if \p this is greather than \p that.
    /// @returns True iff *this > that.
    /// @brief True if this > that.
    int operator > (const TimeValue &that) const {
      if ( this->seconds_ > that.seconds_ ) {
          return 1;
      } else if ( this->seconds_ == that.seconds_ ) {
          if ( this->nanos_ > that.nanos_ ) return 1;
      }
      return 0;
    }

    /// Determine if \p this is less than or equal to \p that.
    /// @returns True iff *this <= that.
    /// @brief True if this <= that.
    int operator <= (const TimeValue &that) const { return that >= *this; }

    /// Determine if \p this is greater than or equal to \p that.
    /// @returns True iff *this >= that.
    int operator >= (const TimeValue &that) const {
      if ( this->seconds_ > that.seconds_ ) {
          return 1;
      } else if ( this->seconds_ == that.seconds_ ) {
          if ( this->nanos_ >= that.nanos_ ) return 1;
      }
      return 0;
    }

    /// Determines if two TimeValue objects represent the same moment in time.
    /// @returns True iff *this == that.
    int operator == (const TimeValue &that) const {
      return (this->seconds_ == that.seconds_) &&
             (this->nanos_ == that.nanos_);
    }

    /// Determines if two TimeValue objects represent times that are not the
    /// same.
    /// @returns True iff *this != that.
    int operator != (const TimeValue &that) const { return !(*this == that); }

    /// Adds two TimeValue objects together.
    /// @returns The sum of the two operands as a new TimeValue
    /// @brief Addition operator.
    friend TimeValue operator + (const TimeValue &tv1, const TimeValue &tv2);

    /// Subtracts two TimeValue objects.
    /// @returns The difference of the two operands as a new TimeValue
    /// @brief Subtraction operator.
    friend TimeValue operator - (const TimeValue &tv1, const TimeValue &tv2);

  /// @}
  /// @name Accessors
  /// @{
  public:

    /// Returns only the seconds component of the TimeValue. The nanoseconds
    /// portion is ignored. No rounding is performed.
    /// @brief Retrieve the seconds component
    SecondsType seconds() const { return seconds_; }

    /// Returns only the nanoseconds component of the TimeValue. The seconds
    /// portion is ignored.
    /// @brief Retrieve the nanoseconds component.
    NanoSecondsType nanoseconds() const { return nanos_; }

    /// Returns only the fractional portion of the TimeValue rounded down to the
    /// nearest microsecond (divide by one thousand).
    /// @brief Retrieve the fractional part as microseconds;
    uint32_t microseconds() const {
      return nanos_ / NANOSECONDS_PER_MICROSECOND;
    }

    /// Returns only the fractional portion of the TimeValue rounded down to the
    /// nearest millisecond (divide by one million).
    /// @brief Retrieve the fractional part as milliseconds;
    uint32_t milliseconds() const {
      return nanos_ / NANOSECONDS_PER_MILLISECOND;
    }

    /// Returns the TimeValue as a number of microseconds. Note that the value
    /// returned can overflow because the range of a uint64_t is smaller than
    /// the range of a TimeValue. Nevertheless, this is useful on some operating
    /// systems and is therefore provided.
    /// @brief Convert to a number of microseconds (can overflow)
    uint64_t usec() const {
      return seconds_ * MICROSECONDS_PER_SECOND +
             ( nanos_ / NANOSECONDS_PER_MICROSECOND );
    }

    /// Returns the TimeValue as a number of milliseconds. Note that the value
    /// returned can overflow because the range of a uint64_t is smaller than
    /// the range of a TimeValue. Nevertheless, this is useful on some operating
    /// systems and is therefore provided.
    /// @brief Convert to a number of milliseconds (can overflow)
    uint64_t msec() const {
      return seconds_ * MILLISECONDS_PER_SECOND +
             ( nanos_ / NANOSECONDS_PER_MILLISECOND );
    }

    /// Converts the TimeValue into the corresponding number of seconds
    /// since the epoch (00:00:00 Jan 1,1970).
    uint64_t toEpochTime() const {
      return seconds_ - PosixZeroTimeSeconds;
    }

    /// Converts the TimeValue into the corresponding number of "ticks" for
    /// Win32 platforms, correcting for the difference in Win32 zero time.
    /// @brief Convert to Win32's FILETIME
    /// (100ns intervals since 00:00:00 Jan 1, 1601 UTC)
    uint64_t toWin32Time() const {
      uint64_t result = (uint64_t)10000000 * (seconds_ - Win32ZeroTimeSeconds);
      result += nanos_ / NANOSECONDS_PER_WIN32_TICK;
      return result;
    }

    /// Provides the seconds and nanoseconds as results in its arguments after
    /// correction for the Posix zero time.
    /// @brief Convert to timespec time (ala POSIX.1b)
    void getTimespecTime( uint64_t& seconds, uint32_t& nanos ) const {
      seconds = seconds_ - PosixZeroTimeSeconds;
      nanos = nanos_;
    }

    /// Provides conversion of the TimeValue into a readable time & date.
    /// @returns std::string containing the readable time value
    /// @brief Convert time to a string.
    std::string str() const;

  /// @}
  /// @name Mutators
  /// @{
  public:
    /// The seconds component of the TimeValue is set to \p sec without
    /// modifying the nanoseconds part.  This is useful for whole second
    /// arithmetic.
    /// @brief Set the seconds component.
    void seconds (SecondsType sec ) {
      this->seconds_ = sec;
      this->normalize();
    }

    /// The nanoseconds component of the TimeValue is set to \p nanos without
    /// modifying the seconds part. This is useful for basic computations
    /// involving just the nanoseconds portion. Note that the TimeValue will be
    /// normalized after this call so that the fractional (nanoseconds) portion
    /// will have the smallest equivalent value.
    /// @brief Set the nanoseconds component using a number of nanoseconds.
    void nanoseconds ( NanoSecondsType nanos ) {
      this->nanos_ = nanos;
      this->normalize();
    }

    /// The seconds component remains unchanged.
    /// @brief Set the nanoseconds component using a number of microseconds.
    void microseconds ( int32_t micros ) {
      this->nanos_ = micros * NANOSECONDS_PER_MICROSECOND;
      this->normalize();
    }

    /// The seconds component remains unchanged.
    /// @brief Set the nanoseconds component using a number of milliseconds.
    void milliseconds ( int32_t millis ) {
      this->nanos_ = millis * NANOSECONDS_PER_MILLISECOND;
      this->normalize();
    }

    /// @brief Converts from microsecond format to TimeValue format
    void usec( int64_t microseconds ) {
      this->seconds_ = microseconds / MICROSECONDS_PER_SECOND;
      this->nanos_ = NanoSecondsType(microseconds % MICROSECONDS_PER_SECOND) *
        NANOSECONDS_PER_MICROSECOND;
      this->normalize();
    }

    /// @brief Converts from millisecond format to TimeValue format
    void msec( int64_t milliseconds ) {
      this->seconds_ = milliseconds / MILLISECONDS_PER_SECOND;
      this->nanos_ = NanoSecondsType(milliseconds % MILLISECONDS_PER_SECOND) *
        NANOSECONDS_PER_MILLISECOND;
      this->normalize();
    }

    /// Converts the \p seconds argument from PosixTime to the corresponding
    /// TimeValue and assigns that value to \p this.
    /// @brief Convert seconds form PosixTime to TimeValue
    void fromEpochTime( SecondsType seconds ) {
      seconds_ = seconds + PosixZeroTimeSeconds;
      nanos_ = 0;
      this->normalize();
    }

    /// Converts the \p win32Time argument from Windows FILETIME to the
    /// corresponding TimeValue and assigns that value to \p this.
    /// @brief Convert seconds form Windows FILETIME to TimeValue
    void fromWin32Time( uint64_t win32Time ) {
      this->seconds_ = win32Time / 10000000 + Win32ZeroTimeSeconds;
      this->nanos_ = NanoSecondsType(win32Time  % 10000000) * 100;
    }

  /// @}
  /// @name Implementation
  /// @{
  private:
    /// This causes the values to be represented so that the fractional
    /// part is minimized, possibly incrementing the seconds part.
    /// @brief Normalize to canonical form.
    void normalize();

  /// @}
  /// @name Data
  /// @{
  private:
    /// Store the values as a <timeval>.
    SecondsType      seconds_;///< Stores the seconds part of the TimeVal
    NanoSecondsType  nanos_;  ///< Stores the nanoseconds part of the TimeVal

    static const SecondsType PosixZeroTimeSeconds;
    static const SecondsType Win32ZeroTimeSeconds;
  /// @}

  };

inline TimeValue operator + (const TimeValue &tv1, const TimeValue &tv2) {
  TimeValue sum (tv1.seconds_ + tv2.seconds_, tv1.nanos_ + tv2.nanos_);
  sum.normalize ();
  return sum;
}

inline TimeValue operator - (const TimeValue &tv1, const TimeValue &tv2) {
  TimeValue difference (tv1.seconds_ - tv2.seconds_, tv1.nanos_ - tv2.nanos_ );
  difference.normalize ();
  return difference;
}

}
}

#endif
