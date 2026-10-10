// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_STATUS_H_
#define INCLUDE_WEBM_STATUS_H_

#include <cstdint>

/**
 \file
 Status information that represents success, failure, etc. for operations
 throughout the API.
 */

namespace webm {

/**
 \addtogroup PUBLIC_API
 @{
 */

/**
 An object used to represent the resulting status of various operations,
 indicating success, failure, or some other result.
 */
struct Status {
  /**
   A list of generic status codes used by the parser. Users are encouraged to
   reuse these values as needed in the derivations of `Callback`. These values
   will always be <= 0.
   */
  enum Code : std::int32_t {
    // General status codes. Range: 0 to -1024.
    /**
     The operation successfully completed.
     */
    kOkCompleted = 0,

    /**
     The operation was successful but only partially completed (for example, a
     read that resulted in fewer bytes than requested).
     */
    kOkPartial = -1,

    /**
     The operation would block, and should be tried again later.
     */
    kWouldBlock = -2,

    /**
     More data was requested but the file has ended.
     */
    kEndOfFile = -3,

    /**
     The reader was unable to seek to the requested location.
     */
    kSeekFailed = -4,

    // Parsing errors. Range: -1025 to -2048.
    /**
     An element's ID is malformed.
     */
    kInvalidElementId = -1025,

    /**
     An element's size is malformed.
     */
    kInvalidElementSize = -1026,

    /**
     An unknown element has unknown size.
     */
    kIndefiniteUnknownElement = -1027,

    /**
     A child element overflowed the parent element's bounds.
     */
    kElementOverflow = -1028,

    /**
     An element's size exceeds the system's memory limits.
     */
    kNotEnoughMemory = -1029,

    /**
     An element's value is illegal/malformed.
     */
    kInvalidElementValue = -1030,

    /**
     A recursive element was so deeply nested that exceeded the parser's limit.
     */
    kExceededRecursionDepthLimit = -1031,

    // The following codes are internal-only and should not be used by users.
    // Additionally, these codes should never be returned to the user; doing so
    // is considered a bug.
    /**
     \internal Parsing should switch from reading to skipping elements.
     */
    kSwitchToSkip = INT32_MIN,
  };

  /**
   Status codes <= 0 are reserved by the parsing library. User error codes
   should be positive (> 0). Users are encouraged to use codes from the `Code`
   enum, but may use a positive error code if some application-specific error is
   encountered.
   */
  std::int32_t code;

  Status() = default;
  Status(const Status&) = default;
  Status(Status&&) = default;
  Status& operator=(const Status&) = default;
  Status& operator=(Status&&) = default;

  /**
   Creates a new `Status` object with the given status code.

   \param code The status code which will be used to set the `code` member.
   */
  constexpr explicit Status(Code code) : code(code) {}

  /**
   Creates a new `Status` object with the given status code.

   \param code The status code which will be used to set the `code` member.
   */
  constexpr explicit Status(std::int32_t code) : code(code) {}

  /**
   Returns true if the status code is either `kOkCompleted` or `kOkPartial`.
   Provided for convenience.
   */
  constexpr bool ok() const {
    return code == kOkCompleted || code == kOkPartial;
  }

  /**
   Returns true if the status code is `kOkCompleted`. Provided for convenience.
   */
  constexpr bool completed_ok() const { return code == kOkCompleted; }

  /**
   Returns true if the status is considered a parsing error. Parsing errors
   represent unrecoverable errors due to malformed data. Only status codes in
   the range -2048 to -1025 (inclusive) are considered parsing errors.
   */
  constexpr bool is_parsing_error() const {
    return -2048 <= code && code <= -1025;
  }
};

/**
 @}
 */

}  // namespace webm

#endif  // INCLUDE_WEBM_STATUS_H_
