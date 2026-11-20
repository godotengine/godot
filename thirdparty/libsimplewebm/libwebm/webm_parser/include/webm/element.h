// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_ELEMENT_H_
#define INCLUDE_WEBM_ELEMENT_H_

#include <cstdint>
#include <limits>
#include <utility>

#include "./id.h"

/**
 \file
 A wrapper around an object that represents a WebM element, including its parsed
 metadata.
 */

namespace webm {

/**
 \addtogroup PUBLIC_API
 @{
 */

/**
 A wrapper around an object that represents a WebM element.

 Since some elements may be absent, this wrapper is used to indicate the
 presence (or lack thereof) of an element in a WebM document. If the element is
 encoded in the file and it has been parsed, `is_present()` will return true.
 Otherwise it will return false since the element was ommitted or skipped when
 parsing.
 */
template <typename T>
class Element {
 public:
  /**
   Value-initializes the element's value and makes `is_present()` false.
   */
  constexpr Element() = default;

  /**
   Creates an element with the given value and makes `is_present()` false.

   \param value The value of the element.
   */
  explicit constexpr Element(const T& value) : value_(value) {}

  /**
   Creates an element with the given value and makes `is_present()` false.

   \param value The value of the element.
   */
  explicit constexpr Element(T&& value) : value_(std::move(value)) {}

  /**
   Creates an element with the given value and presence state.

   \param value The value of the element.
   \param is_present True if the element is present, false if it is absent.
   */
  constexpr Element(const T& value, bool is_present)
      : value_(value), is_present_(is_present) {}

  /**
   Creates an element with the given value and presence state.

   \param value The value of the element.
   \param is_present True if the element is present, false if it is absent.
   */
  constexpr Element(T&& value, bool is_present)
      : value_(std::move(value)), is_present_(is_present) {}

  constexpr Element(const Element<T>& other) = default;
  constexpr Element(Element<T>&& other) = default;

  ~Element() = default;

  Element<T>& operator=(const Element<T>& other) = default;
  Element<T>& operator=(Element<T>&& other) = default;

  /**
   Sets the element's value and state.

   \param value The new value for the element.
   \param is_present The new presence state for the element.
   */
  void Set(const T& value, bool is_present) {
    value_ = value;
    is_present_ = is_present;
  }

  /**
   Sets the element's value and state.

   \param value The new value for the element.
   \param is_present The new presence state for the element.
   */
  void Set(T&& value, bool is_present) {
    value_ = std::move(value);
    is_present_ = is_present;
  }

  /**
   Gets the element's value.
   */
  constexpr const T& value() const { return value_; }

  /**
   Gets a mutuable pointer to the element's value (will never be null).
   */
  T* mutable_value() { return &value_; }

  /**
   Returns true if the element is present, false otherwise.
   */
  constexpr bool is_present() const { return is_present_; }

  bool operator==(const Element<T>& other) const {
    return is_present_ == other.is_present_ && value_ == other.value_;
  }

 private:
  T value_{};
  bool is_present_ = false;
};

/**
 Metadata for WebM elements that are encountered when parsing.
 */
struct ElementMetadata {
  /**
   The EBML ID of the element.
   */
  Id id;

  /**
   The number of bytes that were used to encode the EBML ID and element size.

   If the size of the header is unknown (which is only the case if a seek was
   performed to the middle of an element, so its header was not parsed), this
   will be the value `kUnknownHeaderSize`.
   */
  std::uint32_t header_size;

  /**
   The size of the element.

   This is number of bytes in the element's body, which excludes the header
   bytes.

   If the size of the element's body is unknown, this will be the value
   `kUnknownElementSize`.
   */
  std::uint64_t size;

  /**
   The absolute byte position of the element, starting at the first byte of the
   element's header.

   If the position of the element is unknown (which is only the case if a seek
   was performed to the middle of an element), this will be the value
   `kUnknownElementPosition`.
   */
  std::uint64_t position;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const ElementMetadata& other) const {
    return id == other.id && header_size == other.header_size &&
           size == other.size && position == other.position;
  }
};

/**
 A special value for `ElementMetadata::header_size` indicating the header size
 is not known.
 */
constexpr std::uint64_t kUnknownHeaderSize =
    std::numeric_limits<std::uint32_t>::max();

/**
 A special value for `ElementMetadata::size` indicating the element's size is
 not known.
 */
constexpr std::uint64_t kUnknownElementSize =
    std::numeric_limits<std::uint64_t>::max();

/**
 A special value for `ElementMetadata::position` indicating the element's
 position is not known.
 */
constexpr std::uint64_t kUnknownElementPosition =
    std::numeric_limits<std::uint64_t>::max();

/**
 @}
 */

}  // namespace webm

#endif  // INCLUDE_WEBM_ELEMENT_H_
