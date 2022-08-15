//===- Endian.h - Utilities for IO with endian specific data ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares generic functions to read and write endian specific data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ENDIAN_H
#define LLVM_SUPPORT_ENDIAN_H

#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SwapByteOrder.h"

#pragma warning( push )           // HLSL Change - constant comparisons are done for portability
#pragma warning( disable : 6326 ) // 'Potential comparison of a constant with another constant.'

namespace llvm {
namespace support {
enum endianness {big, little, native};

// These are named values for common alignments.
enum {aligned = 0, unaligned = 1};

namespace detail {
  /// \brief ::value is either alignment, or alignof(T) if alignment is 0.
  template<class T, int alignment>
  struct PickAlignment {
    enum {value = alignment == 0 ? AlignOf<T>::Alignment : alignment};
  };
} // end namespace detail

namespace endian {
/// Swap the bytes of value to match the given endianness.
template<typename value_type, endianness endian>
inline value_type byte_swap(value_type value) {
  if (endian != native && sys::IsBigEndianHost != (endian == big))
    sys::swapByteOrder(value);
  return value;
}

/// Read a value of a particular endianness from memory.
template<typename value_type,
         endianness endian,
         std::size_t alignment>
inline value_type read(const void *memory) {
  value_type ret;

  memcpy(&ret,
         LLVM_ASSUME_ALIGNED(memory,
           (detail::PickAlignment<value_type, alignment>::value)),
         sizeof(value_type));
  return byte_swap<value_type, endian>(ret);
}

/// Read a value of a particular endianness from a buffer, and increment the
/// buffer past that value.
template<typename value_type, endianness endian, std::size_t alignment,
         typename CharT>
inline value_type readNext(const CharT *&memory) {
  value_type ret = read<value_type, endian, alignment>(memory);
  memory += sizeof(value_type);
  return ret;
}

/// Write a value to memory with a particular endianness.
template<typename value_type,
         endianness endian,
         std::size_t alignment>
inline void write(void *memory, value_type value) {
  value = byte_swap<value_type, endian>(value);
  memcpy(LLVM_ASSUME_ALIGNED(memory,
           (detail::PickAlignment<value_type, alignment>::value)),
         &value,
         sizeof(value_type));
}
} // end namespace endian

namespace detail {
template<typename value_type,
         endianness endian,
         std::size_t alignment>
struct packed_endian_specific_integral {
  operator value_type() const {
    return endian::read<value_type, endian, alignment>(
      (const void*)Value.buffer);
  }

  void operator=(value_type newValue) {
    endian::write<value_type, endian, alignment>(
      (void*)Value.buffer, newValue);
  }

  packed_endian_specific_integral &operator+=(value_type newValue) {
    *this = *this + newValue;
    return *this;
  }

  packed_endian_specific_integral &operator-=(value_type newValue) {
    *this = *this - newValue;
    return *this;
  }

  packed_endian_specific_integral &operator|=(value_type newValue) {
    *this = *this | newValue;
    return *this;
  }

  packed_endian_specific_integral &operator&=(value_type newValue) {
    *this = *this & newValue;
    return *this;
  }

private:
  AlignedCharArray<PickAlignment<value_type, alignment>::value,
                   sizeof(value_type)> Value;

public:
  struct ref {
    explicit ref(void *Ptr) : Ptr(Ptr) {}

    operator value_type() const {
      return endian::read<value_type, endian, alignment>(Ptr);
    }

    void operator=(value_type NewValue) {
      endian::write<value_type, endian, alignment>(Ptr, NewValue);
    }

  private:
    void *Ptr;
  };
};

} // end namespace detail

typedef detail::packed_endian_specific_integral
                  <uint16_t, little, unaligned> ulittle16_t;
typedef detail::packed_endian_specific_integral
                  <uint32_t, little, unaligned> ulittle32_t;
typedef detail::packed_endian_specific_integral
                  <uint64_t, little, unaligned> ulittle64_t;

typedef detail::packed_endian_specific_integral
                   <int16_t, little, unaligned> little16_t;
typedef detail::packed_endian_specific_integral
                   <int32_t, little, unaligned> little32_t;
typedef detail::packed_endian_specific_integral
                   <int64_t, little, unaligned> little64_t;

typedef detail::packed_endian_specific_integral
                    <uint16_t, little, aligned> aligned_ulittle16_t;
typedef detail::packed_endian_specific_integral
                    <uint32_t, little, aligned> aligned_ulittle32_t;
typedef detail::packed_endian_specific_integral
                    <uint64_t, little, aligned> aligned_ulittle64_t;

typedef detail::packed_endian_specific_integral
                     <int16_t, little, aligned> aligned_little16_t;
typedef detail::packed_endian_specific_integral
                     <int32_t, little, aligned> aligned_little32_t;
typedef detail::packed_endian_specific_integral
                     <int64_t, little, aligned> aligned_little64_t;

typedef detail::packed_endian_specific_integral
                  <uint16_t, big, unaligned>    ubig16_t;
typedef detail::packed_endian_specific_integral
                  <uint32_t, big, unaligned>    ubig32_t;
typedef detail::packed_endian_specific_integral
                  <uint64_t, big, unaligned>    ubig64_t;

typedef detail::packed_endian_specific_integral
                   <int16_t, big, unaligned>    big16_t;
typedef detail::packed_endian_specific_integral
                   <int32_t, big, unaligned>    big32_t;
typedef detail::packed_endian_specific_integral
                   <int64_t, big, unaligned>    big64_t;

typedef detail::packed_endian_specific_integral
                    <uint16_t, big, aligned>    aligned_ubig16_t;
typedef detail::packed_endian_specific_integral
                    <uint32_t, big, aligned>    aligned_ubig32_t;
typedef detail::packed_endian_specific_integral
                    <uint64_t, big, aligned>    aligned_ubig64_t;

typedef detail::packed_endian_specific_integral
                     <int16_t, big, aligned>    aligned_big16_t;
typedef detail::packed_endian_specific_integral
                     <int32_t, big, aligned>    aligned_big32_t;
typedef detail::packed_endian_specific_integral
                     <int64_t, big, aligned>    aligned_big64_t;

typedef detail::packed_endian_specific_integral
                  <uint16_t, native, unaligned> unaligned_uint16_t;
typedef detail::packed_endian_specific_integral
                  <uint32_t, native, unaligned> unaligned_uint32_t;
typedef detail::packed_endian_specific_integral
                  <uint64_t, native, unaligned> unaligned_uint64_t;

typedef detail::packed_endian_specific_integral
                   <int16_t, native, unaligned> unaligned_int16_t;
typedef detail::packed_endian_specific_integral
                   <int32_t, native, unaligned> unaligned_int32_t;
typedef detail::packed_endian_specific_integral
                   <int64_t, native, unaligned> unaligned_int64_t;

namespace endian {
inline uint16_t read16le(const void *p) { return *(const ulittle16_t *)p; }
inline uint32_t read32le(const void *p) { return *(const ulittle32_t *)p; }
inline uint64_t read64le(const void *p) { return *(const ulittle64_t *)p; }
inline uint16_t read16be(const void *p) { return *(const ubig16_t *)p; }
inline uint32_t read32be(const void *p) { return *(const ubig32_t *)p; }
inline uint64_t read64be(const void *p) { return *(const ubig64_t *)p; }

inline void write16le(void *p, uint16_t v) { *(ulittle16_t *)p = v; }
inline void write32le(void *p, uint32_t v) { *(ulittle32_t *)p = v; }
inline void write64le(void *p, uint64_t v) { *(ulittle64_t *)p = v; }
inline void write16be(void *p, uint16_t v) { *(ubig16_t *)p = v; }
inline void write32be(void *p, uint32_t v) { *(ubig32_t *)p = v; }
inline void write64be(void *p, uint64_t v) { *(ubig64_t *)p = v; }
} // end namespace endian
} // end namespace support
} // end namespace llvm

#pragma warning( pop ) // HLSL Change - pop warning level

#endif
