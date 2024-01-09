// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2017-2019 Valve Corporation
// Copyright (c) 2017-2019 LunarG, Inc.
// Copyright (c) 2019 Collabora, Ltd.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Initial Author: Rylie Pavlik <rylie.pavlik@collabora.com>
//

/*!
 * @file
 *
 * Some utilities, primarily for working with OpenXR handles in a generic way.
 */

#pragma once

#include <openxr/openxr.h>

#include <string>
#include <stdint.h>

inline std::string to_hex(const uint8_t* const data, size_t bytes) {
    std::string out(2 + bytes * 2, '?');
    out[0] = '0';
    out[1] = 'x';
    static const char* hex = "0123456789abcdef";
    auto ch = out.end();
    for (size_t i = 0; i < bytes; ++i) {
        auto b = data[i];
        *--ch = hex[(b >> 0) & 0xf];
        *--ch = hex[(b >> 4) & 0xf];
    }
    return out;
}

template <typename T>
inline std::string to_hex(const T& data) {
    return to_hex(reinterpret_cast<const uint8_t* const>(&data), sizeof(data));
}

#if XR_PTR_SIZE == 8
/// Convert a handle into a same-sized integer.
template <typename T>
static inline uint64_t MakeHandleGeneric(T handle) {
    return reinterpret_cast<uint64_t>(handle);
}

/// Treat an integer as a handle
template <typename T>
static inline T& TreatIntegerAsHandle(uint64_t& handle) {
    return reinterpret_cast<T&>(handle);
}

/// @overload
template <typename T>
static inline T const& TreatIntegerAsHandle(uint64_t const& handle) {
    return reinterpret_cast<T const&>(handle);
}

/// Does a correctly-sized integer represent a null handle?
static inline bool IsIntegerNullHandle(uint64_t handle) { return XR_NULL_HANDLE == reinterpret_cast<void*>(handle); }

#else

/// Convert a handle into a same-sized integer: no-op on 32-bit systems
static inline uint64_t MakeHandleGeneric(uint64_t handle) { return handle; }

/// Treat an integer as a handle: no-op on 32-bit systems
template <typename T>
static inline T& TreatIntegerAsHandle(uint64_t& handle) {
    return handle;
}

/// @overload
template <typename T>
static inline T const& TreatIntegerAsHandle(uint64_t const& handle) {
    return handle;
}

/// Does a correctly-sized integer represent a null handle?
static inline bool IsIntegerNullHandle(uint64_t handle) { return XR_NULL_HANDLE == handle; }

#endif

/// Turns a uint64_t into a string formatted as hex.
///
/// The core of the HandleToHexString implementation is in here.
inline std::string Uint64ToHexString(uint64_t val) { return to_hex(val); }

/// Turns a uint32_t into a string formatted as hex.
inline std::string Uint32ToHexString(uint32_t val) { return to_hex(val); }

/// Turns an OpenXR handle into a string formatted as hex.
template <typename T>
inline std::string HandleToHexString(T handle) {
    return to_hex(handle);
}

/// Turns a pointer-sized integer into a string formatted as hex.
inline std::string UintptrToHexString(uintptr_t val) { return to_hex(val); }

/// Convert a pointer to a string formatted as hex.
template <typename T>
inline std::string PointerToHexString(T const* ptr) {
    return to_hex(ptr);
}
