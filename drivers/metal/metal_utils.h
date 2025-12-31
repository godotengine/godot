/**************************************************************************/
/*  metal_utils.h                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#import <os/log.h>

#import <functional>

/// Godot limits the number of dynamic buffers to 8.
///
/// This is a minimum guarantee for Vulkan.
constexpr uint32_t MAX_DYNAMIC_BUFFERS = 8;

// From rendering/rendering_device/vsync/frame_queue_size
static constexpr uint32_t MAX_FRAME_COUNT = 3;

#pragma mark - Boolean flags

namespace flags {

/*! Sets the flags within the value parameter specified by the mask parameter. */
template <typename Tv, typename Tm>
void set(Tv &p_value, Tm p_mask) {
	using T = std::underlying_type_t<Tv>;
	p_value = static_cast<Tv>(static_cast<T>(p_value) | static_cast<T>(p_mask));
}

/*! Clears the flags within the value parameter specified by the mask parameter. */
template <typename Tv, typename Tm>
void clear(Tv &p_value, Tm p_mask) {
	using T = std::underlying_type_t<Tv>;
	p_value = static_cast<Tv>(static_cast<T>(p_value) & ~static_cast<T>(p_mask));
}

/*! Returns whether the specified value has any of the bits specified in mask set to 1. */
template <typename Tv, typename Tm>
static constexpr bool any(Tv p_value, const Tm p_mask) {
	return ((p_value & p_mask) != 0);
}

/*! Returns whether the specified value has all of the bits specified in mask set to 1. */
template <typename Tv, typename Tm>
static constexpr bool all(Tv p_value, const Tm p_mask) {
	return ((p_value & p_mask) == p_mask);
}

} //namespace flags

#pragma mark - Alignment and Offsets

static constexpr bool is_power_of_two(uint64_t p_value) {
	return p_value && ((p_value & (p_value - 1)) == 0);
}

static constexpr uint64_t round_up_to_alignment(uint64_t p_value, uint64_t p_alignment) {
	DEV_ASSERT(is_power_of_two(p_alignment));

	if (p_alignment == 0) {
		return p_value;
	}

	uint64_t mask = p_alignment - 1;
	uint64_t aligned_value = (p_value + mask) & ~mask;

	return aligned_value;
}

template <typename F>
class Defer {
public:
	explicit Defer(F &&f) :
			func_(std::forward<F>(f)) {}
	~Defer() { func_(); }

	// Non-copyable (correct RAII semantics)
	Defer(const Defer &) = delete;
	Defer &operator=(const Defer &) = delete;

	// Movable
	Defer(Defer &&) = default;
	Defer &operator=(Defer &&) = default;

private:
	F func_;
};

// C++17 class template argument deduction.
template <typename F>
Defer(F &&) -> Defer<std::decay_t<F>>;

#define CONCAT_INTERNAL(x, y) x##y
#define CONCAT(x, y) CONCAT_INTERNAL(x, y)
#define DEFER const auto &CONCAT(defer__, __LINE__) = Defer

extern os_log_t LOG_DRIVER;
// Used for dynamic tracing.
extern os_log_t LOG_INTERVALS;

_FORCE_INLINE_ static constexpr uint32_t make_msl_version(uint32_t p_major, uint32_t p_minor = 0, uint32_t p_patch = 0) {
	return (p_major * 10000) + (p_minor * 100) + p_patch;
}

_FORCE_INLINE_ static constexpr void parse_msl_version(uint32_t p_version, uint32_t &r_major, uint32_t &r_minor) {
	r_major = p_version / 10000;
	r_minor = (p_version % 10000) / 100;
}

constexpr uint32_t MSL_VERSION_23 = make_msl_version(2, 3);
constexpr uint32_t MSL_VERSION_24 = make_msl_version(2, 4);
constexpr uint32_t MSL_VERSION_30 = make_msl_version(3, 0);
constexpr uint32_t MSL_VERSION_31 = make_msl_version(3, 1);
constexpr uint32_t MSL_VERSION_32 = make_msl_version(3, 2);
constexpr uint32_t MSL_VERSION_40 = make_msl_version(4, 0);

/* MSL Language version table
 *
 * | Version |  macOS  |   iOS   |
 * |---------|---------|---------|
 * |   1.0   |         |   9.0   |
 * |   1.1   |  10.11  |   9.0   |
 * |   1.2   |  10.12  |  10.0   |
 * |   2.0   |  10.13  |  11.0   |
 * |   2.1   |  10.14  |  12.0   |
 * |   2.2   |  10.15  |  13.0   |
 * |   2.3   |  11.0   |  14.0   |
 * |   2.4   |  12.0   |  15.0   |
 * |   3.0   |  13.0   |  16.0   |
 * |   3.1   |  14.0   |  17.0   |
 * |   3.2   |  15.0   |  18.0   |
 * |   4.0   |  26.0   |  26.0   |
 * |---------|---------|---------|
 */
