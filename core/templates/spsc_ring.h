/**************************************************************************/
/*  spsc_ring.h                                                           */
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

// SPSCRing — lock-free single-producer / single-consumer ring buffer.
//
// Designed and implemented by Matthew Busel.
//
// Complements the existing RingBuffer<T> (ring_buffer.h), which is
// single-threaded only. SPSCRing is safe to use across exactly two
// threads: one producer and one consumer, with no locks required.
//
// Guarantees:
//   - Lock-free: no Mutex, no Semaphore, no spin-lock.
//   - Zero heap allocation: storage is a fixed inline array.
//   - Cache-friendly: producer and consumer indices live on separate
//     cache lines (alignas(64)) to prevent false sharing.
//   - Non-blocking: push() returns false when full; pop() returns false
//     and leaves p_out unchanged when empty.
//   - noexcept throughout (requires T noexcept move-constructible/assignable).
//
// Template parameters:
//   T    — element type (must be noexcept move-constructible/assignable).
//   SIZE — capacity+1; must be a power of two. Usable slots = SIZE - 1.
//
// Typical use (audio thread ↔ main thread):
//
//   SPSCRing<AudioCommand, 1024> commands;
//
//   // main thread:
//   AudioCommand cmd = ...;
//   commands.push(std::move(cmd));   // false if full — caller drops or retries
//
//   // audio thread:
//   AudioCommand out;
//   while (commands.pop(out)) { execute(out); }

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

// Silence MSVC C4324 "structure padded due to alignment specifier" — the
// padding between tail_ and head_ is intentional (eliminates false sharing).
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

template <typename T, std::size_t SIZE>
class SPSCRing {
	static_assert(SIZE >= 2, "SPSCRing: SIZE must be at least 2");
	static_assert((SIZE & (SIZE - 1u)) == 0u, "SPSCRing: SIZE must be a power of two");
	static_assert(SIZE <= (1u << 30u), "SPSCRing: SIZE exceeds safe limit (2^30)");
	static_assert(std::is_nothrow_move_constructible_v<T>,
			"SPSCRing: T must be noexcept move-constructible");
	static_assert(std::is_nothrow_move_assignable_v<T>,
			"SPSCRing: T must be noexcept move-assignable");

	static constexpr std::size_t MASK = SIZE - 1u;

public:
	// Maximum number of elements the ring can hold simultaneously (SIZE - 1).
	[[nodiscard]] static constexpr std::size_t capacity() noexcept { return SIZE - 1u; }

	// -------------------------------------------------------------------------
	// Producer API — call from the producer thread only.
	// -------------------------------------------------------------------------

	// Try to enqueue p_item. Returns true on success; false when full.
	// Call from the producer thread only.
	[[nodiscard]] bool push(T &&p_item) noexcept {
		const std::size_t tail = tail_.load(std::memory_order_relaxed);
		const std::size_t next = (tail + 1u) & MASK;
		if (next == head_.load(std::memory_order_acquire)) {
			return false; // full
		}
		buf_[tail] = std::move(p_item);
		tail_.store(next, std::memory_order_release);
		return true;
	}

	// Convenience overload: copy-enqueue. Returns true on success; false when full.
	[[nodiscard]] bool push(const T &p_item) noexcept(std::is_nothrow_copy_constructible_v<T>) {
		T copy{ p_item };
		return push(std::move(copy));
	}

	// -------------------------------------------------------------------------
	// Consumer API — call from the consumer thread only.
	// -------------------------------------------------------------------------

	// Try to dequeue into r_out. Returns true on success; false when empty.
	// r_out is left unchanged when the ring is empty.
	// Call from the consumer thread only.
	[[nodiscard]] bool pop(T &r_out) noexcept {
		const std::size_t head = head_.load(std::memory_order_relaxed);
		if (head == tail_.load(std::memory_order_acquire)) {
			return false; // empty
		}
		r_out = std::move(buf_[head]);
		head_.store((head + 1u) & MASK, std::memory_order_release);
		return true;
	}

	// -------------------------------------------------------------------------
	// Diagnostic queries (approximate — non-atomic snapshot).
	// -------------------------------------------------------------------------

	// Approximate number of elements in the ring. Not exact under concurrent use.
	// Suitable for monitoring and metrics only.
	[[nodiscard]] std::size_t size_approx() const noexcept {
		const std::size_t t = tail_.load(std::memory_order_acquire);
		const std::size_t h = head_.load(std::memory_order_acquire);
		return (t - h + SIZE) & MASK;
	}

	[[nodiscard]] bool empty_approx() const noexcept {
		return tail_.load(std::memory_order_acquire) ==
				head_.load(std::memory_order_acquire);
	}

	[[nodiscard]] bool full_approx() const noexcept {
		const std::size_t t = tail_.load(std::memory_order_acquire);
		const std::size_t h = head_.load(std::memory_order_acquire);
		return ((t + 1u) & MASK) == h;
	}

private:
	// Producer index — own cache line prevents false sharing with head_.
	alignas(64) std::atomic<std::size_t> tail_{ 0 };
	// Consumer index — own cache line prevents false sharing with tail_.
	alignas(64) std::atomic<std::size_t> head_{ 0 };
	// Inline element storage — zero heap allocation.
	std::array<T, SIZE> buf_{};
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif
