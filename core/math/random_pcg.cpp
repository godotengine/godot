/**************************************************************************/
/*  random_pcg.cpp                                                        */
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

#include "random_pcg.h"

#include "core/os/os.h"
#include "core/templates/vector.h"

// Platform headers for the OS-provided cryptographic randomness source.
//
// Each platform Godot supports exposes a non-blocking call that returns
// cryptographically strong random bytes from the kernel CSPRNG. We use these
// directly rather than going through std::random_device because:
//
//   1. Every supported platform has a documented, well-defined API for this.
//      Routing through the C++ standard library adds an indirection with no
//      portability benefit, and historically std::random_device has been
//      deterministic on some MinGW builds.
//   2. The dependency surface stays minimal (no <random> include in this
//      core file).
//
// The kernel guarantees its entropy pool is fully initialised before any
// user-mode Godot code runs, so these calls never block in practice.
#if defined(WINDOWS_ENABLED)
#include <bcrypt.h>
#elif defined(UNIX_ENABLED) || defined(WEB_ENABLED)
#include <unistd.h>
#endif

#include <atomic>

RandomPCG::RandomPCG(uint64_t p_seed, uint64_t p_inc) :
		pcg(),
		current_inc(p_inc) {
	seed(p_seed);
}

// Read 8 bytes from the OS cryptographic randomness source into a uint64_t.
// Returns true on success; on failure leaves r_value untouched and returns
// false, allowing the caller to fall back to a deterministic-but-distinct
// alternative seed.
//
// This is a static helper rather than a method on `OS` because it has exactly
// one caller (RandomPCG::randomize) and no other use case justifies broadening
// OS's surface. If a future refactor wants to expose entropy more broadly,
// this helper is the right starting point to lift into `OS::get_entropy()`.
static bool _os_get_entropy_u64(uint64_t &r_value) {
	uint8_t buf[8];
#if defined(WINDOWS_ENABLED)
	// BCRYPT_USE_SYSTEM_PREFERRED_RNG asks BCryptGenRandom to use the
	// OS-default CSPRNG without requiring a provider handle. The function
	// returns 0 (STATUS_SUCCESS) on success; any non-zero NTSTATUS is a
	// failure. There are no documented blocking conditions for the preferred
	// RNG provider on Windows Vista or later. The bcrypt library is already
	// linked by platform/windows/detect.py.
	if (BCryptGenRandom(nullptr, buf, sizeof(buf), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
		return false;
	}
#elif (defined(UNIX_ENABLED) || defined(WEB_ENABLED)) && !(defined(__ANDROID__) && __ANDROID_API__ < 28)
	// getentropy returns 0 on success and -1 on error. The maximum buffer size
	// is 256 bytes; we request 8, so the call cannot fail with EIO. Available
	// since glibc 2.25 (Linux 3.17+), macOS 10.12, iOS 10, all current BSDs,
	// Android API 28+, and Emscripten (which forwards to crypto.getRandomValues
	// in the browser).
	if (getentropy(buf, sizeof(buf)) != 0) {
		return false;
	}
#else
	// Unsupported platform (currently: Android API < 28 or other unrecognised
	// builds). The caller will use the fallback path below, which still
	// guarantees uncorrelated seeds for back-to-back calls within a process.
	(void)buf;
	return false;
#endif
	// Compose the byte array into a uint64 in a portable, alignment-safe way.
	// Using memcpy or a reinterpret_cast would be equivalent on every
	// architecture Godot supports, but the explicit shift makes the intent
	// unambiguous and avoids relying on host endianness.
	r_value = 0;
	for (int i = 0; i < 8; i++) {
		r_value |= ((uint64_t)buf[i]) << (i * 8);
	}
	return true;
}

void RandomPCG::randomize() {
	uint64_t entropy = 0;
	if (likely(_os_get_entropy_u64(entropy))) {
		// Hot path. Each call to randomize() (and therefore every
		// default-constructed RandomNumberGenerator, since its constructor
		// routes through here) gets an independent 64-bit seed drawn from the
		// kernel CSPRNG. Two RNG instances created in the same microsecond
		// are now statistically independent.
		seed(entropy);
		return;
	}

	// Fallback path: OS entropy is unavailable. Reachable only on
	// unrecognised platforms or older Android (API < 28). We retain this
	// branch to ensure unrecognised builds still produce uncorrelated seeds
	// for back-to-back calls, addressing the original bug at minimum even
	// when the preferred source is missing.
	//
	// The previous implementation used "(unix_time + ticks_usec) * pcg.state
	// + INC", which collapsed to identical output for any two calls within
	// the same wall-clock second on any second-resolution unix_time cast. We
	// replace it with a mix of:
	//
	//   1. Wall-clock seconds (varies across runs).
	//   2. Monotonic microsecond ticks (varies within a run, with
	//      sub-microsecond ambiguity under tight loops).
	//   3. A process-wide atomic counter that increments on every call to
	//      randomize(), guaranteeing two calls in the same microsecond
	//      receive distinct mix inputs.
	//
	// The mix is then run through the SplitMix64 finalizer, a bijective hash
	// that diffuses small input differences across all output bits. Without
	// diffusion, sequential counter values would only differ in their low
	// bits, and two seeds that differ by one bit can produce PCG streams that
	// are visibly correlated in the first few outputs.
	static std::atomic<uint64_t> fallback_counter{ 0 };
	uint64_t mix = (uint64_t)OS::get_singleton()->get_unix_time();
	mix ^= OS::get_singleton()->get_ticks_usec();
	// Multiply the counter by SplitMix64's golden-ratio constant before
	// XORing in. This spreads the counter's low bits across the full 64-bit
	// range, preventing partial cancellation against the timing mix when the
	// counter is small and increments by 1.
	mix ^= fallback_counter.fetch_add(1, std::memory_order_relaxed) * 0x9E3779B97F4A7C15ULL;

	// SplitMix64 finalizer. Constants from Steele/Lea/Flood, "Fast Splittable
	// Pseudorandom Number Generators" (OOPSLA 2014). This is a bijective hash
	// over uint64, guaranteeing distinct inputs produce distinct outputs;
	// combined with strong avalanche it ensures even adjacent counter values
	// yield uncorrelated seeds.
	mix = (mix ^ (mix >> 30)) * 0xBF58476D1CE4E5B9ULL;
	mix = (mix ^ (mix >> 27)) * 0x94D049BB133111EBULL;
	mix = mix ^ (mix >> 31);

	seed(mix);
}

int64_t RandomPCG::rand_weighted(const Vector<float> &p_weights) {
	ERR_FAIL_COND_V_MSG(p_weights.is_empty(), -1, "Weights array is empty.");
	int64_t weights_size = p_weights.size();
	const float *weights = p_weights.ptr();
	float weights_sum = 0.0;
	for (int64_t i = 0; i < weights_size; ++i) {
		weights_sum += weights[i];
	}

	float remaining_distance = randf() * weights_sum;
	for (int64_t i = 0; i < weights_size; ++i) {
		remaining_distance -= weights[i];
		if (remaining_distance < 0) {
			return i;
		}
	}

	for (int64_t i = weights_size - 1; i >= 0; --i) {
		if (weights[i] > 0) {
			return i;
		}
	}
	return -1;
}

double RandomPCG::random(double p_from, double p_to) {
	return randd() * (p_to - p_from) + p_from;
}

float RandomPCG::random(float p_from, float p_to) {
	return randf() * (p_to - p_from) + p_from;
}

int RandomPCG::random(int p_from, int p_to) {
	if (p_from == p_to) {
		return p_from;
	}

	int64_t min = MIN(p_from, p_to);
	int64_t max = MAX(p_from, p_to);
	uint32_t diff = static_cast<uint32_t>(max - min);

	if (diff == UINT32_MAX) {
		// Can't add 1 to max uint32_t value for inclusive range, so call rand without passing bounds.
		return static_cast<int64_t>(rand()) + min;
	}

	return static_cast<int64_t>(rand(diff + 1U)) + min;
}
