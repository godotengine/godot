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
#include "core/templates/safe_refcount.h"
#include "core/templates/vector.h"

RandomPCG::RandomPCG(uint64_t p_seed, uint64_t p_inc) :
		pcg(),
		current_inc(p_inc) {
	seed(p_seed);
}

void RandomPCG::randomize() {
	// OS::get_entropy() dispatches to getentropy() or /dev/urandom in
	// drivers/unix/os_unix.cpp, and BCryptGenRandom in
	// platform/windows/os_windows.cpp.
	uint8_t buf[8];
	if (likely(OS::get_singleton()->get_entropy(buf, sizeof(buf)) == OK)) {
		uint64_t entropy = 0;
		for (int i = 0; i < 8; i++) {
			entropy |= ((uint64_t)buf[i]) << (i * 8);
		}
		seed(entropy);
		return;
	}

	// Fallback for platforms with neither getentropy() nor /dev/urandom.
	// The atomic counter keeps seeds distinct when the clock has poor
	// resolution; SplitMix64 diffuses the result.
	static SafeNumeric<uint64_t> fallback_counter;
	uint64_t mix = (uint64_t)OS::get_singleton()->get_unix_time();
	mix ^= OS::get_singleton()->get_ticks_usec();
	mix ^= fallback_counter.postincrement() * 0x9E3779B97F4A7C15ULL;
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
