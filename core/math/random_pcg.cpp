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
#include "core/os/spin_lock.h"
#include "core/templates/vector.h"

namespace {
// komm64: --random-seed (see random_pcg.h). Its own increment keeps the
// stream apart from a generator seeded with the same number.
constexpr uint64_t FIXED_SEED_STREAM_INC = 0x9e3779b97f4a7c15ULL;
bool fixed_seed_set = false;
pcg32_random_t fixed_seed_stream = {};
SpinLock fixed_seed_lock;
} // namespace

void RandomPCG::set_fixed_seed(uint64_t p_seed) {
	fixed_seed_lock.lock();
	pcg32_srandom_r(&fixed_seed_stream, p_seed, FIXED_SEED_STREAM_INC);
	fixed_seed_set = true;
	fixed_seed_lock.unlock();
}

bool RandomPCG::has_fixed_seed() {
	return fixed_seed_set;
}

uint64_t RandomPCG::get_fixed_seed_stream_state() {
	fixed_seed_lock.lock();
	const uint64_t state = fixed_seed_stream.state;
	fixed_seed_lock.unlock();
	return state;
}

void RandomPCG::set_fixed_seed_stream_state(uint64_t p_state) {
	fixed_seed_lock.lock();
	fixed_seed_stream.state = p_state;
	fixed_seed_lock.unlock();
}

RandomPCG::RandomPCG(uint64_t p_seed, uint64_t p_inc) :
		pcg(),
		current_inc(p_inc) {
	seed(p_seed);
}

void RandomPCG::randomize() {
	if (fixed_seed_set) {
		fixed_seed_lock.lock();
		const uint64_t hi = pcg32_random_r(&fixed_seed_stream);
		const uint64_t lo = pcg32_random_r(&fixed_seed_stream);
		fixed_seed_lock.unlock();
		seed((hi << 32) | lo);
		return;
	}
	seed(((uint64_t)OS::get_singleton()->get_unix_time() + OS::get_singleton()->get_ticks_usec()) * pcg.state + PCG_DEFAULT_INC_64);
}

int64_t RandomPCG::rand_weighted(const Vector<float> &p_weights) {
	ERR_FAIL_COND_V_MSG(p_weights.is_empty(), -1, "Weights array is empty.");
	int64_t weights_size = p_weights.size();
	const float *weights = p_weights.ptr();
	float weights_sum = 0.0;
	for (int64_t i = 0; i < weights_size; ++i) {
		ERR_FAIL_COND_V_MSG(weights[i] < 0.0, -1, vformat("Weight array contains the negative value %f at index %d.", weights[i], i));
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
