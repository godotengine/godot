/*************************************************************************/
/*  random_number_generator.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef RANDOM_NUMBER_GENERATOR_H
#define RANDOM_NUMBER_GENERATOR_H

#include "core/math/random_pcg.h"
#include "core/reference.h"

class RandomNumberGenerator : public Reference {
	GDCLASS(RandomNumberGenerator, Reference);

protected:
	RandomPCG randbase;

	static void _bind_methods();

public:
	_FORCE_INLINE_ void set_seed(uint64_t seed) { randbase.seed(seed); }

	_FORCE_INLINE_ uint64_t get_seed() { return randbase.get_seed(); }

	_FORCE_INLINE_ void randomize() { randbase.randomize(); }

	_FORCE_INLINE_ uint32_t randi() { return randbase.rand(); }

	_FORCE_INLINE_ real_t randf() { return randbase.randf(); }

	_FORCE_INLINE_ real_t randf_range(real_t from, real_t to) { return randbase.random(from, to); }

	_FORCE_INLINE_ real_t randfn(real_t mean = 0.0, real_t deviation = 1.0) { return randbase.randfn(mean, deviation); }

	_FORCE_INLINE_ int randi_range(int from, int to) {
		unsigned int ret = randbase.rand();
		if (to < from) {
			return ret % (from - to + 1) + to;
		} else {
			return ret % (to - from + 1) + from;
		}
	}

	_FORCE_INLINE_ Vector2 randv_circle(real_t p_min_radius = 1.0, real_t p_max_radius = 1.0) { // Unit length by default.
		real_t r2_max = p_max_radius * p_max_radius;
		real_t r2_min = p_min_radius * p_min_radius;
		real_t r = Math::sqrt(randbase.random(0.0, 1.0) * (r2_max - r2_min) + r2_min);
		real_t t = randbase.random(0.0, 1.0) * Math_TAU;
		return Vector2(r * Math::cos(t), r * Math::sin(t));
	}

	RandomNumberGenerator() {}
};

#endif // RANDOM_NUMBER_GENERATOR_H
