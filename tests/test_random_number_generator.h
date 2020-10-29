/*************************************************************************/
/*  test_random_number_generator.h                                       */
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

#ifndef TEST_RANDOM_NUMBER_GENERATOR_H
#define TEST_RANDOM_NUMBER_GENERATOR_H

#include "core/math/random_number_generator.h"
#include "tests/test_macros.h"

namespace TestRandomNumberGenerator {

TEST_CASE("[RandomNumberGenerator] On unit circle") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	for (int i = 0; i < 100; ++i) {
		const Vector2 &point = rng->randv_circle(); // Default: 1.0..1.0.
		INFO(point.length());
		CHECK(Math::is_equal_approx(point.length(), 1.0));
		CHECK(point.is_normalized());
	}
}

TEST_CASE("[RandomNumberGenerator] Inside unit circle") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	for (int i = 0; i < 100; ++i) {
		const Vector2 &point = rng->randv_circle(0.0, 1.0);
		INFO(point.length());
		CHECK(point.length() < 1.0);
	}
}

TEST_CASE("[RandomNumberGenerator] Within circle range") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	for (int i = 0; i < 100; ++i) {
		const Vector2 &point = rng->randv_circle(0.5, 1.0);
		INFO(point.length());
		CHECK(point.length() > 0.5);
		CHECK(point.length() < 1.0);
	}
}

TEST_CASE("[RandomNumberGenerator] On circle, non-unit radius") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	for (int i = 0; i < 100; ++i) {
		const Vector2 &point = rng->randv_circle(10.0, 10.0);
		INFO(point.length());
		CHECK(Math::is_equal_approx(point.length(), 10.0));
	}
}

TEST_CASE("[Stress][RandomNumberGenerator] On unit circle, 100000 iterations") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	for (int i = 0; i < 100000; ++i) {
		const Vector2 &point = rng->randv_circle(); // Default: 1.0..1.0.
		INFO(point);
	}
}

TEST_CASE("[Stress][RandomNumberGenerator] Inside unit circle, 100000 iterations") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	for (int i = 0; i < 100000; ++i) {
		const Vector2 &point = rng->randv_circle(0.0, 1.0);
		INFO(point);
	}
}

TEST_CASE("[Stress][RandomNumberGenerator] Within circle range, 100000 iterations") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	for (int i = 0; i < 100000; ++i) {
		const Vector2 &point = rng->randv_circle(0.5, 1.0);
		INFO(point);
	}
}

} // namespace TestRandomNumberGenerator

#endif // TEST_RANDOM_NUMBER_GENERATOR_H
