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

TEST_CASE("[RandomNumberGenerator] Zero for first number immediately after seeding") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	uint32_t n1 = rng->randi();
	uint32_t n2 = rng->randi();
	INFO("Initial random values: " << n1 << " " << n2);
	CHECK(n1 != 0);

	rng->set_seed(1);
	uint32_t n3 = rng->randi();
	uint32_t n4 = rng->randi();
	INFO("Values after changing the seed: " << n3 << " " << n4);
	CHECK(n3 != 0);
}

TEST_CASE("[RandomNumberGenerator] Restore state") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->randomize();
	uint64_t last_seed = rng->get_seed();
	INFO("Current seed: " << last_seed);

	rng->randi();
	rng->randi();

	CHECK_MESSAGE(rng->get_seed() == last_seed,
			"The seed should remain the same after generating some numbers");

	uint64_t saved_state = rng->get_state();
	INFO("Current state: " << saved_state);

	real_t f1_before = rng->randf();
	real_t f2_before = rng->randf();
	INFO("This seed produces: " << f1_before << " " << f2_before);

	// Restore now.
	rng->set_state(saved_state);

	real_t f1_after = rng->randf();
	real_t f2_after = rng->randf();
	INFO("Resetting the state produces: " << f1_after << " " << f2_after);

	String msg = "Should restore the sequence of numbers after resetting the state";
	CHECK_MESSAGE(f1_before == f1_after, msg);
	CHECK_MESSAGE(f2_before == f2_after, msg);
}

TEST_CASE("[RandomNumberGenerator] Restore from seed") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	INFO("Current seed: " << rng->get_seed());
	uint32_t s0_1_before = rng->randi();
	uint32_t s0_2_before = rng->randi();
	INFO("This seed produces: " << s0_1_before << " " << s0_2_before);

	rng->set_seed(9000);
	INFO("Current seed: " << rng->get_seed());
	uint32_t s9000_1 = rng->randi();
	uint32_t s9000_2 = rng->randi();
	INFO("This seed produces: " << s9000_1 << " " << s9000_2);

	rng->set_seed(0);
	INFO("Current seed: " << rng->get_seed());
	uint32_t s0_1_after = rng->randi();
	uint32_t s0_2_after = rng->randi();
	INFO("This seed produces: " << s0_1_after << " " << s0_2_after);

	String msg = "Should restore the sequence of numbers after resetting the seed";
	CHECK_MESSAGE(s0_1_before == s0_1_after, msg);
	CHECK_MESSAGE(s0_2_before == s0_2_after, msg);
}
} // namespace TestRandomNumberGenerator

#endif // TEST_RANDOM_NUMBER_GENERATOR_H
