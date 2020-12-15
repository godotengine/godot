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

TEST_CASE("[RandomNumberGenerator] Check rand_range for expected functionality") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(0);
	uint64_t prev_state = rng->get_state();
	uint32_t seed_before = rng->get_seed();

	INFO("randf_range should give float between -100 and 100, base test.");
	float num_before;
	for (int i = 0; i < 1000; i++) {
		num_before = rng->randf_range(-100, 100);
		CHECK(num_before >= -100);
		CHECK(num_before <= 100);
	}

	float num_after1;
	rng->randomize();
	INFO("randf_range should give float between -75 and 75.");
	INFO("Shouldn't be affected by randomize.");
	for (int i = 0; i < 1000; i++) {
		num_after1 = rng->randf_range(-75, 75);
		CHECK(num_after1 >= -75);
		CHECK(num_after1 <= 75);
	}

	rng->set_state(prev_state);
	int num_after2;
	INFO("randi_range should give int between -50 and 50.");
	INFO("Shouldn't be affected by set_state.");
	for (int i = 0; i < 1000; i++) {
		num_after2 = rng->randi_range(-50, 50);
		CHECK(num_after2 >= -50);
		CHECK(num_after2 <= 50);
	}

	rng->set_seed(seed_before);
	int num_after3;
	INFO("randi_range should give int between -25 and 25.");
	INFO("Shouldn't be affected by set_seed.");
	for (int i = 0; i < 1000; i++) {
		num_after3 = rng->randi_range(-25, 25);
		CHECK(num_after3 >= -25);
		CHECK(num_after3 <= 25);
	}

	rng->randf();
	rng->randf();
	float num_after4;

	INFO("randf_range should give float between -10 and 10.");
	INFO("Shouldn't be affected after generating new numbers.");
	for (int i = 0; i < 1000; i++) {
		num_after4 = rng->randf_range(-10, 10);
		CHECK(num_after4 >= -10);
		CHECK(num_after4 <= 10);
	}

	rng->randi();
	rng->randi();
	int num_after5;

	INFO("randi_range should give int between -5 and 5.");
	INFO("Shouldn't be affected after generating new numbers.");
	for (int i = 0; i < 1000; i++) {
		num_after5 = rng->randf_range(-5, 5);
		CHECK(num_after5 >= -5);
		CHECK(num_after5 <= 5);
	}
}

TEST_CASE("[RandomNumberGenerator] Check randfn for expected functionality") {
	Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
	rng->set_seed(1);
	INFO("randfn should give a number between -5 to 5 (5 std deviations away; above 99.7% chance it will be in this range).");
	INFO("Standard randfn function call.");
	float temp_check;
	for (int i = 0; i < 100; i++) {
		temp_check = rng->randfn();
		CHECK(temp_check >= -5);
		CHECK(temp_check <= 5);
	}

	INFO("Checks if randfn produces number between -5 to 5 after multiple randi/randf calls.");
	INFO("5 std deviations away; above 99.7% chance it will be in this range.");
	rng->randf();
	rng->randi();
	for (int i = 0; i < 100; i++) {
		temp_check = rng->randfn();
		CHECK(temp_check >= -5);
		CHECK(temp_check <= 5);
	}

	INFO("Checks if user defined mean and deviation work properly.");
	INFO("5 std deviations away; above 99.7% chance it will be in this range.");
	for (int i = 0; i < 100; i++) {
		temp_check = rng->randfn(5, 10);
		CHECK(temp_check >= -45);
		CHECK(temp_check <= 55);
	}

	INFO("Checks if randfn works with changed seeds.");
	INFO("5 std deviations away; above 99.7% chance it will be in this range.");
	rng->randomize();
	for (int i = 0; i < 100; i++) {
		temp_check = rng->randfn(3, 3);
		CHECK(temp_check >= -12);
		CHECK(temp_check <= 18);
	}
}
} // namespace TestRandomNumberGenerator

#endif // TEST_RANDOM_NUMBER_GENERATOR_H
