/*************************************************************************/
/*  test_interpolator.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_INTERPOLATOR_H
#define TEST_INTERPOLATOR_H

#include "modules/network_synchronizer/interpolator.h"

#include "tests/test_macros.h"

namespace TestInterpolator {

template <class T>
T generate_value(real_t value) {
	if constexpr (std::is_same_v<T, Vector2> || std::is_same_v<T, Vector2i>) {
		return T(value, value);
	} else if constexpr (std::is_same_v<T, Vector3> || std::is_same_v<T, Vector3i>) {
		return T(value, value, value);
	} else {
		return static_cast<T>(value);
	}
}

// TODO: Add other types
TEST_CASE_TEMPLATE("[Modules][Interpolator] Interpolation", T, int, real_t, Vector2, Vector2i, Vector3) {
	LocalVector<real_t> fractions;
	fractions.reserve(7);
	fractions.push_back(0.0);
	fractions.push_back(1.0);
	fractions.push_back(0.5);
	fractions.push_back(0.001);
	fractions.push_back(0.999);
	fractions.push_back(0.25);
	fractions.push_back(0.75);

	Map<real_t, real_t> values;
	values.insert(0.0, 1.0);
	values.insert(-1.0, 1.0);
	values.insert(0.0, -1.0);
	values.insert(10, 15);

	Interpolator interpolator;
	for (const Map<real_t, real_t>::Element *E = values.front(); E; E = E->next()) {
		for (uint32_t j = 0; j < fractions.size(); ++j) {
			// Skip custom interpolator for now
			for (int k = Interpolator::FALLBACK_INTERPOLATE; k < Interpolator::FALLBACK_CUSTOM_INTERPOLATOR; ++k) {
				const T first_value = generate_value<T>(E->key());
				const T second_value = generate_value<T>(E->value());

				interpolator.reset();
				const int variable_id = interpolator.register_variable(T(), static_cast<Interpolator::Fallback>(k));
				interpolator.terminate_init();
				interpolator.begin_write(0);
				interpolator.epoch_insert(variable_id, first_value);
				interpolator.end_write();

				interpolator.begin_write(1);
				interpolator.epoch_insert(variable_id, second_value);
				interpolator.end_write();

				CAPTURE(k);
				CAPTURE(fractions[j]);
				CAPTURE(first_value);
				CAPTURE(second_value);
				const T result = interpolator.pop_epoch(0, fractions[j])[0];
				switch (k) {
					case Interpolator::FALLBACK_INTERPOLATE: {
						CHECK(result == Interpolator::interpolate(first_value, second_value, fractions[j]).operator T());
					} break;
					case Interpolator::FALLBACK_DEFAULT: {
						if (fractions[j] == 0.0) {
							CHECK(result == first_value);
						} else if (fractions[j] == 1.0) {
							CHECK(result == second_value);
						} else {
							CHECK(result == T());
						}
					} break;
					case Interpolator::FALLBACK_OLD_OR_NEAREST: {
						if (fractions[j] == 0.0) {
							CHECK(result == first_value);
						} else if (fractions[j] == 1.0) {
							CHECK(result == second_value);
						} else {
							CHECK(result == first_value);
						}
					} break;
					case Interpolator::FALLBACK_NEW_OR_NEAREST: {
						if (fractions[j] == 0.0) {
							CHECK(result == first_value);
						} else if (fractions[j] == 1.0) {
							CHECK(result == second_value);
						} else {
							CHECK(result == second_value);
						}
					} break;
				}
			}
		}
	}
}
} // namespace TestInterpolator

#endif // TEST_INTERPOLATOR_H
