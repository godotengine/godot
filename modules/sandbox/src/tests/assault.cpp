/**************************************************************************/
/*  assault.cpp                                                           */
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

#include "../guest_datatypes.h"

#include <random>

/**
 * @brief Assault the sandbox with random GuestVariants.
 *
 * @param test An unused string. TBD.
 * @param iterations The number of GuestVariants to create.
 */
void Sandbox::assault(const String &test, int64_t iterations) {
	Sandbox sandbox;
	Sandbox::CurrentState state;
	sandbox.m_current_state = &state;

	// Create a random number generator.
	std::random_device rd;
	std::uniform_int_distribution<int> rand(0, 256);
	std::uniform_int_distribution<int> type_rand(0, Variant::VARIANT_MAX);

	for (size_t i = 0; i < iterations; i++) {
		std::array<uint8_t, sizeof(GuestVariant)> data;
		std::generate(data.begin(), data.end(), [&]() { return rand(rd); });
		// Create a random GuestVariant
		GuestVariant v;
		std::memcpy(&v, data.data(), data.size());
		// Make the type valid
		v.type = static_cast<Variant::Type>(type_rand(rd));

		try {
			// Try to use the GuestVariant
			v.toVariant(sandbox);
		} catch (const std::exception &e) {
			// If an exception is thrown, the test will just continue
			// We are only interested in knowing if the guest crashes
		}
	}
}
