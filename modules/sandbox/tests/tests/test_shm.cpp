/**************************************************************************/
/*  test_shm.cpp                                                          */
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

#include "api.hpp"

PUBLIC Variant test_shm(float *array, size_t size) {
	// This function is a placeholder for shared memory operations.
	// It assumes that the array is already allocated in shared memory.
	if (array == nullptr || size == 0) {
		return Nil;
	}

	for (size_t i = 0; i < size; ++i) {
		array[i] *= 2.0f; // Example operation: double each element
	}

	return PackedArray<float>(array, size);
}

PUBLIC Variant test_shm2(float *array, size_t size) {
	if (array == nullptr || size == 0) {
		return Nil;
	}

	for (size_t i = 0; i < 5; ++i) {
		array[i] = (1.0f + i) * 2.0f; // Example operation: double each element
	}

	return Nil;
}

PUBLIC Variant verify_shm2(float *array, size_t size) {
	if (array == nullptr || size < 5) {
		return false;
	}

	for (size_t i = 0; i < 5; ++i) {
		if (array[i] != (1.0f + i) * 2.0f) {
			return false; // Verification failed
		}
	}

	return true; // Verification succeeded
}
