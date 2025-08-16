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
