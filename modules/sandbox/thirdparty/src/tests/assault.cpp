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