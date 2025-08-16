#pragma once

#include "function.hpp"
#include "variant.hpp"
struct Object;

struct CallbackTimer {
	using period_t = double;
	using TimerCallback = Function<Variant(Variant)>;
	using TimerNativeCallback = Function<Variant(Object)>;

	// For when all arguments are Variants
	static Variant oneshot(period_t secs, TimerCallback callback);

	static Variant periodic(period_t period, TimerCallback callback);

	// For when native/register-based arguments are enabled
	static Variant native_oneshot(period_t secs, TimerNativeCallback callback);

	static Variant native_periodic(period_t period, TimerNativeCallback callback);

private:
	static Variant create(period_t p, bool oneshot, TimerCallback callback);
	static Variant create_native(period_t p, bool oneshot, TimerNativeCallback callback);
};

inline Variant CallbackTimer::oneshot(period_t secs, TimerCallback callback) {
	return create(secs, true, callback);
}

inline Variant CallbackTimer::periodic(period_t period, TimerCallback callback) {
	return create(period, false, callback);
}

inline Variant CallbackTimer::native_oneshot(period_t secs, TimerNativeCallback callback) {
	return create_native(secs, true, callback);
}

inline Variant CallbackTimer::native_periodic(period_t period, TimerNativeCallback callback) {
	return create_native(period, false, callback);
}
