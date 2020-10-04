#ifndef GODOT_MATH_H
#define GODOT_MATH_H

#include "Defs.hpp"
#include <cmath>

namespace godot {
namespace Math {

// Functions reproduced as in Godot's source code `math_funcs.h`.
// Some are overloads to automatically support changing real_t into either double or float in the way Godot does.

inline double fmod(double p_x, double p_y) {
	return ::fmod(p_x, p_y);
}
inline float fmod(float p_x, float p_y) {
	return ::fmodf(p_x, p_y);
}

inline double floor(double p_x) {
	return ::floor(p_x);
}
inline float floor(float p_x) {
	return ::floorf(p_x);
}

inline double exp(double p_x) {
	return ::exp(p_x);
}
inline float exp(float p_x) {
	return ::expf(p_x);
}

inline double sin(double p_x) {
	return ::sin(p_x);
}
inline float sin(float p_x) {
	return ::sinf(p_x);
}

inline double cos(double p_x) {
	return ::cos(p_x);
}
inline float cos(float p_x) {
	return ::cosf(p_x);
}

inline double tan(double p_x) {
	return ::tan(p_x);
}
inline float tan(float p_x) {
	return ::tanf(p_x);
}

inline double atan2(double p_y, double p_x) {
	return ::atan2(p_y, p_x);
}
inline float atan2(float p_y, float p_x) {
	return ::atan2f(p_y, p_x);
}

inline double sqrt(double p_x) {
	return ::sqrt(p_x);
}
inline float sqrt(float p_x) {
	return ::sqrtf(p_x);
}

inline float lerp(float minv, float maxv, float t) {
	return minv + t * (maxv - minv);
}
inline double lerp(double minv, double maxv, double t) {
	return minv + t * (maxv - minv);
}

inline double lerp_angle(double p_from, double p_to, double p_weight) {
	double difference = fmod(p_to - p_from, Math_TAU);
	double distance = fmod(2.0 * difference, Math_TAU) - difference;
	return p_from + distance * p_weight;
}
inline float lerp_angle(float p_from, float p_to, float p_weight) {
	float difference = fmod(p_to - p_from, (float)Math_TAU);
	float distance = fmod(2.0f * difference, (float)Math_TAU) - difference;
	return p_from + distance * p_weight;
}

template <typename T>
inline T clamp(T x, T minv, T maxv) {
	if (x < minv) {
		return minv;
	}
	if (x > maxv) {
		return maxv;
	}
	return x;
}

template <typename T>
inline T min(T a, T b) {
	return a < b ? a : b;
}

template <typename T>
inline T max(T a, T b) {
	return a > b ? a : b;
}

template <typename T>
inline T sign(T x) {
	return x < 0 ? -1 : 1;
}

inline double deg2rad(double p_y) {
	return p_y * Math_PI / 180.0;
}
inline float deg2rad(float p_y) {
	return p_y * Math_PI / 180.0;
}

inline double rad2deg(double p_y) {
	return p_y * 180.0 / Math_PI;
}
inline float rad2deg(float p_y) {
	return p_y * 180.0 / Math_PI;
}

inline double inverse_lerp(double p_from, double p_to, double p_value) {
	return (p_value - p_from) / (p_to - p_from);
}
inline float inverse_lerp(float p_from, float p_to, float p_value) {
	return (p_value - p_from) / (p_to - p_from);
}

inline double range_lerp(double p_value, double p_istart, double p_istop, double p_ostart, double p_ostop) {
	return Math::lerp(p_ostart, p_ostop, Math::inverse_lerp(p_istart, p_istop, p_value));
}
inline float range_lerp(float p_value, float p_istart, float p_istop, float p_ostart, float p_ostop) {
	return Math::lerp(p_ostart, p_ostop, Math::inverse_lerp(p_istart, p_istop, p_value));
}

inline bool is_equal_approx(real_t a, real_t b) {
	// Check for exact equality first, required to handle "infinity" values.
	if (a == b) {
		return true;
	}
	// Then check for approximate equality.
	real_t tolerance = CMP_EPSILON * std::abs(a);
	if (tolerance < CMP_EPSILON) {
		tolerance = CMP_EPSILON;
	}
	return std::abs(a - b) < tolerance;
}

inline bool is_equal_approx(real_t a, real_t b, real_t tolerance) {
	// Check for exact equality first, required to handle "infinity" values.
	if (a == b) {
		return true;
	}
	// Then check for approximate equality.
	return std::abs(a - b) < tolerance;
}

inline bool is_zero_approx(real_t s) {
	return std::abs(s) < CMP_EPSILON;
}

inline double smoothstep(double p_from, double p_to, double p_weight) {
	if (is_equal_approx(p_from, p_to)) {
		return p_from;
	}
	double x = clamp((p_weight - p_from) / (p_to - p_from), 0.0, 1.0);
	return x * x * (3.0 - 2.0 * x);
}
inline float smoothstep(float p_from, float p_to, float p_weight) {
	if (is_equal_approx(p_from, p_to)) {
		return p_from;
	}
	float x = clamp((p_weight - p_from) / (p_to - p_from), 0.0f, 1.0f);
	return x * x * (3.0f - 2.0f * x);
}

inline double move_toward(double p_from, double p_to, double p_delta) {
	return std::abs(p_to - p_from) <= p_delta ? p_to : p_from + sign(p_to - p_from) * p_delta;
}

inline float move_toward(float p_from, float p_to, float p_delta) {
	return std::abs(p_to - p_from) <= p_delta ? p_to : p_from + sign(p_to - p_from) * p_delta;
}

inline double linear2db(double p_linear) {
	return log(p_linear) * 8.6858896380650365530225783783321;
}
inline float linear2db(float p_linear) {
	return log(p_linear) * 8.6858896380650365530225783783321f;
}

inline double db2linear(double p_db) {
	return exp(p_db * 0.11512925464970228420089957273422);
}
inline float db2linear(float p_db) {
	return exp(p_db * 0.11512925464970228420089957273422f);
}

inline double round(double p_val) {
	return (p_val >= 0) ? floor(p_val + 0.5) : -floor(-p_val + 0.5);
}
inline float round(float p_val) {
	return (p_val >= 0) ? floor(p_val + 0.5) : -floor(-p_val + 0.5);
}

inline int64_t wrapi(int64_t value, int64_t min, int64_t max) {
	int64_t range = max - min;
	return range == 0 ? min : min + ((((value - min) % range) + range) % range);
}

inline double wrapf(double value, double min, double max) {
	double range = max - min;
	return is_zero_approx(range) ? min : value - (range * floor((value - min) / range));
}
inline float wrapf(float value, float min, float max) {
	float range = max - min;
	return is_zero_approx(range) ? min : value - (range * floor((value - min) / range));
}

inline real_t stepify(real_t p_value, real_t p_step) {
	if (p_step != 0) {
		p_value = floor(p_value / p_step + 0.5) * p_step;
	}
	return p_value;
}

inline unsigned int next_power_of_2(unsigned int x) {

	if (x == 0)
		return 0;

	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;

	return ++x;
}

} // namespace Math
} // namespace godot

#endif // GODOT_MATH_H
