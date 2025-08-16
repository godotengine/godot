#include "api.hpp"

PUBLIC Variant test_math_sin(double val) {
	return Math::sin(val);
}
PUBLIC Variant test_math_cos(double val) {
	return Math::cos(val);
}
PUBLIC Variant test_math_tan(double val) {
	return Math::tan(val);
}
PUBLIC Variant test_math_asin(double val) {
	return Math::asin(val);
}
PUBLIC Variant test_math_acos(double val) {
	return Math::acos(val);
}
PUBLIC Variant test_math_atan(double val) {
	return Math::atan(val);
}
PUBLIC Variant test_math_atan2(double x, double y) {
	return Math::atan2(x, y);
}
PUBLIC Variant test_math_pow(double x, double y) {
	return Math::pow(x, y);
}

// NOTE: We can only call with 64-bit floats from GDScript
PUBLIC Variant test_math_sinf(double val) {
	return Math::sinf(val);
}

PUBLIC Variant test_math_lerp(double a, double b, double t) {
	return Math::lerp(a, b, t);
}
PUBLIC Variant test_math_smoothstep(double a, double b, double t) {
	return Math::smoothstep(a, b, t);
}
PUBLIC Variant test_math_clamp(double x, double a, double b) {
	return Math::clamp(x, a, b);
}
PUBLIC Variant test_math_slerp(double a, double b, double t) {
	return Math::slerp(a, b, t);
}
