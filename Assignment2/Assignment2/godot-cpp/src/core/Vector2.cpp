#include "Vector2.hpp"

#include <gdnative/vector2.h>

#include "String.hpp"

namespace godot {

bool Vector2::operator==(const Vector2 &p_vec2) const {
	return x == p_vec2.x && y == p_vec2.y;
}

bool Vector2::operator!=(const Vector2 &p_vec2) const {
	return x != p_vec2.x || y != p_vec2.y;
}

Vector2 Vector2::project(const Vector2 &p_vec) const {
	Vector2 v1 = p_vec;
	Vector2 v2 = *this;
	return v2 * (v1.dot(v2) / v2.dot(v2));
}

Vector2 Vector2::plane_project(real_t p_d, const Vector2 &p_vec) const {
	return p_vec - *this * (dot(p_vec) - p_d);
}

Vector2 Vector2::clamped(real_t p_len) const {
	real_t l = length();
	Vector2 v = *this;
	if (l > 0 && p_len < l) {
		v /= l;
		v *= p_len;
	}
	return v;
}

Vector2 Vector2::cubic_interpolate(const Vector2 &p_b, const Vector2 &p_pre_a, const Vector2 &p_post_b, real_t p_t) const {
	Vector2 p0 = p_pre_a;
	Vector2 p1 = *this;
	Vector2 p2 = p_b;
	Vector2 p3 = p_post_b;

	real_t t = p_t;
	real_t t2 = t * t;
	real_t t3 = t2 * t;

	Vector2 out;
	out = ((p1 * 2.0) +
				  (-p0 + p2) * t +
				  (p0 * 2.0 - p1 * 5.0 + p2 * 4 - p3) * t2 +
				  (-p0 + p1 * 3.0 - p2 * 3.0 + p3) * t3) *
		  0.5;

	return out;
}

Vector2::operator String() const {
	return String::num(x) + ", " + String::num(y);
}

} // namespace godot
