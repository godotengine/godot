/**************************************************************************/
/*  vector.cpp                                                            */
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

#include <vector> // std::hash

#include "syscalls.h"
#include "vector.hpp"

MAKE_SYSCALL(ECALL_VEC2_OPS, void, sys_vec2_ops, Vec2_Op op, Vector2 *v, ...); // NOLINT
MAKE_SYSCALL(ECALL_VEC3_OPS, void, sys_vec3_ops, Vector3 *v, Vector3 *other, Vec3_Op op); // NOLINT

Vector2 Vector2::limit_length(double length) const noexcept {
	Vector2 result = *this;
	sys_vec2_ops(Vec2_Op::LIMIT_LENGTH, &result, length);
	return result;
}

Vector2 Vector2::lerp(const Vector2 &to, double weight) const noexcept {
	Vector2 result = *this;
	sys_vec2_ops(Vec2_Op::LERP, &result, &to, weight);
	return result;
}

Vector2 Vector2::slerp(const Vector2 &to, double weight) const noexcept {
	Vector2 result = *this;
	sys_vec2_ops(Vec2_Op::SLERP, &result, &to, weight);
	return result;
}

Vector2 Vector2::cubic_interpolate(const Vector2 &b, const Vector2 &pre_a, const Vector2 &post_b, double weight) const noexcept {
	Vector2 result = *this;
	sys_vec2_ops(Vec2_Op::CUBIC_INTERPOLATE, &result, &b, &pre_a, &post_b, weight);
	return result;
}

Vector2 Vector2::slide(const Vector2 &normal) const noexcept {
	Vector2 result = *this;
	sys_vec2_ops(Vec2_Op::SLIDE, &result, &normal);
	return result;
}

Vector2 Vector2::bounce(const Vector2 &normal) const noexcept {
	Vector2 result = *this;
	sys_vec2_ops(Vec2_Op::BOUNCE, &result, &normal);
	return result;
}

Vector2 Vector2::reflect(const Vector2 &normal) const noexcept {
	Vector2 result = *this;
	sys_vec2_ops(Vec2_Op::REFLECT, &result, &normal);
	return result;
}

Vector2 Vector2::project(const Vector2 &by) const noexcept {
	Vector2 result = *this;
	sys_vec2_ops(Vec2_Op::PROJECT, &result, &by);
	return result;
}

float Vector3::length() const noexcept {
	register const Vector3 *vptr asm("a0") = this;
	register int op asm("a2") = int(Vec3_Op::LENGTH);
	register float length asm("fa0");
	register int syscall asm("a7") = ECALL_VEC3_OPS;

	__asm__ volatile("ecall"
			: "=f"(length)
			: "r"(op), "r"(vptr), "m"(*vptr), "r"(syscall));
	return length;
}

Vector3 Vector3::normalized() const noexcept {
	Vector3 result;

	register const Vector3 *vptr asm("a0") = this;
	register Vector3 *resptr asm("a1") = &result;
	register int op asm("a2") = int(Vec3_Op::NORMALIZE);
	register int syscall asm("a7") = ECALL_VEC3_OPS;

	__asm__ volatile("ecall"
			: "=m"(*resptr)
			: "r"(op), "r"(vptr), "m"(*vptr), "r"(resptr), "r"(syscall));
	return result;
}

float Vector3::dot(const Vector3 &other) const noexcept {
	register const Vector3 *vptr asm("a0") = this;
	register const Vector3 *otherptr asm("a1") = &other;
	register int op asm("a2") = int(Vec3_Op::DOT);
	register float dot asm("fa0");
	register int syscall asm("a7") = ECALL_VEC3_OPS;

	__asm__ volatile("ecall"
			: "=f"(dot)
			: "r"(op), "r"(vptr), "m"(*vptr), "r"(otherptr), "m"(*otherptr), "r"(syscall));
	return dot;
}

Vector3 Vector3::cross(const Vector3 &other) const noexcept {
	Vector3 result;

	register const Vector3 *vptr asm("a0") = this;
	register const Vector3 *otherptr asm("a1") = &other;
	register int op asm("a2") = int(Vec3_Op::CROSS);
	register Vector3 *resptr asm("a3") = &result;
	register int syscall asm("a7") = ECALL_VEC3_OPS;

	__asm__ volatile("ecall"
			: "=m"(*resptr)
			: "r"(op), "r"(vptr), "m"(*vptr), "r"(otherptr), "m"(*otherptr), "r"(resptr), "r"(syscall));
	return result;
}

float Vector3::distance_to(const Vector3 &other) const noexcept {
	register const Vector3 *vptr asm("a0") = this;
	register const Vector3 *otherptr asm("a1") = &other;
	register int op asm("a2") = int(Vec3_Op::DISTANCE_TO);
	register float distance asm("fa0");
	register int syscall asm("a7") = ECALL_VEC3_OPS;

	__asm__ volatile("ecall"
			: "=f"(distance)
			: "r"(op), "r"(vptr), "m"(*vptr), "r"(otherptr), "m"(*otherptr), "r"(syscall));
	return distance;
}

float Vector3::distance_squared_to(const Vector3 &other) const noexcept {
	register const Vector3 *vptr asm("a0") = this;
	register const Vector3 *otherptr asm("a1") = &other;
	register int op asm("a2") = int(Vec3_Op::DISTANCE_SQ_TO);
	register float distance asm("fa0");
	register int syscall asm("a7") = ECALL_VEC3_OPS;

	__asm__ volatile("ecall"
			: "=f"(distance)
			: "r"(op), "r"(vptr), "m"(*vptr), "r"(otherptr), "m"(*otherptr), "r"(syscall));
	return float(distance);
}

float Vector3::angle_to(const Vector3 &other) const noexcept {
	register const Vector3 *vptr asm("a0") = this;
	register const Vector3 *otherptr asm("a1") = &other;
	register int op asm("a2") = int(Vec3_Op::ANGLE_TO);
	register float angle asm("fa0");
	register int syscall asm("a7") = ECALL_VEC3_OPS;

	__asm__ volatile("ecall"
			: "=f"(angle)
			: "r"(op), "r"(vptr), "m"(*vptr), "r"(otherptr), "m"(*otherptr), "r"(syscall));
	return angle;
}

Vector3 Vector3::direction_to(const Vector3 &other) const noexcept {
	Vector3 ret(other.x - x, other.y - y, other.z - z);
	ret.normalize();
	return ret;
}

static_assert(sizeof(Vector3) == 12, "Vector3 size mismatch");
static_assert(alignof(Vector3) == 4, "Vector3 alignment mismatch");
