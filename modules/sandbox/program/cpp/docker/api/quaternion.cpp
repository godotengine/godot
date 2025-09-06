/**************************************************************************/
/*  quaternion.cpp                                                        */
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

#include "quaternion.hpp"

#include "syscalls.h"

MAKE_SYSCALL(ECALL_QUAT_OPS, void, sys_quat_ops, unsigned idx, Quaternion_Op, ...);

Quaternion Quaternion::identity() {
	// CREATE uses idx for the constructor type
	Quaternion q;
	sys_quat_ops(0, Quaternion_Op::CREATE, &q);
	return q;
}

Quaternion::Quaternion(double p_x, double p_y, double p_z, double p_w) {
	sys_quat_ops(1, Quaternion_Op::CREATE, Vector3(p_x, p_y, p_z), p_w);
}

Quaternion::Quaternion(const Vector3 &axis, double angle) {
	sys_quat_ops(2, Quaternion_Op::CREATE, axis, angle);
}

Quaternion::Quaternion(const Vector3 &euler) {
	sys_quat_ops(3, Quaternion_Op::CREATE, euler);
}

void Quaternion::assign(const Quaternion &quat) {
	// ASSIGN uses idx for the source quaternion, and target quaternion is this
	sys_quat_ops(this->m_idx, Quaternion_Op::ASSIGN, this, quat.get_variant_index());
}

double Quaternion::dot(const Quaternion &q) const {
	double d;
	sys_quat_ops(this->m_idx, Quaternion_Op::DOT, q.get_variant_index(), &d);
	return d;
}

double Quaternion::length_squared() const {
	double d;
	sys_quat_ops(this->m_idx, Quaternion_Op::LENGTH_SQUARED, &d);
	return d;
}

double Quaternion::length() const {
	double d;
	sys_quat_ops(this->m_idx, Quaternion_Op::LENGTH, &d);
	return d;
}

void Quaternion::normalize() {
	sys_quat_ops(this->m_idx, Quaternion_Op::NORMALIZE, this);
}

Quaternion Quaternion::normalized() const {
	Quaternion q;
	sys_quat_ops(this->m_idx, Quaternion_Op::NORMALIZE, &q);
	return q;
}

bool Quaternion::is_normalized() const {
	double d;
	sys_quat_ops(this->m_idx, Quaternion_Op::LENGTH_SQUARED, &d);
	return d > 0.999999f && d < 1.000001f;
}

Quaternion Quaternion::inverse() const {
	Quaternion q;
	sys_quat_ops(this->m_idx, Quaternion_Op::INVERSE, &q);
	return q;
}

Quaternion Quaternion::log() const {
	Quaternion q;
	sys_quat_ops(this->m_idx, Quaternion_Op::LOG, &q);
	return q;
}

Quaternion Quaternion::exp() const {
	Quaternion q;
	sys_quat_ops(this->m_idx, Quaternion_Op::EXP, &q);
	return q;
}

double Quaternion::angle_to(const Quaternion &to) const {
	double d;
	sys_quat_ops(this->m_idx, Quaternion_Op::ANGLE_TO, to.get_variant_index(), &d);
	return d;
}

Quaternion Quaternion::slerp(const Quaternion &to, double t) const {
	Quaternion q;
	sys_quat_ops(this->m_idx, Quaternion_Op::SLERP, to.get_variant_index(), t, &q);
	return q;
}

Quaternion Quaternion::slerpni(const Quaternion &to, double t) const {
	Quaternion q;
	sys_quat_ops(this->m_idx, Quaternion_Op::SLERPNI, to.get_variant_index(), t, &q);
	return q;
}

Quaternion Quaternion::cubic_interpolate(const Quaternion &b, const Quaternion &pre_a, const Quaternion &post_b, double t) const {
	Quaternion q;
	sys_quat_ops(this->m_idx, Quaternion_Op::CUBIC_INTERPOLATE, b.get_variant_index(), pre_a.get_variant_index(), post_b.get_variant_index(), t, &q);
	return q;
}

Quaternion Quaternion::cubic_interpolate_in_time(const Quaternion &b, const Quaternion &pre_a, const Quaternion &post_b, double t, double b_t, double pre_a_t, double post_b_t) const {
	Quaternion q;
	sys_quat_ops(this->m_idx, Quaternion_Op::CUBIC_INTERPOLATE_IN_TIME, b.get_variant_index(), pre_a.get_variant_index(), post_b.get_variant_index(), t, b_t, pre_a_t, post_b_t, &q);
	return q;
}

Vector3 Quaternion::get_axis() const {
	Vector3 v;
	sys_quat_ops(this->m_idx, Quaternion_Op::GET_AXIS, &v);
	return v;
}

double Quaternion::get_angle() const {
	double d;
	sys_quat_ops(this->m_idx, Quaternion_Op::GET_ANGLE, &d);
	return d;
}

void Quaternion::operator*=(const Quaternion &q) {
	sys_quat_ops(this->m_idx, Quaternion_Op::MUL, this, q.get_variant_index());
}

Quaternion Quaternion::operator*(const Quaternion &q) const {
	Quaternion r;
	sys_quat_ops(this->m_idx, Quaternion_Op::MUL, &r, q.get_variant_index());
	return r;
}

double Quaternion::operator[](int idx) const {
	double d;
	sys_quat_ops(this->m_idx, Quaternion_Op::AT, idx, &d);
	return d;
}
