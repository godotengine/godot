/**************************************************************************/
/*  basis.cpp                                                             */
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

#include "basis.hpp"

#include "syscalls.h"

MAKE_SYSCALL(ECALL_BASIS_OPS, void, sys_basis_ops, unsigned idx, Basis_Op, ...);

Basis::Basis(const Vector3 &x, const Vector3 &y, const Vector3 &z) {
	sys_basis_ops(0, Basis_Op::CREATE, this, &x, &y, &z);
}

Basis Basis::identity() {
	Basis b;
	sys_basis_ops(0, Basis_Op::IDENTITY, &b);
	return b;
}

void Basis::assign(const Basis &basis) {
	sys_basis_ops(this->m_idx, Basis_Op::ASSIGN, this, basis.get_variant_index());
}

Vector3 Basis::get_row(int idx) const {
	Vector3 v;
	sys_basis_ops(this->m_idx, Basis_Op::GET_ROW, idx, &v);
	return v;
}

void Basis::set_row(int idx, const Vector3 &axis) {
	sys_basis_ops(this->m_idx, Basis_Op::SET_ROW, this, idx, &axis);
}

Vector3 Basis::get_column(int idx) const {
	Vector3 v;
	sys_basis_ops(this->m_idx, Basis_Op::GET_COLUMN, idx, &v);
	return v;
}

void Basis::set_column(int idx, const Vector3 &axis) {
	sys_basis_ops(this->m_idx, Basis_Op::SET_COLUMN, this, idx, &axis);
}

Basis Basis::inverse() const {
	Basis b;
	sys_basis_ops(this->m_idx, Basis_Op::INVERTED, &b);
	return b;
}

Basis Basis::transposed() const {
	Basis b;
	sys_basis_ops(this->m_idx, Basis_Op::TRANSPOSED, &b);
	return b;
}

double Basis::determinant() const {
	double det;
	sys_basis_ops(this->m_idx, Basis_Op::DETERMINANT, &det);
	return det;
}

Basis Basis::rotated(const Vector3 &axis, double angle) const {
	Basis b;
	sys_basis_ops(this->m_idx, Basis_Op::ROTATED, &b, &axis, angle);
	return b;
}

Basis Basis::lerp(const Basis &to, double t) const {
	Basis b;
	sys_basis_ops(this->m_idx, Basis_Op::LERP, &b, to.get_variant_index(), t);
	return b;
}

Basis Basis::slerp(const Basis &to, double t) const {
	Basis b;
	sys_basis_ops(this->m_idx, Basis_Op::SLERP, &b, to.get_variant_index(), t);
	return b;
}
