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
