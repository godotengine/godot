#include "transform2d.hpp"

#include "syscalls.h"

MAKE_SYSCALL(ECALL_TRANSFORM_2D_OPS, void, sys_transform2d_ops, unsigned, Transform2D_Op, ...);

Transform2D::Transform2D(const Vector2 &x, const Vector2 &y, const Vector2 &origin) {
	sys_transform2d_ops(0, Transform2D_Op::CREATE, this, &x, &y, &origin);
}

Transform2D Transform2D::identity() {
	Transform2D t;
	sys_transform2d_ops(0, Transform2D_Op::IDENTITY, &t);
	return t;
}

void Transform2D::assign(const Transform2D &transform) {
	sys_transform2d_ops(this->m_idx, Transform2D_Op::ASSIGN, this, &transform);
}

Transform2D Transform2D::inverse() const {
	Transform2D t;
	sys_transform2d_ops(this->m_idx, Transform2D_Op::INVERTED, &t);
	return t;
}

void Transform2D::invert() {
	sys_transform2d_ops(this->m_idx, Transform2D_Op::INVERTED, this);
}

Transform2D Transform2D::orthonormalized() const {
	Transform2D t;
	sys_transform2d_ops(this->m_idx, Transform2D_Op::ORTHONORMALIZED, &t);
	return t;
}

void Transform2D::affine_invert() {
	sys_transform2d_ops(this->m_idx, Transform2D_Op::AFFINE_INVERTED, this);
}

Transform2D Transform2D::rotated(double angle) const {
	Transform2D t;
	sys_transform2d_ops(this->m_idx, Transform2D_Op::ROTATED, &t, angle);
	return t;
}

void Transform2D::rotate(double angle) {
	sys_transform2d_ops(this->m_idx, Transform2D_Op::ROTATED, this, angle);
}

Transform2D Transform2D::scaled(const Vector2 &scale) const {
	Transform2D t;
	sys_transform2d_ops(this->m_idx, Transform2D_Op::SCALED, &t, &scale);
	return t;
}

void Transform2D::scale(const Vector2 &scale) {
	sys_transform2d_ops(this->m_idx, Transform2D_Op::SCALED, this, &scale);
}

Transform2D Transform2D::translated(const Vector2 &offset) const {
	Transform2D t;
	sys_transform2d_ops(this->m_idx, Transform2D_Op::TRANSLATED, &t, &offset);
	return t;
}

void Transform2D::translate(const Vector2 &offset) {
	sys_transform2d_ops(this->m_idx, Transform2D_Op::TRANSLATED, this, &offset);
}

Transform2D Transform2D::interpolate_with(const Transform2D &p_transform, double weight) const {
	Transform2D t;
	sys_transform2d_ops(this->m_idx, Transform2D_Op::INTERPOLATE_WITH, &t, &p_transform, weight);
	return t;
}

void Transform2D::interpolate_with(const Transform2D &p_transform, double weight) {
	sys_transform2d_ops(this->m_idx, Transform2D_Op::INTERPOLATE_WITH, this, &p_transform, weight);
}

Vector2 Transform2D::get_column(int idx) const {
	Vector2 v;
	sys_transform2d_ops(this->m_idx, Transform2D_Op::GET_COLUMN, &v, idx);
	return v;
}

void Transform2D::set_column(int idx, const Vector2 &axis) {
	sys_transform2d_ops(this->m_idx, Transform2D_Op::SET_COLUMN, this, idx, &axis);
}
