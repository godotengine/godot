#include "node2d.hpp"

#include "syscalls.h"
#include "transform2d.hpp"

// API call to get/set Node2D properties.
MAKE_SYSCALL(ECALL_NODE2D, void, sys_node2d, Node2D_Op, uint64_t, Variant *);
EXTERN_SYSCALL(void, sys_node, Node_Op, uint64_t, Variant *);
EXTERN_SYSCALL(uint64_t, sys_node_create, Node_Create_Shortlist, const char *, size_t, const char *, size_t);

static inline void node2d(Node2D_Op op, uint64_t address, const Variant &value) {
	sys_node2d(op, address, const_cast<Variant *>(&value));
}

Vector2 Node2D::get_position() const {
	Variant var;
	node2d(Node2D_Op::GET_POSITION, address(), var);
	return var.v2();
}

void Node2D::set_position(const Vector2 &position) {
	Variant value(position);
	node2d(Node2D_Op::SET_POSITION, address(), value);
}

float Node2D::get_rotation() const {
	Variant var;
	node2d(Node2D_Op::GET_ROTATION, address(), var);
	return var.operator float();
}

void Node2D::set_rotation(real_t angle) {
	Variant value(angle);
	node2d(Node2D_Op::SET_ROTATION, address(), value);
}

Vector2 Node2D::get_scale() const {
	Variant var;
	node2d(Node2D_Op::GET_SCALE, address(), var);
	return var.v2();
}

void Node2D::set_scale(const Vector2 &scale) {
	Variant value(scale);
	node2d(Node2D_Op::SET_SCALE, address(), value);
}

float Node2D::get_skew() const {
	Variant var;
	node2d(Node2D_Op::GET_SKEW, address(), var);
	return var.operator float();
}

void Node2D::set_skew(const Variant &value) {
	node2d(Node2D_Op::SET_SKEW, address(), value);
}

void Node2D::set_transform(const Transform2D &value) {
	Variant var(value);
	node2d(Node2D_Op::SET_TRANSFORM, address(), var);
}

Transform2D Node2D::get_transform() const {
	Variant var;
	node2d(Node2D_Op::GET_TRANSFORM, address(), var);
	return var.as_transform2d();
}

Node2D Node2D::duplicate(int flags) const {
	return Node::duplicate(flags);
}

Node2D Node2D::Create(std::string_view path) {
	return Node2D(sys_node_create(Node_Create_Shortlist::CREATE_NODE2D, nullptr, 0, path.data(), path.size()));
}
