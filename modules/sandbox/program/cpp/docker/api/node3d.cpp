/**************************************************************************/
/*  node3d.cpp                                                            */
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

#include "node3d.hpp"

#include "quaternion.hpp"
#include "syscalls.h"
#include "transform3d.hpp"

// API call to get/set Node3D properties.
MAKE_SYSCALL(ECALL_NODE3D, void, sys_node3d, Node3D_Op, uint64_t, Variant *);
EXTERN_SYSCALL(void, sys_node, Node_Op, uint64_t, Variant *);
EXTERN_SYSCALL(uint64_t, sys_node_create, Node_Create_Shortlist, const char *, size_t, const char *, size_t);

static inline void node3d(Node3D_Op op, uint64_t address, const Variant &value) {
	sys_node3d(op, address, const_cast<Variant *>(&value));
}

Vector3 Node3D::get_position() const {
	Variant var;
	node3d(Node3D_Op::GET_POSITION, address(), var);
	return var.v3();
}

void Node3D::set_position(const Variant &value) {
	node3d(Node3D_Op::SET_POSITION, address(), value);
}

Vector3 Node3D::get_rotation() const {
	Variant var;
	node3d(Node3D_Op::GET_ROTATION, address(), var);
	return var.v3();
}

void Node3D::set_rotation(const Variant &value) {
	node3d(Node3D_Op::SET_ROTATION, address(), value);
}

Vector3 Node3D::get_scale() const {
	Variant var;
	node3d(Node3D_Op::GET_SCALE, address(), var);
	return var.v3();
}

void Node3D::set_scale(const Variant &value) {
	node3d(Node3D_Op::SET_SCALE, address(), value);
}

Node3D Node3D::duplicate(int flags) const {
	return Node::duplicate(flags);
}

Node3D Node3D::Create(std::string_view path) {
	return Node3D(sys_node_create(Node_Create_Shortlist::CREATE_NODE3D, nullptr, 0, path.data(), path.size()));
}

Transform3D Node3D::get_transform() const {
	Variant var;
	sys_node3d(Node3D_Op::GET_TRANSFORM, address(), &var);
	return var.as_transform3d();
}

void Node3D::set_transform(const Transform3D &value) {
	Variant var(value);
	sys_node3d(Node3D_Op::SET_TRANSFORM, address(), &var);
}

Quaternion Node3D::get_quaternion() const {
	Variant var;
	sys_node3d(Node3D_Op::GET_QUATERNION, address(), &var);
	return var;
}

void Node3D::set_quaternion(const Quaternion &value) {
	Variant var(value);
	sys_node3d(Node3D_Op::SET_QUATERNION, address(), &var);
}
