/**************************************************************************/
/*  physics_server_3d_rendering_server_handler.cpp                        */
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

#include "physics_server_3d_rendering_server_handler.h"

#include "core/math/aabb.h"
#include "core/math/vector3.h"
#include "core/object/class_db.h"

void PhysicsServer3DRenderingServerHandler::set_vertex(int p_vertex_id, const Vector3 &p_vertex) {
	GDVIRTUAL_CALL(_set_vertex, p_vertex_id, p_vertex);
}
void PhysicsServer3DRenderingServerHandler::set_normal(int p_vertex_id, const Vector3 &p_normal) {
	GDVIRTUAL_CALL(_set_normal, p_vertex_id, p_normal);
}
void PhysicsServer3DRenderingServerHandler::set_aabb(const AABB &p_aabb) {
	GDVIRTUAL_CALL(_set_aabb, p_aabb);
}

void PhysicsServer3DRenderingServerHandler::_bind_methods() {
	GDVIRTUAL_BIND(_set_vertex, "vertex_id", "vertex");
	GDVIRTUAL_BIND(_set_normal, "vertex_id", "normal");
	GDVIRTUAL_BIND(_set_aabb, "aabb");

	ClassDB::bind_method(D_METHOD("set_vertex", "vertex_id", "vertex"), &PhysicsServer3DRenderingServerHandler::set_vertex);
	ClassDB::bind_method(D_METHOD("set_normal", "vertex_id", "normal"), &PhysicsServer3DRenderingServerHandler::set_normal);
	ClassDB::bind_method(D_METHOD("set_aabb", "aabb"), &PhysicsServer3DRenderingServerHandler::set_aabb);
}
