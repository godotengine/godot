/**************************************************************************/
/*  physics_server3d_rendering_server_handler.hpp                         */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/core/object.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

struct AABB;
struct Vector3;

class PhysicsServer3DRenderingServerHandler : public Object {
	GDEXTENSION_CLASS(PhysicsServer3DRenderingServerHandler, Object)

public:
	void set_vertex(int32_t p_vertex_id, const Vector3 &p_vertex);
	void set_normal(int32_t p_vertex_id, const Vector3 &p_normal);
	void set_aabb(const AABB &p_aabb);
	virtual void _set_vertex(int32_t p_vertex_id, const Vector3 &p_vertex);
	virtual void _set_normal(int32_t p_vertex_id, const Vector3 &p_normal);
	virtual void _set_aabb(const AABB &p_aabb);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_set_vertex), decltype(&T::_set_vertex)>) {
			BIND_VIRTUAL_METHOD(T, _set_vertex, 1530502735);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_normal), decltype(&T::_set_normal)>) {
			BIND_VIRTUAL_METHOD(T, _set_normal, 1530502735);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_aabb), decltype(&T::_set_aabb)>) {
			BIND_VIRTUAL_METHOD(T, _set_aabb, 259215842);
		}
	}

public:
};

} // namespace godot

