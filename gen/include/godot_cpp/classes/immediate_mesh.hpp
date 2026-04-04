/**************************************************************************/
/*  immediate_mesh.hpp                                                    */
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

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

struct Color;
struct Plane;
struct Vector2;
struct Vector3;

class ImmediateMesh : public Mesh {
	GDEXTENSION_CLASS(ImmediateMesh, Mesh)

public:
	void surface_begin(Mesh::PrimitiveType p_primitive, const Ref<Material> &p_material = nullptr);
	void surface_set_color(const Color &p_color);
	void surface_set_normal(const Vector3 &p_normal);
	void surface_set_tangent(const Plane &p_tangent);
	void surface_set_uv(const Vector2 &p_uv);
	void surface_set_uv2(const Vector2 &p_uv2);
	void surface_add_vertex(const Vector3 &p_vertex);
	void surface_add_vertex_2d(const Vector2 &p_vertex);
	void surface_end();
	void clear_surfaces();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Mesh::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

