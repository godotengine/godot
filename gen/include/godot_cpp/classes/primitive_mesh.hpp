/**************************************************************************/
/*  primitive_mesh.hpp                                                    */
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

#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Material;

class PrimitiveMesh : public Mesh {
	GDEXTENSION_CLASS(PrimitiveMesh, Mesh)

public:
	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;
	Array get_mesh_arrays() const;
	void set_custom_aabb(const AABB &p_aabb);
	AABB get_custom_aabb() const;
	void set_flip_faces(bool p_flip_faces);
	bool get_flip_faces() const;
	void set_add_uv2(bool p_add_uv2);
	bool get_add_uv2() const;
	void set_uv2_padding(float p_uv2_padding);
	float get_uv2_padding() const;
	void request_update();
	virtual Array _create_mesh_array() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Mesh::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_create_mesh_array), decltype(&T::_create_mesh_array)>) {
			BIND_VIRTUAL_METHOD(T, _create_mesh_array, 3995934104);
		}
	}

public:
};

} // namespace godot

