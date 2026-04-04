/**************************************************************************/
/*  gltf_physics_shape.hpp                                                */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CollisionShape3D;
class ImporterMesh;
class Shape3D;

class GLTFPhysicsShape : public Resource {
	GDEXTENSION_CLASS(GLTFPhysicsShape, Resource)

public:
	static Ref<GLTFPhysicsShape> from_node(CollisionShape3D *p_shape_node);
	CollisionShape3D *to_node(bool p_cache_shapes = false);
	static Ref<GLTFPhysicsShape> from_resource(const Ref<Shape3D> &p_shape_resource);
	Ref<Shape3D> to_resource(bool p_cache_shapes = false);
	static Ref<GLTFPhysicsShape> from_dictionary(const Dictionary &p_dictionary);
	Dictionary to_dictionary() const;
	String get_shape_type() const;
	void set_shape_type(const String &p_shape_type);
	Vector3 get_size() const;
	void set_size(const Vector3 &p_size);
	float get_radius() const;
	void set_radius(float p_radius);
	float get_height() const;
	void set_height(float p_height);
	bool get_is_trigger() const;
	void set_is_trigger(bool p_is_trigger);
	int32_t get_mesh_index() const;
	void set_mesh_index(int32_t p_mesh_index);
	Ref<ImporterMesh> get_importer_mesh() const;
	void set_importer_mesh(const Ref<ImporterMesh> &p_importer_mesh);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

