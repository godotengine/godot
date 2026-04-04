/**************************************************************************/
/*  gltf_node.hpp                                                         */
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
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class GLTFState;
class StringName;

class GLTFNode : public Resource {
	GDEXTENSION_CLASS(GLTFNode, Resource)

public:
	String get_original_name();
	void set_original_name(const String &p_original_name);
	int32_t get_parent();
	void set_parent(int32_t p_parent);
	int32_t get_height();
	void set_height(int32_t p_height);
	Transform3D get_xform();
	void set_xform(const Transform3D &p_xform);
	int32_t get_mesh();
	void set_mesh(int32_t p_mesh);
	int32_t get_camera();
	void set_camera(int32_t p_camera);
	int32_t get_skin();
	void set_skin(int32_t p_skin);
	int32_t get_skeleton();
	void set_skeleton(int32_t p_skeleton);
	Vector3 get_position();
	void set_position(const Vector3 &p_position);
	Quaternion get_rotation();
	void set_rotation(const Quaternion &p_rotation);
	Vector3 get_scale();
	void set_scale(const Vector3 &p_scale);
	PackedInt32Array get_children();
	void set_children(const PackedInt32Array &p_children);
	void append_child_index(int32_t p_child_index);
	int32_t get_light();
	void set_light(int32_t p_light);
	bool get_visible();
	void set_visible(bool p_visible);
	Variant get_additional_data(const StringName &p_extension_name);
	void set_additional_data(const StringName &p_extension_name, const Variant &p_additional_data);
	NodePath get_scene_node_path(const Ref<GLTFState> &p_gltf_state, bool p_handle_skeletons = true);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

