/**************************************************************************/
/*  gltf_skin.hpp                                                         */
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
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Skin;

class GLTFSkin : public Resource {
	GDEXTENSION_CLASS(GLTFSkin, Resource)

public:
	int32_t get_skin_root();
	void set_skin_root(int32_t p_skin_root);
	PackedInt32Array get_joints_original();
	void set_joints_original(const PackedInt32Array &p_joints_original);
	TypedArray<Transform3D> get_inverse_binds();
	void set_inverse_binds(const TypedArray<Transform3D> &p_inverse_binds);
	PackedInt32Array get_joints();
	void set_joints(const PackedInt32Array &p_joints);
	PackedInt32Array get_non_joints();
	void set_non_joints(const PackedInt32Array &p_non_joints);
	PackedInt32Array get_roots();
	void set_roots(const PackedInt32Array &p_roots);
	int32_t get_skeleton();
	void set_skeleton(int32_t p_skeleton);
	Dictionary get_joint_i_to_bone_i();
	void set_joint_i_to_bone_i(const Dictionary &p_joint_i_to_bone_i);
	Dictionary get_joint_i_to_name();
	void set_joint_i_to_name(const Dictionary &p_joint_i_to_name);
	Ref<Skin> get_godot_skin();
	void set_godot_skin(const Ref<Skin> &p_godot_skin);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

