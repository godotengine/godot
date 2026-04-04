/**************************************************************************/
/*  skeleton_ik3d.hpp                                                     */
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

#include <godot_cpp/classes/skeleton_modifier3d.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Skeleton3D;

class SkeletonIK3D : public SkeletonModifier3D {
	GDEXTENSION_CLASS(SkeletonIK3D, SkeletonModifier3D)

public:
	void set_root_bone(const StringName &p_root_bone);
	StringName get_root_bone() const;
	void set_tip_bone(const StringName &p_tip_bone);
	StringName get_tip_bone() const;
	void set_target_transform(const Transform3D &p_target);
	Transform3D get_target_transform() const;
	void set_target_node(const NodePath &p_node);
	NodePath get_target_node();
	void set_override_tip_basis(bool p_override);
	bool is_override_tip_basis() const;
	void set_use_magnet(bool p_use);
	bool is_using_magnet() const;
	void set_magnet_position(const Vector3 &p_local_position);
	Vector3 get_magnet_position() const;
	Skeleton3D *get_parent_skeleton() const;
	bool is_running();
	void set_min_distance(float p_min_distance);
	float get_min_distance() const;
	void set_max_iterations(int32_t p_iterations);
	int32_t get_max_iterations() const;
	void start(bool p_one_time = false);
	void stop();
	void set_interpolation(float p_interpolation);
	float get_interpolation() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SkeletonModifier3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

