/**************************************************************************/
/*  node3d.hpp                                                            */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Node3DGizmo;
class World3D;

class Node3D : public Node {
	GDEXTENSION_CLASS(Node3D, Node)

public:
	enum RotationEditMode {
		ROTATION_EDIT_MODE_EULER = 0,
		ROTATION_EDIT_MODE_QUATERNION = 1,
		ROTATION_EDIT_MODE_BASIS = 2,
	};

	static const int NOTIFICATION_TRANSFORM_CHANGED = 2000;
	static const int NOTIFICATION_ENTER_WORLD = 41;
	static const int NOTIFICATION_EXIT_WORLD = 42;
	static const int NOTIFICATION_VISIBILITY_CHANGED = 43;
	static const int NOTIFICATION_LOCAL_TRANSFORM_CHANGED = 44;

	void set_transform(const Transform3D &p_local);
	Transform3D get_transform() const;
	void set_position(const Vector3 &p_position);
	Vector3 get_position() const;
	void set_rotation(const Vector3 &p_euler_radians);
	Vector3 get_rotation() const;
	void set_rotation_degrees(const Vector3 &p_euler_degrees);
	Vector3 get_rotation_degrees() const;
	void set_rotation_order(EulerOrder p_order);
	EulerOrder get_rotation_order() const;
	void set_rotation_edit_mode(Node3D::RotationEditMode p_edit_mode);
	Node3D::RotationEditMode get_rotation_edit_mode() const;
	void set_scale(const Vector3 &p_scale);
	Vector3 get_scale() const;
	void set_quaternion(const Quaternion &p_quaternion);
	Quaternion get_quaternion() const;
	void set_basis(const Basis &p_basis);
	Basis get_basis() const;
	void set_global_transform(const Transform3D &p_global);
	Transform3D get_global_transform() const;
	Transform3D get_global_transform_interpolated();
	void set_global_position(const Vector3 &p_position);
	Vector3 get_global_position() const;
	void set_global_basis(const Basis &p_basis);
	Basis get_global_basis() const;
	void set_global_rotation(const Vector3 &p_euler_radians);
	Vector3 get_global_rotation() const;
	void set_global_rotation_degrees(const Vector3 &p_euler_degrees);
	Vector3 get_global_rotation_degrees() const;
	Node3D *get_parent_node_3d() const;
	void set_ignore_transform_notification(bool p_enabled);
	void set_as_top_level(bool p_enable);
	bool is_set_as_top_level() const;
	void set_disable_scale(bool p_disable);
	bool is_scale_disabled() const;
	Ref<World3D> get_world_3d() const;
	void force_update_transform();
	void set_visibility_parent(const NodePath &p_path);
	NodePath get_visibility_parent() const;
	void update_gizmos();
	void add_gizmo(const Ref<Node3DGizmo> &p_gizmo);
	TypedArray<Ref<Node3DGizmo>> get_gizmos() const;
	void clear_gizmos();
	void set_subgizmo_selection(const Ref<Node3DGizmo> &p_gizmo, int32_t p_id, const Transform3D &p_transform);
	void clear_subgizmo_selection();
	void set_visible(bool p_visible);
	bool is_visible() const;
	bool is_visible_in_tree() const;
	void show();
	void hide();
	void set_notify_local_transform(bool p_enable);
	bool is_local_transform_notification_enabled() const;
	void set_notify_transform(bool p_enable);
	bool is_transform_notification_enabled() const;
	void rotate(const Vector3 &p_axis, float p_angle);
	void global_rotate(const Vector3 &p_axis, float p_angle);
	void global_scale(const Vector3 &p_scale);
	void global_translate(const Vector3 &p_offset);
	void rotate_object_local(const Vector3 &p_axis, float p_angle);
	void scale_object_local(const Vector3 &p_scale);
	void translate_object_local(const Vector3 &p_offset);
	void rotate_x(float p_angle);
	void rotate_y(float p_angle);
	void rotate_z(float p_angle);
	void translate(const Vector3 &p_offset);
	void orthonormalize();
	void set_identity();
	void look_at(const Vector3 &p_target, const Vector3 &p_up = Vector3(0, 1, 0), bool p_use_model_front = false);
	void look_at_from_position(const Vector3 &p_position, const Vector3 &p_target, const Vector3 &p_up = Vector3(0, 1, 0), bool p_use_model_front = false);
	Vector3 to_local(const Vector3 &p_global_point) const;
	Vector3 to_global(const Vector3 &p_local_point) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Node3D::RotationEditMode);

