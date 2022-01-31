/*************************************************************************/
/*  node_3d.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef NODE_3D_H
#define NODE_3D_H

#include "scene/main/node.h"

class Node3DGizmo : public RefCounted {
	GDCLASS(Node3DGizmo, RefCounted);

public:
	virtual void create() = 0;
	virtual void transform() = 0;
	virtual void clear() = 0;
	virtual void redraw() = 0;
	virtual void free() = 0;

	Node3DGizmo();
	virtual ~Node3DGizmo() {}
};

class Node3D : public Node {
	GDCLASS(Node3D, Node);
	OBJ_CATEGORY("3D");

public:
	enum RotationEditMode {
		ROTATION_EDIT_MODE_EULER,
		ROTATION_EDIT_MODE_QUATERNION,
		ROTATION_EDIT_MODE_BASIS,
	};

	enum RotationOrder {
		ROTATION_ORDER_XYZ,
		ROTATION_ORDER_XZY,
		ROTATION_ORDER_YXZ,
		ROTATION_ORDER_YZX,
		ROTATION_ORDER_ZXY,
		ROTATION_ORDER_ZYX
	};

private:
	enum TransformDirty {
		DIRTY_NONE = 0,
		DIRTY_VECTORS = 1,
		DIRTY_LOCAL = 2,
		DIRTY_GLOBAL = 4
	};

	mutable SelfList<Node> xform_change;

	struct Data {
		mutable Transform3D global_transform;
		mutable Transform3D local_transform;
		mutable Basis::EulerOrder rotation_order = Basis::EULER_ORDER_YXZ;
		mutable Vector3 rotation;
		mutable Vector3 scale = Vector3(1, 1, 1);
		mutable RotationEditMode rotation_edit_mode = ROTATION_EDIT_MODE_EULER;

		mutable int dirty = DIRTY_NONE;

		Viewport *viewport = nullptr;

		bool top_level_active = false;
		bool top_level = false;
		bool inside_world = false;

		RID visibility_parent;

		int children_lock = 0;
		Node3D *parent = nullptr;
		List<Node3D *> children;
		List<Node3D *>::Element *C = nullptr;

		bool ignore_notification = false;
		bool notify_local_transform = false;
		bool notify_transform = false;

		bool visible = true;
		bool disable_scale = false;

#ifdef TOOLS_ENABLED
		Vector<Ref<Node3DGizmo>> gizmos;
		bool gizmos_disabled = false;
		bool gizmos_dirty = false;
		bool transform_gizmo_visible = true;
#endif

	} data;

	NodePath visibility_parent_path;

	void _update_gizmos();
	void _notify_dirty();
	void _propagate_transform_changed(Node3D *p_origin);

	void _propagate_visibility_changed();

	void _propagate_visibility_parent();
	void _update_visibility_parent(bool p_update_root);

protected:
	_FORCE_INLINE_ void set_ignore_transform_notification(bool p_ignore) { data.ignore_notification = p_ignore; }

	_FORCE_INLINE_ void _update_local_transform() const;

	void _notification(int p_what);
	static void _bind_methods();

	virtual void _validate_property(PropertyInfo &property) const override;

	bool property_can_revert(const String &p_name);
	Variant property_get_revert(const String &p_name);

public:
	enum {
		NOTIFICATION_TRANSFORM_CHANGED = SceneTree::NOTIFICATION_TRANSFORM_CHANGED,
		NOTIFICATION_ENTER_WORLD = 41,
		NOTIFICATION_EXIT_WORLD = 42,
		NOTIFICATION_VISIBILITY_CHANGED = 43,
		NOTIFICATION_LOCAL_TRANSFORM_CHANGED = 44,
	};

	Node3D *get_parent_node_3d() const;

	Ref<World3D> get_world_3d() const;

	void set_position(const Vector3 &p_position);

	void set_rotation_edit_mode(RotationEditMode p_mode);
	RotationEditMode get_rotation_edit_mode() const;

	void set_rotation_order(RotationOrder p_order);
	void set_rotation(const Vector3 &p_euler_rad);
	void set_scale(const Vector3 &p_scale);

	Vector3 get_position() const;

	RotationOrder get_rotation_order() const;
	Vector3 get_rotation() const;
	Vector3 get_scale() const;

	void set_transform(const Transform3D &p_transform);
	void set_basis(const Basis &p_basis);
	void set_quaternion(const Quaternion &p_quaternion);
	void set_global_transform(const Transform3D &p_transform);

	Transform3D get_transform() const;
	Basis get_basis() const;
	Quaternion get_quaternion() const;
	Transform3D get_global_transform() const;

#ifdef TOOLS_ENABLED
	virtual Transform3D get_global_gizmo_transform() const;
	virtual Transform3D get_local_gizmo_transform() const;
	virtual void set_transform_gizmo_visible(bool p_enabled) { data.transform_gizmo_visible = p_enabled; };
	virtual bool is_transform_gizmo_visible() const { return data.transform_gizmo_visible; };
#endif

	void set_disable_gizmos(bool p_enabled);
	void update_gizmos();
	void set_subgizmo_selection(Ref<Node3DGizmo> p_gizmo, int p_id, Transform3D p_transform = Transform3D());
	void clear_subgizmo_selection();
	Vector<Ref<Node3DGizmo>> get_gizmos() const;
	Array get_gizmos_bind() const;
	void add_gizmo(Ref<Node3DGizmo> p_gizmo);
	void remove_gizmo(Ref<Node3DGizmo> p_gizmo);
	void clear_gizmos();

	void set_as_top_level(bool p_enabled);
	bool is_set_as_top_level() const;

	void set_disable_scale(bool p_enabled);
	bool is_scale_disabled() const;

	_FORCE_INLINE_ bool is_inside_world() const { return data.inside_world; }

	Transform3D get_relative_transform(const Node *p_parent) const;

	void rotate(const Vector3 &p_axis, real_t p_angle);
	void rotate_x(real_t p_angle);
	void rotate_y(real_t p_angle);
	void rotate_z(real_t p_angle);
	void translate(const Vector3 &p_offset);
	void scale(const Vector3 &p_ratio);

	void rotate_object_local(const Vector3 &p_axis, real_t p_angle);
	void scale_object_local(const Vector3 &p_scale);
	void translate_object_local(const Vector3 &p_offset);

	void global_rotate(const Vector3 &p_axis, real_t p_angle);
	void global_scale(const Vector3 &p_scale);
	void global_translate(const Vector3 &p_offset);

	void look_at(const Vector3 &p_target, const Vector3 &p_up = Vector3(0, 1, 0));
	void look_at_from_position(const Vector3 &p_pos, const Vector3 &p_target, const Vector3 &p_up = Vector3(0, 1, 0));

	Vector3 to_local(Vector3 p_global) const;
	Vector3 to_global(Vector3 p_local) const;

	void set_notify_transform(bool p_enabled);
	bool is_transform_notification_enabled() const;

	void set_notify_local_transform(bool p_enabled);
	bool is_local_transform_notification_enabled() const;

	void orthonormalize();
	void set_identity();

	void set_visible(bool p_visible);
	void show();
	void hide();
	bool is_visible() const;
	bool is_visible_in_tree() const;

	void force_update_transform();

	void set_visibility_parent(const NodePath &p_path);
	NodePath get_visibility_parent() const;

	Node3D();
};

VARIANT_ENUM_CAST(Node3D::RotationEditMode)
VARIANT_ENUM_CAST(Node3D::RotationOrder)

#endif // NODE_3D_H
