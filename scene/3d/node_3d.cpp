/*************************************************************************/
/*  node_3d.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "node_3d.h"

#include "core/object/message_queue.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/main/viewport.h"
#include "scene/scene_string_names.h"

/*

 possible algorithms:

 Algorithm 1: (current)

 definition of invalidation: global is invalid

 1) If a node sets a LOCAL, it produces an invalidation of everything above
 .  a) If above is invalid, don't keep invalidating upwards
 2) If a node sets a GLOBAL, it is converted to LOCAL (and forces validation of everything pending below)

 drawback: setting/reading globals is useful and used very often, and using affine inverses is slow

---

 Algorithm 2: (no longer current)

 definition of invalidation: NONE dirty, LOCAL dirty, GLOBAL dirty

 1) If a node sets a LOCAL, it must climb the tree and set it as GLOBAL dirty
 .  a) marking GLOBALs as dirty up all the tree must be done always
 2) If a node sets a GLOBAL, it marks local as dirty, and that's all?

 //is clearing the dirty state correct in this case?

 drawback: setting a local down the tree forces many tree walks often

--

future: no idea

 */

Node3DGizmo::Node3DGizmo() {
}

void Node3D::_notify_dirty() {
#ifdef TOOLS_ENABLED
	if ((!data.gizmos.is_empty() || data.notify_transform) && !data.ignore_notification && !xform_change.in_list()) {
#else
	if (data.notify_transform && !data.ignore_notification && !xform_change.in_list()) {

#endif
		get_tree()->xform_change_list.add(&xform_change);
	}
}

void Node3D::_update_local_transform() const {
	if (this->get_rotation_edit_mode() != ROTATION_EDIT_MODE_BASIS) {
		data.local_transform = data.local_transform.orthogonalized();
	}
	data.local_transform.basis.set_euler_scale(data.rotation, data.scale);

	data.dirty &= ~DIRTY_LOCAL;
}

void Node3D::_propagate_transform_changed(Node3D *p_origin) {
	if (!is_inside_tree()) {
		return;
	}

	data.children_lock++;

	for (Node3D *&E : data.children) {
		if (E->data.top_level_active) {
			continue; //don't propagate to a top_level
		}
		E->_propagate_transform_changed(p_origin);
	}
#ifdef TOOLS_ENABLED
	if ((!data.gizmos.is_empty() || data.notify_transform) && !data.ignore_notification && !xform_change.in_list()) {
#else
	if (data.notify_transform && !data.ignore_notification && !xform_change.in_list()) {
#endif
		get_tree()->xform_change_list.add(&xform_change);
	}
	data.dirty |= DIRTY_GLOBAL;

	data.children_lock--;
}

void Node3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			ERR_FAIL_COND(!get_tree());

			Node *p = get_parent();
			if (p) {
				data.parent = Object::cast_to<Node3D>(p);
			}

			if (data.parent) {
				data.C = data.parent->data.children.push_back(this);
			} else {
				data.C = nullptr;
			}

			if (data.top_level && !Engine::get_singleton()->is_editor_hint()) {
				if (data.parent) {
					data.local_transform = data.parent->get_global_transform() * get_transform();
					data.dirty = DIRTY_VECTORS; //global is always dirty upon entering a scene
				}
				data.top_level_active = true;
			}

			data.dirty |= DIRTY_GLOBAL; //global is always dirty upon entering a scene
			_notify_dirty();

			notification(NOTIFICATION_ENTER_WORLD);
			_update_visibility_parent(true);

		} break;
		case NOTIFICATION_EXIT_TREE: {
			notification(NOTIFICATION_EXIT_WORLD, true);
			if (xform_change.in_list()) {
				get_tree()->xform_change_list.remove(&xform_change);
			}
			if (data.C) {
				data.parent->data.children.erase(data.C);
			}
			data.parent = nullptr;
			data.C = nullptr;
			data.top_level_active = false;
			_update_visibility_parent(true);
		} break;
		case NOTIFICATION_ENTER_WORLD: {
			data.inside_world = true;
			data.viewport = nullptr;
			Node *parent = get_parent();
			while (parent && !data.viewport) {
				data.viewport = Object::cast_to<Viewport>(parent);
				parent = parent->get_parent();
			}

			ERR_FAIL_COND(!data.viewport);

			if (get_script_instance()) {
				get_script_instance()->call(SceneStringNames::get_singleton()->_enter_world);
			}
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint() && get_tree()->is_node_being_edited(this)) {
				get_tree()->call_group_flags(0, SceneStringNames::get_singleton()->_spatial_editor_group, SceneStringNames::get_singleton()->_request_gizmo, this);
				if (!data.gizmos_disabled) {
					for (int i = 0; i < data.gizmos.size(); i++) {
						data.gizmos.write[i]->create();
						if (is_visible_in_tree()) {
							data.gizmos.write[i]->redraw();
						}
						data.gizmos.write[i]->transform();
					}
				}
			}
#endif

		} break;
		case NOTIFICATION_EXIT_WORLD: {
#ifdef TOOLS_ENABLED
			clear_gizmos();
#endif

			if (get_script_instance()) {
				get_script_instance()->call(SceneStringNames::get_singleton()->_exit_world);
			}

			data.viewport = nullptr;
			data.inside_world = false;

		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
#ifdef TOOLS_ENABLED
			for (int i = 0; i < data.gizmos.size(); i++) {
				data.gizmos.write[i]->transform();
			}
#endif
		} break;

		default: {
		}
	}
}

void Node3D::set_basis(const Basis &p_basis) {
	set_transform(Transform3D(p_basis, data.local_transform.origin));
}
void Node3D::set_quaternion(const Quaternion &p_quaternion) {
	set_transform(Transform3D(Basis(p_quaternion), data.local_transform.origin));
}

void Node3D::set_transform(const Transform3D &p_transform) {
	data.local_transform = p_transform;
	data.dirty |= DIRTY_VECTORS;
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

Basis Node3D::get_basis() const {
	return get_transform().basis;
}
Quaternion Node3D::get_quaternion() const {
	return Quaternion(get_transform().basis);
}

void Node3D::set_global_transform(const Transform3D &p_transform) {
	Transform3D xform = (data.parent && !data.top_level_active)
			? data.parent->get_global_transform().affine_inverse() * p_transform
			: p_transform;

	set_transform(xform);
}

Transform3D Node3D::get_transform() const {
	if (data.dirty & DIRTY_LOCAL) {
		_update_local_transform();
	}

	return data.local_transform;
}
Transform3D Node3D::get_global_transform() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Transform3D());

	if (data.dirty & DIRTY_GLOBAL) {
		if (data.dirty & DIRTY_LOCAL) {
			_update_local_transform();
		}

		if (data.parent && !data.top_level_active) {
			data.global_transform = data.parent->get_global_transform() * data.local_transform;
		} else {
			data.global_transform = data.local_transform;
		}

		if (data.disable_scale) {
			data.global_transform.basis.orthonormalize();
		}

		data.dirty &= ~DIRTY_GLOBAL;
	}

	return data.global_transform;
}

#ifdef TOOLS_ENABLED
Transform3D Node3D::get_global_gizmo_transform() const {
	return get_global_transform();
}

Transform3D Node3D::get_local_gizmo_transform() const {
	return get_transform();
}
#endif

Node3D *Node3D::get_parent_node_3d() const {
	if (data.top_level) {
		return nullptr;
	}

	return Object::cast_to<Node3D>(get_parent());
}

Transform3D Node3D::get_relative_transform(const Node *p_parent) const {
	if (p_parent == this)
		return Transform3D();

	ERR_FAIL_COND_V(!data.parent, Transform3D());

	if (p_parent == data.parent) {
		return get_transform();
	} else {
		return data.parent->get_relative_transform(p_parent) * get_transform();
	}
}

void Node3D::set_position(const Vector3 &p_position) {
	data.local_transform.origin = p_position;
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

void Node3D::set_rotation_edit_mode(RotationEditMode p_mode) {
	if (data.rotation_edit_mode == p_mode) {
		return;
	}
	data.rotation_edit_mode = p_mode;

	// Shearing is not allowed except in ROTATION_EDIT_MODE_BASIS.
	data.dirty |= DIRTY_LOCAL;
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}

	notify_property_list_changed();
}

Node3D::RotationEditMode Node3D::get_rotation_edit_mode() const {
	return data.rotation_edit_mode;
}

void Node3D::set_rotation_order(RotationOrder p_order) {
	Basis::EulerOrder order = Basis::EulerOrder(p_order);

	if (data.rotation_order == order) {
		return;
	}

	ERR_FAIL_INDEX(int32_t(order), 6);

	if (data.dirty & DIRTY_VECTORS) {
		data.rotation = data.local_transform.basis.get_euler_normalized(order);
		data.scale = data.local_transform.basis.get_scale();
		data.dirty &= ~DIRTY_VECTORS;
	} else {
		data.rotation = Basis::from_euler(data.rotation, data.rotation_order).get_euler_normalized(order);
	}

	data.rotation_order = order;
	//changing rotation order should not affect transform

	notify_property_list_changed(); //will change rotation
}

Node3D::RotationOrder Node3D::get_rotation_order() const {
	return RotationOrder(data.rotation_order);
}

void Node3D::set_rotation(const Vector3 &p_euler_rad) {
	if (data.dirty & DIRTY_VECTORS) {
		data.scale = data.local_transform.basis.get_scale();
		data.dirty &= ~DIRTY_VECTORS;
	}

	data.rotation = p_euler_rad;
	data.dirty |= DIRTY_LOCAL;
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

void Node3D::set_scale(const Vector3 &p_scale) {
	if (data.dirty & DIRTY_VECTORS) {
		data.rotation = data.local_transform.basis.get_euler_normalized(data.rotation_order);
		data.dirty &= ~DIRTY_VECTORS;
	}

	data.scale = p_scale;
	data.dirty |= DIRTY_LOCAL;
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

Vector3 Node3D::get_position() const {
	return data.local_transform.origin;
}

Vector3 Node3D::get_rotation() const {
	if (data.dirty & DIRTY_VECTORS) {
		data.scale = data.local_transform.basis.get_scale();
		data.rotation = data.local_transform.basis.get_euler_normalized(data.rotation_order);

		data.dirty &= ~DIRTY_VECTORS;
	}

	return data.rotation;
}

Vector3 Node3D::get_scale() const {
	if (data.dirty & DIRTY_VECTORS) {
		data.scale = data.local_transform.basis.get_scale();
		data.rotation = data.local_transform.basis.get_euler_normalized(data.rotation_order);

		data.dirty &= ~DIRTY_VECTORS;
	}

	return data.scale;
}

void Node3D::update_gizmos() {
#ifdef TOOLS_ENABLED
	if (!is_inside_world()) {
		return;
	}

	if (data.gizmos.is_empty()) {
		return;
	}
	if (data.gizmos_dirty) {
		return;
	}
	data.gizmos_dirty = true;
	MessageQueue::get_singleton()->push_callable(callable_mp(this, &Node3D::_update_gizmos));
#endif
}

void Node3D::set_subgizmo_selection(Ref<Node3DGizmo> p_gizmo, int p_id, Transform3D p_transform) {
#ifdef TOOLS_ENABLED
	if (!is_inside_world()) {
		return;
	}

	if (Engine::get_singleton()->is_editor_hint() && get_tree()->is_node_being_edited(this)) {
		get_tree()->call_group_flags(0, SceneStringNames::get_singleton()->_spatial_editor_group, SceneStringNames::get_singleton()->_set_subgizmo_selection, this, p_gizmo, p_id, p_transform);
	}
#endif
}

void Node3D::clear_subgizmo_selection() {
#ifdef TOOLS_ENABLED
	if (!is_inside_world()) {
		return;
	}

	if (data.gizmos.is_empty()) {
		return;
	}

	if (Engine::get_singleton()->is_editor_hint() && get_tree()->is_node_being_edited(this)) {
		get_tree()->call_group_flags(0, SceneStringNames::get_singleton()->_spatial_editor_group, SceneStringNames::get_singleton()->_clear_subgizmo_selection, this);
	}
#endif
}

void Node3D::add_gizmo(Ref<Node3DGizmo> p_gizmo) {
#ifdef TOOLS_ENABLED

	if (data.gizmos_disabled || p_gizmo.is_null()) {
		return;
	}
	data.gizmos.push_back(p_gizmo);

	if (p_gizmo.is_valid() && is_inside_world()) {
		p_gizmo->create();
		if (is_visible_in_tree()) {
			p_gizmo->redraw();
		}
		p_gizmo->transform();
	}
#endif
}

void Node3D::remove_gizmo(Ref<Node3DGizmo> p_gizmo) {
#ifdef TOOLS_ENABLED

	int idx = data.gizmos.find(p_gizmo);
	if (idx != -1) {
		p_gizmo->free();
		data.gizmos.remove(idx);
	}
#endif
}

void Node3D::clear_gizmos() {
#ifdef TOOLS_ENABLED
	for (int i = 0; i < data.gizmos.size(); i++) {
		data.gizmos.write[i]->free();
	}
	data.gizmos.clear();
#endif
}

Array Node3D::get_gizmos_bind() const {
	Array ret;

#ifdef TOOLS_ENABLED
	for (int i = 0; i < data.gizmos.size(); i++) {
		ret.push_back(Variant(data.gizmos[i].ptr()));
	}
#endif

	return ret;
}

Vector<Ref<Node3DGizmo>> Node3D::get_gizmos() const {
#ifdef TOOLS_ENABLED

	return data.gizmos;
#else

	return Vector<Ref<Node3DGizmo>>();
#endif
}

void Node3D::_update_gizmos() {
#ifdef TOOLS_ENABLED
	if (data.gizmos_disabled || !is_inside_world() || !data.gizmos_dirty) {
		data.gizmos_dirty = false;
		return;
	}
	data.gizmos_dirty = false;
	for (int i = 0; i < data.gizmos.size(); i++) {
		if (is_visible_in_tree()) {
			data.gizmos.write[i]->redraw();
		} else {
			data.gizmos.write[i]->clear();
		}
	}
#endif
}

void Node3D::set_disable_gizmos(bool p_enabled) {
#ifdef TOOLS_ENABLED
	data.gizmos_disabled = p_enabled;
	if (!p_enabled) {
		clear_gizmos();
	}
#endif
}

void Node3D::set_disable_scale(bool p_enabled) {
	data.disable_scale = p_enabled;
}

bool Node3D::is_scale_disabled() const {
	return data.disable_scale;
}

void Node3D::set_as_top_level(bool p_enabled) {
	if (data.top_level == p_enabled) {
		return;
	}
	if (is_inside_tree() && !Engine::get_singleton()->is_editor_hint()) {
		if (p_enabled) {
			set_transform(get_global_transform());
		} else if (data.parent) {
			set_transform(data.parent->get_global_transform().affine_inverse() * get_global_transform());
		}

		data.top_level = p_enabled;
		data.top_level_active = p_enabled;

	} else {
		data.top_level = p_enabled;
	}
}

bool Node3D::is_set_as_top_level() const {
	return data.top_level;
}

Ref<World3D> Node3D::get_world_3d() const {
	ERR_FAIL_COND_V(!is_inside_world(), Ref<World3D>());
	ERR_FAIL_COND_V(!data.viewport, Ref<World3D>());

	return data.viewport->find_world_3d();
}

void Node3D::_propagate_visibility_changed() {
	notification(NOTIFICATION_VISIBILITY_CHANGED);
	emit_signal(SceneStringNames::get_singleton()->visibility_changed);
#ifdef TOOLS_ENABLED
	if (!data.gizmos.is_empty()) {
		data.gizmos_dirty = true;
		_update_gizmos();
	}
#endif

	for (Node3D *c : data.children) {
		if (!c || !c->data.visible) {
			continue;
		}
		c->_propagate_visibility_changed();
	}
}

void Node3D::show() {
	if (data.visible) {
		return;
	}

	data.visible = true;

	if (!is_inside_tree()) {
		return;
	}

	_propagate_visibility_changed();
}

void Node3D::hide() {
	if (!data.visible) {
		return;
	}

	data.visible = false;

	if (!is_inside_tree()) {
		return;
	}

	_propagate_visibility_changed();
}

bool Node3D::is_visible_in_tree() const {
	const Node3D *s = this;

	while (s) {
		if (!s->data.visible) {
			return false;
		}
		s = s->data.parent;
	}

	return true;
}

void Node3D::set_visible(bool p_visible) {
	if (p_visible) {
		show();
	} else {
		hide();
	}
}

bool Node3D::is_visible() const {
	return data.visible;
}

void Node3D::rotate_object_local(const Vector3 &p_axis, real_t p_angle) {
	Transform3D t = get_transform();
	t.basis.rotate_local(p_axis, p_angle);
	set_transform(t);
}

void Node3D::rotate(const Vector3 &p_axis, real_t p_angle) {
	Transform3D t = get_transform();
	t.basis.rotate(p_axis, p_angle);
	set_transform(t);
}

void Node3D::rotate_x(real_t p_angle) {
	Transform3D t = get_transform();
	t.basis.rotate(Vector3(1, 0, 0), p_angle);
	set_transform(t);
}

void Node3D::rotate_y(real_t p_angle) {
	Transform3D t = get_transform();
	t.basis.rotate(Vector3(0, 1, 0), p_angle);
	set_transform(t);
}

void Node3D::rotate_z(real_t p_angle) {
	Transform3D t = get_transform();
	t.basis.rotate(Vector3(0, 0, 1), p_angle);
	set_transform(t);
}

void Node3D::translate(const Vector3 &p_offset) {
	Transform3D t = get_transform();
	t.translate(p_offset);
	set_transform(t);
}

void Node3D::translate_object_local(const Vector3 &p_offset) {
	Transform3D t = get_transform();

	Transform3D s;
	s.translate(p_offset);
	set_transform(t * s);
}

void Node3D::scale(const Vector3 &p_ratio) {
	Transform3D t = get_transform();
	t.basis.scale(p_ratio);
	set_transform(t);
}

void Node3D::scale_object_local(const Vector3 &p_scale) {
	Transform3D t = get_transform();
	t.basis.scale_local(p_scale);
	set_transform(t);
}

void Node3D::global_rotate(const Vector3 &p_axis, real_t p_angle) {
	Transform3D t = get_global_transform();
	t.basis.rotate(p_axis, p_angle);
	set_global_transform(t);
}

void Node3D::global_scale(const Vector3 &p_scale) {
	Transform3D t = get_global_transform();
	t.basis.scale(p_scale);
	set_global_transform(t);
}

void Node3D::global_translate(const Vector3 &p_offset) {
	Transform3D t = get_global_transform();
	t.origin += p_offset;
	set_global_transform(t);
}

void Node3D::orthonormalize() {
	Transform3D t = get_transform();
	t.orthonormalize();
	set_transform(t);
}

void Node3D::set_identity() {
	set_transform(Transform3D());
}

void Node3D::look_at(const Vector3 &p_target, const Vector3 &p_up) {
	Vector3 origin = get_global_transform().origin;
	look_at_from_position(origin, p_target, p_up);
}

void Node3D::look_at_from_position(const Vector3 &p_pos, const Vector3 &p_target, const Vector3 &p_up) {
	ERR_FAIL_COND_MSG(p_pos.is_equal_approx(p_target), "Node origin and target are in the same position, look_at() failed.");
	ERR_FAIL_COND_MSG(p_up.is_equal_approx(Vector3()), "The up vector can't be zero, look_at() failed.");
	ERR_FAIL_COND_MSG(p_up.cross(p_target - p_pos).is_equal_approx(Vector3()), "Up vector and direction between node origin and target are aligned, look_at() failed.");

	Transform3D lookat = Transform3D(Basis::looking_at(p_target - p_pos, p_up), p_pos);
	Vector3 original_scale = get_scale();
	set_global_transform(lookat);
	set_scale(original_scale);
}

Vector3 Node3D::to_local(Vector3 p_global) const {
	return get_global_transform().affine_inverse().xform(p_global);
}

Vector3 Node3D::to_global(Vector3 p_local) const {
	return get_global_transform().xform(p_local);
}

void Node3D::set_notify_transform(bool p_enable) {
	data.notify_transform = p_enable;
}

bool Node3D::is_transform_notification_enabled() const {
	return data.notify_transform;
}

void Node3D::set_notify_local_transform(bool p_enable) {
	data.notify_local_transform = p_enable;
}

bool Node3D::is_local_transform_notification_enabled() const {
	return data.notify_local_transform;
}

void Node3D::force_update_transform() {
	ERR_FAIL_COND(!is_inside_tree());
	if (!xform_change.in_list()) {
		return; //nothing to update
	}
	get_tree()->xform_change_list.remove(&xform_change);

	notification(NOTIFICATION_TRANSFORM_CHANGED);
}

void Node3D::_update_visibility_parent(bool p_update_root) {
	RID new_parent;

	if (!visibility_parent_path.is_empty()) {
		if (!p_update_root) {
			return;
		}
		Node *parent = get_node_or_null(visibility_parent_path);
		ERR_FAIL_COND_MSG(!parent, "Can't find visibility parent node at path: " + visibility_parent_path);
		ERR_FAIL_COND_MSG(parent == this, "The visibility parent can't be the same node.");
		GeometryInstance3D *gi = Object::cast_to<GeometryInstance3D>(parent);
		ERR_FAIL_COND_MSG(!gi, "The visibility parent node must be a GeometryInstance3D, at path: " + visibility_parent_path);
		new_parent = gi ? gi->get_instance() : RID();
	} else if (data.parent) {
		new_parent = data.parent->data.visibility_parent;
	}

	if (new_parent == data.visibility_parent) {
		return;
	}

	data.visibility_parent = new_parent;

	VisualInstance3D *vi = Object::cast_to<VisualInstance3D>(this);
	if (vi) {
		RS::get_singleton()->instance_set_visibility_parent(vi->get_instance(), data.visibility_parent);
	}

	for (Node3D *c : data.children) {
		c->_update_visibility_parent(false);
	}
}

void Node3D::set_visibility_parent(const NodePath &p_path) {
	visibility_parent_path = p_path;
	if (is_inside_tree()) {
		_update_visibility_parent(true);
	}
}

NodePath Node3D::get_visibility_parent() const {
	return visibility_parent_path;
}

void Node3D::_validate_property(PropertyInfo &property) const {
	if (data.rotation_edit_mode != ROTATION_EDIT_MODE_BASIS && property.name == "basis") {
		property.usage = 0;
	}
	if (data.rotation_edit_mode == ROTATION_EDIT_MODE_BASIS && property.name == "scale") {
		property.usage = 0;
	}
	if (data.rotation_edit_mode != ROTATION_EDIT_MODE_QUATERNION && property.name == "quaternion") {
		property.usage = 0;
	}
	if (data.rotation_edit_mode != ROTATION_EDIT_MODE_EULER && property.name == "rotation") {
		property.usage = 0;
	}
	if (data.rotation_edit_mode != ROTATION_EDIT_MODE_EULER && property.name == "rotation_order") {
		property.usage = 0;
	}
}

void Node3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_transform", "local"), &Node3D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &Node3D::get_transform);
	ClassDB::bind_method(D_METHOD("set_position", "position"), &Node3D::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &Node3D::get_position);
	ClassDB::bind_method(D_METHOD("set_rotation", "euler"), &Node3D::set_rotation);
	ClassDB::bind_method(D_METHOD("get_rotation"), &Node3D::get_rotation);
	ClassDB::bind_method(D_METHOD("set_rotation_order", "order"), &Node3D::set_rotation_order);
	ClassDB::bind_method(D_METHOD("get_rotation_order"), &Node3D::get_rotation_order);
	ClassDB::bind_method(D_METHOD("set_rotation_edit_mode", "edit_mode"), &Node3D::set_rotation_edit_mode);
	ClassDB::bind_method(D_METHOD("get_rotation_edit_mode"), &Node3D::get_rotation_edit_mode);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &Node3D::set_scale);
	ClassDB::bind_method(D_METHOD("get_scale"), &Node3D::get_scale);
	ClassDB::bind_method(D_METHOD("set_quaternion", "quaternion"), &Node3D::set_quaternion);
	ClassDB::bind_method(D_METHOD("get_quaternion"), &Node3D::get_quaternion);
	ClassDB::bind_method(D_METHOD("set_basis", "basis"), &Node3D::set_basis);
	ClassDB::bind_method(D_METHOD("get_basis"), &Node3D::get_basis);
	ClassDB::bind_method(D_METHOD("set_global_transform", "global"), &Node3D::set_global_transform);
	ClassDB::bind_method(D_METHOD("get_global_transform"), &Node3D::get_global_transform);
	ClassDB::bind_method(D_METHOD("get_parent_node_3d"), &Node3D::get_parent_node_3d);
	ClassDB::bind_method(D_METHOD("set_ignore_transform_notification", "enabled"), &Node3D::set_ignore_transform_notification);
	ClassDB::bind_method(D_METHOD("set_as_top_level", "enable"), &Node3D::set_as_top_level);
	ClassDB::bind_method(D_METHOD("is_set_as_top_level"), &Node3D::is_set_as_top_level);
	ClassDB::bind_method(D_METHOD("set_disable_scale", "disable"), &Node3D::set_disable_scale);
	ClassDB::bind_method(D_METHOD("is_scale_disabled"), &Node3D::is_scale_disabled);
	ClassDB::bind_method(D_METHOD("get_world_3d"), &Node3D::get_world_3d);

	ClassDB::bind_method(D_METHOD("force_update_transform"), &Node3D::force_update_transform);

	ClassDB::bind_method(D_METHOD("set_visibility_parent", "path"), &Node3D::set_visibility_parent);
	ClassDB::bind_method(D_METHOD("get_visibility_parent"), &Node3D::get_visibility_parent);

	ClassDB::bind_method(D_METHOD("update_gizmos"), &Node3D::update_gizmos);
	ClassDB::bind_method(D_METHOD("add_gizmo", "gizmo"), &Node3D::add_gizmo);
	ClassDB::bind_method(D_METHOD("get_gizmos"), &Node3D::get_gizmos_bind);
	ClassDB::bind_method(D_METHOD("clear_gizmos"), &Node3D::clear_gizmos);
	ClassDB::bind_method(D_METHOD("set_subgizmo_selection", "gizmo", "id", "transform"), &Node3D::set_subgizmo_selection);
	ClassDB::bind_method(D_METHOD("clear_subgizmo_selection"), &Node3D::clear_subgizmo_selection);

	ClassDB::bind_method(D_METHOD("set_visible", "visible"), &Node3D::set_visible);
	ClassDB::bind_method(D_METHOD("is_visible"), &Node3D::is_visible);
	ClassDB::bind_method(D_METHOD("is_visible_in_tree"), &Node3D::is_visible_in_tree);
	ClassDB::bind_method(D_METHOD("show"), &Node3D::show);
	ClassDB::bind_method(D_METHOD("hide"), &Node3D::hide);

	ClassDB::bind_method(D_METHOD("set_notify_local_transform", "enable"), &Node3D::set_notify_local_transform);
	ClassDB::bind_method(D_METHOD("is_local_transform_notification_enabled"), &Node3D::is_local_transform_notification_enabled);

	ClassDB::bind_method(D_METHOD("set_notify_transform", "enable"), &Node3D::set_notify_transform);
	ClassDB::bind_method(D_METHOD("is_transform_notification_enabled"), &Node3D::is_transform_notification_enabled);

	ClassDB::bind_method(D_METHOD("rotate", "axis", "angle"), &Node3D::rotate);
	ClassDB::bind_method(D_METHOD("global_rotate", "axis", "angle"), &Node3D::global_rotate);
	ClassDB::bind_method(D_METHOD("global_scale", "scale"), &Node3D::global_scale);
	ClassDB::bind_method(D_METHOD("global_translate", "offset"), &Node3D::global_translate);
	ClassDB::bind_method(D_METHOD("rotate_object_local", "axis", "angle"), &Node3D::rotate_object_local);
	ClassDB::bind_method(D_METHOD("scale_object_local", "scale"), &Node3D::scale_object_local);
	ClassDB::bind_method(D_METHOD("translate_object_local", "offset"), &Node3D::translate_object_local);
	ClassDB::bind_method(D_METHOD("rotate_x", "angle"), &Node3D::rotate_x);
	ClassDB::bind_method(D_METHOD("rotate_y", "angle"), &Node3D::rotate_y);
	ClassDB::bind_method(D_METHOD("rotate_z", "angle"), &Node3D::rotate_z);
	ClassDB::bind_method(D_METHOD("translate", "offset"), &Node3D::translate);
	ClassDB::bind_method(D_METHOD("orthonormalize"), &Node3D::orthonormalize);
	ClassDB::bind_method(D_METHOD("set_identity"), &Node3D::set_identity);

	ClassDB::bind_method(D_METHOD("look_at", "target", "up"), &Node3D::look_at, DEFVAL(Vector3(0, 1, 0)));
	ClassDB::bind_method(D_METHOD("look_at_from_position", "position", "target", "up"), &Node3D::look_at_from_position, DEFVAL(Vector3(0, 1, 0)));

	ClassDB::bind_method(D_METHOD("to_local", "global_point"), &Node3D::to_local);
	ClassDB::bind_method(D_METHOD("to_global", "local_point"), &Node3D::to_global);

	BIND_CONSTANT(NOTIFICATION_TRANSFORM_CHANGED);
	BIND_CONSTANT(NOTIFICATION_ENTER_WORLD);
	BIND_CONSTANT(NOTIFICATION_EXIT_WORLD);
	BIND_CONSTANT(NOTIFICATION_VISIBILITY_CHANGED);

	BIND_ENUM_CONSTANT(ROTATION_EDIT_MODE_EULER);
	BIND_ENUM_CONSTANT(ROTATION_EDIT_MODE_QUATERNION);
	BIND_ENUM_CONSTANT(ROTATION_EDIT_MODE_BASIS);

	BIND_ENUM_CONSTANT(ROTATION_ORDER_XYZ);
	BIND_ENUM_CONSTANT(ROTATION_ORDER_XZY);
	BIND_ENUM_CONSTANT(ROTATION_ORDER_YXZ);
	BIND_ENUM_CONSTANT(ROTATION_ORDER_YZX);
	BIND_ENUM_CONSTANT(ROTATION_ORDER_ZXY);
	BIND_ENUM_CONSTANT(ROTATION_ORDER_ZYX);

	//ADD_PROPERTY( PropertyInfo(Variant::TRANSFORM3D,"transform/global",PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR ), "set_global_transform", "get_global_transform") ;
	ADD_GROUP("Transform", "");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "global_transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_global_transform", "get_global_transform");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "position", PROPERTY_HINT_RANGE, "-99999,99999,0,or_greater,or_lesser,noslider,suffix:m", PROPERTY_USAGE_EDITOR), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rotation", PROPERTY_HINT_RANGE, "-360,360,0.1,or_lesser,or_greater,radians", PROPERTY_USAGE_EDITOR), "set_rotation", "get_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::QUATERNION, "quaternion", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_quaternion", "get_quaternion");
	ADD_PROPERTY(PropertyInfo(Variant::BASIS, "basis", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_basis", "get_basis");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "scale", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_scale", "get_scale");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rotation_edit_mode", PROPERTY_HINT_ENUM, "Euler,Quaternion,Basis"), "set_rotation_edit_mode", "get_rotation_edit_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rotation_order", PROPERTY_HINT_ENUM, "XYZ,XZY,YXZ,YZX,ZXY,ZYX"), "set_rotation_order", "get_rotation_order");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "top_level"), "set_as_top_level", "is_set_as_top_level");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_transform", "get_transform");
	ADD_GROUP("Visibility", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible"), "set_visible", "is_visible");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "visibility_parent", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "GeometryInstance3D"), "set_visibility_parent", "get_visibility_parent");

	ADD_SIGNAL(MethodInfo("visibility_changed"));
}

Node3D::Node3D() :
		xform_change(this) {}
