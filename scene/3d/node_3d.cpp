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

#include "core/config/engine.h"
#include "core/object/message_queue.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/scene_string_names.h"

/*

 possible algorithms:

 Algorithm 1: (current)

 definition of invalidation: global is invalid

 1) If a node sets a LOCAL, it produces an invalidation of everything above
    a) If above is invalid, don't keep invalidating upwards
 2) If a node sets a GLOBAL, it is converted to LOCAL (and forces validation of everything pending below)

 drawback: setting/reading globals is useful and used very often, and using affine inverses is slow

---

 Algorithm 2: (no longer current)

 definition of invalidation: NONE dirty, LOCAL dirty, GLOBAL dirty

 1) If a node sets a LOCAL, it must climb the tree and set it as GLOBAL dirty
    a) marking GLOBALs as dirty up all the tree must be done always
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
	if ((data.gizmo.is_valid() || data.notify_transform) && !data.ignore_notification && !xform_change.in_list()) {
#else
	if (data.notify_transform && !data.ignore_notification && !xform_change.in_list()) {

#endif
		get_tree()->xform_change_list.add(&xform_change);
	}
}

void Node3D::_update_local_transform() const {
	data.local_transform.basis.set_euler_scale(data.rotation, data.scale);

	data.dirty &= ~DIRTY_LOCAL;
}

void Node3D::_propagate_transform_changed(Node3D *p_origin) {
	if (!is_inside_tree()) {
		return;
	}

	/*
	if (data.dirty&DIRTY_GLOBAL)
		return; //already dirty
	*/

	data.children_lock++;

	for (List<Node3D *>::Element *E = data.children.front(); E; E = E->next()) {
		if (E->get()->data.top_level_active) {
			continue; //don't propagate to a top_level
		}
		E->get()->_propagate_transform_changed(p_origin);
	}
#ifdef TOOLS_ENABLED
	if ((data.gizmo.is_valid() || data.notify_transform) && !data.ignore_notification && !xform_change.in_list()) {
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
				//get_scene()->call_group(SceneMainLoop::GROUP_CALL_REALTIME,SceneStringNames::get_singleton()->_spatial_editor_group,SceneStringNames::get_singleton()->_request_gizmo,this);
				get_tree()->call_group_flags(0, SceneStringNames::get_singleton()->_spatial_editor_group, SceneStringNames::get_singleton()->_request_gizmo, this);
				if (!data.gizmo_disabled) {
					if (data.gizmo.is_valid()) {
						data.gizmo->create();
						if (is_visible_in_tree()) {
							data.gizmo->redraw();
						}
						data.gizmo->transform();
					}
				}
			}
#endif

		} break;
		case NOTIFICATION_EXIT_WORLD: {
#ifdef TOOLS_ENABLED
			if (data.gizmo.is_valid()) {
				data.gizmo->free();
				data.gizmo.unref();
			}
#endif

			if (get_script_instance()) {
				get_script_instance()->call(SceneStringNames::get_singleton()->_exit_world);
			}

			data.viewport = nullptr;
			data.inside_world = false;

		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
#ifdef TOOLS_ENABLED
			if (data.gizmo.is_valid()) {
				data.gizmo->transform();
			}
#endif
		} break;

		default: {
		}
	}
}

void Node3D::set_transform(const Transform &p_transform) {
	data.local_transform = p_transform;
	data.dirty |= DIRTY_VECTORS;
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

void Node3D::set_global_transform(const Transform &p_transform) {
	Transform xform =
			(data.parent && !data.top_level_active) ?
					  data.parent->get_global_transform().affine_inverse() * p_transform :
					  p_transform;

	set_transform(xform);
}

Transform Node3D::get_transform() const {
	if (data.dirty & DIRTY_LOCAL) {
		_update_local_transform();
	}

	return data.local_transform;
}

Transform Node3D::get_global_transform() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Transform());

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
Transform Node3D::get_global_gizmo_transform() const {
	return get_global_transform();
}

Transform Node3D::get_local_gizmo_transform() const {
	return get_transform();
}
#endif

Node3D *Node3D::get_parent_spatial() const {
	return data.parent;
}

Transform Node3D::get_relative_transform(const Node *p_parent) const {
	if (p_parent == this) {
		return Transform();
	}

	ERR_FAIL_COND_V(!data.parent, Transform());

	if (p_parent == data.parent) {
		return get_transform();
	} else {
		return data.parent->get_relative_transform(p_parent) * get_transform();
	}
}

void Node3D::set_translation(const Vector3 &p_translation) {
	data.local_transform.origin = p_translation;
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
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

void Node3D::set_rotation_degrees(const Vector3 &p_euler_deg) {
	set_rotation(p_euler_deg * (Math_PI / 180.0));
}

void Node3D::set_scale(const Vector3 &p_scale) {
	if (data.dirty & DIRTY_VECTORS) {
		data.rotation = data.local_transform.basis.get_rotation();
		data.dirty &= ~DIRTY_VECTORS;
	}

	data.scale = p_scale;
	data.dirty |= DIRTY_LOCAL;
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

Vector3 Node3D::get_translation() const {
	return data.local_transform.origin;
}

Vector3 Node3D::get_rotation() const {
	if (data.dirty & DIRTY_VECTORS) {
		data.scale = data.local_transform.basis.get_scale();
		data.rotation = data.local_transform.basis.get_rotation();

		data.dirty &= ~DIRTY_VECTORS;
	}

	return data.rotation;
}

Vector3 Node3D::get_rotation_degrees() const {
	return get_rotation() * (180.0 / Math_PI);
}

Vector3 Node3D::get_scale() const {
	if (data.dirty & DIRTY_VECTORS) {
		data.scale = data.local_transform.basis.get_scale();
		data.rotation = data.local_transform.basis.get_rotation();

		data.dirty &= ~DIRTY_VECTORS;
	}

	return data.scale;
}

void Node3D::update_gizmo() {
#ifdef TOOLS_ENABLED
	if (!is_inside_world()) {
		return;
	}
	if (!data.gizmo.is_valid()) {
		get_tree()->call_group_flags(SceneTree::GROUP_CALL_REALTIME, SceneStringNames::get_singleton()->_spatial_editor_group, SceneStringNames::get_singleton()->_request_gizmo, this);
	}
	if (!data.gizmo.is_valid()) {
		return;
	}
	if (data.gizmo_dirty) {
		return;
	}
	data.gizmo_dirty = true;
	MessageQueue::get_singleton()->push_call(this, "_update_gizmo");
#endif
}

void Node3D::set_gizmo(const Ref<Node3DGizmo> &p_gizmo) {
#ifdef TOOLS_ENABLED

	if (data.gizmo_disabled) {
		return;
	}
	if (data.gizmo.is_valid() && is_inside_world()) {
		data.gizmo->free();
	}
	data.gizmo = p_gizmo;
	if (data.gizmo.is_valid() && is_inside_world()) {
		data.gizmo->create();
		if (is_visible_in_tree()) {
			data.gizmo->redraw();
		}
		data.gizmo->transform();
	}

#endif
}

Ref<Node3DGizmo> Node3D::get_gizmo() const {
#ifdef TOOLS_ENABLED

	return data.gizmo;
#else

	return Ref<Node3DGizmo>();
#endif
}

void Node3D::_update_gizmo() {
#ifdef TOOLS_ENABLED
	if (!is_inside_world()) {
		return;
	}
	data.gizmo_dirty = false;
	if (data.gizmo.is_valid()) {
		if (is_visible_in_tree()) {
			data.gizmo->redraw();
		} else {
			data.gizmo->clear();
		}
	}
#endif
}

#ifdef TOOLS_ENABLED
void Node3D::set_disable_gizmo(bool p_enabled) {
	data.gizmo_disabled = p_enabled;
	if (!p_enabled && data.gizmo.is_valid()) {
		data.gizmo = Ref<Node3DGizmo>();
	}
}

#endif

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
	if (data.gizmo.is_valid()) {
		_update_gizmo();
	}
#endif

	for (List<Node3D *>::Element *E = data.children.front(); E; E = E->next()) {
		Node3D *c = E->get();
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

void Node3D::rotate_object_local(const Vector3 &p_axis, float p_angle) {
	Transform t = get_transform();
	t.basis.rotate_local(p_axis, p_angle);
	set_transform(t);
}

void Node3D::rotate(const Vector3 &p_axis, float p_angle) {
	Transform t = get_transform();
	t.basis.rotate(p_axis, p_angle);
	set_transform(t);
}

void Node3D::rotate_x(float p_angle) {
	Transform t = get_transform();
	t.basis.rotate(Vector3(1, 0, 0), p_angle);
	set_transform(t);
}

void Node3D::rotate_y(float p_angle) {
	Transform t = get_transform();
	t.basis.rotate(Vector3(0, 1, 0), p_angle);
	set_transform(t);
}

void Node3D::rotate_z(float p_angle) {
	Transform t = get_transform();
	t.basis.rotate(Vector3(0, 0, 1), p_angle);
	set_transform(t);
}

void Node3D::translate(const Vector3 &p_offset) {
	Transform t = get_transform();
	t.translate(p_offset);
	set_transform(t);
}

void Node3D::translate_object_local(const Vector3 &p_offset) {
	Transform t = get_transform();

	Transform s;
	s.translate(p_offset);
	set_transform(t * s);
}

void Node3D::scale(const Vector3 &p_ratio) {
	Transform t = get_transform();
	t.basis.scale(p_ratio);
	set_transform(t);
}

void Node3D::scale_object_local(const Vector3 &p_scale) {
	Transform t = get_transform();
	t.basis.scale_local(p_scale);
	set_transform(t);
}

void Node3D::global_rotate(const Vector3 &p_axis, float p_angle) {
	Transform t = get_global_transform();
	t.basis.rotate(p_axis, p_angle);
	set_global_transform(t);
}

void Node3D::global_scale(const Vector3 &p_scale) {
	Transform t = get_global_transform();
	t.basis.scale(p_scale);
	set_global_transform(t);
}

void Node3D::global_translate(const Vector3 &p_offset) {
	Transform t = get_global_transform();
	t.origin += p_offset;
	set_global_transform(t);
}

void Node3D::orthonormalize() {
	Transform t = get_transform();
	t.orthonormalize();
	set_transform(t);
}

void Node3D::set_identity() {
	set_transform(Transform());
}

void Node3D::look_at(const Vector3 &p_target, const Vector3 &p_up) {
	Vector3 origin(get_global_transform().origin);
	look_at_from_position(origin, p_target, p_up);
}

void Node3D::look_at_from_position(const Vector3 &p_pos, const Vector3 &p_target, const Vector3 &p_up) {
	ERR_FAIL_COND_MSG(p_pos == p_target, "Node origin and target are in the same position, look_at() failed.");
	ERR_FAIL_COND_MSG(p_up.cross(p_target - p_pos) == Vector3(), "Up vector and direction between node origin and target are aligned, look_at() failed.");

	Transform lookat;
	lookat.origin = p_pos;

	Vector3 original_scale(get_scale());
	lookat = lookat.looking_at(p_target, p_up);
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

void Node3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_transform", "local"), &Node3D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &Node3D::get_transform);
	ClassDB::bind_method(D_METHOD("set_translation", "translation"), &Node3D::set_translation);
	ClassDB::bind_method(D_METHOD("get_translation"), &Node3D::get_translation);
	ClassDB::bind_method(D_METHOD("set_rotation", "euler"), &Node3D::set_rotation);
	ClassDB::bind_method(D_METHOD("get_rotation"), &Node3D::get_rotation);
	ClassDB::bind_method(D_METHOD("set_rotation_degrees", "euler_degrees"), &Node3D::set_rotation_degrees);
	ClassDB::bind_method(D_METHOD("get_rotation_degrees"), &Node3D::get_rotation_degrees);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &Node3D::set_scale);
	ClassDB::bind_method(D_METHOD("get_scale"), &Node3D::get_scale);
	ClassDB::bind_method(D_METHOD("set_global_transform", "global"), &Node3D::set_global_transform);
	ClassDB::bind_method(D_METHOD("get_global_transform"), &Node3D::get_global_transform);
	ClassDB::bind_method(D_METHOD("get_parent_spatial"), &Node3D::get_parent_spatial);
	ClassDB::bind_method(D_METHOD("set_ignore_transform_notification", "enabled"), &Node3D::set_ignore_transform_notification);
	ClassDB::bind_method(D_METHOD("set_as_top_level", "enable"), &Node3D::set_as_top_level);
	ClassDB::bind_method(D_METHOD("is_set_as_top_level"), &Node3D::is_set_as_top_level);
	ClassDB::bind_method(D_METHOD("set_disable_scale", "disable"), &Node3D::set_disable_scale);
	ClassDB::bind_method(D_METHOD("is_scale_disabled"), &Node3D::is_scale_disabled);
	ClassDB::bind_method(D_METHOD("get_world_3d"), &Node3D::get_world_3d);

	ClassDB::bind_method(D_METHOD("force_update_transform"), &Node3D::force_update_transform);

	ClassDB::bind_method(D_METHOD("_update_gizmo"), &Node3D::_update_gizmo);

	ClassDB::bind_method(D_METHOD("update_gizmo"), &Node3D::update_gizmo);
	ClassDB::bind_method(D_METHOD("set_gizmo", "gizmo"), &Node3D::set_gizmo);
	ClassDB::bind_method(D_METHOD("get_gizmo"), &Node3D::get_gizmo);

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

	//ADD_PROPERTY( PropertyInfo(Variant::TRANSFORM,"transform/global",PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR ), "set_global_transform", "get_global_transform") ;
	ADD_GROUP("Transform", "");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "global_transform", PROPERTY_HINT_NONE, "", 0), "set_global_transform", "get_global_transform");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "translation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_translation", "get_translation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rotation_degrees", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_rotation_degrees", "get_rotation_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rotation", PROPERTY_HINT_NONE, "", 0), "set_rotation", "get_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "scale", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_scale", "get_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "top_level"), "set_as_top_level", "is_set_as_top_level");
	ADD_GROUP("Matrix", "");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "transform", PROPERTY_HINT_NONE, ""), "set_transform", "get_transform");
	ADD_GROUP("Visibility", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible"), "set_visible", "is_visible");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gizmo", PROPERTY_HINT_RESOURCE_TYPE, "Node3DGizmo", 0), "set_gizmo", "get_gizmo");

	ADD_SIGNAL(MethodInfo("visibility_changed"));
}

Node3D::Node3D() :
		xform_change(this) {}
