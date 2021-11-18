/*************************************************************************/
/*  spatial.cpp                                                          */
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

#include "spatial.h"

#include "core/engine.h"
#include "core/message_queue.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/scene_string_names.h"
#include "servers/visual_server_callbacks.h"

/*

 possible algorithms:

 Algorithm 1: (current)

 definition of invalidation: global is invalid

 1) If a node sets a LOCAL, it produces an invalidation of everything above
 .  a) If above is invalid, don't keep invalidating upwards
 2) If a node sets a GLOBAL, it is converted to LOCAL (and forces validation of everything pending below)

 drawback: setting/reading globals is useful and used very very often, and using affine inverses is slow

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

SpatialGizmo::SpatialGizmo() {
}

void Spatial::_notify_dirty() {
#ifdef TOOLS_ENABLED
	if ((data.gizmo.is_valid() || data.notify_transform) && !data.ignore_notification && !xform_change.in_list()) {
#else
	if (data.notify_transform && !data.ignore_notification && !xform_change.in_list()) {

#endif
		get_tree()->xform_change_list.add(&xform_change);
	}
}

void Spatial::_update_local_transform() const {
	data.local_transform.basis.set_euler_scale(data.rotation, data.scale);

	data.dirty &= ~DIRTY_LOCAL;
}
void Spatial::_propagate_transform_changed(Spatial *p_origin) {
	if (!is_inside_tree()) {
		return;
	}

	/*
	if (data.dirty&DIRTY_GLOBAL)
		return; //already dirty
	*/

	data.children_lock++;

	for (List<Spatial *>::Element *E = data.children.front(); E; E = E->next()) {
		if (E->get()->data.toplevel_active) {
			continue; //don't propagate to a toplevel
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

void Spatial::notification_callback(int p_message_type) {
	switch (p_message_type) {
		default:
			break;
		case VisualServerCallbacks::CALLBACK_NOTIFICATION_ENTER_GAMEPLAY: {
			notification(NOTIFICATION_ENTER_GAMEPLAY);
		} break;
		case VisualServerCallbacks::CALLBACK_NOTIFICATION_EXIT_GAMEPLAY: {
			notification(NOTIFICATION_EXIT_GAMEPLAY);
		} break;
		case VisualServerCallbacks::CALLBACK_SIGNAL_ENTER_GAMEPLAY: {
			emit_signal("gameplay_entered");
		} break;
		case VisualServerCallbacks::CALLBACK_SIGNAL_EXIT_GAMEPLAY: {
			emit_signal("gameplay_exited");
		} break;
	}
}

void Spatial::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			ERR_FAIL_COND(!get_tree());

			Node *p = get_parent();
			if (p) {
				data.parent = Object::cast_to<Spatial>(p);
			}

			if (data.parent) {
				data.C = data.parent->data.children.push_back(this);
			} else {
				data.C = nullptr;
			}

			if (data.toplevel && !Engine::get_singleton()->is_editor_hint()) {
				if (data.parent) {
					data.local_transform = data.parent->get_global_transform() * get_transform();
					data.dirty = DIRTY_VECTORS; //global is always dirty upon entering a scene
				}
				data.toplevel_active = true;
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
			data.toplevel_active = false;
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
				get_script_instance()->call_multilevel(SceneStringNames::get_singleton()->_enter_world, nullptr, 0);
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
				get_script_instance()->call_multilevel(SceneStringNames::get_singleton()->_exit_world, nullptr, 0);
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

void Spatial::set_transform(const Transform &p_transform) {
	data.local_transform = p_transform;
	data.dirty |= DIRTY_VECTORS;
	_change_notify("translation");
	_change_notify("rotation");
	_change_notify("rotation_degrees");
	_change_notify("scale");
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

void Spatial::set_global_transform(const Transform &p_transform) {
	Transform xform = (data.parent && !data.toplevel_active) ? data.parent->get_global_transform().affine_inverse() * p_transform : p_transform;

	set_transform(xform);
}

Transform Spatial::get_transform() const {
	if (data.dirty & DIRTY_LOCAL) {
		_update_local_transform();
	}

	return data.local_transform;
}
Transform Spatial::get_global_transform() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Transform());

	if (data.dirty & DIRTY_GLOBAL) {
		if (data.dirty & DIRTY_LOCAL) {
			_update_local_transform();
		}

		if (data.parent && !data.toplevel_active) {
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
Transform Spatial::get_global_gizmo_transform() const {
	return get_global_transform();
}

Transform Spatial::get_local_gizmo_transform() const {
	return get_transform();
}
#endif

Spatial *Spatial::get_parent_spatial() const {
	return data.parent;
}

Transform Spatial::get_relative_transform(const Node *p_parent) const {
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

void Spatial::set_translation(const Vector3 &p_translation) {
	data.local_transform.origin = p_translation;
	_change_notify("transform");
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

void Spatial::set_rotation(const Vector3 &p_euler_rad) {
	if (data.dirty & DIRTY_VECTORS) {
		data.scale = data.local_transform.basis.get_scale();
		data.dirty &= ~DIRTY_VECTORS;
	}

	data.rotation = p_euler_rad;
	data.dirty |= DIRTY_LOCAL;
	_change_notify("transform");
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

void Spatial::set_rotation_degrees(const Vector3 &p_euler_deg) {
	set_rotation(p_euler_deg * Math_PI / 180.0);
}

void Spatial::set_scale(const Vector3 &p_scale) {
	if (data.dirty & DIRTY_VECTORS) {
		data.rotation = data.local_transform.basis.get_rotation();
		data.dirty &= ~DIRTY_VECTORS;
	}

	data.scale = p_scale;
	data.dirty |= DIRTY_LOCAL;
	_change_notify("transform");
	_propagate_transform_changed(this);
	if (data.notify_local_transform) {
		notification(NOTIFICATION_LOCAL_TRANSFORM_CHANGED);
	}
}

Vector3 Spatial::get_translation() const {
	return data.local_transform.origin;
}

Vector3 Spatial::get_rotation() const {
	if (data.dirty & DIRTY_VECTORS) {
		data.scale = data.local_transform.basis.get_scale();
		data.rotation = data.local_transform.basis.get_rotation();

		data.dirty &= ~DIRTY_VECTORS;
	}

	return data.rotation;
}

Vector3 Spatial::get_rotation_degrees() const {
	return get_rotation() * 180.0 / Math_PI;
}

Vector3 Spatial::get_scale() const {
	if (data.dirty & DIRTY_VECTORS) {
		data.scale = data.local_transform.basis.get_scale();
		data.rotation = data.local_transform.basis.get_rotation();

		data.dirty &= ~DIRTY_VECTORS;
	}

	return data.scale;
}

void Spatial::update_gizmo() {
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

void Spatial::set_gizmo(const Ref<SpatialGizmo> &p_gizmo) {
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

Ref<SpatialGizmo> Spatial::get_gizmo() const {
#ifdef TOOLS_ENABLED

	return data.gizmo;
#else

	return Ref<SpatialGizmo>();
#endif
}

void Spatial::_update_gizmo() {
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

void Spatial::set_disable_gizmo(bool p_enabled) {
#ifdef TOOLS_ENABLED
	data.gizmo_disabled = p_enabled;
	if (!p_enabled && data.gizmo.is_valid()) {
		data.gizmo = Ref<SpatialGizmo>();
	}
#endif
}

void Spatial::set_disable_scale(bool p_enabled) {
	data.disable_scale = p_enabled;
}

bool Spatial::is_scale_disabled() const {
	return data.disable_scale;
}

void Spatial::set_as_toplevel(bool p_enabled) {
	if (data.toplevel == p_enabled) {
		return;
	}
	if (is_inside_tree() && !Engine::get_singleton()->is_editor_hint()) {
		if (p_enabled) {
			set_transform(get_global_transform());
		} else if (data.parent) {
			set_transform(data.parent->get_global_transform().affine_inverse() * get_global_transform());
		}

		data.toplevel = p_enabled;
		data.toplevel_active = p_enabled;

	} else {
		data.toplevel = p_enabled;
	}
}

bool Spatial::is_set_as_toplevel() const {
	return data.toplevel;
}

Ref<World> Spatial::get_world() const {
	ERR_FAIL_COND_V(!is_inside_world(), Ref<World>());
	ERR_FAIL_COND_V(!data.viewport, Ref<World>());

	return data.viewport->find_world();
}

void Spatial::_propagate_visibility_changed() {
	notification(NOTIFICATION_VISIBILITY_CHANGED);
	emit_signal(SceneStringNames::get_singleton()->visibility_changed);
	_change_notify("visible");
#ifdef TOOLS_ENABLED
	if (data.gizmo.is_valid()) {
		_update_gizmo();
	}
#endif

	for (List<Spatial *>::Element *E = data.children.front(); E; E = E->next()) {
		Spatial *c = E->get();
		if (!c || !c->data.visible) {
			continue;
		}
		c->_propagate_visibility_changed();
	}
}

void Spatial::show() {
	if (data.visible) {
		return;
	}

	data.visible = true;

	if (!is_inside_tree()) {
		return;
	}

	_propagate_visibility_changed();
}

void Spatial::hide() {
	if (!data.visible) {
		return;
	}

	data.visible = false;

	if (!is_inside_tree()) {
		return;
	}

	_propagate_visibility_changed();
}

bool Spatial::is_visible_in_tree() const {
	const Spatial *s = this;

	while (s) {
		if (!s->data.visible) {
			return false;
		}
		s = s->data.parent;
	}

	return true;
}

void Spatial::set_visible(bool p_visible) {
	if (p_visible) {
		show();
	} else {
		hide();
	}
}

bool Spatial::is_visible() const {
	return data.visible;
}

void Spatial::rotate_object_local(const Vector3 &p_axis, float p_angle) {
	Transform t = get_transform();
	t.basis.rotate_local(p_axis, p_angle);
	set_transform(t);
}

void Spatial::rotate(const Vector3 &p_axis, float p_angle) {
	Transform t = get_transform();
	t.basis.rotate(p_axis, p_angle);
	set_transform(t);
}

void Spatial::rotate_x(float p_angle) {
	Transform t = get_transform();
	t.basis.rotate(Vector3(1, 0, 0), p_angle);
	set_transform(t);
}

void Spatial::rotate_y(float p_angle) {
	Transform t = get_transform();
	t.basis.rotate(Vector3(0, 1, 0), p_angle);
	set_transform(t);
}
void Spatial::rotate_z(float p_angle) {
	Transform t = get_transform();
	t.basis.rotate(Vector3(0, 0, 1), p_angle);
	set_transform(t);
}

void Spatial::translate(const Vector3 &p_offset) {
	Transform t = get_transform();
	t.translate(p_offset);
	set_transform(t);
}

void Spatial::translate_object_local(const Vector3 &p_offset) {
	Transform t = get_transform();

	Transform s;
	s.translate(p_offset);
	set_transform(t * s);
}

void Spatial::scale(const Vector3 &p_ratio) {
	Transform t = get_transform();
	t.basis.scale(p_ratio);
	set_transform(t);
}

void Spatial::scale_object_local(const Vector3 &p_scale) {
	Transform t = get_transform();
	t.basis.scale_local(p_scale);
	set_transform(t);
}

void Spatial::global_rotate(const Vector3 &p_axis, float p_angle) {
	Transform t = get_global_transform();
	t.basis.rotate(p_axis, p_angle);
	set_global_transform(t);
}

void Spatial::global_scale(const Vector3 &p_scale) {
	Transform t = get_global_transform();
	t.basis.scale(p_scale);
	set_global_transform(t);
}

void Spatial::global_translate(const Vector3 &p_offset) {
	Transform t = get_global_transform();
	t.origin += p_offset;
	set_global_transform(t);
}

void Spatial::orthonormalize() {
	Transform t = get_transform();
	t.orthonormalize();
	set_transform(t);
}

void Spatial::set_identity() {
	set_transform(Transform());
}

void Spatial::look_at(const Vector3 &p_target, const Vector3 &p_up) {
	Vector3 origin(get_global_transform().origin);
	look_at_from_position(origin, p_target, p_up);
}

void Spatial::look_at_from_position(const Vector3 &p_pos, const Vector3 &p_target, const Vector3 &p_up) {
	ERR_FAIL_COND_MSG(p_pos == p_target, "Node origin and target are in the same position, look_at() failed.");
	ERR_FAIL_COND_MSG(p_up.cross(p_target - p_pos) == Vector3(), "Up vector and direction between node origin and target are aligned, look_at() failed.");

	Transform lookat;
	lookat.origin = p_pos;

	Vector3 original_scale(get_scale());
	lookat = lookat.looking_at(p_target, p_up);
	set_global_transform(lookat);
	set_scale(original_scale);
}

Vector3 Spatial::to_local(Vector3 p_global) const {
	return get_global_transform().affine_inverse().xform(p_global);
}

Vector3 Spatial::to_global(Vector3 p_local) const {
	return get_global_transform().xform(p_local);
}

void Spatial::set_notify_transform(bool p_enable) {
	data.notify_transform = p_enable;
}

bool Spatial::is_transform_notification_enabled() const {
	return data.notify_transform;
}

void Spatial::set_notify_local_transform(bool p_enable) {
	data.notify_local_transform = p_enable;
}

bool Spatial::is_local_transform_notification_enabled() const {
	return data.notify_local_transform;
}

void Spatial::force_update_transform() {
	ERR_FAIL_COND(!is_inside_tree());
	if (!xform_change.in_list()) {
		return; //nothing to update
	}
	get_tree()->xform_change_list.remove(&xform_change);

	notification(NOTIFICATION_TRANSFORM_CHANGED);
}

void Spatial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_transform", "local"), &Spatial::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &Spatial::get_transform);
	ClassDB::bind_method(D_METHOD("set_translation", "translation"), &Spatial::set_translation);
	ClassDB::bind_method(D_METHOD("get_translation"), &Spatial::get_translation);
	ClassDB::bind_method(D_METHOD("set_rotation", "euler"), &Spatial::set_rotation);
	ClassDB::bind_method(D_METHOD("get_rotation"), &Spatial::get_rotation);
	ClassDB::bind_method(D_METHOD("set_rotation_degrees", "euler_degrees"), &Spatial::set_rotation_degrees);
	ClassDB::bind_method(D_METHOD("get_rotation_degrees"), &Spatial::get_rotation_degrees);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &Spatial::set_scale);
	ClassDB::bind_method(D_METHOD("get_scale"), &Spatial::get_scale);
	ClassDB::bind_method(D_METHOD("set_global_transform", "global"), &Spatial::set_global_transform);
	ClassDB::bind_method(D_METHOD("get_global_transform"), &Spatial::get_global_transform);
	ClassDB::bind_method(D_METHOD("get_parent_spatial"), &Spatial::get_parent_spatial);
	ClassDB::bind_method(D_METHOD("set_ignore_transform_notification", "enabled"), &Spatial::set_ignore_transform_notification);
	ClassDB::bind_method(D_METHOD("set_as_toplevel", "enable"), &Spatial::set_as_toplevel);
	ClassDB::bind_method(D_METHOD("is_set_as_toplevel"), &Spatial::is_set_as_toplevel);
	ClassDB::bind_method(D_METHOD("set_disable_scale", "disable"), &Spatial::set_disable_scale);
	ClassDB::bind_method(D_METHOD("is_scale_disabled"), &Spatial::is_scale_disabled);
	ClassDB::bind_method(D_METHOD("get_world"), &Spatial::get_world);

	ClassDB::bind_method(D_METHOD("force_update_transform"), &Spatial::force_update_transform);

	ClassDB::bind_method(D_METHOD("_update_gizmo"), &Spatial::_update_gizmo);

	ClassDB::bind_method(D_METHOD("update_gizmo"), &Spatial::update_gizmo);
	ClassDB::bind_method(D_METHOD("set_gizmo", "gizmo"), &Spatial::set_gizmo);
	ClassDB::bind_method(D_METHOD("get_gizmo"), &Spatial::get_gizmo);

	ClassDB::bind_method(D_METHOD("set_visible", "visible"), &Spatial::set_visible);
	ClassDB::bind_method(D_METHOD("is_visible"), &Spatial::is_visible);
	ClassDB::bind_method(D_METHOD("is_visible_in_tree"), &Spatial::is_visible_in_tree);
	ClassDB::bind_method(D_METHOD("show"), &Spatial::show);
	ClassDB::bind_method(D_METHOD("hide"), &Spatial::hide);

	ClassDB::bind_method(D_METHOD("set_notify_local_transform", "enable"), &Spatial::set_notify_local_transform);
	ClassDB::bind_method(D_METHOD("is_local_transform_notification_enabled"), &Spatial::is_local_transform_notification_enabled);

	ClassDB::bind_method(D_METHOD("set_notify_transform", "enable"), &Spatial::set_notify_transform);
	ClassDB::bind_method(D_METHOD("is_transform_notification_enabled"), &Spatial::is_transform_notification_enabled);

	ClassDB::bind_method(D_METHOD("rotate", "axis", "angle"), &Spatial::rotate);
	ClassDB::bind_method(D_METHOD("global_rotate", "axis", "angle"), &Spatial::global_rotate);
	ClassDB::bind_method(D_METHOD("global_scale", "scale"), &Spatial::global_scale);
	ClassDB::bind_method(D_METHOD("global_translate", "offset"), &Spatial::global_translate);
	ClassDB::bind_method(D_METHOD("rotate_object_local", "axis", "angle"), &Spatial::rotate_object_local);
	ClassDB::bind_method(D_METHOD("scale_object_local", "scale"), &Spatial::scale_object_local);
	ClassDB::bind_method(D_METHOD("translate_object_local", "offset"), &Spatial::translate_object_local);
	ClassDB::bind_method(D_METHOD("rotate_x", "angle"), &Spatial::rotate_x);
	ClassDB::bind_method(D_METHOD("rotate_y", "angle"), &Spatial::rotate_y);
	ClassDB::bind_method(D_METHOD("rotate_z", "angle"), &Spatial::rotate_z);
	ClassDB::bind_method(D_METHOD("translate", "offset"), &Spatial::translate);
	ClassDB::bind_method(D_METHOD("orthonormalize"), &Spatial::orthonormalize);
	ClassDB::bind_method(D_METHOD("set_identity"), &Spatial::set_identity);

	ClassDB::bind_method(D_METHOD("look_at", "target", "up"), &Spatial::look_at);
	ClassDB::bind_method(D_METHOD("look_at_from_position", "position", "target", "up"), &Spatial::look_at_from_position);

	ClassDB::bind_method(D_METHOD("to_local", "global_point"), &Spatial::to_local);
	ClassDB::bind_method(D_METHOD("to_global", "local_point"), &Spatial::to_global);

	BIND_CONSTANT(NOTIFICATION_TRANSFORM_CHANGED);
	BIND_CONSTANT(NOTIFICATION_ENTER_WORLD);
	BIND_CONSTANT(NOTIFICATION_EXIT_WORLD);
	BIND_CONSTANT(NOTIFICATION_VISIBILITY_CHANGED);
	BIND_CONSTANT(NOTIFICATION_ENTER_GAMEPLAY);
	BIND_CONSTANT(NOTIFICATION_EXIT_GAMEPLAY);

	//ADD_PROPERTY( PropertyInfo(Variant::TRANSFORM,"transform/global",PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR ), "set_global_transform", "get_global_transform") ;
	ADD_GROUP("Transform", "");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "global_transform", PROPERTY_HINT_NONE, "", 0), "set_global_transform", "get_global_transform");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "translation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_translation", "get_translation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rotation_degrees", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_rotation_degrees", "get_rotation_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rotation", PROPERTY_HINT_NONE, "", 0), "set_rotation", "get_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "scale", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_scale", "get_scale");
	ADD_GROUP("Matrix", "");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "transform", PROPERTY_HINT_NONE, ""), "set_transform", "get_transform");
	ADD_GROUP("Visibility", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible"), "set_visible", "is_visible");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gizmo", PROPERTY_HINT_RESOURCE_TYPE, "SpatialGizmo", 0), "set_gizmo", "get_gizmo");

	ADD_SIGNAL(MethodInfo("visibility_changed"));
	ADD_SIGNAL(MethodInfo("gameplay_entered"));
	ADD_SIGNAL(MethodInfo("gameplay_exited"));
}

Spatial::Spatial() :
		xform_change(this) {
	data.dirty = DIRTY_NONE;
	data.children_lock = 0;

	data.ignore_notification = false;
	data.toplevel = false;
	data.toplevel_active = false;
	data.scale = Vector3(1, 1, 1);
	data.viewport = nullptr;
	data.inside_world = false;
	data.visible = true;
	data.disable_scale = false;

	data.spatial_flags = SPATIAL_FLAG_VI_VISIBLE;

#ifdef TOOLS_ENABLED
	data.gizmo_disabled = false;
	data.gizmo_dirty = false;
#endif
	data.notify_local_transform = false;
	data.notify_transform = false;
	data.parent = nullptr;
	data.C = nullptr;
}

Spatial::~Spatial() {
}
