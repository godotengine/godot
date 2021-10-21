/*************************************************************************/
/*  collision_object_3d.cpp                                              */
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

#include "collision_object_3d.h"

#include "scene/scene_string_names.h"

void CollisionObject3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (_are_collision_shapes_visible()) {
				debug_shape_old_transform = get_global_transform();
				for (const KeyValue<uint32_t, ShapeData> &E : shapes) {
					debug_shapes_to_update.insert(E.key);
				}
				_update_debug_shapes();
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (debug_shapes_count > 0) {
				_clear_debug_shapes();
			}
		} break;

		case NOTIFICATION_ENTER_WORLD: {
			if (area) {
				PhysicsServer3D::get_singleton()->area_set_transform(rid, get_global_transform());
			} else {
				PhysicsServer3D::get_singleton()->body_set_state(rid, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
			}

			bool disabled = !is_enabled();

			if (disabled && (disable_mode != DISABLE_MODE_REMOVE)) {
				_apply_disabled();
			}

			if (!disabled || (disable_mode != DISABLE_MODE_REMOVE)) {
				RID space = get_world_3d()->get_space();
				if (area) {
					PhysicsServer3D::get_singleton()->area_set_space(rid, space);
				} else {
					PhysicsServer3D::get_singleton()->body_set_space(rid, space);
				}
			}

			_update_pickable();
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (only_update_transform_changes) {
				return;
			}

			if (area) {
				PhysicsServer3D::get_singleton()->area_set_transform(rid, get_global_transform());
			} else {
				PhysicsServer3D::get_singleton()->body_set_state(rid, PhysicsServer3D::BODY_STATE_TRANSFORM, get_global_transform());
			}

			_on_transform_changed();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_pickable();
		} break;

		case NOTIFICATION_EXIT_WORLD: {
			bool disabled = !is_enabled();

			if (!disabled || (disable_mode != DISABLE_MODE_REMOVE)) {
				if (area) {
					PhysicsServer3D::get_singleton()->area_set_space(rid, RID());
				} else {
					PhysicsServer3D::get_singleton()->body_set_space(rid, RID());
				}
			}

			if (disabled && (disable_mode != DISABLE_MODE_REMOVE)) {
				_apply_enabled();
			}
		} break;

		case NOTIFICATION_DISABLED: {
			_apply_disabled();
		} break;

		case NOTIFICATION_ENABLED: {
			_apply_enabled();
		} break;
	}
}

void CollisionObject3D::set_collision_layer(uint32_t p_layer) {
	collision_layer = p_layer;
	if (area) {
		PhysicsServer3D::get_singleton()->area_set_collision_layer(get_rid(), p_layer);
	} else {
		PhysicsServer3D::get_singleton()->body_set_collision_layer(get_rid(), p_layer);
	}
}

uint32_t CollisionObject3D::get_collision_layer() const {
	return collision_layer;
}

void CollisionObject3D::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
	if (area) {
		PhysicsServer3D::get_singleton()->area_set_collision_mask(get_rid(), p_mask);
	} else {
		PhysicsServer3D::get_singleton()->body_set_collision_mask(get_rid(), p_mask);
	}
}

uint32_t CollisionObject3D::get_collision_mask() const {
	return collision_mask;
}

void CollisionObject3D::set_collision_layer_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t collision_layer = get_collision_layer();
	if (p_value) {
		collision_layer |= 1 << (p_layer_number - 1);
	} else {
		collision_layer &= ~(1 << (p_layer_number - 1));
	}
	set_collision_layer(collision_layer);
}

bool CollisionObject3D::get_collision_layer_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_layer() & (1 << (p_layer_number - 1));
}

void CollisionObject3D::set_collision_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_collision_mask(mask);
}

bool CollisionObject3D::get_collision_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_mask() & (1 << (p_layer_number - 1));
}

void CollisionObject3D::set_disable_mode(DisableMode p_mode) {
	if (disable_mode == p_mode) {
		return;
	}

	bool disabled = is_inside_tree() && !is_enabled();

	if (disabled) {
		// Cancel previous disable mode.
		_apply_enabled();
	}

	disable_mode = p_mode;

	if (disabled) {
		// Apply new disable mode.
		_apply_disabled();
	}
}

CollisionObject3D::DisableMode CollisionObject3D::get_disable_mode() const {
	return disable_mode;
}

void CollisionObject3D::_apply_disabled() {
	switch (disable_mode) {
		case DISABLE_MODE_REMOVE: {
			if (is_inside_tree()) {
				if (area) {
					PhysicsServer3D::get_singleton()->area_set_space(rid, RID());
				} else {
					PhysicsServer3D::get_singleton()->body_set_space(rid, RID());
				}
			}
		} break;

		case DISABLE_MODE_MAKE_STATIC: {
			if (!area && (body_mode != PhysicsServer3D::BODY_MODE_STATIC)) {
				PhysicsServer3D::get_singleton()->body_set_mode(rid, PhysicsServer3D::BODY_MODE_STATIC);
			}
		} break;

		case DISABLE_MODE_KEEP_ACTIVE: {
			// Nothing to do.
		} break;
	}
}

void CollisionObject3D::_apply_enabled() {
	switch (disable_mode) {
		case DISABLE_MODE_REMOVE: {
			if (is_inside_tree()) {
				RID space = get_world_3d()->get_space();
				if (area) {
					PhysicsServer3D::get_singleton()->area_set_space(rid, space);
				} else {
					PhysicsServer3D::get_singleton()->body_set_space(rid, space);
				}
			}
		} break;

		case DISABLE_MODE_MAKE_STATIC: {
			if (!area && (body_mode != PhysicsServer3D::BODY_MODE_STATIC)) {
				PhysicsServer3D::get_singleton()->body_set_mode(rid, body_mode);
			}
		} break;

		case DISABLE_MODE_KEEP_ACTIVE: {
			// Nothing to do.
		} break;
	}
}

void CollisionObject3D::_input_event_call(Camera3D *p_camera, const Ref<InputEvent> &p_input_event, const Vector3 &p_pos, const Vector3 &p_normal, int p_shape) {
	GDVIRTUAL_CALL(_input_event, p_camera, p_input_event, p_pos, p_normal, p_shape);
	emit_signal(SceneStringNames::get_singleton()->input_event, p_camera, p_input_event, p_pos, p_normal, p_shape);
}

void CollisionObject3D::_mouse_enter() {
	if (get_script_instance()) {
		get_script_instance()->call(SceneStringNames::get_singleton()->_mouse_enter);
	}
	emit_signal(SceneStringNames::get_singleton()->mouse_entered);
}

void CollisionObject3D::_mouse_exit() {
	if (get_script_instance()) {
		get_script_instance()->call(SceneStringNames::get_singleton()->_mouse_exit);
	}
	emit_signal(SceneStringNames::get_singleton()->mouse_exited);
}

void CollisionObject3D::set_body_mode(PhysicsServer3D::BodyMode p_mode) {
	ERR_FAIL_COND(area);

	if (body_mode == p_mode) {
		return;
	}

	body_mode = p_mode;

	if (is_inside_tree() && !is_enabled() && (disable_mode == DISABLE_MODE_MAKE_STATIC)) {
		return;
	}

	PhysicsServer3D::get_singleton()->body_set_mode(rid, p_mode);
}

void CollisionObject3D::set_only_update_transform_changes(bool p_enable) {
	only_update_transform_changes = p_enable;
}

bool CollisionObject3D::is_only_update_transform_changes_enabled() const {
	return only_update_transform_changes;
}

void CollisionObject3D::_update_pickable() {
	if (!is_inside_tree()) {
		return;
	}

	bool pickable = ray_pickable && is_visible_in_tree();
	if (area) {
		PhysicsServer3D::get_singleton()->area_set_ray_pickable(rid, pickable);
	} else {
		PhysicsServer3D::get_singleton()->body_set_ray_pickable(rid, pickable);
	}
}

bool CollisionObject3D::_are_collision_shapes_visible() {
	return is_inside_tree() && get_tree()->is_debugging_collisions_hint() && !Engine::get_singleton()->is_editor_hint();
}

void CollisionObject3D::_update_shape_data(uint32_t p_owner) {
	if (_are_collision_shapes_visible()) {
		if (debug_shapes_to_update.is_empty()) {
			callable_mp(this, &CollisionObject3D::_update_debug_shapes).call_deferred({}, 0);
		}
		debug_shapes_to_update.insert(p_owner);
	}
}

void CollisionObject3D::_shape_changed(const Ref<Shape3D> &p_shape) {
	for (KeyValue<uint32_t, ShapeData> &E : shapes) {
		ShapeData &shapedata = E.value;
		ShapeData::ShapeBase *shapes = shapedata.shapes.ptrw();
		for (int i = 0; i < shapedata.shapes.size(); i++) {
			ShapeData::ShapeBase &s = shapes[i];
			if (s.shape == p_shape && s.debug_shape.is_valid()) {
				Ref<Mesh> mesh = s.shape->get_debug_mesh();
				RS::get_singleton()->instance_set_base(s.debug_shape, mesh->get_rid());
			}
		}
	}
}

void CollisionObject3D::_update_debug_shapes() {
	if (!is_inside_tree()) {
		debug_shapes_to_update.clear();
		return;
	}

	for (Set<uint32_t>::Element *shapedata_idx = debug_shapes_to_update.front(); shapedata_idx; shapedata_idx = shapedata_idx->next()) {
		if (shapes.has(shapedata_idx->get())) {
			ShapeData &shapedata = shapes[shapedata_idx->get()];
			ShapeData::ShapeBase *shapes = shapedata.shapes.ptrw();
			for (int i = 0; i < shapedata.shapes.size(); i++) {
				ShapeData::ShapeBase &s = shapes[i];
				if (s.shape.is_null() || shapedata.disabled) {
					if (s.debug_shape.is_valid()) {
						RS::get_singleton()->free(s.debug_shape);
						s.debug_shape = RID();
						--debug_shapes_count;
					}
					continue;
				}

				if (s.debug_shape.is_null()) {
					s.debug_shape = RS::get_singleton()->instance_create();
					RS::get_singleton()->instance_set_scenario(s.debug_shape, get_world_3d()->get_scenario());

					if (!s.shape->is_connected("changed", callable_mp(this, &CollisionObject3D::_shape_changed))) {
						s.shape->connect("changed", callable_mp(this, &CollisionObject3D::_shape_changed),
								varray(s.shape), CONNECT_DEFERRED);
					}

					++debug_shapes_count;
				}

				Ref<Mesh> mesh = s.shape->get_debug_mesh();
				RS::get_singleton()->instance_set_base(s.debug_shape, mesh->get_rid());
				RS::get_singleton()->instance_set_transform(s.debug_shape, get_global_transform() * shapedata.xform);
			}
		}
	}
	debug_shapes_to_update.clear();
}

void CollisionObject3D::_clear_debug_shapes() {
	for (KeyValue<uint32_t, ShapeData> &E : shapes) {
		ShapeData &shapedata = E.value;
		ShapeData::ShapeBase *shapes = shapedata.shapes.ptrw();
		for (int i = 0; i < shapedata.shapes.size(); i++) {
			ShapeData::ShapeBase &s = shapes[i];
			if (s.debug_shape.is_valid()) {
				RS::get_singleton()->free(s.debug_shape);
				s.debug_shape = RID();
				if (s.shape.is_valid() && s.shape->is_connected("changed", callable_mp(this, &CollisionObject3D::_update_shape_data))) {
					s.shape->disconnect("changed", callable_mp(this, &CollisionObject3D::_update_shape_data));
				}
			}
		}
	}
	debug_shapes_count = 0;
}

void CollisionObject3D::_on_transform_changed() {
	if (debug_shapes_count > 0 && !debug_shape_old_transform.is_equal_approx(get_global_transform())) {
		debug_shape_old_transform = get_global_transform();
		for (KeyValue<uint32_t, ShapeData> &E : shapes) {
			ShapeData &shapedata = E.value;
			const ShapeData::ShapeBase *shapes = shapedata.shapes.ptr();
			for (int i = 0; i < shapedata.shapes.size(); i++) {
				RS::get_singleton()->instance_set_transform(shapes[i].debug_shape, debug_shape_old_transform * shapedata.xform);
			}
		}
	}
}

void CollisionObject3D::set_ray_pickable(bool p_ray_pickable) {
	ray_pickable = p_ray_pickable;
	_update_pickable();
}

bool CollisionObject3D::is_ray_pickable() const {
	return ray_pickable;
}

void CollisionObject3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &CollisionObject3D::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &CollisionObject3D::get_collision_layer);
	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &CollisionObject3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &CollisionObject3D::get_collision_mask);
	ClassDB::bind_method(D_METHOD("set_collision_layer_value", "layer_number", "value"), &CollisionObject3D::set_collision_layer_value);
	ClassDB::bind_method(D_METHOD("get_collision_layer_value", "layer_number"), &CollisionObject3D::get_collision_layer_value);
	ClassDB::bind_method(D_METHOD("set_collision_mask_value", "layer_number", "value"), &CollisionObject3D::set_collision_mask_value);
	ClassDB::bind_method(D_METHOD("get_collision_mask_value", "layer_number"), &CollisionObject3D::get_collision_mask_value);
	ClassDB::bind_method(D_METHOD("set_disable_mode", "mode"), &CollisionObject3D::set_disable_mode);
	ClassDB::bind_method(D_METHOD("get_disable_mode"), &CollisionObject3D::get_disable_mode);
	ClassDB::bind_method(D_METHOD("set_ray_pickable", "ray_pickable"), &CollisionObject3D::set_ray_pickable);
	ClassDB::bind_method(D_METHOD("is_ray_pickable"), &CollisionObject3D::is_ray_pickable);
	ClassDB::bind_method(D_METHOD("set_capture_input_on_drag", "enable"), &CollisionObject3D::set_capture_input_on_drag);
	ClassDB::bind_method(D_METHOD("get_capture_input_on_drag"), &CollisionObject3D::get_capture_input_on_drag);
	ClassDB::bind_method(D_METHOD("get_rid"), &CollisionObject3D::get_rid);
	ClassDB::bind_method(D_METHOD("create_shape_owner", "owner"), &CollisionObject3D::create_shape_owner);
	ClassDB::bind_method(D_METHOD("remove_shape_owner", "owner_id"), &CollisionObject3D::remove_shape_owner);
	ClassDB::bind_method(D_METHOD("get_shape_owners"), &CollisionObject3D::_get_shape_owners);
	ClassDB::bind_method(D_METHOD("shape_owner_set_transform", "owner_id", "transform"), &CollisionObject3D::shape_owner_set_transform);
	ClassDB::bind_method(D_METHOD("shape_owner_get_transform", "owner_id"), &CollisionObject3D::shape_owner_get_transform);
	ClassDB::bind_method(D_METHOD("shape_owner_get_owner", "owner_id"), &CollisionObject3D::shape_owner_get_owner);
	ClassDB::bind_method(D_METHOD("shape_owner_set_disabled", "owner_id", "disabled"), &CollisionObject3D::shape_owner_set_disabled);
	ClassDB::bind_method(D_METHOD("is_shape_owner_disabled", "owner_id"), &CollisionObject3D::is_shape_owner_disabled);
	ClassDB::bind_method(D_METHOD("shape_owner_add_shape", "owner_id", "shape"), &CollisionObject3D::shape_owner_add_shape);
	ClassDB::bind_method(D_METHOD("shape_owner_get_shape_count", "owner_id"), &CollisionObject3D::shape_owner_get_shape_count);
	ClassDB::bind_method(D_METHOD("shape_owner_get_shape", "owner_id", "shape_id"), &CollisionObject3D::shape_owner_get_shape);
	ClassDB::bind_method(D_METHOD("shape_owner_get_shape_index", "owner_id", "shape_id"), &CollisionObject3D::shape_owner_get_shape_index);
	ClassDB::bind_method(D_METHOD("shape_owner_remove_shape", "owner_id", "shape_id"), &CollisionObject3D::shape_owner_remove_shape);
	ClassDB::bind_method(D_METHOD("shape_owner_clear_shapes", "owner_id"), &CollisionObject3D::shape_owner_clear_shapes);
	ClassDB::bind_method(D_METHOD("shape_find_owner", "shape_index"), &CollisionObject3D::shape_find_owner);

	GDVIRTUAL_BIND(_input_event, "camera", "event", "position", "normal", "shape_idx");

	ADD_SIGNAL(MethodInfo("input_event", PropertyInfo(Variant::OBJECT, "camera", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent"), PropertyInfo(Variant::VECTOR3, "position"), PropertyInfo(Variant::VECTOR3, "normal"), PropertyInfo(Variant::INT, "shape_idx")));
	ADD_SIGNAL(MethodInfo("mouse_entered"));
	ADD_SIGNAL(MethodInfo("mouse_exited"));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "disable_mode", PROPERTY_HINT_ENUM, "Remove,MakeStatic,KeepActive"), "set_disable_mode", "get_disable_mode");

	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");

	ADD_GROUP("Input", "input_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "input_ray_pickable"), "set_ray_pickable", "is_ray_pickable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "input_capture_on_drag"), "set_capture_input_on_drag", "get_capture_input_on_drag");

	BIND_ENUM_CONSTANT(DISABLE_MODE_REMOVE);
	BIND_ENUM_CONSTANT(DISABLE_MODE_MAKE_STATIC);
	BIND_ENUM_CONSTANT(DISABLE_MODE_KEEP_ACTIVE);
}

uint32_t CollisionObject3D::create_shape_owner(Object *p_owner) {
	ShapeData sd;
	uint32_t id;

	if (shapes.size() == 0) {
		id = 0;
	} else {
		id = shapes.back()->key() + 1;
	}

	sd.owner = p_owner;

	shapes[id] = sd;

	return id;
}

void CollisionObject3D::remove_shape_owner(uint32_t owner) {
	ERR_FAIL_COND(!shapes.has(owner));

	shape_owner_clear_shapes(owner);

	shapes.erase(owner);
}

void CollisionObject3D::shape_owner_set_disabled(uint32_t p_owner, bool p_disabled) {
	ERR_FAIL_COND(!shapes.has(p_owner));

	ShapeData &sd = shapes[p_owner];
	if (sd.disabled == p_disabled) {
		return;
	}
	sd.disabled = p_disabled;

	for (int i = 0; i < sd.shapes.size(); i++) {
		if (area) {
			PhysicsServer3D::get_singleton()->area_set_shape_disabled(rid, sd.shapes[i].index, p_disabled);
		} else {
			PhysicsServer3D::get_singleton()->body_set_shape_disabled(rid, sd.shapes[i].index, p_disabled);
		}
	}
	_update_shape_data(p_owner);
}

bool CollisionObject3D::is_shape_owner_disabled(uint32_t p_owner) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), false);

	return shapes[p_owner].disabled;
}

void CollisionObject3D::get_shape_owners(List<uint32_t> *r_owners) {
	for (const KeyValue<uint32_t, ShapeData> &E : shapes) {
		r_owners->push_back(E.key);
	}
}

Array CollisionObject3D::_get_shape_owners() {
	Array ret;
	for (const KeyValue<uint32_t, ShapeData> &E : shapes) {
		ret.push_back(E.key);
	}

	return ret;
}

void CollisionObject3D::shape_owner_set_transform(uint32_t p_owner, const Transform3D &p_transform) {
	ERR_FAIL_COND(!shapes.has(p_owner));

	ShapeData &sd = shapes[p_owner];
	sd.xform = p_transform;
	for (int i = 0; i < sd.shapes.size(); i++) {
		if (area) {
			PhysicsServer3D::get_singleton()->area_set_shape_transform(rid, sd.shapes[i].index, p_transform);
		} else {
			PhysicsServer3D::get_singleton()->body_set_shape_transform(rid, sd.shapes[i].index, p_transform);
		}
	}

	_update_shape_data(p_owner);
}
Transform3D CollisionObject3D::shape_owner_get_transform(uint32_t p_owner) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), Transform3D());

	return shapes[p_owner].xform;
}

Object *CollisionObject3D::shape_owner_get_owner(uint32_t p_owner) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), nullptr);

	return shapes[p_owner].owner;
}

void CollisionObject3D::shape_owner_add_shape(uint32_t p_owner, const Ref<Shape3D> &p_shape) {
	ERR_FAIL_COND(!shapes.has(p_owner));
	ERR_FAIL_COND(p_shape.is_null());

	ShapeData &sd = shapes[p_owner];
	ShapeData::ShapeBase s;
	s.index = total_subshapes;
	s.shape = p_shape;

	if (area) {
		PhysicsServer3D::get_singleton()->area_add_shape(rid, p_shape->get_rid(), sd.xform, sd.disabled);
	} else {
		PhysicsServer3D::get_singleton()->body_add_shape(rid, p_shape->get_rid(), sd.xform, sd.disabled);
	}
	sd.shapes.push_back(s);

	total_subshapes++;

	_update_shape_data(p_owner);
}

int CollisionObject3D::shape_owner_get_shape_count(uint32_t p_owner) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), 0);

	return shapes[p_owner].shapes.size();
}

Ref<Shape3D> CollisionObject3D::shape_owner_get_shape(uint32_t p_owner, int p_shape) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), Ref<Shape3D>());
	ERR_FAIL_INDEX_V(p_shape, shapes[p_owner].shapes.size(), Ref<Shape3D>());

	return shapes[p_owner].shapes[p_shape].shape;
}

int CollisionObject3D::shape_owner_get_shape_index(uint32_t p_owner, int p_shape) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), -1);
	ERR_FAIL_INDEX_V(p_shape, shapes[p_owner].shapes.size(), -1);

	return shapes[p_owner].shapes[p_shape].index;
}

void CollisionObject3D::shape_owner_remove_shape(uint32_t p_owner, int p_shape) {
	ERR_FAIL_COND(!shapes.has(p_owner));
	ERR_FAIL_INDEX(p_shape, shapes[p_owner].shapes.size());

	ShapeData::ShapeBase &s = shapes[p_owner].shapes.write[p_shape];
	int index_to_remove = s.index;

	if (area) {
		PhysicsServer3D::get_singleton()->area_remove_shape(rid, index_to_remove);
	} else {
		PhysicsServer3D::get_singleton()->body_remove_shape(rid, index_to_remove);
	}

	if (s.debug_shape.is_valid()) {
		RS::get_singleton()->free(s.debug_shape);
		if (s.shape.is_valid() && s.shape->is_connected("changed", callable_mp(this, &CollisionObject3D::_shape_changed))) {
			s.shape->disconnect("changed", callable_mp(this, &CollisionObject3D::_shape_changed));
		}
		--debug_shapes_count;
	}

	shapes[p_owner].shapes.remove(p_shape);

	for (KeyValue<uint32_t, ShapeData> &E : shapes) {
		for (int i = 0; i < E.value.shapes.size(); i++) {
			if (E.value.shapes[i].index > index_to_remove) {
				E.value.shapes.write[i].index -= 1;
			}
		}
	}

	total_subshapes--;
}

void CollisionObject3D::shape_owner_clear_shapes(uint32_t p_owner) {
	ERR_FAIL_COND(!shapes.has(p_owner));

	while (shape_owner_get_shape_count(p_owner) > 0) {
		shape_owner_remove_shape(p_owner, 0);
	}
}

uint32_t CollisionObject3D::shape_find_owner(int p_shape_index) const {
	ERR_FAIL_INDEX_V(p_shape_index, total_subshapes, UINT32_MAX);

	for (const KeyValue<uint32_t, ShapeData> &E : shapes) {
		for (int i = 0; i < E.value.shapes.size(); i++) {
			if (E.value.shapes[i].index == p_shape_index) {
				return E.key;
			}
		}
	}

	//in theory it should be unreachable
	ERR_FAIL_V_MSG(UINT32_MAX, "Can't find owner for shape index " + itos(p_shape_index) + ".");
}

CollisionObject3D::CollisionObject3D(RID p_rid, bool p_area) {
	rid = p_rid;
	area = p_area;
	set_notify_transform(true);

	if (p_area) {
		PhysicsServer3D::get_singleton()->area_attach_object_instance_id(rid, get_instance_id());
	} else {
		PhysicsServer3D::get_singleton()->body_attach_object_instance_id(rid, get_instance_id());
		PhysicsServer3D::get_singleton()->body_set_mode(rid, body_mode);
	}
}

void CollisionObject3D::set_capture_input_on_drag(bool p_capture) {
	capture_input_on_drag = p_capture;
}

bool CollisionObject3D::get_capture_input_on_drag() const {
	return capture_input_on_drag;
}

TypedArray<String> CollisionObject3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (shapes.is_empty()) {
		warnings.push_back(TTR("This node has no shape, so it can't collide or interact with other objects.\nConsider adding a CollisionShape3D or CollisionPolygon3D as a child to define its shape."));
	}

	return warnings;
}

CollisionObject3D::CollisionObject3D() {
	set_notify_transform(true);
	//owner=

	//set_transform_notify(true);
}

CollisionObject3D::~CollisionObject3D() {
	PhysicsServer3D::get_singleton()->free(rid);
}
