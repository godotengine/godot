/*************************************************************************/
/*  collision_object.cpp                                                 */
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

#include "collision_object.h"

#include "core/engine.h"
#include "mesh_instance.h"
#include "scene/scene_string_names.h"
#include "servers/physics_server.h"

void CollisionObject::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (_are_collision_shapes_visible()) {
				debug_shape_old_transform = get_global_transform();
				for (Map<uint32_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
					debug_shapes_to_update.insert(E->key());
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
				PhysicsServer::get_singleton()->area_set_transform(rid, get_global_transform());
			} else {
				PhysicsServer::get_singleton()->body_set_state(rid, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());
			}

			Ref<World> world_ref = get_world();
			ERR_FAIL_COND(!world_ref.is_valid());
			RID space = world_ref->get_space();
			if (area) {
				PhysicsServer::get_singleton()->area_set_space(rid, space);
			} else {
				PhysicsServer::get_singleton()->body_set_space(rid, space);
			}
			_update_pickable();
			//get space
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (only_update_transform_changes) {
				return;
			}

			if (area) {
				PhysicsServer::get_singleton()->area_set_transform(rid, get_global_transform());
			} else {
				PhysicsServer::get_singleton()->body_set_state(rid, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());
			}

			_on_transform_changed();

		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_pickable();

		} break;
		case NOTIFICATION_EXIT_WORLD: {
			if (area) {
				PhysicsServer::get_singleton()->area_set_space(rid, RID());
			} else {
				PhysicsServer::get_singleton()->body_set_space(rid, RID());
			}

		} break;
	}
}

void CollisionObject::set_collision_layer(uint32_t p_layer) {
	collision_layer = p_layer;
	if (area) {
		PhysicsServer::get_singleton()->area_set_collision_layer(get_rid(), p_layer);
	} else {
		PhysicsServer::get_singleton()->body_set_collision_layer(get_rid(), p_layer);
	}
}

uint32_t CollisionObject::get_collision_layer() const {
	return collision_layer;
}

void CollisionObject::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
	if (area) {
		PhysicsServer::get_singleton()->area_set_collision_mask(get_rid(), p_mask);
	} else {
		PhysicsServer::get_singleton()->body_set_collision_mask(get_rid(), p_mask);
	}
}

uint32_t CollisionObject::get_collision_mask() const {
	return collision_mask;
}

void CollisionObject::set_collision_layer_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision layer bit must be between 0 and 31 inclusive.");
	uint32_t collision_layer = get_collision_layer();
	if (p_value) {
		collision_layer |= 1 << p_bit;
	} else {
		collision_layer &= ~(1 << p_bit);
	}
	set_collision_layer(collision_layer);
}

bool CollisionObject::get_collision_layer_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision layer bit must be between 0 and 31 inclusive.");
	return get_collision_layer() & (1 << p_bit);
}

void CollisionObject::set_collision_mask_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision mask bit must be between 0 and 31 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool CollisionObject::get_collision_mask_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision mask bit must be between 0 and 31 inclusive.");
	return get_collision_mask() & (1 << p_bit);
}

void CollisionObject::_input_event(Node *p_camera, const Ref<InputEvent> &p_input_event, const Vector3 &p_pos, const Vector3 &p_normal, int p_shape) {
	if (get_script_instance()) {
		get_script_instance()->call(SceneStringNames::get_singleton()->_input_event, p_camera, p_input_event, p_pos, p_normal, p_shape);
	}
	emit_signal(SceneStringNames::get_singleton()->input_event, p_camera, p_input_event, p_pos, p_normal, p_shape);
}

void CollisionObject::_mouse_enter() {
	if (get_script_instance()) {
		get_script_instance()->call(SceneStringNames::get_singleton()->_mouse_enter);
	}
	emit_signal(SceneStringNames::get_singleton()->mouse_entered);
}

void CollisionObject::_mouse_exit() {
	if (get_script_instance()) {
		get_script_instance()->call(SceneStringNames::get_singleton()->_mouse_exit);
	}
	emit_signal(SceneStringNames::get_singleton()->mouse_exited);
}

void CollisionObject::set_only_update_transform_changes(bool p_enable) {
	only_update_transform_changes = p_enable;
}

void CollisionObject::_update_pickable() {
	if (!is_inside_tree()) {
		return;
	}

	bool pickable = ray_pickable && is_visible_in_tree();
	if (area) {
		PhysicsServer::get_singleton()->area_set_ray_pickable(rid, pickable);
	} else {
		PhysicsServer::get_singleton()->body_set_ray_pickable(rid, pickable);
	}
}

bool CollisionObject::_are_collision_shapes_visible() {
	return is_inside_tree() && get_tree()->is_debugging_collisions_hint() && !Engine::get_singleton()->is_editor_hint();
}

void CollisionObject::_update_shape_data(uint32_t p_owner) {
	if (_are_collision_shapes_visible()) {
		if (debug_shapes_to_update.empty()) {
			call_deferred("_update_debug_shapes");
		}
		debug_shapes_to_update.insert(p_owner);
	}
}

void CollisionObject::_shape_changed(const Ref<Shape> &p_shape) {
	for (Map<uint32_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
		ShapeData &shapedata = E->get();
		ShapeData::ShapeBase *shapes = shapedata.shapes.ptrw();
		for (int i = 0; i < shapedata.shapes.size(); i++) {
			ShapeData::ShapeBase &s = shapes[i];
			if (s.shape == p_shape && s.debug_shape.is_valid()) {
				Ref<Mesh> mesh = s.shape->get_debug_mesh();
				VS::get_singleton()->instance_set_base(s.debug_shape, mesh->get_rid());
			}
		}
	}
}

void CollisionObject::_update_debug_shapes() {
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
						VS::get_singleton()->free(s.debug_shape);
						s.debug_shape = RID();
						--debug_shapes_count;
					}
				}
				if (!s.debug_shape.is_valid()) {
					s.debug_shape = RID_PRIME(VS::get_singleton()->instance_create());
					VS::get_singleton()->instance_set_scenario(s.debug_shape, get_world()->get_scenario());

					if (!s.shape->is_connected("changed", this, "_shape_changed")) {
						s.shape->connect("changed", this, "_shape_changed", varray(s.shape), CONNECT_DEFERRED);
					}

					++debug_shapes_count;
				}

				Ref<Mesh> mesh = s.shape->get_debug_mesh();
				VS::get_singleton()->instance_set_base(s.debug_shape, mesh->get_rid());
				VS::get_singleton()->instance_set_transform(s.debug_shape, get_global_transform() * shapedata.xform);
				VS::get_singleton()->instance_set_portal_mode(s.debug_shape, VisualServer::INSTANCE_PORTAL_MODE_GLOBAL);
			}
		}
	}
	debug_shapes_to_update.clear();
}

void CollisionObject::_clear_debug_shapes() {
	for (Map<uint32_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
		ShapeData &shapedata = E->get();
		ShapeData::ShapeBase *shapes = shapedata.shapes.ptrw();
		for (int i = 0; i < shapedata.shapes.size(); i++) {
			ShapeData::ShapeBase &s = shapes[i];
			if (s.debug_shape.is_valid()) {
				VS::get_singleton()->free(s.debug_shape);
				s.debug_shape = RID();
				if (s.shape.is_valid() && s.shape->is_connected("changed", this, "_shape_changed")) {
					s.shape->disconnect("changed", this, "_shape_changed");
				}
			}
		}
	}

	debug_shapes_count = 0;
}

void CollisionObject::_on_transform_changed() {
	if (debug_shapes_count > 0 && !debug_shape_old_transform.is_equal_approx(get_global_transform())) {
		debug_shape_old_transform = get_global_transform();
		for (Map<uint32_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
			ShapeData &shapedata = E->get();
			const ShapeData::ShapeBase *shapes = shapedata.shapes.ptr();
			for (int i = 0; i < shapedata.shapes.size(); i++) {
				VS::get_singleton()->instance_set_transform(shapes[i].debug_shape, debug_shape_old_transform * shapedata.xform);
			}
		}
	}
}

void CollisionObject::set_ray_pickable(bool p_ray_pickable) {
	ray_pickable = p_ray_pickable;
	_update_pickable();
}

bool CollisionObject::is_ray_pickable() const {
	return ray_pickable;
}

void CollisionObject::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &CollisionObject::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &CollisionObject::get_collision_layer);
	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &CollisionObject::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &CollisionObject::get_collision_mask);
	ClassDB::bind_method(D_METHOD("set_collision_layer_bit", "bit", "value"), &CollisionObject::set_collision_layer_bit);
	ClassDB::bind_method(D_METHOD("get_collision_layer_bit", "bit"), &CollisionObject::get_collision_layer_bit);
	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &CollisionObject::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &CollisionObject::get_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("set_ray_pickable", "ray_pickable"), &CollisionObject::set_ray_pickable);
	ClassDB::bind_method(D_METHOD("is_ray_pickable"), &CollisionObject::is_ray_pickable);
	ClassDB::bind_method(D_METHOD("set_capture_input_on_drag", "enable"), &CollisionObject::set_capture_input_on_drag);
	ClassDB::bind_method(D_METHOD("get_capture_input_on_drag"), &CollisionObject::get_capture_input_on_drag);
	ClassDB::bind_method(D_METHOD("get_rid"), &CollisionObject::get_rid);
	ClassDB::bind_method(D_METHOD("create_shape_owner", "owner"), &CollisionObject::create_shape_owner);
	ClassDB::bind_method(D_METHOD("remove_shape_owner", "owner_id"), &CollisionObject::remove_shape_owner);
	ClassDB::bind_method(D_METHOD("get_shape_owners"), &CollisionObject::_get_shape_owners);
	ClassDB::bind_method(D_METHOD("shape_owner_set_transform", "owner_id", "transform"), &CollisionObject::shape_owner_set_transform);
	ClassDB::bind_method(D_METHOD("shape_owner_get_transform", "owner_id"), &CollisionObject::shape_owner_get_transform);
	ClassDB::bind_method(D_METHOD("shape_owner_get_owner", "owner_id"), &CollisionObject::shape_owner_get_owner);
	ClassDB::bind_method(D_METHOD("shape_owner_set_disabled", "owner_id", "disabled"), &CollisionObject::shape_owner_set_disabled);
	ClassDB::bind_method(D_METHOD("is_shape_owner_disabled", "owner_id"), &CollisionObject::is_shape_owner_disabled);
	ClassDB::bind_method(D_METHOD("shape_owner_add_shape", "owner_id", "shape"), &CollisionObject::shape_owner_add_shape);
	ClassDB::bind_method(D_METHOD("shape_owner_get_shape_count", "owner_id"), &CollisionObject::shape_owner_get_shape_count);
	ClassDB::bind_method(D_METHOD("shape_owner_get_shape", "owner_id", "shape_id"), &CollisionObject::shape_owner_get_shape);
	ClassDB::bind_method(D_METHOD("shape_owner_get_shape_index", "owner_id", "shape_id"), &CollisionObject::shape_owner_get_shape_index);
	ClassDB::bind_method(D_METHOD("shape_owner_remove_shape", "owner_id", "shape_id"), &CollisionObject::shape_owner_remove_shape);
	ClassDB::bind_method(D_METHOD("shape_owner_clear_shapes", "owner_id"), &CollisionObject::shape_owner_clear_shapes);
	ClassDB::bind_method(D_METHOD("shape_find_owner", "shape_index"), &CollisionObject::shape_find_owner);

	ClassDB::bind_method(D_METHOD("_update_debug_shapes"), &CollisionObject::_update_debug_shapes);
	ClassDB::bind_method(D_METHOD("_shape_changed", "shape"), &CollisionObject::_shape_changed);

	BIND_VMETHOD(MethodInfo("_input_event", PropertyInfo(Variant::OBJECT, "camera"), PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent"), PropertyInfo(Variant::VECTOR3, "position"), PropertyInfo(Variant::VECTOR3, "normal"), PropertyInfo(Variant::INT, "shape_idx")));

	ADD_SIGNAL(MethodInfo("input_event", PropertyInfo(Variant::OBJECT, "camera", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent"), PropertyInfo(Variant::VECTOR3, "position"), PropertyInfo(Variant::VECTOR3, "normal"), PropertyInfo(Variant::INT, "shape_idx")));
	ADD_SIGNAL(MethodInfo("mouse_entered"));
	ADD_SIGNAL(MethodInfo("mouse_exited"));

	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");

	ADD_GROUP("Input", "input_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "input_ray_pickable"), "set_ray_pickable", "is_ray_pickable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "input_capture_on_drag"), "set_capture_input_on_drag", "get_capture_input_on_drag");
}

uint32_t CollisionObject::create_shape_owner(Object *p_owner) {
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

void CollisionObject::remove_shape_owner(uint32_t owner) {
	ERR_FAIL_COND(!shapes.has(owner));

	shape_owner_clear_shapes(owner);

	shapes.erase(owner);
}

void CollisionObject::shape_owner_set_disabled(uint32_t p_owner, bool p_disabled) {
	ERR_FAIL_COND(!shapes.has(p_owner));

	ShapeData &sd = shapes[p_owner];
	if (sd.disabled == p_disabled) {
		return;
	}
	sd.disabled = p_disabled;

	for (int i = 0; i < sd.shapes.size(); i++) {
		if (area) {
			PhysicsServer::get_singleton()->area_set_shape_disabled(rid, sd.shapes[i].index, p_disabled);
		} else {
			PhysicsServer::get_singleton()->body_set_shape_disabled(rid, sd.shapes[i].index, p_disabled);
		}
	}
	_update_shape_data(p_owner);
}

bool CollisionObject::is_shape_owner_disabled(uint32_t p_owner) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), false);

	return shapes[p_owner].disabled;
}

void CollisionObject::get_shape_owners(List<uint32_t> *r_owners) {
	for (Map<uint32_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
		r_owners->push_back(E->key());
	}
}

Array CollisionObject::_get_shape_owners() {
	Array ret;
	for (Map<uint32_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}

	return ret;
}

void CollisionObject::shape_owner_set_transform(uint32_t p_owner, const Transform &p_transform) {
	ERR_FAIL_COND(!shapes.has(p_owner));

	ShapeData &sd = shapes[p_owner];
	sd.xform = p_transform;
	for (int i = 0; i < sd.shapes.size(); i++) {
		if (area) {
			PhysicsServer::get_singleton()->area_set_shape_transform(rid, sd.shapes[i].index, p_transform);
		} else {
			PhysicsServer::get_singleton()->body_set_shape_transform(rid, sd.shapes[i].index, p_transform);
		}
	}

	_update_shape_data(p_owner);
}
Transform CollisionObject::shape_owner_get_transform(uint32_t p_owner) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), Transform());

	return shapes[p_owner].xform;
}

Object *CollisionObject::shape_owner_get_owner(uint32_t p_owner) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), nullptr);

	return shapes[p_owner].owner;
}

void CollisionObject::shape_owner_add_shape(uint32_t p_owner, const Ref<Shape> &p_shape) {
	ERR_FAIL_COND(!shapes.has(p_owner));
	ERR_FAIL_COND(p_shape.is_null());

	ShapeData &sd = shapes[p_owner];
	ShapeData::ShapeBase s;
	s.index = total_subshapes;
	s.shape = p_shape;

	if (area) {
		PhysicsServer::get_singleton()->area_add_shape(rid, p_shape->get_rid(), sd.xform, sd.disabled);
	} else {
		PhysicsServer::get_singleton()->body_add_shape(rid, p_shape->get_rid(), sd.xform, sd.disabled);
	}
	sd.shapes.push_back(s);

	total_subshapes++;

	_update_shape_data(p_owner);
}
int CollisionObject::shape_owner_get_shape_count(uint32_t p_owner) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), 0);

	return shapes[p_owner].shapes.size();
}
Ref<Shape> CollisionObject::shape_owner_get_shape(uint32_t p_owner, int p_shape) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), Ref<Shape>());
	ERR_FAIL_INDEX_V(p_shape, shapes[p_owner].shapes.size(), Ref<Shape>());

	return shapes[p_owner].shapes[p_shape].shape;
}
int CollisionObject::shape_owner_get_shape_index(uint32_t p_owner, int p_shape) const {
	ERR_FAIL_COND_V(!shapes.has(p_owner), -1);
	ERR_FAIL_INDEX_V(p_shape, shapes[p_owner].shapes.size(), -1);

	return shapes[p_owner].shapes[p_shape].index;
}

void CollisionObject::shape_owner_remove_shape(uint32_t p_owner, int p_shape) {
	ERR_FAIL_COND(!shapes.has(p_owner));
	ERR_FAIL_INDEX(p_shape, shapes[p_owner].shapes.size());

	ShapeData::ShapeBase &s = shapes[p_owner].shapes.write[p_shape];
	int index_to_remove = s.index;

	if (area) {
		PhysicsServer::get_singleton()->area_remove_shape(rid, index_to_remove);
	} else {
		PhysicsServer::get_singleton()->body_remove_shape(rid, index_to_remove);
	}

	if (s.debug_shape.is_valid()) {
		VS::get_singleton()->free(s.debug_shape);
		if (s.shape.is_valid() && s.shape->is_connected("changed", this, "_shape_changed")) {
			s.shape->disconnect("changed", this, "_shape_changed");
		}
		--debug_shapes_count;
	}

	shapes[p_owner].shapes.remove(p_shape);

	for (Map<uint32_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
		for (int i = 0; i < E->get().shapes.size(); i++) {
			if (E->get().shapes[i].index > index_to_remove) {
				E->get().shapes.write[i].index -= 1;
			}
		}
	}

	total_subshapes--;
}

void CollisionObject::shape_owner_clear_shapes(uint32_t p_owner) {
	ERR_FAIL_COND(!shapes.has(p_owner));

	while (shape_owner_get_shape_count(p_owner) > 0) {
		shape_owner_remove_shape(p_owner, 0);
	}
}

uint32_t CollisionObject::shape_find_owner(int p_shape_index) const {
	ERR_FAIL_INDEX_V(p_shape_index, total_subshapes, UINT32_MAX);

	for (const Map<uint32_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
		for (int i = 0; i < E->get().shapes.size(); i++) {
			if (E->get().shapes[i].index == p_shape_index) {
				return E->key();
			}
		}
	}

	//in theory it should be unreachable
	ERR_FAIL_V_MSG(UINT32_MAX, "Can't find owner for shape index " + itos(p_shape_index) + ".");
}

CollisionObject::CollisionObject(RID p_rid, bool p_area) {
	rid = p_rid;
	area = p_area;
	capture_input_on_drag = false;
	ray_pickable = true;
	set_notify_transform(true);
	total_subshapes = 0;

	if (p_area) {
		PhysicsServer::get_singleton()->area_attach_object_instance_id(rid, get_instance_id());
	} else {
		PhysicsServer::get_singleton()->body_attach_object_instance_id(rid, get_instance_id());
	}
	//set_transform_notify(true);
}

void CollisionObject::set_capture_input_on_drag(bool p_capture) {
	capture_input_on_drag = p_capture;
}

bool CollisionObject::get_capture_input_on_drag() const {
	return capture_input_on_drag;
}

String CollisionObject::get_configuration_warning() const {
	String warning = Spatial::get_configuration_warning();

	if (shapes.empty()) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("This node has no shape, so it can't collide or interact with other objects.\nConsider adding a CollisionShape or CollisionPolygon as a child to define its shape.");
	}

	return warning;
}

CollisionObject::CollisionObject() {
	capture_input_on_drag = false;
	ray_pickable = true;
	set_notify_transform(true);
	//owner=

	//set_transform_notify(true);
}

CollisionObject::~CollisionObject() {
	PhysicsServer::get_singleton()->free(rid);
}
