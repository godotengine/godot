/*************************************************************************/
/*  collision_object.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "scene/scene_string_names.h"
#include "servers/physics_server.h"
void CollisionObject::_update_shapes_from_children() {

	shapes.clear();
	for (int i = 0; i < get_child_count(); i++) {

		Node *n = get_child(i);
		n->call("_add_to_collision_object", this);
	}

	_update_shapes();
}

void CollisionObject::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_WORLD: {

			if (area)
				PhysicsServer::get_singleton()->area_set_transform(rid, get_global_transform());
			else
				PhysicsServer::get_singleton()->body_set_state(rid, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());

			RID space = get_world()->get_space();
			if (area) {
				PhysicsServer::get_singleton()->area_set_space(rid, space);
			} else
				PhysicsServer::get_singleton()->body_set_space(rid, space);

			_update_pickable();
			//get space
		};

		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (area)
				PhysicsServer::get_singleton()->area_set_transform(rid, get_global_transform());
			else
				PhysicsServer::get_singleton()->body_set_state(rid, PhysicsServer::BODY_STATE_TRANSFORM, get_global_transform());

		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {

			_update_pickable();

		} break;
		case NOTIFICATION_EXIT_WORLD: {

			if (area) {
				PhysicsServer::get_singleton()->area_set_space(rid, RID());
			} else
				PhysicsServer::get_singleton()->body_set_space(rid, RID());

		} break;
	}
}

void CollisionObject::_update_shapes() {

	if (!rid.is_valid())
		return;

	if (area)
		PhysicsServer::get_singleton()->area_clear_shapes(rid);
	else
		PhysicsServer::get_singleton()->body_clear_shapes(rid);

	for (int i = 0; i < shapes.size(); i++) {

		if (shapes[i].shape.is_null())
			continue;
		if (area)
			PhysicsServer::get_singleton()->area_add_shape(rid, shapes[i].shape->get_rid(), shapes[i].xform);
		else {
			PhysicsServer::get_singleton()->body_add_shape(rid, shapes[i].shape->get_rid(), shapes[i].xform);
			if (shapes[i].trigger)
				PhysicsServer::get_singleton()->body_set_shape_as_trigger(rid, i, shapes[i].trigger);
		}
	}
}

bool CollisionObject::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	if (name == "shape_count") {

		shapes.resize(p_value);
		_update_shapes();
		_change_notify();

	} else if (name.begins_with("shapes/")) {

		int idx = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		if (what == "shape")
			set_shape(idx, RefPtr(p_value));
		else if (what == "transform")
			set_shape_transform(idx, p_value);
		else if (what == "trigger")
			set_shape_as_trigger(idx, p_value);

	} else
		return false;

	return true;
}

bool CollisionObject::_get(const StringName &p_name, Variant &r_ret) const {

	String name = p_name;

	if (name == "shape_count") {
		r_ret = shapes.size();
	} else if (name.begins_with("shapes/")) {

		int idx = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		if (what == "shape")
			r_ret = get_shape(idx);
		else if (what == "transform")
			r_ret = get_shape_transform(idx);
		else if (what == "trigger")
			r_ret = is_shape_set_as_trigger(idx);

	} else
		return false;

	return true;
}

void CollisionObject::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::INT, "shape_count", PROPERTY_HINT_RANGE, "0,256,1", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_NO_INSTANCE_STATE));

	for (int i = 0; i < shapes.size(); i++) {
		String path = "shapes/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::OBJECT, path + "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_NO_INSTANCE_STATE));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM, path + "transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_NO_INSTANCE_STATE));
		p_list->push_back(PropertyInfo(Variant::BOOL, path + "trigger", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_NO_INSTANCE_STATE));
	}
}

void CollisionObject::_input_event(Node *p_camera, const InputEvent &p_input_event, const Vector3 &p_pos, const Vector3 &p_normal, int p_shape) {

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

void CollisionObject::_update_pickable() {
	if (!is_inside_tree())
		return;
	bool pickable = ray_pickable && is_inside_tree() && is_visible_in_tree();
	if (area)
		PhysicsServer::get_singleton()->area_set_ray_pickable(rid, pickable);
	else
		PhysicsServer::get_singleton()->body_set_ray_pickable(rid, pickable);
}

void CollisionObject::set_ray_pickable(bool p_ray_pickable) {

	ray_pickable = p_ray_pickable;
	_update_pickable();
}

bool CollisionObject::is_ray_pickable() const {

	return ray_pickable;
}

void CollisionObject::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_shape", "shape:Shape", "transform"), &CollisionObject::add_shape, DEFVAL(Transform()));
	ClassDB::bind_method(D_METHOD("get_shape_count"), &CollisionObject::get_shape_count);
	ClassDB::bind_method(D_METHOD("set_shape", "shape_idx", "shape:Shape"), &CollisionObject::set_shape);
	ClassDB::bind_method(D_METHOD("set_shape_transform", "shape_idx", "transform"), &CollisionObject::set_shape_transform);
	//    ClassDB::bind_method(D_METHOD("set_shape_transform","shape_idx","transform"),&CollisionObject::set_shape_transform);
	ClassDB::bind_method(D_METHOD("set_shape_as_trigger", "shape_idx", "enable"), &CollisionObject::set_shape_as_trigger);
	ClassDB::bind_method(D_METHOD("is_shape_set_as_trigger", "shape_idx"), &CollisionObject::is_shape_set_as_trigger);
	ClassDB::bind_method(D_METHOD("get_shape:Shape", "shape_idx"), &CollisionObject::get_shape);
	ClassDB::bind_method(D_METHOD("get_shape_transform", "shape_idx"), &CollisionObject::get_shape_transform);
	ClassDB::bind_method(D_METHOD("remove_shape", "shape_idx"), &CollisionObject::remove_shape);
	ClassDB::bind_method(D_METHOD("clear_shapes"), &CollisionObject::clear_shapes);
	ClassDB::bind_method(D_METHOD("set_ray_pickable", "ray_pickable"), &CollisionObject::set_ray_pickable);
	ClassDB::bind_method(D_METHOD("is_ray_pickable"), &CollisionObject::is_ray_pickable);
	ClassDB::bind_method(D_METHOD("set_capture_input_on_drag", "enable"), &CollisionObject::set_capture_input_on_drag);
	ClassDB::bind_method(D_METHOD("get_capture_input_on_drag"), &CollisionObject::get_capture_input_on_drag);
	ClassDB::bind_method(D_METHOD("get_rid"), &CollisionObject::get_rid);
	BIND_VMETHOD(MethodInfo("_input_event", PropertyInfo(Variant::OBJECT, "camera"), PropertyInfo(Variant::INPUT_EVENT, "event"), PropertyInfo(Variant::VECTOR3, "click_pos"), PropertyInfo(Variant::VECTOR3, "click_normal"), PropertyInfo(Variant::INT, "shape_idx")));

	ADD_SIGNAL(MethodInfo("input_event", PropertyInfo(Variant::OBJECT, "camera"), PropertyInfo(Variant::INPUT_EVENT, "event"), PropertyInfo(Variant::VECTOR3, "click_pos"), PropertyInfo(Variant::VECTOR3, "click_normal"), PropertyInfo(Variant::INT, "shape_idx")));
	ADD_SIGNAL(MethodInfo("mouse_entered"));
	ADD_SIGNAL(MethodInfo("mouse_exited"));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "input_ray_pickable"), "set_ray_pickable", "is_ray_pickable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "input_capture_on_drag"), "set_capture_input_on_drag", "get_capture_input_on_drag");
}

void CollisionObject::add_shape(const Ref<Shape> &p_shape, const Transform &p_transform) {

	ShapeData sdata;
	sdata.shape = p_shape;
	sdata.xform = p_transform;
	shapes.push_back(sdata);
	_update_shapes();
}
int CollisionObject::get_shape_count() const {

	return shapes.size();
}
void CollisionObject::set_shape(int p_shape_idx, const Ref<Shape> &p_shape) {

	ERR_FAIL_INDEX(p_shape_idx, shapes.size());
	shapes[p_shape_idx].shape = p_shape;
	_update_shapes();
}

void CollisionObject::set_shape_transform(int p_shape_idx, const Transform &p_transform) {

	ERR_FAIL_INDEX(p_shape_idx, shapes.size());
	shapes[p_shape_idx].xform = p_transform;

	_update_shapes();
}

Ref<Shape> CollisionObject::get_shape(int p_shape_idx) const {

	ERR_FAIL_INDEX_V(p_shape_idx, shapes.size(), Ref<Shape>());
	return shapes[p_shape_idx].shape;
}
Transform CollisionObject::get_shape_transform(int p_shape_idx) const {

	ERR_FAIL_INDEX_V(p_shape_idx, shapes.size(), Transform());
	return shapes[p_shape_idx].xform;
}
void CollisionObject::remove_shape(int p_shape_idx) {

	ERR_FAIL_INDEX(p_shape_idx, shapes.size());
	shapes.remove(p_shape_idx);

	_update_shapes();
}

void CollisionObject::clear_shapes() {

	shapes.clear();

	_update_shapes();
}

void CollisionObject::set_shape_as_trigger(int p_shape_idx, bool p_trigger) {

	ERR_FAIL_INDEX(p_shape_idx, shapes.size());
	shapes[p_shape_idx].trigger = p_trigger;
	if (!area && rid.is_valid()) {

		PhysicsServer::get_singleton()->body_set_shape_as_trigger(rid, p_shape_idx, p_trigger);
	}
}

bool CollisionObject::is_shape_set_as_trigger(int p_shape_idx) const {

	ERR_FAIL_INDEX_V(p_shape_idx, shapes.size(), false);
	return shapes[p_shape_idx].trigger;
}

CollisionObject::CollisionObject(RID p_rid, bool p_area) {

	rid = p_rid;
	area = p_area;
	capture_input_on_drag = false;
	ray_pickable = true;
	set_notify_transform(true);
	if (p_area) {
		PhysicsServer::get_singleton()->area_attach_object_instance_ID(rid, get_instance_ID());
	} else {
		PhysicsServer::get_singleton()->body_attach_object_instance_ID(rid, get_instance_ID());
	}
	//set_transform_notify(true);
}

void CollisionObject::set_capture_input_on_drag(bool p_capture) {

	capture_input_on_drag = p_capture;
}

bool CollisionObject::get_capture_input_on_drag() const {

	return capture_input_on_drag;
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
