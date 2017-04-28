/*************************************************************************/
/*  collision_object_2d.cpp                                              */
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
#include "collision_object_2d.h"
#include "scene/scene_string_names.h"
#include "servers/physics_2d_server.h"

void CollisionObject2D::_update_shapes_from_children() {

	shapes.clear();
	for (int i = 0; i < get_child_count(); i++) {

		Node *n = get_child(i);
		n->call("_add_to_collision_object", this);
	}

	_update_shapes();
}

void CollisionObject2D::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			if (area)
				Physics2DServer::get_singleton()->area_set_transform(rid, get_global_transform());
			else
				Physics2DServer::get_singleton()->body_set_state(rid, Physics2DServer::BODY_STATE_TRANSFORM, get_global_transform());

			RID space = get_world_2d()->get_space();
			if (area) {
				Physics2DServer::get_singleton()->area_set_space(rid, space);
			} else
				Physics2DServer::get_singleton()->body_set_space(rid, space);

			_update_pickable();

			//get space
		}

		case NOTIFICATION_VISIBILITY_CHANGED: {

			_update_pickable();
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (area)
				Physics2DServer::get_singleton()->area_set_transform(rid, get_global_transform());
			else
				Physics2DServer::get_singleton()->body_set_state(rid, Physics2DServer::BODY_STATE_TRANSFORM, get_global_transform());

		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (area) {
				Physics2DServer::get_singleton()->area_set_space(rid, RID());
			} else
				Physics2DServer::get_singleton()->body_set_space(rid, RID());

		} break;
	}
}

void CollisionObject2D::_update_shapes() {

	if (!rid.is_valid())
		return;

	if (area)
		Physics2DServer::get_singleton()->area_clear_shapes(rid);
	else
		Physics2DServer::get_singleton()->body_clear_shapes(rid);

	for (int i = 0; i < shapes.size(); i++) {

		if (shapes[i].shape.is_null())
			continue;
		if (area)
			Physics2DServer::get_singleton()->area_add_shape(rid, shapes[i].shape->get_rid(), shapes[i].xform);
		else {
			Physics2DServer::get_singleton()->body_add_shape(rid, shapes[i].shape->get_rid(), shapes[i].xform);
			if (shapes[i].trigger)
				Physics2DServer::get_singleton()->body_set_shape_as_trigger(rid, i, shapes[i].trigger);
		}
	}
}

bool CollisionObject2D::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	if (name.begins_with("shapes/")) {

		int idx = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		if (what == "shape") {
			if (idx >= shapes.size())
				add_shape(RefPtr(p_value));
			else
				set_shape(idx, RefPtr(p_value));
		} else if (what == "transform")
			set_shape_transform(idx, p_value);
		else if (what == "trigger")
			set_shape_as_trigger(idx, p_value);
	} else
		return false;

	return true;
}

bool CollisionObject2D::_get(const StringName &p_name, Variant &r_ret) const {

	String name = p_name;

	if (name.begins_with("shapes/")) {

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

void CollisionObject2D::_get_property_list(List<PropertyInfo> *p_list) const {

	//p_list->push_back( PropertyInfo(Variant::INT,"shape_count",PROPERTY_HINT_RANGE,"0,256,1",PROPERTY_USAGE_NOEDITOR|PROPERTY_USAGE_NO_INSTANCE_STATE) );

	for (int i = 0; i < shapes.size(); i++) {
		String path = "shapes/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::OBJECT, path + "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape2D", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_NO_INSTANCE_STATE));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM, path + "transform", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_NO_INSTANCE_STATE));
		p_list->push_back(PropertyInfo(Variant::BOOL, path + "trigger", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_NO_INSTANCE_STATE));
	}
}

void CollisionObject2D::set_pickable(bool p_enabled) {

	if (pickable == p_enabled)
		return;

	pickable = p_enabled;
	_update_pickable();
}

bool CollisionObject2D::is_pickable() const {

	return pickable;
}

void CollisionObject2D::_input_event(Node *p_viewport, const InputEvent &p_input_event, int p_shape) {

	if (get_script_instance()) {
		get_script_instance()->call(SceneStringNames::get_singleton()->_input_event, p_viewport, p_input_event, p_shape);
	}
	emit_signal(SceneStringNames::get_singleton()->input_event, p_viewport, p_input_event, p_shape);
}

void CollisionObject2D::_mouse_enter() {

	if (get_script_instance()) {
		get_script_instance()->call(SceneStringNames::get_singleton()->_mouse_enter);
	}
	emit_signal(SceneStringNames::get_singleton()->mouse_entered);
}

void CollisionObject2D::_mouse_exit() {

	if (get_script_instance()) {
		get_script_instance()->call(SceneStringNames::get_singleton()->_mouse_exit);
	}
	emit_signal(SceneStringNames::get_singleton()->mouse_exited);
}

void CollisionObject2D::_update_pickable() {
	if (!is_inside_tree())
		return;
	bool pickable = this->pickable && is_inside_tree() && is_visible_in_tree();
	if (area)
		Physics2DServer::get_singleton()->area_set_pickable(rid, pickable);
	else
		Physics2DServer::get_singleton()->body_set_pickable(rid, pickable);
}

void CollisionObject2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_shape", "shape:Shape2D", "transform"), &CollisionObject2D::add_shape, DEFVAL(Transform2D()));
	ClassDB::bind_method(D_METHOD("get_shape_count"), &CollisionObject2D::get_shape_count);
	ClassDB::bind_method(D_METHOD("set_shape", "shape_idx", "shape:Shape"), &CollisionObject2D::set_shape);
	ClassDB::bind_method(D_METHOD("set_shape_transform", "shape_idx", "transform"), &CollisionObject2D::set_shape_transform);
	ClassDB::bind_method(D_METHOD("set_shape_as_trigger", "shape_idx", "enable"), &CollisionObject2D::set_shape_as_trigger);
	ClassDB::bind_method(D_METHOD("get_shape:Shape2D", "shape_idx"), &CollisionObject2D::get_shape);
	ClassDB::bind_method(D_METHOD("get_shape_transform", "shape_idx"), &CollisionObject2D::get_shape_transform);
	ClassDB::bind_method(D_METHOD("is_shape_set_as_trigger", "shape_idx"), &CollisionObject2D::is_shape_set_as_trigger);
	ClassDB::bind_method(D_METHOD("remove_shape", "shape_idx"), &CollisionObject2D::remove_shape);
	ClassDB::bind_method(D_METHOD("clear_shapes"), &CollisionObject2D::clear_shapes);
	ClassDB::bind_method(D_METHOD("get_rid"), &CollisionObject2D::get_rid);

	ClassDB::bind_method(D_METHOD("set_pickable", "enabled"), &CollisionObject2D::set_pickable);
	ClassDB::bind_method(D_METHOD("is_pickable"), &CollisionObject2D::is_pickable);

	BIND_VMETHOD(MethodInfo("_input_event", PropertyInfo(Variant::OBJECT, "viewport"), PropertyInfo(Variant::INPUT_EVENT, "event"), PropertyInfo(Variant::INT, "shape_idx")));

	ADD_SIGNAL(MethodInfo("input_event", PropertyInfo(Variant::OBJECT, "viewport"), PropertyInfo(Variant::INPUT_EVENT, "event"), PropertyInfo(Variant::INT, "shape_idx")));
	ADD_SIGNAL(MethodInfo("mouse_entered"));
	ADD_SIGNAL(MethodInfo("mouse_exited"));

	ADD_GROUP("Pickable", "input_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "input_pickable"), "set_pickable", "is_pickable");
	ADD_GROUP("", "");
}

void CollisionObject2D::add_shape(const Ref<Shape2D> &p_shape, const Transform2D &p_transform) {

	ERR_FAIL_COND(p_shape.is_null());

	ShapeData sdata;
	sdata.shape = p_shape;
	sdata.xform = p_transform;
	sdata.trigger = false;

	if (area)
		Physics2DServer::get_singleton()->area_add_shape(get_rid(), p_shape->get_rid(), p_transform);
	else
		Physics2DServer::get_singleton()->body_add_shape(get_rid(), p_shape->get_rid(), p_transform);

	shapes.push_back(sdata);
}
int CollisionObject2D::get_shape_count() const {

	return shapes.size();
}
void CollisionObject2D::set_shape(int p_shape_idx, const Ref<Shape2D> &p_shape) {

	ERR_FAIL_INDEX(p_shape_idx, shapes.size());
	ERR_FAIL_COND(p_shape.is_null());

	shapes[p_shape_idx].shape = p_shape;
	if (area)
		Physics2DServer::get_singleton()->area_set_shape(get_rid(), p_shape_idx, p_shape->get_rid());
	else
		Physics2DServer::get_singleton()->body_set_shape(get_rid(), p_shape_idx, p_shape->get_rid());

	//_update_shapes();
}

void CollisionObject2D::set_shape_transform(int p_shape_idx, const Transform2D &p_transform) {

	ERR_FAIL_INDEX(p_shape_idx, shapes.size());
	shapes[p_shape_idx].xform = p_transform;

	if (area)
		Physics2DServer::get_singleton()->area_set_shape_transform(get_rid(), p_shape_idx, p_transform);
	else
		Physics2DServer::get_singleton()->body_set_shape_transform(get_rid(), p_shape_idx, p_transform);

	//_update_shapes();
}

Ref<Shape2D> CollisionObject2D::get_shape(int p_shape_idx) const {

	ERR_FAIL_INDEX_V(p_shape_idx, shapes.size(), Ref<Shape2D>());
	return shapes[p_shape_idx].shape;
}
Transform2D CollisionObject2D::get_shape_transform(int p_shape_idx) const {

	ERR_FAIL_INDEX_V(p_shape_idx, shapes.size(), Transform2D());
	return shapes[p_shape_idx].xform;
}
void CollisionObject2D::remove_shape(int p_shape_idx) {

	ERR_FAIL_INDEX(p_shape_idx, shapes.size());
	shapes.remove(p_shape_idx);

	_update_shapes();
}

void CollisionObject2D::set_shape_as_trigger(int p_shape_idx, bool p_trigger) {

	ERR_FAIL_INDEX(p_shape_idx, shapes.size());
	shapes[p_shape_idx].trigger = p_trigger;
	if (!area && rid.is_valid()) {

		Physics2DServer::get_singleton()->body_set_shape_as_trigger(rid, p_shape_idx, p_trigger);
	}
}

bool CollisionObject2D::is_shape_set_as_trigger(int p_shape_idx) const {

	ERR_FAIL_INDEX_V(p_shape_idx, shapes.size(), false);
	return shapes[p_shape_idx].trigger;
}

void CollisionObject2D::clear_shapes() {

	shapes.clear();

	_update_shapes();
}

CollisionObject2D::CollisionObject2D(RID p_rid, bool p_area) {

	rid = p_rid;
	area = p_area;
	pickable = true;
	set_notify_transform(true);

	if (p_area) {

		Physics2DServer::get_singleton()->area_attach_object_instance_ID(rid, get_instance_ID());
	} else {
		Physics2DServer::get_singleton()->body_attach_object_instance_ID(rid, get_instance_ID());
	}
}

CollisionObject2D::CollisionObject2D() {

	//owner=
	set_notify_transform(true);
}

CollisionObject2D::~CollisionObject2D() {

	Physics2DServer::get_singleton()->free(rid);
}
