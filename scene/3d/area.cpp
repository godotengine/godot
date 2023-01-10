/**************************************************************************/
/*  area.cpp                                                              */
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

#include "area.h"
#include "scene/scene_string_names.h"
#include "servers/audio_server.h"
#include "servers/physics_server.h"

void Area::set_space_override_mode(SpaceOverride p_mode) {
	space_override = p_mode;
	PhysicsServer::get_singleton()->area_set_space_override_mode(get_rid(), PhysicsServer::AreaSpaceOverrideMode(p_mode));
}
Area::SpaceOverride Area::get_space_override_mode() const {
	return space_override;
}

void Area::set_gravity_is_point(bool p_enabled) {
	gravity_is_point = p_enabled;
	PhysicsServer::get_singleton()->area_set_param(get_rid(), PhysicsServer::AREA_PARAM_GRAVITY_IS_POINT, p_enabled);
}
bool Area::is_gravity_a_point() const {
	return gravity_is_point;
}

void Area::set_gravity_distance_scale(real_t p_scale) {
	gravity_distance_scale = p_scale;
	PhysicsServer::get_singleton()->area_set_param(get_rid(), PhysicsServer::AREA_PARAM_GRAVITY_DISTANCE_SCALE, p_scale);
}

real_t Area::get_gravity_distance_scale() const {
	return gravity_distance_scale;
}

void Area::set_gravity_vector(const Vector3 &p_vec) {
	gravity_vec = p_vec;
	PhysicsServer::get_singleton()->area_set_param(get_rid(), PhysicsServer::AREA_PARAM_GRAVITY_VECTOR, p_vec);
}
Vector3 Area::get_gravity_vector() const {
	return gravity_vec;
}

void Area::set_gravity(real_t p_gravity) {
	gravity = p_gravity;
	PhysicsServer::get_singleton()->area_set_param(get_rid(), PhysicsServer::AREA_PARAM_GRAVITY, p_gravity);
}
real_t Area::get_gravity() const {
	return gravity;
}
void Area::set_linear_damp(real_t p_linear_damp) {
	linear_damp = p_linear_damp;
	PhysicsServer::get_singleton()->area_set_param(get_rid(), PhysicsServer::AREA_PARAM_LINEAR_DAMP, p_linear_damp);
}
real_t Area::get_linear_damp() const {
	return linear_damp;
}

void Area::set_angular_damp(real_t p_angular_damp) {
	angular_damp = p_angular_damp;
	PhysicsServer::get_singleton()->area_set_param(get_rid(), PhysicsServer::AREA_PARAM_ANGULAR_DAMP, p_angular_damp);
}

real_t Area::get_angular_damp() const {
	return angular_damp;
}

void Area::set_priority(real_t p_priority) {
	priority = p_priority;
	PhysicsServer::get_singleton()->area_set_param(get_rid(), PhysicsServer::AREA_PARAM_PRIORITY, p_priority);
}
real_t Area::get_priority() const {
	return priority;
}

void Area::_body_enter_tree(ObjectID p_id) {
	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);

	Map<ObjectID, BodyState>::Element *E = body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(E->get().in_tree);

	E->get().in_tree = true;
	emit_signal(SceneStringNames::get_singleton()->body_entered, node);
	for (int i = 0; i < E->get().shapes.size(); i++) {
		emit_signal(SceneStringNames::get_singleton()->body_shape_entered, E->get().rid, node, E->get().shapes[i].body_shape, E->get().shapes[i].area_shape);
	}
}

void Area::_body_exit_tree(ObjectID p_id) {
	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);
	Map<ObjectID, BodyState>::Element *E = body_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(!E->get().in_tree);
	E->get().in_tree = false;
	emit_signal(SceneStringNames::get_singleton()->body_exited, node);
	for (int i = 0; i < E->get().shapes.size(); i++) {
		emit_signal(SceneStringNames::get_singleton()->body_shape_exited, E->get().rid, node, E->get().shapes[i].body_shape, E->get().shapes[i].area_shape);
	}
}

void Area::_body_inout(int p_status, const RID &p_body, int p_instance, int p_body_shape, int p_area_shape) {
	bool body_in = p_status == PhysicsServer::AREA_BODY_ADDED;
	ObjectID objid = p_instance;

	Object *obj = ObjectDB::get_instance(objid);
	Node *node = Object::cast_to<Node>(obj);

	Map<ObjectID, BodyState>::Element *E = body_map.find(objid);

	if (!body_in && !E) {
		return; //likely removed from the tree
	}

	locked = true;

	if (body_in) {
		if (!E) {
			E = body_map.insert(objid, BodyState());
			E->get().rid = p_body;
			E->get().rc = 0;
			E->get().in_tree = node && node->is_inside_tree();
			if (node) {
				node->connect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree, make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree, make_binds(objid));
				if (E->get().in_tree) {
					emit_signal(SceneStringNames::get_singleton()->body_entered, node);
				}
			}
		}
		E->get().rc++;
		if (node) {
			E->get().shapes.insert(ShapePair(p_body_shape, p_area_shape));
		}

		if (E->get().in_tree) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_entered, p_body, node, p_body_shape, p_area_shape);
		}

	} else {
		E->get().rc--;

		if (node) {
			E->get().shapes.erase(ShapePair(p_body_shape, p_area_shape));
		}

		bool in_tree = E->get().in_tree;
		if (E->get().rc == 0) {
			body_map.erase(E);
			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree);
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree);
				if (in_tree) {
					emit_signal(SceneStringNames::get_singleton()->body_exited, obj);
				}
			}
		}
		if (node && in_tree) {
			emit_signal(SceneStringNames::get_singleton()->body_shape_exited, p_body, obj, p_body_shape, p_area_shape);
		}
	}

	locked = false;
}

void Area::_clear_monitoring() {
	ERR_FAIL_COND_MSG(locked, "This function can't be used during the in/out signal.");

	{
		Map<ObjectID, BodyState> bmcopy = body_map;
		body_map.clear();
		//disconnect all monitored stuff

		for (Map<ObjectID, BodyState>::Element *E = bmcopy.front(); E; E = E->next()) {
			Object *obj = ObjectDB::get_instance(E->key());
			Node *node = Object::cast_to<Node>(obj);

			if (!node) { //node may have been deleted in previous frame or at other legiminate point
				continue;
			}

			node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_body_enter_tree);
			node->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_body_exit_tree);

			if (!E->get().in_tree) {
				continue;
			}

			for (int i = 0; i < E->get().shapes.size(); i++) {
				emit_signal(SceneStringNames::get_singleton()->body_shape_exited, E->get().rid, node, E->get().shapes[i].body_shape, E->get().shapes[i].area_shape);
			}

			emit_signal(SceneStringNames::get_singleton()->body_exited, node);
		}
	}

	{
		Map<ObjectID, AreaState> bmcopy = area_map;
		area_map.clear();
		//disconnect all monitored stuff

		for (Map<ObjectID, AreaState>::Element *E = bmcopy.front(); E; E = E->next()) {
			Object *obj = ObjectDB::get_instance(E->key());
			Node *node = Object::cast_to<Node>(obj);

			if (!node) { //node may have been deleted in previous frame or at other legiminate point
				continue;
			}

			node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_area_enter_tree);
			node->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_area_exit_tree);

			if (!E->get().in_tree) {
				continue;
			}

			for (int i = 0; i < E->get().shapes.size(); i++) {
				emit_signal(SceneStringNames::get_singleton()->area_shape_exited, E->get().rid, node, E->get().shapes[i].area_shape, E->get().shapes[i].self_shape);
			}

			emit_signal(SceneStringNames::get_singleton()->area_exited, obj);
		}
	}
}
void Area::_notification(int p_what) {
	if (p_what == NOTIFICATION_EXIT_TREE) {
		_clear_monitoring();
	}
}

void Area::set_monitoring(bool p_enable) {
	ERR_FAIL_COND_MSG(locked, "Function blocked during in/out signal. Use set_deferred(\"monitoring\", true/false).");

	if (p_enable == monitoring) {
		return;
	}

	monitoring = p_enable;

	if (monitoring) {
		PhysicsServer::get_singleton()->area_set_monitor_callback(get_rid(), this, SceneStringNames::get_singleton()->_body_inout);
		PhysicsServer::get_singleton()->area_set_area_monitor_callback(get_rid(), this, SceneStringNames::get_singleton()->_area_inout);
	} else {
		PhysicsServer::get_singleton()->area_set_monitor_callback(get_rid(), nullptr, StringName());
		PhysicsServer::get_singleton()->area_set_area_monitor_callback(get_rid(), nullptr, StringName());
		_clear_monitoring();
	}
}

void Area::_area_enter_tree(ObjectID p_id) {
	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);

	Map<ObjectID, AreaState>::Element *E = area_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(E->get().in_tree);

	E->get().in_tree = true;
	emit_signal(SceneStringNames::get_singleton()->area_entered, node);
	for (int i = 0; i < E->get().shapes.size(); i++) {
		emit_signal(SceneStringNames::get_singleton()->area_shape_entered, E->get().rid, node, E->get().shapes[i].area_shape, E->get().shapes[i].self_shape);
	}
}

void Area::_area_exit_tree(ObjectID p_id) {
	Object *obj = ObjectDB::get_instance(p_id);
	Node *node = Object::cast_to<Node>(obj);
	ERR_FAIL_COND(!node);
	Map<ObjectID, AreaState>::Element *E = area_map.find(p_id);
	ERR_FAIL_COND(!E);
	ERR_FAIL_COND(!E->get().in_tree);
	E->get().in_tree = false;
	emit_signal(SceneStringNames::get_singleton()->area_exited, node);
	for (int i = 0; i < E->get().shapes.size(); i++) {
		emit_signal(SceneStringNames::get_singleton()->area_shape_exited, E->get().rid, node, E->get().shapes[i].area_shape, E->get().shapes[i].self_shape);
	}
}

void Area::_area_inout(int p_status, const RID &p_area, int p_instance, int p_area_shape, int p_self_shape) {
	bool area_in = p_status == PhysicsServer::AREA_BODY_ADDED;
	ObjectID objid = p_instance;

	Object *obj = ObjectDB::get_instance(objid);
	Node *node = Object::cast_to<Node>(obj);

	Map<ObjectID, AreaState>::Element *E = area_map.find(objid);

	if (!area_in && !E) {
		return; //likely removed from the tree
	}

	locked = true;

	if (area_in) {
		if (!E) {
			E = area_map.insert(objid, AreaState());
			E->get().rid = p_area;
			E->get().rc = 0;
			E->get().in_tree = node && node->is_inside_tree();
			if (node) {
				node->connect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_area_enter_tree, make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_area_exit_tree, make_binds(objid));
				if (E->get().in_tree) {
					emit_signal(SceneStringNames::get_singleton()->area_entered, node);
				}
			}
		}
		E->get().rc++;
		if (node) {
			E->get().shapes.insert(AreaShapePair(p_area_shape, p_self_shape));
		}

		if (!node || E->get().in_tree) {
			emit_signal(SceneStringNames::get_singleton()->area_shape_entered, p_area, node, p_area_shape, p_self_shape);
		}

	} else {
		E->get().rc--;

		if (node) {
			E->get().shapes.erase(AreaShapePair(p_area_shape, p_self_shape));
		}

		bool in_tree = E->get().in_tree;
		if (E->get().rc == 0) {
			area_map.erase(E);
			if (node) {
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, this, SceneStringNames::get_singleton()->_area_enter_tree);
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, this, SceneStringNames::get_singleton()->_area_exit_tree);
				if (in_tree) {
					emit_signal(SceneStringNames::get_singleton()->area_exited, obj);
				}
			}
		}
		if (!node || in_tree) {
			emit_signal(SceneStringNames::get_singleton()->area_shape_exited, p_area, obj, p_area_shape, p_self_shape);
		}
	}

	locked = false;
}

bool Area::is_monitoring() const {
	return monitoring;
}

Array Area::get_overlapping_bodies() const {
	ERR_FAIL_COND_V(!monitoring, Array());
	Array ret;
	ret.resize(body_map.size());
	int idx = 0;
	for (const Map<ObjectID, BodyState>::Element *E = body_map.front(); E; E = E->next()) {
		Object *obj = ObjectDB::get_instance(E->key());
		if (!obj) {
			ret.resize(ret.size() - 1); //ops
		} else {
			ret[idx++] = obj;
		}
	}

	return ret;
}

void Area::set_monitorable(bool p_enable) {
	ERR_FAIL_COND_MSG(locked || (is_inside_tree() && PhysicsServer::get_singleton()->is_flushing_queries()), "Function blocked during in/out signal. Use set_deferred(\"monitorable\", true/false).");

	if (p_enable == monitorable) {
		return;
	}

	monitorable = p_enable;

	PhysicsServer::get_singleton()->area_set_monitorable(get_rid(), monitorable);
}

bool Area::is_monitorable() const {
	return monitorable;
}

Array Area::get_overlapping_areas() const {
	ERR_FAIL_COND_V(!monitoring, Array());
	Array ret;
	ret.resize(area_map.size());
	int idx = 0;
	for (const Map<ObjectID, AreaState>::Element *E = area_map.front(); E; E = E->next()) {
		Object *obj = ObjectDB::get_instance(E->key());
		if (!obj) {
			ret.resize(ret.size() - 1); //ops
		} else {
			ret[idx++] = obj;
		}
	}

	return ret;
}

bool Area::overlaps_area(Node *p_area) const {
	ERR_FAIL_NULL_V(p_area, false);
	const Map<ObjectID, AreaState>::Element *E = area_map.find(p_area->get_instance_id());
	if (!E) {
		return false;
	}
	return E->get().in_tree;
}

bool Area::overlaps_body(Node *p_body) const {
	ERR_FAIL_NULL_V(p_body, false);
	const Map<ObjectID, BodyState>::Element *E = body_map.find(p_body->get_instance_id());
	if (!E) {
		return false;
	}
	return E->get().in_tree;
}

void Area::set_audio_bus_override(bool p_override) {
	audio_bus_override = p_override;
}

bool Area::is_overriding_audio_bus() const {
	return audio_bus_override;
}

void Area::set_audio_bus(const StringName &p_audio_bus) {
	audio_bus = p_audio_bus;
}
StringName Area::get_audio_bus() const {
	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (AudioServer::get_singleton()->get_bus_name(i) == audio_bus) {
			return audio_bus;
		}
	}
	return "Master";
}

void Area::set_use_reverb_bus(bool p_enable) {
	use_reverb_bus = p_enable;
}
bool Area::is_using_reverb_bus() const {
	return use_reverb_bus;
}

void Area::set_reverb_bus(const StringName &p_audio_bus) {
	reverb_bus = p_audio_bus;
}
StringName Area::get_reverb_bus() const {
	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (AudioServer::get_singleton()->get_bus_name(i) == reverb_bus) {
			return reverb_bus;
		}
	}
	return "Master";
}

void Area::set_reverb_amount(float p_amount) {
	reverb_amount = p_amount;
}
float Area::get_reverb_amount() const {
	return reverb_amount;
}

void Area::set_reverb_uniformity(float p_uniformity) {
	reverb_uniformity = p_uniformity;
}
float Area::get_reverb_uniformity() const {
	return reverb_uniformity;
}

void Area::_validate_property(PropertyInfo &property) const {
	if (property.name == "audio_bus_name" || property.name == "reverb_bus_name") {
		String options;
		for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
			if (i > 0) {
				options += ",";
			}
			String name = AudioServer::get_singleton()->get_bus_name(i);
			options += name;
		}

		property.hint_string = options;
	}
}

void Area::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_body_enter_tree", "id"), &Area::_body_enter_tree);
	ClassDB::bind_method(D_METHOD("_body_exit_tree", "id"), &Area::_body_exit_tree);

	ClassDB::bind_method(D_METHOD("_area_enter_tree", "id"), &Area::_area_enter_tree);
	ClassDB::bind_method(D_METHOD("_area_exit_tree", "id"), &Area::_area_exit_tree);

	ClassDB::bind_method(D_METHOD("set_space_override_mode", "enable"), &Area::set_space_override_mode);
	ClassDB::bind_method(D_METHOD("get_space_override_mode"), &Area::get_space_override_mode);

	ClassDB::bind_method(D_METHOD("set_gravity_is_point", "enable"), &Area::set_gravity_is_point);
	ClassDB::bind_method(D_METHOD("is_gravity_a_point"), &Area::is_gravity_a_point);

	ClassDB::bind_method(D_METHOD("set_gravity_distance_scale", "distance_scale"), &Area::set_gravity_distance_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_distance_scale"), &Area::get_gravity_distance_scale);

	ClassDB::bind_method(D_METHOD("set_gravity_vector", "vector"), &Area::set_gravity_vector);
	ClassDB::bind_method(D_METHOD("get_gravity_vector"), &Area::get_gravity_vector);

	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &Area::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &Area::get_gravity);

	ClassDB::bind_method(D_METHOD("set_angular_damp", "angular_damp"), &Area::set_angular_damp);
	ClassDB::bind_method(D_METHOD("get_angular_damp"), &Area::get_angular_damp);

	ClassDB::bind_method(D_METHOD("set_linear_damp", "linear_damp"), &Area::set_linear_damp);
	ClassDB::bind_method(D_METHOD("get_linear_damp"), &Area::get_linear_damp);

	ClassDB::bind_method(D_METHOD("set_priority", "priority"), &Area::set_priority);
	ClassDB::bind_method(D_METHOD("get_priority"), &Area::get_priority);

	ClassDB::bind_method(D_METHOD("set_monitorable", "enable"), &Area::set_monitorable);
	ClassDB::bind_method(D_METHOD("is_monitorable"), &Area::is_monitorable);

	ClassDB::bind_method(D_METHOD("set_monitoring", "enable"), &Area::set_monitoring);
	ClassDB::bind_method(D_METHOD("is_monitoring"), &Area::is_monitoring);

	ClassDB::bind_method(D_METHOD("get_overlapping_bodies"), &Area::get_overlapping_bodies);
	ClassDB::bind_method(D_METHOD("get_overlapping_areas"), &Area::get_overlapping_areas);

	ClassDB::bind_method(D_METHOD("overlaps_body", "body"), &Area::overlaps_body);
	ClassDB::bind_method(D_METHOD("overlaps_area", "area"), &Area::overlaps_area);

	ClassDB::bind_method(D_METHOD("_body_inout"), &Area::_body_inout);
	ClassDB::bind_method(D_METHOD("_area_inout"), &Area::_area_inout);

	ClassDB::bind_method(D_METHOD("set_audio_bus_override", "enable"), &Area::set_audio_bus_override);
	ClassDB::bind_method(D_METHOD("is_overriding_audio_bus"), &Area::is_overriding_audio_bus);

	ClassDB::bind_method(D_METHOD("set_audio_bus", "name"), &Area::set_audio_bus);
	ClassDB::bind_method(D_METHOD("get_audio_bus"), &Area::get_audio_bus);

	ClassDB::bind_method(D_METHOD("set_use_reverb_bus", "enable"), &Area::set_use_reverb_bus);
	ClassDB::bind_method(D_METHOD("is_using_reverb_bus"), &Area::is_using_reverb_bus);

	ClassDB::bind_method(D_METHOD("set_reverb_bus", "name"), &Area::set_reverb_bus);
	ClassDB::bind_method(D_METHOD("get_reverb_bus"), &Area::get_reverb_bus);

	ClassDB::bind_method(D_METHOD("set_reverb_amount", "amount"), &Area::set_reverb_amount);
	ClassDB::bind_method(D_METHOD("get_reverb_amount"), &Area::get_reverb_amount);

	ClassDB::bind_method(D_METHOD("set_reverb_uniformity", "amount"), &Area::set_reverb_uniformity);
	ClassDB::bind_method(D_METHOD("get_reverb_uniformity"), &Area::get_reverb_uniformity);

	ADD_SIGNAL(MethodInfo("body_shape_entered", PropertyInfo(Variant::_RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("body_shape_exited", PropertyInfo(Variant::_RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("body_entered", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("body_exited", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node")));

	ADD_SIGNAL(MethodInfo("area_shape_entered", PropertyInfo(Variant::_RID, "area_rid"), PropertyInfo(Variant::OBJECT, "area", PROPERTY_HINT_RESOURCE_TYPE, "Area"), PropertyInfo(Variant::INT, "area_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("area_shape_exited", PropertyInfo(Variant::_RID, "area_rid"), PropertyInfo(Variant::OBJECT, "area", PROPERTY_HINT_RESOURCE_TYPE, "Area"), PropertyInfo(Variant::INT, "area_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("area_entered", PropertyInfo(Variant::OBJECT, "area", PROPERTY_HINT_RESOURCE_TYPE, "Area")));
	ADD_SIGNAL(MethodInfo("area_exited", PropertyInfo(Variant::OBJECT, "area", PROPERTY_HINT_RESOURCE_TYPE, "Area")));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "monitoring"), "set_monitoring", "is_monitoring");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "monitorable"), "set_monitorable", "is_monitorable");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "priority", PROPERTY_HINT_RANGE, "0,128,1"), "set_priority", "get_priority");

	ADD_GROUP("Physics Overrides", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "space_override", PROPERTY_HINT_ENUM, "Disabled,Combine,Combine-Replace,Replace,Replace-Combine"), "set_space_override_mode", "get_space_override_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity_point"), "set_gravity_is_point", "is_gravity_a_point");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "gravity_distance_scale", PROPERTY_HINT_EXP_RANGE, "0,1024,0.001,or_greater"), "set_gravity_distance_scale", "get_gravity_distance_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity_vec"), "set_gravity_vector", "get_gravity_vector");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "gravity", PROPERTY_HINT_RANGE, "-32,32,0.001,or_lesser,or_greater"), "set_gravity", "get_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "linear_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_linear_damp", "get_linear_damp");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_angular_damp", "get_angular_damp");

	ADD_GROUP("Audio Bus", "audio_bus_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "audio_bus_override"), "set_audio_bus_override", "is_overriding_audio_bus");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "audio_bus_name", PROPERTY_HINT_ENUM, ""), "set_audio_bus", "get_audio_bus");

	ADD_GROUP("Reverb Bus", "reverb_bus_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "reverb_bus_enable"), "set_use_reverb_bus", "is_using_reverb_bus");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "reverb_bus_name", PROPERTY_HINT_ENUM, ""), "set_reverb_bus", "get_reverb_bus");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "reverb_bus_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_reverb_amount", "get_reverb_amount");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "reverb_bus_uniformity", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_reverb_uniformity", "get_reverb_uniformity");

	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_DISABLED);
	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_COMBINE);
	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_COMBINE_REPLACE);
	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_REPLACE);
	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_REPLACE_COMBINE);
}

Area::Area() :
		CollisionObject(RID_PRIME(PhysicsServer::get_singleton()->area_create()), true) {
	space_override = SPACE_OVERRIDE_DISABLED;
	set_gravity(9.8);
	locked = false;
	set_gravity_vector(Vector3(0, -1, 0));
	gravity_is_point = false;
	gravity_distance_scale = 0;
	linear_damp = 0.1;
	angular_damp = 0.1;
	priority = 0;
	monitoring = false;
	monitorable = false;
	set_monitoring(true);
	set_monitorable(true);

	audio_bus_override = false;
	audio_bus = "Master";

	use_reverb_bus = false;
	reverb_bus = "Master";
	reverb_amount = 0.0;
	reverb_uniformity = 0.0;
}

Area::~Area() {
}
