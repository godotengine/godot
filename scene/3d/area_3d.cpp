/*************************************************************************/
/*  area_3d.cpp                                                          */
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

#include "area_3d.h"

#include "scene/scene_string_names.h"
#include "servers/audio_server.h"

void Area3D::set_space_override_mode(SpaceOverride p_mode) {
	space_override = p_mode;
	PhysicsServer3D::get_singleton()->area_set_space_override_mode(get_rid(), PhysicsServer3D::AreaSpaceOverrideMode(p_mode));
}

Area3D::SpaceOverride Area3D::get_space_override_mode() const {
	return space_override;
}

void Area3D::set_gravity_is_point(bool p_enabled) {
	gravity_is_point = p_enabled;
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_GRAVITY_IS_POINT, p_enabled);
}

bool Area3D::is_gravity_a_point() const {
	return gravity_is_point;
}

void Area3D::set_gravity_distance_scale(real_t p_scale) {
	gravity_distance_scale = p_scale;
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_GRAVITY_DISTANCE_SCALE, p_scale);
}

real_t Area3D::get_gravity_distance_scale() const {
	return gravity_distance_scale;
}

void Area3D::set_gravity_vector(const Vector3 &p_vec) {
	gravity_vec = p_vec;
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_GRAVITY_VECTOR, p_vec);
}

Vector3 Area3D::get_gravity_vector() const {
	return gravity_vec;
}

void Area3D::set_gravity(real_t p_gravity) {
	gravity = p_gravity;
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_GRAVITY, p_gravity);
}

real_t Area3D::get_gravity() const {
	return gravity;
}

void Area3D::set_linear_damp(real_t p_linear_damp) {
	linear_damp = p_linear_damp;
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_LINEAR_DAMP, p_linear_damp);
}

real_t Area3D::get_linear_damp() const {
	return linear_damp;
}

void Area3D::set_angular_damp(real_t p_angular_damp) {
	angular_damp = p_angular_damp;
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_ANGULAR_DAMP, p_angular_damp);
}

real_t Area3D::get_angular_damp() const {
	return angular_damp;
}

void Area3D::set_priority(real_t p_priority) {
	priority = p_priority;
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_PRIORITY, p_priority);
}

real_t Area3D::get_priority() const {
	return priority;
}

void Area3D::set_wind_force_magnitude(real_t p_wind_force_magnitude) {
	wind_force_magnitude = p_wind_force_magnitude;
	if (is_inside_tree()) {
		_initialize_wind();
	}
}

real_t Area3D::get_wind_force_magnitude() const {
	return wind_force_magnitude;
}

void Area3D::set_wind_attenuation_factor(real_t p_wind_force_attenuation_factor) {
	wind_attenuation_factor = p_wind_force_attenuation_factor;
	if (is_inside_tree()) {
		_initialize_wind();
	}
}

real_t Area3D::get_wind_attenuation_factor() const {
	return wind_attenuation_factor;
}

void Area3D::set_wind_source_path(const NodePath &p_wind_source_path) {
	wind_source_path = p_wind_source_path;
	if (is_inside_tree()) {
		_initialize_wind();
	}
}

const NodePath &Area3D::get_wind_source_path() const {
	return wind_source_path;
}

void Area3D::_initialize_wind() {
	real_t temp_magnitude = 0.0;
	Vector3 wind_direction(0., 0., 0.);
	Vector3 wind_source(0., 0., 0.);

	// Overwrite with area-specified info if available
	if (!wind_source_path.is_empty()) {
		Node3D *p_wind_source = Object::cast_to<Node3D>(get_node(wind_source_path));
		ERR_FAIL_NULL(p_wind_source);
		Transform3D global_transform = p_wind_source->get_transform();
		wind_direction = -global_transform.basis.get_axis(Vector3::AXIS_Z).normalized();
		wind_source = global_transform.origin;
		temp_magnitude = wind_force_magnitude;
	}

	// Set force, source and direction in the physics server.
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_WIND_ATTENUATION_FACTOR, wind_attenuation_factor);
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_WIND_SOURCE, wind_source);
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_WIND_DIRECTION, wind_direction);
	PhysicsServer3D::get_singleton()->area_set_param(get_rid(), PhysicsServer3D::AREA_PARAM_WIND_FORCE_MAGNITUDE, temp_magnitude);
}

void Area3D::_body_enter_tree(ObjectID p_id) {
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

void Area3D::_body_exit_tree(ObjectID p_id) {
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

void Area3D::_body_inout(int p_status, const RID &p_body, ObjectID p_instance, int p_body_shape, int p_area_shape) {
	bool body_in = p_status == PhysicsServer3D::AREA_BODY_ADDED;
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
				node->connect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &Area3D::_body_enter_tree), make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Area3D::_body_exit_tree), make_binds(objid));
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
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &Area3D::_body_enter_tree));
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Area3D::_body_exit_tree));
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

void Area3D::_clear_monitoring() {
	ERR_FAIL_COND_MSG(locked, "This function can't be used during the in/out signal.");

	{
		Map<ObjectID, BodyState> bmcopy = body_map;
		body_map.clear();
		//disconnect all monitored stuff

		for (Map<ObjectID, BodyState>::Element *E = bmcopy.front(); E; E = E->next()) {
			Object *obj = ObjectDB::get_instance(E->key());
			Node *node = Object::cast_to<Node>(obj);

			if (!node) { //node may have been deleted in previous frame or at other legitimate point
				continue;
			}
			//ERR_CONTINUE(!node);

			node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &Area3D::_body_enter_tree));
			node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Area3D::_body_exit_tree));

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

			if (!node) { //node may have been deleted in previous frame or at other legitimate point
				continue;
			}
			//ERR_CONTINUE(!node);

			node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &Area3D::_area_enter_tree));
			node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Area3D::_area_exit_tree));

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

void Area3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_EXIT_TREE) {
		_clear_monitoring();
	} else if (p_what == NOTIFICATION_ENTER_TREE) {
		_initialize_wind();
	}
}

void Area3D::set_monitoring(bool p_enable) {
	ERR_FAIL_COND_MSG(locked, "Function blocked during in/out signal. Use set_deferred(\"monitoring\", true/false).");

	if (p_enable == monitoring) {
		return;
	}

	monitoring = p_enable;

	if (monitoring) {
		PhysicsServer3D::get_singleton()->area_set_monitor_callback(get_rid(), this, SceneStringNames::get_singleton()->_body_inout);
		PhysicsServer3D::get_singleton()->area_set_area_monitor_callback(get_rid(), this, SceneStringNames::get_singleton()->_area_inout);
	} else {
		PhysicsServer3D::get_singleton()->area_set_monitor_callback(get_rid(), nullptr, StringName());
		PhysicsServer3D::get_singleton()->area_set_area_monitor_callback(get_rid(), nullptr, StringName());
		_clear_monitoring();
	}
}

void Area3D::_area_enter_tree(ObjectID p_id) {
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

void Area3D::_area_exit_tree(ObjectID p_id) {
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

void Area3D::_area_inout(int p_status, const RID &p_area, ObjectID p_instance, int p_area_shape, int p_self_shape) {
	bool area_in = p_status == PhysicsServer3D::AREA_BODY_ADDED;
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
				node->connect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &Area3D::_area_enter_tree), make_binds(objid));
				node->connect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Area3D::_area_exit_tree), make_binds(objid));
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
				node->disconnect(SceneStringNames::get_singleton()->tree_entered, callable_mp(this, &Area3D::_area_enter_tree));
				node->disconnect(SceneStringNames::get_singleton()->tree_exiting, callable_mp(this, &Area3D::_area_exit_tree));
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

bool Area3D::is_monitoring() const {
	return monitoring;
}

TypedArray<Node3D> Area3D::get_overlapping_bodies() const {
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

void Area3D::set_monitorable(bool p_enable) {
	ERR_FAIL_COND_MSG(locked || (is_inside_tree() && PhysicsServer3D::get_singleton()->is_flushing_queries()), "Function blocked during in/out signal. Use set_deferred(\"monitorable\", true/false).");

	if (p_enable == monitorable) {
		return;
	}

	monitorable = p_enable;

	PhysicsServer3D::get_singleton()->area_set_monitorable(get_rid(), monitorable);
}

bool Area3D::is_monitorable() const {
	return monitorable;
}

TypedArray<Area3D> Area3D::get_overlapping_areas() const {
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

bool Area3D::overlaps_area(Node *p_area) const {
	ERR_FAIL_NULL_V(p_area, false);
	const Map<ObjectID, AreaState>::Element *E = area_map.find(p_area->get_instance_id());
	if (!E) {
		return false;
	}
	return E->get().in_tree;
}

bool Area3D::overlaps_body(Node *p_body) const {
	ERR_FAIL_NULL_V(p_body, false);
	const Map<ObjectID, BodyState>::Element *E = body_map.find(p_body->get_instance_id());
	if (!E) {
		return false;
	}
	return E->get().in_tree;
}

void Area3D::set_audio_bus_override(bool p_override) {
	audio_bus_override = p_override;
}

bool Area3D::is_overriding_audio_bus() const {
	return audio_bus_override;
}

void Area3D::set_audio_bus_name(const StringName &p_audio_bus) {
	audio_bus = p_audio_bus;
}

StringName Area3D::get_audio_bus_name() const {
	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (AudioServer::get_singleton()->get_bus_name(i) == audio_bus) {
			return audio_bus;
		}
	}
	return "Master";
}

void Area3D::set_use_reverb_bus(bool p_enable) {
	use_reverb_bus = p_enable;
}

bool Area3D::is_using_reverb_bus() const {
	return use_reverb_bus;
}

void Area3D::set_reverb_bus(const StringName &p_audio_bus) {
	reverb_bus = p_audio_bus;
}

StringName Area3D::get_reverb_bus() const {
	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (AudioServer::get_singleton()->get_bus_name(i) == reverb_bus) {
			return reverb_bus;
		}
	}
	return "Master";
}

void Area3D::set_reverb_amount(float p_amount) {
	reverb_amount = p_amount;
}

float Area3D::get_reverb_amount() const {
	return reverb_amount;
}

void Area3D::set_reverb_uniformity(float p_uniformity) {
	reverb_uniformity = p_uniformity;
}

float Area3D::get_reverb_uniformity() const {
	return reverb_uniformity;
}

void Area3D::_validate_property(PropertyInfo &property) const {
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

void Area3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_space_override_mode", "enable"), &Area3D::set_space_override_mode);
	ClassDB::bind_method(D_METHOD("get_space_override_mode"), &Area3D::get_space_override_mode);

	ClassDB::bind_method(D_METHOD("set_gravity_is_point", "enable"), &Area3D::set_gravity_is_point);
	ClassDB::bind_method(D_METHOD("is_gravity_a_point"), &Area3D::is_gravity_a_point);

	ClassDB::bind_method(D_METHOD("set_gravity_distance_scale", "distance_scale"), &Area3D::set_gravity_distance_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_distance_scale"), &Area3D::get_gravity_distance_scale);

	ClassDB::bind_method(D_METHOD("set_gravity_vector", "vector"), &Area3D::set_gravity_vector);
	ClassDB::bind_method(D_METHOD("get_gravity_vector"), &Area3D::get_gravity_vector);

	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &Area3D::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &Area3D::get_gravity);

	ClassDB::bind_method(D_METHOD("set_angular_damp", "angular_damp"), &Area3D::set_angular_damp);
	ClassDB::bind_method(D_METHOD("get_angular_damp"), &Area3D::get_angular_damp);

	ClassDB::bind_method(D_METHOD("set_linear_damp", "linear_damp"), &Area3D::set_linear_damp);
	ClassDB::bind_method(D_METHOD("get_linear_damp"), &Area3D::get_linear_damp);

	ClassDB::bind_method(D_METHOD("set_priority", "priority"), &Area3D::set_priority);
	ClassDB::bind_method(D_METHOD("get_priority"), &Area3D::get_priority);

	ClassDB::bind_method(D_METHOD("set_wind_force_magnitude", "wind_force_magnitude"), &Area3D::set_wind_force_magnitude);
	ClassDB::bind_method(D_METHOD("get_wind_force_magnitude"), &Area3D::get_wind_force_magnitude);

	ClassDB::bind_method(D_METHOD("set_wind_attenuation_factor", "wind_attenuation_factor"), &Area3D::set_wind_attenuation_factor);
	ClassDB::bind_method(D_METHOD("get_wind_attenuation_factor"), &Area3D::get_wind_attenuation_factor);

	ClassDB::bind_method(D_METHOD("set_wind_source_path", "wind_source_path"), &Area3D::set_wind_source_path);
	ClassDB::bind_method(D_METHOD("get_wind_source_path"), &Area3D::get_wind_source_path);

	ClassDB::bind_method(D_METHOD("set_monitorable", "enable"), &Area3D::set_monitorable);
	ClassDB::bind_method(D_METHOD("is_monitorable"), &Area3D::is_monitorable);

	ClassDB::bind_method(D_METHOD("set_monitoring", "enable"), &Area3D::set_monitoring);
	ClassDB::bind_method(D_METHOD("is_monitoring"), &Area3D::is_monitoring);

	ClassDB::bind_method(D_METHOD("get_overlapping_bodies"), &Area3D::get_overlapping_bodies);
	ClassDB::bind_method(D_METHOD("get_overlapping_areas"), &Area3D::get_overlapping_areas);

	ClassDB::bind_method(D_METHOD("overlaps_body", "body"), &Area3D::overlaps_body);
	ClassDB::bind_method(D_METHOD("overlaps_area", "area"), &Area3D::overlaps_area);

	ClassDB::bind_method(D_METHOD("_body_inout"), &Area3D::_body_inout);
	ClassDB::bind_method(D_METHOD("_area_inout"), &Area3D::_area_inout);

	ClassDB::bind_method(D_METHOD("set_audio_bus_override", "enable"), &Area3D::set_audio_bus_override);
	ClassDB::bind_method(D_METHOD("is_overriding_audio_bus"), &Area3D::is_overriding_audio_bus);

	ClassDB::bind_method(D_METHOD("set_audio_bus_name", "name"), &Area3D::set_audio_bus_name);
	ClassDB::bind_method(D_METHOD("get_audio_bus_name"), &Area3D::get_audio_bus_name);

	ClassDB::bind_method(D_METHOD("set_use_reverb_bus", "enable"), &Area3D::set_use_reverb_bus);
	ClassDB::bind_method(D_METHOD("is_using_reverb_bus"), &Area3D::is_using_reverb_bus);

	ClassDB::bind_method(D_METHOD("set_reverb_bus", "name"), &Area3D::set_reverb_bus);
	ClassDB::bind_method(D_METHOD("get_reverb_bus"), &Area3D::get_reverb_bus);

	ClassDB::bind_method(D_METHOD("set_reverb_amount", "amount"), &Area3D::set_reverb_amount);
	ClassDB::bind_method(D_METHOD("get_reverb_amount"), &Area3D::get_reverb_amount);

	ClassDB::bind_method(D_METHOD("set_reverb_uniformity", "amount"), &Area3D::set_reverb_uniformity);
	ClassDB::bind_method(D_METHOD("get_reverb_uniformity"), &Area3D::get_reverb_uniformity);

	ADD_SIGNAL(MethodInfo("body_shape_entered", PropertyInfo(Variant::RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node3D"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("body_shape_exited", PropertyInfo(Variant::RID, "body_rid"), PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node3D"), PropertyInfo(Variant::INT, "body_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("body_entered", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node3D")));
	ADD_SIGNAL(MethodInfo("body_exited", PropertyInfo(Variant::OBJECT, "body", PROPERTY_HINT_RESOURCE_TYPE, "Node3D")));

	ADD_SIGNAL(MethodInfo("area_shape_entered", PropertyInfo(Variant::RID, "area_rid"), PropertyInfo(Variant::OBJECT, "area", PROPERTY_HINT_RESOURCE_TYPE, "Area3D"), PropertyInfo(Variant::INT, "area_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("area_shape_exited", PropertyInfo(Variant::RID, "area_rid"), PropertyInfo(Variant::OBJECT, "area", PROPERTY_HINT_RESOURCE_TYPE, "Area3D"), PropertyInfo(Variant::INT, "area_shape_index"), PropertyInfo(Variant::INT, "local_shape_index")));
	ADD_SIGNAL(MethodInfo("area_entered", PropertyInfo(Variant::OBJECT, "area", PROPERTY_HINT_RESOURCE_TYPE, "Area3D")));
	ADD_SIGNAL(MethodInfo("area_exited", PropertyInfo(Variant::OBJECT, "area", PROPERTY_HINT_RESOURCE_TYPE, "Area3D")));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "monitoring"), "set_monitoring", "is_monitoring");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "monitorable"), "set_monitorable", "is_monitorable");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "priority", PROPERTY_HINT_RANGE, "0,128,1"), "set_priority", "get_priority");

	ADD_GROUP("Physics Overrides", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "space_override", PROPERTY_HINT_ENUM, "Disabled,Combine,Combine-Replace,Replace,Replace-Combine"), "set_space_override_mode", "get_space_override_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gravity_point"), "set_gravity_is_point", "is_gravity_a_point");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity_distance_scale", PROPERTY_HINT_RANGE, "0,1024,0.001,or_greater,exp"), "set_gravity_distance_scale", "get_gravity_distance_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity_vec"), "set_gravity_vector", "get_gravity_vector");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity", PROPERTY_HINT_RANGE, "-32,32,0.001,or_lesser,or_greater"), "set_gravity", "get_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "linear_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_linear_damp", "get_linear_damp");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_damp", PROPERTY_HINT_RANGE, "0,100,0.001,or_greater"), "set_angular_damp", "get_angular_damp");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wind_force_magnitude", PROPERTY_HINT_RANGE, "0,10,0.001,or_greater"), "set_wind_force_magnitude", "get_wind_force_magnitude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wind_attenuation_factor", PROPERTY_HINT_RANGE, "0.0,3.0,0.001,or_greater"), "set_wind_attenuation_factor", "get_wind_attenuation_factor");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "wind_source_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"), "set_wind_source_path", "get_wind_source_path");

	ADD_GROUP("Audio Bus", "audio_bus_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "audio_bus_override"), "set_audio_bus_override", "is_overriding_audio_bus");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "audio_bus_name", PROPERTY_HINT_ENUM, ""), "set_audio_bus_name", "get_audio_bus_name");

	ADD_GROUP("Reverb Bus", "reverb_bus_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "reverb_bus_enable"), "set_use_reverb_bus", "is_using_reverb_bus");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "reverb_bus_name", PROPERTY_HINT_ENUM, ""), "set_reverb_bus", "get_reverb_bus");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "reverb_bus_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_reverb_amount", "get_reverb_amount");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "reverb_bus_uniformity", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_reverb_uniformity", "get_reverb_uniformity");

	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_DISABLED);
	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_COMBINE);
	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_COMBINE_REPLACE);
	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_REPLACE);
	BIND_ENUM_CONSTANT(SPACE_OVERRIDE_REPLACE_COMBINE);
}

Area3D::Area3D() :
		CollisionObject3D(PhysicsServer3D::get_singleton()->area_create(), true) {
	set_gravity(9.8);
	set_gravity_vector(Vector3(0, -1, 0));
	set_monitoring(true);
	set_monitorable(true);
}

Area3D::~Area3D() {
}
