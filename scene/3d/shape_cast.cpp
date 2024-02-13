/**************************************************************************/
/*  shape_cast.cpp                                                        */
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

#include "shape_cast.h"

#include "collision_object.h"
#include "core/engine.h"
#include "mesh_instance.h"
#include "scene/resources/concave_polygon_shape.h"

void ShapeCast::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (Engine::get_singleton()->is_editor_hint()) {
				_update_debug_shape_vertices();
			}
			if (enabled && !Engine::get_singleton()->is_editor_hint()) {
				set_physics_process_internal(true);
			} else {
				set_physics_process_internal(false);
			}

			if (get_tree()->is_debugging_collisions_hint()) {
				_update_debug_shape();
			}

			if (Object::cast_to<CollisionObject>(get_parent())) {
				if (exclude_parent_body) {
					exclude.insert(Object::cast_to<CollisionObject>(get_parent())->get_rid());
				} else {
					exclude.erase(Object::cast_to<CollisionObject>(get_parent())->get_rid());
				}
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (enabled) {
				set_physics_process_internal(false);
			}

			if (debug_shape) {
				_clear_debug_shape();
			}
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (!enabled) {
				break;
			}

			bool prev_collision_state = collided;
			_update_shapecast_state();
			if (get_tree()->is_debugging_collisions_hint()) {
				if (prev_collision_state != collided) {
					_update_debug_shape_material(true);
				}
				if (collided) {
					_update_debug_shape();
				}
				if (prev_collision_state == collided && !collided) {
					_update_debug_shape();
				}
			}
		} break;
	}
}

void ShapeCast::_bind_methods() {
	ClassDB::bind_method(D_METHOD("resource_changed", "resource"), &ShapeCast::resource_changed);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &ShapeCast::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &ShapeCast::is_enabled);

	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &ShapeCast::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &ShapeCast::get_shape);

	ClassDB::bind_method(D_METHOD("set_target_position", "local_point"), &ShapeCast::set_target_position);
	ClassDB::bind_method(D_METHOD("get_target_position"), &ShapeCast::get_target_position);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &ShapeCast::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &ShapeCast::get_margin);

	ClassDB::bind_method(D_METHOD("set_max_results", "max_results"), &ShapeCast::set_max_results);
	ClassDB::bind_method(D_METHOD("get_max_results"), &ShapeCast::get_max_results);

	ClassDB::bind_method(D_METHOD("is_colliding"), &ShapeCast::is_colliding);
	ClassDB::bind_method(D_METHOD("get_collision_count"), &ShapeCast::get_collision_count);

	ClassDB::bind_method(D_METHOD("force_shapecast_update"), &ShapeCast::force_shapecast_update);

	ClassDB::bind_method(D_METHOD("get_collider", "index"), &ShapeCast::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_rid", "index"), &ShapeCast::get_collider_rid);
	ClassDB::bind_method(D_METHOD("get_collider_shape", "index"), &ShapeCast::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collision_point", "index"), &ShapeCast::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal", "index"), &ShapeCast::get_collision_normal);

	ClassDB::bind_method(D_METHOD("get_closest_collision_safe_fraction"), &ShapeCast::get_closest_collision_safe_fraction);
	ClassDB::bind_method(D_METHOD("get_closest_collision_unsafe_fraction"), &ShapeCast::get_closest_collision_unsafe_fraction);

	ClassDB::bind_method(D_METHOD("add_exception_rid", "rid"), &ShapeCast::add_exception_rid);
	ClassDB::bind_method(D_METHOD("add_exception", "node"), &ShapeCast::add_exception);

	ClassDB::bind_method(D_METHOD("remove_exception_rid", "rid"), &ShapeCast::remove_exception_rid);
	ClassDB::bind_method(D_METHOD("remove_exception", "node"), &ShapeCast::remove_exception);

	ClassDB::bind_method(D_METHOD("clear_exceptions"), &ShapeCast::clear_exceptions);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &ShapeCast::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &ShapeCast::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_value", "layer_number", "value"), &ShapeCast::set_collision_mask_value);
	ClassDB::bind_method(D_METHOD("get_collision_mask_value", "layer_number"), &ShapeCast::get_collision_mask_value);

	ClassDB::bind_method(D_METHOD("set_exclude_parent_body", "mask"), &ShapeCast::set_exclude_parent_body);
	ClassDB::bind_method(D_METHOD("get_exclude_parent_body"), &ShapeCast::get_exclude_parent_body);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &ShapeCast::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &ShapeCast::is_collide_with_areas_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &ShapeCast::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &ShapeCast::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("_get_collision_result"), &ShapeCast::_get_collision_result);

	ClassDB::bind_method(D_METHOD("set_debug_shape_custom_color", "debug_shape_custom_color"), &ShapeCast::set_debug_shape_custom_color);
	ClassDB::bind_method(D_METHOD("get_debug_shape_custom_color"), &ShapeCast::get_debug_shape_custom_color);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exclude_parent"), "set_exclude_parent_body", "get_exclude_parent_body");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target_position", PROPERTY_HINT_NONE), "set_target_position", "get_target_position");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "margin", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_results"), "set_max_results", "get_max_results");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "collision_result", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_SCRIPT_VARIABLE), "", "_get_collision_result");

	ADD_GROUP("Collide With", "collide_with");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collide_with_areas", "is_collide_with_areas_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collide_with_bodies", "is_collide_with_bodies_enabled");

	ADD_GROUP("Debug Shape", "debug_shape");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "debug_shape_custom_color"), "set_debug_shape_custom_color", "get_debug_shape_custom_color");
}

String ShapeCast::get_configuration_warning() const {
	String warning = Spatial::get_configuration_warning();

	if (shape.is_null()) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("This node cannot interact with other objects unless a Shape is assigned.");
	}

	if (shape.is_valid() && Object::cast_to<ConcavePolygonShape>(*shape)) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("ShapeCast does not support ConcavePolygonShapes. Collisions will not be reported.");
	}

	return warning;
}

void ShapeCast::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	update_gizmo();

	if (is_inside_tree() && !Engine::get_singleton()->is_editor_hint()) {
		set_physics_process_internal(p_enabled);
	}
	if (!p_enabled) {
		collided = false;
	}

	if (is_inside_tree() && get_tree()->is_debugging_collisions_hint()) {
		if (p_enabled) {
			_update_debug_shape();
		} else {
			_clear_debug_shape();
		}
	}
}

bool ShapeCast::is_enabled() const {
	return enabled;
}

void ShapeCast::set_target_position(const Vector3 &p_point) {
	target_position = p_point;
	if (is_inside_tree()) {
		_update_debug_shape();
	}
	update_gizmo();

	if (Engine::get_singleton()->is_editor_hint()) {
		if (is_inside_tree()) {
			_update_debug_shape_vertices();
		}
	} else if (debug_shape) {
		_update_debug_shape();
	}
}

Vector3 ShapeCast::get_target_position() const {
	return target_position;
}

void ShapeCast::set_margin(real_t p_margin) {
	margin = p_margin;
}

real_t ShapeCast::get_margin() const {
	return margin;
}

void ShapeCast::set_max_results(int p_max_results) {
	max_results = p_max_results;
}

int ShapeCast::get_max_results() const {
	return max_results;
}

void ShapeCast::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
}

uint32_t ShapeCast::get_collision_mask() const {
	return collision_mask;
}

void ShapeCast::set_collision_mask_value(int p_layer_number, bool p_value) {
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

bool ShapeCast::get_collision_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_mask() & (1 << (p_layer_number - 1));
}

int ShapeCast::get_collision_count() const {
	return result.size();
}

bool ShapeCast::is_colliding() const {
	return collided;
}

Object *ShapeCast::get_collider(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, result.size(), nullptr, "No collider found.");

	if (result[p_idx].collider_id == 0) {
		return nullptr;
	}
	return ObjectDB::get_instance(result[p_idx].collider_id);
}

RID ShapeCast::get_collider_rid(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, result.size(), RID(), "No collider RID found.");
	return result[p_idx].rid;
}

int ShapeCast::get_collider_shape(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, result.size(), -1, "No collider shape found.");
	return result[p_idx].shape;
}

Vector3 ShapeCast::get_collision_point(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, result.size(), Vector3(), "No collision point found.");
	return result[p_idx].point;
}

Vector3 ShapeCast::get_collision_normal(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, result.size(), Vector3(), "No collision normal found.");
	return result[p_idx].normal;
}

real_t ShapeCast::get_closest_collision_safe_fraction() const {
	return collision_safe_fraction;
}

real_t ShapeCast::get_closest_collision_unsafe_fraction() const {
	return collision_unsafe_fraction;
}

void ShapeCast::resource_changed(Ref<Resource> p_res) {
	if (is_inside_tree()) {
		_update_debug_shape();
	}
	update_gizmo();
}

void ShapeCast::set_shape(const Ref<Shape> &p_shape) {
	if (p_shape == shape) {
		return;
	}
	if (!shape.is_null()) {
		shape->unregister_owner(this);
	}
	shape = p_shape;
	if (!shape.is_null()) {
		shape->register_owner(this);
	}
	if (p_shape.is_valid()) {
		shape_rid = shape->get_rid();
	}

	if (is_inside_tree()) {
		_update_debug_shape();
	}

	update_gizmo();
	update_configuration_warning();
}

Ref<Shape> ShapeCast::get_shape() const {
	return shape;
}

void ShapeCast::set_exclude_parent_body(bool p_exclude_parent_body) {
	if (exclude_parent_body == p_exclude_parent_body) {
		return;
	}
	exclude_parent_body = p_exclude_parent_body;

	if (!is_inside_tree()) {
		return;
	}
	if (Object::cast_to<CollisionObject>(get_parent())) {
		if (exclude_parent_body) {
			exclude.insert(Object::cast_to<CollisionObject>(get_parent())->get_rid());
		} else {
			exclude.erase(Object::cast_to<CollisionObject>(get_parent())->get_rid());
		}
	}
}

bool ShapeCast::get_exclude_parent_body() const {
	return exclude_parent_body;
}

void ShapeCast::_update_shapecast_state() {
	result.clear();

	ERR_FAIL_COND_MSG(shape.is_null(), "Null reference to shape. ShapeCast requires a Shape3D to sweep for collisions.");

	Ref<World> w3d = get_world();
	ERR_FAIL_COND(w3d.is_null());

	PhysicsDirectSpaceState *dss = PhysicsServer::get_singleton()->space_get_direct_state(w3d->get_space());
	ERR_FAIL_COND(!dss);

	Transform gt = get_global_transform();

	collision_safe_fraction = 0.0;
	collision_unsafe_fraction = 0.0;

	if (target_position != Vector3()) {
		dss->cast_motion(shape_rid, gt, target_position, margin, collision_safe_fraction, collision_unsafe_fraction, exclude, collision_mask, collide_with_bodies, collide_with_areas);
		if (collision_unsafe_fraction < 1.0) {
			// Move shape transform to the point of impact,
			// so we can collect contact info at that point.
			gt.set_origin(gt.get_origin() + target_position * (collision_unsafe_fraction + CMP_EPSILON));
		}
	}

	// Regardless of whether the shape is stuck or it's moved along
	// the motion vector, we'll only consider static collisions from now on.
	bool intersected = true;
	Set<RID> intersected_objects = exclude;
	while (intersected && result.size() < max_results) {
		PhysicsDirectSpaceState::ShapeRestInfo info;
		intersected = dss->rest_info(shape_rid, gt, margin, &info, intersected_objects, collision_mask, collide_with_bodies, collide_with_areas);
		if (intersected) {
			result.push_back(info);
			intersected_objects.insert(info.rid);
		}
	}
	collided = !result.empty();
}

void ShapeCast::force_shapecast_update() {
	_update_shapecast_state();
}

void ShapeCast::add_exception_rid(const RID &p_rid) {
	exclude.insert(p_rid);
}

void ShapeCast::add_exception(const Object *p_object) {
	ERR_FAIL_NULL(p_object);
	const CollisionObject *co = Object::cast_to<CollisionObject>(p_object);
	ERR_FAIL_COND_MSG(!co, "The passed Node must be an instance of CollisionObject.");
	add_exception_rid(co->get_rid());
}

void ShapeCast::remove_exception_rid(const RID &p_rid) {
	exclude.erase(p_rid);
}

void ShapeCast::remove_exception(const Object *p_object) {
	ERR_FAIL_NULL(p_object);
	const CollisionObject *co = Object::cast_to<CollisionObject>(p_object);
	ERR_FAIL_COND_MSG(!co, "The passed Node must be an instance of CollisionObject.");
	remove_exception_rid(co->get_rid());
}

void ShapeCast::clear_exceptions() {
	exclude.clear();
}

void ShapeCast::set_collide_with_areas(bool p_clip) {
	collide_with_areas = p_clip;
}

bool ShapeCast::is_collide_with_areas_enabled() const {
	return collide_with_areas;
}

void ShapeCast::set_collide_with_bodies(bool p_clip) {
	collide_with_bodies = p_clip;
}

bool ShapeCast::is_collide_with_bodies_enabled() const {
	return collide_with_bodies;
}

Array ShapeCast::_get_collision_result() const {
	Array ret;

	for (int i = 0; i < result.size(); ++i) {
		const PhysicsDirectSpaceState::ShapeRestInfo &sri = result[i];

		Dictionary col;
		col["point"] = sri.point;
		col["normal"] = sri.normal;
		col["rid"] = sri.rid;
		col["collider"] = ObjectDB::get_instance(sri.collider_id);
		col["collider_id"] = sri.collider_id;
		col["shape"] = sri.shape;
		col["linear_velocity"] = sri.linear_velocity;

		ret.push_back(col);
	}
	return ret;
}

void ShapeCast::_update_debug_shape_vertices() {
	debug_shape_vertices.clear();
	debug_line_vertices.clear();

	if (!shape.is_null()) {
		debug_shape_vertices.append_array(shape->get_debug_mesh_lines());
		for (int i = 0; i < debug_shape_vertices.size(); i++) {
			debug_shape_vertices.set(i, debug_shape_vertices[i] + Vector3(target_position * get_closest_collision_safe_fraction()));
		}
	}

	if (target_position == Vector3()) {
		return;
	}

	debug_line_vertices.push_back(Vector3());
	debug_line_vertices.push_back(target_position);
}

const Vector<Vector3> &ShapeCast::get_debug_shape_vertices() const {
	return debug_shape_vertices;
}

const Vector<Vector3> &ShapeCast::get_debug_line_vertices() const {
	return debug_line_vertices;
}

void ShapeCast::set_debug_shape_custom_color(const Color &p_color) {
	debug_shape_custom_color = p_color;
	if (debug_material.is_valid()) {
		_update_debug_shape_material();
	}
}

Ref<Material3D> ShapeCast::get_debug_material() {
	_update_debug_shape_material();
	return debug_material;
}

const Color &ShapeCast::get_debug_shape_custom_color() const {
	return debug_shape_custom_color;
}

void ShapeCast::_create_debug_shape() {
	_update_debug_shape_material();

	Ref<ArrayMesh> mesh = memnew(ArrayMesh);

	MeshInstance *mi = memnew(MeshInstance);
	mi->set_mesh(mesh);

	add_child(mi);
	debug_shape = mi;
}

void ShapeCast::_update_debug_shape_material(bool p_check_collision) {
	if (!debug_material.is_valid()) {
		Ref<SpatialMaterial> material = memnew(SpatialMaterial);
		debug_material = material;

		material->set_flag(Material3D::FLAG_UNSHADED, true);
		material->set_feature(Material3D::FEATURE_TRANSPARENT, true);
		// Use double-sided rendering so that the RayCast can be seen if the camera is inside.
		material->set_cull_mode(Material3D::CULL_DISABLED);
	}

	Color color = debug_shape_custom_color;
	if (color == Color(0.0, 0.0, 0.0)) {
		// Use the default debug shape color defined in the Project Settings.
		color = get_tree()->get_debug_collisions_color();
	}

	if (p_check_collision && collided) {
		if ((color.get_h() < 0.055 || color.get_h() > 0.945) && color.get_s() > 0.5 && color.get_v() > 0.5) {
			// If base color is already quite reddish, highlight collision with green color
			color = Color(0.0, 1.0, 0.0, color.a);
		} else {
			// Else, highlight collision with red color
			color = Color(1.0, 0, 0, color.a);
		}
	}

	Ref<Material3D> material = static_cast<Ref<Material3D>>(debug_material);
	material->set_albedo(color);
}

void ShapeCast::_update_debug_shape() {
	if (!enabled) {
		return;
	}

	if (!debug_shape) {
		_create_debug_shape();
	}

	_update_debug_shape_vertices();

	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	MeshInstance *mi = static_cast<MeshInstance *>(debug_shape);
	Ref<ArrayMesh> mesh = mi->get_mesh();
	if (!mesh.is_valid()) {
		return;
	}

	mesh->clear_surfaces();

	Array a;
	a.resize(Mesh::ARRAY_MAX);

	uint32_t flags = 0;
	int surface_count = 0;

	if (!debug_shape_vertices.empty()) {
		a[Mesh::ARRAY_VERTEX] = debug_shape_vertices;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, a, Array(), flags);
		mesh->surface_set_material(surface_count, debug_material);
		++surface_count;
	}

	if (!debug_line_vertices.empty()) {
		a[Mesh::ARRAY_VERTEX] = debug_line_vertices;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, a, Array(), flags);
		mesh->surface_set_material(surface_count, debug_material);
		++surface_count;
	}
}

void ShapeCast::_clear_debug_shape() {
	if (!debug_shape) {
		return;
	}

	MeshInstance *mi = static_cast<MeshInstance *>(debug_shape);
	if (mi->is_inside_tree()) {
		mi->queue_delete();
	} else {
		memdelete(mi);
	}

	debug_shape = nullptr;
}

ShapeCast::~ShapeCast() {
	if (!shape.is_null()) {
		shape->unregister_owner(this);
	}
}
