/*************************************************************************/
/*  soft_body_3d.cpp                                                     */
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

#include "soft_body_3d.h"

#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/templates/list.h"
#include "core/templates/rid.h"
#include "scene/3d/collision_object_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/3d/skeleton_3d.h"

SoftBodyRenderingServerHandler::SoftBodyRenderingServerHandler() {}

void SoftBodyRenderingServerHandler::prepare(RID p_mesh, int p_surface) {
	clear();

	ERR_FAIL_COND(!p_mesh.is_valid());

	mesh = p_mesh;
	surface = p_surface;

	RS::SurfaceData surface_data = RS::get_singleton()->mesh_get_surface(mesh, surface);

	uint32_t surface_offsets[RS::ARRAY_MAX];
	uint32_t vertex_stride;
	uint32_t attrib_stride;
	uint32_t skin_stride;
	RS::get_singleton()->mesh_surface_make_offsets_from_format(surface_data.format, surface_data.vertex_count, surface_data.index_count, surface_offsets, vertex_stride, attrib_stride, skin_stride);

	buffer = surface_data.vertex_data;
	stride = vertex_stride;
	offset_vertices = surface_offsets[RS::ARRAY_VERTEX];
	offset_normal = surface_offsets[RS::ARRAY_NORMAL];
}

void SoftBodyRenderingServerHandler::clear() {
	buffer.resize(0);
	stride = 0;
	offset_vertices = 0;
	offset_normal = 0;

	surface = 0;
	mesh = RID();
}

void SoftBodyRenderingServerHandler::open() {
	write_buffer = buffer.ptrw();
}

void SoftBodyRenderingServerHandler::close() {
	write_buffer = nullptr;
}

void SoftBodyRenderingServerHandler::commit_changes() {
	RS::get_singleton()->mesh_surface_update_region(mesh, surface, 0, buffer);
}

void SoftBodyRenderingServerHandler::set_vertex(int p_vertex_id, const void *p_vector3) {
	memcpy(&write_buffer[p_vertex_id * stride + offset_vertices], p_vector3, sizeof(float) * 3);
}

void SoftBodyRenderingServerHandler::set_normal(int p_vertex_id, const void *p_vector3) {
	memcpy(&write_buffer[p_vertex_id * stride + offset_normal], p_vector3, sizeof(float) * 3);
}

void SoftBodyRenderingServerHandler::set_aabb(const AABB &p_aabb) {
	RS::get_singleton()->mesh_set_custom_aabb(mesh, p_aabb);
}

SoftBody3D::PinnedPoint::PinnedPoint() {
}

SoftBody3D::PinnedPoint::PinnedPoint(const PinnedPoint &obj_tocopy) {
	point_index = obj_tocopy.point_index;
	spatial_attachment_path = obj_tocopy.spatial_attachment_path;
	spatial_attachment = obj_tocopy.spatial_attachment;
	offset = obj_tocopy.offset;
}

SoftBody3D::PinnedPoint &SoftBody3D::PinnedPoint::operator=(const PinnedPoint &obj) {
	point_index = obj.point_index;
	spatial_attachment_path = obj.spatial_attachment_path;
	spatial_attachment = obj.spatial_attachment;
	offset = obj.offset;
	return *this;
}

void SoftBody3D::_update_pickable() {
	if (!is_inside_tree()) {
		return;
	}
	bool pickable = ray_pickable && is_visible_in_tree();
	PhysicsServer3D::get_singleton()->soft_body_set_ray_pickable(physics_rid, pickable);
}

bool SoftBody3D::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	String which = name.get_slicec('/', 0);

	if ("pinned_points" == which) {
		return _set_property_pinned_points_indices(p_value);

	} else if ("attachments" == which) {
		int idx = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);

		return _set_property_pinned_points_attachment(idx, what, p_value);
	}

	return false;
}

bool SoftBody3D::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	String which = name.get_slicec('/', 0);

	if ("pinned_points" == which) {
		Array arr_ret;
		const int pinned_points_indices_size = pinned_points.size();
		const PinnedPoint *r = pinned_points.ptr();
		arr_ret.resize(pinned_points_indices_size);

		for (int i = 0; i < pinned_points_indices_size; ++i) {
			arr_ret[i] = r[i].point_index;
		}

		r_ret = arr_ret;
		return true;

	} else if ("attachments" == which) {
		int idx = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);

		return _get_property_pinned_points(idx, what, r_ret);
	}

	return false;
}

void SoftBody3D::_get_property_list(List<PropertyInfo> *p_list) const {
	const int pinned_points_indices_size = pinned_points.size();

	p_list->push_back(PropertyInfo(Variant::PACKED_INT32_ARRAY, "pinned_points"));

	for (int i = 0; i < pinned_points_indices_size; ++i) {
		p_list->push_back(PropertyInfo(Variant::INT, "attachments/" + itos(i) + "/point_index"));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, "attachments/" + itos(i) + "/spatial_attachment_path"));
		p_list->push_back(PropertyInfo(Variant::VECTOR3, "attachments/" + itos(i) + "/offset"));
	}
}

bool SoftBody3D::_set_property_pinned_points_indices(const Array &p_indices) {
	const int p_indices_size = p_indices.size();

	{ // Remove the pined points on physics server that will be removed by resize
		const PinnedPoint *r = pinned_points.ptr();
		if (p_indices_size < pinned_points.size()) {
			for (int i = pinned_points.size() - 1; i >= p_indices_size; --i) {
				pin_point(r[i].point_index, false);
			}
		}
	}

	pinned_points.resize(p_indices_size);

	PinnedPoint *w = pinned_points.ptrw();
	int point_index;
	for (int i = 0; i < p_indices_size; ++i) {
		point_index = p_indices.get(i);
		if (w[i].point_index != point_index) {
			if (-1 != w[i].point_index) {
				pin_point(w[i].point_index, false);
			}
			w[i].point_index = point_index;
			pin_point(w[i].point_index, true);
		}
	}
	return true;
}

bool SoftBody3D::_set_property_pinned_points_attachment(int p_item, const String &p_what, const Variant &p_value) {
	if (pinned_points.size() <= p_item) {
		return false;
	}

	if ("spatial_attachment_path" == p_what) {
		PinnedPoint *w = pinned_points.ptrw();
		pin_point(w[p_item].point_index, true, p_value);
		_make_cache_dirty();
	} else if ("offset" == p_what) {
		PinnedPoint *w = pinned_points.ptrw();
		w[p_item].offset = p_value;
	} else {
		return false;
	}

	return true;
}

bool SoftBody3D::_get_property_pinned_points(int p_item, const String &p_what, Variant &r_ret) const {
	if (pinned_points.size() <= p_item) {
		return false;
	}
	const PinnedPoint *r = pinned_points.ptr();

	if ("point_index" == p_what) {
		r_ret = r[p_item].point_index;
	} else if ("spatial_attachment_path" == p_what) {
		r_ret = r[p_item].spatial_attachment_path;
	} else if ("offset" == p_what) {
		r_ret = r[p_item].offset;
	} else {
		return false;
	}

	return true;
}

void SoftBody3D::_softbody_changed() {
	prepare_physics_server();
	_reset_points_offsets();
#ifdef TOOLS_ENABLED
	update_configuration_warnings();
#endif
}

void SoftBody3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_WORLD: {
			if (Engine::get_singleton()->is_editor_hint()) {
				// I have no idea what this is supposed to do, it's really weird
				// leaving for upcoming PK work on physics
				//add_change_receptor(this);
			}

			RID space = get_world_3d()->get_space();
			PhysicsServer3D::get_singleton()->soft_body_set_space(physics_rid, space);
			prepare_physics_server();
		} break;
		case NOTIFICATION_READY: {
			if (!parent_collision_ignore.is_empty()) {
				add_collision_exception_with(get_node(parent_collision_ignore));
			}

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (Engine::get_singleton()->is_editor_hint()) {
				_reset_points_offsets();
				return;
			}

			PhysicsServer3D::get_singleton()->soft_body_set_transform(physics_rid, get_global_transform());

			set_notify_transform(false);
			// Required to be top level with Transform at center of world in order to modify RenderingServer only to support custom Transform
			set_as_top_level(true);
			set_transform(Transform());
			set_notify_transform(true);

		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_pickable();

		} break;
		case NOTIFICATION_EXIT_WORLD: {
			PhysicsServer3D::get_singleton()->soft_body_set_space(physics_rid, RID());

		} break;
	}

#ifdef TOOLS_ENABLED

	if (p_what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED) {
		if (Engine::get_singleton()->is_editor_hint()) {
			update_configuration_warnings();
		}
	}

#endif
}

void SoftBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_physics_rid"), &SoftBody3D::get_physics_rid);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "collision_mask"), &SoftBody3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &SoftBody3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "collision_layer"), &SoftBody3D::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &SoftBody3D::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &SoftBody3D::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &SoftBody3D::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("set_collision_layer_bit", "bit", "value"), &SoftBody3D::set_collision_layer_bit);
	ClassDB::bind_method(D_METHOD("get_collision_layer_bit", "bit"), &SoftBody3D::get_collision_layer_bit);

	ClassDB::bind_method(D_METHOD("set_parent_collision_ignore", "parent_collision_ignore"), &SoftBody3D::set_parent_collision_ignore);
	ClassDB::bind_method(D_METHOD("get_parent_collision_ignore"), &SoftBody3D::get_parent_collision_ignore);

	ClassDB::bind_method(D_METHOD("get_collision_exceptions"), &SoftBody3D::get_collision_exceptions);
	ClassDB::bind_method(D_METHOD("add_collision_exception_with", "body"), &SoftBody3D::add_collision_exception_with);
	ClassDB::bind_method(D_METHOD("remove_collision_exception_with", "body"), &SoftBody3D::remove_collision_exception_with);

	ClassDB::bind_method(D_METHOD("set_simulation_precision", "simulation_precision"), &SoftBody3D::set_simulation_precision);
	ClassDB::bind_method(D_METHOD("get_simulation_precision"), &SoftBody3D::get_simulation_precision);

	ClassDB::bind_method(D_METHOD("set_total_mass", "mass"), &SoftBody3D::set_total_mass);
	ClassDB::bind_method(D_METHOD("get_total_mass"), &SoftBody3D::get_total_mass);

	ClassDB::bind_method(D_METHOD("set_linear_stiffness", "linear_stiffness"), &SoftBody3D::set_linear_stiffness);
	ClassDB::bind_method(D_METHOD("get_linear_stiffness"), &SoftBody3D::get_linear_stiffness);

	ClassDB::bind_method(D_METHOD("set_pressure_coefficient", "pressure_coefficient"), &SoftBody3D::set_pressure_coefficient);
	ClassDB::bind_method(D_METHOD("get_pressure_coefficient"), &SoftBody3D::get_pressure_coefficient);

	ClassDB::bind_method(D_METHOD("set_damping_coefficient", "damping_coefficient"), &SoftBody3D::set_damping_coefficient);
	ClassDB::bind_method(D_METHOD("get_damping_coefficient"), &SoftBody3D::get_damping_coefficient);

	ClassDB::bind_method(D_METHOD("set_drag_coefficient", "drag_coefficient"), &SoftBody3D::set_drag_coefficient);
	ClassDB::bind_method(D_METHOD("get_drag_coefficient"), &SoftBody3D::get_drag_coefficient);

	ClassDB::bind_method(D_METHOD("set_ray_pickable", "ray_pickable"), &SoftBody3D::set_ray_pickable);
	ClassDB::bind_method(D_METHOD("is_ray_pickable"), &SoftBody3D::is_ray_pickable);

	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "parent_collision_ignore", PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE, "Parent collision object"), "set_parent_collision_ignore", "get_parent_collision_ignore");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "simulation_precision", PROPERTY_HINT_RANGE, "1,100,1"), "set_simulation_precision", "get_simulation_precision");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "total_mass", PROPERTY_HINT_RANGE, "0.01,10000,1"), "set_total_mass", "get_total_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "linear_stiffness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_linear_stiffness", "get_linear_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pressure_coefficient"), "set_pressure_coefficient", "get_pressure_coefficient");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping_coefficient", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_damping_coefficient", "get_damping_coefficient");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "drag_coefficient", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_coefficient", "get_drag_coefficient");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ray_pickable"), "set_ray_pickable", "is_ray_pickable");
}

TypedArray<String> SoftBody3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (get_mesh().is_null()) {
		warnings.push_back(TTR("This body will be ignored until you set a mesh."));
	}

	Transform t = get_transform();
	if ((ABS(t.basis.get_axis(0).length() - 1.0) > 0.05 || ABS(t.basis.get_axis(1).length() - 1.0) > 0.05 || ABS(t.basis.get_axis(2).length() - 1.0) > 0.05)) {
		warnings.push_back(TTR("Size changes to SoftBody3D will be overridden by the physics engine when running.\nChange the size in children collision shapes instead."));
	}

	return warnings;
}

void SoftBody3D::_update_physics_server() {
	if (!simulation_started) {
		return;
	}

	_update_cache_pin_points_datas();
	// Submit bone attachment
	const int pinned_points_indices_size = pinned_points.size();
	const PinnedPoint *r = pinned_points.ptr();
	for (int i = 0; i < pinned_points_indices_size; ++i) {
		if (r[i].spatial_attachment) {
			PhysicsServer3D::get_singleton()->soft_body_move_point(physics_rid, r[i].point_index, r[i].spatial_attachment->get_global_transform().xform(r[i].offset));
		}
	}
}

void SoftBody3D::_draw_soft_mesh() {
	if (get_mesh().is_null()) {
		return;
	}

	if (!rendering_server_handler.is_ready()) {
		rendering_server_handler.prepare(get_mesh()->get_rid(), 0);

		/// Necessary in order to render the mesh correctly (Soft body nodes are in global space)
		simulation_started = true;
		call_deferred("set_as_top_level", true);
		call_deferred("set_transform", Transform());
	}

	_update_physics_server();

	rendering_server_handler.open();
	PhysicsServer3D::get_singleton()->soft_body_update_rendering_server(physics_rid, &rendering_server_handler);
	rendering_server_handler.close();

	rendering_server_handler.commit_changes();
}

void SoftBody3D::prepare_physics_server() {
	if (Engine::get_singleton()->is_editor_hint()) {
		if (get_mesh().is_valid()) {
			PhysicsServer3D::get_singleton()->soft_body_set_mesh(physics_rid, get_mesh());
		} else {
			PhysicsServer3D::get_singleton()->soft_body_set_mesh(physics_rid, nullptr);
		}

		return;
	}

	if (get_mesh().is_valid()) {
		become_mesh_owner();
		PhysicsServer3D::get_singleton()->soft_body_set_mesh(physics_rid, get_mesh());
		RS::get_singleton()->connect("frame_pre_draw", callable_mp(this, &SoftBody3D::_draw_soft_mesh));
	} else {
		PhysicsServer3D::get_singleton()->soft_body_set_mesh(physics_rid, nullptr);
		if (RS::get_singleton()->is_connected("frame_pre_draw", callable_mp(this, &SoftBody3D::_draw_soft_mesh))) {
			RS::get_singleton()->disconnect("frame_pre_draw", callable_mp(this, &SoftBody3D::_draw_soft_mesh));
		}
	}
}

void SoftBody3D::become_mesh_owner() {
	if (mesh.is_null()) {
		return;
	}

	if (!mesh_owner) {
		mesh_owner = true;

		Vector<Ref<Material>> copy_materials;
		copy_materials.append_array(surface_override_materials);

		ERR_FAIL_COND(!mesh->get_surface_count());

		// Get current mesh array and create new mesh array with necessary flag for softbody
		Array surface_arrays = mesh->surface_get_arrays(0);
		Array surface_blend_arrays = mesh->surface_get_blend_shape_arrays(0);
		Dictionary surface_lods = mesh->surface_get_lods(0);
		uint32_t surface_format = mesh->surface_get_format(0);

		surface_format |= Mesh::ARRAY_FLAG_USE_DYNAMIC_UPDATE;

		Ref<ArrayMesh> soft_mesh;
		soft_mesh.instance();
		soft_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface_arrays, surface_blend_arrays, surface_lods, surface_format);
		soft_mesh->surface_set_material(0, mesh->surface_get_material(0));

		set_mesh(soft_mesh);

		for (int i = copy_materials.size() - 1; 0 <= i; --i) {
			set_surface_override_material(i, copy_materials[i]);
		}
	}
}

void SoftBody3D::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
	PhysicsServer3D::get_singleton()->soft_body_set_collision_mask(physics_rid, p_mask);
}

uint32_t SoftBody3D::get_collision_mask() const {
	return collision_mask;
}

void SoftBody3D::set_collision_layer(uint32_t p_layer) {
	collision_layer = p_layer;
	PhysicsServer3D::get_singleton()->soft_body_set_collision_layer(physics_rid, p_layer);
}

uint32_t SoftBody3D::get_collision_layer() const {
	return collision_layer;
}

void SoftBody3D::set_collision_mask_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision mask bit must be between 0 and 31 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool SoftBody3D::get_collision_mask_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision mask bit must be between 0 and 31 inclusive.");
	return get_collision_mask() & (1 << p_bit);
}

void SoftBody3D::set_collision_layer_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision layer bit must be between 0 and 31 inclusive.");
	uint32_t layer = get_collision_layer();
	if (p_value) {
		layer |= 1 << p_bit;
	} else {
		layer &= ~(1 << p_bit);
	}
	set_collision_layer(layer);
}

bool SoftBody3D::get_collision_layer_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision layer bit must be between 0 and 31 inclusive.");
	return get_collision_layer() & (1 << p_bit);
}

void SoftBody3D::set_parent_collision_ignore(const NodePath &p_parent_collision_ignore) {
	parent_collision_ignore = p_parent_collision_ignore;
}

const NodePath &SoftBody3D::get_parent_collision_ignore() const {
	return parent_collision_ignore;
}

void SoftBody3D::set_pinned_points_indices(Vector<SoftBody3D::PinnedPoint> p_pinned_points_indices) {
	pinned_points = p_pinned_points_indices;
	for (int i = pinned_points.size() - 1; 0 <= i; --i) {
		pin_point(p_pinned_points_indices[i].point_index, true);
	}
}

Vector<SoftBody3D::PinnedPoint> SoftBody3D::get_pinned_points_indices() {
	return pinned_points;
}

Array SoftBody3D::get_collision_exceptions() {
	List<RID> exceptions;
	PhysicsServer3D::get_singleton()->soft_body_get_collision_exceptions(physics_rid, &exceptions);
	Array ret;
	for (List<RID>::Element *E = exceptions.front(); E; E = E->next()) {
		RID body = E->get();
		ObjectID instance_id = PhysicsServer3D::get_singleton()->body_get_object_instance_id(body);
		Object *obj = ObjectDB::get_instance(instance_id);
		PhysicsBody3D *physics_body = Object::cast_to<PhysicsBody3D>(obj);
		ret.append(physics_body);
	}
	return ret;
}

void SoftBody3D::add_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	CollisionObject3D *collision_object = Object::cast_to<CollisionObject3D>(p_node);
	ERR_FAIL_COND_MSG(!collision_object, "Collision exception only works between two CollisionObject3Ds.");
	PhysicsServer3D::get_singleton()->soft_body_add_collision_exception(physics_rid, collision_object->get_rid());
}

void SoftBody3D::remove_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	CollisionObject3D *collision_object = Object::cast_to<CollisionObject3D>(p_node);
	ERR_FAIL_COND_MSG(!collision_object, "Collision exception only works between two CollisionObject3Ds.");
	PhysicsServer3D::get_singleton()->soft_body_remove_collision_exception(physics_rid, collision_object->get_rid());
}

int SoftBody3D::get_simulation_precision() {
	return PhysicsServer3D::get_singleton()->soft_body_get_simulation_precision(physics_rid);
}

void SoftBody3D::set_simulation_precision(int p_simulation_precision) {
	PhysicsServer3D::get_singleton()->soft_body_set_simulation_precision(physics_rid, p_simulation_precision);
}

real_t SoftBody3D::get_total_mass() {
	return PhysicsServer3D::get_singleton()->soft_body_get_total_mass(physics_rid);
}

void SoftBody3D::set_total_mass(real_t p_total_mass) {
	PhysicsServer3D::get_singleton()->soft_body_set_total_mass(physics_rid, p_total_mass);
}

void SoftBody3D::set_linear_stiffness(real_t p_linear_stiffness) {
	PhysicsServer3D::get_singleton()->soft_body_set_linear_stiffness(physics_rid, p_linear_stiffness);
}

real_t SoftBody3D::get_linear_stiffness() {
	return PhysicsServer3D::get_singleton()->soft_body_get_linear_stiffness(physics_rid);
}

real_t SoftBody3D::get_pressure_coefficient() {
	return PhysicsServer3D::get_singleton()->soft_body_get_pressure_coefficient(physics_rid);
}

void SoftBody3D::set_pressure_coefficient(real_t p_pressure_coefficient) {
	PhysicsServer3D::get_singleton()->soft_body_set_pressure_coefficient(physics_rid, p_pressure_coefficient);
}

real_t SoftBody3D::get_damping_coefficient() {
	return PhysicsServer3D::get_singleton()->soft_body_get_damping_coefficient(physics_rid);
}

void SoftBody3D::set_damping_coefficient(real_t p_damping_coefficient) {
	PhysicsServer3D::get_singleton()->soft_body_set_damping_coefficient(physics_rid, p_damping_coefficient);
}

real_t SoftBody3D::get_drag_coefficient() {
	return PhysicsServer3D::get_singleton()->soft_body_get_drag_coefficient(physics_rid);
}

void SoftBody3D::set_drag_coefficient(real_t p_drag_coefficient) {
	PhysicsServer3D::get_singleton()->soft_body_set_drag_coefficient(physics_rid, p_drag_coefficient);
}

Vector3 SoftBody3D::get_point_transform(int p_point_index) {
	return PhysicsServer3D::get_singleton()->soft_body_get_point_global_position(physics_rid, p_point_index);
}

void SoftBody3D::pin_point_toggle(int p_point_index) {
	pin_point(p_point_index, !(-1 != _has_pinned_point(p_point_index)));
}

void SoftBody3D::pin_point(int p_point_index, bool pin, const NodePath &p_spatial_attachment_path) {
	_pin_point_on_physics_server(p_point_index, pin);
	if (pin) {
		_add_pinned_point(p_point_index, p_spatial_attachment_path);
	} else {
		_remove_pinned_point(p_point_index);
	}
}

bool SoftBody3D::is_point_pinned(int p_point_index) const {
	return -1 != _has_pinned_point(p_point_index);
}

void SoftBody3D::set_ray_pickable(bool p_ray_pickable) {
	ray_pickable = p_ray_pickable;
	_update_pickable();
}

bool SoftBody3D::is_ray_pickable() const {
	return ray_pickable;
}

SoftBody3D::SoftBody3D() :
		physics_rid(PhysicsServer3D::get_singleton()->soft_body_create()) {
	PhysicsServer3D::get_singleton()->body_attach_object_instance_id(physics_rid, get_instance_id());
}

SoftBody3D::~SoftBody3D() {
	PhysicsServer3D::get_singleton()->free(physics_rid);
}

void SoftBody3D::reset_softbody_pin() {
	PhysicsServer3D::get_singleton()->soft_body_remove_all_pinned_points(physics_rid);
	const PinnedPoint *pps = pinned_points.ptr();
	for (int i = pinned_points.size() - 1; 0 < i; --i) {
		PhysicsServer3D::get_singleton()->soft_body_pin_point(physics_rid, pps[i].point_index, true);
	}
}

void SoftBody3D::_make_cache_dirty() {
	pinned_points_cache_dirty = true;
}

void SoftBody3D::_update_cache_pin_points_datas() {
	if (!pinned_points_cache_dirty) {
		return;
	}

	pinned_points_cache_dirty = false;

	PinnedPoint *w = pinned_points.ptrw();
	for (int i = pinned_points.size() - 1; 0 <= i; --i) {
		if (!w[i].spatial_attachment_path.is_empty()) {
			w[i].spatial_attachment = Object::cast_to<Node3D>(get_node(w[i].spatial_attachment_path));
		}
		if (!w[i].spatial_attachment) {
			ERR_PRINT("Node3D node not defined in the pinned point, this is undefined behavior for SoftBody3D!");
		}
	}
}

void SoftBody3D::_pin_point_on_physics_server(int p_point_index, bool pin) {
	PhysicsServer3D::get_singleton()->soft_body_pin_point(physics_rid, p_point_index, pin);
}

void SoftBody3D::_add_pinned_point(int p_point_index, const NodePath &p_spatial_attachment_path) {
	SoftBody3D::PinnedPoint *pinned_point;
	if (-1 == _get_pinned_point(p_point_index, pinned_point)) {
		// Create new
		PinnedPoint pp;
		pp.point_index = p_point_index;
		pp.spatial_attachment_path = p_spatial_attachment_path;

		if (!p_spatial_attachment_path.is_empty() && has_node(p_spatial_attachment_path)) {
			pp.spatial_attachment = Object::cast_to<Node3D>(get_node(p_spatial_attachment_path));
			pp.offset = (pp.spatial_attachment->get_global_transform().affine_inverse() * get_global_transform()).xform(PhysicsServer3D::get_singleton()->soft_body_get_point_global_position(physics_rid, pp.point_index));
		}

		pinned_points.push_back(pp);

	} else {
		pinned_point->point_index = p_point_index;
		pinned_point->spatial_attachment_path = p_spatial_attachment_path;

		if (!p_spatial_attachment_path.is_empty() && has_node(p_spatial_attachment_path)) {
			pinned_point->spatial_attachment = Object::cast_to<Node3D>(get_node(p_spatial_attachment_path));
			pinned_point->offset = (pinned_point->spatial_attachment->get_global_transform().affine_inverse() * get_global_transform()).xform(PhysicsServer3D::get_singleton()->soft_body_get_point_global_position(physics_rid, pinned_point->point_index));
		}
	}
}

void SoftBody3D::_reset_points_offsets() {
	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	const PinnedPoint *r = pinned_points.ptr();
	PinnedPoint *w = pinned_points.ptrw();
	for (int i = pinned_points.size() - 1; 0 <= i; --i) {
		if (!r[i].spatial_attachment) {
			if (!r[i].spatial_attachment_path.is_empty() && has_node(r[i].spatial_attachment_path)) {
				w[i].spatial_attachment = Object::cast_to<Node3D>(get_node(r[i].spatial_attachment_path));
			}
		}

		if (!r[i].spatial_attachment) {
			continue;
		}

		w[i].offset = (r[i].spatial_attachment->get_global_transform().affine_inverse() * get_global_transform()).xform(PhysicsServer3D::get_singleton()->soft_body_get_point_global_position(physics_rid, r[i].point_index));
	}
}

void SoftBody3D::_remove_pinned_point(int p_point_index) {
	const int id(_has_pinned_point(p_point_index));
	if (-1 != id) {
		pinned_points.remove(id);
	}
}

int SoftBody3D::_get_pinned_point(int p_point_index, SoftBody3D::PinnedPoint *&r_point) const {
	const int id = _has_pinned_point(p_point_index);
	if (-1 == id) {
		r_point = nullptr;
		return -1;
	} else {
		r_point = const_cast<SoftBody3D::PinnedPoint *>(&pinned_points.ptr()[id]);
		return id;
	}
}

int SoftBody3D::_has_pinned_point(int p_point_index) const {
	const PinnedPoint *r = pinned_points.ptr();
	for (int i = pinned_points.size() - 1; 0 <= i; --i) {
		if (p_point_index == r[i].point_index) {
			return i;
		}
	}
	return -1;
}
