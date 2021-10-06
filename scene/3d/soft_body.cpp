/*************************************************************************/
/*  soft_body.cpp                                                        */
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

#include "soft_body.h"
#include "core/list.h"
#include "core/object.h"
#include "core/os/os.h"
#include "core/rid.h"
#include "scene/3d/collision_object.h"
#include "scene/3d/physics_body.h"
#include "scene/3d/skeleton.h"
#include "servers/physics_server.h"

SoftBodyVisualServerHandler::SoftBodyVisualServerHandler() {}

void SoftBodyVisualServerHandler::prepare(RID p_mesh, int p_surface) {
	clear();

	ERR_FAIL_COND(!p_mesh.is_valid());

	mesh = p_mesh;
	surface = p_surface;

	const uint32_t surface_format = VS::get_singleton()->mesh_surface_get_format(mesh, surface);
	const int surface_vertex_len = VS::get_singleton()->mesh_surface_get_array_len(mesh, p_surface);
	const int surface_index_len = VS::get_singleton()->mesh_surface_get_array_index_len(mesh, p_surface);
	uint32_t surface_offsets[VS::ARRAY_MAX];
	uint32_t surface_strides[VS::ARRAY_MAX];

	buffer = VS::get_singleton()->mesh_surface_get_array(mesh, surface);
	VS::get_singleton()->mesh_surface_make_offsets_from_format(surface_format, surface_vertex_len, surface_index_len, surface_offsets, surface_strides);
	ERR_FAIL_COND(surface_strides[VS::ARRAY_VERTEX] != surface_strides[VS::ARRAY_NORMAL]);
	stride = surface_strides[VS::ARRAY_VERTEX];
	offset_vertices = surface_offsets[VS::ARRAY_VERTEX];
	offset_normal = surface_offsets[VS::ARRAY_NORMAL];
}

void SoftBodyVisualServerHandler::clear() {
	if (mesh.is_valid()) {
		buffer.resize(0);
	}

	mesh = RID();
}

void SoftBodyVisualServerHandler::open() {
	write_buffer = buffer.write();
}

void SoftBodyVisualServerHandler::close() {
	write_buffer.release();
}

void SoftBodyVisualServerHandler::commit_changes() {
	VS::get_singleton()->mesh_surface_update_region(mesh, surface, 0, buffer);
}

void SoftBodyVisualServerHandler::set_vertex(int p_vertex_id, const void *p_vector3) {
	memcpy(&write_buffer[p_vertex_id * stride + offset_vertices], p_vector3, sizeof(float) * 3);
}

void SoftBodyVisualServerHandler::set_normal(int p_vertex_id, const void *p_vector3) {
	Vector2 normal_oct = VisualServer::get_singleton()->norm_to_oct(*(Vector3 *)p_vector3);
	int16_t v_normal[2] = {
		(int16_t)CLAMP(normal_oct.x * 32767, -32768, 32767),
		(int16_t)CLAMP(normal_oct.y * 32767, -32768, 32767),
	};
	memcpy(&write_buffer[p_vertex_id * stride + offset_normal], v_normal, sizeof(uint16_t) * 2);
}

void SoftBodyVisualServerHandler::set_aabb(const AABB &p_aabb) {
	VS::get_singleton()->mesh_set_custom_aabb(mesh, p_aabb);
}

SoftBody::PinnedPoint::PinnedPoint() :
		point_index(-1),
		spatial_attachment(nullptr) {
}

SoftBody::PinnedPoint::PinnedPoint(const PinnedPoint &obj_tocopy) {
	point_index = obj_tocopy.point_index;
	spatial_attachment_path = obj_tocopy.spatial_attachment_path;
	spatial_attachment = obj_tocopy.spatial_attachment;
	offset = obj_tocopy.offset;
}

SoftBody::PinnedPoint SoftBody::PinnedPoint::operator=(const PinnedPoint &obj) {
	point_index = obj.point_index;
	spatial_attachment_path = obj.spatial_attachment_path;
	spatial_attachment = obj.spatial_attachment;
	offset = obj.offset;
	return *this;
}

void SoftBody::_update_pickable() {
	if (!is_inside_tree()) {
		return;
	}
	bool pickable = ray_pickable && is_visible_in_tree();
	PhysicsServer::get_singleton()->soft_body_set_ray_pickable(physics_rid, pickable);
}

bool SoftBody::_set(const StringName &p_name, const Variant &p_value) {
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

bool SoftBody::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	String which = name.get_slicec('/', 0);

	if ("pinned_points" == which) {
		Array arr_ret;
		const int pinned_points_indices_size = pinned_points.size();
		PoolVector<PinnedPoint>::Read r = pinned_points.read();
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

void SoftBody::_get_property_list(List<PropertyInfo> *p_list) const {
	const int pinned_points_indices_size = pinned_points.size();

	p_list->push_back(PropertyInfo(Variant::POOL_INT_ARRAY, "pinned_points"));

	for (int i = 0; i < pinned_points_indices_size; ++i) {
		p_list->push_back(PropertyInfo(Variant::INT, "attachments/" + itos(i) + "/point_index"));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, "attachments/" + itos(i) + "/spatial_attachment_path"));
		p_list->push_back(PropertyInfo(Variant::VECTOR3, "attachments/" + itos(i) + "/offset"));
	}
}

bool SoftBody::_set_property_pinned_points_indices(const Array &p_indices) {
	const int p_indices_size = p_indices.size();

	{ // Remove the pined points on physics server that will be removed by resize
		PoolVector<PinnedPoint>::Read r = pinned_points.read();
		if (p_indices_size < pinned_points.size()) {
			for (int i = pinned_points.size() - 1; i >= p_indices_size; --i) {
				pin_point(r[i].point_index, false);
			}
		}
	}

	pinned_points.resize(p_indices_size);

	PoolVector<PinnedPoint>::Write w = pinned_points.write();
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

bool SoftBody::_set_property_pinned_points_attachment(int p_item, const String &p_what, const Variant &p_value) {
	if (pinned_points.size() <= p_item) {
		return false;
	}

	if ("spatial_attachment_path" == p_what) {
		PoolVector<PinnedPoint>::Write w = pinned_points.write();
		pin_point(w[p_item].point_index, true, p_value);
		_make_cache_dirty();
	} else if ("offset" == p_what) {
		PoolVector<PinnedPoint>::Write w = pinned_points.write();
		w[p_item].offset = p_value;
	} else {
		return false;
	}

	return true;
}

bool SoftBody::_get_property_pinned_points(int p_item, const String &p_what, Variant &r_ret) const {
	if (pinned_points.size() <= p_item) {
		return false;
	}
	PoolVector<PinnedPoint>::Read r = pinned_points.read();

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

void SoftBody::_changed_callback(Object *p_changed, const char *p_prop) {
	prepare_physics_server();
	_reset_points_offsets();
#ifdef TOOLS_ENABLED
	if (p_changed == this) {
		update_configuration_warning();
	}
#endif
}

void SoftBody::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_WORLD: {
			if (Engine::get_singleton()->is_editor_hint()) {
				add_change_receptor(this);
			}

			RID space = get_world()->get_space();
			PhysicsServer::get_singleton()->soft_body_set_space(physics_rid, space);
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

			PhysicsServer::get_singleton()->soft_body_set_transform(physics_rid, get_global_transform());

			set_notify_transform(false);
			// Required to be top level with Transform at center of world in order to modify VisualServer only to support custom Transform
			set_as_toplevel(true);
			set_transform(Transform());
			set_notify_transform(true);

		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_pickable();

		} break;
		case NOTIFICATION_EXIT_WORLD: {
			PhysicsServer::get_singleton()->soft_body_set_space(physics_rid, RID());

		} break;
	}

#ifdef TOOLS_ENABLED

	if (p_what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED) {
		if (Engine::get_singleton()->is_editor_hint()) {
			update_configuration_warning();
		}
	}

#endif
}

void SoftBody::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_draw_soft_mesh"), &SoftBody::_draw_soft_mesh);

	ClassDB::bind_method(D_METHOD("set_physics_enabled", "enabled"), &SoftBody::set_physics_enabled);
	ClassDB::bind_method(D_METHOD("is_physics_enabled"), &SoftBody::is_physics_enabled);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "collision_mask"), &SoftBody::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &SoftBody::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "collision_layer"), &SoftBody::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &SoftBody::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &SoftBody::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &SoftBody::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("set_collision_layer_bit", "bit", "value"), &SoftBody::set_collision_layer_bit);
	ClassDB::bind_method(D_METHOD("get_collision_layer_bit", "bit"), &SoftBody::get_collision_layer_bit);

	ClassDB::bind_method(D_METHOD("set_parent_collision_ignore", "parent_collision_ignore"), &SoftBody::set_parent_collision_ignore);
	ClassDB::bind_method(D_METHOD("get_parent_collision_ignore"), &SoftBody::get_parent_collision_ignore);

	ClassDB::bind_method(D_METHOD("get_collision_exceptions"), &SoftBody::get_collision_exceptions);
	ClassDB::bind_method(D_METHOD("add_collision_exception_with", "body"), &SoftBody::add_collision_exception_with);
	ClassDB::bind_method(D_METHOD("remove_collision_exception_with", "body"), &SoftBody::remove_collision_exception_with);

	ClassDB::bind_method(D_METHOD("set_simulation_precision", "simulation_precision"), &SoftBody::set_simulation_precision);
	ClassDB::bind_method(D_METHOD("get_simulation_precision"), &SoftBody::get_simulation_precision);

	ClassDB::bind_method(D_METHOD("set_total_mass", "mass"), &SoftBody::set_total_mass);
	ClassDB::bind_method(D_METHOD("get_total_mass"), &SoftBody::get_total_mass);

	ClassDB::bind_method(D_METHOD("set_linear_stiffness", "linear_stiffness"), &SoftBody::set_linear_stiffness);
	ClassDB::bind_method(D_METHOD("get_linear_stiffness"), &SoftBody::get_linear_stiffness);

	ClassDB::bind_method(D_METHOD("set_areaAngular_stiffness", "areaAngular_stiffness"), &SoftBody::set_areaAngular_stiffness);
	ClassDB::bind_method(D_METHOD("get_areaAngular_stiffness"), &SoftBody::get_areaAngular_stiffness);

	ClassDB::bind_method(D_METHOD("set_volume_stiffness", "volume_stiffness"), &SoftBody::set_volume_stiffness);
	ClassDB::bind_method(D_METHOD("get_volume_stiffness"), &SoftBody::get_volume_stiffness);

	ClassDB::bind_method(D_METHOD("set_pressure_coefficient", "pressure_coefficient"), &SoftBody::set_pressure_coefficient);
	ClassDB::bind_method(D_METHOD("get_pressure_coefficient"), &SoftBody::get_pressure_coefficient);

	ClassDB::bind_method(D_METHOD("set_pose_matching_coefficient", "pose_matching_coefficient"), &SoftBody::set_pose_matching_coefficient);
	ClassDB::bind_method(D_METHOD("get_pose_matching_coefficient"), &SoftBody::get_pose_matching_coefficient);

	ClassDB::bind_method(D_METHOD("set_damping_coefficient", "damping_coefficient"), &SoftBody::set_damping_coefficient);
	ClassDB::bind_method(D_METHOD("get_damping_coefficient"), &SoftBody::get_damping_coefficient);

	ClassDB::bind_method(D_METHOD("set_drag_coefficient", "drag_coefficient"), &SoftBody::set_drag_coefficient);
	ClassDB::bind_method(D_METHOD("get_drag_coefficient"), &SoftBody::get_drag_coefficient);

	ClassDB::bind_method(D_METHOD("get_point_transform", "point_index"), &SoftBody::get_point_transform);

	ClassDB::bind_method(D_METHOD("set_point_pinned", "point_index", "pinned", "attachment_path"), &SoftBody::pin_point, DEFVAL(NodePath()));
	ClassDB::bind_method(D_METHOD("is_point_pinned", "point_index"), &SoftBody::is_point_pinned);

	ClassDB::bind_method(D_METHOD("set_ray_pickable", "ray_pickable"), &SoftBody::set_ray_pickable);
	ClassDB::bind_method(D_METHOD("is_ray_pickable"), &SoftBody::is_ray_pickable);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "physics_enabled"), "set_physics_enabled", "is_physics_enabled");

	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "parent_collision_ignore", PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE, "Parent collision object"), "set_parent_collision_ignore", "get_parent_collision_ignore");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "simulation_precision", PROPERTY_HINT_RANGE, "1,100,1"), "set_simulation_precision", "get_simulation_precision");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "total_mass", PROPERTY_HINT_RANGE, "0.01,10000,1"), "set_total_mass", "get_total_mass");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "linear_stiffness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_linear_stiffness", "get_linear_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "areaAngular_stiffness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_areaAngular_stiffness", "get_areaAngular_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_stiffness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_volume_stiffness", "get_volume_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "pressure_coefficient"), "set_pressure_coefficient", "get_pressure_coefficient");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "damping_coefficient", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_damping_coefficient", "get_damping_coefficient");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "drag_coefficient", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_drag_coefficient", "get_drag_coefficient");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "pose_matching_coefficient", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_pose_matching_coefficient", "get_pose_matching_coefficient");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ray_pickable"), "set_ray_pickable", "is_ray_pickable");
}

String SoftBody::get_configuration_warning() const {
	String warning = MeshInstance::get_configuration_warning();

	if (get_mesh().is_null()) {
		if (!warning.empty()) {
			warning += "\n\n";
		}

		warning += TTR("This body will be ignored until you set a mesh.");
	}

	Transform t = get_transform();
	if ((ABS(t.basis.get_axis(0).length() - 1.0) > 0.05 || ABS(t.basis.get_axis(1).length() - 1.0) > 0.05 || ABS(t.basis.get_axis(2).length() - 1.0) > 0.05)) {
		if (!warning.empty()) {
			warning += "\n\n";
		}

		warning += TTR("Size changes to SoftBody will be overridden by the physics engine when running.\nChange the size in children collision shapes instead.");
	}

	return warning;
}

void SoftBody::_update_physics_server() {
	if (!simulation_started) {
		return;
	}

	_update_cache_pin_points_datas();
	// Submit bone attachment
	const int pinned_points_indices_size = pinned_points.size();
	PoolVector<PinnedPoint>::Read r = pinned_points.read();
	for (int i = 0; i < pinned_points_indices_size; ++i) {
		if (r[i].spatial_attachment) {
			PhysicsServer::get_singleton()->soft_body_move_point(physics_rid, r[i].point_index, r[i].spatial_attachment->get_global_transform().xform(r[i].offset));
		}
	}
}

void SoftBody::_draw_soft_mesh() {
	if (get_mesh().is_null()) {
		return;
	}

	const RID mesh_rid = get_mesh()->get_rid();
	if (!visual_server_handler.is_ready(mesh_rid)) {
		visual_server_handler.prepare(mesh_rid, 0);

		/// Necessary in order to render the mesh correctly (Soft body nodes are in global space)
		simulation_started = true;
		call_deferred("set_as_toplevel", true);
		call_deferred("set_transform", Transform());
	}

	_update_physics_server();

	visual_server_handler.open();
	PhysicsServer::get_singleton()->soft_body_update_visual_server(physics_rid, &visual_server_handler);
	visual_server_handler.close();

	visual_server_handler.commit_changes();
}

void SoftBody::prepare_physics_server() {
	if (Engine::get_singleton()->is_editor_hint()) {
		if (get_mesh().is_valid()) {
			PhysicsServer::get_singleton()->soft_body_set_mesh(physics_rid, get_mesh());
		} else {
			PhysicsServer::get_singleton()->soft_body_set_mesh(physics_rid, nullptr);
		}

		return;
	}

	if (get_mesh().is_valid() && physics_enabled) {
		become_mesh_owner();
		PhysicsServer::get_singleton()->soft_body_set_mesh(physics_rid, get_mesh());
		VS::get_singleton()->connect("frame_pre_draw", this, "_draw_soft_mesh");
	} else {
		PhysicsServer::get_singleton()->soft_body_set_mesh(physics_rid, nullptr);
		if (VS::get_singleton()->is_connected("frame_pre_draw", this, "_draw_soft_mesh")) {
			VS::get_singleton()->disconnect("frame_pre_draw", this, "_draw_soft_mesh");
		}
	}
}

void SoftBody::become_mesh_owner() {
	if (mesh.is_null()) {
		return;
	}

	if (!mesh_owner) {
		mesh_owner = true;

		Vector<Ref<Material>> copy_materials;
		copy_materials.append_array(materials);

		ERR_FAIL_COND(!mesh->get_surface_count());

		// Get current mesh array and create new mesh array with necessary flag for softbody
		Array surface_arrays = mesh->surface_get_arrays(0);
		Array surface_blend_arrays = mesh->surface_get_blend_shape_arrays(0);
		uint32_t surface_format = mesh->surface_get_format(0);

		surface_format &= ~(Mesh::ARRAY_COMPRESS_VERTEX | Mesh::ARRAY_COMPRESS_NORMAL);
		surface_format |= Mesh::ARRAY_FLAG_USE_DYNAMIC_UPDATE;

		Ref<ArrayMesh> soft_mesh;
		soft_mesh.instance();
		soft_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface_arrays, surface_blend_arrays, surface_format);
		soft_mesh->surface_set_material(0, mesh->surface_get_material(0));

		set_mesh(soft_mesh);

		for (int i = copy_materials.size() - 1; 0 <= i; --i) {
			set_surface_material(i, copy_materials[i]);
		}
	}
}

void SoftBody::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
	PhysicsServer::get_singleton()->soft_body_set_collision_mask(physics_rid, p_mask);
}

uint32_t SoftBody::get_collision_mask() const {
	return collision_mask;
}
void SoftBody::set_collision_layer(uint32_t p_layer) {
	collision_layer = p_layer;
	PhysicsServer::get_singleton()->soft_body_set_collision_layer(physics_rid, p_layer);
}

uint32_t SoftBody::get_collision_layer() const {
	return collision_layer;
}

void SoftBody::set_collision_mask_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision mask bit must be between 0 and 31 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool SoftBody::get_collision_mask_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision mask bit must be between 0 and 31 inclusive.");
	return get_collision_mask() & (1 << p_bit);
}

void SoftBody::set_collision_layer_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision layer bit must be between 0 and 31 inclusive.");
	uint32_t layer = get_collision_layer();
	if (p_value) {
		layer |= 1 << p_bit;
	} else {
		layer &= ~(1 << p_bit);
	}
	set_collision_layer(layer);
}

bool SoftBody::get_collision_layer_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision layer bit must be between 0 and 31 inclusive.");
	return get_collision_layer() & (1 << p_bit);
}

void SoftBody::set_parent_collision_ignore(const NodePath &p_parent_collision_ignore) {
	parent_collision_ignore = p_parent_collision_ignore;
}

const NodePath &SoftBody::get_parent_collision_ignore() const {
	return parent_collision_ignore;
}

void SoftBody::set_physics_enabled(bool p_enabled) {
	if (p_enabled == physics_enabled) {
		return;
	}

	physics_enabled = p_enabled;

	if (is_inside_tree()) {
		prepare_physics_server();
	}
}

bool SoftBody::is_physics_enabled() const {
	return physics_enabled;
}

void SoftBody::set_pinned_points_indices(PoolVector<SoftBody::PinnedPoint> p_pinned_points_indices) {
	pinned_points = p_pinned_points_indices;
	PoolVector<PinnedPoint>::Read w = pinned_points.read();
	for (int i = pinned_points.size() - 1; 0 <= i; --i) {
		pin_point(p_pinned_points_indices[i].point_index, true);
	}
}

PoolVector<SoftBody::PinnedPoint> SoftBody::get_pinned_points_indices() {
	return pinned_points;
}

Array SoftBody::get_collision_exceptions() {
	List<RID> exceptions;
	PhysicsServer::get_singleton()->soft_body_get_collision_exceptions(physics_rid, &exceptions);
	Array ret;
	for (List<RID>::Element *E = exceptions.front(); E; E = E->next()) {
		RID body = E->get();
		ObjectID instance_id = PhysicsServer::get_singleton()->body_get_object_instance_id(body);
		Object *obj = ObjectDB::get_instance(instance_id);
		PhysicsBody *physics_body = Object::cast_to<PhysicsBody>(obj);
		ret.append(physics_body);
	}
	return ret;
}

void SoftBody::add_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	CollisionObject *collision_object = Object::cast_to<CollisionObject>(p_node);
	ERR_FAIL_COND_MSG(!collision_object, "Collision exception only works between two CollisionObject.");
	PhysicsServer::get_singleton()->soft_body_add_collision_exception(physics_rid, collision_object->get_rid());
}

void SoftBody::remove_collision_exception_with(Node *p_node) {
	ERR_FAIL_NULL(p_node);
	CollisionObject *collision_object = Object::cast_to<CollisionObject>(p_node);
	ERR_FAIL_COND_MSG(!collision_object, "Collision exception only works between two CollisionObject.");
	PhysicsServer::get_singleton()->soft_body_remove_collision_exception(physics_rid, collision_object->get_rid());
}

int SoftBody::get_simulation_precision() {
	return PhysicsServer::get_singleton()->soft_body_get_simulation_precision(physics_rid);
}

void SoftBody::set_simulation_precision(int p_simulation_precision) {
	PhysicsServer::get_singleton()->soft_body_set_simulation_precision(physics_rid, p_simulation_precision);
}

real_t SoftBody::get_total_mass() {
	return PhysicsServer::get_singleton()->soft_body_get_total_mass(physics_rid);
}

void SoftBody::set_total_mass(real_t p_total_mass) {
	PhysicsServer::get_singleton()->soft_body_set_total_mass(physics_rid, p_total_mass);
}

void SoftBody::set_linear_stiffness(real_t p_linear_stiffness) {
	PhysicsServer::get_singleton()->soft_body_set_linear_stiffness(physics_rid, p_linear_stiffness);
}

real_t SoftBody::get_linear_stiffness() {
	return PhysicsServer::get_singleton()->soft_body_get_linear_stiffness(physics_rid);
}

void SoftBody::set_areaAngular_stiffness(real_t p_areaAngular_stiffness) {
	PhysicsServer::get_singleton()->soft_body_set_areaAngular_stiffness(physics_rid, p_areaAngular_stiffness);
}

real_t SoftBody::get_areaAngular_stiffness() {
	return PhysicsServer::get_singleton()->soft_body_get_areaAngular_stiffness(physics_rid);
}

void SoftBody::set_volume_stiffness(real_t p_volume_stiffness) {
	PhysicsServer::get_singleton()->soft_body_set_volume_stiffness(physics_rid, p_volume_stiffness);
}

real_t SoftBody::get_volume_stiffness() {
	return PhysicsServer::get_singleton()->soft_body_get_volume_stiffness(physics_rid);
}

real_t SoftBody::get_pressure_coefficient() {
	return PhysicsServer::get_singleton()->soft_body_get_pressure_coefficient(physics_rid);
}

void SoftBody::set_pose_matching_coefficient(real_t p_pose_matching_coefficient) {
	PhysicsServer::get_singleton()->soft_body_set_pose_matching_coefficient(physics_rid, p_pose_matching_coefficient);
}

real_t SoftBody::get_pose_matching_coefficient() {
	return PhysicsServer::get_singleton()->soft_body_get_pose_matching_coefficient(physics_rid);
}

void SoftBody::set_pressure_coefficient(real_t p_pressure_coefficient) {
	PhysicsServer::get_singleton()->soft_body_set_pressure_coefficient(physics_rid, p_pressure_coefficient);
}

real_t SoftBody::get_damping_coefficient() {
	return PhysicsServer::get_singleton()->soft_body_get_damping_coefficient(physics_rid);
}

void SoftBody::set_damping_coefficient(real_t p_damping_coefficient) {
	PhysicsServer::get_singleton()->soft_body_set_damping_coefficient(physics_rid, p_damping_coefficient);
}

real_t SoftBody::get_drag_coefficient() {
	return PhysicsServer::get_singleton()->soft_body_get_drag_coefficient(physics_rid);
}

void SoftBody::set_drag_coefficient(real_t p_drag_coefficient) {
	PhysicsServer::get_singleton()->soft_body_set_drag_coefficient(physics_rid, p_drag_coefficient);
}

Vector3 SoftBody::get_point_transform(int p_point_index) {
	return PhysicsServer::get_singleton()->soft_body_get_point_global_position(physics_rid, p_point_index);
}

void SoftBody::pin_point_toggle(int p_point_index) {
	pin_point(p_point_index, !(-1 != _has_pinned_point(p_point_index)));
}

void SoftBody::pin_point(int p_point_index, bool pin, const NodePath &p_spatial_attachment_path) {
	_pin_point_on_physics_server(p_point_index, pin);
	if (pin) {
		_add_pinned_point(p_point_index, p_spatial_attachment_path);
	} else {
		_remove_pinned_point(p_point_index);
	}
}

bool SoftBody::is_point_pinned(int p_point_index) const {
	return -1 != _has_pinned_point(p_point_index);
}

void SoftBody::set_ray_pickable(bool p_ray_pickable) {
	ray_pickable = p_ray_pickable;
	_update_pickable();
}

bool SoftBody::is_ray_pickable() const {
	return ray_pickable;
}

SoftBody::SoftBody() :
		physics_rid(PhysicsServer::get_singleton()->soft_body_create()),
		mesh_owner(false),
		collision_mask(1),
		collision_layer(1),
		simulation_started(false),
		pinned_points_cache_dirty(true),
		ray_pickable(true) {
	PhysicsServer::get_singleton()->body_attach_object_instance_id(physics_rid, get_instance_id());
}

SoftBody::~SoftBody() {
	PhysicsServer::get_singleton()->free(physics_rid);
}

void SoftBody::reset_softbody_pin() {
	PhysicsServer::get_singleton()->soft_body_remove_all_pinned_points(physics_rid);
	PoolVector<PinnedPoint>::Read pps = pinned_points.read();
	for (int i = pinned_points.size() - 1; 0 < i; --i) {
		PhysicsServer::get_singleton()->soft_body_pin_point(physics_rid, pps[i].point_index, true);
	}
}

void SoftBody::_make_cache_dirty() {
	pinned_points_cache_dirty = true;
}

void SoftBody::_update_cache_pin_points_datas() {
	if (!pinned_points_cache_dirty) {
		return;
	}

	pinned_points_cache_dirty = false;

	PoolVector<PinnedPoint>::Write w = pinned_points.write();
	for (int i = pinned_points.size() - 1; 0 <= i; --i) {
		if (!w[i].spatial_attachment_path.is_empty()) {
			w[i].spatial_attachment = Object::cast_to<Spatial>(get_node(w[i].spatial_attachment_path));
		}
		if (!w[i].spatial_attachment) {
			ERR_PRINT("Spatial node not defined in the pinned point, this is undefined behavior for SoftBody!");
		}
	}
}

void SoftBody::_pin_point_on_physics_server(int p_point_index, bool pin) {
	PhysicsServer::get_singleton()->soft_body_pin_point(physics_rid, p_point_index, pin);
}

void SoftBody::_add_pinned_point(int p_point_index, const NodePath &p_spatial_attachment_path) {
	SoftBody::PinnedPoint *pinned_point;
	if (-1 == _get_pinned_point(p_point_index, pinned_point)) {
		// Create new
		PinnedPoint pp;
		pp.point_index = p_point_index;
		pp.spatial_attachment_path = p_spatial_attachment_path;

		if (!p_spatial_attachment_path.is_empty() && has_node(p_spatial_attachment_path)) {
			pp.spatial_attachment = Object::cast_to<Spatial>(get_node(p_spatial_attachment_path));
			pp.offset = (pp.spatial_attachment->get_global_transform().affine_inverse() * get_global_transform()).xform(PhysicsServer::get_singleton()->soft_body_get_point_global_position(physics_rid, pp.point_index));
		}

		pinned_points.push_back(pp);

	} else {
		pinned_point->point_index = p_point_index;
		pinned_point->spatial_attachment_path = p_spatial_attachment_path;

		if (!p_spatial_attachment_path.is_empty() && has_node(p_spatial_attachment_path)) {
			pinned_point->spatial_attachment = Object::cast_to<Spatial>(get_node(p_spatial_attachment_path));
			pinned_point->offset = (pinned_point->spatial_attachment->get_global_transform().affine_inverse() * get_global_transform()).xform(PhysicsServer::get_singleton()->soft_body_get_point_global_position(physics_rid, pinned_point->point_index));
		}
	}
}

void SoftBody::_reset_points_offsets() {
	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	PoolVector<PinnedPoint>::Read r = pinned_points.read();
	PoolVector<PinnedPoint>::Write w = pinned_points.write();
	for (int i = pinned_points.size() - 1; 0 <= i; --i) {
		if (!r[i].spatial_attachment) {
			if (!r[i].spatial_attachment_path.is_empty() && has_node(r[i].spatial_attachment_path)) {
				w[i].spatial_attachment = Object::cast_to<Spatial>(get_node(r[i].spatial_attachment_path));
			}
		}

		if (!r[i].spatial_attachment) {
			continue;
		}

		w[i].offset = (r[i].spatial_attachment->get_global_transform().affine_inverse() * get_global_transform()).xform(PhysicsServer::get_singleton()->soft_body_get_point_global_position(physics_rid, r[i].point_index));
	}
}

void SoftBody::_remove_pinned_point(int p_point_index) {
	const int id(_has_pinned_point(p_point_index));
	if (-1 != id) {
		pinned_points.remove(id);
	}
}

int SoftBody::_get_pinned_point(int p_point_index, SoftBody::PinnedPoint *&r_point) const {
	const int id = _has_pinned_point(p_point_index);
	if (-1 == id) {
		r_point = nullptr;
		return -1;
	} else {
		r_point = const_cast<SoftBody::PinnedPoint *>(&pinned_points.read()[id]);
		return id;
	}
}

int SoftBody::_has_pinned_point(int p_point_index) const {
	PoolVector<PinnedPoint>::Read r = pinned_points.read();
	for (int i = pinned_points.size() - 1; 0 <= i; --i) {
		if (p_point_index == r[i].point_index) {
			return i;
		}
	}
	return -1;
}
