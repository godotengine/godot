/*************************************************************************/
/*  soft_body_3d.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SOFT_PHYSICS_BODY_H
#define SOFT_PHYSICS_BODY_H

#include "scene/3d/mesh_instance_3d.h"

class SoftBody3D;

class SoftBodyRenderingServerHandler {
	friend class SoftBody3D;

	RID mesh;
	int surface;
	Vector<uint8_t> buffer;
	uint32_t stride;
	uint32_t offset_vertices;
	uint32_t offset_normal;

	uint8_t *write_buffer;

private:
	SoftBodyRenderingServerHandler();
	bool is_ready() { return mesh.is_valid(); }
	void prepare(RID p_mesh_rid, int p_surface);
	void clear();
	void open();
	void close();
	void commit_changes();

public:
	void set_vertex(int p_vertex_id, const void *p_vector3);
	void set_normal(int p_vertex_id, const void *p_vector3);
	void set_aabb(const AABB &p_aabb);
};

class SoftBody3D : public MeshInstance3D {
	GDCLASS(SoftBody3D, MeshInstance3D);

public:
	struct PinnedPoint {
		int point_index = -1;
		NodePath spatial_attachment_path;
		Node3D *spatial_attachment = nullptr; // Cache
		Vector3 offset;

		PinnedPoint();
		PinnedPoint(const PinnedPoint &obj_tocopy);
		PinnedPoint &operator=(const PinnedPoint &obj);
	};

private:
	SoftBodyRenderingServerHandler rendering_server_handler;

	RID physics_rid;

	bool mesh_owner = false;
	uint32_t collision_mask = 1;
	uint32_t collision_layer = 1;
	NodePath parent_collision_ignore;
	Vector<PinnedPoint> pinned_points;
	bool simulation_started = false;
	bool pinned_points_cache_dirty = true;

	Ref<ArrayMesh> debug_mesh_cache;
	class MeshInstance3D *debug_mesh;

	bool capture_input_on_drag;
	bool ray_pickable = true;

	void _update_pickable();

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	bool _set_property_pinned_points_indices(const Array &p_indices);
	bool _set_property_pinned_points_attachment(int p_item, const String &p_what, const Variant &p_value);
	bool _get_property_pinned_points(int p_item, const String &p_what, Variant &r_ret) const;

	virtual void _changed_callback(Object *p_changed, const char *p_prop) override;

	void _notification(int p_what);
	static void _bind_methods();

	virtual String get_configuration_warning() const override;

protected:
	void _update_physics_server();
	void _draw_soft_mesh();

public:
	void prepare_physics_server();
	void become_mesh_owner();

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;

	void set_collision_mask_bit(int p_bit, bool p_value);
	bool get_collision_mask_bit(int p_bit) const;

	void set_collision_layer_bit(int p_bit, bool p_value);
	bool get_collision_layer_bit(int p_bit) const;

	void set_parent_collision_ignore(const NodePath &p_parent_collision_ignore);
	const NodePath &get_parent_collision_ignore() const;

	void set_pinned_points_indices(Vector<PinnedPoint> p_pinned_points_indices);
	Vector<PinnedPoint> get_pinned_points_indices();

	void set_simulation_precision(int p_simulation_precision);
	int get_simulation_precision();

	void set_total_mass(real_t p_total_mass);
	real_t get_total_mass();

	void set_linear_stiffness(real_t p_linear_stiffness);
	real_t get_linear_stiffness();

	void set_areaAngular_stiffness(real_t p_areaAngular_stiffness);
	real_t get_areaAngular_stiffness();

	void set_volume_stiffness(real_t p_volume_stiffness);
	real_t get_volume_stiffness();

	void set_pressure_coefficient(real_t p_pressure_coefficient);
	real_t get_pressure_coefficient();

	void set_pose_matching_coefficient(real_t p_pose_matching_coefficient);
	real_t get_pose_matching_coefficient();

	void set_damping_coefficient(real_t p_damping_coefficient);
	real_t get_damping_coefficient();

	void set_drag_coefficient(real_t p_drag_coefficient);
	real_t get_drag_coefficient();

	Array get_collision_exceptions();
	void add_collision_exception_with(Node *p_node);
	void remove_collision_exception_with(Node *p_node);

	Vector3 get_point_transform(int p_point_index);

	void pin_point_toggle(int p_point_index);
	void pin_point(int p_point_index, bool pin, const NodePath &p_spatial_attachment_path = NodePath());
	bool is_point_pinned(int p_point_index) const;

	void set_ray_pickable(bool p_ray_pickable);
	bool is_ray_pickable() const;

	SoftBody3D();
	~SoftBody3D();

private:
	void reset_softbody_pin();

	void _make_cache_dirty();
	void _update_cache_pin_points_datas();

	void _pin_point_on_physics_server(int p_point_index, bool pin);
	void _add_pinned_point(int p_point_index, const NodePath &p_spatial_attachment_path);
	void _reset_points_offsets();

	void _remove_pinned_point(int p_point_index);
	int _get_pinned_point(int p_point_index, PinnedPoint *&r_point) const;
	int _has_pinned_point(int p_point_index) const;
};

#endif // SOFT_PHYSICS_BODY_H
