/**************************************************************************/
/*  spatial_editor_gizmos.cpp                                             */
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

#include "spatial_editor_gizmos.h"

#include "core/math/convex_hull.h"
#include "core/math/geometry.h"
#include "scene/3d/audio_stream_player_3d.h"
#include "scene/3d/baked_lightmap.h"
#include "scene/3d/collision_polygon.h"
#include "scene/3d/collision_shape.h"
#include "scene/3d/cpu_particles.h"
#include "scene/3d/gi_probe.h"
#include "scene/3d/label_3d.h"
#include "scene/3d/light.h"
#include "scene/3d/listener.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/navigation_mesh_instance.h"
#include "scene/3d/occluder.h"
#include "scene/3d/particles.h"
#include "scene/3d/physics_joint.h"
#include "scene/3d/portal.h"
#include "scene/3d/position_3d.h"
#include "scene/3d/ray_cast.h"
#include "scene/3d/reflection_probe.h"
#include "scene/3d/room.h"
#include "scene/3d/shape_cast.h"
#include "scene/3d/soft_body.h"
#include "scene/3d/spring_arm.h"
#include "scene/3d/sprite_3d.h"
#include "scene/3d/vehicle_body.h"
#include "scene/3d/visibility_notifier.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/capsule_shape.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "scene/resources/cylinder_shape.h"
#include "scene/resources/height_map_shape.h"
#include "scene/resources/occluder_shape.h"
#include "scene/resources/occluder_shape_polygon.h"
#include "scene/resources/plane_shape.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/ray_shape.h"
#include "scene/resources/sphere_shape.h"
#include "scene/resources/surface_tool.h"

#define HANDLE_HALF_SIZE 9.5

bool EditorSpatialGizmo::is_editable() const {
	ERR_FAIL_COND_V(!spatial_node, false);
	Node *edited_root = spatial_node->get_tree()->get_edited_scene_root();
	if (spatial_node == edited_root) {
		return true;
	}
	if (spatial_node->get_owner() == edited_root) {
		return true;
	}

	if (edited_root->is_editable_instance(spatial_node->get_owner())) {
		return true;
	}

	return false;
}

void EditorSpatialGizmo::clear() {
	for (int i = 0; i < instances.size(); i++) {
		if (instances[i].instance.is_valid()) {
			VS::get_singleton()->free(instances[i].instance);
			instances.write[i].instance = RID();
		}
	}

	billboard_handle = false;
	collision_segments.clear();
	collision_mesh = Ref<TriangleMesh>();
	instances.clear();
	handles.clear();
	secondary_handles.clear();
}

void EditorSpatialGizmo::redraw() {
	if (get_script_instance() && get_script_instance()->has_method("redraw")) {
		get_script_instance()->call("redraw");
		return;
	}

	ERR_FAIL_COND(!gizmo_plugin);
	gizmo_plugin->redraw(this);
}

String EditorSpatialGizmo::get_handle_name(int p_idx) const {
	if (get_script_instance() && get_script_instance()->has_method("get_handle_name")) {
		return get_script_instance()->call("get_handle_name", p_idx);
	}

	ERR_FAIL_COND_V(!gizmo_plugin, "");
	return gizmo_plugin->get_handle_name(this, p_idx);
}

bool EditorSpatialGizmo::is_handle_highlighted(int p_idx) const {
	if (get_script_instance() && get_script_instance()->has_method("is_handle_highlighted")) {
		return get_script_instance()->call("is_handle_highlighted", p_idx);
	}

	ERR_FAIL_COND_V(!gizmo_plugin, false);
	return gizmo_plugin->is_handle_highlighted(this, p_idx);
}

Variant EditorSpatialGizmo::get_handle_value(int p_idx) {
	if (get_script_instance() && get_script_instance()->has_method("get_handle_value")) {
		return get_script_instance()->call("get_handle_value", p_idx);
	}

	ERR_FAIL_COND_V(!gizmo_plugin, Variant());
	return gizmo_plugin->get_handle_value(this, p_idx);
}

void EditorSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {
	if (get_script_instance() && get_script_instance()->has_method("set_handle")) {
		get_script_instance()->call("set_handle", p_idx, p_camera, p_point);
		return;
	}

	ERR_FAIL_COND(!gizmo_plugin);
	gizmo_plugin->set_handle(this, p_idx, p_camera, p_point);
}

void EditorSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {
	if (get_script_instance() && get_script_instance()->has_method("commit_handle")) {
		get_script_instance()->call("commit_handle", p_idx, p_restore, p_cancel);
		return;
	}

	ERR_FAIL_COND(!gizmo_plugin);
	gizmo_plugin->commit_handle(this, p_idx, p_restore, p_cancel);
}

void EditorSpatialGizmo::set_spatial_node(Spatial *p_node) {
	ERR_FAIL_NULL(p_node);
	spatial_node = p_node;
}

void EditorSpatialGizmo::Instance::create_instance(Spatial *p_base, bool p_hidden) {
	instance = VS::get_singleton()->instance_create2(mesh->get_rid(), p_base->get_world()->get_scenario());
	VS::get_singleton()->instance_set_portal_mode(instance, VisualServer::INSTANCE_PORTAL_MODE_GLOBAL);
	VS::get_singleton()->instance_attach_object_instance_id(instance, p_base->get_instance_id());
	if (skin_reference.is_valid()) {
		VS::get_singleton()->instance_attach_skeleton(instance, skin_reference->get_skeleton());
	}
	if (extra_margin) {
		VS::get_singleton()->instance_set_extra_visibility_margin(instance, 1);
	}
	VS::get_singleton()->instance_geometry_set_cast_shadows_setting(instance, VS::SHADOW_CASTING_SETTING_OFF);
	int layer = p_hidden ? 0 : 1 << SpatialEditorViewport::GIZMO_EDIT_LAYER;
	VS::get_singleton()->instance_set_layer_mask(instance, layer); //gizmos are 26
}

void EditorSpatialGizmo::add_mesh(const Ref<Mesh> &p_mesh, bool p_billboard, const Ref<SkinReference> &p_skin_reference, const Ref<Material> &p_material) {
	ERR_FAIL_COND(!spatial_node);
	ERR_FAIL_COND_MSG(!p_mesh.is_valid(), "EditorSpatialGizmo.add_mesh() requires a valid Mesh resource.");

	Instance ins;

	ins.billboard = p_billboard;
	ins.mesh = p_mesh;
	ins.skin_reference = p_skin_reference;
	ins.material = p_material;
	if (valid) {
		ins.create_instance(spatial_node, hidden);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
		if (ins.material.is_valid()) {
			VS::get_singleton()->instance_geometry_set_material_override(ins.instance, p_material->get_rid());
		}
	}

	instances.push_back(ins);
}

void EditorSpatialGizmo::add_lines(const Vector<Vector3> &p_lines, const Ref<Material> &p_material, bool p_billboard, const Color &p_modulate) {
	if (p_lines.empty()) {
		return;
	}

	ERR_FAIL_COND(!spatial_node);
	Instance ins;

	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	Array a;
	a.resize(Mesh::ARRAY_MAX);

	a[Mesh::ARRAY_VERTEX] = p_lines;

	PoolVector<Color> color;
	color.resize(p_lines.size());
	{
		PoolVector<Color>::Write w = color.write();
		for (int i = 0; i < p_lines.size(); i++) {
			if (is_selected()) {
				w[i] = Color(1, 1, 1, 0.8) * p_modulate;
			} else {
				w[i] = Color(1, 1, 1, 0.2) * p_modulate;
			}
		}
	}

	a[Mesh::ARRAY_COLOR] = color;

	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, a);
	mesh->surface_set_material(0, p_material);

	if (p_billboard) {
		float md = 0;
		for (int i = 0; i < p_lines.size(); i++) {
			md = MAX(0, p_lines[i].length());
		}
		if (md) {
			mesh->set_custom_aabb(AABB(Vector3(-md, -md, -md), Vector3(md, md, md) * 2.0));
		}
	}

	ins.billboard = p_billboard;
	ins.mesh = mesh;
	if (valid) {
		ins.create_instance(spatial_node, hidden);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
	}

	instances.push_back(ins);
}

void EditorSpatialGizmo::add_vertices(const Vector<Vector3> &p_vertices, const Ref<Material> &p_material, Mesh::PrimitiveType p_primitive_type, bool p_billboard, const Color &p_modulate) {
	if (p_vertices.empty()) {
		return;
	}

	ERR_FAIL_COND(!spatial_node);
	Instance ins;

	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	Array a;
	a.resize(Mesh::ARRAY_MAX);

	a[Mesh::ARRAY_VERTEX] = p_vertices;

	PoolVector<Color> color;
	color.resize(p_vertices.size());
	{
		PoolVector<Color>::Write w = color.write();
		for (int i = 0; i < p_vertices.size(); i++) {
			if (is_selected()) {
				w[i] = Color(1, 1, 1, 0.8) * p_modulate;
			} else {
				w[i] = Color(1, 1, 1, 0.2) * p_modulate;
			}
		}
	}

	a[Mesh::ARRAY_COLOR] = color;

	mesh->add_surface_from_arrays(p_primitive_type, a);
	mesh->surface_set_material(0, p_material);

	if (p_billboard) {
		float md = 0;
		for (int i = 0; i < p_vertices.size(); i++) {
			md = MAX(0, p_vertices[i].length());
		}
		if (md) {
			mesh->set_custom_aabb(AABB(Vector3(-md, -md, -md), Vector3(md, md, md) * 2.0));
		}
	}

	ins.billboard = p_billboard;
	ins.mesh = mesh;
	if (valid) {
		ins.create_instance(spatial_node, hidden);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
	}

	instances.push_back(ins);
}

void EditorSpatialGizmo::add_unscaled_billboard(const Ref<Material> &p_material, float p_scale, const Color &p_modulate) {
	ERR_FAIL_COND(!spatial_node);
	Instance ins;

	Vector<Vector3> vs;
	Vector<Vector2> uv;
	Vector<Color> colors;

	vs.push_back(Vector3(-p_scale, p_scale, 0));
	vs.push_back(Vector3(p_scale, p_scale, 0));
	vs.push_back(Vector3(p_scale, -p_scale, 0));
	vs.push_back(Vector3(-p_scale, -p_scale, 0));

	uv.push_back(Vector2(0, 0));
	uv.push_back(Vector2(1, 0));
	uv.push_back(Vector2(1, 1));
	uv.push_back(Vector2(0, 1));

	colors.push_back(p_modulate);
	colors.push_back(p_modulate);
	colors.push_back(p_modulate);
	colors.push_back(p_modulate);

	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	Array a;
	a.resize(Mesh::ARRAY_MAX);
	a[Mesh::ARRAY_VERTEX] = vs;
	a[Mesh::ARRAY_TEX_UV] = uv;
	a[Mesh::ARRAY_COLOR] = colors;
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLE_FAN, a);
	mesh->surface_set_material(0, p_material);

	float md = 0;
	for (int i = 0; i < vs.size(); i++) {
		md = MAX(0, vs[i].length());
	}
	if (md) {
		mesh->set_custom_aabb(AABB(Vector3(-md, -md, -md), Vector3(md, md, md) * 2.0));
	}

	selectable_icon_size = p_scale;
	mesh->set_custom_aabb(AABB(Vector3(-selectable_icon_size, -selectable_icon_size, -selectable_icon_size) * 100.0f, Vector3(selectable_icon_size, selectable_icon_size, selectable_icon_size) * 200.0f));

	ins.mesh = mesh;
	ins.unscaled = true;
	ins.billboard = true;
	if (valid) {
		ins.create_instance(spatial_node, hidden);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
	}

	selectable_icon_size = p_scale;

	instances.push_back(ins);
}

void EditorSpatialGizmo::add_collision_triangles(const Ref<TriangleMesh> &p_tmesh) {
	collision_mesh = p_tmesh;
}

void EditorSpatialGizmo::add_collision_segments(const Vector<Vector3> &p_lines) {
	int from = collision_segments.size();
	collision_segments.resize(from + p_lines.size());
	for (int i = 0; i < p_lines.size(); i++) {
		collision_segments.write[from + i] = p_lines[i];
	}
}

void EditorSpatialGizmo::add_handles(const Vector<Vector3> &p_handles, const Ref<Material> &p_material, bool p_billboard, bool p_secondary) {
	billboard_handle = p_billboard;

	if (!is_selected() || !is_editable()) {
		return;
	}

	ERR_FAIL_COND(!spatial_node);

	Instance ins;

	Ref<ArrayMesh> mesh = memnew(ArrayMesh);

	Array a;
	a.resize(VS::ARRAY_MAX);
	a[VS::ARRAY_VERTEX] = p_handles;
	PoolVector<Color> colors;
	{
		colors.resize(p_handles.size());
		PoolVector<Color>::Write w = colors.write();
		for (int i = 0; i < p_handles.size(); i++) {
			Color col(1, 1, 1, 1);
			if (is_handle_highlighted(i)) {
				col = Color(0, 0, 1, 0.9);
			}

			if (SpatialEditor::get_singleton()->get_over_gizmo_handle() != i) {
				col.a = 0.8;
			}

			w[i] = col;
		}
	}
	a[VS::ARRAY_COLOR] = colors;
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_POINTS, a);
	mesh->surface_set_material(0, p_material);

	if (p_billboard) {
		float md = 0;
		for (int i = 0; i < p_handles.size(); i++) {
			md = MAX(0, p_handles[i].length());
		}
		if (md) {
			mesh->set_custom_aabb(AABB(Vector3(-md, -md, -md), Vector3(md, md, md) * 2.0));
		}
	}

	ins.mesh = mesh;
	ins.billboard = p_billboard;
	ins.extra_margin = true;
	if (valid) {
		ins.create_instance(spatial_node, hidden);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
	}
	instances.push_back(ins);
	if (!p_secondary) {
		int chs = handles.size();
		handles.resize(chs + p_handles.size());
		for (int i = 0; i < p_handles.size(); i++) {
			handles.write[i + chs] = p_handles[i];
		}
	} else {
		int chs = secondary_handles.size();
		secondary_handles.resize(chs + p_handles.size());
		for (int i = 0; i < p_handles.size(); i++) {
			secondary_handles.write[i + chs] = p_handles[i];
		}
	}
}

void EditorSpatialGizmo::add_solid_box(Ref<Material> &p_material, Vector3 p_size, Vector3 p_position) {
	ERR_FAIL_COND(!spatial_node);

	CubeMesh cubem;
	cubem.set_size(p_size);

	Array arrays = cubem.surface_get_arrays(0);
	PoolVector3Array vertex = arrays[VS::ARRAY_VERTEX];
	PoolVector3Array::Write w = vertex.write();

	for (int i = 0; i < vertex.size(); ++i) {
		w[i] += p_position;
	}

	arrays[VS::ARRAY_VERTEX] = vertex;

	Ref<ArrayMesh> m = memnew(ArrayMesh);
	m->add_surface_from_arrays(cubem.surface_get_primitive_type(0), arrays);
	m->surface_set_material(0, p_material);
	add_mesh(m);
}

bool EditorSpatialGizmo::intersect_frustum(const Camera *p_camera, const Vector<Plane> &p_frustum) {
	ERR_FAIL_COND_V(!spatial_node, false);
	ERR_FAIL_COND_V(!valid, false);

	if (hidden && !gizmo_plugin->is_selectable_when_hidden()) {
		return false;
	}

	if (selectable_icon_size > 0.0f) {
		Vector3 origin = spatial_node->get_global_transform().get_origin();

		const Plane *p = p_frustum.ptr();
		int fc = p_frustum.size();

		bool any_out = false;

		for (int j = 0; j < fc; j++) {
			if (p[j].is_point_over(origin)) {
				any_out = true;
				break;
			}
		}

		return !any_out;
	}

	if (collision_segments.size()) {
		const Plane *p = p_frustum.ptr();
		int fc = p_frustum.size();

		int vc = collision_segments.size();
		const Vector3 *vptr = collision_segments.ptr();
		Transform t = spatial_node->get_global_transform();

		bool any_out = false;
		for (int j = 0; j < fc; j++) {
			for (int i = 0; i < vc; i++) {
				Vector3 v = t.xform(vptr[i]);
				if (p[j].is_point_over(v)) {
					any_out = true;
					break;
				}
			}
			if (any_out) {
				break;
			}
		}

		if (!any_out) {
			return true;
		}
	}

	if (collision_mesh.is_valid()) {
		Transform t = spatial_node->get_global_transform();

		Vector3 mesh_scale = t.get_basis().get_scale();
		t.orthonormalize();

		Transform it = t.affine_inverse();

		Vector<Plane> transformed_frustum;

		for (int i = 0; i < p_frustum.size(); i++) {
			transformed_frustum.push_back(it.xform(p_frustum[i]));
		}

		Vector<Vector3> convex_points = Geometry::compute_convex_mesh_points(p_frustum.ptr(), p_frustum.size());

		if (collision_mesh->inside_convex_shape(transformed_frustum.ptr(), transformed_frustum.size(), convex_points.ptr(), convex_points.size(), mesh_scale)) {
			return true;
		}
	}

	return false;
}

bool EditorSpatialGizmo::intersect_ray(Camera *p_camera, const Point2 &p_point, Vector3 &r_pos, Vector3 &r_normal, int *r_gizmo_handle, bool p_sec_first) {
	ERR_FAIL_COND_V(!spatial_node, false);
	ERR_FAIL_COND_V(!valid, false);

	if (hidden && !gizmo_plugin->is_selectable_when_hidden()) {
		return false;
	}

	if (r_gizmo_handle && !hidden) {
		Transform t = spatial_node->get_global_transform();
		if (billboard_handle) {
			t.set_look_at(t.origin, t.origin - p_camera->get_transform().basis.get_axis(2), p_camera->get_transform().basis.get_axis(1));
		}

		float min_d = 1e20;
		int idx = -1;

		for (int i = 0; i < secondary_handles.size(); i++) {
			Vector3 hpos = t.xform(secondary_handles[i]);
			Vector2 p = p_camera->unproject_position(hpos);

			if (p.distance_to(p_point) < HANDLE_HALF_SIZE) {
				real_t dp = p_camera->get_transform().origin.distance_to(hpos);
				if (dp < min_d) {
					r_pos = t.xform(hpos);
					r_normal = p_camera->get_transform().basis.get_axis(2);
					min_d = dp;
					idx = i + handles.size();
				}
			}
		}

		if (p_sec_first && idx != -1) {
			*r_gizmo_handle = idx;
			return true;
		}

		min_d = 1e20;

		for (int i = 0; i < handles.size(); i++) {
			Vector3 hpos = t.xform(handles[i]);
			Vector2 p;
			if (!p_camera->safe_unproject_position(hpos, p)) {
				continue;
			}

			if (p.distance_to(p_point) < HANDLE_HALF_SIZE) {
				real_t dp = p_camera->get_transform().origin.distance_to(hpos);
				if (dp < min_d) {
					r_pos = t.xform(hpos);
					r_normal = p_camera->get_transform().basis.get_axis(2);
					min_d = dp;
					idx = i;
				}
			}
		}

		if (idx >= 0) {
			*r_gizmo_handle = idx;
			return true;
		}
	}

	if (selectable_icon_size > 0.0f) {
		Transform t = spatial_node->get_global_transform();
		Vector3 camera_position = p_camera->get_camera_transform().origin;
		if (camera_position.distance_squared_to(t.origin) > 0.01) {
			t.set_look_at(t.origin, camera_position, Vector3(0, 1, 0));
		}

		float scale = t.origin.distance_to(p_camera->get_camera_transform().origin);

		if (p_camera->get_projection() == Camera::PROJECTION_ORTHOGONAL) {
			float aspect = p_camera->get_viewport()->get_visible_rect().size.aspect();
			float size = p_camera->get_size();
			scale = size / aspect;
		}

		Point2 center = p_camera->unproject_position(t.origin);

		Transform orig_camera_transform = p_camera->get_camera_transform();

		if (orig_camera_transform.origin.distance_squared_to(t.origin) > 0.01 &&
				ABS(orig_camera_transform.basis.get_axis(Vector3::AXIS_Z).dot(Vector3(0, 1, 0))) < 0.99) {
			p_camera->look_at(t.origin, Vector3(0, 1, 0));
		}

		Vector3 c0 = t.xform(Vector3(selectable_icon_size, selectable_icon_size, 0) * scale);
		Vector3 c1 = t.xform(Vector3(-selectable_icon_size, -selectable_icon_size, 0) * scale);

		Point2 p0 = p_camera->unproject_position(c0);
		Point2 p1 = p_camera->unproject_position(c1);

		p_camera->set_global_transform(orig_camera_transform);

		Rect2 rect(p0, (p1 - p0).abs());

		rect.set_position(center - rect.get_size() / 2.0);

		if (rect.has_point(p_point)) {
			r_pos = t.origin;
			r_normal = -p_camera->project_ray_normal(p_point);
			return true;
		}
	}

	if (collision_segments.size()) {
		Plane camp(p_camera->get_transform().origin, (-p_camera->get_transform().basis.get_axis(2)).normalized());

		int vc = collision_segments.size();
		const Vector3 *vptr = collision_segments.ptr();
		Transform t = spatial_node->get_global_transform();
		if (billboard_handle) {
			t.set_look_at(t.origin, t.origin - p_camera->get_transform().basis.get_axis(2), p_camera->get_transform().basis.get_axis(1));
		}

		Vector3 cp;
		float cpd = 1e20;

		for (int i = 0; i < vc / 2; i++) {
			Vector3 a = t.xform(vptr[i * 2 + 0]);
			Vector3 b = t.xform(vptr[i * 2 + 1]);
			Vector2 s[2];
			if (!p_camera->safe_unproject_position(a, s[0])) {
				continue;
			}
			if (!p_camera->safe_unproject_position(b, s[1])) {
				continue;
			}

			Vector2 p = Geometry::get_closest_point_to_segment_2d(p_point, s);

			float pd = p.distance_to(p_point);

			if (pd < cpd) {
				float d = s[0].distance_to(s[1]);
				Vector3 tcp;
				if (d > 0) {
					float d2 = s[0].distance_to(p) / d;
					tcp = a + (b - a) * d2;

				} else {
					tcp = a;
				}

				if (camp.distance_to(tcp) < p_camera->get_znear()) {
					continue;
				}
				cp = tcp;
				cpd = pd;
			}
		}

		if (cpd < 8) {
			r_pos = cp;
			r_normal = -p_camera->project_ray_normal(p_point);
			return true;
		}
	}

	if (collision_mesh.is_valid()) {
		Transform gt = spatial_node->get_global_transform();

		if (billboard_handle) {
			gt.set_look_at(gt.origin, gt.origin - p_camera->get_transform().basis.get_axis(2), p_camera->get_transform().basis.get_axis(1));
		}

		Transform ai = gt.affine_inverse();
		Vector3 ray_from = ai.xform(p_camera->project_ray_origin(p_point));
		Vector3 ray_dir = ai.basis.xform(p_camera->project_ray_normal(p_point)).normalized();
		Vector3 rpos, rnorm;

		if (collision_mesh->intersect_ray(ray_from, ray_dir, rpos, rnorm)) {
			r_pos = gt.xform(rpos);
			r_normal = gt.basis.xform(rnorm).normalized();
			return true;
		}
	}

	return false;
}

void EditorSpatialGizmo::create() {
	ERR_FAIL_COND(!spatial_node);
	ERR_FAIL_COND(valid);
	valid = true;

	for (int i = 0; i < instances.size(); i++) {
		instances.write[i].create_instance(spatial_node, hidden);
	}

	transform();
}

void EditorSpatialGizmo::transform() {
	ERR_FAIL_COND(!spatial_node);
	ERR_FAIL_COND(!valid);
	for (int i = 0; i < instances.size(); i++) {
		VS::get_singleton()->instance_set_transform(instances[i].instance, spatial_node->get_global_transform());
	}
}

void EditorSpatialGizmo::free() {
	ERR_FAIL_COND(!spatial_node);
	ERR_FAIL_COND(!valid);

	for (int i = 0; i < instances.size(); i++) {
		if (instances[i].instance.is_valid()) {
			VS::get_singleton()->free(instances[i].instance);
			instances.write[i].instance = RID();
		}
	}

	clear();

	valid = false;
}

void EditorSpatialGizmo::set_hidden(bool p_hidden) {
	hidden = p_hidden;
	int layer = hidden ? 0 : 1 << SpatialEditorViewport::GIZMO_EDIT_LAYER;
	for (int i = 0; i < instances.size(); ++i) {
		VS::get_singleton()->instance_set_layer_mask(instances[i].instance, layer);
	}
}

void EditorSpatialGizmo::set_plugin(EditorSpatialGizmoPlugin *p_plugin) {
	gizmo_plugin = p_plugin;
}

void EditorSpatialGizmo::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_lines", "lines", "material", "billboard", "modulate"), &EditorSpatialGizmo::add_lines, DEFVAL(false), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_mesh", "mesh", "billboard", "skeleton", "material"), &EditorSpatialGizmo::add_mesh, DEFVAL(false), DEFVAL(Ref<SkinReference>()), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("add_collision_segments", "segments"), &EditorSpatialGizmo::add_collision_segments);
	ClassDB::bind_method(D_METHOD("add_collision_triangles", "triangles"), &EditorSpatialGizmo::add_collision_triangles);
	ClassDB::bind_method(D_METHOD("add_unscaled_billboard", "material", "default_scale", "modulate"), &EditorSpatialGizmo::add_unscaled_billboard, DEFVAL(1), DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_handles", "handles", "material", "billboard", "secondary"), &EditorSpatialGizmo::add_handles, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_spatial_node", "node"), &EditorSpatialGizmo::_set_spatial_node);
	ClassDB::bind_method(D_METHOD("get_spatial_node"), &EditorSpatialGizmo::get_spatial_node);
	ClassDB::bind_method(D_METHOD("get_plugin"), &EditorSpatialGizmo::get_plugin);
	ClassDB::bind_method(D_METHOD("clear"), &EditorSpatialGizmo::clear);
	ClassDB::bind_method(D_METHOD("set_hidden", "hidden"), &EditorSpatialGizmo::set_hidden);

	BIND_VMETHOD(MethodInfo("redraw"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "get_handle_name", PropertyInfo(Variant::INT, "index")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "is_handle_highlighted", PropertyInfo(Variant::INT, "index")));

	MethodInfo hvget(Variant::NIL, "get_handle_value", PropertyInfo(Variant::INT, "index"));
	hvget.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
	BIND_VMETHOD(hvget);

	BIND_VMETHOD(MethodInfo("set_handle", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::OBJECT, "camera", PROPERTY_HINT_RESOURCE_TYPE, "Camera"), PropertyInfo(Variant::VECTOR2, "point")));
	MethodInfo cm = MethodInfo("commit_handle", PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::NIL, "restore"), PropertyInfo(Variant::BOOL, "cancel"));
	cm.default_arguments.push_back(false);
	BIND_VMETHOD(cm);
}

EditorSpatialGizmo::EditorSpatialGizmo() {
	valid = false;
	billboard_handle = false;
	hidden = false;
	base = nullptr;
	selected = false;
	instanced = false;
	spatial_node = nullptr;
	gizmo_plugin = nullptr;
	selectable_icon_size = -1.0f;
}

EditorSpatialGizmo::~EditorSpatialGizmo() {
	if (gizmo_plugin != nullptr) {
		gizmo_plugin->unregister_gizmo(this);
	}
	clear();
}

Vector3 EditorSpatialGizmo::get_handle_pos(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, handles.size(), Vector3());

	return handles[p_idx];
}

//// light gizmo

LightSpatialGizmoPlugin::LightSpatialGizmoPlugin() {
	// Enable vertex colors for the materials below as the gizmo color depends on the light color.
	create_material("lines_primary", Color(1, 1, 1), false, false, true);
	create_material("lines_secondary", Color(1, 1, 1, 0.35), false, false, true);
	create_material("lines_billboard", Color(1, 1, 1), true, false, true);

	create_icon_material("light_directional_icon", SpatialEditor::get_singleton()->get_icon("GizmoDirectionalLight", "EditorIcons"));
	create_icon_material("light_omni_icon", SpatialEditor::get_singleton()->get_icon("GizmoLight", "EditorIcons"));
	create_icon_material("light_spot_icon", SpatialEditor::get_singleton()->get_icon("GizmoSpotLight", "EditorIcons"));

	create_handle_material("handles");
	create_handle_material("handles_billboard", true);
}

bool LightSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<Light>(p_spatial) != nullptr;
}

String LightSpatialGizmoPlugin::get_name() const {
	return "Lights";
}

int LightSpatialGizmoPlugin::get_priority() const {
	return -1;
}

String LightSpatialGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	if (p_idx == 0) {
		return "Radius";
	} else {
		return "Aperture";
	}
}

Variant LightSpatialGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	Light *light = Object::cast_to<Light>(p_gizmo->get_spatial_node());
	if (p_idx == 0) {
		return light->get_param(Light::PARAM_RANGE);
	}
	if (p_idx == 1) {
		return light->get_param(Light::PARAM_SPOT_ANGLE);
	}

	return Variant();
}

static float _find_closest_angle_to_half_pi_arc(const Vector3 &p_from, const Vector3 &p_to, float p_arc_radius, const Transform &p_arc_xform) {
	//bleh, discrete is simpler
	static const int arc_test_points = 64;
	float min_d = 1e20;
	Vector3 min_p;

	for (int i = 0; i < arc_test_points; i++) {
		float a = i * Math_PI * 0.5 / arc_test_points;
		float an = (i + 1) * Math_PI * 0.5 / arc_test_points;
		Vector3 p = Vector3(Math::cos(a), 0, -Math::sin(a)) * p_arc_radius;
		Vector3 n = Vector3(Math::cos(an), 0, -Math::sin(an)) * p_arc_radius;

		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(p, n, p_from, p_to, ra, rb);

		float d = ra.distance_to(rb);
		if (d < min_d) {
			min_d = d;
			min_p = ra;
		}
	}

	//min_p = p_arc_xform.affine_inverse().xform(min_p);
	float a = (Math_PI * 0.5) - Vector2(min_p.x, -min_p.z).angle();
	return a * 180.0 / Math_PI;
}

void LightSpatialGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {
	Light *light = Object::cast_to<Light>(p_gizmo->get_spatial_node());
	Transform gt = light->get_global_transform();
	Transform gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 s[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };
	if (p_idx == 0) {
		if (Object::cast_to<SpotLight>(light)) {
			Vector3 ra, rb;
			Geometry::get_closest_points_between_segments(Vector3(), Vector3(0, 0, -4096), s[0], s[1], ra, rb);

			float d = -ra.z;
			if (SpatialEditor::get_singleton()->is_snap_enabled()) {
				d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
			}

			if (d <= 0) { // Equal is here for negative zero.
				d = 0;
			}

			light->set_param(Light::PARAM_RANGE, d);
		} else if (Object::cast_to<OmniLight>(light)) {
			Plane cp = Plane(gt.origin, p_camera->get_transform().basis.get_axis(2));

			Vector3 inters;
			if (cp.intersects_ray(ray_from, ray_dir, &inters)) {
				float r = inters.distance_to(gt.origin);
				if (SpatialEditor::get_singleton()->is_snap_enabled()) {
					r = Math::stepify(r, SpatialEditor::get_singleton()->get_translate_snap());
				}

				light->set_param(Light::PARAM_RANGE, r);
			}
		}

	} else if (p_idx == 1) {
		float a = _find_closest_angle_to_half_pi_arc(s[0], s[1], light->get_param(Light::PARAM_RANGE), gt);
		light->set_param(Light::PARAM_SPOT_ANGLE, CLAMP(a, 0.01, 89.99));
	}
}

void LightSpatialGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	Light *light = Object::cast_to<Light>(p_gizmo->get_spatial_node());
	if (p_cancel) {
		light->set_param(p_idx == 0 ? Light::PARAM_RANGE : Light::PARAM_SPOT_ANGLE, p_restore);

	} else if (p_idx == 0) {
		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Light Radius"));
		ur->add_do_method(light, "set_param", Light::PARAM_RANGE, light->get_param(Light::PARAM_RANGE));
		ur->add_undo_method(light, "set_param", Light::PARAM_RANGE, p_restore);
		ur->commit_action();
	} else if (p_idx == 1) {
		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Light Radius"));
		ur->add_do_method(light, "set_param", Light::PARAM_SPOT_ANGLE, light->get_param(Light::PARAM_SPOT_ANGLE));
		ur->add_undo_method(light, "set_param", Light::PARAM_SPOT_ANGLE, p_restore);
		ur->commit_action();
	}
}

void LightSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	Light *light = Object::cast_to<Light>(p_gizmo->get_spatial_node());

	Color color = light->get_color();
	// Make the gizmo color as bright as possible for better visibility
	color.set_hsv(color.get_h(), color.get_s(), 1);

	p_gizmo->clear();

	if (Object::cast_to<DirectionalLight>(light)) {
		Ref<Material> material = get_material("lines_primary", p_gizmo);
		Ref<Material> icon = get_material("light_directional_icon", p_gizmo);

		const int arrow_points = 7;
		const float arrow_length = 1.5;

		Vector3 arrow[arrow_points] = {
			Vector3(0, 0, -1),
			Vector3(0, 0.8, 0),
			Vector3(0, 0.3, 0),
			Vector3(0, 0.3, arrow_length),
			Vector3(0, -0.3, arrow_length),
			Vector3(0, -0.3, 0),
			Vector3(0, -0.8, 0)
		};

		int arrow_sides = 2;

		Vector<Vector3> lines;

		for (int i = 0; i < arrow_sides; i++) {
			for (int j = 0; j < arrow_points; j++) {
				Basis ma(Vector3(0, 0, 1), Math_PI * i / arrow_sides);

				Vector3 v1 = arrow[j] - Vector3(0, 0, arrow_length);
				Vector3 v2 = arrow[(j + 1) % arrow_points] - Vector3(0, 0, arrow_length);

				lines.push_back(ma.xform(v1));
				lines.push_back(ma.xform(v2));
			}
		}

		p_gizmo->add_lines(lines, material, false, color);
		p_gizmo->add_unscaled_billboard(icon, 0.05, color);
	}

	if (Object::cast_to<OmniLight>(light)) {
		// Use both a billboard circle and 3 non-billboard circles for a better sphere-like representation
		const Ref<Material> lines_material = get_material("lines_secondary", p_gizmo);
		const Ref<Material> lines_billboard_material = get_material("lines_billboard", p_gizmo);
		const Ref<Material> icon = get_material("light_omni_icon", p_gizmo);

		OmniLight *on = Object::cast_to<OmniLight>(light);
		const float r = on->get_param(Light::PARAM_RANGE);
		Vector<Vector3> points;
		Vector<Vector3> points_billboard;

		for (int i = 0; i < 120; i++) {
			// Create a circle
			const float ra = Math::deg2rad((float)(i * 3));
			const float rb = Math::deg2rad((float)((i + 1) * 3));
			const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
			const Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

			// Draw axis-aligned circles
			points.push_back(Vector3(a.x, 0, a.y));
			points.push_back(Vector3(b.x, 0, b.y));
			points.push_back(Vector3(0, a.x, a.y));
			points.push_back(Vector3(0, b.x, b.y));
			points.push_back(Vector3(a.x, a.y, 0));
			points.push_back(Vector3(b.x, b.y, 0));

			// Draw a billboarded circle
			points_billboard.push_back(Vector3(a.x, a.y, 0));
			points_billboard.push_back(Vector3(b.x, b.y, 0));
		}

		p_gizmo->add_lines(points, lines_material, true, color);
		p_gizmo->add_lines(points_billboard, lines_billboard_material, true, color);
		p_gizmo->add_unscaled_billboard(icon, 0.05, color);

		Vector<Vector3> handles;
		handles.push_back(Vector3(r, 0, 0));
		p_gizmo->add_handles(handles, get_material("handles_billboard"), true);
	}

	if (Object::cast_to<SpotLight>(light)) {
		const Ref<Material> material_primary = get_material("lines_primary", p_gizmo);
		const Ref<Material> material_secondary = get_material("lines_secondary", p_gizmo);
		const Ref<Material> icon = get_material("light_spot_icon", p_gizmo);

		Vector<Vector3> points_primary;
		Vector<Vector3> points_secondary;
		SpotLight *sl = Object::cast_to<SpotLight>(light);

		float r = sl->get_param(Light::PARAM_RANGE);
		float w = r * Math::sin(Math::deg2rad(sl->get_param(Light::PARAM_SPOT_ANGLE)));
		float d = r * Math::cos(Math::deg2rad(sl->get_param(Light::PARAM_SPOT_ANGLE)));

		for (int i = 0; i < 120; i++) {
			// Draw a circle
			const float ra = Math::deg2rad((float)(i * 3));
			const float rb = Math::deg2rad((float)((i + 1) * 3));
			const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
			const Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * w;

			points_primary.push_back(Vector3(a.x, a.y, -d));
			points_primary.push_back(Vector3(b.x, b.y, -d));

			if (i % 15 == 0) {
				// Draw 8 lines from the cone origin to the sides of the circle
				points_secondary.push_back(Vector3(a.x, a.y, -d));
				points_secondary.push_back(Vector3());
			}
		}

		points_primary.push_back(Vector3(0, 0, -r));
		points_primary.push_back(Vector3());

		p_gizmo->add_lines(points_primary, material_primary, false, color);
		p_gizmo->add_lines(points_secondary, material_secondary, false, color);

		const float ra = 16 * Math_PI * 2.0 / 64.0;
		const Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;

		Vector<Vector3> handles;
		handles.push_back(Vector3(0, 0, -r));
		handles.push_back(Vector3(a.x, a.y, -d));

		p_gizmo->add_handles(handles, get_material("handles"));
		p_gizmo->add_unscaled_billboard(icon, 0.05, color);
	}
}

//////

//// player gizmo
AudioStreamPlayer3DSpatialGizmoPlugin::AudioStreamPlayer3DSpatialGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/stream_player_3d", Color(0.4, 0.8, 1));

	create_icon_material("stream_player_3d_icon", SpatialEditor::get_singleton()->get_icon("GizmoSpatialSamplePlayer", "EditorIcons"));
	create_material("stream_player_3d_material_primary", gizmo_color);
	create_material("stream_player_3d_material_secondary", gizmo_color * Color(1, 1, 1, 0.35));
	create_handle_material("handles");
}

bool AudioStreamPlayer3DSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<AudioStreamPlayer3D>(p_spatial) != nullptr;
}

String AudioStreamPlayer3DSpatialGizmoPlugin::get_name() const {
	return "AudioStreamPlayer3D";
}

int AudioStreamPlayer3DSpatialGizmoPlugin::get_priority() const {
	return -1;
}

String AudioStreamPlayer3DSpatialGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	return "Emission Radius";
}

Variant AudioStreamPlayer3DSpatialGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	AudioStreamPlayer3D *player = Object::cast_to<AudioStreamPlayer3D>(p_gizmo->get_spatial_node());
	return player->get_emission_angle();
}

void AudioStreamPlayer3DSpatialGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {
	AudioStreamPlayer3D *player = Object::cast_to<AudioStreamPlayer3D>(p_gizmo->get_spatial_node());

	Transform gt = player->get_global_transform();
	Transform gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);
	Vector3 ray_to = ray_from + ray_dir * 4096;

	ray_from = gi.xform(ray_from);
	ray_to = gi.xform(ray_to);

	float closest_dist = 1e20;
	float closest_angle = 1e20;

	for (int i = 0; i < 180; i++) {
		float a = i * Math_PI / 180.0;
		float an = (i + 1) * Math_PI / 180.0;

		Vector3 from(Math::sin(a), 0, -Math::cos(a));
		Vector3 to(Math::sin(an), 0, -Math::cos(an));

		Vector3 r1, r2;
		Geometry::get_closest_points_between_segments(from, to, ray_from, ray_to, r1, r2);
		float d = r1.distance_to(r2);
		if (d < closest_dist) {
			closest_dist = d;
			closest_angle = i;
		}
	}

	if (closest_angle < 91) {
		player->set_emission_angle(closest_angle);
	}
}

void AudioStreamPlayer3DSpatialGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	AudioStreamPlayer3D *player = Object::cast_to<AudioStreamPlayer3D>(p_gizmo->get_spatial_node());

	if (p_cancel) {
		player->set_emission_angle(p_restore);

	} else {
		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change AudioStreamPlayer3D Emission Angle"));
		ur->add_do_method(player, "set_emission_angle", player->get_emission_angle());
		ur->add_undo_method(player, "set_emission_angle", p_restore);
		ur->commit_action();
	}
}

void AudioStreamPlayer3DSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	const AudioStreamPlayer3D *player = Object::cast_to<AudioStreamPlayer3D>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	const Ref<Material> icon = get_material("stream_player_3d_icon", p_gizmo);

	if (player->is_emission_angle_enabled()) {
		const float pc = player->get_emission_angle();
		const float ofs = -Math::cos(Math::deg2rad(pc));
		const float radius = Math::sin(Math::deg2rad(pc));

		Vector<Vector3> points_primary;
		points_primary.resize(200);

		for (int i = 0; i < 100; i++) {
			const float a = i * 2.0 * Math_PI / 100.0;
			const float an = (i + 1) * 2.0 * Math_PI / 100.0;

			const Vector3 from(Math::sin(a) * radius, Math::cos(a) * radius, ofs);
			const Vector3 to(Math::sin(an) * radius, Math::cos(an) * radius, ofs);

			points_primary.write[i * 2 + 0] = from;
			points_primary.write[i * 2 + 1] = to;
		}

		const Ref<Material> material_primary = get_material("stream_player_3d_material_primary", p_gizmo);
		p_gizmo->add_lines(points_primary, material_primary);

		Vector<Vector3> points_secondary;
		points_secondary.resize(16);

		for (int i = 0; i < 8; i++) {
			const float a = i * 2.0 * Math_PI / 8.0;
			const Vector3 from(Math::sin(a) * radius, Math::cos(a) * radius, ofs);

			points_secondary.write[i * 2 + 0] = from;
			points_secondary.write[i * 2 + 1] = Vector3();
		}

		const Ref<Material> material_secondary = get_material("stream_player_3d_material_secondary", p_gizmo);
		p_gizmo->add_lines(points_secondary, material_secondary);

		Vector<Vector3> handles;
		const float ha = Math::deg2rad(player->get_emission_angle());
		handles.push_back(Vector3(Math::sin(ha), 0, -Math::cos(ha)));
		p_gizmo->add_handles(handles, get_material("handles"));
	}

	p_gizmo->add_unscaled_billboard(icon, 0.05);
}

//////

ListenerSpatialGizmoPlugin::ListenerSpatialGizmoPlugin() {
	create_icon_material("listener_icon", SpatialEditor::get_singleton()->get_icon("GizmoListener", "EditorIcons"));
}

bool ListenerSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<Listener>(p_spatial) != nullptr;
}

String ListenerSpatialGizmoPlugin::get_name() const {
	return "Listener";
}

int ListenerSpatialGizmoPlugin::get_priority() const {
	return -1;
}

void ListenerSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	const Ref<Material> icon = get_material("listener_icon", p_gizmo);
	p_gizmo->add_unscaled_billboard(icon, 0.05);
}

//////

CameraSpatialGizmoPlugin::CameraSpatialGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/camera", Color(0.8, 0.4, 0.8));

	create_material("camera_material", gizmo_color);
	create_handle_material("handles");
}

bool CameraSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<Camera>(p_spatial) != nullptr;
}

String CameraSpatialGizmoPlugin::get_name() const {
	return "Camera";
}

int CameraSpatialGizmoPlugin::get_priority() const {
	return -1;
}

String CameraSpatialGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	Camera *camera = Object::cast_to<Camera>(p_gizmo->get_spatial_node());

	if (camera->get_projection() == Camera::PROJECTION_PERSPECTIVE) {
		return "FOV";
	} else {
		return "Size";
	}
}

Variant CameraSpatialGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	Camera *camera = Object::cast_to<Camera>(p_gizmo->get_spatial_node());

	if (camera->get_projection() == Camera::PROJECTION_PERSPECTIVE) {
		return camera->get_fov();
	} else {
		return camera->get_size();
	}
}

void CameraSpatialGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {
	Camera *camera = Object::cast_to<Camera>(p_gizmo->get_spatial_node());

	Transform gt = camera->get_global_transform();
	Transform gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 s[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

	if (camera->get_projection() == Camera::PROJECTION_PERSPECTIVE) {
		Transform gt2 = camera->get_global_transform();
		float a = _find_closest_angle_to_half_pi_arc(s[0], s[1], 1.0, gt2);
		camera->set("fov", CLAMP(a * 2.0, 1, 179));
	} else {
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(0, 0, -1), Vector3(4096, 0, -1), s[0], s[1], ra, rb);
		float d = ra.x * 2.0;
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		d = CLAMP(d, 0.1, 16384);

		camera->set("size", d);
	}
}

void CameraSpatialGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	Camera *camera = Object::cast_to<Camera>(p_gizmo->get_spatial_node());

	if (camera->get_projection() == Camera::PROJECTION_PERSPECTIVE) {
		if (p_cancel) {
			camera->set("fov", p_restore);
		} else {
			UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
			ur->create_action(TTR("Change Camera FOV"));
			ur->add_do_property(camera, "fov", camera->get_fov());
			ur->add_undo_property(camera, "fov", p_restore);
			ur->commit_action();
		}

	} else {
		if (p_cancel) {
			camera->set("size", p_restore);
		} else {
			UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
			ur->create_action(TTR("Change Camera Size"));
			ur->add_do_property(camera, "size", camera->get_size());
			ur->add_undo_property(camera, "size", p_restore);
			ur->commit_action();
		}
	}
}

void CameraSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	Camera *camera = Object::cast_to<Camera>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Vector<Vector3> lines;
	Vector<Vector3> handles;

	Ref<Material> material = get_material("camera_material", p_gizmo);

#define ADD_TRIANGLE(m_a, m_b, m_c) \
	{                               \
		lines.push_back(m_a);       \
		lines.push_back(m_b);       \
		lines.push_back(m_b);       \
		lines.push_back(m_c);       \
		lines.push_back(m_c);       \
		lines.push_back(m_a);       \
	}

#define ADD_QUAD(m_a, m_b, m_c, m_d) \
	{                                \
		lines.push_back(m_a);        \
		lines.push_back(m_b);        \
		lines.push_back(m_b);        \
		lines.push_back(m_c);        \
		lines.push_back(m_c);        \
		lines.push_back(m_d);        \
		lines.push_back(m_d);        \
		lines.push_back(m_a);        \
	}

	switch (camera->get_projection()) {
		case Camera::PROJECTION_PERSPECTIVE: {
			// The real FOV is halved for accurate representation
			float fov = camera->get_fov() / 2.0;

			Vector3 side = Vector3(Math::sin(Math::deg2rad(fov)), 0, -Math::cos(Math::deg2rad(fov)));
			Vector3 nside = side;
			nside.x = -nside.x;
			Vector3 up = Vector3(0, side.x, 0);

			ADD_TRIANGLE(Vector3(), side + up, side - up);
			ADD_TRIANGLE(Vector3(), nside + up, nside - up);
			ADD_TRIANGLE(Vector3(), side + up, nside + up);
			ADD_TRIANGLE(Vector3(), side - up, nside - up);

			handles.push_back(side);
			side.x *= 0.25;
			nside.x *= 0.25;
			Vector3 tup(0, up.y * 3 / 2, side.z);
			ADD_TRIANGLE(tup, side + up, nside + up);

		} break;
		case Camera::PROJECTION_ORTHOGONAL: {
			float size = camera->get_size();

			float hsize = size * 0.5;
			Vector3 right(hsize, 0, 0);
			Vector3 up(0, hsize, 0);
			Vector3 back(0, 0, -1.0);
			Vector3 front(0, 0, 0);

			ADD_QUAD(-up - right, -up + right, up + right, up - right);
			ADD_QUAD(-up - right + back, -up + right + back, up + right + back, up - right + back);
			ADD_QUAD(up + right, up + right + back, up - right + back, up - right);
			ADD_QUAD(-up + right, -up + right + back, -up - right + back, -up - right);

			handles.push_back(right + back);

			right.x *= 0.25;
			Vector3 tup(0, up.y * 3 / 2, back.z);
			ADD_TRIANGLE(tup, right + up + back, -right + up + back);

		} break;
		case Camera::PROJECTION_FRUSTUM: {
			float hsize = camera->get_size() / 2.0;

			Vector3 side = Vector3(hsize, 0, -camera->get_znear()).normalized();
			Vector3 nside = side;
			nside.x = -nside.x;
			Vector3 up = Vector3(0, side.x, 0);
			Vector3 offset = Vector3(camera->get_frustum_offset().x, camera->get_frustum_offset().y, 0.0);

			ADD_TRIANGLE(Vector3(), side + up + offset, side - up + offset);
			ADD_TRIANGLE(Vector3(), nside + up + offset, nside - up + offset);
			ADD_TRIANGLE(Vector3(), side + up + offset, nside + up + offset);
			ADD_TRIANGLE(Vector3(), side - up + offset, nside - up + offset);

			side.x *= 0.25;
			nside.x *= 0.25;
			Vector3 tup(0, up.y * 3 / 2, side.z);
			ADD_TRIANGLE(tup + offset, side + up + offset, nside + up + offset);
		}
	}

#undef ADD_TRIANGLE
#undef ADD_QUAD

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_collision_segments(lines);
	p_gizmo->add_handles(handles, get_material("handles"));

	ClippedCamera *clipcam = Object::cast_to<ClippedCamera>(camera);
	if (clipcam) {
		Spatial *parent = Object::cast_to<Spatial>(camera->get_parent());
		if (!parent) {
			return;
		}
		Vector3 cam_normal = -camera->get_global_transform().basis.get_axis(Vector3::AXIS_Z).normalized();
		Vector3 cam_x = camera->get_global_transform().basis.get_axis(Vector3::AXIS_X).normalized();
		Vector3 cam_y = camera->get_global_transform().basis.get_axis(Vector3::AXIS_Y).normalized();
		Vector3 cam_pos = camera->get_global_transform().origin;
		Vector3 parent_pos = parent->get_global_transform().origin;

		Plane parent_plane(parent_pos, cam_normal);
		Vector3 ray_from = parent_plane.project(cam_pos);

		lines.clear();
		lines.push_back(ray_from + cam_x * 0.5 + cam_y * 0.5);
		lines.push_back(ray_from + cam_x * 0.5 + cam_y * -0.5);

		lines.push_back(ray_from + cam_x * 0.5 + cam_y * -0.5);
		lines.push_back(ray_from + cam_x * -0.5 + cam_y * -0.5);

		lines.push_back(ray_from + cam_x * -0.5 + cam_y * -0.5);
		lines.push_back(ray_from + cam_x * -0.5 + cam_y * 0.5);

		lines.push_back(ray_from + cam_x * -0.5 + cam_y * 0.5);
		lines.push_back(ray_from + cam_x * 0.5 + cam_y * 0.5);

		if (parent_plane.distance_to(cam_pos) < 0) {
			lines.push_back(ray_from);
			lines.push_back(cam_pos);
		}

		Transform local = camera->get_global_transform().affine_inverse();
		for (int i = 0; i < lines.size(); i++) {
			lines.write[i] = local.xform(lines[i]);
		}

		p_gizmo->add_lines(lines, material);
	}
}

//////

MeshInstanceSpatialGizmoPlugin::MeshInstanceSpatialGizmoPlugin() {
}

bool MeshInstanceSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<MeshInstance>(p_spatial) != nullptr && Object::cast_to<SoftBody>(p_spatial) == nullptr;
}

String MeshInstanceSpatialGizmoPlugin::get_name() const {
	return "MeshInstance";
}

int MeshInstanceSpatialGizmoPlugin::get_priority() const {
	return -1;
}

bool MeshInstanceSpatialGizmoPlugin::can_be_hidden() const {
	return false;
}

void MeshInstanceSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	MeshInstance *mesh = Object::cast_to<MeshInstance>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Ref<Mesh> m = mesh->get_mesh();

	if (!m.is_valid()) {
		return; //none
	}

	Ref<TriangleMesh> tm = m->generate_triangle_mesh();
	if (tm.is_valid()) {
		p_gizmo->add_collision_triangles(tm);
	}
}

/////
SpriteBase3DSpatialGizmoPlugin::SpriteBase3DSpatialGizmoPlugin() {
}

bool SpriteBase3DSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<SpriteBase3D>(p_spatial) != nullptr;
}

String SpriteBase3DSpatialGizmoPlugin::get_name() const {
	return "SpriteBase3D";
}

int SpriteBase3DSpatialGizmoPlugin::get_priority() const {
	return -1;
}

bool SpriteBase3DSpatialGizmoPlugin::can_be_hidden() const {
	return false;
}

void SpriteBase3DSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	SpriteBase3D *sprite_base = Object::cast_to<SpriteBase3D>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Ref<TriangleMesh> tm = sprite_base->generate_triangle_mesh();
	if (tm.is_valid()) {
		p_gizmo->add_collision_triangles(tm);
	}
}

///

Label3DSpatialGizmoPlugin::Label3DSpatialGizmoPlugin() {
}

bool Label3DSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<Label3D>(p_spatial) != nullptr;
}

String Label3DSpatialGizmoPlugin::get_name() const {
	return "Label3D";
}

int Label3DSpatialGizmoPlugin::get_priority() const {
	return -1;
}

bool Label3DSpatialGizmoPlugin::can_be_hidden() const {
	return false;
}

void Label3DSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	Label3D *label = Object::cast_to<Label3D>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Ref<TriangleMesh> tm = label->generate_triangle_mesh();
	if (tm.is_valid()) {
		p_gizmo->add_collision_triangles(tm);
	}
}

///

Position3DSpatialGizmoPlugin::Position3DSpatialGizmoPlugin() {
	pos3d_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));
	cursor_points = Vector<Vector3>();

	PoolVector<Color> cursor_colors;
	const float cs = 0.25;
	// Add more points to create a "hard stop" in the color gradient.
	cursor_points.push_back(Vector3(+cs, 0, 0));
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3(-cs, 0, 0));

	cursor_points.push_back(Vector3(0, +cs, 0));
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3(0, -cs, 0));

	cursor_points.push_back(Vector3(0, 0, +cs));
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3());
	cursor_points.push_back(Vector3(0, 0, -cs));

	// Use the axis color which is brighter for the positive axis.
	// Use a darkened axis color for the negative axis.
	// This makes it possible to see in which direction the Position3D node is rotated
	// (which can be important depending on how it's used).
	const Color color_x = EditorNode::get_singleton()->get_gui_base()->get_color("axis_x_color", "Editor");
	cursor_colors.push_back(color_x);
	cursor_colors.push_back(color_x);
	// FIXME: Use less strong darkening factor once GH-48573 is fixed.
	// The current darkening factor compensates for lines being too bright in the 3D editor.
	cursor_colors.push_back(color_x.linear_interpolate(Color(0, 0, 0), 0.75));
	cursor_colors.push_back(color_x.linear_interpolate(Color(0, 0, 0), 0.75));

	const Color color_y = EditorNode::get_singleton()->get_gui_base()->get_color("axis_y_color", "Editor");
	cursor_colors.push_back(color_y);
	cursor_colors.push_back(color_y);
	cursor_colors.push_back(color_y.linear_interpolate(Color(0, 0, 0), 0.75));
	cursor_colors.push_back(color_y.linear_interpolate(Color(0, 0, 0), 0.75));

	const Color color_z = EditorNode::get_singleton()->get_gui_base()->get_color("axis_z_color", "Editor");
	cursor_colors.push_back(color_z);
	cursor_colors.push_back(color_z);
	cursor_colors.push_back(color_z.linear_interpolate(Color(0, 0, 0), 0.75));
	cursor_colors.push_back(color_z.linear_interpolate(Color(0, 0, 0), 0.75));

	Ref<Material3D> mat = memnew(SpatialMaterial);
	mat->set_flag(Material3D::FLAG_UNSHADED, true);
	mat->set_flag(Material3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	mat->set_flag(Material3D::FLAG_SRGB_VERTEX_COLOR, true);
	mat->set_feature(Material3D::FEATURE_TRANSPARENT, true);
	mat->set_line_width(3);
	Array d;
	d.resize(VS::ARRAY_MAX);
	d[Mesh::ARRAY_VERTEX] = cursor_points;
	d[Mesh::ARRAY_COLOR] = cursor_colors;
	pos3d_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, d);
	pos3d_mesh->surface_set_material(0, mat);
}

bool Position3DSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<Position3D>(p_spatial) != nullptr;
}

String Position3DSpatialGizmoPlugin::get_name() const {
	return "Position3D";
}

int Position3DSpatialGizmoPlugin::get_priority() const {
	return -1;
}

void Position3DSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	p_gizmo->clear();
	p_gizmo->add_mesh(pos3d_mesh);
	p_gizmo->add_collision_segments(cursor_points);
}

/////

SkeletonSpatialGizmoPlugin::SkeletonSpatialGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/skeleton", Color(1, 0.8, 0.4));
	create_material("skeleton_material", gizmo_color);
}

bool SkeletonSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<Skeleton>(p_spatial) != nullptr;
}

String SkeletonSpatialGizmoPlugin::get_name() const {
	return "Skeleton";
}

int SkeletonSpatialGizmoPlugin::get_priority() const {
	return -1;
}

void SkeletonSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	Skeleton *skel = Object::cast_to<Skeleton>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Ref<Material> material = get_material("skeleton_material", p_gizmo);

	Ref<SurfaceTool> surface_tool(memnew(SurfaceTool));

	surface_tool->begin(Mesh::PRIMITIVE_LINES);
	surface_tool->set_material(material);
	Vector<Transform> grests;
	grests.resize(skel->get_bone_count());

	Vector<int> bones;
	Vector<float> weights;
	bones.resize(4);
	weights.resize(4);

	for (int i = 0; i < 4; i++) {
		bones.write[i] = 0;
		weights.write[i] = 0;
	}

	weights.write[0] = 1;

	AABB aabb;

	Color bonecolor = Color(1.0, 0.4, 0.4, 0.3);
	Color rootcolor = Color(0.4, 1.0, 0.4, 0.1);

	for (int i_bone = 0; i_bone < skel->get_bone_count(); i_bone++) {
		int i = skel->get_process_order(i_bone);

		int parent = skel->get_bone_parent(i);

		if (parent >= 0) {
			grests.write[i] = grests[parent] * skel->get_bone_rest(i);

			Vector3 v0 = grests[parent].origin;
			Vector3 v1 = grests[i].origin;
			Vector3 d = (v1 - v0).normalized();
			float dist = v0.distance_to(v1);

			//find closest axis
			int closest = -1;
			float closest_d = 0.0;

			for (int j = 0; j < 3; j++) {
				float dp = Math::abs(grests[parent].basis[j].normalized().dot(d));
				if (j == 0 || dp > closest_d) {
					closest = j;
				}
			}

			//find closest other
			Vector3 first;
			Vector3 points[4];
			int pointidx = 0;
			for (int j = 0; j < 3; j++) {
				bones.write[0] = parent;
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(rootcolor);
				surface_tool->add_vertex(v0 - grests[parent].basis[j].normalized() * dist * 0.05);
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(rootcolor);
				surface_tool->add_vertex(v0 + grests[parent].basis[j].normalized() * dist * 0.05);

				if (j == closest) {
					continue;
				}

				Vector3 axis;
				if (first == Vector3()) {
					axis = d.cross(d.cross(grests[parent].basis[j])).normalized();
					first = axis;
				} else {
					axis = d.cross(first).normalized();
				}

				for (int k = 0; k < 2; k++) {
					if (k == 1) {
						axis = -axis;
					}
					Vector3 point = v0 + d * dist * 0.2;
					point += axis * dist * 0.1;

					bones.write[0] = parent;
					surface_tool->add_bones(bones);
					surface_tool->add_weights(weights);
					surface_tool->add_color(bonecolor);
					surface_tool->add_vertex(v0);
					surface_tool->add_bones(bones);
					surface_tool->add_weights(weights);
					surface_tool->add_color(bonecolor);
					surface_tool->add_vertex(point);

					bones.write[0] = parent;
					surface_tool->add_bones(bones);
					surface_tool->add_weights(weights);
					surface_tool->add_color(bonecolor);
					surface_tool->add_vertex(point);
					bones.write[0] = i;
					surface_tool->add_bones(bones);
					surface_tool->add_weights(weights);
					surface_tool->add_color(bonecolor);
					surface_tool->add_vertex(v1);
					points[pointidx++] = point;
				}
			}

			SWAP(points[1], points[2]);
			for (int j = 0; j < 4; j++) {
				bones.write[0] = parent;
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(bonecolor);
				surface_tool->add_vertex(points[j]);
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(bonecolor);
				surface_tool->add_vertex(points[(j + 1) % 4]);
			}
		} else {
			grests.write[i] = skel->get_bone_rest(i);
			bones.write[0] = i;
		}
	}

	Ref<ArrayMesh> m = surface_tool->commit();
	p_gizmo->add_mesh(m, false, skel->register_skin(Ref<Skin>()));
}

////

PhysicalBoneSpatialGizmoPlugin::PhysicalBoneSpatialGizmoPlugin() {
	create_material("joint_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/joint"));
}

bool PhysicalBoneSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<PhysicalBone>(p_spatial) != nullptr;
}

String PhysicalBoneSpatialGizmoPlugin::get_name() const {
	return "PhysicalBones";
}

int PhysicalBoneSpatialGizmoPlugin::get_priority() const {
	return -1;
}

void PhysicalBoneSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	p_gizmo->clear();

	PhysicalBone *physical_bone = Object::cast_to<PhysicalBone>(p_gizmo->get_spatial_node());

	if (!physical_bone) {
		return;
	}

	Skeleton *sk(physical_bone->find_skeleton_parent());
	if (!sk) {
		return;
	}

	PhysicalBone *pb(sk->get_physical_bone(physical_bone->get_bone_id()));
	if (!pb) {
		return;
	}

	PhysicalBone *pbp(sk->get_physical_bone_parent(physical_bone->get_bone_id()));
	if (!pbp) {
		return;
	}

	Vector<Vector3> points;

	switch (physical_bone->get_joint_type()) {
		case PhysicalBone::JOINT_TYPE_PIN: {
			JointSpatialGizmoPlugin::CreatePinJointGizmo(physical_bone->get_joint_offset(), points);
		} break;
		case PhysicalBone::JOINT_TYPE_CONE: {
			const PhysicalBone::ConeJointData *cjd(static_cast<const PhysicalBone::ConeJointData *>(physical_bone->get_joint_data()));
			JointSpatialGizmoPlugin::CreateConeTwistJointGizmo(
					physical_bone->get_joint_offset(),
					physical_bone->get_global_transform() * physical_bone->get_joint_offset(),
					pb->get_global_transform(),
					pbp->get_global_transform(),
					cjd->swing_span,
					cjd->twist_span,
					&points,
					&points);
		} break;
		case PhysicalBone::JOINT_TYPE_HINGE: {
			const PhysicalBone::HingeJointData *hjd(static_cast<const PhysicalBone::HingeJointData *>(physical_bone->get_joint_data()));
			JointSpatialGizmoPlugin::CreateHingeJointGizmo(
					physical_bone->get_joint_offset(),
					physical_bone->get_global_transform() * physical_bone->get_joint_offset(),
					pb->get_global_transform(),
					pbp->get_global_transform(),
					hjd->angular_limit_lower,
					hjd->angular_limit_upper,
					hjd->angular_limit_enabled,
					points,
					&points,
					&points);
		} break;
		case PhysicalBone::JOINT_TYPE_SLIDER: {
			const PhysicalBone::SliderJointData *sjd(static_cast<const PhysicalBone::SliderJointData *>(physical_bone->get_joint_data()));
			JointSpatialGizmoPlugin::CreateSliderJointGizmo(
					physical_bone->get_joint_offset(),
					physical_bone->get_global_transform() * physical_bone->get_joint_offset(),
					pb->get_global_transform(),
					pbp->get_global_transform(),
					sjd->angular_limit_lower,
					sjd->angular_limit_upper,
					sjd->linear_limit_lower,
					sjd->linear_limit_upper,
					points,
					&points,
					&points);
		} break;
		case PhysicalBone::JOINT_TYPE_6DOF: {
			const PhysicalBone::SixDOFJointData *sdofjd(static_cast<const PhysicalBone::SixDOFJointData *>(physical_bone->get_joint_data()));
			JointSpatialGizmoPlugin::CreateGeneric6DOFJointGizmo(
					physical_bone->get_joint_offset(),

					physical_bone->get_global_transform() * physical_bone->get_joint_offset(),
					pb->get_global_transform(),
					pbp->get_global_transform(),

					sdofjd->axis_data[0].angular_limit_lower,
					sdofjd->axis_data[0].angular_limit_upper,
					sdofjd->axis_data[0].linear_limit_lower,
					sdofjd->axis_data[0].linear_limit_upper,
					sdofjd->axis_data[0].angular_limit_enabled,
					sdofjd->axis_data[0].linear_limit_enabled,

					sdofjd->axis_data[1].angular_limit_lower,
					sdofjd->axis_data[1].angular_limit_upper,
					sdofjd->axis_data[1].linear_limit_lower,
					sdofjd->axis_data[1].linear_limit_upper,
					sdofjd->axis_data[1].angular_limit_enabled,
					sdofjd->axis_data[1].linear_limit_enabled,

					sdofjd->axis_data[2].angular_limit_lower,
					sdofjd->axis_data[2].angular_limit_upper,
					sdofjd->axis_data[2].linear_limit_lower,
					sdofjd->axis_data[2].linear_limit_upper,
					sdofjd->axis_data[2].angular_limit_enabled,
					sdofjd->axis_data[2].linear_limit_enabled,

					points,
					&points,
					&points);
		} break;
		default:
			return;
	}

	Ref<Material> material = get_material("joint_material", p_gizmo);

	p_gizmo->add_collision_segments(points);
	p_gizmo->add_lines(points, material);
}

/////

RayCastSpatialGizmoPlugin::RayCastSpatialGizmoPlugin() {
	const Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	create_material("shape_material", gizmo_color);
	const float gizmo_value = gizmo_color.get_v();
	const Color gizmo_color_disabled = Color(gizmo_value, gizmo_value, gizmo_value, 0.65);
	create_material("shape_material_disabled", gizmo_color_disabled);
}

bool RayCastSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<RayCast>(p_spatial) != nullptr;
}

String RayCastSpatialGizmoPlugin::get_name() const {
	return "RayCast";
}

int RayCastSpatialGizmoPlugin::get_priority() const {
	return -1;
}

void RayCastSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	RayCast *raycast = Object::cast_to<RayCast>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	const Ref<Material3D> material = raycast->is_enabled() ? raycast->get_debug_material() : get_material("shape_material_disabled");

	p_gizmo->add_lines(raycast->get_debug_line_vertices(), material);

	if (raycast->get_debug_shape_thickness() > 1) {
		p_gizmo->add_vertices(raycast->get_debug_shape_vertices(), material, Mesh::PRIMITIVE_TRIANGLE_STRIP);
	}

	p_gizmo->add_collision_segments(raycast->get_debug_line_vertices());
}

/////

ShapeCastGizmoPlugin::ShapeCastGizmoPlugin() {
	const Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	create_material("shape_material", gizmo_color);
	const float gizmo_value = gizmo_color.get_v();
	const Color gizmo_color_disabled = Color(gizmo_value, gizmo_value, gizmo_value, 0.65);
	create_material("shape_material_disabled", gizmo_color_disabled);
}

bool ShapeCastGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<ShapeCast>(p_spatial) != nullptr;
}

String ShapeCastGizmoPlugin::get_name() const {
	return "ShapeCast";
}

int ShapeCastGizmoPlugin::get_priority() const {
	return -1;
}

void ShapeCastGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	ShapeCast *shapecast = Object::cast_to<ShapeCast>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	const Ref<Material3D> material = shapecast->is_enabled() ? shapecast->get_debug_material() : get_material("shape_material_disabled");

	p_gizmo->add_lines(shapecast->get_debug_line_vertices(), material);

	if (shapecast->get_shape().is_valid()) {
		p_gizmo->add_lines(shapecast->get_debug_shape_vertices(), material);
	}

	p_gizmo->add_collision_segments(shapecast->get_debug_line_vertices());
}

/////

void SpringArmSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	SpringArm *spring_arm = Object::cast_to<SpringArm>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Vector<Vector3> lines;

	lines.push_back(Vector3());
	lines.push_back(Vector3(0, 0, 1.0) * spring_arm->get_length());

	Ref<Material3D> material = get_material("shape_material", p_gizmo);

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_collision_segments(lines);
}

SpringArmSpatialGizmoPlugin::SpringArmSpatialGizmoPlugin() {
	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	create_material("shape_material", gizmo_color);
}

bool SpringArmSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<SpringArm>(p_spatial) != nullptr;
}

String SpringArmSpatialGizmoPlugin::get_name() const {
	return "SpringArm";
}

int SpringArmSpatialGizmoPlugin::get_priority() const {
	return -1;
}

/////

VehicleWheelSpatialGizmoPlugin::VehicleWheelSpatialGizmoPlugin() {
	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	create_material("shape_material", gizmo_color);
}

bool VehicleWheelSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<VehicleWheel>(p_spatial) != nullptr;
}

String VehicleWheelSpatialGizmoPlugin::get_name() const {
	return "VehicleWheel";
}

int VehicleWheelSpatialGizmoPlugin::get_priority() const {
	return -1;
}

void VehicleWheelSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	VehicleWheel *car_wheel = Object::cast_to<VehicleWheel>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Vector<Vector3> points;

	float r = car_wheel->get_radius();
	const int skip = 10;
	for (int i = 0; i <= 360; i += skip) {
		float ra = Math::deg2rad((float)i);
		float rb = Math::deg2rad((float)i + skip);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

		points.push_back(Vector3(0, a.x, a.y));
		points.push_back(Vector3(0, b.x, b.y));

		const int springsec = 4;

		for (int j = 0; j < springsec; j++) {
			float t = car_wheel->get_suspension_rest_length() * 5;
			points.push_back(Vector3(a.x, i / 360.0 * t / springsec + j * (t / springsec), a.y) * 0.2);
			points.push_back(Vector3(b.x, (i + skip) / 360.0 * t / springsec + j * (t / springsec), b.y) * 0.2);
		}
	}

	//travel
	points.push_back(Vector3(0, 0, 0));
	points.push_back(Vector3(0, car_wheel->get_suspension_rest_length(), 0));

	//axis
	points.push_back(Vector3(r * 0.2, car_wheel->get_suspension_rest_length(), 0));
	points.push_back(Vector3(-r * 0.2, car_wheel->get_suspension_rest_length(), 0));
	//axis
	points.push_back(Vector3(r * 0.2, 0, 0));
	points.push_back(Vector3(-r * 0.2, 0, 0));

	//forward line
	points.push_back(Vector3(0, -r, 0));
	points.push_back(Vector3(0, -r, r * 2));
	points.push_back(Vector3(0, -r, r * 2));
	points.push_back(Vector3(r * 2 * 0.2, -r, r * 2 * 0.8));
	points.push_back(Vector3(0, -r, r * 2));
	points.push_back(Vector3(-r * 2 * 0.2, -r, r * 2 * 0.8));

	Ref<Material> material = get_material("shape_material", p_gizmo);

	p_gizmo->add_lines(points, material);
	p_gizmo->add_collision_segments(points);
}

///////////

SoftBodySpatialGizmoPlugin::SoftBodySpatialGizmoPlugin() {
	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	create_material("shape_material", gizmo_color);
	create_handle_material("handles");
}

bool SoftBodySpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<SoftBody>(p_spatial) != nullptr;
}

String SoftBodySpatialGizmoPlugin::get_name() const {
	return "SoftBody";
}

int SoftBodySpatialGizmoPlugin::get_priority() const {
	return -1;
}

bool SoftBodySpatialGizmoPlugin::is_selectable_when_hidden() const {
	return true;
}

void SoftBodySpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	SoftBody *soft_body = Object::cast_to<SoftBody>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	if (!soft_body || soft_body->get_mesh().is_null()) {
		return;
	}

	// find mesh

	Vector<Vector3> lines;

	soft_body->get_mesh()->generate_debug_mesh_lines(lines);

	if (!lines.size()) {
		return;
	}

	Ref<TriangleMesh> tm = soft_body->get_mesh()->generate_triangle_mesh();

	Vector<Vector3> points;
	for (int i = 0; i < soft_body->get_mesh()->get_surface_count(); i++) {
		Array arrays = soft_body->get_mesh()->surface_get_arrays(i);
		ERR_CONTINUE(arrays.empty());

		const PoolVector<Vector3> &vertices = arrays[Mesh::ARRAY_VERTEX];
		PoolVector<Vector3>::Read vertices_read = vertices.read();
		int vertex_count = vertices.size();
		for (int index = 0; index < vertex_count; ++index) {
			points.push_back(vertices_read[index]);
		}
	}

	Ref<Material> material = get_material("shape_material", p_gizmo);

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_handles(points, get_material("handles"));
	p_gizmo->add_collision_triangles(tm);
}

String SoftBodySpatialGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	return "SoftBody pin point";
}

Variant SoftBodySpatialGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	SoftBody *soft_body = Object::cast_to<SoftBody>(p_gizmo->get_spatial_node());
	return Variant(soft_body->is_point_pinned(p_idx));
}

void SoftBodySpatialGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	SoftBody *soft_body = Object::cast_to<SoftBody>(p_gizmo->get_spatial_node());
	soft_body->pin_point_toggle(p_idx);
}

bool SoftBodySpatialGizmoPlugin::is_handle_highlighted(const EditorSpatialGizmo *p_gizmo, int idx) const {
	SoftBody *soft_body = Object::cast_to<SoftBody>(p_gizmo->get_spatial_node());
	return soft_body->is_point_pinned(idx);
}

///////////

VisibilityNotifierGizmoPlugin::VisibilityNotifierGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/visibility_notifier", Color(0.8, 0.5, 0.7));
	create_material("visibility_notifier_material", gizmo_color);
	gizmo_color.a = 0.1;
	create_material("visibility_notifier_solid_material", gizmo_color);
	create_handle_material("handles");
}

bool VisibilityNotifierGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<VisibilityNotifier>(p_spatial) != nullptr;
}

String VisibilityNotifierGizmoPlugin::get_name() const {
	return "VisibilityNotifier";
}

int VisibilityNotifierGizmoPlugin::get_priority() const {
	return -1;
}

String VisibilityNotifierGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	switch (p_idx) {
		case 0:
			return "Size X";
		case 1:
			return "Size Y";
		case 2:
			return "Size Z";
		case 3:
			return "Pos X";
		case 4:
			return "Pos Y";
		case 5:
			return "Pos Z";
	}

	return "";
}

Variant VisibilityNotifierGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	VisibilityNotifier *notifier = Object::cast_to<VisibilityNotifier>(p_gizmo->get_spatial_node());
	return notifier->get_aabb();
}
void VisibilityNotifierGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {
	VisibilityNotifier *notifier = Object::cast_to<VisibilityNotifier>(p_gizmo->get_spatial_node());

	Transform gt = notifier->get_global_transform();

	Transform gi = gt.affine_inverse();

	bool move = p_idx >= 3;
	p_idx = p_idx % 3;

	AABB aabb = notifier->get_aabb();
	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

	Vector3 ofs = aabb.position + aabb.size * 0.5;

	Vector3 axis;
	axis[p_idx] = 1.0;

	if (move) {
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(ofs - axis * 4096, ofs + axis * 4096, sg[0], sg[1], ra, rb);

		float d = ra[p_idx];
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		aabb.position[p_idx] = d - 1.0 - aabb.size[p_idx] * 0.5;
		notifier->set_aabb(aabb);

	} else {
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(ofs, ofs + axis * 4096, sg[0], sg[1], ra, rb);

		float d = ra[p_idx] - ofs[p_idx];
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}
		//resize
		aabb.position[p_idx] = (aabb.position[p_idx] + aabb.size[p_idx] * 0.5) - d;
		aabb.size[p_idx] = d * 2;
		notifier->set_aabb(aabb);
	}
}

void VisibilityNotifierGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	VisibilityNotifier *notifier = Object::cast_to<VisibilityNotifier>(p_gizmo->get_spatial_node());

	if (p_cancel) {
		notifier->set_aabb(p_restore);
		return;
	}

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Notifier AABB"));
	ur->add_do_method(notifier, "set_aabb", notifier->get_aabb());
	ur->add_undo_method(notifier, "set_aabb", p_restore);
	ur->commit_action();
}

void VisibilityNotifierGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	VisibilityNotifier *notifier = Object::cast_to<VisibilityNotifier>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Vector<Vector3> lines;
	AABB aabb = notifier->get_aabb();

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	Vector<Vector3> handles;

	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = aabb.position[i] + aabb.size[i];
		ax[(i + 1) % 3] = aabb.position[(i + 1) % 3] + aabb.size[(i + 1) % 3] * 0.5;
		ax[(i + 2) % 3] = aabb.position[(i + 2) % 3] + aabb.size[(i + 2) % 3] * 0.5;
		handles.push_back(ax);
	}

	Vector3 center = aabb.position + aabb.size * 0.5;
	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = 1.0;
		handles.push_back(center + ax);
		lines.push_back(center);
		lines.push_back(center + ax);
	}

	Ref<Material> material = get_material("visibility_notifier_material", p_gizmo);

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_collision_segments(lines);

	if (p_gizmo->is_selected()) {
		Ref<Material> solid_material = get_material("visibility_notifier_solid_material", p_gizmo);
		p_gizmo->add_solid_box(solid_material, aabb.get_size(), aabb.get_position() + aabb.get_size() / 2.0);
	}

	p_gizmo->add_handles(handles, get_material("handles"));
}

////

CPUParticlesGizmoPlugin::CPUParticlesGizmoPlugin() {
	create_icon_material("particles_icon", SpatialEditor::get_singleton()->get_icon("GizmoCPUParticles", "EditorIcons"));
}

bool CPUParticlesGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<CPUParticles>(p_spatial) != nullptr;
}

String CPUParticlesGizmoPlugin::get_name() const {
	return "CPUParticles";
}

int CPUParticlesGizmoPlugin::get_priority() const {
	return -1;
}

bool CPUParticlesGizmoPlugin::is_selectable_when_hidden() const {
	return true;
}

void CPUParticlesGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	Ref<Material> icon = get_material("particles_icon", p_gizmo);
	p_gizmo->add_unscaled_billboard(icon, 0.05);
}

////

ParticlesGizmoPlugin::ParticlesGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/particles", Color(0.8, 0.7, 0.4));
	create_material("particles_material", gizmo_color);
	gizmo_color.a = 0.1;
	create_material("particles_solid_material", gizmo_color);
	create_icon_material("particles_icon", SpatialEditor::get_singleton()->get_icon("GizmoParticles", "EditorIcons"));
	create_handle_material("handles");
}

bool ParticlesGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<Particles>(p_spatial) != nullptr;
}

String ParticlesGizmoPlugin::get_name() const {
	return "Particles";
}

int ParticlesGizmoPlugin::get_priority() const {
	return -1;
}

bool ParticlesGizmoPlugin::is_selectable_when_hidden() const {
	return true;
}

String ParticlesGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	switch (p_idx) {
		case 0:
			return "Size X";
		case 1:
			return "Size Y";
		case 2:
			return "Size Z";
		case 3:
			return "Pos X";
		case 4:
			return "Pos Y";
		case 5:
			return "Pos Z";
	}

	return "";
}
Variant ParticlesGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	Particles *particles = Object::cast_to<Particles>(p_gizmo->get_spatial_node());
	return particles->get_visibility_aabb();
}
void ParticlesGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {
	Particles *particles = Object::cast_to<Particles>(p_gizmo->get_spatial_node());

	Transform gt = particles->get_global_transform();
	Transform gi = gt.affine_inverse();

	bool move = p_idx >= 3;
	p_idx = p_idx % 3;

	AABB aabb = particles->get_visibility_aabb();
	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

	Vector3 ofs = aabb.position + aabb.size * 0.5;

	Vector3 axis;
	axis[p_idx] = 1.0;

	if (move) {
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(ofs - axis * 4096, ofs + axis * 4096, sg[0], sg[1], ra, rb);

		float d = ra[p_idx];
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		aabb.position[p_idx] = d - 1.0 - aabb.size[p_idx] * 0.5;
		particles->set_visibility_aabb(aabb);

	} else {
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(ofs, ofs + axis * 4096, sg[0], sg[1], ra, rb);

		float d = ra[p_idx] - ofs[p_idx];
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}
		//resize
		aabb.position[p_idx] = (aabb.position[p_idx] + aabb.size[p_idx] * 0.5) - d;
		aabb.size[p_idx] = d * 2;
		particles->set_visibility_aabb(aabb);
	}
}

void ParticlesGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	Particles *particles = Object::cast_to<Particles>(p_gizmo->get_spatial_node());

	if (p_cancel) {
		particles->set_visibility_aabb(p_restore);
		return;
	}

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Particles AABB"));
	ur->add_do_method(particles, "set_visibility_aabb", particles->get_visibility_aabb());
	ur->add_undo_method(particles, "set_visibility_aabb", p_restore);
	ur->commit_action();
}

void ParticlesGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	Particles *particles = Object::cast_to<Particles>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Vector<Vector3> lines;
	AABB aabb = particles->get_visibility_aabb();

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	Vector<Vector3> handles;

	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = aabb.position[i] + aabb.size[i];
		ax[(i + 1) % 3] = aabb.position[(i + 1) % 3] + aabb.size[(i + 1) % 3] * 0.5;
		ax[(i + 2) % 3] = aabb.position[(i + 2) % 3] + aabb.size[(i + 2) % 3] * 0.5;
		handles.push_back(ax);
	}

	Vector3 center = aabb.position + aabb.size * 0.5;
	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = 1.0;
		handles.push_back(center + ax);
		lines.push_back(center);
		lines.push_back(center + ax);
	}

	Ref<Material> material = get_material("particles_material", p_gizmo);
	Ref<Material> icon = get_material("particles_icon", p_gizmo);

	p_gizmo->add_lines(lines, material);

	if (p_gizmo->is_selected()) {
		Ref<Material> solid_material = get_material("particles_solid_material", p_gizmo);
		p_gizmo->add_solid_box(solid_material, aabb.get_size(), aabb.get_position() + aabb.get_size() / 2.0);
	}

	p_gizmo->add_handles(handles, get_material("handles"));
	p_gizmo->add_unscaled_billboard(icon, 0.05);
}
////

ReflectionProbeGizmoPlugin::ReflectionProbeGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/reflection_probe", Color(0.6, 1, 0.5));

	create_material("reflection_probe_material", gizmo_color);

	gizmo_color.a = 0.5;
	create_material("reflection_internal_material", gizmo_color);

	gizmo_color.a = 0.1;
	create_material("reflection_probe_solid_material", gizmo_color);

	create_icon_material("reflection_probe_icon", SpatialEditor::get_singleton()->get_icon("GizmoReflectionProbe", "EditorIcons"));
	create_handle_material("handles");
}

bool ReflectionProbeGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<ReflectionProbe>(p_spatial) != nullptr;
}

String ReflectionProbeGizmoPlugin::get_name() const {
	return "ReflectionProbe";
}

int ReflectionProbeGizmoPlugin::get_priority() const {
	return -1;
}

String ReflectionProbeGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	switch (p_idx) {
		case 0:
			return "Extents X";
		case 1:
			return "Extents Y";
		case 2:
			return "Extents Z";
		case 3:
			return "Origin X";
		case 4:
			return "Origin Y";
		case 5:
			return "Origin Z";
	}

	return "";
}
Variant ReflectionProbeGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	ReflectionProbe *probe = Object::cast_to<ReflectionProbe>(p_gizmo->get_spatial_node());
	return AABB(probe->get_extents(), probe->get_origin_offset());
}
void ReflectionProbeGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {
	ReflectionProbe *probe = Object::cast_to<ReflectionProbe>(p_gizmo->get_spatial_node());
	Transform gt = probe->get_global_transform();

	Transform gi = gt.affine_inverse();

	if (p_idx < 3) {
		Vector3 extents = probe->get_extents();

		Vector3 ray_from = p_camera->project_ray_origin(p_point);
		Vector3 ray_dir = p_camera->project_ray_normal(p_point);

		Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 16384) };

		Vector3 axis;
		axis[p_idx] = 1.0;

		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), axis * 16384, sg[0], sg[1], ra, rb);
		float d = ra[p_idx];
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		extents[p_idx] = d;
		probe->set_extents(extents);
	} else {
		p_idx -= 3;

		Vector3 origin = probe->get_origin_offset();
		origin[p_idx] = 0;

		Vector3 ray_from = p_camera->project_ray_origin(p_point);
		Vector3 ray_dir = p_camera->project_ray_normal(p_point);

		Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 16384) };

		Vector3 axis;
		axis[p_idx] = 1.0;

		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(origin - axis * 16384, origin + axis * 16384, sg[0], sg[1], ra, rb);
		// Adjust the actual position to account for the gizmo handle position
		float d = ra[p_idx] + 0.25;
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		origin[p_idx] = d;
		probe->set_origin_offset(origin);
	}
}

void ReflectionProbeGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	ReflectionProbe *probe = Object::cast_to<ReflectionProbe>(p_gizmo->get_spatial_node());

	AABB restore = p_restore;

	if (p_cancel) {
		probe->set_extents(restore.position);
		probe->set_origin_offset(restore.size);
		return;
	}

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Probe Extents"));
	ur->add_do_method(probe, "set_extents", probe->get_extents());
	ur->add_do_method(probe, "set_origin_offset", probe->get_origin_offset());
	ur->add_undo_method(probe, "set_extents", restore.position);
	ur->add_undo_method(probe, "set_origin_offset", restore.size);
	ur->commit_action();
}

void ReflectionProbeGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	ReflectionProbe *probe = Object::cast_to<ReflectionProbe>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Vector<Vector3> lines;
	Vector<Vector3> internal_lines;
	Vector3 extents = probe->get_extents();

	AABB aabb;
	aabb.position = -extents;
	aabb.size = extents * 2;

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	for (int i = 0; i < 8; i++) {
		Vector3 ep = aabb.get_endpoint(i);
		internal_lines.push_back(probe->get_origin_offset());
		internal_lines.push_back(ep);
	}

	Vector<Vector3> handles;

	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = aabb.position[i] + aabb.size[i];
		handles.push_back(ax);
	}

	for (int i = 0; i < 3; i++) {
		Vector3 orig_handle = probe->get_origin_offset();
		orig_handle[i] -= 0.25;
		lines.push_back(orig_handle);
		handles.push_back(orig_handle);

		orig_handle[i] += 0.5;
		lines.push_back(orig_handle);
	}

	Ref<Material> material = get_material("reflection_probe_material", p_gizmo);
	Ref<Material> material_internal = get_material("reflection_internal_material", p_gizmo);
	Ref<Material> icon = get_material("reflection_probe_icon", p_gizmo);

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_lines(internal_lines, material_internal);

	if (p_gizmo->is_selected()) {
		Ref<Material> solid_material = get_material("reflection_probe_solid_material", p_gizmo);
		p_gizmo->add_solid_box(solid_material, probe->get_extents() * 2.0);
	}

	p_gizmo->add_unscaled_billboard(icon, 0.05);
	p_gizmo->add_handles(handles, get_material("handles"));
}

GIProbeGizmoPlugin::GIProbeGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/gi_probe", Color(0.5, 1, 0.6));

	create_material("gi_probe_material", gizmo_color);

	gizmo_color.a = 0.5;
	create_material("gi_probe_internal_material", gizmo_color);

	gizmo_color.a = 0.1;
	create_material("gi_probe_solid_material", gizmo_color);

	create_icon_material("gi_probe_icon", SpatialEditor::get_singleton()->get_icon("GizmoGIProbe", "EditorIcons"));
	create_handle_material("handles");
}

bool GIProbeGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<GIProbe>(p_spatial) != nullptr;
}

String GIProbeGizmoPlugin::get_name() const {
	return "GIProbe";
}

int GIProbeGizmoPlugin::get_priority() const {
	return -1;
}

String GIProbeGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	switch (p_idx) {
		case 0:
			return "Extents X";
		case 1:
			return "Extents Y";
		case 2:
			return "Extents Z";
	}

	return "";
}
Variant GIProbeGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	GIProbe *probe = Object::cast_to<GIProbe>(p_gizmo->get_spatial_node());
	return probe->get_extents();
}
void GIProbeGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {
	GIProbe *probe = Object::cast_to<GIProbe>(p_gizmo->get_spatial_node());

	Transform gt = probe->get_global_transform();
	Transform gi = gt.affine_inverse();

	Vector3 extents = probe->get_extents();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 16384) };

	Vector3 axis;
	axis[p_idx] = 1.0;

	Vector3 ra, rb;
	Geometry::get_closest_points_between_segments(Vector3(), axis * 16384, sg[0], sg[1], ra, rb);
	float d = ra[p_idx];
	if (SpatialEditor::get_singleton()->is_snap_enabled()) {
		d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
	}

	if (d < 0.001) {
		d = 0.001;
	}

	extents[p_idx] = d;
	probe->set_extents(extents);
}

void GIProbeGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	GIProbe *probe = Object::cast_to<GIProbe>(p_gizmo->get_spatial_node());

	Vector3 restore = p_restore;

	if (p_cancel) {
		probe->set_extents(restore);
		return;
	}

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Probe Extents"));
	ur->add_do_method(probe, "set_extents", probe->get_extents());
	ur->add_undo_method(probe, "set_extents", restore);
	ur->commit_action();
}

void GIProbeGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	GIProbe *probe = Object::cast_to<GIProbe>(p_gizmo->get_spatial_node());

	Ref<Material> material = get_material("gi_probe_material", p_gizmo);
	Ref<Material> icon = get_material("gi_probe_icon", p_gizmo);
	Ref<Material> material_internal = get_material("gi_probe_internal_material", p_gizmo);

	p_gizmo->clear();

	Vector<Vector3> lines;
	Vector3 extents = probe->get_extents();

	static const int subdivs[GIProbe::SUBDIV_MAX] = { 64, 128, 256, 512 };

	AABB aabb = AABB(-extents, extents * 2);
	int subdiv = subdivs[probe->get_subdiv()];
	float cell_size = aabb.get_longest_axis_size() / subdiv;

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	p_gizmo->add_lines(lines, material);

	lines.clear();

	for (int i = 1; i < subdiv; i++) {
		for (int j = 0; j < 3; j++) {
			if (cell_size * i > aabb.size[j]) {
				continue;
			}

			int j_n1 = (j + 1) % 3;
			int j_n2 = (j + 2) % 3;

			for (int k = 0; k < 4; k++) {
				Vector3 from = aabb.position, to = aabb.position;
				from[j] += cell_size * i;
				to[j] += cell_size * i;

				if (k & 1) {
					to[j_n1] += aabb.size[j_n1];
				} else {
					to[j_n2] += aabb.size[j_n2];
				}

				if (k & 2) {
					from[j_n1] += aabb.size[j_n1];
					from[j_n2] += aabb.size[j_n2];
				}

				lines.push_back(from);
				lines.push_back(to);
			}
		}
	}

	p_gizmo->add_lines(lines, material_internal);

	Vector<Vector3> handles;

	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = aabb.position[i] + aabb.size[i];
		handles.push_back(ax);
	}

	if (p_gizmo->is_selected()) {
		Ref<Material> solid_material = get_material("gi_probe_solid_material", p_gizmo);
		p_gizmo->add_solid_box(solid_material, aabb.get_size());
	}

	p_gizmo->add_unscaled_billboard(icon, 0.05);
	p_gizmo->add_handles(handles, get_material("handles"));
}

////

BakedIndirectLightGizmoPlugin::BakedIndirectLightGizmoPlugin() {
	Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/baked_indirect_light", Color(0.5, 0.6, 1));

	create_material("baked_indirect_light_material", gizmo_color);

	gizmo_color.a = 0.1;
	create_material("baked_indirect_light_internal_material", gizmo_color);

	create_icon_material("baked_indirect_light_icon", SpatialEditor::get_singleton()->get_icon("GizmoBakedLightmap", "EditorIcons"));
	create_handle_material("handles");
}

String BakedIndirectLightGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	switch (p_idx) {
		case 0:
			return "Extents X";
		case 1:
			return "Extents Y";
		case 2:
			return "Extents Z";
	}

	return "";
}
Variant BakedIndirectLightGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	BakedLightmap *baker = Object::cast_to<BakedLightmap>(p_gizmo->get_spatial_node());
	return baker->get_extents();
}
void BakedIndirectLightGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {
	BakedLightmap *baker = Object::cast_to<BakedLightmap>(p_gizmo->get_spatial_node());

	Transform gt = baker->get_global_transform();
	Transform gi = gt.affine_inverse();

	Vector3 extents = baker->get_extents();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 16384) };

	Vector3 axis;
	axis[p_idx] = 1.0;

	Vector3 ra, rb;
	Geometry::get_closest_points_between_segments(Vector3(), axis * 16384, sg[0], sg[1], ra, rb);
	float d = ra[p_idx];
	if (SpatialEditor::get_singleton()->is_snap_enabled()) {
		d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
	}

	if (d < 0.001) {
		d = 0.001;
	}

	extents[p_idx] = d;
	baker->set_extents(extents);
}

void BakedIndirectLightGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	BakedLightmap *baker = Object::cast_to<BakedLightmap>(p_gizmo->get_spatial_node());

	Vector3 restore = p_restore;

	if (p_cancel) {
		baker->set_extents(restore);
		return;
	}

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Probe Extents"));
	ur->add_do_method(baker, "set_extents", baker->get_extents());
	ur->add_undo_method(baker, "set_extents", restore);
	ur->commit_action();
}

bool BakedIndirectLightGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<BakedLightmap>(p_spatial) != nullptr;
}

String BakedIndirectLightGizmoPlugin::get_name() const {
	return "BakedLightmap";
}

int BakedIndirectLightGizmoPlugin::get_priority() const {
	return -1;
}

void BakedIndirectLightGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	BakedLightmap *baker = Object::cast_to<BakedLightmap>(p_gizmo->get_spatial_node());

	Ref<Material> material = get_material("baked_indirect_light_material", p_gizmo);
	Ref<Material> icon = get_material("baked_indirect_light_icon", p_gizmo);
	Ref<Material> material_internal = get_material("baked_indirect_light_internal_material", p_gizmo);

	p_gizmo->clear();

	Vector<Vector3> lines;
	Vector3 extents = baker->get_extents();

	AABB aabb = AABB(-extents, extents * 2);

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	p_gizmo->add_lines(lines, material);

	Vector<Vector3> handles;

	for (int i = 0; i < 3; i++) {
		Vector3 ax;
		ax[i] = aabb.position[i] + aabb.size[i];
		handles.push_back(ax);
	}

	if (p_gizmo->is_selected()) {
		p_gizmo->add_solid_box(material_internal, aabb.get_size());
	}

	p_gizmo->add_unscaled_billboard(icon, 0.05);
	p_gizmo->add_handles(handles, get_material("handles"));
}

////

CollisionObjectGizmoPlugin::CollisionObjectGizmoPlugin() {
	const Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	create_material("shape_material", gizmo_color);
	const float gizmo_value = gizmo_color.get_v();
	const Color gizmo_color_disabled = Color(gizmo_value, gizmo_value, gizmo_value, 0.65);
	create_material("shape_material_disabled", gizmo_color_disabled);
}

bool CollisionObjectGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<CollisionObject>(p_spatial) != nullptr;
}

String CollisionObjectGizmoPlugin::get_name() const {
	return "CollisionObject";
}

int CollisionObjectGizmoPlugin::get_priority() const {
	return -2;
}

void CollisionObjectGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	CollisionObject *co = Object::cast_to<CollisionObject>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	List<uint32_t> owners;
	co->get_shape_owners(&owners);
	for (List<uint32_t>::Element *E = owners.front(); E; E = E->next()) {
		uint32_t owner_id = E->get();
		Transform xform = co->shape_owner_get_transform(owner_id);
		Object *owner = co->shape_owner_get_owner(owner_id);
		// Exclude CollisionShape and CollisionPolygon as they have their gizmo.
		if (!Object::cast_to<CollisionShape>(owner) && !Object::cast_to<CollisionPolygon>(owner)) {
			Ref<Material> material = get_material(!co->is_shape_owner_disabled(owner_id) ? "shape_material" : "shape_material_disabled", p_gizmo);
			for (int shape_id = 0; shape_id < co->shape_owner_get_shape_count(owner_id); shape_id++) {
				Ref<Shape> s = co->shape_owner_get_shape(owner_id, shape_id);
				if (s.is_null()) {
					continue;
				}
				SurfaceTool st;
				st.append_from(s->get_debug_mesh(), 0, xform);

				p_gizmo->add_mesh(st.commit(), false, Ref<SkinReference>(), material);
				p_gizmo->add_collision_segments(s->get_debug_mesh_lines());
			}
		}
	}
}

////

CollisionShapeSpatialGizmoPlugin::CollisionShapeSpatialGizmoPlugin() {
	const Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	create_material("shape_material", gizmo_color);
	const float gizmo_value = gizmo_color.get_v();
	const Color gizmo_color_disabled = Color(gizmo_value, gizmo_value, gizmo_value, 0.65);
	create_material("shape_material_disabled", gizmo_color_disabled);
	create_handle_material("handles");
}

bool CollisionShapeSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<CollisionShape>(p_spatial) != nullptr;
}

String CollisionShapeSpatialGizmoPlugin::get_name() const {
	return "CollisionShape";
}

int CollisionShapeSpatialGizmoPlugin::get_priority() const {
	return -1;
}

String CollisionShapeSpatialGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {
	const CollisionShape *cs = Object::cast_to<CollisionShape>(p_gizmo->get_spatial_node());

	Ref<Shape> s = cs->get_shape();
	if (s.is_null()) {
		return "";
	}

	if (Object::cast_to<SphereShape>(*s)) {
		return "Radius";
	}

	if (Object::cast_to<BoxShape>(*s)) {
		return "Extents";
	}

	if (Object::cast_to<CapsuleShape>(*s)) {
		return p_idx == 0 ? "Radius" : "Height";
	}

	if (Object::cast_to<CylinderShape>(*s)) {
		return p_idx == 0 ? "Radius" : "Height";
	}

	if (Object::cast_to<RayShape>(*s)) {
		return "Length";
	}

	return "";
}

Variant CollisionShapeSpatialGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {
	CollisionShape *cs = Object::cast_to<CollisionShape>(p_gizmo->get_spatial_node());

	Ref<Shape> s = cs->get_shape();
	if (s.is_null()) {
		return Variant();
	}

	if (Object::cast_to<SphereShape>(*s)) {
		Ref<SphereShape> ss = s;
		return ss->get_radius();
	}

	if (Object::cast_to<BoxShape>(*s)) {
		Ref<BoxShape> bs = s;
		return bs->get_extents();
	}

	if (Object::cast_to<CapsuleShape>(*s)) {
		Ref<CapsuleShape> cs2 = s;
		return p_idx == 0 ? cs2->get_radius() : cs2->get_height();
	}

	if (Object::cast_to<CylinderShape>(*s)) {
		Ref<CylinderShape> cs2 = s;
		return p_idx == 0 ? cs2->get_radius() : cs2->get_height();
	}

	if (Object::cast_to<RayShape>(*s)) {
		Ref<RayShape> cs2 = s;
		return cs2->get_length();
	}

	return Variant();
}
void CollisionShapeSpatialGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {
	CollisionShape *cs = Object::cast_to<CollisionShape>(p_gizmo->get_spatial_node());

	Ref<Shape> s = cs->get_shape();
	if (s.is_null()) {
		return;
	}

	Transform gt = cs->get_global_transform();
	Transform gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

	if (Object::cast_to<SphereShape>(*s)) {
		Ref<SphereShape> ss = s;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), sg[0], sg[1], ra, rb);
		float d = ra.x;
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		ss->set_radius(d);
	}

	if (Object::cast_to<RayShape>(*s)) {
		Ref<RayShape> rs = s;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), Vector3(0, 0, 4096), sg[0], sg[1], ra, rb);
		float d = ra.z;
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		rs->set_length(d);
	}

	if (Object::cast_to<BoxShape>(*s)) {
		Vector3 axis;
		axis[p_idx] = 1.0;
		Ref<BoxShape> bs = s;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = ra[p_idx];
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		Vector3 he = bs->get_extents();
		he[p_idx] = d;
		bs->set_extents(he);
	}

	if (Object::cast_to<CapsuleShape>(*s)) {
		Vector3 axis;
		axis[p_idx == 0 ? 0 : 2] = 1.0;
		Ref<CapsuleShape> cs2 = s;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = axis.dot(ra);
		if (p_idx == 1) {
			d -= cs2->get_radius();
		}

		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		if (p_idx == 0) {
			cs2->set_radius(d);
		} else if (p_idx == 1) {
			cs2->set_height(d * 2.0);
		}
	}

	if (Object::cast_to<CylinderShape>(*s)) {
		Vector3 axis;
		axis[p_idx == 0 ? 0 : 1] = 1.0;
		Ref<CylinderShape> cs2 = s;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = axis.dot(ra);
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		if (p_idx == 0) {
			cs2->set_radius(d);
		} else if (p_idx == 1) {
			cs2->set_height(d * 2.0);
		}
	}
}
void CollisionShapeSpatialGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {
	CollisionShape *cs = Object::cast_to<CollisionShape>(p_gizmo->get_spatial_node());

	Ref<Shape> s = cs->get_shape();
	if (s.is_null()) {
		return;
	}

	if (Object::cast_to<SphereShape>(*s)) {
		Ref<SphereShape> ss = s;
		if (p_cancel) {
			ss->set_radius(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Sphere Shape Radius"));
		ur->add_do_method(ss.ptr(), "set_radius", ss->get_radius());
		ur->add_undo_method(ss.ptr(), "set_radius", p_restore);
		ur->commit_action();
	}

	if (Object::cast_to<BoxShape>(*s)) {
		Ref<BoxShape> ss = s;
		if (p_cancel) {
			ss->set_extents(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Box Shape Extents"));
		ur->add_do_method(ss.ptr(), "set_extents", ss->get_extents());
		ur->add_undo_method(ss.ptr(), "set_extents", p_restore);
		ur->commit_action();
	}

	if (Object::cast_to<CapsuleShape>(*s)) {
		Ref<CapsuleShape> ss = s;
		if (p_cancel) {
			if (p_idx == 0) {
				ss->set_radius(p_restore);
			} else {
				ss->set_height(p_restore);
			}
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		if (p_idx == 0) {
			ur->create_action(TTR("Change Capsule Shape Radius"));
			ur->add_do_method(ss.ptr(), "set_radius", ss->get_radius());
			ur->add_undo_method(ss.ptr(), "set_radius", p_restore);
		} else {
			ur->create_action(TTR("Change Capsule Shape Height"));
			ur->add_do_method(ss.ptr(), "set_height", ss->get_height());
			ur->add_undo_method(ss.ptr(), "set_height", p_restore);
		}

		ur->commit_action();
	}

	if (Object::cast_to<CylinderShape>(*s)) {
		Ref<CylinderShape> ss = s;
		if (p_cancel) {
			if (p_idx == 0) {
				ss->set_radius(p_restore);
			} else {
				ss->set_height(p_restore);
			}
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		if (p_idx == 0) {
			ur->create_action(TTR("Change Cylinder Shape Radius"));
			ur->add_do_method(ss.ptr(), "set_radius", ss->get_radius());
			ur->add_undo_method(ss.ptr(), "set_radius", p_restore);
		} else {
			ur->create_action(
					///

					////////
					TTR("Change Cylinder Shape Height"));
			ur->add_do_method(ss.ptr(), "set_height", ss->get_height());
			ur->add_undo_method(ss.ptr(), "set_height", p_restore);
		}

		ur->commit_action();
	}

	if (Object::cast_to<RayShape>(*s)) {
		Ref<RayShape> ss = s;
		if (p_cancel) {
			ss->set_length(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Ray Shape Length"));
		ur->add_do_method(ss.ptr(), "set_length", ss->get_length());
		ur->add_undo_method(ss.ptr(), "set_length", p_restore);
		ur->commit_action();
	}
}
void CollisionShapeSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	CollisionShape *cs = Object::cast_to<CollisionShape>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Ref<Shape> s = cs->get_shape();
	if (s.is_null()) {
		return;
	}

	const Ref<Material> material =
			get_material(!cs->is_disabled() ? "shape_material" : "shape_material_disabled", p_gizmo);
	Ref<Material> handles_material = get_material("handles");

	if (Object::cast_to<SphereShape>(*s)) {
		Ref<SphereShape> sp = s;
		float r = sp->get_radius();

		Vector<Vector3> points;

		for (int i = 0; i <= 360; i++) {
			float ra = Math::deg2rad((float)i);
			float rb = Math::deg2rad((float)i + 1);
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

			points.push_back(Vector3(a.x, 0, a.y));
			points.push_back(Vector3(b.x, 0, b.y));
			points.push_back(Vector3(0, a.x, a.y));
			points.push_back(Vector3(0, b.x, b.y));
			points.push_back(Vector3(a.x, a.y, 0));
			points.push_back(Vector3(b.x, b.y, 0));
		}

		Vector<Vector3> collision_segments;

		for (int i = 0; i < 64; i++) {
			float ra = i * Math_PI * 2.0 / 64.0;
			float rb = (i + 1) * Math_PI * 2.0 / 64.0;
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

			collision_segments.push_back(Vector3(a.x, 0, a.y));
			collision_segments.push_back(Vector3(b.x, 0, b.y));
			collision_segments.push_back(Vector3(0, a.x, a.y));
			collision_segments.push_back(Vector3(0, b.x, b.y));
			collision_segments.push_back(Vector3(a.x, a.y, 0));
			collision_segments.push_back(Vector3(b.x, b.y, 0));
		}

		p_gizmo->add_lines(points, material);
		p_gizmo->add_collision_segments(collision_segments);
		Vector<Vector3> handles;
		handles.push_back(Vector3(r, 0, 0));
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<BoxShape>(*s)) {
		Ref<BoxShape> bs = s;
		Vector<Vector3> lines;
		AABB aabb;
		aabb.position = -bs->get_extents();
		aabb.size = aabb.position * -2;

		for (int i = 0; i < 12; i++) {
			Vector3 a, b;
			aabb.get_edge(i, a, b);
			lines.push_back(a);
			lines.push_back(b);
		}

		Vector<Vector3> handles;

		for (int i = 0; i < 3; i++) {
			Vector3 ax;
			ax[i] = bs->get_extents()[i];
			handles.push_back(ax);
		}

		p_gizmo->add_lines(lines, material);
		p_gizmo->add_collision_segments(lines);
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<CapsuleShape>(*s)) {
		Ref<CapsuleShape> cs2 = s;
		float radius = cs2->get_radius();
		float height = cs2->get_height();

		Vector<Vector3> points;

		Vector3 d(0, 0, height * 0.5);
		for (int i = 0; i < 360; i++) {
			float ra = Math::deg2rad((float)i);
			float rb = Math::deg2rad((float)i + 1);
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * radius;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * radius;

			points.push_back(Vector3(a.x, a.y, 0) + d);
			points.push_back(Vector3(b.x, b.y, 0) + d);

			points.push_back(Vector3(a.x, a.y, 0) - d);
			points.push_back(Vector3(b.x, b.y, 0) - d);

			if (i % 90 == 0) {
				points.push_back(Vector3(a.x, a.y, 0) + d);
				points.push_back(Vector3(a.x, a.y, 0) - d);
			}

			Vector3 dud = i < 180 ? d : -d;

			points.push_back(Vector3(0, a.y, a.x) + dud);
			points.push_back(Vector3(0, b.y, b.x) + dud);
			points.push_back(Vector3(a.y, 0, a.x) + dud);
			points.push_back(Vector3(b.y, 0, b.x) + dud);
		}

		p_gizmo->add_lines(points, material);

		Vector<Vector3> collision_segments;

		for (int i = 0; i < 64; i++) {
			float ra = i * Math_PI * 2.0 / 64.0;
			float rb = (i + 1) * Math_PI * 2.0 / 64.0;
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * radius;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * radius;

			collision_segments.push_back(Vector3(a.x, a.y, 0) + d);
			collision_segments.push_back(Vector3(b.x, b.y, 0) + d);

			collision_segments.push_back(Vector3(a.x, a.y, 0) - d);
			collision_segments.push_back(Vector3(b.x, b.y, 0) - d);

			if (i % 16 == 0) {
				collision_segments.push_back(Vector3(a.x, a.y, 0) + d);
				collision_segments.push_back(Vector3(a.x, a.y, 0) - d);
			}

			Vector3 dud = i < 32 ? d : -d;

			collision_segments.push_back(Vector3(0, a.y, a.x) + dud);
			collision_segments.push_back(Vector3(0, b.y, b.x) + dud);
			collision_segments.push_back(Vector3(a.y, 0, a.x) + dud);
			collision_segments.push_back(Vector3(b.y, 0, b.x) + dud);
		}

		p_gizmo->add_collision_segments(collision_segments);

		Vector<Vector3> handles;
		handles.push_back(Vector3(cs2->get_radius(), 0, 0));
		handles.push_back(Vector3(0, 0, cs2->get_height() * 0.5 + cs2->get_radius()));
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<CylinderShape>(*s)) {
		Ref<CylinderShape> cs2 = s;
		float radius = cs2->get_radius();
		float height = cs2->get_height();

		Vector<Vector3> points;

		Vector3 d(0, height * 0.5, 0);
		for (int i = 0; i < 360; i++) {
			float ra = Math::deg2rad((float)i);
			float rb = Math::deg2rad((float)i + 1);
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * radius;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * radius;

			points.push_back(Vector3(a.x, 0, a.y) + d);
			points.push_back(Vector3(b.x, 0, b.y) + d);

			points.push_back(Vector3(a.x, 0, a.y) - d);
			points.push_back(Vector3(b.x, 0, b.y) - d);

			if (i % 90 == 0) {
				points.push_back(Vector3(a.x, 0, a.y) + d);
				points.push_back(Vector3(a.x, 0, a.y) - d);
			}
		}

		p_gizmo->add_lines(points, material);

		Vector<Vector3> collision_segments;

		for (int i = 0; i < 64; i++) {
			float ra = i * Math_PI * 2.0 / 64.0;
			float rb = (i + 1) * Math_PI * 2.0 / 64.0;
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * radius;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * radius;

			collision_segments.push_back(Vector3(a.x, 0, a.y) + d);
			collision_segments.push_back(Vector3(b.x, 0, b.y) + d);

			collision_segments.push_back(Vector3(a.x, 0, a.y) - d);
			collision_segments.push_back(Vector3(b.x, 0, b.y) - d);

			if (i % 16 == 0) {
				collision_segments.push_back(Vector3(a.x, 0, a.y) + d);
				collision_segments.push_back(Vector3(a.x, 0, a.y) - d);
			}
		}

		p_gizmo->add_collision_segments(collision_segments);

		Vector<Vector3> handles;
		handles.push_back(Vector3(cs2->get_radius(), 0, 0));
		handles.push_back(Vector3(0, cs2->get_height() * 0.5, 0));
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<PlaneShape>(*s)) {
		Ref<PlaneShape> ps = s;
		Plane p = ps->get_plane();
		Vector<Vector3> points;

		Vector3 n1 = p.get_any_perpendicular_normal();
		Vector3 n2 = p.normal.cross(n1).normalized();

		Vector3 pface[4] = {
			p.normal * p.d + n1 * 10.0 + n2 * 10.0,
			p.normal * p.d + n1 * 10.0 + n2 * -10.0,
			p.normal * p.d + n1 * -10.0 + n2 * -10.0,
			p.normal * p.d + n1 * -10.0 + n2 * 10.0,
		};

		points.push_back(pface[0]);
		points.push_back(pface[1]);
		points.push_back(pface[1]);
		points.push_back(pface[2]);
		points.push_back(pface[2]);
		points.push_back(pface[3]);
		points.push_back(pface[3]);
		points.push_back(pface[0]);
		points.push_back(p.normal * p.d);
		points.push_back(p.normal * p.d + p.normal * 3);

		p_gizmo->add_lines(points, material);
		p_gizmo->add_collision_segments(points);
	}

	if (Object::cast_to<ConvexPolygonShape>(*s)) {
		PoolVector<Vector3> points = Object::cast_to<ConvexPolygonShape>(*s)->get_points();

		if (points.size() > 3) {
			Vector<Vector3> varr = Variant(points);
			Geometry::MeshData md;
			Error err = ConvexHullComputer::convex_hull(varr, md);
			if (err == OK) {
				Vector<Vector3> points2;
				points2.resize(md.edges.size() * 2);
				for (int i = 0; i < md.edges.size(); i++) {
					points2.write[i * 2 + 0] = md.vertices[md.edges[i].a];
					points2.write[i * 2 + 1] = md.vertices[md.edges[i].b];
				}

				p_gizmo->add_lines(points2, material);
				p_gizmo->add_collision_segments(points2);
			}
		}
	}

	if (Object::cast_to<ConcavePolygonShape>(*s)) {
		Ref<ConcavePolygonShape> cs2 = s;
		Ref<ArrayMesh> mesh = cs2->get_debug_mesh();
		p_gizmo->add_mesh(mesh, false, Ref<SkinReference>(), material);
		p_gizmo->add_collision_segments(cs2->get_debug_mesh_lines());
	}

	if (Object::cast_to<RayShape>(*s)) {
		Ref<RayShape> rs = s;

		Vector<Vector3> points;
		points.push_back(Vector3());
		points.push_back(Vector3(0, 0, rs->get_length()));
		p_gizmo->add_lines(points, material);
		p_gizmo->add_collision_segments(points);
		Vector<Vector3> handles;
		handles.push_back(Vector3(0, 0, rs->get_length()));
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<HeightMapShape>(*s)) {
		Ref<HeightMapShape> hms = s;

		Ref<ArrayMesh> mesh = hms->get_debug_mesh();
		p_gizmo->add_mesh(mesh, false, Ref<SkinReference>(), material);
	}
}

/////

CollisionPolygonSpatialGizmoPlugin::CollisionPolygonSpatialGizmoPlugin() {
	const Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	create_material("shape_material", gizmo_color);
	const float gizmo_value = gizmo_color.get_v();
	const Color gizmo_color_disabled = Color(gizmo_value, gizmo_value, gizmo_value, 0.65);
	create_material("shape_material_disabled", gizmo_color_disabled);
}

bool CollisionPolygonSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<CollisionPolygon>(p_spatial) != nullptr;
}

String CollisionPolygonSpatialGizmoPlugin::get_name() const {
	return "CollisionPolygon";
}

int CollisionPolygonSpatialGizmoPlugin::get_priority() const {
	return -1;
}

void CollisionPolygonSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	CollisionPolygon *polygon = Object::cast_to<CollisionPolygon>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Vector<Vector2> points = polygon->get_polygon();
	float depth = polygon->get_depth() * 0.5;

	Vector<Vector3> lines;
	for (int i = 0; i < points.size(); i++) {
		int n = (i + 1) % points.size();
		lines.push_back(Vector3(points[i].x, points[i].y, depth));
		lines.push_back(Vector3(points[n].x, points[n].y, depth));
		lines.push_back(Vector3(points[i].x, points[i].y, -depth));
		lines.push_back(Vector3(points[n].x, points[n].y, -depth));
		lines.push_back(Vector3(points[i].x, points[i].y, depth));
		lines.push_back(Vector3(points[i].x, points[i].y, -depth));
	}

	const Ref<Material> material =
			get_material(!polygon->is_disabled() ? "shape_material" : "shape_material_disabled", p_gizmo);

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_collision_segments(lines);
}

////

NavigationMeshSpatialGizmoPlugin::NavigationMeshSpatialGizmoPlugin() {
	create_material("navigation_edge_material", EDITOR_DEF("editors/3d_gizmos/gizmo_colors/navigation_edge", Color(0.5, 1, 1)));
	create_material("navigation_edge_material_disabled", EDITOR_DEF("editors/3d_gizmos/gizmo_colors/navigation_edge_disabled", Color(0.7, 0.7, 0.7)));
	create_material("navigation_solid_material", EDITOR_DEF("editors/3d_gizmos/gizmo_colors/navigation_solid", Color(0.5, 1, 1, 0.4)));
	create_material("navigation_solid_material_disabled", EDITOR_DEF("editors/3d_gizmos/gizmo_colors/navigation_solid_disabled", Color(0.7, 0.7, 0.7, 0.4)));
}

bool NavigationMeshSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<NavigationMeshInstance>(p_spatial) != nullptr;
}

String NavigationMeshSpatialGizmoPlugin::get_name() const {
	return "NavigationMeshInstance";
}

int NavigationMeshSpatialGizmoPlugin::get_priority() const {
	return -1;
}

void NavigationMeshSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	NavigationMeshInstance *navmesh = Object::cast_to<NavigationMeshInstance>(p_gizmo->get_spatial_node());

	Ref<Material> edge_material = get_material("navigation_edge_material", p_gizmo);
	Ref<Material> edge_material_disabled = get_material("navigation_edge_material_disabled", p_gizmo);
	Ref<Material> solid_material = get_material("navigation_solid_material", p_gizmo);
	Ref<Material> solid_material_disabled = get_material("navigation_solid_material_disabled", p_gizmo);

	p_gizmo->clear();
	Ref<NavigationMesh> navmeshie = navmesh->get_navigation_mesh();
	if (navmeshie.is_null()) {
		return;
	}

	PoolVector<Vector3> vertices = navmeshie->get_vertices();
	PoolVector<Vector3>::Read vr = vertices.read();
	List<Face3> faces;
	for (int i = 0; i < navmeshie->get_polygon_count(); i++) {
		Vector<int> p = navmeshie->get_polygon(i);

		for (int j = 2; j < p.size(); j++) {
			Face3 f;
			f.vertex[0] = vr[p[0]];
			f.vertex[1] = vr[p[j - 1]];
			f.vertex[2] = vr[p[j]];

			faces.push_back(f);
		}
	}

	if (faces.empty()) {
		return;
	}

	Map<_EdgeKey, bool> edge_map;
	PoolVector<Vector3> tmeshfaces;
	tmeshfaces.resize(faces.size() * 3);

	{
		PoolVector<Vector3>::Write tw = tmeshfaces.write();
		int tidx = 0;

		for (List<Face3>::Element *E = faces.front(); E; E = E->next()) {
			const Face3 &f = E->get();

			for (int j = 0; j < 3; j++) {
				tw[tidx++] = f.vertex[j];
				_EdgeKey ek;
				ek.from = f.vertex[j].snapped(Vector3(CMP_EPSILON, CMP_EPSILON, CMP_EPSILON));
				ek.to = f.vertex[(j + 1) % 3].snapped(Vector3(CMP_EPSILON, CMP_EPSILON, CMP_EPSILON));
				if (ek.from < ek.to) {
					SWAP(ek.from, ek.to);
				}

				Map<_EdgeKey, bool>::Element *F = edge_map.find(ek);

				if (F) {
					F->get() = false;

				} else {
					edge_map[ek] = true;
				}
			}
		}
	}
	Vector<Vector3> lines;

	for (Map<_EdgeKey, bool>::Element *E = edge_map.front(); E; E = E->next()) {
		if (E->get()) {
			lines.push_back(E->key().from);
			lines.push_back(E->key().to);
		}
	}

	Ref<TriangleMesh> tmesh = memnew(TriangleMesh);
	tmesh->create(tmeshfaces);

	if (lines.size()) {
		p_gizmo->add_lines(lines, navmesh->is_enabled() ? edge_material : edge_material_disabled);
	}
	p_gizmo->add_collision_triangles(tmesh);
	Ref<ArrayMesh> m = memnew(ArrayMesh);
	Array a;
	a.resize(Mesh::ARRAY_MAX);
	a[0] = tmeshfaces;
	m->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, a);
	m->surface_set_material(0, navmesh->is_enabled() ? solid_material : solid_material_disabled);
	p_gizmo->add_mesh(m);
	p_gizmo->add_collision_segments(lines);
}

//////

#define BODY_A_RADIUS 0.25
#define BODY_B_RADIUS 0.27

Basis JointGizmosDrawer::look_body(const Transform &p_joint_transform, const Transform &p_body_transform) {
	const Vector3 &p_eye(p_joint_transform.origin);
	const Vector3 &p_target(p_body_transform.origin);

	Vector3 v_x, v_y, v_z;

	// Look the body with X
	v_x = p_target - p_eye;
	v_x.normalize();

	v_z = v_x.cross(Vector3(0, 1, 0));
	v_z.normalize();

	v_y = v_z.cross(v_x);
	v_y.normalize();

	Basis base;
	base.set(v_x, v_y, v_z);

	// Absorb current joint transform
	base = p_joint_transform.basis.inverse() * base;

	return base;
}

Basis JointGizmosDrawer::look_body_toward(Vector3::Axis p_axis, const Transform &joint_transform, const Transform &body_transform) {
	switch (p_axis) {
		case Vector3::AXIS_X:
			return look_body_toward_x(joint_transform, body_transform);
		case Vector3::AXIS_Y:
			return look_body_toward_y(joint_transform, body_transform);
		case Vector3::AXIS_Z:
			return look_body_toward_z(joint_transform, body_transform);
		default:
			return Basis();
	}
}

Basis JointGizmosDrawer::look_body_toward_x(const Transform &p_joint_transform, const Transform &p_body_transform) {
	const Vector3 &p_eye(p_joint_transform.origin);
	const Vector3 &p_target(p_body_transform.origin);

	const Vector3 p_front(p_joint_transform.basis.get_axis(0));

	Vector3 v_x, v_y, v_z;

	// Look the body with X
	v_x = p_target - p_eye;
	v_x.normalize();

	v_y = p_front.cross(v_x);
	v_y.normalize();

	v_z = v_y.cross(p_front);
	v_z.normalize();

	// Clamp X to FRONT axis
	v_x = p_front;
	v_x.normalize();

	Basis base;
	base.set(v_x, v_y, v_z);

	// Absorb current joint transform
	base = p_joint_transform.basis.inverse() * base;

	return base;
}

Basis JointGizmosDrawer::look_body_toward_y(const Transform &p_joint_transform, const Transform &p_body_transform) {
	const Vector3 &p_eye(p_joint_transform.origin);
	const Vector3 &p_target(p_body_transform.origin);

	const Vector3 p_up(p_joint_transform.basis.get_axis(1));

	Vector3 v_x, v_y, v_z;

	// Look the body with X
	v_x = p_target - p_eye;
	v_x.normalize();

	v_z = v_x.cross(p_up);
	v_z.normalize();

	v_x = p_up.cross(v_z);
	v_x.normalize();

	// Clamp Y to UP axis
	v_y = p_up;
	v_y.normalize();

	Basis base;
	base.set(v_x, v_y, v_z);

	// Absorb current joint transform
	base = p_joint_transform.basis.inverse() * base;

	return base;
}

Basis JointGizmosDrawer::look_body_toward_z(const Transform &p_joint_transform, const Transform &p_body_transform) {
	const Vector3 &p_eye(p_joint_transform.origin);
	const Vector3 &p_target(p_body_transform.origin);

	const Vector3 p_lateral(p_joint_transform.basis.get_axis(2));

	Vector3 v_x, v_y, v_z;

	// Look the body with X
	v_x = p_target - p_eye;
	v_x.normalize();

	v_z = p_lateral;
	v_z.normalize();

	v_y = v_z.cross(v_x);
	v_y.normalize();

	// Clamp X to Z axis
	v_x = v_y.cross(v_z);
	v_x.normalize();

	Basis base;
	base.set(v_x, v_y, v_z);

	// Absorb current joint transform
	base = p_joint_transform.basis.inverse() * base;

	return base;
}

void JointGizmosDrawer::draw_circle(Vector3::Axis p_axis, real_t p_radius, const Transform &p_offset, const Basis &p_base, real_t p_limit_lower, real_t p_limit_upper, Vector<Vector3> &r_points, bool p_inverse) {
	if (p_limit_lower == p_limit_upper) {
		r_points.push_back(p_offset.translated(Vector3()).origin);
		r_points.push_back(p_offset.translated(p_base.xform(Vector3(0.5, 0, 0))).origin);

	} else {
		if (p_limit_lower > p_limit_upper) {
			p_limit_lower = -Math_PI;
			p_limit_upper = Math_PI;
		}

		const int points = 32;

		for (int i = 0; i < points; i++) {
			real_t s = p_limit_lower + i * (p_limit_upper - p_limit_lower) / points;
			real_t n = p_limit_lower + (i + 1) * (p_limit_upper - p_limit_lower) / points;

			Vector3 from;
			Vector3 to;
			switch (p_axis) {
				case Vector3::AXIS_X:
					if (p_inverse) {
						from = p_base.xform(Vector3(0, Math::sin(s), Math::cos(s))) * p_radius;
						to = p_base.xform(Vector3(0, Math::sin(n), Math::cos(n))) * p_radius;
					} else {
						from = p_base.xform(Vector3(0, -Math::sin(s), Math::cos(s))) * p_radius;
						to = p_base.xform(Vector3(0, -Math::sin(n), Math::cos(n))) * p_radius;
					}
					break;
				case Vector3::AXIS_Y:
					if (p_inverse) {
						from = p_base.xform(Vector3(Math::cos(s), 0, -Math::sin(s))) * p_radius;
						to = p_base.xform(Vector3(Math::cos(n), 0, -Math::sin(n))) * p_radius;
					} else {
						from = p_base.xform(Vector3(Math::cos(s), 0, Math::sin(s))) * p_radius;
						to = p_base.xform(Vector3(Math::cos(n), 0, Math::sin(n))) * p_radius;
					}
					break;
				case Vector3::AXIS_Z:
					from = p_base.xform(Vector3(Math::cos(s), Math::sin(s), 0)) * p_radius;
					to = p_base.xform(Vector3(Math::cos(n), Math::sin(n), 0)) * p_radius;
					break;
			}

			if (i == points - 1) {
				r_points.push_back(p_offset.translated(to).origin);
				r_points.push_back(p_offset.translated(Vector3()).origin);
			}
			if (i == 0) {
				r_points.push_back(p_offset.translated(from).origin);
				r_points.push_back(p_offset.translated(Vector3()).origin);
			}

			r_points.push_back(p_offset.translated(from).origin);
			r_points.push_back(p_offset.translated(to).origin);
		}

		r_points.push_back(p_offset.translated(Vector3(0, p_radius * 1.5, 0)).origin);
		r_points.push_back(p_offset.translated(Vector3()).origin);
	}
}

void JointGizmosDrawer::draw_cone(const Transform &p_offset, const Basis &p_base, real_t p_swing, real_t p_twist, Vector<Vector3> &r_points) {
	float r = 1.0;
	float w = r * Math::sin(p_swing);
	float d = r * Math::cos(p_swing);

	//swing
	for (int i = 0; i < 360; i += 10) {
		float ra = Math::deg2rad((float)i);
		float rb = Math::deg2rad((float)i + 10);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * w;

		r_points.push_back(p_offset.translated(p_base.xform(Vector3(d, a.x, a.y))).origin);
		r_points.push_back(p_offset.translated(p_base.xform(Vector3(d, b.x, b.y))).origin);

		if (i % 90 == 0) {
			r_points.push_back(p_offset.translated(p_base.xform(Vector3(d, a.x, a.y))).origin);
			r_points.push_back(p_offset.translated(p_base.xform(Vector3())).origin);
		}
	}

	r_points.push_back(p_offset.translated(p_base.xform(Vector3())).origin);
	r_points.push_back(p_offset.translated(p_base.xform(Vector3(1, 0, 0))).origin);

	/// Twist
	float ts = Math::rad2deg(p_twist);
	ts = MIN(ts, 720);

	for (int i = 0; i < int(ts); i += 5) {
		float ra = Math::deg2rad((float)i);
		float rb = Math::deg2rad((float)i + 5);
		float c = i / 720.0;
		float cn = (i + 5) / 720.0;
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w * c;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * w * cn;

		r_points.push_back(p_offset.translated(p_base.xform(Vector3(c, a.x, a.y))).origin);
		r_points.push_back(p_offset.translated(p_base.xform(Vector3(cn, b.x, b.y))).origin);
	}
}

////

JointSpatialGizmoPlugin::JointSpatialGizmoPlugin() {
	create_material("joint_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/joint"));
	create_material("joint_body_a_material", EDITOR_DEF("editors/3d_gizmos/gizmo_colors/joint_body_a", Color(0.6, 0.8, 1)));
	create_material("joint_body_b_material", EDITOR_DEF("editors/3d_gizmos/gizmo_colors/joint_body_b", Color(0.6, 0.9, 1)));

	update_timer = memnew(Timer);
	update_timer->set_name("JointGizmoUpdateTimer");
	update_timer->set_wait_time(1.0 / 120.0);
	update_timer->connect("timeout", this, "incremental_update_gizmos");
	update_timer->set_autostart(true);
	EditorNode::get_singleton()->call_deferred("add_child", update_timer);
}

void JointSpatialGizmoPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("incremental_update_gizmos"), &JointSpatialGizmoPlugin::incremental_update_gizmos);
}

void JointSpatialGizmoPlugin::incremental_update_gizmos() {
	if (!current_gizmos.empty()) {
		update_idx++;
		update_idx = update_idx % current_gizmos.size();
		redraw(current_gizmos[update_idx]);
	}
}

bool JointSpatialGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<Joint>(p_spatial) != nullptr;
}

String JointSpatialGizmoPlugin::get_name() const {
	return "Joints";
}

int JointSpatialGizmoPlugin::get_priority() const {
	return -1;
}

void JointSpatialGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	Joint *joint = Object::cast_to<Joint>(p_gizmo->get_spatial_node());

	p_gizmo->clear();

	Spatial *node_body_a = nullptr;
	if (!joint->get_node_a().is_empty()) {
		node_body_a = Object::cast_to<Spatial>(joint->get_node(joint->get_node_a()));
	}

	Spatial *node_body_b = nullptr;
	if (!joint->get_node_b().is_empty()) {
		node_body_b = Object::cast_to<Spatial>(joint->get_node(joint->get_node_b()));
	}

	if (!node_body_a && !node_body_b) {
		return;
	}

	Ref<Material> common_material = get_material("joint_material", p_gizmo);
	Ref<Material> body_a_material = get_material("joint_body_a_material", p_gizmo);
	Ref<Material> body_b_material = get_material("joint_body_b_material", p_gizmo);

	Vector<Vector3> points;
	Vector<Vector3> body_a_points;
	Vector<Vector3> body_b_points;

	if (Object::cast_to<PinJoint>(joint)) {
		CreatePinJointGizmo(Transform(), points);
		p_gizmo->add_collision_segments(points);
		p_gizmo->add_lines(points, common_material);
	}

	HingeJoint *hinge = Object::cast_to<HingeJoint>(joint);
	if (hinge) {
		CreateHingeJointGizmo(
				Transform(),
				hinge->get_global_transform(),
				node_body_a ? node_body_a->get_global_transform() : Transform(),
				node_body_b ? node_body_b->get_global_transform() : Transform(),
				hinge->get_param(HingeJoint::PARAM_LIMIT_LOWER),
				hinge->get_param(HingeJoint::PARAM_LIMIT_UPPER),
				hinge->get_flag(HingeJoint::FLAG_USE_LIMIT),
				points,
				node_body_a ? &body_a_points : nullptr,
				node_body_b ? &body_b_points : nullptr);

		p_gizmo->add_collision_segments(points);
		p_gizmo->add_collision_segments(body_a_points);
		p_gizmo->add_collision_segments(body_b_points);

		p_gizmo->add_lines(points, common_material);
		p_gizmo->add_lines(body_a_points, body_a_material);
		p_gizmo->add_lines(body_b_points, body_b_material);
	}

	SliderJoint *slider = Object::cast_to<SliderJoint>(joint);
	if (slider) {
		CreateSliderJointGizmo(
				Transform(),
				slider->get_global_transform(),
				node_body_a ? node_body_a->get_global_transform() : Transform(),
				node_body_b ? node_body_b->get_global_transform() : Transform(),
				slider->get_param(SliderJoint::PARAM_ANGULAR_LIMIT_LOWER),
				slider->get_param(SliderJoint::PARAM_ANGULAR_LIMIT_UPPER),
				slider->get_param(SliderJoint::PARAM_LINEAR_LIMIT_LOWER),
				slider->get_param(SliderJoint::PARAM_LINEAR_LIMIT_UPPER),
				points,
				node_body_a ? &body_a_points : nullptr,
				node_body_b ? &body_b_points : nullptr);

		p_gizmo->add_collision_segments(points);
		p_gizmo->add_collision_segments(body_a_points);
		p_gizmo->add_collision_segments(body_b_points);

		p_gizmo->add_lines(points, common_material);
		p_gizmo->add_lines(body_a_points, body_a_material);
		p_gizmo->add_lines(body_b_points, body_b_material);
	}

	ConeTwistJoint *cone = Object::cast_to<ConeTwistJoint>(joint);
	if (cone) {
		CreateConeTwistJointGizmo(
				Transform(),
				cone->get_global_transform(),
				node_body_a ? node_body_a->get_global_transform() : Transform(),
				node_body_b ? node_body_b->get_global_transform() : Transform(),
				cone->get_param(ConeTwistJoint::PARAM_SWING_SPAN),
				cone->get_param(ConeTwistJoint::PARAM_TWIST_SPAN),
				node_body_a ? &body_a_points : nullptr,
				node_body_b ? &body_b_points : nullptr);

		p_gizmo->add_collision_segments(body_a_points);
		p_gizmo->add_collision_segments(body_b_points);

		p_gizmo->add_lines(body_a_points, body_a_material);
		p_gizmo->add_lines(body_b_points, body_b_material);
	}

	Generic6DOFJoint *gen = Object::cast_to<Generic6DOFJoint>(joint);
	if (gen) {
		CreateGeneric6DOFJointGizmo(
				Transform(),
				gen->get_global_transform(),
				node_body_a ? node_body_a->get_global_transform() : Transform(),
				node_body_b ? node_body_b->get_global_transform() : Transform(),

				gen->get_param_x(Generic6DOFJoint::PARAM_ANGULAR_LOWER_LIMIT),
				gen->get_param_x(Generic6DOFJoint::PARAM_ANGULAR_UPPER_LIMIT),
				gen->get_param_x(Generic6DOFJoint::PARAM_LINEAR_LOWER_LIMIT),
				gen->get_param_x(Generic6DOFJoint::PARAM_LINEAR_UPPER_LIMIT),
				gen->get_flag_x(Generic6DOFJoint::FLAG_ENABLE_ANGULAR_LIMIT),
				gen->get_flag_x(Generic6DOFJoint::FLAG_ENABLE_LINEAR_LIMIT),

				gen->get_param_y(Generic6DOFJoint::PARAM_ANGULAR_LOWER_LIMIT),
				gen->get_param_y(Generic6DOFJoint::PARAM_ANGULAR_UPPER_LIMIT),
				gen->get_param_y(Generic6DOFJoint::PARAM_LINEAR_LOWER_LIMIT),
				gen->get_param_y(Generic6DOFJoint::PARAM_LINEAR_UPPER_LIMIT),
				gen->get_flag_y(Generic6DOFJoint::FLAG_ENABLE_ANGULAR_LIMIT),
				gen->get_flag_y(Generic6DOFJoint::FLAG_ENABLE_LINEAR_LIMIT),

				gen->get_param_z(Generic6DOFJoint::PARAM_ANGULAR_LOWER_LIMIT),
				gen->get_param_z(Generic6DOFJoint::PARAM_ANGULAR_UPPER_LIMIT),
				gen->get_param_z(Generic6DOFJoint::PARAM_LINEAR_LOWER_LIMIT),
				gen->get_param_z(Generic6DOFJoint::PARAM_LINEAR_UPPER_LIMIT),
				gen->get_flag_z(Generic6DOFJoint::FLAG_ENABLE_ANGULAR_LIMIT),
				gen->get_flag_z(Generic6DOFJoint::FLAG_ENABLE_LINEAR_LIMIT),

				points,
				node_body_a ? &body_a_points : nullptr,
				node_body_a ? &body_b_points : nullptr);

		p_gizmo->add_collision_segments(points);
		p_gizmo->add_collision_segments(body_a_points);
		p_gizmo->add_collision_segments(body_b_points);

		p_gizmo->add_lines(points, common_material);
		p_gizmo->add_lines(body_a_points, body_a_material);
		p_gizmo->add_lines(body_b_points, body_b_material);
	}
}

void JointSpatialGizmoPlugin::CreatePinJointGizmo(const Transform &p_offset, Vector<Vector3> &r_cursor_points) {
	float cs = 0.25;

	r_cursor_points.push_back(p_offset.translated(Vector3(+cs, 0, 0)).origin);
	r_cursor_points.push_back(p_offset.translated(Vector3(-cs, 0, 0)).origin);
	r_cursor_points.push_back(p_offset.translated(Vector3(0, +cs, 0)).origin);
	r_cursor_points.push_back(p_offset.translated(Vector3(0, -cs, 0)).origin);
	r_cursor_points.push_back(p_offset.translated(Vector3(0, 0, +cs)).origin);
	r_cursor_points.push_back(p_offset.translated(Vector3(0, 0, -cs)).origin);
}

void JointSpatialGizmoPlugin::CreateHingeJointGizmo(const Transform &p_offset, const Transform &p_trs_joint, const Transform &p_trs_body_a, const Transform &p_trs_body_b, real_t p_limit_lower, real_t p_limit_upper, bool p_use_limit, Vector<Vector3> &r_common_points, Vector<Vector3> *r_body_a_points, Vector<Vector3> *r_body_b_points) {
	r_common_points.push_back(p_offset.translated(Vector3(0, 0, 0.5)).origin);
	r_common_points.push_back(p_offset.translated(Vector3(0, 0, -0.5)).origin);

	if (!p_use_limit) {
		p_limit_upper = -1;
		p_limit_lower = 0;
	}

	if (r_body_a_points) {
		JointGizmosDrawer::draw_circle(Vector3::AXIS_Z,
				BODY_A_RADIUS,
				p_offset,
				JointGizmosDrawer::look_body_toward_z(p_trs_joint, p_trs_body_a),
				p_limit_lower,
				p_limit_upper,
				*r_body_a_points);
	}

	if (r_body_b_points) {
		JointGizmosDrawer::draw_circle(Vector3::AXIS_Z,
				BODY_B_RADIUS,
				p_offset,
				JointGizmosDrawer::look_body_toward_z(p_trs_joint, p_trs_body_b),
				p_limit_lower,
				p_limit_upper,
				*r_body_b_points);
	}
}

void JointSpatialGizmoPlugin::CreateSliderJointGizmo(const Transform &p_offset, const Transform &p_trs_joint, const Transform &p_trs_body_a, const Transform &p_trs_body_b, real_t p_angular_limit_lower, real_t p_angular_limit_upper, real_t p_linear_limit_lower, real_t p_linear_limit_upper, Vector<Vector3> &r_points, Vector<Vector3> *r_body_a_points, Vector<Vector3> *r_body_b_points) {
	p_linear_limit_lower = -p_linear_limit_lower;
	p_linear_limit_upper = -p_linear_limit_upper;

	float cs = 0.25;
	r_points.push_back(p_offset.translated(Vector3(0, 0, 0.5)).origin);
	r_points.push_back(p_offset.translated(Vector3(0, 0, -0.5)).origin);

	if (p_linear_limit_lower >= p_linear_limit_upper) {
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_upper, 0, 0)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_lower, 0, 0)).origin);

		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_upper, -cs, -cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_upper, -cs, cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_upper, -cs, cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_upper, cs, cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_upper, cs, cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_upper, cs, -cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_upper, cs, -cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_upper, -cs, -cs)).origin);

		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_lower, -cs, -cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_lower, -cs, cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_lower, -cs, cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_lower, cs, cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_lower, cs, cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_lower, cs, -cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_lower, cs, -cs)).origin);
		r_points.push_back(p_offset.translated(Vector3(p_linear_limit_lower, -cs, -cs)).origin);

	} else {
		r_points.push_back(p_offset.translated(Vector3(+cs * 2, 0, 0)).origin);
		r_points.push_back(p_offset.translated(Vector3(-cs * 2, 0, 0)).origin);
	}

	if (r_body_a_points) {
		JointGizmosDrawer::draw_circle(
				Vector3::AXIS_X,
				BODY_A_RADIUS,
				p_offset,
				JointGizmosDrawer::look_body_toward(Vector3::AXIS_X, p_trs_joint, p_trs_body_a),
				p_angular_limit_lower,
				p_angular_limit_upper,
				*r_body_a_points);
	}

	if (r_body_b_points) {
		JointGizmosDrawer::draw_circle(
				Vector3::AXIS_X,
				BODY_B_RADIUS,
				p_offset,
				JointGizmosDrawer::look_body_toward(Vector3::AXIS_X, p_trs_joint, p_trs_body_b),
				p_angular_limit_lower,
				p_angular_limit_upper,
				*r_body_b_points,
				true);
	}
}

void JointSpatialGizmoPlugin::CreateConeTwistJointGizmo(const Transform &p_offset, const Transform &p_trs_joint, const Transform &p_trs_body_a, const Transform &p_trs_body_b, real_t p_swing, real_t p_twist, Vector<Vector3> *r_body_a_points, Vector<Vector3> *r_body_b_points) {
	if (r_body_a_points) {
		JointGizmosDrawer::draw_cone(
				p_offset,
				JointGizmosDrawer::look_body(p_trs_joint, p_trs_body_a),
				p_swing,
				p_twist,
				*r_body_a_points);
	}

	if (r_body_b_points) {
		JointGizmosDrawer::draw_cone(
				p_offset,
				JointGizmosDrawer::look_body(p_trs_joint, p_trs_body_b),
				p_swing,
				p_twist,
				*r_body_b_points);
	}
}

void JointSpatialGizmoPlugin::CreateGeneric6DOFJointGizmo(
		const Transform &p_offset,
		const Transform &p_trs_joint,
		const Transform &p_trs_body_a,
		const Transform &p_trs_body_b,
		real_t p_angular_limit_lower_x,
		real_t p_angular_limit_upper_x,
		real_t p_linear_limit_lower_x,
		real_t p_linear_limit_upper_x,
		bool p_enable_angular_limit_x,
		bool p_enable_linear_limit_x,
		real_t p_angular_limit_lower_y,
		real_t p_angular_limit_upper_y,
		real_t p_linear_limit_lower_y,
		real_t p_linear_limit_upper_y,
		bool p_enable_angular_limit_y,
		bool p_enable_linear_limit_y,
		real_t p_angular_limit_lower_z,
		real_t p_angular_limit_upper_z,
		real_t p_linear_limit_lower_z,
		real_t p_linear_limit_upper_z,
		bool p_enable_angular_limit_z,
		bool p_enable_linear_limit_z,
		Vector<Vector3> &r_points,
		Vector<Vector3> *r_body_a_points,
		Vector<Vector3> *r_body_b_points) {
	float cs = 0.25;

	for (int ax = 0; ax < 3; ax++) {
		float ll = 0;
		float ul = 0;
		float lll = 0;
		float lul = 0;

		int a1 = 0;
		int a2 = 0;
		int a3 = 0;
		bool enable_ang = false;
		bool enable_lin = false;

		switch (ax) {
			case 0:
				ll = p_angular_limit_lower_x;
				ul = p_angular_limit_upper_x;
				lll = -p_linear_limit_lower_x;
				lul = -p_linear_limit_upper_x;
				enable_ang = p_enable_angular_limit_x;
				enable_lin = p_enable_linear_limit_x;
				a1 = 0;
				a2 = 1;
				a3 = 2;
				break;
			case 1:
				ll = p_angular_limit_lower_y;
				ul = p_angular_limit_upper_y;
				lll = -p_linear_limit_lower_y;
				lul = -p_linear_limit_upper_y;
				enable_ang = p_enable_angular_limit_y;
				enable_lin = p_enable_linear_limit_y;
				a1 = 1;
				a2 = 2;
				a3 = 0;
				break;
			case 2:
				ll = p_angular_limit_lower_z;
				ul = p_angular_limit_upper_z;
				lll = -p_linear_limit_lower_z;
				lul = -p_linear_limit_upper_z;
				enable_ang = p_enable_angular_limit_z;
				enable_lin = p_enable_linear_limit_z;
				a1 = 2;
				a2 = 0;
				a3 = 1;
				break;
		}

#define ADD_VTX(x, y, z)                                   \
	{                                                      \
		Vector3 v;                                         \
		v[a1] = (x);                                       \
		v[a2] = (y);                                       \
		v[a3] = (z);                                       \
		r_points.push_back(p_offset.translated(v).origin); \
	}

		if (enable_lin && lll >= lul) {
			ADD_VTX(lul, 0, 0);
			ADD_VTX(lll, 0, 0);

			ADD_VTX(lul, -cs, -cs);
			ADD_VTX(lul, -cs, cs);
			ADD_VTX(lul, -cs, cs);
			ADD_VTX(lul, cs, cs);
			ADD_VTX(lul, cs, cs);
			ADD_VTX(lul, cs, -cs);
			ADD_VTX(lul, cs, -cs);
			ADD_VTX(lul, -cs, -cs);

			ADD_VTX(lll, -cs, -cs);
			ADD_VTX(lll, -cs, cs);
			ADD_VTX(lll, -cs, cs);
			ADD_VTX(lll, cs, cs);
			ADD_VTX(lll, cs, cs);
			ADD_VTX(lll, cs, -cs);
			ADD_VTX(lll, cs, -cs);
			ADD_VTX(lll, -cs, -cs);

		} else {
			ADD_VTX(+cs * 2, 0, 0);
			ADD_VTX(-cs * 2, 0, 0);
		}

		if (!enable_ang) {
			ll = 0;
			ul = -1;
		}

		if (r_body_a_points) {
			JointGizmosDrawer::draw_circle(
					static_cast<Vector3::Axis>(ax),
					BODY_A_RADIUS,
					p_offset,
					JointGizmosDrawer::look_body_toward(static_cast<Vector3::Axis>(ax), p_trs_joint, p_trs_body_a),
					ll,
					ul,
					*r_body_a_points,
					true);
		}

		if (r_body_b_points) {
			JointGizmosDrawer::draw_circle(
					static_cast<Vector3::Axis>(ax),
					BODY_B_RADIUS,
					p_offset,
					JointGizmosDrawer::look_body_toward(static_cast<Vector3::Axis>(ax), p_trs_joint, p_trs_body_b),
					ll,
					ul,
					*r_body_b_points);
		}
	}

#undef ADD_VTX
}

////

RoomGizmoPlugin::RoomGizmoPlugin() {
	Color color_room = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/room_edge", Color(0.5, 1.0, 0.0));
	Color color_overlap = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/room_overlap", Color(1.0, 0.0, 0.0));

	create_material("room", color_room, false, true, false);
	create_material("room_overlap", color_overlap, false, false, false);

	create_handle_material("room_handle");
}

Ref<EditorSpatialGizmo> RoomGizmoPlugin::create_gizmo(Spatial *p_spatial) {
	Ref<RoomSpatialGizmo> ref;

	Room *room = Object::cast_to<Room>(p_spatial);
	if (room) {
		ref = Ref<RoomSpatialGizmo>(memnew(RoomSpatialGizmo(room)));
	}

	return ref;
}

bool RoomGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	if (Object::cast_to<Room>(p_spatial)) {
		return true;
	}

	return false;
}

String RoomGizmoPlugin::get_name() const {
	return "Room";
}

int RoomGizmoPlugin::get_priority() const {
	return -1;
}

//////////////////////

String RoomSpatialGizmo::get_handle_name(int p_idx) const {
	return "Point " + itos(p_idx);
}

Variant RoomSpatialGizmo::get_handle_value(int p_idx) {
	if (!_room) {
		return Vector3(0, 0, 0);
	}

	int num_points = _room->_bound_pts.size();
	if (p_idx >= num_points) {
		return Vector3(0, 0, 0);
	}

	return _room->_bound_pts[p_idx];
}

void RoomSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {
	if (!_room || (p_idx >= _room->_bound_pts.size())) {
		return;
	}

	Transform tr = _room->get_global_transform();
	Transform tr_inv = tr.affine_inverse();

	Vector3 pt_world = _room->_bound_pts[p_idx];
	pt_world = tr.xform(pt_world);

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 camera_dir = p_camera->get_transform().basis.get_axis(2);

	// find the smallest camera axis, we will only transform the handles on 2 axes max,
	// to try and make things more user friendly (it is confusing trying to change 3d position
	// from a 2d view)
	int biggest_axis = 0;
	real_t biggest = 0.0;
	for (int n = 0; n < 3; n++) {
		real_t val = Math::abs(camera_dir.get_axis(n));
		if (val > biggest) {
			biggest = val;
			biggest_axis = n;
		}
	}

	{
		Plane plane(pt_world, camera_dir);
		Vector3 inters;

		if (plane.intersects_ray(ray_from, ray_dir, &inters)) {
			if (SpatialEditor::get_singleton()->is_snap_enabled()) {
				float snap = SpatialEditor::get_singleton()->get_translate_snap();
				inters.snap(Vector3(snap, snap, snap));
			}

			for (int n = 0; n < 3; n++) {
				if (n != biggest_axis) {
					pt_world.set_axis(n, inters.get_axis(n));
				}
			}

			Vector3 pt_local = tr_inv.xform(pt_world);
			_room->set_point(p_idx, pt_local);
		}

		return;
	}
}

void RoomSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {
	if (!_room || (p_idx >= _room->_bound_pts.size())) {
		return;
	}

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();

	ur->create_action(TTR("Set Room Point Position"));
	ur->add_do_method(_room, "set_point", p_idx, _room->_bound_pts[p_idx]);
	ur->add_undo_method(_room, "set_point", p_idx, p_restore);
	ur->commit_action();

	_room->property_list_changed_notify();
}

void RoomSpatialGizmo::redraw() {
	clear();

	if (!_room) {
		return;
	}

	const Geometry::MeshData &md = _room->_bound_mesh_data;
	if (!md.edges.size())
		return;

	Vector<Vector3> lines;
	Transform tr = _room->get_global_transform();
	Transform tr_inv = tr.affine_inverse();

	Ref<Material> material = gizmo_plugin->get_material("room", this);
	Ref<Material> material_overlap = gizmo_plugin->get_material("room_overlap", this);
	Color color(1, 1, 1, 1);

	for (int n = 0; n < md.edges.size(); n++) {
		Vector3 a = md.vertices[md.edges[n].a];
		Vector3 b = md.vertices[md.edges[n].b];

		// xform
		a = tr_inv.xform(a);
		b = tr_inv.xform(b);

		lines.push_back(a);
		lines.push_back(b);
	}

	if (lines.size()) {
		add_lines(lines, material, false, color);
	}

	// overlap zones
	for (int z = 0; z < _room->_gizmo_overlap_zones.size(); z++) {
		const Geometry::MeshData &md_overlap = _room->_gizmo_overlap_zones[z];
		Vector<Vector3> pts;

		for (int f = 0; f < md_overlap.faces.size(); f++) {
			const Geometry::MeshData::Face &face = md_overlap.faces[f];

			for (int c = 0; c < face.indices.size() - 2; c++) {
				pts.push_back(tr_inv.xform(md_overlap.vertices[face.indices[0]]));
				pts.push_back(tr_inv.xform(md_overlap.vertices[face.indices[c + 1]]));
				pts.push_back(tr_inv.xform(md_overlap.vertices[face.indices[c + 2]]));
			}
		}

		Ref<ArrayMesh> mesh = memnew(ArrayMesh);
		Array array;
		array.resize(Mesh::ARRAY_MAX);
		array[Mesh::ARRAY_VERTEX] = pts;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, array);
		add_mesh(mesh, false, Ref<SkinReference>(), material_overlap);
	}

	Vector<Vector3> handles;
	// draw the handles separately because these must correspond to the raw points
	// for editing
	for (int n = 0; n < _room->_bound_pts.size(); n++) {
		handles.push_back(_room->_bound_pts[n]);
	}

	// handles
	if (handles.size()) {
		Ref<Material> material_handle = gizmo_plugin->get_material("room_handle", this);
		add_handles(handles, material_handle);
	}
}

RoomSpatialGizmo::RoomSpatialGizmo(Room *p_room) {
	_room = p_room;
	set_spatial_node(p_room);
}

////

PortalGizmoPlugin::PortalGizmoPlugin() {
	Color color_portal_margin = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/portal_margin", Color(1.0, 0.1, 0.1, 0.3));
	Color color_portal_edge = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/portal_edge", Color(0.0, 0.0, 0.0, 0.3));
	Color color_portal_arrow = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/portal_arrow", Color(1.0, 1.0, 1.0, 1.0));

	create_icon_material("portal_icon", SpatialEditor::get_singleton()->get_icon("GizmoPortal", "EditorIcons"), true);
	create_material("portal", Color(1.0, 1.0, 1.0, 1.0), false, false, true);
	create_material("portal_margin", color_portal_margin, false, false, false);
	create_material("portal_edge", color_portal_edge, false, false, false);
	create_material("portal_arrow", color_portal_arrow, false, false, false);

	create_handle_material("portal_handle");
}

Ref<EditorSpatialGizmo> PortalGizmoPlugin::create_gizmo(Spatial *p_spatial) {
	Ref<PortalSpatialGizmo> ref;

	Portal *portal = Object::cast_to<Portal>(p_spatial);
	if (portal) {
		ref = Ref<PortalSpatialGizmo>(memnew(PortalSpatialGizmo(portal)));
	}

	return ref;
}

bool PortalGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	if (Object::cast_to<Portal>(p_spatial)) {
		return true;
	}

	return false;
}

String PortalGizmoPlugin::get_name() const {
	return "Portal";
}

int PortalGizmoPlugin::get_priority() const {
	return -1;
}

//////////////////////

String PortalSpatialGizmo::get_handle_name(int p_idx) const {
	return "Point " + itos(p_idx);
}

Variant PortalSpatialGizmo::get_handle_value(int p_idx) {
	if (!_portal) {
		return Vector2(0, 0);
	}

	int num_points = _portal->_pts_local_raw.size();
	if (p_idx >= num_points) {
		return Vector2(0, 0);
	}

	return _portal->_pts_local_raw[p_idx];
}

void PortalSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {
	if (!_portal || (p_idx >= _portal->_pts_local_raw.size())) {
		return;
	}

	Transform tr = _portal->get_global_transform();
	Transform tr_inv = tr.affine_inverse();

	Vector3 pt_local = Portal::_vec2to3(_portal->_pts_local_raw[p_idx]);
	Vector3 pt_world = tr.xform(pt_local);

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	// get a normal from the global transform
	Plane plane(Vector3(0, 0, 0), Vector3(0, 0, 1));
	plane = tr.xform(plane);

	// construct the plane that the 2d portal is defined in
	plane = Plane(pt_world, plane.normal);

	Vector3 inters;

	if (plane.intersects_ray(ray_from, ray_dir, &inters)) {
		// back calculate from the 3d intersection to the 2d portal plane
		inters = tr_inv.xform(inters);

		// snapping will be in 2d for portals, and the scale may make less sense,
		// but better to offer at least some functionality
		if (SpatialEditor::get_singleton()->is_snap_enabled()) {
			float snap = SpatialEditor::get_singleton()->get_translate_snap();
			inters.snap(Vector3(snap, snap, snap));
		}

		_portal->set_point(p_idx, Vector2(inters.x, inters.y));

		return;
	}
}

void PortalSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {
	if (!_portal || (p_idx >= _portal->_pts_local_raw.size())) {
		return;
	}

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();

	ur->create_action(TTR("Set Portal Point Position"));
	ur->add_do_method(_portal, "set_point", p_idx, _portal->_pts_local_raw[p_idx]);
	ur->add_undo_method(_portal, "set_point", p_idx, p_restore);
	ur->commit_action();

	_portal->property_list_changed_notify();
}

void PortalSpatialGizmo::redraw() {
	clear();

	if (!_portal) {
		return;
	}

	// warnings
	if (_portal->_warning_outside_room_aabb || _portal->_warning_facing_wrong_way || _portal->_warning_autolink_failed) {
		Ref<Material> icon = gizmo_plugin->get_material("portal_icon", this);
		add_unscaled_billboard(icon, 0.05);
	}

	Transform tr = _portal->get_global_transform();
	Transform tr_inv = tr.affine_inverse();

	Ref<Material> material_portal = gizmo_plugin->get_material("portal", this);
	Ref<Material> material_margin = gizmo_plugin->get_material("portal_margin", this);
	Ref<Material> material_edge = gizmo_plugin->get_material("portal_edge", this);
	Ref<Material> material_arrow = gizmo_plugin->get_material("portal_arrow", this);
	Color color(1, 1, 1, 1);

	// make sure world points are up to date
	_portal->portal_update();

	int num_points = _portal->_pts_world.size();

	// prevent compiler warnings later on
	if (num_points < 3) {
		return;
	}

	// margins
	real_t margin = _portal->get_active_portal_margin();
	bool show_margins = Portal::_settings_gizmo_show_margins;

	if (margin < 0.05f) {
		show_margins = false;
	}

	PoolVector<Vector3> pts_portal;
	PoolVector<Color> cols_portal;
	PoolVector<Vector3> pts_margin;
	Vector<Vector3> edge_pts;

	Vector<Vector3> handles;

	Vector3 portal_normal_world_space = _portal->_plane.normal;
	portal_normal_world_space *= margin;

	// this may not be necessary, dealing with non uniform scales,
	// possible the affine_invert dealt with this earlier .. but it's just for
	// the editor so not performance critical
	Basis normal_basis = tr_inv.basis;

	Vector3 portal_normal = normal_basis.xform(portal_normal_world_space);
	Vector3 pt_portal_first = tr_inv.xform(_portal->_pts_world[0]);

	for (int n = 0; n < num_points; n++) {
		Vector3 pt = _portal->_pts_world[n];
		pt = tr_inv.xform(pt);

		// CI for visual studio can't seem to get around the possibility
		// that this could cause a divide by zero, so using a local to preclude the
		// possibility of aliasing from another thread
		int m = (n + 1) % num_points;
		Vector3 pt_next = _portal->_pts_world[m];
		pt_next = tr_inv.xform(pt_next);

		// don't need the first and last triangles
		if ((n != 0) && (n != (num_points - 1))) {
			pts_portal.push_back(pt_portal_first);
			pts_portal.push_back(pt);
			pts_portal.push_back(pt_next);
			cols_portal.push_back(_color_portal_front);
			cols_portal.push_back(_color_portal_front);
			cols_portal.push_back(_color_portal_front);

			pts_portal.push_back(pt_next);
			pts_portal.push_back(pt);
			pts_portal.push_back(pt_portal_first);
			cols_portal.push_back(_color_portal_back);
			cols_portal.push_back(_color_portal_back);
			cols_portal.push_back(_color_portal_back);
		}

		if (show_margins) {
			Vector3 pt0 = pt - portal_normal;
			Vector3 pt1 = pt + portal_normal;
			Vector3 pt2 = pt_next - portal_normal;
			Vector3 pt3 = pt_next + portal_normal;

			pts_margin.push_back(pt0);
			pts_margin.push_back(pt2);
			pts_margin.push_back(pt1);

			pts_margin.push_back(pt2);
			pts_margin.push_back(pt3);
			pts_margin.push_back(pt1);

			edge_pts.push_back(pt0);
			edge_pts.push_back(pt2);
			edge_pts.push_back(pt1);
			edge_pts.push_back(pt3);
		}
	}

	// draw the handles separately because these must correspond to the raw points
	// for editing
	for (int n = 0; n < _portal->_pts_local_raw.size(); n++) {
		Vector3 pt = Portal::_vec2to3(_portal->_pts_local_raw[n]);
		handles.push_back(pt);
	}

	// portal itself
	{
		Ref<ArrayMesh> mesh = memnew(ArrayMesh);
		Array array;
		array.resize(Mesh::ARRAY_MAX);
		array[Mesh::ARRAY_VERTEX] = pts_portal;
		array[Mesh::ARRAY_COLOR] = cols_portal;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, array);
		add_mesh(mesh, false, Ref<SkinReference>(), material_portal);

		// handles
		Ref<Material> material_handle = gizmo_plugin->get_material("portal_handle", this);
		add_handles(handles, material_handle);
	}

	if (show_margins) {
		Ref<ArrayMesh> mesh = memnew(ArrayMesh);
		Array array;
		array.resize(Mesh::ARRAY_MAX);
		array[Mesh::ARRAY_VERTEX] = pts_margin;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, array);
		add_mesh(mesh, false, Ref<SkinReference>(), material_margin);

		// lines around the outside of mesh
		add_lines(edge_pts, material_edge, false, color);
	} // only if the margin is sufficient to be worth drawing

	// arrow
	if (show_margins) {
		const int arrow_points = 7;
		const float arrow_length = 0.5; // 1.5
		const float arrow_width = 0.1; // 0.3
		const float arrow_barb = 0.27; // 0.8

		Vector3 arrow[arrow_points] = {
			Vector3(0, 0, -1),
			Vector3(0, arrow_barb, 0),
			Vector3(0, arrow_width, 0),
			Vector3(0, arrow_width, arrow_length),
			Vector3(0, -arrow_width, arrow_length),
			Vector3(0, -arrow_width, 0),
			Vector3(0, -arrow_barb, 0)
		};

		int arrow_sides = 2;

		Vector<Vector3> lines;

		for (int i = 0; i < arrow_sides; i++) {
			for (int j = 0; j < arrow_points; j++) {
				Basis ma(Vector3(0, 0, 1), Math_PI * i / arrow_sides);

				Vector3 v1 = arrow[j] - Vector3(0, 0, arrow_length);
				Vector3 v2 = arrow[(j + 1) % arrow_points] - Vector3(0, 0, arrow_length);

				lines.push_back(ma.xform(v1));
				lines.push_back(ma.xform(v2));
			}
		}

		add_lines(lines, material_arrow, false, color);
	}
}

PortalSpatialGizmo::PortalSpatialGizmo(Portal *p_portal) {
	_portal = p_portal;
	set_spatial_node(p_portal);

	_color_portal_front = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/portal_front", Color(0.05, 0.05, 1.0, 0.3));
	_color_portal_back = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/portal_back", Color(1.0, 1.0, 0.0, 0.15));
}

/////////////////////

OccluderGizmoPlugin::OccluderGizmoPlugin() {
	Color color_occluder = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/occluder", Color(1.0, 0.0, 1.0));
	create_material("occluder", color_occluder, false, true, false);

	create_material("occluder_poly", Color(1, 1, 1, 1), false, false, true);

	create_handle_material("occluder_handle");
	create_handle_material("extra_handle", false, SpatialEditor::get_singleton()->get_icon("EditorInternalHandle", "EditorIcons"));
}

Ref<EditorSpatialGizmo> OccluderGizmoPlugin::create_gizmo(Spatial *p_spatial) {
	Ref<OccluderSpatialGizmo> ref;

	Occluder *occluder = Object::cast_to<Occluder>(p_spatial);
	if (occluder) {
		ref = Ref<OccluderSpatialGizmo>(memnew(OccluderSpatialGizmo(occluder)));
	}

	return ref;
}

bool OccluderGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	if (Object::cast_to<Occluder>(p_spatial)) {
		return true;
	}

	return false;
}

String OccluderGizmoPlugin::get_name() const {
	return "Occluder";
}

int OccluderGizmoPlugin::get_priority() const {
	return -1;
}

//////////////////////

String OccluderSpatialGizmo::get_handle_name(int p_idx) const {
	const OccluderShapeSphere *occ_sphere = get_occluder_shape_sphere();
	if (occ_sphere) {
		int num_spheres = occ_sphere->get_spheres().size();

		if (p_idx >= num_spheres) {
			p_idx -= num_spheres;
			return "Radius " + itos(p_idx);
		} else {
			return "Sphere " + itos(p_idx);
		}
	}

	const OccluderShapePolygon *occ_poly = get_occluder_shape_poly();
	if (occ_poly) {
		if (p_idx < occ_poly->_poly_pts_local_raw.size()) {
			return "Poly Point " + itos(p_idx);
		} else {
			return "Hole Point " + itos(p_idx - occ_poly->_poly_pts_local_raw.size());
		}
	}

	return "Unknown";
}

Variant OccluderSpatialGizmo::get_handle_value(int p_idx) {
	const OccluderShapeSphere *occ_sphere = get_occluder_shape_sphere();
	if (occ_sphere) {
		Vector<Plane> spheres = occ_sphere->get_spheres();
		int num_spheres = spheres.size();

		if (p_idx >= num_spheres) {
			p_idx -= num_spheres;
			return spheres[p_idx].d;
		} else {
			return spheres[p_idx].normal;
		}
	}

	const OccluderShapePolygon *occ_poly = get_occluder_shape_poly();
	if (occ_poly) {
		if (p_idx < occ_poly->_poly_pts_local_raw.size()) {
			return occ_poly->_poly_pts_local_raw[p_idx];
		} else {
			p_idx -= occ_poly->_poly_pts_local_raw.size();
			if (p_idx < occ_poly->_hole_pts_local_raw.size()) {
				return occ_poly->_hole_pts_local_raw[p_idx];
			}
			return Vector2(0, 0);
		}
	}

	return 0;
}

void OccluderSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {
	if (!_occluder) {
		return;
	}

	Transform tr = _occluder->get_global_transform();
	Transform tr_inv = tr.affine_inverse();

	// selection ray
	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);
	Vector3 camera_dir = p_camera->get_transform().basis.get_axis(2);

	// find the smallest camera axis, we will only transform the handles on 2 axes max,
	// to try and make things more user friendly (it is confusing trying to change 3d position
	// from a 2d view)
	int biggest_axis = 0;
	real_t biggest = 0.0;
	for (int n = 0; n < 3; n++) {
		real_t val = Math::abs(camera_dir.get_axis(n));
		if (val > biggest) {
			biggest = val;
			biggest_axis = n;
		}
	}

	// find world space of selected point
	OccluderShapeSphere *occ_sphere = get_occluder_shape_sphere();
	if (occ_sphere) {
		Vector<Plane> spheres = occ_sphere->get_spheres();
		int num_spheres = spheres.size();

		// radius?
		bool is_radius = false;

		if (p_idx >= num_spheres) {
			p_idx -= num_spheres;
			is_radius = true;
		}

		Vector3 pt_world = spheres[p_idx].normal;
		pt_world = tr.xform(pt_world);
		Vector3 pt_world_center = pt_world;

		// a plane between the radius point and the centre
		Plane plane;
		if (is_radius) {
			plane = Plane(Vector3(0, 0, 1), pt_world.z);
		} else {
			plane = Plane(pt_world, camera_dir);
		}

		Vector3 inters;
		if (plane.intersects_ray(ray_from, ray_dir, &inters)) {
			if (SpatialEditor::get_singleton()->is_snap_enabled()) {
				float snap = SpatialEditor::get_singleton()->get_translate_snap();
				inters.snap(Vector3(snap, snap, snap));
			}

			if (is_radius) {
				pt_world = inters;

				// new radius is simply the dist between this point and the centre of the sphere
				real_t radius = (pt_world - pt_world_center).length();
				occ_sphere->set_sphere_radius(p_idx, radius);
			} else {
				for (int n = 0; n < 3; n++) {
					if (n != biggest_axis) {
						pt_world.set_axis(n, inters.get_axis(n));
					}
				}

				Vector3 pt_local = tr_inv.xform(pt_world);
				occ_sphere->set_sphere_position(p_idx, pt_local);
			}

			return;
		}
	}

	OccluderShapePolygon *occ_poly = get_occluder_shape_poly();
	if (occ_poly) {
		Vector3 pt_local;

		bool hole = p_idx >= occ_poly->_poly_pts_local_raw.size();
		if (hole) {
			p_idx -= occ_poly->_poly_pts_local_raw.size();
			if (p_idx >= occ_poly->_hole_pts_local_raw.size()) {
				return;
			}
			pt_local = OccluderShapePolygon::_vec2to3(occ_poly->_hole_pts_local_raw[p_idx]);
		} else {
			pt_local = OccluderShapePolygon::_vec2to3(occ_poly->_poly_pts_local_raw[p_idx]);
		}

		Vector3 pt_world = tr.xform(pt_local);

		// get a normal from the global transform
		Plane plane(Vector3(0, 0, 0), Vector3(0, 0, 1));
		plane = tr.xform(plane);

		// construct the plane that the 2d portal is defined in
		plane = Plane(pt_world, plane.normal);

		Vector3 inters;

		if (plane.intersects_ray(ray_from, ray_dir, &inters)) {
			// back calculate from the 3d intersection to the 2d portal plane
			inters = tr_inv.xform(inters);

			// snapping will be in 2d for portals, and the scale may make less sense,
			// but better to offer at least some functionality
			if (SpatialEditor::get_singleton()->is_snap_enabled()) {
				float snap = SpatialEditor::get_singleton()->get_translate_snap();
				inters.snap(Vector3(snap, snap, snap));
			}

			if (hole) {
				occ_poly->set_hole_point(p_idx, Vector2(inters.x, inters.y));
			} else {
				occ_poly->set_polygon_point(p_idx, Vector2(inters.x, inters.y));
			}

			return;
		}
	}
}

void OccluderSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {
	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();

	OccluderShapeSphere *occ_sphere = get_occluder_shape_sphere();
	if (occ_sphere) {
		Vector<Plane> spheres = occ_sphere->get_spheres();
		int num_spheres = spheres.size();

		if (p_idx >= num_spheres) {
			p_idx -= num_spheres;

			ur->create_action(TTR("Set Occluder Sphere Radius"));
			ur->add_do_method(occ_sphere, "set_sphere_radius", p_idx, spheres[p_idx].d);
			ur->add_undo_method(occ_sphere, "set_sphere_radius", p_idx, p_restore);
		} else {
			ur->create_action(TTR("Set Occluder Sphere Position"));
			ur->add_do_method(occ_sphere, "set_sphere_position", p_idx, spheres[p_idx].normal);
			ur->add_undo_method(occ_sphere, "set_sphere_position", p_idx, p_restore);
		}

		ur->commit_action();
		_occluder->property_list_changed_notify();
	}

	OccluderShapePolygon *occ_poly = get_occluder_shape_poly();
	if (occ_poly) {
		if (p_idx < occ_poly->_poly_pts_local_raw.size()) {
			ur->create_action(TTR("Set Occluder Polygon Point Position"));
			ur->add_do_method(occ_poly, "set_polygon_point", p_idx, occ_poly->_poly_pts_local_raw[p_idx]);
			ur->add_undo_method(occ_poly, "set_polygon_point", p_idx, p_restore);
			ur->commit_action();
			_occluder->property_list_changed_notify();
		} else {
			p_idx -= occ_poly->_poly_pts_local_raw.size();
			if (p_idx < occ_poly->_hole_pts_local_raw.size()) {
				ur->create_action(TTR("Set Occluder Hole Point Position"));
				ur->add_do_method(occ_poly, "set_hole_point", p_idx, occ_poly->_hole_pts_local_raw[p_idx]);
				ur->add_undo_method(occ_poly, "set_hole_point", p_idx, p_restore);
				ur->commit_action();
				_occluder->property_list_changed_notify();
			}
		}
	}
}

OccluderShapeSphere *OccluderSpatialGizmo::get_occluder_shape_sphere() {
	OccluderShapeSphere *occ_sphere = Object::cast_to<OccluderShapeSphere>(get_occluder_shape());
	return occ_sphere;
}

const OccluderShapePolygon *OccluderSpatialGizmo::get_occluder_shape_poly() const {
	const OccluderShapePolygon *occ_poly = Object::cast_to<OccluderShapePolygon>(get_occluder_shape());
	return occ_poly;
}

OccluderShapePolygon *OccluderSpatialGizmo::get_occluder_shape_poly() {
	OccluderShapePolygon *occ_poly = Object::cast_to<OccluderShapePolygon>(get_occluder_shape());
	return occ_poly;
}

const OccluderShapeSphere *OccluderSpatialGizmo::get_occluder_shape_sphere() const {
	const OccluderShapeSphere *occ_sphere = Object::cast_to<OccluderShapeSphere>(get_occluder_shape());
	return occ_sphere;
}

const OccluderShape *OccluderSpatialGizmo::get_occluder_shape() const {
	if (!_occluder) {
		return nullptr;
	}

	Ref<OccluderShape> rshape = _occluder->get_shape();
	if (rshape.is_null() || !rshape.is_valid()) {
		return nullptr;
	}

	return rshape.ptr();
}

OccluderShape *OccluderSpatialGizmo::get_occluder_shape() {
	if (!_occluder) {
		return nullptr;
	}

	Ref<OccluderShape> rshape = _occluder->get_shape();
	if (rshape.is_null() || !rshape.is_valid()) {
		return nullptr;
	}

	return rshape.ptr();
}

void OccluderSpatialGizmo::redraw() {
	clear();

	if (!_occluder) {
		return;
	}

	Ref<Material> material_occluder = gizmo_plugin->get_material("occluder", this);
	Color color(1, 1, 1, 1);

	const OccluderShapeSphere *occ_sphere = get_occluder_shape_sphere();
	if (occ_sphere) {
		Vector<Plane> spheres = occ_sphere->get_spheres();
		if (!spheres.size()) {
			return;
		}

		Vector<Vector3> points;
		Vector<Vector3> handles;
		Vector<Vector3> radius_handles;

		for (int n = 0; n < spheres.size(); n++) {
			const Plane &p = spheres[n];

			real_t r = p.d;
			Vector3 offset = p.normal;
			handles.push_back(offset);

			// add a handle for the radius
			radius_handles.push_back(offset + Vector3(r, 0, 0));

			const int deg_change = 4;

			for (int i = 0; i <= 360; i += deg_change) {
				real_t ra = Math::deg2rad((real_t)i);
				real_t rb = Math::deg2rad((real_t)i + deg_change);
				Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
				Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

				points.push_back(offset + Vector3(a.x, 0, a.y));
				points.push_back(offset + Vector3(b.x, 0, b.y));
				points.push_back(offset + Vector3(0, a.x, a.y));
				points.push_back(offset + Vector3(0, b.x, b.y));
				points.push_back(offset + Vector3(a.x, a.y, 0));
				points.push_back(offset + Vector3(b.x, b.y, 0));
			}
		} // for n through spheres

		add_lines(points, material_occluder, false, color);

		// handles
		Ref<Material> material_handle = gizmo_plugin->get_material("occluder_handle", this);
		Ref<Material> material_extra_handle = gizmo_plugin->get_material("extra_handle", this);
		add_handles(handles, material_handle);
		add_handles(radius_handles, material_extra_handle, false, true);
	}

	const OccluderShapePolygon *occ_poly = get_occluder_shape_poly();
	if (occ_poly) {
		// main poly
		if (occ_poly->_poly_pts_local_raw.size()) {
			_redraw_poly(false, occ_poly->_poly_pts_local, occ_poly->_poly_pts_local_raw);
		}

		// hole
		if (occ_poly->_hole_pts_local_raw.size()) {
			_redraw_poly(true, occ_poly->_hole_pts_local, occ_poly->_hole_pts_local_raw);
		}
	}
}

void OccluderSpatialGizmo::_redraw_poly(bool p_hole, const Vector<Vector2> &p_pts, const PoolVector<Vector2> &p_pts_raw) {
	PoolVector<Vector3> pts_edge;
	PoolVector<Color> cols;

	Color col_front = _color_poly_front;
	Color col_back = _color_poly_back;

	if (p_hole) {
		col_front = _color_hole;
		col_back = _color_hole;
	}

	if (p_pts.size() > 2) {
		Vector3 pt_first = OccluderShapePolygon::_vec2to3(p_pts[0]);
		Vector3 pt_prev = OccluderShapePolygon::_vec2to3(p_pts[p_pts.size() - 1]);
		for (int n = 0; n < p_pts.size(); n++) {
			Vector3 pt_curr = OccluderShapePolygon::_vec2to3(p_pts[n]);
			pts_edge.push_back(pt_first);
			pts_edge.push_back(pt_prev);
			pts_edge.push_back(pt_curr);
			cols.push_back(col_front);
			cols.push_back(col_front);
			cols.push_back(col_front);

			pts_edge.push_back(pt_first);
			pts_edge.push_back(pt_curr);
			pts_edge.push_back(pt_prev);
			cols.push_back(col_back);
			cols.push_back(col_back);
			cols.push_back(col_back);

			pt_prev = pt_curr;
		}
	}

	// draw the handles separately because these must correspond to the raw points
	// for editing
	Vector<Vector3> handles;
	for (int n = 0; n < p_pts_raw.size(); n++) {
		Vector3 pt = OccluderShapePolygon::_vec2to3(p_pts_raw[n]);
		handles.push_back(pt);
	}

	// poly itself
	{
		if (pts_edge.size() > 2) {
			Ref<ArrayMesh> mesh = memnew(ArrayMesh);
			Array array;
			array.resize(Mesh::ARRAY_MAX);
			array[Mesh::ARRAY_VERTEX] = pts_edge;
			array[Mesh::ARRAY_COLOR] = cols;
			mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, array);

			Ref<Material> material_poly = gizmo_plugin->get_material("occluder_poly", this);
			add_mesh(mesh, false, Ref<SkinReference>(), material_poly);
		}

		// handles
		if (!p_hole) {
			Ref<Material> material_handle = gizmo_plugin->get_material("occluder_handle", this);
			add_handles(handles, material_handle);
		} else {
			Ref<Material> material_extra_handle = gizmo_plugin->get_material("extra_handle", this);
			add_handles(handles, material_extra_handle, false, true);
		}
	}
}

OccluderSpatialGizmo::OccluderSpatialGizmo(Occluder *p_occluder) {
	_occluder = p_occluder;
	set_spatial_node(p_occluder);

	_color_poly_front = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/occluder_polygon_front", Color(1.0, 0.25, 0.8, 0.3));
	_color_poly_back = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/occluder_polygon_back", Color(0.85, 0.1, 1.0, 0.3));
	_color_hole = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/occluder_hole", Color(0.0, 1.0, 1.0, 0.3));
}
