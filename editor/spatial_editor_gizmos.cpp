/*************************************************************************/
/*  spatial_editor_gizmos.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "spatial_editor_gizmos.h"

#include "geometry.h"
#include "quick_hull.h"
#include "scene/3d/camera.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/capsule_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "scene/resources/plane_shape.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/ray_shape.h"
#include "scene/resources/sphere_shape.h"
#include "scene/resources/surface_tool.h"

// Keep small children away from this file.
// It's so ugly it will eat them alive

#define HANDLE_HALF_SIZE 0.05

bool EditorSpatialGizmo::can_draw() const {
	return is_editable();
}
bool EditorSpatialGizmo::is_editable() const {

	ERR_FAIL_COND_V(!spatial_node, false);
	Node *edited_root = spatial_node->get_tree()->get_edited_scene_root();
	if (spatial_node == edited_root)
		return true;
	if (spatial_node->get_owner() == edited_root)
		return true;

	if (edited_root->is_editable_instance(spatial_node->get_owner()))
		return true;

	return false;
}

void EditorSpatialGizmo::clear() {

	for (int i = 0; i < instances.size(); i++) {

		if (instances[i].instance.is_valid())
			VS::get_singleton()->free(instances[i].instance);
	}

	billboard_handle = false;
	collision_segments.clear();
	collision_mesh = Ref<TriangleMesh>();
	instances.clear();
	handles.clear();
	secondary_handles.clear();
}

void EditorSpatialGizmo::redraw() {

	if (get_script_instance() && get_script_instance()->has_method("redraw"))
		get_script_instance()->call("redraw");
}

void EditorSpatialGizmo::Instance::create_instance(Spatial *p_base) {

	instance = VS::get_singleton()->instance_create2(mesh->get_rid(), p_base->get_world()->get_scenario());
	VS::get_singleton()->instance_attach_object_instance_id(instance, p_base->get_instance_id());
	if (skeleton.is_valid())
		VS::get_singleton()->instance_attach_skeleton(instance, skeleton);
	if (extra_margin)
		VS::get_singleton()->instance_set_extra_visibility_margin(instance, 1);
	VS::get_singleton()->instance_geometry_set_cast_shadows_setting(instance, VS::SHADOW_CASTING_SETTING_OFF);
	VS::get_singleton()->instance_set_layer_mask(instance, 1 << SpatialEditorViewport::GIZMO_EDIT_LAYER); //gizmos are 26
}

void EditorSpatialGizmo::add_mesh(const Ref<ArrayMesh> &p_mesh, bool p_billboard, const RID &p_skeleton) {

	ERR_FAIL_COND(!spatial_node);
	Instance ins;

	ins.billboard = p_billboard;
	ins.mesh = p_mesh;
	ins.skeleton = p_skeleton;
	if (valid) {
		ins.create_instance(spatial_node);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
	}

	instances.push_back(ins);
}

void EditorSpatialGizmo::add_lines(const Vector<Vector3> &p_lines, const Ref<Material> &p_material, bool p_billboard) {

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
			if (is_selected())
				w[i] = Color(1, 1, 1, 0.8);
			else
				w[i] = Color(1, 1, 1, 0.2);
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
			mesh->set_custom_aabb(Rect3(Vector3(-md, -md, -md), Vector3(md, md, md) * 2.0));
		}
	}

	ins.billboard = p_billboard;
	ins.mesh = mesh;
	if (valid) {
		ins.create_instance(spatial_node);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
	}

	instances.push_back(ins);
}

void EditorSpatialGizmo::add_unscaled_billboard(const Ref<Material> &p_material, float p_scale) {

	ERR_FAIL_COND(!spatial_node);
	Instance ins;

	Vector<Vector3> vs;
	Vector<Vector2> uv;

	vs.push_back(Vector3(-p_scale, p_scale, 0));
	vs.push_back(Vector3(p_scale, p_scale, 0));
	vs.push_back(Vector3(p_scale, -p_scale, 0));
	vs.push_back(Vector3(-p_scale, -p_scale, 0));

	uv.push_back(Vector2(0, 0));
	uv.push_back(Vector2(1, 0));
	uv.push_back(Vector2(1, 1));
	uv.push_back(Vector2(0, 1));

	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	Array a;
	a.resize(Mesh::ARRAY_MAX);
	a[Mesh::ARRAY_VERTEX] = vs;
	a[Mesh::ARRAY_TEX_UV] = uv;
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLE_FAN, a);
	mesh->surface_set_material(0, p_material);

	if (true) {
		float md = 0;
		for (int i = 0; i < vs.size(); i++) {

			md = MAX(0, vs[i].length());
		}
		if (md) {
			mesh->set_custom_aabb(Rect3(Vector3(-md, -md, -md), Vector3(md, md, md) * 2.0));
		}
	}

	ins.mesh = mesh;
	ins.unscaled = true;
	ins.billboard = true;
	if (valid) {
		ins.create_instance(spatial_node);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
	}

	instances.push_back(ins);
}

void EditorSpatialGizmo::add_collision_triangles(const Ref<TriangleMesh> &p_tmesh, const Rect3 &p_bounds) {

	collision_mesh = p_tmesh;
	collision_mesh_bounds = p_bounds;
}

void EditorSpatialGizmo::add_collision_segments(const Vector<Vector3> &p_lines) {

	int from = collision_segments.size();
	collision_segments.resize(from + p_lines.size());
	for (int i = 0; i < p_lines.size(); i++) {

		collision_segments[from + i] = p_lines[i];
	}
}

void EditorSpatialGizmo::add_handles(const Vector<Vector3> &p_handles, bool p_billboard, bool p_secondary) {

	billboard_handle = p_billboard;

	if (!is_selected() || !is_editable())
		return;

	ERR_FAIL_COND(!spatial_node);

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
			if (SpatialEditor::get_singleton()->get_over_gizmo_handle() != i)
				col = Color(0.9, 0.9, 0.9, 0.9);
			w[i] = col;
		}
	}
	a[VS::ARRAY_COLOR] = colors;
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_POINTS, a);
	if (p_billboard)
		mesh->surface_set_material(0, SpatialEditorGizmos::singleton->handle2_material_billboard);
	else
		mesh->surface_set_material(0, SpatialEditorGizmos::singleton->handle2_material);

	if (p_billboard) {
		float md = 0;
		for (int i = 0; i < p_handles.size(); i++) {

			md = MAX(0, p_handles[i].length());
		}
		if (md) {
			mesh->set_custom_aabb(Rect3(Vector3(-md, -md, -md), Vector3(md, md, md) * 2.0));
		}
	}

	ins.mesh = mesh;
	ins.billboard = p_billboard;
	ins.extra_margin = true;
	if (valid) {
		ins.create_instance(spatial_node);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
	}
	instances.push_back(ins);
	if (!p_secondary) {
		int chs = handles.size();
		handles.resize(chs + p_handles.size());
		for (int i = 0; i < p_handles.size(); i++) {
			handles[i + chs] = p_handles[i];
		}
	} else {

		int chs = secondary_handles.size();
		secondary_handles.resize(chs + p_handles.size());
		for (int i = 0; i < p_handles.size(); i++) {
			secondary_handles[i + chs] = p_handles[i];
		}
	}
}

void EditorSpatialGizmo::add_solid_box(Ref<Material> &p_material, Vector3 p_size) {
	ERR_FAIL_COND(!spatial_node);

	CubeMesh cubem;
	cubem.set_size(p_size);
	Ref<ArrayMesh> m = memnew(ArrayMesh);
	m->add_surface_from_arrays(cubem.surface_get_primitive_type(0), cubem.surface_get_arrays(0));
	m->surface_set_material(0, p_material);
	add_mesh(m);

	Instance ins;
	ins.mesh = m;
	if (valid) {
		ins.create_instance(spatial_node);
		VS::get_singleton()->instance_set_transform(ins.instance, spatial_node->get_global_transform());
	}

	instances.push_back(ins);
}

void EditorSpatialGizmo::set_spatial_node(Spatial *p_node) {

	ERR_FAIL_NULL(p_node);
	spatial_node = p_node;
}

bool EditorSpatialGizmo::intersect_frustum(const Camera *p_camera, const Vector<Plane> &p_frustum) {

	ERR_FAIL_COND_V(!spatial_node, false);
	ERR_FAIL_COND_V(!valid, false);

	if (collision_segments.size()) {

		const Plane *p = p_frustum.ptr();
		int fc = p_frustum.size();

		int vc = collision_segments.size();
		const Vector3 *vptr = collision_segments.ptr();
		Transform t = spatial_node->get_global_transform();

		for (int i = 0; i < vc / 2; i++) {

			Vector3 a = t.xform(vptr[i * 2 + 0]);
			Vector3 b = t.xform(vptr[i * 2 + 1]);

			bool any_out = false;
			for (int j = 0; j < fc; j++) {

				if (p[j].distance_to(a) > 0 && p[j].distance_to(b) > 0) {

					any_out = true;
					break;
				}
			}

			if (!any_out)
				return true;
		}

		return false;
	}

	if (collision_mesh_bounds.size != Vector3(0.0, 0.0, 0.0)) {
		Transform t = spatial_node->get_global_transform();
		const Plane *p = p_frustum.ptr();
		int fc = p_frustum.size();

		Vector3 mins = t.xform(collision_mesh_bounds.get_position());
		Vector3 max = t.xform(collision_mesh_bounds.get_position() + collision_mesh_bounds.get_size());

		bool any_out = false;

		for (int j = 0; j < fc; j++) {

			if (p[j].distance_to(mins) > 0 || p[j].distance_to(max) > 0) {

				any_out = true;
				break;
			}
		}

		if (!any_out)
			return true;
	}

	return false;
}

bool EditorSpatialGizmo::intersect_ray(const Camera *p_camera, const Point2 &p_point, Vector3 &r_pos, Vector3 &r_normal, int *r_gizmo_handle, bool p_sec_first) {

	ERR_FAIL_COND_V(!spatial_node, false);
	ERR_FAIL_COND_V(!valid, false);

	if (r_gizmo_handle) {

		Transform t = spatial_node->get_global_transform();
		t.orthonormalize();
		if (billboard_handle) {
			t.set_look_at(t.origin, t.origin - p_camera->get_transform().basis.get_axis(2), p_camera->get_transform().basis.get_axis(1));
		}

		float min_d = 1e20;
		int idx = -1;

		for (int i = 0; i < secondary_handles.size(); i++) {

			Vector3 hpos = t.xform(secondary_handles[i]);
			Vector2 p = p_camera->unproject_position(hpos);
			if (p.distance_to(p_point) < SpatialEditorGizmos::singleton->handle_t->get_width() * 0.6) {

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
			Vector2 p = p_camera->unproject_position(hpos);
			if (p.distance_to(p_point) < SpatialEditorGizmos::singleton->handle_t->get_width() * 0.6) {

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
			s[0] = p_camera->unproject_position(a);
			s[1] = p_camera->unproject_position(b);

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

				if (camp.distance_to(tcp) < p_camera->get_znear())
					continue;
				cp = tcp;
				cpd = pd;
			}
		}

		if (cpd < 8) {

			r_pos = cp;
			r_normal = -p_camera->project_ray_normal(p_point);
			return true;
		}

		return false;
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

		instances[i].create_instance(spatial_node);
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

		if (instances[i].instance.is_valid())
			VS::get_singleton()->free(instances[i].instance);
		instances[i].instance = RID();
	}

	valid = false;
}

Ref<SpatialMaterial> EditorSpatialGizmo::create_material(const String &p_name, const Color &p_color, bool p_billboard, bool p_on_top, bool p_use_vertex_color) {

	String name = p_name;

	if (!is_editable()) {
		name += "@readonly";
	} else if (is_selected()) {
		name += "@selected";
	}

	if (SpatialEditorGizmos::singleton->material_cache.has(name)) {
		return SpatialEditorGizmos::singleton->material_cache[name];
	}

	Color color = p_color;

	if (!is_editable()) {
		color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/instanced");
	}
	if (!is_selected()) {
		color.a *= 0.3;
	}

	Ref<SpatialMaterial> line_material;
	line_material.instance();
	line_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	line_material->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	if (p_use_vertex_color) {
		line_material->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		line_material->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
	}

	if (p_billboard) {
		line_material->set_billboard_mode(SpatialMaterial::BILLBOARD_ENABLED);
	}

	if (p_on_top && is_selected()) {
		line_material->set_on_top_of_alpha();
	}

	line_material->set_albedo(color);

	SpatialEditorGizmos::singleton->material_cache[name] = line_material;

	return line_material;
}

Ref<SpatialMaterial> EditorSpatialGizmo::create_icon_material(const String &p_name, const Ref<Texture> &p_texture, bool p_on_top, const Color &p_albedo) {

	String name = p_name;

	if (!is_editable()) {
		name += "@readonly";
	} else if (is_selected()) {
		name += "@selected";
	}

	if (SpatialEditorGizmos::singleton->material_cache.has(name)) {
		return SpatialEditorGizmos::singleton->material_cache[name];
	}

	Color color = p_albedo;

	if (!is_editable()) {
		color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/instanced");
	} else if (!is_selected()) {
		color.a *= 0.3;
	}

	Ref<SpatialMaterial> icon;
	icon.instance();
	icon->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	icon->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	icon->set_depth_draw_mode(SpatialMaterial::DEPTH_DRAW_DISABLED);
	icon->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	icon->set_albedo(color);
	icon->set_texture(SpatialMaterial::TEXTURE_ALBEDO, p_texture);
	icon->set_flag(SpatialMaterial::FLAG_FIXED_SIZE, true);
	icon->set_billboard_mode(SpatialMaterial::BILLBOARD_ENABLED);

	if (p_on_top && is_selected()) {
		icon->set_on_top_of_alpha();
	}

	SpatialEditorGizmos::singleton->material_cache[name] = icon;

	return icon;
}

void EditorSpatialGizmo::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_lines", "lines", "material", "billboard"), &EditorSpatialGizmo::add_lines, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_mesh", "mesh", "billboard", "skeleton"), &EditorSpatialGizmo::add_mesh, DEFVAL(false), DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("add_collision_segments", "segments"), &EditorSpatialGizmo::add_collision_segments);
	ClassDB::bind_method(D_METHOD("add_collision_triangles", "triangles", "bounds"), &EditorSpatialGizmo::add_collision_triangles);
	ClassDB::bind_method(D_METHOD("add_unscaled_billboard", "material", "default_scale"), &EditorSpatialGizmo::add_unscaled_billboard, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("add_handles", "handles", "billboard", "secondary"), &EditorSpatialGizmo::add_handles, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_spatial_node", "node"), &EditorSpatialGizmo::_set_spatial_node);
	ClassDB::bind_method(D_METHOD("clear"), &EditorSpatialGizmo::clear);

	BIND_VMETHOD(MethodInfo("redraw"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "get_handle_name", PropertyInfo(Variant::INT, "index")));

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
	base = NULL;
	spatial_node = NULL;
}

EditorSpatialGizmo::~EditorSpatialGizmo() {

	clear();
}

Vector3 EditorSpatialGizmo::get_handle_pos(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, handles.size(), Vector3());

	return handles[p_idx];
}

//// light gizmo

String LightSpatialGizmo::get_handle_name(int p_idx) const {

	if (p_idx == 0)
		return "Radius";
	else
		return "Aperture";
}

Variant LightSpatialGizmo::get_handle_value(int p_idx) const {

	if (p_idx == 0)
		return light->get_param(Light::PARAM_RANGE);
	if (p_idx == 1)
		return light->get_param(Light::PARAM_SPOT_ANGLE);

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

void LightSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	Transform gt = light->get_global_transform();
	gt.orthonormalize();
	Transform gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 s[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };
	if (p_idx == 0) {

		if (Object::cast_to<SpotLight>(light)) {
			Vector3 ra, rb;
			Geometry::get_closest_points_between_segments(Vector3(), Vector3(0, 0, -4096), s[0], s[1], ra, rb);

			float d = -ra.z;
			if (d < 0)
				d = 0;

			light->set_param(Light::PARAM_RANGE, d);
		} else if (Object::cast_to<OmniLight>(light)) {

			Plane cp = Plane(gt.origin, p_camera->get_transform().basis.get_axis(2));

			Vector3 inters;
			if (cp.intersects_ray(ray_from, ray_dir, &inters)) {

				float r = inters.distance_to(gt.origin);
				light->set_param(Light::PARAM_RANGE, r);
			}
		}

	} else if (p_idx == 1) {

		float a = _find_closest_angle_to_half_pi_arc(s[0], s[1], light->get_param(Light::PARAM_RANGE), gt);
		light->set_param(Light::PARAM_SPOT_ANGLE, CLAMP(a, 0.01, 89.99));
	}
}

void LightSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

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

void LightSpatialGizmo::redraw() {

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/light");

	if (Object::cast_to<DirectionalLight>(light)) {

		Ref<Material> material = create_material("light_directional_material", gizmo_color);
		Ref<Material> icon = create_icon_material("light_directional_icon", SpatialEditor::get_singleton()->get_icon("GizmoDirectionalLight", "EditorIcons"));

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

		add_lines(lines, material);
		add_collision_segments(lines);
		add_unscaled_billboard(icon, 0.05);
	}

	if (Object::cast_to<OmniLight>(light)) {

		Ref<Material> material = create_material("light_omni_material", gizmo_color, true);
		Ref<Material> icon = create_icon_material("light_omni_icon", SpatialEditor::get_singleton()->get_icon("GizmoLight", "EditorIcons"));
		clear();

		OmniLight *on = Object::cast_to<OmniLight>(light);

		float r = on->get_param(Light::PARAM_RANGE);

		Vector<Vector3> points;

		for (int i = 0; i <= 360; i++) {

			float ra = Math::deg2rad((float)i);
			float rb = Math::deg2rad((float)i + 1);
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

			/*points.push_back(Vector3(a.x,0,a.y));
			points.push_back(Vector3(b.x,0,b.y));
			points.push_back(Vector3(0,a.x,a.y));
			points.push_back(Vector3(0,b.x,b.y));*/
			points.push_back(Vector3(a.x, a.y, 0));
			points.push_back(Vector3(b.x, b.y, 0));
		}

		add_lines(points, material, true);
		add_collision_segments(points);

		add_unscaled_billboard(icon, 0.05);

		Vector<Vector3> handles;
		handles.push_back(Vector3(r, 0, 0));
		add_handles(handles, true);
	}

	if (Object::cast_to<SpotLight>(light)) {

		Ref<Material> material = create_material("light_spot_material", gizmo_color);
		Ref<Material> icon = create_icon_material("light_spot_icon", SpatialEditor::get_singleton()->get_icon("GizmoSpotLight", "EditorIcons"));

		clear();

		Vector<Vector3> points;
		SpotLight *on = Object::cast_to<SpotLight>(light);

		float r = on->get_param(Light::PARAM_RANGE);
		float w = r * Math::sin(Math::deg2rad(on->get_param(Light::PARAM_SPOT_ANGLE)));
		float d = r * Math::cos(Math::deg2rad(on->get_param(Light::PARAM_SPOT_ANGLE)));

		for (int i = 0; i < 360; i++) {

			float ra = Math::deg2rad((float)i);
			float rb = Math::deg2rad((float)i + 1);
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * w;

			points.push_back(Vector3(a.x, a.y, -d));
			points.push_back(Vector3(b.x, b.y, -d));

			if (i % 90 == 0) {

				points.push_back(Vector3(a.x, a.y, -d));
				points.push_back(Vector3());
			}
		}

		points.push_back(Vector3(0, 0, -r));
		points.push_back(Vector3());

		add_lines(points, material);

		Vector<Vector3> handles;
		handles.push_back(Vector3(0, 0, -r));

		Vector<Vector3> collision_segments;

		for (int i = 0; i < 64; i++) {

			float ra = i * Math_PI * 2.0 / 64.0;
			float rb = (i + 1) * Math_PI * 2.0 / 64.0;
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * w;

			collision_segments.push_back(Vector3(a.x, a.y, -d));
			collision_segments.push_back(Vector3(b.x, b.y, -d));

			if (i % 16 == 0) {

				collision_segments.push_back(Vector3(a.x, a.y, -d));
				collision_segments.push_back(Vector3());
			}

			if (i == 16) {

				handles.push_back(Vector3(a.x, a.y, -d));
			}
		}

		collision_segments.push_back(Vector3(0, 0, -r));
		collision_segments.push_back(Vector3());

		add_handles(handles);
		add_collision_segments(collision_segments);
		add_unscaled_billboard(icon, 0.05);
	}
}

LightSpatialGizmo::LightSpatialGizmo(Light *p_light) {

	light = p_light;
	set_spatial_node(p_light);
}

//////

//// player gizmo

String AudioStreamPlayer3DSpatialGizmo::get_handle_name(int p_idx) const {

	return "Emission Radius";
}

Variant AudioStreamPlayer3DSpatialGizmo::get_handle_value(int p_idx) const {

	return player->get_emission_angle();
}

void AudioStreamPlayer3DSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	Transform gt = player->get_global_transform();
	gt.orthonormalize();
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

void AudioStreamPlayer3DSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

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

void AudioStreamPlayer3DSpatialGizmo::redraw() {

	clear();

	Ref<Material> icon = create_icon_material("stream_player_3d_material", SpatialEditor::get_singleton()->get_icon("GizmoSpatialSamplePlayer", "EditorIcons"));

	if (player->is_emission_angle_enabled()) {

		Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/stream_player_3d");
		Ref<Material> material = create_material("stream_player_3d_material", gizmo_color);

		float pc = player->get_emission_angle();

		Vector<Vector3> points;
		points.resize(208);

		float ofs = -Math::cos(Math::deg2rad(pc));
		float radius = Math::sin(Math::deg2rad(pc));

		for (int i = 0; i < 100; i++) {

			float a = i * 2.0 * Math_PI / 100.0;
			float an = (i + 1) * 2.0 * Math_PI / 100.0;

			Vector3 from(Math::sin(a) * radius, Math::cos(a) * radius, ofs);
			Vector3 to(Math::sin(an) * radius, Math::cos(an) * radius, ofs);

			points[i * 2 + 0] = from;
			points[i * 2 + 1] = to;
		}

		for (int i = 0; i < 4; i++) {

			float a = i * 2.0 * Math_PI / 4.0;

			Vector3 from(Math::sin(a) * radius, Math::cos(a) * radius, ofs);

			points[200 + i * 2 + 0] = from;
			points[200 + i * 2 + 1] = Vector3();
		}

		add_lines(points, material);
		add_collision_segments(points);

		Vector<Vector3> handles;
		float ha = Math::deg2rad(player->get_emission_angle());
		handles.push_back(Vector3(Math::sin(ha), 0, -Math::cos(ha)));
		add_handles(handles);
	}

	add_unscaled_billboard(icon, 0.05);
}

AudioStreamPlayer3DSpatialGizmo::AudioStreamPlayer3DSpatialGizmo(AudioStreamPlayer3D *p_player) {

	player = p_player;
	set_spatial_node(p_player);
}

//////

String CameraSpatialGizmo::get_handle_name(int p_idx) const {

	if (camera->get_projection() == Camera::PROJECTION_PERSPECTIVE) {
		return "FOV";
	} else {
		return "Size";
	}
}
Variant CameraSpatialGizmo::get_handle_value(int p_idx) const {

	if (camera->get_projection() == Camera::PROJECTION_PERSPECTIVE) {
		return camera->get_fov();
	} else {

		return camera->get_size();
	}
}
void CameraSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	Transform gt = camera->get_global_transform();
	gt.orthonormalize();
	Transform gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 s[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

	if (camera->get_projection() == Camera::PROJECTION_PERSPECTIVE) {
		Transform gt = camera->get_global_transform();
		float a = _find_closest_angle_to_half_pi_arc(s[0], s[1], 1.0, gt);
		camera->set("fov", a);
	} else {

		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(0, 0, -1), Vector3(4096, 0, -1), s[0], s[1], ra, rb);
		float d = ra.x * 2.0;
		if (d < 0)
			d = 0;

		camera->set("size", d);
	}
}
void CameraSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

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

void CameraSpatialGizmo::redraw() {

	clear();

	Vector<Vector3> lines;
	Vector<Vector3> handles;

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/camera");
	Ref<Material> material = create_material("camera_material", gizmo_color);
	Ref<Material> icon = create_icon_material("camera_icon", SpatialEditor::get_singleton()->get_icon("GizmoCamera", "EditorIcons"));

	switch (camera->get_projection()) {

		case Camera::PROJECTION_PERSPECTIVE: {

			float fov = camera->get_fov();

			Vector3 side = Vector3(Math::sin(Math::deg2rad(fov)), 0, -Math::cos(Math::deg2rad(fov)));
			Vector3 nside = side;
			nside.x = -nside.x;
			Vector3 up = Vector3(0, side.x, 0);

#define ADD_TRIANGLE(m_a, m_b, m_c) \
	{                               \
		lines.push_back(m_a);       \
		lines.push_back(m_b);       \
		lines.push_back(m_b);       \
		lines.push_back(m_c);       \
		lines.push_back(m_c);       \
		lines.push_back(m_a);       \
	}

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
	}

	add_lines(lines, material);
	add_collision_segments(lines);
	add_unscaled_billboard(icon, 0.05);
	add_handles(handles);
}

CameraSpatialGizmo::CameraSpatialGizmo(Camera *p_camera) {

	camera = p_camera;
	set_spatial_node(camera);
}

//////

bool MeshInstanceSpatialGizmo::can_draw() const {
	return true; //mesh can always draw (even though nothing is displayed)
}
void MeshInstanceSpatialGizmo::redraw() {

	Ref<Mesh> m = mesh->get_mesh();
	if (!m.is_valid())
		return; //none

	Ref<TriangleMesh> tm = m->generate_triangle_mesh();
	if (tm.is_valid()) {
		Rect3 aabb;
		add_collision_triangles(tm, aabb);
	}
}

MeshInstanceSpatialGizmo::MeshInstanceSpatialGizmo(MeshInstance *p_mesh) {

	mesh = p_mesh;
	set_spatial_node(p_mesh);
}

/////

void Position3DSpatialGizmo::redraw() {

	clear();
	add_mesh(SpatialEditorGizmos::singleton->pos3d_mesh);
	Vector<Vector3> cursor_points;
	float cs = 0.25;
	cursor_points.push_back(Vector3(+cs, 0, 0));
	cursor_points.push_back(Vector3(-cs, 0, 0));
	cursor_points.push_back(Vector3(0, +cs, 0));
	cursor_points.push_back(Vector3(0, -cs, 0));
	cursor_points.push_back(Vector3(0, 0, +cs));
	cursor_points.push_back(Vector3(0, 0, -cs));
	add_collision_segments(cursor_points);
}

Position3DSpatialGizmo::Position3DSpatialGizmo(Position3D *p_p3d) {

	p3d = p_p3d;
	set_spatial_node(p3d);
}

/////

void SkeletonSpatialGizmo::redraw() {

	clear();

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/skeleton");
	Ref<Material> material = create_material("skeleton_material", gizmo_color);

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
		bones[i] = 0;
		weights[i] = 0;
	}

	weights[0] = 1;

	Rect3 aabb;

	Color bonecolor = Color(1.0, 0.4, 0.4, 0.3);
	Color rootcolor = Color(0.4, 1.0, 0.4, 0.1);

	for (int i = 0; i < skel->get_bone_count(); i++) {

		int parent = skel->get_bone_parent(i);

		if (parent >= 0) {
			grests[i] = grests[parent] * skel->get_bone_rest(i);

			Vector3 v0 = grests[parent].origin;
			Vector3 v1 = grests[i].origin;
			Vector3 d = (v1 - v0).normalized();
			float dist = v0.distance_to(v1);

			//find closest axis
			int closest = -1;
			float closest_d = 0.0;

			for (int j = 0; j < 3; j++) {
				float dp = Math::abs(grests[parent].basis[j].normalized().dot(d));
				if (j == 0 || dp > closest_d)
					closest = j;
			}

			//find closest other
			Vector3 first;
			Vector3 points[4];
			int pointidx = 0;
			for (int j = 0; j < 3; j++) {

				bones[0] = parent;
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(rootcolor);
				surface_tool->add_vertex(v0 - grests[parent].basis[j].normalized() * dist * 0.05);
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(rootcolor);
				surface_tool->add_vertex(v0 + grests[parent].basis[j].normalized() * dist * 0.05);

				if (j == closest)
					continue;

				Vector3 axis;
				if (first == Vector3()) {
					axis = d.cross(d.cross(grests[parent].basis[j])).normalized();
					first = axis;
				} else {
					axis = d.cross(first).normalized();
				}

				for (int k = 0; k < 2; k++) {

					if (k == 1)
						axis = -axis;
					Vector3 point = v0 + d * dist * 0.2;
					point += axis * dist * 0.1;

					bones[0] = parent;
					surface_tool->add_bones(bones);
					surface_tool->add_weights(weights);
					surface_tool->add_color(bonecolor);
					surface_tool->add_vertex(v0);
					surface_tool->add_bones(bones);
					surface_tool->add_weights(weights);
					surface_tool->add_color(bonecolor);
					surface_tool->add_vertex(point);

					bones[0] = parent;
					surface_tool->add_bones(bones);
					surface_tool->add_weights(weights);
					surface_tool->add_color(bonecolor);
					surface_tool->add_vertex(point);
					bones[0] = i;
					surface_tool->add_bones(bones);
					surface_tool->add_weights(weights);
					surface_tool->add_color(bonecolor);
					surface_tool->add_vertex(v1);
					points[pointidx++] = point;
				}
			}

			SWAP(points[1], points[2]);
			for (int j = 0; j < 4; j++) {

				bones[0] = parent;
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(bonecolor);
				surface_tool->add_vertex(points[j]);
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(bonecolor);
				surface_tool->add_vertex(points[(j + 1) % 4]);
			}

			/*
			bones[0]=parent;
			surface_tool->add_bones(bones);
			surface_tool->add_weights(weights);
			surface_tool->add_color(Color(0.4,1,0.4,0.4));
			surface_tool->add_vertex(v0);
			bones[0]=i;
			surface_tool->add_bones(bones);
			surface_tool->add_weights(weights);
			surface_tool->add_color(Color(0.4,1,0.4,0.4));
			surface_tool->add_vertex(v1);
*/
		} else {

			grests[i] = skel->get_bone_rest(i);
			bones[0] = i;
		}
		/*
		Transform  t = grests[i];
		t.orthonormalize();

		for (int i=0;i<6;i++) {


			Vector3 face_points[4];

			for (int j=0;j<4;j++) {

				float v[3];
				v[0]=1.0;
				v[1]=1-2*((j>>1)&1);
				v[2]=v[1]*(1-2*(j&1));

				for (int k=0;k<3;k++) {

					if (i<3)
						face_points[j][(i+k)%3]=v[k]*(i>=3?-1:1);
					else
						face_points[3-j][(i+k)%3]=v[k]*(i>=3?-1:1);
				}
			}

			for(int j=0;j<4;j++) {
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(Color(1.0,0.4,0.4,0.4));
				surface_tool->add_vertex(t.xform(face_points[j]*0.04));
				surface_tool->add_bones(bones);
				surface_tool->add_weights(weights);
				surface_tool->add_color(Color(1.0,0.4,0.4,0.4));
				surface_tool->add_vertex(t.xform(face_points[(j+1)%4]*0.04));
			}

		}
		*/
	}

	Ref<ArrayMesh> m = surface_tool->commit();
	add_mesh(m, false, skel->get_skeleton());
}

SkeletonSpatialGizmo::SkeletonSpatialGizmo(Skeleton *p_skel) {

	skel = p_skel;
	set_spatial_node(p_skel);
}
#if 0
void RoomSpatialGizmo::redraw() {

	clear();
	Ref<RoomBounds> roomie = room->get_room();
	if (roomie.is_null())
		return;
	PoolVector<Face3> faces = roomie->get_geometry_hint();

	Vector<Vector3> lines;
	int fc = faces.size();
	PoolVector<Face3>::Read r = faces.read();

	Map<_EdgeKey, Vector3> edge_map;

	for (int i = 0; i < fc; i++) {

		Vector3 fn = r[i].get_plane().normal;

		for (int j = 0; j < 3; j++) {

			_EdgeKey ek;
			ek.from = r[i].vertex[j].snapped(Vector3(CMP_EPSILON, CMP_EPSILON, CMP_EPSILON));
			ek.to = r[i].vertex[(j + 1) % 3].snapped(Vector3(CMP_EPSILON, CMP_EPSILON, CMP_EPSILON));
			if (ek.from < ek.to)
				SWAP(ek.from, ek.to);

			Map<_EdgeKey, Vector3>::Element *E = edge_map.find(ek);

			if (E) {

				if (E->get().dot(fn) > 0.9) {

					E->get() = Vector3();
				}

			} else {

				edge_map[ek] = fn;
			}
		}
	}

	for (Map<_EdgeKey, Vector3>::Element *E = edge_map.front(); E; E = E->next()) {

		if (E->get() != Vector3()) {
			lines.push_back(E->key().from);
			lines.push_back(E->key().to);
		}
	}

	add_lines(lines, SpatialEditorGizmos::singleton->room_material);
	add_collision_segments(lines);
}

RoomSpatialGizmo::RoomSpatialGizmo(Room *p_room) {

	set_spatial_node(p_room);
	room = p_room;
}

/////

void PortalSpatialGizmo::redraw() {

	clear();

	Vector<Point2> points = portal->get_shape();
	if (points.size() == 0) {
		return;
	}

	Vector<Vector3> lines;

	Vector3 center;
	for (int i = 0; i < points.size(); i++) {

		Vector3 f;
		f.x = points[i].x;
		f.y = points[i].y;
		Vector3 fn;
		fn.x = points[(i + 1) % points.size()].x;
		fn.y = points[(i + 1) % points.size()].y;
		center += f;

		lines.push_back(f);
		lines.push_back(fn);
	}

	center /= points.size();
	lines.push_back(center);
	lines.push_back(center + Vector3(0, 0, 1));

	add_lines(lines, SpatialEditorGizmos::singleton->portal_material);
	add_collision_segments(lines);
}

PortalSpatialGizmo::PortalSpatialGizmo(Portal *p_portal) {

	set_spatial_node(p_portal);
	portal = p_portal;
}

#endif
/////

void RayCastSpatialGizmo::redraw() {

	clear();

	Vector<Vector3> lines;

	lines.push_back(Vector3());
	lines.push_back(raycast->get_cast_to());

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	Ref<Material> material = create_material("shape_material", gizmo_color);

	add_lines(lines, material);
	add_collision_segments(lines);
}

RayCastSpatialGizmo::RayCastSpatialGizmo(RayCast *p_raycast) {

	set_spatial_node(p_raycast);
	raycast = p_raycast;
}

/////

void VehicleWheelSpatialGizmo::redraw() {

	clear();

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

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	Ref<Material> material = create_material("shape_material", gizmo_color);

	add_lines(points, material);
	add_collision_segments(points);
}

VehicleWheelSpatialGizmo::VehicleWheelSpatialGizmo(VehicleWheel *p_car_wheel) {

	set_spatial_node(p_car_wheel);
	car_wheel = p_car_wheel;
}

///////////

String CollisionShapeSpatialGizmo::get_handle_name(int p_idx) const {

	Ref<Shape> s = cs->get_shape();
	if (s.is_null())
		return "";

	if (Object::cast_to<SphereShape>(*s)) {

		return "Radius";
	}

	if (Object::cast_to<BoxShape>(*s)) {

		return "Extents";
	}

	if (Object::cast_to<CapsuleShape>(*s)) {

		return p_idx == 0 ? "Radius" : "Height";
	}

	if (Object::cast_to<RayShape>(*s)) {

		return "Length";
	}

	return "";
}
Variant CollisionShapeSpatialGizmo::get_handle_value(int p_idx) const {

	Ref<Shape> s = cs->get_shape();
	if (s.is_null())
		return Variant();

	if (Object::cast_to<SphereShape>(*s)) {

		Ref<SphereShape> ss = s;
		return ss->get_radius();
	}

	if (Object::cast_to<BoxShape>(*s)) {

		Ref<BoxShape> bs = s;
		return bs->get_extents();
	}

	if (Object::cast_to<CapsuleShape>(*s)) {

		Ref<CapsuleShape> cs = s;
		return p_idx == 0 ? cs->get_radius() : cs->get_height();
	}

	if (Object::cast_to<RayShape>(*s)) {

		Ref<RayShape> cs = s;
		return cs->get_length();
	}

	return Variant();
}
void CollisionShapeSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {
	Ref<Shape> s = cs->get_shape();
	if (s.is_null())
		return;

	Transform gt = cs->get_global_transform();
	gt.orthonormalize();
	Transform gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

	if (Object::cast_to<SphereShape>(*s)) {

		Ref<SphereShape> ss = s;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), sg[0], sg[1], ra, rb);
		float d = ra.x;
		if (d < 0.001)
			d = 0.001;

		ss->set_radius(d);
	}

	if (Object::cast_to<RayShape>(*s)) {

		Ref<RayShape> rs = s;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), Vector3(0, 0, 4096), sg[0], sg[1], ra, rb);
		float d = ra.z;
		if (d < 0.001)
			d = 0.001;

		rs->set_length(d);
	}

	if (Object::cast_to<BoxShape>(*s)) {

		Vector3 axis;
		axis[p_idx] = 1.0;
		Ref<BoxShape> bs = s;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = ra[p_idx];
		if (d < 0.001)
			d = 0.001;

		Vector3 he = bs->get_extents();
		he[p_idx] = d;
		bs->set_extents(he);
	}

	if (Object::cast_to<CapsuleShape>(*s)) {

		Vector3 axis;
		axis[p_idx == 0 ? 0 : 2] = 1.0;
		Ref<CapsuleShape> cs = s;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = axis.dot(ra);
		if (p_idx == 1)
			d -= cs->get_radius();
		if (d < 0.001)
			d = 0.001;

		if (p_idx == 0)
			cs->set_radius(d);
		else if (p_idx == 1)
			cs->set_height(d * 2.0);
	}
}
void CollisionShapeSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {
	Ref<Shape> s = cs->get_shape();
	if (s.is_null())
		return;

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
			if (p_idx == 0)
				ss->set_radius(p_restore);
			else
				ss->set_height(p_restore);
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
void CollisionShapeSpatialGizmo::redraw() {

	clear();

	Ref<Shape> s = cs->get_shape();
	if (s.is_null())
		return;

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	Ref<Material> material = create_material("shape_material", gizmo_color);

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

		add_lines(points, material);
		add_collision_segments(collision_segments);
		Vector<Vector3> handles;
		handles.push_back(Vector3(r, 0, 0));
		add_handles(handles);
	}

	if (Object::cast_to<BoxShape>(*s)) {

		Ref<BoxShape> bs = s;
		Vector<Vector3> lines;
		Rect3 aabb;
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

		add_lines(lines, material);
		add_collision_segments(lines);
		add_handles(handles);
	}

	if (Object::cast_to<CapsuleShape>(*s)) {

		Ref<CapsuleShape> cs = s;
		float radius = cs->get_radius();
		float height = cs->get_height();

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

		add_lines(points, material);

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

		add_collision_segments(collision_segments);

		Vector<Vector3> handles;
		handles.push_back(Vector3(cs->get_radius(), 0, 0));
		handles.push_back(Vector3(0, 0, cs->get_height() * 0.5 + cs->get_radius()));
		add_handles(handles);
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

		add_lines(points, material);
		add_collision_segments(points);
	}

	if (Object::cast_to<ConvexPolygonShape>(*s)) {

		PoolVector<Vector3> points = Object::cast_to<ConvexPolygonShape>(*s)->get_points();

		if (points.size() > 3) {

			QuickHull qh;
			Vector<Vector3> varr = Variant(points);
			Geometry::MeshData md;
			Error err = qh.build(varr, md);
			if (err == OK) {
				Vector<Vector3> points;
				points.resize(md.edges.size() * 2);
				for (int i = 0; i < md.edges.size(); i++) {
					points[i * 2 + 0] = md.vertices[md.edges[i].a];
					points[i * 2 + 1] = md.vertices[md.edges[i].b];
				}

				add_lines(points, material);
				add_collision_segments(points);
			}
		}
	}

	if (Object::cast_to<RayShape>(*s)) {

		Ref<RayShape> rs = s;

		Vector<Vector3> points;
		points.push_back(Vector3());
		points.push_back(Vector3(0, 0, rs->get_length()));
		add_lines(points, material);
		add_collision_segments(points);
		Vector<Vector3> handles;
		handles.push_back(Vector3(0, 0, rs->get_length()));
		add_handles(handles);
	}
}
CollisionShapeSpatialGizmo::CollisionShapeSpatialGizmo(CollisionShape *p_cs) {

	cs = p_cs;
	set_spatial_node(p_cs);
}

/////

void CollisionPolygonSpatialGizmo::redraw() {

	clear();

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

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/shape");
	Ref<Material> material = create_material("shape_material", gizmo_color);

	add_lines(lines, material);
	add_collision_segments(lines);
}

CollisionPolygonSpatialGizmo::CollisionPolygonSpatialGizmo(CollisionPolygon *p_polygon) {

	set_spatial_node(p_polygon);
	polygon = p_polygon;
}
///

String VisibilityNotifierGizmo::get_handle_name(int p_idx) const {

	switch (p_idx) {
		case 0: return "X";
		case 1: return "Y";
		case 2: return "Z";
	}

	return "";
}
Variant VisibilityNotifierGizmo::get_handle_value(int p_idx) const {

	return notifier->get_aabb();
}
void VisibilityNotifierGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	Transform gt = notifier->get_global_transform();
	//gt.orthonormalize();
	Transform gi = gt.affine_inverse();

	Rect3 aabb = notifier->get_aabb();
	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };
	Vector3 ofs = aabb.position + aabb.size * 0.5;

	Vector3 axis;
	axis[p_idx] = 1.0;

	Vector3 ra, rb;
	Geometry::get_closest_points_between_segments(ofs, ofs + axis * 4096, sg[0], sg[1], ra, rb);
	float d = ra[p_idx];
	if (d < 0.001)
		d = 0.001;

	aabb.position[p_idx] = (aabb.position[p_idx] + aabb.size[p_idx] * 0.5) - d;
	aabb.size[p_idx] = d * 2;
	notifier->set_aabb(aabb);
}

void VisibilityNotifierGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

	if (p_cancel) {
		notifier->set_aabb(p_restore);
		return;
	}

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Notifier Extents"));
	ur->add_do_method(notifier, "set_aabb", notifier->get_aabb());
	ur->add_undo_method(notifier, "set_aabb", p_restore);
	ur->commit_action();
}

void VisibilityNotifierGizmo::redraw() {

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/visibility_notifier");
	Ref<Material> material = create_material("visibility_notifier_material", gizmo_color);

	clear();

	Vector<Vector3> lines;
	Rect3 aabb = notifier->get_aabb();

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
		handles.push_back(ax);
	}

	add_lines(lines, material);
	//add_unscaled_billboard(SpatialEditorGizmos::singleton->visi,0.05);
	add_collision_segments(lines);
	add_handles(handles);
}
VisibilityNotifierGizmo::VisibilityNotifierGizmo(VisibilityNotifier *p_notifier) {

	notifier = p_notifier;
	set_spatial_node(p_notifier);
}

////////

///

String ParticlesGizmo::get_handle_name(int p_idx) const {

	switch (p_idx) {
		case 0: return "Size X";
		case 1: return "Size Y";
		case 2: return "Size Z";
		case 3: return "Pos X";
		case 4: return "Pos Y";
		case 5: return "Pos Z";
	}

	return "";
}
Variant ParticlesGizmo::get_handle_value(int p_idx) const {

	return particles->get_visibility_aabb();
}
void ParticlesGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	Transform gt = particles->get_global_transform();
	//gt.orthonormalize();
	Transform gi = gt.affine_inverse();

	bool move = p_idx >= 3;
	p_idx = p_idx % 3;

	Rect3 aabb = particles->get_visibility_aabb();
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

		aabb.position[p_idx] = d - 1.0 - aabb.size[p_idx] * 0.5;
		particles->set_visibility_aabb(aabb);

	} else {
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(ofs, ofs + axis * 4096, sg[0], sg[1], ra, rb);

		float d = ra[p_idx] - ofs[p_idx];
		if (d < 0.001)
			d = 0.001;
		//resize
		aabb.position[p_idx] = (aabb.position[p_idx] + aabb.size[p_idx] * 0.5) - d;
		aabb.size[p_idx] = d * 2;
		particles->set_visibility_aabb(aabb);
	}
}

void ParticlesGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

	if (p_cancel) {
		particles->set_visibility_aabb(p_restore);
		return;
	}

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Change Particles AABB"));
	ur->add_do_method(particles, "set_custom_aabb", particles->get_visibility_aabb());
	ur->add_undo_method(particles, "set_custom_aabb", p_restore);
	ur->commit_action();
}

void ParticlesGizmo::redraw() {

	clear();

	Vector<Vector3> lines;
	Rect3 aabb = particles->get_visibility_aabb();

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

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/particles");
	Ref<Material> material = create_material("particles_material", gizmo_color);
	Ref<Material> icon = create_icon_material("particles_icon", SpatialEditor::get_singleton()->get_icon("GizmoParticles", "EditorIcons"));

	add_lines(lines, material);
	add_collision_segments(lines);

	if (is_selected()) {

		gizmo_color.a = 0.1;
		Ref<Material> solid_material = create_material("particles_solid_material", gizmo_color);
		add_solid_box(solid_material, aabb.get_size());
	}

	//add_unscaled_billboard(SpatialEditorGizmos::singleton->visi,0.05);
	add_unscaled_billboard(icon, 0.05);
	add_handles(handles);
}
ParticlesGizmo::ParticlesGizmo(Particles *p_particles) {

	particles = p_particles;
	set_spatial_node(p_particles);
}

////////

///

String ReflectionProbeGizmo::get_handle_name(int p_idx) const {

	switch (p_idx) {
		case 0: return "Extents X";
		case 1: return "Extents Y";
		case 2: return "Extents Z";
		case 3: return "Origin X";
		case 4: return "Origin Y";
		case 5: return "Origin Z";
	}

	return "";
}
Variant ReflectionProbeGizmo::get_handle_value(int p_idx) const {

	return Rect3(probe->get_extents(), probe->get_origin_offset());
}
void ReflectionProbeGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	Transform gt = probe->get_global_transform();
	//gt.orthonormalize();
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
		if (d < 0.001)
			d = 0.001;

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
		float d = ra[p_idx];
		d += 0.25;

		origin[p_idx] = d;
		probe->set_origin_offset(origin);
	}
}

void ReflectionProbeGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

	Rect3 restore = p_restore;

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

void ReflectionProbeGizmo::redraw() {

	clear();

	Vector<Vector3> lines;
	Vector<Vector3> internal_lines;
	Vector3 extents = probe->get_extents();

	Rect3 aabb;
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

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/reflection_probe");
	Ref<Material> material = create_material("reflection_probe_material", gizmo_color);
	Ref<Material> icon = create_icon_material("reflection_probe_icon", SpatialEditor::get_singleton()->get_icon("GizmoReflectionProbe", "EditorIcons"));

	Color gizmo_color_internal = gizmo_color;
	gizmo_color_internal.a = 0.5;
	Ref<Material> material_internal = create_material("reflection_internal_material", gizmo_color_internal);

	add_lines(lines, material);
	add_lines(internal_lines, material_internal);

	if (is_selected()) {

		gizmo_color.a = 0.1;
		Ref<Material> solid_material = create_material("reflection_probe_solid_material", gizmo_color);
		add_solid_box(solid_material, probe->get_extents() * 2.0);
	}

	//add_unscaled_billboard(SpatialEditorGizmos::singleton->visi,0.05);
	add_unscaled_billboard(icon, 0.05);
	add_collision_segments(lines);
	add_handles(handles);
}
ReflectionProbeGizmo::ReflectionProbeGizmo(ReflectionProbe *p_probe) {

	probe = p_probe;
	set_spatial_node(p_probe);
}

////////

///

String GIProbeGizmo::get_handle_name(int p_idx) const {

	switch (p_idx) {
		case 0: return "Extents X";
		case 1: return "Extents Y";
		case 2: return "Extents Z";
	}

	return "";
}
Variant GIProbeGizmo::get_handle_value(int p_idx) const {

	return probe->get_extents();
}
void GIProbeGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	Transform gt = probe->get_global_transform();
	//gt.orthonormalize();
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
	if (d < 0.001)
		d = 0.001;

	extents[p_idx] = d;
	probe->set_extents(extents);
}

void GIProbeGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

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

void GIProbeGizmo::redraw() {

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/gi_probe");
	Ref<Material> material = create_material("gi_probe_material", gizmo_color);
	Ref<Material> icon = create_icon_material("gi_probe_icon", SpatialEditor::get_singleton()->get_icon("GizmoGIProbe", "EditorIcons"));
	Color gizmo_color_internal = gizmo_color;
	gizmo_color_internal.a = 0.1;
	Ref<Material> material_internal = create_material("gi_probe_internal_material", gizmo_color_internal);

	clear();

	Vector<Vector3> lines;
	Vector3 extents = probe->get_extents();

	static const int subdivs[GIProbe::SUBDIV_MAX] = { 64, 128, 256, 512 };

	Rect3 aabb = Rect3(-extents, extents * 2);
	int subdiv = subdivs[probe->get_subdiv()];
	float cell_size = aabb.get_longest_axis_size() / subdiv;

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	add_lines(lines, material);
	add_collision_segments(lines);

	lines.clear();

	for (int i = 1; i < subdiv; i++) {

		for (int j = 0; j < 3; j++) {

			if (cell_size * i > aabb.size[j]) {
				continue;
			}

			Vector2 dir;
			dir[j] = 1.0;
			Vector2 ta, tb;
			int j_n1 = (j + 1) % 3;
			int j_n2 = (j + 2) % 3;
			ta[j_n1] = 1.0;
			tb[j_n2] = 1.0;

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

	add_lines(lines, material_internal);

	Vector<Vector3> handles;

	for (int i = 0; i < 3; i++) {

		Vector3 ax;
		ax[i] = aabb.position[i] + aabb.size[i];
		handles.push_back(ax);
	}

	if (is_selected()) {

		gizmo_color.a = 0.1;
		Ref<Material> solid_material = create_material("gi_probe_solid_material", gizmo_color);
		add_solid_box(solid_material, aabb.get_size());
	}

	add_unscaled_billboard(icon, 0.05);
	add_handles(handles);
}
GIProbeGizmo::GIProbeGizmo(GIProbe *p_probe) {

	probe = p_probe;
	set_spatial_node(p_probe);
}

////////

void NavigationMeshSpatialGizmo::redraw() {

	Ref<Material> edge_material = create_material("navigation_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/navigation_edge"));
	Ref<Material> edge_material_disabled = create_material("navigation_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/navigation_edge_disabled"));
	Ref<Material> solid_material = create_material("navigation_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/navigation_solid"));
	Ref<Material> solid_material_disabled = create_material("navigation_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/navigation_solid_disabled"));

	clear();
	Ref<NavigationMesh> navmeshie = navmesh->get_navigation_mesh();
	if (navmeshie.is_null())
		return;

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

	if (faces.empty())
		return;

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
				if (ek.from < ek.to)
					SWAP(ek.from, ek.to);

				Map<_EdgeKey, bool>::Element *E = edge_map.find(ek);

				if (E) {

					E->get() = false;

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

	if (lines.size())
		add_lines(lines, navmesh->is_enabled() ? edge_material : edge_material_disabled);
	add_collision_triangles(tmesh);
	Ref<ArrayMesh> m = memnew(ArrayMesh);
	Array a;
	a.resize(Mesh::ARRAY_MAX);
	a[0] = tmeshfaces;
	m->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, a);
	m->surface_set_material(0, navmesh->is_enabled() ? solid_material : solid_material_disabled);
	add_mesh(m);
	add_collision_segments(lines);
}

NavigationMeshSpatialGizmo::NavigationMeshSpatialGizmo(NavigationMeshInstance *p_navmesh) {

	set_spatial_node(p_navmesh);
	navmesh = p_navmesh;
}

//////
///
///

void PinJointSpatialGizmo::redraw() {

	clear();
	Vector<Vector3> cursor_points;
	float cs = 0.25;
	cursor_points.push_back(Vector3(+cs, 0, 0));
	cursor_points.push_back(Vector3(-cs, 0, 0));
	cursor_points.push_back(Vector3(0, +cs, 0));
	cursor_points.push_back(Vector3(0, -cs, 0));
	cursor_points.push_back(Vector3(0, 0, +cs));
	cursor_points.push_back(Vector3(0, 0, -cs));
	add_collision_segments(cursor_points);

	Ref<Material> material = create_material("joint_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/joint"));

	add_lines(cursor_points, material);
}

PinJointSpatialGizmo::PinJointSpatialGizmo(PinJoint *p_p3d) {

	p3d = p_p3d;
	set_spatial_node(p3d);
}

////

void HingeJointSpatialGizmo::redraw() {

	clear();
	Vector<Vector3> cursor_points;
	float cs = 0.25;
	/*cursor_points.push_back(Vector3(+cs,0,0));
	cursor_points.push_back(Vector3(-cs,0,0));
	cursor_points.push_back(Vector3(0,+cs,0));
	cursor_points.push_back(Vector3(0,-cs,0));*/
	cursor_points.push_back(Vector3(0, 0, +cs * 2));
	cursor_points.push_back(Vector3(0, 0, -cs * 2));

	float ll = p3d->get_param(HingeJoint::PARAM_LIMIT_LOWER);
	float ul = p3d->get_param(HingeJoint::PARAM_LIMIT_UPPER);

	if (p3d->get_flag(HingeJoint::FLAG_USE_LIMIT) && ll < ul) {

		const int points = 32;

		for (int i = 0; i < points; i++) {

			float s = ll + i * (ul - ll) / points;
			float n = ll + (i + 1) * (ul - ll) / points;

			Vector3 from = Vector3(-Math::sin(s), Math::cos(s), 0) * cs;
			Vector3 to = Vector3(-Math::sin(n), Math::cos(n), 0) * cs;

			if (i == points - 1) {
				cursor_points.push_back(to);
				cursor_points.push_back(Vector3());
			}
			if (i == 0) {
				cursor_points.push_back(from);
				cursor_points.push_back(Vector3());
			}

			cursor_points.push_back(from);
			cursor_points.push_back(to);
		}

		cursor_points.push_back(Vector3(0, cs * 1.5, 0));
		cursor_points.push_back(Vector3());

	} else {

		const int points = 32;

		for (int i = 0; i < points; i++) {

			float s = ll + i * (Math_PI * 2.0) / points;
			float n = ll + (i + 1) * (Math_PI * 2.0) / points;

			Vector3 from = Vector3(-Math::sin(s), Math::cos(s), 0) * cs;
			Vector3 to = Vector3(-Math::sin(n), Math::cos(n), 0) * cs;

			cursor_points.push_back(from);
			cursor_points.push_back(to);
		}
	}

	Ref<Material> material = create_material("joint_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/joint"));

	add_collision_segments(cursor_points);
	add_lines(cursor_points, material);
}

HingeJointSpatialGizmo::HingeJointSpatialGizmo(HingeJoint *p_p3d) {

	p3d = p_p3d;
	set_spatial_node(p3d);
}

///////
///
////

void SliderJointSpatialGizmo::redraw() {

	clear();
	Vector<Vector3> cursor_points;
	float cs = 0.25;
	/*cursor_points.push_back(Vector3(+cs,0,0));
	cursor_points.push_back(Vector3(-cs,0,0));
	cursor_points.push_back(Vector3(0,+cs,0));
	cursor_points.push_back(Vector3(0,-cs,0));*/
	cursor_points.push_back(Vector3(0, 0, +cs * 2));
	cursor_points.push_back(Vector3(0, 0, -cs * 2));

	float ll = p3d->get_param(SliderJoint::PARAM_ANGULAR_LIMIT_LOWER);
	float ul = p3d->get_param(SliderJoint::PARAM_ANGULAR_LIMIT_UPPER);
	float lll = -p3d->get_param(SliderJoint::PARAM_LINEAR_LIMIT_LOWER);
	float lul = -p3d->get_param(SliderJoint::PARAM_LINEAR_LIMIT_UPPER);

	if (lll > lul) {

		cursor_points.push_back(Vector3(lul, 0, 0));
		cursor_points.push_back(Vector3(lll, 0, 0));

		cursor_points.push_back(Vector3(lul, -cs, -cs));
		cursor_points.push_back(Vector3(lul, -cs, cs));
		cursor_points.push_back(Vector3(lul, -cs, cs));
		cursor_points.push_back(Vector3(lul, cs, cs));
		cursor_points.push_back(Vector3(lul, cs, cs));
		cursor_points.push_back(Vector3(lul, cs, -cs));
		cursor_points.push_back(Vector3(lul, cs, -cs));
		cursor_points.push_back(Vector3(lul, -cs, -cs));

		cursor_points.push_back(Vector3(lll, -cs, -cs));
		cursor_points.push_back(Vector3(lll, -cs, cs));
		cursor_points.push_back(Vector3(lll, -cs, cs));
		cursor_points.push_back(Vector3(lll, cs, cs));
		cursor_points.push_back(Vector3(lll, cs, cs));
		cursor_points.push_back(Vector3(lll, cs, -cs));
		cursor_points.push_back(Vector3(lll, cs, -cs));
		cursor_points.push_back(Vector3(lll, -cs, -cs));

	} else {

		cursor_points.push_back(Vector3(+cs * 2, 0, 0));
		cursor_points.push_back(Vector3(-cs * 2, 0, 0));
	}

	if (ll < ul) {

		const int points = 32;

		for (int i = 0; i < points; i++) {

			float s = ll + i * (ul - ll) / points;
			float n = ll + (i + 1) * (ul - ll) / points;

			Vector3 from = Vector3(0, Math::cos(s), -Math::sin(s)) * cs;
			Vector3 to = Vector3(0, Math::cos(n), -Math::sin(n)) * cs;

			if (i == points - 1) {
				cursor_points.push_back(to);
				cursor_points.push_back(Vector3());
			}
			if (i == 0) {
				cursor_points.push_back(from);
				cursor_points.push_back(Vector3());
			}

			cursor_points.push_back(from);
			cursor_points.push_back(to);
		}

		cursor_points.push_back(Vector3(0, cs * 1.5, 0));
		cursor_points.push_back(Vector3());

	} else {

		const int points = 32;

		for (int i = 0; i < points; i++) {

			float s = ll + i * (Math_PI * 2.0) / points;
			float n = ll + (i + 1) * (Math_PI * 2.0) / points;

			Vector3 from = Vector3(0, Math::cos(s), -Math::sin(s)) * cs;
			Vector3 to = Vector3(0, Math::cos(n), -Math::sin(n)) * cs;

			cursor_points.push_back(from);
			cursor_points.push_back(to);
		}
	}

	Ref<Material> material = create_material("joint_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/joint"));

	add_collision_segments(cursor_points);
	add_lines(cursor_points, material);
}

SliderJointSpatialGizmo::SliderJointSpatialGizmo(SliderJoint *p_p3d) {

	p3d = p_p3d;
	set_spatial_node(p3d);
}

///////
///
////

void ConeTwistJointSpatialGizmo::redraw() {

	clear();
	Vector<Vector3> points;

	float r = 1.0;
	float w = r * Math::sin(p3d->get_param(ConeTwistJoint::PARAM_SWING_SPAN));
	float d = r * Math::cos(p3d->get_param(ConeTwistJoint::PARAM_SWING_SPAN));

	//swing
	for (int i = 0; i < 360; i += 10) {

		float ra = Math::deg2rad((float)i);
		float rb = Math::deg2rad((float)i + 10);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * w;

		/*points.push_back(Vector3(a.x,0,a.y));
		points.push_back(Vector3(b.x,0,b.y));
		points.push_back(Vector3(0,a.x,a.y));
		points.push_back(Vector3(0,b.x,b.y));*/
		points.push_back(Vector3(d, a.x, a.y));
		points.push_back(Vector3(d, b.x, b.y));

		if (i % 90 == 0) {

			points.push_back(Vector3(d, a.x, a.y));
			points.push_back(Vector3());
		}
	}

	points.push_back(Vector3());
	points.push_back(Vector3(1, 0, 0));

	//twist
	/*
	 */
	float ts = Math::rad2deg(p3d->get_param(ConeTwistJoint::PARAM_TWIST_SPAN));
	ts = MIN(ts, 720);

	for (int i = 0; i < int(ts); i += 5) {

		float ra = Math::deg2rad((float)i);
		float rb = Math::deg2rad((float)i + 5);
		float c = i / 720.0;
		float cn = (i + 5) / 720.0;
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * w * c;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * w * cn;

		/*points.push_back(Vector3(a.x,0,a.y));
		points.push_back(Vector3(b.x,0,b.y));
		points.push_back(Vector3(0,a.x,a.y));
		points.push_back(Vector3(0,b.x,b.y));*/

		points.push_back(Vector3(c, a.x, a.y));
		points.push_back(Vector3(cn, b.x, b.y));
	}

	Ref<Material> material = create_material("joint_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/joint"));
	add_collision_segments(points);
	add_lines(points, material);
}

ConeTwistJointSpatialGizmo::ConeTwistJointSpatialGizmo(ConeTwistJoint *p_p3d) {

	p3d = p_p3d;
	set_spatial_node(p3d);
}

////////
/// \brief SpatialEditorGizmos::singleton
///
///////
///
////

void Generic6DOFJointSpatialGizmo::redraw() {

	clear();
	Vector<Vector3> cursor_points;
	float cs = 0.25;

	for (int ax = 0; ax < 3; ax++) {
		/*cursor_points.push_back(Vector3(+cs,0,0));
		cursor_points.push_back(Vector3(-cs,0,0));
		cursor_points.push_back(Vector3(0,+cs,0));
		cursor_points.push_back(Vector3(0,-cs,0));
		cursor_points.push_back(Vector3(0,0,+cs*2));
		cursor_points.push_back(Vector3(0,0,-cs*2)); */

		float ll;
		float ul;
		float lll;
		float lul;

		int a1, a2, a3;
		bool enable_ang;
		bool enable_lin;

		switch (ax) {
			case 0:
				ll = p3d->get_param_x(Generic6DOFJoint::PARAM_ANGULAR_LOWER_LIMIT);
				ul = p3d->get_param_x(Generic6DOFJoint::PARAM_ANGULAR_UPPER_LIMIT);
				lll = -p3d->get_param_x(Generic6DOFJoint::PARAM_LINEAR_LOWER_LIMIT);
				lul = -p3d->get_param_x(Generic6DOFJoint::PARAM_LINEAR_UPPER_LIMIT);
				enable_ang = p3d->get_flag_x(Generic6DOFJoint::FLAG_ENABLE_ANGULAR_LIMIT);
				enable_lin = p3d->get_flag_x(Generic6DOFJoint::FLAG_ENABLE_LINEAR_LIMIT);
				a1 = 0;
				a2 = 1;
				a3 = 2;
				break;
			case 1:
				ll = p3d->get_param_y(Generic6DOFJoint::PARAM_ANGULAR_LOWER_LIMIT);
				ul = p3d->get_param_y(Generic6DOFJoint::PARAM_ANGULAR_UPPER_LIMIT);
				lll = -p3d->get_param_y(Generic6DOFJoint::PARAM_LINEAR_LOWER_LIMIT);
				lul = -p3d->get_param_y(Generic6DOFJoint::PARAM_LINEAR_UPPER_LIMIT);
				enable_ang = p3d->get_flag_y(Generic6DOFJoint::FLAG_ENABLE_ANGULAR_LIMIT);
				enable_lin = p3d->get_flag_y(Generic6DOFJoint::FLAG_ENABLE_LINEAR_LIMIT);
				a1 = 2;
				a2 = 0;
				a3 = 1;
				break;
			case 2:
				ll = p3d->get_param_z(Generic6DOFJoint::PARAM_ANGULAR_LOWER_LIMIT);
				ul = p3d->get_param_z(Generic6DOFJoint::PARAM_ANGULAR_UPPER_LIMIT);
				lll = -p3d->get_param_z(Generic6DOFJoint::PARAM_LINEAR_LOWER_LIMIT);
				lul = -p3d->get_param_z(Generic6DOFJoint::PARAM_LINEAR_UPPER_LIMIT);
				enable_ang = p3d->get_flag_z(Generic6DOFJoint::FLAG_ENABLE_ANGULAR_LIMIT);
				enable_lin = p3d->get_flag_z(Generic6DOFJoint::FLAG_ENABLE_LINEAR_LIMIT);

				a1 = 1;
				a2 = 2;
				a3 = 0;
				break;
		}

#define ADD_VTX(x, y, z)            \
	{                               \
		Vector3 v;                  \
		v[a1] = (x);                \
		v[a2] = (y);                \
		v[a3] = (z);                \
		cursor_points.push_back(v); \
	}

#define SET_VTX(what, x, y, z) \
	{                          \
		Vector3 v;             \
		v[a1] = (x);           \
		v[a2] = (y);           \
		v[a3] = (z);           \
		what = v;              \
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

		if (enable_ang && ll <= ul) {

			const int points = 32;

			for (int i = 0; i < points; i++) {

				float s = ll + i * (ul - ll) / points;
				float n = ll + (i + 1) * (ul - ll) / points;

				Vector3 from;
				SET_VTX(from, 0, Math::cos(s), -Math::sin(s));
				from *= cs;
				Vector3 to;
				SET_VTX(to, 0, Math::cos(n), -Math::sin(n));
				to *= cs;

				if (i == points - 1) {
					cursor_points.push_back(to);
					cursor_points.push_back(Vector3());
				}
				if (i == 0) {
					cursor_points.push_back(from);
					cursor_points.push_back(Vector3());
				}

				cursor_points.push_back(from);
				cursor_points.push_back(to);
			}

			ADD_VTX(0, cs * 1.5, 0);
			cursor_points.push_back(Vector3());

		} else {

			const int points = 32;

			for (int i = 0; i < points; i++) {

				float s = ll + i * (Math_PI * 2.0) / points;
				float n = ll + (i + 1) * (Math_PI * 2.0) / points;

				//Vector3 from=Vector3(0,Math::cos(s),-Math::sin(s) )*cs;
				//Vector3 to=Vector3( 0,Math::cos(n),-Math::sin(n) )*cs;

				Vector3 from;
				SET_VTX(from, 0, Math::cos(s), -Math::sin(s));
				from *= cs;
				Vector3 to;
				SET_VTX(to, 0, Math::cos(n), -Math::sin(n));
				to *= cs;

				cursor_points.push_back(from);
				cursor_points.push_back(to);
			}
		}
	}

#undef ADD_VTX
#undef SET_VTX

	Ref<Material> material = create_material("joint_material", EDITOR_GET("editors/3d_gizmos/gizmo_colors/joint"));
	add_collision_segments(cursor_points);
	add_lines(cursor_points, material);
}

Generic6DOFJointSpatialGizmo::Generic6DOFJointSpatialGizmo(Generic6DOFJoint *p_p3d) {

	p3d = p_p3d;
	set_spatial_node(p3d);
}

///////
///
////

SpatialEditorGizmos *SpatialEditorGizmos::singleton = NULL;

Ref<SpatialEditorGizmo> SpatialEditorGizmos::get_gizmo(Spatial *p_spatial) {

	if (Object::cast_to<Light>(p_spatial)) {

		Ref<LightSpatialGizmo> lsg = memnew(LightSpatialGizmo(Object::cast_to<Light>(p_spatial)));
		return lsg;
	}

	if (Object::cast_to<Camera>(p_spatial)) {

		Ref<CameraSpatialGizmo> lsg = memnew(CameraSpatialGizmo(Object::cast_to<Camera>(p_spatial)));
		return lsg;
	}

	if (Object::cast_to<Skeleton>(p_spatial)) {

		Ref<SkeletonSpatialGizmo> lsg = memnew(SkeletonSpatialGizmo(Object::cast_to<Skeleton>(p_spatial)));
		return lsg;
	}

	if (Object::cast_to<Position3D>(p_spatial)) {

		Ref<Position3DSpatialGizmo> lsg = memnew(Position3DSpatialGizmo(Object::cast_to<Position3D>(p_spatial)));
		return lsg;
	}

	if (Object::cast_to<MeshInstance>(p_spatial)) {

		Ref<MeshInstanceSpatialGizmo> misg = memnew(MeshInstanceSpatialGizmo(Object::cast_to<MeshInstance>(p_spatial)));
		return misg;
	}

	/*if (Object::cast_to<Room>(p_spatial)) {

		Ref<RoomSpatialGizmo> misg = memnew(RoomSpatialGizmo(Object::cast_to<Room>(p_spatial)));
		return misg;
	}*/

	if (Object::cast_to<NavigationMeshInstance>(p_spatial)) {

		Ref<NavigationMeshSpatialGizmo> misg = memnew(NavigationMeshSpatialGizmo(Object::cast_to<NavigationMeshInstance>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<RayCast>(p_spatial)) {

		Ref<RayCastSpatialGizmo> misg = memnew(RayCastSpatialGizmo(Object::cast_to<RayCast>(p_spatial)));
		return misg;
	}
	/*
	if (Object::cast_to<Portal>(p_spatial)) {

		Ref<PortalSpatialGizmo> misg = memnew(PortalSpatialGizmo(Object::cast_to<Portal>(p_spatial)));
		return misg;
	}
*/
	if (Object::cast_to<CollisionShape>(p_spatial)) {

		Ref<CollisionShapeSpatialGizmo> misg = memnew(CollisionShapeSpatialGizmo(Object::cast_to<CollisionShape>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<VisibilityNotifier>(p_spatial)) {

		Ref<VisibilityNotifierGizmo> misg = memnew(VisibilityNotifierGizmo(Object::cast_to<VisibilityNotifier>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<Particles>(p_spatial)) {

		Ref<ParticlesGizmo> misg = memnew(ParticlesGizmo(Object::cast_to<Particles>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<ReflectionProbe>(p_spatial)) {

		Ref<ReflectionProbeGizmo> misg = memnew(ReflectionProbeGizmo(Object::cast_to<ReflectionProbe>(p_spatial)));
		return misg;
	}
	if (Object::cast_to<GIProbe>(p_spatial)) {

		Ref<GIProbeGizmo> misg = memnew(GIProbeGizmo(Object::cast_to<GIProbe>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<VehicleWheel>(p_spatial)) {

		Ref<VehicleWheelSpatialGizmo> misg = memnew(VehicleWheelSpatialGizmo(Object::cast_to<VehicleWheel>(p_spatial)));
		return misg;
	}
	if (Object::cast_to<PinJoint>(p_spatial)) {

		Ref<PinJointSpatialGizmo> misg = memnew(PinJointSpatialGizmo(Object::cast_to<PinJoint>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<HingeJoint>(p_spatial)) {

		Ref<HingeJointSpatialGizmo> misg = memnew(HingeJointSpatialGizmo(Object::cast_to<HingeJoint>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<SliderJoint>(p_spatial)) {

		Ref<SliderJointSpatialGizmo> misg = memnew(SliderJointSpatialGizmo(Object::cast_to<SliderJoint>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<ConeTwistJoint>(p_spatial)) {

		Ref<ConeTwistJointSpatialGizmo> misg = memnew(ConeTwistJointSpatialGizmo(Object::cast_to<ConeTwistJoint>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<Generic6DOFJoint>(p_spatial)) {

		Ref<Generic6DOFJointSpatialGizmo> misg = memnew(Generic6DOFJointSpatialGizmo(Object::cast_to<Generic6DOFJoint>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<CollisionPolygon>(p_spatial)) {

		Ref<CollisionPolygonSpatialGizmo> misg = memnew(CollisionPolygonSpatialGizmo(Object::cast_to<CollisionPolygon>(p_spatial)));
		return misg;
	}

	if (Object::cast_to<AudioStreamPlayer3D>(p_spatial)) {

		Ref<AudioStreamPlayer3DSpatialGizmo> misg = memnew(AudioStreamPlayer3DSpatialGizmo(Object::cast_to<AudioStreamPlayer3D>(p_spatial)));
		return misg;
	}

	return Ref<SpatialEditorGizmo>();
}

SpatialEditorGizmos::SpatialEditorGizmos() {

	singleton = this;

	handle_material = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	handle_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	handle_material->set_on_top_of_alpha();
	handle_material->set_albedo(Color(0.8, 0.8, 0.8));
	handle_material_billboard = handle_material->duplicate();
	handle_material_billboard->set_billboard_mode(SpatialMaterial::BILLBOARD_ENABLED);

	handle2_material = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	handle2_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	handle2_material->set_flag(SpatialMaterial::FLAG_USE_POINT_SIZE, true);
	handle_t = SpatialEditor::get_singleton()->get_icon("Editor3DHandle", "EditorIcons");
	handle2_material->set_point_size(handle_t->get_width());
	handle2_material->set_texture(SpatialMaterial::TEXTURE_ALBEDO, handle_t);
	handle2_material->set_albedo(Color(1, 1, 1));
	handle2_material->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	handle2_material->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	handle2_material->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
	handle2_material->set_on_top_of_alpha();
	handle2_material_billboard = handle2_material->duplicate();
	handle2_material_billboard->set_billboard_mode(SpatialMaterial::BILLBOARD_ENABLED);
	handle2_material_billboard->set_billboard_mode(SpatialMaterial::BILLBOARD_ENABLED);
	handle2_material_billboard->set_on_top_of_alpha();

	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/light", Color(1, 1, 0.2));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/stream_player_3d", Color(0.4, 0.8, 1));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/camera", Color(0.8, 0.4, 0.8));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/skeleton", Color(1, 0.8, 0.4));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/visibility_notifier", Color(0.8, 0.5, 0.7));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/particles", Color(0.8, 0.7, 0.4));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/reflection_probe", Color(0.6, 1, 0.5));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/gi_probe", Color(0.5, 1, 0.6));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/shape", Color(0.5, 0.7, 1));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/joint", Color(0.5, 0.8, 1));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/navigation_edge", Color(0.5, 1, 1));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/navigation_edge_disabled", Color(0.7, 0.7, 0.7));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/navigation_solid", Color(0.5, 1, 1, 0.4));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/navigation_solid_disabled", Color(0.7, 0.7, 0.7, 0.4));
	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/instanced", Color(0.7, 0.7, 0.7, 0.5));

#if 0
	light_material = create_line_material(Color(1, 1, 0.2));
	light_material_omni = create_line_material(Color(1, 1, 0.2));
	light_material_omni->set_billboard_mode(SpatialMaterial::BILLBOARD_ENABLED);

	light_material_omni_icon = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	light_material_omni_icon->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	light_material_omni_icon->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	light_material_omni_icon->set_depth_draw_mode(SpatialMaterial::DEPTH_DRAW_DISABLED);
	light_material_omni_icon->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	light_material_omni_icon->set_albedo(Color(1, 1, 1, 0.9));
	light_material_omni_icon->set_texture(SpatialMaterial::TEXTURE_ALBEDO, SpatialEditor::get_singleton()->get_icon("GizmoLight", "EditorIcons"));
	light_material_omni_icon->set_flag(SpatialMaterial::FLAG_FIXED_SIZE, true);
	light_material_omni_icon->set_billboard_mode(SpatialMaterial::BILLBOARD_ENABLED);

	light_material_directional_icon = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	light_material_directional_icon->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	light_material_directional_icon->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	light_material_directional_icon->set_depth_draw_mode(SpatialMaterial::DEPTH_DRAW_DISABLED);
	light_material_directional_icon->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	light_material_directional_icon->set_albedo(Color(1, 1, 1, 0.9));
	light_material_directional_icon->set_texture(SpatialMaterial::TEXTURE_ALBEDO, SpatialEditor::get_singleton()->get_icon("GizmoDirectionalLight", "EditorIcons"));
	light_material_directional_icon->set_billboard_mode(SpatialMaterial::BILLBOARD_ENABLED);
	light_material_directional_icon->set_depth_scale(1);

	camera_material = create_line_material(Color(1.0, 0.5, 1.0));

	navmesh_edge_material = create_line_material(Color(0.1, 0.8, 1.0));
	navmesh_solid_material = create_solid_material(Color(0.1, 0.8, 1.0, 0.4));
	navmesh_edge_material->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, false);
	navmesh_edge_material->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, false);
	navmesh_solid_material->set_cull_mode(SpatialMaterial::CULL_DISABLED);

	navmesh_edge_material_disabled = create_line_material(Color(1.0, 0.8, 0.1));
	navmesh_solid_material_disabled = create_solid_material(Color(1.0, 0.8, 0.1, 0.4));
	navmesh_edge_material_disabled->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, false);
	navmesh_edge_material_disabled->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, false);
	navmesh_solid_material_disabled->set_cull_mode(SpatialMaterial::CULL_DISABLED);

	skeleton_material = create_line_material(Color(0.6, 1.0, 0.3));
	skeleton_material->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	skeleton_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	skeleton_material->set_on_top_of_alpha();
	skeleton_material->set_depth_draw_mode(SpatialMaterial::DEPTH_DRAW_DISABLED);

	//position 3D Shared mesh

	pos3d_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));
	{

		PoolVector<Vector3> cursor_points;
		PoolVector<Color> cursor_colors;
		float cs = 0.25;
		cursor_points.push_back(Vector3(+cs, 0, 0));
		cursor_points.push_back(Vector3(-cs, 0, 0));
		cursor_points.push_back(Vector3(0, +cs, 0));
		cursor_points.push_back(Vector3(0, -cs, 0));
		cursor_points.push_back(Vector3(0, 0, +cs));
		cursor_points.push_back(Vector3(0, 0, -cs));
		cursor_colors.push_back(Color(1, 0.5, 0.5, 0.7));
		cursor_colors.push_back(Color(1, 0.5, 0.5, 0.7));
		cursor_colors.push_back(Color(0.5, 1, 0.5, 0.7));
		cursor_colors.push_back(Color(0.5, 1, 0.5, 0.7));
		cursor_colors.push_back(Color(0.5, 0.5, 1, 0.7));
		cursor_colors.push_back(Color(0.5, 0.5, 1, 0.7));

		Ref<SpatialMaterial> mat = memnew(SpatialMaterial);
		mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
		mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
		mat->set_line_width(3);
		Array d;
		d.resize(VS::ARRAY_MAX);
		d[Mesh::ARRAY_VERTEX] = cursor_points;
		d[Mesh::ARRAY_COLOR] = cursor_colors;
		pos3d_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, d);
		pos3d_mesh->surface_set_material(0, mat);
	}

	listener_line_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));
	{

		PoolVector<Vector3> cursor_points;
		PoolVector<Color> cursor_colors;
		cursor_points.push_back(Vector3(0, 0, 0));
		cursor_points.push_back(Vector3(0, 0, -1.0));
		cursor_colors.push_back(Color(0.5, 0.5, 0.5, 0.7));
		cursor_colors.push_back(Color(0.5, 0.5, 0.5, 0.7));

		Ref<SpatialMaterial> mat = memnew(SpatialMaterial);
		mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
		mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
		mat->set_line_width(3);
		Array d;
		d.resize(VS::ARRAY_MAX);
		d[Mesh::ARRAY_VERTEX] = cursor_points;
		d[Mesh::ARRAY_COLOR] = cursor_colors;
		listener_line_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, d);
		listener_line_mesh->surface_set_material(0, mat);
	}

	room_material = create_line_material(Color(1.0, 0.6, 0.9));
	portal_material = create_line_material(Color(1.0, 0.8, 0.6));
	raycast_material = create_line_material(Color(1.0, 0.8, 0.6));
	car_wheel_material = create_line_material(Color(0.6, 0.8, 1.0));
	visibility_notifier_material = create_line_material(Color(1.0, 0.5, 1.0));
	particles_material = create_line_material(Color(1.0, 1.0, 0.5));
	reflection_probe_material = create_line_material(Color(0.5, 1.0, 0.7));
	reflection_probe_material_internal = create_line_material(Color(0.3, 0.8, 0.5, 0.15));
	gi_probe_material = create_line_material(Color(0.7, 1.0, 0.5));
	gi_probe_material_internal = create_line_material(Color(0.5, 0.8, 0.3, 0.1));
	joint_material = create_line_material(Color(0.6, 0.8, 1.0));

	stream_player_icon = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	stream_player_icon->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	stream_player_icon->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	stream_player_icon->set_depth_draw_mode(SpatialMaterial::DEPTH_DRAW_DISABLED);
	stream_player_icon->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	stream_player_icon->set_albedo(Color(1, 1, 1, 0.9));
	stream_player_icon->set_texture(SpatialMaterial::TEXTURE_ALBEDO, SpatialEditor::get_singleton()->get_icon("GizmoSpatialStreamPlayer", "EditorIcons"));

	visibility_notifier_icon = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	visibility_notifier_icon->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	visibility_notifier_icon->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	visibility_notifier_icon->set_depth_draw_mode(SpatialMaterial::DEPTH_DRAW_DISABLED);
	visibility_notifier_icon->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	visibility_notifier_icon->set_albedo(Color(1, 1, 1, 0.9));
	visibility_notifier_icon->set_texture(SpatialMaterial::TEXTURE_ALBEDO, SpatialEditor::get_singleton()->get_icon("Visible", "EditorIcons"));

	listener_icon = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	listener_icon->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	listener_icon->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	listener_icon->set_depth_draw_mode(SpatialMaterial::DEPTH_DRAW_DISABLED);
	listener_icon->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	listener_icon->set_albedo(Color(1, 1, 1, 0.9));
	listener_icon->set_texture(SpatialMaterial::TEXTURE_ALBEDO, SpatialEditor::get_singleton()->get_icon("GizmoListener", "EditorIcons"));

	{

		PoolVector<Vector3> vertices;

#undef ADD_VTX
#define ADD_VTX(m_idx) \
	vertices.push_back(face_points[m_idx]);

		for (int i = 0; i < 6; i++) {

			Vector3 face_points[4];

			for (int j = 0; j < 4; j++) {

				float v[3];
				v[0] = 1.0;
				v[1] = 1 - 2 * ((j >> 1) & 1);
				v[2] = v[1] * (1 - 2 * (j & 1));

				for (int k = 0; k < 3; k++) {

					if (i < 3)
						face_points[j][(i + k) % 3] = v[k] * (i >= 3 ? -1 : 1);
					else
						face_points[3 - j][(i + k) % 3] = v[k] * (i >= 3 ? -1 : 1);
				}
			}
			//tri 1
			ADD_VTX(0);
			ADD_VTX(1);
			ADD_VTX(2);
			//tri 2
			ADD_VTX(2);
			ADD_VTX(3);
			ADD_VTX(0);
		}

		test_cube_tm = Ref<TriangleMesh>(memnew(TriangleMesh));
		test_cube_tm->create(vertices);
	}

	shape_material = create_line_material(Color(0.2, 1, 1.0));
#endif

	pos3d_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));
	{

		PoolVector<Vector3> cursor_points;
		PoolVector<Color> cursor_colors;
		float cs = 0.25;
		cursor_points.push_back(Vector3(+cs, 0, 0));
		cursor_points.push_back(Vector3(-cs, 0, 0));
		cursor_points.push_back(Vector3(0, +cs, 0));
		cursor_points.push_back(Vector3(0, -cs, 0));
		cursor_points.push_back(Vector3(0, 0, +cs));
		cursor_points.push_back(Vector3(0, 0, -cs));
		cursor_colors.push_back(Color(1, 0.5, 0.5, 0.7));
		cursor_colors.push_back(Color(1, 0.5, 0.5, 0.7));
		cursor_colors.push_back(Color(0.5, 1, 0.5, 0.7));
		cursor_colors.push_back(Color(0.5, 1, 0.5, 0.7));
		cursor_colors.push_back(Color(0.5, 0.5, 1, 0.7));
		cursor_colors.push_back(Color(0.5, 0.5, 1, 0.7));

		Ref<SpatialMaterial> mat = memnew(SpatialMaterial);
		mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
		mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
		mat->set_line_width(3);
		Array d;
		d.resize(VS::ARRAY_MAX);
		d[Mesh::ARRAY_VERTEX] = cursor_points;
		d[Mesh::ARRAY_COLOR] = cursor_colors;
		pos3d_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, d);
		pos3d_mesh->surface_set_material(0, mat);
	}

	listener_line_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));
	{

		PoolVector<Vector3> cursor_points;
		PoolVector<Color> cursor_colors;
		cursor_points.push_back(Vector3(0, 0, 0));
		cursor_points.push_back(Vector3(0, 0, -1.0));
		cursor_colors.push_back(Color(0.5, 0.5, 0.5, 0.7));
		cursor_colors.push_back(Color(0.5, 0.5, 0.5, 0.7));

		Ref<SpatialMaterial> mat = memnew(SpatialMaterial);
		mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
		mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
		mat->set_line_width(3);
		Array d;
		d.resize(VS::ARRAY_MAX);
		d[Mesh::ARRAY_VERTEX] = cursor_points;
		d[Mesh::ARRAY_COLOR] = cursor_colors;
		listener_line_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, d);
		listener_line_mesh->surface_set_material(0, mat);
	}
}
