/**************************************************************************/
/*  line_3d.cpp                                                           */
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

#include "line_3d.h"

#include "scene/3d/camera_3d.h"
#include "scene/resources/3d/shape_3d.h"

void Line3D::init_shaders() {
	billboard_additive_shader.instantiate();
	billboard_additive_shader->set_code(R"(
shader_type spatial;
render_mode
	blend_add,
	depth_draw_never,
	unshaded,
	skip_vertex_transform,
	cull_disabled;

void vertex() {
	vec3 p = (MODELVIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec3 t = (MODELVIEW_MATRIX * vec4(NORMAL, 0.0)).xyz;
	VERTEX = p + UV.y * normalize(cross(p, t));
	NORMAL = (VIEW_MATRIX * vec4(0, 1, 0, 0)).xyz;
	UV.y = (sign(UV.y) + 1.0) / 2.0;
}

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
)");
	billboard_shader.instantiate();
	billboard_shader->set_code(R"(
shader_type spatial;
render_mode
	blend_mix,
	depth_draw_never,
	unshaded,
	skip_vertex_transform,
	cull_disabled;

void vertex() {
	vec3 p = (MODELVIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec3 t = (MODELVIEW_MATRIX * vec4(NORMAL, 0.0)).xyz;
	VERTEX = p + UV.y * normalize(cross(p, t));
	NORMAL = (VIEW_MATRIX * vec4(0, 1, 0, 0)).xyz;
	UV.y = (sign(UV.y) + 1.0) / 2.0;
}

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
)");
	local_additive_shader.instantiate();
	local_additive_shader->set_code(R"(
shader_type spatial;
render_mode
	blend_add,
	depth_draw_never,
	unshaded,
	cull_disabled;

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
)");
	local_shader.instantiate();
	local_shader->set_code(R"(
shader_type spatial;
render_mode
	blend_mix,
	depth_draw_never,
	unshaded,
	skip_vertex_transform,
	cull_disabled;

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
)");

	billboard_additive_material.instantiate();
	billboard_additive_material->set_shader(billboard_additive_shader);
	billboard_material.instantiate();
	billboard_material->set_shader(billboard_shader);
	local_additive_material.instantiate();
	local_additive_material->set_shader(local_additive_shader);
	local_material.instantiate();
	local_material->set_shader(local_shader);
}

void Line3D::finish_shaders() {
	billboard_additive_material.unref();
	billboard_material.unref();
	local_additive_material.unref();
	local_material.unref();
	billboard_shader.unref();
	billboard_additive_shader.unref();
	local_shader.unref();
	local_additive_shader.unref();
}

void Line3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			switch (line_mode) {
				case Line3D::LINE_MODE_TRAIL: {
					_process_trail();
				} break;
			}
			if (_needs_rebuilding) {
				_do_rebuild();
			}
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (global_space) {
				if (line_mode == Line3D::LINE_MODE_BEAM) {
					_process_beam();
				}
				rebuild();
			}
		}
	}
}

void Line3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_global_space", "global_space"), &Line3D::set_global_space);
	ClassDB::bind_method(D_METHOD("get_global_space"), &Line3D::get_global_space);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &Line3D::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &Line3D::get_width);
	ClassDB::bind_method(D_METHOD("set_width_curve", "curve"), &Line3D::set_width_curve);
	ClassDB::bind_method(D_METHOD("get_width_curve"), &Line3D::get_width_curve);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &Line3D::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &Line3D::get_color);
	ClassDB::bind_method(D_METHOD("set_color_gradient", "gradient"), &Line3D::set_color_gradient);
	ClassDB::bind_method(D_METHOD("get_color_gradient"), &Line3D::get_color_gradient);

	ClassDB::bind_method(D_METHOD("set_material_mode", "material_mode"), &Line3D::set_material_mode);
	ClassDB::bind_method(D_METHOD("get_material_mode"), &Line3D::get_material_mode);
	ClassDB::bind_method(D_METHOD("set_material", "material"), &Line3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &Line3D::get_material);

	ClassDB::bind_method(D_METHOD("set_mesh_alignment", "slignment"), &Line3D::set_mesh_alignment);
	ClassDB::bind_method(D_METHOD("get_alignment"), &Line3D::get_alignment);

	ClassDB::bind_method(D_METHOD("set_tiling_mode", "tiling_mode"), &Line3D::set_tiling_mode);
	ClassDB::bind_method(D_METHOD("get_tiling_mode"), &Line3D::get_tiling_mode);
	ClassDB::bind_method(D_METHOD("set_tiling_multiplier", "tiling_multiplier"), &Line3D::set_tiling_multiplier);
	ClassDB::bind_method(D_METHOD("get_tiling_multiplier"), &Line3D::get_tiling_multiplier);
	ClassDB::bind_method(D_METHOD("set_tiling_offset", "offset"), &Line3D::set_tiling_offset);
	ClassDB::bind_method(D_METHOD("get_tiling_offset"), &Line3D::get_tiling_offset);

	ClassDB::bind_method(D_METHOD("set_points", "points"), &Line3D::set_points);
	ClassDB::bind_method(D_METHOD("get_points"), &Line3D::get_points);

	ClassDB::bind_method(D_METHOD("set_normals", "normals"), &Line3D::set_normals);
	ClassDB::bind_method(D_METHOD("get_normals"), &Line3D::get_normals);

	ClassDB::bind_method(D_METHOD("set_line_mode", "line_mode"), &Line3D::set_line_mode);
	ClassDB::bind_method(D_METHOD("get_line_mode"), &Line3D::get_line_mode);

	ClassDB::bind_method(D_METHOD("rebuild"), &Line3D::rebuild);
	ClassDB::bind_method(D_METHOD("clear"), &Line3D::clear);

	// Beam and Trail
	ClassDB::bind_method(D_METHOD("set_max_section_length", "max_section_length"), &Line3D::set_max_section_length);
	ClassDB::bind_method(D_METHOD("get_max_section_length"), &Line3D::get_max_section_length);

	// Beam
	ClassDB::bind_method(D_METHOD("set_target", "target"), &Line3D::set_target);
	ClassDB::bind_method(D_METHOD("get_target"), &Line3D::get_target);

	// Trail
	ClassDB::bind_method(D_METHOD("set_emitting", "emitting"), &Line3D::set_emitting);
	ClassDB::bind_method(D_METHOD("get_emitting"), &Line3D::get_emitting);

	ClassDB::bind_method(D_METHOD("set_lifetime", "lifetime"), &Line3D::set_lifetime);
	ClassDB::bind_method(D_METHOD("get_lifetime"), &Line3D::get_lifetime);

	ClassDB::bind_method(D_METHOD("set_max_length", "max_length"), &Line3D::set_max_length);
	ClassDB::bind_method(D_METHOD("get_max_length"), &Line3D::get_max_length);

	ClassDB::bind_method(D_METHOD("get_current_length"), &Line3D::get_current_length);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "line_mode", PROPERTY_HINT_ENUM, "Trail, Beam, Manual"), "set_line_mode", "get_line_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "global_space"), "set_global_space", "get_global_space");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "width_curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_width_curve", "get_width_curve");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color", PROPERTY_HINT_NONE), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "color_gradient", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_color_gradient", "get_color_gradient");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "material_mode", PROPERTY_HINT_ENUM, "Default, Default Additive, Manual"), "set_material_mode", "get_material_mode");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "ShaderMaterial"), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_section_length"), "set_max_section_length", "get_max_section_length");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh_alignment", PROPERTY_HINT_ENUM, "Local, Billboard"), "set_mesh_alignment", "get_alignment");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "tiling_mode", PROPERTY_HINT_ENUM, "Unit, Length"), "set_tiling_mode", "get_tiling_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tiling_multiplier"), "set_tiling_multiplier", "get_tiling_multiplier");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tiling_offset"), "set_tiling_offset", "get_tiling_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "emitting"), "set_emitting", "get_emitting");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lifetime"), "set_lifetime", "get_lifetime");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_length"), "set_max_length", "get_max_length");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target"), "set_target", "get_target");

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "points"), "set_points", "get_points");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "normals"), "set_normals", "get_normals");

	BIND_ENUM_CONSTANT(LINE_MODE_TRAIL);
	BIND_ENUM_CONSTANT(LINE_MODE_BEAM);
	BIND_ENUM_CONSTANT(LINE_MODE_MANUAL);
	BIND_ENUM_CONSTANT(LINE_MODE_MAX);

	BIND_ENUM_CONSTANT(MESH_ALIGNMENT_LOCAL);
	BIND_ENUM_CONSTANT(MESH_ALIGNMENT_BILLBOARD);
	BIND_ENUM_CONSTANT(MESH_ALIGNMENT_MAX);

	BIND_ENUM_CONSTANT(TLING_UNIT);
	BIND_ENUM_CONSTANT(TILING_LENGTH);
	BIND_ENUM_CONSTANT(TILING_MAX);

	BIND_ENUM_CONSTANT(LIMIT_MODE_LIFETIME);
	BIND_ENUM_CONSTANT(LIMIT_MODE_MAX_LENGTH);
	BIND_ENUM_CONSTANT(LIMIT_MODE_MAX);
}

void Line3D::set_global_space(bool p_global_space) {
	if (global_space != p_global_space) {
		_needs_rebuilding = true;
	}
	global_space = p_global_space;
}
bool Line3D::get_global_space() const {
	return global_space;
}

void Line3D::set_width(float p_width) {
	width = p_width;
}
float Line3D::get_width() const {
	return width;
}

void Line3D::set_width_curve(Ref<Curve> p_width_curve) {
	if (p_width_curve == width_curve) {
		return;
	}
	if (width_curve.is_valid()) {
		width_curve->disconnect(CoreStringName(changed), callable_mp((Line3D *)this, &Line3D::rebuild));
	}
	width_curve = p_width_curve;
	if (width_curve.is_valid()) {
		width_curve->connect(CoreStringName(changed), callable_mp((Line3D *)this, &Line3D::rebuild));
	}
	_needs_rebuilding = true;
}

Ref<Curve> Line3D::get_width_curve() {
	return width_curve;
}

void Line3D::set_color(Color p_color) {
	if (p_color != color) {
		_needs_rebuilding = true;
	}
	color = p_color;
}
Color Line3D::get_color() const {
	return color;
}

void Line3D::set_color_gradient(Ref<Gradient> p_color_gradient) {
	if (p_color_gradient == color_gradient) {
		return;
	}
	if (color_gradient.is_valid()) {
		color_gradient->disconnect(CoreStringName(changed), callable_mp((Line3D *)this, &Line3D::rebuild));
	}
	color_gradient = p_color_gradient;
	if (color_gradient.is_valid()) {
		color_gradient->connect(CoreStringName(changed), callable_mp((Line3D *)this, &Line3D::rebuild));
	}
	_needs_rebuilding = true;
}

Ref<Gradient> Line3D::get_color_gradient() {
	return color_gradient;
}

void Line3D::set_material_mode(Line3D::MaterialMode p_material_mode) {
	material_mode = p_material_mode;
	notify_property_list_changed();
	_ensure_material();
}
Line3D::MaterialMode Line3D::get_material_mode() const {
	return material_mode;
}

void Line3D::set_material(Ref<ShaderMaterial> p_material) {
	if (material_mode != Line3D::MaterialMode::MATERIAL_MODE_CUSTOM) {
		return;
	}
	material = p_material;
	_ensure_material();
}
Ref<ShaderMaterial> Line3D::get_material() {
	return material;
}

void Line3D::set_mesh_alignment(MeshAlignment p_alignment) {
	if (p_alignment != alignment) {
		_needs_rebuilding = true;
	}
	alignment = p_alignment;
}

Line3D::MeshAlignment Line3D::get_alignment() const {
	return alignment;
}

void Line3D::set_tiling_mode(Tiling p_tiling_mode) {
	if (p_tiling_mode != tiling_mode) {
		_needs_rebuilding = true;
	}
	tiling_mode = p_tiling_mode;
	notify_property_list_changed();
}
Line3D::Tiling Line3D::get_tiling_mode() const {
	return tiling_mode;
}

void Line3D::set_tiling_multiplier(float p_tiling_multiplier) {
	if (p_tiling_multiplier != tiling_multiplier) {
		_needs_rebuilding = true;
	}
	tiling_multiplier = p_tiling_multiplier;
}
float Line3D::get_tiling_multiplier() const {
	return tiling_multiplier;
}

void Line3D::set_tiling_offset(float p_tiling_offset) {
	if (p_tiling_offset != p_tiling_offset) {
		_needs_rebuilding = true;
	}
	tiling_offset = p_tiling_offset;
}
float Line3D::get_tiling_offset() const {
	return tiling_offset;
}

void Line3D::set_points(PackedVector3Array p_points) {
	if (line_mode != LineMode::LINE_MODE_MANUAL) {
		if (Engine::get_singleton()->is_editor_hint()) {
			print_error("Setting points on line3d not in manual mode.");
		}
	}
	points = p_points;
	_needs_rebuilding = true;
}

PackedVector3Array Line3D::get_points() const {
	return points;
}

void Line3D::set_normals(PackedVector3Array p_normals) {
	if (line_mode != LineMode::LINE_MODE_MANUAL) {
		if (Engine::get_singleton()->is_editor_hint()) {
			print_error("Setting normals on line3d not in manual mode.");
		}
	}
	normals = p_normals;
	_needs_rebuilding = true;
}

PackedVector3Array Line3D::get_normals() const {
	return normals;
}

void Line3D::set_line_mode(Line3D::LineMode p_line_mode) {
	if (p_line_mode == line_mode) {
		return;
	}
	set_process_internal(p_line_mode != Line3D::LineMode::LINE_MODE_MANUAL);
	line_mode = p_line_mode;
	notify_property_list_changed();
}

Line3D::LineMode Line3D::get_line_mode() const {
	return line_mode;
}

void Line3D::rebuild() {
	_needs_rebuilding = true;
	_do_rebuild();
}

void Line3D::clear() {
	points = PackedVector3Array();
	normals = PackedVector3Array();
	rebuild();
}

// Beam

void Line3D::set_target(Vector3 p_target) {
	target = p_target;
	if (line_mode == Line3D::LineMode::LINE_MODE_BEAM) {
		_process_beam();
		_needs_rebuilding = true;
	}
}
Vector3 Line3D::get_target() const {
	return target;
}

// Trail3D
void Line3D::set_emitting(bool p_emitting) {
	emitting = p_emitting;
}
bool Line3D::get_emitting() const {
	return emitting;
}

void Line3D::set_max_section_length(real_t p_max_section_length) {
	if (p_max_section_length != max_section_length) {
		_needs_rebuilding = true;
	}
	max_section_length = p_max_section_length;
}
real_t Line3D::get_max_section_length() const {
	return max_section_length;
}

void Line3D::set_lifetime(real_t p_lifetime) {
	lifetime = p_lifetime;
}
real_t Line3D::get_lifetime() const {
	return lifetime;
}

void Line3D::set_max_length(real_t p_max_length) {
	max_length = p_max_length;
}
real_t Line3D::get_max_length() const {
	return max_length;
}

real_t Line3D::_calc_current_length() {
	real_t length = 0.0;
	for (int i = 0; i < points.size() - 1; i++) {
		length += points[i].distance_to(points[i + 1]);
	}
	return length;
}

real_t Line3D::get_current_length() {
	return _calc_current_length();
}

void Line3D::_do_rebuild() {
	if (!is_inside_tree() || !is_ready() || !_needs_rebuilding) {
		return;
	}
	if (mesh == nullptr || (Ref<ArrayMesh>)mesh == nullptr) {
		mesh = new ArrayMesh();
	}
	Ref<ArrayMesh> am = (Ref<ArrayMesh>)mesh;
	am->clear_surfaces();

	int points_count = points.size();

	if (points_count < 2) {
		return;
	}

	_needs_rebuilding = false;

	PackedVector3Array mesh_vertices;
	PackedVector3Array mesh_normals;
	PackedColorArray mesh_colors;
	PackedVector2Array mesh_uvs;
	PackedInt32Array mesh_indices;

	mesh_vertices.resize(points_count * 3);
	mesh_normals.resize(points_count * 3);
	mesh_colors.resize(points_count * 3);
	mesh_uvs.resize(points_count * 3);
	mesh_indices.resize((points_count - 1) * 12);

	Vector3 *_vertices = mesh_vertices.ptrw();
	Vector3 *_normals = mesh_normals.ptrw();
	Color *_colors = mesh_colors.ptrw();
	Vector2 *_uvs = mesh_uvs.ptrw();
	int *_indices = mesh_indices.ptrw();

	for (int i = 0; i < points_count - 1; i++) {
		int j = i * 12;
		int k = i * 3;
		_indices[j] = k;
		_indices[j + 1] = k + 3;
		_indices[j + 2] = k + 1;
		_indices[j + 3] = k + 1;
		_indices[j + 4] = k + 3;
		_indices[j + 5] = k + 4;
		_indices[j + 6] = k + 1;
		_indices[j + 7] = k + 4;
		_indices[j + 8] = k + 2;
		_indices[j + 9] = k + 2;
		_indices[j + 10] = k + 4;
		_indices[j + 11] = k + 5;
	}

	Transform3D inv_global_tf = get_global_transform().inverse();

	real_t length = _calc_current_length();

	real_t dist = 0.0;

	for (int i = 0; i < points_count; i++) {
		int j0 = i * 3;
		int j1 = j0 + 1;
		int j2 = j0 + 2;

		Vector3 p = points[i];

		if (i > 0) {
			dist += points[i - 1].distance_to(p);
		}

		real_t ratio = length > 0.0 ? dist / length : 0.0;
		real_t length_uv = 0;

		switch (tiling_mode) {
			case Line3D::Tiling::TILING_LENGTH: {
				length_uv = dist * tiling_multiplier;
			} break;
			case Line3D::Tiling::TLING_UNIT: {
				length_uv = ratio * tiling_multiplier;
			} break;
			default: {
			}
		}

		real_t half_width = width * 0.5;

		if (width_curve.is_valid()) {
			half_width *= width_curve->sample(ratio);
		}

		Vector3 tangent;

		if (i == 0) {
			tangent = (points[i + 1] - p).normalized();
		} else if (i == points_count - 1) {
			tangent = (p - points[i - 1]).normalized();
		} else {
			tangent = (p - points[i - 1]).lerp(points[i + 1] - p, 0.5).normalized();
		}
		if (global_space) {
			p = inv_global_tf.xform(p);
			tangent = inv_global_tf.basis.xform(tangent);
		}

		if (alignment == Line3D::MeshAlignment::MESH_ALIGNMENT_BILLBOARD) {
			_vertices[j0] = p;
			_vertices[j1] = p;
			_vertices[j2] = p;

			_normals[j0] = tangent;
			_normals[j1] = tangent;
			_normals[j2] = tangent;

			_uvs[j0] = Vector2(length_uv, -half_width);
			_uvs[j1] = Vector2(length_uv, 0);
			_uvs[j2] = Vector2(length_uv, half_width);
		} else {
			ERR_FAIL_COND_MSG(normals.size() != points.size(), "Non-billboarded Line3D has a different number of points and normal. Please ensure that normals and points are the same amount");
			Vector3 normal = normals[i];
			if (global_space) {
				normal = inv_global_tf.basis.xform(normal);
			}

			Vector3 curve_binormal = tangent.cross(normal);
			_vertices[j0] = p + half_width * curve_binormal;
			_vertices[j1] = p;
			_vertices[j2] = p - half_width * curve_binormal;

			_normals[j0] = normal;
			_normals[j1] = normal;
			_normals[j2] = normal;

			_uvs[j0] = Vector2(length_uv, 0);
			_uvs[j1] = Vector2(length_uv, 0.5);
			_uvs[j2] = Vector2(length_uv, 1);
		}
		Color c = color.srgb_to_linear();
		if (color_gradient.is_valid()) {
			c *= color_gradient->get_color_at_offset(ratio).srgb_to_linear();
		}

		_colors[j0] = c;
		_colors[j1] = c;
		_colors[j2] = c;
	}

	Array arrays;
	arrays.resize(RS::ARRAY_MAX);
	arrays[RS::ARRAY_VERTEX] = mesh_vertices;
	arrays[RS::ARRAY_NORMAL] = mesh_normals;
	arrays[RS::ARRAY_TEX_UV] = mesh_uvs;
	arrays[RS::ARRAY_COLOR] = mesh_colors;
	arrays[RS::ARRAY_INDEX] = mesh_indices;

	am->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);

	_ensure_material();
}

void Line3D::_process_beam() {
	int subdivisions = 1;
	if (global_space) {
		Vector3 origin = get_global_transform().origin;
		subdivisions = (int)(origin.distance_to(target) / max_section_length);
		points.resize(subdivisions + 1);
		normals.resize(subdivisions + 1);
		Vector3 *_points = points.ptrw();
		Vector3 *_normals = normals.ptrw();
		Vector3 beam_dir = target - origin;
		real_t total_beam_length = beam_dir.length();
		real_t remaining_len = 0.0;
		beam_dir.normalize();
		for (int i = 0; remaining_len < total_beam_length; i++) {
			_points[i] = origin + beam_dir * remaining_len;
			remaining_len += max_section_length;
			_normals[i] = Vector3(0.0, 1.0, 0.0);
		}
		_points[subdivisions] = target;
		_normals[subdivisions] = Vector3(0.0, 1.0, 0.0);
	} else {
		subdivisions = (int)(target.length() / max_section_length);
		points.resize(subdivisions + 1);
		normals.resize(subdivisions + 1);
		Vector3 *_points = points.ptrw();
		Vector3 *_normals = normals.ptrw();
		Vector3 beam_dir = target;
		real_t total_beam_length = beam_dir.length();
		real_t remaining_len = 0.0;
		beam_dir.normalize();
		for (int i = 0; remaining_len < total_beam_length; i++) {
			_points[i] = beam_dir * remaining_len;
			remaining_len += max_section_length;
			_normals[i] = Vector3(0.0, 1.0, 0.0);
		}
		_points[subdivisions] = target;
		_normals[subdivisions] = Vector3(0.0, 1.0, 0.0);
	}
	_needs_rebuilding = true;
}

void Line3D::_process_trail() {
}

void Line3D::_ensure_material() {
	switch (material_mode) {
		case Line3D::MaterialMode::MATERIAL_MODE_ADD: {
			if (alignment == MeshAlignment::MESH_ALIGNMENT_BILLBOARD) {
				mesh->surface_set_material(0, billboard_additive_material);
			} else {
				mesh->surface_set_material(0, local_additive_material);
			}

		} break;
		case Line3D::MaterialMode::MATERIAL_MODE_MIX: {
			if (alignment == MeshAlignment::MESH_ALIGNMENT_BILLBOARD) {
				mesh->surface_set_material(0, billboard_material);
			} else {
				mesh->surface_set_material(0, local_material);
			}
		} break;
		case Line3D::MaterialMode::MATERIAL_MODE_CUSTOM: {
			mesh->surface_set_material(0, material);
		} break;
	}
}

Line3D::Line3D() {
	line_mode = Line3D::LineMode::LINE_MODE_TRAIL;
	width = 1.0;
	width_curve = nullptr;
	color = Color(1.0, 1.0, 1.0, 0.5);
	color_gradient = nullptr;
	alignment = Line3D::MeshAlignment::MESH_ALIGNMENT_BILLBOARD;
	tiling_mode = Line3D::Tiling::TILING_LENGTH;
	tiling_multiplier = 1.0;
	tiling_offset = 0.0;
	points = PackedVector3Array();
	normals = PackedVector3Array();
	_needs_rebuilding = false;
	max_section_length = 0.2;
	emitting = true;
	lifetime = 0.5;
	max_length = 2.0;
	target = Vector3();
	mesh = nullptr;
	set_process_internal(true);
}

Line3D::~Line3D() {}
