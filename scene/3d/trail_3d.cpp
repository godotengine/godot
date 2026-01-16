/**************************************************************************/
/*  trail_3d.cpp                                                          */
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

#include "trail_3d.h"

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "servers/rendering/rendering_server.h"

// Points calculation, line construction and shaders are taken
// and adapted from CozyCubeGames.

/**************************************************************************/
/* Copyright (c) 2024 CozyCubeGames                                       */
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

void Trail3D::init_shaders() {
	billboard_additive_shader.instantiate();
	billboard_additive_shader->set_code(R"(
shader_type spatial;
render_mode blend_add, depth_draw_never, unshaded, skip_vertex_transform, cull_disabled;

void vertex() {
	if (length(NORMAL) > 0.0) {
		vec3 p = (MODELVIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
		vec3 t = (MODELVIEW_MATRIX * vec4(NORMAL, 0.0)).xyz;
		VERTEX = p + UV.y * normalize(cross(p, t));
		NORMAL = (VIEW_MATRIX * vec4(0, 1, 0, 0)).xyz;
		UV.y = (sign(UV.y) + 1.0) / 2.0;
	} else {
		VERTEX = vec3(0.);
		NORMAL = vec3(0.);
	}
}

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
)");
	billboard_shader.instantiate();
	billboard_shader->set_code(R"(
shader_type spatial;
render_mode blend_mix, depth_draw_never, unshaded, skip_vertex_transform, cull_disabled;

void vertex() {
	if (length(NORMAL) > 0.0) {
		vec3 p = (MODELVIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
		vec3 t = (MODELVIEW_MATRIX * vec4(NORMAL, 0.0)).xyz;
		VERTEX = p + UV.y * normalize(cross(p, t));
		NORMAL = (VIEW_MATRIX * vec4(0, 1, 0, 0)).xyz;
		UV.y = (sign(UV.y) + 1.0) / 2.0;
	} else {
		VERTEX = vec3(0.);
		NORMAL = vec3(0.);
	}
}

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
)");
	local_additive_shader.instantiate();
	local_additive_shader->set_code(R"(
shader_type spatial;
render_mode blend_add, depth_draw_never, unshaded, cull_disabled;

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
}
)");
	local_shader.instantiate();
	local_shader->set_code(R"(
shader_type spatial;
render_mode blend_mix, depth_draw_never, unshaded, cull_disabled;

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

void Trail3D::finish_shaders() {
	billboard_additive_material.unref();
	billboard_material.unref();
	local_additive_material.unref();
	local_material.unref();
	billboard_shader.unref();
	billboard_additive_shader.unref();
	local_shader.unref();
	local_additive_shader.unref();
}

void Trail3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			_process_trail(get_process_delta_time());
			if (_needs_rebuilding) {
				_do_rebuild();
			}
		} break;
		case NOTIFICATION_READY: {
			clear();
		} break;
	}
}

void Trail3D::set_width(float p_width) {
	width = p_width;
	_needs_rebuilding = true;
}

float Trail3D::get_width() const {
	return width;
}

void Trail3D::set_emitting(bool p_emitting) {
	if (emitting != p_emitting && p_emitting) {
		clear();
	}
	emitting = p_emitting;
}

bool Trail3D::is_emitting() const {
	return emitting;
}

void Trail3D::set_width_curve(Ref<Curve> p_width_curve) {
	if (p_width_curve == width_curve) {
		return;
	}
	if (width_curve.is_valid()) {
		width_curve->disconnect_changed(callable_mp(this, &Trail3D::rebuild));
	}
	width_curve = p_width_curve;
	if (width_curve.is_valid()) {
		width_curve->connect_changed(callable_mp(this, &Trail3D::rebuild));
	}
	_needs_rebuilding = true;
}

Ref<Curve> Trail3D::get_width_curve() const {
	return width_curve;
}

void Trail3D::set_color(const Color &p_color) {
	if (p_color != color) {
		_needs_rebuilding = true;
	}
	color = p_color;
}

Color Trail3D::get_color() const {
	return color;
}

void Trail3D::set_color_gradient(Ref<Gradient> p_color_gradient) {
	if (p_color_gradient == color_gradient) {
		return;
	}
	if (color_gradient.is_valid()) {
		color_gradient->disconnect_changed(callable_mp(this, &Trail3D::rebuild));
	}
	color_gradient = p_color_gradient;
	if (color_gradient.is_valid()) {
		color_gradient->connect_changed(callable_mp(this, &Trail3D::rebuild));
	}
	_needs_rebuilding = true;
}

Ref<Gradient> Trail3D::get_color_gradient() const {
	return color_gradient;
}

void Trail3D::set_material_mode(MaterialMode p_material_mode) {
	if (material_mode == p_material_mode) {
		return;
	}
	material_mode = p_material_mode;
	notify_property_list_changed();
	_ensure_material();
}

Trail3D::MaterialMode Trail3D::get_material_mode() const {
	return material_mode;
}

void Trail3D::set_material(Ref<ShaderMaterial> p_material) {
	if (material_mode != MATERIAL_MODE_CUSTOM) {
		return;
	}
	material = p_material;
	_ensure_material();
}

Ref<ShaderMaterial> Trail3D::get_material() const {
	return material;
}

void Trail3D::set_mesh_alignment(MeshAlignment p_alignment) {
	if (p_alignment != alignment) {
		_needs_rebuilding = true;
		alignment = p_alignment;
		_ensure_material();
	}
}

Trail3D::MeshAlignment Trail3D::get_mesh_alignment() const {
	return alignment;
}

void Trail3D::set_tiling_mode(TilingMode p_tiling_mode) {
	if (p_tiling_mode != tiling_mode) {
		_needs_rebuilding = true;
	}
	tiling_mode = p_tiling_mode;
	notify_property_list_changed();
}

Trail3D::TilingMode Trail3D::get_tiling_mode() const {
	return tiling_mode;
}

void Trail3D::set_tiling_multiplier(float p_tiling_multiplier) {
	if (p_tiling_multiplier != tiling_multiplier) {
		_needs_rebuilding = true;
	}
	tiling_multiplier = p_tiling_multiplier;
}

float Trail3D::get_tiling_multiplier() const {
	return tiling_multiplier;
}

void Trail3D::rebuild() {
	_needs_rebuilding = true;
	_do_rebuild();
}

void Trail3D::clear() {
	points.clear();
	normals.clear();
	_times.clear();
	tangents.clear();
	velocities.clear();
	tiling_offset = 0.;
	_last_vertex_count = 600;
	_init_clear_mesh();
	rebuild();
}

// Trail3D

void Trail3D::set_limit_mode(LimitMode p_limit_mode) {
	if (limit_mode == p_limit_mode) {
		return;
	}
	limit_mode = p_limit_mode;
	clear();
	notify_property_list_changed();
}

Trail3D::LimitMode Trail3D::get_limit_mode() const {
	return limit_mode;
}

void Trail3D::set_min_section_length(real_t p_min_section_length) {
	if (p_min_section_length != min_section_length) {
		_needs_rebuilding = true;
	}
	min_section_length = p_min_section_length;
}

real_t Trail3D::get_min_section_length() const {
	return min_section_length;
}

void Trail3D::set_lifetime(real_t p_lifetime) {
	lifetime = p_lifetime;
}

real_t Trail3D::get_lifetime() const {
	return lifetime;
}

void Trail3D::set_max_length(real_t p_max_length) {
	max_length = p_max_length;
}

real_t Trail3D::get_max_length() const {
	return max_length;
}

void Trail3D::set_pin_uv(bool p_pin_uv) {
	if (p_pin_uv != pin_uv) {
		_needs_rebuilding = true;
	}
	pin_uv = p_pin_uv;
}

bool Trail3D::get_pin_uv() const {
	return pin_uv;
}

real_t Trail3D::_calc_current_length() const {
	real_t length = 0.0;
	for (int i = 0; i < points.size() - 1; i++) {
		length += points.get(i).distance_to(points.get(i + 1));
	}

	return length;
}

real_t Trail3D::get_current_length() const {
	return _calc_current_length();
}

void Trail3D::_init_clear_mesh() {
	if (Object::cast_to<ArrayMesh>(_mesh.ptr()) == nullptr) {
		_mesh = (memnew(ArrayMesh));
		set_base(_mesh->get_rid());
	}
	Array arrays;

	PackedVector3Array mesh_vertices;
	PackedVector3Array mesh_normals;
	PackedColorArray mesh_colors;
	PackedVector2Array mesh_uvs;
	PackedInt32Array mesh_indices;

	mesh_vertices.resize(_last_vertex_count);
	mesh_normals.resize(_last_vertex_count);
	mesh_uvs.resize(_last_vertex_count);
	mesh_colors.resize(_last_vertex_count);
	mesh_indices.resize(_last_vertex_count * 4);

	mesh_vertices.fill(Vector3(0.0, 0.0, 0.0));
	mesh_normals.fill(Vector3(0.0, 0.0, 0.0));
	mesh_uvs.fill(Vector2(0.0, 0.0));
	mesh_colors.fill(Color(1.0, 1.0, 1.0, 0.0));
	mesh_indices.fill(0);

	arrays.resize(RSE::ARRAY_MAX);
	arrays[RSE::ARRAY_VERTEX] = mesh_vertices;
	arrays[RSE::ARRAY_NORMAL] = mesh_normals;
	arrays[RSE::ARRAY_TEX_UV] = mesh_uvs;
	arrays[RSE::ARRAY_COLOR] = mesh_colors;
	arrays[RSE::ARRAY_INDEX] = mesh_indices;

	RenderingServerTypes::SurfaceData sd;
	Error err = RS::get_singleton()->mesh_create_surface_data_from_arrays(&sd, RSE::PRIMITIVE_TRIANGLES, arrays, Array(), Dictionary());

	ERR_FAIL_COND_MSG(err != OK, "Trail3D failed mesh initialization. Please open a bug report");

	mesh_surface_format = sd.format;
	vertex_buffer = sd.vertex_data;
	attribute_buffer = sd.attribute_data;
	index_buffer = sd.index_data;

	// We need to have it calculate the offset info
	RS::get_singleton()->mesh_surface_make_offsets_from_format(sd.format, sd.vertex_count, sd.index_count, mesh_surface_offsets, vertex_stride, normal_tangent_stride, attrib_stride, skin_stride);

	_mesh->clear_surfaces();
	_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);

	_ensure_material();
}

void Trail3D::_do_rebuild() {
	if (!_needs_rebuilding || !is_inside_tree() || !is_ready()) {
		return;
	}

	int points_count = points.size();

	if (points_count < 2) {
		return;
	}

	if (points_count * 3 > _last_vertex_count) {
		while (points_count * 3 > _last_vertex_count) {
			_last_vertex_count *= 2;
		}
		if (_last_vertex_count >= 65536) {
			WARN_PRINT("Vertex count for trail tried to exceed max vertex size.");
		}
		_last_vertex_count = MIN(_last_vertex_count, 65536);

		_init_clear_mesh();

		points_count = MIN(points_count, _last_vertex_count / 3);
	}

	_needs_rebuilding = false;

	uint8_t *index_write_buffer = index_buffer.ptrw();

	for (int i = 0; i < points_count - 1; i++) {
		int j = i * 12;
		uint16_t k = (uint16_t)i * 3;

		uint16_t idx = k;
		memcpy(&index_write_buffer[j * 2], &idx, sizeof(uint16_t));
		idx = k + 3;
		memcpy(&index_write_buffer[(j + 1) * 2], &idx, sizeof(uint16_t));
		idx = k + 1;
		memcpy(&index_write_buffer[(j + 2) * 2], &idx, sizeof(uint16_t));
		idx = k + 1;
		memcpy(&index_write_buffer[(j + 3) * 2], &idx, sizeof(uint16_t));
		idx = k + 3;
		memcpy(&index_write_buffer[(j + 4) * 2], &idx, sizeof(uint16_t));
		idx = k + 4;
		memcpy(&index_write_buffer[(j + 5) * 2], &idx, sizeof(uint16_t));
		idx = k + 1;
		memcpy(&index_write_buffer[(j + 6) * 2], &idx, sizeof(uint16_t));
		idx = k + 4;
		memcpy(&index_write_buffer[(j + 7) * 2], &idx, sizeof(uint16_t));
		idx = k + 2;
		memcpy(&index_write_buffer[(j + 8) * 2], &idx, sizeof(uint16_t));
		idx = k + 2;
		memcpy(&index_write_buffer[(j + 9) * 2], &idx, sizeof(uint16_t));
		idx = k + 4;
		memcpy(&index_write_buffer[(j + 10) * 2], &idx, sizeof(uint16_t));
		idx = k + 5;
		memcpy(&index_write_buffer[(j + 11) * 2], &idx, sizeof(uint16_t));
	}

	Transform3D inv_global_tf = get_global_transform_interpolated().inverse();

	real_t length = _calc_current_length();

	real_t dist = 0.0;
	Vector3 min_bounds = Vector3();
	Vector3 max_bounds = Vector3();
	Vector3 width_bounds = Vector3(width, width, width);

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
			case TILING_MODE_LENGTH: {
				if (pin_uv) {
					length_uv = (dist + tiling_offset) * tiling_multiplier;
				} else {
					length_uv = dist * tiling_multiplier;
				}
			} break;
			case TILING_MODE_UNIT: {
				length_uv = ratio * tiling_multiplier;
			} break;
			default: {
			}
		}

		real_t half_width = width * 0.5;

		if (width_curve.is_valid()) {
			half_width *= width_curve->sample(ratio);
		}

		Vector3 tangent = tangents[i];

		p = inv_global_tf.xform(p);
		tangent = inv_global_tf.basis.xform(tangent);
		min_bounds = min_bounds.min(p - width_bounds);
		max_bounds = max_bounds.max(p + width_bounds);

		if (alignment == MESH_ALIGNMENT_BILLBOARD) {
			_encode_vertex(p, j0);
			_encode_vertex(p, j1);
			_encode_vertex(p, j2);

			_encode_normal(tangent, j0);
			_encode_normal(tangent, j1);
			_encode_normal(tangent, j2);

			_encode_uv(Vector2(length_uv, -half_width), j0);
			_encode_uv(Vector2(length_uv, 0), j1);
			_encode_uv(Vector2(length_uv, half_width), j2);

		} else {
			Vector3 normal = normals[i];
			normal = inv_global_tf.basis.xform(normal);

			Vector3 curve_binormal = normal.cross(tangent);
			_encode_vertex(p + half_width * curve_binormal, j0);
			_encode_vertex(p, j1);
			_encode_vertex(p - half_width * curve_binormal, j2);

			_encode_normal(normal, j0);
			_encode_normal(normal, j1);
			_encode_normal(normal, j2);

			_encode_uv(Vector2(length_uv, 0), j0);
			_encode_uv(Vector2(length_uv, 0.5), j1);
			_encode_uv(Vector2(length_uv, 1), j2);
		}
		Color c = color.srgb_to_linear();
		if (color_gradient.is_valid()) {
			c *= color_gradient->get_color_at_offset(ratio).srgb_to_linear();
		}

		_encode_color(c, j0);
		_encode_color(c, j1);
		_encode_color(c, j2);
	}

	uint16_t idx = 0;
	for (int i = points_count * 3; i < _last_vertex_count; i++) {
		_encode_vertex(Vector3(), i);
		_encode_normal(Vector3(), i);
		_encode_uv(Vector2(), i);
		_encode_color(Color(), i);
	}
	for (int i = (points_count - 1) * 12; i < _last_vertex_count * 4; i++) {
		memcpy(&index_write_buffer[i * 2], &idx, sizeof(uint16_t));
	}

	RID rid = _mesh->get_rid();

	RS::get_singleton()->mesh_surface_update_vertex_region(rid, 0, 0, vertex_buffer);
	RS::get_singleton()->mesh_surface_update_attribute_region(rid, 0, 0, attribute_buffer);
	RS::get_singleton()->mesh_surface_update_index_region(rid, 0, 0, index_buffer);

	set_custom_aabb(AABB(min_bounds, (max_bounds - min_bounds)));
}

void Trail3D::_process_trail(real_t p_delta) {
	if (!is_inside_tree() || !is_ready()) {
		return;
	}

	Transform3D tf = get_global_transform_interpolated();
	Vector3 pos = tf.origin;
	Vector3 up = tf.basis.get_column(1);
	_time += p_delta;

	if (points.size() < 2) {
		while (points.size() < 2) {
			points.insert(0, pos);
			normals.insert(0, up);
			tangents.insert(0, Vector3(0.0, 0.0, 1.0));
			if (limit_mode == LIMIT_MODE_LIFETIME) {
				_times.insert(0, _time);
				velocities.insert(0, 0.);
			}
		}
	} else if (emitting) {
		points.write[0] = pos;
		normals.write[0] = up;
		if (pos.distance_squared_to(points[1]) > CMP_EPSILON) {
			Vector3 t = (pos - points[1]).normalized();
			tangents.write[0] = t;
			if (points.size() == 2) {
				tangents.write[1] = t;
			}
		} else {
			tangents.write[0] = tangents[1];
		}
		if (limit_mode == LIMIT_MODE_LIFETIME) {
			_times.write[0] = _time;
		}
	}

	Vector3 leading = points.get(1);
	Vector3 from_leading = pos - leading;
	real_t dist_from_leading = from_leading.length();

	if (dist_from_leading < CMP_EPSILON && limit_mode == LIMIT_MODE_LIFETIME) {
		_times.write[1] = _time;
	}

	if (pin_uv) {
		tiling_offset = -_last_pinned_u - min_section_length;
	} else {
		_last_pinned_u = -tiling_offset - min_section_length;
	}

	switch (limit_mode) {
		case LIMIT_MODE_MAX_LENGTH:
			if (dist_from_leading > min_section_length && emitting) {
				tangents.insert(1, (pos - points[1]).normalized());
				points.insert(1, pos);
				normals.insert(1, up);
				_last_pinned_u += dist_from_leading;
				if (points.size() > 3) {
					tangents.write[2] = (points[1] - points[3]).normalized();
				}
			}
			break;
		case LIMIT_MODE_LIFETIME:
			if (dist_from_leading > min_section_length && emitting) {
				tangents.insert(1, (pos - points[1]).normalized());
				points.insert(1, pos);
				normals.insert(1, up);
				_times.insert(1, _time);
				_last_pinned_u += dist_from_leading;
				velocities.insert(1, (points[1] - points[2]).length() / (_times[1] - _times[2]));
				if (points.size() > 3) {
					tangents.write[2] = (points[1] - points[3]).normalized();
				}
			}
			break;
		default:
			break;
	}

	if (pin_uv) {
		tiling_offset -= dist_from_leading;
	}

	if (limit_mode == LIMIT_MODE_MAX_LENGTH) {
		int last_index = points.size() - 1;
		real_t total_length = 0.0;
		for (int i = 0; i < last_index; i++) {
			total_length += points[i].distance_to(points[i + 1]);
		}

		real_t extra_length = total_length - max_length;

		while (extra_length > 0 && points.size() > 1) {
			Vector3 last_point = points[last_index];
			Vector3 second_last_point = points[last_index - 1];
			real_t last_section_length = last_point.distance_to(second_last_point);
			if (last_section_length > extra_length) {
				real_t shortened_section_length = last_section_length - extra_length;
				points.write[last_index] = second_last_point + (last_point - second_last_point) * (shortened_section_length / last_section_length);
			} else {
				points.remove_at(last_index);
				normals.remove_at(last_index);
				tangents.remove_at(last_index);
				last_index -= 1;
			}
			extra_length -= last_section_length;
		}

	} else if (limit_mode == LIMIT_MODE_LIFETIME) {
		// ATTEMPTS: 5
		// Add +1 to this counter for every failed attempt to improve this code
		int last_index = _times.size() - 1;
		real_t second_last_time = _times.get(last_index - 1);
		while ((second_last_time < _time - lifetime) && points.size() > 2) {
			points.remove_at(last_index);
			normals.remove_at(last_index);
			tangents.remove_at(last_index);
			_times.remove_at(last_index);
			velocities.remove_at(last_index);
			last_index -= 1;
			_last_section_speed = (points[last_index - 1] - points[last_index]).length() / (_times[last_index - 1] - _times[last_index]);
			second_last_time = _times.get(last_index - 1);
		}

		real_t last_time = _times[last_index];
		if (last_time <= _time - lifetime) {
			Vector3 dir = points[last_index - 1] - points[last_index];
			real_t distance = dir.length();

			if (distance > 0.) {
				real_t speed_distance = velocities[last_index] * p_delta;
				points.write[last_index] += dir / distance * MIN(distance, speed_distance);
			}
		}
	}
	_needs_rebuilding = true;
}

void Trail3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "material" && material_mode != MATERIAL_MODE_CUSTOM) {
		p_property.usage = PROPERTY_USAGE_NONE;
	} else if (p_property.name == "lifetime" && limit_mode == LIMIT_MODE_MAX_LENGTH) {
		p_property.usage = PROPERTY_USAGE_NONE;
	} else if (p_property.name == "max_length" && limit_mode == LIMIT_MODE_LIFETIME) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void Trail3D::_ensure_material() {
	if (!_mesh.is_valid()) {
		return;
	}
	switch (material_mode) {
		case MATERIAL_MODE_ADD: {
			if (alignment == MESH_ALIGNMENT_BILLBOARD) {
				_mesh->surface_set_material(0, billboard_additive_material);
			} else {
				_mesh->surface_set_material(0, local_additive_material);
			}

		} break;
		case MATERIAL_MODE_MIX: {
			if (alignment == MESH_ALIGNMENT_BILLBOARD) {
				_mesh->surface_set_material(0, billboard_material);
			} else {
				_mesh->surface_set_material(0, local_material);
			}
		} break;
		case MATERIAL_MODE_CUSTOM: {
			_mesh->surface_set_material(0, material);
		} break;
		case MATERIAL_MODE_MAX: {
		} break;
	}
}

void Trail3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &Trail3D::set_width);
	ClassDB::bind_method(D_METHOD("get_width"), &Trail3D::get_width);
	ClassDB::bind_method(D_METHOD("set_width_curve", "curve"), &Trail3D::set_width_curve);
	ClassDB::bind_method(D_METHOD("get_width_curve"), &Trail3D::get_width_curve);
	ClassDB::bind_method(D_METHOD("set_emitting", "emitting"), &Trail3D::set_emitting);
	ClassDB::bind_method(D_METHOD("is_emitting"), &Trail3D::is_emitting);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &Trail3D::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &Trail3D::get_color);
	ClassDB::bind_method(D_METHOD("set_color_gradient", "gradient"), &Trail3D::set_color_gradient);
	ClassDB::bind_method(D_METHOD("get_color_gradient"), &Trail3D::get_color_gradient);

	ClassDB::bind_method(D_METHOD("set_material_mode", "material_mode"), &Trail3D::set_material_mode);
	ClassDB::bind_method(D_METHOD("get_material_mode"), &Trail3D::get_material_mode);
	ClassDB::bind_method(D_METHOD("set_material", "material"), &Trail3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &Trail3D::get_material);

	ClassDB::bind_method(D_METHOD("set_mesh_alignment", "alignment"), &Trail3D::set_mesh_alignment);
	ClassDB::bind_method(D_METHOD("get_mesh_alignment"), &Trail3D::get_mesh_alignment);

	ClassDB::bind_method(D_METHOD("set_tiling_multiplier", "tiling_multiplier"), &Trail3D::set_tiling_multiplier);
	ClassDB::bind_method(D_METHOD("get_tiling_multiplier"), &Trail3D::get_tiling_multiplier);
	ClassDB::bind_method(D_METHOD("set_tiling_mode", "tiling_mode"), &Trail3D::set_tiling_mode);
	ClassDB::bind_method(D_METHOD("get_tiling_mode"), &Trail3D::get_tiling_mode);

	ClassDB::bind_method(D_METHOD("rebuild"), &Trail3D::rebuild);
	ClassDB::bind_method(D_METHOD("clear"), &Trail3D::clear);

	// Beam and Trail
	ClassDB::bind_method(D_METHOD("set_min_section_length", "min_section_length"), &Trail3D::set_min_section_length);
	ClassDB::bind_method(D_METHOD("get_min_section_length"), &Trail3D::get_min_section_length);

	// Trail

	ClassDB::bind_method(D_METHOD("set_limit_mode", "limit_mode"), &Trail3D::set_limit_mode);
	ClassDB::bind_method(D_METHOD("get_limit_mode"), &Trail3D::get_limit_mode);

	ClassDB::bind_method(D_METHOD("set_lifetime", "lifetime"), &Trail3D::set_lifetime);
	ClassDB::bind_method(D_METHOD("get_lifetime"), &Trail3D::get_lifetime);

	ClassDB::bind_method(D_METHOD("set_max_length", "max_length"), &Trail3D::set_max_length);
	ClassDB::bind_method(D_METHOD("get_max_length"), &Trail3D::get_max_length);

	ClassDB::bind_method(D_METHOD("set_pin_uv", "pin_uv"), &Trail3D::set_pin_uv);
	ClassDB::bind_method(D_METHOD("get_pin_uv"), &Trail3D::get_pin_uv);

	ClassDB::bind_method(D_METHOD("get_current_length"), &Trail3D::get_current_length);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "emitting"), "set_emitting", "is_emitting");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "limit_mode", PROPERTY_HINT_ENUM, "Time,Length"), "set_limit_mode", "get_limit_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lifetime", PROPERTY_HINT_RANGE, "0.0,10.0,0.001,or_greater"), "set_lifetime", "get_lifetime");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_length", PROPERTY_HINT_RANGE, "0.0,10.0,0.01,or_greater"), "set_max_length", "get_max_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "width"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "width_curve", PROPERTY_HINT_RESOURCE_TYPE, Curve::get_class_static()), "set_width_curve", "get_width_curve");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color", PROPERTY_HINT_NONE), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "color_gradient", PROPERTY_HINT_RESOURCE_TYPE, Gradient::get_class_static()), "set_color_gradient", "get_color_gradient");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh_alignment", PROPERTY_HINT_ENUM, "Local,Billboard"), "set_mesh_alignment", "get_mesh_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "material_mode", PROPERTY_HINT_ENUM, "Default,Default Additive,Custom"), "set_material_mode", "get_material_mode");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, ShaderMaterial::get_class_static()), "set_material", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tiling_mode", PROPERTY_HINT_ENUM, "Unit,Length"), "set_tiling_mode", "get_tiling_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tiling_multiplier"), "set_tiling_multiplier", "get_tiling_multiplier");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pin_uv"), "set_pin_uv", "get_pin_uv");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_section_length", PROPERTY_HINT_RANGE, "0.05,1.0,0.01"), "set_min_section_length", "get_min_section_length");

	BIND_ENUM_CONSTANT(MESH_ALIGNMENT_LOCAL);
	BIND_ENUM_CONSTANT(MESH_ALIGNMENT_BILLBOARD);
	BIND_ENUM_CONSTANT(MESH_ALIGNMENT_MAX);

	BIND_ENUM_CONSTANT(TILING_MODE_UNIT);
	BIND_ENUM_CONSTANT(TILING_MODE_LENGTH);
	BIND_ENUM_CONSTANT(TILING_MAX);

	BIND_ENUM_CONSTANT(LIMIT_MODE_LIFETIME);
	BIND_ENUM_CONSTANT(LIMIT_MODE_MAX_LENGTH);
	BIND_ENUM_CONSTANT(LIMIT_MODE_MAX);

	BIND_ENUM_CONSTANT(MATERIAL_MODE_MIX);
	BIND_ENUM_CONSTANT(MATERIAL_MODE_ADD);
	BIND_ENUM_CONSTANT(MATERIAL_MODE_CUSTOM);
	BIND_ENUM_CONSTANT(MATERIAL_MODE_MAX);
}

Trail3D::Trail3D() {
	set_process_internal(true);
	_init_clear_mesh();
}

void Trail3D::_encode_vertex(const Vector3 &p_vertex, int p_index) {
	DEV_ASSERT((p_index * vertex_stride + mesh_surface_offsets[RSE::ARRAY_COLOR] + 12) <= attribute_buffer.size());
	float v_vertex[3] = { (float)p_vertex.x, (float)p_vertex.y, (float)p_vertex.z };
	memcpy(&(vertex_buffer.ptrw())[p_index * vertex_stride + mesh_surface_offsets[RSE::ARRAY_VERTEX]], &v_vertex, sizeof(float) * 3);
}

void Trail3D::_encode_normal(const Vector3 &p_normal, int p_index) {
	DEV_ASSERT((p_index * normal_tangent_stride + mesh_surface_offsets[RSE::ARRAY_COLOR] + 4) <= attribute_buffer.size());
	uint32_t v_normal = 0;
	Vector2 res = p_normal.octahedron_encode();
	v_normal |= (uint16_t)CLAMP(res.x * 65535, 0, 65535);
	v_normal |= (uint16_t)CLAMP(res.y * 65535, 0, 65535) << 16;
	memcpy(&(vertex_buffer.ptrw())[p_index * normal_tangent_stride + mesh_surface_offsets[RSE::ARRAY_NORMAL]], &v_normal, 4);
}

void Trail3D::_encode_uv(const Vector2 &p_uv, int p_index) {
	DEV_ASSERT((p_index * attrib_stride + mesh_surface_offsets[RSE::ARRAY_TEX_UV] + 8) <= attribute_buffer.size());
	float v_uv[2] = { (float)p_uv.x, (float)p_uv.y };
	memcpy(&(attribute_buffer.ptrw())[p_index * attrib_stride + mesh_surface_offsets[RSE::ARRAY_TEX_UV]], v_uv, 8);
}

void Trail3D::_encode_color(const Color &p_color, int p_index) {
	DEV_ASSERT((p_index * attrib_stride + mesh_surface_offsets[RSE::ARRAY_COLOR] + 4) <= attribute_buffer.size());
	uint8_t v_color[4] = {
		uint8_t(CLAMP(p_color.r * 255.0, 0.0, 255.0)),
		uint8_t(CLAMP(p_color.g * 255.0, 0.0, 255.0)),
		uint8_t(CLAMP(p_color.b * 255.0, 0.0, 255.0)),
		uint8_t(CLAMP(p_color.a * 255.0, 0.0, 255.0))
	};
	memcpy(&(attribute_buffer.ptrw())[p_index * attrib_stride + mesh_surface_offsets[RSE::ARRAY_COLOR]], v_color, 4);
}
