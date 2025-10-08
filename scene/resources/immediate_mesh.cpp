/**************************************************************************/
/*  immediate_mesh.cpp                                                    */
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

#include "immediate_mesh.h"

void ImmediateMesh::surface_begin(PrimitiveType p_primitive, const Ref<Material> &p_material) {
	ERR_FAIL_COND_MSG(surface_active, "Already creating a new surface.");
	active_surface_data.primitive = p_primitive;
	active_surface_data.material = p_material;
	surface_active = true;
}
void ImmediateMesh::surface_set_color(const Color &p_color) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");

	if (!uses_colors) {
		colors.resize(vertices.size());
		for (Color &color : colors) {
			color = p_color;
		}
		uses_colors = true;
	}

	current_color = p_color;
}
void ImmediateMesh::surface_set_normal(const Vector3 &p_normal) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");

	if (!uses_normals) {
		normals.resize(vertices.size());
		for (Vector3 &normal : normals) {
			normal = p_normal;
		}
		uses_normals = true;
	}

	current_normal = p_normal;
}
void ImmediateMesh::surface_set_tangent(const Plane &p_tangent) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	if (!uses_tangents) {
		tangents.resize(vertices.size());
		for (Plane &tangent : tangents) {
			tangent = p_tangent;
		}
		uses_tangents = true;
	}

	current_tangent = p_tangent;
}
void ImmediateMesh::surface_set_uv(const Vector2 &p_uv) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	if (!uses_uvs) {
		uvs.resize(vertices.size());
		for (Vector2 &uv : uvs) {
			uv = p_uv;
		}
		uses_uvs = true;
	}

	current_uv = p_uv;
}
void ImmediateMesh::surface_set_uv2(const Vector2 &p_uv2) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	if (!uses_uv2s) {
		uv2s.resize(vertices.size());
		for (Vector2 &uv : uv2s) {
			uv = p_uv2;
		}
		uses_uv2s = true;
	}

	current_uv2 = p_uv2;
}
void ImmediateMesh::surface_set_custom0(const Vector4 &p_custom0) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	if (!uses_custom0) {
		custom0s.resize(vertices.size());
		for (Vector4 &custom0 : custom0s) {
			custom0 = p_custom0;
		}
		uses_custom0 = true;
	}

	current_custom0 = p_custom0;
}
void ImmediateMesh::surface_set_custom1(const Vector4 &p_custom1) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	if (!uses_custom1) {
		custom1s.resize(vertices.size());
		for (Vector4 &custom1 : custom1s) {
			custom1 = p_custom1;
		}
		uses_custom1 = true;
	}

	current_custom1 = p_custom1;
}
void ImmediateMesh::surface_set_custom2(const Vector4 &p_custom2) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	if (!uses_custom2) {
		custom2s.resize(vertices.size());
		for (Vector4 &custom2 : custom2s) {
			custom2 = p_custom2;
		}
		uses_custom2 = true;
	}

	current_custom2 = p_custom2;
}
void ImmediateMesh::surface_set_custom3(const Vector4 &p_custom3) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	if (!uses_custom3) {
		custom3s.resize(vertices.size());
		for (Vector4 &custom3 : custom3s) {
			custom3 = p_custom3;
		}
		uses_custom3 = true;
	}

	current_custom3 = p_custom3;
}
void ImmediateMesh::surface_add_vertex(const Vector3 &p_vertex) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	ERR_FAIL_COND_MSG(vertices.size() && active_surface_data.vertex_2d, "Can't mix 2D and 3D vertices in a surface.");

	if (uses_colors) {
		colors.push_back(current_color);
	}
	if (uses_normals) {
		normals.push_back(current_normal);
	}
	if (uses_tangents) {
		tangents.push_back(current_tangent);
	}
	if (uses_uvs) {
		uvs.push_back(current_uv);
	}
	if (uses_uv2s) {
		uv2s.push_back(current_uv2);
	}
	if (uses_custom0) {
		custom0s.push_back(current_custom0);
	}
	if (uses_custom1) {
		custom1s.push_back(current_custom1);
	}
	if (uses_custom2) {
		custom2s.push_back(current_custom2);
	}
	if (uses_custom3) {
		custom3s.push_back(current_custom3);
	}
	vertices.push_back(p_vertex);
}

void ImmediateMesh::surface_add_vertex_2d(const Vector2 &p_vertex) {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	ERR_FAIL_COND_MSG(vertices.size() && !active_surface_data.vertex_2d, "Can't mix 2D and 3D vertices in a surface.");

	if (uses_colors) {
		colors.push_back(current_color);
	}
	if (uses_normals) {
		normals.push_back(current_normal);
	}
	if (uses_tangents) {
		tangents.push_back(current_tangent);
	}
	if (uses_uvs) {
		uvs.push_back(current_uv);
	}
	if (uses_uv2s) {
		uv2s.push_back(current_uv2);
	}
	if (uses_custom0) {
		custom0s.push_back(current_custom0);
	}
	if (uses_custom1) {
		custom1s.push_back(current_custom1);
	}
	if (uses_custom2) {
		custom2s.push_back(current_custom2);
	}
	if (uses_custom3) {
		custom3s.push_back(current_custom3);
	}
	Vector3 v(p_vertex.x, p_vertex.y, 0);
	vertices.push_back(v);

	active_surface_data.vertex_2d = true;
}

void ImmediateMesh::surface_end() {
	ERR_FAIL_COND_MSG(!surface_active, "Not creating any surface. Use surface_begin() to do it.");
	ERR_FAIL_COND_MSG(vertices.is_empty(), "No vertices were added, surface can't be created.");

	uint64_t format = ARRAY_FORMAT_VERTEX | ARRAY_FLAG_FORMAT_CURRENT_VERSION;

	uint32_t vertex_stride = 0;
	if (active_surface_data.vertex_2d) {
		format |= ARRAY_FLAG_USE_2D_VERTICES;
		vertex_stride = sizeof(float) * 2;
	} else {
		vertex_stride = sizeof(float) * 3;
	}
	uint32_t normal_tangent_stride = 0;
	uint32_t normal_offset = 0;
	if (uses_normals) {
		format |= ARRAY_FORMAT_NORMAL;
		normal_offset = vertex_stride * vertices.size();
		normal_tangent_stride += sizeof(uint32_t);
	}
	uint32_t tangent_offset = 0;
	if (uses_tangents || uses_normals) {
		format |= ARRAY_FORMAT_TANGENT;
		tangent_offset = vertex_stride * vertices.size() + normal_tangent_stride;
		normal_tangent_stride += sizeof(uint32_t);
	}

	AABB aabb;

	{
		surface_vertex_create_cache.resize((vertex_stride + normal_tangent_stride) * vertices.size());
		uint8_t *surface_vertex_ptr = surface_vertex_create_cache.ptrw();
		for (uint32_t i = 0; i < vertices.size(); i++) {
			{
				float *vtx = (float *)&surface_vertex_ptr[i * vertex_stride];
				vtx[0] = vertices[i].x;
				vtx[1] = vertices[i].y;
				if (!active_surface_data.vertex_2d) {
					vtx[2] = vertices[i].z;
				}
				if (i == 0) {
					aabb = AABB(vertices[i], SMALL_VEC3); // Must have a bit of size.
				} else {
					aabb.expand_to(vertices[i]);
				}
			}
			if (uses_normals) {
				uint32_t *normal = (uint32_t *)&surface_vertex_ptr[i * normal_tangent_stride + normal_offset];

				Vector2 n = normals[i].octahedron_encode();

				uint32_t value = 0;
				value |= (uint16_t)CLAMP(n.x * 65535, 0, 65535);
				value |= (uint16_t)CLAMP(n.y * 65535, 0, 65535) << 16;

				*normal = value;
			}
			if (uses_tangents || uses_normals) {
				uint32_t *tangent = (uint32_t *)&surface_vertex_ptr[i * normal_tangent_stride + tangent_offset];
				Vector2 t;
				if (uses_tangents) {
					t = tangents[i].normal.octahedron_tangent_encode(tangents[i].d);
				} else {
					Vector3 tan = Vector3(normals[i].z, -normals[i].x, normals[i].y).cross(normals[i].normalized()).normalized();
					t = tan.octahedron_tangent_encode(1.0);
				}

				uint32_t value = 0;
				value |= (uint16_t)CLAMP(t.x * 65535, 0, 65535);
				value |= (uint16_t)CLAMP(t.y * 65535, 0, 65535) << 16;
				if (value == 4294901760) {
					// (1, 1) and (0, 1) decode to the same value, but (0, 1) messes with our compression detection.
					// So we sanitize here.
					value = 4294967295;
				}

				*tangent = value;
			}
		}
	}

	if (uses_colors || uses_uvs || uses_uv2s) {
		uint32_t attribute_stride = 0;

		if (uses_colors) {
			format |= ARRAY_FORMAT_COLOR;
			attribute_stride += sizeof(uint8_t) * 4;
		}
		uint32_t uv_offset = 0;
		if (uses_uvs) {
			format |= ARRAY_FORMAT_TEX_UV;
			uv_offset = attribute_stride;
			attribute_stride += sizeof(float) * 2;
		}
		uint32_t uv2_offset = 0;
		if (uses_uv2s) {
			format |= ARRAY_FORMAT_TEX_UV2;
			uv2_offset = attribute_stride;
			attribute_stride += sizeof(float) * 2;
		}
		uint32_t custom0_offset = 0;
		if (uses_custom0) {
			format |= ARRAY_FORMAT_CUSTOM0 | (ARRAY_CUSTOM_RGBA_FLOAT << ARRAY_FORMAT_CUSTOM0_SHIFT);
			custom0_offset = attribute_stride;
			attribute_stride += sizeof(float) * 4;
		}
		uint32_t custom1_offset = 0;
		if (uses_custom1) {
			format |= ARRAY_FORMAT_CUSTOM1 | (ARRAY_CUSTOM_RGBA_FLOAT << ARRAY_FORMAT_CUSTOM1_SHIFT);
			custom1_offset = attribute_stride;
			attribute_stride += sizeof(float) * 4;
		}
		uint32_t custom2_offset = 0;
		if (uses_custom2) {
			format |= ARRAY_FORMAT_CUSTOM2 | (ARRAY_CUSTOM_RGBA_FLOAT << ARRAY_FORMAT_CUSTOM2_SHIFT);
			custom2_offset = attribute_stride;
			attribute_stride += sizeof(float) * 4;
		}
		uint32_t custom3_offset = 0;
		if (uses_custom3) {
			format |= ARRAY_FORMAT_CUSTOM3 | (ARRAY_CUSTOM_RGBA_FLOAT << ARRAY_FORMAT_CUSTOM3_SHIFT);
			custom3_offset = attribute_stride;
			attribute_stride += sizeof(float) * 4;
		}

		surface_attribute_create_cache.resize(vertices.size() * attribute_stride);

		uint8_t *surface_attribute_ptr = surface_attribute_create_cache.ptrw();

		for (uint32_t i = 0; i < vertices.size(); i++) {
			if (uses_colors) {
				uint8_t *color8 = (uint8_t *)&surface_attribute_ptr[i * attribute_stride];

				color8[0] = uint8_t(CLAMP(colors[i].r * 255.0, 0.0, 255.0));
				color8[1] = uint8_t(CLAMP(colors[i].g * 255.0, 0.0, 255.0));
				color8[2] = uint8_t(CLAMP(colors[i].b * 255.0, 0.0, 255.0));
				color8[3] = uint8_t(CLAMP(colors[i].a * 255.0, 0.0, 255.0));
			}
			if (uses_uvs) {
				float *uv = (float *)&surface_attribute_ptr[i * attribute_stride + uv_offset];

				uv[0] = uvs[i].x;
				uv[1] = uvs[i].y;
			}

			if (uses_uv2s) {
				float *uv2 = (float *)&surface_attribute_ptr[i * attribute_stride + uv2_offset];

				uv2[0] = uv2s[i].x;
				uv2[1] = uv2s[i].y;
			}

			if (uses_custom0) {
				float *custom0 = (float *)&surface_attribute_ptr[i * attribute_stride + custom0_offset];

				custom0[0] = custom0s[i].x;
				custom0[1] = custom0s[i].y;
				custom0[2] = custom0s[i].z;
				custom0[3] = custom0s[i].w;
			}
			if (uses_custom1) {
				float *custom1 = (float *)&surface_attribute_ptr[i * attribute_stride + custom1_offset];

				custom1[0] = custom1s[i].x;
				custom1[1] = custom1s[i].y;
				custom1[2] = custom1s[i].z;
				custom1[3] = custom1s[i].w;
			}
			if (uses_custom2) {
				float *custom2 = (float *)&surface_attribute_ptr[i * attribute_stride + custom2_offset];

				custom2[0] = custom2s[i].x;
				custom2[1] = custom2s[i].y;
				custom2[2] = custom2s[i].z;
				custom2[3] = custom2s[i].w;
			}
			if (uses_custom3) {
				float *custom3 = (float *)&surface_attribute_ptr[i * attribute_stride + custom3_offset];

				custom3[0] = custom3s[i].x;
				custom3[1] = custom3s[i].y;
				custom3[2] = custom3s[i].z;
				custom3[3] = custom3s[i].w;
			}
		}
	}

	RS::SurfaceData sd;

	sd.primitive = RS::PrimitiveType(active_surface_data.primitive);
	sd.format = format;
	sd.vertex_data = surface_vertex_create_cache;
	if (uses_colors || uses_uvs || uses_uv2s || uses_custom0 || uses_custom1 || uses_custom2 || uses_custom3) {
		sd.attribute_data = surface_attribute_create_cache;
	}
	sd.vertex_count = vertices.size();
	sd.aabb = aabb;
	if (active_surface_data.material.is_valid()) {
		sd.material = active_surface_data.material->get_rid();
	}

	RS::get_singleton()->mesh_add_surface(mesh, sd);

	active_surface_data.aabb = aabb;

	active_surface_data.format = format;
	active_surface_data.array_len = vertices.size();

	surfaces.push_back(active_surface_data);

	colors.clear();
	normals.clear();
	tangents.clear();
	uvs.clear();
	uv2s.clear();
	custom0s.clear();
	custom1s.clear();
	custom2s.clear();
	custom3s.clear();
	vertices.clear();

	uses_colors = false;
	uses_normals = false;
	uses_tangents = false;
	uses_uvs = false;
	uses_uv2s = false;
	uses_custom0 = false;
	uses_custom1 = false;
	uses_custom2 = false;
	uses_custom3 = false;

	surface_active = false;

	emit_changed();
}

void ImmediateMesh::clear_surfaces() {
	RS::get_singleton()->mesh_clear(mesh);
	surfaces.clear();
	surface_active = false;

	colors.clear();
	normals.clear();
	tangents.clear();
	uvs.clear();
	uv2s.clear();
	custom0s.clear();
	custom1s.clear();
	custom2s.clear();
	custom3s.clear();
	vertices.clear();

	uses_colors = false;
	uses_normals = false;
	uses_tangents = false;
	uses_uvs = false;
	uses_uv2s = false;
	uses_custom0 = false;
	uses_custom1 = false;
	uses_custom2 = false;
	uses_custom3 = false;
}

int ImmediateMesh::get_surface_count() const {
	return surfaces.size();
}
int ImmediateMesh::surface_get_array_len(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, int(surfaces.size()), -1);
	return surfaces[p_idx].array_len;
}
int ImmediateMesh::surface_get_array_index_len(int p_idx) const {
	return 0;
}
Array ImmediateMesh::surface_get_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, int(surfaces.size()), Array());
	return RS::get_singleton()->mesh_surface_get_arrays(mesh, p_surface);
}
TypedArray<Array> ImmediateMesh::surface_get_blend_shape_arrays(int p_surface) const {
	return TypedArray<Array>();
}
Dictionary ImmediateMesh::surface_get_lods(int p_surface) const {
	return Dictionary();
}
BitField<Mesh::ArrayFormat> ImmediateMesh::surface_get_format(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, int(surfaces.size()), 0);
	return surfaces[p_idx].format;
}
Mesh::PrimitiveType ImmediateMesh::surface_get_primitive_type(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, int(surfaces.size()), PRIMITIVE_MAX);
	return surfaces[p_idx].primitive;
}
void ImmediateMesh::surface_set_material(int p_idx, const Ref<Material> &p_material) {
	ERR_FAIL_INDEX(p_idx, int(surfaces.size()));
	surfaces[p_idx].material = p_material;
	RID mat;
	if (p_material.is_valid()) {
		mat = p_material->get_rid();
	}
	RS::get_singleton()->mesh_surface_set_material(mesh, p_idx, mat);
}
Ref<Material> ImmediateMesh::surface_get_material(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, int(surfaces.size()), Ref<Material>());
	return surfaces[p_idx].material;
}
int ImmediateMesh::get_blend_shape_count() const {
	return 0;
}
StringName ImmediateMesh::get_blend_shape_name(int p_index) const {
	return StringName();
}
void ImmediateMesh::set_blend_shape_name(int p_index, const StringName &p_name) {
}

AABB ImmediateMesh::get_aabb() const {
	AABB aabb;
	for (uint32_t i = 0; i < surfaces.size(); i++) {
		if (i == 0) {
			aabb = surfaces[i].aabb;
		} else {
			aabb = aabb.merge(surfaces[i].aabb);
		}
	}
	return aabb;
}

void ImmediateMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("surface_begin", "primitive", "material"), &ImmediateMesh::surface_begin, DEFVAL(Ref<Material>()));
	ClassDB::bind_method(D_METHOD("surface_set_color", "color"), &ImmediateMesh::surface_set_color);
	ClassDB::bind_method(D_METHOD("surface_set_normal", "normal"), &ImmediateMesh::surface_set_normal);
	ClassDB::bind_method(D_METHOD("surface_set_tangent", "tangent"), &ImmediateMesh::surface_set_tangent);
	ClassDB::bind_method(D_METHOD("surface_set_uv", "uv"), &ImmediateMesh::surface_set_uv);
	ClassDB::bind_method(D_METHOD("surface_set_uv2", "uv2"), &ImmediateMesh::surface_set_uv2);
	ClassDB::bind_method(D_METHOD("surface_set_custom0", "custom0"), &ImmediateMesh::surface_set_custom0);
	ClassDB::bind_method(D_METHOD("surface_set_custom1", "custom1"), &ImmediateMesh::surface_set_custom1);
	ClassDB::bind_method(D_METHOD("surface_set_custom2", "custom2"), &ImmediateMesh::surface_set_custom2);
	ClassDB::bind_method(D_METHOD("surface_set_custom3", "custom3"), &ImmediateMesh::surface_set_custom3);
	ClassDB::bind_method(D_METHOD("surface_add_vertex", "vertex"), &ImmediateMesh::surface_add_vertex);
	ClassDB::bind_method(D_METHOD("surface_add_vertex_2d", "vertex"), &ImmediateMesh::surface_add_vertex_2d);
	ClassDB::bind_method(D_METHOD("surface_end"), &ImmediateMesh::surface_end);

	ClassDB::bind_method(D_METHOD("clear_surfaces"), &ImmediateMesh::clear_surfaces);
}

RID ImmediateMesh::get_rid() const {
	return mesh;
}

ImmediateMesh::ImmediateMesh() {
	mesh = RS::get_singleton()->mesh_create();
}
ImmediateMesh::~ImmediateMesh() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	RS::get_singleton()->free_rid(mesh);
}
