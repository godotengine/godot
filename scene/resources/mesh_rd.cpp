/**************************************************************************/
/*  mesh_rd.cpp                                                           */
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

#include "mesh_rd.h"

#include "core/object/class_db.h"
#include "servers/rendering/rendering_server.h"

void MeshRD::_create_if_empty() const {
	if (!mesh.is_valid()) {
		mesh = RS::get_singleton()->mesh_create();
		RS::get_singleton()->mesh_set_path(mesh, get_path());
	}
}

void MeshRD::_recompute_aabb() {
	aabb = AABB();

	for (int i = 0; i < surfaces.size(); i++) {
		if (i == 0) {
			aabb = surfaces[i].aabb;
		} else {
			aabb.merge_with(surfaces[i].aabb);
		}
	}
}

void MeshRD::add_surface(BitField<ArrayFormat> p_format, PrimitiveType p_primitive, int p_vertex_count, RID p_vertex_buffer, const AABB &p_aabb, RID p_attribute_buffer, int p_index_count, RID p_index_buffer, const Ref<Material> &p_material, const Vector4 &p_uv_scale, RID p_indirect_buffer, int p_indirect_buffer_offset) {
	ERR_FAIL_COND(surfaces.size() == RSE::MAX_MESH_SURFACES);
	ERR_FAIL_COND_MSG(get_blend_shape_count() != 0, "MeshRD does not support blend shapes.");
	ERR_FAIL_COND_MSG(p_vertex_count < 0, "MeshRD surface vertex count must be non-negative.");
	ERR_FAIL_COND_MSG(p_index_count < 0, "MeshRD surface index count must be non-negative.");
	ERR_FAIL_COND_MSG(p_indirect_buffer_offset < 0, "Indirect buffer offset must be non-negative.");
	const BitField<ArrayFormat> skinning_flags(uint64_t(ARRAY_FORMAT_BONES) | uint64_t(ARRAY_FORMAT_WEIGHTS));
	ERR_FAIL_COND_MSG(p_format.has_flag(skinning_flags), "MeshRD does not support skeleton skinning.");

	p_format = BitField<ArrayFormat>((uint64_t(p_format) & ~(uint64_t(ARRAY_FLAG_FORMAT_VERSION_MASK) << uint64_t(ARRAY_FLAG_FORMAT_VERSION_SHIFT))) | uint64_t(ARRAY_FLAG_FORMAT_CURRENT_VERSION));

	const bool uses_empty_vertex_array = p_format.has_flag(ARRAY_FLAG_USES_EMPTY_VERTEX_ARRAY);
	const BitField<ArrayFormat> attribute_flags(uint64_t(ARRAY_FORMAT_COLOR) | uint64_t(ARRAY_FORMAT_TEX_UV) | uint64_t(ARRAY_FORMAT_TEX_UV2) | uint64_t(ARRAY_FORMAT_CUSTOM0) | uint64_t(ARRAY_FORMAT_CUSTOM1) | uint64_t(ARRAY_FORMAT_CUSTOM2) | uint64_t(ARRAY_FORMAT_CUSTOM3));
	const bool needs_attribute_buffer = p_format.has_flag(attribute_flags);
	ERR_FAIL_COND_MSG(p_vertex_count == 0 && p_index_count == 0, "MeshRD surfaces must contain vertices, indices, or both.");
	ERR_FAIL_COND_MSG(p_vertex_count > 0 && !uses_empty_vertex_array && p_vertex_buffer.is_null(), "MeshRD surfaces require a vertex buffer when vertex_count is greater than zero.");
	ERR_FAIL_COND_MSG(needs_attribute_buffer && p_attribute_buffer.is_null(), "MeshRD surface format requires an attribute buffer.");
	ERR_FAIL_COND_MSG(p_index_count > 0 && p_index_buffer.is_null(), "MeshRD surfaces require an index buffer when index_count is greater than zero.");

	_create_if_empty();

	Surface s;
	s.aabb = p_aabb;
	s.primitive = p_primitive;
	s.vertex_count = p_vertex_count;
	s.index_count = p_index_count;
	s.format = p_format;
	s.material = p_material;

	RenderingServerTypes::SurfaceBuffers surface;
	surface.format = uint64_t(p_format);
	surface.primitive = RSE::PrimitiveType(p_primitive);
	surface.vertex_count = p_vertex_count;
	surface.vertex_buffer = p_vertex_buffer;
	surface.attribute_buffer = p_attribute_buffer;
	surface.index_count = p_index_count;
	surface.index_buffer = p_index_buffer;
	surface.indirect_buffer = p_indirect_buffer;
	surface.indirect_buffer_offset = p_indirect_buffer_offset;
	surface.aabb = p_aabb;
	surface.uv_scale = p_uv_scale;
	surface.material = p_material.is_valid() ? p_material->get_rid() : RID();

	RS::get_singleton()->mesh_add_surface_from_buffers(mesh, surface);

	surfaces.push_back(s);
	_recompute_aabb();
	clear_cache();
	emit_changed();
}

void MeshRD::clear_surfaces() {
	if (!mesh.is_valid()) {
		return;
	}
	RS::get_singleton()->mesh_clear(mesh);
	surfaces.clear();
	aabb = AABB();
	clear_cache();
	emit_changed();
}

void MeshRD::surface_remove(int p_surface) {
	ERR_FAIL_INDEX(p_surface, surfaces.size());
	RS::get_singleton()->mesh_surface_remove(mesh, p_surface);
	surfaces.remove_at(p_surface);

	clear_cache();
	_recompute_aabb();
	emit_changed();
}

void MeshRD::surface_set_indirect_buffer(int p_surface, RID p_indirect_buffer, int p_offset) {
	ERR_FAIL_INDEX(p_surface, surfaces.size());
	ERR_FAIL_COND_MSG(p_offset < 0, "Indirect buffer offset must be non-negative.");

	RS::get_singleton()->mesh_surface_set_indirect_buffer(mesh, p_surface, p_indirect_buffer, p_offset);
	emit_changed();
}

Array MeshRD::surface_get_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Array());
	return Array();
}

TypedArray<Array> MeshRD::surface_get_blend_shape_arrays(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), TypedArray<Array>());
	return TypedArray<Array>();
}

Dictionary MeshRD::surface_get_lods(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), Dictionary());
	return Dictionary();
}

int MeshRD::get_surface_count() const {
	return surfaces.size();
}

int MeshRD::surface_get_array_len(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), -1);
	return surfaces[p_idx].vertex_count;
}

int MeshRD::surface_get_array_index_len(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), -1);
	return surfaces[p_idx].index_count;
}

BitField<Mesh::ArrayFormat> MeshRD::surface_get_format(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), 0);
	return surfaces[p_idx].format;
}

MeshRD::PrimitiveType MeshRD::surface_get_primitive_type(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), PRIMITIVE_LINES);
	return surfaces[p_idx].primitive;
}

void MeshRD::surface_set_material(int p_idx, const Ref<Material> &p_material) {
	ERR_FAIL_INDEX(p_idx, surfaces.size());
	if (surfaces[p_idx].material == p_material) {
		return;
	}

	surfaces.write[p_idx].material = p_material;
	RS::get_singleton()->mesh_surface_set_material(mesh, p_idx, p_material.is_valid() ? p_material->get_rid() : RID());
	emit_changed();
}

Ref<Material> MeshRD::surface_get_material(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), Ref<Material>());
	return surfaces[p_idx].material;
}

int MeshRD::get_blend_shape_count() const {
	return 0;
}

StringName MeshRD::get_blend_shape_name(int p_index) const {
	ERR_FAIL_V(StringName());
}

void MeshRD::set_blend_shape_name(int p_index, const StringName &p_name) {
	ERR_FAIL();
}

void MeshRD::set_custom_aabb(const AABB &p_custom) {
	_create_if_empty();
	custom_aabb = p_custom;
	RS::get_singleton()->mesh_set_custom_aabb(mesh, custom_aabb);
	emit_changed();
}

AABB MeshRD::get_custom_aabb() const {
	return custom_aabb;
}

AABB MeshRD::get_aabb() const {
	return aabb;
}

RID MeshRD::get_rid() const {
	_create_if_empty();
	return mesh;
}

void MeshRD::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_surface", "format", "primitive", "vertex_count", "vertex_buffer", "aabb", "attribute_buffer", "index_count", "index_buffer", "material", "uv_scale", "indirect_buffer", "indirect_buffer_offset"), &MeshRD::add_surface, DEFVAL(RID()), DEFVAL(0), DEFVAL(RID()), DEFVAL(Ref<Material>()), DEFVAL(Vector4()), DEFVAL(RID()), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("clear_surfaces"), &MeshRD::clear_surfaces);
	ClassDB::bind_method(D_METHOD("surface_remove", "surf_idx"), &MeshRD::surface_remove);
	ClassDB::bind_method(D_METHOD("surface_set_indirect_buffer", "surf_idx", "indirect_buffer", "offset"), &MeshRD::surface_set_indirect_buffer, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("set_custom_aabb", "aabb"), &MeshRD::set_custom_aabb);
	ClassDB::bind_method(D_METHOD("get_custom_aabb"), &MeshRD::get_custom_aabb);

	ADD_PROPERTY(PropertyInfo(Variant::AABB, "custom_aabb", PROPERTY_HINT_NO_NODEPATH, "suffix:m"), "set_custom_aabb", "get_custom_aabb");
}

MeshRD::MeshRD() {
}

MeshRD::~MeshRD() {
	if (mesh.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RenderingServer::get_singleton()->free_rid(mesh);
	}
}
