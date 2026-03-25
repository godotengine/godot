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
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server.h"

void MeshRD::_create_if_empty() const {
	if (!mesh.is_valid()) {
		mesh = RS::get_singleton()->mesh_create();
		RS::get_singleton()->mesh_set_path(mesh, get_path());
	}
}

Array MeshRD::_get_surfaces_data() const {
	Array ret;
	for (int i = 0; i < surfaces.size(); i++) {
		const Surface &surface = surfaces[i];
		if (!surface.serializable) {
			continue;
		}

		Dictionary data;
		data["format"] = surface.format;
		data["primitive"] = int(surface.primitive);
		data["vertex_count"] = surface.max_array_length;
		data["index_count"] = surface.max_index_array_length;
		data["aabb"] = surface.aabb;
		data["uv_scale"] = surface.uv_scale;
		data["vertex_buffer_creation_bits"] = surface.vertex_buffer_creation_bits;
		data["attribute_buffer_creation_bits"] = surface.attribute_buffer_creation_bits;
		data["index_buffer_creation_bits"] = surface.index_buffer_creation_bits;
		if (surface.material.is_valid()) {
			data["material"] = surface.material;
		}
		ret.push_back(data);
	}
	return ret;
}

void MeshRD::_set_surfaces_data(const Array &p_surfaces) {
	clear_surfaces();
	for (int i = 0; i < p_surfaces.size(); i++) {
		Dictionary data = p_surfaces[i];
		ERR_FAIL_COND(!data.has("format"));
		ERR_FAIL_COND(!data.has("primitive"));
		ERR_FAIL_COND(!data.has("vertex_count"));
		ERR_FAIL_COND(!data.has("aabb"));

		_add_surface_storage_internal(
				BitField<ArrayFormat>(uint64_t(data["format"])),
				PrimitiveType(int(data["primitive"])),
				int(data["vertex_count"]),
				int(data.get("index_count", 0)),
				data["aabb"],
				data.get("material", Ref<Material>()),
				data.get("uv_scale", Vector4()),
				uint32_t(int(data.get("vertex_buffer_creation_bits", 0))),
				uint32_t(int(data.get("attribute_buffer_creation_bits", 0))),
				uint32_t(int(data.get("index_buffer_creation_bits", 0))),
				false);
	}
	if (!p_surfaces.is_empty()) {
		emit_changed();
	}
}

void MeshRD::_add_surface_storage_internal(BitField<ArrayFormat> p_format, PrimitiveType p_primitive, int p_vertex_count, int p_index_count, const AABB &p_aabb, const Ref<Material> &p_material, const Vector4 &p_uv_scale, uint32_t p_vertex_buffer_creation_bits, uint32_t p_attribute_buffer_creation_bits, uint32_t p_index_buffer_creation_bits, bool p_emit_changed) {
	ERR_FAIL_COND(surfaces.size() == RSE::MAX_MESH_SURFACES);
	ERR_FAIL_COND_MSG(p_vertex_count < 0, "MeshRD surface vertex count must be non-negative.");
	ERR_FAIL_COND_MSG(p_index_count < 0, "MeshRD surface index count must be non-negative.");

	RenderingDevice *rd = RS::get_singleton()->get_rendering_device();
	ERR_FAIL_NULL(rd);

	// Ensure current format version is set.
	p_format = BitField<ArrayFormat>((uint64_t(p_format) & ~(uint64_t(RSE::ARRAY_FLAG_FORMAT_VERSION_MASK) << RSE::ARRAY_FLAG_FORMAT_VERSION_SHIFT)) | RSE::ARRAY_FLAG_FORMAT_CURRENT_VERSION);

	BitField<RSE::ArrayFormat> rs_format = BitField<RSE::ArrayFormat>(uint64_t(p_format));
	const uint32_t vertex_stride = RS::get_singleton()->mesh_surface_get_format_vertex_stride(rs_format, p_vertex_count);
	const uint32_t normal_tangent_stride = RS::get_singleton()->mesh_surface_get_format_normal_tangent_stride(rs_format, p_vertex_count);
	const uint32_t attribute_stride = RS::get_singleton()->mesh_surface_get_format_attribute_stride(rs_format, p_vertex_count);
	const uint32_t index_stride = p_index_count > 0 ? RS::get_singleton()->mesh_surface_get_format_index_stride(rs_format, p_vertex_count) : 0;

	RID vertex_buffer;
	RID attribute_buffer;
	RID index_buffer;

	const uint32_t vertex_buffer_size = p_vertex_count * (vertex_stride + normal_tangent_stride);
	if (vertex_buffer_size > 0) {
		vertex_buffer = rd->vertex_buffer_create(vertex_buffer_size, Vector<uint8_t>(), RD::BufferCreationBits(p_vertex_buffer_creation_bits));
		ERR_FAIL_COND_MSG(!vertex_buffer.is_valid(), "Failed to create MeshRD vertex buffer from descriptor.");
	}
	const uint32_t attribute_buffer_size = p_vertex_count * attribute_stride;
	if (attribute_buffer_size > 0) {
		attribute_buffer = rd->vertex_buffer_create(attribute_buffer_size, Vector<uint8_t>(), RD::BufferCreationBits(p_attribute_buffer_creation_bits));
		if (!attribute_buffer.is_valid()) {
			if (vertex_buffer.is_valid()) {
				rd->free_rid(vertex_buffer);
			}
			ERR_FAIL_MSG("Failed to create MeshRD attribute buffer from descriptor.");
		}
	}
	if (p_index_count > 0) {
		RD::IndexBufferFormat index_format = index_stride == 2 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32;
		if (index_stride != 2 && index_stride != 4) {
			if (vertex_buffer.is_valid()) {
				rd->free_rid(vertex_buffer);
			}
			if (attribute_buffer.is_valid()) {
				rd->free_rid(attribute_buffer);
			}
			ERR_FAIL_MSG("MeshRD descriptor-based indexed surfaces require 16-bit or 32-bit indices.");
		}
		index_buffer = rd->index_buffer_create(p_index_count, index_format, Vector<uint8_t>(), false, RD::BufferCreationBits(p_index_buffer_creation_bits));
		if (!index_buffer.is_valid()) {
			if (vertex_buffer.is_valid()) {
				rd->free_rid(vertex_buffer);
			}
			if (attribute_buffer.is_valid()) {
				rd->free_rid(attribute_buffer);
			}
			ERR_FAIL_MSG("Failed to create MeshRD index buffer from descriptor.");
		}
	}

	_create_if_empty();

	Surface s;
	s.serializable = true;
	s.vertex_buffer_creation_bits = p_vertex_buffer_creation_bits;
	s.attribute_buffer_creation_bits = p_attribute_buffer_creation_bits;
	s.index_buffer_creation_bits = p_index_buffer_creation_bits;
	s.vertex_buffer = vertex_buffer;
	s.attribute_buffer = attribute_buffer;
	s.index_buffer = index_buffer;
	s.aabb = p_aabb;
	s.primitive = p_primitive;
	s.array_length = p_vertex_count;
	s.index_array_length = p_index_count;
	s.max_array_length = p_vertex_count;
	s.max_index_array_length = p_index_count;
	s.format = p_format;
	s.material = p_material;
	s.uv_scale = p_uv_scale;

	surfaces.push_back(s);
	_recompute_aabb();

	RenderingServerTypes::SurfaceDataRD sd;
	sd.format = p_format;
	sd.primitive = RSE::PrimitiveType(p_primitive);
	sd.vertex_count = p_vertex_count;
	sd.vertex_buffer = vertex_buffer;
	sd.owns_vertex_buffer = vertex_buffer.is_valid();
	sd.attribute_buffer = attribute_buffer;
	sd.owns_attribute_buffer = attribute_buffer.is_valid();
	sd.index_count = p_index_count;
	sd.index_buffer = index_buffer;
	sd.owns_index_buffer = index_buffer.is_valid();
	sd.aabb = p_aabb;
	sd.uv_scale = p_uv_scale;
	sd.material = p_material.is_valid() ? p_material->get_rid() : RID();

	RS::get_singleton()->mesh_add_surface_rd(mesh, sd);

	clear_cache();
	if (p_emit_changed) {
		emit_changed();
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

void MeshRD::add_surface(BitField<ArrayFormat> p_format, PrimitiveType p_primitive, int p_vertex_count, RID p_vertex_buffer, RID p_attribute_buffer, int p_index_count, RID p_index_buffer, const AABB &p_aabb, const Ref<Material> &p_material, const Vector4 &p_uv_scale) {
	ERR_FAIL_COND(surfaces.size() == RSE::MAX_MESH_SURFACES);
	ERR_FAIL_COND_MSG(get_blend_shape_count() != 0, "MeshRD does not support blend shapes.");

	// Ensure current format version is set.
	p_format = BitField<ArrayFormat>((uint64_t(p_format) & ~(uint64_t(RSE::ARRAY_FLAG_FORMAT_VERSION_MASK) << RSE::ARRAY_FLAG_FORMAT_VERSION_SHIFT)) | RSE::ARRAY_FLAG_FORMAT_CURRENT_VERSION);

	_create_if_empty();

	Surface s;
	s.serializable = false;
	s.vertex_buffer = p_vertex_buffer;
	s.attribute_buffer = p_attribute_buffer;
	s.index_buffer = p_index_buffer;
	s.aabb = p_aabb;
	s.primitive = p_primitive;
	s.array_length = p_vertex_count;
	s.index_array_length = p_index_count;
	s.max_array_length = p_vertex_count;
	s.max_index_array_length = p_index_count;
	s.format = p_format;
	s.material = p_material;
	s.uv_scale = p_uv_scale;

	surfaces.push_back(s);
	_recompute_aabb();

	RenderingServerTypes::SurfaceDataRD sd;
	sd.format = p_format;
	sd.primitive = RSE::PrimitiveType(p_primitive);
	sd.vertex_count = p_vertex_count;
	sd.vertex_buffer = p_vertex_buffer;
	sd.attribute_buffer = p_attribute_buffer;
	sd.index_count = p_index_count;
	sd.index_buffer = p_index_buffer;
	sd.aabb = p_aabb;
	sd.uv_scale = p_uv_scale;
	sd.material = p_material.is_valid() ? p_material->get_rid() : RID();

	RS::get_singleton()->mesh_add_surface_rd(mesh, sd);

	clear_cache();
	emit_changed();
}

void MeshRD::add_surface_storage(BitField<ArrayFormat> p_format, PrimitiveType p_primitive, int p_vertex_count, int p_index_count, const AABB &p_aabb, const Ref<Material> &p_material, const Vector4 &p_uv_scale, uint32_t p_vertex_buffer_creation_bits, uint32_t p_attribute_buffer_creation_bits, uint32_t p_index_buffer_creation_bits) {
	_add_surface_storage_internal(p_format, p_primitive, p_vertex_count, p_index_count, p_aabb, p_material, p_uv_scale, p_vertex_buffer_creation_bits, p_attribute_buffer_creation_bits, p_index_buffer_creation_bits, true);
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

void MeshRD::surface_set_active_range(int p_surface, int p_vertex_count, int p_index_count) {
	ERR_FAIL_INDEX(p_surface, surfaces.size());
	ERR_FAIL_COND_MSG(p_vertex_count < 0 || p_vertex_count > surfaces[p_surface].max_array_length, "Active vertex count exceeds MeshRD surface capacity.");
	ERR_FAIL_COND_MSG(p_index_count < 0 || p_index_count > surfaces[p_surface].max_index_array_length, "Active index count exceeds MeshRD surface capacity.");

	surfaces.write[p_surface].array_length = p_vertex_count;
	surfaces.write[p_surface].index_array_length = p_index_count;
	RS::get_singleton()->mesh_surface_set_active_range(mesh, p_surface, p_vertex_count, p_index_count);
	emit_changed();
}

void MeshRD::surface_set_indirect_buffer(int p_surface, RID p_indirect_buffer, int p_offset) {
	ERR_FAIL_INDEX(p_surface, surfaces.size());
	ERR_FAIL_COND_MSG(p_offset < 0, "Indirect buffer offset must be non-negative.");

	surfaces.write[p_surface].indirect_buffer = p_indirect_buffer;
	surfaces.write[p_surface].indirect_buffer_offset = p_offset;
	RS::get_singleton()->mesh_surface_set_indirect_buffer(mesh, p_surface, p_indirect_buffer, p_offset);
	emit_changed();
}

RID MeshRD::surface_get_vertex_buffer(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), RID());
	return surfaces[p_surface].vertex_buffer;
}

RID MeshRD::surface_get_attribute_buffer(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), RID());
	return surfaces[p_surface].attribute_buffer;
}

RID MeshRD::surface_get_index_buffer(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), RID());
	return surfaces[p_surface].index_buffer;
}

RID MeshRD::surface_get_indirect_buffer(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), RID());
	return surfaces[p_surface].indirect_buffer;
}

int MeshRD::surface_get_indirect_buffer_offset(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	return surfaces[p_surface].indirect_buffer_offset;
}

int MeshRD::surface_get_max_vertex_count(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	return surfaces[p_surface].max_array_length;
}

int MeshRD::surface_get_max_index_count(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	return surfaces[p_surface].max_index_array_length;
}

int MeshRD::surface_get_active_vertex_count(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	return surfaces[p_surface].array_length;
}

int MeshRD::surface_get_active_index_count(int p_surface) const {
	ERR_FAIL_INDEX_V(p_surface, surfaces.size(), 0);
	return surfaces[p_surface].index_array_length;
}

void MeshRD::surface_mark_dirty(int p_surface) {
	ERR_FAIL_INDEX(p_surface, surfaces.size());
	RS::get_singleton()->mesh_surface_mark_dirty(mesh, p_surface);
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
	return surfaces[p_idx].array_length;
}

int MeshRD::surface_get_array_index_len(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, surfaces.size(), -1);
	return surfaces[p_idx].index_array_length;
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
	ClassDB::bind_method(D_METHOD("add_surface", "format", "primitive", "vertex_count", "vertex_buffer", "attribute_buffer", "index_count", "index_buffer", "aabb", "material", "uv_scale"), &MeshRD::add_surface, DEFVAL(Ref<Material>()), DEFVAL(Vector4()));
	ClassDB::bind_method(D_METHOD("add_surface_storage", "format", "primitive", "vertex_count", "index_count", "aabb", "material", "uv_scale", "vertex_buffer_creation_bits", "attribute_buffer_creation_bits", "index_buffer_creation_bits"), &MeshRD::add_surface_storage, DEFVAL(Ref<Material>()), DEFVAL(Vector4()), DEFVAL(0), DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("_set_surfaces_data", "surfaces"), &MeshRD::_set_surfaces_data);
	ClassDB::bind_method(D_METHOD("_get_surfaces_data"), &MeshRD::_get_surfaces_data);
	ClassDB::bind_method(D_METHOD("clear_surfaces"), &MeshRD::clear_surfaces);
	ClassDB::bind_method(D_METHOD("surface_remove", "surf_idx"), &MeshRD::surface_remove);
	ClassDB::bind_method(D_METHOD("surface_set_active_range", "surf_idx", "vertex_count", "index_count"), &MeshRD::surface_set_active_range);
	ClassDB::bind_method(D_METHOD("surface_set_indirect_buffer", "surf_idx", "indirect_buffer", "offset"), &MeshRD::surface_set_indirect_buffer, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("surface_get_vertex_buffer", "surf_idx"), &MeshRD::surface_get_vertex_buffer);
	ClassDB::bind_method(D_METHOD("surface_get_attribute_buffer", "surf_idx"), &MeshRD::surface_get_attribute_buffer);
	ClassDB::bind_method(D_METHOD("surface_get_index_buffer", "surf_idx"), &MeshRD::surface_get_index_buffer);
	ClassDB::bind_method(D_METHOD("surface_get_indirect_buffer", "surf_idx"), &MeshRD::surface_get_indirect_buffer);
	ClassDB::bind_method(D_METHOD("surface_get_indirect_buffer_offset", "surf_idx"), &MeshRD::surface_get_indirect_buffer_offset);
	ClassDB::bind_method(D_METHOD("surface_get_max_vertex_count", "surf_idx"), &MeshRD::surface_get_max_vertex_count);
	ClassDB::bind_method(D_METHOD("surface_get_max_index_count", "surf_idx"), &MeshRD::surface_get_max_index_count);
	ClassDB::bind_method(D_METHOD("surface_get_active_vertex_count", "surf_idx"), &MeshRD::surface_get_active_vertex_count);
	ClassDB::bind_method(D_METHOD("surface_get_active_index_count", "surf_idx"), &MeshRD::surface_get_active_index_count);
	ClassDB::bind_method(D_METHOD("surface_mark_dirty", "surf_idx"), &MeshRD::surface_mark_dirty);
	ClassDB::bind_method(D_METHOD("set_custom_aabb", "aabb"), &MeshRD::set_custom_aabb);
	ClassDB::bind_method(D_METHOD("get_custom_aabb"), &MeshRD::get_custom_aabb);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "_surfaces_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "_set_surfaces_data", "_get_surfaces_data");
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
