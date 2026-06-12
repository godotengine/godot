/**************************************************************************/
/*  mesh_rd.h                                                             */
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

#pragma once

#include "scene/resources/mesh.h"

class MeshRD : public Mesh {
	GDCLASS(MeshRD, Mesh);
	RES_BASE_EXTENSION("mesh");

	struct Surface {
		uint64_t format = 0;
		int array_length = 0;
		int index_array_length = 0;
		int max_array_length = 0;
		int max_index_array_length = 0;
		bool serializable = false;
		uint32_t vertex_buffer_creation_bits = 0;
		uint32_t attribute_buffer_creation_bits = 0;
		uint32_t index_buffer_creation_bits = 0;
		RID vertex_buffer;
		RID attribute_buffer;
		RID index_buffer;
		RID indirect_buffer;
		int indirect_buffer_offset = 0;
		PrimitiveType primitive = PrimitiveType::PRIMITIVE_MAX;
		AABB aabb;
		Vector4 uv_scale;
		Ref<Material> material;
	};

	Vector<Surface> surfaces;
	mutable RID mesh;
	AABB aabb;
	AABB custom_aabb;

	void _create_if_empty() const;
	void _recompute_aabb();
	Array _get_surfaces_data() const;
	void _set_surfaces_data(const Array &p_surfaces);
	void _add_surface_storage_internal(BitField<ArrayFormat> p_format, PrimitiveType p_primitive, int p_vertex_count, int p_index_count, const AABB &p_aabb, const Ref<Material> &p_material, const Vector4 &p_uv_scale, uint32_t p_vertex_buffer_creation_bits, uint32_t p_attribute_buffer_creation_bits, uint32_t p_index_buffer_creation_bits, bool p_emit_changed);

protected:
	static void _bind_methods();

public:
	void add_surface(BitField<ArrayFormat> p_format, PrimitiveType p_primitive, int p_vertex_count, RID p_vertex_buffer, RID p_attribute_buffer, int p_index_count, RID p_index_buffer, const AABB &p_aabb, const Ref<Material> &p_material = Ref<Material>(), const Vector4 &p_uv_scale = Vector4());
	void add_surface_storage(BitField<ArrayFormat> p_format, PrimitiveType p_primitive, int p_vertex_count, int p_index_count, const AABB &p_aabb, const Ref<Material> &p_material = Ref<Material>(), const Vector4 &p_uv_scale = Vector4(), uint32_t p_vertex_buffer_creation_bits = 0, uint32_t p_attribute_buffer_creation_bits = 0, uint32_t p_index_buffer_creation_bits = 0);
	void clear_surfaces();
	void surface_remove(int p_surface);
	void surface_set_active_range(int p_surface, int p_vertex_count, int p_index_count);
	void surface_set_indirect_buffer(int p_surface, RID p_indirect_buffer, int p_offset = 0);
	RID surface_get_vertex_buffer(int p_surface) const;
	RID surface_get_attribute_buffer(int p_surface) const;
	RID surface_get_index_buffer(int p_surface) const;
	RID surface_get_indirect_buffer(int p_surface) const;
	int surface_get_indirect_buffer_offset(int p_surface) const;
	int surface_get_max_vertex_count(int p_surface) const;
	int surface_get_max_index_count(int p_surface) const;
	int surface_get_active_vertex_count(int p_surface) const;
	int surface_get_active_index_count(int p_surface) const;
	void surface_mark_dirty(int p_surface);

	Array surface_get_arrays(int p_surface) const override;
	TypedArray<Array> surface_get_blend_shape_arrays(int p_surface) const override;
	Dictionary surface_get_lods(int p_surface) const override;

	int get_surface_count() const override;
	int surface_get_array_len(int p_idx) const override;
	int surface_get_array_index_len(int p_idx) const override;
	BitField<ArrayFormat> surface_get_format(int p_idx) const override;
	PrimitiveType surface_get_primitive_type(int p_idx) const override;
	void surface_set_material(int p_idx, const Ref<Material> &p_material) override;
	Ref<Material> surface_get_material(int p_idx) const override;

	int get_blend_shape_count() const override;
	StringName get_blend_shape_name(int p_index) const override;
	void set_blend_shape_name(int p_index, const StringName &p_name) override;

	void set_custom_aabb(const AABB &p_custom);
	AABB get_custom_aabb() const;

	AABB get_aabb() const override;
	RID get_rid() const override;

	MeshRD();
	~MeshRD();
};
