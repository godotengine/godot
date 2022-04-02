/*************************************************************************/
/*  mesh_storage.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef MESH_STORAGE_GLES3_H
#define MESH_STORAGE_GLES3_H

#ifdef GLES3_ENABLED

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "servers/rendering/storage/mesh_storage.h"

namespace GLES3 {

class MeshStorage : public RendererMeshStorage {
private:
	static MeshStorage *singleton;

public:
	static MeshStorage *get_singleton();

	MeshStorage();
	virtual ~MeshStorage();

	/* MESH API */

	virtual RID mesh_allocate() override;
	virtual void mesh_initialize(RID p_rid) override;
	virtual void mesh_free(RID p_rid) override;

	virtual void mesh_set_blend_shape_count(RID p_mesh, int p_blend_shape_count) override;
	virtual bool mesh_needs_instance(RID p_mesh, bool p_has_skeleton) override;

	virtual void mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) override;

	virtual int mesh_get_blend_shape_count(RID p_mesh) const override;

	virtual void mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) override;
	virtual RS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const override;

	virtual void mesh_surface_update_vertex_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) override;
	virtual void mesh_surface_update_attribute_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) override;
	virtual void mesh_surface_update_skin_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) override;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) override;
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const override;

	virtual RS::SurfaceData mesh_get_surface(RID p_mesh, int p_surface) const override;
	virtual int mesh_get_surface_count(RID p_mesh) const override;

	virtual void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) override;
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const override;

	virtual AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID()) override;
	virtual void mesh_set_shadow_mesh(RID p_mesh, RID p_shadow_mesh) override;
	virtual void mesh_clear(RID p_mesh) override;

	/* MESH INSTANCE API */

	virtual RID mesh_instance_create(RID p_base) override;
	virtual void mesh_instance_free(RID p_rid) override;
	virtual void mesh_instance_set_skeleton(RID p_mesh_instance, RID p_skeleton) override;
	virtual void mesh_instance_set_blend_shape_weight(RID p_mesh_instance, int p_shape, float p_weight) override;
	virtual void mesh_instance_check_for_update(RID p_mesh_instance) override;
	virtual void update_mesh_instances() override;

	/* MULTIMESH API */

	struct MultiMesh {
		RID mesh;
		int instances = 0;
		RS::MultimeshTransformFormat xform_format = RS::MULTIMESH_TRANSFORM_3D;
		bool uses_colors = false;
		bool uses_custom_data = false;
		int visible_instances = -1;
		AABB aabb;
		bool aabb_dirty = false;
		bool buffer_set = false;
		uint32_t stride_cache = 0;
		uint32_t color_offset_cache = 0;
		uint32_t custom_data_offset_cache = 0;

		Vector<float> data_cache; //used if individual setting is used
		bool *data_cache_dirty_regions = nullptr;
		uint32_t data_cache_used_dirty_regions = 0;

		RID buffer; //storage buffer
		RID uniform_set_3d;
		RID uniform_set_2d;

		bool dirty = false;
		MultiMesh *dirty_list = nullptr;

		RendererStorage::Dependency dependency;
	};

	mutable RID_Owner<MultiMesh, true> multimesh_owner;

	MultiMesh *multimesh_dirty_list = nullptr;

	_FORCE_INLINE_ void _multimesh_make_local(MultiMesh *multimesh) const;
	_FORCE_INLINE_ void _multimesh_mark_dirty(MultiMesh *multimesh, int p_index, bool p_aabb);
	_FORCE_INLINE_ void _multimesh_mark_all_dirty(MultiMesh *multimesh, bool p_data, bool p_aabb);
	_FORCE_INLINE_ void _multimesh_re_create_aabb(MultiMesh *multimesh, const float *p_data, int p_instances);
	void _update_dirty_multimeshes();

	virtual RID multimesh_allocate() override;
	virtual void multimesh_initialize(RID p_rid) override;
	virtual void multimesh_free(RID p_rid) override;
	virtual void multimesh_allocate_data(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors = false, bool p_use_custom_data = false) override;
	virtual int multimesh_get_instance_count(RID p_multimesh) const override;

	virtual void multimesh_set_mesh(RID p_multimesh, RID p_mesh) override;
	virtual void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform3D &p_transform) override;
	virtual void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) override;
	virtual void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) override;
	virtual void multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) override;

	virtual RID multimesh_get_mesh(RID p_multimesh) const override;
	virtual AABB multimesh_get_aabb(RID p_multimesh) const override;

	virtual Transform3D multimesh_instance_get_transform(RID p_multimesh, int p_index) const override;
	virtual Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const override;
	virtual Color multimesh_instance_get_color(RID p_multimesh, int p_index) const override;
	virtual Color multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const override;
	virtual void multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) override;
	virtual Vector<float> multimesh_get_buffer(RID p_multimesh) const override;

	virtual void multimesh_set_visible_instances(RID p_multimesh, int p_visible) override;
	virtual int multimesh_get_visible_instances(RID p_multimesh) const override;

	_FORCE_INLINE_ RS::MultimeshTransformFormat multimesh_get_transform_format(RID p_multimesh) const {
		MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
		return multimesh->xform_format;
	}

	_FORCE_INLINE_ bool multimesh_uses_colors(RID p_multimesh) const {
		MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
		return multimesh->uses_colors;
	}

	_FORCE_INLINE_ bool multimesh_uses_custom_data(RID p_multimesh) const {
		MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
		return multimesh->uses_custom_data;
	}

	_FORCE_INLINE_ uint32_t multimesh_get_instances_to_draw(RID p_multimesh) const {
		MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
		if (multimesh->visible_instances >= 0) {
			return multimesh->visible_instances;
		}
		return multimesh->instances;
	}

	/* SKELETON API */

	virtual RID skeleton_allocate() override;
	virtual void skeleton_initialize(RID p_rid) override;
	virtual void skeleton_free(RID p_rid) override;

	virtual void skeleton_allocate_data(RID p_skeleton, int p_bones, bool p_2d_skeleton = false) override;
	virtual void skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) override;
	virtual int skeleton_get_bone_count(RID p_skeleton) const override;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform3D &p_transform) override;
	virtual Transform3D skeleton_bone_get_transform(RID p_skeleton, int p_bone) const override;
	virtual void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) override;
	virtual Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const override;

	virtual void skeleton_update_dependency(RID p_base, RendererStorage::DependencyTracker *p_instance) override;
};

} // namespace GLES3

#endif // GLES3_ENABLED

#endif // !MESH_STORAGE_GLES3_H
