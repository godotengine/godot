/**************************************************************************/
/*  mesh_storage.h                                                        */
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

#ifndef MESH_STORAGE_H
#define MESH_STORAGE_H

#include "servers/rendering_server.h"
#include "utilities.h"

class RendererMeshStorage {
public:
	virtual ~RendererMeshStorage() {}

	/* MESH API */

	virtual RID mesh_allocate() = 0;
	virtual void mesh_initialize(RID p_rid) = 0;
	virtual void mesh_free(RID p_rid) = 0;

	virtual void mesh_set_blend_shape_count(RID p_mesh, int p_blend_shape_count) = 0;

	/// Returns stride
	virtual void mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) = 0;

	virtual int mesh_get_blend_shape_count(RID p_mesh) const = 0;

	virtual void mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) = 0;
	virtual RS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const = 0;

	virtual void mesh_surface_update_vertex_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) = 0;
	virtual void mesh_surface_update_attribute_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) = 0;
	virtual void mesh_surface_update_skin_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) = 0;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) = 0;
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const = 0;

	virtual RS::SurfaceData mesh_get_surface(RID p_mesh, int p_surface) const = 0;

	virtual int mesh_get_surface_count(RID p_mesh) const = 0;

	virtual void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) = 0;
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const = 0;
	virtual AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID()) = 0;

	virtual void mesh_set_path(RID p_mesh, const String &p_path) = 0;
	virtual String mesh_get_path(RID p_mesh) const = 0;

	virtual void mesh_set_shadow_mesh(RID p_mesh, RID p_shadow_mesh) = 0;

	virtual void mesh_clear(RID p_mesh) = 0;

	virtual bool mesh_needs_instance(RID p_mesh, bool p_has_skeleton) = 0;

	/* MESH INSTANCE */

	virtual RID mesh_instance_create(RID p_base) = 0;
	virtual void mesh_instance_free(RID p_rid) = 0;
	virtual void mesh_instance_set_skeleton(RID p_mesh_instance, RID p_skeleton) = 0;
	virtual void mesh_instance_set_blend_shape_weight(RID p_mesh_instance, int p_shape, float p_weight) = 0;
	virtual void mesh_instance_check_for_update(RID p_mesh_instance) = 0;
	virtual void mesh_instance_set_canvas_item_transform(RID p_mesh_instance, const Transform2D &p_transform) = 0;
	virtual void update_mesh_instances() = 0;

	/* MULTIMESH API */

	virtual RID multimesh_allocate() = 0;
	virtual void multimesh_initialize(RID p_rid) = 0;
	virtual void multimesh_free(RID p_rid) = 0;

	virtual void multimesh_allocate_data(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors = false, bool p_use_custom_data = false) = 0;

	virtual int multimesh_get_instance_count(RID p_multimesh) const = 0;

	virtual void multimesh_set_mesh(RID p_multimesh, RID p_mesh) = 0;
	virtual void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform3D &p_transform) = 0;
	virtual void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) = 0;
	virtual void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) = 0;
	virtual void multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) = 0;

	virtual RID multimesh_get_mesh(RID p_multimesh) const = 0;

	virtual Transform3D multimesh_instance_get_transform(RID p_multimesh, int p_index) const = 0;
	virtual Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const = 0;
	virtual Color multimesh_instance_get_color(RID p_multimesh, int p_index) const = 0;
	virtual Color multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const = 0;

	virtual void multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) = 0;
	virtual Vector<float> multimesh_get_buffer(RID p_multimesh) const = 0;

	virtual void multimesh_set_visible_instances(RID p_multimesh, int p_visible) = 0;
	virtual int multimesh_get_visible_instances(RID p_multimesh) const = 0;

	virtual AABB multimesh_get_aabb(RID p_multimesh) const = 0;

	/* SKELETON API */

	virtual RID skeleton_allocate() = 0;
	virtual void skeleton_initialize(RID p_rid) = 0;
	virtual void skeleton_free(RID p_rid) = 0;

	virtual void skeleton_allocate_data(RID p_skeleton, int p_bones, bool p_2d_skeleton = false) = 0;
	virtual int skeleton_get_bone_count(RID p_skeleton) const = 0;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform3D &p_transform) = 0;
	virtual Transform3D skeleton_bone_get_transform(RID p_skeleton, int p_bone) const = 0;
	virtual void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) = 0;
	virtual Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const = 0;
	virtual void skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) = 0;

	virtual void skeleton_update_dependency(RID p_base, DependencyTracker *p_instance) = 0;
};

#endif // MESH_STORAGE_H
