/*************************************************************************/
/*  mesh_storage.cpp                                                     */
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

#ifdef GLES3_ENABLED

#include "mesh_storage.h"

using namespace GLES3;

MeshStorage *MeshStorage::singleton = nullptr;

MeshStorage *MeshStorage::get_singleton() {
	return singleton;
}

MeshStorage::MeshStorage() {
	singleton = this;
}

MeshStorage::~MeshStorage() {
	singleton = nullptr;
}

/* MESH API */

RID MeshStorage::mesh_allocate() {
	return RID();
}

void MeshStorage::mesh_initialize(RID p_rid) {
}

void MeshStorage::mesh_free(RID p_rid) {
}

void MeshStorage::mesh_set_blend_shape_count(RID p_mesh, int p_blend_shape_count) {
}

bool MeshStorage::mesh_needs_instance(RID p_mesh, bool p_has_skeleton) {
	return false;
}

void MeshStorage::mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) {
}

int MeshStorage::mesh_get_blend_shape_count(RID p_mesh) const {
	return 0;
}

void MeshStorage::mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) {
}

RS::BlendShapeMode MeshStorage::mesh_get_blend_shape_mode(RID p_mesh) const {
	return RS::BLEND_SHAPE_MODE_NORMALIZED;
}

void MeshStorage::mesh_surface_update_vertex_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
}

void MeshStorage::mesh_surface_update_attribute_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
}

void MeshStorage::mesh_surface_update_skin_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
}

void MeshStorage::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) {
}

RID MeshStorage::mesh_surface_get_material(RID p_mesh, int p_surface) const {
	return RID();
}

RS::SurfaceData MeshStorage::mesh_get_surface(RID p_mesh, int p_surface) const {
	return RS::SurfaceData();
}

int MeshStorage::mesh_get_surface_count(RID p_mesh) const {
	return 1;
}

void MeshStorage::mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) {
}

AABB MeshStorage::mesh_get_custom_aabb(RID p_mesh) const {
	return AABB();
}

AABB MeshStorage::mesh_get_aabb(RID p_mesh, RID p_skeleton) {
	return AABB();
}

void MeshStorage::mesh_set_shadow_mesh(RID p_mesh, RID p_shadow_mesh) {
}

void MeshStorage::mesh_clear(RID p_mesh) {
}

/* MESH INSTANCE API */

RID MeshStorage::mesh_instance_create(RID p_base) {
	return RID();
}

void MeshStorage::mesh_instance_free(RID p_rid) {
}

void MeshStorage::mesh_instance_set_skeleton(RID p_mesh_instance, RID p_skeleton) {
}

void MeshStorage::mesh_instance_set_blend_shape_weight(RID p_mesh_instance, int p_shape, float p_weight) {
}

void MeshStorage::mesh_instance_check_for_update(RID p_mesh_instance) {
}

void MeshStorage::update_mesh_instances() {
}

/* MULTIMESH API */

RID MeshStorage::multimesh_allocate() {
	return RID();
}

void MeshStorage::multimesh_initialize(RID p_rid) {
}

void MeshStorage::multimesh_free(RID p_rid) {
}

void MeshStorage::multimesh_allocate_data(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors, bool p_use_custom_data) {
}

int MeshStorage::multimesh_get_instance_count(RID p_multimesh) const {
	return 0;
}

void MeshStorage::multimesh_set_mesh(RID p_multimesh, RID p_mesh) {
}

void MeshStorage::multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform3D &p_transform) {
}

void MeshStorage::multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) {
}

void MeshStorage::multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {
}

void MeshStorage::multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) {
}

RID MeshStorage::multimesh_get_mesh(RID p_multimesh) const {
	return RID();
}

AABB MeshStorage::multimesh_get_aabb(RID p_multimesh) const {
	return AABB();
}

Transform3D MeshStorage::multimesh_instance_get_transform(RID p_multimesh, int p_index) const {
	return Transform3D();
}

Transform2D MeshStorage::multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const {
	return Transform2D();
}

Color MeshStorage::multimesh_instance_get_color(RID p_multimesh, int p_index) const {
	return Color();
}

Color MeshStorage::multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const {
	return Color();
}

void MeshStorage::multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) {
}

Vector<float> MeshStorage::multimesh_get_buffer(RID p_multimesh) const {
	return Vector<float>();
}

void MeshStorage::multimesh_set_visible_instances(RID p_multimesh, int p_visible) {
}

int MeshStorage::multimesh_get_visible_instances(RID p_multimesh) const {
	return 0;
}

/* SKELETON API */

RID MeshStorage::skeleton_allocate() {
	return RID();
}

void MeshStorage::skeleton_initialize(RID p_rid) {
}

void MeshStorage::skeleton_free(RID p_rid) {
}

void MeshStorage::skeleton_allocate_data(RID p_skeleton, int p_bones, bool p_2d_skeleton) {
}

void MeshStorage::skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) {
}

int MeshStorage::skeleton_get_bone_count(RID p_skeleton) const {
	return 0;
}

void MeshStorage::skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform3D &p_transform) {
}

Transform3D MeshStorage::skeleton_bone_get_transform(RID p_skeleton, int p_bone) const {
	return Transform3D();
}

void MeshStorage::skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) {
}

Transform2D MeshStorage::skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const {
	return Transform2D();
}

void MeshStorage::skeleton_update_dependency(RID p_base, RendererStorage::DependencyTracker *p_instance) {
}

#endif // GLES3_ENABLED
