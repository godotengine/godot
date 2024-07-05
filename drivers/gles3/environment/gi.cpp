/**************************************************************************/
/*  gi.cpp                                                                */
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

#ifdef GLES3_ENABLED

#include "gi.h"

using namespace GLES3;

/* VOXEL GI API */

RID GI::voxel_gi_allocate() {
	return RID();
}

void GI::voxel_gi_free(RID p_rid) {
}

void GI::voxel_gi_initialize(RID p_rid) {
}

void GI::voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) {
}

AABB GI::voxel_gi_get_bounds(RID p_voxel_gi) const {
	return AABB();
}

Vector3i GI::voxel_gi_get_octree_size(RID p_voxel_gi) const {
	return Vector3i();
}

Vector<uint8_t> GI::voxel_gi_get_octree_cells(RID p_voxel_gi) const {
	return Vector<uint8_t>();
}

Vector<uint8_t> GI::voxel_gi_get_data_cells(RID p_voxel_gi) const {
	return Vector<uint8_t>();
}

Vector<uint8_t> GI::voxel_gi_get_distance_field(RID p_voxel_gi) const {
	return Vector<uint8_t>();
}

Vector<int> GI::voxel_gi_get_level_counts(RID p_voxel_gi) const {
	return Vector<int>();
}

Transform3D GI::voxel_gi_get_to_cell_xform(RID p_voxel_gi) const {
	return Transform3D();
}

void GI::voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) {
}

float GI::voxel_gi_get_dynamic_range(RID p_voxel_gi) const {
	return 0;
}

void GI::voxel_gi_set_propagation(RID p_voxel_gi, float p_range) {
}

float GI::voxel_gi_get_propagation(RID p_voxel_gi) const {
	return 0;
}

void GI::voxel_gi_set_energy(RID p_voxel_gi, float p_range) {
}

float GI::voxel_gi_get_energy(RID p_voxel_gi) const {
	return 0.0;
}

void GI::voxel_gi_set_baked_exposure_normalization(RID p_voxel_gi, float p_baked_exposure) {
}

float GI::voxel_gi_get_baked_exposure_normalization(RID p_voxel_gi) const {
	return 1.0;
}

void GI::voxel_gi_set_bias(RID p_voxel_gi, float p_range) {
}

float GI::voxel_gi_get_bias(RID p_voxel_gi) const {
	return 0.0;
}

void GI::voxel_gi_set_normal_bias(RID p_voxel_gi, float p_range) {
}

float GI::voxel_gi_get_normal_bias(RID p_voxel_gi) const {
	return 0.0;
}

void GI::voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) {
}

bool GI::voxel_gi_is_interior(RID p_voxel_gi) const {
	return false;
}

void GI::voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) {
}

bool GI::voxel_gi_is_using_two_bounces(RID p_voxel_gi) const {
	return false;
}

uint32_t GI::voxel_gi_get_version(RID p_voxel_gi) const {
	return 0;
}

void GI::hddagi_reset() {
}

#endif // GLES3_ENABLED
