/*************************************************************************/
/*  light_storage.cpp                                                    */
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

#include "light_storage.h"
#include "config.h"

using namespace GLES3;

LightStorage *LightStorage::singleton = nullptr;

LightStorage *LightStorage::get_singleton() {
	return singleton;
}

LightStorage::LightStorage() {
	singleton = this;
}

LightStorage::~LightStorage() {
	singleton = nullptr;
}

/* Light API */

RID LightStorage::directional_light_allocate() {
	return RID();
}

void LightStorage::directional_light_initialize(RID p_rid) {
}

RID LightStorage::omni_light_allocate() {
	return RID();
}

void LightStorage::omni_light_initialize(RID p_rid) {
}

RID LightStorage::spot_light_allocate() {
	return RID();
}

void LightStorage::spot_light_initialize(RID p_rid) {
}

void LightStorage::light_free(RID p_rid) {
}

void LightStorage::light_set_color(RID p_light, const Color &p_color) {
}

void LightStorage::light_set_param(RID p_light, RS::LightParam p_param, float p_value) {
}

void LightStorage::light_set_shadow(RID p_light, bool p_enabled) {
}

void LightStorage::light_set_projector(RID p_light, RID p_texture) {
}

void LightStorage::light_set_negative(RID p_light, bool p_enable) {
}

void LightStorage::light_set_cull_mask(RID p_light, uint32_t p_mask) {
}

void LightStorage::light_set_distance_fade(RID p_light, bool p_enabled, float p_begin, float p_shadow, float p_length) {
}

void LightStorage::light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {
}

void LightStorage::light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) {
}

void LightStorage::light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) {
}

void LightStorage::light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) {
}

void LightStorage::light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) {
}

void LightStorage::light_directional_set_blend_splits(RID p_light, bool p_enable) {
}

bool LightStorage::light_directional_get_blend_splits(RID p_light) const {
	return false;
}

void LightStorage::light_directional_set_sky_mode(RID p_light, RS::LightDirectionalSkyMode p_mode) {
}

RS::LightDirectionalSkyMode LightStorage::light_directional_get_sky_mode(RID p_light) const {
	return RS::LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_AND_SKY;
}

RS::LightDirectionalShadowMode LightStorage::light_directional_get_shadow_mode(RID p_light) {
	return RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL;
}

RS::LightOmniShadowMode LightStorage::light_omni_get_shadow_mode(RID p_light) {
	return RS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID;
}

bool LightStorage::light_has_shadow(RID p_light) const {
	return false;
}

bool LightStorage::light_has_projector(RID p_light) const {
	return false;
}

RS::LightType LightStorage::light_get_type(RID p_light) const {
	return RS::LIGHT_OMNI;
}

AABB LightStorage::light_get_aabb(RID p_light) const {
	return AABB();
}

float LightStorage::light_get_param(RID p_light, RS::LightParam p_param) {
	return 0.0;
}

Color LightStorage::light_get_color(RID p_light) {
	return Color();
}

RS::LightBakeMode LightStorage::light_get_bake_mode(RID p_light) {
	return RS::LIGHT_BAKE_DISABLED;
}

uint32_t LightStorage::light_get_max_sdfgi_cascade(RID p_light) {
	return 0;
}

uint64_t LightStorage::light_get_version(RID p_light) const {
	return 0;
}

/* PROBE API */

RID LightStorage::reflection_probe_allocate() {
	return RID();
}

void LightStorage::reflection_probe_initialize(RID p_rid) {
}

void LightStorage::reflection_probe_free(RID p_rid) {
}

void LightStorage::reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) {
}

void LightStorage::reflection_probe_set_intensity(RID p_probe, float p_intensity) {
}

void LightStorage::reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) {
}

void LightStorage::reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) {
}

void LightStorage::reflection_probe_set_ambient_energy(RID p_probe, float p_energy) {
}

void LightStorage::reflection_probe_set_max_distance(RID p_probe, float p_distance) {
}

void LightStorage::reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) {
}

void LightStorage::reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) {
}

void LightStorage::reflection_probe_set_as_interior(RID p_probe, bool p_enable) {
}

void LightStorage::reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) {
}

void LightStorage::reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) {
}

void LightStorage::reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) {
}

void LightStorage::reflection_probe_set_resolution(RID p_probe, int p_resolution) {
}

AABB LightStorage::reflection_probe_get_aabb(RID p_probe) const {
	return AABB();
}

RS::ReflectionProbeUpdateMode LightStorage::reflection_probe_get_update_mode(RID p_probe) const {
	return RenderingServer::REFLECTION_PROBE_UPDATE_ONCE;
}

uint32_t LightStorage::reflection_probe_get_cull_mask(RID p_probe) const {
	return 0;
}

Vector3 LightStorage::reflection_probe_get_extents(RID p_probe) const {
	return Vector3();
}

Vector3 LightStorage::reflection_probe_get_origin_offset(RID p_probe) const {
	return Vector3();
}

float LightStorage::reflection_probe_get_origin_max_distance(RID p_probe) const {
	return 0.0;
}

bool LightStorage::reflection_probe_renders_shadows(RID p_probe) const {
	return false;
}

void LightStorage::reflection_probe_set_mesh_lod_threshold(RID p_probe, float p_ratio) {
}

float LightStorage::reflection_probe_get_mesh_lod_threshold(RID p_probe) const {
	return 0.0;
}

/* LIGHTMAP CAPTURE */

RID LightStorage::lightmap_allocate() {
	return RID();
}

void LightStorage::lightmap_initialize(RID p_rid) {
}

void LightStorage::lightmap_free(RID p_rid) {
}

void LightStorage::lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) {
}

void LightStorage::lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) {
}

void LightStorage::lightmap_set_probe_interior(RID p_lightmap, bool p_interior) {
}

void LightStorage::lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) {
}

PackedVector3Array LightStorage::lightmap_get_probe_capture_points(RID p_lightmap) const {
	return PackedVector3Array();
}

PackedColorArray LightStorage::lightmap_get_probe_capture_sh(RID p_lightmap) const {
	return PackedColorArray();
}

PackedInt32Array LightStorage::lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const {
	return PackedInt32Array();
}

PackedInt32Array LightStorage::lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const {
	return PackedInt32Array();
}

AABB LightStorage::lightmap_get_aabb(RID p_lightmap) const {
	return AABB();
}

void LightStorage::lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) {
}

bool LightStorage::lightmap_is_interior(RID p_lightmap) const {
	return false;
}

void LightStorage::lightmap_set_probe_capture_update_speed(float p_speed) {
}

float LightStorage::lightmap_get_probe_capture_update_speed() const {
	return 0;
}

#endif // !GLES3_ENABLED
