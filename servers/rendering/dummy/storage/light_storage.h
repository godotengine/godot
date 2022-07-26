/*************************************************************************/
/*  light_storage.h                                                      */
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

#ifndef LIGHT_STORAGE_DUMMY_H
#define LIGHT_STORAGE_DUMMY_H

#include "servers/rendering/storage/light_storage.h"

namespace RendererDummy {

class LightStorage : public RendererLightStorage {
public:
	/* Light API */

	virtual RID directional_light_allocate() override { return RID(); }
	virtual void directional_light_initialize(RID p_rid) override {}
	virtual RID omni_light_allocate() override { return RID(); }
	virtual void omni_light_initialize(RID p_rid) override {}
	virtual RID spot_light_allocate() override { return RID(); }
	virtual void spot_light_initialize(RID p_rid) override {}

	virtual void light_free(RID p_rid) override {}

	virtual void light_set_color(RID p_light, const Color &p_color) override {}
	virtual void light_set_param(RID p_light, RS::LightParam p_param, float p_value) override {}
	virtual void light_set_shadow(RID p_light, bool p_enabled) override {}
	virtual void light_set_projector(RID p_light, RID p_texture) override {}
	virtual void light_set_negative(RID p_light, bool p_enable) override {}
	virtual void light_set_cull_mask(RID p_light, uint32_t p_mask) override {}
	virtual void light_set_distance_fade(RID p_light, bool p_enabled, float p_begin, float p_shadow, float p_length) override {}
	virtual void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) override {}
	virtual void light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) override {}
	virtual void light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) override {}

	virtual void light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) override {}

	virtual void light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) override {}
	virtual void light_directional_set_blend_splits(RID p_light, bool p_enable) override {}
	virtual bool light_directional_get_blend_splits(RID p_light) const override { return false; }
	virtual void light_directional_set_sky_mode(RID p_light, RS::LightDirectionalSkyMode p_mode) override {}
	virtual RS::LightDirectionalSkyMode light_directional_get_sky_mode(RID p_light) const override { return RS::LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_AND_SKY; }

	virtual RS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) override { return RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL; }
	virtual RS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) override { return RS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID; }

	virtual bool light_has_shadow(RID p_light) const override { return false; }
	virtual bool light_has_projector(RID p_light) const override { return false; }

	virtual RS::LightType light_get_type(RID p_light) const override { return RS::LIGHT_OMNI; }
	virtual AABB light_get_aabb(RID p_light) const override { return AABB(); }
	virtual float light_get_param(RID p_light, RS::LightParam p_param) override { return 0.0; }
	virtual Color light_get_color(RID p_light) override { return Color(); }
	virtual RS::LightBakeMode light_get_bake_mode(RID p_light) override { return RS::LIGHT_BAKE_DISABLED; }
	virtual uint32_t light_get_max_sdfgi_cascade(RID p_light) override { return 0; }
	virtual uint64_t light_get_version(RID p_light) const override { return 0; }

	/* PROBE API */
	virtual RID reflection_probe_allocate() override { return RID(); }
	virtual void reflection_probe_initialize(RID p_rid) override {}
	virtual void reflection_probe_free(RID p_rid) override {}

	virtual void reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) override {}
	virtual void reflection_probe_set_intensity(RID p_probe, float p_intensity) override {}
	virtual void reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) override {}
	virtual void reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) override {}
	virtual void reflection_probe_set_ambient_energy(RID p_probe, float p_energy) override {}
	virtual void reflection_probe_set_max_distance(RID p_probe, float p_distance) override {}
	virtual void reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) override {}
	virtual void reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) override {}
	virtual void reflection_probe_set_as_interior(RID p_probe, bool p_enable) override {}
	virtual void reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) override {}
	virtual void reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) override {}
	virtual void reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) override {}
	virtual void reflection_probe_set_resolution(RID p_probe, int p_resolution) override {}
	virtual void reflection_probe_set_mesh_lod_threshold(RID p_probe, float p_ratio) override {}
	virtual float reflection_probe_get_mesh_lod_threshold(RID p_probe) const override { return 0.0; }

	virtual AABB reflection_probe_get_aabb(RID p_probe) const override { return AABB(); }
	virtual RS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const override { return RenderingServer::REFLECTION_PROBE_UPDATE_ONCE; }
	virtual uint32_t reflection_probe_get_cull_mask(RID p_probe) const override { return 0; }
	virtual Vector3 reflection_probe_get_extents(RID p_probe) const override { return Vector3(); }
	virtual Vector3 reflection_probe_get_origin_offset(RID p_probe) const override { return Vector3(); }
	virtual float reflection_probe_get_origin_max_distance(RID p_probe) const override { return 0.0; }
	virtual bool reflection_probe_renders_shadows(RID p_probe) const override { return false; }

	/* LIGHTMAP CAPTURE */
	virtual RID lightmap_allocate() override { return RID(); }
	virtual void lightmap_initialize(RID p_rid) override {}
	virtual void lightmap_free(RID p_rid) override {}

	virtual void lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) override {}
	virtual void lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) override {}
	virtual void lightmap_set_probe_interior(RID p_lightmap, bool p_interior) override {}
	virtual void lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) override {}
	virtual PackedVector3Array lightmap_get_probe_capture_points(RID p_lightmap) const override { return PackedVector3Array(); }
	virtual PackedColorArray lightmap_get_probe_capture_sh(RID p_lightmap) const override { return PackedColorArray(); }
	virtual PackedInt32Array lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const override { return PackedInt32Array(); }
	virtual PackedInt32Array lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const override { return PackedInt32Array(); }
	virtual AABB lightmap_get_aabb(RID p_lightmap) const override { return AABB(); }
	virtual void lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) override {}
	virtual bool lightmap_is_interior(RID p_lightmap) const override { return false; }
	virtual void lightmap_set_probe_capture_update_speed(float p_speed) override {}
	virtual float lightmap_get_probe_capture_update_speed() const override { return 0; }
};

} // namespace RendererDummy

#endif // LIGHT_STORAGE_DUMMY_H
