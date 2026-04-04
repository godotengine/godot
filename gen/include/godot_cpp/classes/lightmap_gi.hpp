/**************************************************************************/
/*  lightmap_gi.hpp                                                       */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/lightmap_gi_data.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/visual_instance3d.hpp>
#include <godot_cpp/variant/color.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CameraAttributes;
class Sky;

class LightmapGI : public VisualInstance3D {
	GDEXTENSION_CLASS(LightmapGI, VisualInstance3D)

public:
	enum BakeQuality {
		BAKE_QUALITY_LOW = 0,
		BAKE_QUALITY_MEDIUM = 1,
		BAKE_QUALITY_HIGH = 2,
		BAKE_QUALITY_ULTRA = 3,
	};

	enum GenerateProbes {
		GENERATE_PROBES_DISABLED = 0,
		GENERATE_PROBES_SUBDIV_4 = 1,
		GENERATE_PROBES_SUBDIV_8 = 2,
		GENERATE_PROBES_SUBDIV_16 = 3,
		GENERATE_PROBES_SUBDIV_32 = 4,
	};

	enum BakeError {
		BAKE_ERROR_OK = 0,
		BAKE_ERROR_NO_SCENE_ROOT = 1,
		BAKE_ERROR_FOREIGN_DATA = 2,
		BAKE_ERROR_NO_LIGHTMAPPER = 3,
		BAKE_ERROR_NO_SAVE_PATH = 4,
		BAKE_ERROR_NO_MESHES = 5,
		BAKE_ERROR_MESHES_INVALID = 6,
		BAKE_ERROR_CANT_CREATE_IMAGE = 7,
		BAKE_ERROR_USER_ABORTED = 8,
		BAKE_ERROR_TEXTURE_SIZE_TOO_SMALL = 9,
		BAKE_ERROR_LIGHTMAP_TOO_SMALL = 10,
		BAKE_ERROR_ATLAS_TOO_SMALL = 11,
	};

	enum EnvironmentMode {
		ENVIRONMENT_MODE_DISABLED = 0,
		ENVIRONMENT_MODE_SCENE = 1,
		ENVIRONMENT_MODE_CUSTOM_SKY = 2,
		ENVIRONMENT_MODE_CUSTOM_COLOR = 3,
	};

	void set_light_data(const Ref<LightmapGIData> &p_data);
	Ref<LightmapGIData> get_light_data() const;
	void set_bake_quality(LightmapGI::BakeQuality p_bake_quality);
	LightmapGI::BakeQuality get_bake_quality() const;
	void set_bounces(int32_t p_bounces);
	int32_t get_bounces() const;
	void set_bounce_indirect_energy(float p_bounce_indirect_energy);
	float get_bounce_indirect_energy() const;
	void set_generate_probes(LightmapGI::GenerateProbes p_subdivision);
	LightmapGI::GenerateProbes get_generate_probes() const;
	void set_bias(float p_bias);
	float get_bias() const;
	void set_environment_mode(LightmapGI::EnvironmentMode p_mode);
	LightmapGI::EnvironmentMode get_environment_mode() const;
	void set_environment_custom_sky(const Ref<Sky> &p_sky);
	Ref<Sky> get_environment_custom_sky() const;
	void set_environment_custom_color(const Color &p_color);
	Color get_environment_custom_color() const;
	void set_environment_custom_energy(float p_energy);
	float get_environment_custom_energy() const;
	void set_texel_scale(float p_texel_scale);
	float get_texel_scale() const;
	void set_max_texture_size(int32_t p_max_texture_size);
	int32_t get_max_texture_size() const;
	void set_supersampling_enabled(bool p_enable);
	bool is_supersampling_enabled() const;
	void set_supersampling_factor(float p_factor);
	float get_supersampling_factor() const;
	void set_use_denoiser(bool p_use_denoiser);
	bool is_using_denoiser() const;
	void set_denoiser_strength(float p_denoiser_strength);
	float get_denoiser_strength() const;
	void set_denoiser_range(int32_t p_denoiser_range);
	int32_t get_denoiser_range() const;
	void set_interior(bool p_enable);
	bool is_interior() const;
	void set_directional(bool p_directional);
	bool is_directional() const;
	void set_shadowmask_mode(LightmapGIData::ShadowmaskMode p_mode);
	LightmapGIData::ShadowmaskMode get_shadowmask_mode() const;
	void set_use_texture_for_bounces(bool p_use_texture_for_bounces);
	bool is_using_texture_for_bounces() const;
	void set_camera_attributes(const Ref<CameraAttributes> &p_camera_attributes);
	Ref<CameraAttributes> get_camera_attributes() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualInstance3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(LightmapGI::BakeQuality);
VARIANT_ENUM_CAST(LightmapGI::GenerateProbes);
VARIANT_ENUM_CAST(LightmapGI::BakeError);
VARIANT_ENUM_CAST(LightmapGI::EnvironmentMode);

