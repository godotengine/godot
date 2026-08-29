/**************************************************************************/
/*  vertex_performance_profile.h                                          */
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

#include "core/object/ref_counted.h"
#include "core/variant/variant.h"

// A serializable performance profile. Encodes the per-tier trade-offs the
// Vertex performance system applies to rendering, physics, audio and memory.
class VertexPerformanceProfile : public RefCounted {
	GDCLASS(VertexPerformanceProfile, RefCounted)

public:
	enum Preset {
		PRESET_ULTRA_LOW,
		PRESET_LOW,
		PRESET_BALANCED,
		PRESET_HIGH,
		PRESET_ULTRA,
		PRESET_CUSTOM,
	};

private:
	Preset preset = PRESET_BALANCED;
	int resolution_width = 1280;
	int resolution_height = 720;
	float render_scale = 1.0f; // Viewport render scale multiplier (0.5 - 1.0).
	int texture_quality = 2; // 0=lowest .. 3=highest.
	int texture_compression = 1; // 0=none/fast, 1=ETC2/ASTC, 2=ASTC high quality.
	bool particles_enabled = true;
	int max_particles = 2000;
	int shadow_quality = 2; // 0=off .. 3=high.
	int lighting_quality = 2; // 0=minimal .. 3=full.
	int post_processing_quality = 2; // 0=off .. 3=full.
	int anti_aliasing = 2; // 0=off, 1=FXAA, 2=MSAA 2x, 3=MSAA 4x, 4=TAA.
	bool vsync = true;
	int target_fps = 60;
	int physics_ticks_per_second = 60;
	int audio_quality = 2; // 0=low .. 2=high.
	int shader_quality = 2; // 0=minimal .. 2=full.
	int texture_memory_budget_mb = 512; // 0 = unlimited.
	int asset_cache_budget_mb = 256;
	bool adaptive_quality = true;
	float adaptive_frame_time_threshold_ms = 33.3f; // ~30 FPS sustained.
	int adaptive_min_render_scale_pct = 50;

protected:
	static void _bind_methods();

public:
	void apply_preset(Preset p_preset);
	Preset get_preset() const { return preset; }

	Dictionary to_dictionary() const;
	void from_dictionary(const Dictionary &p_dict);

	// Property getters/setters (bound).
	void set_resolution_width(int p_w) { resolution_width = p_w; }
	int get_resolution_width() const { return resolution_width; }
	void set_resolution_height(int p_h) { resolution_height = p_h; }
	int get_resolution_height() const { return resolution_height; }
	void set_render_scale(float p_s) { render_scale = p_s; }
	float get_render_scale() const { return render_scale; }
	void set_texture_quality(int p_q) { texture_quality = p_q; }
	int get_texture_quality() const { return texture_quality; }
	void set_texture_compression(int p_c) { texture_compression = p_c; }
	int get_texture_compression() const { return texture_compression; }
	void set_particles_enabled(bool p_e) { particles_enabled = p_e; }
	bool get_particles_enabled() const { return particles_enabled; }
	void set_max_particles(int p_m) { max_particles = p_m; }
	int get_max_particles() const { return max_particles; }
	void set_shadow_quality(int p_q) { shadow_quality = p_q; }
	int get_shadow_quality() const { return shadow_quality; }
	void set_lighting_quality(int p_q) { lighting_quality = p_q; }
	int get_lighting_quality() const { return lighting_quality; }
	void set_post_processing_quality(int p_q) { post_processing_quality = p_q; }
	int get_post_processing_quality() const { return post_processing_quality; }
	void set_anti_aliasing(int p_a) { anti_aliasing = p_a; }
	int get_anti_aliasing() const { return anti_aliasing; }
	void set_vsync(bool p_v) { vsync = p_v; }
	bool get_vsync() const { return vsync; }
	void set_target_fps(int p_f) { target_fps = p_f; }
	int get_target_fps() const { return target_fps; }
	void set_physics_ticks_per_second(int p_t) { physics_ticks_per_second = p_t; }
	int get_physics_ticks_per_second() const { return physics_ticks_per_second; }
	void set_audio_quality(int p_q) { audio_quality = p_q; }
	int get_audio_quality() const { return audio_quality; }
	void set_shader_quality(int p_q) { shader_quality = p_q; }
	int get_shader_quality() const { return shader_quality; }
	void set_texture_memory_budget_mb(int p_m) { texture_memory_budget_mb = p_m; }
	int get_texture_memory_budget_mb() const { return texture_memory_budget_mb; }
	void set_asset_cache_budget_mb(int p_m) { asset_cache_budget_mb = p_m; }
	int get_asset_cache_budget_mb() const { return asset_cache_budget_mb; }
	void set_adaptive_quality(bool p_a) { adaptive_quality = p_a; }
	bool get_adaptive_quality() const { return adaptive_quality; }
	void set_adaptive_frame_time_threshold_ms(float p_t) { adaptive_frame_time_threshold_ms = p_t; }
	float get_adaptive_frame_time_threshold_ms() const { return adaptive_frame_time_threshold_ms; }
	void set_adaptive_min_render_scale_pct(int p_p) { adaptive_min_render_scale_pct = p_p; }
	int get_adaptive_min_render_scale_pct() const { return adaptive_min_render_scale_pct; }

	VertexPerformanceProfile();
};

VARIANT_ENUM_CAST(VertexPerformanceProfile::Preset);
