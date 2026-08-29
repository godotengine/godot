/**************************************************************************/
/*  vertex_performance_profile.cpp                                        */
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

#include "vertex_performance_profile.h"

#include "core/object/class_db.h"
#include "core/variant/variant.h"

void VertexPerformanceProfile::apply_preset(Preset p_preset) {
	preset = p_preset;
	switch (p_preset) {
		case PRESET_ULTRA_LOW:
			resolution_width = 960;
			resolution_height = 540;
			render_scale = 0.5f;
			texture_quality = 0;
			texture_compression = 1;
			particles_enabled = true;
			max_particles = 256;
			shadow_quality = 0;
			lighting_quality = 0;
			post_processing_quality = 0;
			anti_aliasing = 0;
			vsync = true;
			target_fps = 30;
			physics_ticks_per_second = 30;
			audio_quality = 0;
			shader_quality = 0;
			texture_memory_budget_mb = 128;
			asset_cache_budget_mb = 64;
			adaptive_quality = true;
			break;
		case PRESET_LOW:
			resolution_width = 1280;
			resolution_height = 720;
			render_scale = 0.6f;
			texture_quality = 1;
			texture_compression = 1;
			particles_enabled = true;
			max_particles = 512;
			shadow_quality = 1;
			lighting_quality = 1;
			post_processing_quality = 1;
			anti_aliasing = 0;
			vsync = true;
			target_fps = 30;
			physics_ticks_per_second = 60;
			audio_quality = 1;
			shader_quality = 1;
			texture_memory_budget_mb = 192;
			asset_cache_budget_mb = 96;
			adaptive_quality = true;
			break;
		case PRESET_BALANCED:
			resolution_width = 1920;
			resolution_height = 1080;
			render_scale = 0.75f;
			texture_quality = 2;
			texture_compression = 1;
			particles_enabled = true;
			max_particles = 2000;
			shadow_quality = 2;
			lighting_quality = 2;
			post_processing_quality = 2;
			anti_aliasing = 2;
			vsync = true;
			target_fps = 60;
			physics_ticks_per_second = 60;
			audio_quality = 2;
			shader_quality = 2;
			texture_memory_budget_mb = 512;
			asset_cache_budget_mb = 256;
			adaptive_quality = true;
			break;
		case PRESET_HIGH:
			resolution_width = 1920;
			resolution_height = 1080;
			render_scale = 1.0f;
			texture_quality = 3;
			texture_compression = 2;
			particles_enabled = true;
			max_particles = 8000;
			shadow_quality = 3;
			lighting_quality = 3;
			post_processing_quality = 3;
			anti_aliasing = 3;
			vsync = true;
			target_fps = 60;
			physics_ticks_per_second = 60;
			audio_quality = 2;
			shader_quality = 2;
			texture_memory_budget_mb = 1024;
			asset_cache_budget_mb = 512;
			adaptive_quality = false;
			break;
		case PRESET_ULTRA:
		case PRESET_CUSTOM:
		default:
			resolution_width = 2560;
			resolution_height = 1440;
			render_scale = 1.0f;
			texture_quality = 3;
			texture_compression = 2;
			particles_enabled = true;
			max_particles = 20000;
			shadow_quality = 3;
			lighting_quality = 3;
			post_processing_quality = 3;
			anti_aliasing = 4;
			vsync = true;
			target_fps = 120;
			physics_ticks_per_second = 60;
			audio_quality = 2;
			shader_quality = 2;
			texture_memory_budget_mb = 2048;
			asset_cache_budget_mb = 1024;
			adaptive_quality = false;
			break;
	}
}

Dictionary VertexPerformanceProfile::to_dictionary() const {
	Dictionary d;
	d["resolution_width"] = resolution_width;
	d["resolution_height"] = resolution_height;
	d["render_scale"] = render_scale;
	d["texture_quality"] = texture_quality;
	d["texture_compression"] = texture_compression;
	d["particles_enabled"] = particles_enabled;
	d["max_particles"] = max_particles;
	d["shadow_quality"] = shadow_quality;
	d["lighting_quality"] = lighting_quality;
	d["post_processing_quality"] = post_processing_quality;
	d["anti_aliasing"] = anti_aliasing;
	d["vsync"] = vsync;
	d["target_fps"] = target_fps;
	d["physics_ticks_per_second"] = physics_ticks_per_second;
	d["audio_quality"] = audio_quality;
	d["shader_quality"] = shader_quality;
	d["texture_memory_budget_mb"] = texture_memory_budget_mb;
	d["asset_cache_budget_mb"] = asset_cache_budget_mb;
	d["adaptive_quality"] = adaptive_quality;
	d["adaptive_frame_time_threshold_ms"] = adaptive_frame_time_threshold_ms;
	d["adaptive_min_render_scale_pct"] = adaptive_min_render_scale_pct;
	return d;
}

void VertexPerformanceProfile::from_dictionary(const Dictionary &p_dict) {
	if (p_dict.has("resolution_width")) {
		resolution_width = p_dict["resolution_width"];
	}
	if (p_dict.has("resolution_height")) {
		resolution_height = p_dict["resolution_height"];
	}
	if (p_dict.has("render_scale")) {
		render_scale = p_dict["render_scale"];
	}
	if (p_dict.has("texture_quality")) {
		texture_quality = p_dict["texture_quality"];
	}
	if (p_dict.has("texture_compression")) {
		texture_compression = p_dict["texture_compression"];
	}
	if (p_dict.has("particles_enabled")) {
		particles_enabled = p_dict["particles_enabled"];
	}
	if (p_dict.has("max_particles")) {
		max_particles = p_dict["max_particles"];
	}
	if (p_dict.has("shadow_quality")) {
		shadow_quality = p_dict["shadow_quality"];
	}
	if (p_dict.has("lighting_quality")) {
		lighting_quality = p_dict["lighting_quality"];
	}
	if (p_dict.has("post_processing_quality")) {
		post_processing_quality = p_dict["post_processing_quality"];
	}
	if (p_dict.has("anti_aliasing")) {
		anti_aliasing = p_dict["anti_aliasing"];
	}
	if (p_dict.has("vsync")) {
		vsync = p_dict["vsync"];
	}
	if (p_dict.has("target_fps")) {
		target_fps = p_dict["target_fps"];
	}
	if (p_dict.has("physics_ticks_per_second")) {
		physics_ticks_per_second = p_dict["physics_ticks_per_second"];
	}
	if (p_dict.has("audio_quality")) {
		audio_quality = p_dict["audio_quality"];
	}
	if (p_dict.has("shader_quality")) {
		shader_quality = p_dict["shader_quality"];
	}
	if (p_dict.has("texture_memory_budget_mb")) {
		texture_memory_budget_mb = p_dict["texture_memory_budget_mb"];
	}
	if (p_dict.has("asset_cache_budget_mb")) {
		asset_cache_budget_mb = p_dict["asset_cache_budget_mb"];
	}
	if (p_dict.has("adaptive_quality")) {
		adaptive_quality = p_dict["adaptive_quality"];
	}
	if (p_dict.has("adaptive_frame_time_threshold_ms")) {
		adaptive_frame_time_threshold_ms = p_dict["adaptive_frame_time_threshold_ms"];
	}
	if (p_dict.has("adaptive_min_render_scale_pct")) {
		adaptive_min_render_scale_pct = p_dict["adaptive_min_render_scale_pct"];
	}
}

VertexPerformanceProfile::VertexPerformanceProfile() {
	apply_preset(PRESET_BALANCED);
}

void VertexPerformanceProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("apply_preset", "preset"), &VertexPerformanceProfile::apply_preset);
	ClassDB::bind_method(D_METHOD("get_preset"), &VertexPerformanceProfile::get_preset);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &VertexPerformanceProfile::to_dictionary);
	ClassDB::bind_method(D_METHOD("from_dictionary", "dict"), &VertexPerformanceProfile::from_dictionary);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "resolution_width"), "set_resolution_width", "get_resolution_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "resolution_height"), "set_resolution_height", "get_resolution_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "render_scale"), "set_render_scale", "get_render_scale");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_quality"), "set_texture_quality", "get_texture_quality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_compression"), "set_texture_compression", "get_texture_compression");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "particles_enabled"), "set_particles_enabled", "get_particles_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_particles"), "set_max_particles", "get_max_particles");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_quality"), "set_shadow_quality", "get_shadow_quality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "lighting_quality"), "set_lighting_quality", "get_lighting_quality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "post_processing_quality"), "set_post_processing_quality", "get_post_processing_quality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "anti_aliasing"), "set_anti_aliasing", "get_anti_aliasing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vsync"), "set_vsync", "get_vsync");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "target_fps"), "set_target_fps", "get_target_fps");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "physics_ticks_per_second"), "set_physics_ticks_per_second", "get_physics_ticks_per_second");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "audio_quality"), "set_audio_quality", "get_audio_quality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shader_quality"), "set_shader_quality", "get_shader_quality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_memory_budget_mb"), "set_texture_memory_budget_mb", "get_texture_memory_budget_mb");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "asset_cache_budget_mb"), "set_asset_cache_budget_mb", "get_asset_cache_budget_mb");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "adaptive_quality"), "set_adaptive_quality", "get_adaptive_quality");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adaptive_frame_time_threshold_ms"), "set_adaptive_frame_time_threshold_ms", "get_adaptive_frame_time_threshold_ms");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "adaptive_min_render_scale_pct"), "set_adaptive_min_render_scale_pct", "get_adaptive_min_render_scale_pct");

	BIND_ENUM_CONSTANT(PRESET_ULTRA_LOW);
	BIND_ENUM_CONSTANT(PRESET_LOW);
	BIND_ENUM_CONSTANT(PRESET_BALANCED);
	BIND_ENUM_CONSTANT(PRESET_HIGH);
	BIND_ENUM_CONSTANT(PRESET_ULTRA);
	BIND_ENUM_CONSTANT(PRESET_CUSTOM);
}
