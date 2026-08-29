/**************************************************************************/
/*  vertex_performance_manager.cpp                                        */
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

#include "vertex_performance_manager.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

#include <cmath>

// Map our generic quality enums to existing Godot project setting values.
// MSAA: 0=disabled,1=2x,2=4x,3=8x (matches Godot's rendering/anti_aliasing/quality/msaa_3d enum index).

void VertexPerformanceManager::_apply_render_scale_to_root(float p_scale) {
	current_render_scale = CLAMP(p_scale, 0.25f, 2.0f);
	if (SceneTree *tree = SceneTree::get_singleton()) {
		if (Window *root = tree->get_root()) {
			root->set_scaling_3d_scale(current_render_scale);
		}
	}
}

void VertexPerformanceManager::_recompute_adaptive() {
	if (!monitoring_enabled || profile.is_null() || !profile->get_adaptive_quality()) {
		return;
	}
	Engine *engine = Engine::get_singleton();
	if (!engine) {
		return;
	}
	// Derive an instantaneous frame time from the reported FPS. This is an
	// inexpensive proxy that works across all renderers (no GPU timer needed).
	double fps = engine->get_frames_per_second();
	float frame_time_ms = fps > 0.0 ? float(1000.0 / fps) : 0.0f;
	float threshold = profile->get_adaptive_frame_time_threshold_ms();

	if (frame_time_ms > threshold) {
		slow_frame_streak++;
		fast_frame_streak = 0;
	} else if (frame_time_ms < threshold * 0.75f) {
		fast_frame_streak++;
		slow_frame_streak = 0;
	} else {
		slow_frame_streak = 0;
		fast_frame_streak = 0;
	}

	float min_scale = profile->get_adaptive_min_render_scale_pct() / 100.0f;
	// Scale down after a sustained streak of slow frames.
	if (slow_frame_streak >= 60 && current_render_scale > min_scale) {
		_apply_render_scale_to_root(current_render_scale - 0.1f);
		slow_frame_streak = 0;
	}
	// Scale back up when the engine is comfortably fast and below the profile baseline.
	if (fast_frame_streak >= 300 && current_render_scale < original_render_scale) {
		_apply_render_scale_to_root(current_render_scale + 0.1f);
		fast_frame_streak = 0;
	}
}

void VertexPerformanceManager::_arm_monitor_timer() {
	SceneTree *tree = SceneTree::get_singleton();
	if (!tree) {
		return;
	}
	// Re-arm roughly every ~150ms. The timer fires its timeout signal, which
	// calls VertexPerformanceManager.tick(), which re-arms the next timer.
	Ref<SceneTreeTimer> timer = tree->create_timer(0.15, true, false, true);
	if (timer.is_valid()) {
		timer->connect("timeout", Callable(this, "tick"));
	}
}

void VertexPerformanceManager::tick() {
	_recompute_adaptive();
	if (monitoring_enabled) {
		_arm_monitor_timer();
	}
}

void VertexPerformanceManager::set_profile(const Ref<VertexPerformanceProfile> &p_profile) {
	profile = p_profile;
	if (profile.is_valid()) {
		original_render_scale = profile->get_render_scale();
		current_render_scale = profile->get_render_scale();
		current_texture_quality = profile->get_texture_quality();
	}
}

void VertexPerformanceManager::apply_to_project_settings() {
	// Writes the active profile into existing Godot project settings so the
	// next launch (and subsystems that GLOBAL_GET these keys) pick them up.
	ProjectSettings *ps = ProjectSettings::get_singleton();
	if (!ps || profile.is_null()) {
		return;
	}
	ps->set_setting("display/window/size/viewport_width", profile->get_resolution_width());
	ps->set_setting("display/window/size/viewport_height", profile->get_resolution_height());
	ps->set_setting("rendering/scaling_3d/scale", profile->get_render_scale());
	ps->set_setting("rendering/anti_aliasing/quality/msaa_2d", MIN(profile->get_anti_aliasing(), 3));
	ps->set_setting("rendering/anti_aliasing/quality/msaa_3d", MIN(profile->get_anti_aliasing(), 3));

	// Frame rate / physics / VSync.
	Engine::get_singleton()->set_max_fps(profile->get_target_fps());
	Engine::get_singleton()->set_physics_ticks_per_second(profile->get_physics_ticks_per_second());

	ps->set_setting("Vertex/Performance/texture_memory_budget_mb", profile->get_texture_memory_budget_mb());
	ps->set_setting("Vertex/Performance/asset_cache_budget_mb", profile->get_asset_cache_budget_mb());
	ps->set_setting("Vertex/Performance/shader_quality", profile->get_shader_quality());
	ps->set_setting("Vertex/Performance/audio_quality", profile->get_audio_quality());
}

void VertexPerformanceManager::apply_live() {
	// Applies settings that can take effect immediately on the running tree.
	if (profile.is_null()) {
		return;
	}
	original_render_scale = profile->get_render_scale();
	_apply_render_scale_to_root(profile->get_render_scale());

	Engine *engine = Engine::get_singleton();
	if (engine) {
		engine->set_max_fps(profile->get_target_fps());
		engine->set_physics_ticks_per_second(profile->get_physics_ticks_per_second());
	}
}

void VertexPerformanceManager::set_monitoring_enabled(bool p_enabled) {
	monitoring_enabled = p_enabled;
	if (p_enabled) {
		_arm_monitor_timer();
	}
}

void VertexPerformanceManager::set_current_render_scale(float p_scale) {
	_apply_render_scale_to_root(p_scale);
}

void VertexPerformanceManager::reset_adaptive() {
	slow_frame_streak = 0;
	fast_frame_streak = 0;
	if (profile.is_valid()) {
		_apply_render_scale_to_root(profile->get_render_scale());
	}
}

Dictionary VertexPerformanceManager::get_status() const {
	Dictionary d;
	d["monitoring_enabled"] = monitoring_enabled;
	d["current_render_scale"] = current_render_scale;
	d["profile_render_scale"] = profile.is_valid() ? profile->get_render_scale() : 1.0f;
	d["fps"] = Engine::get_singleton() ? Engine::get_singleton()->get_frames_per_second() : 0.0;
	d["slow_frame_streak"] = slow_frame_streak;
	d["fast_frame_streak"] = fast_frame_streak;
	return d;
}

Ref<VertexPerformanceProfile> VertexPerformanceManager::create_profile(int p_preset) const {
	Ref<VertexPerformanceProfile> prof = memnew(VertexPerformanceProfile);
	prof->apply_preset((VertexPerformanceProfile::Preset)p_preset);
	return prof;
}

VertexPerformanceManager::VertexPerformanceManager() {
	if (profile.is_null()) {
		profile = memnew(VertexPerformanceProfile);
	}
	original_render_scale = profile->get_render_scale();
	current_render_scale = original_render_scale;
}

VertexPerformanceManager::~VertexPerformanceManager() {}

void VertexPerformanceManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_profile", "profile"), &VertexPerformanceManager::set_profile);
	ClassDB::bind_method(D_METHOD("get_profile"), &VertexPerformanceManager::get_profile);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "profile", PROPERTY_HINT_RESOURCE_TYPE, "VertexPerformanceProfile"), "set_profile", "get_profile");

	ClassDB::bind_method(D_METHOD("apply_to_project_settings"), &VertexPerformanceManager::apply_to_project_settings);
	ClassDB::bind_method(D_METHOD("apply_live"), &VertexPerformanceManager::apply_live);
	ClassDB::bind_method(D_METHOD("set_monitoring_enabled", "enabled"), &VertexPerformanceManager::set_monitoring_enabled);
	ClassDB::bind_method(D_METHOD("get_monitoring_enabled"), &VertexPerformanceManager::get_monitoring_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "monitoring_enabled"), "set_monitoring_enabled", "get_monitoring_enabled");

	ClassDB::bind_method(D_METHOD("get_current_render_scale"), &VertexPerformanceManager::get_current_render_scale);
	ClassDB::bind_method(D_METHOD("set_current_render_scale", "scale"), &VertexPerformanceManager::set_current_render_scale);
	ClassDB::bind_method(D_METHOD("reset_adaptive"), &VertexPerformanceManager::reset_adaptive);
	ClassDB::bind_method(D_METHOD("get_status"), &VertexPerformanceManager::get_status);
	ClassDB::bind_method(D_METHOD("tick"), &VertexPerformanceManager::tick);
	ClassDB::bind_method(D_METHOD("create_profile", "preset"), &VertexPerformanceManager::create_profile);

	ADD_SIGNAL(MethodInfo("adaptive_quality_changed", PropertyInfo(Variant::FLOAT, "new_render_scale")));
}
