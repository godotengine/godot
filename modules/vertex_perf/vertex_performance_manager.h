/**************************************************************************/
/*  vertex_performance_manager.h                                          */
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

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "vertex_performance_profile.h"

// Drives Vertex's adaptive performance system. Holds the active profile,
// pushes its values into engine/project settings, and monitors sustained
// frame-time degradation to scale rendering cost down (and back up) at runtime.
class VertexPerformanceManager : public Object {
	GDCLASS(VertexPerformanceManager, Object)

private:
	Ref<VertexPerformanceProfile> profile;
	bool monitoring_enabled = false;
	float current_render_scale = 1.0f;
	int current_texture_quality = 3;

	// Adaptive monitoring state.
	int slow_frame_streak = 0; // Consecutive frames above the threshold.
	int fast_frame_streak = 0; // Consecutive frames comfortably below target.
	float original_render_scale = 1.0f;

	void _apply_render_scale_to_root(float p_scale);
	void _recompute_adaptive();
	void _arm_monitor_timer();

protected:
	static void _bind_methods();

public:
	void set_profile(const Ref<VertexPerformanceProfile> &p_profile);
	Ref<VertexPerformanceProfile> get_profile() const { return profile; }

	void apply_to_project_settings();
	void apply_live();
	void set_monitoring_enabled(bool p_enabled);
	bool get_monitoring_enabled() const { return monitoring_enabled; }

	// Adaptive live controls.
	float get_current_render_scale() const { return current_render_scale; }
	void set_current_render_scale(float p_scale);
	void reset_adaptive();
	Dictionary get_status() const;

	// Called periodically by the internal monitor timer. Exposed so it can be
	// invoked manually from scripting or tests.
	void tick();

	// Helpers exposed to scripting / AI assistant.
	Ref<VertexPerformanceProfile> create_profile(int p_preset) const;

	VertexPerformanceManager();
	~VertexPerformanceManager();
};
