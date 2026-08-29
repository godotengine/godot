/**************************************************************************/
/*  vertex_project_optimizer.h                                           */
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
#include "vertex_optimization_report.h"

// Vertex Project Optimizer. Scans the project directory for large/expensive
// assets and reads live engine performance counters, then produces a report
// with recommendations and (optionally) applies safe automatic optimizations.
// Destructive or irreversible actions always require explicit confirmation.
class VertexProjectOptimizer : public Object {
	GDCLASS(VertexProjectOptimizer, Object)

private:
	int64_t large_texture_threshold_kb = 4096; // 4 MB
	int large_particle_threshold = 5000;
	int high_draw_call_threshold = 500;
	int64_t large_asset_threshold_kb = 10240; // 10 MB

	Dictionary _collect_runtime_metrics() const;
	void _scan_directory(const String &p_path, VertexOptimizationReport *p_report) const;
	int64_t _file_size_kb(const String &p_path) const;
	bool _is_image_extension(const String &p_ext) const;
	bool _is_audio_extension(const String &p_ext) const;
	bool _is_scene_extension(const String &p_ext) const;
	bool _is_shader_extension(const String &p_ext) const;

protected:
	static void _bind_methods();

public:
	Ref<VertexOptimizationReport> analyze_project(const String &p_root = "res://") const;
	Dictionary get_runtime_metrics() const { return _collect_runtime_metrics(); }

	// Safe, reversible automatic optimizations. None delete files.
	Dictionary apply_safe_optimizations(const Ref<VertexOptimizationReport> &p_report, bool p_dry_run = true);

	void set_large_texture_threshold_kb(int64_t p_kb) { large_texture_threshold_kb = p_kb; }
	int64_t get_large_texture_threshold_kb() const { return large_texture_threshold_kb; }
	void set_large_particle_threshold(int p_n) { large_particle_threshold = p_n; }
	int get_large_particle_threshold() const { return large_particle_threshold; }
	void set_high_draw_call_threshold(int p_n) { high_draw_call_threshold = p_n; }
	int get_high_draw_call_threshold() const { return high_draw_call_threshold; }
	void set_large_asset_threshold_kb(int64_t p_kb) { large_asset_threshold_kb = p_kb; }
	int64_t get_large_asset_threshold_kb() const { return large_asset_threshold_kb; }

	VertexProjectOptimizer();
	~VertexProjectOptimizer();
};
