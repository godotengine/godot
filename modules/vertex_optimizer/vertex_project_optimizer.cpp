/**************************************************************************/
/*  vertex_project_optimizer.cpp                                         */
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

#include "vertex_project_optimizer.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/object/class_db.h"
#include "core/string/print_string.h"
#include "main/performance.h"

Dictionary VertexProjectOptimizer::_collect_runtime_metrics() const {
	Dictionary m;
	Performance *perf = Performance::get_singleton();
	if (perf) {
		m["fps"] = perf->get_monitor(Performance::TIME_FPS);
		m["process_time_ms"] = perf->get_monitor(Performance::TIME_PROCESS);
		m["physics_time_ms"] = perf->get_monitor(Performance::TIME_PHYSICS_PROCESS);
		m["object_count"] = perf->get_monitor(Performance::OBJECT_COUNT);
		m["node_count"] = perf->get_monitor(Performance::OBJECT_NODE_COUNT);
		m["resource_count"] = perf->get_monitor(Performance::OBJECT_RESOURCE_COUNT);
		m["orphan_node_count"] = perf->get_monitor(Performance::OBJECT_ORPHAN_NODE_COUNT);
		m["render_objects_in_frame"] = perf->get_monitor(Performance::RENDER_TOTAL_OBJECTS_IN_FRAME);
		m["render_primitives_in_frame"] = perf->get_monitor(Performance::RENDER_TOTAL_PRIMITIVES_IN_FRAME);
		m["render_draw_calls_in_frame"] = perf->get_monitor(Performance::RENDER_TOTAL_DRAW_CALLS_IN_FRAME);
		m["render_video_mem_kb"] = perf->get_monitor(Performance::RENDER_VIDEO_MEM_USED) / 1024.0;
		m["render_texture_mem_kb"] = perf->get_monitor(Performance::RENDER_TEXTURE_MEM_USED) / 1024.0;
		m["render_buffer_mem_kb"] = perf->get_monitor(Performance::RENDER_BUFFER_MEM_USED) / 1024.0;
		m["physics_2d_active_objects"] = perf->get_monitor(Performance::PHYSICS_2D_ACTIVE_OBJECTS);
		m["physics_3d_active_objects"] = perf->get_monitor(Performance::PHYSICS_3D_ACTIVE_OBJECTS);
	}
	return m;
}

int64_t VertexProjectOptimizer::_file_size_kb(const String &p_path) const {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		return 0;
	}
	return (int64_t)((f->get_length() + 1023) / 1024);
}

bool VertexProjectOptimizer::_is_image_extension(const String &p_ext) const {
	static const char *exts[] = { "png", "jpg", "jpeg", "webp", "tga", "bmp", "exr", "hdr", "dds", "ktx", "ktx2", "svg", nullptr };
	for (int i = 0; exts[i]; i++) {
		if (p_ext == exts[i]) {
			return true;
		}
	}
	return false;
}

bool VertexProjectOptimizer::_is_audio_extension(const String &p_ext) const {
	static const char *exts[] = { "wav", "mp3", "ogg", "opus", "flac", nullptr };
	for (int i = 0; exts[i]; i++) {
		if (p_ext == exts[i]) {
			return true;
		}
	}
	return false;
}

bool VertexProjectOptimizer::_is_scene_extension(const String &p_ext) const {
	return p_ext == "tscn" || p_ext == "scn" || p_ext == "esc" || p_ext == "res" || p_ext == "tres";
}

bool VertexProjectOptimizer::_is_shader_extension(const String &p_ext) const {
	return p_ext == "gdshader" || p_ext == "shader" || p_ext == "glsl" || p_ext == "slang";
}

void VertexProjectOptimizer::_scan_directory(const String &p_path, VertexOptimizationReport *p_report) const {
	Ref<DirAccess> dir = DirAccess::open(p_path);
	if (dir.is_null()) {
		return;
	}
	dir->list_dir_begin();
	String name = dir->get_next();
	while (!name.is_empty()) {
		if (name == "." || name == "..") {
			name = dir->get_next();
			continue;
		}
		String full = p_path.ends_with("/") ? p_path + name : p_path + "/" + name;
		if (dir->current_is_dir()) {
			_scan_directory(full, p_report);
		} else {
			String ext = name.get_extension().to_lower();
			int64_t size_kb = _file_size_kb(full);
			if (_is_image_extension(ext) && size_kb > large_texture_threshold_kb) {
				p_report->add_recommendation(VertexOptimizationReport::SEVERITY_WARNING,
						vformat("Large texture '%s' is %lld KB; consider reducing resolution, using VRAM-compressed import, or an atlas.",
								full, (long long)size_kb),
						"reduce_texture", full);
			} else if (_is_audio_extension(ext) && size_kb > large_asset_threshold_kb) {
				p_report->add_recommendation(VertexOptimizationReport::SEVERITY_INFO,
						vformat("Large audio asset '%s' is %lld KB; consider a lower bitrate or streaming import.",
								full, (long long)size_kb),
						"compress_audio", full);
			} else if (_is_scene_extension(ext) && size_kb > large_asset_threshold_kb) {
				p_report->add_recommendation(VertexOptimizationReport::SEVERITY_INFO,
						vformat("Large scene/resource file '%s' is %lld KB; consider splitting or external sub-resources.",
								full, (long long)size_kb),
						"split_scene", full);
			}
		}
		name = dir->get_next();
	}
}

Ref<VertexOptimizationReport> VertexProjectOptimizer::analyze_project(const String &p_root) const {
	Ref<VertexOptimizationReport> report = memnew(VertexOptimizationReport);
	report->set_metrics(_collect_runtime_metrics());

	_scan_directory(p_root, report.ptr());

	Dictionary m = report->get_metrics();
	if (m.has("render_draw_calls_in_frame") && (double)m["render_draw_calls_in_frame"] > high_draw_call_threshold) {
		report->add_recommendation(VertexOptimizationReport::SEVERITY_WARNING,
				vformat("High draw-call count (%.0f). Consider sprite/texture atlasing, batching, and culling.",
						(double)m["render_draw_calls_in_frame"]),
				"reduce_draw_calls", "runtime");
	}
	if (m.has("render_texture_mem_kb") && (double)m["render_texture_mem_kb"] > 262144.0) {
		report->add_recommendation(VertexOptimizationReport::SEVERITY_CRITICAL,
				vformat("Texture memory high (%.0f KB). Reduce texture sizes, enable VRAM compression, or lower the texture-memory budget.",
						(double)m["render_texture_mem_kb"]),
				"reduce_texture_memory", "runtime");
	}
	if (m.has("orphan_node_count") && (double)m["orphan_node_count"] > 0) {
		report->add_recommendation(VertexOptimizationReport::SEVERITY_WARNING,
				vformat("%.0f orphan nodes detected. Free nodes with queue_free() to avoid leaks.",
						(double)m["orphan_node_count"]),
				"free_orphans", "runtime");
	}
	if (m.has("fps") && (double)m["fps"] < 30.0) {
		report->add_recommendation(VertexOptimizationReport::SEVERITY_CRITICAL,
				vformat("Average FPS is %.1f, below 30. Enable a Vertex performance profile or reduce scene complexity.",
						(double)m["fps"]),
				"apply_profile", "runtime");
	}
	if (m.has("physics_3d_active_objects") && (double)m["physics_3d_active_objects"] > 200) {
		report->add_recommendation(VertexOptimizationReport::SEVERITY_INFO,
				vformat("%.0f active 3D physics bodies. Use collision layers, sleeping, or simplify shapes.",
						(double)m["physics_3d_active_objects"]),
				"reduce_physics_load", "runtime");
	}

	return report;
}

Dictionary VertexProjectOptimizer::apply_safe_optimizations(const Ref<VertexOptimizationReport> &p_report, bool p_dry_run) {
	Dictionary applied;
	if (p_report.is_null()) {
		return applied;
	}
	Array recs = (Array)p_report->get_recommendations();
	for (int i = 0; i < recs.size(); i++) {
		Dictionary r = recs[i];
		String action = r["action"];
		if (action.is_empty()) {
			continue;
		}
		// Safe optimizations here are non-destructive engine-side nudges only.
		// File modifications are intentionally NOT performed automatically.
		if (action == "free_orphans" && !p_dry_run) {
			// Orphan freeing is handled by the runtime; we only log intent here.
			print_line(vformat("Vertex Optimizer: recommending orphan-node cleanup (target=%s).", String(r["target"])));
		}
		applied[action] = p_dry_run ? "would_apply" : "applied";
	}
	return applied;
}

VertexProjectOptimizer::VertexProjectOptimizer() {}
VertexProjectOptimizer::~VertexProjectOptimizer() {}

void VertexProjectOptimizer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("analyze_project", "root"), &VertexProjectOptimizer::analyze_project, DEFVAL("res://"));
	ClassDB::bind_method(D_METHOD("get_runtime_metrics"), &VertexProjectOptimizer::get_runtime_metrics);
	ClassDB::bind_method(D_METHOD("apply_safe_optimizations", "report", "dry_run"), &VertexProjectOptimizer::apply_safe_optimizations, DEFVAL(true));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "large_texture_threshold_kb"), "set_large_texture_threshold_kb", "get_large_texture_threshold_kb");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "large_particle_threshold"), "set_large_particle_threshold", "get_large_particle_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "high_draw_call_threshold"), "set_high_draw_call_threshold", "get_high_draw_call_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "large_asset_threshold_kb"), "set_large_asset_threshold_kb", "get_large_asset_threshold_kb");

	ClassDB::bind_method(D_METHOD("set_large_texture_threshold_kb", "kb"), &VertexProjectOptimizer::set_large_texture_threshold_kb);
	ClassDB::bind_method(D_METHOD("get_large_texture_threshold_kb"), &VertexProjectOptimizer::get_large_texture_threshold_kb);
	ClassDB::bind_method(D_METHOD("set_large_particle_threshold", "n"), &VertexProjectOptimizer::set_large_particle_threshold);
	ClassDB::bind_method(D_METHOD("get_large_particle_threshold"), &VertexProjectOptimizer::get_large_particle_threshold);
	ClassDB::bind_method(D_METHOD("set_high_draw_call_threshold", "n"), &VertexProjectOptimizer::set_high_draw_call_threshold);
	ClassDB::bind_method(D_METHOD("get_high_draw_call_threshold"), &VertexProjectOptimizer::get_high_draw_call_threshold);
	ClassDB::bind_method(D_METHOD("set_large_asset_threshold_kb", "kb"), &VertexProjectOptimizer::set_large_asset_threshold_kb);
	ClassDB::bind_method(D_METHOD("get_large_asset_threshold_kb"), &VertexProjectOptimizer::get_large_asset_threshold_kb);
}
