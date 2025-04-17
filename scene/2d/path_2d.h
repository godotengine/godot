/**************************************************************************/
/*  path_2d.h                                                             */
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

#include "scene/2d/node_2d.h"
#include "scene/resources/curve.h"

class Path2D;
class Timer;

class PathDebug2D {
	static bool debug_enabled;
	static Color debug_paths_color;
	static float debug_paths_width;
	static float debug_paths_sample_interval;
	static bool debug_paths_fish_bones_enabled;
	static int debug_paths_fish_bones_interval;

	static void emit_changed();
	static bool _emitting_changed;
	void _emit_changed_deferred();

	static Mutex update_callbacks_mutex;
	static HashMap<Path2D *, Callable> update_callbacks;
	static void emit_update_callbacks();

public:
	static void add_update_callback(Path2D *p_path, Callable p_callback);
	static void remove_update_callback(Path2D *p_path);

	static void init_settings();

	static void set_debug_enabled(bool p_enabled);
	static bool is_debug_enabled();

	static void set_debug_paths_color(const Color &p_color);
	static Color get_debug_paths_color();

	static void set_debug_paths_width(float p_width);
	static float get_debug_paths_width();

	static void set_debug_paths_sample_interval(float p_interval);
	static float get_debug_paths_sample_interval();

	static void set_debug_paths_fish_bones_enabled(bool p_enabled);
	static bool get_debug_paths_fish_bones_enabled();

	static void set_debug_paths_fish_bones_interval(int p_interval);
	static int get_debug_paths_fish_bones_interval();

	PathDebug2D();
	~PathDebug2D();
};

class Path2D : public Node2D {
	GDCLASS(Path2D, Node2D);

	Ref<Curve2D> curve;

	void _curve_changed();

	RID debug_mesh_rid;
	RID debug_instance;
	bool debug_enabled = true;
	bool debug_custom_enabled = false;
	Color debug_custom_color = Color(1.0, 1.0, 1.0, 1.0);

	void _debug_create();
	void _debug_update();
	void _debug_clear();
	void _debug_free();

	bool _emitting_debug_changed = false;
	void _on_debug_global_changed();
	void _emit_debug_changed_deferred();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
#ifdef DEBUG_ENABLED
	virtual Rect2 _edit_get_rect() const override;
	virtual bool _edit_use_rect() const override;
	virtual bool _edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const override;
#endif

	void set_curve(const Ref<Curve2D> &p_curve);
	Ref<Curve2D> get_curve() const;

	void set_debug_enabled(bool p_enabled);
	bool get_debug_enabled() const;

	void set_debug_custom_enabled(bool p_enabled);
	bool get_debug_custom_enabled() const;

	void set_debug_custom_color(const Color &p_color);
	const Color &get_debug_custom_color() const;

	Path2D() {}
};

class PathFollow2D : public Node2D {
	GDCLASS(PathFollow2D, Node2D);

public:
private:
	Path2D *path = nullptr;
	real_t progress = 0.0;
	Timer *update_timer = nullptr;
	real_t h_offset = 0.0;
	real_t v_offset = 0.0;
	bool cubic = true;
	bool loop = true;
	bool rotates = true;

	void _update_transform();

protected:
	void _validate_property(PropertyInfo &p_property) const;

	void _notification(int p_what);
	static void _bind_methods();

public:
	void path_changed();

	void set_progress(real_t p_progress);
	real_t get_progress() const;

	void set_h_offset(real_t p_h_offset);
	real_t get_h_offset() const;

	void set_v_offset(real_t p_v_offset);
	real_t get_v_offset() const;

	void set_progress_ratio(real_t p_ratio);
	real_t get_progress_ratio() const;

	void set_loop(bool p_loop);
	bool has_loop() const;

	void set_rotation_enabled(bool p_enabled);
	bool is_rotation_enabled() const;

	void set_cubic_interpolation_enabled(bool p_enabled);
	bool is_cubic_interpolation_enabled() const;

	PackedStringArray get_configuration_warnings() const override;

	PathFollow2D() {}
};
