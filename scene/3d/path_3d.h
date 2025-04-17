/**************************************************************************/
/*  path_3d.h                                                             */
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

#include "scene/3d/node_3d.h"
#include "scene/resources/curve.h"

class Path3D;

class PathDebug3D {
	static bool debug_enabled;
	static Color debug_paths_color;
	static float debug_paths_sample_interval;
	static bool debug_paths_fish_bones_enabled;
	static int debug_paths_fish_bones_interval;
	static Ref<Material> debug_default_material;

	static void emit_changed();
	static bool _emitting_changed;
	void _emit_changed_deferred();

	static Mutex update_callbacks_mutex;
	static HashMap<Path3D *, Callable> update_callbacks;
	static void emit_update_callbacks();

public:
	static void add_update_callback(Path3D *p_path, Callable p_callback);
	static void remove_update_callback(Path3D *p_path);

	static void init_settings();
	static void init_materials();
	static void finish_materials();

	static void set_debug_enabled(bool p_enabled);
	static bool is_debug_enabled();

	static void set_debug_paths_color(const Color &p_color);
	static Color get_debug_paths_color();

	static void set_debug_paths_sample_interval(float p_interval);
	static float get_debug_paths_sample_interval();

	static void set_debug_paths_fish_bones_enabled(bool p_enabled);
	static bool get_debug_paths_fish_bones_enabled();

	static void set_debug_paths_fish_bones_interval(int p_interval);
	static int get_debug_paths_fish_bones_interval();

	static Ref<Material> get_debug_material();

	PathDebug3D();
	~PathDebug3D();
};

class Path3D : public Node3D {
	GDCLASS(Path3D, Node3D);

	Ref<Curve3D> curve;

	Callable update_callback; // Used only by CSG currently.

	void _curve_changed();

	RID debug_mesh_rid;
	RID debug_instance;
	bool debug_enabled = true;
	bool debug_custom_enabled = false;
	Color debug_custom_color = Color(1.0, 1.0, 1.0, 1.0);
	Ref<StandardMaterial3D> debug_custom_material;

	void _debug_create();
	void _debug_update();
	void _debug_clear();
	void _debug_free();

	Ref<StandardMaterial3D> get_debug_material();

	bool _emitting_debug_changed = false;
	void _on_debug_global_changed();
	void _emit_debug_changed_deferred();

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	void set_update_callback(Callable p_callback);

	void set_curve(const Ref<Curve3D> &p_curve);
	Ref<Curve3D> get_curve() const;

	void set_debug_enabled(bool p_enabled);
	bool get_debug_enabled() const;

	void set_debug_custom_enabled(bool p_enabled);
	bool get_debug_custom_enabled() const;

	void set_debug_custom_color(const Color &p_color);
	const Color &get_debug_custom_color() const;

	Path3D();
	~Path3D();
};

class PathFollow3D : public Node3D {
	GDCLASS(PathFollow3D, Node3D);

public:
	enum RotationMode {
		ROTATION_NONE,
		ROTATION_Y,
		ROTATION_XY,
		ROTATION_XYZ,
		ROTATION_ORIENTED
	};

private:
	Path3D *path = nullptr;
	real_t progress = 0.0;
	real_t h_offset = 0.0;
	real_t v_offset = 0.0;
	bool cubic = true;
	bool loop = true;
	bool tilt_enabled = true;
	bool transform_dirty = true;
	bool use_model_front = false;
	RotationMode rotation_mode = ROTATION_XYZ;

protected:
	void _validate_property(PropertyInfo &p_property) const;

	void _notification(int p_what);

	static void _bind_methods();

public:
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

	void set_tilt_enabled(bool p_enabled);
	bool is_tilt_enabled() const;

	void set_rotation_mode(RotationMode p_rotation_mode);
	RotationMode get_rotation_mode() const;

	void set_use_model_front(bool p_use_model_front);
	bool is_using_model_front() const;

	void set_cubic_interpolation_enabled(bool p_enabled);
	bool is_cubic_interpolation_enabled() const;

	PackedStringArray get_configuration_warnings() const override;

	void update_transform();

	static Transform3D correct_posture(Transform3D p_transform, PathFollow3D::RotationMode p_rotation_mode);

	PathFollow3D() {}
};

VARIANT_ENUM_CAST(PathFollow3D::RotationMode);
