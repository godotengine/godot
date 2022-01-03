/*************************************************************************/
/*  editor_visual_profiler.h                                             */
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

#ifndef EDITOR_FRAME_PROFILER_H
#define EDITOR_FRAME_PROFILER_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"

class EditorVisualProfiler : public VBoxContainer {
	GDCLASS(EditorVisualProfiler, VBoxContainer);

public:
	struct Metric {
		bool valid = false;

		uint64_t frame_number = 0;

		struct Area {
			String name;
			Color color_cache;
			StringName fullpath_cache;
			float cpu_time = 0;
			float gpu_time = 0;
		};

		Vector<Area> areas;
	};

	enum DisplayTimeMode {
		DISPLAY_FRAME_TIME,
		DISPLAY_FRAME_PERCENT,
	};

private:
	Button *activate;
	Button *clear_button;

	TextureRect *graph;
	Ref<ImageTexture> graph_texture;
	Vector<uint8_t> graph_image;
	Tree *variables;
	HSplitContainer *h_split;
	CheckBox *frame_relative;
	CheckBox *linked;

	OptionButton *display_mode;

	SpinBox *cursor_metric_edit;

	Vector<Metric> frame_metrics;
	int last_metric;

	StringName selected_area;

	bool updating_frame;

	//int cursor_metric;
	int hover_metric;

	float graph_height_cpu;
	float graph_height_gpu;

	float graph_limit;

	bool seeking;

	Timer *frame_delay;
	Timer *plot_delay;

	void _update_frame(bool p_focus_selected = false);

	void _activate_pressed();
	void _clear_pressed();

	String _get_time_as_text(float p_time);

	//void _make_metric_ptrs(Metric &m);
	void _item_selected();

	void _update_plot();

	void _graph_tex_mouse_exit();

	void _graph_tex_draw();
	void _graph_tex_input(const Ref<InputEvent> &p_ev);

	int _get_cursor_index() const;

	Color _get_color_from_signature(const StringName &p_signature) const;

	void _cursor_metric_changed(double);

	void _combo_changed(int);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void add_frame_metric(const Metric &p_metric);
	void set_enabled(bool p_enable);
	bool is_profiling();
	bool is_seeking() { return seeking; }
	void disable_seeking();

	void clear();

	Vector<Vector<String>> get_data_as_csv() const;

	EditorVisualProfiler();
};

#endif // EDITOR_FRAME_PROFILER_H
