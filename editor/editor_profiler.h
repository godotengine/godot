/*************************************************************************/
/*  editor_profiler.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef EDITORPROFILER_H
#define EDITORPROFILER_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"

class EditorProfiler : public VBoxContainer {

	GDCLASS(EditorProfiler, VBoxContainer)

public:
	struct Metric {

		bool valid;

		int frame_number;
		float frame_time;
		float idle_time;
		float fixed_time;
		float fixed_frame_time;

		struct Category {

			StringName signature;
			String name;
			float total_time; //total for category

			struct Item {

				StringName signature;
				String name;
				String script;
				int line;
				float self;
				float total;
				int calls;
			};

			Vector<Item> items;
		};

		Vector<Category> categories;

		Map<StringName, Category *> category_ptrs;
		Map<StringName, Category::Item *> item_ptrs;

		Metric() {
			valid = false;
			frame_number = 0;
		}
	};

	enum DisplayMode {
		DISPLAY_FRAME_TIME,
		DISPLAY_AVERAGE_TIME,
		DISPLAY_FRAME_PERCENT,
		DISPLAY_FIXED_FRAME_PERCENT,
	};

	enum DisplayTime {
		DISPLAY_TOTAL_TIME,
		DISPLAY_SELF_TIME,
	};

private:
	Button *activate;
	TextureRect *graph;
	Ref<ImageTexture> graph_texture;
	PoolVector<uint8_t> graph_image;
	Tree *variables;
	HSplitContainer *h_split;

	Set<StringName> plot_sigs;

	OptionButton *display_mode;
	OptionButton *display_time;

	SpinBox *cursor_metric_edit;

	Vector<Metric> frame_metrics;
	int last_metric;

	int max_functions;

	bool updating_frame;

	//int cursor_metric;
	int hover_metric;

	float graph_height;

	bool seeking;

	Timer *frame_delay;
	Timer *plot_delay;

	void _update_frame();

	void _activate_pressed();

	String _get_time_as_text(Metric &m, float p_time, int p_calls);

	void _make_metric_ptrs(Metric &m);
	void _item_edited();

	void _update_plot();

	void _graph_tex_mouse_exit();

	void _graph_tex_draw();
	void _graph_tex_input(const InputEvent &p_ev);

	int _get_cursor_index() const;

	Color _get_color_from_signature(const StringName &p_signature) const;

	void _cursor_metric_changed(double);

	void _combo_changed(int);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void add_frame_metric(const Metric &p_metric, bool p_final = false);
	void set_enabled(bool p_enable);
	bool is_profiling();
	bool is_seeking() { return seeking; }
	void disable_seeking();

	void clear();

	EditorProfiler();
};

#endif // EDITORPROFILER_H
