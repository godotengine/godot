/*************************************************************************/
/*  editor_profiler.cpp                                                  */
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

#include "editor_profiler.h"

#include "core/os/os.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"

void EditorProfiler::_make_metric_ptrs(Metric &m) {
	for (int i = 0; i < m.categories.size(); i++) {
		m.category_ptrs[m.categories[i].signature] = &m.categories.write[i];
		for (int j = 0; j < m.categories[i].items.size(); j++) {
			m.item_ptrs[m.categories[i].items[j].signature] = &m.categories.write[i].items.write[j];
		}
	}
}

EditorProfiler::Metric EditorProfiler::_get_frame_metric(int index) {
	return frame_metrics[(frame_metrics.size() + last_metric - (total_metrics - 1) + index) % frame_metrics.size()];
}

void EditorProfiler::add_frame_metric(const Metric &p_metric, bool p_final) {
	++last_metric;
	if (last_metric >= frame_metrics.size()) {
		last_metric = 0;
	}

	total_metrics++;
	if (total_metrics > frame_metrics.size()) {
		total_metrics = frame_metrics.size();
	}

	frame_metrics.write[last_metric] = p_metric;
	_make_metric_ptrs(frame_metrics.write[last_metric]);

	updating_frame = true;
	clear_button->set_disabled(false);
	cursor_metric_edit->set_editable(true);
	cursor_metric_edit->set_max(p_metric.frame_number);
	cursor_metric_edit->set_min(_get_frame_metric(0).frame_number);

	if (!seeking) {
		cursor_metric_edit->set_value(p_metric.frame_number);
	}

	updating_frame = false;

	if (frame_delay->is_stopped()) {
		frame_delay->set_wait_time(p_final ? 0.1 : 1);
		frame_delay->start();
	}

	if (plot_delay->is_stopped()) {
		plot_delay->set_wait_time(0.1);
		plot_delay->start();
	}
}

void EditorProfiler::clear() {
	int metric_size = EditorSettings::get_singleton()->get("debugger/profiler_frame_history_size");
	metric_size = CLAMP(metric_size, 60, 1024);
	frame_metrics.clear();
	frame_metrics.resize(metric_size);
	total_metrics = 0;
	last_metric = -1;
	variables->clear();
	plot_sigs.clear();
	plot_sigs.insert("physics_frame_time");
	plot_sigs.insert("category_frame_time");

	updating_frame = true;
	cursor_metric_edit->set_min(0);
	cursor_metric_edit->set_max(100); // Doesn't make much sense, but we can't have min == max. Doesn't hurt.
	cursor_metric_edit->set_value(0);
	cursor_metric_edit->set_editable(false);
	updating_frame = false;
	hover_metric = -1;
	seeking = false;
}

static String _get_percent_txt(float p_value, float p_total) {
	if (p_total == 0) {
		p_total = 0.00001;
	}

	return TS->format_number(String::num((p_value / p_total) * 100, 1)) + TS->percent_sign();
}

String EditorProfiler::_get_time_as_text(const Metric &m, float p_time, int p_calls) {
	const int dmode = display_mode->get_selected();

	if (dmode == DISPLAY_FRAME_TIME) {
		return TS->format_number(rtos(p_time * 1000).pad_decimals(2)) + " " + RTR("ms");
	} else if (dmode == DISPLAY_AVERAGE_TIME) {
		if (p_calls == 0) {
			return TS->format_number("0.00") + " " + RTR("ms");
		} else {
			return TS->format_number(rtos((p_time / p_calls) * 1000).pad_decimals(2)) + " " + RTR("ms");
		}
	} else if (dmode == DISPLAY_FRAME_PERCENT) {
		return _get_percent_txt(p_time, m.frame_time);
	} else if (dmode == DISPLAY_PHYSICS_FRAME_PERCENT) {
		return _get_percent_txt(p_time, m.physics_frame_time);
	}

	return "err";
}

Color EditorProfiler::_get_color_from_signature(const StringName &p_signature) const {
	Color bc = get_theme_color(SNAME("error_color"), SNAME("Editor"));
	double rot = ABS(double(p_signature.hash()) / double(0x7FFFFFFF));
	Color c;
	c.set_hsv(rot, bc.get_s(), bc.get_v());
	return c.lerp(get_theme_color(SNAME("base_color"), SNAME("Editor")), 0.07);
}

void EditorProfiler::_item_edited() {
	if (updating_frame) {
		return;
	}

	TreeItem *item = variables->get_edited();
	if (!item) {
		return;
	}
	StringName signature = item->get_metadata(0);
	bool checked = item->is_checked(0);

	if (checked) {
		plot_sigs.insert(signature);
	} else {
		plot_sigs.erase(signature);
	}

	if (!frame_delay->is_processing()) {
		frame_delay->set_wait_time(0.1);
		frame_delay->start();
	}

	_update_plot();
}

void EditorProfiler::_update_plot() {
	const int w = graph->get_size().width;
	const int h = graph->get_size().height;
	bool reset_texture = false;
	const int desired_len = w * h * 4;

	if (graph_image.size() != desired_len) {
		reset_texture = true;
		graph_image.resize(desired_len);
	}

	uint8_t *wr = graph_image.ptrw();
	const Color background_color = get_theme_color(SNAME("dark_color_2"), SNAME("Editor"));

	// Clear the previous frame and set the background color.
	for (int i = 0; i < desired_len; i += 4) {
		wr[i + 0] = Math::fast_ftoi(background_color.r * 255);
		wr[i + 1] = Math::fast_ftoi(background_color.g * 255);
		wr[i + 2] = Math::fast_ftoi(background_color.b * 255);
		wr[i + 3] = 255;
	}

	//find highest value

	const bool use_self = display_time->get_selected() == DISPLAY_SELF_TIME;
	float highest = 0;

	for (int i = 0; i < total_metrics; i++) {
		const Metric &m = _get_frame_metric(i);

		for (Set<StringName>::Element *E = plot_sigs.front(); E; E = E->next()) {
			const Map<StringName, Metric::Category *>::Element *F = m.category_ptrs.find(E->get());
			if (F) {
				highest = MAX(F->get()->total_time, highest);
			}

			const Map<StringName, Metric::Category::Item *>::Element *G = m.item_ptrs.find(E->get());
			if (G) {
				if (use_self) {
					highest = MAX(G->get()->self, highest);
				} else {
					highest = MAX(G->get()->total, highest);
				}
			}
		}
	}

	if (highest > 0) {
		//means some data exists..
		highest *= 1.2; //leave some upper room
		graph_height = highest;

		Vector<int> columnv;
		columnv.resize(h * 4);

		int *column = columnv.ptrw();

		Map<StringName, int> prev_plots;

		for (int i = 0; i < total_metrics * w / frame_metrics.size() - 1; i++) {
			for (int j = 0; j < h * 4; j++) {
				column[j] = 0;
			}

			int current = i * frame_metrics.size() / w;

			for (Set<StringName>::Element *E = plot_sigs.front(); E; E = E->next()) {
				const Metric &m = _get_frame_metric(current);

				float value = 0;

				const Map<StringName, Metric::Category *>::Element *F = m.category_ptrs.find(E->get());
				if (F) {
					value = F->get()->total_time;
				}

				const Map<StringName, Metric::Category::Item *>::Element *G = m.item_ptrs.find(E->get());
				if (G) {
					if (use_self) {
						value = G->get()->self;
					} else {
						value = G->get()->total;
					}
				}

				int plot_pos = CLAMP(int(value * h / highest), 0, h - 1);

				int prev_plot = plot_pos;
				Map<StringName, int>::Element *H = prev_plots.find(E->get());
				if (H) {
					prev_plot = H->get();
					H->get() = plot_pos;
				} else {
					prev_plots[E->get()] = plot_pos;
				}

				plot_pos = h - plot_pos - 1;
				prev_plot = h - prev_plot - 1;

				if (prev_plot > plot_pos) {
					SWAP(prev_plot, plot_pos);
				}

				Color col = _get_color_from_signature(E->get());

				for (int j = prev_plot; j <= plot_pos; j++) {
					column[j * 4 + 0] += Math::fast_ftoi(CLAMP(col.r * 255, 0, 255));
					column[j * 4 + 1] += Math::fast_ftoi(CLAMP(col.g * 255, 0, 255));
					column[j * 4 + 2] += Math::fast_ftoi(CLAMP(col.b * 255, 0, 255));
					column[j * 4 + 3] += 1;
				}
			}

			for (int j = 0; j < h * 4; j += 4) {
				const int a = column[j + 3];
				if (a > 0) {
					column[j + 0] /= a;
					column[j + 1] /= a;
					column[j + 2] /= a;
				}

				const uint8_t red = uint8_t(column[j + 0]);
				const uint8_t green = uint8_t(column[j + 1]);
				const uint8_t blue = uint8_t(column[j + 2]);
				const bool is_filled = red >= 1 || green >= 1 || blue >= 1;
				const int widx = ((j >> 2) * w + i) * 4;

				// If the pixel isn't filled by any profiler line, apply the background color instead.
				wr[widx + 0] = is_filled ? red : Math::fast_ftoi(background_color.r * 255);
				wr[widx + 1] = is_filled ? green : Math::fast_ftoi(background_color.g * 255);
				wr[widx + 2] = is_filled ? blue : Math::fast_ftoi(background_color.b * 255);
				wr[widx + 3] = 255;
			}
		}
	}

	Ref<Image> img;
	img.instantiate();
	img->create(w, h, false, Image::FORMAT_RGBA8, graph_image);

	if (reset_texture) {
		if (graph_texture.is_null()) {
			graph_texture.instantiate();
		}
		graph_texture->create_from_image(img);
	}

	graph_texture->update(img);

	graph->set_texture(graph_texture);
	graph->update();
}

void EditorProfiler::_update_frame() {
	int cursor_metric = cursor_metric_edit->get_value() - _get_frame_metric(0).frame_number;

	updating_frame = true;
	variables->clear();

	TreeItem *root = variables->create_item();
	const Metric &m = _get_frame_metric(cursor_metric);

	int dtime = display_time->get_selected();

	for (int i = 0; i < m.categories.size(); i++) {
		TreeItem *category = variables->create_item(root);
		category->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		category->set_editable(0, true);
		category->set_metadata(0, m.categories[i].signature);
		category->set_text(0, String(m.categories[i].name));
		category->set_text(1, _get_time_as_text(m, m.categories[i].total_time, 1));

		if (plot_sigs.has(m.categories[i].signature)) {
			category->set_checked(0, true);
			category->set_custom_color(0, _get_color_from_signature(m.categories[i].signature));
		}

		for (int j = m.categories[i].items.size() - 1; j >= 0; j--) {
			const Metric::Category::Item &it = m.categories[i].items[j];

			TreeItem *item = variables->create_item(category);
			item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
			item->set_editable(0, true);
			item->set_text(0, it.name);
			item->set_metadata(0, it.signature);
			item->set_metadata(1, it.script);
			item->set_metadata(2, it.line);
			item->set_text_alignment(2, HORIZONTAL_ALIGNMENT_RIGHT);
			item->set_tooltip(0, it.name + "\n" + it.script + ":" + itos(it.line));

			float time = dtime == DISPLAY_SELF_TIME ? it.self : it.total;

			item->set_text(1, _get_time_as_text(m, time, it.calls));

			item->set_text(2, itos(it.calls));

			if (plot_sigs.has(it.signature)) {
				item->set_checked(0, true);
				item->set_custom_color(0, _get_color_from_signature(it.signature));
			}
		}
	}

	updating_frame = false;
}

void EditorProfiler::_activate_pressed() {
	if (activate->is_pressed()) {
		activate->set_icon(get_theme_icon(SNAME("Stop"), SNAME("EditorIcons")));
		activate->set_text(TTR("Stop"));
		_clear_pressed();
	} else {
		activate->set_icon(get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
		activate->set_text(TTR("Start"));
	}
	emit_signal(SNAME("enable_profiling"), activate->is_pressed());
}

void EditorProfiler::_clear_pressed() {
	clear_button->set_disabled(true);
	clear();
	_update_plot();
}

void EditorProfiler::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_LAYOUT_DIRECTION_CHANGED || p_what == NOTIFICATION_TRANSLATION_CHANGED) {
		activate->set_icon(get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
		clear_button->set_icon(get_theme_icon(SNAME("Clear"), SNAME("EditorIcons")));
	}
}

void EditorProfiler::_graph_tex_draw() {
	if (total_metrics == 0) {
		return;
	}
	if (seeking) {
		int frame = cursor_metric_edit->get_value() - _get_frame_metric(0).frame_number;
		int cur_x = (2 * frame + 1) * graph->get_size().x / (2 * frame_metrics.size()) + 1;
		graph->draw_line(Vector2(cur_x, 0), Vector2(cur_x, graph->get_size().y), Color(1, 1, 1, 0.8));
	}
	if (hover_metric > -1 && hover_metric < total_metrics) {
		int cur_x = (2 * hover_metric + 1) * graph->get_size().x / (2 * frame_metrics.size()) + 1;
		graph->draw_line(Vector2(cur_x, 0), Vector2(cur_x, graph->get_size().y), Color(1, 1, 1, 0.4));
	}
}

void EditorProfiler::_graph_tex_mouse_exit() {
	hover_metric = -1;
	graph->update();
}

void EditorProfiler::_cursor_metric_changed(double) {
	if (updating_frame) {
		return;
	}

	graph->update();
	_update_frame();
}

void EditorProfiler::_graph_tex_input(const Ref<InputEvent> &p_ev) {
	if (last_metric < 0) {
		return;
	}

	Ref<InputEventMouse> me = p_ev;
	Ref<InputEventMouseButton> mb = p_ev;
	Ref<InputEventMouseMotion> mm = p_ev;

	if (
			(mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) ||
			(mm.is_valid())) {
		int x = me->get_position().x - 1;
		x = x * frame_metrics.size() / graph->get_size().width;

		hover_metric = x;

		if (x < 0) {
			x = 0;
		}

		if (x >= frame_metrics.size()) {
			x = frame_metrics.size() - 1;
		}

		if (mb.is_valid() || (mm->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE) {
			updating_frame = true;

			if (x < total_metrics)
				cursor_metric_edit->set_value(_get_frame_metric(x).frame_number);
			updating_frame = false;

			if (activate->is_pressed()) {
				if (!seeking) {
					emit_signal(SNAME("break_request"));
				}
			}

			seeking = true;

			if (!frame_delay->is_processing()) {
				frame_delay->set_wait_time(0.1);
				frame_delay->start();
			}
		}

		graph->update();
	}
}

void EditorProfiler::disable_seeking() {
	seeking = false;
	graph->update();
}

void EditorProfiler::_combo_changed(int) {
	_update_frame();
	_update_plot();
}

void EditorProfiler::_bind_methods() {
	ADD_SIGNAL(MethodInfo("enable_profiling", PropertyInfo(Variant::BOOL, "enable")));
	ADD_SIGNAL(MethodInfo("break_request"));
}

void EditorProfiler::set_enabled(bool p_enable) {
	activate->set_disabled(!p_enable);
}

bool EditorProfiler::is_profiling() {
	return activate->is_pressed();
}

Vector<Vector<String>> EditorProfiler::get_data_as_csv() const {
	Vector<Vector<String>> res;

	if (frame_metrics.is_empty()) {
		return res;
	}

	// Different metrics may contain different number of categories.
	Set<StringName> possible_signatures;
	for (int i = 0; i < frame_metrics.size(); i++) {
		const Metric &m = frame_metrics[i];
		if (!m.valid) {
			continue;
		}
		for (const KeyValue<StringName, Metric::Category *> &E : m.category_ptrs) {
			possible_signatures.insert(E.key);
		}
		for (const KeyValue<StringName, Metric::Category::Item *> &E : m.item_ptrs) {
			possible_signatures.insert(E.key);
		}
	}

	// Generate CSV header and cache indices.
	Map<StringName, int> sig_map;
	Vector<String> signatures;
	signatures.resize(possible_signatures.size());
	int sig_index = 0;
	for (const Set<StringName>::Element *E = possible_signatures.front(); E; E = E->next()) {
		signatures.write[sig_index] = E->get();
		sig_map[E->get()] = sig_index;
		sig_index++;
	}
	res.push_back(signatures);

	// values
	Vector<String> values;

	int index = last_metric;

	for (int i = 0; i < frame_metrics.size(); i++) {
		++index;

		if (index >= frame_metrics.size()) {
			index = 0;
		}

		const Metric &m = frame_metrics[index];

		if (!m.valid) {
			continue;
		}

		// Don't keep old values since there may be empty cells.
		values.clear();
		values.resize(possible_signatures.size());

		for (const KeyValue<StringName, Metric::Category *> &E : m.category_ptrs) {
			values.write[sig_map[E.key]] = String::num_real(E.value->total_time);
		}
		for (const KeyValue<StringName, Metric::Category::Item *> &E : m.item_ptrs) {
			values.write[sig_map[E.key]] = String::num_real(E.value->total);
		}

		res.push_back(values);
	}

	return res;
}

EditorProfiler::EditorProfiler() {
	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);
	activate = memnew(Button);
	activate->set_toggle_mode(true);
	activate->set_text(TTR("Start"));
	activate->connect("pressed", callable_mp(this, &EditorProfiler::_activate_pressed));
	hb->add_child(activate);

	clear_button = memnew(Button);
	clear_button->set_text(TTR("Clear"));
	clear_button->connect("pressed", callable_mp(this, &EditorProfiler::_clear_pressed));
	clear_button->set_disabled(true);
	hb->add_child(clear_button);

	hb->add_child(memnew(Label(TTR("Measure:"))));

	display_mode = memnew(OptionButton);
	display_mode->add_item(TTR("Frame Time (ms)"));
	display_mode->add_item(TTR("Average Time (ms)"));
	display_mode->add_item(TTR("Frame %"));
	display_mode->add_item(TTR("Physics Frame %"));
	display_mode->connect("item_selected", callable_mp(this, &EditorProfiler::_combo_changed));

	hb->add_child(display_mode);

	hb->add_child(memnew(Label(TTR("Time:"))));

	display_time = memnew(OptionButton);
	display_time->add_item(TTR("Inclusive"));
	display_time->add_item(TTR("Self"));
	display_time->set_tooltip(TTR("Inclusive: Includes time from other functions called by this function.\nUse this to spot bottlenecks.\n\nSelf: Only count the time spent in the function itself, not in other functions called by that function.\nUse this to find individual functions to optimize."));
	display_time->connect("item_selected", callable_mp(this, &EditorProfiler::_combo_changed));

	hb->add_child(display_time);

	hb->add_spacer();

	hb->add_child(memnew(Label(TTR("Frame #:"))));

	cursor_metric_edit = memnew(SpinBox);
	cursor_metric_edit->set_h_size_flags(SIZE_FILL);
	cursor_metric_edit->set_value(0);
	cursor_metric_edit->set_editable(false);
	hb->add_child(cursor_metric_edit);
	cursor_metric_edit->connect("value_changed", callable_mp(this, &EditorProfiler::_cursor_metric_changed));

	hb->add_theme_constant_override("separation", 8 * EDSCALE);

	h_split = memnew(HSplitContainer);
	add_child(h_split);
	h_split->set_v_size_flags(SIZE_EXPAND_FILL);

	variables = memnew(Tree);
	variables->set_custom_minimum_size(Size2(320, 0) * EDSCALE);
	variables->set_hide_folding(true);
	h_split->add_child(variables);
	variables->set_hide_root(true);
	variables->set_columns(3);
	variables->set_column_titles_visible(true);
	variables->set_column_title(0, TTR("Name"));
	variables->set_column_expand(0, true);
	variables->set_column_clip_content(0, true);
	variables->set_column_expand_ratio(0, 60);
	variables->set_column_title(1, TTR("Time"));
	variables->set_column_expand(1, false);
	variables->set_column_clip_content(1, true);
	variables->set_column_expand_ratio(1, 100);
	variables->set_column_title(2, TTR("Calls"));
	variables->set_column_expand(2, false);
	variables->set_column_clip_content(2, true);
	variables->set_column_expand_ratio(2, 60);
	variables->connect("item_edited", callable_mp(this, &EditorProfiler::_item_edited));

	graph = memnew(TextureRect);
	graph->set_expand(true);
	graph->set_mouse_filter(MOUSE_FILTER_STOP);
	graph->connect("draw", callable_mp(this, &EditorProfiler::_graph_tex_draw));
	graph->connect("gui_input", callable_mp(this, &EditorProfiler::_graph_tex_input));
	graph->connect("mouse_exited", callable_mp(this, &EditorProfiler::_graph_tex_mouse_exit));

	h_split->add_child(graph);
	graph->set_h_size_flags(SIZE_EXPAND_FILL);

	int metric_size = CLAMP(int(EDITOR_DEF("debugger/profiler_frame_history_size", 600)), 60, 1024);
	frame_metrics.resize(metric_size);
	total_metrics = 0;
	last_metric = -1;
	hover_metric = -1;

	EDITOR_DEF("debugger/profiler_frame_max_functions", 64);

	frame_delay = memnew(Timer);
	frame_delay->set_wait_time(0.1);
	frame_delay->set_one_shot(true);
	add_child(frame_delay);
	frame_delay->connect("timeout", callable_mp(this, &EditorProfiler::_update_frame));

	plot_delay = memnew(Timer);
	plot_delay->set_wait_time(0.1);
	plot_delay->set_one_shot(true);
	add_child(plot_delay);
	plot_delay->connect("timeout", callable_mp(this, &EditorProfiler::_update_plot));

	plot_sigs.insert("physics_frame_time");
	plot_sigs.insert("category_frame_time");

	seeking = false;
	graph_height = 1;
}
