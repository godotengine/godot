/**************************************************************************/
/*  editor_quick_open_dialog.cpp                                          */
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

#include "editor_quick_open_dialog.h"

#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_resource_preview.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/center_container.h"
#include "scene/gui/check_button.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"

const Vector2i MAGIC_HIGHLIGHT_OFFSET = { 4, 5 };
const String DISPLAY_MODE_SETTING = "filesystem/quick_open_dialog/default_display_mode";
const String FUZZY_MATCHING_SETTING = "filesystem/quick_open_dialog/enable_fuzzy_matching";
const String INCLUDE_ADDONS_SETTING = "filesystem/quick_open_dialog/include_addons";
const String MAX_RESULTS_SETTING = "filesystem/quick_open_dialog/max_results";
const String MAX_MISSES_SETTING = "filesystem/quick_open_dialog/max_fuzzy_misses";
const String SEARCH_HIGHLIGHT_SETTING = "filesystem/quick_open_dialog/show_search_highlight";

EditorQuickOpenDialog::EditorQuickOpenDialog() {
	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->add_theme_constant_override("separation", 0);
	add_child(vbc);

	{
		// Search bar
		MarginContainer *mc = memnew(MarginContainer);
		mc->add_theme_constant_override("margin_top", 6);
		mc->add_theme_constant_override("margin_bottom", 6);
		mc->add_theme_constant_override("margin_left", 1);
		mc->add_theme_constant_override("margin_right", 1);
		vbc->add_child(mc);

		search_box = memnew(LineEdit);
		search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		search_box->set_placeholder(TTR("Search files..."));
		search_box->set_clear_button_enabled(true);
		mc->add_child(search_box);
	}

	{
		container = memnew(QuickOpenResultContainer);
		container->connect("result_clicked", callable_mp(this, &EditorQuickOpenDialog::ok_pressed));
		vbc->add_child(container);
	}

	search_box->connect(SceneStringName(text_changed), callable_mp(this, &EditorQuickOpenDialog::_search_box_text_changed));
	search_box->connect(SceneStringName(gui_input), callable_mp(container, &QuickOpenResultContainer::handle_search_box_input));
	register_text_enter(search_box);
	get_ok_button()->hide();
}

String EditorQuickOpenDialog::get_dialog_title(const Vector<StringName> &p_base_types) {
	if (p_base_types.size() > 1) {
		return TTR("Select Resource");
	}

	if (p_base_types[0] == SNAME("PackedScene")) {
		return TTR("Select Scene");
	}

	return TTR("Select") + " " + p_base_types[0];
}

void EditorQuickOpenDialog::popup_dialog(const Vector<StringName> &p_base_types, const Callable &p_item_selected_callback) {
	ERR_FAIL_COND(p_base_types.is_empty());
	ERR_FAIL_COND(!p_item_selected_callback.is_valid());

	item_selected_callback = p_item_selected_callback;

	container->init(p_base_types);
	get_ok_button()->set_disabled(container->has_nothing_selected());

	set_title(get_dialog_title(p_base_types));
	popup_centered_clamped(Size2(655, 650) * EDSCALE, 0.8f);
	search_box->grab_focus();
}

void EditorQuickOpenDialog::ok_pressed() {
	item_selected_callback.call(container->get_selected());

	container->save_selected_item();
	container->cleanup();
	search_box->clear();
	hide();
}

void EditorQuickOpenDialog::cancel_pressed() {
	container->cleanup();
	search_box->clear();
}

void EditorQuickOpenDialog::_search_box_text_changed(const String &p_query) {
	container->set_query_and_update(p_query);
	get_ok_button()->set_disabled(container->has_nothing_selected());
}

//------------------------- Result Container

void style_button(Button *p_button) {
	p_button->set_flat(true);
	p_button->set_focus_mode(Control::FOCUS_NONE);
	p_button->set_default_cursor_shape(Control::CURSOR_POINTING_HAND);
}

QuickOpenResultContainer::QuickOpenResultContainer() {
	fuzzy_search.instantiate();
	set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_theme_constant_override("separation", 0);

	{
		// Results section
		panel_container = memnew(PanelContainer);
		panel_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		add_child(panel_container);

		{
			// No search results
			no_results_container = memnew(CenterContainer);
			no_results_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			no_results_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			panel_container->add_child(no_results_container);

			no_results_label = memnew(Label);
			no_results_label->add_theme_font_size_override(SceneStringName(font_size), 24 * EDSCALE);
			no_results_container->add_child(no_results_label);
			no_results_container->hide();
		}

		{
			// Search results
			scroll_container = memnew(ScrollContainer);
			scroll_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			scroll_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			scroll_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
			scroll_container->hide();
			panel_container->add_child(scroll_container);

			list = memnew(VBoxContainer);
			list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			list->hide();
			scroll_container->add_child(list);

			grid = memnew(HFlowContainer);
			grid->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			grid->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			grid->add_theme_constant_override("v_separation", 18);
			grid->add_theme_constant_override("h_separation", 4);
			grid->hide();
			scroll_container->add_child(grid);
		}
	}

	{
		// Bottom bar
		HBoxContainer *bottom_bar = memnew(HBoxContainer);
		bottom_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		add_child(bottom_bar);

		file_details_path = memnew(Label);
		file_details_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		file_details_path->set_horizontal_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
		file_details_path->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
		bottom_bar->add_child(file_details_path);

		{
			HBoxContainer *hbc = memnew(HBoxContainer);
			hbc->add_theme_constant_override("separation", 3);
			bottom_bar->add_child(hbc);

			fuzzy_search_toggle = memnew(CheckButton);
			style_button(fuzzy_search_toggle);
			fuzzy_search_toggle->set_tooltip_text(TTR("Enable fuzzy matching"));
			fuzzy_search_toggle->connect(SceneStringName(toggled), callable_mp(this, &QuickOpenResultContainer::_toggle_fuzzy_search));
			hbc->add_child(fuzzy_search_toggle);

			include_addons_toggle = memnew(CheckButton);
			style_button(include_addons_toggle);
			include_addons_toggle->set_tooltip_text(TTR("Include files from addons"));
			include_addons_toggle->connect(SceneStringName(toggled), callable_mp(this, &QuickOpenResultContainer::_toggle_include_addons));
			hbc->add_child(include_addons_toggle);

			VSeparator *vsep = memnew(VSeparator);
			vsep->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
			vsep->set_custom_minimum_size(Size2i(0, 14 * EDSCALE));
			hbc->add_child(vsep);

			display_mode_toggle = memnew(Button);
			style_button(display_mode_toggle);
			display_mode_toggle->connect(SceneStringName(pressed), callable_mp(this, &QuickOpenResultContainer::_toggle_display_mode));
			hbc->add_child(display_mode_toggle);
		}
	}

	// Creating and deleting nodes while searching is slow, so we allocate
	// a bunch of result nodes and fill in the content based on result ranking.
	result_items.resize(EDITOR_GET(MAX_RESULTS_SETTING));
	for (int i = 0; i < result_items.size(); i++) {
		QuickOpenResultItem *item = memnew(QuickOpenResultItem);
		item->connect(SceneStringName(gui_input), callable_mp(this, &QuickOpenResultContainer::_item_input).bind(i));
		result_items.write[i] = item;
	}
}

QuickOpenResultContainer::~QuickOpenResultContainer() {
	if (never_opened) {
		for (QuickOpenResultItem *E : result_items) {
			memdelete(E);
		}
	}
}

void QuickOpenResultContainer::init(const Vector<StringName> &p_base_types) {
	base_types = p_base_types;

	const int display_mode_behavior = EDITOR_GET(DISPLAY_MODE_SETTING);
	const bool adaptive_display_mode = (display_mode_behavior == 0);

	if (adaptive_display_mode) {
		_set_display_mode(adaptive_display_mode ? get_adaptive_display_mode(p_base_types) : QuickOpenDisplayMode::LIST);
	} else if (never_opened) {
		int last = EditorSettings::get_singleton()->get_project_metadata("quick_open_dialog", "last_mode", (int)QuickOpenDisplayMode::LIST);
		_set_display_mode((QuickOpenDisplayMode)last);
	}

	const bool fuzzy_matching = EDITOR_GET(FUZZY_MATCHING_SETTING);
	const bool include_addons = EDITOR_GET(INCLUDE_ADDONS_SETTING);
	fuzzy_search_toggle->set_pressed_no_signal(fuzzy_matching);
	include_addons_toggle->set_pressed_no_signal(include_addons);
	never_opened = false;

	const bool enable_highlights = EDITOR_GET(SEARCH_HIGHLIGHT_SETTING);
	for (QuickOpenResultItem *E : result_items) {
		E->enable_highlights = enable_highlights;
	}

	_create_initial_results();
}

void QuickOpenResultContainer::_create_initial_results() {
	file_type_icons.clear();
	file_type_icons.insert("__default_icon", get_editor_theme_icon(SNAME("Object")));
	filepaths.clear();
	filetypes.clear();
	_find_filepaths_in_folder(EditorFileSystem::get_singleton()->get_filesystem(), include_addons_toggle->is_pressed());
	max_total_results = MIN(filepaths.size(), result_items.size());
	update_results();
}

void QuickOpenResultContainer::_find_filepaths_in_folder(EditorFileSystemDirectory *p_directory, bool p_include_addons) {
	for (int i = 0; i < p_directory->get_subdir_count(); i++) {
		if (p_include_addons || p_directory->get_name() != "addons") {
			_find_filepaths_in_folder(p_directory->get_subdir(i), p_include_addons);
		}
	}

	for (int i = 0; i < p_directory->get_file_count(); i++) {
		String file_path = p_directory->get_file_path(i);

		const StringName engine_type = p_directory->get_file_type(i);
		const StringName script_type = p_directory->get_file_resource_script_class(i);

		const bool is_engine_type = script_type == StringName();
		const StringName &actual_type = is_engine_type ? engine_type : script_type;

		for (const StringName &parent_type : base_types) {
			bool is_valid = ClassDB::is_parent_class(engine_type, parent_type) || (!is_engine_type && EditorNode::get_editor_data().script_class_is_parent(script_type, parent_type));

			if (is_valid) {
				filepaths.append(file_path);
				filetypes.insert(file_path, actual_type);
				break; // Stop testing base types as soon as we get a match.
			}
		}
	}
}

void QuickOpenResultContainer::set_query_and_update(const String &p_query) {
	query = p_query;
	update_results();
}

void QuickOpenResultContainer::update_results() {
	_score_and_sort_candidates();
	_update_result_items(MIN(candidates.size(), max_total_results), 0);
}

void QuickOpenResultContainer::_score_and_sort_candidates() {
	candidates.clear();

	if (query.is_empty()) {
		return;
	}

	fuzzy_search->set_query(query);
	fuzzy_search->max_results = max_total_results;
	bool fuzzy_matching = EDITOR_GET(FUZZY_MATCHING_SETTING);
	int max_misses = EDITOR_GET(MAX_MISSES_SETTING);
	fuzzy_search->allow_subsequences = fuzzy_matching;
	fuzzy_search->max_misses = fuzzy_matching ? max_misses : 0;

	Vector<Ref<FuzzySearchResult>> results = fuzzy_search->search_all(filepaths);
	candidates.resize(results.size());

	for (int i = 0; i < results.size(); i++) {
		String filename = results[i]->target;
		StringName actual_type = *filetypes.lookup_ptr(results[i]->target);
		QuickOpenResultCandidate candidate;
		candidate.file_name = filename.get_file();
		candidate.file_directory = filename.get_base_dir();
		candidate.result = results[i];

		EditorResourcePreview::PreviewItem item = EditorResourcePreview::get_singleton()->get_resource_preview_if_available(filename);
		if (item.preview.is_valid()) {
			candidate.thumbnail = item.preview;
		} else if (file_type_icons.has(actual_type)) {
			candidate.thumbnail = *file_type_icons.lookup_ptr(actual_type);
		} else if (has_theme_icon(actual_type, EditorStringName(EditorIcons))) {
			candidate.thumbnail = get_editor_theme_icon(actual_type);
			file_type_icons.insert(actual_type, candidate.thumbnail);
		} else {
			candidate.thumbnail = *file_type_icons.lookup_ptr("__default_icon");
		}

		candidates.set(i, candidate);
	}
}

void QuickOpenResultContainer::_update_result_items(int p_new_visible_results_count, int p_new_selection_index) {
	List<QuickOpenResultCandidate> *type_history = nullptr;

	showing_history = false;

	if (query.is_empty()) {
		if (candidates.size() <= SHOW_ALL_FILES_THRESHOLD) {
			p_new_visible_results_count = candidates.size();
		} else {
			p_new_visible_results_count = 0;

			if (base_types.size() == 1) {
				type_history = selected_history.lookup_ptr(base_types[0]);
				if (type_history) {
					p_new_visible_results_count = type_history->size();
					showing_history = true;
				}
			}
		}
	}

	// Only need to update items that were not hidden in previous update.
	int num_items_needing_updates = MAX(num_visible_results, p_new_visible_results_count);
	num_visible_results = p_new_visible_results_count;

	for (int i = 0; i < num_items_needing_updates; i++) {
		QuickOpenResultItem *item = result_items[i];

		if (i < num_visible_results) {
			item->set_content(type_history ? type_history->get(i) : candidates[i]);
		} else {
			item->reset();
		}
	};

	const bool any_results = num_visible_results > 0;
	_select_item(any_results ? p_new_selection_index : -1);

	scroll_container->set_visible(any_results);
	no_results_container->set_visible(!any_results);

	if (!any_results) {
		if (filepaths.is_empty()) {
			no_results_label->set_text(TTR("No files found for this type"));
		} else if (query.is_empty()) {
			no_results_label->set_text(TTR("Start searching to find files..."));
		} else {
			no_results_label->set_text(TTR("No results found"));
		}
	}
}

void QuickOpenResultContainer::handle_search_box_input(const Ref<InputEvent> &p_ie) {
	if (num_visible_results < 0) {
		return;
	}

	Ref<InputEventKey> key_event = p_ie;
	if (key_event.is_valid() && key_event->is_pressed()) {
		bool move_selection = false;

		switch (key_event->get_keycode()) {
			case Key::UP:
			case Key::DOWN:
			case Key::PAGEUP:
			case Key::PAGEDOWN: {
				move_selection = true;
			} break;
			case Key::LEFT:
			case Key::RIGHT: {
				if (content_display_mode == QuickOpenDisplayMode::GRID) {
					if (key_event->get_modifiers_mask() == 0) {
						move_selection = true;
					}
				}
			} break;
			default:
				break; // Let the event through so it will reach the search box.
		}

		if (move_selection) {
			_move_selection_index(key_event->get_keycode());
			queue_redraw();
			accept_event();
		}
	}
}

void QuickOpenResultContainer::_move_selection_index(Key p_key) {
	const int max_index = num_visible_results - 1;

	int idx = selection_index;
	if (content_display_mode == QuickOpenDisplayMode::LIST) {
		if (p_key == Key::UP) {
			idx = (idx == 0) ? max_index : (idx - 1);
		} else if (p_key == Key::DOWN) {
			idx = (idx == max_index) ? 0 : (idx + 1);
		} else if (p_key == Key::PAGEUP) {
			idx = (idx == 0) ? idx : MAX(idx - 10, 0);
		} else if (p_key == Key::PAGEDOWN) {
			idx = (idx == max_index) ? idx : MIN(idx + 10, max_index);
		}
	} else {
		int column_count = grid->get_line_max_child_count();

		if (p_key == Key::LEFT) {
			idx = (idx == 0) ? max_index : (idx - 1);
		} else if (p_key == Key::RIGHT) {
			idx = (idx == max_index) ? 0 : (idx + 1);
		} else if (p_key == Key::UP) {
			idx = (idx == 0) ? max_index : MAX(idx - column_count, 0);
		} else if (p_key == Key::DOWN) {
			idx = (idx == max_index) ? 0 : MIN(idx + column_count, max_index);
		} else if (p_key == Key::PAGEUP) {
			idx = (idx == 0) ? idx : MAX(idx - (3 * column_count), 0);
		} else if (p_key == Key::PAGEDOWN) {
			idx = (idx == max_index) ? idx : MIN(idx + (3 * column_count), max_index);
		}
	}

	_select_item(idx);
}

void QuickOpenResultContainer::_select_item(int p_index) {
	if (!has_nothing_selected()) {
		result_items[selection_index]->highlight_item(false);
	}

	selection_index = p_index;

	if (has_nothing_selected()) {
		file_details_path->set_text("");
		return;
	}

	result_items[selection_index]->highlight_item(true);
	file_details_path->set_text(get_selected() + (showing_history ? TTR(" (recently opened)") : ""));

	const QuickOpenResultItem *item = result_items[selection_index];

	// Copied from Tree.
	const int selected_position = item->get_position().y;
	const int selected_size = item->get_size().y;
	const int scroll_window_size = scroll_container->get_size().y;
	const int scroll_position = scroll_container->get_v_scroll();

	if (selected_position <= scroll_position) {
		scroll_container->set_v_scroll(selected_position);
	} else if (selected_position + selected_size > scroll_position + scroll_window_size) {
		scroll_container->set_v_scroll(selected_position + selected_size - scroll_window_size);
	}
}

void QuickOpenResultContainer::_item_input(const Ref<InputEvent> &p_ev, int p_index) {
	Ref<InputEventMouseButton> mb = p_ev;

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		_select_item(p_index);
		emit_signal(SNAME("result_clicked"));
	}
}

void QuickOpenResultContainer::_toggle_fuzzy_search(bool p_pressed) {
	EditorSettings::get_singleton()->set(FUZZY_MATCHING_SETTING, p_pressed);
	update_results();
}

void QuickOpenResultContainer::_toggle_include_addons(bool p_pressed) {
	EditorSettings::get_singleton()->set(INCLUDE_ADDONS_SETTING, p_pressed);
	cleanup();
	_create_initial_results();
}

void QuickOpenResultContainer::_toggle_display_mode() {
	QuickOpenDisplayMode new_display_mode = (content_display_mode == QuickOpenDisplayMode::LIST) ? QuickOpenDisplayMode::GRID : QuickOpenDisplayMode::LIST;
	_set_display_mode(new_display_mode);
}

void QuickOpenResultContainer::_set_display_mode(QuickOpenDisplayMode p_display_mode) {
	content_display_mode = p_display_mode;
	EditorSettings::get_singleton()->set_project_metadata("quick_open_dialog", "last_mode", (int)content_display_mode);

	const bool show_list = (content_display_mode == QuickOpenDisplayMode::LIST);
	if ((show_list && list->is_visible()) || (!show_list && grid->is_visible())) {
		return;
	}

	hide();

	// Move result item nodes from one container to the other.
	CanvasItem *prev_root;
	CanvasItem *next_root;
	if (content_display_mode == QuickOpenDisplayMode::LIST) {
		prev_root = Object::cast_to<CanvasItem>(grid);
		next_root = Object::cast_to<CanvasItem>(list);
	} else {
		prev_root = Object::cast_to<CanvasItem>(list);
		next_root = Object::cast_to<CanvasItem>(grid);
	}

	const bool first_time = !list->is_visible() && !grid->is_visible();

	prev_root->hide();
	for (QuickOpenResultItem *item : result_items) {
		item->set_display_mode(content_display_mode);

		if (!first_time) {
			prev_root->remove_child(item);
		}

		next_root->add_child(item);
	}
	next_root->show();
	show();

	_update_result_items(num_visible_results, selection_index);

	if (content_display_mode == QuickOpenDisplayMode::LIST) {
		display_mode_toggle->set_icon(get_editor_theme_icon(SNAME("FileThumbnail")));
		display_mode_toggle->set_tooltip_text(TTR("Grid view"));
	} else {
		display_mode_toggle->set_icon(get_editor_theme_icon(SNAME("FileList")));
		display_mode_toggle->set_tooltip_text(TTR("List view"));
	}
}

bool QuickOpenResultContainer::has_nothing_selected() const {
	return selection_index < 0;
}

String QuickOpenResultContainer::get_selected() const {
	ERR_FAIL_COND_V_MSG(has_nothing_selected(), String(), "Tried to get selected file, but nothing was selected.");

	if (showing_history) {
		const List<QuickOpenResultCandidate> *type_history = selected_history.lookup_ptr(base_types[0]);

		const QuickOpenResultCandidate &c = type_history->get(selection_index);
		return c.file_directory.path_join(c.file_name);
	} else {
		const QuickOpenResultCandidate &c = candidates[selection_index];
		return c.file_directory.path_join(c.file_name);
	}
}

QuickOpenDisplayMode QuickOpenResultContainer::get_adaptive_display_mode(const Vector<StringName> &p_base_types) {
	static const Vector<StringName> grid_preferred_types = {
		"Font",
		"Texture2D",
		"Material",
		"Mesh"
	};

	for (const StringName &type : grid_preferred_types) {
		for (const StringName &base_type : p_base_types) {
			if (base_type == type || ClassDB::is_parent_class(base_type, type))
				return QuickOpenDisplayMode::GRID;
		}
	}

	return QuickOpenDisplayMode::LIST;
}

void QuickOpenResultContainer::save_selected_item() {
	if (base_types.size() > 1) {
		// Getting the type of the file and checking which base type it belongs to should be possible.
		// However, for now these are not supported, and we don't record this.
		return;
	}

	if (showing_history) {
		// Selecting from history, so already added.
		return;
	}

	const StringName &base_type = base_types[0];

	List<QuickOpenResultCandidate> *type_history = selected_history.lookup_ptr(base_type);
	if (!type_history) {
		selected_history.insert(base_type, List<QuickOpenResultCandidate>());
		type_history = selected_history.lookup_ptr(base_type);
	} else {
		const QuickOpenResultCandidate &selected = candidates[selection_index];

		for (const QuickOpenResultCandidate &candidate : *type_history) {
			if (candidate.file_directory == selected.file_directory && candidate.file_name == selected.file_name) {
				return;
			}
		}

		if (type_history->size() > 8) {
			type_history->pop_back();
		}
	}

	type_history->push_front(candidates[selection_index]);
}

void QuickOpenResultContainer::cleanup() {
	num_visible_results = 0;
	candidates.clear();
	_select_item(-1);

	for (QuickOpenResultItem *item : result_items) {
		item->reset();
	}
}

void QuickOpenResultContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			Color text_color = get_theme_color("font_readonly_color", EditorStringName(Editor));
			file_details_path->add_theme_color_override(SceneStringName(font_color), text_color);
			no_results_label->add_theme_color_override(SceneStringName(font_color), text_color);

			panel_container->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SceneStringName(panel), SNAME("Tree")));

			if (content_display_mode == QuickOpenDisplayMode::LIST) {
				display_mode_toggle->set_icon(get_editor_theme_icon(SNAME("FileThumbnail")));
			} else {
				display_mode_toggle->set_icon(get_editor_theme_icon(SNAME("FileList")));
			}
		} break;
	}
}

void QuickOpenResultContainer::_bind_methods() {
	ADD_SIGNAL(MethodInfo("result_clicked"));
}

//------------------------- Result Item

QuickOpenResultItem::QuickOpenResultItem() {
	set_focus_mode(FocusMode::FOCUS_ALL);
	_set_enabled(false);
	set_default_cursor_shape(CURSOR_POINTING_HAND);

	list_item = memnew(QuickOpenResultListItem);
	list_item->hide();
	add_child(list_item);

	grid_item = memnew(QuickOpenResultGridItem);
	grid_item->hide();
	add_child(grid_item);
}

void QuickOpenResultItem::set_display_mode(QuickOpenDisplayMode p_display_mode) {
	if (p_display_mode == QuickOpenDisplayMode::LIST) {
		grid_item->hide();
		grid_item->reset();
		list_item->show();
	} else {
		list_item->hide();
		list_item->reset();
		grid_item->show();
	}

	dirty_highlights = true;
	queue_redraw();
}

void QuickOpenResultItem::set_content(const QuickOpenResultCandidate &p_candidate) {
	_set_enabled(true);

	if (list_item->is_visible()) {
		list_item->set_content(p_candidate);
	} else {
		grid_item->set_content(p_candidate);
	}

	queue_redraw();
}

void QuickOpenResultItem::reset() {
	_set_enabled(false);

	is_hovering = false;
	is_selected = false;
	list_item->reset();
	grid_item->reset();
}

void QuickOpenResultItem::highlight_item(bool p_enabled) {
	is_selected = p_enabled;

	if (list_item->is_visible()) {
		if (p_enabled) {
			list_item->highlight_item(highlighted_font_color);
		} else {
			list_item->remove_highlight();
		}
	} else {
		if (p_enabled) {
			grid_item->highlight_item(highlighted_font_color);
		} else {
			grid_item->remove_highlight();
		}
	}

	queue_redraw();
}

void QuickOpenResultItem::draw_search_highlights() {
	if (dirty_highlights) {
		// When initially switching layouts, the new sub-item has not yet been positioned, so this
		// delays finding and drawing highlights until after that happens.
		dirty_highlights = false;
		callable_mp(Object::cast_to<CanvasItem>(this), &CanvasItem::queue_redraw).call_deferred();
		return;
	}

	Control *item;
	Vector<Rect2i> highlights;

	if (list_item->is_visible()) {
		item = list_item;
		highlights = list_item->get_search_highlights();
	} else {
		item = grid_item;
		highlights = grid_item->get_search_highlights();
	}

	Vector2i offset = item->get_position() + MAGIC_HIGHLIGHT_OFFSET * EDSCALE;
	for (Rect2i rect : highlights) {
		rect.position += offset;
		draw_rect(rect, Color(1, 1, 1, 0.07), true);
		draw_rect(rect, Color(0.5, 0.7, 1.0, 0.4), false, 1);
	}
}

void QuickOpenResultItem::_set_enabled(bool p_enabled) {
	set_visible(p_enabled);
	set_process(p_enabled);
	set_process_input(p_enabled);
}

void QuickOpenResultItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_MOUSE_ENTER:
		case NOTIFICATION_MOUSE_EXIT: {
			is_hovering = is_visible() && p_what == NOTIFICATION_MOUSE_ENTER;
			queue_redraw();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			selected_stylebox = get_theme_stylebox("selected", "Tree");
			hovering_stylebox = get_theme_stylebox("hover", "Tree");
			highlighted_font_color = get_theme_color("font_focus_color", EditorStringName(Editor));
		} break;
		case NOTIFICATION_DRAW: {
			if (enable_highlights) {
				draw_search_highlights();
			}
			if (is_selected) {
				draw_style_box(selected_stylebox, Rect2(Point2(), get_size()));
			} else if (is_hovering) {
				draw_style_box(hovering_stylebox, Rect2(Point2(), get_size()));
			}
		} break;
	}
}

//----------------- List item

Vector2i get_path_interval(const Vector2i p_interval, const int p_dir_index) {
	if (p_interval.x >= p_dir_index || p_interval.y < 1) {
		return { -1, -1 };
	}
	return { p_interval.x, MIN(p_interval.x + p_interval.y, p_dir_index) - p_interval.x };
}

Vector2i get_name_interval(const Vector2i p_interval, const int p_dir_index) {
	if (p_interval.x + p_interval.y <= p_dir_index || p_interval.y < 1) {
		return { -1, -1 };
	}
	int first_name_idx = p_dir_index + 1;
	int start = MAX(p_interval.x, first_name_idx);
	return { start - first_name_idx, p_interval.y - start + p_interval.x };
}

Rect2i get_highlight_region(Ref<Font> &p_font, const int p_font_size, const String &p_string, const Vector2i p_substr) {
	Vector2i prefix = p_font->get_string_size(p_string.substr(0, p_substr.x), HORIZONTAL_ALIGNMENT_LEFT, -1, p_font_size);
	prefix.y = 0;
	Vector2i size = p_font->get_string_size(p_string.substr(p_substr.x, p_substr.y), HORIZONTAL_ALIGNMENT_LEFT, -1, p_font_size);
	return { prefix, size };
}

QuickOpenResultListItem::QuickOpenResultListItem() {
	set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_theme_constant_override("separation", 4 * EDSCALE);

	{
		image_container = memnew(MarginContainer);
		image_container->add_theme_constant_override("margin_top", 2 * EDSCALE);
		image_container->add_theme_constant_override("margin_bottom", 2 * EDSCALE);
		image_container->add_theme_constant_override("margin_left", CONTAINER_MARGIN * EDSCALE);
		image_container->add_theme_constant_override("margin_right", 0);
		add_child(image_container);

		thumbnail = memnew(TextureRect);
		thumbnail->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
		thumbnail->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
		thumbnail->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
		thumbnail->set_stretch_mode(TextureRect::StretchMode::STRETCH_SCALE);
		image_container->add_child(thumbnail);
	}

	{
		text_container = memnew(VBoxContainer);
		text_container->add_theme_constant_override("separation", -6 * EDSCALE);
		text_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		text_container->set_v_size_flags(Control::SIZE_FILL);
		add_child(text_container);

		name = memnew(Label);
		name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		name->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
		name->set_horizontal_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_LEFT);
		text_container->add_child(name);

		path = memnew(Label);
		path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		path->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
		path->add_theme_font_size_override(SceneStringName(font_size), 12 * EDSCALE);
		text_container->add_child(path);
	}
}

void QuickOpenResultListItem::set_content(const QuickOpenResultCandidate &p_candidate) {
	result = p_candidate.result;
	thumbnail->set_texture(p_candidate.thumbnail);
	name->set_text(p_candidate.file_name);
	path->set_text(p_candidate.file_directory);

	const int max_size = 32 * EDSCALE;
	bool uses_icon = p_candidate.thumbnail->get_width() < max_size;

	if (uses_icon) {
		thumbnail->set_custom_minimum_size(p_candidate.thumbnail->get_size());

		int margin_needed = (max_size - p_candidate.thumbnail->get_width()) / 2;
		image_container->add_theme_constant_override("margin_left", CONTAINER_MARGIN + margin_needed);
		image_container->add_theme_constant_override("margin_right", margin_needed);
	} else {
		thumbnail->set_custom_minimum_size(Size2i(max_size, max_size));
		image_container->add_theme_constant_override("margin_left", CONTAINER_MARGIN);
		image_container->add_theme_constant_override("margin_right", 0);
	}
}

Vector<Rect2i> QuickOpenResultListItem::get_search_highlights() {
	Vector<Rect2i> highlights;
	Ref<Font> font = get_theme_font(SceneStringName(font));

	if (result.is_null() || font.is_null()) {
		return highlights;
	}

	int path_font_size = path->get_theme_font_size(SceneStringName(font_size));
	int name_font_size = name->get_theme_font_size(SceneStringName(font_size));
	Vector2i path_position = path->get_screen_position() - get_screen_position();
	Vector2i name_position = name->get_screen_position() - get_screen_position();

	for (Ref<FuzzyTokenMatch> match : result->token_matches) {
		for (Vector2i interval : match->substrings) {
			Vector2i path_interval = get_path_interval(interval, result->dir_index);
			Vector2i name_interval = get_name_interval(interval, result->dir_index);
			if (path_interval.x != -1) {
				Rect2i path_highlight = get_highlight_region(font, path_font_size, path->get_text(), path_interval);
				path_highlight.position += path_position;
				highlights.append(path_highlight);
			}
			if (name_interval.x != -1) {
				Rect2i name_highlight = get_highlight_region(font, name_font_size, name->get_text(), name_interval);
				name_highlight.position += name_position;
				highlights.append(name_highlight);
			}
		}
	}
	return highlights;
}

void QuickOpenResultListItem::reset() {
	thumbnail->set_texture(nullptr);
	name->set_text("");
	path->set_text("");
	result = nullptr;
}

void QuickOpenResultListItem::highlight_item(const Color &p_color) {
	name->add_theme_color_override(SceneStringName(font_color), p_color);
}

void QuickOpenResultListItem::remove_highlight() {
	name->remove_theme_color_override(SceneStringName(font_color));
}

void QuickOpenResultListItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			path->add_theme_color_override(SceneStringName(font_color), get_theme_color("font_disabled_color", EditorStringName(Editor)));
		} break;
	}
}

//--------------- Grid Item

QuickOpenResultGridItem::QuickOpenResultGridItem() {
	set_h_size_flags(Control::SIZE_FILL);
	set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_theme_constant_override("separation", -2 * EDSCALE);

	thumbnail = memnew(TextureRect);
	thumbnail->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	thumbnail->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	thumbnail->set_custom_minimum_size(Size2i(120 * EDSCALE, 64 * EDSCALE));
	add_child(thumbnail);

	name = memnew(Label);
	name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	name->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	name->set_horizontal_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
	name->add_theme_font_size_override(SceneStringName(font_size), 13 * EDSCALE);
	add_child(name);
}

void QuickOpenResultGridItem::set_content(const QuickOpenResultCandidate &p_candidate) {
	result = p_candidate.result;
	thumbnail->set_texture(p_candidate.thumbnail);
	name->set_text(p_candidate.file_name);
	name->set_tooltip_text(p_candidate.file_name);

	bool uses_icon = p_candidate.thumbnail->get_width() < (32 * EDSCALE);

	if (uses_icon || p_candidate.thumbnail->get_height() <= thumbnail->get_custom_minimum_size().y) {
		thumbnail->set_expand_mode(TextureRect::EXPAND_KEEP_SIZE);
		thumbnail->set_stretch_mode(TextureRect::StretchMode::STRETCH_KEEP_CENTERED);
	} else {
		thumbnail->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
		thumbnail->set_stretch_mode(TextureRect::StretchMode::STRETCH_SCALE);
	}
}

Vector<Rect2i> QuickOpenResultGridItem::get_search_highlights() {
	Vector<Rect2i> highlights;
	Ref<Font> font = get_theme_font(SceneStringName(font));

	if (result.is_null() || font.is_null()) {
		return highlights;
	}

	int font_size = name->get_theme_font_size(SceneStringName(font_size));
	Rect2i name_rect = name->get_rect();
	// Rect and string offsets are to handle centered text and trailing ellipsis
	name_rect.size.x -= (MAGIC_HIGHLIGHT_OFFSET.x + 5) * EDSCALE;
	int name_width = font->get_string_size(name->get_text(), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x;
	// The lower bound was tested to work well with EDSCALE = 1 and EDSCALE = 2
	int str_offset = MAX(2 * EDSCALE - 1, (name_rect.size.x - name_width) / 2);

	if (!name_rect.has_area()) {
		return highlights;
	}

	for (Ref<FuzzyTokenMatch> match : result->token_matches) {
		for (Vector2i interval : match->substrings) {
			Vector2i name_interval = get_name_interval(interval, result->dir_index);
			if (name_interval.x != -1) {
				Rect2i name_highlight = get_highlight_region(font, font_size, name->get_text(), name_interval);
				name_highlight.position += name_rect.position;
				name_highlight.position.x += str_offset;
				highlights.append(name_rect.intersection(name_highlight));
			}
		}
	}
	return highlights;
}

void QuickOpenResultGridItem::reset() {
	name->set_text("");
	thumbnail->set_texture(nullptr);
	result = nullptr;
}

void QuickOpenResultGridItem::highlight_item(const Color &p_color) {
	name->add_theme_color_override(SceneStringName(font_color), p_color);
}

void QuickOpenResultGridItem::remove_highlight() {
	name->remove_theme_color_override(SceneStringName(font_color));
}
