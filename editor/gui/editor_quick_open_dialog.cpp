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

#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/string/fuzzy_search.h"
#include "editor/docks/filesystem_dock.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/file_system/editor_paths.h"
#include "editor/gui/editor_toaster.h"
#include "editor/inspector/editor_resource_preview.h"
#include "editor/inspector/multi_node_edit.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/center_container.h"
#include "scene/gui/check_button.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"

void HighlightedLabel::draw_substr_rects(const Vector2i &p_substr, Vector2 p_offset, int p_line_limit, int line_spacing) {
	for (int i = get_lines_skipped(); i < p_line_limit; i++) {
		RID line = get_line_rid(i);
		Vector<Vector2> ranges = TS->shaped_text_get_selection(line, p_substr.x, p_substr.x + p_substr.y);
		Rect2 line_rect = get_line_rect(i);
		for (const Vector2 &range : ranges) {
			Rect2 rect = Rect2(Point2(range.x, 0) + line_rect.position, Size2(range.y - range.x, line_rect.size.y));
			rect.position = p_offset + line_rect.position;
			rect.position.x += range.x;
			rect.size = Size2(range.y - range.x, line_rect.size.y);
			rect.size.x = MIN(rect.size.x, line_rect.size.x - range.x);
			if (rect.size.x > 0) {
				draw_rect(rect, Color(1, 1, 1, 0.07), true);
				draw_rect(rect, Color(0.5, 0.7, 1.0, 0.4), false, 1);
			}
		}
		p_offset.y += line_spacing + TS->shaped_text_get_ascent(line) + TS->shaped_text_get_descent(line);
	}
}

void HighlightedLabel::add_highlight(const Vector2i &p_interval) {
	if (p_interval.y > 0) {
		highlights.append(p_interval);
		queue_redraw();
	}
}

void HighlightedLabel::reset_highlights() {
	highlights.clear();
	queue_redraw();
}

void HighlightedLabel::_notification(int p_notification) {
	if (p_notification == NOTIFICATION_DRAW) {
		if (highlights.is_empty()) {
			return;
		}

		Vector2 offset;
		int line_limit;
		int line_spacing;
		get_layout_data(offset, line_limit, line_spacing);

		for (const Vector2i &substr : highlights) {
			draw_substr_rects(substr, offset, line_limit, line_spacing);
		}
	}
}

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
		search_box->set_accessibility_name(TTRC("Search"));
		search_box->set_clear_button_enabled(true);
		mc->add_child(search_box);
	}

	{
		container = memnew(QuickOpenResultContainer);
		container->connect("selection_changed", callable_mp(this, &EditorQuickOpenDialog::selection_changed));
		container->connect("result_clicked", callable_mp(this, &EditorQuickOpenDialog::item_pressed));
		vbc->add_child(container);
	}

	search_box->connect(SceneStringName(text_changed), callable_mp(this, &EditorQuickOpenDialog::_search_box_text_changed));
	search_box->connect(SceneStringName(gui_input), callable_mp(container, &QuickOpenResultContainer::handle_search_box_input));
	register_text_enter(search_box);
	get_ok_button()->hide();
}

String EditorQuickOpenDialog::get_dialog_title() const {
	Vector<StringName> allowed_types = container->get_base_types();
	if (allowed_types.size() > 1) {
		String joined_types;
		for (int i = 0; i < allowed_types.size(); i++) {
			joined_types += String(allowed_types[i]) + (i < allowed_types.size() - 1 ? ", " : "");
		}
		return vformat(TTR("Select %s"), joined_types);
	}

	if (allowed_types[0] == SNAME("PackedScene")) {
		return TTR("Select Scene");
	}

	return vformat(TTR("Select %s"), allowed_types[0]);
}

void EditorQuickOpenDialog::popup_dialog(const Vector<StringName> &p_base_types, const Callable &p_item_selected_callback, bool p_allow_type_switching) {
	ERR_FAIL_COND(p_base_types.is_empty());
	ERR_FAIL_COND(!p_item_selected_callback.is_valid());

	property_object = nullptr;
	property_path = "";
	item_selected_callback = p_item_selected_callback;
	allow_type_switching = p_allow_type_switching;

	is_cycling_items = false;
	container->init(p_base_types);
	container->set_instant_preview_toggle_visible(false);
	_finish_dialog_setup(p_base_types);
}

void EditorQuickOpenDialog::popup_dialog_for_property(const Vector<StringName> &p_base_types, Object *p_obj, const StringName &p_path, const Callable &p_item_selected_callback) {
	ERR_FAIL_NULL(p_obj);
	ERR_FAIL_COND(p_base_types.is_empty());
	ERR_FAIL_COND(!p_item_selected_callback.is_valid());

	property_object = p_obj;
	property_path = p_path;
	item_selected_callback = p_item_selected_callback;
	initial_property_value = property_object->get(property_path);
	allow_type_switching = false;

	if (!initial_property_value.is_null()) {
		container->set_assigned_resource_id(EditorFileSystem::get_singleton()->get_file_uid(initial_property_value.call("get_path")));
	}
	container->init(p_base_types);
	container->set_instant_preview_toggle_visible(true);
	_finish_dialog_setup(p_base_types);
}

void EditorQuickOpenDialog::_finish_dialog_setup(const Vector<StringName> &p_base_types) {
	set_process_shortcut_input(allow_type_switching);
	get_ok_button()->set_disabled(container->has_nothing_selected());
	set_title(get_dialog_title());
	popup_centered_clamped(Size2(780, 650) * EDSCALE, 0.8f);
	search_box->grab_focus();
}

void EditorQuickOpenDialog::ok_pressed() {
	container->save_selected_item();

	update_property();
	container->cleanup();
	search_box->clear();
	hide();
	// Don't clear when toggling include addons
	container->set_assigned_resource_id(ResourceUID::INVALID_ID);
}

bool EditorQuickOpenDialog::_is_instant_preview_active() const {
	return property_object != nullptr && container->is_instant_preview_enabled();
}

void EditorQuickOpenDialog::selection_changed() const {
	if (!_is_instant_preview_active()) {
		return;
	}

	preview_property();
}

void EditorQuickOpenDialog::item_pressed(bool p_double_click) {
	// A double-click should always be taken as a "confirm" action.
	if (p_double_click) {
		ok_pressed();
		return;
	}

	// Single-clicks should be taken as a "confirm" action only if Instant Preview
	// isn't currently enabled, or the property object is null for some reason.
	if (!_is_instant_preview_active()) {
		ok_pressed();
	}
}

void EditorQuickOpenDialog::preview_property() const {
	Variant preview_resource;
	if (!container->has_nothing_selected()) {
		String path = container->get_selected_path();

		preview_resource = ResourceLoader::load(path);
		ERR_FAIL_COND_MSG(preview_resource.is_null(), "Cannot load resource from path '" + path + "'.");

		Resource *res = Object::cast_to<Resource>(property_object);
		if (res) {
			HashSet<Resource *> resources_found;
			resources_found.insert(res);
			if (EditorNode::find_recursive_resources(preview_resource, resources_found)) {
				EditorToaster::get_singleton()->popup_str(TTR("Recursion detected, Instant Preview failed."), EditorToaster::SEVERITY_ERROR);
				preview_resource = Ref<Resource>();
			}
		}
	} else {
		// Reset preview resource to initial value.
		preview_resource = initial_property_value;
	}

	// MultiNodeEdit has adding to the undo/redo stack baked into its set function.
	// As such, we have to specifically call a version of its setter that doesn't
	// create undo/redo actions.
	property_object->set_block_signals(true);
	if (Object::cast_to<MultiNodeEdit>(property_object)) {
		Object::cast_to<MultiNodeEdit>(property_object)->_set_impl(property_path, preview_resource, "", false);
	} else {
		property_object->set(property_path, preview_resource);
	}
	property_object->set_block_signals(false);
}

void EditorQuickOpenDialog::update_property() const {
	// Set the property back to the initial value first, so that the undo action
	// has the correct object.
	if (property_object) {
		if (Object::cast_to<MultiNodeEdit>(property_object)) {
			Object::cast_to<MultiNodeEdit>(property_object)->_set_impl(property_path, initial_property_value, "", false);
		} else {
			property_object->set(property_path, initial_property_value);
		}
	}

	if (!item_selected_callback.is_valid()) {
		String err_msg = "The callback provided to the Quick Open dialog was invalid.";
		if (_is_instant_preview_active()) {
			err_msg += " Try disabling \"Instant Preview\" as a workaround.";
		}
		ERR_FAIL_MSG(err_msg);
	}

	item_selected_callback.call(container->get_selected_path());
}

void EditorQuickOpenDialog::cancel_pressed() {
	if (property_object) {
		if (Object::cast_to<MultiNodeEdit>(property_object)) {
			Object::cast_to<MultiNodeEdit>(property_object)->_set_impl(property_path, initial_property_value, "", false);
		} else {
			property_object->set(property_path, initial_property_value);
		}
	}
	container->cleanup();
	search_box->clear();
	container->set_assigned_resource_id(ResourceUID::INVALID_ID);
}

void EditorQuickOpenDialog::shortcut_input(const Ref<InputEvent> &p_event) {
	// If the user is cycling through items (with up/down arrows), confirm selection when releasing the keys.
	Ref<InputEventWithModifiers> iewm = p_event;
	if (is_cycling_items && iewm.is_valid() && p_event->is_released() && iewm->get_modifiers_mask().is_empty()) {
		ok_pressed();
		return;
	}

	if (p_event.is_null() || !p_event->is_pressed() || p_event->is_echo()) {
		return;
	}

	Vector<StringName> new_base_types;
	if (EditorSettings *settings = EditorSettings::get_singleton()) {
		if (settings->is_shortcut("editor/quick_open", p_event)) {
			new_base_types.push_back("Resource");
		} else if (settings->is_shortcut("editor/quick_open_scene", p_event)) {
			new_base_types.push_back("PackedScene");
		} else if (settings->is_shortcut("editor/quick_open_script", p_event)) {
			new_base_types.push_back("Script");
		}
	}

	if (new_base_types.size() != 1) {
		return;
	}

	// Check if we're already showing this dialog type.
	const Vector<StringName> &current_base_types = container->get_base_types();
	if (current_base_types.size() == 1 && current_base_types[0] == new_base_types[0]) {
		// Already showing the requested dialog type, move next.
		Ref<InputEventKey> down_event = memnew(InputEventKey);
		down_event->set_keycode(Key::DOWN);
		down_event->set_pressed(true);
		container->handle_search_box_input(down_event);
		is_cycling_items = true;
	} else {
		// Switch to the new dialog type.
		container->init(new_base_types);
		container->set_instant_preview_toggle_visible(false);
		is_cycling_items = false;
		set_title(get_dialog_title());
		search_box->clear();
		search_box->grab_focus();
	}

	set_input_as_handled();
}

void EditorQuickOpenDialog::_search_box_text_changed(const String &p_query) {
	container->set_query_and_update(p_query);
	get_ok_button()->set_disabled(container->has_nothing_selected());
}

//------------------------- Result Container

void style_button(Button *p_button) {
	p_button->set_flat(true);
	p_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
}

QuickOpenResultContainer::QuickOpenResultContainer() {
	set_h_size_flags(Control::SIZE_EXPAND_FILL);
	set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_theme_constant_override("separation", 0);
	history_file.instantiate();

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
			no_results_label->set_focus_mode(FOCUS_ACCESSIBILITY);
			no_results_label->add_theme_font_size_override(SceneStringName(font_size), 24 * EDSCALE);
			no_results_container->add_child(no_results_label);
			no_results_container->hide();
		}

		{
			MarginContainer *mc = memnew(MarginContainer);
			mc->set_theme_type_variation("NoBorderHorizontalWindow");
			mc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			panel_container->add_child(mc);

			// Search results
			scroll_container = memnew(ScrollContainer);
			scroll_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
			scroll_container->set_scroll_hint_mode(ScrollContainer::SCROLL_HINT_MODE_ALL);
			scroll_container->hide();
			panel_container->add_child(scroll_container);

			list = memnew(VBoxContainer);
			list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			list->add_theme_constant_override(SNAME("separation"), 0);
			list->hide();
			scroll_container->add_child(list);

			grid = memnew(HFlowContainer);
			grid->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			grid->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			grid->add_theme_constant_override(SNAME("v_separation"), 0);
			grid->add_theme_constant_override(SNAME("h_separation"), 0);
			grid->hide();
			scroll_container->add_child(grid);

			file_context_menu = memnew(PopupMenu);
			file_context_menu->add_item(TTR("Show in FileSystem"), FILE_SHOW_IN_FILESYSTEM);
			file_context_menu->add_item(OS::get_singleton()->get_platform_string(OS::PLATFORM_STRING_FILE_MANAGER_SHOW), FILE_SHOW_IN_FILE_MANAGER);
			file_context_menu->connect(SceneStringName(id_pressed), callable_mp(this, &QuickOpenResultContainer::_menu_option));
			file_context_menu->hide();
			scroll_container->add_child(file_context_menu);
		}
	}

	{
		// Selected filepath
		file_details_path = memnew(Label);
		file_details_path->set_focus_mode(FOCUS_ACCESSIBILITY);
		file_details_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		file_details_path->set_horizontal_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
		file_details_path->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
		add_child(file_details_path);
	}

	{
		// Bottom bar
		HBoxContainer *bottom_bar = memnew(HBoxContainer);
		bottom_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		bottom_bar->set_alignment(ALIGNMENT_END);
		bottom_bar->add_theme_constant_override("separation", 3);
		add_child(bottom_bar);

		instant_preview_toggle = memnew(CheckButton);
		style_button(instant_preview_toggle);
		instant_preview_toggle->set_text(TTRC("Instant Preview"));
		instant_preview_toggle->set_tooltip_text(TTRC("Selected resource will be previewed in the editor before accepting."));
		instant_preview_toggle->connect(SceneStringName(toggled), callable_mp(this, &QuickOpenResultContainer::_toggle_instant_preview));
		bottom_bar->add_child(instant_preview_toggle);

		fuzzy_search_toggle = memnew(CheckButton);
		style_button(fuzzy_search_toggle);
		fuzzy_search_toggle->set_text(TTR("Fuzzy Search"));
		fuzzy_search_toggle->set_tooltip_text(TTRC("Include approximate matches."));
		fuzzy_search_toggle->connect(SceneStringName(toggled), callable_mp(this, &QuickOpenResultContainer::_toggle_fuzzy_search));
		bottom_bar->add_child(fuzzy_search_toggle);

		include_addons_toggle = memnew(CheckButton);
		style_button(include_addons_toggle);
		include_addons_toggle->set_text(TTR("Addons"));
		include_addons_toggle->set_tooltip_text(TTR("Include files from addons"));
		include_addons_toggle->connect(SceneStringName(toggled), callable_mp(this, &QuickOpenResultContainer::_toggle_include_addons));
		bottom_bar->add_child(include_addons_toggle);

		VSeparator *vsep = memnew(VSeparator);
		vsep->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
		vsep->set_custom_minimum_size(Size2i(0, 14 * EDSCALE));
		bottom_bar->add_child(vsep);

		display_mode_toggle = memnew(Button);
		display_mode_toggle->set_accessibility_name(TTRC("Display Mode"));
		style_button(display_mode_toggle);
		display_mode_toggle->connect(SceneStringName(pressed), callable_mp(this, &QuickOpenResultContainer::_toggle_display_mode));
		bottom_bar->add_child(display_mode_toggle);
	}
}

void QuickOpenResultContainer::_menu_option(int p_option) {
	ERR_FAIL_COND(get_selected() == ResourceUID::INVALID_ID);
	String selected_path = get_selected_path();

	switch (p_option) {
		case FILE_SHOW_IN_FILESYSTEM: {
			FileSystemDock::get_singleton()->navigate_to_path(selected_path);
		} break;
		case FILE_SHOW_IN_FILE_MANAGER: {
			String dir = ProjectSettings::get_singleton()->globalize_path(selected_path);
			OS::get_singleton()->shell_show_in_file_manager(dir, true);
		} break;
	}
}

void QuickOpenResultContainer::_ensure_result_vector_capacity() {
	const int target_size = EDITOR_GET("filesystem/quick_open_dialog/max_results");
	const int initial_size = result_items.size();
	for (int i = target_size; i < initial_size; i++) {
		result_items[i]->queue_free();
	}
	result_items.resize(target_size);
	for (int i = initial_size; i < target_size; i++) {
		QuickOpenResultItem *item = memnew(QuickOpenResultItem);
		item->connect(SceneStringName(gui_input), callable_mp(this, &QuickOpenResultContainer::_item_input).bind(i));
		result_items.write[i] = item;
		if (!never_opened) {
			_layout_result_item(item);
		}
	}
}

void QuickOpenResultContainer::_resolve_actual_type(const Vector<StringName> &p_base_types) {
	Vector<StringName> allowed_types;
	for (const StringName &type : p_base_types) {
		if (ClassDB::is_abstract(type)) {
			List<StringName> inheriters;
			ClassDB::get_direct_inheriters_from_class(type, &inheriters);
			for (const StringName &inheritor : inheriters) {
				allowed_types.push_back(inheritor);
			}
		} else {
			allowed_types.push_back(type);
		}
	}
	actual_types = allowed_types;
}

void QuickOpenResultContainer::init(const Vector<StringName> &p_base_types) {
	_ensure_result_vector_capacity();
	if (p_base_types.size() > 1) {
		// Material
		_resolve_actual_type(p_base_types);
	} else {
		actual_types = p_base_types;
	}

	const int display_mode_behavior = EDITOR_GET("filesystem/quick_open_dialog/default_display_mode");
	const bool adaptive_display_mode = (display_mode_behavior == 0);
	const bool first_open = never_opened;

	if (adaptive_display_mode) {
		_set_display_mode(get_adaptive_display_mode(actual_types));
	} else if (never_opened) {
		int last = EditorSettings::get_singleton()->get_project_metadata("quick_open_dialog", "last_mode", QuickOpenDisplayMode::LIST);
		_set_display_mode(static_cast<QuickOpenDisplayMode>(last));
	}

	const bool do_instant_preview = EDITOR_GET("filesystem/quick_open_dialog/instant_preview");
	const bool fuzzy_matching = EDITOR_GET("filesystem/quick_open_dialog/enable_fuzzy_matching");
	const bool include_addons = EDITOR_GET("filesystem/quick_open_dialog/include_addons");
	instant_preview_toggle->set_pressed_no_signal(do_instant_preview);
	fuzzy_search_toggle->set_pressed_no_signal(fuzzy_matching);
	include_addons_toggle->set_pressed_no_signal(include_addons);
	never_opened = false;

	const bool enable_highlights = EDITOR_GET("filesystem/quick_open_dialog/show_search_highlight");
	for (QuickOpenResultItem *E : result_items) {
		E->enable_highlights = enable_highlights;
	}

	bool history_modified = false;

	if (first_open && history_file->load(_get_cache_file_path()) == OK) {
		// Load history when opening for the first time.
		Vector<String> history_keys = history_file->get_section_keys("selected_history");
		for (const String &type : history_keys) {
			const PackedStringArray history_uids = history_file->get_value("selected_history", type);

			PackedStringArray cleaned_text_uids;
			cleaned_text_uids.resize(history_uids.size());

			Vector<ResourceUID::ID> cleaned_ids;
			cleaned_ids.resize(history_uids.size());

			{
				String *text_write = cleaned_text_uids.ptrw();
				ResourceUID::ID *id_write = cleaned_ids.ptrw();
				int i = 0;
				for (String uid : history_uids) {
#ifndef DISABLE_DEPRECATED
					if (!uid.begins_with("uid://")) {
						// uid might be a path here, if config was written by older editor version
						ResourceUID::ID id = EditorFileSystem::get_singleton()->get_file_uid(uid);
						if (id == ResourceUID::INVALID_ID) {
							continue;
						}
						uid = ResourceUID::get_singleton()->id_to_text(id);
					}
#endif

					ResourceUID::ID id = ResourceUID::get_singleton()->text_to_id(uid);
					if (id == ResourceUID::INVALID_ID || !ResourceUID::get_singleton()->has_id(id)) {
						continue;
					}

					text_write[i] = uid;
					id_write[i] = id;
					i++;
				}

				cleaned_text_uids.resize(i);
				selected_history.insert(type, cleaned_ids);

				if (i < history_uids.size()) {
					// Some paths removed, need to update history.
					if (i == 0) {
						history_file->erase_section_key("selected_history", type);
					} else {
						history_file->set_value("selected_history", type, cleaned_text_uids);
					}
					history_modified = true;
				}
			}
		}
	} else if (!first_open) {
		for (const StringName &type : actual_types) {
			Vector<ResourceUID::ID> *history = selected_history.getptr(type);
			if (history) {
				Vector<ResourceUID::ID> clean_history;
				clean_history.reserve(history->size());
				for (const ResourceUID::ID &id : *history) {
					if (ResourceUID::get_singleton()->has_id(id)) {
						clean_history.push_back(id);
					} else {
						history_modified = true;
					}
				}

				if (clean_history.is_empty()) {
					selected_history.erase(type);
				} else if (history_modified) {
					*history = clean_history;
				}
			}
		}
	}

	if (history_modified) {
		history_file->save(_get_cache_file_path());
	}

	_create_initial_results();
}

void QuickOpenResultContainer::_sort_ids(int p_max_results) {
	struct FilepathComparator {
		bool operator()(const ResourceUID::ID &p_lhs, const ResourceUID::ID &p_rhs) const {
			String lhs_path = ResourceUID::get_singleton()->get_id_path(p_lhs);
			String rhs_path = ResourceUID::get_singleton()->get_id_path(p_rhs);

			// Sort on (length, alphanumeric) to prioritize shorter filepaths
			return lhs_path.length() == rhs_path.length() ? lhs_path < rhs_path : lhs_path.length() < rhs_path.length();
		}
	};

	SortArray<ResourceUID::ID, FilepathComparator> sorter{};

	if (static_cast<int>(ids.size()) > p_max_results) {
		sorter.partial_sort(0, ids.size(), p_max_results, ids.ptr());
	} else {
		sorter.sort(ids.ptr(), ids.size());
	}
}

void QuickOpenResultContainer::_create_initial_results() {
	file_type_icons.clear();
	file_type_icons.insert(SNAME("__default_icon"), get_editor_theme_icon(SNAME("Object")));
	ids.clear();
	filetypes.clear();

	_update_history();

	_find_ids_in_folder(EditorFileSystem::get_singleton()->get_filesystem(), include_addons_toggle->is_pressed());
	_sort_ids(result_items.size());
	max_total_results = MIN(ids.size(), result_items.size());
	update_results();
}

void QuickOpenResultContainer::_find_ids_in_folder(EditorFileSystemDirectory *p_directory, bool p_include_addons) {
	for (int i = 0; i < p_directory->get_subdir_count(); i++) {
		if (p_include_addons || p_directory->get_name() != "addons") {
			_find_ids_in_folder(p_directory->get_subdir(i), p_include_addons);
		}
	}

	for (int i = 0; i < p_directory->get_file_count(); i++) {
		ResourceUID::ID id = p_directory->get_file_uid(i);
		if (id == ResourceUID::INVALID_ID) {
			continue;
		}

		const StringName engine_type = p_directory->get_file_type(i);
		const StringName script_type = p_directory->get_file_resource_script_class(i);

		const bool is_engine_type = script_type == StringName();
		const StringName &actual_type = is_engine_type ? engine_type : script_type;

		for (const StringName &parent_type : actual_types) {
			bool is_valid = ClassDB::is_parent_class(engine_type, parent_type) || (!is_engine_type && EditorNode::get_editor_data().script_class_is_parent(script_type, parent_type));

			if (is_valid) {
				ids.push_back(id);
				filetypes.insert(id, actual_type);
				break; // Stop testing base types as soon as we get a match.
			}
		}
	}
}

void QuickOpenResultContainer::set_assigned_resource_id(ResourceUID::ID p_id) {
	assigned_resource_id = p_id;
}

void QuickOpenResultContainer::set_query_and_update(const String &p_query) {
	query = p_query;
	update_results();
}

void QuickOpenResultContainer::_update_history() {
	const bool include_addons = include_addons_toggle->is_pressed();
	for (const StringName &type : actual_types) {
		Vector<ResourceUID::ID> *history = selected_history.getptr(type);
		if (!history) {
			continue;
		}

		visible_history.reserve(visible_history.size() + static_cast<int>(history->size()));
		for (const ResourceUID::ID &id : *history) {
			if (!include_addons && ResourceUID::get_singleton()->get_id_path(id).begins_with("res://addons/")) {
				continue;
			}
			visible_history.push_back(id);
			history_set.insert(id);
		}
	}
}

QuickOpenResultCandidate QuickOpenResultCandidate::from_id(const ResourceUID::ID &p_id, bool &r_success) {
	if (p_id == ResourceUID::INVALID_ID || !ResourceUID::get_singleton()->has_id(p_id)) {
		r_success = false;
		return {};
	}

	QuickOpenResultCandidate candidate;
	candidate.id = p_id;
	candidate.result = nullptr;
	r_success = true;
	return candidate;
}

QuickOpenResultCandidate QuickOpenResultCandidate::from_result(const FuzzySearchResult &p_result, bool &r_success) {
	const ResourceUID::ID id = EditorFileSystem::get_singleton()->get_file_uid(p_result.target);

	QuickOpenResultCandidate candidate = from_id(id, r_success);
	if (!r_success) {
		return candidate;
	}

	candidate.result = &p_result;
	return candidate;
}

void QuickOpenResultContainer::_add_candidate(QuickOpenResultCandidate &p_candidate) {
	ERR_FAIL_COND(!ResourceUID::get_singleton()->has_id(p_candidate.id));

	StringName actual_type;
	{
		StringName *actual_type_ptr = filetypes.getptr(p_candidate.id);
		if (actual_type_ptr) {
			actual_type = *actual_type_ptr;
		} else {
			ERR_PRINT(vformat("EditorQuickOpenDialog: No type for path %s.", ResourceUID::get_singleton()->get_id_path(p_candidate.id)));
		}
	}

	String file_path = ResourceUID::get_singleton()->get_id_path(p_candidate.id);
	EditorResourcePreview::PreviewItem item = EditorResourcePreview::get_singleton()->get_resource_preview_if_available(file_path);
	if (item.preview.is_valid()) {
		p_candidate.thumbnail = item.preview;
	} else if (file_type_icons.has(actual_type)) {
		p_candidate.thumbnail = *file_type_icons.getptr(actual_type);
	} else if (has_theme_icon(actual_type, EditorStringName(EditorIcons))) {
		p_candidate.thumbnail = get_editor_theme_icon(actual_type);
		file_type_icons.insert(actual_type, p_candidate.thumbnail);
	} else {
		p_candidate.thumbnail = *file_type_icons.getptr(SNAME("__default_icon"));
	}

	candidates.push_back(p_candidate);
	candidates_ids.insert(p_candidate.id);
}

void QuickOpenResultContainer::update_results() {
	candidates.clear();
	candidates_ids.clear();

	if (query.is_empty()) {
		_use_default_candidates();
	} else {
		_score_and_sort_candidates();
	}

	if (has_nothing_selected()) {
		reset_preview = true;
	}
	_update_result_items(MIN(candidates.size(), max_total_results));
}

void QuickOpenResultContainer::_use_default_candidates() {
	candidates.reserve(visible_history.size() + ids.size());
	const bool has_valid_resource_id = assigned_resource_id != ResourceUID::INVALID_ID;

	// When no resource is selected, show history first.
	if (!visible_history.is_empty() && !has_valid_resource_id) {
		for (const ResourceUID::ID &id : visible_history) {
			bool success;
			QuickOpenResultCandidate candidate = QuickOpenResultCandidate::from_id(id, success);
			if (!success) {
				continue;
			}
			_add_candidate(candidate);
		}
	}

	for (const ResourceUID::ID &id : ids) {
		if (candidates.size() >= max_total_results) {
			break;
		}
		if (candidates_ids.has(id)) {
			continue;
		}

		bool success;
		QuickOpenResultCandidate candidate = QuickOpenResultCandidate::from_id(id, success);
		if (!success) {
			continue;
		}

		if (has_valid_resource_id && id == assigned_resource_id) {
			selection_index = static_cast<int>(candidates.size());
		}
		_add_candidate(candidate);
	}
}

void QuickOpenResultContainer::_update_fuzzy_search_results() {
	FuzzySearch fuzzy_search;
	fuzzy_search.start_offset = 6; // Don't match against "res://" at the start of each filepath.
	fuzzy_search.set_query(query);
	fuzzy_search.max_results = max_total_results;
	bool fuzzy_matching = EDITOR_GET("filesystem/quick_open_dialog/enable_fuzzy_matching");
	int max_misses = EDITOR_GET("filesystem/quick_open_dialog/max_fuzzy_misses");
	fuzzy_search.allow_subsequences = fuzzy_matching;
	fuzzy_search.max_misses = fuzzy_matching ? max_misses : 0;

	PackedStringArray paths;
	paths.reserve_exact(ids.size());
	for (const ResourceUID::ID &id : ids) {
		paths.push_back(ResourceUID::get_singleton()->get_id_path(id));
	}

	fuzzy_search.search_all(paths, search_results);
}

void QuickOpenResultContainer::_score_and_sort_candidates() {
	_update_fuzzy_search_results();

	candidates.reserve(search_results.size());
	for (const FuzzySearchResult &result : search_results) {
		bool success;
		QuickOpenResultCandidate candidate = QuickOpenResultCandidate::from_result(result, success);
		if (!success) {
			continue;
		}

		_add_candidate(candidate);
	}
}

void QuickOpenResultContainer::_update_result_items(int p_new_visible_results_count) {
	// Only need to update items that were not hidden in previous update.
	int num_items_needing_updates = MAX(num_visible_results, p_new_visible_results_count);
	num_visible_results = p_new_visible_results_count;

	for (int i = 0; i < num_items_needing_updates; i++) {
		QuickOpenResultItem *item = result_items[i];

		if (i < num_visible_results) {
			item->set_content(candidates[i]);
		} else {
			item->reset();
		}
	}

	const bool any_results = num_visible_results > 0;
	_select_item(any_results ? selection_index : -1, true);

	scroll_container->set_visible(any_results);
	no_results_container->set_visible(!any_results);

	if (!any_results) {
		if (ids.is_empty()) {
			no_results_label->set_text(TTR("No files found for this type"));
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
					// Maybe strip off the shift modifier to allow non-selecting navigation by character?
					if (key_event->get_modifiers_mask().is_empty()) {
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
	// Don't move selection if there are no results.
	if (num_visible_results <= 0) {
		return;
	}
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

void QuickOpenResultContainer::_select_item(int p_index, bool p_center_on_scroll) {
	if (!has_nothing_selected()) {
		result_items[selection_index]->highlight_item(false);
	}

	selection_index = p_index;

	if (has_nothing_selected()) {
		if (reset_preview) {
			emit_signal(SNAME("selection_changed"));
			reset_preview = false;
		}
		file_details_path->set_text("");
		return;
	}

	QuickOpenResultItem *selection_item = result_items[selection_index];
	selection_item->highlight_item(true);
	bool in_history = history_set.has(candidates[selection_index].id);
	file_details_path->set_text(get_selected_path() + (in_history ? TTR(" (recently opened)") : ""));

	emit_signal(SNAME("selection_changed"));

	if (p_center_on_scroll) {
		// Wait for layout pending finished before scrolling.
		selection_item->call_on_all_layout_pending_finished(callable_mp(this, &QuickOpenResultContainer::_scroll_to_center).bind(selection_item));
	} else {
		// Copied from Tree.
		const int selection_position = selection_item->get_position().y;
		const int selection_size = selection_item->get_size().y;
		const int scroll_window_size = scroll_container->get_size().y;
		const int scroll_position = scroll_container->get_v_scroll();

		if (selection_position <= scroll_position) {
			scroll_container->set_v_scroll(selection_position);
		} else if (selection_position + selection_size > scroll_position + scroll_window_size) {
			scroll_container->set_v_scroll(selection_position + selection_size - scroll_window_size);
		}
	}
}

void QuickOpenResultContainer::_scroll_to_center(QuickOpenResultItem *p_item) const {
	const int selection_position = p_item->get_position().y;
	const int selection_size = p_item->get_size().y;
	const int scroll_window_size = scroll_container->get_size().y;

	const int centered_scroll = selection_position + selection_size / 2 - scroll_window_size / 2;
	scroll_container->set_v_scroll(centered_scroll);
}

void QuickOpenResultContainer::_item_input(const Ref<InputEvent> &p_ev, int p_index) {
	Ref<InputEventMouseButton> mb = p_ev;

	if (mb.is_valid() && mb->is_pressed()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			_select_item(p_index);
			emit_signal(SNAME("result_clicked"), mb->is_double_click());
		} else if (mb->get_button_index() == MouseButton::RIGHT) {
			_select_item(p_index);
			file_context_menu->set_position(result_items[p_index]->get_screen_position() + mb->get_position());
			file_context_menu->reset_size();
			file_context_menu->popup();
		}
	}
}

void QuickOpenResultContainer::_toggle_instant_preview(bool p_pressed) {
	EditorSettings::get_singleton()->set("filesystem/quick_open_dialog/instant_preview", p_pressed);
}

void QuickOpenResultContainer::_toggle_fuzzy_search(bool p_pressed) {
	EditorSettings::get_singleton()->set("filesystem/quick_open_dialog/enable_fuzzy_matching", p_pressed);
	update_results();
}

String QuickOpenResultContainer::_get_cache_file_path() const {
	return EditorPaths::get_singleton()->get_project_settings_dir().path_join("quick_open_dialog_cache.cfg");
}

void QuickOpenResultContainer::_toggle_include_addons(bool p_pressed) {
	EditorSettings::get_singleton()->set("filesystem/quick_open_dialog/include_addons", p_pressed);
	cleanup();
	_create_initial_results();
}

void QuickOpenResultContainer::_toggle_display_mode() {
	QuickOpenDisplayMode new_display_mode = (content_display_mode == QuickOpenDisplayMode::LIST) ? QuickOpenDisplayMode::GRID : QuickOpenDisplayMode::LIST;
	_set_display_mode(new_display_mode);
}

CanvasItem *QuickOpenResultContainer::_get_result_root() {
	if (content_display_mode == QuickOpenDisplayMode::LIST) {
		return list;
	} else {
		return grid;
	}
}

void QuickOpenResultContainer::_layout_result_item(QuickOpenResultItem *item) {
	item->set_display_mode(content_display_mode);
	Node *parent = item->get_parent();
	if (parent) {
		parent->remove_child(item);
	}
	_get_result_root()->add_child(item);
}

void QuickOpenResultContainer::_set_display_mode(QuickOpenDisplayMode p_display_mode) {
	CanvasItem *prev_root = _get_result_root();

	if (prev_root->is_visible() && content_display_mode == p_display_mode) {
		return;
	}

	content_display_mode = p_display_mode;
	CanvasItem *next_root = _get_result_root();

	EditorSettings::get_singleton()->set_project_metadata("quick_open_dialog", "last_mode", content_display_mode);

	prev_root->hide();
	next_root->show();

	for (QuickOpenResultItem *item : result_items) {
		_layout_result_item(item);
	}

	_update_result_items(num_visible_results);

	if (content_display_mode == QuickOpenDisplayMode::LIST) {
		display_mode_toggle->set_button_icon(get_editor_theme_icon(SNAME("FileThumbnail")));
		display_mode_toggle->set_tooltip_text(TTR("Grid view"));
	} else {
		display_mode_toggle->set_button_icon(get_editor_theme_icon(SNAME("FileList")));
		display_mode_toggle->set_tooltip_text(TTR("List view"));
	}
}

bool QuickOpenResultContainer::has_nothing_selected() const {
	return selection_index < 0;
}

ResourceUID::ID QuickOpenResultContainer::get_selected() const {
	ERR_FAIL_COND_V_MSG(has_nothing_selected(), ResourceUID::INVALID_ID, "Tried to get selected file, but nothing was selected.");
	return candidates[selection_index].id;
}

String QuickOpenResultContainer::get_selected_path() const {
	ERR_FAIL_COND_V_MSG(has_nothing_selected(), "", "Tried to get selected file path, but nothing was selected.");
	String path = ResourceUID::get_singleton()->get_id_path(candidates[selection_index].id);
	ERR_FAIL_COND_V_MSG(path.is_empty(), "", "Failed to get selected file path.");
	return path;
}

const Vector<StringName> &QuickOpenResultContainer::get_base_types() const {
	return actual_types;
}

QuickOpenDisplayMode QuickOpenResultContainer::get_adaptive_display_mode(const Vector<StringName> &p_base_types) {
	static const Vector<StringName> grid_preferred_types = {
		StringName("Font", true),
		StringName("Texture2D", true),
		StringName("Material", true),
		StringName("Mesh", true),
	};

	for (const StringName &type : grid_preferred_types) {
		for (const StringName &base_type : p_base_types) {
			if (base_type == type || ClassDB::is_parent_class(base_type, type)) {
				return QuickOpenDisplayMode::GRID;
			}
		}
	}

	return QuickOpenDisplayMode::LIST;
}

bool QuickOpenResultContainer::is_instant_preview_enabled() const {
	return instant_preview_toggle && instant_preview_toggle->is_visible() && instant_preview_toggle->is_pressed();
}

void QuickOpenResultContainer::set_instant_preview_toggle_visible(bool p_visible) const {
	instant_preview_toggle->set_visible(p_visible);
}

void QuickOpenResultContainer::save_selected_item() {
	const ResourceUID::ID selection_id = get_selected();
	const StringName &selection_type = filetypes.get(selection_id);
	Vector<ResourceUID::ID> *type_history = selected_history.getptr(selection_type);

	if (!type_history) {
		selected_history.insert(selection_type, Vector<ResourceUID::ID>());
		type_history = selected_history.getptr(selection_type);
	} else {
		for (int i = 0; i < type_history->size(); i++) {
			if (selection_id == type_history->get(i)) {
				type_history->remove_at(i);
				break;
			}
		}
	}

	type_history->insert(0, selection_id);
	if (type_history->size() > MAX_HISTORY_SIZE) {
		type_history->resize(MAX_HISTORY_SIZE);
	}

	PackedStringArray history_uids;
	history_uids.resize(type_history->size());
	{
		String *uids_write = history_uids.ptrw();

		int i = 0;
		for (const ResourceUID::ID &id : *type_history) {
			uids_write[i] = ResourceUID::get_singleton()->id_to_text(id);
			i++;
		}
	}
	history_file->set_value("selected_history", selection_type, history_uids);
	history_file->save(_get_cache_file_path());
}

void QuickOpenResultContainer::cleanup() {
	num_visible_results = 0;
	candidates.clear();
	history_set.clear();
	visible_history.clear();
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

			file_context_menu->set_item_icon(FILE_SHOW_IN_FILESYSTEM, get_editor_theme_icon(SNAME("ShowInFileSystem")));
			file_context_menu->set_item_icon(FILE_SHOW_IN_FILE_MANAGER, get_editor_theme_icon(SNAME("Filesystem")));

			panel_container->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SceneStringName(panel), SNAME("Tree")));

			if (content_display_mode == QuickOpenDisplayMode::LIST) {
				display_mode_toggle->set_button_icon(get_editor_theme_icon(SNAME("FileThumbnail")));
			} else {
				display_mode_toggle->set_button_icon(get_editor_theme_icon(SNAME("FileList")));
			}
		} break;
	}
}

void QuickOpenResultContainer::_bind_methods() {
	ADD_SIGNAL(MethodInfo("selection_changed"));
	ADD_SIGNAL(MethodInfo("result_clicked", PropertyInfo(Variant::BOOL, "double_click")));
}

//------------------------- Result Item

QuickOpenResultItem::QuickOpenResultItem() {
	set_focus_mode(FocusMode::FOCUS_NONE);
	_set_enabled(false);

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

	queue_redraw();
}

void QuickOpenResultItem::set_content(const QuickOpenResultCandidate &p_candidate) {
	_set_enabled(true);

	if (list_item->is_visible()) {
		list_item->set_content(p_candidate, enable_highlights);
	} else {
		grid_item->set_content(p_candidate, enable_highlights);
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
			hovering_stylebox = get_theme_stylebox(SNAME("hovered"), "Tree");
			highlighted_font_color = get_theme_color("font_focus_color", EditorStringName(Editor));
		} break;
		case NOTIFICATION_DRAW: {
			if (is_selected) {
				draw_style_box(selected_stylebox, Rect2(Point2(), get_size()));
			} else if (is_hovering) {
				draw_style_box(hovering_stylebox, Rect2(Point2(), get_size()));
			}
		} break;
	}
}

//----------------- List item

static Vector2i _get_path_interval(const Vector2i &p_interval, int p_dir_index) {
	if (p_interval.x >= p_dir_index || p_interval.y < 1) {
		return { -1, -1 };
	}
	return { p_interval.x, MIN(p_interval.x + p_interval.y, p_dir_index) - p_interval.x };
}

static Vector2i _get_name_interval(const Vector2i &p_interval, int p_dir_index) {
	if (p_interval.x + p_interval.y <= p_dir_index || p_interval.y < 1) {
		return { -1, -1 };
	}
	int first_name_idx = p_dir_index + 1;
	int start = MAX(p_interval.x, first_name_idx);
	return { start - first_name_idx, p_interval.y - start + p_interval.x };
}

QuickOpenResultListItem::QuickOpenResultListItem() {
	set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_theme_constant_override("margin_left", 6 * EDSCALE);
	add_theme_constant_override("margin_right", 6 * EDSCALE);

	hbc = memnew(HBoxContainer);
	hbc->add_theme_constant_override(SNAME("separation"), 4 * EDSCALE);
	add_child(hbc);

	const int max_size = 36 * EDSCALE;

	thumbnail = memnew(TextureRect);
	thumbnail->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	thumbnail->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	thumbnail->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	thumbnail->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	thumbnail->set_custom_minimum_size(Size2i(max_size, max_size));
	hbc->add_child(thumbnail);

	text_container = memnew(VBoxContainer);
	text_container->add_theme_constant_override(SNAME("separation"), -7 * EDSCALE);
	text_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	text_container->set_v_size_flags(Control::SIZE_FILL);
	hbc->add_child(text_container);

	name = memnew(HighlightedLabel);
	name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	name->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	name->set_horizontal_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_LEFT);
	text_container->add_child(name);

	path = memnew(HighlightedLabel);
	path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	path->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	path->add_theme_font_size_override(SceneStringName(font_size), 12 * EDSCALE);
	text_container->add_child(path);
}

void QuickOpenResultListItem::set_content(const QuickOpenResultCandidate &p_candidate, bool p_highlight) {
	thumbnail->set_texture(p_candidate.thumbnail);

	String file_path = ResourceUID::get_singleton()->get_id_path(p_candidate.id);
	name->set_text(file_path.get_file());
	path->set_text(file_path.get_base_dir());
	name->reset_highlights();
	path->reset_highlights();

	if (p_highlight && p_candidate.result != nullptr) {
		for (const FuzzyTokenMatch &match : p_candidate.result->token_matches) {
			for (const Vector2i &interval : match.substrings) {
				path->add_highlight(_get_path_interval(interval, p_candidate.result->dir_index));
				name->add_highlight(_get_name_interval(interval, p_candidate.result->dir_index));
			}
		}
	}
}

void QuickOpenResultListItem::reset() {
	thumbnail->set_texture(nullptr);
	name->set_text("");
	path->set_text("");
	name->reset_highlights();
	path->reset_highlights();
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
	set_custom_minimum_size(Size2i(120 * EDSCALE, 0));
	add_theme_constant_override("margin_top", 6 * EDSCALE);
	add_theme_constant_override("margin_left", 2 * EDSCALE);
	add_theme_constant_override("margin_right", 2 * EDSCALE);

	vbc = memnew(VBoxContainer);
	vbc->set_h_size_flags(Control::SIZE_FILL);
	vbc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_theme_constant_override(SNAME("separation"), 0);
	add_child(vbc);

	const int max_size = 64 * EDSCALE;

	thumbnail = memnew(TextureRect);
	thumbnail->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	thumbnail->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	thumbnail->set_custom_minimum_size(Size2i(max_size, max_size));
	vbc->add_child(thumbnail);

	name = memnew(HighlightedLabel);
	name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	name->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	name->set_horizontal_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
	name->add_theme_font_size_override(SceneStringName(font_size), 13 * EDSCALE);
	vbc->add_child(name);
}

void QuickOpenResultGridItem::set_content(const QuickOpenResultCandidate &p_candidate, bool p_highlight) {
	thumbnail->set_texture(p_candidate.thumbnail);

	String file_path = ResourceUID::get_singleton()->get_id_path(p_candidate.id);
	name->set_text(file_path.get_file());
	name->set_tooltip_text(file_path);
	name->reset_highlights();

	if (p_highlight && p_candidate.result != nullptr) {
		for (const FuzzyTokenMatch &match : p_candidate.result->token_matches) {
			for (const Vector2i &interval : match.substrings) {
				name->add_highlight(_get_name_interval(interval, p_candidate.result->dir_index));
			}
		}
	}

	bool uses_icon = p_candidate.thumbnail->get_width() < (32 * EDSCALE);

	if (uses_icon || p_candidate.thumbnail->get_height() <= thumbnail->get_custom_minimum_size().y) {
		thumbnail->set_expand_mode(TextureRect::EXPAND_KEEP_SIZE);
		thumbnail->set_stretch_mode(TextureRect::StretchMode::STRETCH_KEEP_CENTERED);
	} else {
		thumbnail->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
		thumbnail->set_stretch_mode(TextureRect::StretchMode::STRETCH_SCALE);
	}
}

void QuickOpenResultGridItem::reset() {
	thumbnail->set_texture(nullptr);
	name->set_text("");
	name->reset_highlights();
}

void QuickOpenResultGridItem::highlight_item(const Color &p_color) {
	name->add_theme_color_override(SceneStringName(font_color), p_color);
}

void QuickOpenResultGridItem::remove_highlight() {
	name->remove_theme_color_override(SceneStringName(font_color));
}
