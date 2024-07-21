/**
 * task_palette.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "task_palette.h"

#include "../util/limbo_compat.h"
#include "../util/limbo_string_names.h"
#include "../util/limbo_task_db.h"
#include "../util/limbo_utility.h"

#ifdef LIMBOAI_MODULE
#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "editor/editor_help.h"
#include "editor/editor_node.h"
#include "editor/editor_paths.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/check_box.h"
#endif // LIMBO_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/button_group.hpp>
#include <godot_cpp/classes/check_box.hpp>
#include <godot_cpp/classes/config_file.hpp>
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/editor_paths.hpp>
#include <godot_cpp/classes/font.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/h_box_container.hpp>
#include <godot_cpp/classes/h_flow_container.hpp>
#include <godot_cpp/classes/input_event_mouse_button.hpp>
#include <godot_cpp/classes/label.hpp>
#include <godot_cpp/classes/popup_menu.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/script_editor.hpp>
#include <godot_cpp/classes/script_editor_base.hpp>
#include <godot_cpp/classes/style_box.hpp>
#include <godot_cpp/core/error_macros.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

//**** TaskButton

void TaskButton::_bind_methods() {
}

Control *TaskButton::_do_make_tooltip() const {
#ifdef LIMBOAI_MODULE
	String help_symbol;
	bool is_resource = task_meta.begins_with("res://");

	if (is_resource) {
		help_symbol = "class|\"" + task_meta.lstrip("res://") + "\"|";
	} else {
		help_symbol = "class|" + task_meta + "|";
	}

	EditorHelpBit *help_bit = memnew(EditorHelpBit(help_symbol));
	help_bit->set_content_height_limits(1, 360 * EDSCALE);

	String desc = _module_get_help_description(task_meta);
	if (desc.is_empty() && is_resource) {
		// ! HACK: Force documentation parsing.
		Ref<Script> s = ResourceLoader::load(task_meta);
		if (s.is_valid()) {
			Vector<DocData::ClassDoc> docs = s->get_documentation();
			for (int i = 0; i < docs.size(); i++) {
				const DocData::ClassDoc &doc = docs.get(i);
				EditorHelp::get_doc_data()->add_doc(doc);
			}
			desc = _module_get_help_description(task_meta);
		}
	}
	if (desc.is_empty() && help_bit->get_description().is_empty()) {
		desc = "[i]" + TTR("No description.") + "[/i]";
	}
	if (!desc.is_empty()) {
		help_bit->set_description(desc);
	}

	EditorHelpBitTooltip::show_tooltip(help_bit, const_cast<TaskButton *>(this));
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
	// TODO: When we figure out how to retrieve documentation in GDEXTENSION, should add a tooltip control here.
#endif // LIMBOAI_GDEXTENSION

	return memnew(Control); // Make the standard tooltip invisible.
}

#ifdef LIMBOAI_MODULE

String TaskButton::_module_get_help_description(const String &p_class_or_script_path) const {
	String descr;

	DocTools *dd = EditorHelp::get_doc_data();
	HashMap<String, DocData::ClassDoc>::Iterator E;

	if (p_class_or_script_path.begins_with("res://")) {
		// Try to find by script path.
		E = dd->class_list.find(vformat("\"%s\"", p_class_or_script_path.trim_prefix("res://")));
		if (!E) {
			// Try to guess global script class from filename.
			String maybe_class_name = p_class_or_script_path.get_file().get_basename().to_pascal_case();
			E = dd->class_list.find(maybe_class_name);
		}
	} else {
		// Try to find core class or global class.
		E = dd->class_list.find(p_class_or_script_path);
	}

	if (E) {
		if (E->value.description.is_empty()) {
			descr = DTR(E->value.brief_description);
		} else {
			descr = DTR(E->value.description);
		}
	}

	// TODO: Documentation tooltips are only available in the module variant. Find a way to show em in GDExtension.

	return descr;
}

#endif // LIMBOAI_MODULE

TaskButton::TaskButton() {
	set_focus_mode(FOCUS_NONE);
}

//**** TaskButton ^

//**** TaskPaletteSection

void TaskPaletteSection::_on_task_button_pressed(const String &p_task) {
	emit_signal(LW_NAME(task_button_pressed), p_task);
}

void TaskPaletteSection::_on_task_button_gui_input(const Ref<InputEvent> &p_event, const String &p_task) {
	if (!p_event->is_pressed()) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == LW_MBTN(RIGHT)) {
		emit_signal(LW_NAME(task_button_rmb), p_task);
	}
}

void TaskPaletteSection::_on_header_pressed() {
	set_collapsed(!is_collapsed());
}

void TaskPaletteSection::set_filter(String p_filter_text) {
	int num_hidden = 0;
	if (p_filter_text.is_empty()) {
		for (int i = 0; i < tasks_container->get_child_count(); i++) {
			Object::cast_to<Button>(tasks_container->get_child(i))->show();
		}
		set_visible(tasks_container->get_child_count() > 0);
	} else {
		for (int i = 0; i < tasks_container->get_child_count(); i++) {
			Button *btn = Object::cast_to<Button>(tasks_container->get_child(i));
			btn->set_visible(btn->get_text().findn(p_filter_text) != -1);
			num_hidden += !btn->is_visible();
		}
		set_visible(num_hidden < tasks_container->get_child_count());
	}
}

void TaskPaletteSection::add_task_button(const String &p_name, const Ref<Texture> &icon, const String &p_meta) {
	TaskButton *btn = memnew(TaskButton);
	btn->set_text(p_name);
	BUTTON_SET_ICON(btn, icon);
	btn->set_tooltip_text("dummy_text"); // Force tooltip to be shown.
	btn->set_task_meta(p_meta);
	btn->add_theme_constant_override(LW_NAME(icon_max_width), 16 * EDSCALE); // Force user icons to  be of the proper size.
	btn->connect(LW_NAME(pressed), callable_mp(this, &TaskPaletteSection::_on_task_button_pressed).bind(p_meta));
	btn->connect(LW_NAME(gui_input), callable_mp(this, &TaskPaletteSection::_on_task_button_gui_input).bind(p_meta));
	tasks_container->add_child(btn);
}

void TaskPaletteSection::set_collapsed(bool p_collapsed) {
	tasks_container->set_visible(!p_collapsed);
	BUTTON_SET_ICON(section_header, (p_collapsed ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon));
}

bool TaskPaletteSection::is_collapsed() const {
	return !tasks_container->is_visible();
}

void TaskPaletteSection::_do_update_theme_item_cache() {
	theme_cache.arrow_down_icon = get_theme_icon(LW_NAME(GuiTreeArrowDown), LW_NAME(EditorIcons));
	theme_cache.arrow_right_icon = get_theme_icon(LW_NAME(GuiTreeArrowRight), LW_NAME(EditorIcons));
}

void TaskPaletteSection::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			section_header->connect(LW_NAME(pressed), callable_mp(this, &TaskPaletteSection::_on_header_pressed));
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_do_update_theme_item_cache();
			BUTTON_SET_ICON(section_header, (is_collapsed() ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon));
			section_header->add_theme_font_override(LW_NAME(font), get_theme_font(LW_NAME(bold), LW_NAME(EditorFonts)));
		} break;
	}
}

void TaskPaletteSection::_bind_methods() {
	ADD_SIGNAL(MethodInfo("task_button_pressed"));
	ADD_SIGNAL(MethodInfo("task_button_rmb"));
}

TaskPaletteSection::TaskPaletteSection() {
	section_header = memnew(Button);
	add_child(section_header);
	section_header->set_focus_mode(FOCUS_NONE);

	tasks_container = memnew(HFlowContainer);
	add_child(tasks_container);
}

TaskPaletteSection::~TaskPaletteSection() {
}

//**** TaskPaletteSection ^

//**** TaskPalette

void TaskPalette::_menu_action_selected(int p_id) {
	ERR_FAIL_COND(context_task.is_empty());
	switch (p_id) {
		case MENU_OPEN_DOC: {
			LimboUtility::get_singleton()->open_doc_class(context_task);
		} break;
		case MENU_EDIT_SCRIPT: {
			ERR_FAIL_COND(!context_task.begins_with("res://"));
			EDIT_SCRIPT(context_task);
		} break;
		case MENU_FAVORITE: {
			PackedStringArray favorite_tasks = GLOBAL_GET("limbo_ai/behavior_tree/favorite_tasks");
			if (favorite_tasks.has(context_task)) {
				int idx = favorite_tasks.find(context_task);
				if (idx >= 0) {
					favorite_tasks.remove_at(idx);
				}
			} else {
				favorite_tasks.append(context_task);
			}
			ProjectSettings::get_singleton()->set_setting("limbo_ai/behavior_tree/favorite_tasks", favorite_tasks);
			ProjectSettings::get_singleton()->save();
			emit_signal(LW_NAME(favorite_tasks_changed));
		} break;
	}
}

void TaskPalette::_on_task_button_pressed(const String &p_task) {
	emit_signal(LW_NAME(task_selected), p_task);
}

void TaskPalette::_on_task_button_rmb(const String &p_task) {
	if (dialog_mode) {
		return;
	}

	ERR_FAIL_COND(p_task.is_empty());

	context_task = p_task;
	menu->clear();

	menu->add_icon_item(theme_cache.edit_script_icon, TTR("Edit Script"), MENU_EDIT_SCRIPT);
	menu->set_item_disabled(MENU_EDIT_SCRIPT, !context_task.begins_with("res://"));
	menu->add_icon_item(theme_cache.open_doc_icon, TTR("Open Documentation"), MENU_OPEN_DOC);

	menu->add_separator();
	Array favorite_tasks = GLOBAL_GET("limbo_ai/behavior_tree/favorite_tasks");
	if (favorite_tasks.has(context_task)) {
		menu->add_icon_item(theme_cache.remove_from_favorites_icon, TTR("Remove from Favorites"), MENU_FAVORITE);
	} else {
		menu->add_icon_item(theme_cache.add_to_favorites_icon, TTR("Add to Favorites"), MENU_FAVORITE);
	}

	menu->reset_size();
	menu->set_position(get_screen_position() + get_local_mouse_position());
	menu->popup();
}

void TaskPalette::_apply_filter(const String &p_text) {
	for (int i = 0; i < sections->get_child_count(); i++) {
		TaskPaletteSection *sec = Object::cast_to<TaskPaletteSection>(sections->get_child(i));
		ERR_FAIL_NULL(sec);
		sec->set_filter(p_text);
	}
}

void TaskPalette::_update_filter_popup() {
	switch (filter_settings.type_filter) {
		case FilterSettings::TypeFilter::TYPE_ALL: {
			type_all->set_pressed(true);
		} break;
		case FilterSettings::TypeFilter::TYPE_CORE: {
			type_core->set_pressed(true);
		} break;
		case FilterSettings::TypeFilter::TYPE_USER: {
			type_user->set_pressed(true);
		} break;
	}

	switch (filter_settings.category_filter) {
		case FilterSettings::CategoryFilter::CATEGORY_ALL: {
			category_all->set_pressed(true);
		} break;
		case FilterSettings::CategoryFilter::CATEGORY_INCLUDE: {
			category_include->set_pressed(true);
		} break;
		case FilterSettings::CategoryFilter::CATEGORY_EXCLUDE: {
			category_exclude->set_pressed(true);
		} break;
	}

	while (category_list->get_child_count() > 0) {
		Node *item = category_list->get_child(0);
		category_list->remove_child(item);
		item->queue_free();
	}
	for (String &cat : LimboTaskDB::get_categories()) {
		CheckBox *category_item = memnew(CheckBox);
		category_item->set_text(cat);
		category_item->set_focus_mode(FocusMode::FOCUS_NONE);
		category_item->set_pressed_no_signal(LOGICAL_XOR(
				filter_settings.excluded_categories.has(cat),
				filter_settings.category_filter == FilterSettings::CategoryFilter::CATEGORY_INCLUDE));
		category_item->connect(LW_NAME(toggled), callable_mp(this, &TaskPalette::_category_item_toggled).bind(cat));
		category_list->add_child(category_item);
	}

	category_list->reset_size();
	Size2 size = category_list->get_size() + Size2(8, 8);
	size.width = MIN(size.width, 400 * EDSCALE);
	size.height = MIN(size.height, 600 * EDSCALE);
	category_scroll->set_custom_minimum_size(size);

	category_choice->set_visible(filter_settings.category_filter != FilterSettings::CATEGORY_ALL);
}

void TaskPalette::_show_filter_popup() {
	_update_filter_popup();

	tool_filters->set_pressed_no_signal(true);

	Transform2D xform = tool_filters->get_screen_transform();
	Rect2i rect = Rect2(xform.get_origin(), xform.get_scale() * tool_filters->get_size());

	rect.position.y += rect.size.height;
	rect.size.height = 0;
	filter_popup->reset_size();
	filter_popup->popup(rect);
}

void TaskPalette::_category_filter_changed() {
	if (category_all->is_pressed()) {
		filter_settings.category_filter = FilterSettings::CategoryFilter::CATEGORY_ALL;
	} else if (category_include->is_pressed()) {
		filter_settings.category_filter = FilterSettings::CategoryFilter::CATEGORY_INCLUDE;
	} else if (category_exclude->is_pressed()) {
		filter_settings.category_filter = FilterSettings::CategoryFilter::CATEGORY_EXCLUDE;
	}

	for (int i = 0; i < category_list->get_child_count(); i++) {
		CheckBox *item = Object::cast_to<CheckBox>(category_list->get_child(i));
		item->set_pressed_no_signal(LOGICAL_XOR(
				filter_settings.excluded_categories.has(item->get_text()),
				filter_settings.category_filter == FilterSettings::CATEGORY_INCLUDE));
	}

	category_choice->set_visible(filter_settings.category_filter != FilterSettings::CATEGORY_ALL);
	filter_popup->reset_size();
	_filter_data_changed();
}

void TaskPalette::_set_all_filter_categories(bool p_selected) {
	for (int i = 0; i < category_list->get_child_count(); i++) {
		CheckBox *item = Object::cast_to<CheckBox>(category_list->get_child(i));
		item->set_pressed_no_signal(p_selected);
		bool excluded = LOGICAL_XOR(p_selected, filter_settings.category_filter == FilterSettings::CATEGORY_INCLUDE);
		_set_category_excluded(item->get_text(), excluded);
	}
	_filter_data_changed();
}

void TaskPalette::_type_filter_changed() {
	if (type_all->is_pressed()) {
		filter_settings.type_filter = FilterSettings::TypeFilter::TYPE_ALL;
	} else if (type_core->is_pressed()) {
		filter_settings.type_filter = FilterSettings::TypeFilter::TYPE_CORE;
	} else if (type_user->is_pressed()) {
		filter_settings.type_filter = FilterSettings::TypeFilter::TYPE_USER;
	}
	_filter_data_changed();
}

void TaskPalette::_category_item_toggled(bool p_pressed, const String &p_category) {
	bool excluded = LOGICAL_XOR(p_pressed, filter_settings.category_filter == FilterSettings::CATEGORY_INCLUDE);
	_set_category_excluded(p_category, excluded);
	_filter_data_changed();
}

void TaskPalette::_filter_data_changed() {
	call_deferred(LW_NAME(refresh));
	_update_filter_button();
}

void TaskPalette::_draw_filter_popup_background() {
	theme_cache.category_choice_background->draw(category_choice->get_canvas_item(), Rect2(Point2(), category_choice->get_size()));
}

void TaskPalette::_update_filter_button() {
	tool_filters->set_pressed_no_signal(filter_popup->is_visible() ||
			filter_settings.type_filter != FilterSettings::TYPE_ALL ||
			(filter_settings.category_filter != FilterSettings::CATEGORY_ALL && !filter_settings.excluded_categories.is_empty()));
}

void TaskPalette::refresh() {
	HashSet<String> collapsed_sections;
	if (sections->get_child_count() == 0) {
		// Restore collapsed state from config.
		Ref<ConfigFile> cf;
		cf.instantiate();
		String conf_path = PROJECT_CONFIG_FILE();
		if (cf->load(conf_path) == OK) {
			Variant value = cf->get_value("task_palette", "collapsed_sections", Array());
			if (VARIANT_IS_ARRAY(value)) {
				Array arr = value;
				for (int i = 0; i < arr.size(); i++) {
					if (arr[i].get_type() == Variant::STRING) {
						collapsed_sections.insert(arr[i]);
					}
				}
			}
		}
	} else {
		for (int i = 0; i < sections->get_child_count(); i++) {
			TaskPaletteSection *sec = Object::cast_to<TaskPaletteSection>(sections->get_child(i));
			ERR_FAIL_NULL(sec);
			if (sec->is_collapsed()) {
				collapsed_sections.insert(sec->get_category_name());
			}
			sections->get_child(i)->queue_free();
		}
	}

	LimboTaskDB::scan_user_tasks();
	List<String> categories = LimboTaskDB::get_categories();
	categories.sort();

	for (String cat : categories) {
		if (filter_settings.category_filter != FilterSettings::CATEGORY_ALL && filter_settings.excluded_categories.has(cat)) {
			continue;
		}

		List<String> tasks = LimboTaskDB::get_tasks_in_category(cat);
		if (tasks.size() == 0) {
			continue;
		}

		TaskPaletteSection *sec = memnew(TaskPaletteSection());
		sec->set_category_name(cat);
		for (const String &task_meta : tasks) {
			Ref<Texture2D> icon = LimboUtility::get_singleton()->get_task_icon(task_meta);

			String tname;

			if (task_meta.begins_with("res:")) {
				if (filter_settings.type_filter == FilterSettings::TYPE_CORE) {
					continue;
				}
				tname = task_meta.get_file().get_basename().trim_prefix("BT").to_pascal_case();
			} else {
				if (filter_settings.type_filter == FilterSettings::TYPE_USER) {
					continue;
				}
				tname = task_meta.trim_prefix("BT");
			}

			sec->add_task_button(tname, icon, task_meta);
		}
		sec->set_filter("");
		sec->connect(LW_NAME(task_button_pressed), callable_mp(this, &TaskPalette::_on_task_button_pressed));
		sec->connect(LW_NAME(task_button_rmb), callable_mp(this, &TaskPalette::_on_task_button_rmb));
		sections->add_child(sec);
		sec->set_collapsed(!dialog_mode && collapsed_sections.has(cat));
	}

	if (!dialog_mode && !filter_edit->get_text().is_empty()) {
		_apply_filter(filter_edit->get_text());
	}
}

void TaskPalette::use_dialog_mode() {
	tool_filters->hide();
	tool_refresh->hide();
	dialog_mode = true;
}

void TaskPalette::_do_update_theme_item_cache() {
	theme_cache.add_to_favorites_icon = get_theme_icon(LW_NAME(Favorites), LW_NAME(EditorIcons));
	theme_cache.edit_script_icon = get_theme_icon(LW_NAME(Script), LW_NAME(EditorIcons));
	theme_cache.open_doc_icon = get_theme_icon(LW_NAME(Help), LW_NAME(EditorIcons));
	theme_cache.remove_from_favorites_icon = get_theme_icon(LW_NAME(NonFavorite), LW_NAME(EditorIcons));

	theme_cache.category_choice_background = get_theme_stylebox(LW_NAME(normal), LW_NAME(LineEdit));
}

void TaskPalette::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			// **** Signals
			tool_filters->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::_show_filter_popup));
			filter_edit->connect(LW_NAME(text_changed), callable_mp(this, &TaskPalette::_apply_filter));
			tool_refresh->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::refresh));
			menu->connect(LW_NAME(id_pressed), callable_mp(this, &TaskPalette::_menu_action_selected));
			type_all->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::_type_filter_changed));
			type_core->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::_type_filter_changed));
			type_user->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::_type_filter_changed));
			category_all->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::_category_filter_changed));
			category_include->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::_category_filter_changed));
			category_exclude->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::_category_filter_changed));
			category_choice->connect(LW_NAME(draw), callable_mp(this, &TaskPalette::_draw_filter_popup_background));
			select_all->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::_set_all_filter_categories).bind(true));
			deselect_all->connect(LW_NAME(pressed), callable_mp(this, &TaskPalette::_set_all_filter_categories).bind(false));
			filter_popup->connect(LW_NAME(popup_hide), callable_mp(this, &TaskPalette::_update_filter_button));
		} break;
		case NOTIFICATION_ENTER_TREE: {
			Ref<ConfigFile> cf;
			cf.instantiate();
			String conf_path = PROJECT_CONFIG_FILE();
			if (cf->load(conf_path) == OK) {
				Variant value = cf->get_value("task_palette", "type_filter", FilterSettings::TypeFilter(0));
				if (VARIANT_IS_NUM(value)) {
					filter_settings.type_filter = (FilterSettings::TypeFilter)(int)value;
				}

				value = cf->get_value("task_palette", "category_filter", FilterSettings::CategoryFilter(0));
				if (VARIANT_IS_NUM(value)) {
					filter_settings.category_filter = (FilterSettings::CategoryFilter)(int)value;
				}

				value = cf->get_value("task_palette", "excluded_categories", Array());
				if (VARIANT_IS_ARRAY(value)) {
					Array arr = value;
					for (int i = 0; i < arr.size(); i++) {
						if (arr[i].get_type() == Variant::STRING) {
							filter_settings.excluded_categories.insert(arr[i]);
						}
					}
				}
			}
			_update_filter_button();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			Ref<ConfigFile> cf;
			cf.instantiate();
			String conf_path = PROJECT_CONFIG_FILE();
			cf->load(conf_path);

			Array collapsed_sections;
			for (int i = 0; i < sections->get_child_count(); i++) {
				TaskPaletteSection *sec = Object::cast_to<TaskPaletteSection>(sections->get_child(i));
				if (sec->is_collapsed()) {
					collapsed_sections.push_back(sec->get_category_name());
				}
			}
			cf->set_value("task_palette", "collapsed_sections", collapsed_sections);

			cf->set_value("task_palette", "type_filter", filter_settings.type_filter);
			cf->set_value("task_palette", "category_filter", filter_settings.category_filter);

			Array excluded_categories;
			for (const String &cat : filter_settings.excluded_categories) {
				excluded_categories.append(cat);
			}
			cf->set_value("task_palette", "excluded_categories", excluded_categories);

			cf->save(conf_path);
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_do_update_theme_item_cache();

			BUTTON_SET_ICON(tool_filters, get_theme_icon(LW_NAME(AnimationFilter), LW_NAME(EditorIcons)));
			BUTTON_SET_ICON(tool_refresh, get_theme_icon(LW_NAME(Reload), LW_NAME(EditorIcons)));

			filter_edit->set_right_icon(get_theme_icon(LW_NAME(Search), LW_NAME(EditorIcons)));

			BUTTON_SET_ICON(select_all, LimboUtility::get_singleton()->get_task_icon("LimboSelectAll"));
			BUTTON_SET_ICON(deselect_all, LimboUtility::get_singleton()->get_task_icon("LimboDeselectAll"));

			category_choice->queue_redraw();

			if (is_visible_in_tree()) {
				refresh();
			}
		} break;
	}
}

void TaskPalette::_bind_methods() {
	ClassDB::bind_method(D_METHOD("refresh"), &TaskPalette::refresh);

	ADD_SIGNAL(MethodInfo("task_selected"));
	ADD_SIGNAL(MethodInfo("favorite_tasks_changed"));
}

TaskPalette::TaskPalette() {
	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	HBoxContainer *hb = memnew(HBoxContainer);
	vb->add_child(hb);

	tool_filters = memnew(Button);
	tool_filters->set_tooltip_text(TTR("Show filters"));
	tool_filters->set_flat(true);
	tool_filters->set_toggle_mode(true);
	tool_filters->set_focus_mode(FocusMode::FOCUS_NONE);
	hb->add_child(tool_filters);

	filter_edit = memnew(LineEdit);
	filter_edit->set_clear_button_enabled(true);
	filter_edit->set_placeholder(TTR("Filter tasks"));
	filter_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(filter_edit);

	tool_refresh = memnew(Button);
	tool_refresh->set_tooltip_text(TTR("Refresh tasks"));
	tool_refresh->set_flat(true);
	tool_refresh->set_focus_mode(FocusMode::FOCUS_NONE);
	hb->add_child(tool_refresh);

	ScrollContainer *sc = memnew(ScrollContainer);
	sc->set_h_size_flags(SIZE_EXPAND_FILL);
	sc->set_v_size_flags(SIZE_EXPAND_FILL);
	vb->add_child(sc);

	sections = memnew(VBoxContainer);
	sections->set_h_size_flags(SIZE_EXPAND_FILL);
	sections->set_v_size_flags(SIZE_EXPAND_FILL);
	sc->add_child(sections);

	menu = memnew(PopupMenu);
	add_child(menu);

	filter_popup = memnew(PopupPanel);
	{
		VBoxContainer *vbox = memnew(VBoxContainer);
		filter_popup->add_child(vbox);

		Label *type_header = memnew(Label);
		type_header->set_text(TTR("Type"));
		type_header->set_theme_type_variation("HeaderSmall");
		vbox->add_child(type_header);

		HBoxContainer *type_filter = memnew(HBoxContainer);
		vbox->add_child(type_filter);

		Ref<ButtonGroup> type_filter_group;
		type_filter_group.instantiate();

		type_all = memnew(Button);
		type_all->set_text(TTR("All"));
		type_all->set_tooltip_text(TTR("Show tasks of all types"));
		type_all->set_toggle_mode(true);
		type_all->set_focus_mode(FocusMode::FOCUS_NONE);
		type_all->set_button_group(type_filter_group);
		type_all->set_pressed(true);
		type_filter->add_child(type_all);

		type_core = memnew(Button);
		type_core->set_text(TTR("Core"));
		type_core->set_tooltip_text(TTR("Show only core tasks"));
		type_core->set_toggle_mode(true);
		type_core->set_focus_mode(FocusMode::FOCUS_NONE);
		type_core->set_button_group(type_filter_group);
		type_filter->add_child(type_core);

		type_user = memnew(Button);
		type_user->set_text(TTR("User"));
		type_user->set_tooltip_text(TTR("Show only user-implemented tasks (aka scripts)"));
		type_user->set_toggle_mode(true);
		type_user->set_focus_mode(FocusMode::FOCUS_NONE);
		type_user->set_button_group(type_filter_group);
		type_filter->add_child(type_user);

		Control *space1 = memnew(Control);
		space1->set_custom_minimum_size(Size2(0, 4));
		vbox->add_child(space1);

		Label *category_header = memnew(Label);
		category_header->set_text(TTR("Categories"));
		category_header->set_theme_type_variation("HeaderSmall");
		vbox->add_child(category_header);

		HBoxContainer *category_filter = memnew(HBoxContainer);
		vbox->add_child(category_filter);

		Ref<ButtonGroup> category_filter_group;
		category_filter_group.instantiate();

		category_all = memnew(Button);
		category_all->set_text(TTR("All"));
		category_all->set_tooltip_text(TTR("Show tasks of all categories"));
		category_all->set_toggle_mode(true);
		category_all->set_focus_mode(FocusMode::FOCUS_NONE);
		category_all->set_pressed(true);
		category_all->set_button_group(category_filter_group);
		category_filter->add_child(category_all);

		category_include = memnew(Button);
		category_include->set_text(TTR("Include"));
		category_include->set_tooltip_text(TTR("Show tasks from selected categories"));
		category_include->set_toggle_mode(true);
		category_include->set_focus_mode(FocusMode::FOCUS_NONE);
		category_include->set_button_group(category_filter_group);
		category_filter->add_child(category_include);

		category_exclude = memnew(Button);
		category_exclude->set_text(TTR("Exclude"));
		category_exclude->set_tooltip_text(TTR("Don't show tasks from selected categories"));
		category_exclude->set_toggle_mode(true);
		category_exclude->set_focus_mode(FocusMode::FOCUS_NONE);
		category_exclude->set_button_group(category_filter_group);
		category_filter->add_child(category_exclude);

		category_choice = memnew(VBoxContainer);
		vbox->add_child(category_choice);

		HBoxContainer *selection_controls = memnew(HBoxContainer);
		selection_controls->set_alignment(BoxContainer::ALIGNMENT_CENTER);
		category_choice->add_child(selection_controls);

		select_all = memnew(Button);
		select_all->set_tooltip_text(TTR("Select all categories"));
		select_all->set_focus_mode(FocusMode::FOCUS_NONE);
		selection_controls->add_child(select_all);

		deselect_all = memnew(Button);
		deselect_all->set_tooltip_text(TTR("Deselect all categories"));
		deselect_all->set_focus_mode(FocusMode::FOCUS_NONE);
		selection_controls->add_child(deselect_all);

		category_scroll = memnew(ScrollContainer);
		category_choice->add_child(category_scroll);

		category_list = memnew(VBoxContainer);
		category_scroll->add_child(category_list);
	}
	add_child(filter_popup);
}

TaskPalette::~TaskPalette() {
}

#endif // ! TOOLS_ENABLED
