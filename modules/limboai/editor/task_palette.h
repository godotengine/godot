/**
 * task_palette.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#ifndef TASK_PALETTE_H
#define TASK_PALETTE_H

#ifdef LIMBOAI_MODULE
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/button.hpp>
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/flow_container.hpp>
#include <godot_cpp/classes/line_edit.hpp>
#include <godot_cpp/classes/panel_container.hpp>
#include <godot_cpp/classes/popup_panel.hpp>
#include <godot_cpp/classes/scroll_container.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/v_box_container.hpp>
#include <godot_cpp/templates/hash_set.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class TaskButton : public Button {
	GDCLASS(TaskButton, Button);

private:
	String task_meta;

	Control *_do_make_tooltip() const;

#ifdef LIMBOAI_MODULE
	String _module_get_help_description(const String &p_class_or_script_path) const;
#endif

protected:
	static void _bind_methods();

public:
#ifdef LIMBOAI_MODULE
	virtual Control *make_custom_tooltip(const String &p_text) const override { return _do_make_tooltip(); }
#elif LIMBOAI_GDEXTENSION
	virtual Object *_make_custom_tooltip(const String &p_text) const override { return _do_make_tooltip(); }
#endif

	String get_task_meta() const { return task_meta; }
	void set_task_meta(const String &p_task_meta) { task_meta = p_task_meta; }

	TaskButton();
};

class TaskPaletteSection : public VBoxContainer {
	GDCLASS(TaskPaletteSection, VBoxContainer);

private:
	struct ThemeCache {
		Ref<Texture2D> arrow_down_icon;
		Ref<Texture2D> arrow_right_icon;
	} theme_cache;

	FlowContainer *tasks_container;
	Button *section_header;

	void _on_task_button_pressed(const String &p_task);
	void _on_task_button_gui_input(const Ref<InputEvent> &p_event, const String &p_task);
	void _on_header_pressed();

protected:
	static void _bind_methods();

	void _notification(int p_what);

	virtual void _do_update_theme_item_cache();

public:
	void set_filter(String p_filter);
	void add_task_button(const String &p_name, const Ref<Texture> &icon, const String &p_meta);

	void set_collapsed(bool p_collapsed);
	bool is_collapsed() const;

	String get_category_name() const { return section_header->get_text(); }
	void set_category_name(const String &p_cat) { section_header->set_text(p_cat); }

	TaskPaletteSection();
	~TaskPaletteSection();
};

class TaskPalette : public PanelContainer {
	GDCLASS(TaskPalette, PanelContainer)

private:
	enum MenuAction {
		MENU_EDIT_SCRIPT,
		MENU_OPEN_DOC,
		MENU_FAVORITE,
	};

	struct FilterSettings {
		enum TypeFilter {
			TYPE_ALL,
			TYPE_CORE,
			TYPE_USER,
		} type_filter;

		enum CategoryFilter {
			CATEGORY_ALL,
			CATEGORY_INCLUDE,
			CATEGORY_EXCLUDE,
		} category_filter;

		HashSet<String> excluded_categories;
	} filter_settings;

	struct ThemeCache {
		Ref<Texture2D> add_to_favorites_icon;
		Ref<Texture2D> edit_script_icon;
		Ref<Texture2D> open_doc_icon;
		Ref<Texture2D> remove_from_favorites_icon;

		Ref<StyleBox> category_choice_background;
	} theme_cache;

	LineEdit *filter_edit;
	VBoxContainer *sections;
	PopupMenu *menu;
	Button *tool_filters;
	Button *tool_refresh;

	// Filter popup
	PopupPanel *filter_popup;
	Button *type_all;
	Button *type_core;
	Button *type_user;
	Button *category_all;
	Button *category_include;
	Button *category_exclude;
	VBoxContainer *category_choice;
	Button *select_all;
	Button *deselect_all;
	ScrollContainer *category_scroll;
	VBoxContainer *category_list;

	String context_task;
	bool dialog_mode = false;

	void _menu_action_selected(int p_id);
	void _on_task_button_pressed(const String &p_task);
	void _on_task_button_rmb(const String &p_task);
	void _apply_filter(const String &p_text);
	void _update_filter_popup();
	void _show_filter_popup();
	void _type_filter_changed();
	void _category_filter_changed();
	void _set_all_filter_categories(bool p_selected);
	void _category_item_toggled(bool p_pressed, const String &p_category);
	void _filter_data_changed();
	void _draw_filter_popup_background();
	void _update_filter_button();

	_FORCE_INLINE_ void _set_category_excluded(const String &p_category, bool p_excluded) {
		if (p_excluded) {
			filter_settings.excluded_categories.insert(p_category);
		} else {
			filter_settings.excluded_categories.erase(p_category);
		}
	}

protected:
	virtual void _do_update_theme_item_cache();

	void _notification(int p_what);
	static void _bind_methods();

public:
	void refresh();
	void use_dialog_mode();
	void clear_filter() { filter_edit->set_text(""); }

	TaskPalette();
	~TaskPalette();
};

#endif // TASK_PALETTE_H

#endif // ! TOOLS_ENABLED
