/**
 * blackboard_plan_editor.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BLACKBOARD_PLAN_EDITOR_H
#define BLACKBOARD_PLAN_EDITOR_H

#ifdef TOOLS_ENABLED

#include "../blackboard/blackboard_plan.h"

#ifdef LIMBOAI_MODULE
#include "editor/editor_inspector.h"
#include "scene/gui/check_box.h"
#include "scene/gui/dialogs.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/accept_dialog.hpp>
#include <godot_cpp/classes/check_box.hpp>
#include <godot_cpp/classes/editor_inspector_plugin.hpp>
#include <godot_cpp/classes/line_edit.hpp>
#include <godot_cpp/classes/panel_container.hpp>
#include <godot_cpp/classes/popup_menu.hpp>
#include <godot_cpp/classes/scroll_container.hpp>
#include <godot_cpp/classes/style_box_flat.hpp>
#include <godot_cpp/classes/v_box_container.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

// *****

class BlackboardPlanEditor : public AcceptDialog {
	GDCLASS(BlackboardPlanEditor, AcceptDialog);

private:
	static BlackboardPlanEditor *singleton;

private:
	struct ThemeCache {
		Ref<Texture2D> trash_icon;
		Ref<Texture2D> grab_icon;
		Ref<StyleBoxFlat> odd_style;
		Ref<StyleBoxFlat> even_style;
		Ref<StyleBoxFlat> header_style;
	} theme_cache;

	int last_index = 0;
	int drag_mouse_y_delta = 0;
	int drag_start = -1;
	int drag_index = -1;

	Ref<BlackboardPlan> plan;
	StringName default_var_name;
	Variant::Type default_type = Variant::NIL;
	PropertyHint default_hint = PROPERTY_HINT_NONE;
	String default_hint_string;
	Variant default_value;

	VBoxContainer *rows_vbox;
	Button *add_var_tool;
	CheckBox *nodepath_prefetching;
	PanelContainer *header_row;
	ScrollContainer *scroll_container;
	PopupMenu *type_menu;
	PopupMenu *hint_menu;

	LineEdit *_get_name_edit(int p_row_index) const;

	void _add_var();
	void _trash_var(int p_index);
	void _rename_var(const StringName &p_new_name, int p_index);
	void _change_var_type(Variant::Type p_new_type, int p_index);
	void _change_var_hint(PropertyHint p_new_hint, int p_index);
	void _change_var_hint_string(const String &p_new_hint_string, int p_index);

	void _show_button_popup(Button *p_button, PopupMenu *p_popup, int p_index);
	void _type_chosen(int id);
	void _hint_chosen(int id);
	void _add_var_pressed();
	void _prefetching_toggled(bool p_toggle_on);

	void _drag_button_down(Control *p_row);
	void _drag_button_up();
	void _drag_button_gui_input(const Ref<InputEvent> &p_event);

	void _refresh();
	void _visibility_changed();

protected:
	static void _bind_methods() {}

	void _notification(int p_what);

public:
	_FORCE_INLINE_ static BlackboardPlanEditor *get_singleton() { return singleton; }

	void edit_plan(const Ref<BlackboardPlan> &p_plan);
	void set_defaults(const StringName &p_name, Variant::Type p_type = Variant::FLOAT, PropertyHint p_hint = PROPERTY_HINT_NONE, String p_hint_string = "", Variant p_value = Variant());
	void reset_defaults();

	BlackboardPlanEditor();
};

// *****

class EditorInspectorPluginBBPlan : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginBBPlan, EditorInspectorPlugin);

private:
	BlackboardPlanEditor *plan_editor;
	Ref<StyleBoxFlat> toolbar_style;

	void _edit_plan(const Ref<BlackboardPlan> &p_plan);
	void _open_base_plan(const Ref<BlackboardPlan> &p_plan);

protected:
	static void _bind_methods() {}

public:
#ifdef LIMBOAI_MODULE
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
#elif LIMBOAI_GDEXTENSION
	virtual bool _can_handle(Object *p_object) const override;
	virtual void _parse_begin(Object *p_object) override;
#endif

	EditorInspectorPluginBBPlan();
};

#endif // TOOLS_ENABLED

#endif // BLACKBOARD_PLAN_EDITOR_H
