/**
 * blackboard_plan_editor.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "blackboard_plan_editor.h"

#include "../util/limbo_compat.h"
#include "../util/limbo_string_names.h"
#include "../util/limbo_utility.h"

#ifdef LIMBOAI_MODULE
#include "editor/editor_interface.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/style_box_flat.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/button.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/h_box_container.hpp>
#include <godot_cpp/classes/input.hpp>
#include <godot_cpp/classes/input_event_mouse_motion.hpp>
#include <godot_cpp/classes/label.hpp>
#include <godot_cpp/classes/line_edit.hpp>
#include <godot_cpp/classes/margin_container.hpp>
#include <godot_cpp/classes/theme.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

BlackboardPlanEditor *BlackboardPlanEditor::singleton = nullptr;

LineEdit *BlackboardPlanEditor::_get_name_edit(int p_row_index) const {
	return Object::cast_to<LineEdit>(rows_vbox->get_child(p_row_index)->get_child(0)->get_child(1));
}

void BlackboardPlanEditor::_add_var() {
	ERR_FAIL_NULL(plan);

	int suffix = 1;
	StringName var_name = default_var_name == StringName() ? "var" : default_var_name;
	while (plan->has_var(var_name)) {
		suffix += 1;
		var_name = String(default_var_name) + itos(suffix);
	}

	BBVariable var(default_type, Variant(), default_hint, default_hint_string);
	if (default_value.get_type() == default_type) {
		var.set_value(default_value);
	}
	plan->add_var(var_name, var);
	reset_defaults();
	_refresh();
}

void BlackboardPlanEditor::_trash_var(int p_index) {
	ERR_FAIL_NULL(plan);
	StringName var_name = plan->get_var_by_index(p_index).first;
	plan->remove_var(var_name);
	_refresh();
}

void BlackboardPlanEditor::_rename_var(const StringName &p_new_name, int p_index) {
	ERR_FAIL_NULL(plan);

	LineEdit *name_edit = _get_name_edit(p_index);
	ERR_FAIL_NULL(name_edit);

	bool is_valid_var_name = plan->is_valid_var_name(p_new_name);
	if (is_valid_var_name) {
		plan->rename_var(plan->get_var_by_index(p_index).first, p_new_name);
	}

	if (is_valid_var_name || plan->get_var_by_index(p_index).first == p_new_name) {
		if (name_edit->has_theme_color_override(LW_NAME(font_color))) {
			name_edit->remove_theme_color_override(LW_NAME(font_color));
			name_edit->queue_redraw();
		}
	} else {
		if (!name_edit->has_theme_color_override(LW_NAME(font_color))) {
			name_edit->add_theme_color_override(LW_NAME(font_color), Color(1, 0, 0));
			name_edit->queue_redraw();
		}
	}
}

void BlackboardPlanEditor::_change_var_type(Variant::Type p_new_type, int p_index) {
	ERR_FAIL_NULL(plan);

	BBVariable var = plan->get_var_by_index(p_index).second;
	if (var.get_type() == p_new_type) {
		return;
	}

	PackedInt32Array allowed_hints = LimboUtility::get_singleton()->get_property_hints_allowed_for_type(p_new_type);
	if (!allowed_hints.has(var.get_hint())) {
		var.set_hint(PROPERTY_HINT_NONE);
	}

	var.set_type(p_new_type);

	plan->notify_property_list_changed();
	_refresh();
}

void BlackboardPlanEditor::_change_var_hint(PropertyHint p_new_hint, int p_index) {
	ERR_FAIL_NULL(plan);
	plan->get_var_by_index(p_index).second.set_hint(p_new_hint);
	plan->notify_property_list_changed();
	_refresh();
}

void BlackboardPlanEditor::_change_var_hint_string(const String &p_new_hint_string, int p_index) {
	ERR_FAIL_NULL(plan);
	plan->get_var_by_index(p_index).second.set_hint_string(p_new_hint_string);
	plan->notify_property_list_changed();
}

void BlackboardPlanEditor::edit_plan(const Ref<BlackboardPlan> &p_plan) {
	plan = p_plan;
	_refresh();
}

void BlackboardPlanEditor::set_defaults(const StringName &p_var_name, Variant::Type p_type, PropertyHint p_hint, String p_hint_string, Variant p_value) {
	default_var_name = p_var_name;
	default_type = p_type;
	default_hint = p_hint;
	default_hint_string = p_hint_string;
	default_value = p_value;
}

void BlackboardPlanEditor::reset_defaults() {
	default_var_name = "var";
	default_type = Variant::FLOAT;
	default_hint = PROPERTY_HINT_NONE;
	default_hint_string = "";
}

void BlackboardPlanEditor::_show_button_popup(Button *p_button, PopupMenu *p_popup, int p_index) {
	ERR_FAIL_NULL(p_button);
	ERR_FAIL_NULL(p_popup);

	Transform2D xform = p_button->get_screen_transform();
	Rect2 rect(xform.get_origin(), xform.get_scale() * p_button->get_size());
	rect.position.y += rect.size.height;
	rect.size.height = 0;
	p_popup->set_size(rect.size);
	p_popup->set_position(rect.position);

	if (p_popup == hint_menu) {
		hint_menu->clear();
		hint_menu->reset_size();
		Variant::Type t = plan->get_var_by_index(p_index).second.get_type();
		PackedInt32Array hints = LimboUtility::get_singleton()->get_property_hints_allowed_for_type(t);
		for (int i = 0; i < hints.size(); i++) {
			hint_menu->add_item(LimboUtility::get_singleton()->get_property_hint_text(PropertyHint(hints[i])), hints[i]);
		}
	}

	last_index = p_index;
	p_popup->popup();
}

void BlackboardPlanEditor::_type_chosen(int id) {
	_change_var_type(Variant::Type(id), last_index);
}

void BlackboardPlanEditor::_hint_chosen(int id) {
	_change_var_hint(PropertyHint(id), last_index);
}

void BlackboardPlanEditor::_add_var_pressed() {
	_add_var();
	LineEdit *name_edit = _get_name_edit(rows_vbox->get_child_count() - 1);
	name_edit->grab_focus();
	name_edit->select_all();

	// Note: Scroll to the end, delay is necessary here.
	scroll_container->call_deferred(LW_NAME(call_deferred), LW_NAME(set_v_scroll), 888888888);
}

void BlackboardPlanEditor::_prefetching_toggled(bool p_toggle_on) {
	ERR_FAIL_COND(plan.is_null());
	plan->set_prefetch_nodepath_vars(p_toggle_on);
}

void BlackboardPlanEditor::_drag_button_down(Control *p_row) {
	drag_index = p_row->get_index();
	drag_start = drag_index;
	drag_mouse_y_delta = 0.0;
	Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
}

void BlackboardPlanEditor::_drag_button_up() {
	Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
	plan->move_var(drag_start, drag_index);
	drag_index = -1;
	drag_start = -1;
	_refresh();
}

void BlackboardPlanEditor::_drag_button_gui_input(const Ref<InputEvent> &p_event) {
	if (drag_index < 0) {
		return;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_null()) {
		return;
	}

	drag_mouse_y_delta += mm->get_relative().y;

	if ((drag_index == 0 && drag_mouse_y_delta < 0.0) || (drag_index == (plan->get_var_count() - 1) && drag_mouse_y_delta > 0.0)) {
		drag_mouse_y_delta = 0.0;
		return;
	}

	float required_distance = 30.0f * EDSCALE;
	if (ABS(drag_mouse_y_delta) > required_distance) {
		int drag_dir = drag_mouse_y_delta > 0.0f ? 1 : -1;
		drag_mouse_y_delta -= required_distance * drag_dir;

		Control *row = Object::cast_to<Control>(rows_vbox->get_child(drag_index));
		Control *other_row = Object::cast_to<Control>(rows_vbox->get_child(drag_index + drag_dir));
		ERR_FAIL_NULL(row);
		ERR_FAIL_NULL(other_row);
		rows_vbox->move_child(row, drag_index + drag_dir);
		ADD_STYLEBOX_OVERRIDE(row, LW_NAME(panel), row->get_index() % 2 ? theme_cache.odd_style : theme_cache.even_style);
		ADD_STYLEBOX_OVERRIDE(other_row, LW_NAME(panel), other_row->get_index() % 2 ? theme_cache.odd_style : theme_cache.even_style);

		drag_index += drag_dir;
	}
}

void BlackboardPlanEditor::_visibility_changed() {
	if (!is_visible() && plan.is_valid()) {
		plan->notify_property_list_changed();
		reset_defaults();
	}
}

void BlackboardPlanEditor::_refresh() {
	for (int i = 0; i < rows_vbox->get_child_count(); i++) {
		Control *child = Object::cast_to<Control>(rows_vbox->get_child(i));
		ERR_FAIL_NULL(child);
		child->hide();
		child->queue_free();
	}

	nodepath_prefetching->set_pressed(plan->is_prefetching_nodepath_vars());

	TypedArray<StringName> names = plan->list_vars();
	for (int i = 0; i < names.size(); i++) {
		const String &var_name = names[i];
		BBVariable var = plan->get_var(var_name);

		PanelContainer *row_panel = memnew(PanelContainer);
		rows_vbox->add_child(row_panel);
		ADD_STYLEBOX_OVERRIDE(row_panel, LW_NAME(panel), i % 2 ? theme_cache.odd_style : theme_cache.even_style);
		row_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		HBoxContainer *props_hbox = memnew(HBoxContainer);
		row_panel->add_child(props_hbox);
		props_hbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		Button *drag_button = memnew(Button);
		props_hbox->add_child(drag_button);
		drag_button->set_custom_minimum_size(Size2(28.0, 28.0) * EDSCALE);
		BUTTON_SET_ICON(drag_button, theme_cache.grab_icon);
		drag_button->connect(LW_NAME(gui_input), callable_mp(this, &BlackboardPlanEditor::_drag_button_gui_input));
		drag_button->connect(LW_NAME(button_down), callable_mp(this, &BlackboardPlanEditor::_drag_button_down).bind(row_panel));
		drag_button->connect(LW_NAME(button_up), callable_mp(this, &BlackboardPlanEditor::_drag_button_up));

		LineEdit *name_edit = memnew(LineEdit);
		props_hbox->add_child(name_edit);
		name_edit->set_text(var_name);
		name_edit->set_placeholder(TTR("Variable name"));
		name_edit->set_flat(true);
		name_edit->set_custom_minimum_size(Size2(300.0, 0.0) * EDSCALE);
		name_edit->connect(LW_NAME(text_changed), callable_mp(this, &BlackboardPlanEditor::_rename_var).bind(i));
		name_edit->connect(LW_NAME(text_submitted), callable_mp(this, &BlackboardPlanEditor::_refresh).unbind(1));

		Button *type_choice = memnew(Button);
		props_hbox->add_child(type_choice);
		type_choice->set_custom_minimum_size(Size2(170, 0.0) * EDSCALE);
		type_choice->set_text(Variant::get_type_name(var.get_type()));
		type_choice->set_tooltip_text(Variant::get_type_name(var.get_type()));
		BUTTON_SET_ICON(type_choice, get_theme_icon(Variant::get_type_name(var.get_type()), LW_NAME(EditorIcons)));
		type_choice->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
		type_choice->set_flat(true);
		type_choice->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
		type_choice->connect(LW_NAME(pressed), callable_mp(this, &BlackboardPlanEditor::_show_button_popup).bind(type_choice, type_menu, i));

		Button *hint_choice = memnew(Button);
		props_hbox->add_child(hint_choice);
		hint_choice->set_custom_minimum_size(Size2(150.0, 0.0) * EDSCALE);
		hint_choice->set_text(LimboUtility::get_singleton()->get_property_hint_text(var.get_hint()));
		hint_choice->set_tooltip_text(LimboUtility::get_singleton()->get_property_hint_text(var.get_hint()));
		hint_choice->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
		hint_choice->set_flat(true);
		hint_choice->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
		hint_choice->connect(LW_NAME(pressed), callable_mp(this, &BlackboardPlanEditor::_show_button_popup).bind(hint_choice, hint_menu, i));

		LineEdit *hint_string_edit = memnew(LineEdit);
		props_hbox->add_child(hint_string_edit);
		hint_string_edit->set_custom_minimum_size(Size2(300.0, 0.0) * EDSCALE);
		hint_string_edit->set_text(var.get_hint_string());
		hint_string_edit->set_placeholder(TTR("Hint string"));
		hint_string_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hint_string_edit->set_flat(true);
		hint_string_edit->connect(LW_NAME(text_changed), callable_mp(this, &BlackboardPlanEditor::_change_var_hint_string).bind(i));
		hint_string_edit->connect(LW_NAME(text_submitted), callable_mp(this, &BlackboardPlanEditor::_refresh).unbind(1));

		Button *trash_button = memnew(Button);
		props_hbox->add_child(trash_button);
		trash_button->set_custom_minimum_size(Size2(24.0, 0.0) * EDSCALE);
		BUTTON_SET_ICON(trash_button, theme_cache.trash_icon);
		trash_button->connect(LW_NAME(pressed), callable_mp(this, &BlackboardPlanEditor::_trash_var).bind(i));
	}
}

void BlackboardPlanEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			theme_cache.trash_icon = get_theme_icon(LW_NAME(Remove), LW_NAME(EditorIcons));
			theme_cache.grab_icon = get_theme_icon(LW_NAME(TripleBar), LW_NAME(EditorIcons));

			BUTTON_SET_ICON(add_var_tool, get_theme_icon(LW_NAME(Add), LW_NAME(EditorIcons)));

			type_menu->clear();
			for (int i = 0; i < Variant::VARIANT_MAX; i++) {
				if (i == Variant::RID || i == Variant::CALLABLE || i == Variant::SIGNAL) {
					continue;
				}
				String type = Variant::get_type_name(Variant::Type(i));
				type_menu->add_icon_item(get_theme_icon(type, LW_NAME(EditorIcons)), type, i);
			}

			ADD_STYLEBOX_OVERRIDE(scroll_container, LW_NAME(panel), get_theme_stylebox(LW_NAME(panel), LW_NAME(Tree)));

			Color bg_color = get_theme_color(LW_NAME(dark_color_2), LW_NAME(Editor));
			theme_cache.odd_style->set_bg_color(bg_color.darkened(-0.05));
			theme_cache.even_style->set_bg_color(bg_color.darkened(0.05));
			theme_cache.header_style->set_bg_color(bg_color.darkened(-0.2));

			ADD_STYLEBOX_OVERRIDE(header_row, LW_NAME(panel), theme_cache.header_style);
		} break;
		case NOTIFICATION_ENTER_TREE: {
			add_var_tool->connect(LW_NAME(pressed), callable_mp(this, &BlackboardPlanEditor::_add_var_pressed));
			connect(LW_NAME(visibility_changed), callable_mp(this, &BlackboardPlanEditor::_visibility_changed));
			type_menu->connect(LW_NAME(id_pressed), callable_mp(this, &BlackboardPlanEditor::_type_chosen));
			hint_menu->connect(LW_NAME(id_pressed), callable_mp(this, &BlackboardPlanEditor::_hint_chosen));
			nodepath_prefetching->connect(LW_NAME(toggled), callable_mp(this, &BlackboardPlanEditor::_prefetching_toggled));

			for (int i = 0; i < PropertyHint::PROPERTY_HINT_MAX; i++) {
				hint_menu->add_item(LimboUtility::get_singleton()->get_property_hint_text(PropertyHint(i)), i);
			}

			singleton = this;
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (singleton == this) {
				singleton = nullptr;
			}
		} break;
	}
}

BlackboardPlanEditor::BlackboardPlanEditor() {
	reset_defaults();

	set_title(TTR("Manage Blackboard Plan"));

	VBoxContainer *vbox = memnew(VBoxContainer);
	vbox->add_theme_constant_override(LW_NAME(separation), 8 * EDSCALE);
	add_child(vbox);

	HBoxContainer *toolbar = memnew(HBoxContainer);
	vbox->add_child(toolbar);

	add_var_tool = memnew(Button);
	toolbar->add_child(add_var_tool);
	add_var_tool->set_focus_mode(Control::FOCUS_NONE);
	add_var_tool->set_text(TTR("Add variable"));

	nodepath_prefetching = memnew(CheckBox);
	toolbar->add_child(nodepath_prefetching);
	nodepath_prefetching->set_text(TTR("NodePath Prefetching"));
	nodepath_prefetching->set_tooltip_text(TTR("If checked, NodePath variables will be prefetched on Blackboard initialization."));
	nodepath_prefetching->set_h_size_flags(Control::SIZE_EXPAND | Control::SIZE_SHRINK_END);
	nodepath_prefetching->set_focus_mode(Control::FOCUS_NONE);

	{
		// * Header
		header_row = memnew(PanelContainer);
		vbox->add_child(header_row);
		header_row->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		HBoxContainer *labels_hbox = memnew(HBoxContainer);
		header_row->add_child(labels_hbox);
		labels_hbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		Control *offset = memnew(Control);
		labels_hbox->add_child(offset);
		offset->set_custom_minimum_size(Size2(2.0, 0.0) * EDSCALE);

		Label *drag_header = memnew(Label);
		labels_hbox->add_child(drag_header);
		drag_header->set_custom_minimum_size(Size2(28.0, 28.0) * EDSCALE);

		Label *name_header = memnew(Label);
		labels_hbox->add_child(name_header);
		name_header->set_text(TTR("Name"));
		name_header->set_custom_minimum_size(Size2(300.0, 0.0) * EDSCALE);
		name_header->set_theme_type_variation(LW_NAME(HeaderSmall));

		Label *type_header = memnew(Label);
		labels_hbox->add_child(type_header);
		type_header->set_text(TTR("Type"));
		type_header->set_custom_minimum_size(Size2(170, 0.0) * EDSCALE);
		type_header->set_theme_type_variation(LW_NAME(HeaderSmall));

		Label *hint_header = memnew(Label);
		labels_hbox->add_child(hint_header);
		hint_header->set_text(TTR("Hint"));
		hint_header->set_custom_minimum_size(Size2(150.0, 0.0) * EDSCALE);
		hint_header->set_theme_type_variation(LW_NAME(HeaderSmall));

		Label *hint_string_header = memnew(Label);
		labels_hbox->add_child(hint_string_header);
		hint_string_header->set_text(TTR("Hint string"));
		hint_string_header->set_custom_minimum_size(Size2(300.0, 0.0) * EDSCALE);
		hint_string_header->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hint_string_header->set_theme_type_variation(LW_NAME(HeaderSmall));
	}

	scroll_container = memnew(ScrollContainer);
	vbox->add_child(scroll_container);
	scroll_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	scroll_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	scroll_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	scroll_container->set_custom_minimum_size(Size2(0.0, 600.0) * EDSCALE);

	rows_vbox = memnew(VBoxContainer);
	scroll_container->add_child(rows_vbox);
	rows_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	rows_vbox->add_theme_constant_override(LW_NAME(separation), 0);

	type_menu = memnew(PopupMenu);
	add_child(type_menu);

	hint_menu = memnew(PopupMenu);
	add_child(hint_menu);

	theme_cache.odd_style.instantiate();
	theme_cache.even_style.instantiate();
	theme_cache.header_style.instantiate();
}

// ***** EditorInspectorPluginBBPlan *****

void EditorInspectorPluginBBPlan::_edit_plan(const Ref<BlackboardPlan> &p_plan) {
	ERR_FAIL_NULL(p_plan);
	plan_editor->edit_plan(p_plan);
	plan_editor->popup_centered();
}

void EditorInspectorPluginBBPlan::_open_base_plan(const Ref<BlackboardPlan> &p_plan) {
	ERR_FAIL_NULL(p_plan);
	ERR_FAIL_NULL(p_plan->get_base_plan());
	EditorInterface::get_singleton()->call_deferred("edit_resource", p_plan->get_base_plan());
}

#ifdef LIMBOAI_MODULE
bool EditorInspectorPluginBBPlan::can_handle(Object *p_object) {
#elif LIMBOAI_GDEXTENSION
bool EditorInspectorPluginBBPlan::_can_handle(Object *p_object) const {
#endif
	Ref<BlackboardPlan> plan = Object::cast_to<BlackboardPlan>(p_object);
	if (plan.is_valid()) {
		plan->sync_with_base_plan();
	}
	return plan.is_valid();
}

#ifdef LIMBOAI_MODULE
void EditorInspectorPluginBBPlan::parse_begin(Object *p_object) {
#elif LIMBOAI_GDEXTENSION
void EditorInspectorPluginBBPlan::_parse_begin(Object *p_object) {
#endif
	Ref<BlackboardPlan> plan = Object::cast_to<BlackboardPlan>(p_object);
	ERR_FAIL_NULL(plan);

	PanelContainer *panel = memnew(PanelContainer);
	ADD_STYLEBOX_OVERRIDE(panel, LW_NAME(panel), toolbar_style);

	MarginContainer *margin_container = memnew(MarginContainer);
	panel->add_child(margin_container);
	margin_container->set_theme_type_variation("MarginContainer4px");
	margin_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	margin_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	VBoxContainer *toolbar = memnew(VBoxContainer);
	margin_container->add_child(toolbar);
	toolbar->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	if (plan->is_derived()) {
		Button *goto_btn = memnew(Button);
		toolbar->add_child(goto_btn);
		goto_btn->set_text(TTR("Edit Base"));
		goto_btn->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
		goto_btn->set_custom_minimum_size(Size2(150.0, 0.0) * EDSCALE);
		BUTTON_SET_ICON(goto_btn, EditorInterface::get_singleton()->get_editor_theme()->get_icon(LW_NAME(Edit), LW_NAME(EditorIcons)));
		goto_btn->connect(LW_NAME(pressed), callable_mp(this, &EditorInspectorPluginBBPlan::_open_base_plan).bind(plan));
	} else {
		Button *edit_btn = memnew(Button);
		toolbar->add_child(edit_btn);
		edit_btn->set_text(TTR("Manage..."));
		edit_btn->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
		edit_btn->set_custom_minimum_size(Size2(150.0, 0.0) * EDSCALE);
		BUTTON_SET_ICON(edit_btn, EditorInterface::get_singleton()->get_editor_theme()->get_icon(LW_NAME(EditAddRemove), LW_NAME(EditorIcons)));
		edit_btn->connect(LW_NAME(pressed), callable_mp(this, &EditorInspectorPluginBBPlan::_edit_plan).bind(plan));
	}

	add_custom_control(panel);
}

EditorInspectorPluginBBPlan::EditorInspectorPluginBBPlan() {
	plan_editor = memnew(BlackboardPlanEditor);
	EditorInterface::get_singleton()->get_base_control()->add_child(plan_editor);
	plan_editor->hide();

	toolbar_style = Ref<StyleBoxFlat>(memnew(StyleBoxFlat));
	Color bg = EditorInterface::get_singleton()->get_editor_theme()->get_color(LW_NAME(accent_color), LW_NAME(Editor));
	bg = bg.darkened(-0.1);
	bg.a *= 0.2;
	toolbar_style->set_bg_color(bg);
}

#endif // TOOLS_ENABLED
