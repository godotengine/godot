/**
 * editor_property_variable_name.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "editor_property_variable_name.h"

#include "../logic/blackboard/bb_param/bb_param.h"
#include "blackboard_plan_editor.h"

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/popup_menu.h"

int EditorPropertyVariableName::last_caret_column = 0;

//***** EditorPropertyVariableName

void EditorPropertyVariableName::_show_variables_popup() {
	ERR_FAIL_NULL(plan);

	variables_popup->clear();
	variables_popup->reset_size();
	TypedArray<StringName> var_names = plan->list_vars();
	for (int i = 0; i < var_names.size(); i++) {
		variables_popup->add_item(var_names[i], i);
	}

	Transform2D xform = name_edit->get_screen_transform();
	Rect2 rect(xform.get_origin(), xform.get_scale() * name_edit->get_size());
	rect.position.y += rect.size.height;
	rect.size.height = 0;
	variables_popup->set_size(rect.size);
	variables_popup->set_position(rect.position);

	variables_popup->popup(rect);
}

void EditorPropertyVariableName::_name_changed(const String &p_new_name) {
	if (updating) {
		return;
	}

	emit_changed(get_edited_property(), p_new_name);
	last_caret_column = name_edit->get_caret_column();
	_update_status();
}

void EditorPropertyVariableName::_name_submitted() {
	_name_changed(name_edit->get_text());
	if (name_edit->has_focus()) {
		name_edit->release_focus();
	}
}

void EditorPropertyVariableName::_variable_selected(int p_id) {
	String var_name = plan->get_var_by_index(p_id).first;
	name_edit->set_text(var_name);
	_name_submitted();
}

void EditorPropertyVariableName::_update_status() {
	status_btn->set_visible(plan.is_valid());
	drop_btn->set_visible(plan.is_valid());
	if (plan.is_null()) {
		return;
	}
	String var_name = name_edit->get_text();
	if (var_name.is_empty() && allow_empty) {
		status_btn->set_icon(theme_cache.var_empty_icon);
		status_btn->set_tooltip_text(TTR("Variable name not specified.\nClick to open the blackboard plan."));
	} else if (plan->has_var(var_name)) {
		if (expected_type == Variant::NIL || plan->get_var(var_name).get_type() == expected_type) {
			status_btn->set_icon( theme_cache.var_exists_icon);
			status_btn->set_tooltip_text(TTR("This variable is present in the blackboard plan.\nClick to open the blackboard plan."));
		} else {
			status_btn->set_icon( theme_cache.var_error_icon);
			status_btn->set_tooltip_text(TTR(vformat(
					"The %s variable in the blackboard plan should be of type %s.\nClick to open the blackboard plan.",
					BBParam::decorate_var(var_name),
					Variant::get_type_name(expected_type))));
		}
	} else if (name_edit->get_text().begins_with("_")) {
		status_btn->set_icon(theme_cache.var_private_icon);
		status_btn->set_tooltip_text(TTR("This variable is private and is not included in the blackboard plan.\nClick to open the blackboard plan."));
	} else {
		status_btn->set_icon( theme_cache.var_not_found_icon);
		status_btn->set_tooltip_text(TTR("No matching variable found in the blackboard plan!\nClick to open the blackboard plan."));
	}
}

void EditorPropertyVariableName::_status_pressed() {
	ERR_FAIL_NULL(plan);
	if (!plan->has_var(name_edit->get_text())) {
		BlackboardPlanEditor::get_singleton()->set_defaults(name_edit->get_text(),
				expected_type == Variant::NIL ? Variant::FLOAT : expected_type,
				default_hint, default_hint_string, default_value);
	}
	BlackboardPlanEditor::get_singleton()->edit_plan(plan);
	BlackboardPlanEditor::get_singleton()->popup_centered();
}

void EditorPropertyVariableName::_status_mouse_entered() {
	ERR_FAIL_NULL(plan);
	if (!plan->has_var(name_edit->get_text())) {
		status_btn->set_icon( theme_cache.var_add_icon);
	}
}

void EditorPropertyVariableName::_status_mouse_exited() {
	ERR_FAIL_NULL(plan);
	_update_status();
}

void EditorPropertyVariableName::update_property() {
	String s = get_edited_object()->get(get_edited_property());
	updating = true;
	if (name_edit->get_text() != s) {
		int caret = name_edit->get_caret_column();
		if (caret == 0) {
			caret = last_caret_column;
		}
		name_edit->set_text(s);
		name_edit->set_caret_column(caret);
	}
	name_edit->set_editable(!is_read_only());
	_update_status();
	updating = false;
}

void EditorPropertyVariableName::setup(const Ref<BlackboardPlan> &p_plan, bool p_allow_empty, Variant::Type p_type, PropertyHint p_hint, String p_hint_string, Variant p_default_value) {
	plan = p_plan;
	allow_empty = p_allow_empty;
	expected_type = p_type;
	default_hint = p_hint;
	default_hint_string = p_hint_string;
	default_value = p_default_value;
	_update_status();
}

void EditorPropertyVariableName::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			name_edit->connect(SNAME("text_changed"), callable_mp(this, &EditorPropertyVariableName::_name_changed));
			name_edit->connect(SNAME("text_submitted"), callable_mp(this, &EditorPropertyVariableName::_name_submitted).unbind(1));
			name_edit->connect(SNAME("focus_exited"), callable_mp(this, &EditorPropertyVariableName::_name_submitted));
			variables_popup->connect(SNAME("id_pressed"), callable_mp(this, &EditorPropertyVariableName::_variable_selected));
			drop_btn->connect(SNAME("pressed"), callable_mp(this, &EditorPropertyVariableName::_show_variables_popup));
			status_btn->connect(SNAME("pressed"), callable_mp(this, &EditorPropertyVariableName::_status_pressed));
			status_btn->connect(SNAME("mouse_entered"), callable_mp(this, &EditorPropertyVariableName::_status_mouse_entered));
			status_btn->connect(SNAME("mouse_exited"), callable_mp(this, &EditorPropertyVariableName::_status_mouse_exited));
		} break;
		case NOTIFICATION_ENTER_TREE: {
			BlackboardPlanEditor::get_singleton()->connect(SNAME("visibility_changed"), callable_mp(this, &EditorPropertyVariableName::_update_status));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (BlackboardPlanEditor::get_singleton()) {
				BlackboardPlanEditor::get_singleton()->disconnect(SNAME("visibility_changed"), callable_mp(this, &EditorPropertyVariableName::_update_status));
			}
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			drop_btn->set_icon(get_theme_icon(SNAME("GuiOptionArrow"), SNAME("EditorIcons")));
			theme_cache.var_add_icon = BBParam::get_task_icon(SNAME("LimboVarAdd"));
			theme_cache.var_exists_icon = BBParam::get_task_icon(SNAME("LimboVarExists"));
			theme_cache.var_not_found_icon = BBParam::get_task_icon(SNAME("LimboVarNotFound"));
			theme_cache.var_private_icon = BBParam::get_task_icon(SNAME("LimboVarPrivate"));
			theme_cache.var_empty_icon = BBParam::get_task_icon(SNAME("LimboVarEmpty"));
			theme_cache.var_error_icon = BBParam::get_task_icon(SNAME("LimboVarError"));
		} break;
	}
}

EditorPropertyVariableName::EditorPropertyVariableName() {
	HBoxContainer *hbox = memnew(HBoxContainer);
	add_child(hbox);
	hbox->add_theme_constant_override(SNAME("separation"), 0);

	name_edit = memnew(LineEdit);
	hbox->add_child(name_edit);
	add_focusable(name_edit);
	name_edit->set_h_size_flags(SIZE_EXPAND_FILL);
	name_edit->set_placeholder(TTR("Variable name"));

	drop_btn = memnew(Button);
	hbox->add_child(drop_btn);
	drop_btn->set_flat(true);
	drop_btn->set_focus_mode(FOCUS_NONE);

	status_btn = memnew(Button);
	hbox->add_child(status_btn);
	status_btn->set_flat(true);
	status_btn->set_focus_mode(FOCUS_NONE);

	variables_popup = memnew(PopupMenu);
	add_child(variables_popup);
}

//***** EditorInspectorPluginVariableName

bool EditorInspectorPluginVariableName::can_handle(Object *p_object) {
	Ref<BBParam> param = Object::cast_to<BBParam>(p_object);
	if (param.is_valid()) {
		return true;
	}
	Ref<BlackboardPlan> plan = Object::cast_to<BlackboardPlan>(p_object);
	if (plan.is_valid()) {
		return true;
	}
	return false;
}

bool EditorInspectorPluginVariableName::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	bool is_mapping = p_path.begins_with("mapping/");
	if (!(p_type == Variant::Type::STRING_NAME || p_type == Variant::Type::STRING) || !(is_mapping || p_path.ends_with("_var") || p_path.ends_with("variable"))) {
		return false;
	}

	Ref<BlackboardPlan> plan;
	Variant::Type expected_type = Variant::NIL;
	PropertyHint default_hint = PROPERTY_HINT_NONE;
	String default_hint_string;
	Variant default_value;
	if (is_mapping) {
		plan.reference_ptr(Object::cast_to<BlackboardPlan>(p_object));
		ERR_FAIL_NULL_V(plan, false);
		String var_name = p_path.trim_prefix("mapping/");
		if (plan->has_var(var_name)) {
			BBVariable variable = plan->get_var(var_name);
			expected_type = variable.get_type();
			default_hint = variable.get_hint();
			default_hint_string = variable.get_hint_string();
			default_value = variable.get_value();
		}
		if (plan->get_parent_scope_plan_provider().is_valid()) {
			Ref<BlackboardPlan> parent_plan = plan->get_parent_scope_plan_provider().call();
			if (parent_plan.is_valid()) {
				plan = parent_plan;
			}
		}
		ERR_FAIL_NULL_V(plan, false);
	} else {
		plan = editor_plan_provider.call();
	}

	EditorPropertyVariableName *ed = memnew(EditorPropertyVariableName);
	ed->setup(plan, is_mapping, expected_type, default_hint, default_hint_string, default_value);
	add_property_editor(p_path, ed, expected_type);

	return true;
}

#endif // TOOLS_ENABLED
