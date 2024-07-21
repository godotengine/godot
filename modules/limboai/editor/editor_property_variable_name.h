/**
 * editor_property_variable_name.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef EDITOR_PROPERTY_VARIABLE_NAME_H
#define EDITOR_PROPERTY_VARIABLE_NAME_H

#ifdef TOOLS_ENABLED

#include "../blackboard/blackboard_plan.h"

#ifdef LIMBOAI_MODULE
#include "editor/editor_inspector.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/button.hpp>
#include <godot_cpp/classes/editor_inspector_plugin.hpp>
#include <godot_cpp/classes/editor_property.hpp>
#include <godot_cpp/classes/line_edit.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class EditorPropertyVariableName : public EditorProperty {
	GDCLASS(EditorPropertyVariableName, EditorProperty);

private:
	static int last_caret_column;

private:
	struct ThemeCache {
		Ref<Texture2D> var_add_icon;
		Ref<Texture2D> var_empty_icon;
		Ref<Texture2D> var_error_icon;
		Ref<Texture2D> var_exists_icon;
		Ref<Texture2D> var_not_found_icon;
		Ref<Texture2D> var_private_icon;
	};
	ThemeCache theme_cache;

	Ref<BlackboardPlan> plan;

	bool allow_empty = false;
	Variant::Type expected_type = Variant::NIL;
	PropertyHint default_hint = PROPERTY_HINT_NONE;
	String default_hint_string;
	Variant default_value;

	bool updating = false;

	LineEdit *name_edit;
	Button *drop_btn;
	Button *status_btn;
	PopupMenu *variables_popup;

	void _show_variables_popup();
	void _name_changed(const String &p_new_name);
	void _name_submitted();
	void _variable_selected(int p_id);
	void _update_status();

	void _status_pressed();
	void _status_mouse_entered();
	void _status_mouse_exited();

protected:
	static void _bind_methods() {}

	void _notification(int p_what);

public:
#ifdef LIMBOAI_MODULE
	virtual void update_property() override;
#elif LIMBOAI_GDEXTENSION
	virtual void _update_property() override;
#endif

	void setup(const Ref<BlackboardPlan> &p_plan, bool p_allow_empty, Variant::Type p_type = Variant::NIL, PropertyHint p_hint = PROPERTY_HINT_NONE, String p_hint_string = "", Variant p_default_value = Variant());
	EditorPropertyVariableName();
};

class EditorInspectorPluginVariableName : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginVariableName, EditorInspectorPlugin);

private:
	Callable editor_plan_provider;

protected:
	static void _bind_methods() {}

public:
#ifdef LIMBOAI_MODULE
	virtual bool can_handle(Object *p_object) override;
	virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide = false) override;
#elif LIMBOAI_GDEXTENSION
	virtual bool _can_handle(Object *p_object) const override;
	virtual bool _parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide = false) override;
#endif

	void set_editor_plan_provider(const Callable &p_getter) { editor_plan_provider = p_getter; }

	EditorInspectorPluginVariableName() = default;
};

#endif // TOOLS_ENABLED

#endif // EDITOR_PROPERTY_VARIABLE_NAME_H
