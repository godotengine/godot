/**
 * editor_property_bb_param.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#ifndef EDITOR_PROPERTY_BB_PARAM_H
#define EDITOR_PROPERTY_BB_PARAM_H

#ifdef LIMBOAI_MODULE

#include "editor/editor_inspector.h"

#include "modules/limboai/blackboard/bb_param/bb_param.h"
#include "modules/limboai/editor/mode_switch_button.h"

#include "scene/gui/box_container.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/menu_button.h"

class EditorPropertyBBParam : public EditorProperty {
	GDCLASS(EditorPropertyBBParam, EditorProperty);

private:
	enum Mode {
		SPECIFY_VALUE,
		BIND_VAR,
	};

	StringName param_type;
	PropertyHint property_hint = PROPERTY_HINT_NONE;
	Mode mode = Mode::SPECIFY_VALUE;

	HBoxContainer *hbox = nullptr;
	MarginContainer *bottom_container = nullptr;
	HBoxContainer *editor_hbox = nullptr;
	ModeSwitchButton *mode_button = nullptr;
	EditorProperty *value_editor = nullptr;
	LineEdit *variable_edit = nullptr;
	MenuButton *type_choice = nullptr;

	Ref<BBParam> _get_edited_param();

	void _create_value_editor(Variant::Type p_type);
	void _remove_value_editor();

	void _value_edited(const String &p_property, Variant p_value, const String &p_name = "", bool p_changing = false);
	void _variable_edited(const String &p_text);
	void _mode_changed();
	void _type_selected(int p_index);

protected:
	void _notification(int p_what);

public:
	virtual void update_property() override;
	void setup(PropertyHint p_hint, const String &p_hint_text);

	EditorPropertyBBParam();
};

class EditorInspectorPluginBBParam : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginBBParam, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide = false) override;
};

#endif // ! LIMBOAI_MODULE

#endif // ! EDITOR_PROPERTY_BB_PARAM_H

#endif // ! TOOLS_ENABLED
