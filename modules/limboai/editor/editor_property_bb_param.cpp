/**
 * editor_property_bb_param.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#ifdef LIMBOAI_MODULE

#include "editor_property_bb_param.h"

#include "../blackboard/bb_param/bb_param.h"
#include "../blackboard/bb_param/bb_variant.h"
#include "../util/limbo_string_names.h"
#include "editor_property_variable_name.h"
#include "mode_switch_button.h"

#include "core/error/error_macros.h"
#include "core/io/marshalls.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/os/memory.h"
#include "core/string/print_string.h"
#include "core/variant/variant.h"
#include "editor/editor_inspector.h"
#include "editor/editor_properties.h"
#include "editor/editor_properties_array_dict.h"
#include "editor/editor_properties_vector.h"
#include "editor/editor_settings.h"
#include "scene/gui/base_button.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/menu_button.h"

Ref<BBParam> EditorPropertyBBParam::_get_edited_param() {
	Ref<BBParam> param = get_edited_property_value();
	if (param.is_null()) {
		// Create parameter resource if null.
		param = ClassDB::instantiate(param_type);
		get_edited_object()->set(get_edited_property(), param);
	}
	return param;
}

void EditorPropertyBBParam::_create_value_editor(Variant::Type p_type) {
	if (value_editor) {
		if (value_editor->get_meta(SNAME("_param_type")) == Variant(p_type)) {
			return;
		}
		_remove_value_editor();
	}

	bool is_bottom = false;

	switch (p_type) {
		case Variant::NIL: {
			value_editor = memnew(EditorPropertyNil);
		} break;
		case Variant::BOOL: {
			value_editor = memnew(EditorPropertyCheck);
		} break;
		case Variant::INT: {
			EditorPropertyInteger *editor = memnew(EditorPropertyInteger);
			editor->setup(-100000, 100000, 1, false, true, true);
			value_editor = editor;
		} break;
		case Variant::FLOAT: {
			EditorPropertyFloat *editor = memnew(EditorPropertyFloat);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true, false, true, true);
			value_editor = editor;
		} break;
		case Variant::STRING: {
			if (property_hint == PROPERTY_HINT_MULTILINE_TEXT) {
				value_editor = memnew(EditorPropertyMultilineText);
			} else {
				value_editor = memnew(EditorPropertyText);
			}
			is_bottom = (property_hint == PROPERTY_HINT_MULTILINE_TEXT);
		} break;
		case Variant::VECTOR2: {
			EditorPropertyVector2 *editor = memnew(EditorPropertyVector2);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
		} break;
		case Variant::VECTOR2I: {
			EditorPropertyVector2i *editor = memnew(EditorPropertyVector2i);
			editor->setup(-100000, 100000);
			value_editor = editor;
		} break;
		case Variant::RECT2: {
			EditorPropertyRect2 *editor = memnew(EditorPropertyRect2);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::RECT2I: {
			EditorPropertyRect2i *editor = memnew(EditorPropertyRect2i);
			editor->setup(-100000, 100000);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::VECTOR3: {
			EditorPropertyVector3 *editor = memnew(EditorPropertyVector3);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::VECTOR3I: {
			EditorPropertyVector3i *editor = memnew(EditorPropertyVector3i);
			editor->setup(-100000, 100000);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::VECTOR4: {
			EditorPropertyVector4 *editor = memnew(EditorPropertyVector4);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::VECTOR4I: {
			EditorPropertyVector4i *editor = memnew(EditorPropertyVector4i);
			editor->setup(-100000, 100000);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::TRANSFORM2D: {
			EditorPropertyTransform2D *editor = memnew(EditorPropertyTransform2D);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::PLANE: {
			EditorPropertyPlane *editor = memnew(EditorPropertyPlane);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::QUATERNION: {
			EditorPropertyQuaternion *editor = memnew(EditorPropertyQuaternion);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::AABB: {
			EditorPropertyAABB *editor = memnew(EditorPropertyAABB);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::BASIS: {
			EditorPropertyBasis *editor = memnew(EditorPropertyBasis);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::TRANSFORM3D: {
			EditorPropertyTransform3D *editor = memnew(EditorPropertyTransform3D);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::PROJECTION: {
			EditorPropertyProjection *editor = memnew(EditorPropertyProjection);
			editor->setup(-100000, 100000, EDITOR_GET("interface/inspector/default_float_step"), true);
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::COLOR: {
			value_editor = memnew(EditorPropertyColor);
		} break;
		case Variant::STRING_NAME: {
			EditorPropertyText *editor = memnew(EditorPropertyText);
			editor->set_string_name(true);
			value_editor = editor;
			is_bottom = (property_hint == PROPERTY_HINT_MULTILINE_TEXT);
		} break;
		case Variant::NODE_PATH: {
			value_editor = memnew(EditorPropertyNodePath);
		} break;
			// case Variant::RID: {
			// } break;
			// case Variant::SIGNAL: {
			// } break;
			// case Variant::CALLABLE: {
			// } break;
		case Variant::OBJECT: {
			// Only resources are supported.
			EditorPropertyResource *editor = memnew(EditorPropertyResource);
			editor->setup(_get_edited_param().ptr(), SNAME("saved_value"), "Resource");
			value_editor = editor;
			is_bottom = true;
		} break;
		case Variant::DICTIONARY: {
			value_editor = memnew(EditorPropertyDictionary);
			is_bottom = true;
		} break;

		case Variant::ARRAY:
		case Variant::PACKED_BYTE_ARRAY:
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
		case Variant::PACKED_STRING_ARRAY:
		case Variant::PACKED_VECTOR2_ARRAY:
		case Variant::PACKED_VECTOR3_ARRAY:
		case Variant::PACKED_COLOR_ARRAY: {
			EditorPropertyArray *editor = memnew(EditorPropertyArray);
			editor->setup(p_type);
			value_editor = editor;
			is_bottom = true;
		} break;

		default: {
			ERR_PRINT("Unexpected variant type!");
			value_editor = memnew(EditorPropertyNil);
		}
	}
	value_editor->set_name_split_ratio(0.0);
	value_editor->set_use_folding(is_using_folding());
	value_editor->set_selectable(false);
	value_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	value_editor->set_meta(SNAME("_param_type"), p_type);
	value_editor->connect(SNAME("property_changed"), callable_mp(this, &EditorPropertyBBParam::_value_edited));
	if (is_bottom) {
		bottom_container->add_child(value_editor);
		set_bottom_editor(bottom_container);
		bottom_container->show();
	} else {
		set_bottom_editor(nullptr);
		editor_hbox->add_child(value_editor);
		bottom_container->hide();
	}
}

void EditorPropertyBBParam::_remove_value_editor() {
	if (value_editor) {
		value_editor->get_parent()->remove_child(value_editor);
		value_editor->queue_free();
		value_editor = nullptr;
	}
}

void EditorPropertyBBParam::_value_edited(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	_get_edited_param()->set_saved_value(p_value);
}

void EditorPropertyBBParam::_type_selected(int p_index) {
	Ref<BBParam> param = _get_edited_param();
	ERR_FAIL_COND(param.is_null());
	if (p_index == ID_BIND_VAR) {
		param->set_value_source(BBParam::BLACKBOARD_VAR);
	} else {
		param->set_value_source(BBParam::SAVED_VALUE);
		Ref<BBVariant> variant_param = param;
		if (variant_param.is_valid()) {
			variant_param->set_type(Variant::Type(p_index));
		}
	}
	update_property();
}

void EditorPropertyBBParam::_variable_edited(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	_get_edited_param()->set_variable(p_value);
}

void EditorPropertyBBParam::update_property() {
	if (!initialized) {
		// Initialize UI -- needed after https://github.com/godotengine/godot/commit/db7175458a0532f1efe733f303ad2b55a02a52a5
		_notification(NOTIFICATION_THEME_CHANGED);
	}

	Ref<BBParam> param = _get_edited_param();

	if (param->get_value_source() == BBParam::BLACKBOARD_VAR) {
		_remove_value_editor();
		variable_editor->set_object_and_property(param.ptr(), SNAME("variable"));
		variable_editor->setup(plan, false, param->get_variable_expected_type());
		variable_editor->update_property();
		variable_editor->show();
		bottom_container->hide();
		type_choice->set_icon(get_editor_theme_icon(SNAME("LimboExtraVariable")));
	} else {
		_create_value_editor(param->get_type());
		variable_editor->hide();
		value_editor->show();
		value_editor->set_object_and_property(param.ptr(), SNAME("saved_value"));
		value_editor->update_property();
		type_choice->set_icon(get_editor_theme_icon(Variant::get_type_name(param->get_type())));
	}
}

void EditorPropertyBBParam::setup(PropertyHint p_hint, const String &p_hint_text, const Ref<BlackboardPlan> &p_plan) {
	param_type = p_hint_text;
	property_hint = p_hint;
	plan = p_plan;
	variable_editor->set_name_split_ratio(0.0);
}

void EditorPropertyBBParam::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			if (!get_edited_object()) {
				// Null check needed after https://github.com/godotengine/godot/commit/db7175458a0532f1efe733f303ad2b55a02a52a5
				return;
			}

			{
				String type = Variant::get_type_name(_get_edited_param()->get_type());
				type_choice->set_icon(get_editor_theme_icon(type));
			}

			// Initialize type choice.
			PopupMenu *type_menu = type_choice->get_popup();
			type_menu->clear();
			type_menu->add_icon_item(get_editor_theme_icon(SNAME("LimboExtraVariable")), TTR("Blackboard Variable"), ID_BIND_VAR);
			type_menu->add_separator();
			Ref<BBParam> param = _get_edited_param();
			bool is_variant_param = param->is_class_ptr(BBVariant::get_class_ptr_static());
			if (is_variant_param) {
				for (int i = 0; i < Variant::VARIANT_MAX; i++) {
					if (i == Variant::RID || i == Variant::CALLABLE || i == Variant::SIGNAL) {
						continue;
					}
					String type = Variant::get_type_name(Variant::Type(i));
					type_menu->add_icon_item(get_editor_theme_icon(type), type, i);
				}
			} else { // Not a variant param.
				String type = Variant::get_type_name(param->get_type());
				type_menu->add_icon_item(get_editor_theme_icon(type), type, param->get_type());
			}

			initialized = true;
		} break;
	}
}

EditorPropertyBBParam::EditorPropertyBBParam() {
	hbox = memnew(HBoxContainer);
	add_child(hbox);
	hbox->add_theme_constant_override(LW_NAME(separation), 0);

	bottom_container = memnew(MarginContainer);
	bottom_container->set_theme_type_variation("MarginContainer4px");
	add_child(bottom_container);

	type_choice = memnew(MenuButton);
	hbox->add_child(type_choice);
	type_choice->get_popup()->connect(LW_NAME(id_pressed), callable_mp(this, &EditorPropertyBBParam::_type_selected));
	type_choice->set_tooltip_text(TTR("Click to choose type"));
	type_choice->set_flat(false);

	editor_hbox = memnew(HBoxContainer);
	hbox->add_child(editor_hbox);
	editor_hbox->set_h_size_flags(SIZE_EXPAND_FILL);
	editor_hbox->add_theme_constant_override(LW_NAME(separation), 0);

	variable_editor = memnew(EditorPropertyVariableName);
	editor_hbox->add_child(variable_editor);
	variable_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	variable_editor->connect(SNAME("property_changed"), callable_mp(this, &EditorPropertyBBParam::_variable_edited));

	param_type = SNAME("BBString");
}

//***** EditorInspectorPluginBBParam

bool EditorInspectorPluginBBParam::can_handle(Object *p_object) {
	return true; // Handles everything.
}

bool EditorInspectorPluginBBParam::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide) {
	if (p_hint == PROPERTY_HINT_RESOURCE_TYPE && p_hint_text.begins_with("BB")) {
		// TODO: Add more rigid hint check.
		EditorPropertyBBParam *editor = memnew(EditorPropertyBBParam());
		editor->setup(p_hint, p_hint_text, plan_getter.call());
		add_property_editor(p_path, editor);
		return true;
	}
	return false;
}

#endif // ! LIMBOAI_MODULE

#endif // ! TOOLS_ENABLED
