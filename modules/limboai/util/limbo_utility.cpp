/**
 * limbo_utility.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "limbo_utility.h"

#include "../bt/tasks/bt_task.h"
#include "../util/limbo_compat.h"
#include "limboai_version.h"

#ifdef LIMBOAI_MODULE
#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "core/variant/variant.h"
#include "scene/resources/texture.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif // TOOLS_ENABLED

#endif // ! LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include "godot_cpp/classes/input_event_key.hpp"
#include "godot_cpp/classes/project_settings.hpp"
#include "godot_cpp/variant/dictionary.hpp"
#include "godot_cpp/variant/utility_functions.hpp"
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/theme.hpp>
#include <godot_cpp/core/error_macros.hpp>
#endif // ! LIMBOAI_GDEXTENSION

LimboUtility *LimboUtility::singleton = nullptr;

LimboUtility *LimboUtility::get_singleton() {
	return singleton;
}

String LimboUtility::decorate_var(String p_variable) const {
	String var = p_variable.trim_prefix("$").trim_prefix("\"").trim_suffix("\"");
	if (var.find(" ") == -1 && !var.is_empty()) {
		return vformat("$%s", var);
	} else {
		return vformat("$\"%s\"", var);
	}
}

String LimboUtility::decorate_output_var(String p_variable) const {
	return LW_NAME(output_var_prefix) + decorate_var(p_variable);
}

String LimboUtility::get_status_name(int p_status) const {
	switch (p_status) {
		case BTTask::FRESH:
			return "FRESH";
		case BTTask::RUNNING:
			return "RUNNING";
		case BTTask::FAILURE:
			return "FAILURE";
		case BTTask::SUCCESS:
			return "SUCCESS";
		default:
			return "";
	}
}

Ref<Texture2D> LimboUtility::get_task_icon(String p_class_or_script_path) const {
	ERR_FAIL_COND_V_MSG(p_class_or_script_path.is_empty(), Variant(), "BTTask: script path or class cannot be empty.");

#if defined(TOOLS_ENABLED) && defined(LIMBOAI_MODULE)
	// * Using editor theme
	if (Engine::get_singleton()->is_editor_hint()) {
		Ref<Theme> theme = EditorNode::get_singleton()->get_editor_theme();
		ERR_FAIL_COND_V(theme.is_null(), nullptr);

		if (p_class_or_script_path.begins_with("res:")) {
			Ref<Script> s = ResourceLoader::load(p_class_or_script_path, "Script");
			if (s.is_null()) {
				return theme->get_icon(SNAME("FileBroken"), SNAME("EditorIcons"));
			}

			EditorData &ed = EditorNode::get_editor_data();
			Ref<Texture2D> script_icon = ed.get_script_icon(s);
			if (script_icon.is_valid()) {
				return script_icon;
			}

			StringName base_type = s->get_instance_base_type();
			if (theme->has_icon(base_type, SNAME("EditorIcons"))) {
				return theme->get_icon(base_type, SNAME("EditorIcons"));
			}
		}

		if (theme->has_icon(p_class_or_script_path, SNAME("EditorIcons"))) {
			return theme->get_icon(p_class_or_script_path, SNAME("EditorIcons"));
		}

		// Use an icon of one of the base classes: look up max 3 parents.
		StringName class_name = p_class_or_script_path;
		for (int i = 0; i < 3; i++) {
			class_name = ClassDB::get_parent_class(class_name);
			if (theme->has_icon(class_name, SNAME("EditorIcons"))) {
				return theme->get_icon(class_name, SNAME("EditorIcons"));
			}
		}

		// Return generic resource icon as a fallback.
		return theme->get_icon(SNAME("Resource"), SNAME("EditorIcons"));
	}
#endif // ! TOOLS_ENABLED && LIMBOAI_MODULE

	String path;

	if (p_class_or_script_path.begins_with("res://")) {
		TypedArray<Dictionary> classes = ProjectSettings::get_singleton()->get_global_class_list();
		for (int i = 0; i < classes.size(); i++) {
			if (classes[i].get("path") == p_class_or_script_path) {
				path = classes[i].get("icon");
				break;
			}
		}
		if (path.is_empty()) {
			Ref<Script> sc = RESOURCE_LOAD(p_class_or_script_path, "Script");
			if (sc.is_valid()) {
				path = "res://addons/limboai/icons/" + sc->get_instance_base_type() + ".svg";
			}
		}
	} else {
		// Trying addon icons
		path = "res://addons/limboai/icons/" + p_class_or_script_path + ".svg";
	}

	if (RESOURCE_EXISTS(path, "Texture2D")) {
		Ref<Texture2D> icon = RESOURCE_LOAD(path, "Texture2D");
		return icon;
	}

	return nullptr;
}

String LimboUtility::get_check_operator_string(CheckType p_check_type) const {
	switch (p_check_type) {
		case LimboUtility::CheckType::CHECK_EQUAL: {
			return "==";
		} break;
		case LimboUtility::CheckType::CHECK_LESS_THAN: {
			return "<";
		} break;
		case LimboUtility::CheckType::CHECK_LESS_THAN_OR_EQUAL: {
			return "<=";
		} break;
		case LimboUtility::CheckType::CHECK_GREATER_THAN: {
			return ">";
		} break;
		case LimboUtility::CheckType::CHECK_GREATER_THAN_OR_EQUAL: {
			return ">=";
		} break;
		case LimboUtility::CheckType::CHECK_NOT_EQUAL: {
			return "!=";
		} break;
		default: {
			return "?";
		} break;
	}
}

bool LimboUtility::perform_check(CheckType p_check_type, const Variant &left_value, const Variant &right_value) {
	Variant ret;
	switch (p_check_type) {
		case LimboUtility::CheckType::CHECK_EQUAL: {
			VARIANT_EVALUATE(Variant::OP_EQUAL, left_value, right_value, ret);
		} break;
		case LimboUtility::CheckType::CHECK_LESS_THAN: {
			VARIANT_EVALUATE(Variant::OP_LESS, left_value, right_value, ret);
		} break;
		case LimboUtility::CheckType::CHECK_LESS_THAN_OR_EQUAL: {
			VARIANT_EVALUATE(Variant::OP_LESS_EQUAL, left_value, right_value, ret);
		} break;
		case LimboUtility::CheckType::CHECK_GREATER_THAN: {
			VARIANT_EVALUATE(Variant::OP_GREATER, left_value, right_value, ret);
		} break;
		case LimboUtility::CheckType::CHECK_GREATER_THAN_OR_EQUAL: {
			VARIANT_EVALUATE(Variant::OP_GREATER_EQUAL, left_value, right_value, ret);
		} break;
		case LimboUtility::CheckType::CHECK_NOT_EQUAL: {
			VARIANT_EVALUATE(Variant::OP_NOT_EQUAL, left_value, right_value, ret);
		} break;
		default: {
			return false;
		} break;
	}

	return ret;
}

String LimboUtility::get_operation_string(Operation p_operation) const {
	switch (p_operation) {
		case OPERATION_NONE: {
			return "";
		} break;
		case OPERATION_ADDITION: {
			return "+";
		} break;
		case OPERATION_SUBTRACTION: {
			return "-";
		} break;
		case OPERATION_MULTIPLICATION: {
			return "*";
		} break;
		case OPERATION_DIVISION: {
			return "/";
		} break;
		case OPERATION_MODULO: {
			return "%";
		} break;
		case OPERATION_POWER: {
			return "**";
		} break;
		case OPERATION_BIT_SHIFT_LEFT: {
			return "<<";
		} break;
		case OPERATION_BIT_SHIFT_RIGHT: {
			return ">>";
		} break;
		case OPERATION_BIT_AND: {
			return "&";
		} break;
		case OPERATION_BIT_OR: {
			return "|";
		} break;
		case OPERATION_BIT_XOR: {
			return "^";
		} break;
	}
	return "";
}

Variant LimboUtility::perform_operation(Operation p_operation, const Variant &left_value, const Variant &right_value) {
	Variant ret;
	switch (p_operation) {
		case OPERATION_NONE: {
			ret = right_value;
		} break;
		case OPERATION_ADDITION: {
			VARIANT_EVALUATE(Variant::OP_ADD, left_value, right_value, ret);
		} break;
		case OPERATION_SUBTRACTION: {
			VARIANT_EVALUATE(Variant::OP_SUBTRACT, left_value, right_value, ret);
		} break;
		case OPERATION_MULTIPLICATION: {
			VARIANT_EVALUATE(Variant::OP_MULTIPLY, left_value, right_value, ret);
		} break;
		case OPERATION_DIVISION: {
			VARIANT_EVALUATE(Variant::OP_DIVIDE, left_value, right_value, ret);
		} break;
		case OPERATION_MODULO: {
			VARIANT_EVALUATE(Variant::OP_MODULE, left_value, right_value, ret);
		} break;
		case OPERATION_POWER: {
// TODO: Fix when godot-cpp https://github.com/godotengine/godot-cpp/issues/1348 is resolved.
#ifdef LIMBOAI_MODULE
			VARIANT_EVALUATE(Variant::OP_POWER, left_value, right_value, ret);
#elif LIMBOAI_GDEXTENSION
			ERR_PRINT("LimboUtility: Operation POWER is not available due to https://github.com/godotengine/godot-cpp/issues/1348");
			ret = left_value;
#endif
		} break;
		case OPERATION_BIT_SHIFT_LEFT: {
			VARIANT_EVALUATE(Variant::OP_SHIFT_LEFT, left_value, right_value, ret);
		} break;
		case OPERATION_BIT_SHIFT_RIGHT: {
			VARIANT_EVALUATE(Variant::OP_SHIFT_RIGHT, left_value, right_value, ret);
		} break;
		case OPERATION_BIT_AND: {
			VARIANT_EVALUATE(Variant::OP_BIT_AND, left_value, right_value, ret);
		} break;
		case OPERATION_BIT_OR: {
			VARIANT_EVALUATE(Variant::OP_BIT_OR, left_value, right_value, ret);
		} break;
		case OPERATION_BIT_XOR: {
			VARIANT_EVALUATE(Variant::OP_BIT_XOR, left_value, right_value, ret);
		} break;
	}
	return ret;
}

String LimboUtility::get_property_hint_text(PropertyHint p_hint) const {
	switch (p_hint) {
		case PROPERTY_HINT_NONE: {
			return "NONE";
		}
		case PROPERTY_HINT_RANGE: {
			return "RANGE";
		}
		case PROPERTY_HINT_ENUM: {
			return "ENUM";
		}
		case PROPERTY_HINT_ENUM_SUGGESTION: {
			return "ENUM_SUGGESTION";
		}
		case PROPERTY_HINT_EXP_EASING: {
			return "EXP_EASING";
		}
		case PROPERTY_HINT_LINK: {
			return "LINK";
		}
		case PROPERTY_HINT_FLAGS: {
			return "FLAGS";
		}
		case PROPERTY_HINT_LAYERS_2D_RENDER: {
			return "LAYERS_2D_RENDER";
		}
		case PROPERTY_HINT_LAYERS_2D_PHYSICS: {
			return "LAYERS_2D_PHYSICS";
		}
		case PROPERTY_HINT_LAYERS_2D_NAVIGATION: {
			return "LAYERS_2D_NAVIGATION";
		}
		case PROPERTY_HINT_LAYERS_3D_RENDER: {
			return "LAYERS_3D_RENDER";
		}
		case PROPERTY_HINT_LAYERS_3D_PHYSICS: {
			return "LAYERS_3D_PHYSICS";
		}
		case PROPERTY_HINT_LAYERS_3D_NAVIGATION: {
			return "LAYERS_3D_NAVIGATION";
		}
		case PROPERTY_HINT_FILE: {
			return "FILE";
		}
		case PROPERTY_HINT_DIR: {
			return "DIR";
		}
		case PROPERTY_HINT_GLOBAL_FILE: {
			return "GLOBAL_FILE";
		}
		case PROPERTY_HINT_GLOBAL_DIR: {
			return "GLOBAL_DIR";
		}
		case PROPERTY_HINT_RESOURCE_TYPE: {
			return "RESOURCE_TYPE";
		}
		case PROPERTY_HINT_MULTILINE_TEXT: {
			return "MULTILINE_TEXT";
		}
		case PROPERTY_HINT_EXPRESSION: {
			return "EXPRESSION";
		}
		case PROPERTY_HINT_PLACEHOLDER_TEXT: {
			return "PLACEHOLDER_TEXT";
		}
		case PROPERTY_HINT_COLOR_NO_ALPHA: {
			return "COLOR_NO_ALPHA";
		}
		case PROPERTY_HINT_OBJECT_ID: {
			return "OBJECT_ID";
		}
		case PROPERTY_HINT_TYPE_STRING: {
			return "TYPE_STRING";
		}
		case PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE: {
			return "NODE_PATH_TO_EDITED_NODE";
		}
		case PROPERTY_HINT_OBJECT_TOO_BIG: {
			return "OBJECT_TOO_BIG";
		}
		case PROPERTY_HINT_NODE_PATH_VALID_TYPES: {
			return "NODE_PATH_VALID_TYPES";
		}
		case PROPERTY_HINT_SAVE_FILE: {
			return "SAVE_FILE";
		}
		case PROPERTY_HINT_GLOBAL_SAVE_FILE: {
			return "GLOBAL_SAVE_FILE";
		}
		case PROPERTY_HINT_INT_IS_OBJECTID: {
			return "INT_IS_OBJECTID";
		}
		case PROPERTY_HINT_INT_IS_POINTER: {
			return "INT_IS_POINTER";
		}
		case PROPERTY_HINT_ARRAY_TYPE: {
			return "ARRAY_TYPE";
		}
		case PROPERTY_HINT_LOCALE_ID: {
			return "LOCALE_ID";
		}
		case PROPERTY_HINT_LOCALIZABLE_STRING: {
			return "LOCALIZABLE_STRING";
		}
		case PROPERTY_HINT_NODE_TYPE: {
			return "NODE_TYPE";
		}
		case PROPERTY_HINT_HIDE_QUATERNION_EDIT: {
			return "HIDE_QUATERNION_EDIT";
		}
		case PROPERTY_HINT_PASSWORD: {
			return "PASSWORD";
		}
		case PROPERTY_HINT_LAYERS_AVOIDANCE: {
			return "LAYERS_AVOIDANCE";
		}
		case PROPERTY_HINT_MAX: {
			return "MAX";
		}
	}
	return "";
}

PackedInt32Array LimboUtility::get_property_hints_allowed_for_type(Variant::Type p_type) const {
	PackedInt32Array hints;
	hints.append(PROPERTY_HINT_NONE);

	// * According to editor/editor_properties.cpp
	switch (p_type) {
		case Variant::Type::NIL:
		case Variant::Type::RID:
		case Variant::Type::CALLABLE:
		case Variant::Type::SIGNAL:
		case Variant::Type::BOOL: {
		} break;
		case Variant::Type::INT: {
			hints.append(PROPERTY_HINT_RANGE);
			hints.append(PROPERTY_HINT_ENUM);
			hints.append(PROPERTY_HINT_FLAGS);
			hints.append(PROPERTY_HINT_LAYERS_2D_RENDER);
			hints.append(PROPERTY_HINT_LAYERS_2D_PHYSICS);
			hints.append(PROPERTY_HINT_LAYERS_2D_NAVIGATION);
			hints.append(PROPERTY_HINT_LAYERS_3D_RENDER);
			hints.append(PROPERTY_HINT_LAYERS_3D_PHYSICS);
			hints.append(PROPERTY_HINT_LAYERS_3D_NAVIGATION);
			hints.append(PROPERTY_HINT_LAYERS_AVOIDANCE);
		} break;
		case Variant::Type::FLOAT: {
			hints.append(PROPERTY_HINT_RANGE);
			hints.append(PROPERTY_HINT_EXP_EASING);
		} break;
		case Variant::Type::STRING: {
			hints.append(PROPERTY_HINT_ENUM);
			hints.append(PROPERTY_HINT_ENUM_SUGGESTION);
			hints.append(PROPERTY_HINT_FILE);
			hints.append(PROPERTY_HINT_DIR);
			hints.append(PROPERTY_HINT_GLOBAL_FILE);
			hints.append(PROPERTY_HINT_GLOBAL_DIR);
			hints.append(PROPERTY_HINT_MULTILINE_TEXT);
			hints.append(PROPERTY_HINT_EXPRESSION);
			hints.append(PROPERTY_HINT_PLACEHOLDER_TEXT);
			// hints.append(PROPERTY_HINT_TYPE_STRING); // ! Causes a crash.
			hints.append(PROPERTY_HINT_SAVE_FILE);
			hints.append(PROPERTY_HINT_GLOBAL_SAVE_FILE);
			hints.append(PROPERTY_HINT_LOCALE_ID);
			hints.append(PROPERTY_HINT_PASSWORD);
		} break;
		case Variant::Type::VECTOR2:
		case Variant::Type::VECTOR2I:
		case Variant::Type::VECTOR3:
		case Variant::Type::VECTOR3I:
		case Variant::Type::VECTOR4:
		case Variant::Type::VECTOR4I: {
			hints.append(PROPERTY_HINT_RANGE);
			hints.append(PROPERTY_HINT_LINK);
		} break;
		case Variant::Type::RECT2:
		case Variant::Type::RECT2I:
		case Variant::Type::TRANSFORM2D:
		case Variant::Type::PLANE:
		case Variant::Type::AABB:
		case Variant::Type::BASIS:
		case Variant::Type::TRANSFORM3D:
		case Variant::Type::PROJECTION: {
			hints.append(PROPERTY_HINT_RANGE);
		} break;
		case Variant::Type::QUATERNION: {
			hints.append(PROPERTY_HINT_RANGE);
			hints.append(PROPERTY_HINT_HIDE_QUATERNION_EDIT);
		} break;
		case Variant::Type::COLOR: {
			hints.append(PROPERTY_HINT_COLOR_NO_ALPHA);
		} break;
		case Variant::Type::STRING_NAME: {
			hints.append(PROPERTY_HINT_ENUM);
			hints.append(PROPERTY_HINT_ENUM_SUGGESTION);
			hints.append(PROPERTY_HINT_PLACEHOLDER_TEXT);
			hints.append(PROPERTY_HINT_PASSWORD);
		} break;
		case Variant::Type::NODE_PATH: {
			hints.append(PROPERTY_HINT_NODE_PATH_VALID_TYPES);
			hints.append(PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT);

		} break;
		case Variant::Type::OBJECT: {
			hints.append(PROPERTY_HINT_RESOURCE_TYPE);
			hints.append(PROPERTY_HINT_NODE_TYPE);
		} break;
		case Variant::Type::DICTIONARY: {
			hints.append(PROPERTY_HINT_LOCALIZABLE_STRING);
		} break;
		case Variant::Type::ARRAY:
		case Variant::Type::PACKED_BYTE_ARRAY:
		case Variant::Type::PACKED_INT32_ARRAY:
		case Variant::Type::PACKED_INT64_ARRAY:
		case Variant::Type::PACKED_FLOAT32_ARRAY:
		case Variant::Type::PACKED_FLOAT64_ARRAY:
		case Variant::Type::PACKED_STRING_ARRAY:
		case Variant::Type::PACKED_VECTOR2_ARRAY:
		case Variant::Type::PACKED_VECTOR3_ARRAY:
		case Variant::Type::PACKED_VECTOR4_ARRAY:
		case Variant::Type::PACKED_COLOR_ARRAY:
		case Variant::Type::VARIANT_MAX: {
		} break;
	}
	return hints;
}

#ifdef TOOLS_ENABLED

Ref<Shortcut> LimboUtility::add_shortcut(const String &p_path, const String &p_name, Key p_keycode) {
	Ref<Shortcut> sc = memnew(Shortcut);
	sc->set_name(p_name);

	Array events;

	Key keycode = p_keycode;
	Ref<InputEventKey> ev = memnew(InputEventKey);
	if (((int)LW_KEY_MASK(CMD_OR_CTRL) & (int)keycode) == (int)LW_KEY_MASK(CMD_OR_CTRL)) {
		keycode = (Key)((int)keycode & (~((int)LW_KEY_MASK(CMD_OR_CTRL))));
		ev->set_ctrl_pressed(true);
	}
	if (((int)LW_KEY_MASK(ALT) & (int)keycode) == (int)LW_KEY_MASK(ALT)) {
		keycode = (Key)((int)keycode & (~((int)LW_KEY_MASK(ALT))));
		ev->set_alt_pressed(true);
	}
	if (((int)LW_KEY_MASK(SHIFT) & (int)keycode) == (int)LW_KEY_MASK(SHIFT)) {
		keycode = (Key)((int)keycode & (~((int)LW_KEY_MASK(SHIFT))));
		ev->set_shift_pressed(true);
	}
	ev->set_keycode(keycode);
	ev->set_pressed(true);

	events.append(ev);
	sc->set_events(events);
	shortcuts[p_path] = sc;
	return sc;
}

bool LimboUtility::is_shortcut(const String &p_path, const Ref<InputEvent> &p_event) const {
	HashMap<String, Ref<Shortcut>>::ConstIterator E = shortcuts.find(p_path);
	ERR_FAIL_COND_V_MSG(!E, false, vformat("LimboUtility: Shortcut not found: %s.", p_path));
	return E->value->matches_event(p_event);
}

Ref<Shortcut> LimboUtility::get_shortcut(const String &p_path) const {
	HashMap<String, Ref<Shortcut>>::ConstIterator SC = shortcuts.find(p_path);
	if (SC) {
		return SC->value;
	}
	return nullptr;
}

void LimboUtility::open_doc_introduction() {
	OS::get_singleton()->shell_open(vformat("%s/getting-started/introduction.html", LIMBOAI_VERSION_DOC_URL));
}

void LimboUtility::open_doc_online() {
	OS::get_singleton()->shell_open(vformat("%s/index.html", LIMBOAI_VERSION_DOC_URL));
}

void LimboUtility::open_doc_gdextension_limitations() {
	OS::get_singleton()->shell_open(vformat("%s/getting-started/gdextension.html#limitations-of-the-gdextension-version", LIMBOAI_VERSION_DOC_URL));
}

void LimboUtility::open_doc_custom_tasks() {
	OS::get_singleton()->shell_open(vformat("%s/getting-started/custom-tasks.html", LIMBOAI_VERSION_DOC_URL));
}

void LimboUtility::open_doc_class(const String &p_class_name) {
	if (p_class_name.begins_with("res://")) {
		SHOW_DOC(vformat("class_name:\"%s\"", p_class_name));
		return;
	}

#ifdef LIMBOAI_MODULE
	SHOW_DOC("class_name:" + p_class_name);
#elif LIMBOAI_GDEXTENSION
	OS::get_singleton()->shell_open(vformat("%s/classes/class_%s.html", LIMBOAI_VERSION_DOC_URL, p_class_name.to_lower()));
#endif
}

#endif // ! TOOLS_ENABLED

void LimboUtility::_bind_methods() {
	ClassDB::bind_method(D_METHOD("decorate_var", "variable"), &LimboUtility::decorate_var);
	ClassDB::bind_method(D_METHOD("decorate_output_var", "variable"), &LimboUtility::decorate_output_var);
	ClassDB::bind_method(D_METHOD("get_status_name", "status"), &LimboUtility::get_status_name);
	ClassDB::bind_method(D_METHOD("get_task_icon", "class_or_script_path"), &LimboUtility::get_task_icon);
	ClassDB::bind_method(D_METHOD("get_check_operator_string", "check"), &LimboUtility::get_check_operator_string);
	ClassDB::bind_method(D_METHOD("perform_check", "check", "a", "b"), &LimboUtility::perform_check);
	ClassDB::bind_method(D_METHOD("get_operation_string", "operation"), &LimboUtility::get_operation_string);
	ClassDB::bind_method(D_METHOD("perform_operation", "operation", "a", "b"), &LimboUtility::perform_operation);

	BIND_ENUM_CONSTANT(CHECK_EQUAL);
	BIND_ENUM_CONSTANT(CHECK_LESS_THAN);
	BIND_ENUM_CONSTANT(CHECK_LESS_THAN_OR_EQUAL);
	BIND_ENUM_CONSTANT(CHECK_GREATER_THAN);
	BIND_ENUM_CONSTANT(CHECK_GREATER_THAN_OR_EQUAL);
	BIND_ENUM_CONSTANT(CHECK_NOT_EQUAL);

	BIND_ENUM_CONSTANT(OPERATION_NONE);
	BIND_ENUM_CONSTANT(OPERATION_ADDITION);
	BIND_ENUM_CONSTANT(OPERATION_SUBTRACTION);
	BIND_ENUM_CONSTANT(OPERATION_MULTIPLICATION);
	BIND_ENUM_CONSTANT(OPERATION_DIVISION);
	BIND_ENUM_CONSTANT(OPERATION_MODULO);
	BIND_ENUM_CONSTANT(OPERATION_POWER);
	BIND_ENUM_CONSTANT(OPERATION_BIT_SHIFT_LEFT);
	BIND_ENUM_CONSTANT(OPERATION_BIT_SHIFT_RIGHT);
	BIND_ENUM_CONSTANT(OPERATION_BIT_AND);
	BIND_ENUM_CONSTANT(OPERATION_BIT_OR);
	BIND_ENUM_CONSTANT(OPERATION_BIT_XOR);
}

LimboUtility::LimboUtility() {
	singleton = this;
}

LimboUtility::~LimboUtility() {
	singleton = nullptr;
}
