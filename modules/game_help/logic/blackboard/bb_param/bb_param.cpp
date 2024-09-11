/**
 * bb_param.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bb_param.h"

#if defined(TOOLS_ENABLED)
#include "editor/editor_node.h"
#endif


VARIANT_ENUM_CAST(BBParam::ValueSource);


String BBParam::decorate_var(String p_variable) {
	String var = p_variable.trim_prefix("$").trim_prefix("\"").trim_suffix("\"");
	if (var.find(" ") == -1 && !var.is_empty()) {
		return vformat("$%s", var);
	}
	else {
		return vformat("$\"%s\"", var);
	}
}

Variant BBParam::VARIANT_DEFAULT(Variant::Type p_type) {
	switch (p_type) {
	case Variant::Type::NIL: {
		return Variant();
	} break;
	case Variant::Type::BOOL: {
		return Variant(false);
	} break;
	case Variant::Type::INT: {
		return Variant(0);
	} break;
	case Variant::Type::FLOAT: {
		return Variant(0.0);
	} break;
	case Variant::Type::STRING: {
		return Variant("");
	} break;
	case Variant::Type::VECTOR2: {
		return Variant(Vector2());
	} break;
	case Variant::Type::VECTOR2I: {
		return Variant(Vector2i());
	} break;
	case Variant::Type::RECT2: {
		return Variant(Rect2());
	} break;
	case Variant::Type::RECT2I: {
		return Variant(Rect2i());
	} break;
	case Variant::Type::VECTOR3: {
		return Variant(Vector3());
	} break;
	case Variant::Type::VECTOR3I: {
		return Variant(Vector3i());
	} break;
	case Variant::Type::TRANSFORM2D: {
		return Variant(Transform2D());
	} break;
	case Variant::Type::VECTOR4: {
		return Variant(Vector4());
	} break;
	case Variant::Type::VECTOR4I: {
		return Variant(Vector4i());
	} break;
	case Variant::Type::PLANE: {
		return Variant(Plane());
	} break;
	case Variant::Type::QUATERNION: {
		return Variant(Quaternion());
	} break;
	case Variant::Type::AABB: {
		return Variant(AABB());
	} break;
	case Variant::Type::BASIS: {
		return Variant(Basis());
	} break;
	case Variant::Type::TRANSFORM3D: {
		return Variant(Transform3D());
	} break;
	case Variant::Type::PROJECTION: {
		return Variant(Projection());
	} break;
	case Variant::Type::COLOR: {
		return Variant(Color());
	} break;
	case Variant::Type::STRING_NAME: {
		return Variant(StringName());
	} break;
	case Variant::Type::NODE_PATH: {
		return Variant(NodePath());
	} break;
	case Variant::Type::RID: {
		return Variant(RID());
	} break;
	case Variant::Type::OBJECT: {
		return Variant();
	} break;
	case Variant::Type::CALLABLE: {
		return Variant();
	} break;
	case Variant::Type::SIGNAL: {
		return Variant();
	} break;
	case Variant::Type::DICTIONARY: {
		return Variant(Dictionary());
	} break;
	case Variant::Type::ARRAY: {
		return Variant(Array());
	} break;
	case Variant::Type::PACKED_BYTE_ARRAY: {
		return Variant(PackedByteArray());
	} break;
	case Variant::Type::PACKED_INT32_ARRAY: {
		return Variant(PackedInt32Array());
	} break;
	case Variant::Type::PACKED_INT64_ARRAY: {
		return Variant(PackedInt64Array());
	} break;
	case Variant::Type::PACKED_FLOAT32_ARRAY: {
		return Variant(PackedFloat32Array());
	} break;
	case Variant::Type::PACKED_FLOAT64_ARRAY: {
		return Variant(PackedFloat64Array());
	} break;
	case Variant::Type::PACKED_STRING_ARRAY: {
		return Variant(PackedStringArray());
	} break;
	case Variant::Type::PACKED_VECTOR2_ARRAY: {
		return Variant(PackedVector2Array());
	} break;
	case Variant::Type::PACKED_VECTOR3_ARRAY: {
		return Variant(PackedVector3Array());
	} break;
	case Variant::Type::PACKED_COLOR_ARRAY: {
		return Variant(PackedColorArray());
	} break;
	default: {
		return Variant();
	}
	}
}
	
PackedInt32Array BBParam::get_property_hints_allowed_for_type(Variant::Type p_type) {
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

String BBParam::get_property_hint_text(PropertyHint p_hint) {
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

Ref<Texture2D> BBParam::get_task_icon(String p_class_or_script_path)  {
ERR_FAIL_COND_V_MSG(p_class_or_script_path.is_empty(), Variant(), "BTTask: script path or class cannot be empty.");

#if defined(TOOLS_ENABLED) 
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
#endif
}





void BBParam::set_value_source(ValueSource p_value) {
	value_source = p_value;
	notify_property_list_changed();
	_update_name();
	emit_changed();
}

Variant BBParam::get_saved_value() {
	return saved_value;
}

void BBParam::set_saved_value(Variant p_value) {
	if (p_value.get_type() == Variant::NIL) {
		_assign_default_value();
	} else {
		saved_value = p_value;
	}
	_update_name();
	emit_changed();
}

void BBParam::set_variable(const StringName &p_variable) {
	variable = p_variable;
	_update_name();
	emit_changed();
}

String BBParam::to_string() {
	if (value_source == SAVED_VALUE) {
		String s = saved_value.stringify();
		switch (get_type()) {
			case Variant::STRING: {
				s = "\"" + s.c_escape() + "\"";
			} break;
			case Variant::STRING_NAME: {
				s = "&\"" + s.c_escape() + "\"";
			} break;
			case Variant::NODE_PATH: {
				s = "^\"" + s.c_escape() + "\"";
			} break;
			default: {
			} break;
		}
		return s;
	} else {
		return decorate_var(variable);
	}
}

Variant BBParam::get_value(Node *p_scene_root, const Ref<Blackboard> &p_blackboard, const Variant &p_default) {
	ERR_FAIL_COND_V(!p_blackboard.is_valid(), p_default);

	if (value_source == SAVED_VALUE) {
		if (saved_value == Variant()) {
			_assign_default_value();
		}
		return saved_value;
	} else {
		ERR_FAIL_COND_V_MSG(!p_blackboard->has_var(variable), p_default, vformat("BBParam: Blackboard variable \"%s\" doesn't exist.", variable));
		return p_blackboard->get_var(variable, p_default);
	}
}

void BBParam::_get_property_list(List<PropertyInfo> *p_list) const {
	if (value_source == ValueSource::SAVED_VALUE) {
		p_list->push_back(PropertyInfo(get_type(), "saved_value"));
	} else {
		p_list->push_back(PropertyInfo(Variant::STRING_NAME, "variable"));
	}
}

void BBParam::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_value_source", "value_source"), &BBParam::set_value_source);
	ClassDB::bind_method(D_METHOD("get_value_source"), &BBParam::get_value_source);
	ClassDB::bind_method(D_METHOD("set_saved_value", "value"), &BBParam::set_saved_value);
	ClassDB::bind_method(D_METHOD("get_saved_value"), &BBParam::get_saved_value);
	ClassDB::bind_method(D_METHOD("set_variable", "variable_name"), &BBParam::set_variable);
	ClassDB::bind_method(D_METHOD("get_variable"), &BBParam::get_variable);
	ClassDB::bind_method(D_METHOD("get_type"), &BBParam::get_type);
	ClassDB::bind_method(D_METHOD("get_value", "scene_root", "blackboard", "default"), &BBParam::get_value, Variant());

	ADD_PROPERTY(PropertyInfo(Variant::INT, "value_source", PROPERTY_HINT_ENUM, "Saved Value,Blackboard Var"), "set_value_source", "get_value_source");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "variable", PROPERTY_HINT_NONE, "", 0), "set_variable", "get_variable");
	ADD_PROPERTY(PropertyInfo(Variant::NIL, "saved_value", PROPERTY_HINT_NONE, "", 0), "set_saved_value", "get_saved_value");

	BIND_ENUM_CONSTANT(SAVED_VALUE);
	BIND_ENUM_CONSTANT(BLACKBOARD_VAR);
}

BBParam::BBParam() {
	value_source = SAVED_VALUE;
}
