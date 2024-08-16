/**
 * limbo_compat.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "limbo_compat.h"

#ifdef LIMBOAI_MODULE

#ifdef TOOLS_ENABLED
#include "core/io/resource.h"
#include "core/variant/variant.h"
#include "editor/editor_node.h"
#include "editor/plugins/script_editor_plugin.h"
#endif // TOOLS_ENABLED

#endif // ! LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION

#include "godot_cpp/classes/editor_interface.hpp"
#include "godot_cpp/core/error_macros.hpp"
#include "godot_cpp/variant/typed_array.hpp"
#include <godot_cpp/classes/editor_settings.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/classes/script_editor.hpp>
#include <godot_cpp/classes/script_editor_base.hpp>
#include <godot_cpp/classes/translation_server.hpp>

using namespace godot;

Variant _EDITOR_GET(const String &p_setting) {
	Ref<EditorSettings> es = EditorInterface::get_singleton()->get_editor_settings();
	ERR_FAIL_COND_V(es.is_null() || !es->has_setting(p_setting), Variant());
	return es->get(p_setting);
}

String TTR(const String &p_text, const String &p_context) {
	if (TranslationServer::get_singleton()) {
		return TranslationServer::get_singleton()->translate(p_text, p_context);
	}

	return p_text;
}

Variant _GLOBAL_DEF(const String &p_var, const Variant &p_default, bool p_restart_if_changed, bool p_ignore_value_in_docs, bool p_basic, bool p_internal) {
	Variant ret;
	if (!ProjectSettings::get_singleton()->has_setting(p_var)) {
		ProjectSettings::get_singleton()->set(p_var, p_default);
	}
	ret = GLOBAL_GET(p_var);

	ProjectSettings::get_singleton()->set_initial_value(p_var, p_default);
	// ProjectSettings::get_singleton()->set_builtin_order(p_var);
	ProjectSettings::get_singleton()->set_as_basic(p_var, p_basic);
	ProjectSettings::get_singleton()->set_restart_if_changed(p_var, p_restart_if_changed);
	// ProjectSettings::get_singleton()->set_ignore_value_in_docs(p_var, p_ignore_value_in_docs);
	ProjectSettings::get_singleton()->set_as_internal(p_var, p_internal);
	return ret;
}

Variant _GLOBAL_DEF(const PropertyInfo &p_info, const Variant &p_default, bool p_restart_if_changed, bool p_ignore_value_in_docs, bool p_basic, bool p_internal) {
	Variant ret = _GLOBAL_DEF(p_info.name, p_default, p_restart_if_changed, p_ignore_value_in_docs, p_basic, p_internal);

	Dictionary dic_info;
	dic_info["type"] = p_info.type;
	dic_info["name"] = p_info.name;
	dic_info["class_name"] = p_info.class_name;
	dic_info["hint"] = p_info.hint;
	dic_info["hint_string"] = p_info.hint_string;
	dic_info["usage"] = p_info.usage;

	ProjectSettings::get_singleton()->add_property_info(dic_info);
	return ret;
}

#endif // ! LIMBOAI_GDEXTENSION

// **** Shared

Variant VARIANT_DEFAULT(Variant::Type p_type) {
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

#ifdef TOOLS_ENABLED

void SHOW_BUILTIN_DOC(const String &p_topic) {
#ifdef LIMBOAI_MODULE
	ScriptEditor::get_singleton()->goto_help(p_topic);
	EditorNode::get_singleton()->set_visible_editor(EditorNode::EDITOR_SCRIPT);
#elif LIMBOAI_GDEXTENSION
	TypedArray<ScriptEditorBase> open_editors = EditorInterface::get_singleton()->get_script_editor()->get_open_script_editors();
	ERR_FAIL_COND_MSG(open_editors.size() == 0, "Can't open help page. Need at least one script open in the script editor.");
	ScriptEditorBase *seb = Object::cast_to<ScriptEditorBase>(open_editors.front());
	ERR_FAIL_NULL(seb);
	seb->emit_signal("go_to_help", p_topic);
#endif
}

void EDIT_SCRIPT(const String &p_path) {
#ifdef LIMBOAI_MODULE
	Ref<Resource> res = ScriptEditor::get_singleton()->open_file(p_path);
	ERR_FAIL_COND_MSG(res.is_null(), "Failed to load script: " + p_path);
	EditorNode::get_singleton()->edit_resource(res);
#elif LIMBOAI_GDEXTENSION
	Ref<Script> res = RESOURCE_LOAD(p_path, "Script");
	ERR_FAIL_COND_MSG(res.is_null(), "Failed to load script: " + p_path);
	EditorInterface::get_singleton()->edit_script(res);
	EditorInterface::get_singleton()->set_main_screen_editor("Script");
#endif
}

#endif // ! TOOLS_ENABLED
