#/**
 * limbo_compat.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/*
 *	Defines and funcs that help to bridge some differences between GDExtension and Godot APIs.
 *  This helps us writing compatible code with both module and GDExtension.
 */

#ifndef LIMBO_COMPAT_H
#define LIMBO_COMPAT_H

#ifdef LIMBOAI_MODULE

#include "core/object/ref_counted.h"
#include "core/string/print_string.h"

// *** API abstractions: Module edition

#define SCRIPT_EDITOR() (ScriptEditor::get_singleton())
#define EDITOR_FILE_SYSTEM() (EditorFileSystem::get_singleton())
#define EDITOR_SETTINGS() (EditorSettings::get_singleton())
#define BASE_CONTROL() (EditorNode::get_singleton()->get_gui_base())
#define MAIN_SCREEN_CONTROL() (EditorNode::get_singleton()->get_main_screen_control())
#define SCENE_TREE() (SceneTree::get_singleton())
#define IS_DEBUGGER_ACTIVE() (EngineDebugger::is_active())
#define FS_DOCK_SELECT_FILE(m_path) FileSystemDock::get_singleton()->select_file(m_path)

#define PRINT_LINE(...) (print_line(__VA_ARGS__))
#define IS_CLASS(m_obj, m_class) (m_obj->is_class_ptr(m_class::get_class_ptr_static()))
#define RAND_RANGE(m_from, m_to) (Math::random(m_from, m_to))
#define RANDF() (Math::randf())
#define BUTTON_SET_ICON(m_btn, m_icon) m_btn->set_icon(m_icon)
#define RESOURCE_LOAD(m_path, m_hint) ResourceLoader::load(m_path, m_hint)
#define RESOURCE_LOAD_NO_CACHE(m_path, m_hint) ResourceLoader::load(m_path, m_hint, ResourceFormatLoader::CACHE_MODE_IGNORE)
#define RESOURCE_SAVE(m_res, m_path, m_flags) ResourceSaver::save(m_res, m_path, m_flags)
#define RESOURCE_IS_CACHED(m_path) (ResourceCache::has(m_path))
#define RESOURCE_EXISTS(m_path, m_type_hint) (ResourceLoader::exists(m_path, m_type_hint))
#define RESOURCE_IS_SCENE_FILE(m_path) (ResourceLoader::get_resource_type(m_path) == "PackedScene")
#define GET_PROJECT_SETTINGS_DIR() EditorPaths::get_singleton()->get_project_settings_dir()
#define EDIT_RESOURCE(m_res) EditorNode::get_singleton()->edit_resource(m_res)
#define INSPECTOR_GET_EDITED_OBJECT() (InspectorDock::get_inspector_singleton()->get_edited_object())
#define SET_MAIN_SCREEN_EDITOR(m_name) (EditorNode::get_singleton()->select_editor_by_name(m_name))
#define FILE_EXISTS(m_path) FileAccess::exists(m_path)
#define DIR_ACCESS_CREATE() DirAccess::create(DirAccess::ACCESS_RESOURCES)
#define PERFORMANCE_ADD_CUSTOM_MONITOR(m_id, m_callable) (Performance::get_singleton()->add_custom_monitor(m_id, m_callable, Variant()))
#define GET_SCRIPT(m_obj) (m_obj->get_script_instance() ? m_obj->get_script_instance()->get_script() : nullptr)
#define ADD_STYLEBOX_OVERRIDE(m_control, m_name, m_stylebox) (m_control->add_theme_style_override(m_name, m_stylebox))
#define GET_NODE(m_parent, m_path) m_parent->get_node(m_path)

_FORCE_INLINE_ bool OBJECT_HAS_PROPERTY(Object *p_obj, const StringName &p_prop) {
	bool r_valid;
	return Variant(p_obj).has_key(p_prop, r_valid);
}

#define VARIANT_EVALUATE(m_op, m_lvalue, m_rvalue, r_ret) r_ret = Variant::evaluate(m_op, m_lvalue, m_rvalue)

// * Enum

#define LW_KEY(key) (Key::key)
#define LW_KEY_MASK(mask) (KeyModifierMask::mask)
#define LW_MBTN(key) (MouseButton::key)

#endif // ! LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/variant.hpp>

using namespace godot;

// *** API abstractions: GDExtension edition

#define SCRIPT_EDITOR() (EditorInterface::get_singleton()->get_script_editor())
#define EDITOR_FILE_SYSTEM() (EditorInterface::get_singleton()->get_resource_filesystem())
#define EDITOR_SETTINGS() (EditorInterface::get_singleton()->get_editor_settings())
#define BASE_CONTROL() (EditorInterface::get_singleton()->get_base_control())
#define MAIN_SCREEN_CONTROL() (EditorInterface::get_singleton()->get_editor_main_screen())
#define SCENE_TREE() ((SceneTree *)(Engine::get_singleton()->get_main_loop()))
#define IS_DEBUGGER_ACTIVE() (EngineDebugger::get_singleton()->is_active())
#define FS_DOCK_SELECT_FILE(m_path) EditorInterface::get_singleton()->get_file_system_dock()->navigate_to_path(m_path)

#define PRINT_LINE(...) (UtilityFunctions::print(__VA_ARGS__))
// TODO: Use this def if https://github.com/godotengine/godot-cpp/pull/1356 gets merged:
// #define IS_CLASS(m_obj, m_class) (m_obj->is_class_static(m_class::get_class_static()))
#define IS_CLASS(m_obj, m_class) (m_obj->is_class(#m_class))
#define RAND_RANGE(m_from, m_to) (UtilityFunctions::randf_range(m_from, m_to))
#define RANDF() (UtilityFunctions::randf())
#define BUTTON_SET_ICON(m_btn, m_icon) m_btn->set_button_icon(m_icon)
#define RESOURCE_LOAD(m_path, m_hint) ResourceLoader::get_singleton()->load(m_path, m_hint)
#define RESOURCE_LOAD_NO_CACHE(m_path, m_hint) ResourceLoader::get_singleton()->load(m_path, m_hint, ResourceLoader::CACHE_MODE_IGNORE)
#define RESOURCE_SAVE(m_res, m_path, m_flags) ResourceSaver::get_singleton()->save(m_res, m_path, m_flags)
#define RESOURCE_IS_CACHED(m_path) (ResourceLoader::get_singleton()->has_cached(res_path))
#define RESOURCE_IS_SCENE_FILE(m_path) (ResourceLoader::get_singleton()->get_recognized_extensions_for_type("PackedScene").has(m_path.get_extension()))
#define RESOURCE_EXISTS(m_path, m_type_hint) (ResourceLoader::get_singleton()->exists(m_path, m_type_hint))
#define GET_PROJECT_SETTINGS_DIR() EditorInterface::get_singleton()->get_editor_paths()->get_project_settings_dir()
#define EDIT_RESOURCE(m_res) EditorInterface::get_singleton()->edit_resource(m_res)
#define INSPECTOR_GET_EDITED_OBJECT() (EditorInterface::get_singleton()->get_inspector()->get_edited_object())
#define SET_MAIN_SCREEN_EDITOR(m_name) (EditorInterface::get_singleton()->set_main_screen_editor(m_name))
#define FILE_EXISTS(m_path) FileAccess::file_exists(m_path)
#define DIR_ACCESS_CREATE() DirAccess::open("res://")
#define PERFORMANCE_ADD_CUSTOM_MONITOR(m_id, m_callable) (Performance::get_singleton()->add_custom_monitor(m_id, m_callable))
#define GET_SCRIPT(m_obj) (m_obj->get_script())
#define ADD_STYLEBOX_OVERRIDE(m_control, m_name, m_stylebox) (m_control->add_theme_stylebox_override(m_name, m_stylebox))
#define GET_NODE(m_parent, m_path) m_parent->get_node_internal(m_path)

_FORCE_INLINE_ bool OBJECT_HAS_PROPERTY(Object *p_obj, const StringName &p_prop) {
	return Variant(p_obj).has_key(p_prop);
}

#define VARIANT_EVALUATE(m_op, m_lvalue, m_rvalue, r_ret)            \
	{                                                                \
		bool r_valid;                                                \
		Variant::evaluate(m_op, m_lvalue, m_rvalue, r_ret, r_valid); \
	}

// * Enum

#define LW_KEY(key) (Key::KEY_##key)
#define LW_KEY_MASK(mask) (KeyModifierMask::KEY_MASK_##mask)
#define LW_MBTN(key) (MouseButton::MOUSE_BUTTON_##key)

// * Missing defines

#define EDITOR_GET(m_var) _EDITOR_GET(m_var)
Variant _EDITOR_GET(const String &p_setting);

#define GLOBAL_GET(m_var) ProjectSettings::get_singleton()->get_setting_with_override(m_var)

#define GLOBAL_DEF(m_var, m_value) _GLOBAL_DEF(m_var, m_value)
Variant _GLOBAL_DEF(const String &p_var, const Variant &p_default, bool p_restart_if_changed = false, bool p_ignore_value_in_docs = false, bool p_basic = false, bool p_internal = false);
Variant _GLOBAL_DEF(const PropertyInfo &p_info, const Variant &p_default, bool p_restart_if_changed = false, bool p_ignore_value_in_docs = false, bool p_basic = false, bool p_internal = false);

#define EDSCALE (EditorInterface::get_singleton()->get_editor_scale())

String TTR(const String &p_text, const String &p_context = "");

#endif // ! LIMBOAI_GDEXTENSION

// *** API abstractions: Shared

#define VARIANT_IS_ARRAY(m_variant) (m_variant.get_type() >= Variant::ARRAY)
#define VARIANT_IS_NUM(m_variant) (m_variant.get_type() == Variant::INT || m_variant.get_type() == Variant::FLOAT)

inline void VARIANT_DELETE_IF_OBJECT(Variant m_variant) {
	if (m_variant.get_type() == Variant::OBJECT) {
		Ref<RefCounted> r = m_variant;
		if (r.is_null()) {
			memdelete((Object *)m_variant);
		}
	}
}

Variant VARIANT_DEFAULT(Variant::Type p_type);

#define PROJECT_CONFIG_FILE() GET_PROJECT_SETTINGS_DIR().path_join("limbo_ai.cfg")
#define IS_RESOURCE_FILE(m_path) (m_path.begins_with("res://") && m_path.find("::") == -1)
#define RESOURCE_TYPE_HINT(m_type) vformat("%s/%s:%s", Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, m_type)
#define RESOURCE_IS_BUILT_IN(m_res) (m_res->get_path().is_empty() || m_res->get_path().contains("::"))
#define RESOURCE_PATH_IS_BUILT_IN(m_path) (m_path.is_empty() || m_path.contains("::"))

#ifdef TOOLS_ENABLED

void SHOW_DOC(const String &p_topic);
void EDIT_SCRIPT(const String &p_path);

#endif // TOOLS_ENABLED

#endif // LIMBO_COMPAT_H
