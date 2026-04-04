/**************************************************************************/
/*  editor_interface.hpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/rect2i.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class Control;
class EditorCommandPalette;
class EditorFileSystem;
class EditorInspector;
class EditorPaths;
class EditorResourcePreview;
class EditorSelection;
class EditorSettings;
class EditorToaster;
class EditorUndoRedoManager;
class FileSystemDock;
class Mesh;
class Resource;
class Script;
class ScriptEditor;
class SubViewport;
class Texture2D;
class Theme;
class VBoxContainer;
class Window;

class EditorInterface : public Object {
	GDEXTENSION_CLASS(EditorInterface, Object)

	static EditorInterface *singleton;

public:
	static EditorInterface *get_singleton();

	void restart_editor(bool p_save = true);
	EditorCommandPalette *get_command_palette() const;
	EditorFileSystem *get_resource_filesystem() const;
	EditorPaths *get_editor_paths() const;
	EditorResourcePreview *get_resource_previewer() const;
	EditorSelection *get_selection() const;
	Ref<EditorSettings> get_editor_settings() const;
	EditorToaster *get_editor_toaster() const;
	EditorUndoRedoManager *get_editor_undo_redo() const;
	TypedArray<Ref<Texture2D>> make_mesh_previews(const TypedArray<Ref<Mesh>> &p_meshes, int32_t p_preview_size);
	void set_plugin_enabled(const String &p_plugin, bool p_enabled);
	bool is_plugin_enabled(const String &p_plugin) const;
	Ref<Theme> get_editor_theme() const;
	Control *get_base_control() const;
	VBoxContainer *get_editor_main_screen() const;
	ScriptEditor *get_script_editor() const;
	SubViewport *get_editor_viewport_2d() const;
	SubViewport *get_editor_viewport_3d(int32_t p_idx = 0) const;
	void set_main_screen_editor(const String &p_name);
	void set_distraction_free_mode(bool p_enter);
	bool is_distraction_free_mode_enabled() const;
	bool is_multi_window_enabled() const;
	float get_editor_scale() const;
	String get_editor_language() const;
	bool is_node_3d_snap_enabled() const;
	float get_node_3d_translate_snap() const;
	float get_node_3d_rotate_snap() const;
	float get_node_3d_scale_snap() const;
	void popup_dialog(Window *p_dialog, const Rect2i &p_rect = Rect2i(0, 0, 0, 0));
	void popup_dialog_centered(Window *p_dialog, const Vector2i &p_minsize = Vector2i(0, 0));
	void popup_dialog_centered_ratio(Window *p_dialog, float p_ratio = 0.8);
	void popup_dialog_centered_clamped(Window *p_dialog, const Vector2i &p_minsize = Vector2i(0, 0), float p_fallback_ratio = 0.75);
	String get_current_feature_profile() const;
	void set_current_feature_profile(const String &p_profile_name);
	void popup_node_selector(const Callable &p_callback, const TypedArray<StringName> &p_valid_types = {}, Node *p_current_value = nullptr);
	void popup_property_selector(Object *p_object, const Callable &p_callback, const PackedInt32Array &p_type_filter = PackedInt32Array(), const String &p_current_value = String());
	void popup_method_selector(Object *p_object, const Callable &p_callback, const String &p_current_value = String());
	void popup_quick_open(const Callable &p_callback, const TypedArray<StringName> &p_base_types = {});
	void popup_create_dialog(const Callable &p_callback, const StringName &p_base_type = String(), const String &p_current_type = String(), const String &p_dialog_title = String(), const TypedArray<StringName> &p_type_blocklist = {});
	FileSystemDock *get_file_system_dock() const;
	void select_file(const String &p_file);
	PackedStringArray get_selected_paths() const;
	String get_current_path() const;
	String get_current_directory() const;
	EditorInspector *get_inspector() const;
	void inspect_object(Object *p_object, const String &p_for_property = String(), bool p_inspector_only = false);
	void edit_resource(const Ref<Resource> &p_resource);
	void edit_node(Node *p_node);
	void edit_script(const Ref<Script> &p_script, int32_t p_line = -1, int32_t p_column = 0, bool p_grab_focus = true);
	void open_scene_from_path(const String &p_scene_filepath, bool p_set_inherited = false);
	void reload_scene_from_path(const String &p_scene_filepath);
	void set_object_edited(Object *p_object, bool p_edited);
	bool is_object_edited(Object *p_object) const;
	PackedStringArray get_open_scenes() const;
	TypedArray<Node> get_open_scene_roots() const;
	Node *get_edited_scene_root() const;
	void add_root_node(Node *p_node);
	Error save_scene();
	void save_scene_as(const String &p_path, bool p_with_preview = true);
	void save_all_scenes();
	Error close_scene();
	void mark_scene_as_unsaved();
	void play_main_scene();
	void play_current_scene();
	void play_custom_scene(const String &p_scene_filepath);
	void stop_playing_scene();
	bool is_playing_scene() const;
	String get_playing_scene() const;
	void set_movie_maker_enabled(bool p_enabled);
	bool is_movie_maker_enabled() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~EditorInterface();

public:
};

} // namespace godot

