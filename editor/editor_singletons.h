/*************************************************************************/
/*  editor_singletons.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef EDITOR_SINGLETONS_H
#define EDITOR_SINGLETONS_H

#include "editor/editor_resource_preview.h"
//#include "editor/scene_tree_dock.h"
//#include "editor/inspector_dock.h"
//#include "editor/node_dock.h"
#include "editor/editor_file_system.h"
#include "editor/editor_inspector.h"
#include "editor/filesystem_dock.h"
#include "scene/main/node.h"

class ScriptEditor;

class SceneTreeDock;
class NodeDock;
class ConnectionsDock;
class InspectorDock;
class ImportDock;
class FileSystemDock;

class EditorData;
class EditorSelection;
class EditorSettings;

class EditorInterface : public Node {
	GDCLASS(EditorInterface, Node);

protected:
	static void _bind_methods();
	static EditorInterface *singleton;

	Array _make_mesh_previews(const Array &p_meshes, int p_preview_size);

public:
	static EditorInterface *get_singleton() { return singleton; }

	Control *get_editor_main_control();
	void edit_resource(const Ref<Resource> &p_resource);
	void open_scene_from_path(const String &scene_path);
	void reload_scene_from_path(const String &scene_path);

	void play_main_scene();
	void play_current_scene();
	void play_custom_scene(const String &scene_path);
	void stop_playing_scene();
	bool is_playing_scene() const;
	String get_playing_scene() const;

	Node *get_edited_scene_root();
	Array get_open_scenes() const;
	ScriptEditor *get_script_editor();

	void select_file(const String &p_file);
	String get_selected_path() const;
	String get_current_path() const;

	void inspect_object(Object *p_obj, const String &p_for_property = String(), bool p_inspector_only = false);

	EditorSelection *get_selection();
	//EditorImportExport *get_import_export();
	Ref<EditorSettings> get_editor_settings();
	EditorResourcePreview *get_resource_previewer();
	EditorFileSystem *get_resource_file_system();

	Control *get_base_control();
	float get_editor_scale() const;

	void set_plugin_enabled(const String &p_plugin, bool p_enabled);
	bool is_plugin_enabled(const String &p_plugin) const;

	EditorInspector *get_inspector() const;

	Error save_scene();
	void save_scene_as(const String &p_scene, bool p_with_preview = true);

	Vector<Ref<Texture2D>> make_mesh_previews(const Vector<Ref<Mesh>> &p_meshes, Vector<Transform> *p_transforms, int p_preview_size);

	void set_main_screen_editor(const String &p_name);
	void set_distraction_free_mode(bool p_enter);
	bool is_distraction_free_mode_enabled() const;

	EditorInterface();
};

class EditorDocks : public Node {
	GDCLASS(EditorDocks, Node);

	friend class EditorNode;

	SceneTreeDock *scene_tree_dock = nullptr;
	FileSystemDock *filesystem_dock = nullptr;
	NodeDock *node_dock = nullptr;
	ConnectionsDock *connection_dock = nullptr;
	InspectorDock *inspector_dock = nullptr;
	ImportDock *import_dock = nullptr;

protected:
	static void _bind_methods();
	static EditorDocks *singleton;

public:
	static EditorDocks *get_singleton() { return singleton; }

	SceneTreeDock *get_scene_tree_dock();
	FileSystemDock *get_filesystem_dock();
	NodeDock *get_node_dock();
	ConnectionsDock *get_connection_dock();
	InspectorDock *get_inspector_dock();
	ImportDock *get_import_dock();

	void add_dock(String p_name, Control p_control);
	void remove_dock(String p_name);

	EditorDocks();
};

#endif // EDITOR_SINGLETONS_H
