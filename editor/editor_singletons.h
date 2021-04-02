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

#include "editor/editor_file_system.h"
#include "editor/editor_inspector.h"
#include "editor/editor_resource_preview.h"
#include "scene/gui/button.h"
#include "scene/gui/panel_container.h"
#include "scene/main/node.h"

class EditorNode;
class EditorData;
class EditorSelection;
class EditorSettings;

class EditorInterface : public Node {
	GDCLASS(EditorInterface, Node);

	EditorNode *editor;

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

	EditorInterface(EditorNode *p_editor);
};

class SceneTreeDock;
class NodeDock;
class ConnectionsDock;
class InspectorDock;
class ImportDock;
class FileSystemDock;

class EditorDocks : public Node {
	GDCLASS(EditorDocks, Node);

public:
	enum DockSlot {
		DOCK_SLOT_LEFT_UL,
		DOCK_SLOT_LEFT_BL,
		DOCK_SLOT_LEFT_UR,
		DOCK_SLOT_LEFT_BR,
		DOCK_SLOT_RIGHT_UL,
		DOCK_SLOT_RIGHT_BL,
		DOCK_SLOT_RIGHT_UR,
		DOCK_SLOT_RIGHT_BR,
		DOCK_SLOT_MAX
	};

private:
	friend class EditorNode;

	EditorNode *editor;

	// EditorNode added
	SceneTreeDock *scene_tree_dock = nullptr;
	FileSystemDock *filesystem_dock = nullptr;
	NodeDock *node_dock = nullptr;
	ConnectionsDock *connection_dock = nullptr;
	InspectorDock *inspector_dock = nullptr;
	ImportDock *import_dock = nullptr;

	// Plugin added
	VBoxContainer *version_control_dock = nullptr;

protected:
	static void _bind_methods();
	static EditorDocks *singleton;

public:
	static EditorDocks *get_singleton() { return singleton; }

	// Internal setters
	void set_version_control_dock(VBoxContainer *p_dock);

	// Getters
	SceneTreeDock *get_scene_tree_dock();
	FileSystemDock *get_filesystem_dock();
	NodeDock *get_node_dock();
	ConnectionsDock *get_connection_dock();
	InspectorDock *get_inspector_dock();
	ImportDock *get_import_dock();
	VBoxContainer *get_version_control_dock();

	// Modifiers
	void add_control(DockSlot p_slot, Control *p_control);
	void remove_control(Control *p_control);

	EditorDocks(EditorNode *p_editor);
};

VARIANT_ENUM_CAST(EditorDocks::DockSlot);

class EditorAudioBuses;
class EditorLog;
class AnimationPlayerEditor;
class AnimationTreeEditor;
class EditorDebuggerNode;
class ResourcePreloaderEditor;
class FindInFilesPanel;
class ShaderEditor;
class ShaderFileEditor;
class SpriteFramesEditor;
class TextureRegionEditor;
class ThemeEditor;
class TileSetEditor;
class VisualShaderEditor;

class EditorBottomPanels : public Node {
	GDCLASS(EditorBottomPanels, Node);

	EditorNode *editor;

	EditorLog *output_panel = nullptr;
	EditorAudioBuses *audio_panel = nullptr;
	AnimationPlayerEditor *animation_panel = nullptr;
	AnimationTreeEditor *animation_tree_panel = nullptr;
	EditorDebuggerNode *debugger_panel = nullptr;
	ResourcePreloaderEditor *resource_preloader_panel = nullptr;
	FindInFilesPanel *find_in_files_panel = nullptr;
	ShaderEditor *shader_panel = nullptr;
	ShaderFileEditor *shader_file_panel = nullptr;
	SpriteFramesEditor *sprite_frames_panel = nullptr;
	TextureRegionEditor *texture_region_panel = nullptr;
	ThemeEditor *theme_panel = nullptr;
	TileSetEditor *tileset_panel = nullptr;
	PanelContainer *version_control_panel = nullptr;
	VisualShaderEditor *visual_shader_panel = nullptr;

protected:
	static void _bind_methods();
	static EditorBottomPanels *singleton;

public:
	static EditorBottomPanels *get_singleton() { return singleton; }

	// Internal setters
	void set_output_panel(EditorLog *p_panel);
	void set_audio_panel(EditorAudioBuses *p_panel);
	void set_animation_panel(AnimationPlayerEditor *p_panel);
	void set_animation_tree_panel(AnimationTreeEditor *p_panel);
	void set_debugger_panel(EditorDebuggerNode *p_panel);
	void set_resource_preloader_panel(ResourcePreloaderEditor *p_panel);
	void set_find_in_files_panel(FindInFilesPanel *p_panel);
	void set_shader_panel(ShaderEditor *p_panel);
	void set_shader_file_panel(ShaderFileEditor *p_panel);
	void set_sprite_frames_panel(SpriteFramesEditor *p_panel);
	void set_texture_region_panel(TextureRegionEditor *p_panel);
	void set_theme_panel(ThemeEditor *p_panel);
	void set_tileset_panel(TileSetEditor *p_panel);
	void set_version_control_panel(PanelContainer *p_panel);
	void set_visual_shader_panel(VisualShaderEditor *p_panel);

	// Getters
	EditorLog *get_output_panel();
	EditorAudioBuses *get_audio_panel();
	AnimationPlayerEditor *get_animation_panel();
	AnimationTreeEditor *get_animation_tree_panel();
	EditorDebuggerNode *get_debugger_panel();
	ResourcePreloaderEditor *get_resource_preloader_panel();
	FindInFilesPanel *get_find_in_files_panel();
	ShaderEditor *get_shader_panel();
	ShaderFileEditor *get_shader_file_panel();
	SpriteFramesEditor *get_sprite_frames_panel();
	TextureRegionEditor *get_texture_region_panel();
	ThemeEditor *get_theme_panel();
	TileSetEditor *get_tileset_panel();
	PanelContainer *get_version_control_panel();
	VisualShaderEditor *get_visual_shader_panel();

	// Modifiers
	Button *add_control(const String p_title, Control *p_control);
	void remove_control(Control *p_control);
	void make_bottom_panel_item_visible(Control *p_item);
	void hide_bottom_panel();

	EditorBottomPanels(EditorNode *p_editor);
};

class CanvasItemEditor;
class Node3DEditor;
class ScriptEditor;
class EditorAssetLibrary;

class EditorWorkspaces : public Node {
	GDCLASS(EditorWorkspaces, Node);

	EditorNode *editor;

	CanvasItemEditor *canvas_item_workspace = nullptr;
	Node3DEditor *node_3d_workspace = nullptr;
	ScriptEditor *script_workspace = nullptr;
	EditorAssetLibrary *asset_library_workspace = nullptr;

protected:
	static void _bind_methods();
	static EditorWorkspaces *singleton;

public:
	static EditorWorkspaces *get_singleton() { return singleton; }

	// Internal setters
	void set_canvas_item_workspace(CanvasItemEditor *p_panel);
	void set_node_3d_workspace(Node3DEditor *p_panel);
	void set_script_workspace(ScriptEditor *p_panel);
	void set_asset_library_workspace(EditorAssetLibrary *p_panel);

	// Getters
	CanvasItemEditor *get_canvas_item_workspace();
	Node3DEditor *get_node_3d_workspace();
	ScriptEditor *get_script_workspace();
	EditorAssetLibrary *get_asset_library_workspace();

	// Modifiers
	void add_control(Control *p_control);
	void remove_control(Control *p_control);

	EditorWorkspaces(EditorNode *p_editor);
};

#endif // EDITOR_SINGLETONS_H
