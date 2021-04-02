/*************************************************************************/
/*  editor_singletons.cpp                                                */
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

#include "editor_singletons.h"

#include "editor/editor_audio_buses.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/filesystem_dock.h"
#include "editor/find_in_files.h"
#include "editor/import_dock.h"
#include "editor/node_dock.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/animation_tree_editor_plugin.h"
#include "editor/plugins/asset_library_editor_plugin.h"
#include "editor/plugins/canvas_item_editor_plugin.h"
#include "editor/plugins/editor_debugger_plugin.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "editor/plugins/resource_preloader_editor_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/plugins/shader_editor_plugin.h"
#include "editor/plugins/shader_file_editor_plugin.h"
#include "editor/plugins/sprite_frames_editor_plugin.h"
#include "editor/plugins/texture_region_editor_plugin.h"
#include "editor/plugins/theme_editor_plugin.h"
#include "editor/plugins/tile_set_editor_plugin.h"
#include "editor/plugins/visual_shader_editor_plugin.h"

#include "main/main.h"

Array EditorInterface::_make_mesh_previews(const Array &p_meshes, int p_preview_size) {
	Vector<Ref<Mesh>> meshes;

	for (int i = 0; i < p_meshes.size(); i++) {
		meshes.push_back(p_meshes[i]);
	}

	Vector<Ref<Texture2D>> textures = make_mesh_previews(meshes, nullptr, p_preview_size);
	Array ret;
	for (int i = 0; i < textures.size(); i++) {
		ret.push_back(textures[i]);
	}

	return ret;
}

Vector<Ref<Texture2D>> EditorInterface::make_mesh_previews(const Vector<Ref<Mesh>> &p_meshes, Vector<Transform> *p_transforms, int p_preview_size) {
	int size = p_preview_size;

	RID scenario = RS::get_singleton()->scenario_create();

	RID viewport = RS::get_singleton()->viewport_create();
	RS::get_singleton()->viewport_set_update_mode(viewport, RS::VIEWPORT_UPDATE_ALWAYS);
	RS::get_singleton()->viewport_set_scenario(viewport, scenario);
	RS::get_singleton()->viewport_set_size(viewport, size, size);
	RS::get_singleton()->viewport_set_transparent_background(viewport, true);
	RS::get_singleton()->viewport_set_active(viewport, true);
	RID viewport_texture = RS::get_singleton()->viewport_get_texture(viewport);

	RID camera = RS::get_singleton()->camera_create();
	RS::get_singleton()->viewport_attach_camera(viewport, camera);

	RID light = RS::get_singleton()->directional_light_create();
	RID light_instance = RS::get_singleton()->instance_create2(light, scenario);

	RID light2 = RS::get_singleton()->directional_light_create();
	RS::get_singleton()->light_set_color(light2, Color(0.7, 0.7, 0.7));
	RID light_instance2 = RS::get_singleton()->instance_create2(light2, scenario);

	EditorProgress ep("mlib", TTR("Creating Mesh Previews"), p_meshes.size());

	Vector<Ref<Texture2D>> textures;

	for (int i = 0; i < p_meshes.size(); i++) {
		Ref<Mesh> mesh = p_meshes[i];
		if (!mesh.is_valid()) {
			textures.push_back(Ref<Texture2D>());
			continue;
		}

		Transform mesh_xform;
		if (p_transforms != nullptr) {
			mesh_xform = (*p_transforms)[i];
		}

		RID inst = RS::get_singleton()->instance_create2(mesh->get_rid(), scenario);
		RS::get_singleton()->instance_set_transform(inst, mesh_xform);

		AABB aabb = mesh->get_aabb();
		Vector3 ofs = aabb.position + aabb.size * 0.5;
		aabb.position -= ofs;
		Transform xform;
		xform.basis = Basis().rotated(Vector3(0, 1, 0), -Math_PI / 6);
		xform.basis = Basis().rotated(Vector3(1, 0, 0), Math_PI / 6) * xform.basis;
		AABB rot_aabb = xform.xform(aabb);
		float m = MAX(rot_aabb.size.x, rot_aabb.size.y) * 0.5;
		if (m == 0) {
			textures.push_back(Ref<Texture2D>());
			continue;
		}
		xform.origin = -xform.basis.xform(ofs); //-ofs*m;
		xform.origin.z -= rot_aabb.size.z * 2;
		xform.invert();
		xform = mesh_xform * xform;

		RS::get_singleton()->camera_set_transform(camera, xform * Transform(Basis(), Vector3(0, 0, 3)));
		RS::get_singleton()->camera_set_orthogonal(camera, m * 2, 0.01, 1000.0);

		RS::get_singleton()->instance_set_transform(light_instance, xform * Transform().looking_at(Vector3(-2, -1, -1), Vector3(0, 1, 0)));
		RS::get_singleton()->instance_set_transform(light_instance2, xform * Transform().looking_at(Vector3(+1, -1, -2), Vector3(0, 1, 0)));

		ep.step(TTR("Thumbnail..."), i);
		Main::iteration();
		Main::iteration();
		Ref<Image> img = RS::get_singleton()->texture_2d_get(viewport_texture);
		ERR_CONTINUE(!img.is_valid() || img->is_empty());
		Ref<ImageTexture> it(memnew(ImageTexture));
		it->create_from_image(img);

		RS::get_singleton()->free(inst);

		textures.push_back(it);
	}

	RS::get_singleton()->free(viewport);
	RS::get_singleton()->free(light);
	RS::get_singleton()->free(light_instance);
	RS::get_singleton()->free(light2);
	RS::get_singleton()->free(light_instance2);
	RS::get_singleton()->free(camera);
	RS::get_singleton()->free(scenario);

	return textures;
}

void EditorInterface::set_main_screen_editor(const String &p_name) {
	editor->select_editor_by_name(p_name);
}

Control *EditorInterface::get_editor_main_control() {
	return editor->get_main_control();
}

void EditorInterface::edit_resource(const Ref<Resource> &p_resource) {
	editor->edit_resource(p_resource);
}

void EditorInterface::open_scene_from_path(const String &scene_path) {
	if (editor->is_changing_scene()) {
		return;
	}

	editor->open_request(scene_path);
}

void EditorInterface::reload_scene_from_path(const String &scene_path) {
	if (editor->is_changing_scene()) {
		return;
	}

	editor->reload_scene(scene_path);
}

void EditorInterface::play_main_scene() {
	editor->run_play();
}

void EditorInterface::play_current_scene() {
	editor->run_play_current();
}

void EditorInterface::play_custom_scene(const String &scene_path) {
	editor->run_play_custom(scene_path);
}

void EditorInterface::stop_playing_scene() {
	editor->run_stop();
}

bool EditorInterface::is_playing_scene() const {
	return editor->is_run_playing();
}

String EditorInterface::get_playing_scene() const {
	return editor->get_run_playing_scene();
}

Node *EditorInterface::get_edited_scene_root() {
	return editor->get_edited_scene();
}

Array EditorInterface::get_open_scenes() const {
	Array ret;
	Vector<EditorData::EditedScene> scenes = EditorNode::get_editor_data().get_edited_scenes();

	int scns_amount = scenes.size();
	for (int idx_scn = 0; idx_scn < scns_amount; idx_scn++) {
		if (scenes[idx_scn].root == nullptr) {
			continue;
		}
		ret.push_back(scenes[idx_scn].root->get_filename());
	}
	return ret;
}

void EditorInterface::select_file(const String &p_file) {
	EditorDocks::get_singleton()->get_filesystem_dock()->select_file(p_file);
}

String EditorInterface::get_selected_path() const {
	return EditorDocks::get_singleton()->get_filesystem_dock()->get_selected_path();
}

String EditorInterface::get_current_path() const {
	return EditorDocks::get_singleton()->get_filesystem_dock()->get_current_path();
}

void EditorInterface::inspect_object(Object *p_obj, const String &p_for_property, bool p_inspector_only) {
	editor->push_item(p_obj, p_for_property, p_inspector_only);
}

EditorFileSystem *EditorInterface::get_resource_file_system() {
	return EditorFileSystem::get_singleton();
}

EditorSelection *EditorInterface::get_selection() {
	return editor->get_editor_selection();
}

Ref<EditorSettings> EditorInterface::get_editor_settings() {
	return EditorSettings::get_singleton();
}

EditorResourcePreview *EditorInterface::get_resource_previewer() {
	return EditorResourcePreview::get_singleton();
}

Control *EditorInterface::get_base_control() {
	return editor->get_gui_base();
}

float EditorInterface::get_editor_scale() const {
	return EDSCALE;
}

void EditorInterface::set_plugin_enabled(const String &p_plugin, bool p_enabled) {
	editor->set_addon_plugin_enabled(p_plugin, p_enabled, true);
}

bool EditorInterface::is_plugin_enabled(const String &p_plugin) const {
	return editor->is_addon_plugin_enabled(p_plugin);
}

EditorInspector *EditorInterface::get_inspector() const {
	return editor->get_inspector();
}

Error EditorInterface::save_scene() {
	if (!get_edited_scene_root()) {
		return ERR_CANT_CREATE;
	}
	if (get_edited_scene_root()->get_filename() == String()) {
		return ERR_CANT_CREATE;
	}

	save_scene_as(get_edited_scene_root()->get_filename());
	return OK;
}

void EditorInterface::save_scene_as(const String &p_scene, bool p_with_preview) {
	editor->save_scene_to_path(p_scene, p_with_preview);
}

void EditorInterface::set_distraction_free_mode(bool p_enter) {
	editor->set_distraction_free_mode(p_enter);
}

bool EditorInterface::is_distraction_free_mode_enabled() const {
	return editor->is_distraction_free_mode_enabled();
}

EditorInterface *EditorInterface::singleton = nullptr;

void EditorInterface::_bind_methods() {
	ClassDB::bind_method(D_METHOD("inspect_object", "object", "for_property", "inspector_only"), &EditorInterface::inspect_object, DEFVAL(String()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_selection"), &EditorInterface::get_selection);
	ClassDB::bind_method(D_METHOD("get_editor_settings"), &EditorInterface::get_editor_settings);
	ClassDB::bind_method(D_METHOD("get_base_control"), &EditorInterface::get_base_control);
	ClassDB::bind_method(D_METHOD("get_editor_scale"), &EditorInterface::get_editor_scale);
	ClassDB::bind_method(D_METHOD("edit_resource", "resource"), &EditorInterface::edit_resource);
	ClassDB::bind_method(D_METHOD("open_scene_from_path", "scene_filepath"), &EditorInterface::open_scene_from_path);
	ClassDB::bind_method(D_METHOD("reload_scene_from_path", "scene_filepath"), &EditorInterface::reload_scene_from_path);
	ClassDB::bind_method(D_METHOD("play_main_scene"), &EditorInterface::play_main_scene);
	ClassDB::bind_method(D_METHOD("play_current_scene"), &EditorInterface::play_current_scene);
	ClassDB::bind_method(D_METHOD("play_custom_scene", "scene_filepath"), &EditorInterface::play_custom_scene);
	ClassDB::bind_method(D_METHOD("stop_playing_scene"), &EditorInterface::stop_playing_scene);
	ClassDB::bind_method(D_METHOD("is_playing_scene"), &EditorInterface::is_playing_scene);
	ClassDB::bind_method(D_METHOD("get_playing_scene"), &EditorInterface::get_playing_scene);
	ClassDB::bind_method(D_METHOD("get_open_scenes"), &EditorInterface::get_open_scenes);
	ClassDB::bind_method(D_METHOD("get_edited_scene_root"), &EditorInterface::get_edited_scene_root);
	ClassDB::bind_method(D_METHOD("get_resource_previewer"), &EditorInterface::get_resource_previewer);
	ClassDB::bind_method(D_METHOD("get_resource_filesystem"), &EditorInterface::get_resource_file_system);
	ClassDB::bind_method(D_METHOD("get_editor_main_control"), &EditorInterface::get_editor_main_control);
	ClassDB::bind_method(D_METHOD("make_mesh_previews", "meshes", "preview_size"), &EditorInterface::_make_mesh_previews);
	ClassDB::bind_method(D_METHOD("select_file", "file"), &EditorInterface::select_file);
	ClassDB::bind_method(D_METHOD("get_selected_path"), &EditorInterface::get_selected_path);
	ClassDB::bind_method(D_METHOD("get_current_path"), &EditorInterface::get_current_path);

	ClassDB::bind_method(D_METHOD("set_plugin_enabled", "plugin", "enabled"), &EditorInterface::set_plugin_enabled);
	ClassDB::bind_method(D_METHOD("is_plugin_enabled", "plugin"), &EditorInterface::is_plugin_enabled);

	ClassDB::bind_method(D_METHOD("get_inspector"), &EditorInterface::get_inspector);

	ClassDB::bind_method(D_METHOD("save_scene"), &EditorInterface::save_scene);
	ClassDB::bind_method(D_METHOD("save_scene_as", "path", "with_preview"), &EditorInterface::save_scene_as, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("set_main_screen_editor", "name"), &EditorInterface::set_main_screen_editor);
	ClassDB::bind_method(D_METHOD("set_distraction_free_mode", "enter"), &EditorInterface::set_distraction_free_mode);
	ClassDB::bind_method(D_METHOD("is_distraction_free_mode_enabled"), &EditorInterface::is_distraction_free_mode_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "distraction_free_mode"), "set_distraction_free_mode", "is_distraction_free_mode_enabled");
}

EditorInterface::EditorInterface(EditorNode *p_editor) {
	singleton = this;
	editor = p_editor;
}

///////////////////////////////////////////

SceneTreeDock *EditorDocks::get_scene_tree_dock() {
	return scene_tree_dock;
}

FileSystemDock *EditorDocks::get_filesystem_dock() {
	return filesystem_dock;
}

NodeDock *EditorDocks::get_node_dock() {
	return node_dock;
}

ConnectionsDock *EditorDocks::get_connection_dock() {
	return connection_dock;
}

InspectorDock *EditorDocks::get_inspector_dock() {
	return inspector_dock;
}

ImportDock *EditorDocks::get_import_dock() {
	return import_dock;
}

void EditorDocks::add_control(DockSlot p_slot, Control *p_control) {
	ERR_FAIL_NULL(p_control);
	editor->add_control_to_dock(EditorNode::DockSlot(p_slot), p_control);
}

void EditorDocks::remove_control(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	editor->remove_control_from_dock(p_control);
}

void EditorDocks::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_scene_tree_dock"), &EditorDocks::get_scene_tree_dock);
	ClassDB::bind_method(D_METHOD("get_filesystem_dock"), &EditorDocks::get_filesystem_dock);
	ClassDB::bind_method(D_METHOD("get_node_dock"), &EditorDocks::get_node_dock);
	ClassDB::bind_method(D_METHOD("get_connection_dock"), &EditorDocks::get_connection_dock);
	ClassDB::bind_method(D_METHOD("get_inspector_dock"), &EditorDocks::get_inspector_dock);
	ClassDB::bind_method(D_METHOD("get_import_dock"), &EditorDocks::get_import_dock);

	ClassDB::bind_method(D_METHOD("add_control", "slot", "control"), &EditorDocks::add_control);
	ClassDB::bind_method(D_METHOD("remove_control", "control"), &EditorDocks::remove_control);

	BIND_ENUM_CONSTANT(DOCK_SLOT_LEFT_UL);
	BIND_ENUM_CONSTANT(DOCK_SLOT_LEFT_BL);
	BIND_ENUM_CONSTANT(DOCK_SLOT_LEFT_UR);
	BIND_ENUM_CONSTANT(DOCK_SLOT_LEFT_BR);
	BIND_ENUM_CONSTANT(DOCK_SLOT_RIGHT_UL);
	BIND_ENUM_CONSTANT(DOCK_SLOT_RIGHT_BL);
	BIND_ENUM_CONSTANT(DOCK_SLOT_RIGHT_UR);
	BIND_ENUM_CONSTANT(DOCK_SLOT_RIGHT_BR);
	BIND_ENUM_CONSTANT(DOCK_SLOT_MAX);
}

void EditorDocks::set_version_control_dock(VBoxContainer *p_dock) {
	ERR_FAIL_COND(!version_control_dock);
	version_control_dock = p_dock;
}

EditorDocks *EditorDocks::singleton = nullptr;

EditorDocks::EditorDocks(EditorNode *p_editor) {
	singleton = this;
	editor = p_editor;
}

///////////////////////////////////////////

void EditorBottomPanels::set_output_panel(EditorLog *p_panel) {
	ERR_FAIL_COND(output_panel);
	output_panel = p_panel;
}

void EditorBottomPanels::set_audio_panel(EditorAudioBuses *p_panel) {
	ERR_FAIL_COND(audio_panel);
	audio_panel = p_panel;
}

void EditorBottomPanels::set_animation_panel(AnimationPlayerEditor *p_panel) {
	ERR_FAIL_COND(animation_panel);
	animation_panel = p_panel;
}

void EditorBottomPanels::set_animation_tree_panel(AnimationTreeEditor *p_panel) {
	ERR_FAIL_COND(animation_tree_panel);
	animation_tree_panel = p_panel;
}

void EditorBottomPanels::set_debugger_panel(EditorDebuggerNode *p_panel) {
	ERR_FAIL_COND(debugger_panel);
	debugger_panel = p_panel;
}

void EditorBottomPanels::set_resource_preloader_panel(ResourcePreloaderEditor *p_panel) {
	ERR_FAIL_COND(resource_preloader_panel);
	resource_preloader_panel = p_panel;
}

void EditorBottomPanels::set_find_in_files_panel(FindInFilesPanel *p_panel) {
	ERR_FAIL_COND(find_in_files_panel);
	find_in_files_panel = p_panel;
}

void EditorBottomPanels::set_shader_panel(ShaderEditor *p_panel) {
	ERR_FAIL_COND(shader_panel);
	shader_panel = p_panel;
}

void EditorBottomPanels::set_shader_file_panel(ShaderFileEditor *p_panel) {
	ERR_FAIL_COND(shader_file_panel);
	shader_file_panel = p_panel;
}

void EditorBottomPanels::set_sprite_frames_panel(SpriteFramesEditor *p_panel) {
	ERR_FAIL_COND(sprite_frames_panel);
	sprite_frames_panel = p_panel;
}

void EditorBottomPanels::set_texture_region_panel(TextureRegionEditor *p_panel) {
	ERR_FAIL_COND(texture_region_panel);
	texture_region_panel = p_panel;
}

void EditorBottomPanels::set_theme_panel(ThemeEditor *p_panel) {
	ERR_FAIL_COND(theme_panel);
	theme_panel = p_panel;
}

void EditorBottomPanels::set_tileset_panel(TileSetEditor *p_panel) {
	ERR_FAIL_COND(tileset_panel);
	tileset_panel = p_panel;
}

void EditorBottomPanels::set_version_control_panel(PanelContainer *p_panel) {
	ERR_FAIL_COND(version_control_panel);
	version_control_panel = p_panel;
}

void EditorBottomPanels::set_visual_shader_panel(VisualShaderEditor *p_panel) {
	ERR_FAIL_COND(visual_shader_panel);
	visual_shader_panel = p_panel;
}

EditorLog *EditorBottomPanels::get_output_panel() {
	return output_panel;
}

EditorAudioBuses *EditorBottomPanels::get_audio_panel() {
	return audio_panel;
}

AnimationPlayerEditor *EditorBottomPanels::get_animation_panel() {
	return animation_panel;
}

AnimationTreeEditor *EditorBottomPanels::get_animation_tree_panel() {
	return animation_tree_panel;
}

EditorDebuggerNode *EditorBottomPanels::get_debugger_panel() {
	return debugger_panel;
}

ResourcePreloaderEditor *EditorBottomPanels::get_resource_preloader_panel() {
	return resource_preloader_panel;
}

FindInFilesPanel *EditorBottomPanels::get_find_in_files_panel() {
	return find_in_files_panel;
}

ShaderEditor *EditorBottomPanels::get_shader_panel() {
	return shader_panel;
}

ShaderFileEditor *EditorBottomPanels::get_shader_file_panel() {
	return shader_file_panel;
}

SpriteFramesEditor *EditorBottomPanels::get_sprite_frames_panel() {
	return sprite_frames_panel;
}

TextureRegionEditor *EditorBottomPanels::get_texture_region_panel() {
	return texture_region_panel;
}

ThemeEditor *EditorBottomPanels::get_theme_panel() {
	return theme_panel;
}

TileSetEditor *EditorBottomPanels::get_tileset_panel() {
	return tileset_panel;
}

PanelContainer *EditorBottomPanels::get_version_control_panel() {
	return version_control_panel;
}

VisualShaderEditor *EditorBottomPanels::get_visual_shader_panel() {
	return visual_shader_panel;
}

Button *EditorBottomPanels::add_control(const String p_title, Control *p_control) {
	ERR_FAIL_NULL_V(p_control, nullptr);
	return editor->add_bottom_panel_item(p_title, p_control);
}

void EditorBottomPanels::remove_control(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	editor->remove_bottom_panel_item(p_control);
}

void EditorBottomPanels::make_bottom_panel_item_visible(Control *p_item) {
	editor->make_bottom_panel_item_visible(p_item);
}

void EditorBottomPanels::hide_bottom_panel() {
	editor->hide_bottom_panel();
}

void EditorBottomPanels::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_output_panel"), &EditorBottomPanels::get_output_panel);
	ClassDB::bind_method(D_METHOD("get_audio_panel"), &EditorBottomPanels::get_audio_panel);
	ClassDB::bind_method(D_METHOD("get_animation_panel"), &EditorBottomPanels::get_animation_panel);
	ClassDB::bind_method(D_METHOD("get_animation_tree_panel"), &EditorBottomPanels::get_animation_tree_panel);
	ClassDB::bind_method(D_METHOD("get_debugger_panel"), &EditorBottomPanels::get_debugger_panel);
	ClassDB::bind_method(D_METHOD("get_resource_preloader_panel"), &EditorBottomPanels::get_resource_preloader_panel);
	ClassDB::bind_method(D_METHOD("get_find_in_files_panel"), &EditorBottomPanels::get_find_in_files_panel);
	ClassDB::bind_method(D_METHOD("get_shader_panel"), &EditorBottomPanels::get_shader_panel);
	ClassDB::bind_method(D_METHOD("get_sprite_frames_panel"), &EditorBottomPanels::get_sprite_frames_panel);
	ClassDB::bind_method(D_METHOD("get_texture_region_panel"), &EditorBottomPanels::get_texture_region_panel);
	ClassDB::bind_method(D_METHOD("get_theme_panel"), &EditorBottomPanels::get_theme_panel);
	ClassDB::bind_method(D_METHOD("get_tileset_panel"), &EditorBottomPanels::get_tileset_panel);
	ClassDB::bind_method(D_METHOD("get_version_control_panel"), &EditorBottomPanels::get_version_control_panel);
	ClassDB::bind_method(D_METHOD("get_visual_shader_panel"), &EditorBottomPanels::get_visual_shader_panel);

	ClassDB::bind_method(D_METHOD("add_control", "title", "control"), &EditorBottomPanels::add_control);
	ClassDB::bind_method(D_METHOD("remove_control", "control"), &EditorBottomPanels::remove_control);

	ClassDB::bind_method(D_METHOD("make_bottom_panel_item_visible", "control"), &EditorBottomPanels::make_bottom_panel_item_visible);
	ClassDB::bind_method(D_METHOD("hide_bottom_panel"), &EditorBottomPanels::hide_bottom_panel);
}

EditorBottomPanels *EditorBottomPanels::singleton = nullptr;

EditorBottomPanels::EditorBottomPanels(EditorNode *p_editor) {
	singleton = this;
	editor = p_editor;
}

///////////////////////////////////////////

void EditorWorkspaces::set_canvas_item_workspace(CanvasItemEditor *p_panel) {
	ERR_FAIL_COND(canvas_item_workspace);
	canvas_item_workspace = p_panel;
}

void EditorWorkspaces::set_node_3d_workspace(Node3DEditor *p_panel) {
	ERR_FAIL_COND(node_3d_workspace);
	node_3d_workspace = p_panel;
}

void EditorWorkspaces::set_script_workspace(ScriptEditor *p_panel) {
	ERR_FAIL_COND(script_workspace);
	script_workspace = p_panel;
}

void EditorWorkspaces::set_asset_library_workspace(EditorAssetLibrary *p_panel) {
	ERR_FAIL_COND(asset_library_workspace);
	asset_library_workspace = p_panel;
}

CanvasItemEditor *EditorWorkspaces::get_canvas_item_workspace() {
	return canvas_item_workspace;
}

Node3DEditor *EditorWorkspaces::get_node_3d_workspace() {
	return node_3d_workspace;
}

ScriptEditor *EditorWorkspaces::get_script_workspace() {
	return script_workspace;
}

EditorAssetLibrary *EditorWorkspaces::get_asset_library_workspace() {
	return asset_library_workspace;
}

void EditorWorkspaces::add_control(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	ERR_FAIL_COND(p_control->get_parent());
	editor->get_main_control()->add_child(p_control);
}

void EditorWorkspaces::remove_control(Control *p_control) {
	ERR_FAIL_COND(p_control->get_parent() != editor->get_main_control());
	p_control->get_parent()->remove_child(p_control);
}

void EditorWorkspaces::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_canvas_item_workspace"), &EditorWorkspaces::get_canvas_item_workspace);
	ClassDB::bind_method(D_METHOD("get_node_3d_workspace"), &EditorWorkspaces::get_node_3d_workspace);
	ClassDB::bind_method(D_METHOD("get_script_workspace"), &EditorWorkspaces::get_script_workspace);
	ClassDB::bind_method(D_METHOD("get_asset_library_workspace"), &EditorWorkspaces::get_asset_library_workspace);

	ClassDB::bind_method(D_METHOD("add_control", "title", "control"), &EditorWorkspaces::add_control);
	ClassDB::bind_method(D_METHOD("remove_control", "control"), &EditorWorkspaces::remove_control);
}

EditorWorkspaces *EditorWorkspaces::singleton = nullptr;

EditorWorkspaces::EditorWorkspaces(EditorNode *p_editor) {
	singleton = this;
	editor = p_editor;
}
