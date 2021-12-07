/*************************************************************************/
/*  editor_plugin.cpp                                                    */
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

#include "editor_plugin.h"

#include "editor/editor_command_palette.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_paths.h"
#include "editor/editor_settings.h"
#include "editor/filesystem_dock.h"
#include "editor/project_settings_editor.h"
#include "editor_resource_preview.h"
#include "main/main.h"
#include "plugins/canvas_item_editor_plugin.h"
#include "plugins/node_3d_editor_plugin.h"
#include "scene/3d/camera_3d.h"
#include "scene/gui/popup_menu.h"
#include "servers/rendering_server.h"

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

Vector<Ref<Texture2D>> EditorInterface::make_mesh_previews(const Vector<Ref<Mesh>> &p_meshes, Vector<Transform3D> *p_transforms, int p_preview_size) {
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

		Transform3D mesh_xform;
		if (p_transforms != nullptr) {
			mesh_xform = (*p_transforms)[i];
		}

		RID inst = RS::get_singleton()->instance_create2(mesh->get_rid(), scenario);
		RS::get_singleton()->instance_set_transform(inst, mesh_xform);

		AABB aabb = mesh->get_aabb();
		Vector3 ofs = aabb.get_center();
		aabb.position -= ofs;
		Transform3D xform;
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

		RS::get_singleton()->camera_set_transform(camera, xform * Transform3D(Basis(), Vector3(0, 0, 3)));
		RS::get_singleton()->camera_set_orthogonal(camera, m * 2, 0.01, 1000.0);

		RS::get_singleton()->instance_set_transform(light_instance, xform * Transform3D().looking_at(Vector3(-2, -1, -1), Vector3(0, 1, 0)));
		RS::get_singleton()->instance_set_transform(light_instance2, xform * Transform3D().looking_at(Vector3(+1, -1, -2), Vector3(0, 1, 0)));

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
	EditorNode::get_singleton()->select_editor_by_name(p_name);
}

Control *EditorInterface::get_editor_main_control() {
	return EditorNode::get_singleton()->get_main_control();
}

void EditorInterface::edit_resource(const Ref<Resource> &p_resource) {
	EditorNode::get_singleton()->edit_resource(p_resource);
}

void EditorInterface::edit_node(Node *p_node) {
	EditorNode::get_singleton()->edit_node(p_node);
}

void EditorInterface::edit_script(const Ref<Script> &p_script, int p_line, int p_col, bool p_grab_focus) {
	ScriptEditor::get_singleton()->edit(p_script, p_line, p_col, p_grab_focus);
}

void EditorInterface::open_scene_from_path(const String &scene_path) {
	if (EditorNode::get_singleton()->is_changing_scene()) {
		return;
	}

	EditorNode::get_singleton()->open_request(scene_path);
}

void EditorInterface::reload_scene_from_path(const String &scene_path) {
	if (EditorNode::get_singleton()->is_changing_scene()) {
		return;
	}

	EditorNode::get_singleton()->reload_scene(scene_path);
}

void EditorInterface::play_main_scene() {
	EditorNode::get_singleton()->run_play();
}

void EditorInterface::play_current_scene() {
	EditorNode::get_singleton()->run_play_current();
}

void EditorInterface::play_custom_scene(const String &scene_path) {
	EditorNode::get_singleton()->run_play_custom(scene_path);
}

void EditorInterface::stop_playing_scene() {
	EditorNode::get_singleton()->run_stop();
}

bool EditorInterface::is_playing_scene() const {
	return EditorNode::get_singleton()->is_run_playing();
}

String EditorInterface::get_playing_scene() const {
	return EditorNode::get_singleton()->get_run_playing_scene();
}

Node *EditorInterface::get_edited_scene_root() {
	return EditorNode::get_singleton()->get_edited_scene();
}

Array EditorInterface::get_open_scenes() const {
	Array ret;
	Vector<EditorData::EditedScene> scenes = EditorNode::get_editor_data().get_edited_scenes();

	int scns_amount = scenes.size();
	for (int idx_scn = 0; idx_scn < scns_amount; idx_scn++) {
		if (scenes[idx_scn].root == nullptr) {
			continue;
		}
		ret.push_back(scenes[idx_scn].root->get_scene_file_path());
	}
	return ret;
}

ScriptEditor *EditorInterface::get_script_editor() {
	return ScriptEditor::get_singleton();
}

void EditorInterface::select_file(const String &p_file) {
	EditorNode::get_singleton()->get_filesystem_dock()->select_file(p_file);
}

String EditorInterface::get_selected_path() const {
	return EditorNode::get_singleton()->get_filesystem_dock()->get_selected_path();
}

String EditorInterface::get_current_path() const {
	return EditorNode::get_singleton()->get_filesystem_dock()->get_current_path();
}

void EditorInterface::inspect_object(Object *p_obj, const String &p_for_property, bool p_inspector_only) {
	EditorNode::get_singleton()->push_item(p_obj, p_for_property, p_inspector_only);
}

EditorFileSystem *EditorInterface::get_resource_file_system() {
	return EditorFileSystem::get_singleton();
}

FileSystemDock *EditorInterface::get_file_system_dock() {
	return EditorNode::get_singleton()->get_filesystem_dock();
}

EditorSelection *EditorInterface::get_selection() {
	return EditorNode::get_singleton()->get_editor_selection();
}

Ref<EditorSettings> EditorInterface::get_editor_settings() {
	return EditorSettings::get_singleton();
}
EditorPaths *EditorInterface::get_editor_paths() {
	return EditorPaths::get_singleton();
}

EditorResourcePreview *EditorInterface::get_resource_previewer() {
	return EditorResourcePreview::get_singleton();
}

Control *EditorInterface::get_base_control() {
	return EditorNode::get_singleton()->get_gui_base();
}

float EditorInterface::get_editor_scale() const {
	return EDSCALE;
}

void EditorInterface::set_plugin_enabled(const String &p_plugin, bool p_enabled) {
	EditorNode::get_singleton()->set_addon_plugin_enabled(p_plugin, p_enabled, true);
}

bool EditorInterface::is_plugin_enabled(const String &p_plugin) const {
	return EditorNode::get_singleton()->is_addon_plugin_enabled(p_plugin);
}

EditorInspector *EditorInterface::get_inspector() const {
	return EditorNode::get_singleton()->get_inspector();
}

Error EditorInterface::save_scene() {
	if (!get_edited_scene_root()) {
		return ERR_CANT_CREATE;
	}
	if (get_edited_scene_root()->get_scene_file_path() == String()) {
		return ERR_CANT_CREATE;
	}

	save_scene_as(get_edited_scene_root()->get_scene_file_path());
	return OK;
}

void EditorInterface::save_scene_as(const String &p_scene, bool p_with_preview) {
	EditorNode::get_singleton()->save_scene_to_path(p_scene, p_with_preview);
}

void EditorInterface::set_distraction_free_mode(bool p_enter) {
	EditorNode::get_singleton()->set_distraction_free_mode(p_enter);
}

bool EditorInterface::is_distraction_free_mode_enabled() const {
	return EditorNode::get_singleton()->is_distraction_free_mode_enabled();
}

EditorCommandPalette *EditorInterface::get_command_palette() const {
	return EditorCommandPalette::get_singleton();
}

EditorInterface *EditorInterface::singleton = nullptr;

void EditorInterface::_bind_methods() {
	ClassDB::bind_method(D_METHOD("inspect_object", "object", "for_property", "inspector_only"), &EditorInterface::inspect_object, DEFVAL(String()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_selection"), &EditorInterface::get_selection);
	ClassDB::bind_method(D_METHOD("get_editor_settings"), &EditorInterface::get_editor_settings);
	ClassDB::bind_method(D_METHOD("get_script_editor"), &EditorInterface::get_script_editor);
	ClassDB::bind_method(D_METHOD("get_base_control"), &EditorInterface::get_base_control);
	ClassDB::bind_method(D_METHOD("get_editor_scale"), &EditorInterface::get_editor_scale);
	ClassDB::bind_method(D_METHOD("edit_resource", "resource"), &EditorInterface::edit_resource);
	ClassDB::bind_method(D_METHOD("edit_node", "node"), &EditorInterface::edit_node);
	ClassDB::bind_method(D_METHOD("edit_script", "script", "line", "column", "grab_focus"), &EditorInterface::edit_script, DEFVAL(-1), DEFVAL(0), DEFVAL(true));
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
	ClassDB::bind_method(D_METHOD("get_file_system_dock"), &EditorInterface::get_file_system_dock);
	ClassDB::bind_method(D_METHOD("get_editor_paths"), &EditorInterface::get_editor_paths);
	ClassDB::bind_method(D_METHOD("get_command_palette"), &EditorInterface::get_command_palette);

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

EditorInterface::EditorInterface() {
	singleton = this;
}

///////////////////////////////////////////
void EditorPlugin::add_custom_type(const String &p_type, const String &p_base, const Ref<Script> &p_script, const Ref<Texture2D> &p_icon) {
	EditorNode::get_editor_data().add_custom_type(p_type, p_base, p_script, p_icon);
}

void EditorPlugin::remove_custom_type(const String &p_type) {
	EditorNode::get_editor_data().remove_custom_type(p_type);
}

void EditorPlugin::add_autoload_singleton(const String &p_name, const String &p_path) {
	EditorNode::get_singleton()->get_project_settings()->get_autoload_settings()->autoload_add(p_name, p_path);
}

void EditorPlugin::remove_autoload_singleton(const String &p_name) {
	EditorNode::get_singleton()->get_project_settings()->get_autoload_settings()->autoload_remove(p_name);
}

Button *EditorPlugin::add_control_to_bottom_panel(Control *p_control, const String &p_title) {
	ERR_FAIL_NULL_V(p_control, nullptr);
	return EditorNode::get_singleton()->add_bottom_panel_item(p_title, p_control);
}

void EditorPlugin::add_control_to_dock(DockSlot p_slot, Control *p_control) {
	ERR_FAIL_NULL(p_control);
	EditorNode::get_singleton()->add_control_to_dock(EditorNode::DockSlot(p_slot), p_control);
}

void EditorPlugin::remove_control_from_docks(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	EditorNode::get_singleton()->remove_control_from_dock(p_control);
}

void EditorPlugin::remove_control_from_bottom_panel(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	EditorNode::get_singleton()->remove_bottom_panel_item(p_control);
}

void EditorPlugin::add_control_to_container(CustomControlContainer p_location, Control *p_control) {
	ERR_FAIL_NULL(p_control);

	switch (p_location) {
		case CONTAINER_TOOLBAR: {
			EditorNode::get_menu_hb()->add_child(p_control);
		} break;

		case CONTAINER_SPATIAL_EDITOR_MENU: {
			Node3DEditor::get_singleton()->add_control_to_menu_panel(p_control);

		} break;
		case CONTAINER_SPATIAL_EDITOR_SIDE_LEFT: {
			Node3DEditor::get_singleton()->get_palette_split()->add_child(p_control);
			Node3DEditor::get_singleton()->get_palette_split()->move_child(p_control, 0);

		} break;
		case CONTAINER_SPATIAL_EDITOR_SIDE_RIGHT: {
			Node3DEditor::get_singleton()->get_palette_split()->add_child(p_control);
			Node3DEditor::get_singleton()->get_palette_split()->move_child(p_control, 1);

		} break;
		case CONTAINER_SPATIAL_EDITOR_BOTTOM: {
			Node3DEditor::get_singleton()->get_shader_split()->add_child(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_MENU: {
			CanvasItemEditor::get_singleton()->add_control_to_menu_panel(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_SIDE_LEFT: {
			CanvasItemEditor::get_singleton()->get_palette_split()->add_child(p_control);
			CanvasItemEditor::get_singleton()->get_palette_split()->move_child(p_control, 0);

		} break;
		case CONTAINER_CANVAS_EDITOR_SIDE_RIGHT: {
			CanvasItemEditor::get_singleton()->get_palette_split()->add_child(p_control);
			CanvasItemEditor::get_singleton()->get_palette_split()->move_child(p_control, 1);

		} break;
		case CONTAINER_CANVAS_EDITOR_BOTTOM: {
			CanvasItemEditor::get_singleton()->get_bottom_split()->add_child(p_control);

		} break;
		case CONTAINER_PROPERTY_EDITOR_BOTTOM: {
			EditorNode::get_singleton()->get_inspector_dock_addon_area()->add_child(p_control);

		} break;
		case CONTAINER_PROJECT_SETTING_TAB_LEFT: {
			ProjectSettingsEditor::get_singleton()->get_tabs()->add_child(p_control);
			ProjectSettingsEditor::get_singleton()->get_tabs()->move_child(p_control, 0);

		} break;
		case CONTAINER_PROJECT_SETTING_TAB_RIGHT: {
			ProjectSettingsEditor::get_singleton()->get_tabs()->add_child(p_control);
			ProjectSettingsEditor::get_singleton()->get_tabs()->move_child(p_control, 1);

		} break;
	}
}

void EditorPlugin::remove_control_from_container(CustomControlContainer p_location, Control *p_control) {
	ERR_FAIL_NULL(p_control);

	switch (p_location) {
		case CONTAINER_TOOLBAR: {
			EditorNode::get_menu_hb()->remove_child(p_control);
		} break;

		case CONTAINER_SPATIAL_EDITOR_MENU: {
			Node3DEditor::get_singleton()->remove_control_from_menu_panel(p_control);

		} break;
		case CONTAINER_SPATIAL_EDITOR_SIDE_LEFT:
		case CONTAINER_SPATIAL_EDITOR_SIDE_RIGHT: {
			Node3DEditor::get_singleton()->get_palette_split()->remove_child(p_control);

		} break;
		case CONTAINER_SPATIAL_EDITOR_BOTTOM: {
			Node3DEditor::get_singleton()->get_shader_split()->remove_child(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_MENU: {
			CanvasItemEditor::get_singleton()->remove_control_from_menu_panel(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_SIDE_LEFT:
		case CONTAINER_CANVAS_EDITOR_SIDE_RIGHT: {
			CanvasItemEditor::get_singleton()->get_palette_split()->remove_child(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_BOTTOM: {
			CanvasItemEditor::get_singleton()->get_bottom_split()->remove_child(p_control);

		} break;
		case CONTAINER_PROPERTY_EDITOR_BOTTOM: {
			EditorNode::get_singleton()->get_inspector_dock_addon_area()->remove_child(p_control);

		} break;
		case CONTAINER_PROJECT_SETTING_TAB_LEFT:
		case CONTAINER_PROJECT_SETTING_TAB_RIGHT: {
			ProjectSettingsEditor::get_singleton()->get_tabs()->remove_child(p_control);

		} break;
	}
}

void EditorPlugin::add_tool_menu_item(const String &p_name, const Callable &p_callable) {
	EditorNode::get_singleton()->add_tool_menu_item(p_name, p_callable);
}

void EditorPlugin::add_tool_submenu_item(const String &p_name, Object *p_submenu) {
	ERR_FAIL_NULL(p_submenu);
	PopupMenu *submenu = Object::cast_to<PopupMenu>(p_submenu);
	ERR_FAIL_NULL(submenu);
	EditorNode::get_singleton()->add_tool_submenu_item(p_name, submenu);
}

void EditorPlugin::remove_tool_menu_item(const String &p_name) {
	EditorNode::get_singleton()->remove_tool_menu_item(p_name);
}

void EditorPlugin::set_input_event_forwarding_always_enabled() {
	input_event_forwarding_always_enabled = true;
	EditorPluginList *always_input_forwarding_list = EditorNode::get_singleton()->get_editor_plugins_force_input_forwarding();
	always_input_forwarding_list->add_plugin(this);
}

void EditorPlugin::set_force_draw_over_forwarding_enabled() {
	force_draw_over_forwarding_enabled = true;
	EditorPluginList *always_draw_over_forwarding_list = EditorNode::get_singleton()->get_editor_plugins_force_over();
	always_draw_over_forwarding_list->add_plugin(this);
}

void EditorPlugin::notify_scene_changed(const Node *scn_root) {
	emit_signal(SNAME("scene_changed"), scn_root);
}

void EditorPlugin::notify_main_screen_changed(const String &screen_name) {
	if (screen_name == last_main_screen_name) {
		return;
	}

	emit_signal(SNAME("main_screen_changed"), screen_name);
	last_main_screen_name = screen_name;
}

void EditorPlugin::notify_scene_closed(const String &scene_filepath) {
	emit_signal(SNAME("scene_closed"), scene_filepath);
}

void EditorPlugin::notify_resource_saved(const Ref<Resource> &p_resource) {
	emit_signal(SNAME("resource_saved"), p_resource);
}

bool EditorPlugin::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	bool success;
	if (GDVIRTUAL_CALL(_forward_canvas_gui_input, p_event, success)) {
		return success;
	}
	return false;
}

void EditorPlugin::forward_canvas_draw_over_viewport(Control *p_overlay) {
	GDVIRTUAL_CALL(_forward_canvas_draw_over_viewport, p_overlay);
}

void EditorPlugin::forward_canvas_force_draw_over_viewport(Control *p_overlay) {
	GDVIRTUAL_CALL(_forward_canvas_force_draw_over_viewport, p_overlay);
}

// Updates the overlays of the 2D viewport or, if in 3D mode, of every 3D viewport.
int EditorPlugin::update_overlays() const {
	if (Node3DEditor::get_singleton()->is_visible()) {
		int count = 0;
		for (uint32_t i = 0; i < Node3DEditor::VIEWPORTS_COUNT; i++) {
			Node3DEditorViewport *vp = Node3DEditor::get_singleton()->get_editor_viewport(i);
			if (vp->is_visible()) {
				vp->update_surface();
				count++;
			}
		}
		return count;
	} else {
		// This will update the normal viewport itself as well
		CanvasItemEditor::get_singleton()->get_viewport_control()->update();
		return 1;
	}
}

EditorPlugin::AfterGUIInput EditorPlugin::forward_spatial_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) {
	int success;

	if (GDVIRTUAL_CALL(_forward_3d_gui_input, p_camera, p_event, success)) {
		return static_cast<EditorPlugin::AfterGUIInput>(success);
	}

	return EditorPlugin::AFTER_GUI_INPUT_PASS;
}

void EditorPlugin::forward_spatial_draw_over_viewport(Control *p_overlay) {
	GDVIRTUAL_CALL(_forward_3d_draw_over_viewport, p_overlay);
}

void EditorPlugin::forward_spatial_force_draw_over_viewport(Control *p_overlay) {
	GDVIRTUAL_CALL(_forward_3d_force_draw_over_viewport, p_overlay);
}

String EditorPlugin::get_name() const {
	String name;
	if (GDVIRTUAL_CALL(_get_plugin_name, name)) {
		return name;
	}

	return String();
}

const Ref<Texture2D> EditorPlugin::get_icon() const {
	Ref<Texture2D> icon;
	if (GDVIRTUAL_CALL(_get_plugin_icon, icon)) {
		return icon;
	}

	return Ref<Texture2D>();
}

bool EditorPlugin::has_main_screen() const {
	bool success;
	if (GDVIRTUAL_CALL(_has_main_screen, success)) {
		return success;
	}

	return false;
}

void EditorPlugin::make_visible(bool p_visible) {
	GDVIRTUAL_CALL(_make_visible, p_visible);
}

void EditorPlugin::edit(Object *p_object) {
	if (p_object->is_class("Resource")) {
		GDVIRTUAL_CALL(_edit, Ref<Resource>(Object::cast_to<Resource>(p_object)));
	} else {
		GDVIRTUAL_CALL(_edit, p_object);
	}
}

bool EditorPlugin::handles(Object *p_object) const {
	bool success;
	if (GDVIRTUAL_CALL(_handles, p_object, success)) {
		return success;
	}

	return false;
}

Dictionary EditorPlugin::get_state() const {
	Dictionary state;
	if (GDVIRTUAL_CALL(_get_state, state)) {
		return state;
	}

	return Dictionary();
}

void EditorPlugin::set_state(const Dictionary &p_state) {
	GDVIRTUAL_CALL(_set_state, p_state);
}

void EditorPlugin::clear() {
	GDVIRTUAL_CALL(_clear);
}

// if editor references external resources/scenes, save them
void EditorPlugin::save_external_data() {
	GDVIRTUAL_CALL(_save_external_data);
}

// if changes are pending in editor, apply them
void EditorPlugin::apply_changes() {
	GDVIRTUAL_CALL(_apply_changes);
}

void EditorPlugin::get_breakpoints(List<String> *p_breakpoints) {
	PackedStringArray arr;
	if (GDVIRTUAL_CALL(_get_breakpoints, arr)) {
		for (int i = 0; i < arr.size(); i++) {
			p_breakpoints->push_back(arr[i]);
		}
	}
}

bool EditorPlugin::get_remove_list(List<Node *> *p_list) {
	return false;
}

void EditorPlugin::restore_global_state() {}
void EditorPlugin::save_global_state() {}

void EditorPlugin::add_undo_redo_inspector_hook_callback(Callable p_callable) {
	EditorNode::get_singleton()->get_editor_data().add_undo_redo_inspector_hook_callback(p_callable);
}

void EditorPlugin::remove_undo_redo_inspector_hook_callback(Callable p_callable) {
	EditorNode::get_singleton()->get_editor_data().remove_undo_redo_inspector_hook_callback(p_callable);
}

void EditorPlugin::add_translation_parser_plugin(const Ref<EditorTranslationParserPlugin> &p_parser) {
	ERR_FAIL_COND(!p_parser.is_valid());
	EditorTranslationParser::get_singleton()->add_parser(p_parser, EditorTranslationParser::CUSTOM);
}

void EditorPlugin::remove_translation_parser_plugin(const Ref<EditorTranslationParserPlugin> &p_parser) {
	ERR_FAIL_COND(!p_parser.is_valid());
	EditorTranslationParser::get_singleton()->remove_parser(p_parser, EditorTranslationParser::CUSTOM);
}

void EditorPlugin::add_import_plugin(const Ref<EditorImportPlugin> &p_importer) {
	ERR_FAIL_COND(!p_importer.is_valid());
	ResourceFormatImporter::get_singleton()->add_importer(p_importer);
	EditorFileSystem::get_singleton()->call_deferred(SNAME("scan"));
}

void EditorPlugin::remove_import_plugin(const Ref<EditorImportPlugin> &p_importer) {
	ERR_FAIL_COND(!p_importer.is_valid());
	ResourceFormatImporter::get_singleton()->remove_importer(p_importer);
	EditorFileSystem::get_singleton()->call_deferred(SNAME("scan"));
}

void EditorPlugin::add_export_plugin(const Ref<EditorExportPlugin> &p_exporter) {
	ERR_FAIL_COND(!p_exporter.is_valid());
	EditorExport::get_singleton()->add_export_plugin(p_exporter);
}

void EditorPlugin::remove_export_plugin(const Ref<EditorExportPlugin> &p_exporter) {
	ERR_FAIL_COND(!p_exporter.is_valid());
	EditorExport::get_singleton()->remove_export_plugin(p_exporter);
}

void EditorPlugin::add_spatial_gizmo_plugin(const Ref<EditorNode3DGizmoPlugin> &p_gizmo_plugin) {
	ERR_FAIL_COND(!p_gizmo_plugin.is_valid());
	Node3DEditor::get_singleton()->add_gizmo_plugin(p_gizmo_plugin);
}

void EditorPlugin::remove_spatial_gizmo_plugin(const Ref<EditorNode3DGizmoPlugin> &p_gizmo_plugin) {
	ERR_FAIL_COND(!p_gizmo_plugin.is_valid());
	Node3DEditor::get_singleton()->remove_gizmo_plugin(p_gizmo_plugin);
}

void EditorPlugin::add_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin) {
	ERR_FAIL_COND(!p_plugin.is_valid());
	EditorInspector::add_inspector_plugin(p_plugin);
}

void EditorPlugin::remove_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin) {
	ERR_FAIL_COND(!p_plugin.is_valid());
	EditorInspector::remove_inspector_plugin(p_plugin);
}

void EditorPlugin::add_scene_format_importer_plugin(const Ref<EditorSceneFormatImporter> &p_importer) {
	ERR_FAIL_COND(!p_importer.is_valid());
	ResourceImporterScene::get_singleton()->add_importer(p_importer);
}

void EditorPlugin::remove_scene_format_importer_plugin(const Ref<EditorSceneFormatImporter> &p_importer) {
	ERR_FAIL_COND(!p_importer.is_valid());
	ResourceImporterScene::get_singleton()->remove_importer(p_importer);
}

void EditorPlugin::add_scene_post_import_plugin(const Ref<EditorScenePostImportPlugin> &p_plugin) {
	ResourceImporterScene::get_singleton()->add_post_importer_plugin(p_plugin);
}
void EditorPlugin::remove_scene_post_import_plugin(const Ref<EditorScenePostImportPlugin> &p_plugin) {
	ResourceImporterScene::get_singleton()->remove_post_importer_plugin(p_plugin);
}

int find(const PackedStringArray &a, const String &v) {
	const String *r = a.ptr();
	for (int j = 0; j < a.size(); ++j) {
		if (r[j] == v) {
			return j;
		}
	}
	return -1;
}

void EditorPlugin::enable_plugin() {
	// Called when the plugin gets enabled in project settings, after it's added to the tree.
	// You can implement it to register autoloads.
	GDVIRTUAL_CALL(_enable_plugin);
}

void EditorPlugin::disable_plugin() {
	// Last function called when the plugin gets disabled in project settings.
	// Implement it to cleanup things from the project, such as unregister autoloads.
	GDVIRTUAL_CALL(_disable_plugin);
}

void EditorPlugin::set_window_layout(Ref<ConfigFile> p_layout) {
	GDVIRTUAL_CALL(_set_window_layout, p_layout);
}

void EditorPlugin::get_window_layout(Ref<ConfigFile> p_layout) {
	GDVIRTUAL_CALL(_get_window_layout, p_layout);
}

bool EditorPlugin::build() {
	bool success;
	if (GDVIRTUAL_CALL(_build, success)) {
		return success;
	}
	return true;
}

void EditorPlugin::queue_save_layout() {
	EditorNode::get_singleton()->save_layout();
}

void EditorPlugin::make_bottom_panel_item_visible(Control *p_item) {
	EditorNode::get_singleton()->make_bottom_panel_item_visible(p_item);
}

void EditorPlugin::hide_bottom_panel() {
	EditorNode::get_singleton()->hide_bottom_panel();
}

EditorInterface *EditorPlugin::get_editor_interface() {
	return EditorInterface::get_singleton();
}

ScriptCreateDialog *EditorPlugin::get_script_create_dialog() {
	return EditorNode::get_singleton()->get_script_create_dialog();
}

void EditorPlugin::add_debugger_plugin(const Ref<Script> &p_script) {
	EditorDebuggerNode::get_singleton()->add_debugger_plugin(p_script);
}

void EditorPlugin::remove_debugger_plugin(const Ref<Script> &p_script) {
	EditorDebuggerNode::get_singleton()->remove_debugger_plugin(p_script);
}

void EditorPlugin::_editor_project_settings_changed() {
	emit_signal(SNAME("project_settings_changed"));
}
void EditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		EditorNode::get_singleton()->connect("project_settings_changed", callable_mp(this, &EditorPlugin::_editor_project_settings_changed));
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {
		EditorNode::get_singleton()->disconnect("project_settings_changed", callable_mp(this, &EditorPlugin::_editor_project_settings_changed));
	}
}

void EditorPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_control_to_container", "container", "control"), &EditorPlugin::add_control_to_container);
	ClassDB::bind_method(D_METHOD("add_control_to_bottom_panel", "control", "title"), &EditorPlugin::add_control_to_bottom_panel);
	ClassDB::bind_method(D_METHOD("add_control_to_dock", "slot", "control"), &EditorPlugin::add_control_to_dock);
	ClassDB::bind_method(D_METHOD("remove_control_from_docks", "control"), &EditorPlugin::remove_control_from_docks);
	ClassDB::bind_method(D_METHOD("remove_control_from_bottom_panel", "control"), &EditorPlugin::remove_control_from_bottom_panel);
	ClassDB::bind_method(D_METHOD("remove_control_from_container", "container", "control"), &EditorPlugin::remove_control_from_container);
	ClassDB::bind_method(D_METHOD("add_tool_menu_item", "name", "callable"), &EditorPlugin::add_tool_menu_item);
	ClassDB::bind_method(D_METHOD("add_tool_submenu_item", "name", "submenu"), &EditorPlugin::add_tool_submenu_item);
	ClassDB::bind_method(D_METHOD("remove_tool_menu_item", "name"), &EditorPlugin::remove_tool_menu_item);
	ClassDB::bind_method(D_METHOD("add_custom_type", "type", "base", "script", "icon"), &EditorPlugin::add_custom_type);
	ClassDB::bind_method(D_METHOD("remove_custom_type", "type"), &EditorPlugin::remove_custom_type);

	ClassDB::bind_method(D_METHOD("add_autoload_singleton", "name", "path"), &EditorPlugin::add_autoload_singleton);
	ClassDB::bind_method(D_METHOD("remove_autoload_singleton", "name"), &EditorPlugin::remove_autoload_singleton);

	ClassDB::bind_method(D_METHOD("update_overlays"), &EditorPlugin::update_overlays);

	ClassDB::bind_method(D_METHOD("make_bottom_panel_item_visible", "item"), &EditorPlugin::make_bottom_panel_item_visible);
	ClassDB::bind_method(D_METHOD("hide_bottom_panel"), &EditorPlugin::hide_bottom_panel);

	ClassDB::bind_method(D_METHOD("get_undo_redo"), &EditorPlugin::_get_undo_redo);
	ClassDB::bind_method(D_METHOD("add_undo_redo_inspector_hook_callback", "callable"), &EditorPlugin::add_undo_redo_inspector_hook_callback);
	ClassDB::bind_method(D_METHOD("remove_undo_redo_inspector_hook_callback", "callable"), &EditorPlugin::remove_undo_redo_inspector_hook_callback);
	ClassDB::bind_method(D_METHOD("queue_save_layout"), &EditorPlugin::queue_save_layout);
	ClassDB::bind_method(D_METHOD("add_translation_parser_plugin", "parser"), &EditorPlugin::add_translation_parser_plugin);
	ClassDB::bind_method(D_METHOD("remove_translation_parser_plugin", "parser"), &EditorPlugin::remove_translation_parser_plugin);
	ClassDB::bind_method(D_METHOD("add_import_plugin", "importer"), &EditorPlugin::add_import_plugin);
	ClassDB::bind_method(D_METHOD("remove_import_plugin", "importer"), &EditorPlugin::remove_import_plugin);
	ClassDB::bind_method(D_METHOD("add_scene_format_importer_plugin", "scene_format_importer"), &EditorPlugin::add_scene_format_importer_plugin);
	ClassDB::bind_method(D_METHOD("remove_scene_format_importer_plugin", "scene_format_importer"), &EditorPlugin::remove_scene_format_importer_plugin);
	ClassDB::bind_method(D_METHOD("add_scene_post_import_plugin", "scene_import_plugin"), &EditorPlugin::add_scene_post_import_plugin);
	ClassDB::bind_method(D_METHOD("remove_scene_post_import_plugin", "scene_import_plugin"), &EditorPlugin::remove_scene_post_import_plugin);
	ClassDB::bind_method(D_METHOD("add_export_plugin", "plugin"), &EditorPlugin::add_export_plugin);
	ClassDB::bind_method(D_METHOD("remove_export_plugin", "plugin"), &EditorPlugin::remove_export_plugin);
	ClassDB::bind_method(D_METHOD("add_spatial_gizmo_plugin", "plugin"), &EditorPlugin::add_spatial_gizmo_plugin);
	ClassDB::bind_method(D_METHOD("remove_spatial_gizmo_plugin", "plugin"), &EditorPlugin::remove_spatial_gizmo_plugin);
	ClassDB::bind_method(D_METHOD("add_inspector_plugin", "plugin"), &EditorPlugin::add_inspector_plugin);
	ClassDB::bind_method(D_METHOD("remove_inspector_plugin", "plugin"), &EditorPlugin::remove_inspector_plugin);
	ClassDB::bind_method(D_METHOD("set_input_event_forwarding_always_enabled"), &EditorPlugin::set_input_event_forwarding_always_enabled);
	ClassDB::bind_method(D_METHOD("set_force_draw_over_forwarding_enabled"), &EditorPlugin::set_force_draw_over_forwarding_enabled);

	ClassDB::bind_method(D_METHOD("get_editor_interface"), &EditorPlugin::get_editor_interface);
	ClassDB::bind_method(D_METHOD("get_script_create_dialog"), &EditorPlugin::get_script_create_dialog);
	ClassDB::bind_method(D_METHOD("add_debugger_plugin", "script"), &EditorPlugin::add_debugger_plugin);
	ClassDB::bind_method(D_METHOD("remove_debugger_plugin", "script"), &EditorPlugin::remove_debugger_plugin);

	GDVIRTUAL_BIND(_forward_canvas_gui_input, "event");
	GDVIRTUAL_BIND(_forward_canvas_draw_over_viewport, "viewport_control");
	GDVIRTUAL_BIND(_forward_canvas_force_draw_over_viewport, "viewport_control");
	GDVIRTUAL_BIND(_forward_3d_gui_input, "viewport_camera", "event");
	GDVIRTUAL_BIND(_forward_3d_draw_over_viewport, "viewport_control");
	GDVIRTUAL_BIND(_forward_3d_force_draw_over_viewport, "viewport_control");
	GDVIRTUAL_BIND(_get_plugin_name);
	GDVIRTUAL_BIND(_get_plugin_icon);
	GDVIRTUAL_BIND(_has_main_screen);
	GDVIRTUAL_BIND(_make_visible, "visible");
	GDVIRTUAL_BIND(_edit, "object");
	GDVIRTUAL_BIND(_handles, "object");
	GDVIRTUAL_BIND(_get_state);
	GDVIRTUAL_BIND(_set_state, "state");
	GDVIRTUAL_BIND(_clear);
	GDVIRTUAL_BIND(_save_external_data);
	GDVIRTUAL_BIND(_apply_changes);
	GDVIRTUAL_BIND(_get_breakpoints);
	GDVIRTUAL_BIND(_set_window_layout, "configuration");
	GDVIRTUAL_BIND(_get_window_layout, "configuration");
	GDVIRTUAL_BIND(_build);
	GDVIRTUAL_BIND(_enable_plugin);
	GDVIRTUAL_BIND(_disable_plugin);

	ADD_SIGNAL(MethodInfo("scene_changed", PropertyInfo(Variant::OBJECT, "scene_root", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("scene_closed", PropertyInfo(Variant::STRING, "filepath")));
	ADD_SIGNAL(MethodInfo("main_screen_changed", PropertyInfo(Variant::STRING, "screen_name")));
	ADD_SIGNAL(MethodInfo("resource_saved", PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource")));
	ADD_SIGNAL(MethodInfo("project_settings_changed"));

	BIND_ENUM_CONSTANT(CONTAINER_TOOLBAR);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_MENU);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_SIDE_LEFT);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_SIDE_RIGHT);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_BOTTOM);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_MENU);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_SIDE_LEFT);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_SIDE_RIGHT);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_BOTTOM);
	BIND_ENUM_CONSTANT(CONTAINER_PROPERTY_EDITOR_BOTTOM);
	BIND_ENUM_CONSTANT(CONTAINER_PROJECT_SETTING_TAB_LEFT);
	BIND_ENUM_CONSTANT(CONTAINER_PROJECT_SETTING_TAB_RIGHT);

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

EditorPluginCreateFunc EditorPlugins::creation_funcs[MAX_CREATE_FUNCS];

int EditorPlugins::creation_func_count = 0;
