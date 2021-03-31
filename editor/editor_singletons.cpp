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

#include "editor/editor_node.h"
#include "editor/import_dock.h"
#include "editor/node_dock.h"
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
	EditorNode::get_singleton()->select_editor_by_name(p_name);
}

Control *EditorInterface::get_editor_main_control() {
	return EditorNode::get_singleton()->get_main_control();
}

void EditorInterface::edit_resource(const Ref<Resource> &p_resource) {
	EditorNode::get_singleton()->edit_resource(p_resource);
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
		ret.push_back(scenes[idx_scn].root->get_filename());
	}
	return ret;
}

ScriptEditor *EditorInterface::get_script_editor() {
	return ScriptEditor::get_singleton();
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
	EditorNode::get_singleton()->push_item(p_obj, p_for_property, p_inspector_only);
}

EditorFileSystem *EditorInterface::get_resource_file_system() {
	return EditorFileSystem::get_singleton();
}

EditorSelection *EditorInterface::get_selection() {
	return EditorNode::get_singleton()->get_editor_selection();
}

Ref<EditorSettings> EditorInterface::get_editor_settings() {
	return EditorSettings::get_singleton();
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
	if (get_edited_scene_root()->get_filename() == String()) {
		return ERR_CANT_CREATE;
	}

	save_scene_as(get_edited_scene_root()->get_filename());
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

EditorInterface *EditorInterface::singleton = nullptr;

void EditorInterface::_bind_methods() {
	ClassDB::bind_method(D_METHOD("inspect_object", "object", "for_property", "inspector_only"), &EditorInterface::inspect_object, DEFVAL(String()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_selection"), &EditorInterface::get_selection);
	ClassDB::bind_method(D_METHOD("get_editor_settings"), &EditorInterface::get_editor_settings);
	ClassDB::bind_method(D_METHOD("get_script_editor"), &EditorInterface::get_script_editor);
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

EditorInterface::EditorInterface() {
	singleton = this;
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

void EditorDocks::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_scene_tree_dock"), &EditorDocks::get_scene_tree_dock);
	ClassDB::bind_method(D_METHOD("get_filesystem_dock"), &EditorDocks::get_filesystem_dock);
	ClassDB::bind_method(D_METHOD("get_node_dock"), &EditorDocks::get_node_dock);
	ClassDB::bind_method(D_METHOD("get_connection_dock"), &EditorDocks::get_connection_dock);
	ClassDB::bind_method(D_METHOD("get_inspector_dock"), &EditorDocks::get_inspector_dock);
	ClassDB::bind_method(D_METHOD("get_import_dock"), &EditorDocks::get_import_dock);
}

EditorDocks *EditorDocks::singleton = nullptr;

EditorDocks::EditorDocks() {
	singleton = this;
}
