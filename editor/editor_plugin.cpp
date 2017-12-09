/*************************************************************************/
/*  editor_plugin.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor_resource_preview.h"
#include "main/main.h"
#include "plugins/canvas_item_editor_plugin.h"
#include "plugins/spatial_editor_plugin.h"
#include "scene/3d/camera.h"
#include "scene/gui/popup_menu.h"
#include "servers/visual_server.h"
Array EditorInterface::_make_mesh_previews(const Array &p_meshes, int p_preview_size) {

	Vector<Ref<Mesh> > meshes;

	for (int i = 0; i < p_meshes.size(); i++) {
		meshes.push_back(p_meshes[i]);
	}

	Vector<Ref<Texture> > textures = make_mesh_previews(meshes, p_preview_size);
	Array ret;
	for (int i = 0; i < textures.size(); i++) {
		ret.push_back(textures[i]);
	}

	return ret;
}

Vector<Ref<Texture> > EditorInterface::make_mesh_previews(const Vector<Ref<Mesh> > &p_meshes, int p_preview_size) {

	int size = p_preview_size;

	RID scenario = VS::get_singleton()->scenario_create();

	RID viewport = VS::get_singleton()->viewport_create();
	VS::get_singleton()->viewport_set_update_mode(viewport, VS::VIEWPORT_UPDATE_ALWAYS);
	VS::get_singleton()->viewport_set_vflip(viewport, true);
	VS::get_singleton()->viewport_set_scenario(viewport, scenario);
	VS::get_singleton()->viewport_set_size(viewport, size, size);
	VS::get_singleton()->viewport_set_transparent_background(viewport, true);
	VS::get_singleton()->viewport_set_active(viewport, true);
	RID viewport_texture = VS::get_singleton()->viewport_get_texture(viewport);

	RID camera = VS::get_singleton()->camera_create();
	VS::get_singleton()->viewport_attach_camera(viewport, camera);
	VS::get_singleton()->camera_set_transform(camera, Transform(Basis(), Vector3(0, 0, 3)));
	//VS::get_singleton()->camera_set_perspective(camera,45,0.1,10);
	VS::get_singleton()->camera_set_orthogonal(camera, 1.0, 0.01, 1000.0);

	RID light = VS::get_singleton()->directional_light_create();
	RID light_instance = VS::get_singleton()->instance_create2(light, scenario);
	VS::get_singleton()->instance_set_transform(light_instance, Transform().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));

	RID light2 = VS::get_singleton()->directional_light_create();
	VS::get_singleton()->light_set_color(light2, Color(0.7, 0.7, 0.7));
	//VS::get_singleton()->light_set_color(light2, VS::LIGHT_COLOR_SPECULAR, Color(0.0, 0.0, 0.0));
	RID light_instance2 = VS::get_singleton()->instance_create2(light2, scenario);

	VS::get_singleton()->instance_set_transform(light_instance2, Transform().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));

	//sphere = VS::get_singleton()->mesh_create();
	RID mesh_instance = VS::get_singleton()->instance_create();
	VS::get_singleton()->instance_set_scenario(mesh_instance, scenario);

	EditorProgress ep("mlib", TTR("Creating Mesh Previews"), p_meshes.size());

	Vector<Ref<Texture> > textures;

	for (int i = 0; i < p_meshes.size(); i++) {

		Ref<Mesh> mesh = p_meshes[i];
		if (!mesh.is_valid()) {
			textures.push_back(Ref<Texture>());
			continue;
		}
		AABB aabb = mesh->get_aabb();
		print_line("aabb: " + aabb);
		Vector3 ofs = aabb.position + aabb.size * 0.5;
		aabb.position -= ofs;
		Transform xform;
		xform.basis = Basis().rotated(Vector3(0, 1, 0), -Math_PI * 0.25);
		xform.basis = Basis().rotated(Vector3(1, 0, 0), Math_PI * 0.25) * xform.basis;
		AABB rot_aabb = xform.xform(aabb);
		print_line("rot_aabb: " + rot_aabb);
		float m = MAX(rot_aabb.size.x, rot_aabb.size.y) * 0.5;
		if (m == 0) {
			textures.push_back(Ref<Texture>());
			continue;
		}
		m = 1.0 / m;
		m *= 0.5;
		print_line("scale: " + rtos(m));
		xform.basis.scale(Vector3(m, m, m));
		xform.origin = -xform.basis.xform(ofs); //-ofs*m;
		xform.origin.z -= rot_aabb.size.z * 2;
		RID inst = VS::get_singleton()->instance_create2(mesh->get_rid(), scenario);
		VS::get_singleton()->instance_set_transform(inst, xform);
		ep.step(TTR("Thumbnail.."), i);
		Main::iteration();
		Main::iteration();
		Ref<Image> img = VS::get_singleton()->texture_get_data(viewport_texture);
		ERR_CONTINUE(!img.is_valid() || img->empty());
		Ref<ImageTexture> it(memnew(ImageTexture));
		it->create_from_image(img);

		//print_line("loaded image, size: "+rtos(m)+" dist: "+rtos(dist)+" empty?"+itos(img.empty())+" w: "+itos(it->get_width())+" h: "+itos(it->get_height()));
		VS::get_singleton()->free(inst);

		textures.push_back(it);
	}

	VS::get_singleton()->free(mesh_instance);
	VS::get_singleton()->free(viewport);
	VS::get_singleton()->free(light);
	VS::get_singleton()->free(light_instance);
	VS::get_singleton()->free(light2);
	VS::get_singleton()->free(light_instance2);
	VS::get_singleton()->free(camera);
	VS::get_singleton()->free(scenario);

	return textures;
}

Control *EditorInterface::get_editor_viewport() {

	return EditorNode::get_singleton()->get_viewport();
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

Node *EditorInterface::get_edited_scene_root() {
	return EditorNode::get_singleton()->get_edited_scene();
}

Array EditorInterface::get_open_scenes() const {

	Array ret;
	Vector<EditorData::EditedScene> scenes = EditorNode::get_singleton()->get_editor_data().get_edited_scenes();

	int scns_amount = scenes.size();
	for (int idx_scn = 0; idx_scn < scns_amount; idx_scn++) {
		if (scenes[idx_scn].root == NULL)
			continue;
		ret.push_back(scenes[idx_scn].root->get_filename());
	}
	return ret;
}

ScriptEditor *EditorInterface::get_script_editor() {
	return ScriptEditor::get_singleton();
}

void EditorInterface::inspect_object(Object *p_obj, const String &p_for_property) {

	EditorNode::get_singleton()->push_item(p_obj, p_for_property);
}

EditorFileSystem *EditorInterface::get_resource_file_system() {
	return EditorFileSystem::get_singleton();
}

EditorSelection *EditorInterface::get_selection() {
	return EditorNode::get_singleton()->get_editor_selection();
}

EditorSettings *EditorInterface::get_editor_settings() {
	return EditorSettings::get_singleton();
}

EditorResourcePreview *EditorInterface::get_resource_previewer() {
	return EditorResourcePreview::get_singleton();
}

Control *EditorInterface::get_base_control() {

	return EditorNode::get_singleton()->get_gui_base();
}

Error EditorInterface::save_scene() {
	if (!get_edited_scene_root())
		return ERR_CANT_CREATE;
	if (get_edited_scene_root()->get_filename() == String())
		return ERR_CANT_CREATE;

	save_scene_as(get_edited_scene_root()->get_filename());
	return OK;
}

void EditorInterface::save_scene_as(const String &p_scene, bool p_with_preview) {

	EditorNode::get_singleton()->save_scene_to_path(p_scene, p_with_preview);
}

EditorInterface *EditorInterface::singleton = NULL;

void EditorInterface::_bind_methods() {

	ClassDB::bind_method(D_METHOD("inspect_object", "object", "for_property"), &EditorInterface::inspect_object, DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("get_selection"), &EditorInterface::get_selection);
	ClassDB::bind_method(D_METHOD("get_editor_settings"), &EditorInterface::get_editor_settings);
	ClassDB::bind_method(D_METHOD("get_script_editor"), &EditorInterface::get_script_editor);
	ClassDB::bind_method(D_METHOD("get_base_control"), &EditorInterface::get_base_control);
	ClassDB::bind_method(D_METHOD("edit_resource", "resource"), &EditorInterface::edit_resource);
	ClassDB::bind_method(D_METHOD("open_scene_from_path", "scene_filepath"), &EditorInterface::open_scene_from_path);
	ClassDB::bind_method(D_METHOD("reload_scene_from_path", "scene_filepath"), &EditorInterface::reload_scene_from_path);
	ClassDB::bind_method(D_METHOD("get_open_scenes"), &EditorInterface::get_open_scenes);
	ClassDB::bind_method(D_METHOD("get_edited_scene_root"), &EditorInterface::get_edited_scene_root);
	ClassDB::bind_method(D_METHOD("get_resource_previewer"), &EditorInterface::get_resource_previewer);
	ClassDB::bind_method(D_METHOD("get_resource_filesystem"), &EditorInterface::get_resource_file_system);
	ClassDB::bind_method(D_METHOD("get_editor_viewport"), &EditorInterface::get_editor_viewport);
	ClassDB::bind_method(D_METHOD("make_mesh_previews", "meshes", "preview_size"), &EditorInterface::_make_mesh_previews);

	ClassDB::bind_method(D_METHOD("save_scene"), &EditorInterface::save_scene);
	ClassDB::bind_method(D_METHOD("save_scene_as", "path", "with_preview"), &EditorInterface::save_scene_as, DEFVAL(true));
}

EditorInterface::EditorInterface() {
	singleton = this;
}

///////////////////////////////////////////
void EditorPlugin::add_custom_type(const String &p_type, const String &p_base, const Ref<Script> &p_script, const Ref<Texture> &p_icon) {

	EditorNode::get_editor_data().add_custom_type(p_type, p_base, p_script, p_icon);
}

void EditorPlugin::remove_custom_type(const String &p_type) {

	EditorNode::get_editor_data().remove_custom_type(p_type);
}

ToolButton *EditorPlugin::add_control_to_bottom_panel(Control *p_control, const String &p_title) {

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

	switch (p_location) {

		case CONTAINER_TOOLBAR: {

			EditorNode::get_menu_hb()->add_child(p_control);
		} break;

		case CONTAINER_SPATIAL_EDITOR_MENU: {

			SpatialEditor::get_singleton()->add_control_to_menu_panel(p_control);

		} break;
		case CONTAINER_SPATIAL_EDITOR_SIDE: {

			SpatialEditor::get_singleton()->get_palette_split()->add_child(p_control);
			SpatialEditor::get_singleton()->get_palette_split()->move_child(p_control, 0);

		} break;
		case CONTAINER_SPATIAL_EDITOR_BOTTOM: {

			SpatialEditor::get_singleton()->get_shader_split()->add_child(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_MENU: {

			CanvasItemEditor::get_singleton()->add_control_to_menu_panel(p_control);

		} break;
		case CONTAINER_CANVAS_EDITOR_SIDE: {

			CanvasItemEditor::get_singleton()->get_palette_split()->add_child(p_control);
			CanvasItemEditor::get_singleton()->get_palette_split()->move_child(p_control, 0);

		} break;
		case CONTAINER_CANVAS_EDITOR_BOTTOM: {

			CanvasItemEditor::get_singleton()->get_bottom_split()->add_child(p_control);

		} break;
		case CONTAINER_PROPERTY_EDITOR_BOTTOM: {

			EditorNode::get_singleton()->get_property_editor_vb()->add_child(p_control);

		} break;
	}
}

void EditorPlugin::add_tool_menu_item(const String &p_name, Object *p_handler, const String &p_callback, const Variant &p_ud) {

	//EditorNode::get_singleton()->add_tool_menu_item(p_name, p_handler, p_callback, p_ud);
}

void EditorPlugin::add_tool_submenu_item(const String &p_name, Object *p_submenu) {

	ERR_FAIL_NULL(p_submenu);
	PopupMenu *submenu = Object::cast_to<PopupMenu>(p_submenu);
	ERR_FAIL_NULL(submenu);
	//EditorNode::get_singleton()->add_tool_submenu_item(p_name, submenu);
}

void EditorPlugin::remove_tool_menu_item(const String &p_name) {

	//EditorNode::get_singleton()->remove_tool_menu_item(p_name);
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
	if (scn_root == NULL) return;
	emit_signal("scene_changed", scn_root);
}

void EditorPlugin::notify_main_screen_changed(const String &screen_name) {

	if (screen_name == last_main_screen_name)
		return;

	emit_signal("main_screen_changed", screen_name);
	last_main_screen_name = screen_name;
}

void EditorPlugin::notify_scene_closed(const String &scene_filepath) {
	emit_signal("scene_closed", scene_filepath);
}

Ref<SpatialEditorGizmo> EditorPlugin::create_spatial_gizmo(Spatial *p_spatial) {
	//??
	if (get_script_instance() && get_script_instance()->has_method("create_spatial_gizmo")) {
		return get_script_instance()->call("create_spatial_gizmo", p_spatial);
	}

	return Ref<SpatialEditorGizmo>();
}

bool EditorPlugin::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {

	if (get_script_instance() && get_script_instance()->has_method("forward_canvas_gui_input")) {
		return get_script_instance()->call("forward_canvas_gui_input", p_event);
	}
	return false;
}

void EditorPlugin::forward_draw_over_viewport(Control *p_overlay) {

	if (get_script_instance() && get_script_instance()->has_method("forward_draw_over_viewport")) {
		get_script_instance()->call("forward_draw_over_viewport", p_overlay);
	}
}

void EditorPlugin::forward_force_draw_over_viewport(Control *p_overlay) {

	if (get_script_instance() && get_script_instance()->has_method("forward_force_draw_over_viewport")) {
		get_script_instance()->call("forward_force_draw_over_viewport", p_overlay);
	}
}

// Updates the overlays of the 2D viewport or, if in 3D mode, of every 3D viewport.
int EditorPlugin::update_overlays() const {

	if (SpatialEditor::get_singleton()->is_visible()) {
		int count = 0;
		for (int i = 0; i < SpatialEditor::VIEWPORTS_COUNT; i++) {
			SpatialEditorViewport *vp = SpatialEditor::get_singleton()->get_editor_viewport(i);
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

bool EditorPlugin::forward_spatial_gui_input(Camera *p_camera, const Ref<InputEvent> &p_event) {

	if (get_script_instance() && get_script_instance()->has_method("forward_spatial_gui_input")) {
		return get_script_instance()->call("forward_spatial_gui_input", p_camera, p_event);
	}

	return false;
}
String EditorPlugin::get_name() const {

	if (get_script_instance() && get_script_instance()->has_method("get_plugin_name")) {
		return get_script_instance()->call("get_plugin_name");
	}

	return String();
}
const Ref<Texture> EditorPlugin::get_icon() const {

	if (get_script_instance() && get_script_instance()->has_method("get_plugin_icon")) {
		return get_script_instance()->call("get_plugin_icon");
	}

	return Ref<Texture>();
}
bool EditorPlugin::has_main_screen() const {

	if (get_script_instance() && get_script_instance()->has_method("has_main_screen")) {
		return get_script_instance()->call("has_main_screen");
	}

	return false;
}
void EditorPlugin::make_visible(bool p_visible) {

	if (get_script_instance() && get_script_instance()->has_method("make_visible")) {
		get_script_instance()->call("make_visible", p_visible);
	}
}

void EditorPlugin::edit(Object *p_object) {

	if (get_script_instance() && get_script_instance()->has_method("edit")) {
		if (p_object->is_class("Resource")) {
			get_script_instance()->call("edit", Ref<Resource>(Object::cast_to<Resource>(p_object)));
		} else {
			get_script_instance()->call("edit", p_object);
		}
	}
}

bool EditorPlugin::handles(Object *p_object) const {

	if (get_script_instance() && get_script_instance()->has_method("handles")) {
		return get_script_instance()->call("handles", p_object);
	}

	return false;
}
Dictionary EditorPlugin::get_state() const {

	if (get_script_instance() && get_script_instance()->has_method("get_state")) {
		return get_script_instance()->call("get_state");
	}

	return Dictionary();
}

void EditorPlugin::set_state(const Dictionary &p_state) {

	if (get_script_instance() && get_script_instance()->has_method("set_state")) {
		get_script_instance()->call("set_state", p_state);
	}
}

void EditorPlugin::clear() {

	if (get_script_instance() && get_script_instance()->has_method("clear")) {
		get_script_instance()->call("clear");
	}
}

// if editor references external resources/scenes, save them
void EditorPlugin::save_external_data() {

	if (get_script_instance() && get_script_instance()->has_method("save_external_data")) {
		get_script_instance()->call("save_external_data");
	}
}

// if changes are pending in editor, apply them
void EditorPlugin::apply_changes() {

	if (get_script_instance() && get_script_instance()->has_method("apply_changes")) {
		get_script_instance()->call("apply_changes");
	}
}

void EditorPlugin::get_breakpoints(List<String> *p_breakpoints) {

	if (get_script_instance() && get_script_instance()->has_method("get_breakpoints")) {
		PoolStringArray arr = get_script_instance()->call("get_breakpoints");
		for (int i = 0; i < arr.size(); i++)
			p_breakpoints->push_back(arr[i]);
	}
}
bool EditorPlugin::get_remove_list(List<Node *> *p_list) {

	return false;
}

void EditorPlugin::restore_global_state() {}
void EditorPlugin::save_global_state() {}

void EditorPlugin::add_import_plugin(const Ref<EditorImportPlugin> &p_importer) {
	ResourceFormatImporter::get_singleton()->add_importer(p_importer);
	EditorFileSystem::get_singleton()->call_deferred("scan");
}

void EditorPlugin::remove_import_plugin(const Ref<EditorImportPlugin> &p_importer) {
	ResourceFormatImporter::get_singleton()->remove_importer(p_importer);
	EditorFileSystem::get_singleton()->call_deferred("scan");
}

void EditorPlugin::add_export_plugin(const Ref<EditorExportPlugin> &p_exporter) {
	EditorExport::get_singleton()->add_export_plugin(p_exporter);
}

void EditorPlugin::remove_export_plugin(const Ref<EditorExportPlugin> &p_exporter) {
	EditorExport::get_singleton()->remove_export_plugin(p_exporter);
}

void EditorPlugin::add_scene_import_plugin(const Ref<EditorSceneImporter> &p_importer) {
	ResourceImporterScene::get_singleton()->add_importer(p_importer);
}

void EditorPlugin::remove_scene_import_plugin(const Ref<EditorSceneImporter> &p_importer) {
	ResourceImporterScene::get_singleton()->remove_importer(p_importer);
}

void EditorPlugin::set_window_layout(Ref<ConfigFile> p_layout) {

	if (get_script_instance() && get_script_instance()->has_method("set_window_layout")) {
		get_script_instance()->call("set_window_layout", p_layout);
	}
}

void EditorPlugin::get_window_layout(Ref<ConfigFile> p_layout) {

	if (get_script_instance() && get_script_instance()->has_method("get_window_layout")) {
		get_script_instance()->call("get_window_layout", p_layout);
	}
}

void EditorPlugin::queue_save_layout() const {

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

void EditorPlugin::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_control_to_container", "container", "control"), &EditorPlugin::add_control_to_container);
	ClassDB::bind_method(D_METHOD("add_control_to_bottom_panel", "control", "title"), &EditorPlugin::add_control_to_bottom_panel);
	ClassDB::bind_method(D_METHOD("add_control_to_dock", "slot", "control"), &EditorPlugin::add_control_to_dock);
	ClassDB::bind_method(D_METHOD("remove_control_from_docks", "control"), &EditorPlugin::remove_control_from_docks);
	ClassDB::bind_method(D_METHOD("remove_control_from_bottom_panel", "control"), &EditorPlugin::remove_control_from_bottom_panel);
	//ClassDB::bind_method(D_METHOD("add_tool_menu_item", "name", "handler", "callback", "ud"),&EditorPlugin::add_tool_menu_item,DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("add_tool_submenu_item", "name", "submenu"), &EditorPlugin::add_tool_submenu_item);
	//ClassDB::bind_method(D_METHOD("remove_tool_menu_item", "name"),&EditorPlugin::remove_tool_menu_item);
	ClassDB::bind_method(D_METHOD("add_custom_type", "type", "base", "script", "icon"), &EditorPlugin::add_custom_type);
	ClassDB::bind_method(D_METHOD("remove_custom_type", "type"), &EditorPlugin::remove_custom_type);

	ClassDB::bind_method(D_METHOD("update_overlays"), &EditorPlugin::update_overlays);

	ClassDB::bind_method(D_METHOD("make_bottom_panel_item_visible", "item"), &EditorPlugin::make_bottom_panel_item_visible);
	ClassDB::bind_method(D_METHOD("hide_bottom_panel"), &EditorPlugin::hide_bottom_panel);

	ClassDB::bind_method(D_METHOD("get_undo_redo"), &EditorPlugin::_get_undo_redo);
	ClassDB::bind_method(D_METHOD("queue_save_layout"), &EditorPlugin::queue_save_layout);
	ClassDB::bind_method(D_METHOD("add_import_plugin", "importer"), &EditorPlugin::add_import_plugin);
	ClassDB::bind_method(D_METHOD("remove_import_plugin", "importer"), &EditorPlugin::remove_import_plugin);
	ClassDB::bind_method(D_METHOD("add_scene_import_plugin", "scene_importer"), &EditorPlugin::add_scene_import_plugin);
	ClassDB::bind_method(D_METHOD("remove_scene_import_plugin", "scene_importer"), &EditorPlugin::remove_scene_import_plugin);
	ClassDB::bind_method(D_METHOD("add_export_plugin", "exporter"), &EditorPlugin::add_export_plugin);
	ClassDB::bind_method(D_METHOD("remove_export_plugin", "exporter"), &EditorPlugin::remove_export_plugin);
	ClassDB::bind_method(D_METHOD("set_input_event_forwarding_always_enabled"), &EditorPlugin::set_input_event_forwarding_always_enabled);
	ClassDB::bind_method(D_METHOD("set_force_draw_over_forwarding_enabled"), &EditorPlugin::set_force_draw_over_forwarding_enabled);

	ClassDB::bind_method(D_METHOD("get_editor_interface"), &EditorPlugin::get_editor_interface);

	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "forward_canvas_gui_input", PropertyInfo(Variant::TRANSFORM2D, "canvas_xform"), PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("forward_draw_over_viewport", PropertyInfo(Variant::OBJECT, "overlay", PROPERTY_HINT_RESOURCE_TYPE, "Control")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("forward_force_draw_over_viewport", PropertyInfo(Variant::OBJECT, "overlay", PROPERTY_HINT_RESOURCE_TYPE, "Control")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "forward_spatial_gui_input", PropertyInfo(Variant::OBJECT, "camera", PROPERTY_HINT_RESOURCE_TYPE, "Camera"), PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	MethodInfo gizmo = MethodInfo(Variant::OBJECT, "create_spatial_gizmo", PropertyInfo(Variant::OBJECT, "for_spatial", PROPERTY_HINT_RESOURCE_TYPE, "Spatial"));
	gizmo.return_val.hint = PROPERTY_HINT_RESOURCE_TYPE;
	gizmo.return_val.hint_string = "EditorSpatialGizmo";
	ClassDB::add_virtual_method(get_class_static(), gizmo);
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::STRING, "get_plugin_name"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::OBJECT, "get_plugin_icon"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "has_main_screen"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("make_visible", PropertyInfo(Variant::BOOL, "visible")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("edit", PropertyInfo(Variant::OBJECT, "object")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "handles", PropertyInfo(Variant::OBJECT, "object")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::DICTIONARY, "get_state"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("set_state", PropertyInfo(Variant::DICTIONARY, "state")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("clear"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("save_external_data"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("apply_changes"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::POOL_STRING_ARRAY, "get_breakpoints"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("set_window_layout", PropertyInfo(Variant::OBJECT, "layout", PROPERTY_HINT_RESOURCE_TYPE, "ConfigFile")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("get_window_layout", PropertyInfo(Variant::OBJECT, "layout", PROPERTY_HINT_RESOURCE_TYPE, "ConfigFile")));

	ADD_SIGNAL(MethodInfo("scene_changed", PropertyInfo(Variant::OBJECT, "scene_root", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("scene_closed", PropertyInfo(Variant::STRING, "filepath")));
	ADD_SIGNAL(MethodInfo("main_screen_changed", PropertyInfo(Variant::STRING, "screen_name")));

	BIND_ENUM_CONSTANT(CONTAINER_TOOLBAR);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_MENU);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_SIDE);
	BIND_ENUM_CONSTANT(CONTAINER_SPATIAL_EDITOR_BOTTOM);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_MENU);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_SIDE);
	BIND_ENUM_CONSTANT(CONTAINER_CANVAS_EDITOR_BOTTOM);
	BIND_ENUM_CONSTANT(CONTAINER_PROPERTY_EDITOR_BOTTOM);

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

EditorPlugin::EditorPlugin() {
	undo_redo = NULL;
	input_event_forwarding_always_enabled = false;
	force_draw_over_forwarding_enabled = false;
	last_main_screen_name = "";
}

EditorPlugin::~EditorPlugin() {
}

EditorPluginCreateFunc EditorPlugins::creation_funcs[MAX_CREATE_FUNCS];

int EditorPlugins::creation_func_count = 0;
