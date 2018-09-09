/*************************************************************************/
/*  editor_scene_template_settings.cpp                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor_scene_template_settings.h"

#include "core/global_constants.h"
#include "core/project_settings.h"
#include "editor/editor_node.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

#define PREVIEW_LIST_MAX_SIZE 10

void EditorSceneTemplateSettings::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		List<String> afn;
		ResourceLoader::get_recognized_extensions_for_type("PackedScene", &afn);

		EditorFileDialog *file_dialog = scene_template_add_path->get_file_dialog();

		for (List<String>::Element *E = afn.front(); E; E = E->next()) {
			file_dialog->add_filter("*." + E->get());
		}
	}
}

bool EditorSceneTemplateSettings::_scene_template_name_is_valid(const String &p_name, String *r_error) {

	if (!p_name.is_valid_identifier()) {
		if (r_error)
			*r_error = TTR("Invalid name.") + "\n" + TTR("Valid characters:") + " a-z, A-Z, 0-9 or _";

		return false;
	}

	if (ClassDB::class_exists(p_name)) {
		if (r_error)
			*r_error = TTR("Invalid name. Must not collide with an existing engine class name.");

		return false;
	}

	if (ScriptServer::is_global_class(p_name) && !ClassDB::is_parent_class(ScriptServer::get_global_class_base(p_name), "Node")) {
		if (r_error)
			*r_error = TTR("Invalid name. Must not collide with an existing non-Node script class name.");

		return false;
	}

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (Variant::get_type_name(Variant::Type(i)) == p_name) {
			if (r_error)
				*r_error = TTR("Invalid name. Must not collide with an existing buit-in type name.");

			return false;
		}
	}

	for (int i = 0; i < GlobalConstants::get_global_constant_count(); i++) {
		if (GlobalConstants::get_global_constant_name(i) == p_name) {
			if (r_error)
				*r_error = TTR("Invalid name. Must not collide with an existing global constant name.");

			return false;
		}
	}

	return true;
}

void EditorSceneTemplateSettings::_scene_template_add() {

	scene_template_add(scene_template_add_name->get_text(), scene_template_add_path->get_line_edit()->get_text());

	scene_template_add_path->get_line_edit()->set_text("");
	scene_template_add_name->set_text("");
}

void EditorSceneTemplateSettings::_scene_template_selected() {

	TreeItem *ti = tree->get_selected();

	if (!ti)
		return;

	selected_scene_template = "scene_template/" + ti->get_text(0);
}

void EditorSceneTemplateSettings::_scene_template_edited() {

	if (updating_scene_template)
		return;

	TreeItem *ti = tree->get_edited();
	int column = tree->get_edited_column();

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	if (column == 0) {
		String name = ti->get_text(0);
		String old_name = selected_scene_template.get_slice("/", 1);

		if (name == old_name)
			return;

		String error;
		if (!_scene_template_name_is_valid(name, &error)) {
			ti->set_text(0, old_name);
			EditorNode::get_singleton()->show_warning(error);
			return;
		}

		if (ProjectSettings::get_singleton()->has_setting("scene_template/" + name)) {
			ti->set_text(0, old_name);
			EditorNode::get_singleton()->show_warning(vformat(TTR("SceneTemplate '%s' already exists!"), name));
			return;
		}

		updating_scene_template = true;

		name = "scene_template/" + name;

		int order = ProjectSettings::get_singleton()->get_order(selected_scene_template);
		String path = ProjectSettings::get_singleton()->get(selected_scene_template);

		undo_redo->create_action(TTR("Rename SceneTemplate"));

		undo_redo->add_do_property(ProjectSettings::get_singleton(), name, path);
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "set_order", name, order);
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "clear", selected_scene_template);

		undo_redo->add_undo_property(ProjectSettings::get_singleton(), selected_scene_template, path);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", selected_scene_template, order);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "clear", name);

		undo_redo->add_do_method(this, "call_deferred", "update_scene_templates");
		undo_redo->add_undo_method(this, "call_deferred", "update_scene_templates");

		undo_redo->add_do_method(this, "emit_signal", scene_template_changed);
		undo_redo->add_undo_method(this, "emit_signal", scene_template_changed);

		undo_redo->commit_action();

		selected_scene_template = name;
	}

	updating_scene_template = false;
}

void EditorSceneTemplateSettings::_scene_template_button_pressed(Object *p_item, int p_column, int p_button) {

	TreeItem *ti = Object::cast_to<TreeItem>(p_item);

	String name = "scene_template/" + ti->get_text(0);

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	switch (p_button) {
		case BUTTON_OPEN: {
			_scene_template_open(ti->get_text(1));
		} break;
		case BUTTON_MOVE_UP:
		case BUTTON_MOVE_DOWN: {

			TreeItem *swap = NULL;

			if (p_button == BUTTON_MOVE_UP) {
				swap = ti->get_prev();
			} else {
				swap = ti->get_next();
			}

			if (!swap)
				return;

			String swap_name = "scene_template/" + swap->get_text(0);

			int order = ProjectSettings::get_singleton()->get_order(name);
			int swap_order = ProjectSettings::get_singleton()->get_order(swap_name);

			undo_redo->create_action(TTR("Move SceneTemplate"));

			undo_redo->add_do_method(ProjectSettings::get_singleton(), "set_order", name, swap_order);
			undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", name, order);

			undo_redo->add_do_method(ProjectSettings::get_singleton(), "set_order", swap_name, order);
			undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", swap_name, swap_order);

			undo_redo->add_do_method(this, "update_scene_templates");
			undo_redo->add_undo_method(this, "update_scene_templates");

			undo_redo->add_do_method(this, "emit_signal", scene_template_changed);
			undo_redo->add_undo_method(this, "emit_signal", scene_template_changed);

			undo_redo->commit_action();
		} break;
		case BUTTON_DELETE: {

			int order = ProjectSettings::get_singleton()->get_order(name);

			undo_redo->create_action(TTR("Remove SceneTemplate"));

			undo_redo->add_do_property(ProjectSettings::get_singleton(), name, Variant());

			undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, ProjectSettings::get_singleton()->get(name));
			undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_persisting", name, true);
			undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", order);

			undo_redo->add_do_method(this, "update_scene_templates");
			undo_redo->add_undo_method(this, "update_scene_templates");

			undo_redo->add_do_method(this, "emit_signal", scene_template_changed);
			undo_redo->add_undo_method(this, "emit_signal", scene_template_changed);

			undo_redo->commit_action();
		} break;
	}
}

void EditorSceneTemplateSettings::_scene_template_activated() {
	TreeItem *ti = tree->get_selected();
	if (!ti)
		return;
	_scene_template_open(ti->get_text(1));
}

void EditorSceneTemplateSettings::_scene_template_open(const String &fpath) {
	if (ResourceLoader::get_resource_type(fpath) == "PackedScene") {
		EditorNode::get_singleton()->open_request(fpath);
	}
	ProjectSettingsEditor::get_singleton()->hide();
}
void EditorSceneTemplateSettings::_scene_template_file_callback(const String &p_path) {

	StringName name = scene_template_get_base_file(p_path);
	scene_template_add_name->set_text(p_path.get_file().get_basename());
}

StringName EditorSceneTemplateSettings::scene_template_get_base(const StringName &p_name) const {
	if (p_name == StringName() || !scene_template_map.has(p_name))
		return StringName();
	ERR_FAIL_COND_V(!scene_template_map[p_name], StringName());

	Ref<PackedScene> scene = ResourceLoader::load(scene_template_map[p_name]->path, "PackedScene");
	if (scene.is_null())
		return StringName();

	Ref<SceneState> state = scene->get_state();
	if (state.is_null())
		return StringName();

	Ref<PackedScene> base_scene = state->get_node_instance(0);
	while (base_scene.is_valid()) {
		for (const List<SceneTemplateInfo>::Element *E = scene_template_cache.front(); E; E = E->next()) {
			if (E->get().path == base_scene->get_path()) {
				return E->get().name;
			}
		}
		base_scene = base_scene->get_state()->get_node_instance(0);
	}

	for (int i = 0; i < state->get_node_property_count(0); i++) {
		if ("script" == state->get_node_property_name(0, i)) {
			Ref<Script> script = state->get_node_property_value(0, i);
			if (script.is_valid()) {
				StringName name = EditorNode::get_editor_data().script_class_get_name(script->get_path());
				if (ScriptServer::is_global_class(name))
					return name;
			}
			break;
		}
	}
	return state->get_node_type(0);
}

StringName EditorSceneTemplateSettings::scene_template_get_base_file(const String &p_path) const {
	return scene_template_get_base(scene_template_get_name(p_path));
}

StringName EditorSceneTemplateSettings::scene_template_get_name(const String &p_path) const {
	for (const List<SceneTemplateInfo>::Element *E = scene_template_cache.front(); E; E = E->next()) {
		if (E->get().path == p_path)
			return E->get().name;
	}
	return StringName();
}

Node *EditorSceneTemplateSettings::scene_template_instance(const StringName &p_name) {
	return _create_scene_template(scene_template_map[p_name]->path);
}

bool EditorSceneTemplateSettings::scene_template_is_parent(const StringName &p_name, const StringName &p_inherits) {
	if (!scene_template_map.has(p_name))
		return false;

	return EditorNode::get_editor_data().script_class_is_parent(scene_template_get_base(p_name), p_inherits);
}

Node *EditorSceneTemplateSettings::_create_scene_template(const String &p_path) {
	RES res = ResourceLoader::load(p_path);
	ERR_EXPLAIN("Can't instance SceneTemplate: " + p_path);
	ERR_FAIL_COND_V(res.is_null(), NULL);
	Node *n = NULL;
	if (res->is_class("PackedScene")) {
		Ref<PackedScene> ps = res;
		n = ps->instance();
	}

	ERR_EXPLAIN("Path in SceneTemplate is not a node: " + p_path);
	ERR_FAIL_COND_V(!n, NULL);

	return n;
}

void EditorSceneTemplateSettings::update_scene_templates() {

	if (updating_scene_template)
		return;

	updating_scene_template = true;

	Map<String, SceneTemplateInfo> to_remove;
	List<SceneTemplateInfo *> to_add;

	for (List<SceneTemplateInfo>::Element *E = scene_template_cache.front(); E; E = E->next()) {
		SceneTemplateInfo &info = E->get();
		to_remove.insert(info.name, info);
	}

	scene_template_cache.clear();
	scene_template_map.clear();

	tree->clear();
	TreeItem *root = tree->create_item();

	List<PropertyInfo> props;
	ProjectSettings::get_singleton()->get_property_list(&props);

	for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {

		const PropertyInfo &pi = E->get();

		if (!pi.name.begins_with("scene_template/"))
			continue;

		String name = pi.name.get_slice("/", 1);
		String path = ProjectSettings::get_singleton()->get(pi.name);
		int order = ProjectSettings::get_singleton()->get_order(pi.name);

		if (name.empty())
			continue;

		SceneTemplateInfo info;

		info.name = name;
		info.path = path;
		info.order = order;

		scene_template_cache.push_back(info);
		scene_template_map[name] = &scene_template_cache.back()->get();

		TreeItem *item = tree->create_item(root);
		item->set_text(0, name);
		item->set_editable(0, true);

		item->set_text(1, path);
		item->set_selectable(1, true);

		item->add_button(2, get_icon("FileList", "EditorIcons"), BUTTON_OPEN);
		item->add_button(2, get_icon("MoveUp", "EditorIcons"), BUTTON_MOVE_UP);
		item->add_button(2, get_icon("MoveDown", "EditorIcons"), BUTTON_MOVE_DOWN);
		item->add_button(2, get_icon("Remove", "EditorIcons"), BUTTON_DELETE);
		item->set_selectable(2, true);
	}

	updating_scene_template = false;
}

Variant EditorSceneTemplateSettings::get_drag_data_fw(const Point2 &p_point, Control *p_control) {

	if (scene_template_cache.size() <= 1)
		return false;

	PoolStringArray scene_templates;

	TreeItem *next = tree->get_next_selected(NULL);

	while (next) {
		scene_templates.push_back(next->get_text(0));
		next = tree->get_next_selected(next);
	}

	if (scene_templates.size() == 0 || scene_templates.size() == scene_template_cache.size())
		return Variant();

	VBoxContainer *preview = memnew(VBoxContainer);

	int max_size = MIN(PREVIEW_LIST_MAX_SIZE, scene_templates.size());

	for (int i = 0; i < max_size; i++) {
		Label *label = memnew(Label(scene_templates[i]));
		label->set_self_modulate(Color(1, 1, 1, Math::lerp(1, 0, float(i) / PREVIEW_LIST_MAX_SIZE)));

		preview->add_child(label);
	}

	tree->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);
	tree->set_drag_preview(preview);

	Dictionary drop_data;
	drop_data["type"] = "scene_template";
	drop_data["scene_templates"] = scene_templates;

	return drop_data;
}

bool EditorSceneTemplateSettings::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control) const {
	if (updating_scene_template)
		return false;

	Dictionary drop_data = p_data;

	if (!drop_data.has("type"))
		return false;

	if (drop_data.has("type")) {
		TreeItem *ti = tree->get_item_at_position(p_point);

		if (!ti)
			return false;

		int section = tree->get_drop_section_at_position(p_point);

		if (section < -1)
			return false;

		return true;
	}

	return false;
}

void EditorSceneTemplateSettings::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_control) {

	TreeItem *ti = tree->get_item_at_position(p_point);

	if (!ti)
		return;

	int section = tree->get_drop_section_at_position(p_point);

	if (section < -1)
		return;

	String name;
	bool move_to_back = false;

	if (section < 0) {
		name = ti->get_text(0);
	} else if (ti->get_next()) {
		name = ti->get_next()->get_text(0);
	} else {
		name = ti->get_text(0);
		move_to_back = true;
	}

	int order = ProjectSettings::get_singleton()->get_order("scene_template/" + name);

	SceneTemplateInfo aux;
	List<SceneTemplateInfo>::Element *E = NULL;

	if (!move_to_back) {
		aux.order = order;
		E = scene_template_cache.find(aux);
	}

	Dictionary drop_data = p_data;
	PoolStringArray scene_templates = drop_data["scene_templates"];

	Vector<int> orders;
	orders.resize(scene_template_cache.size());

	for (int i = 0; i < scene_templates.size(); i++) {
		aux.order = ProjectSettings::get_singleton()->get_order("scene_template/" + scene_templates[i]);

		List<SceneTemplateInfo>::Element *I = scene_template_cache.find(aux);

		if (move_to_back) {
			scene_template_cache.move_to_back(I);
		} else if (E != I) {
			scene_template_cache.move_before(I, E);
		} else if (E->next()) {
			E = E->next();
		} else {
			break;
		}
	}

	int i = 0;

	for (List<SceneTemplateInfo>::Element *E = scene_template_cache.front(); E; E = E->next()) {
		orders.write[i++] = E->get().order;
	}

	orders.sort();

	UndoRedo *undo_redo = EditorNode::get_undo_redo();

	undo_redo->create_action(TTR("Rearrange SceneTemplates"));

	i = 0;

	for (List<SceneTemplateInfo>::Element *E = scene_template_cache.front(); E; E = E->next()) {
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "set_order", "scene_template/" + E->get().name, orders[i++]);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", "scene_template/" + E->get().name, E->get().order);
	}

	orders.clear();

	undo_redo->add_do_method(this, "update_scene_templates");
	undo_redo->add_undo_method(this, "update_scene_templates");

	undo_redo->add_do_method(this, "emit_signal", scene_template_changed);
	undo_redo->add_undo_method(this, "emit_signal", scene_template_changed);

	undo_redo->commit_action();
}

void EditorSceneTemplateSettings::scene_template_add(const String &p_name, const String &p_path) {

	String name = p_name;

	String error;
	if (!_scene_template_name_is_valid(name, &error)) {
		EditorNode::get_singleton()->show_warning(error);
		return;
	}

	String path = p_path;
	if (!FileAccess::exists(path)) {
		EditorNode::get_singleton()->show_warning(TTR("Invalid Path.") + "\n" + TTR("File does not exist."));
		return;
	}

	if (!path.begins_with("res://")) {
		EditorNode::get_singleton()->show_warning(TTR("Invalid Path.") + "\n" + TTR("Not in resource path."));
		return;
	}

	name = "scene_template/" + name;

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

	undo_redo->create_action(TTR("Add SceneTemplate"));
	undo_redo->add_do_property(ProjectSettings::get_singleton(), name, path);

	if (ProjectSettings::get_singleton()->has_setting(name)) {
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, ProjectSettings::get_singleton()->get(name));
	} else {
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, Variant());
	}

	undo_redo->add_do_method(this, "update_scene_templates");
	undo_redo->add_undo_method(this, "update_scene_templates");

	undo_redo->add_do_method(this, "emit_signal", scene_template_changed);
	undo_redo->add_undo_method(this, "emit_signal", scene_template_changed);

	undo_redo->commit_action();
}

void EditorSceneTemplateSettings::scene_template_remove(const String &p_name) {

	String name = "scene_template/" + p_name;

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

	int order = ProjectSettings::get_singleton()->get_order(name);

	undo_redo->create_action(TTR("Remove SceneTemplate"));

	undo_redo->add_do_property(ProjectSettings::get_singleton(), name, Variant());

	undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, ProjectSettings::get_singleton()->get(name));
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_persisting", name, true);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", order);

	undo_redo->add_do_method(this, "update_scene_templates");
	undo_redo->add_undo_method(this, "update_scene_templates");

	undo_redo->add_do_method(this, "emit_signal", scene_template_changed);
	undo_redo->add_undo_method(this, "emit_signal", scene_template_changed);

	undo_redo->commit_action();
}

void EditorSceneTemplateSettings::set_template_name(const String &p_name) {
	scene_template_add_name->set_text(p_name);
}

String EditorSceneTemplateSettings::get_template_name() const {
	return scene_template_add_name->get_text();
}

void EditorSceneTemplateSettings::set_template_path(const String &p_path) {
	scene_template_add_path->get_line_edit()->set_text(p_path);
}

String EditorSceneTemplateSettings::get_template_path() const {
	return scene_template_add_path->get_line_edit()->get_text();
}

void EditorSceneTemplateSettings::_bind_methods() {

	ClassDB::bind_method("_scene_template_add", &EditorSceneTemplateSettings::_scene_template_add);
	ClassDB::bind_method("_scene_template_selected", &EditorSceneTemplateSettings::_scene_template_selected);
	ClassDB::bind_method("_scene_template_edited", &EditorSceneTemplateSettings::_scene_template_edited);
	ClassDB::bind_method("_scene_template_button_pressed", &EditorSceneTemplateSettings::_scene_template_button_pressed);
	ClassDB::bind_method("_scene_template_file_callback", &EditorSceneTemplateSettings::_scene_template_file_callback);
	ClassDB::bind_method("_scene_template_activated", &EditorSceneTemplateSettings::_scene_template_activated);
	ClassDB::bind_method("_scene_template_open", &EditorSceneTemplateSettings::_scene_template_open);

	ClassDB::bind_method("get_drag_data_fw", &EditorSceneTemplateSettings::get_drag_data_fw);
	ClassDB::bind_method("can_drop_data_fw", &EditorSceneTemplateSettings::can_drop_data_fw);
	ClassDB::bind_method("drop_data_fw", &EditorSceneTemplateSettings::drop_data_fw);

	ClassDB::bind_method("update_scene_templates", &EditorSceneTemplateSettings::update_scene_templates);
	ClassDB::bind_method("scene_template_add", &EditorSceneTemplateSettings::scene_template_add);
	ClassDB::bind_method("scene_template_remove", &EditorSceneTemplateSettings::scene_template_remove);

	ADD_SIGNAL(MethodInfo("scene_template_changed"));
}

EditorSceneTemplateSettings::EditorSceneTemplateSettings() {

	// Make first cache
	List<PropertyInfo> props;
	ProjectSettings::get_singleton()->get_property_list(&props);
	for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {

		const PropertyInfo &pi = E->get();

		if (!pi.name.begins_with("scene_template/"))
			continue;

		String name = pi.name.get_slice("/", 1);

		if (name.empty())
			continue;

		String path = ProjectSettings::get_singleton()->get(pi.name);
		int order = ProjectSettings::get_singleton()->get_order(pi.name);

		scene_template_cache.push_back(SceneTemplateInfo(name, path, order));
		scene_template_map[name] = &scene_template_cache.back()->get();
	}

	scene_template_changed = "scene_template_changed";

	updating_scene_template = false;
	selected_scene_template = "";

	HBoxContainer *hbc = memnew(HBoxContainer);
	add_child(hbc);

	Label *l = memnew(Label);
	l->set_text(TTR("Name:"));
	hbc->add_child(l);

	scene_template_add_name = memnew(LineEdit);
	scene_template_add_name->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(scene_template_add_name);

	l = memnew(Label);
	l->set_text(TTR("Path:"));
	hbc->add_child(l);

	scene_template_add_path = memnew(EditorLineEditFileChooser);
	scene_template_add_path->set_h_size_flags(SIZE_EXPAND_FILL);
	scene_template_add_path->get_file_dialog()->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	scene_template_add_path->get_file_dialog()->connect("file_selected", this, "_scene_template_file_callback");
	hbc->add_child(scene_template_add_path);

	Button *add_scene_template = memnew(Button);
	add_scene_template->set_text(TTR("Add"));
	add_scene_template->connect("pressed", this, "_scene_template_add");
	hbc->add_child(add_scene_template);

	tree = memnew(Tree);
	tree->set_hide_root(true);
	tree->set_select_mode(Tree::SELECT_MULTI);
	tree->set_allow_reselect(true);

	tree->set_drag_forwarding(this);

	tree->set_columns(3);
	tree->set_column_titles_visible(true);

	tree->set_column_title(0, TTR("Name"));
	tree->set_column_expand(0, true);
	tree->set_column_min_width(0, 100);

	tree->set_column_title(1, TTR("Path"));
	tree->set_column_expand(1, true);
	tree->set_column_min_width(1, 100);

	tree->set_column_expand(2, false);
	tree->set_column_min_width(2, 120 * EDSCALE);

	tree->connect("cell_selected", this, "_scene_template_selected");
	tree->connect("item_edited", this, "_scene_template_edited");
	tree->connect("button_pressed", this, "_scene_template_button_pressed");
	tree->connect("item_activated", this, "_scene_template_activated");
	tree->set_v_size_flags(SIZE_EXPAND_FILL);

	add_child(tree, true);
}

EditorSceneTemplateSettings::~EditorSceneTemplateSettings() {
}
