/**************************************************************************/
/*  blueprint_editor_plugin.cpp                                           */
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

#include "blueprint_editor_plugin.h"

#include "../blueprint_player.h"

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/object/callable_mp.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/gui/editor_file_dialog.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/graph_node.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/tab_bar.h"
#include "scene/resources/style_box_flat.h"
#include "scene/scene_string_names.h"

static const int PORT_TYPE_EXEC = 0;
static const int PORT_TYPE_DATA = 1;

static const Color EXEC_COLOR = Color(0.95, 0.95, 0.95);
static const Color DATA_COLOR = Color(0.45, 0.78, 1.0);

static String _node_gui_name(int p_id) {
	return "bp_" + itos(p_id);
}

static int _node_id_from_gui_name(const StringName &p_name) {
	return String(p_name).trim_prefix("bp_").to_int();
}

static Color _category_color(const String &p_category) {
	if (p_category == "event") {
		return Color(0.55, 0.21, 0.17); // Red: entry points.
	}
	if (p_category == "action") {
		return Color(0.16, 0.35, 0.53); // Blue: actions.
	}
	if (p_category == "flow") {
		return Color(0.38, 0.27, 0.5); // Purple: control flow.
	}
	return Color(0.15, 0.4, 0.28); // Green: data.
}

static String _category_label(const String &p_category) {
	if (p_category == "event") {
		return "Evenements";
	}
	if (p_category == "action") {
		return "Actions";
	}
	if (p_category == "flow") {
		return "Flux";
	}
	return "Donnees";
}

String BlueprintEditor::_tab_title(const Ref<Blueprint> &p_blueprint) const {
	const String path = p_blueprint->get_path();
	if (path.is_empty()) {
		return "(nouveau)";
	}
	if (path.contains("::")) {
		return path.get_slice("::", 0).get_file() + " (scene)";
	}
	return path.get_file();
}

void BlueprintEditor::_update_state() {
	const bool has_blueprint = blueprint.is_valid();
	empty_hint->set_visible(!has_blueprint);
	graph->set_visible(has_blueprint);
	tabs->set_visible(open_blueprints.size() > 0);
	add_menu->set_disabled(!has_blueprint);
	save_button->set_disabled(!has_blueprint);

	if (has_blueprint) {
		const String path = blueprint->get_path();
		if (path.is_empty()) {
			info_label->set_text("(pas encore enregistre)");
		} else if (path.contains("::")) {
			info_label->set_text("Integre a la scene " + path.get_slice("::", 0).get_file() + " - sauvegarde avec Ctrl+S");
		} else {
			info_label->set_text(path);
		}
	} else {
		info_label->set_text("");
	}
}

void BlueprintEditor::edit(const Ref<Blueprint> &p_blueprint) {
	if (p_blueprint.is_null()) {
		return;
	}

	for (int i = 0; i < open_blueprints.size(); i++) {
		if (open_blueprints[i] == p_blueprint) {
			// set_current_tab() only emits tab_changed when the index changes.
			if (tabs->get_current_tab() == i) {
				_on_tab_changed(i);
			} else {
				tabs->set_current_tab(i);
			}
			return;
		}
	}

	open_blueprints.push_back(p_blueprint);
	tabs->add_tab(_tab_title(p_blueprint));
	const int idx = tabs->get_tab_count() - 1;
	if (tabs->get_current_tab() == idx) {
		_on_tab_changed(idx);
	} else {
		tabs->set_current_tab(idx);
	}
}

void BlueprintEditor::_on_tab_changed(int p_tab) {
	if (p_tab < 0 || p_tab >= open_blueprints.size()) {
		blueprint.unref();
	} else {
		blueprint = open_blueprints[p_tab];
	}
	_update_state();
	_rebuild();
}

void BlueprintEditor::_on_tab_close_pressed(int p_tab) {
	if (p_tab < 0 || p_tab >= open_blueprints.size()) {
		return;
	}
	open_blueprints.remove_at(p_tab);
	tabs->remove_tab(p_tab);

	if (open_blueprints.is_empty()) {
		blueprint.unref();
		_update_state();
		_rebuild();
	} else {
		_on_tab_changed(tabs->get_current_tab());
	}
}

void BlueprintEditor::_on_new_pressed() {
	new_dialog->popup_centered_ratio(0.7);
}

void BlueprintEditor::_on_open_pressed() {
	open_dialog->popup_centered_ratio(0.7);
}

void BlueprintEditor::_on_save_pressed() {
	if (blueprint.is_null()) {
		return;
	}
	const String path = blueprint->get_path();
	if (path.is_empty()) {
		new_dialog->popup_centered_ratio(0.7); // No file yet: pick one.
		return;
	}
	if (path.contains("::")) {
		EditorNode::get_singleton()->show_warning("Ce blueprint est integre a une scene.\nSauvegarde la scene (Ctrl+S) pour l'enregistrer.");
		return;
	}
	const Error err = ResourceSaver::save(blueprint, path);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(vformat("Echec de l'enregistrement du blueprint (erreur %d).", err));
	}
}

void BlueprintEditor::_on_new_file_selected(const String &p_path) {
	Ref<Blueprint> bp = blueprint.is_valid() && blueprint->get_path().is_empty() ? blueprint : Ref<Blueprint>();
	if (bp.is_null()) {
		bp.instantiate();
	}
	const Error err = ResourceSaver::save(bp, p_path, ResourceSaver::FLAG_CHANGE_PATH);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(vformat("Impossible de creer le blueprint (erreur %d).", err));
		return;
	}
	EditorFileSystem::get_singleton()->update_file(p_path);
	edit(bp);
	// The tab may already exist (saving a "(nouveau)" tab): refresh its title.
	tabs->set_tab_title(tabs->get_current_tab(), _tab_title(bp));
	_update_state();
}

void BlueprintEditor::_on_open_file_selected(const String &p_path) {
	Ref<Blueprint> bp = ResourceLoader::load(p_path);
	if (bp.is_null()) {
		EditorNode::get_singleton()->show_warning("Ce fichier n'est pas une ressource Blueprint.");
		return;
	}
	edit(bp);
}

Vector2 BlueprintEditor::_center_graph_position() const {
	return (graph->get_scroll_offset() + graph->get_size() * 0.5) / graph->get_zoom();
}

void BlueprintEditor::_fill_node_menu(PopupMenu *p_menu) {
	p_menu->clear();
	const Vector<BlueprintNodeDef> &defs = blueprint_get_node_defs();

	String current_category;
	for (int i = 0; i < defs.size(); i++) {
		if (defs[i].category != current_category) {
			current_category = defs[i].category;
			p_menu->add_separator(_category_label(current_category));
		}
		p_menu->add_item(defs[i].title, i);
	}
}

void BlueprintEditor::_rebuild() {
	graph->clear_connections();
	for (int i = graph->get_child_count() - 1; i >= 0; i--) {
		GraphNode *gn = Object::cast_to<GraphNode>(graph->get_child(i));
		if (gn) {
			graph->remove_child(gn);
			memdelete(gn);
		}
	}

	if (blueprint.is_null()) {
		return;
	}

	const Array nodes = blueprint->get_nodes();
	for (int i = 0; i < nodes.size(); i++) {
		_make_graph_node(nodes[i]);
	}

	const Array conns = blueprint->get_connections();
	for (int i = 0; i < conns.size(); i++) {
		const Dictionary conn = conns[i];
		graph->connect_node(_node_gui_name(conn["from_node"]), conn["from_port"], _node_gui_name(conn["to_node"]), conn["to_port"]);
	}
}

void BlueprintEditor::_style_graph_node(GraphNode *p_gn, const String &p_category) {
	const Color color = _category_color(p_category);

	Ref<StyleBoxFlat> titlebar = p_gn->get_theme_stylebox(SNAME("titlebar"))->duplicate();
	if (titlebar.is_valid()) {
		titlebar->set_bg_color(color);
		p_gn->add_theme_style_override(SNAME("titlebar"), titlebar);
	}
	Ref<StyleBoxFlat> titlebar_selected = p_gn->get_theme_stylebox(SNAME("titlebar_selected"))->duplicate();
	if (titlebar_selected.is_valid()) {
		titlebar_selected->set_bg_color(color.lightened(0.2));
		p_gn->add_theme_style_override(SNAME("titlebar_selected"), titlebar_selected);
	}
}

void BlueprintEditor::_make_graph_node(const Dictionary &p_data) {
	const int id = p_data["id"];
	const String type = p_data["type"];
	const BlueprintNodeDef *def = blueprint_get_node_def(type);
	if (!def) {
		return;
	}
	const Dictionary params = p_data["params"];

	GraphNode *gn = memnew(GraphNode);
	gn->set_name(_node_gui_name(id));
	gn->set_title(def->title);
	gn->set_position_offset(p_data["position"]);
	gn->set_custom_minimum_size(Size2(190, 0));

	// Portless settings first (node path, method name...): rows without slots.
	for (const String &key : def->config_params) {
		HBoxContainer *row = memnew(HBoxContainer);
		row->set_custom_minimum_size(Size2(170, 26));
		row->add_theme_constant_override(SNAME("separation"), 8);

		Label *label = memnew(Label);
		label->set_text(key);
		row->add_child(label);

		LineEdit *edit = memnew(LineEdit);
		edit->set_custom_minimum_size(Size2(100, 0));
		edit->set_h_size_flags(SIZE_EXPAND_FILL);
		edit->set_placeholder(key);
		edit->set_text(params.get(key, ""));
		edit->connect(SceneStringName(text_changed), callable_mp(this, &BlueprintEditor::_on_param_changed).bind(id, key));
		row->add_child(edit);

		gn->add_child(row);
	}
	const int slot_offset = def->config_params.size();

	const int rows = MAX(def->inputs.size(), def->outputs.size());
	for (int i = 0; i < rows; i++) {
		HBoxContainer *row = memnew(HBoxContainer);
		row->set_custom_minimum_size(Size2(170, 26));
		row->add_theme_constant_override(SNAME("separation"), 8);

		const bool has_input = i < def->inputs.size();
		const bool has_output = i < def->outputs.size();

		if (has_input) {
			Label *label = memnew(Label);
			label->set_text(def->inputs[i].name);
			row->add_child(label);

			if (!def->inputs[i].param_key.is_empty()) {
				LineEdit *edit = memnew(LineEdit);
				edit->set_custom_minimum_size(Size2(80, 0));
				edit->set_h_size_flags(SIZE_EXPAND_FILL);
				edit->set_placeholder(def->inputs[i].param_key);
				edit->set_text(params.get(def->inputs[i].param_key, ""));
				edit->connect(SceneStringName(text_changed), callable_mp(this, &BlueprintEditor::_on_param_changed).bind(id, def->inputs[i].param_key));
				row->add_child(edit);
			}
		}

		Control *spacer = memnew(Control);
		spacer->set_h_size_flags(SIZE_EXPAND_FILL);
		row->add_child(spacer);

		if (has_output) {
			if (!def->outputs[i].param_key.is_empty()) {
				LineEdit *edit = memnew(LineEdit);
				edit->set_custom_minimum_size(Size2(80, 0));
				edit->set_h_size_flags(SIZE_EXPAND_FILL);
				edit->set_placeholder(def->outputs[i].param_key);
				edit->set_text(params.get(def->outputs[i].param_key, ""));
				edit->connect(SceneStringName(text_changed), callable_mp(this, &BlueprintEditor::_on_param_changed).bind(id, def->outputs[i].param_key));
				row->add_child(edit);
			}

			Label *label = memnew(Label);
			label->set_text(def->outputs[i].name);
			row->add_child(label);
		}

		gn->add_child(row);

		const bool in_exec = has_input && def->inputs[i].exec;
		const bool out_exec = has_output && def->outputs[i].exec;
		gn->set_slot(slot_offset + i,
				has_input, in_exec ? PORT_TYPE_EXEC : PORT_TYPE_DATA, in_exec ? EXEC_COLOR : DATA_COLOR,
				has_output, out_exec ? PORT_TYPE_EXEC : PORT_TYPE_DATA, out_exec ? EXEC_COLOR : DATA_COLOR);
	}

	gn->connect(SNAME("dragged"), callable_mp(this, &BlueprintEditor::_on_node_dragged).bind(id));
	graph->add_child(gn);
	_style_graph_node(gn, def->category);
}

void BlueprintEditor::_on_connection_request(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	if (blueprint.is_null()) {
		return;
	}
	const int from_id = _node_id_from_gui_name(p_from);
	const int to_id = _node_id_from_gui_name(p_to);
	if (from_id == to_id) {
		return;
	}

	const BlueprintNodeDef *from_def = blueprint_get_node_def(String(blueprint->get_node_data(from_id)["type"]));
	const BlueprintNodeDef *to_def = blueprint_get_node_def(String(blueprint->get_node_data(to_id)["type"]));
	if (!from_def || !to_def || p_from_port >= from_def->outputs.size() || p_to_port >= to_def->inputs.size()) {
		return;
	}
	const bool from_exec = from_def->outputs[p_from_port].exec;
	const bool to_exec = to_def->inputs[p_to_port].exec;
	if (from_exec != to_exec) {
		return;
	}

	// An exec output drives a single target; a data input has a single source.
	const Array conns = blueprint->get_connections();
	for (int i = 0; i < conns.size(); i++) {
		const Dictionary conn = conns[i];
		const bool same_exec_source = from_exec && int(conn["from_node"]) == from_id && int(conn["from_port"]) == p_from_port;
		const bool same_data_target = !from_exec && int(conn["to_node"]) == to_id && int(conn["to_port"]) == p_to_port;
		if (same_exec_source || same_data_target) {
			blueprint->remove_connection(conn["from_node"], conn["from_port"], conn["to_node"], conn["to_port"]);
		}
	}

	blueprint->add_connection(from_id, p_from_port, to_id, p_to_port);
	_rebuild();
}

void BlueprintEditor::_on_disconnection_request(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	if (blueprint.is_null()) {
		return;
	}
	blueprint->remove_connection(_node_id_from_gui_name(p_from), p_from_port, _node_id_from_gui_name(p_to), p_to_port);
	_rebuild();
}

void BlueprintEditor::_on_delete_nodes_request(const TypedArray<StringName> &p_nodes) {
	if (blueprint.is_null()) {
		return;
	}
	for (int i = 0; i < p_nodes.size(); i++) {
		blueprint->remove_node(_node_id_from_gui_name(p_nodes[i]));
	}
	_rebuild();
}

void BlueprintEditor::_on_popup_request(const Vector2 &p_position) {
	if (blueprint.is_null()) {
		return;
	}
	pending_add_position = (p_position + graph->get_scroll_offset()) / graph->get_zoom();
	context_menu->set_position(graph->get_screen_position() + p_position);
	context_menu->popup();
}

void BlueprintEditor::_on_add_menu_pressed(int p_id) {
	_add_node_of_type(p_id, _center_graph_position());
}

void BlueprintEditor::_on_context_menu_pressed(int p_id) {
	_add_node_of_type(p_id, pending_add_position);
}

void BlueprintEditor::_add_node_of_type(int p_type_index, const Vector2 &p_graph_position) {
	if (blueprint.is_null()) {
		return;
	}
	const Vector<BlueprintNodeDef> &defs = blueprint_get_node_defs();
	if (p_type_index < 0 || p_type_index >= defs.size()) {
		return;
	}
	blueprint->add_node(defs[p_type_index].type, p_graph_position);
	_rebuild();
}

void BlueprintEditor::_on_node_dragged(const Vector2 &p_from, const Vector2 &p_to, int p_id) {
	if (blueprint.is_valid()) {
		blueprint->set_node_position(p_id, p_to);
	}
}

void BlueprintEditor::_on_param_changed(const String &p_text, int p_id, const String &p_key) {
	if (blueprint.is_valid()) {
		blueprint->set_node_param(p_id, p_key, p_text);
	}
}

BlueprintEditor::BlueprintEditor() {
	HBoxContainer *toolbar = memnew(HBoxContainer);
	add_child(toolbar);

	Button *new_button = memnew(Button);
	new_button->set_text("Nouveau");
	toolbar->add_child(new_button);

	Button *open_button = memnew(Button);
	open_button->set_text("Ouvrir");
	toolbar->add_child(open_button);

	save_button = memnew(Button);
	save_button->set_text("Enregistrer");
	save_button->set_disabled(true);
	toolbar->add_child(save_button);

	add_menu = memnew(MenuButton);
	add_menu->set_text("Ajouter un noeud");
	add_menu->set_flat(false);
	add_menu->set_disabled(true);
	toolbar->add_child(add_menu);

	Label *tip = memnew(Label);
	tip->set_text("Clic droit dans le graphe pour ajouter un noeud. Suppr pour effacer.");
	tip->set_modulate(Color(1, 1, 1, 0.5));
	toolbar->add_child(tip);

	info_label = memnew(Label);
	info_label->set_text("");
	info_label->set_h_size_flags(SIZE_EXPAND_FILL);
	info_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
	toolbar->add_child(info_label);

	tabs = memnew(TabBar);
	tabs->set_tab_close_display_policy(TabBar::CLOSE_BUTTON_SHOW_ALWAYS);
	tabs->set_visible(false);
	add_child(tabs);

	graph = memnew(GraphEdit);
	graph->set_v_size_flags(SIZE_EXPAND_FILL);
	graph->set_right_disconnects(true);
	graph->add_valid_connection_type(PORT_TYPE_EXEC, PORT_TYPE_EXEC);
	graph->add_valid_connection_type(PORT_TYPE_DATA, PORT_TYPE_DATA);
	graph->set_visible(false);
	add_child(graph);

	empty_hint = memnew(Label);
	empty_hint->set_text("Clique sur \"Nouveau\" pour creer un blueprint, \"Ouvrir\" pour en charger un,\nou selectionne un BlueprintPlayer dans une scene.");
	empty_hint->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	empty_hint->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	empty_hint->set_v_size_flags(SIZE_EXPAND_FILL);
	empty_hint->set_modulate(Color(1, 1, 1, 0.6));
	add_child(empty_hint);

	context_menu = memnew(PopupMenu);
	add_child(context_menu);

	new_dialog = memnew(EditorFileDialog);
	new_dialog->set_file_mode(FileDialog::FILE_MODE_SAVE_FILE);
	new_dialog->set_title("Nouveau blueprint");
	new_dialog->add_filter("*.tres", "Blueprint");
	new_dialog->set_current_file("nouveau_blueprint.tres");
	add_child(new_dialog);

	open_dialog = memnew(EditorFileDialog);
	open_dialog->set_file_mode(FileDialog::FILE_MODE_OPEN_FILE);
	open_dialog->set_title("Ouvrir un blueprint");
	open_dialog->add_filter("*.tres", "Blueprint");
	add_child(open_dialog);

	_fill_node_menu(add_menu->get_popup());
	_fill_node_menu(context_menu);

	new_button->connect(SceneStringName(pressed), callable_mp(this, &BlueprintEditor::_on_new_pressed));
	open_button->connect(SceneStringName(pressed), callable_mp(this, &BlueprintEditor::_on_open_pressed));
	save_button->connect(SceneStringName(pressed), callable_mp(this, &BlueprintEditor::_on_save_pressed));
	new_dialog->connect(SNAME("file_selected"), callable_mp(this, &BlueprintEditor::_on_new_file_selected));
	open_dialog->connect(SNAME("file_selected"), callable_mp(this, &BlueprintEditor::_on_open_file_selected));

	tabs->connect(SNAME("tab_changed"), callable_mp(this, &BlueprintEditor::_on_tab_changed));
	tabs->connect(SNAME("tab_close_pressed"), callable_mp(this, &BlueprintEditor::_on_tab_close_pressed));

	add_menu->get_popup()->connect(SNAME("id_pressed"), callable_mp(this, &BlueprintEditor::_on_add_menu_pressed));
	context_menu->connect(SNAME("id_pressed"), callable_mp(this, &BlueprintEditor::_on_context_menu_pressed));

	graph->connect(SNAME("connection_request"), callable_mp(this, &BlueprintEditor::_on_connection_request));
	graph->connect(SNAME("disconnection_request"), callable_mp(this, &BlueprintEditor::_on_disconnection_request));
	graph->connect(SNAME("delete_nodes_request"), callable_mp(this, &BlueprintEditor::_on_delete_nodes_request));
	graph->connect(SNAME("popup_request"), callable_mp(this, &BlueprintEditor::_on_popup_request));
}

const Ref<Texture2D> BlueprintEditorPlugin::get_plugin_icon() const {
	return EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GraphEdit"), EditorStringName(EditorIcons));
}

void BlueprintEditorPlugin::edit(Object *p_object) {
	Ref<Blueprint> bp;

	BlueprintPlayer *player = Object::cast_to<BlueprintPlayer>(p_object);
	if (player) {
		bp = player->get_blueprint();
		if (bp.is_null()) {
			bp.instantiate();
			player->set_blueprint(bp);
		}
	} else {
		bp = Ref<Blueprint>(Object::cast_to<Blueprint>(p_object));
	}

	blueprint_editor->edit(bp);
}

bool BlueprintEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<Blueprint>(p_object) != nullptr || Object::cast_to<BlueprintPlayer>(p_object) != nullptr;
}

void BlueprintEditorPlugin::make_visible(bool p_visible) {
	blueprint_editor->set_visible(p_visible);
}

BlueprintEditorPlugin::BlueprintEditorPlugin() {
	blueprint_editor = memnew(BlueprintEditor);
	blueprint_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(blueprint_editor);
	blueprint_editor->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	blueprint_editor->hide();
}
