/**************************************************************************/
/*  import_pipeline_editor_plugin.cpp                                     */
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

#include "import_pipeline_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/io/resource_importer.h"
#include "core/math/math_funcs.h"
#include "core/math/projection.h"
#include "core/os/keyboard.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_quick_open.h"
#include "editor/editor_resource_preview.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/import/import_pipeline.h"
#include "editor/import/import_pipeline_plugin.h"
#include "editor/import_dock.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/node_3d_editor_gizmos.h"
#include "editor/scene_tree_dock.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/decal.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/3d/world_environment.h"
#include "scene/gui/center_container.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/sky_material.h"
#include "scene/resources/surface_tool.h"

class ImporterStep : public ImportPipelineStep {
	GDCLASS(ImporterStep, ImportPipelineStep)
	friend class ImportPipelineEditor;

	List<ResourceImporter::ImportOption> options;
	HashMap<StringName, Variant> parameters;
	Ref<ResourceImporter> importer;
	String path;
	Ref<Resource> result;

	bool _set(const StringName &p_name, const Variant &p_value) {
		parameters[p_name] = p_value;
		return true;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (p_name == "result") {
			r_ret = result;
			return true;
		}
		if (parameters.has(p_name)) {
			r_ret = parameters[p_name];
			return true;
		}
		return false;
	}

	String get_result_type() const {
		return importer->get_resource_type();
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		for (ResourceImporter::ImportOption option : options) {
			p_list->push_back(option.option);
		}
		p_list->push_back(PropertyInfo(Variant::OBJECT, "result", PROPERTY_HINT_RESOURCE_TYPE, get_result_type()));
	}

	void update() override {
		String temp_path = ResourceFormatImporter::get_singleton()->get_import_base_path(path) + "-temp";
		importer->import(path, temp_path, parameters, nullptr);
		temp_path += "." + importer->get_save_extension();
		result = ResourceLoader::load(temp_path, "");
		result->set_path("");
		Ref<DirAccess> dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		dir_access->remove(temp_path);
	}

	PackedStringArray get_outputs() override {
		PackedStringArray results;
		results.append("result");
		return results;
	}

	void set_path(const String &p_path) {
		path = p_path;
		set_step_name(vformat("Import '%s'  ", path));
		options.clear();
		parameters.clear();
		if (FileAccess::exists(path + ".import")) {
			Ref<ConfigFile> config;
			config.instantiate();
			config->load(path + ".import");
			List<String> keys;
			config->get_section_keys("params", &keys);
			for (const String &E : keys) {
				Variant value = config->get_value("params", E);
				parameters[E] = value;
			}
			String importer_name = config->get_value("remap", "importer");
			importer = ResourceFormatImporter::get_singleton()->get_importer_by_name(importer_name);
			importer->get_import_options(path, &options);

			for (const ResourceImporter::ImportOption &E : options) {
				if (!parameters.has(E.option.name)) {
					parameters[E.option.name] = E.default_value;
				}
			}
		}
		notify_property_list_changed();
	}
};

class LoaderStep : public ImportPipelineStep {
	GDCLASS(LoaderStep, ImportPipelineStep)
	friend class ImportPipelineEditor;

	Ref<Resource> result;
	String path;

	bool _set(const StringName &p_name, const Variant &p_value) {
		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (p_name == "result") {
			r_ret = result;
			return true;
		}
		return false;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "result", PROPERTY_HINT_RESOURCE_TYPE, result->get_class()));
	}

	void update() override {
		result = ResourceLoader::load(path);
	}

	PackedStringArray get_outputs() override {
		PackedStringArray results;
		results.append("result");
		return results;
	}

	void load(const String &p_path) {
		path = p_path;
		set_step_name(vformat("Load '%s'", path));
		update();
	}
};

class OverwriterStep : public ImportPipelineStep {
	GDCLASS(OverwriterStep, ImportPipelineStep)
	friend class ImportPipelineEditor;

	String type;

	void _get_property_list(List<PropertyInfo> *p_list) const {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "source", PROPERTY_HINT_RESOURCE_TYPE, type));
	}

	PackedStringArray get_inputs() override {
		PackedStringArray results;
		results.append("source");
		return results;
	}
};

class SaverStep : public ImportPipelineStep {
	GDCLASS(SaverStep, ImportPipelineStep)
	friend class ImportPipelineEditor;

	String name;

	bool _set(const StringName &p_name, const Variant &p_value) {
		if (p_name == "source") {
			return true;
		} else if (p_name == "name") {
			name = p_value;
			emit_signal("name_changed", get_display_name());
			return true;
		}
		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (p_name == "name") {
			r_ret = name;
			return true;
		}
		return false;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "source", PROPERTY_HINT_RESOURCE_TYPE));
		p_list->push_back(PropertyInfo(Variant::STRING, "name"));
	}

	String get_display_name() {
		if (name.is_empty()) {
			return vformat("Save (empty name)");
		} else {
			return vformat("Save \"%s\"", name);
		}
	}

	PackedStringArray get_inputs() override {
		PackedStringArray results;
		results.append("source");
		return results;
	}
};

//wrapper object to filter the in- and outputs of the step
class NodeData : public Object {
	GDCLASS(NodeData, Object)
	friend class ImportPipelineEditor;

	List<PropertyInfo> options;
	Ref<ImportPipelineStep> step;
	bool results_valid = false;
	ImportPipelineEditor *pipeline;
	String name;
	HashSet<String> connection_names;

	struct Source {
		StringName name;
		String type;

		//set if connected
		String source_node;
		int source_idx;
	};
	List<Source> sources;

	struct Result {
		StringName name;
		String type;
	};

	List<Result> results;

	bool _set(const StringName &p_name, const Variant &p_value) {
		bool valid;
		step->set(p_name, p_value, &valid);
		results_valid = false;
		if (!connection_names.has(p_name)) {
			pipeline->_update_preview();
		}
		return valid;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		bool valid;
		r_ret = step->get(p_name, &valid);
		return valid;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		for (PropertyInfo option : options) {
			p_list->push_back(option);
		}
	}

	bool _property_can_revert(const StringName &p_name) const {
		return step->property_can_revert(p_name);
	}

	bool _property_get_revert(const StringName &p_name, Variant &r_property) const {
		r_property = step->property_get_revert(p_name);
		return true;
	}

	void update() {
		options.clear();
		connection_names.clear();
		PackedStringArray inputs = step->get_inputs();
		PackedStringArray outputs = step->get_outputs();
		for (const String &conn : inputs) {
			connection_names.insert(conn);
		}
		for (const String &conn : outputs) {
			connection_names.insert(conn);
		}
		List<PropertyInfo> properties;
		step->get_property_list(&properties, true);
		for (PropertyInfo property : properties) {
			if (inputs.has(property.name) || outputs.has(property.name)) {
				continue;
			}
			options.push_back(property);
		}
		notify_property_list_changed();
	}

	void update_connections() {
		sources.clear();
		results.clear();
		List<PropertyInfo> properties;
		step->get_property_list(&properties, true);

		PackedStringArray inputs = step->get_inputs();
		PackedStringArray outputs = step->get_outputs();

		for (PropertyInfo property : properties) {
			if (inputs.has(property.name)) {
				Source source;
				source.name = property.name;
				source.type = property.hint_string;
				if (source.type.is_empty()) {
					source.type = "Resource";
				}
				source.source_node = "";
				source.source_idx = -1;
				sources.push_back(source);
			} else if (outputs.has(property.name)) {
				Result result;
				result.name = property.name;
				result.type = property.hint_string;
				if (result.type.is_empty()) {
					result.type = "Resource";
				}
				results.push_back(result);
			}
		}
	}
};

ImportPipelineEditor *ImportPipelineEditor::singleton = nullptr;

void ImportPipelineEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			List<String> extensions;
			ResourceLoader::get_recognized_extensions_for_type("Resource", &extensions);
			for (String extension : extensions) {
				source_dialog->add_filter("*." + extension);
				load_dialog->add_filter("*." + extension);
			}

			extensions.clear();
			ResourceLoader::get_recognized_extensions_for_type("Script", &extensions);
			for (const String &extension : extensions) {
				if (extension == "res" || extension == "tres") {
					//script could use these, but nobody should
					continue;
				}
				script_dialog->add_filter("*." + extension, "Script");
			}

			_import_plugins_changed();
			ImportPipelinePlugins::get_singleton()->connect("plugins_changed", callable_mp(this, &ImportPipelineEditor::_import_plugins_changed));
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				_update_preview();
			}
		} break;
	}
}

void ImportPipelineEditor::_import_plugins_changed() {
	add_menu->clear();
	for (int i = add_menu->get_child_count() - 1; i >= 0; i--) {
		PopupMenu *sub_menu = Object::cast_to<PopupMenu>(add_menu->get_child(i));
		if (sub_menu != nullptr) {
			sub_menu->queue_free();
		}
	}
	add_menu->add_item("Source");
	add_menu->add_separator();
	add_menu->add_item("Sink");
	add_menu->add_separator();
	add_menu->add_item("Custom Script");
	add_menu->add_separator();

	for (int i = 0; i < ImportPipelinePlugins::get_singleton()->get_plugin_count(); i++) {
		Ref<ImportPipelinePlugin> plugin = ImportPipelinePlugins::get_singleton()->get_plugin(i);
		String category = plugin->get_category();
		PackedStringArray avaible_steps = plugin->get_avaible_steps();
		PopupMenu *sub_menu = Object::cast_to<PopupMenu>(add_menu->get_node_or_null(category));
		if (sub_menu == nullptr) {
			sub_menu = memnew(PopupMenu);
			sub_menu->set_name(category);
			add_menu->add_child(sub_menu);
			//sub_menu->connect("id_pressed", callable_mp(this, &ImportPipelineEditor::_create_step).bind(sub_menu));
			sub_menu->connect("id_pressed", callable_mp(this, &ImportPipelineEditor::_create_step).bind(sub_menu));
			add_menu->add_submenu_item(category, category);
		}
		for (String step : avaible_steps) {
			sub_menu->add_item(step);
		}
	}
}

void ImportPipelineEditor::_reset_pipeline() {
	if (load_path == "clear") {
		pipeline_data = Ref<ImportPipeline>();
	} else if (load_path.is_empty() || !FileAccess::exists(load_path + ".import")) {
		print_error(vformat("Can not create pipeline for '%s', because it is not imported from an external file.", load_path));
		return;
	} else {
		ConfigFile cf;
		cf.load(load_path + ".import");
		if (cf.has_section("post_import")) {
			pipeline_data = cf.get_value("post_import", "pipeline");
		} else {
			pipeline_data.instantiate();
		}
	}
	_load();
}

void ImportPipelineEditor::_node_selected(Node *p_node) {
	_update_settings(p_node->get_name());
}

void ImportPipelineEditor::_node_deselected(Node *p_node) {
	_update_settings("");
}

void ImportPipelineEditor::_update_settings(const String &p_node) {
	if (p_node != "") {
		NodeData *node_data = steps[p_node];
		_get_result(p_node, -1);
		Node *tree = node_data->step->get_tree();
		if (tree != nullptr && tree->get_parent() == nullptr) {
			sub_viewport->add_child(tree);
		}
		settings_inspector->edit(node_data, tree);
	} else {
		settings_inspector->edit(nullptr);
	}
	settings_node = p_node;
}

bool ImportPipelineEditor::_has_cycle(const String &current, const String &target) {
	if (current == target) {
		return true;
	}
	NodeData *node_data = steps[current];
	for (NodeData::Source source : node_data->sources) {
		if (source.source_node != "" && _has_cycle(source.source_node, target)) {
			return true;
		}
	}
	return false;
}

void ImportPipelineEditor::_connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index) {
	NodeData *to = steps[p_to];
	NodeData::Source current = to->sources[p_to_index];
	if (current.source_node == p_from && current.source_idx == p_from_index) {
		graph->disconnect_node(p_from, p_from_index, p_to, p_to_index);
		to->sources[p_to_index].source_node = "";
	} else {
		if (_has_cycle(p_from, p_to)) {
			print_line("cycle");
			return;
		}
		String source_type = steps[p_from]->results[p_from_index].type;
		String target_type = current.type;
		if (!ClassDB::is_parent_class(source_type, target_type)) {
			print_line("types mismatch", source_type, target_type);
			return;
		}
		if (current.source_node != "") {
			graph->disconnect_node(current.source_node, current.source_idx, p_to, p_to_index);
		}
		graph->connect_node(p_from, p_from_index, p_to, p_to_index);
		to->sources[p_to_index].source_node = p_from;
		to->sources[p_to_index].source_idx = p_from_index;
	}
	_get_result(p_to, -1);
	pipeline_state = PIPELINE_UNSAVED;
	_update_preview();
	_update_settings(settings_node);
}

Ref<Resource> ImportPipelineEditor::_get_result(StringName p_node, int p_idx) {
	NodeData *current = steps[p_node];
	bool valid = current->results_valid;

	for (NodeData::Source source : current->sources) {
		Ref<Resource> source_res;
		if (source.source_node == "") {
			source_res = Ref<Resource>();
		} else {
			source_res = _get_result(source.source_node, source.source_idx);
		}
		if (current->step->get(source.name) != source_res) {
			current->set(source.name, source_res);
			current->step->source_changed();
			valid = false;
		}
	}

	if (!valid) {
		current->step->update();
		current->results_valid = true;
	}
	if (p_idx >= 0) {
		return current->step->get(current->results[p_idx].name);
	} else {
		return nullptr;
	}
}

void ImportPipelineEditor::_update_preview() {
	if (preview_node.is_empty()) {
		preview_inspector->edit(nullptr);
		preview_resource = Ref<Resource>();
	} else {
		Ref<Resource> result = _get_result(preview_node, preview_idx);
		preview_inspector->edit(result.ptr());
		preview_resource = result;
	}
}

void ImportPipelineEditor::_result_button_pressed(StringName p_node, int p_idx) {
	if (preview_node == p_node && preview_idx == p_idx) {
		preview_node = "";
	} else {
		if (!preview_node.is_empty()) {
			NodeData *old = steps[preview_node];
			Button *old_button = Object::cast_to<Button>(graph->get_node(NodePath(preview_node))->get_child(preview_idx + old->sources.size()));
			old_button->set_pressed(false);
		}
		preview_node = p_node;
		preview_idx = p_idx;
	}
	_update_preview();
}

StringName ImportPipelineEditor::_create_importer_node(Vector2 p_position, const String &p_path) {
	Ref<ImporterStep> step = memnew(ImporterStep);
	step->set_path(p_path);
	StringName name = _create_node(step, p_position, false);
	importer_node_name = name;
	return name;
}

StringName ImportPipelineEditor::_create_overwritter_node(Vector2 p_position) {
	Ref<OverwriterStep> step = memnew(OverwriterStep);
	step->set_step_name("Overwrite Imported  ");
	StringName name = _create_node(step, p_position, false);
	return name;
}

StringName ImportPipelineEditor::_create_loader_node(Vector2 p_position, const String &p_path) {
	Ref<LoaderStep> step = memnew(LoaderStep);
	step->load(p_path);
	return _create_node(step, p_position);
}

StringName ImportPipelineEditor::_create_saver_node(Vector2 p_position, const String &p_name) {
	Ref<SaverStep> step = memnew(SaverStep);
	step->name = p_name;
	step->set_step_name(step->get_display_name());
	return _create_node(step, p_position, "");
}

void ImportPipelineEditor ::_remove_node(StringName p_node) {
	if (preview_node == p_node) {
		preview_node = "";
		_update_preview();
	}
	if (settings_node == p_node) {
		_node_deselected(graph->get_node(NodePath(p_node)));
	}

	NodeData *node_data = steps[p_node];
	memfree(node_data);
	steps.erase(p_node);

	List<GraphEdit::Connection> connection_list;
	graph->get_connection_list(&connection_list);
	for (GraphEdit::Connection connection : connection_list) {
		if (connection.from == p_node) {
			steps[connection.to]->sources[connection.to_port].source_node = "";
			graph->disconnect_node(connection.from, connection.from_port, connection.to, connection.to_port);
		} else if (connection.to == p_node) {
			graph->disconnect_node(connection.from, connection.from_port, connection.to, connection.to_port);
		}
	}
	Node *node = graph->get_node(NodePath(p_node));
	graph->remove_child(node);
	memdelete(node);
	pipeline_state = PIPELINE_UNSAVED;
}

void ImportPipelineEditor::_update_node(StringName p_node) {
	NodeData *node_data = steps[p_node];
	GraphNode *node = Object::cast_to<GraphNode>(graph->get_node(NodePath(p_node)));
	for (int i = node->get_child_count() - 1; i >= 0; i--) {
		node->set_slot_enabled_left(i, false);
		node->set_slot_enabled_right(i, false);
		Node *child = node->get_child(i);
		node->remove_child(child);
		memdelete(child);
	}
	node->reset_size();

	struct Connection {
		String property;
		String type;
		StringName node;
		int index;
		int self_index;
	};

	List<Connection> sources;
	List<Connection> results;

	List<GraphEdit::Connection> graph_connections;
	graph->get_connection_list(&graph_connections);
	for (GraphEdit::Connection connection : graph_connections) {
		if (connection.to == p_node) {
			Connection conn;
			NodeData::Source source = node_data->sources[connection.to_port];
			conn.property = source.name;
			conn.type = source.type;
			conn.node = source.source_node;
			conn.index = source.source_idx;
			conn.self_index = connection.to_port;
			sources.push_back(conn);
			source.source_idx = -1;
			source.source_node = "";
			node_data->sources[connection.to_port] = source;
		} else if (connection.from == p_node) {
			Connection conn;
			NodeData::Result result = node_data->results[connection.from_port];
			conn.property = result.name;
			conn.type = result.type;
			conn.node = connection.to;
			conn.index = connection.to_port;
			conn.self_index = connection.from_port;
			results.push_back(conn);
		}
	}
	for (Connection conn : sources) {
		graph->disconnect_node(conn.node, conn.index, p_node, conn.self_index);
	}
	for (Connection conn : results) {
		graph->disconnect_node(p_node, conn.self_index, conn.node, conn.index);
	}

	node_data->update_connections();
	node_data->update();
	node_data->results_valid = false;

	HashMap<String, int> indices;

	for (NodeData::Source source : node_data->sources) {
		Label *label = memnew(Label);
		label->set_text(source.name);
		node->add_child(label);
		int index = label->get_index();
		node->set_slot(index, true, 0, Color(1.0, 1.0, 1.0), false, 0, Color(1.0, 1.0, 1.0));
		indices[source.name] = index;
	}
	for (int i = 0; i < node_data->results.size(); i++) {
		Button *button = memnew(Button);
		button->set_text(node_data->results[i].name);
		button->set_text_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
		button->set_toggle_mode(true);
		node->add_child(button);
		node->set_slot(button->get_index(), false, 0, Color(1.0, 1.0, 1.0), true, 0, Color(1.0, 1.0, 1.0));
		button->connect("pressed", callable_mp(this, &ImportPipelineEditor::_result_button_pressed).bind(node->get_name(), i));
		indices[node_data->results[i].name] = i;
	}
	Control *spacer = memnew(Control);
	spacer->set_custom_minimum_size(Vector2(0.0, 5.0));
	node->add_child(spacer);

	for (Connection conn : sources) {
		int index = indices[conn.property];
		NodeData::Source source = node_data->sources[index];
		if (source.type != conn.type) {
			print_line("DEBUG: source connection type changed"); //todo: handle or better print error
			continue;
		}
		graph->connect_node(conn.node, conn.index, p_node, index);
		source.source_node = conn.node;
		source.source_idx = conn.index;
		node_data->sources[index] = source;
	}
	for (Connection conn : results) {
		int index = indices[conn.property];
		NodeData::Result result = node_data->results[index];
		if (result.type != conn.type) {
			print_line("DEBUG: result connection type changed"); //todo: handle or better print error
			continue;
		}
		graph->connect_node(p_node, index, conn.node, conn.index);
	}
}

StringName ImportPipelineEditor::_create_node(Ref<ImportPipelineStep> p_step, Vector2 p_position, bool p_closable) {
	GraphNode *node = memnew(GraphNode);
	graph->add_child(node);
	StringName name = node->get_name();
	node->set_position_offset(p_position);
	if (Ref<Script>(p_step->get_script()).is_valid()) {
		node->set_title(Ref<Script>(p_step->get_script())->get_path().get_file().get_basename());
	} else {
		node->set_title(p_step->get_step_name());
	}
	if (p_closable) {
		node->set_show_close_button(true);
		node->connect("close_request", callable_mp(this, &ImportPipelineEditor ::_remove_node).bind(name), CONNECT_DEFERRED);
	}

	NodeData *node_data = memnew(NodeData);
	node_data->step = p_step;
	node_data->pipeline = this;

	node_data->name = name;
	p_step->connect("name_changed", callable_mp(node, &GraphNode::set_title));
	p_step->connect("property_list_changed", callable_mp(this, &ImportPipelineEditor::_update_node).bind(name));
	steps[name] = node_data;
	_update_node(name);
	pipeline_state = PIPELINE_UNSAVED;
	return name;
}

Vector2 ImportPipelineEditor::_get_creation_position() {
	return (graph->get_scroll_ofs() + add_menu->get_position() - graph->get_screen_position()) / graph->get_zoom();
}

void ImportPipelineEditor::_create_step(int p_idx, PopupMenu *p_menu) {
	Ref<ImportPipelineStep> step = ImportPipelinePlugins::get_singleton()->create_step(p_menu->get_name(), p_menu->get_item_text(p_idx));
	_create_node(step, _get_creation_position());
}

void ImportPipelineEditor::_create_script_step(const String &p_path) {
	if (!ResourceLoader::exists(p_path)) {
		print_error(vformat("Script '%s' does not exist.", p_path));
		return;
	}
	Ref<Script> custom_script = ResourceLoader::load(p_path);
	if (!custom_script.is_valid()) {
		print_error(vformat("Could not load script '%s'.", p_path));
		return;
	}
	if (!custom_script->is_tool()) {
		print_error(vformat("Script '%s' is not in tool mode.", p_path));
		return;
	}
	Ref<ImportPipelineStep> step = memnew(ImportPipelineStep);
	step->set_category_name("");
	step->set_step_name("");
	if (custom_script->instance_create(step.ptr()) != nullptr) {
		_create_node(step, _get_creation_position());
	}
}

void ImportPipelineEditor::_create_special_step(int p_idx) {
	if (p_idx == 0) {
		source_dialog->popup_file_dialog();
	} else if (p_idx == 2) {
		_create_saver_node(_get_creation_position(), "");
	} else if (p_idx == 4) {
		script_dialog->popup_file_dialog();
	}
}

void ImportPipelineEditor::_create_add_popup(Vector2 position) {
	if (pipeline_state == PIPELINE_NOTHING) {
		return;
	}
	add_menu->set_position(position + graph->get_screen_position());
	add_menu->popup();
}

void ImportPipelineEditor::_settings_property_changed(const String &p_name) {
	pipeline_state = PIPELINE_UNSAVED;
}

void ImportPipelineEditor::_add_source(const String &p_path) {
	_create_loader_node(_get_creation_position(), p_path);
}

Dictionary ImportPipelineEditor::get_state() const {
	Dictionary state;
	state["zoom"] = graph->get_zoom();
	state["offsets"] = graph->get_scroll_ofs();
	return state;
}

void ImportPipelineEditor::set_state(const Dictionary &p_state) {
	graph->set_zoom(p_state["zoom"]);
	graph->set_scroll_ofs(p_state["offsets"]);
}

void ImportPipelineEditor::_save() {
	if (pipeline_state != PIPELINE_UNSAVED) {
		return;
	}

	HashMap<StringName, int> step_ids;

	/*save steps*/ {
		TypedArray<Dictionary> array;
		for (KeyValue<String, NodeData *> entry : steps) {
			GraphNode *node = Object::cast_to<GraphNode>(graph->get_node(entry.key));
			NodeData *node_data = entry.value;
			Ref<ImportPipelineStep> step = node_data->step;
			Dictionary dict;
			dict["position"] = node->get_position_offset();
			if (Object::cast_to<ImporterStep>(step.ptr()) != nullptr) {
				dict["type"] = ImportPipelineStep::STEP_IMPORTER;
			} else if (Object::cast_to<OverwriterStep>(step.ptr()) != nullptr) {
				dict["type"] = ImportPipelineStep::STEP_OVERWRITER;
			} else if (Object::cast_to<LoaderStep>(step.ptr()) != nullptr) {
				dict["type"] = ImportPipelineStep::STEP_LOADER;
				dict["path"] = Object::cast_to<LoaderStep>(step.ptr())->path;
			} else if (Object::cast_to<SaverStep>(step.ptr()) != nullptr) {
				dict["type"] = ImportPipelineStep::STEP_SAVER;
				dict["name"] = Object::cast_to<SaverStep>(step.ptr())->name;
			} else {
				dict["type"] = ImportPipelineStep::STEP_DEFAULT;
				dict["category"] = step->get_category_name();
				dict["name"] = step->get_step_name();
				Ref<Script> custom_script = step->get_script();
				if (custom_script.is_valid()) {
					dict["script"] = custom_script;
				}
				PackedStringArray inputs = step->get_inputs();
				PackedStringArray outputs = step->get_outputs();
				Dictionary parameters;
				List<PropertyInfo> properties;
				step->get_property_list(&properties);
				for (PropertyInfo property : properties) {
					if ((property.usage & PROPERTY_USAGE_STORAGE) == 0) {
						continue;
					}
					if (property.name == "script" || inputs.has(property.name) || outputs.has(property.name)) {
						continue;
					}
					parameters[property.name] = step->get(property.name);
				}
				dict["parameters"] = parameters;
			}
			step_ids[node->get_name()] = array.size();
			array.push_back(dict);
		}

		pipeline_data->set_steps(array);
	}

	/*save connections*/ {
		//could be saved as one int-array, but would be less readable
		TypedArray<Dictionary> array;
		List<GraphEdit::Connection> connection_list;
		graph->get_connection_list(&connection_list);
		for (GraphEdit::Connection connection : connection_list) {
			Dictionary dict;
			dict["from"] = step_ids[connection.from];
			dict["from_port"] = connection.from_port;
			dict["to"] = step_ids[connection.to];
			dict["to_port"] = connection.to_port;
			array.push_back(dict);
		}
		pipeline_data->set_connections(array);
	}

	Ref<ImporterStep> step = steps[importer_node_name]->step;
	EditorFileSystem::get_singleton()->reimport_file_with_custom_parameters(
			path,
			step->importer->get_importer_name(),
			step->parameters,
			pipeline_data);
	pipeline_state = PIPELINE_SAVED;
}

void ImportPipelineEditor::_save_with_path(const String &p_save) {
	EditorNode::get_singleton()->save_resource_in_path(pipeline_data, p_save);
}

void ImportPipelineEditor::_node_moved() {
	pipeline_state = PIPELINE_UNSAVED;
}

void ImportPipelineEditor::change(const String &p_path) {
	load_path = p_path;
	if (pipeline_state == PIPELINE_UNSAVED) {
		overwrite_dialog->popup_centered();
	} else {
		_reset_pipeline();
	}
}

void ImportPipelineEditor::_load() {
	path = load_path;
	graph->clear_connections();
	for (int i = graph->get_child_count() - 1; i >= 0; i--) {
		if (Object::cast_to<GraphNode>(graph->get_child(i))) {
			graph->get_child(i)->queue_free();
		}
	}
	for (KeyValue<String, NodeData *> entry : steps) {
		memfree(entry.value);
	}
	steps.clear();
	preview_inspector->edit(nullptr);
	preview_node = "";
	settings_inspector->edit(nullptr);
	settings_node = "";

	if (!pipeline_data.is_valid()) {
		pipeline_state = PIPELINE_NOTHING;
		name_label->set_text("No Pipeline loaded");
		return;
	}

	TypedArray<Dictionary> steps_data = pipeline_data->get_steps();
	TypedArray<Dictionary> connection_list = pipeline_data->get_connections();
	Vector<StringName> step_names;
	step_names.resize(steps_data.size());

	String importer;
	String overwriter;
	for (int i = 0; i < steps_data.size(); i++) {
		Dictionary step_data = steps_data[i];
		StringName name;
		switch (ImportPipelineStep::StepType(int(step_data["type"]))) {
			case ImportPipelineStep::STEP_IMPORTER: {
				if (!importer.is_empty()) {
					continue;
				}
				name = _create_importer_node(step_data["position"], path);
				importer = name;
			} break;
			case ImportPipelineStep::STEP_LOADER: {
				name = _create_loader_node(step_data["position"], step_data["path"]);
			} break;
			case ImportPipelineStep::STEP_OVERWRITER: {
				if (!overwriter.is_empty()) {
					continue;
				}
				name = _create_overwritter_node(step_data["position"]);
				overwriter = name;
			} break;
			case ImportPipelineStep::STEP_SAVER: {
				name = _create_saver_node(step_data["position"], step_data["name"]);
			} break;
			case ImportPipelineStep::STEP_DEFAULT: {
				Ref<ImportPipelineStep> step = ImportPipelinePlugins::get_singleton()->create_step(step_data["category"], step_data["name"]);
				if (!step.is_valid()) {
					name = "invalid";
					break;
				}
				if (step_data.has("script")) {
					step->set_script(step_data["script"]);
				}
				Dictionary parameters = step_data["parameters"];
				Array keys = parameters.keys();
				for (int j = 0; j < keys.size(); j++) {
					String key = keys[j];
					step->set(key, parameters[key]);
				}
				name = _create_node(step, step_data["position"]);
			} break;
		}
		step_names.set(i, name);
	}

	if (importer.is_empty() && overwriter.is_empty()) {
		Dictionary connection;
		connection["from"] = step_names.size();
		connection["from_port"] = 0;
		connection["to"] = step_names.size() + 1;
		connection["to_port"] = 0;
		connection_list.append(connection);
	}
	if (importer.is_empty()) {
		String name = _create_importer_node(Vector2(300, 300), path);
		importer = name;
		step_names.push_back(name);
	}
	if (overwriter.is_empty()) {
		String name = _create_overwritter_node(Vector2(600, 300));
		overwriter = name;
		step_names.push_back(name);
	}

	steps[overwriter]->sources[0].type = steps[importer]->results[0].type;

	for (int i = 0; i < connection_list.size(); i++) {
		Dictionary con = connection_list[i];
		String from = step_names[con["from"]];
		String to = step_names[con["to"]];
		if (from == "invalid " || to == "invalid") {
			continue;
		}
		_connection_request(from, con["from_port"], to, con["to_port"]);
	}
	_get_result(importer, -1);
	name_label->set_text(vformat("Loaded: '%s'", path));
	pipeline_state = PIPELINE_SAVED;
}

ImportPipelineEditor::ImportPipelineEditor() {
	singleton = this;

	set_v_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);

	load_dialog = memnew(EditorFileDialog);
	load_dialog->connect("file_selected", callable_mp(this, &ImportPipelineEditor::change));
	load_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	load_dialog->set_show_hidden_files(false);
	load_dialog->set_display_mode(EditorFileDialog::DISPLAY_THUMBNAILS);
	load_dialog->set_title(TTR("Select Resource"));
	add_child(load_dialog);

	overwrite_dialog = memnew(ConfirmationDialog);
	add_child(overwrite_dialog);
	overwrite_dialog->set_text("Overwrite current pipeline?");
	overwrite_dialog->connect("confirmed", callable_mp(this, &ImportPipelineEditor::_reset_pipeline));

	source_dialog = memnew(EditorFileDialog);
	source_dialog->connect("file_selected", callable_mp(this, &ImportPipelineEditor::_add_source));
	source_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	source_dialog->set_show_hidden_files(false);
	source_dialog->set_display_mode(EditorFileDialog::DISPLAY_THUMBNAILS);
	source_dialog->set_title(TTR("Select Source"));
	add_child(source_dialog);

	sub_viewport = memnew(SubViewport);
	sub_viewport->set_use_own_world_3d(true);
	add_child(sub_viewport);

	add_menu = memnew(PopupMenu);
	add_child(add_menu);
	add_menu->connect("id_pressed", callable_mp(this, &ImportPipelineEditor::_create_special_step));

	script_dialog = memnew(EditorFileDialog);
	script_dialog->connect("file_selected", callable_mp(this, &ImportPipelineEditor::_create_script_step));
	script_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	script_dialog->set_show_hidden_files(false);
	script_dialog->set_display_mode(EditorFileDialog::DISPLAY_LIST);
	script_dialog->set_title(TTR("Select Script"));
	add_child(script_dialog);

	HSplitContainer *content = memnew(HSplitContainer);
	add_child(content);
	content->set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);

	VBoxContainer *left_side = memnew(VBoxContainer);
	content->add_child(left_side);

	Label *left_label = memnew(Label);
	left_side->add_child(left_label);
	left_label->set_text(TTR("Step Settings"));

	Panel *left_panel = memnew(Panel);
	left_label->add_child(left_panel);
	left_panel->set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);
	left_panel->set_draw_behind_parent(true);

	settings_inspector = memnew(EditorInspector);
	left_side->add_child(settings_inspector);
	settings_inspector->set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);
	settings_inspector->set_custom_minimum_size(Size2(300 * EDSCALE, 0));
	settings_inspector->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	settings_inspector->set_autoclear(true);
	settings_inspector->set_show_categories(false);
	settings_inspector->set_use_doc_hints(false);
	settings_inspector->set_hide_script(true);
	settings_inspector->set_hide_metadata(true);
	settings_inspector->set_property_name_style(EditorPropertyNameProcessor::get_default_inspector_style());
	settings_inspector->connect("property_edited", callable_mp(this, &ImportPipelineEditor::_settings_property_changed), CONNECT_DEFERRED);

	HSplitContainer *right_split = memnew(HSplitContainer);
	content->add_child(right_split);

	VBoxContainer *middle = memnew(VBoxContainer);
	right_split->add_child(middle);
	middle->set_v_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	middle->set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);

	HBoxContainer *top_bar = memnew(HBoxContainer);
	middle->add_child(top_bar);

	Button *load_button = memnew(Button);
	top_bar->add_child(load_button);
	load_button->set_flat(true);
	load_button->set_text(TTR("Load"));
	load_button->connect("pressed", callable_mp(load_dialog, &EditorFileDialog::popup_file_dialog));

	Button *save_button = memnew(Button);
	top_bar->add_child(save_button);
	save_button->set_flat(true);
	save_button->set_text(TTR("Save"));
	save_button->connect("pressed", callable_mp(this, &ImportPipelineEditor::_save));

	Button *clear_button = memnew(Button);
	top_bar->add_child(clear_button);
	clear_button->set_flat(true);
	clear_button->set_text(TTR("Clear"));
	clear_button->connect("pressed", callable_mp(this, &ImportPipelineEditor::change).bind("clear"));

	top_bar->add_spacer();

	name_label = memnew(Label);
	top_bar->add_child(name_label);
	name_label->set_text("No Pipeline loaded");

	graph = memnew(GraphEdit);
	middle->add_child(graph);
	graph->set_v_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	graph->connect("connection_request", callable_mp(this, &ImportPipelineEditor::_connection_request));
	graph->connect("node_selected", callable_mp(this, &ImportPipelineEditor::_node_selected));
	graph->connect("node_deselected", callable_mp(this, &ImportPipelineEditor::_node_deselected));
	graph->connect("popup_request", callable_mp(this, &ImportPipelineEditor::_create_add_popup));
	graph->connect("end_node_move", callable_mp(this, &ImportPipelineEditor::_node_moved));

	VBoxContainer *right_side = memnew(VBoxContainer);
	right_split->add_child(right_side);

	Label *right_label = memnew(Label);
	right_side->add_child(right_label);
	right_label->set_text(TTR("Preview"));

	Panel *right_panel = memnew(Panel);
	right_label->add_child(right_panel);
	right_panel->set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);
	right_panel->set_draw_behind_parent(true);

	preview_inspector = memnew(EditorInspector);
	right_side->add_child(preview_inspector);
	preview_inspector->set_custom_minimum_size(Size2(300 * EDSCALE, 0));
	preview_inspector->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	preview_inspector->set_autoclear(true);
	preview_inspector->set_show_categories(true);
	preview_inspector->set_use_doc_hints(false);
	preview_inspector->set_hide_script(false);
	preview_inspector->set_hide_metadata(true);
	preview_inspector->set_property_name_style(EditorPropertyNameProcessor::get_default_inspector_style());
	preview_inspector->set_read_only(true);
}

ImportPipelineEditor::~ImportPipelineEditor() {
	singleton = nullptr;
	for (KeyValue<String, NodeData *> entry : steps) {
		memfree(entry.value);
	}
}

///////////////////////////////////////

void ImportPipelineEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		ImportPipelineEditor::get_singleton()->show();
	} else {
		ImportPipelineEditor::get_singleton()->hide();
	}
}

void ImportPipelineEditorPlugin::edit(Object *p_object) {
	//todo: change to .import files
	//ImportPipelineEditor::get_singleton()->change(Object::cast_to<Resource>(p_object)->get_path());
}

bool ImportPipelineEditorPlugin::handles(Object *p_object) const {
	//todo: change to .import files
	//return Object::cast_to<ImportPipeline>(p_object) != nullptr;
	return false;
}

Dictionary ImportPipelineEditorPlugin::get_state() const {
	return ImportPipelineEditor::get_singleton()->get_state();
}

void ImportPipelineEditorPlugin::set_state(const Dictionary &p_state) {
	ImportPipelineEditor::get_singleton()->set_state(p_state);
}

ImportPipelineEditorPlugin::ImportPipelineEditorPlugin() {
	if (ImportPipelineEditor::get_singleton() == nullptr) {
		memnew(ImportPipelineEditor);
	}
	EditorNode::get_singleton()->get_main_screen_control()->add_child(ImportPipelineEditor::get_singleton());
	ImportPipelineEditor::get_singleton()->hide();
}
