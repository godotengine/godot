/**************************************************************************/
/*  import_pipeline_steps_scene.cpp                                       */
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

#include "import_pipeline_steps_scene.h"

#include "core/config/project_settings.h"
#include "editor/editor_scale.h"
#include "editor/plugins/scene_preview.h"
#include "scene/gui/texture_button.h"
#include "scene/main/viewport.h"

void recursive_set_owner(Node *p_node, Node *p_owner) {
	if (p_node == p_owner) {
		p_node->set_owner(nullptr);
	} else {
		p_node->set_owner(p_owner);
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {
		recursive_set_owner(p_node->get_child(i), p_owner);
	}
}

class ImportPipelineStepRemoveNode : public ImportPipelineStep {
	GDCLASS(ImportPipelineStepRemoveNode, ImportPipelineStep);

protected:
	Ref<PackedScene> source = nullptr;
	Ref<PackedScene> result = nullptr;
	Ref<PackedScene> removed = nullptr;
	NodePath path = NodePath();
	Node *tree = nullptr;

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("get_source"), &ImportPipelineStepRemoveNode::get_source);
		ClassDB::bind_method(D_METHOD("set_source", "source"), &ImportPipelineStepRemoveNode::set_source);

		ClassDB::bind_method(D_METHOD("get_result"), &ImportPipelineStepRemoveNode::get_result);
		ClassDB::bind_method(D_METHOD("set_result", "result"), &ImportPipelineStepRemoveNode::set_result);

		ClassDB::bind_method(D_METHOD("get_removed"), &ImportPipelineStepRemoveNode::get_removed);
		ClassDB::bind_method(D_METHOD("set_removed", "removed"), &ImportPipelineStepRemoveNode::set_removed);

		ClassDB::bind_method(D_METHOD("get_path"), &ImportPipelineStepRemoveNode::get_path);
		ClassDB::bind_method(D_METHOD("set_path", "path"), &ImportPipelineStepRemoveNode::set_path);

		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "source", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_source", "get_source");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "result", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_result", "get_result");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "removed", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_removed", "get_removed");
		ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "path"), "set_path", "get_path");
	}

public:
	void source_changed() override {
		if (tree != nullptr) {
			tree->queue_free();
			tree = nullptr;
		}
		if (source != nullptr) {
			tree = source->instantiate();
		}
	}

	void update() override {
		result = Ref<Resource>();
		removed = Ref<Resource>();
		if (tree == nullptr) {
			return;
		}
		if (path.is_empty() || !tree->has_node(path)) {
			result.instantiate();
			result->pack(tree);
			return;
		}
		Node *node = tree->get_node(path);
		if (node == tree) {
			removed.instantiate();
			removed->pack(tree);
			return;
		}
		result.instantiate();
		removed.instantiate();
		recursive_set_owner(node, node);
		result->pack(tree);
		removed->pack(node);
		recursive_set_owner(node, tree);
	}

	PackedStringArray get_inputs() override {
		PackedStringArray sources;
		sources.push_back("source");
		return sources;
	}

	PackedStringArray get_outputs() override {
		PackedStringArray sources;
		sources.push_back("result");
		sources.push_back("removed");
		return sources;
	}

	Node *get_tree() override { return tree; }

	void set_source(Ref<PackedScene> p_source) { source = p_source; }
	Ref<PackedScene> get_source() { return source; }
	void set_result(Ref<PackedScene> p_result) { result = p_result; }
	Ref<PackedScene> get_result() { return result; }
	void set_removed(Ref<PackedScene> p_removed) { result = p_removed; }
	Ref<PackedScene> get_removed() { return removed; }
	void set_path(NodePath p_path) { path = p_path; }
	NodePath get_path() { return path; }

	~ImportPipelineStepRemoveNode() {
		if (tree != nullptr) {
			tree->queue_free();
		}
	}
};

class ImportPipelineStepAddNode : public ImportPipelineStep {
	GDCLASS(ImportPipelineStepAddNode, ImportPipelineStep);

protected:
	Ref<PackedScene> source = nullptr;
	Ref<PackedScene> addition = nullptr;
	Ref<PackedScene> result = nullptr;
	NodePath path = NodePath();
	Node *tree = nullptr;

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("get_source"), &ImportPipelineStepAddNode::get_source);
		ClassDB::bind_method(D_METHOD("set_source", "source"), &ImportPipelineStepAddNode::set_source);

		ClassDB::bind_method(D_METHOD("get_addition"), &ImportPipelineStepAddNode::get_addition);
		ClassDB::bind_method(D_METHOD("set_addition", "addition"), &ImportPipelineStepAddNode::set_addition);

		ClassDB::bind_method(D_METHOD("get_result"), &ImportPipelineStepAddNode::get_result);
		ClassDB::bind_method(D_METHOD("set_result", "result"), &ImportPipelineStepAddNode::set_result);

		ClassDB::bind_method(D_METHOD("get_path"), &ImportPipelineStepAddNode::get_path);
		ClassDB::bind_method(D_METHOD("set_path", "path"), &ImportPipelineStepAddNode::set_path);

		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "source", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_source", "get_source");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "addition", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_addition", "get_addition");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "result", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_result", "get_result");
		ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "path"), "set_path", "get_path");
	}

public:
	void source_changed() override {
		if (tree != nullptr) {
			tree->queue_free();
			tree = nullptr;
		}
		if (source != nullptr) {
			tree = source->instantiate();
		}
	}

	void update() override {
		result = Ref<Resource>();
		if (tree == nullptr) {
			return;
		}
		result.instantiate();
		if (path.is_empty() || !tree->has_node(path) || !addition.is_valid()) {
			result->pack(tree);
			return;
		}
		Node *node = tree->get_node(path);
		Node *add = addition->instantiate();
		node->add_child(add);
		recursive_set_owner(add, tree);
		result->pack(tree);
		add->queue_free();
	}

	PackedStringArray get_inputs() override {
		PackedStringArray sources;
		sources.push_back("source");
		sources.push_back("addition");
		return sources;
	}

	PackedStringArray get_outputs() override {
		PackedStringArray sources;
		sources.push_back("result");
		return sources;
	}

	Node *get_tree() override { return tree; }

	void set_source(Ref<PackedScene> p_source) { source = p_source; }
	Ref<PackedScene> get_source() { return source; }
	void set_result(Ref<PackedScene> p_result) { result = p_result; }
	Ref<PackedScene> get_result() { return result; }
	void set_addition(Ref<PackedScene> p_addition) { addition = p_addition; }
	Ref<PackedScene> get_addition() { return addition; }
	void set_path(NodePath p_path) { path = p_path; }
	NodePath get_path() { return path; }

	~ImportPipelineStepAddNode() {
		if (tree != nullptr) {
			tree->queue_free();
		}
	}
};

class ImportPipelineStepGetNodes : public ImportPipelineStep {
	GDCLASS(ImportPipelineStepGetNodes, ImportPipelineStep);

protected:
	Ref<PackedScene> scene;

	PackedStringArray names;
	HashMap<String, Ref<PackedScene>> nodes;

	bool _set(const StringName &p_name, const Variant &p_value) {
		if (p_name == "scene") {
			scene = p_value;
			return true;
		}
		if (nodes.has(p_name)) {
			nodes[p_name] = p_value;
			return true;
		}
		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (p_name == "scene") {
			r_ret = scene;
			return true;
		}
		if (nodes.has(p_name)) {
			r_ret = nodes[p_name];
			return true;
		}
		return false;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "scene", PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"));
		for (const String &name : names) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, name, PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"));
		}
	}

public:
	void iterate(Node *p_node, const String &p_prefix) {
		String name = p_prefix + p_node->get_name();

		Ref<PackedScene> packed_scene;
		packed_scene.instantiate();
		packed_scene->pack(p_node);
		nodes[name] = packed_scene;
		names.append(name);

		name += "-";

		for (int i = 0; i < p_node->get_child_count(); i++) {
			iterate(p_node->get_child(i), name);
		}
	}

	void source_changed() override {
		names.clear();
		nodes.clear();
		if (scene == nullptr) {
			return;
		}
		Node *node = scene->instantiate();
		recursive_set_owner(node, nullptr);
		iterate(node, "");
		node->queue_free();
		notify_property_list_changed();
	}

	PackedStringArray get_inputs() override {
		PackedStringArray sources;
		sources.push_back("scene");
		return sources;
	}

	PackedStringArray get_outputs() override {
		return names;
	}
};

PackedStringArray ImportPipelinePluginScene::get_avaible_steps() {
	PackedStringArray steps;
	steps.append("Add Node");
	steps.append("Remove Node");
	steps.append("Get Nodes");
	return steps;
}

Ref<ImportPipelineStep> ImportPipelinePluginScene::get_step(const String &p_name) {
	if (p_name == "Add Node") {
		return memnew(ImportPipelineStepAddNode);
	} else if (p_name == "Remove Node") {
		return memnew(ImportPipelineStepRemoveNode);
	} else if (p_name == "Get Nodes") {
		return memnew(ImportPipelineStepGetNodes);
	} else {
		return nullptr;
	}
}
