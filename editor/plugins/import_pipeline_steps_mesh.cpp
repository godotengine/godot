/**************************************************************************/
/*  import_pipeline_steps_mesh.cpp                                        */
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

#include "import_pipeline_steps_mesh.h"

#include "core/config/project_settings.h"
#include "editor/editor_scale.h"
#include "editor/plugins/scene_preview.h"
#include "scene/gui/texture_button.h"
#include "scene/main/viewport.h"

class ImportPipelineStepExtractMesh : public ImportPipelineStep {
	GDCLASS(ImportPipelineStepExtractMesh, ImportPipelineStep);

	Ref<PackedScene> scene = nullptr;
	Ref<Mesh> mesh = nullptr;
	NodePath path = NodePath();

	Node *tree = nullptr;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("get_scene"), &ImportPipelineStepExtractMesh::get_scene);
		ClassDB::bind_method(D_METHOD("set_scene", "scene"), &ImportPipelineStepExtractMesh::set_scene);
		ClassDB::bind_method(D_METHOD("get_mesh"), &ImportPipelineStepExtractMesh::get_mesh);
		ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &ImportPipelineStepExtractMesh::set_mesh);
		ClassDB::bind_method(D_METHOD("get_path"), &ImportPipelineStepExtractMesh::get_path);
		ClassDB::bind_method(D_METHOD("set_path", "path"), &ImportPipelineStepExtractMesh::set_path);

		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "scene", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_scene", "get_scene");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
		ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "MeshInstance3D"), "set_path", "get_path");
	}

public:
	void source_changed() override {
		if (tree != nullptr) {
			tree->queue_free();
			tree = nullptr;
		}
		if (scene != nullptr) {
			tree = scene->instantiate();
		}
	}
	void update() override {
		mesh = Ref<Mesh>();
		if (tree == nullptr) {
			return;
		}
		if (path.is_empty() || !tree->has_node(path)) {
			return;
		}
		MeshInstance3D *node = Object::cast_to<MeshInstance3D>(tree->get_node(path));
		if (node == nullptr) {
			return;
		}
		mesh = node->get_mesh()->duplicate();
	}
	PackedStringArray get_inputs() override {
		PackedStringArray scenes;
		scenes.push_back("scene");
		return scenes;
	}
	PackedStringArray get_outputs() override {
		PackedStringArray scenes;
		scenes.push_back("mesh");
		return scenes;
	}
	Node *get_tree() override { return tree; }

	void set_scene(Ref<PackedScene> p_scene) { scene = p_scene; }
	Ref<PackedScene> get_scene() { return scene; }
	void set_mesh(Ref<Mesh> p_mesh) { mesh = p_mesh; }
	Ref<Mesh> get_mesh() { return mesh; }
	void set_path(NodePath p_path) { path = p_path; }
	NodePath get_path() { return path; }

	ImportPipelineStepExtractMesh() {}
	~ImportPipelineStepExtractMesh() {
		if (tree != nullptr) {
			tree->queue_free();
		}
	}
};

class ImportPipelineStepSetMaterial : public ImportPipelineStep {
	GDCLASS(ImportPipelineStepSetMaterial, ImportPipelineStep);

	Ref<Mesh> mesh = nullptr;
	Ref<Material> material = nullptr;
	Ref<Mesh> result = nullptr;
	int index = 0;

protected:
	bool _set(const StringName &p_name, const Variant &p_value) {
		if (p_name == "mesh") {
			mesh = p_value;
			notify_property_list_changed();
			return true;
		} else if (p_name == "material") {
			material = p_value;
			return true;
		} else if (p_name == "result") {
			result = p_value;
			return true;
		} else if (p_name == "index") {
			index = p_value;
			return true;
		}
		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (p_name == "mesh") {
			r_ret = mesh;
			return true;
		} else if (p_name == "material") {
			r_ret = material;
			return true;
		} else if (p_name == "result") {
			r_ret = result;
			return true;
		} else if (p_name == "index") {
			r_ret = index;
			return true;
		}
		return true;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "Material"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, "result", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"));
		if (mesh.is_valid() && mesh->get_surface_count() > 1) {
			p_list->push_back(PropertyInfo(Variant::INT, "index", PROPERTY_HINT_RANGE, "0," + itos(mesh->get_surface_count() - 1)));
		}
	}

public:
	void update() override {
		if (!mesh.is_valid()) {
			result = Ref<Mesh>();
			return;
		}
		result = mesh->duplicate();
		result->surface_set_material(index, material);
	}

	PackedStringArray get_inputs() override {
		PackedStringArray scenes;
		scenes.push_back("mesh");
		scenes.push_back("material");
		return scenes;
	}
	PackedStringArray get_outputs() override {
		PackedStringArray scenes;
		scenes.push_back("result");
		return scenes;
	}
};

class ImportPipelineStepCreateMeshInstance : public ImportPipelineStep {
	GDCLASS(ImportPipelineStepCreateMeshInstance, ImportPipelineStep);

	Ref<Mesh> mesh = nullptr;
	Transform3D transform;
	Ref<PackedScene> scene = nullptr;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("get_mesh"), &ImportPipelineStepCreateMeshInstance::get_mesh);
		ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &ImportPipelineStepCreateMeshInstance::set_mesh);
		ClassDB::bind_method(D_METHOD("get_transform"), &ImportPipelineStepCreateMeshInstance::get_transform);
		ClassDB::bind_method(D_METHOD("set_transform", "transform"), &ImportPipelineStepCreateMeshInstance::set_transform);
		ClassDB::bind_method(D_METHOD("get_scene"), &ImportPipelineStepCreateMeshInstance::get_scene);
		ClassDB::bind_method(D_METHOD("set_scene", "scene"), &ImportPipelineStepCreateMeshInstance::set_scene);

		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
		ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "transform"), "set_transform", "get_transform");
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "scene", PropertyHint::PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_scene", "get_scene");
	}

public:
	void update() override {
		if (!mesh.is_valid()) {
			scene = Ref<PackedScene>();
			return;
		}
		scene.instantiate();
		MeshInstance3D *instance = memnew(MeshInstance3D);
		instance->set_mesh(mesh);
		instance->set_transform(transform);
		instance->set_name("mesh"); //todo: make parameter
		scene->pack(instance);
		instance->queue_free();
	}

	PackedStringArray get_inputs() override {
		PackedStringArray scenes;
		scenes.push_back("mesh");
		return scenes;
	}
	PackedStringArray get_outputs() override {
		PackedStringArray scenes;
		scenes.push_back("scene");
		return scenes;
	}

	void set_mesh(Ref<Mesh> p_mesh) { mesh = p_mesh; }
	Ref<Mesh> get_mesh() { return mesh; }
	void set_transform(Transform3D p_transform) { transform = p_transform; }
	Transform3D get_transform() { return transform; }
	void set_scene(Ref<PackedScene> p_scene) { scene = p_scene; }
	Ref<PackedScene> get_scene() { return scene; }
};

class ImportPipelineStepExtractMaterial : public ImportPipelineStep {
	GDCLASS(ImportPipelineStepExtractMaterial, ImportPipelineStep);

	Ref<Mesh> mesh = nullptr;
	Ref<Material> material = nullptr;
	int index = 0;

protected:
	bool _set(const StringName &p_name, const Variant &p_value) {
		if (p_name == "mesh") {
			mesh = p_value;
			notify_property_list_changed();
			return true;
		} else if (p_name == "material") {
			material = p_value;
			return true;
		} else if (p_name == "index") {
			index = p_value;
			return true;
		}
		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (p_name == "mesh") {
			r_ret = mesh;
			return true;
		} else if (p_name == "material") {
			r_ret = material;
			return true;
		} else if (p_name == "index") {
			r_ret = index;
			return true;
		}
		return true;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "Material"));
		if (mesh.is_valid() && mesh->get_surface_count() > 1) {
			p_list->push_back(PropertyInfo(Variant::INT, "index", PROPERTY_HINT_RANGE, "0," + itos(mesh->get_surface_count() - 1)));
		}
	}

public:
	void update() override {
		if (!mesh.is_valid()) {
			material = Ref<Material>();
			return;
		}
		material = mesh->surface_get_material(index);
	}

	PackedStringArray get_inputs() override {
		PackedStringArray scenes;
		scenes.push_back("mesh");
		return scenes;
	}
	PackedStringArray get_outputs() override {
		PackedStringArray scenes;
		scenes.push_back("material");
		return scenes;
	}

	ImportPipelineStepExtractMaterial() {}
	~ImportPipelineStepExtractMaterial() {}
};

PackedStringArray ImportPipelinePluginMesh::get_avaible_steps() {
	PackedStringArray steps;
	steps.append("Extract Mesh");
	steps.append("Set Material");
	steps.append("Extract Material");
	steps.append("Create MeshInstance3D");
	return steps;
}

Ref<ImportPipelineStep> ImportPipelinePluginMesh::get_step(const String &p_name) {
	if (p_name == "Extract Mesh") {
		return memnew(ImportPipelineStepExtractMesh);
	} else if (p_name == "Set Material") {
		return memnew(ImportPipelineStepSetMaterial);
	} else if (p_name == "Extract Material") {
		return memnew(ImportPipelineStepExtractMaterial);
	} else if (p_name == "Create MeshInstance3D") {
		return memnew(ImportPipelineStepCreateMeshInstance);
	} else {
		return nullptr;
	}
}
