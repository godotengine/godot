/**************************************************************************/
/*  test_gltf_extras.h                                                    */
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

#ifndef TEST_GLTF_EXTRAS_H
#define TEST_GLTF_EXTRAS_H

#include "tests/test_macros.h"

#ifdef TOOLS_ENABLED

#include "core/os/os.h"
#include "editor/import/3d/resource_importer_scene.h"
#include "modules/gltf/editor/editor_scene_importer_gltf.h"
#include "modules/gltf/gltf_document.h"
#include "modules/gltf/gltf_state.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/main/window.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/material.h"
#include "scene/resources/packed_scene.h"

namespace TestGltfExtras {

static Node *_gltf_export_then_import(Node *p_root, String &p_tempfilebase) {
	Ref<GLTFDocument> doc;
	doc.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	Error err = doc->append_from_scene(p_root, state, EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS);
	CHECK_MESSAGE(err == OK, "GLTF state generation failed.");
	err = doc->write_to_filesystem(state, p_tempfilebase + ".gltf");
	CHECK_MESSAGE(err == OK, "Writing GLTF to cache dir failed.");

	// Setting up importers.
	Ref<ResourceImporterScene> import_scene = memnew(ResourceImporterScene("PackedScene", true));
	ResourceFormatImporter::get_singleton()->add_importer(import_scene);
	Ref<EditorSceneFormatImporterGLTF> import_gltf;
	import_gltf.instantiate();
	ResourceImporterScene::add_scene_importer(import_gltf);

	// GTLF importer behaves differently outside of editor, it's too late to modify Engine::get_editor_hint
	// as the registration of runtime extensions already happened, so remove them. See modules/gltf/register_types.cpp
	GLTFDocument::unregister_all_gltf_document_extensions();

	HashMap<StringName, Variant> options(20);
	options["nodes/root_type"] = "";
	options["nodes/root_name"] = "";
	options["nodes/apply_root_scale"] = true;
	options["nodes/root_scale"] = 1.0;
	options["meshes/ensure_tangents"] = true;
	options["meshes/generate_lods"] = false;
	options["meshes/create_shadow_meshes"] = true;
	options["meshes/light_baking"] = 1;
	options["meshes/lightmap_texel_size"] = 0.2;
	options["meshes/force_disable_compression"] = false;
	options["skins/use_named_skins"] = true;
	options["animation/import"] = true;
	options["animation/fps"] = 30;
	options["animation/trimming"] = false;
	options["animation/remove_immutable_tracks"] = true;
	options["import_script/path"] = "";
	options["_subresources"] = Dictionary();
	options["gltf/naming_version"] = 1;

	// Process gltf file, note that this generates `.scn` resource from the 2nd argument.
	err = import_scene->import(p_tempfilebase + ".gltf", p_tempfilebase, options, nullptr, nullptr, nullptr);
	CHECK_MESSAGE(err == OK, "GLTF import failed.");
	ResourceImporterScene::remove_scene_importer(import_gltf);

	Ref<PackedScene> packed_scene = ResourceLoader::load(p_tempfilebase + ".scn", "", ResourceFormatLoader::CACHE_MODE_REPLACE, &err);
	CHECK_MESSAGE(err == OK, "Loading scene failed.");
	Node *p_scene = packed_scene->instantiate();
	return p_scene;
}

TEST_CASE("[SceneTree][Node] GLTF test mesh and material meta export and import") {
	// Setup scene.
	Ref<StandardMaterial3D> original_material = memnew(StandardMaterial3D);
	original_material->set_albedo(Color(1.0, .0, .0));
	original_material->set_name("material");
	Dictionary material_dict;
	material_dict["node_type"] = "material";
	original_material->set_meta("extras", material_dict);

	Ref<PlaneMesh> original_meshdata = memnew(PlaneMesh);
	original_meshdata->set_name("planemesh");
	Dictionary meshdata_dict;
	meshdata_dict["node_type"] = "planemesh";
	original_meshdata->set_meta("extras", meshdata_dict);
	original_meshdata->surface_set_material(0, original_material);

	MeshInstance3D *original_mesh_instance = memnew(MeshInstance3D);
	original_mesh_instance->set_mesh(original_meshdata);
	original_mesh_instance->set_name("mesh_instance_3d");
	Dictionary mesh_instance_dict;
	mesh_instance_dict["node_type"] = "mesh_instance_3d";
	original_mesh_instance->set_meta("extras", mesh_instance_dict);

	Node3D *original = memnew(Node3D);
	SceneTree::get_singleton()->get_root()->add_child(original);
	original->add_child(original_mesh_instance);
	original->set_name("node3d");
	Dictionary node_dict;
	node_dict["node_type"] = "node3d";
	original->set_meta("extras", node_dict);
	original->set_meta("meta_not_nested_under_extras", "should not propagate");

	// Convert to GLFT and back.
	String tempfile = OS::get_singleton()->get_cache_path().path_join("gltf_extras");
	Node *loaded = _gltf_export_then_import(original, tempfile);

	// Compare the results.
	CHECK(loaded->get_name() == "node3d");
	CHECK(Dictionary(loaded->get_meta("extras")).size() == 1);
	CHECK(Dictionary(loaded->get_meta("extras"))["node_type"] == "node3d");
	CHECK_FALSE(loaded->has_meta("meta_not_nested_under_extras"));
	CHECK_FALSE(Dictionary(loaded->get_meta("extras")).has("meta_not_nested_under_extras"));

	MeshInstance3D *mesh_instance_3d = Object::cast_to<MeshInstance3D>(loaded->find_child("mesh_instance_3d", false, true));
	CHECK(mesh_instance_3d->get_name() == "mesh_instance_3d");
	CHECK(Dictionary(mesh_instance_3d->get_meta("extras"))["node_type"] == "mesh_instance_3d");

	Ref<Mesh> mesh = mesh_instance_3d->get_mesh();
	CHECK(Dictionary(mesh->get_meta("extras"))["node_type"] == "planemesh");

	Ref<Material> material = mesh->surface_get_material(0);
	CHECK(material->get_name() == "material");
	CHECK(Dictionary(material->get_meta("extras"))["node_type"] == "material");

	memdelete(original_mesh_instance);
	memdelete(original);
	memdelete(loaded);
}

TEST_CASE("[SceneTree][Node] GLTF test skeleton and bone export and import") {
	// Setup scene.
	Skeleton3D *skeleton = memnew(Skeleton3D);
	skeleton->set_name("skeleton");
	Dictionary skeleton_extras;
	skeleton_extras["node_type"] = "skeleton";
	skeleton->set_meta("extras", skeleton_extras);

	skeleton->add_bone("parent");
	skeleton->set_bone_rest(0, Transform3D());
	Dictionary parent_bone_extras;
	parent_bone_extras["bone"] = "i_am_parent_bone";
	skeleton->set_bone_meta(0, "extras", parent_bone_extras);

	skeleton->add_bone("child");
	skeleton->set_bone_rest(1, Transform3D());
	skeleton->set_bone_parent(1, 0);
	Dictionary child_bone_extras;
	child_bone_extras["bone"] = "i_am_child_bone";
	skeleton->set_bone_meta(1, "extras", child_bone_extras);

	// We have to have a mesh to link with skeleton or it will not get imported.
	Ref<PlaneMesh> meshdata = memnew(PlaneMesh);
	meshdata->set_name("planemesh");

	MeshInstance3D *mesh = memnew(MeshInstance3D);
	mesh->set_mesh(meshdata);
	mesh->set_name("mesh_instance_3d");

	Node3D *scene = memnew(Node3D);
	SceneTree::get_singleton()->get_root()->add_child(scene);
	scene->add_child(skeleton);
	scene->add_child(mesh);
	scene->set_name("node3d");

	// Now that both skeleton and mesh are part of scene, link them.
	mesh->set_skeleton_path(mesh->get_path_to(skeleton));

	// Convert to GLFT and back.
	String tempfile = OS::get_singleton()->get_cache_path().path_join("gltf_bone_extras");
	Node *loaded = _gltf_export_then_import(scene, tempfile);

	// Compare the results.
	CHECK(loaded->get_name() == "node3d");
	Skeleton3D *result = Object::cast_to<Skeleton3D>(loaded->find_child("Skeleton3D", false, true));
	CHECK(result->get_bone_name(0) == "parent");
	CHECK(Dictionary(result->get_bone_meta(0, "extras"))["bone"] == "i_am_parent_bone");
	CHECK(result->get_bone_name(1) == "child");
	CHECK(Dictionary(result->get_bone_meta(1, "extras"))["bone"] == "i_am_child_bone");

	memdelete(skeleton);
	memdelete(mesh);
	memdelete(scene);
	memdelete(loaded);
}
} // namespace TestGltfExtras

#endif // TOOLS_ENABLED

#endif // TEST_GLTF_EXTRAS_H
