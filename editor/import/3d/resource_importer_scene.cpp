/**************************************************************************/
/*  resource_importer_scene.cpp                                           */
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

#include "resource_importer_scene.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/io/resource_saver.h"
#include "core/object/script_language.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/import/3d/scene_import_settings.h"
#include "editor/renames_map_3_to_4.h"
#include "scene/3d/area_3d.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/navigation_region_3d.h"
#include "scene/3d/occluder_instance_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/3d/vehicle_body_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/box_shape_3d.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/resource_format_text.h"
#include "scene/resources/separation_ray_shape_3d.h"
#include "scene/resources/sphere_shape_3d.h"
#include "scene/resources/surface_tool.h"
#include "scene/resources/world_boundary_shape_3d.h"
#include "servers/rendering/shader_language.h"
#ifdef MODULE_REGEX_ENABLED
#include "modules/regex/regex.h"
#endif

uint32_t EditorSceneFormatImporter::get_import_flags() const {
	uint32_t ret;
	if (GDVIRTUAL_CALL(_get_import_flags, ret)) {
		return ret;
	}

	ERR_FAIL_V(0);
}

void EditorSceneFormatImporter::get_extensions(List<String> *r_extensions) const {
	Vector<String> arr;
	if (GDVIRTUAL_CALL(_get_extensions, arr)) {
		for (int i = 0; i < arr.size(); i++) {
			r_extensions->push_back(arr[i]);
		}
		return;
	}

	ERR_FAIL();
}

Node *EditorSceneFormatImporter::import_scene(const String &p_path, uint32_t p_flags, const HashMap<StringName, Variant> &p_options, List<String> *r_missing_deps, Error *r_err) {
	Dictionary options_dict;
	for (const KeyValue<StringName, Variant> &elem : p_options) {
		options_dict[elem.key] = elem.value;
	}
	Object *ret = nullptr;
	if (GDVIRTUAL_CALL(_import_scene, p_path, p_flags, options_dict, ret)) {
		return Object::cast_to<Node>(ret);
	}

	ERR_FAIL_V(nullptr);
}

void EditorSceneFormatImporter::get_import_options(const String &p_path, List<ResourceImporter::ImportOption> *r_options) {
	GDVIRTUAL_CALL(_get_import_options, p_path);
}

Variant EditorSceneFormatImporter::get_option_visibility(const String &p_path, bool p_for_animation, const String &p_option, const HashMap<StringName, Variant> &p_options) {
	Variant ret;
	GDVIRTUAL_CALL(_get_option_visibility, p_path, p_for_animation, p_option, ret);
	return ret;
}

void EditorSceneFormatImporter::_bind_methods() {
	GDVIRTUAL_BIND(_get_import_flags);
	GDVIRTUAL_BIND(_get_extensions);
	GDVIRTUAL_BIND(_import_scene, "path", "flags", "options");
	GDVIRTUAL_BIND(_get_import_options, "path");
	GDVIRTUAL_BIND(_get_option_visibility, "path", "for_animation", "option");

	BIND_CONSTANT(IMPORT_SCENE);
	BIND_CONSTANT(IMPORT_ANIMATION);
	BIND_CONSTANT(IMPORT_FAIL_ON_MISSING_DEPENDENCIES);
	BIND_CONSTANT(IMPORT_GENERATE_TANGENT_ARRAYS);
	BIND_CONSTANT(IMPORT_USE_NAMED_SKIN_BINDS);
	BIND_CONSTANT(IMPORT_DISCARD_MESHES_AND_MATERIALS);
	BIND_CONSTANT(IMPORT_FORCE_DISABLE_MESH_COMPRESSION);
}

/////////////////////////////////
void EditorScenePostImport::_bind_methods() {
	GDVIRTUAL_BIND(_post_import, "scene")
	ClassDB::bind_method(D_METHOD("get_source_file"), &EditorScenePostImport::get_source_file);
}

Node *EditorScenePostImport::post_import(Node *p_scene) {
	Object *ret;
	if (GDVIRTUAL_CALL(_post_import, p_scene, ret)) {
		return Object::cast_to<Node>(ret);
	}

	return p_scene;
}

String EditorScenePostImport::get_source_file() const {
	return source_file;
}

void EditorScenePostImport::init(const String &p_source_file) {
	source_file = p_source_file;
}

EditorScenePostImport::EditorScenePostImport() {
}

///////////////////////////////////////////////////////

Variant EditorScenePostImportPlugin::get_option_value(const StringName &p_name) const {
	ERR_FAIL_COND_V_MSG(current_options == nullptr && current_options_dict == nullptr, Variant(), "get_option_value called from a function where option values are not available.");
	ERR_FAIL_COND_V_MSG(current_options && !current_options->has(p_name), Variant(), "get_option_value called with unexisting option argument: " + String(p_name));
	ERR_FAIL_COND_V_MSG(current_options_dict && !current_options_dict->has(p_name), Variant(), "get_option_value called with unexisting option argument: " + String(p_name));
	if (current_options && current_options->has(p_name)) {
		return (*current_options)[p_name];
	}
	if (current_options_dict && current_options_dict->has(p_name)) {
		return (*current_options_dict)[p_name];
	}
	return Variant();
}
void EditorScenePostImportPlugin::add_import_option(const String &p_name, Variant p_default_value) {
	ERR_FAIL_NULL_MSG(current_option_list, "add_import_option() can only be called from get_import_options().");
	add_import_option_advanced(p_default_value.get_type(), p_name, p_default_value);
}
void EditorScenePostImportPlugin::add_import_option_advanced(Variant::Type p_type, const String &p_name, Variant p_default_value, PropertyHint p_hint, const String &p_hint_string, int p_usage_flags) {
	ERR_FAIL_NULL_MSG(current_option_list, "add_import_option_advanced() can only be called from get_import_options().");
	current_option_list->push_back(ResourceImporter::ImportOption(PropertyInfo(p_type, p_name, p_hint, p_hint_string, p_usage_flags), p_default_value));
}

void EditorScenePostImportPlugin::get_internal_import_options(InternalImportCategory p_category, List<ResourceImporter::ImportOption> *r_options) {
	current_option_list = r_options;
	GDVIRTUAL_CALL(_get_internal_import_options, p_category);
	current_option_list = nullptr;
}
Variant EditorScenePostImportPlugin::get_internal_option_visibility(InternalImportCategory p_category, bool p_for_animation, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	current_options = &p_options;
	Variant ret;
	GDVIRTUAL_CALL(_get_internal_option_visibility, p_category, p_for_animation, p_option, ret);
	current_options = nullptr;
	return ret;
}
Variant EditorScenePostImportPlugin::get_internal_option_update_view_required(InternalImportCategory p_category, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	current_options = &p_options;
	Variant ret;
	GDVIRTUAL_CALL(_get_internal_option_update_view_required, p_category, p_option, ret);
	current_options = nullptr;
	return ret;
}

void EditorScenePostImportPlugin::internal_process(InternalImportCategory p_category, Node *p_base_scene, Node *p_node, Ref<Resource> p_resource, const Dictionary &p_options) {
	current_options_dict = &p_options;
	GDVIRTUAL_CALL(_internal_process, p_category, p_base_scene, p_node, p_resource);
	current_options_dict = nullptr;
}

void EditorScenePostImportPlugin::get_import_options(const String &p_path, List<ResourceImporter::ImportOption> *r_options) {
	current_option_list = r_options;
	GDVIRTUAL_CALL(_get_import_options, p_path);
	current_option_list = nullptr;
}
Variant EditorScenePostImportPlugin::get_option_visibility(const String &p_path, bool p_for_animation, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	current_options = &p_options;
	Variant ret;
	GDVIRTUAL_CALL(_get_option_visibility, p_path, p_for_animation, p_option, ret);
	current_options = nullptr;
	return ret;
}

void EditorScenePostImportPlugin::pre_process(Node *p_scene, const HashMap<StringName, Variant> &p_options) {
	current_options = &p_options;
	GDVIRTUAL_CALL(_pre_process, p_scene);
	current_options = nullptr;
}
void EditorScenePostImportPlugin::post_process(Node *p_scene, const HashMap<StringName, Variant> &p_options) {
	current_options = &p_options;
	GDVIRTUAL_CALL(_post_process, p_scene);
	current_options = nullptr;
}

void EditorScenePostImportPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_option_value", "name"), &EditorScenePostImportPlugin::get_option_value);

	ClassDB::bind_method(D_METHOD("add_import_option", "name", "value"), &EditorScenePostImportPlugin::add_import_option);
	ClassDB::bind_method(D_METHOD("add_import_option_advanced", "type", "name", "default_value", "hint", "hint_string", "usage_flags"), &EditorScenePostImportPlugin::add_import_option_advanced, DEFVAL(PROPERTY_HINT_NONE), DEFVAL(""), DEFVAL(PROPERTY_USAGE_DEFAULT));

	GDVIRTUAL_BIND(_get_internal_import_options, "category");
	GDVIRTUAL_BIND(_get_internal_option_visibility, "category", "for_animation", "option");
	GDVIRTUAL_BIND(_get_internal_option_update_view_required, "category", "option");
	GDVIRTUAL_BIND(_internal_process, "category", "base_node", "node", "resource");
	GDVIRTUAL_BIND(_get_import_options, "path");
	GDVIRTUAL_BIND(_get_option_visibility, "path", "for_animation", "option");
	GDVIRTUAL_BIND(_pre_process, "scene");
	GDVIRTUAL_BIND(_post_process, "scene");

	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_NODE);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_MESH);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_MATERIAL);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_ANIMATION);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_MAX);
}

/////////////////////////////////////////////////////////

String ResourceImporterScene::get_importer_name() const {
	return animation_importer ? "animation_library" : "scene";
}

String ResourceImporterScene::get_visible_name() const {
	return animation_importer ? "Animation Library" : "Scene";
}

void ResourceImporterScene::get_recognized_extensions(List<String> *p_extensions) const {
	get_scene_importer_extensions(p_extensions);
}

String ResourceImporterScene::get_save_extension() const {
	return animation_importer ? "res" : "scn";
}

String ResourceImporterScene::get_resource_type() const {
	return animation_importer ? "AnimationLibrary" : "PackedScene";
}

int ResourceImporterScene::get_format_version() const {
	return 1;
}

bool ResourceImporterScene::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	if (animation_importer) {
		if (p_option == "animation/import") { // Option ignored, animation always imported.
			return false;
		}
	} else if (p_option.begins_with("animation/")) {
		if (p_option != "animation/import" && !bool(p_options["animation/import"])) {
			return false;
		}
	}

	if (animation_importer && (p_option.begins_with("nodes/") || p_option.begins_with("meshes/") || p_option.begins_with("skins/"))) {
		return false; // Nothing to do here for animations.
	}

	if (p_option == "meshes/lightmap_texel_size" && int(p_options["meshes/light_baking"]) != 2) {
		// Only display the lightmap texel size import option when using the Static Lightmaps light baking mode.
		return false;
	}

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		Variant ret = post_importer_plugins.write[i]->get_option_visibility(p_path, animation_importer, p_option, p_options);
		if (ret.get_type() == Variant::BOOL) {
			return ret;
		}
	}

	for (Ref<EditorSceneFormatImporter> importer : scene_importers) {
		Variant ret = importer->get_option_visibility(p_path, animation_importer, p_option, p_options);
		if (ret.get_type() == Variant::BOOL) {
			return ret;
		}
	}

	return true;
}

int ResourceImporterScene::get_preset_count() const {
	return 0;
}

String ResourceImporterScene::get_preset_name(int p_idx) const {
	return String();
}

static bool _teststr(const String &p_what, const String &p_str) {
	String what = p_what;

	// Remove trailing spaces and numbers, some apps like blender add ".number" to duplicates
	// (dot is replaced with _ as invalid character) so also compensate for this.
	while (what.length() && (is_digit(what[what.length() - 1]) || what[what.length() - 1] <= 32 || what[what.length() - 1] == '_')) {
		what = what.substr(0, what.length() - 1);
	}

	if (what.findn("$" + p_str) != -1) { //blender and other stuff
		return true;
	}
	if (what.to_lower().ends_with("-" + p_str)) { //collada only supports "_" and "-" besides letters
		return true;
	}
	if (what.to_lower().ends_with("_" + p_str)) { //collada only supports "_" and "-" besides letters
		return true;
	}
	return false;
}

static String _fixstr(const String &p_what, const String &p_str) {
	String what = p_what;

	// Remove trailing spaces and numbers, some apps like blender add ".number" to duplicates
	// (dot is replaced with _ as invalid character) so also compensate for this.
	while (what.length() && (is_digit(what[what.length() - 1]) || what[what.length() - 1] <= 32 || what[what.length() - 1] == '_')) {
		what = what.substr(0, what.length() - 1);
	}

	String end = p_what.substr(what.length(), p_what.length() - what.length());

	if (what.findn("$" + p_str) != -1) { //blender and other stuff
		return what.replace("$" + p_str, "") + end;
	}
	if (what.to_lower().ends_with("-" + p_str)) { //collada only supports "_" and "-" besides letters
		return what.substr(0, what.length() - (p_str.length() + 1)) + end;
	}
	if (what.to_lower().ends_with("_" + p_str)) { //collada only supports "_" and "-" besides letters
		return what.substr(0, what.length() - (p_str.length() + 1)) + end;
	}
	return what;
}

static void _pre_gen_shape_list(Ref<ImporterMesh> &mesh, Vector<Ref<Shape3D>> &r_shape_list, bool p_convex) {
	ERR_FAIL_NULL_MSG(mesh, "Cannot generate shape list with null mesh value.");
	if (!p_convex) {
		Ref<ConcavePolygonShape3D> shape = mesh->create_trimesh_shape();
		r_shape_list.push_back(shape);
	} else {
		Vector<Ref<Shape3D>> cd;
		cd.push_back(mesh->create_convex_shape(true, /*Passing false, otherwise VHACD will be used to simplify (Decompose) the Mesh.*/ false));
		if (cd.size()) {
			for (int i = 0; i < cd.size(); i++) {
				r_shape_list.push_back(cd[i]);
			}
		}
	}
}

struct ScalableNodeCollection {
	HashSet<Node3D *> node_3ds;
	HashSet<Ref<ImporterMesh>> importer_meshes;
	HashSet<Ref<Skin>> skins;
	HashSet<Ref<Animation>> animations;
};

void _rescale_importer_mesh(Vector3 p_scale, Ref<ImporterMesh> p_mesh, bool is_shadow = false) {
	// MESH and SKIN data divide, to compensate for object position multiplying.

	const int surf_count = p_mesh->get_surface_count();
	const int blendshape_count = p_mesh->get_blend_shape_count();
	struct LocalSurfData {
		Mesh::PrimitiveType prim = {};
		Array arr;
		Array bsarr;
		Dictionary lods;
		String name;
		Ref<Material> mat;
		uint64_t fmt_compress_flags = 0;
	};

	Vector<LocalSurfData> surf_data_by_mesh;

	Vector<String> blendshape_names;
	for (int bsidx = 0; bsidx < blendshape_count; bsidx++) {
		blendshape_names.append(p_mesh->get_blend_shape_name(bsidx));
	}

	for (int surf_idx = 0; surf_idx < surf_count; surf_idx++) {
		Mesh::PrimitiveType prim = p_mesh->get_surface_primitive_type(surf_idx);
		const uint64_t fmt_compress_flags = p_mesh->get_surface_format(surf_idx);
		Array arr = p_mesh->get_surface_arrays(surf_idx);
		String name = p_mesh->get_surface_name(surf_idx);
		Dictionary lods;
		Ref<Material> mat = p_mesh->get_surface_material(surf_idx);
		{
			Vector<Vector3> vertex_array = arr[ArrayMesh::ARRAY_VERTEX];
			for (int vert_arr_i = 0; vert_arr_i < vertex_array.size(); vert_arr_i++) {
				vertex_array.write[vert_arr_i] = vertex_array[vert_arr_i] * p_scale;
			}
			arr[ArrayMesh::ARRAY_VERTEX] = vertex_array;
		}
		Array blendshapes;
		for (int bsidx = 0; bsidx < blendshape_count; bsidx++) {
			Array current_bsarr = p_mesh->get_surface_blend_shape_arrays(surf_idx, bsidx);
			Vector<Vector3> current_bs_vertex_array = current_bsarr[ArrayMesh::ARRAY_VERTEX];
			int current_bs_vert_arr_len = current_bs_vertex_array.size();
			for (int32_t bs_vert_arr_i = 0; bs_vert_arr_i < current_bs_vert_arr_len; bs_vert_arr_i++) {
				current_bs_vertex_array.write[bs_vert_arr_i] = current_bs_vertex_array[bs_vert_arr_i] * p_scale;
			}
			current_bsarr[ArrayMesh::ARRAY_VERTEX] = current_bs_vertex_array;
			blendshapes.push_back(current_bsarr);
		}

		LocalSurfData surf_data_dictionary = LocalSurfData();
		surf_data_dictionary.prim = prim;
		surf_data_dictionary.arr = arr;
		surf_data_dictionary.bsarr = blendshapes;
		surf_data_dictionary.lods = lods;
		surf_data_dictionary.fmt_compress_flags = fmt_compress_flags;
		surf_data_dictionary.name = name;
		surf_data_dictionary.mat = mat;

		surf_data_by_mesh.push_back(surf_data_dictionary);
	}

	p_mesh->clear();

	for (int bsidx = 0; bsidx < blendshape_count; bsidx++) {
		p_mesh->add_blend_shape(blendshape_names[bsidx]);
	}

	for (int surf_idx = 0; surf_idx < surf_count; surf_idx++) {
		const Mesh::PrimitiveType prim = surf_data_by_mesh[surf_idx].prim;
		const Array arr = surf_data_by_mesh[surf_idx].arr;
		const Array bsarr = surf_data_by_mesh[surf_idx].bsarr;
		const Dictionary lods = surf_data_by_mesh[surf_idx].lods;
		const uint64_t fmt_compress_flags = surf_data_by_mesh[surf_idx].fmt_compress_flags;
		const String name = surf_data_by_mesh[surf_idx].name;
		const Ref<Material> mat = surf_data_by_mesh[surf_idx].mat;

		p_mesh->add_surface(prim, arr, bsarr, lods, mat, name, fmt_compress_flags);
	}

	if (!is_shadow && p_mesh->get_shadow_mesh() != p_mesh && p_mesh->get_shadow_mesh().is_valid()) {
		_rescale_importer_mesh(p_scale, p_mesh->get_shadow_mesh(), true);
	}
}

void _rescale_skin(Vector3 p_scale, Ref<Skin> p_skin) {
	// MESH and SKIN data divide, to compensate for object position multiplying.
	for (int i = 0; i < p_skin->get_bind_count(); i++) {
		Transform3D transform = p_skin->get_bind_pose(i);
		p_skin->set_bind_pose(i, Transform3D(transform.basis, p_scale * transform.origin));
	}
}

void _rescale_animation(Vector3 p_scale, Ref<Animation> p_animation) {
	for (int track_idx = 0; track_idx < p_animation->get_track_count(); track_idx++) {
		if (p_animation->track_get_type(track_idx) == Animation::TYPE_POSITION_3D) {
			for (int key_idx = 0; key_idx < p_animation->track_get_key_count(track_idx); key_idx++) {
				Vector3 value = p_animation->track_get_key_value(track_idx, key_idx);
				value = p_scale * value;
				p_animation->track_set_key_value(track_idx, key_idx, value);
			}
		}
	}
}

void _apply_scale_to_scalable_node_collection(ScalableNodeCollection &p_collection, Vector3 p_scale) {
	for (Node3D *node_3d : p_collection.node_3ds) {
		node_3d->set_position(p_scale * node_3d->get_position());
		Skeleton3D *skeleton_3d = Object::cast_to<Skeleton3D>(node_3d);
		if (skeleton_3d) {
			for (int i = 0; i < skeleton_3d->get_bone_count(); i++) {
				Transform3D rest = skeleton_3d->get_bone_rest(i);
				skeleton_3d->set_bone_rest(i, Transform3D(rest.basis, p_scale * rest.origin));
				skeleton_3d->set_bone_pose_position(i, p_scale * rest.origin);
			}
		}
	}
	for (Ref<ImporterMesh> mesh : p_collection.importer_meshes) {
		_rescale_importer_mesh(p_scale, mesh, false);
	}
	for (Ref<Skin> skin : p_collection.skins) {
		_rescale_skin(p_scale, skin);
	}
	for (Ref<Animation> animation : p_collection.animations) {
		_rescale_animation(p_scale, animation);
	}
}

void _populate_scalable_nodes_collection(Node *p_node, ScalableNodeCollection &p_collection) {
	if (!p_node) {
		return;
	}
	Node3D *node_3d = Object::cast_to<Node3D>(p_node);
	if (node_3d) {
		p_collection.node_3ds.insert(node_3d);
		ImporterMeshInstance3D *mesh_instance_3d = Object::cast_to<ImporterMeshInstance3D>(p_node);
		if (mesh_instance_3d) {
			Ref<ImporterMesh> mesh = mesh_instance_3d->get_mesh();
			if (mesh.is_valid()) {
				p_collection.importer_meshes.insert(mesh);
			}
			Ref<Skin> skin = mesh_instance_3d->get_skin();
			if (skin.is_valid()) {
				p_collection.skins.insert(skin);
			}
		}
	}
	AnimationPlayer *animation_player = Object::cast_to<AnimationPlayer>(p_node);
	if (animation_player) {
		List<StringName> animation_list;
		animation_player->get_animation_list(&animation_list);

		for (const StringName &E : animation_list) {
			Ref<Animation> animation = animation_player->get_animation(E);
			p_collection.animations.insert(animation);
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		_populate_scalable_nodes_collection(child, p_collection);
	}
}

void _apply_permanent_scale_to_descendants(Node *p_root_node, Vector3 p_scale) {
	ScalableNodeCollection scalable_node_collection;
	_populate_scalable_nodes_collection(p_root_node, scalable_node_collection);
	_apply_scale_to_scalable_node_collection(scalable_node_collection, p_scale);
}

Node *ResourceImporterScene::_pre_fix_node(Node *p_node, Node *p_root, HashMap<Ref<ImporterMesh>, Vector<Ref<Shape3D>>> &r_collision_map, Pair<PackedVector3Array, PackedInt32Array> *r_occluder_arrays, List<Pair<NodePath, Node *>> &r_node_renames) {
	// Children first.
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *r = _pre_fix_node(p_node->get_child(i), p_root, r_collision_map, r_occluder_arrays, r_node_renames);
		if (!r) {
			i--; // Was erased.
		}
	}

	String name = p_node->get_name();
	NodePath original_path = p_root->get_path_to(p_node); // Used to detect renames due to import hints.

	bool isroot = p_node == p_root;

	if (!isroot && _teststr(name, "noimp")) {
		p_node->set_owner(nullptr);
		memdelete(p_node);
		return nullptr;
	}

	if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);

		Ref<ImporterMesh> m = mi->get_mesh();

		if (m.is_valid()) {
			for (int i = 0; i < m->get_surface_count(); i++) {
				Ref<BaseMaterial3D> mat = m->get_surface_material(i);
				if (!mat.is_valid()) {
					continue;
				}

				if (_teststr(mat->get_name(), "alpha")) {
					mat->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
					mat->set_name(_fixstr(mat->get_name(), "alpha"));
				}
				if (_teststr(mat->get_name(), "vcol")) {
					mat->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
					mat->set_flag(BaseMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
					mat->set_name(_fixstr(mat->get_name(), "vcol"));
				}
			}
		}
	}

	if (Object::cast_to<AnimationPlayer>(p_node)) {
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(p_node);

		// Node paths in animation tracks are relative to the following path (this is used to fix node paths below).
		Node *ap_root = ap->get_node(ap->get_root_node());
		NodePath path_prefix = p_root->get_path_to(ap_root);

		bool nodes_were_renamed = r_node_renames.size() != 0;

		List<StringName> anims;
		ap->get_animation_list(&anims);
		for (const StringName &E : anims) {
			Ref<Animation> anim = ap->get_animation(E);
			ERR_CONTINUE(anim.is_null());

			// Remove animation tracks referencing non-importable nodes.
			for (int i = 0; i < anim->get_track_count(); i++) {
				NodePath path = anim->track_get_path(i);

				for (int j = 0; j < path.get_name_count(); j++) {
					String node = path.get_name(j);
					if (_teststr(node, "noimp")) {
						anim->remove_track(i);
						i--;
						break;
					}
				}
			}

			// Fix node paths in animations, in case nodes were renamed earlier due to import hints.
			if (nodes_were_renamed) {
				for (int i = 0; i < anim->get_track_count(); i++) {
					NodePath path = anim->track_get_path(i);
					// Convert track path to absolute node path without subnames (some manual work because we are not in the scene tree).
					Vector<StringName> absolute_path_names = path_prefix.get_names();
					absolute_path_names.append_array(path.get_names());
					NodePath absolute_path(absolute_path_names, false);
					absolute_path.simplify();
					// Fix paths to renamed nodes.
					for (const Pair<NodePath, Node *> &F : r_node_renames) {
						if (F.first == absolute_path) {
							NodePath new_path(ap_root->get_path_to(F.second).get_names(), path.get_subnames(), false);
							print_verbose(vformat("Fix: Correcting node path in animation track: %s should be %s", path, new_path));
							anim->track_set_path(i, new_path);
							break; // Only one match is possible.
						}
					}
				}
			}

			String animname = E;
			const int loop_string_count = 3;
			static const char *loop_strings[loop_string_count] = { "loop_mode", "loop", "cycle" };
			for (int i = 0; i < loop_string_count; i++) {
				if (_teststr(animname, loop_strings[i])) {
					anim->set_loop_mode(Animation::LOOP_LINEAR);
					animname = _fixstr(animname, loop_strings[i]);

					Ref<AnimationLibrary> library = ap->get_animation_library(ap->find_animation_library(anim));
					library->rename_animation(E, animname);
				}
			}
		}
	}

	if (_teststr(name, "colonly") || _teststr(name, "convcolonly")) {
		if (isroot) {
			return p_node;
		}

		String fixed_name;
		if (_teststr(name, "colonly")) {
			fixed_name = _fixstr(name, "colonly");
		} else if (_teststr(name, "convcolonly")) {
			fixed_name = _fixstr(name, "convcolonly");
		}

		if (fixed_name.is_empty()) {
			p_node->set_owner(nullptr);
			memdelete(p_node);
			ERR_FAIL_V_MSG(nullptr, vformat("Skipped node `%s` because its name is empty after removing the suffix.", name));
		}

		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);
		if (mi) {
			Ref<ImporterMesh> mesh = mi->get_mesh();

			if (mesh.is_valid()) {
				Vector<Ref<Shape3D>> shapes;
				if (r_collision_map.has(mesh)) {
					shapes = r_collision_map[mesh];
				} else if (_teststr(name, "colonly")) {
					_pre_gen_shape_list(mesh, shapes, false);
					r_collision_map[mesh] = shapes;
				} else if (_teststr(name, "convcolonly")) {
					_pre_gen_shape_list(mesh, shapes, true);
					r_collision_map[mesh] = shapes;
				}

				if (shapes.size()) {
					StaticBody3D *col = memnew(StaticBody3D);
					col->set_transform(mi->get_transform());
					col->set_name(fixed_name);
					p_node->replace_by(col);
					p_node->set_owner(nullptr);
					memdelete(p_node);
					p_node = col;

					_add_shapes(col, shapes);
				}
			}

		} else if (p_node->has_meta("empty_draw_type")) {
			String empty_draw_type = String(p_node->get_meta("empty_draw_type"));
			StaticBody3D *sb = memnew(StaticBody3D);
			sb->set_name(fixed_name);
			Object::cast_to<Node3D>(sb)->set_transform(Object::cast_to<Node3D>(p_node)->get_transform());
			p_node->replace_by(sb);
			p_node->set_owner(nullptr);
			memdelete(p_node);
			p_node = sb;
			CollisionShape3D *colshape = memnew(CollisionShape3D);
			if (empty_draw_type == "CUBE") {
				BoxShape3D *boxShape = memnew(BoxShape3D);
				boxShape->set_size(Vector3(2, 2, 2));
				colshape->set_shape(boxShape);
			} else if (empty_draw_type == "SINGLE_ARROW") {
				SeparationRayShape3D *rayShape = memnew(SeparationRayShape3D);
				rayShape->set_length(1);
				colshape->set_shape(rayShape);
				Object::cast_to<Node3D>(sb)->rotate_x(Math_PI / 2);
			} else if (empty_draw_type == "IMAGE") {
				WorldBoundaryShape3D *world_boundary_shape = memnew(WorldBoundaryShape3D);
				colshape->set_shape(world_boundary_shape);
			} else {
				SphereShape3D *sphereShape = memnew(SphereShape3D);
				sphereShape->set_radius(1);
				colshape->set_shape(sphereShape);
			}
			sb->add_child(colshape, true);
			colshape->set_owner(sb->get_owner());
		}

	} else if (_teststr(name, "rigid") && Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		if (isroot) {
			return p_node;
		}

		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);
		Ref<ImporterMesh> mesh = mi->get_mesh();

		if (mesh.is_valid()) {
			Vector<Ref<Shape3D>> shapes;
			if (r_collision_map.has(mesh)) {
				shapes = r_collision_map[mesh];
			} else {
				_pre_gen_shape_list(mesh, shapes, true);
			}

			RigidBody3D *rigid_body = memnew(RigidBody3D);
			rigid_body->set_name(_fixstr(name, "rigid_body"));
			p_node->replace_by(rigid_body);
			rigid_body->set_transform(mi->get_transform());
			p_node = rigid_body;
			mi->set_transform(Transform3D());
			rigid_body->add_child(mi, true);
			mi->set_owner(rigid_body->get_owner());

			_add_shapes(rigid_body, shapes);
		}

	} else if ((_teststr(name, "col") || (_teststr(name, "convcol"))) && Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);

		Ref<ImporterMesh> mesh = mi->get_mesh();

		if (mesh.is_valid()) {
			Vector<Ref<Shape3D>> shapes;
			String fixed_name;
			if (r_collision_map.has(mesh)) {
				shapes = r_collision_map[mesh];
			} else if (_teststr(name, "col")) {
				_pre_gen_shape_list(mesh, shapes, false);
				r_collision_map[mesh] = shapes;
			} else if (_teststr(name, "convcol")) {
				_pre_gen_shape_list(mesh, shapes, true);
				r_collision_map[mesh] = shapes;
			}

			if (_teststr(name, "col")) {
				fixed_name = _fixstr(name, "col");
			} else if (_teststr(name, "convcol")) {
				fixed_name = _fixstr(name, "convcol");
			}

			if (!fixed_name.is_empty()) {
				if (mi->get_parent() && !mi->get_parent()->has_node(fixed_name)) {
					mi->set_name(fixed_name);
				}
			}

			if (shapes.size()) {
				StaticBody3D *col = memnew(StaticBody3D);
				mi->add_child(col, true);
				col->set_owner(mi->get_owner());

				_add_shapes(col, shapes);
			}
		}

	} else if (_teststr(name, "navmesh") && Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		if (isroot) {
			return p_node;
		}

		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);

		Ref<ImporterMesh> mesh = mi->get_mesh();
		ERR_FAIL_COND_V(mesh.is_null(), nullptr);
		NavigationRegion3D *nmi = memnew(NavigationRegion3D);

		nmi->set_name(_fixstr(name, "navmesh"));
		Ref<NavigationMesh> nmesh = mesh->create_navigation_mesh();
		nmi->set_navigation_mesh(nmesh);
		Object::cast_to<Node3D>(nmi)->set_transform(mi->get_transform());
		p_node->replace_by(nmi);
		p_node->set_owner(nullptr);
		memdelete(p_node);
		p_node = nmi;
	} else if (_teststr(name, "occ") || _teststr(name, "occonly")) {
		if (isroot) {
			return p_node;
		}
		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);
		if (mi) {
			Ref<ImporterMesh> mesh = mi->get_mesh();

			if (mesh.is_valid()) {
				if (r_occluder_arrays) {
					OccluderInstance3D::bake_single_node(mi, 0.0f, r_occluder_arrays->first, r_occluder_arrays->second);
				}
				if (_teststr(name, "occ")) {
					String fixed_name = _fixstr(name, "occ");
					if (!fixed_name.is_empty()) {
						if (mi->get_parent() && !mi->get_parent()->has_node(fixed_name)) {
							mi->set_name(fixed_name);
						}
					}
				} else {
					p_node->set_owner(nullptr);
					memdelete(p_node);
					p_node = nullptr;
				}
			}
		}
	} else if (_teststr(name, "vehicle")) {
		if (isroot) {
			return p_node;
		}

		Node *owner = p_node->get_owner();
		Node3D *s = Object::cast_to<Node3D>(p_node);
		VehicleBody3D *bv = memnew(VehicleBody3D);
		String n = _fixstr(p_node->get_name(), "vehicle");
		bv->set_name(n);
		p_node->replace_by(bv);
		p_node->set_name(n);
		bv->add_child(p_node);
		bv->set_owner(owner);
		p_node->set_owner(owner);
		bv->set_transform(s->get_transform());
		s->set_transform(Transform3D());

		p_node = bv;
	} else if (_teststr(name, "wheel")) {
		if (isroot) {
			return p_node;
		}

		Node *owner = p_node->get_owner();
		Node3D *s = Object::cast_to<Node3D>(p_node);
		VehicleWheel3D *bv = memnew(VehicleWheel3D);
		String n = _fixstr(p_node->get_name(), "wheel");
		bv->set_name(n);
		p_node->replace_by(bv);
		p_node->set_name(n);
		bv->add_child(p_node);
		bv->set_owner(owner);
		p_node->set_owner(owner);
		bv->set_transform(s->get_transform());
		s->set_transform(Transform3D());

		p_node = bv;
	} else if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		//last attempt, maybe collision inside the mesh data

		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);

		Ref<ImporterMesh> mesh = mi->get_mesh();
		if (!mesh.is_null()) {
			Vector<Ref<Shape3D>> shapes;
			if (r_collision_map.has(mesh)) {
				shapes = r_collision_map[mesh];
			} else if (_teststr(mesh->get_name(), "col")) {
				_pre_gen_shape_list(mesh, shapes, false);
				r_collision_map[mesh] = shapes;
				mesh->set_name(_fixstr(mesh->get_name(), "col"));
			} else if (_teststr(mesh->get_name(), "convcol")) {
				_pre_gen_shape_list(mesh, shapes, true);
				r_collision_map[mesh] = shapes;
				mesh->set_name(_fixstr(mesh->get_name(), "convcol"));
			} else if (_teststr(mesh->get_name(), "occ")) {
				if (r_occluder_arrays) {
					OccluderInstance3D::bake_single_node(mi, 0.0f, r_occluder_arrays->first, r_occluder_arrays->second);
				}
				mesh->set_name(_fixstr(mesh->get_name(), "occ"));
			}

			if (shapes.size()) {
				StaticBody3D *col = memnew(StaticBody3D);
				p_node->add_child(col, true);
				col->set_owner(p_node->get_owner());

				_add_shapes(col, shapes);
			}
		}
	}

	if (p_node) {
		NodePath new_path = p_root->get_path_to(p_node);
		if (new_path != original_path) {
			print_verbose(vformat("Fix: Renamed %s to %s", original_path, new_path));
			r_node_renames.push_back({ original_path, p_node });
		}
	}

	return p_node;
}

Node *ResourceImporterScene::_pre_fix_animations(Node *p_node, Node *p_root, const Dictionary &p_node_data, const Dictionary &p_animation_data, float p_animation_fps) {
	// children first
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *r = _pre_fix_animations(p_node->get_child(i), p_root, p_node_data, p_animation_data, p_animation_fps);
		if (!r) {
			i--; //was erased
		}
	}

	String import_id = p_node->get_meta("import_id", "PATH:" + p_root->get_path_to(p_node));

	Dictionary node_settings;
	if (p_node_data.has(import_id)) {
		node_settings = p_node_data[import_id];
	}

	{
		//make sure this is unique
		node_settings = node_settings.duplicate(true);
		//fill node settings for this node with default values
		List<ImportOption> iopts;
		get_internal_import_options(INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE, &iopts);
		for (const ImportOption &E : iopts) {
			if (!node_settings.has(E.option.name)) {
				node_settings[E.option.name] = E.default_value;
			}
		}
	}

	if (Object::cast_to<AnimationPlayer>(p_node)) {
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(p_node);
		List<StringName> anims;
		ap->get_animation_list(&anims);

		AnimationImportTracks import_tracks_mode[TRACK_CHANNEL_MAX] = {
			AnimationImportTracks(int(node_settings["import_tracks/position"])),
			AnimationImportTracks(int(node_settings["import_tracks/rotation"])),
			AnimationImportTracks(int(node_settings["import_tracks/scale"]))
		};

		if (!anims.is_empty() && (import_tracks_mode[0] != ANIMATION_IMPORT_TRACKS_IF_PRESENT || import_tracks_mode[1] != ANIMATION_IMPORT_TRACKS_IF_PRESENT || import_tracks_mode[2] != ANIMATION_IMPORT_TRACKS_IF_PRESENT)) {
			_optimize_track_usage(ap, import_tracks_mode);
		}
	}

	return p_node;
}

Node *ResourceImporterScene::_post_fix_animations(Node *p_node, Node *p_root, const Dictionary &p_node_data, const Dictionary &p_animation_data, float p_animation_fps) {
	// children first
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *r = _post_fix_animations(p_node->get_child(i), p_root, p_node_data, p_animation_data, p_animation_fps);
		if (!r) {
			i--; //was erased
		}
	}

	String import_id = p_node->get_meta("import_id", "PATH:" + p_root->get_path_to(p_node));

	Dictionary node_settings;
	if (p_node_data.has(import_id)) {
		node_settings = p_node_data[import_id];
	}

	{
		//make sure this is unique
		node_settings = node_settings.duplicate(true);
		//fill node settings for this node with default values
		List<ImportOption> iopts;
		get_internal_import_options(INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE, &iopts);
		for (const ImportOption &E : iopts) {
			if (!node_settings.has(E.option.name)) {
				node_settings[E.option.name] = E.default_value;
			}
		}
	}

	if (Object::cast_to<AnimationPlayer>(p_node)) {
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(p_node);

		bool use_optimizer = node_settings["optimizer/enabled"];
		float anim_optimizer_linerr = node_settings["optimizer/max_velocity_error"];
		float anim_optimizer_angerr = node_settings["optimizer/max_angular_error"];
		int anim_optimizer_preerr = node_settings["optimizer/max_precision_error"];

		if (use_optimizer) {
			_optimize_animations(ap, anim_optimizer_linerr, anim_optimizer_angerr, anim_optimizer_preerr);
		}

		bool use_compression = node_settings["compression/enabled"];
		int anim_compression_page_size = node_settings["compression/page_size"];

		if (use_compression) {
			_compress_animations(ap, anim_compression_page_size);
		}

		List<StringName> anims;
		ap->get_animation_list(&anims);
		for (const StringName &name : anims) {
			Ref<Animation> anim = ap->get_animation(name);
			Array animation_slices;

			if (p_animation_data.has(name)) {
				Dictionary anim_settings = p_animation_data[name];

				{
					int slices_count = anim_settings["slices/amount"];

					for (int i = 0; i < slices_count; i++) {
						String slice_name = anim_settings["slice_" + itos(i + 1) + "/name"];
						int from_frame = anim_settings["slice_" + itos(i + 1) + "/start_frame"];
						int end_frame = anim_settings["slice_" + itos(i + 1) + "/end_frame"];
						Animation::LoopMode loop_mode = static_cast<Animation::LoopMode>((int)anim_settings["slice_" + itos(i + 1) + "/loop_mode"]);
						bool save_to_file = anim_settings["slice_" + itos(i + 1) + "/save_to_file/enabled"];
						String save_to_path = anim_settings["slice_" + itos(i + 1) + "/save_to_file/path"];
						bool save_to_file_keep_custom = anim_settings["slice_" + itos(i + 1) + "/save_to_file/keep_custom_tracks"];

						animation_slices.push_back(slice_name);
						animation_slices.push_back(from_frame / p_animation_fps);
						animation_slices.push_back(end_frame / p_animation_fps);
						animation_slices.push_back(loop_mode);
						animation_slices.push_back(save_to_file);
						animation_slices.push_back(save_to_path);
						animation_slices.push_back(save_to_file_keep_custom);
					}

					if (animation_slices.size() > 0) {
						_create_slices(ap, anim, animation_slices, true);
					}
				}
				{
					//fill with default values
					List<ImportOption> iopts;
					get_internal_import_options(INTERNAL_IMPORT_CATEGORY_ANIMATION, &iopts);
					for (const ImportOption &F : iopts) {
						if (!anim_settings.has(F.option.name)) {
							anim_settings[F.option.name] = F.default_value;
						}
					}
				}

				anim->set_loop_mode(static_cast<Animation::LoopMode>((int)anim_settings["settings/loop_mode"]));
				bool save = anim_settings["save_to_file/enabled"];
				String path = anim_settings["save_to_file/path"];
				bool keep_custom = anim_settings["save_to_file/keep_custom_tracks"];

				Ref<Animation> saved_anim = _save_animation_to_file(anim, save, path, keep_custom);

				if (saved_anim != anim) {
					Ref<AnimationLibrary> al = ap->get_animation_library(ap->find_animation_library(anim));
					al->add_animation(name, saved_anim); //replace
				}
			}
		}
	}

	return p_node;
}

Node *ResourceImporterScene::_post_fix_node(Node *p_node, Node *p_root, HashMap<Ref<ImporterMesh>, Vector<Ref<Shape3D>>> &collision_map, Pair<PackedVector3Array, PackedInt32Array> &r_occluder_arrays, HashSet<Ref<ImporterMesh>> &r_scanned_meshes, const Dictionary &p_node_data, const Dictionary &p_material_data, const Dictionary &p_animation_data, float p_animation_fps, float p_applied_root_scale) {
	// children first
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *r = _post_fix_node(p_node->get_child(i), p_root, collision_map, r_occluder_arrays, r_scanned_meshes, p_node_data, p_material_data, p_animation_data, p_animation_fps, p_applied_root_scale);
		if (!r) {
			i--; //was erased
		}
	}

	bool isroot = p_node == p_root;

	String import_id = p_node->get_meta("import_id", "PATH:" + p_root->get_path_to(p_node));

	Dictionary node_settings;
	if (p_node_data.has(import_id)) {
		node_settings = p_node_data[import_id];
	}

	if (!isroot && (node_settings.has("import/skip_import") && bool(node_settings["import/skip_import"]))) {
		p_node->set_owner(nullptr);
		memdelete(p_node);
		return nullptr;
	}

	{
		//make sure this is unique
		node_settings = node_settings.duplicate(true);
		//fill node settings for this node with default values
		List<ImportOption> iopts;
		if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
			get_internal_import_options(INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE, &iopts);
		} else if (Object::cast_to<AnimationPlayer>(p_node)) {
			get_internal_import_options(INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE, &iopts);
		} else if (Object::cast_to<Skeleton3D>(p_node)) {
			get_internal_import_options(INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE, &iopts);
		} else {
			get_internal_import_options(INTERNAL_IMPORT_CATEGORY_NODE, &iopts);
		}
		for (const ImportOption &E : iopts) {
			if (!node_settings.has(E.option.name)) {
				node_settings[E.option.name] = E.default_value;
			}
		}
	}

	{
		ObjectID node_id = p_node->get_instance_id();
		for (int i = 0; i < post_importer_plugins.size(); i++) {
			post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_NODE, p_root, p_node, Ref<Resource>(), node_settings);
			if (ObjectDB::get_instance(node_id) == nullptr) { //may have been erased, so do not continue
				break;
			}
		}
	}

	if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		ObjectID node_id = p_node->get_instance_id();
		for (int i = 0; i < post_importer_plugins.size(); i++) {
			post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE, p_root, p_node, Ref<Resource>(), node_settings);
			if (ObjectDB::get_instance(node_id) == nullptr) { //may have been erased, so do not continue
				break;
			}
		}
	}

	if (Object::cast_to<Skeleton3D>(p_node)) {
		ObjectID node_id = p_node->get_instance_id();
		for (int i = 0; i < post_importer_plugins.size(); i++) {
			post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE, p_root, p_node, Ref<Resource>(), node_settings);
			if (ObjectDB::get_instance(node_id) == nullptr) { //may have been erased, so do not continue
				break;
			}
		}
	}

	if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);

		Ref<ImporterMesh> m = mi->get_mesh();

		if (m.is_valid()) {
			if (!r_scanned_meshes.has(m)) {
				for (int i = 0; i < m->get_surface_count(); i++) {
					Ref<Material> mat = m->get_surface_material(i);
					if (mat.is_valid()) {
						String mat_id = mat->get_meta("import_id", mat->get_name());

						if (!mat_id.is_empty() && p_material_data.has(mat_id)) {
							Dictionary matdata = p_material_data[mat_id];
							{
								//fill node settings for this node with default values
								List<ImportOption> iopts;
								get_internal_import_options(INTERNAL_IMPORT_CATEGORY_MATERIAL, &iopts);
								for (const ImportOption &E : iopts) {
									if (!matdata.has(E.option.name)) {
										matdata[E.option.name] = E.default_value;
									}
								}
							}

							for (int j = 0; j < post_importer_plugins.size(); j++) {
								post_importer_plugins.write[j]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_MATERIAL, p_root, p_node, mat, matdata);
							}

							if (matdata.has("use_external/enabled") && bool(matdata["use_external/enabled"]) && matdata.has("use_external/path")) {
								String path = matdata["use_external/path"];
								Ref<Material> external_mat = ResourceLoader::load(path);
								if (external_mat.is_valid()) {
									m->set_surface_material(i, external_mat);
								}
							}
						}
					}
				}

				r_scanned_meshes.insert(m);
			}

			if (node_settings.has("generate/physics")) {
				int mesh_physics_mode = MeshPhysicsMode::MESH_PHYSICS_DISABLED;

				const bool generate_collider = node_settings["generate/physics"];
				if (generate_collider) {
					mesh_physics_mode = MeshPhysicsMode::MESH_PHYSICS_MESH_AND_STATIC_COLLIDER;
					if (node_settings.has("physics/body_type")) {
						const BodyType body_type = (BodyType)node_settings["physics/body_type"].operator int();
						switch (body_type) {
							case BODY_TYPE_STATIC:
								mesh_physics_mode = MeshPhysicsMode::MESH_PHYSICS_MESH_AND_STATIC_COLLIDER;
								break;
							case BODY_TYPE_DYNAMIC:
								mesh_physics_mode = MeshPhysicsMode::MESH_PHYSICS_RIGID_BODY_AND_MESH;
								break;
							case BODY_TYPE_AREA:
								mesh_physics_mode = MeshPhysicsMode::MESH_PHYSICS_AREA_ONLY;
								break;
						}
					}
				}

				if (mesh_physics_mode != MeshPhysicsMode::MESH_PHYSICS_DISABLED) {
					Vector<Ref<Shape3D>> shapes;
					if (collision_map.has(m)) {
						shapes = collision_map[m];
					} else {
						shapes = get_collision_shapes(
								m,
								node_settings,
								p_applied_root_scale);
					}

					if (shapes.size()) {
						CollisionObject3D *base = nullptr;
						switch (mesh_physics_mode) {
							case MESH_PHYSICS_MESH_AND_STATIC_COLLIDER: {
								StaticBody3D *col = memnew(StaticBody3D);
								p_node->add_child(col, true);
								col->set_owner(p_node->get_owner());
								col->set_transform(get_collision_shapes_transform(node_settings));
								col->set_position(p_applied_root_scale * col->get_position());
								const Ref<PhysicsMaterial> &pmo = node_settings["physics/physics_material_override"];
								if (!pmo.is_null()) {
									col->set_physics_material_override(pmo);
								}
								base = col;
							} break;
							case MESH_PHYSICS_RIGID_BODY_AND_MESH: {
								RigidBody3D *rigid_body = memnew(RigidBody3D);
								rigid_body->set_name(p_node->get_name());
								p_node->replace_by(rigid_body);
								rigid_body->set_transform(mi->get_transform() * get_collision_shapes_transform(node_settings));
								rigid_body->set_position(p_applied_root_scale * rigid_body->get_position());
								p_node = rigid_body;
								mi->set_transform(Transform3D());
								rigid_body->add_child(mi, true);
								mi->set_owner(rigid_body->get_owner());
								const Ref<PhysicsMaterial> &pmo = node_settings["physics/physics_material_override"];
								if (!pmo.is_null()) {
									rigid_body->set_physics_material_override(pmo);
								}
								base = rigid_body;
							} break;
							case MESH_PHYSICS_STATIC_COLLIDER_ONLY: {
								StaticBody3D *col = memnew(StaticBody3D);
								col->set_transform(mi->get_transform() * get_collision_shapes_transform(node_settings));
								col->set_position(p_applied_root_scale * col->get_position());
								col->set_name(p_node->get_name());
								p_node->replace_by(col);
								p_node->set_owner(nullptr);
								memdelete(p_node);
								p_node = col;
								const Ref<PhysicsMaterial> &pmo = node_settings["physics/physics_material_override"];
								if (!pmo.is_null()) {
									col->set_physics_material_override(pmo);
								}
								base = col;
							} break;
							case MESH_PHYSICS_AREA_ONLY: {
								Area3D *area = memnew(Area3D);
								area->set_transform(mi->get_transform() * get_collision_shapes_transform(node_settings));
								area->set_position(p_applied_root_scale * area->get_position());
								area->set_name(p_node->get_name());
								p_node->replace_by(area);
								p_node->set_owner(nullptr);
								memdelete(p_node);
								p_node = area;
								base = area;

							} break;
						}

						base->set_collision_layer(node_settings["physics/layer"]);
						base->set_collision_mask(node_settings["physics/mask"]);

						for (const Ref<Shape3D> &E : shapes) {
							CollisionShape3D *cshape = memnew(CollisionShape3D);
							cshape->set_shape(E);
							base->add_child(cshape, true);

							cshape->set_owner(base->get_owner());
						}
					}
				}
			}
		}
	}

	//navmesh (node may have changed type above)
	if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);

		Ref<ImporterMesh> m = mi->get_mesh();

		if (m.is_valid()) {
			if (node_settings.has("generate/navmesh")) {
				int navmesh_mode = node_settings["generate/navmesh"];

				if (navmesh_mode != NAVMESH_DISABLED) {
					NavigationRegion3D *nmi = memnew(NavigationRegion3D);

					Ref<NavigationMesh> nmesh = m->create_navigation_mesh();
					nmi->set_navigation_mesh(nmesh);

					if (navmesh_mode == NAVMESH_NAVMESH_ONLY) {
						nmi->set_transform(mi->get_transform());
						p_node->replace_by(nmi);
						p_node->set_owner(nullptr);
						memdelete(p_node);
						p_node = nmi;
					} else {
						mi->add_child(nmi, true);
						nmi->set_owner(mi->get_owner());
					}
				}
			}
		}
	}

	if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);

		Ref<ImporterMesh> m = mi->get_mesh();

		if (m.is_valid()) {
			if (node_settings.has("generate/occluder")) {
				int occluder_mode = node_settings["generate/occluder"];

				if (occluder_mode != OCCLUDER_DISABLED) {
					float simplification_dist = 0.0f;
					if (node_settings.has("occluder/simplification_distance")) {
						simplification_dist = node_settings["occluder/simplification_distance"];
					}

					OccluderInstance3D::bake_single_node(mi, simplification_dist, r_occluder_arrays.first, r_occluder_arrays.second);

					if (occluder_mode == OCCLUDER_OCCLUDER_ONLY) {
						p_node->set_owner(nullptr);
						memdelete(p_node);
						p_node = nullptr;
					}
				}
			}
		}
	}

	if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);

		if (node_settings.has("mesh_instance/layers")) {
			mi->set_layer_mask(node_settings["mesh_instance/layers"]);
		}

		if (node_settings.has("mesh_instance/visibility_range_begin")) {
			mi->set_visibility_range_begin(node_settings["mesh_instance/visibility_range_begin"]);
		}

		if (node_settings.has("mesh_instance/visibility_range_begin_margin")) {
			mi->set_visibility_range_begin_margin(node_settings["mesh_instance/visibility_range_begin_margin"]);
		}

		if (node_settings.has("mesh_instance/visibility_range_end")) {
			mi->set_visibility_range_end(node_settings["mesh_instance/visibility_range_end"]);
		}

		if (node_settings.has("mesh_instance/visibility_range_end_margin")) {
			mi->set_visibility_range_end_margin(node_settings["mesh_instance/visibility_range_end_margin"]);
		}

		if (node_settings.has("mesh_instance/visibility_range_fade_mode")) {
			const GeometryInstance3D::VisibilityRangeFadeMode range_fade_mode = (GeometryInstance3D::VisibilityRangeFadeMode)node_settings["mesh_instance/visibility_range_fade_mode"].operator int();
			mi->set_visibility_range_fade_mode(range_fade_mode);
		}

		if (node_settings.has("mesh_instance/cast_shadow")) {
			const GeometryInstance3D::ShadowCastingSetting cast_shadows = (GeometryInstance3D::ShadowCastingSetting)node_settings["mesh_instance/cast_shadow"].operator int();
			mi->set_cast_shadows_setting(cast_shadows);
		}
	}

	if (Object::cast_to<AnimationPlayer>(p_node)) {
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(p_node);

		for (int i = 0; i < post_importer_plugins.size(); i++) {
			post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE, p_root, p_node, Ref<Resource>(), node_settings);
		}

		if (post_importer_plugins.size()) {
			List<StringName> anims;
			ap->get_animation_list(&anims);
			for (const StringName &name : anims) {
				if (p_animation_data.has(name)) {
					Ref<Animation> anim = ap->get_animation(name);
					Dictionary anim_settings = p_animation_data[name];
					{
						//fill with default values
						List<ImportOption> iopts;
						get_internal_import_options(INTERNAL_IMPORT_CATEGORY_ANIMATION, &iopts);
						for (const ImportOption &F : iopts) {
							if (!anim_settings.has(F.option.name)) {
								anim_settings[F.option.name] = F.default_value;
							}
						}
					}

					for (int i = 0; i < post_importer_plugins.size(); i++) {
						post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_ANIMATION, p_root, p_node, anim, anim_settings);
					}
				}
			}
		}
	}

	return p_node;
}

Ref<Animation> ResourceImporterScene::_save_animation_to_file(Ref<Animation> anim, bool p_save_to_file, String p_save_to_path, bool p_keep_custom_tracks) {
	if (!p_save_to_file || !p_save_to_path.is_resource_file()) {
		return anim;
	}

	if (FileAccess::exists(p_save_to_path) && p_keep_custom_tracks) {
		// Copy custom animation tracks from previously imported files.
		Ref<Animation> old_anim = ResourceLoader::load(p_save_to_path, "Animation", ResourceFormatLoader::CACHE_MODE_IGNORE);
		if (old_anim.is_valid()) {
			for (int i = 0; i < old_anim->get_track_count(); i++) {
				if (!old_anim->track_is_imported(i)) {
					old_anim->copy_track(i, anim);
				}
			}
			anim->set_loop_mode(old_anim->get_loop_mode());
		}
	}

	if (ResourceCache::has(p_save_to_path)) {
		Ref<Animation> old_anim = ResourceCache::get_ref(p_save_to_path);
		if (old_anim.is_valid()) {
			old_anim->copy_from(anim);
			anim = old_anim;
		}
	}
	anim->set_path(p_save_to_path, true); // Set path to save externally.
	Error err = ResourceSaver::save(anim, p_save_to_path, ResourceSaver::FLAG_CHANGE_PATH);
	ERR_FAIL_COND_V_MSG(err != OK, anim, "Saving of animation failed: " + p_save_to_path);
	return anim;
}

void ResourceImporterScene::_create_slices(AnimationPlayer *ap, Ref<Animation> anim, const Array &p_slices, bool p_bake_all) {
	Ref<AnimationLibrary> al = ap->get_animation_library(ap->find_animation_library(anim));

	for (int i = 0; i < p_slices.size(); i += 7) {
		String name = p_slices[i];
		float from = p_slices[i + 1];
		float to = p_slices[i + 2];
		Animation::LoopMode loop_mode = static_cast<Animation::LoopMode>((int)p_slices[i + 3]);
		bool save_to_file = p_slices[i + 4];
		String save_to_path = p_slices[i + 5];
		bool keep_current = p_slices[i + 6];
		if (from >= to) {
			continue;
		}

		Ref<Animation> new_anim = memnew(Animation);

		for (int j = 0; j < anim->get_track_count(); j++) {
			List<float> keys;
			int kc = anim->track_get_key_count(j);
			int dtrack = -1;
			for (int k = 0; k < kc; k++) {
				float kt = anim->track_get_key_time(j, k);
				if (kt >= from && kt < to) {
					//found a key within range, so create track
					if (dtrack == -1) {
						new_anim->add_track(anim->track_get_type(j));
						dtrack = new_anim->get_track_count() - 1;
						new_anim->track_set_path(dtrack, anim->track_get_path(j));
						new_anim->track_set_imported(dtrack, true);

						if (kt > (from + 0.01) && k > 0) {
							if (anim->track_get_type(j) == Animation::TYPE_POSITION_3D) {
								Vector3 p;
								anim->try_position_track_interpolate(j, from, &p);
								new_anim->position_track_insert_key(dtrack, 0, p);
							} else if (anim->track_get_type(j) == Animation::TYPE_ROTATION_3D) {
								Quaternion r;
								anim->try_rotation_track_interpolate(j, from, &r);
								new_anim->rotation_track_insert_key(dtrack, 0, r);
							} else if (anim->track_get_type(j) == Animation::TYPE_SCALE_3D) {
								Vector3 s;
								anim->try_scale_track_interpolate(j, from, &s);
								new_anim->scale_track_insert_key(dtrack, 0, s);
							} else if (anim->track_get_type(j) == Animation::TYPE_VALUE) {
								Variant var = anim->value_track_interpolate(j, from);
								new_anim->track_insert_key(dtrack, 0, var);
							} else if (anim->track_get_type(j) == Animation::TYPE_BLEND_SHAPE) {
								float interp;
								anim->try_blend_shape_track_interpolate(j, from, &interp);
								new_anim->blend_shape_track_insert_key(dtrack, 0, interp);
							}
						}
					}

					if (anim->track_get_type(j) == Animation::TYPE_POSITION_3D) {
						Vector3 p;
						anim->position_track_get_key(j, k, &p);
						new_anim->position_track_insert_key(dtrack, kt - from, p);
					} else if (anim->track_get_type(j) == Animation::TYPE_ROTATION_3D) {
						Quaternion r;
						anim->rotation_track_get_key(j, k, &r);
						new_anim->rotation_track_insert_key(dtrack, kt - from, r);
					} else if (anim->track_get_type(j) == Animation::TYPE_SCALE_3D) {
						Vector3 s;
						anim->scale_track_get_key(j, k, &s);
						new_anim->scale_track_insert_key(dtrack, kt - from, s);
					} else if (anim->track_get_type(j) == Animation::TYPE_VALUE) {
						Variant var = anim->track_get_key_value(j, k);
						new_anim->track_insert_key(dtrack, kt - from, var);
					} else if (anim->track_get_type(j) == Animation::TYPE_BLEND_SHAPE) {
						float interp;
						anim->blend_shape_track_get_key(j, k, &interp);
						new_anim->blend_shape_track_insert_key(dtrack, kt - from, interp);
					}
				}

				if (dtrack != -1 && kt >= to) {
					if (anim->track_get_type(j) == Animation::TYPE_POSITION_3D) {
						Vector3 p;
						anim->try_position_track_interpolate(j, to, &p);
						new_anim->position_track_insert_key(dtrack, to - from, p);
					} else if (anim->track_get_type(j) == Animation::TYPE_ROTATION_3D) {
						Quaternion r;
						anim->try_rotation_track_interpolate(j, to, &r);
						new_anim->rotation_track_insert_key(dtrack, to - from, r);
					} else if (anim->track_get_type(j) == Animation::TYPE_SCALE_3D) {
						Vector3 s;
						anim->try_scale_track_interpolate(j, to, &s);
						new_anim->scale_track_insert_key(dtrack, to - from, s);
					} else if (anim->track_get_type(j) == Animation::TYPE_VALUE) {
						Variant var = anim->value_track_interpolate(j, to);
						new_anim->track_insert_key(dtrack, to - from, var);
					} else if (anim->track_get_type(j) == Animation::TYPE_BLEND_SHAPE) {
						float interp;
						anim->try_blend_shape_track_interpolate(j, to, &interp);
						new_anim->blend_shape_track_insert_key(dtrack, to - from, interp);
					}
				}
			}

			if (dtrack == -1 && p_bake_all) {
				new_anim->add_track(anim->track_get_type(j));
				dtrack = new_anim->get_track_count() - 1;
				new_anim->track_set_path(dtrack, anim->track_get_path(j));
				new_anim->track_set_imported(dtrack, true);
				if (anim->track_get_type(j) == Animation::TYPE_POSITION_3D) {
					Vector3 p;
					anim->try_position_track_interpolate(j, from, &p);
					new_anim->position_track_insert_key(dtrack, 0, p);
					anim->try_position_track_interpolate(j, to, &p);
					new_anim->position_track_insert_key(dtrack, to - from, p);
				} else if (anim->track_get_type(j) == Animation::TYPE_ROTATION_3D) {
					Quaternion r;
					anim->try_rotation_track_interpolate(j, from, &r);
					new_anim->rotation_track_insert_key(dtrack, 0, r);
					anim->try_rotation_track_interpolate(j, to, &r);
					new_anim->rotation_track_insert_key(dtrack, to - from, r);
				} else if (anim->track_get_type(j) == Animation::TYPE_SCALE_3D) {
					Vector3 s;
					anim->try_scale_track_interpolate(j, from, &s);
					new_anim->scale_track_insert_key(dtrack, 0, s);
					anim->try_scale_track_interpolate(j, to, &s);
					new_anim->scale_track_insert_key(dtrack, to - from, s);
				} else if (anim->track_get_type(j) == Animation::TYPE_VALUE) {
					Variant var = anim->value_track_interpolate(j, from);
					new_anim->track_insert_key(dtrack, 0, var);
					Variant to_var = anim->value_track_interpolate(j, to);
					new_anim->track_insert_key(dtrack, to - from, to_var);
				} else if (anim->track_get_type(j) == Animation::TYPE_BLEND_SHAPE) {
					float interp;
					anim->try_blend_shape_track_interpolate(j, from, &interp);
					new_anim->blend_shape_track_insert_key(dtrack, 0, interp);
					anim->try_blend_shape_track_interpolate(j, to, &interp);
					new_anim->blend_shape_track_insert_key(dtrack, to - from, interp);
				}
			}
		}

		new_anim->set_loop_mode(loop_mode);
		new_anim->set_length(to - from);

		al->add_animation(name, new_anim);

		Ref<Animation> saved_anim = _save_animation_to_file(new_anim, save_to_file, save_to_path, keep_current);
		if (saved_anim != new_anim) {
			al->add_animation(name, saved_anim);
		}
	}

	al->remove_animation(ap->find_animation(anim)); // Remove original animation (no longer needed).
}

void ResourceImporterScene::_optimize_animations(AnimationPlayer *anim, float p_max_vel_error, float p_max_ang_error, int p_prc_error) {
	List<StringName> anim_names;
	anim->get_animation_list(&anim_names);
	for (const StringName &E : anim_names) {
		Ref<Animation> a = anim->get_animation(E);
		a->optimize(p_max_vel_error, p_max_ang_error, p_prc_error);
	}
}

void ResourceImporterScene::_compress_animations(AnimationPlayer *anim, int p_page_size_kb) {
	List<StringName> anim_names;
	anim->get_animation_list(&anim_names);
	for (const StringName &E : anim_names) {
		Ref<Animation> a = anim->get_animation(E);
		a->compress(p_page_size_kb * 1024);
	}
}

void ResourceImporterScene::get_internal_import_options(InternalImportCategory p_category, List<ImportOption> *r_options) const {
	switch (p_category) {
		case INTERNAL_IMPORT_CATEGORY_NODE: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "import/skip_import", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
		} break;
		case INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "import/skip_import", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "generate/physics", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/navmesh", PROPERTY_HINT_ENUM, "Disabled,Mesh + NavMesh,NavMesh Only"), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "physics/body_type", PROPERTY_HINT_ENUM, "Static,Dynamic,Area"), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "physics/shape_type", PROPERTY_HINT_ENUM, "Decompose Convex,Simple Convex,Trimesh,Box,Sphere,Cylinder,Capsule", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::OBJECT, "physics/physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), Variant()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "physics/layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), 1));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "physics/mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), 1));

			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "mesh_instance/layers", PROPERTY_HINT_LAYERS_3D_RENDER), 1));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "mesh_instance/visibility_range_begin", PROPERTY_HINT_RANGE, "0.0,4096.0,0.01,or_greater,suffix:m"), 0.0f));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "mesh_instance/visibility_range_begin_margin", PROPERTY_HINT_RANGE, "0.0,4096.0,0.01,or_greater,suffix:m"), 0.0f));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "mesh_instance/visibility_range_end", PROPERTY_HINT_RANGE, "0.0,4096.0,0.01,or_greater,suffix:m"), 0.0f));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "mesh_instance/visibility_range_end_margin", PROPERTY_HINT_RANGE, "0.0,4096.0,0.01,or_greater,suffix:m"), 0.0f));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "mesh_instance/visibility_range_fade_mode", PROPERTY_HINT_ENUM, "Disabled,Self,Dependencies"), GeometryInstance3D::VISIBILITY_RANGE_FADE_DISABLED));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "mesh_instance/cast_shadow", PROPERTY_HINT_ENUM, "Off,On,Double-Sided,Shadows Only"), GeometryInstance3D::SHADOW_CASTING_SETTING_ON));

			// Decomposition
			Ref<MeshConvexDecompositionSettings> decomposition_default = Ref<MeshConvexDecompositionSettings>();
			decomposition_default.instantiate();
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "decomposition/advanced", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/precision", PROPERTY_HINT_RANGE, "1,10,1"), 5));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "decomposition/max_concavity", PROPERTY_HINT_RANGE, "0.0,1.0,0.001", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_max_concavity()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "decomposition/symmetry_planes_clipping_bias", PROPERTY_HINT_RANGE, "0.0,1.0,0.001", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_symmetry_planes_clipping_bias()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "decomposition/revolution_axes_clipping_bias", PROPERTY_HINT_RANGE, "0.0,1.0,0.001", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_revolution_axes_clipping_bias()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "decomposition/min_volume_per_convex_hull", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_min_volume_per_convex_hull()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/resolution", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_resolution()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/max_num_vertices_per_convex_hull", PROPERTY_HINT_RANGE, "5,512,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_max_num_vertices_per_convex_hull()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/plane_downsampling", PROPERTY_HINT_RANGE, "1,16,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_plane_downsampling()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/convexhull_downsampling", PROPERTY_HINT_RANGE, "1,16,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_convex_hull_downsampling()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "decomposition/normalize_mesh", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_normalize_mesh()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/mode", PROPERTY_HINT_ENUM, "Voxel,Tetrahedron", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), static_cast<int>(decomposition_default->get_mode())));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "decomposition/convexhull_approximation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_convex_hull_approximation()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/max_convex_hulls", PROPERTY_HINT_RANGE, "1,100,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_max_convex_hulls()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "decomposition/project_hull_vertices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default->get_project_hull_vertices()));

			// Primitives: Box, Sphere, Cylinder, Capsule.
			r_options->push_back(ImportOption(PropertyInfo(Variant::VECTOR3, "primitive/size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), Vector3(2.0, 2.0, 2.0)));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "primitive/height", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 1.0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "primitive/radius", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 1.0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::VECTOR3, "primitive/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), Vector3()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::VECTOR3, "primitive/rotation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), Vector3()));

			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/occluder", PROPERTY_HINT_ENUM, "Disabled,Mesh + Occluder,Occluder Only", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "occluder/simplification_distance", PROPERTY_HINT_RANGE, "0.0,2.0,0.01", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0.1f));
		} break;
		case INTERNAL_IMPORT_CATEGORY_MESH: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "save_to_file/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "save_to_file/path", PROPERTY_HINT_SAVE_FILE, "*.res,*.tres"), ""));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "save_to_file/make_streamable"), ""));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/shadow_meshes", PROPERTY_HINT_ENUM, "Default,Enable,Disable"), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/lightmap_uv", PROPERTY_HINT_ENUM, "Default,Enable,Disable"), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/lods", PROPERTY_HINT_ENUM, "Default,Enable,Disable"), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "lods/normal_split_angle", PROPERTY_HINT_RANGE, "0,180,0.1,degrees"), 25.0f));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "lods/normal_merge_angle", PROPERTY_HINT_RANGE, "0,180,0.1,degrees"), 60.0f));
		} break;
		case INTERNAL_IMPORT_CATEGORY_MATERIAL: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "use_external/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "use_external/path", PROPERTY_HINT_FILE, "*.material,*.res,*.tres"), ""));
		} break;
		case INTERNAL_IMPORT_CATEGORY_ANIMATION: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "settings/loop_mode", PROPERTY_HINT_ENUM, "None,Linear,Pingpong"), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "save_to_file/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "save_to_file/path", PROPERTY_HINT_SAVE_FILE, "*.res,*.tres"), ""));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "save_to_file/keep_custom_tracks"), ""));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "slices/amount", PROPERTY_HINT_RANGE, "0,256,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));

			for (int i = 0; i < 256; i++) {
				r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "slice_" + itos(i + 1) + "/name"), ""));
				r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "slice_" + itos(i + 1) + "/start_frame"), 0));
				r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "slice_" + itos(i + 1) + "/end_frame"), 0));
				r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "slice_" + itos(i + 1) + "/loop_mode", PROPERTY_HINT_ENUM, "None,Linear,Pingpong"), 0));
				r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "slice_" + itos(i + 1) + "/save_to_file/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
				r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "slice_" + itos(i + 1) + "/save_to_file/path", PROPERTY_HINT_SAVE_FILE, ".res,*.tres"), ""));
				r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "slice_" + itos(i + 1) + "/save_to_file/keep_custom_tracks"), false));
			}
		} break;
		case INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "import/skip_import", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "optimizer/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), true));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "optimizer/max_velocity_error", PROPERTY_HINT_RANGE, "0,1,0.01"), 0.01));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "optimizer/max_angular_error", PROPERTY_HINT_RANGE, "0,1,0.01"), 0.01));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "optimizer/max_precision_error", PROPERTY_HINT_NONE, "1,6,1"), 3));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compression/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compression/page_size", PROPERTY_HINT_RANGE, "4,512,1,suffix:kb"), 8));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "import_tracks/position", PROPERTY_HINT_ENUM, "IfPresent,IfPresentForAll,Never"), 1));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "import_tracks/rotation", PROPERTY_HINT_ENUM, "IfPresent,IfPresentForAll,Never"), 1));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "import_tracks/scale", PROPERTY_HINT_ENUM, "IfPresent,IfPresentForAll,Never"), 1));
		} break;
		case INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "import/skip_import", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::OBJECT, "retarget/bone_map", PROPERTY_HINT_RESOURCE_TYPE, "BoneMap", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), Variant()));
		} break;
		default: {
		}
	}

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		post_importer_plugins.write[i]->get_internal_import_options(EditorScenePostImportPlugin::InternalImportCategory(p_category), r_options);
	}
}

bool ResourceImporterScene::get_internal_option_visibility(InternalImportCategory p_category, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	if (p_options.has("import/skip_import") && p_option != "import/skip_import" && bool(p_options["import/skip_import"])) {
		return false; //if skip import
	}
	switch (p_category) {
		case INTERNAL_IMPORT_CATEGORY_NODE: {
		} break;
		case INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE: {
			const bool generate_physics =
					p_options.has("generate/physics") &&
					p_options["generate/physics"].operator bool();

			if (p_option.find("physics/") >= 0) {
				// Show if need to generate collisions.
				return generate_physics;
			}

			if (p_option.find("decomposition/") >= 0) {
				// Show if need to generate collisions.
				if (generate_physics &&
						// Show if convex is enabled.
						p_options["physics/shape_type"] == Variant(SHAPE_TYPE_DECOMPOSE_CONVEX)) {
					if (p_option == "decomposition/advanced") {
						return true;
					}

					const bool decomposition_advanced =
							p_options.has("decomposition/advanced") &&
							p_options["decomposition/advanced"].operator bool();

					if (p_option == "decomposition/precision") {
						return !decomposition_advanced;
					} else {
						return decomposition_advanced;
					}
				}

				return false;
			}

			if (p_option == "primitive/position" || p_option == "primitive/rotation") {
				const ShapeType physics_shape = (ShapeType)p_options["physics/shape_type"].operator int();
				return generate_physics &&
						physics_shape >= SHAPE_TYPE_BOX;
			}

			if (p_option == "primitive/size") {
				const ShapeType physics_shape = (ShapeType)p_options["physics/shape_type"].operator int();
				return generate_physics &&
						physics_shape == SHAPE_TYPE_BOX;
			}

			if (p_option == "primitive/radius") {
				const ShapeType physics_shape = (ShapeType)p_options["physics/shape_type"].operator int();
				return generate_physics &&
						(physics_shape == SHAPE_TYPE_SPHERE ||
								physics_shape == SHAPE_TYPE_CYLINDER ||
								physics_shape == SHAPE_TYPE_CAPSULE);
			}

			if (p_option == "primitive/height") {
				const ShapeType physics_shape = (ShapeType)p_options["physics/shape_type"].operator int();
				return generate_physics &&
						(physics_shape == SHAPE_TYPE_CYLINDER ||
								physics_shape == SHAPE_TYPE_CAPSULE);
			}

			if (p_option == "occluder/simplification_distance") {
				// Show only if occluder generation is enabled
				return p_options.has("generate/occluder") && p_options["generate/occluder"].operator signed int() != OCCLUDER_DISABLED;
			}
		} break;
		case INTERNAL_IMPORT_CATEGORY_MESH: {
			if (p_option == "save_to_file/path" || p_option == "save_to_file/make_streamable") {
				return p_options["save_to_file/enabled"];
			}
		} break;
		case INTERNAL_IMPORT_CATEGORY_MATERIAL: {
			if (p_option == "use_external/path") {
				return p_options["use_external/enabled"];
			}
		} break;
		case INTERNAL_IMPORT_CATEGORY_ANIMATION: {
			if (p_option == "save_to_file/path" || p_option == "save_to_file/keep_custom_tracks") {
				return p_options["save_to_file/enabled"];
			}
			if (p_option.begins_with("slice_")) {
				int max_slice = p_options["slices/amount"];
				int slice = p_option.get_slice("_", 1).to_int() - 1;
				if (slice >= max_slice) {
					return false;
				}
			}
		} break;
		case INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE: {
			if (p_option.begins_with("optimizer/") && p_option != "optimizer/enabled" && !bool(p_options["optimizer/enabled"])) {
				return false;
			}
			if (p_option.begins_with("compression/") && p_option != "compression/enabled" && !bool(p_options["compression/enabled"])) {
				return false;
			}
		} break;
		case INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE: {
			const bool use_retarget = p_options["retarget/bone_map"].get_validated_object() != nullptr;
			if (p_option != "retarget/bone_map" && p_option.begins_with("retarget/")) {
				return use_retarget;
			}
		} break;
		default: {
		}
	}

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		Variant ret = post_importer_plugins.write[i]->get_internal_option_visibility(EditorScenePostImportPlugin::InternalImportCategory(p_category), animation_importer, p_option, p_options);
		if (ret.get_type() == Variant::BOOL) {
			return ret;
		}
	}

	return true;
}

bool ResourceImporterScene::get_internal_option_update_view_required(InternalImportCategory p_category, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	switch (p_category) {
		case INTERNAL_IMPORT_CATEGORY_NODE: {
		} break;
		case INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE: {
			if (
					p_option == "generate/physics" ||
					p_option == "physics/shape_type" ||
					p_option.find("decomposition/") >= 0 ||
					p_option.find("primitive/") >= 0) {
				return true;
			}
		} break;
		case INTERNAL_IMPORT_CATEGORY_MESH: {
		} break;
		case INTERNAL_IMPORT_CATEGORY_MATERIAL: {
		} break;
		case INTERNAL_IMPORT_CATEGORY_ANIMATION: {
		} break;
		case INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE: {
		} break;
		case INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE: {
		} break;
		default: {
		}
	}

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		Variant ret = post_importer_plugins.write[i]->get_internal_option_update_view_required(EditorScenePostImportPlugin::InternalImportCategory(p_category), p_option, p_options);
		if (ret.get_type() == Variant::BOOL) {
			return ret;
		}
	}

	return false;
}

void ResourceImporterScene::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_type", PROPERTY_HINT_TYPE_STRING, "Node"), ""));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_name"), ""));

	List<String> script_extentions;
	ResourceLoader::get_recognized_extensions_for_type("Script", &script_extentions);

	String script_ext_hint;

	for (const String &E : script_extentions) {
		if (!script_ext_hint.is_empty()) {
			script_ext_hint += ",";
		}
		script_ext_hint += "*." + E;
	}

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "nodes/apply_root_scale"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "nodes/root_scale", PROPERTY_HINT_RANGE, "0.001,1000,0.001"), 1.0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/ensure_tangents"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/generate_lods"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/create_shadow_meshes"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "meshes/light_baking", PROPERTY_HINT_ENUM, "Disabled,Static,Static Lightmaps,Dynamic", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "meshes/lightmap_texel_size", PROPERTY_HINT_RANGE, "0.001,100,0.001"), 0.2));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/force_disable_compression"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "skins/use_named_skins"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/import"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "animation/fps", PROPERTY_HINT_RANGE, "1,120,1"), 30));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/trimming"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/remove_immutable_tracks"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "import_script/path", PROPERTY_HINT_FILE, script_ext_hint), ""));

	r_options->push_back(ImportOption(PropertyInfo(Variant::DICTIONARY, "_subresources", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), Dictionary()));

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		post_importer_plugins.write[i]->get_import_options(p_path, r_options);
	}

	for (Ref<EditorSceneFormatImporter> importer_elem : scene_importers) {
		importer_elem->get_import_options(p_path, r_options);
	}
}

void ResourceImporterScene::handle_compatibility_options(HashMap<StringName, Variant> &p_import_params) const {
	for (Ref<EditorSceneFormatImporter> importer_elem : scene_importers) {
		importer_elem->handle_compatibility_options(p_import_params);
	}
}

void ResourceImporterScene::_replace_owner(Node *p_node, Node *p_scene, Node *p_new_owner) {
	if (p_node != p_new_owner && p_node->get_owner() == p_scene) {
		p_node->set_owner(p_new_owner);
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *n = p_node->get_child(i);
		_replace_owner(n, p_scene, p_new_owner);
	}
}

Array ResourceImporterScene::_get_skinned_pose_transforms(ImporterMeshInstance3D *p_src_mesh_node) {
	Array skin_pose_transform_array;

	const Ref<Skin> skin = p_src_mesh_node->get_skin();
	if (skin.is_valid()) {
		NodePath skeleton_path = p_src_mesh_node->get_skeleton_path();
		const Node *node = p_src_mesh_node->get_node_or_null(skeleton_path);
		const Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(node);
		if (skeleton) {
			int bind_count = skin->get_bind_count();

			for (int i = 0; i < bind_count; i++) {
				Transform3D bind_pose = skin->get_bind_pose(i);
				String bind_name = skin->get_bind_name(i);

				int bone_idx = bind_name.is_empty() ? skin->get_bind_bone(i) : skeleton->find_bone(bind_name);
				ERR_FAIL_COND_V(bone_idx >= skeleton->get_bone_count(), Array());

				Transform3D bp_global_rest;
				if (bone_idx >= 0) {
					bp_global_rest = skeleton->get_bone_global_pose(bone_idx);
				} else {
					bp_global_rest = skeleton->get_bone_global_pose(i);
				}

				skin_pose_transform_array.push_back(bp_global_rest * bind_pose);
			}
		}
	}

	return skin_pose_transform_array;
}

Node *ResourceImporterScene::_generate_meshes(Node *p_node, const Dictionary &p_mesh_data, bool p_generate_lods, bool p_create_shadow_meshes, LightBakeMode p_light_bake_mode, float p_lightmap_texel_size, const Vector<uint8_t> &p_src_lightmap_cache, Vector<Vector<uint8_t>> &r_lightmap_caches) {
	ImporterMeshInstance3D *src_mesh_node = Object::cast_to<ImporterMeshInstance3D>(p_node);
	if (src_mesh_node) {
		//is mesh
		MeshInstance3D *mesh_node = memnew(MeshInstance3D);
		mesh_node->set_name(src_mesh_node->get_name());
		mesh_node->set_transform(src_mesh_node->get_transform());
		mesh_node->set_skin(src_mesh_node->get_skin());
		mesh_node->set_skeleton_path(src_mesh_node->get_skeleton_path());
		if (src_mesh_node->get_mesh().is_valid()) {
			Ref<ArrayMesh> mesh;
			if (!src_mesh_node->get_mesh()->has_mesh()) {
				//do mesh processing

				bool generate_lods = p_generate_lods;
				float split_angle = 25.0f;
				float merge_angle = 60.0f;
				bool create_shadow_meshes = p_create_shadow_meshes;
				bool bake_lightmaps = p_light_bake_mode == LIGHT_BAKE_STATIC_LIGHTMAPS;
				String save_to_file;

				String mesh_id = src_mesh_node->get_mesh()->get_meta("import_id", src_mesh_node->get_mesh()->get_name());

				if (!mesh_id.is_empty() && p_mesh_data.has(mesh_id)) {
					Dictionary mesh_settings = p_mesh_data[mesh_id];
					{
						//fill node settings for this node with default values
						List<ImportOption> iopts;
						get_internal_import_options(INTERNAL_IMPORT_CATEGORY_MESH, &iopts);
						for (const ImportOption &E : iopts) {
							if (!mesh_settings.has(E.option.name)) {
								mesh_settings[E.option.name] = E.default_value;
							}
						}
					}

					if (mesh_settings.has("generate/shadow_meshes")) {
						int shadow_meshes = mesh_settings["generate/shadow_meshes"];
						if (shadow_meshes == MESH_OVERRIDE_ENABLE) {
							create_shadow_meshes = true;
						} else if (shadow_meshes == MESH_OVERRIDE_DISABLE) {
							create_shadow_meshes = false;
						}
					}

					if (mesh_settings.has("generate/lightmap_uv")) {
						int lightmap_uv = mesh_settings["generate/lightmap_uv"];
						if (lightmap_uv == MESH_OVERRIDE_ENABLE) {
							bake_lightmaps = true;
						} else if (lightmap_uv == MESH_OVERRIDE_DISABLE) {
							bake_lightmaps = false;
						}
					}

					if (mesh_settings.has("generate/lods")) {
						int lods = mesh_settings["generate/lods"];
						if (lods == MESH_OVERRIDE_ENABLE) {
							generate_lods = true;
						} else if (lods == MESH_OVERRIDE_DISABLE) {
							generate_lods = false;
						}
					}

					if (mesh_settings.has("lods/normal_split_angle")) {
						split_angle = mesh_settings["lods/normal_split_angle"];
					}

					if (mesh_settings.has("lods/normal_merge_angle")) {
						merge_angle = mesh_settings["lods/normal_merge_angle"];
					}

					if (mesh_settings.has("save_to_file/enabled") && bool(mesh_settings["save_to_file/enabled"]) && mesh_settings.has("save_to_file/path")) {
						save_to_file = mesh_settings["save_to_file/path"];
						if (!save_to_file.is_resource_file()) {
							save_to_file = "";
						}
					}

					for (int i = 0; i < post_importer_plugins.size(); i++) {
						post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_MESH, nullptr, src_mesh_node, src_mesh_node->get_mesh(), mesh_settings);
					}
				}

				if (bake_lightmaps) {
					Transform3D xf;
					Node3D *n = src_mesh_node;
					while (n) {
						xf = n->get_transform() * xf;
						n = n->get_parent_node_3d();
					}

					Vector<uint8_t> lightmap_cache;
					src_mesh_node->get_mesh()->lightmap_unwrap_cached(xf, p_lightmap_texel_size, p_src_lightmap_cache, lightmap_cache);

					if (!lightmap_cache.is_empty()) {
						if (r_lightmap_caches.is_empty()) {
							r_lightmap_caches.push_back(lightmap_cache);
						} else {
							String new_md5 = String::md5(lightmap_cache.ptr()); // MD5 is stored at the beginning of the cache data

							for (int i = 0; i < r_lightmap_caches.size(); i++) {
								String md5 = String::md5(r_lightmap_caches[i].ptr());
								if (new_md5 < md5) {
									r_lightmap_caches.insert(i, lightmap_cache);
									break;
								}

								if (new_md5 == md5) {
									break;
								}
							}
						}
					}
				}

				if (generate_lods) {
					Array skin_pose_transform_array = _get_skinned_pose_transforms(src_mesh_node);
					src_mesh_node->get_mesh()->generate_lods(merge_angle, split_angle, skin_pose_transform_array);
				}

				if (create_shadow_meshes) {
					src_mesh_node->get_mesh()->create_shadow_mesh();
				}

				if (!save_to_file.is_empty()) {
					Ref<Mesh> existing = ResourceCache::get_ref(save_to_file);
					if (existing.is_valid()) {
						//if somehow an existing one is useful, create
						existing->reset_state();
					}
					mesh = src_mesh_node->get_mesh()->get_mesh(existing);

					ResourceSaver::save(mesh, save_to_file); //override

					mesh->set_path(save_to_file, true); //takeover existing, if needed

				} else {
					mesh = src_mesh_node->get_mesh()->get_mesh();
				}
			} else {
				mesh = src_mesh_node->get_mesh()->get_mesh();
			}

			if (mesh.is_valid()) {
				mesh_node->set_mesh(mesh);
				for (int i = 0; i < mesh->get_surface_count(); i++) {
					mesh_node->set_surface_override_material(i, src_mesh_node->get_surface_material(i));
				}
			}
		}

		switch (p_light_bake_mode) {
			case LIGHT_BAKE_DISABLED: {
				mesh_node->set_gi_mode(GeometryInstance3D::GI_MODE_DISABLED);
			} break;
			case LIGHT_BAKE_DYNAMIC: {
				mesh_node->set_gi_mode(GeometryInstance3D::GI_MODE_DYNAMIC);
			} break;
			case LIGHT_BAKE_STATIC:
			case LIGHT_BAKE_STATIC_LIGHTMAPS: {
				mesh_node->set_gi_mode(GeometryInstance3D::GI_MODE_STATIC);
			} break;
		}

		mesh_node->set_layer_mask(src_mesh_node->get_layer_mask());
		mesh_node->set_cast_shadows_setting(src_mesh_node->get_cast_shadows_setting());
		mesh_node->set_visibility_range_begin(src_mesh_node->get_visibility_range_begin());
		mesh_node->set_visibility_range_begin_margin(src_mesh_node->get_visibility_range_begin_margin());
		mesh_node->set_visibility_range_end(src_mesh_node->get_visibility_range_end());
		mesh_node->set_visibility_range_end_margin(src_mesh_node->get_visibility_range_end_margin());
		mesh_node->set_visibility_range_fade_mode(src_mesh_node->get_visibility_range_fade_mode());

		p_node->replace_by(mesh_node);
		p_node->set_owner(nullptr);
		memdelete(p_node);
		p_node = mesh_node;
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_generate_meshes(p_node->get_child(i), p_mesh_data, p_generate_lods, p_create_shadow_meshes, p_light_bake_mode, p_lightmap_texel_size, p_src_lightmap_cache, r_lightmap_caches);
	}

	return p_node;
}

void ResourceImporterScene::_add_shapes(Node *p_node, const Vector<Ref<Shape3D>> &p_shapes) {
	for (const Ref<Shape3D> &E : p_shapes) {
		CollisionShape3D *cshape = memnew(CollisionShape3D);
		cshape->set_shape(E);
		p_node->add_child(cshape, true);

		cshape->set_owner(p_node->get_owner());
	}
}

void ResourceImporterScene::_optimize_track_usage(AnimationPlayer *p_player, AnimationImportTracks *p_track_actions) {
	List<StringName> anims;
	p_player->get_animation_list(&anims);
	Node *parent = p_player->get_parent();
	ERR_FAIL_NULL(parent);
	HashMap<NodePath, uint32_t> used_tracks[TRACK_CHANNEL_MAX];
	bool tracks_to_add = false;
	static const Animation::TrackType track_types[TRACK_CHANNEL_MAX] = { Animation::TYPE_POSITION_3D, Animation::TYPE_ROTATION_3D, Animation::TYPE_SCALE_3D, Animation::TYPE_BLEND_SHAPE };
	for (const StringName &I : anims) {
		Ref<Animation> anim = p_player->get_animation(I);
		for (int i = 0; i < anim->get_track_count(); i++) {
			for (int j = 0; j < TRACK_CHANNEL_MAX; j++) {
				if (anim->track_get_type(i) != track_types[j]) {
					continue;
				}
				switch (p_track_actions[j]) {
					case ANIMATION_IMPORT_TRACKS_IF_PRESENT: {
						// Do Nothing.
					} break;
					case ANIMATION_IMPORT_TRACKS_IF_PRESENT_FOR_ALL: {
						used_tracks[j].insert(anim->track_get_path(i), 0);
						tracks_to_add = true;
					} break;
					case ANIMATION_IMPORT_TRACKS_NEVER: {
						anim->remove_track(i);
						i--;
					} break;
				}
			}
		}
	}

	if (!tracks_to_add) {
		return;
	}

	uint32_t pass = 0;
	for (const StringName &I : anims) {
		Ref<Animation> anim = p_player->get_animation(I);
		for (int j = 0; j < TRACK_CHANNEL_MAX; j++) {
			if (p_track_actions[j] != ANIMATION_IMPORT_TRACKS_IF_PRESENT_FOR_ALL) {
				continue;
			}

			pass++;

			for (int i = 0; i < anim->get_track_count(); i++) {
				if (anim->track_get_type(i) != track_types[j]) {
					continue;
				}

				NodePath path = anim->track_get_path(i);

				ERR_CONTINUE(!used_tracks[j].has(path)); // Should never happen.

				used_tracks[j][path] = pass;
			}

			for (const KeyValue<NodePath, uint32_t> &J : used_tracks[j]) {
				if (J.value == pass) {
					continue;
				}

				NodePath path = J.key;
				Node *n = parent->get_node(path);

				if (j == TRACK_CHANNEL_BLEND_SHAPE) {
					MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(n);
					if (mi && path.get_subname_count() > 0) {
						StringName bs = path.get_subname(0);
						bool valid;
						float value = mi->get(bs, &valid);
						if (valid) {
							int track_idx = anim->add_track(track_types[j]);
							anim->track_set_path(track_idx, path);
							anim->track_set_imported(track_idx, true);
							anim->blend_shape_track_insert_key(track_idx, 0, value);
						}
					}

				} else {
					Skeleton3D *skel = Object::cast_to<Skeleton3D>(n);
					Node3D *n3d = Object::cast_to<Node3D>(n);
					Vector3 loc;
					Quaternion rot;
					Vector3 scale;
					if (skel && path.get_subname_count() > 0) {
						StringName bone = path.get_subname(0);
						int bone_idx = skel->find_bone(bone);
						if (bone_idx == -1) {
							continue;
						}
						// Note that this is using get_bone_pose to update the bone pose cache.
						_ALLOW_DISCARD_ skel->get_bone_pose(bone_idx);
						loc = skel->get_bone_pose_position(bone_idx);
						rot = skel->get_bone_pose_rotation(bone_idx);
						scale = skel->get_bone_pose_scale(bone_idx);
					} else if (n3d) {
						loc = n3d->get_position();
						rot = n3d->get_transform().basis.get_rotation_quaternion();
						scale = n3d->get_scale();
					} else {
						continue;
					}

					// Ensure insertion keeps tracks together and ordered by type (loc/rot/scale)
					int insert_at_pos = -1;
					for (int k = 0; k < anim->get_track_count(); k++) {
						NodePath tpath = anim->track_get_path(k);

						if (path == tpath) {
							Animation::TrackType ttype = anim->track_get_type(k);
							if (insert_at_pos == -1) {
								// First insert, determine whether replacing or kicking back
								if (track_types[j] < ttype) {
									insert_at_pos = k;
									break; // No point in continuing.
								} else {
									insert_at_pos = k + 1;
								}
							} else if (ttype < track_types[j]) {
								// Kick back.
								insert_at_pos = k + 1;
							}
						} else if (insert_at_pos >= 0) {
							break;
						}
					}
					int track_idx = anim->add_track(track_types[j], insert_at_pos);

					anim->track_set_path(track_idx, path);
					anim->track_set_imported(track_idx, true);
					switch (j) {
						case TRACK_CHANNEL_POSITION: {
							anim->position_track_insert_key(track_idx, 0, loc);
						} break;
						case TRACK_CHANNEL_ROTATION: {
							anim->rotation_track_insert_key(track_idx, 0, rot);
						} break;
						case TRACK_CHANNEL_SCALE: {
							anim->scale_track_insert_key(track_idx, 0, scale);
						} break;
						default: {
						}
					}
				}
			}
		}
	}
}

Node *ResourceImporterScene::pre_import(const String &p_source_file, const HashMap<StringName, Variant> &p_options) {
	Ref<EditorSceneFormatImporter> importer;
	String ext = p_source_file.get_extension().to_lower();

	// TRANSLATORS: This is an editor progress label.
	EditorProgress progress("pre-import", TTR("Pre-Import Scene"), 0);
	progress.step(TTR("Importing Scene..."), 0);

	for (Ref<EditorSceneFormatImporter> importer_elem : scene_importers) {
		List<String> extensions;
		importer_elem->get_extensions(&extensions);

		for (const String &F : extensions) {
			if (F.to_lower() == ext) {
				importer = importer_elem;
				break;
			}
		}

		if (importer.is_valid()) {
			break;
		}
	}

	ERR_FAIL_COND_V(!importer.is_valid(), nullptr);

	Error err = OK;

	Node *scene = importer->import_scene(p_source_file, EditorSceneFormatImporter::IMPORT_ANIMATION | EditorSceneFormatImporter::IMPORT_GENERATE_TANGENT_ARRAYS | EditorSceneFormatImporter::IMPORT_FORCE_DISABLE_MESH_COMPRESSION, p_options, nullptr, &err);
	if (!scene || err != OK) {
		return nullptr;
	}

	HashMap<Ref<ImporterMesh>, Vector<Ref<Shape3D>>> collision_map;
	List<Pair<NodePath, Node *>> node_renames;
	_pre_fix_node(scene, scene, collision_map, nullptr, node_renames);

	return scene;
}

Error ResourceImporterScene::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	const String &src_path = p_source_file;

	Ref<EditorSceneFormatImporter> importer;
	String ext = src_path.get_extension().to_lower();

	EditorProgress progress("import", TTR("Import Scene"), 104);
	progress.step(TTR("Importing Scene..."), 0);

	for (Ref<EditorSceneFormatImporter> importer_elem : scene_importers) {
		List<String> extensions;
		importer_elem->get_extensions(&extensions);

		for (const String &F : extensions) {
			if (F.to_lower() == ext) {
				importer = importer_elem;
				break;
			}
		}

		if (importer.is_valid()) {
			break;
		}
	}

	ERR_FAIL_COND_V(!importer.is_valid(), ERR_FILE_UNRECOGNIZED);

	int import_flags = 0;

	if (animation_importer) {
		import_flags |= EditorSceneFormatImporter::IMPORT_ANIMATION;
		import_flags |= EditorSceneFormatImporter::IMPORT_DISCARD_MESHES_AND_MATERIALS;
	} else {
		if (bool(p_options["animation/import"])) {
			import_flags |= EditorSceneFormatImporter::IMPORT_ANIMATION;
		}
	}

	if (bool(p_options["skins/use_named_skins"])) {
		import_flags |= EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS;
	}

	bool ensure_tangents = p_options["meshes/ensure_tangents"];
	if (ensure_tangents) {
		import_flags |= EditorSceneFormatImporter::IMPORT_GENERATE_TANGENT_ARRAYS;
	}

	bool force_disable_compression = p_options["meshes/force_disable_compression"];
	if (force_disable_compression) {
		import_flags |= EditorSceneFormatImporter::IMPORT_FORCE_DISABLE_MESH_COMPRESSION;
	}

	Error err = OK;
	List<String> missing_deps; // for now, not much will be done with this
	Node *scene = importer->import_scene(src_path, import_flags, p_options, &missing_deps, &err);
	if (!scene || err != OK) {
		return err;
	}

	bool apply_root = true;
	if (p_options.has("nodes/apply_root_scale")) {
		apply_root = p_options["nodes/apply_root_scale"];
	}
	real_t root_scale = 1;
	if (p_options.has("nodes/root_scale")) {
		root_scale = p_options["nodes/root_scale"];
	}
	if (Object::cast_to<Node3D>(scene)) {
		Node3D *scene_3d = Object::cast_to<Node3D>(scene);
		Vector3 scale = Vector3(root_scale, root_scale, root_scale);
		if (apply_root) {
			_apply_permanent_scale_to_descendants(scene, scale);
		} else {
			scene_3d->scale(scale);
		}
	}
	Dictionary subresources = p_options["_subresources"];

	Dictionary node_data;
	if (subresources.has("nodes")) {
		node_data = subresources["nodes"];
	}

	Dictionary material_data;
	if (subresources.has("materials")) {
		material_data = subresources["materials"];
	}

	Dictionary animation_data;
	if (subresources.has("animations")) {
		animation_data = subresources["animations"];
	}

	HashSet<Ref<ImporterMesh>> scanned_meshes;
	HashMap<Ref<ImporterMesh>, Vector<Ref<Shape3D>>> collision_map;
	Pair<PackedVector3Array, PackedInt32Array> occluder_arrays;
	List<Pair<NodePath, Node *>> node_renames;

	_pre_fix_node(scene, scene, collision_map, &occluder_arrays, node_renames);

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		post_importer_plugins.write[i]->pre_process(scene, p_options);
	}

	float fps = 30;
	if (p_options.has(SNAME("animation/fps"))) {
		fps = (float)p_options[SNAME("animation/fps")];
	}
	_pre_fix_animations(scene, scene, node_data, animation_data, fps);
	_post_fix_node(scene, scene, collision_map, occluder_arrays, scanned_meshes, node_data, material_data, animation_data, fps, apply_root ? root_scale : 1.0);
	_post_fix_animations(scene, scene, node_data, animation_data, fps);

	String root_type = p_options["nodes/root_type"];
	if (!root_type.is_empty()) {
		root_type = root_type.split(" ")[0]; // Full root_type is "ClassName (filename.gd)" for a script global class.
		Ref<Script> root_script = nullptr;
		if (ScriptServer::is_global_class(root_type)) {
			root_script = ResourceLoader::load(ScriptServer::get_global_class_path(root_type));
			root_type = ScriptServer::get_global_class_base(root_type);
		}
		if (scene->get_class_name() != root_type) {
			// If the user specified a Godot node type that does not match
			// what the scene import gave us, replace the root node.
			Node *base_node = Object::cast_to<Node>(ClassDB::instantiate(root_type));
			if (base_node) {
				scene->replace_by(base_node);
				scene->set_owner(nullptr);
				memdelete(scene);
				scene = base_node;
			}
		}
		if (root_script.is_valid()) {
			scene->set_script(Variant(root_script));
		}
	}

	String root_name = p_options["nodes/root_name"];
	if (!root_name.is_empty() && root_name != "Scene Root") {
		// TODO: Remove `&& root_name != "Scene Root"` for Godot 5.0.
		// For backwards compatibility with existing .import files,
		// treat "Scene Root" as having no root name override.
		scene->set_name(root_name);
	} else if (String(scene->get_name()).is_empty()) {
		scene->set_name(p_save_path.get_file().get_basename());
	}

	if (!occluder_arrays.first.is_empty() && !occluder_arrays.second.is_empty()) {
		Ref<ArrayOccluder3D> occ = memnew(ArrayOccluder3D);
		occ->set_arrays(occluder_arrays.first, occluder_arrays.second);
		OccluderInstance3D *occluder_instance = memnew(OccluderInstance3D);
		occluder_instance->set_occluder(occ);
		scene->add_child(occluder_instance, true);
		occluder_instance->set_owner(scene);
	}

	bool gen_lods = bool(p_options["meshes/generate_lods"]);
	bool create_shadow_meshes = bool(p_options["meshes/create_shadow_meshes"]);
	int light_bake_mode = p_options["meshes/light_baking"];
	float texel_size = p_options["meshes/lightmap_texel_size"];
	float lightmap_texel_size = MAX(0.001, texel_size);

	Vector<uint8_t> src_lightmap_cache;
	Vector<Vector<uint8_t>> mesh_lightmap_caches;

	{
		src_lightmap_cache = FileAccess::get_file_as_bytes(p_source_file + ".unwrap_cache", &err);
		if (err != OK) {
			src_lightmap_cache.clear();
		}
	}

	Dictionary mesh_data;
	if (subresources.has("meshes")) {
		mesh_data = subresources["meshes"];
	}
	scene = _generate_meshes(scene, mesh_data, gen_lods, create_shadow_meshes, LightBakeMode(light_bake_mode), lightmap_texel_size, src_lightmap_cache, mesh_lightmap_caches);

	if (mesh_lightmap_caches.size()) {
		Ref<FileAccess> f = FileAccess::open(p_source_file + ".unwrap_cache", FileAccess::WRITE);
		if (f.is_valid()) {
			f->store_32(mesh_lightmap_caches.size());
			for (int i = 0; i < mesh_lightmap_caches.size(); i++) {
				String md5 = String::md5(mesh_lightmap_caches[i].ptr());
				f->store_buffer(mesh_lightmap_caches[i].ptr(), mesh_lightmap_caches[i].size());
			}
		}
	}
	err = OK;

	progress.step(TTR("Running Custom Script..."), 2);

	String post_import_script_path = p_options["import_script/path"];
	Ref<EditorScenePostImport> post_import_script;

	if (!post_import_script_path.is_empty()) {
		Ref<Script> scr = ResourceLoader::load(post_import_script_path);
		if (!scr.is_valid()) {
			EditorNode::add_io_error(TTR("Couldn't load post-import script:") + " " + post_import_script_path);
		} else {
			post_import_script = Ref<EditorScenePostImport>(memnew(EditorScenePostImport));
			post_import_script->set_script(scr);
			if (!post_import_script->get_script_instance()) {
				EditorNode::add_io_error(TTR("Invalid/broken script for post-import (check console):") + " " + post_import_script_path);
				post_import_script.unref();
				return ERR_CANT_CREATE;
			}
		}
	}

	if (post_import_script.is_valid()) {
		post_import_script->init(p_source_file);
		scene = post_import_script->post_import(scene);
		if (!scene) {
			EditorNode::add_io_error(
					TTR("Error running post-import script:") + " " + post_import_script_path + "\n" +
					TTR("Did you return a Node-derived object in the `_post_import()` method?"));
			return err;
		}
	}

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		post_importer_plugins.write[i]->post_process(scene, p_options);
	}

	progress.step(TTR("Saving..."), 104);

	int flags = 0;
	if (EDITOR_GET("filesystem/on_save/compress_binary_resources")) {
		flags |= ResourceSaver::FLAG_COMPRESS;
	}

	if (animation_importer) {
		Ref<AnimationLibrary> library;
		for (int i = 0; i < scene->get_child_count(); i++) {
			AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(scene->get_child(i));
			if (ap) {
				List<StringName> libs;
				ap->get_animation_library_list(&libs);
				if (libs.size()) {
					library = ap->get_animation_library(libs.front()->get());
					break;
				}
			}
		}

		if (!library.is_valid()) {
			library.instantiate(); // Will be empty
		}

		print_verbose("Saving animation to: " + p_save_path + ".res");
		err = ResourceSaver::save(library, p_save_path + ".res", flags); //do not take over, let the changed files reload themselves
		ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save animation to file '" + p_save_path + ".res'.");

	} else {
		Ref<PackedScene> packer = memnew(PackedScene);
		packer->pack(scene);
		print_verbose("Saving scene to: " + p_save_path + ".scn");
		err = ResourceSaver::save(packer, p_save_path + ".scn", flags); //do not take over, let the changed files reload themselves
		ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save scene to file '" + p_save_path + ".scn'.");
	}

	memdelete(scene);

	//this is not the time to reimport, wait until import process is done, import file is saved, etc.
	//EditorNode::get_singleton()->reload_scene(p_source_file);

	return OK;
}

ResourceImporterScene *ResourceImporterScene::scene_singleton = nullptr;
ResourceImporterScene *ResourceImporterScene::animation_singleton = nullptr;

Vector<Ref<EditorSceneFormatImporter>> ResourceImporterScene::scene_importers;
Vector<Ref<EditorScenePostImportPlugin>> ResourceImporterScene::post_importer_plugins;

bool ResourceImporterScene::has_advanced_options() const {
	return true;
}

void ResourceImporterScene::show_advanced_options(const String &p_path) {
	SceneImportSettingsDialog::get_singleton()->open_settings(p_path, animation_importer);
}

ResourceImporterScene::ResourceImporterScene(bool p_animation_import, bool p_singleton) {
	// This should only be set through the EditorNode.
	if (p_singleton) {
		if (p_animation_import) {
			animation_singleton = this;
		} else {
			scene_singleton = this;
		}
	}

	animation_importer = p_animation_import;
}

ResourceImporterScene::~ResourceImporterScene() {
	if (animation_singleton == this) {
		animation_singleton = nullptr;
	}
	if (scene_singleton == this) {
		scene_singleton = nullptr;
	}
}

void ResourceImporterScene::add_scene_importer(Ref<EditorSceneFormatImporter> p_importer, bool p_first_priority) {
	ERR_FAIL_COND(p_importer.is_null());
	if (p_first_priority) {
		scene_importers.insert(0, p_importer);
	} else {
		scene_importers.push_back(p_importer);
	}
}

void ResourceImporterScene::remove_post_importer_plugin(const Ref<EditorScenePostImportPlugin> &p_plugin) {
	post_importer_plugins.erase(p_plugin);
}

void ResourceImporterScene::add_post_importer_plugin(const Ref<EditorScenePostImportPlugin> &p_plugin, bool p_first_priority) {
	ERR_FAIL_COND(p_plugin.is_null());
	if (p_first_priority) {
		post_importer_plugins.insert(0, p_plugin);
	} else {
		post_importer_plugins.push_back(p_plugin);
	}
}

void ResourceImporterScene::remove_scene_importer(Ref<EditorSceneFormatImporter> p_importer) {
	scene_importers.erase(p_importer);
}

void ResourceImporterScene::clean_up_importer_plugins() {
	scene_importers.clear();
	post_importer_plugins.clear();
}

void ResourceImporterScene::get_scene_importer_extensions(List<String> *p_extensions) {
	for (Ref<EditorSceneFormatImporter> importer_elem : scene_importers) {
		importer_elem->get_extensions(p_extensions);
	}
}

///////////////////////////////////////

static const char *token_to_str[ShaderLanguage::TK_MAX] = {
	"", // TK_EMPTY
	"", // TK_IDENTIFIER
	"true",
	"false",
	"", // TK_FLOAT_CONSTANT
	"", // TK_INT_CONSTANT
	"", // TK_UINT_CONSTANT
	"void",
	"bool",
	"bvec2",
	"bvec3",
	"bvec4",
	"int",
	"ivec2",
	"ivec3",
	"ivec4",
	"uint",
	"uvec2",
	"uvec3",
	"uvec4",
	"float",
	"vec2",
	"vec3",
	"vec4",
	"mat2",
	"mat3",
	"mat4",
	"sampler2D",
	"isampler2D",
	"usampler2D",
	"sampler2DArray",
	"isampler2DArray",
	"usampler2DArray",
	"sampler3D",
	"isampler3D",
	"usampler3D",
	"samplerCube",
	"samplerCubeArray",
	"flat",
	"smooth",
	"const",
	"struct",
	"lowp",
	"mediump",
	"highp",
	"==",
	"!=",
	"<",
	"<=",
	">",
	">=",
	"&&",
	"||",
	"!",
	"+",
	"-",
	"*",
	"/",
	"%",
	"<<",
	">>",
	"=",
	"+=",
	"-=",
	"*=",
	"/=",
	"%=",
	"<<=",
	">>=",
	"&=",
	"|=",
	"^=",
	"&",
	"|",
	"^",
	"~",
	"++",
	"--",
	"if",
	"else",
	"for",
	"while",
	"do",
	"switch",
	"case",
	"default",
	"break",
	"continue",
	"return",
	"discard",
	"[",
	"]",
	"{",
	"}",
	"(",
	")",
	"?",
	",",
	":",
	";",
	".",
	"uniform",
	"group_uniforms",
	"instance",
	"global",
	"varying",
	"in",
	"out",
	"inout",
	"render_mode",
	"hint_default_white",
	"hint_default_black",
	"hint_default_transparent",
	"hint_normal",
	"hint_roughness_normal",
	"hint_roughness_r",
	"hint_roughness_g",
	"hint_roughness_b",
	"hint_roughness_a",
	"hint_roughness_gray",
	"hint_anisotropy",
	"source_color",
	"hint_range",
	"instance_index",
	"hint_screen_texture",
	"hint_normal_roughness_texture",
	"hint_depth_texture",
	"filter_nearest",
	"filter_linear",
	"filter_nearest_mipmap",
	"filter_linear_mipmap",
	"filter_nearest_mipmap_anisotropic",
	"filter_linear_mipmap_anisotropic",
	"repeat_enable",
	"repeat_disable",
	"shader_type",
	"", // TK_CURSOR
	"", // TK_ERROR
	"", // TK_EOF
	"\t",
	"\r",
	" ",
	"\n",
	"", // TK_BLOCK_COMMENT
	"", // TK_LINE_COMMENT
	"", // TK_PREPROC_DIRECTIVE
};

class TokenStreamManip {
	using TokenType = ShaderLanguage::TokenType;
	using Token = ShaderLanguage::Token;
	using TT = TokenType;

	List<Token> code_tokens;
	List<Token>::Element *curr_ptr;
	const String old_code;
	bool first = false;
	Token eof_token{ ShaderLanguage::TK_EOF, {}, 0, 0, 0, 0 };
	bool token_is_skippable(const Token &tk) const {
		switch (tk.type) {
			case ShaderLanguage::TK_TAB:
			case ShaderLanguage::TK_CR:
			case ShaderLanguage::TK_SPACE:
			case ShaderLanguage::TK_NEWLINE:
			case ShaderLanguage::TK_BLOCK_COMMENT:
			case ShaderLanguage::TK_LINE_COMMENT:
			case ShaderLanguage::TK_PREPROC_DIRECTIVE:
				return true;
			default:
				break;
		}
		return false;
	};
	List<Token>::Element *_get_next_token_ptr(List<Token>::Element *_curr_ptr, bool skip_first = false) const {
		ERR_FAIL_COND_V(_curr_ptr == nullptr, _curr_ptr);
		if (_curr_ptr->next() == nullptr) {
			return _curr_ptr;
		}
		_curr_ptr = _curr_ptr->next();
		while (token_is_skippable(_curr_ptr->get())) {
			if (_curr_ptr->next() == nullptr) {
				return _curr_ptr;
			}
			_curr_ptr = _curr_ptr->next();
		}
		return _curr_ptr;
	};
	List<Token>::Element *_get_prev_token_ptr(List<Token>::Element *_curr_ptr) const {
		ERR_FAIL_COND_V(_curr_ptr == nullptr, _curr_ptr);
		if (_curr_ptr->prev() == nullptr) {
			return _curr_ptr;
		}
		_curr_ptr = _curr_ptr->prev();
		while (token_is_skippable(_curr_ptr->get())) {
			if (_curr_ptr->prev() == nullptr) {
				return _curr_ptr;
			}
			_curr_ptr = _curr_ptr->prev();
		}
		return _curr_ptr;
	};
	List<Token>::Element *get_next_token() {
		curr_ptr = _get_next_token_ptr(curr_ptr, first);
		return curr_ptr;
	};
	List<Token>::Element *get_prev_token() {
		curr_ptr = _get_prev_token_ptr(curr_ptr);
		return curr_ptr;
	};
	List<Token>::Element *remove_cur_and_get_next() {
		ERR_FAIL_COND_V(!curr_ptr, nullptr);
		List<Token>::Element *prev = curr_ptr->prev();
		if (!prev) {
			prev = curr_ptr->next();
			code_tokens.erase(curr_ptr);
			while (token_is_skippable(prev->get())) {
				if (prev->next() == nullptr) {
					return prev;
				}
				prev = prev->next();
			}
			return prev;
		}
		code_tokens.erase(curr_ptr);
		curr_ptr = prev;
		return get_next_token();
	};
	TokenType _peek_tk_type(int64_t count, List<Token>::Element **r_pos = nullptr) const {
		ERR_FAIL_COND_V(!curr_ptr, ShaderLanguage::TK_EOF);
		if (count == 0) {
			return curr_ptr->get().type;
		}

		bool backwards = count < 0;
		uint64_t max_count = abs(count);
		auto start_ptr = curr_ptr;
		for (int i = 0; i < max_count; i++) {
			auto _ptr = backwards ? _get_prev_token_ptr(start_ptr) : _get_next_token_ptr(start_ptr);
			if (!_ptr) {
				if (r_pos) {
					*r_pos = start_ptr;
				}
				return ShaderLanguage::TK_EOF;
			}
			start_ptr = _ptr;
		}
		if (r_pos) {
			*r_pos = start_ptr;
		}
		return start_ptr->get().type;
	}
	TokenType peek_next_tk_type(uint32_t count = 1) const {
		return _peek_tk_type(count);
	};
	TokenType peek_prev_tk_type(uint32_t count = 1) const {
		return _peek_tk_type(-((int64_t)count));
	};
	List<Token>::Element *get_pos() const {
		ERR_FAIL_COND_V(!curr_ptr, nullptr);
		return curr_ptr;
	};
	enum {
		NEW_IDENT = -1
	};
	bool reset_to(List<Token>::Element *p_pos) {
		ERR_FAIL_COND_V(p_pos == nullptr, false);
		curr_ptr = p_pos;
		return true;
	};
	bool insert_after(const Vector<Token> &token_list, List<Token>::Element *p_pos) {
		ERR_FAIL_COND_V(p_pos == nullptr, false);
		for (int i = token_list.size() - 1; i >= 0; i--) {
			const Token &tk = token_list[i];
			code_tokens.insert_after(p_pos, { tk.type, tk.text, tk.constant, tk.line, tk.length, NEW_IDENT });
		}
		return true;
	};
	bool insert_before(const Vector<Token> &token_list, List<Token>::Element *p_pos) {
		ERR_FAIL_COND_V(p_pos == nullptr, false);
		for (const Token &tk : token_list) {
			code_tokens.insert_before(p_pos, { tk.type, tk.text, tk.constant, tk.line, tk.length, NEW_IDENT });
		}
		return true;
	};
	bool insert_after(const Token &token, List<Token>::Element *p_pos) {
		ERR_FAIL_COND_V(p_pos == nullptr, false);
		Token new_token = token;
		new_token.pos = NEW_IDENT;
		code_tokens.insert_after(p_pos, new_token);
		return true;
	};
	bool insert_before(const Token &token, List<Token>::Element *p_pos) {
		ERR_FAIL_COND_V(p_pos == nullptr, false);
		Token new_token = token;
		new_token.pos = NEW_IDENT;
		code_tokens.insert_before(p_pos, new_token);
		return true;
	};
	List<Token>::Element *replace_curr(const Token &token) {
		ERR_FAIL_COND_V(curr_ptr == nullptr, nullptr);
		Token new_token = token;
		new_token.pos = NEW_IDENT;
		List<Token>::Element *prev = curr_ptr;
		curr_ptr = code_tokens.insert_before(curr_ptr, new_token);
		code_tokens.erase(prev);
		return curr_ptr;
	};

	bool _insert_uniform_declaration(const String &p_name, TokenType hint, List<Token>::Element *p_shader_decl_end_pos) {
		if (p_shader_decl_end_pos == nullptr) {
			return false;
		}
		//	"\nuniform sampler2D %s : hint_%s, filter_linear_mipmap;\n";
		return insert_after({ { TT::TK_NEWLINE }, { TT::TK_UNIFORM }, { TT::TK_SPACE }, { TT::TK_TYPE_SAMPLER2D },
									{ TT::TK_SPACE }, { TT::TK_IDENTIFIER, p_name }, { TT::TK_SPACE }, { TT::TK_COLON },
									{ TT::TK_SPACE }, { hint }, { TT::TK_COMMA }, { TT::TK_SPACE },
									{ TT::TK_FILTER_LINEAR_MIPMAP }, { TT::TK_SEMICOLON },
									{ TT::TK_NEWLINE } },
				p_shader_decl_end_pos);
	}
	List<Token>::Element *_remove_from_curr_to(List<Token>::Element *p_end) {
		ERR_FAIL_COND_V(p_end == nullptr, nullptr);
		while (curr_ptr != p_end) {
			auto next = curr_ptr->next();
			code_tokens.erase(curr_ptr);
			curr_ptr = next;
		}
		return curr_ptr;
	}

	List<Token>::Element *_get_end_of_closure() {
		int additional_closures = 0;
		int iters = 0;
		bool found = false;
		List<Token>::Element *ptr = curr_ptr;
		for (; ptr; ptr = ptr->next()) {
			switch (ptr->get().type) {
				case TT::TK_CURLY_BRACKET_OPEN:
				case TT::TK_PARENTHESIS_OPEN:
				case TT::TK_BRACKET_OPEN: {
					additional_closures++;
				} break;
				case TT::TK_CURLY_BRACKET_CLOSE:
				case TT::TK_PARENTHESIS_CLOSE:
				case TT::TK_BRACKET_CLOSE: {
					if (additional_closures != 0) {
						additional_closures--;
						break;
					}
					return ptr;
				} break;
				case TT::TK_SEMICOLON: {
					return _get_prev_token_ptr(ptr);
				} break;
				case TT::TK_EOF:
				case TT::TK_ERROR: {
					ptr = _get_prev_token_ptr(ptr);
					ERR_FAIL_V(ptr);
				} break;
				default:
					break;
			}
		}
		return ptr;
	}

public:
	bool convert_code(String &r_err_str) {
		/**
		 * We need to do the following:
		 *  * Replace everything in RenamesMap3To4::shaders_renames
		 *	* the usage of SCREEN_TEXTURE, DEPTH_TEXTURE, and NORMAL_ROUGHNESS_TEXTURE necessitates adding a uniform declaration at the top of the file
		 *	* async_visible and async_hidden render modes need to be removed
		 *	* If shader_type is "particles", need to rename the function "void vertex()" to "void process()"
		 *  * Invert all usages of CLEARCOAT_GLOSS
		 *    * invert all lefthand assignments
		 * 			- `CLEARCOAT_GLOSS = 5.0 / foo;`
		 * 			becomes: `CLEARCOAT_ROUGHNESS = (1.0 - (5.0 / foo));`,
		 *          - `CLEARCOAT_GLOSS *= 1.1;`
		 * 			becomes `CLEARCOAT_ROUGHNESS = (1.0 - ((1.0 - CLEARCOAT_ROUGHNESS) * 1.1));`
		 *    * invert all righthand usages
		 * 			- `foo = CLEARCOAT_GLOSS;`
		 * 			becomes: `foo = (1.0 - CLEARCOAT_ROUGHNESS);`
		 *	* Check for use of `specular_blinn` and `specular_phong` render modes; not supported in 4.x, throw an error
		 *	* Check for use of `MODULATE`; not supported in 4.x, throw an error
		 */
#define SDCONV_COND_FAIL(cond, msg)                                 \
	if (unlikely(cond)) {                                           \
		r_err_str = "3.x Shader conversion failed: " + String(msg); \
		return false;                                               \
	}
#define SDCONV_COND_LINE_FAIL(cond, line, msg)                                               \
	if (unlikely(cond)) {                                                                    \
		r_err_str = "3.x Shader conversion failed: Line " + itos(line) + ": " + String(msg); \
		return false;                                                                        \
	}
#define SDCONV_LINE_FAIL(line, msg)                                                      \
	r_err_str = "3.x Shader conversion failed: Line " + itos(line) + ": " + String(msg); \
	return false;
		SDCONV_COND_FAIL(code_tokens.size() == 0, "Empty shader file");
		if (code_tokens.back()->get().type != TT::TK_EOF) {
			SDCONV_COND_LINE_FAIL(code_tokens.back()->get().type == TT::TK_ERROR, code_tokens.back()->get().line, "Parser error (" + code_tokens.back()->get().text + ")");
			code_tokens.push_back(eof_token);
		}
		// TK_SHADER_TYPE, TK_IDENTIFIER, TK_SEMICOLON are always the first three tokens in a 3.x shader file
		curr_ptr = code_tokens.front();

		String shader_type;
		RenderingServer::ShaderMode shader_mode;
		{
			SDCONV_COND_FAIL(code_tokens.size() < 3, "Invalid shader file");
			auto first_token = get_next_token();
			SDCONV_COND_LINE_FAIL(first_token->get().type != TT::TK_SHADER_TYPE, first_token->get().line, "Shader type must be first token");
			auto id_token = get_next_token();
			SDCONV_COND_LINE_FAIL(id_token->get().type != TT::TK_IDENTIFIER, id_token->get().line, "Invalid shader type");
			String shader_type = id_token->get().text;
			auto token = get_next_token();
			SDCONV_COND_LINE_FAIL(token->get().type != TT::TK_SEMICOLON, token->get().line, "Expected semi-colon after shader type");
			if (shader_type == "spatial") {
				shader_mode = RenderingServer::ShaderMode::SHADER_SPATIAL;
			} else if (shader_type == "particles") {
				shader_mode = RenderingServer::ShaderMode::SHADER_PARTICLES;
			} else if (shader_type == "canvas_item") {
				shader_mode = RenderingServer::ShaderMode::SHADER_CANVAS_ITEM;
			} else { // 3.x didn't support any other shader types
				SDCONV_LINE_FAIL(id_token->get().line, "Invalid 3.x shader type");
			}
		}
		List<Token>::Element *after_type_decl = get_pos();

		HashMap<String, String> renames;
		HashMap<String, Token> hint_renames;
		for (unsigned int current_index = 0; RenamesMap3To4::shaders_renames[current_index][0]; current_index++) {
			String old_name = RenamesMap3To4::shaders_renames[current_index][0];
			if (old_name.begins_with("hint_")) {
				if (old_name == "hint_albedo") {
					hint_renames.insert(old_name, Token{ TT::TK_HINT_SOURCE_COLOR, {}, 0, 0, 0, 0 });
				} else if (old_name == "hint_aniso") {
					hint_renames.insert(old_name, Token{ TT::TK_HINT_ANISOTROPY_TEXTURE, {}, 0, 0, 0, 0 });
				} else if (old_name == "hint_black") {
					hint_renames.insert(old_name, Token{ TT::TK_HINT_DEFAULT_BLACK_TEXTURE, {}, 0, 0, 0, 0 });
				} else if (old_name == "hint_black_albedo") {
					hint_renames.insert(old_name, Token{ TT::TK_HINT_DEFAULT_BLACK_TEXTURE, {}, 0, 0, 0, 0 });
				} else if (old_name == "hint_color") {
					hint_renames.insert(old_name, Token{ TT::TK_HINT_SOURCE_COLOR, {}, 0, 0, 0, 0 });
				} else if (old_name == "hint_white") {
					hint_renames.insert(old_name, Token{ TT::TK_HINT_DEFAULT_WHITE_TEXTURE, {}, 0, 0, 0, 0 });
				} else { // this shouldn't ever happen
					r_err_str = "No hint rename!?!?!?";
					ERR_FAIL_V_MSG(false, r_err_str);
				}
			} else {
				renames.insert(RenamesMap3To4::shaders_renames[current_index][0], RenamesMap3To4::shaders_renames[current_index][1]);
			}
		}
		bool has_screen_texture = false;
		bool has_depth_texture = false;
		bool has_roughness_texture = false;

		while (true) {
			auto cur_tok = get_next_token();
			if (cur_tok->get().type == TT::TK_EOF) {
				break;
			}
			switch (cur_tok->get().type) {
				case TT::TK_RENDER_MODE: {
					// we only care about the ones for spatial
					if (shader_mode == RenderingServer::ShaderMode::SHADER_SPATIAL) {
						while (true) {
							auto next_tk = get_next_token();
							if (next_tk->get().type == TT::TK_IDENTIFIER) {
								SDCONV_COND_LINE_FAIL(next_tk->get().text == "specular_blinn" || next_tk->get().text == "specular_phong", next_tk->get().line, "render mode" + next_tk->get().text + "is not supported by this version of Godot.");
								if (next_tk->get().text == "async_visible" || next_tk->get().text == "async_hidden") {
									next_tk = remove_cur_and_get_next();
									if (next_tk->get().type == TT::TK_COMMA) {
										next_tk = remove_cur_and_get_next();
									} else {
										if (peek_prev_tk_type() == TT::TK_RENDER_MODE && next_tk->get().type == TT::TK_SEMICOLON) {
											// we need to remove this line entirely
											auto end = get_pos()->next();
											next_tk = get_prev_token();
											next_tk = _remove_from_curr_to(end);
											break;
										}
									}
								}
							} else {
								SDCONV_COND_LINE_FAIL(next_tk->get().type != TT::TK_COMMA && next_tk->get().type != TT::TK_SEMICOLON, next_tk->get().line, "Invalid render mode declaration");
							}
							if (next_tk->get().type == TT::TK_SEMICOLON) {
								break;
							}
						}
					}
				} break;
				case TT::TK_IDENTIFIER: {
					if (cur_tok->get().text == "SCREEN_TEXTURE" && !has_screen_texture) {
						has_screen_texture = true;
						SDCONV_COND_LINE_FAIL(!_insert_uniform_declaration("SCREEN_TEXTURE", TT::TK_HINT_SCREEN_TEXTURE, after_type_decl), cur_tok->get().line, "Failed to insert uniform declaration");
					} else if (cur_tok->get().text == "DEPTH_TEXTURE" && !has_depth_texture) {
						has_depth_texture = true;
						SDCONV_COND_LINE_FAIL(!_insert_uniform_declaration("DEPTH_TEXTURE", TT::TK_HINT_DEPTH_TEXTURE, after_type_decl), cur_tok->get().line, "Failed to insert uniform declaration");
					} else if (cur_tok->get().text == "NORMAL_ROUGHNESS_TEXTURE" && !has_roughness_texture) {
						has_roughness_texture = true;
						SDCONV_COND_LINE_FAIL(!_insert_uniform_declaration("NORMAL_ROUGHNESS_TEXTURE", TT::TK_HINT_NORMAL_ROUGHNESS_TEXTURE, after_type_decl), cur_tok->get().line, "Failed to insert uniform declaration");
					} else if (shader_mode == RenderingServer::ShaderMode::SHADER_PARTICLES && cur_tok->get().text == "vertex") {
						if (peek_prev_tk_type() == TT::TK_TYPE_VOID || peek_next_tk_type() == TT::TK_PARENTHESIS_OPEN) {
							replace_curr({ TT::TK_IDENTIFIER, "process" });
						}
					} else if (shader_mode == RenderingServer::ShaderMode::SHADER_CANVAS_ITEM && cur_tok->get().text == "MODULATE") {
						// This is not supported in Godot 4.x (yet, may be re-added).
						SDCONV_LINE_FAIL(cur_tok->get().line, "MODULATE is not supported by this version of Godot")
					} else if (cur_tok->get().text == "CLEARCOAT_GLOSS") {
						cur_tok = replace_curr({ TT::TK_IDENTIFIER, "CLEARCOAT_ROUGHNESS" });
						Token end_token;
						List<Token>::Element *assign_closure_end = nullptr;
						switch (peek_next_tk_type()) {
							case TT::TK_OP_ASSIGN:
							case TT::TK_OP_ASSIGN_ADD:
							case TT::TK_OP_ASSIGN_SUB:
							case TT::TK_OP_ASSIGN_MUL:
							case TT::TK_OP_ASSIGN_DIV: {
								assign_closure_end = _get_end_of_closure();
								{
									auto assign_tk = get_next_token();
									// " = (1.0 - ("
									Vector<Token> pending_closures = {
										{ TT::TK_OP_ASSIGN },
										{ TT::TK_SPACE },
										{ TT::TK_PARENTHESIS_OPEN },
										{ TT::TK_FLOAT_CONSTANT, {}, 1.0 },
										{ TT::TK_SPACE },
										{ TT::TK_OP_SUB },
										{ TT::TK_SPACE },
										{ TT::TK_PARENTHESIS_OPEN },
									};
									if (assign_tk->get().type != TT::TK_OP_ASSIGN) {
										// " = (1.0 - ((1.0 - CLEARCOAT_ROUGHNESS) {op}
										pending_closures.append_array(
												{ { TT::TK_PARENTHESIS_OPEN },
														{ TT::TK_FLOAT_CONSTANT, {}, 1.0 },
														{ TT::TK_SPACE },
														{ TT::TK_OP_SUB },
														{ TT::TK_SPACE },
														{ TT::TK_IDENTIFIER, "CLEARCOAT_ROUGHNESS" },
														{ TT::TK_PARENTHESIS_CLOSE },
														{ TT::TK_SPACE } });
									}
									switch (assign_tk->get().type) {
										case TT::TK_OP_ASSIGN_ADD: {
											pending_closures.push_back({ TT::TK_OP_ADD });
										} break;
										case TT::TK_OP_ASSIGN_SUB: {
											pending_closures.push_back({ TT::TK_OP_SUB });
										} break;
										case TT::TK_OP_ASSIGN_MUL: {
											pending_closures.push_back({ TT::TK_OP_MUL });
										} break;
										case TT::TK_OP_ASSIGN_DIV: {
											pending_closures.push_back({ TT::TK_OP_DIV });
										} break;
										default:
											break;
									}
									insert_before(pending_closures, assign_tk);
								}
								remove_cur_and_get_next();
								insert_after({ { TT::TK_PARENTHESIS_CLOSE }, { TT::TK_PARENTHESIS_CLOSE } }, assign_closure_end);
								reset_to(cur_tok);

							} break;
							default:
								break;
						}

						// now we need to check the previous token
						// if this is anything but a `{` or `;`, we need to invert it
						if (peek_prev_tk_type() == TT::TK_SEMICOLON || peek_prev_tk_type() == TT::TK_CURLY_BRACKET_OPEN) {
							break;
						}
						Vector<Token> pending_closures = {
							{ TT::TK_PARENTHESIS_OPEN },
							{ TT::TK_FLOAT_CONSTANT, {}, 1.0 },
							{ TT::TK_SPACE },
							{ TT::TK_OP_SUB },
							{ TT::TK_SPACE }
						};
						if (assign_closure_end) {
							// invert_str = "(1.0 - (" + assign_str;
							pending_closures.append_array({ { TT::TK_PARENTHESIS_OPEN }, { TT::TK_SPACE } });
							insert_after({ { TT::TK_PARENTHESIS_CLOSE }, { TT::TK_PARENTHESIS_CLOSE } }, assign_closure_end);
						} else {
							insert_after({ TT::TK_PARENTHESIS_CLOSE }, cur_tok);
						}
						insert_before(pending_closures, cur_tok);
					} else if (renames.has(cur_tok->get().text)) {
						replace_curr({ TT::TK_IDENTIFIER, renames[cur_tok->get().text] });
					} else if (hint_renames.has(cur_tok->get().text)) {
						replace_curr(hint_renames[cur_tok->get().text]);
					}
				} break; // end of identifier case
				case TT::TK_ERROR: {
					SDCONV_LINE_FAIL(cur_tok->get().line, "Parser error ( " + cur_tok->get().text + ")");
				} break;
				default:
					break;
			}
		}
		return true;
	}

	String emit_code() const {
		if (code_tokens.size() == 0) {
			return "";
		}
		String new_code = "";
		const List<ShaderLanguage::Token>::Element *start = code_tokens.front()->next(); // skip TK_EOF token at start
		for (auto E = start; E; E = E->next()) {
			const Token &tk = E->get();
			ERR_FAIL_COND_V(tk.type < 0 || tk.type > TT::TK_MAX, "");
			bool end = false;
			switch (tk.type) {
				// remember that we can't trust the tk.name unless it's a newly inserted token
				// same with constants
				case TT::TK_PREPROC_DIRECTIVE:
				case TT::TK_LINE_COMMENT:
				case TT::TK_BLOCK_COMMENT:
				case TT::TK_IDENTIFIER: {
					if (tk.pos == NEW_IDENT) {
						new_code += tk.text;
					} else {
						// don't trust the token text because it may have been modified by the ShaderLanguage parser
						new_code += old_code.substr(tk.pos, tk.length);
					}
				} break;
				case TT::TK_INT_CONSTANT:
				case TT::TK_FLOAT_CONSTANT:
				case TT::TK_UINT_CONSTANT: {
					if (tk.pos == NEW_IDENT) {
						String const_str = rtos(tk.constant);
						if (!tk.is_integer_constant() && !const_str.contains(".")) {
							const_str += ".0";
						}
						new_code += const_str;
					} else {
						new_code += old_code.substr(tk.pos, tk.length);
					}
				} break;
				case TT::TK_ERROR:
				case TT::TK_EOF: {
					end = true;
					new_code += "";
				} break;

				default: {
					new_code += token_to_str[tk.type];
				} break;
			}
			if (end) {
				break;
			}
		}

		return new_code;
	}

	TokenStreamManip() = delete;
	TokenStreamManip(const String &p_code) :
			old_code(p_code) {
		ShaderLanguage sl;
		sl.token_debug_stream(old_code, code_tokens, true);
		code_tokens.push_back(eof_token);
		code_tokens.push_front(eof_token);
		curr_ptr = code_tokens.front();
	}
};

String EditorSceneFormatImporterESCN::convert_old_shader_code(const String &p_code, String &r_err_str) {
	// TODO: remove this
	TokenStreamManip tsm(p_code);
	if (!tsm.convert_code(r_err_str)) {
		return String();
	}
	return tsm.emit_code();
}

int _get_starting_index(const List<PropertyInfo> &p_properties) {
	// we need to get past MissingResource's properties
	bool fmissingResource = false;
	bool foundoriginal_class = false;
	bool found_recording_property = false;
	for (int i = 0; i < p_properties.size(); i++) {
		if (p_properties[i].usage & PROPERTY_USAGE_CATEGORY && p_properties[i].name == "MissingResource") {
			fmissingResource = true;
		} else if (fmissingResource) {
			if (p_properties[i].name == "original_class") {
				foundoriginal_class = true;
			}
			if (p_properties[i].name == "recording_properties") {
				found_recording_property = true;
			}
		}
		if (foundoriginal_class && found_recording_property) {
			return i + 1;
		}
	}
	return 0;
}

Ref<Resource> EditorSceneFormatImporterESCN::convert_old_shader(const Ref<MissingResource> &p_res, Error &r_err, String &r_err_str) {
	// Shader conversion
	// 3.x shaders will fail to compile with the 4.x compiler
	// This is done upon `set_code()` during resource load, and this will result in errors when loading normally
	Ref<Shader> shader;
	String code;
	List<PropertyInfo> missingres_properties;
	shader.instantiate();
	// get the props
	p_res->get_property_list(&missingres_properties);
	int start_idx = _get_starting_index(missingres_properties);
	// set resource_local_to_scene and resource_name; resource_path gets set by the resource loader
	shader->set("resource_local_to_scene", p_res->get("resource_local_to_scene"));
	shader->set("resource_name", p_res->get("resource_name"));
	for (int i = start_idx; i < missingres_properties.size(); i++) {
		const PropertyInfo &prop = missingres_properties[i];
		if (prop.name == "code") {
			code = p_res->get(prop.name);
		} else {
			shader->set(prop.name, p_res->get(prop.name));
		}
	}
	// String new_code = convert_old_shader_code(code, r_err_str);
	TokenStreamManip tsm(code);
	if (!tsm.convert_code(r_err_str)) {
		r_err = ERR_FILE_CORRUPT;
		return Ref<Resource>();
	}
	// sl.token_debug_stream(new_code, new_code_tokens);
	shader->set_code(tsm.emit_code());
	r_err = OK;
	return shader;
}

Ref<Resource> EditorSceneFormatImporterESCN::convert_old_animation(const Ref<MissingResource> &p_res, Error &r_err, String &r_err_str) {
	// Converts old scene animation format
	// `transform` track type was removed in 4.x and will result in loading errors if loaded normally
	// We need to convert any `transform` tracks to separate position, rotation, and scale tracks
	List<PropertyInfo> missingres_properties;
	Vector<Dictionary> tracks;
	Ref<Animation> animation;

	animation.instantiate();
	p_res->get_property_list(&missingres_properties);
	// ClassDB::get_property_list("Animation", &animation_properties); // Animation is derived from `Resource`, so no inheritance

	int start_idx = _get_starting_index(missingres_properties);
	// Set resource properties
	// Set resource_local_to_scene and resource_name; resource_path gets set by the resource loader because it pollutes the cache.
	animation->set("resource_local_to_scene", p_res->get("resource_local_to_scene"));
	animation->set("resource_name", p_res->get("resource_name"));
	// Set recorded properties.
	for (int i = start_idx; i < missingres_properties.size(); i++) {
		const PropertyInfo &prop = missingres_properties[i];
		if (prop.name.begins_with("tracks/")) {
			int id = prop.name.get_slicec('/', 1).to_int();
			while (id >= tracks.size()) {
				tracks.push_back(Dictionary());
			}
			tracks.write[id][prop.name.get_slicec('/', 2)] = p_res->get(prop.name);
		} else {
			animation->set(prop.name, p_res->get(prop.name));
		}
	}
	for (int i = 0; i < tracks.size(); i++) {
		// now that we have all the tracks, we need to split the transform tracks into separate tracks
		// this is because the current animation player doesn't support transform tracks
		// so we need to split them into separate position, rotation, and scale tracks
		// TODO: we also need to split the blend shape tracks into separate tracks
		if (tracks[i].has("type") && tracks[i]["type"] == "transform") {
			// split the transform track into separate tracks

			// Old scene format only used 32-bit floats, did not have configurable real_t.
			Vector<float> keys = tracks[i]["keys"];
			int vcount = keys.size();
			int tcount = vcount / 12;
			if ((vcount % 12) != 0) { // should be multiple of 12
				r_err_str = "Failed to convert animation: invalid number of keys in transform track.";
				r_err = ERR_FILE_CORRUPT;
				ERR_FAIL_V(Ref<Resource>());
			}

			Vector<real_t> position_keys;
			Vector<real_t> rotation_keys;
			Vector<real_t> scale_keys;
			position_keys.resize(tcount * 5); // time + transition + xyz
			rotation_keys.resize(tcount * 6); // time + transition + xyzw
			scale_keys.resize(tcount * 5); // time + transition + xyz
			// split the keys into separate tracks
			for (int j = 0; j < tcount; j++) {
				// it's position (Vector3, xyz), then rotation (Quaternion, xyzw), then scale (Vector3, xyz)
				// each track has time and transition values, so get those
				const float *ofs = &(keys.ptr()[j * 12]);
				float time = ofs[0];
				float transition = ofs[1];

				position_keys.write[j * 5 + 0] = time;
				position_keys.write[j * 5 + 1] = transition;
				position_keys.write[j * 5 + 2] = ofs[2]; // x
				position_keys.write[j * 5 + 3] = ofs[3]; // y
				position_keys.write[j * 5 + 4] = ofs[4]; // z

				rotation_keys.write[j * 6 + 0] = time;
				rotation_keys.write[j * 6 + 1] = transition;
				rotation_keys.write[j * 6 + 2] = ofs[5]; // x
				rotation_keys.write[j * 6 + 3] = ofs[6]; // y
				rotation_keys.write[j * 6 + 4] = ofs[7]; // z
				rotation_keys.write[j * 6 + 5] = ofs[8]; // w

				scale_keys.write[j * 5 + 0] = time;
				scale_keys.write[j * 5 + 1] = transition;
				scale_keys.write[j * 5 + 2] = ofs[9]; // x
				scale_keys.write[j * 5 + 3] = ofs[10]; // y
				scale_keys.write[j * 5 + 4] = ofs[11]; // z
			}

			Dictionary c_track;
			{
				auto dict_keys = tracks[i].keys();
				for (int j = 0; j < dict_keys.size(); j++) {
					if (dict_keys[j] == "type" || dict_keys[j] == "keys") {
						continue;
					}
					c_track[dict_keys[j]] = tracks[i][dict_keys[j]];
				}
			}
			auto c_track_keys = c_track.keys();
			tracks.remove_at(i);
			// scale, then rotation, then position
			if (scale_keys.size() > 0) {
				Dictionary track;
				track["type"] = "scale_3d";
				for (int j = 0; j < c_track_keys.size(); j++) {
					String key = c_track_keys[j];
					track[key] = c_track[key];
				}
				track["keys"] = scale_keys;
				tracks.insert(i, track);
			}
			if (rotation_keys.size() > 0) {
				Dictionary track;
				track["type"] = "rotation_3d";
				for (int j = 0; j < c_track_keys.size(); j++) {
					String key = c_track_keys[j];
					track[key] = c_track[key];
				}
				track["keys"] = rotation_keys;
				tracks.insert(i, track);
			}
			if (position_keys.size() > 0) {
				Dictionary track;
				track["type"] = "position_3d";
				for (int j = 0; j < c_track_keys.size(); j++) {
					String key = c_track_keys[j];
					track[key] = c_track[key];
				}
				track["keys"] = position_keys;
				tracks.insert(i, track);
			}
		}
	}
	// now, set all the track data
	for (int i = 0; i < tracks.size(); i++) {
		const Dictionary &track = tracks[i];
		String track_prefix = "tracks/" + itos(i) + "/";
		animation->set(track_prefix + "type", track["type"]);
		// iterate over all the dictionary keys
		TypedArray<String> dict_keys = track.keys();
		for (int j = 0; j < dict_keys.size(); j++) {
			const String &key = dict_keys[j];
			if (key != "type" && key != "keys") {
				animation->set(track_prefix + key, track[key]);
			}
		}
		// set keys at the end
		animation->set(track_prefix + "keys", track["keys"]);
	}
	r_err = OK;
	return animation;
}

// Converts old 3.x animations transforms relative to the bone rest to absolute.
void EditorSceneFormatImporterESCN::_recompute_animation_tracks(AnimationPlayer *p_player) {
	List<StringName> anims;
	p_player->get_animation_list(&anims);
	Node *parent = p_player->get_parent();
	ERR_FAIL_NULL(parent);

	// we iterate over all the animations in the player, then all the tracks in the player
	// if it is position, rotation, or scale, we get the skeleton path and the bone name
	// then we get the node at the skeleton path, and get the bone index from the bone name
	// we then get the rest position, rotation, or scale from the skeleton
	// then we iterate over all the keys in the track, and multiply the key * rest position, rotation, or scale and assign it to the key
	for (List<StringName>::Element *E = anims.front(); E; E = E->next()) {
		StringName anim_name = E->get();
		Ref<Animation> anim = p_player->get_animation(anim_name);
		ERR_CONTINUE(anim.is_null());
		for (int i = 0; i < anim->get_track_count(); i++) {
			int track_type = anim->track_get_type(i);
			if (track_type == Animation::TYPE_POSITION_3D || track_type == Animation::TYPE_ROTATION_3D || track_type == Animation::TYPE_SCALE_3D) {
				NodePath path = anim->track_get_path(i);
				Node *node = parent->get_node(path);
				if (!node) {
					continue;
				}
				Skeleton3D *skel = Object::cast_to<Skeleton3D>(node);
				ERR_CONTINUE(!skel);

				StringName bone = path.get_subname(0);
				int bone_idx = skel->find_bone(bone);
				if (bone_idx == -1) {
					continue;
				}

				Transform3D rest = skel->get_bone_rest(bone_idx);
				for (int j = 0; j < anim->track_get_key_count(i); j++) {
					Variant val;
					if (track_type == Animation::TYPE_POSITION_3D) {
						Vector3 a_pos = anim->track_get_key_value(i, j);
						Transform3D t = Transform3D();
						t.set_origin(a_pos);
						Vector3 new_a_pos = (rest * t).origin;
						anim->track_set_key_value(i, j, new_a_pos);
					} else if (track_type == Animation::TYPE_ROTATION_3D) {
						Quaternion q = anim->track_get_key_value(i, j);
						Transform3D t = Transform3D();
						t.basis.rotate(q);
						Quaternion new_q = (rest * t).basis.get_rotation_quaternion();
						anim->track_set_key_value(i, j, new_q);
					} else if (track_type == Animation::TYPE_SCALE_3D) {
						Vector3 v = anim->track_get_key_value(i, j);
						Transform3D t = Transform3D();
						t.scale(v);
						Vector3 new_v = (rest * t).basis.get_scale(); // is this right? I have no idea how 3d works :pensive:
						anim->track_set_key_value(i, j, new_v);
					}
				}
			}
		}
	}
}

void EditorSceneFormatImporterESCN::_fix_old_format_scene(Node *scene) {
	TypedArray<Node> nodes = scene->find_children("*", "MeshInstance3D");
	for (int32_t node_i = 0; node_i < nodes.size(); node_i++) {
		MeshInstance3D *mesh_3d = cast_to<MeshInstance3D>(nodes[node_i]);
		auto skel_path = mesh_3d->get_skeleton_path();
		Node *skel_node = skel_path.is_absolute() ? scene->get_node(skel_path) : mesh_3d->get_node(skel_path);
		if (skel_node) {
			Skeleton3D *skel = Object::cast_to<Skeleton3D>(skel_node);
			if (!skel) {
				WARN_PRINT("MeshInstance3D.skeleton_path is not a Skeleton3D.");
				continue;
			}
			// Checking to see if the `pose` parameter was not set in the imported skeleton bones
			// 3.x did not export skeletons with the `pose` parameter set if `pose == Transform3D()`
			// If it wasn't, we need to set it to the rest pose if `rest != Transform3D()`
			// poses were relative to the rests in 3.x, but they're absolute in 4.x
			for (int i = 0; i < skel->get_bone_count(); i++) {
				Transform3D rest = skel->get_bone_rest(i);
				Vector3 pos = skel->get_bone_pose_position(i);
				Quaternion rot = skel->get_bone_pose_rotation(i);
				Vector3 scale = skel->get_bone_pose_scale(i);
				if (rest != Transform3D() && pos == Vector3() && rot == Quaternion() && scale == Vector3(1, 1, 1)) {
					// we need to set the position, rotation, and scale to the rest position, rotation, and scale
					skel->set_bone_pose_position(i, rest.origin);
					skel->set_bone_pose_rotation(i, rest.basis.get_rotation_quaternion());
					skel->set_bone_pose_scale(i, rest.basis.get_scale());
				}
			}
		}
	}
	// 3.x animations keyframe transforms were similarly relative to the `rest` pose
	// need to convert them to absolute
	TypedArray<Node> skel_nodes = scene->find_children("*", "AnimationPlayer");
	for (int32_t node_i = 0; node_i < skel_nodes.size(); node_i++) {
		AnimationPlayer *player = cast_to<AnimationPlayer>(skel_nodes[node_i]);
		_recompute_animation_tracks(player);
	}
}

uint32_t EditorSceneFormatImporterESCN::get_import_flags() const {
	return IMPORT_SCENE;
}

void EditorSceneFormatImporterESCN::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("escn");
}

int get_text_format_version(String p_path) {
	Error error;
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ, &error);
	ERR_FAIL_COND_V_MSG(error != OK || f.is_null(), -1, "Cannot open file '" + p_path + "'.");
	String line = f->get_line().strip_edges();
	// skip empty lines and comments
	while (line.is_empty() || line.begins_with(";")) {
		line = f->get_line().strip_edges();
		if (f->eof_reached())
			break;
	}
	int format_index = line.find("format");
	ERR_FAIL_COND_V_MSG(format_index == -1, -1, "No format specifier in file '" + p_path + "'.");
	String format_str = line.substr(format_index).get_slicec('=', 1).strip_edges();
	ERR_FAIL_COND_V_MSG(!format_str.substr(0, 1).is_numeric(), -1, "Invalid format in file '" + p_path + "'.");
	int format = format_str.to_int();
	return format;
}

Node *EditorSceneFormatImporterESCN::import_scene(const String &p_path, uint32_t p_flags, const HashMap<StringName, Variant> &p_options, List<String> *r_missing_deps, Error *r_err) {
	Error error;
	Ref<PackedScene> ps;

	int format = get_text_format_version(p_path);
	ERR_FAIL_COND_V(format == -1, nullptr);
	if (format == 2) { // 3.x escn export
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ, &error);
		ERR_FAIL_COND_V_MSG(error != OK, nullptr, "Cannot open file '" + p_path + "'.");
		ResourceLoaderText loader;
		String path = p_path;
		loader.local_path = ProjectSettings::get_singleton()->localize_path(path);
		loader.res_path = loader.local_path;
		loader.cache_mode = ResourceFormatLoader::CACHE_MODE_IGNORE; // So we don't recalculate the animations twice
		loader._set_special_handler("Animation", convert_old_animation);
		loader._set_special_handler("Shader", convert_old_shader);
		loader.open(f);
		error = loader.load();
		ERR_FAIL_COND_V_MSG(error != OK, nullptr, "Cannot load scene as text resource from path '" + p_path + "'.");
		ps = loader.get_resource();
	} else {
		ps = ResourceFormatLoaderText::singleton->load(p_path, p_path, &error);
	}
	ERR_FAIL_COND_V_MSG(!ps.is_valid(), nullptr, "Cannot load scene as text resource from path '" + p_path + "'.");
	Node *scene = ps->instantiate();
	ERR_FAIL_COND_V(!scene, nullptr);
	TypedArray<Node> nodes = scene->find_children("*", "MeshInstance3D");
	if (format == 2) {
		_fix_old_format_scene(scene);
	}
	for (int32_t node_i = 0; node_i < nodes.size(); node_i++) {
		MeshInstance3D *mesh_3d = cast_to<MeshInstance3D>(nodes[node_i]);
		Ref<ImporterMesh> mesh;
		mesh.instantiate();
		// Ignore the aabb, it will be recomputed.
		ImporterMeshInstance3D *importer_mesh_3d = memnew(ImporterMeshInstance3D);
		importer_mesh_3d->set_name(mesh_3d->get_name());
		Node *parent = mesh_3d->get_parent();
		Transform3D rel_transform = mesh_3d->get_relative_transform(parent);
		if (rel_transform == Transform3D() && parent && parent != mesh_3d) {
			// If we're here, we probably got a "data.parent is null" error
			// Node3D.data.parent hasn't been set yet but Node.data.parent has, so we need to get the transform manually
			Node3D *parent_3d = mesh_3d->get_parent_node_3d();
			if (parent == parent_3d) {
				rel_transform = mesh_3d->get_transform();
			} else if (parent_3d) {
				rel_transform = parent_3d->get_relative_transform(parent) * mesh_3d->get_transform();
			} // otherwise parent isn't a Node3D
		}
		importer_mesh_3d->set_transform(rel_transform);
		Ref<Skin> skin = mesh_3d->get_skin();
		importer_mesh_3d->set_skeleton_path(mesh_3d->get_skeleton_path());
		importer_mesh_3d->set_skin(skin);
		Ref<ArrayMesh> array_mesh_3d_mesh = mesh_3d->get_mesh();
		if (array_mesh_3d_mesh.is_valid()) {
			// For the MeshInstance3D nodes, we need to convert the ArrayMesh to an ImporterMesh specially.
			mesh->set_name(array_mesh_3d_mesh->get_name());
			for (int32_t blend_i = 0; blend_i < array_mesh_3d_mesh->get_blend_shape_count(); blend_i++) {
				mesh->add_blend_shape(array_mesh_3d_mesh->get_blend_shape_name(blend_i));
			}
			for (int32_t surface_i = 0; surface_i < array_mesh_3d_mesh->get_surface_count(); surface_i++) {
				mesh->add_surface(array_mesh_3d_mesh->surface_get_primitive_type(surface_i),
						array_mesh_3d_mesh->surface_get_arrays(surface_i),
						array_mesh_3d_mesh->surface_get_blend_shape_arrays(surface_i),
						array_mesh_3d_mesh->surface_get_lods(surface_i),
						array_mesh_3d_mesh->surface_get_material(surface_i),
						array_mesh_3d_mesh->surface_get_name(surface_i),
						array_mesh_3d_mesh->surface_get_format(surface_i));
			}
			mesh->set_blend_shape_mode(array_mesh_3d_mesh->get_blend_shape_mode());
			importer_mesh_3d->set_mesh(mesh);
			mesh_3d->replace_by(importer_mesh_3d);
			continue;
		}
		Ref<Mesh> mesh_3d_mesh = mesh_3d->get_mesh();
		if (mesh_3d_mesh.is_valid()) {
			// For the MeshInstance3D nodes, we need to convert the Mesh to an ImporterMesh specially.
			mesh->set_name(mesh_3d_mesh->get_name());
			for (int32_t surface_i = 0; surface_i < mesh_3d_mesh->get_surface_count(); surface_i++) {
				mesh->add_surface(mesh_3d_mesh->surface_get_primitive_type(surface_i),
						mesh_3d_mesh->surface_get_arrays(surface_i),
						Array(),
						mesh_3d_mesh->surface_get_lods(surface_i),
						mesh_3d_mesh->surface_get_material(surface_i),
						mesh_3d_mesh->surface_get_material(surface_i).is_valid() ? mesh_3d_mesh->surface_get_material(surface_i)->get_name() : String(),
						mesh_3d_mesh->surface_get_format(surface_i));
			}
			importer_mesh_3d->set_mesh(mesh);
			mesh_3d->replace_by(importer_mesh_3d);
			continue;
		}
	}

	return scene;
}
