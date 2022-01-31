/*************************************************************************/
/*  resource_importer_scene.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "resource_importer_scene.h"

#include "core/error/error_macros.h"
#include "core/io/resource_saver.h"
#include "editor/editor_node.h"
#include "editor/import/scene_import_settings.h"
#include "scene/3d/area_3d.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/navigation_region_3d.h"
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

uint32_t EditorSceneFormatImporter::get_import_flags() const {
	int ret;
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

Node *EditorSceneFormatImporter::import_scene(const String &p_path, uint32_t p_flags, const Map<StringName, Variant> &p_options, int p_bake_fps, List<String> *r_missing_deps, Error *r_err) {
	Dictionary options_dict;
	for (const KeyValue<StringName, Variant> &elem : p_options) {
		options_dict[elem.key] = elem.value;
	}
	Object *ret = nullptr;
	if (GDVIRTUAL_CALL(_import_scene, p_path, p_flags, options_dict, p_bake_fps, ret)) {
		return Object::cast_to<Node>(ret);
	}

	ERR_FAIL_V(nullptr);
}

Ref<Animation> EditorSceneFormatImporter::import_animation(const String &p_path, uint32_t p_flags, const Map<StringName, Variant> &p_options, int p_bake_fps) {
	Dictionary options_dict;
	for (const KeyValue<StringName, Variant> &elem : p_options) {
		options_dict[elem.key] = elem.value;
	}
	Ref<Animation> ret;
	if (GDVIRTUAL_CALL(_import_animation, p_path, p_flags, options_dict, p_bake_fps, ret)) {
		return ret;
	}

	ERR_FAIL_V(nullptr);
}

void EditorSceneFormatImporter::get_import_options(const String &p_path, List<ResourceImporter::ImportOption> *r_options) {
	GDVIRTUAL_CALL(_get_import_options, p_path);
}

Variant EditorSceneFormatImporter::get_option_visibility(const String &p_path, const String &p_option, const Map<StringName, Variant> &p_options) {
	Variant ret;
	GDVIRTUAL_CALL(_get_option_visibility, p_path, p_option, ret);
	return ret;
}

void EditorSceneFormatImporter::_bind_methods() {
	GDVIRTUAL_BIND(_get_import_flags);
	GDVIRTUAL_BIND(_get_extensions);
	GDVIRTUAL_BIND(_import_scene, "path", "flags", "options", "bake_fps");
	GDVIRTUAL_BIND(_import_animation, "path", "flags", "options", "bake_fps");
	GDVIRTUAL_BIND(_get_import_options, "path");
	GDVIRTUAL_BIND(_get_option_visibility, "path", "option");

	BIND_CONSTANT(IMPORT_SCENE);
	BIND_CONSTANT(IMPORT_ANIMATION);
	BIND_CONSTANT(IMPORT_FAIL_ON_MISSING_DEPENDENCIES);
	BIND_CONSTANT(IMPORT_GENERATE_TANGENT_ARRAYS);
	BIND_CONSTANT(IMPORT_USE_NAMED_SKIN_BINDS);
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
	if (current_options) {
		(*current_options)[p_name];
	}
	if (current_options_dict) {
		(*current_options_dict)[p_name];
	}
	return Variant();
}
void EditorScenePostImportPlugin::add_import_option(const String &p_name, Variant p_default_value) {
	ERR_FAIL_COND_MSG(current_option_list == nullptr, "add_import_option() can only be called from get_import_options()");
	add_import_option_advanced(p_default_value.get_type(), p_name, p_default_value);
}
void EditorScenePostImportPlugin::add_import_option_advanced(Variant::Type p_type, const String &p_name, Variant p_default_value, PropertyHint p_hint, const String &p_hint_string, int p_usage_flags) {
	ERR_FAIL_COND_MSG(current_option_list == nullptr, "add_import_option_advanced() can only be called from get_import_options()");
	current_option_list->push_back(ResourceImporter::ImportOption(PropertyInfo(p_type, p_name, p_hint, p_hint_string, p_usage_flags), p_default_value));
}

void EditorScenePostImportPlugin::get_internal_import_options(InternalImportCategory p_category, List<ResourceImporter::ImportOption> *r_options) {
	current_option_list = r_options;
	GDVIRTUAL_CALL(_get_internal_import_options, p_category);
	current_option_list = nullptr;
}
Variant EditorScenePostImportPlugin::get_internal_option_visibility(InternalImportCategory p_category, const String &p_option, const Map<StringName, Variant> &p_options) const {
	current_options = &p_options;
	Variant ret;
	GDVIRTUAL_CALL(_get_internal_option_visibility, p_category, p_option, ret);
	current_options = nullptr;
	return ret;
}
Variant EditorScenePostImportPlugin::get_internal_option_update_view_required(InternalImportCategory p_category, const String &p_option, const Map<StringName, Variant> &p_options) const {
	current_options = &p_options;
	Variant ret;
	GDVIRTUAL_CALL(_get_internal_option_update_view_required, p_category, p_option, ret);
	current_options = nullptr;
	return ret;
}

void EditorScenePostImportPlugin::internal_process(InternalImportCategory p_category, Node *p_base_scene, Node *p_node, RES p_resource, const Dictionary &p_options) {
	current_options_dict = &p_options;
	GDVIRTUAL_CALL(_internal_process, p_category, p_base_scene, p_node, p_resource);
	current_options_dict = nullptr;
}

void EditorScenePostImportPlugin::get_import_options(const String &p_path, List<ResourceImporter::ImportOption> *r_options) {
	current_option_list = r_options;
	GDVIRTUAL_CALL(_get_import_options, p_path);
	current_option_list = nullptr;
}
Variant EditorScenePostImportPlugin::get_option_visibility(const String &p_path, const String &p_option, const Map<StringName, Variant> &p_options) const {
	current_options = &p_options;
	Variant ret;
	GDVIRTUAL_CALL(_get_option_visibility, p_path, p_option, ret);
	current_options = nullptr;
	return ret;
}

void EditorScenePostImportPlugin::pre_process(Node *p_scene, const Map<StringName, Variant> &p_options) {
	current_options = &p_options;
	GDVIRTUAL_CALL(_pre_process, p_scene);
	current_options = nullptr;
}
void EditorScenePostImportPlugin::post_process(Node *p_scene, const Map<StringName, Variant> &p_options) {
	current_options = &p_options;
	GDVIRTUAL_CALL(_post_process, p_scene);
	current_options = nullptr;
}

void EditorScenePostImportPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_option_value", "name"), &EditorScenePostImportPlugin::get_option_value);

	ClassDB::bind_method(D_METHOD("add_import_option", "name", "value"), &EditorScenePostImportPlugin::add_import_option);
	ClassDB::bind_method(D_METHOD("add_import_option_advanced", "type", "name", "default_value", "hint", "hint_string", "usage_flags"), &EditorScenePostImportPlugin::add_import_option_advanced, DEFVAL(PROPERTY_HINT_NONE), DEFVAL(""), DEFVAL(PROPERTY_USAGE_DEFAULT));

	GDVIRTUAL_BIND(_get_internal_import_options, "category");
	GDVIRTUAL_BIND(_get_internal_option_visibility, "category", "option");
	GDVIRTUAL_BIND(_get_internal_option_update_view_required, "category", "option");
	GDVIRTUAL_BIND(_internal_process, "category", "base_node", "node", "resource");
	GDVIRTUAL_BIND(_get_import_options, "path");
	GDVIRTUAL_BIND(_get_option_visibility, "path", "option");
	GDVIRTUAL_BIND(_pre_process, "scene");
	GDVIRTUAL_BIND(_post_process, "scene");

	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_NODE);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_MESH);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_MATERIAL);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_ANIMATION);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE);
	BIND_ENUM_CONSTANT(INTERNAL_IMPORT_CATEGORY_MAX);
}

/////////////////////////////////////////////////////////

String ResourceImporterScene::get_importer_name() const {
	return "scene";
}

String ResourceImporterScene::get_visible_name() const {
	return "Scene";
}

void ResourceImporterScene::get_recognized_extensions(List<String> *p_extensions) const {
	for (Set<Ref<EditorSceneFormatImporter>>::Element *E = importers.front(); E; E = E->next()) {
		E->get()->get_extensions(p_extensions);
	}
}

String ResourceImporterScene::get_save_extension() const {
	return "scn";
}

String ResourceImporterScene::get_resource_type() const {
	return "PackedScene";
}

int ResourceImporterScene::get_format_version() const {
	return 1;
}

bool ResourceImporterScene::get_option_visibility(const String &p_path, const String &p_option, const Map<StringName, Variant> &p_options) const {
	if (p_option.begins_with("animation/")) {
		if (p_option != "animation/import" && !bool(p_options["animation/import"])) {
			return false;
		}
	}

	if (p_option == "meshes/lightmap_texel_size" && int(p_options["meshes/light_baking"]) != 2) {
		// Only display the lightmap texel size import option when using the Static Lightmaps light baking mode.
		return false;
	}

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		Variant ret = post_importer_plugins.write[i]->get_option_visibility(p_path, p_option, p_options);
		if (ret.get_type() == Variant::BOOL) {
			return ret;
		}
	}

	for (Ref<EditorSceneFormatImporter> importer : importers) {
		Variant ret = importer->get_option_visibility(p_path, p_option, p_options);
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

	//remove trailing spaces and numbers, some apps like blender add ".number" to duplicates so also compensate for this
	while (what.length() && ((what[what.length() - 1] >= '0' && what[what.length() - 1] <= '9') || what[what.length() - 1] <= 32 || what[what.length() - 1] == '.')) {
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

	//remove trailing spaces and numbers, some apps like blender add ".number" to duplicates so also compensate for this
	while (what.length() && ((what[what.length() - 1] >= '0' && what[what.length() - 1] <= '9') || what[what.length() - 1] <= 32 || what[what.length() - 1] == '.')) {
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
	ERR_FAIL_NULL_MSG(mesh, "Cannot generate shape list with null mesh value");
	ERR_FAIL_NULL_MSG(mesh->get_mesh(), "Cannot generate shape list with null mesh value");
	if (!p_convex) {
		Ref<Shape3D> shape = mesh->create_trimesh_shape();
		r_shape_list.push_back(shape);
	} else {
		Vector<Ref<Shape3D>> cd;
		cd.push_back(mesh->get_mesh()->create_convex_shape(true, /*Passing false, otherwise VHACD will be used to simplify (Decompose) the Mesh.*/ false));
		if (cd.size()) {
			for (int i = 0; i < cd.size(); i++) {
				r_shape_list.push_back(cd[i]);
			}
		}
	}
}

Node *ResourceImporterScene::_pre_fix_node(Node *p_node, Node *p_root, Map<Ref<ImporterMesh>, Vector<Ref<Shape3D>>> &collision_map, List<Pair<NodePath, Node *>> &r_node_renames) {
	// Children first.
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *r = _pre_fix_node(p_node->get_child(i), p_root, collision_map, r_node_renames);
		if (!r) {
			i--; // Was erased.
		}
	}

	String name = p_node->get_name();
	NodePath original_path = p_root->get_path_to(p_node); // Used to detect renames due to import hints.

	bool isroot = p_node == p_root;

	if (!isroot && _teststr(name, "noimp")) {
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
		Node *ap_root = ap->get_node(ap->get_root());
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
					anim->set_loop_mode(Animation::LoopMode::LOOP_LINEAR);
					animname = _fixstr(animname, loop_strings[i]);
					ap->rename_animation(E, animname);
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

		ERR_FAIL_COND_V(fixed_name.is_empty(), nullptr);

		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);
		if (mi) {
			Ref<ImporterMesh> mesh = mi->get_mesh();

			if (mesh.is_valid()) {
				Vector<Ref<Shape3D>> shapes;
				if (collision_map.has(mesh)) {
					shapes = collision_map[mesh];
				} else if (_teststr(name, "colonly")) {
					_pre_gen_shape_list(mesh, shapes, false);
					collision_map[mesh] = shapes;
				} else if (_teststr(name, "convcolonly")) {
					_pre_gen_shape_list(mesh, shapes, true);
					collision_map[mesh] = shapes;
				}

				if (shapes.size()) {
					StaticBody3D *col = memnew(StaticBody3D);
					col->set_transform(mi->get_transform());
					col->set_name(fixed_name);
					p_node->replace_by(col);
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
			if (collision_map.has(mesh)) {
				shapes = collision_map[mesh];
			} else {
				_pre_gen_shape_list(mesh, shapes, true);
			}

			RigidDynamicBody3D *rigid_body = memnew(RigidDynamicBody3D);
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
			if (collision_map.has(mesh)) {
				shapes = collision_map[mesh];
			} else if (_teststr(name, "col")) {
				_pre_gen_shape_list(mesh, shapes, false);
				collision_map[mesh] = shapes;
			} else if (_teststr(name, "convcol")) {
				_pre_gen_shape_list(mesh, shapes, true);
				collision_map[mesh] = shapes;
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
		memdelete(p_node);
		p_node = nmi;

	} else if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		//last attempt, maybe collision inside the mesh data

		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);

		Ref<ImporterMesh> mesh = mi->get_mesh();
		if (!mesh.is_null()) {
			Vector<Ref<Shape3D>> shapes;
			if (collision_map.has(mesh)) {
				shapes = collision_map[mesh];
			} else if (_teststr(mesh->get_name(), "col")) {
				_pre_gen_shape_list(mesh, shapes, false);
				collision_map[mesh] = shapes;
				mesh->set_name(_fixstr(mesh->get_name(), "col"));
			} else if (_teststr(mesh->get_name(), "convcol")) {
				_pre_gen_shape_list(mesh, shapes, true);
				collision_map[mesh] = shapes;
				mesh->set_name(_fixstr(mesh->get_name(), "convcol"));
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

Node *ResourceImporterScene::_post_fix_node(Node *p_node, Node *p_root, Map<Ref<ImporterMesh>, Vector<Ref<Shape3D>>> &collision_map, Set<Ref<ImporterMesh>> &r_scanned_meshes, const Dictionary &p_node_data, const Dictionary &p_material_data, const Dictionary &p_animation_data, float p_animation_fps) {
	// children first
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *r = _post_fix_node(p_node->get_child(i), p_root, collision_map, r_scanned_meshes, p_node_data, p_material_data, p_animation_data, p_animation_fps);
		if (!r) {
			i--; //was erased
		}
	}

	bool isroot = p_node == p_root;

	String import_id;

	if (p_node->has_meta("import_id")) {
		import_id = p_node->get_meta("import_id");
	} else {
		import_id = "PATH:" + p_root->get_path_to(p_node);
	}

	Dictionary node_settings;
	if (p_node_data.has(import_id)) {
		node_settings = p_node_data[import_id];
	}

	if (!isroot && (node_settings.has("import/skip_import") && bool(node_settings["import/skip_import"]))) {
		memdelete(p_node);
		return nullptr;
	}

	{
		ObjectID node_id = p_node->get_instance_id();
		for (int i = 0; i < post_importer_plugins.size(); i++) {
			post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_NODE, p_root, p_node, RES(), node_settings);
			if (ObjectDB::get_instance(node_id) == nullptr) { //may have been erased, so do not continue
				break;
			}
		}
	}

	if (Object::cast_to<ImporterMeshInstance3D>(p_node)) {
		ObjectID node_id = p_node->get_instance_id();
		for (int i = 0; i < post_importer_plugins.size(); i++) {
			post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE, p_root, p_node, RES(), node_settings);
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
						String mat_id;
						if (mat->has_meta("import_id")) {
							mat_id = mat->get_meta("import_id");
						} else {
							mat_id = mat->get_name();
						}

						if (!mat_id.is_empty() && p_material_data.has(mat_id)) {
							Dictionary matdata = p_material_data[mat_id];

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
								m->get_mesh(),
								node_settings);
					}

					if (shapes.size()) {
						CollisionObject3D *base = nullptr;
						switch (mesh_physics_mode) {
							case MESH_PHYSICS_MESH_AND_STATIC_COLLIDER: {
								StaticBody3D *col = memnew(StaticBody3D);
								p_node->add_child(col, true);
								col->set_owner(p_node->get_owner());
								col->set_transform(get_collision_shapes_transform(node_settings));
								base = col;
							} break;
							case MESH_PHYSICS_RIGID_BODY_AND_MESH: {
								RigidDynamicBody3D *rigid_body = memnew(RigidDynamicBody3D);
								rigid_body->set_name(p_node->get_name());
								p_node->replace_by(rigid_body);
								rigid_body->set_transform(mi->get_transform() * get_collision_shapes_transform(node_settings));
								p_node = rigid_body;
								mi->set_transform(Transform3D());
								rigid_body->add_child(mi, true);
								mi->set_owner(rigid_body->get_owner());
								base = rigid_body;
							} break;
							case MESH_PHYSICS_STATIC_COLLIDER_ONLY: {
								StaticBody3D *col = memnew(StaticBody3D);
								col->set_transform(mi->get_transform() * get_collision_shapes_transform(node_settings));
								col->set_name(p_node->get_name());
								p_node->replace_by(col);
								memdelete(p_node);
								p_node = col;
								base = col;
							} break;
							case MESH_PHYSICS_AREA_ONLY: {
								Area3D *area = memnew(Area3D);
								area->set_transform(mi->get_transform() * get_collision_shapes_transform(node_settings));
								area->set_name(p_node->get_name());
								p_node->replace_by(area);
								memdelete(p_node);
								p_node = area;
								base = area;

							} break;
						}

						int idx = 0;
						for (const Ref<Shape3D> &E : shapes) {
							CollisionShape3D *cshape = memnew(CollisionShape3D);
							cshape->set_shape(E);
							base->add_child(cshape, true);

							cshape->set_owner(base->get_owner());
							idx++;
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

	if (Object::cast_to<AnimationPlayer>(p_node)) {
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(p_node);

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

		for (int i = 0; i < post_importer_plugins.size(); i++) {
			post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE, p_root, p_node, RES(), node_settings);
		}

		bool use_optimizer = node_settings["optimizer/enabled"];
		float anim_optimizer_linerr = node_settings["optimizer/max_linear_error"];
		float anim_optimizer_angerr = node_settings["optimizer/max_angular_error"];
		float anim_optimizer_maxang = node_settings["optimizer/max_angle"];

		if (use_optimizer) {
			_optimize_animations(ap, anim_optimizer_linerr, anim_optimizer_angerr, anim_optimizer_maxang);
		}

		Array animation_clips;
		{
			int clip_count = node_settings["clips/amount"];

			for (int i = 0; i < clip_count; i++) {
				String name = node_settings["clip_" + itos(i + 1) + "/name"];
				int from_frame = node_settings["clip_" + itos(i + 1) + "/start_frame"];
				int end_frame = node_settings["clip_" + itos(i + 1) + "/end_frame"];
				Animation::LoopMode loop_mode = static_cast<Animation::LoopMode>((int)node_settings["clip_" + itos(i + 1) + "/loop_mode"]);
				bool save_to_file = node_settings["clip_" + itos(i + 1) + "/save_to_file/enabled"];
				bool save_to_path = node_settings["clip_" + itos(i + 1) + "/save_to_file/path"];
				bool save_to_file_keep_custom = node_settings["clip_" + itos(i + 1) + "/save_to_file/keep_custom_tracks"];

				animation_clips.push_back(name);
				animation_clips.push_back(from_frame / p_animation_fps);
				animation_clips.push_back(end_frame / p_animation_fps);
				animation_clips.push_back(loop_mode);
				animation_clips.push_back(save_to_file);
				animation_clips.push_back(save_to_path);
				animation_clips.push_back(save_to_file_keep_custom);
			}
		}

		if (animation_clips.size()) {
			_create_clips(ap, animation_clips, true);
		} else {
			List<StringName> anims;
			ap->get_animation_list(&anims);
			for (const StringName &name : anims) {
				Ref<Animation> anim = ap->get_animation(name);
				if (p_animation_data.has(name)) {
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

					anim->set_loop_mode(static_cast<Animation::LoopMode>((int)anim_settings["settings/loop_mode"]));
					bool save = anim_settings["save_to_file/enabled"];
					String path = anim_settings["save_to_file/path"];
					bool keep_custom = anim_settings["save_to_file/keep_custom_tracks"];

					Ref<Animation> saved_anim = _save_animation_to_file(anim, save, path, keep_custom);

					if (saved_anim != anim) {
						ap->add_animation(name, saved_anim); //replace
					}
				}
			}

			AnimationImportTracks import_tracks_mode[TRACK_CHANNEL_MAX] = {
				AnimationImportTracks(int(node_settings["import_tracks/position"])),
				AnimationImportTracks(int(node_settings["import_tracks/rotation"])),
				AnimationImportTracks(int(node_settings["import_tracks/scale"]))
			};

			if (anims.size() > 1 && (import_tracks_mode[0] != ANIMATION_IMPORT_TRACKS_IF_PRESENT || import_tracks_mode[1] != ANIMATION_IMPORT_TRACKS_IF_PRESENT || import_tracks_mode[2] != ANIMATION_IMPORT_TRACKS_IF_PRESENT)) {
				_optimize_track_usage(ap, import_tracks_mode);
			}
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
						post_importer_plugins.write[i]->internal_process(EditorScenePostImportPlugin::INTERNAL_IMPORT_CATEGORY_ANIMATION, p_root, p_node, anim, node_settings);
					}
				}
			}
		}

		bool use_compression = node_settings["compression/enabled"];
		int anim_compression_page_size = node_settings["compression/page_size"];

		if (use_compression) {
			_compress_animations(ap, anim_compression_page_size);
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
		Ref<Animation> old_anim = Ref<Resource>(ResourceCache::get(p_save_to_path));
		if (old_anim.is_valid()) {
			old_anim->copy_from(anim);
			anim = old_anim;
		}
	}
	anim->set_path(p_save_to_path, true); // Set path to save externally.
	Error err = ResourceSaver::save(p_save_to_path, anim, ResourceSaver::FLAG_CHANGE_PATH);
	ERR_FAIL_COND_V_MSG(err != OK, anim, "Saving of animation failed: " + p_save_to_path);
	return anim;
}

void ResourceImporterScene::_create_clips(AnimationPlayer *anim, const Array &p_clips, bool p_bake_all) {
	if (!anim->has_animation("default")) {
		ERR_FAIL_COND_MSG(p_clips.size() > 0, "To create clips, animations must be named \"default\".");
		return;
	}

	Ref<Animation> default_anim = anim->get_animation("default");

	for (int i = 0; i < p_clips.size(); i += 7) {
		String name = p_clips[i];
		float from = p_clips[i + 1];
		float to = p_clips[i + 2];
		Animation::LoopMode loop_mode = static_cast<Animation::LoopMode>((int)p_clips[i + 3]);
		bool save_to_file = p_clips[i + 4];
		String save_to_path = p_clips[i + 5];
		bool keep_current = p_clips[i + 6];
		if (from >= to) {
			continue;
		}

		Ref<Animation> new_anim = memnew(Animation);

		for (int j = 0; j < default_anim->get_track_count(); j++) {
			List<float> keys;
			int kc = default_anim->track_get_key_count(j);
			int dtrack = -1;
			for (int k = 0; k < kc; k++) {
				float kt = default_anim->track_get_key_time(j, k);
				if (kt >= from && kt < to) {
					//found a key within range, so create track
					if (dtrack == -1) {
						new_anim->add_track(default_anim->track_get_type(j));
						dtrack = new_anim->get_track_count() - 1;
						new_anim->track_set_path(dtrack, default_anim->track_get_path(j));

						if (kt > (from + 0.01) && k > 0) {
							if (default_anim->track_get_type(j) == Animation::TYPE_POSITION_3D) {
								Vector3 p;
								default_anim->position_track_interpolate(j, from, &p);
								new_anim->position_track_insert_key(dtrack, 0, p);
							} else if (default_anim->track_get_type(j) == Animation::TYPE_ROTATION_3D) {
								Quaternion r;
								default_anim->rotation_track_interpolate(j, from, &r);
								new_anim->rotation_track_insert_key(dtrack, 0, r);
							} else if (default_anim->track_get_type(j) == Animation::TYPE_SCALE_3D) {
								Vector3 s;
								default_anim->scale_track_interpolate(j, from, &s);
								new_anim->scale_track_insert_key(dtrack, 0, s);
							} else if (default_anim->track_get_type(j) == Animation::TYPE_VALUE) {
								Variant var = default_anim->value_track_interpolate(j, from);
								new_anim->track_insert_key(dtrack, 0, var);
							} else if (default_anim->track_get_type(j) == Animation::TYPE_BLEND_SHAPE) {
								float interp;
								default_anim->blend_shape_track_interpolate(j, from, &interp);
								new_anim->blend_shape_track_insert_key(dtrack, 0, interp);
							}
						}
					}

					if (default_anim->track_get_type(j) == Animation::TYPE_POSITION_3D) {
						Vector3 p;
						default_anim->position_track_get_key(j, k, &p);
						new_anim->position_track_insert_key(dtrack, kt - from, p);
					} else if (default_anim->track_get_type(j) == Animation::TYPE_ROTATION_3D) {
						Quaternion r;
						default_anim->rotation_track_get_key(j, k, &r);
						new_anim->rotation_track_insert_key(dtrack, kt - from, r);
					} else if (default_anim->track_get_type(j) == Animation::TYPE_SCALE_3D) {
						Vector3 s;
						default_anim->scale_track_get_key(j, k, &s);
						new_anim->scale_track_insert_key(dtrack, kt - from, s);
					} else if (default_anim->track_get_type(j) == Animation::TYPE_VALUE) {
						Variant var = default_anim->track_get_key_value(j, k);
						new_anim->track_insert_key(dtrack, kt - from, var);
					} else if (default_anim->track_get_type(j) == Animation::TYPE_BLEND_SHAPE) {
						float interp;
						default_anim->blend_shape_track_get_key(j, k, &interp);
						new_anim->blend_shape_track_insert_key(dtrack, kt - from, interp);
					}
				}

				if (dtrack != -1 && kt >= to) {
					if (default_anim->track_get_type(j) == Animation::TYPE_POSITION_3D) {
						Vector3 p;
						default_anim->position_track_interpolate(j, to, &p);
						new_anim->position_track_insert_key(dtrack, to - from, p);
					} else if (default_anim->track_get_type(j) == Animation::TYPE_ROTATION_3D) {
						Quaternion r;
						default_anim->rotation_track_interpolate(j, to, &r);
						new_anim->rotation_track_insert_key(dtrack, to - from, r);
					} else if (default_anim->track_get_type(j) == Animation::TYPE_SCALE_3D) {
						Vector3 s;
						default_anim->scale_track_interpolate(j, to, &s);
						new_anim->scale_track_insert_key(dtrack, to - from, s);
					} else if (default_anim->track_get_type(j) == Animation::TYPE_VALUE) {
						Variant var = default_anim->value_track_interpolate(j, to);
						new_anim->track_insert_key(dtrack, to - from, var);
					} else if (default_anim->track_get_type(j) == Animation::TYPE_BLEND_SHAPE) {
						float interp;
						default_anim->blend_shape_track_interpolate(j, to, &interp);
						new_anim->blend_shape_track_insert_key(dtrack, to - from, interp);
					}
				}
			}

			if (dtrack == -1 && p_bake_all) {
				new_anim->add_track(default_anim->track_get_type(j));
				dtrack = new_anim->get_track_count() - 1;
				new_anim->track_set_path(dtrack, default_anim->track_get_path(j));
				if (default_anim->track_get_type(j) == Animation::TYPE_POSITION_3D) {
					Vector3 p;
					default_anim->position_track_interpolate(j, from, &p);
					new_anim->position_track_insert_key(dtrack, 0, p);
					default_anim->position_track_interpolate(j, to, &p);
					new_anim->position_track_insert_key(dtrack, to - from, p);
				} else if (default_anim->track_get_type(j) == Animation::TYPE_ROTATION_3D) {
					Quaternion r;
					default_anim->rotation_track_interpolate(j, from, &r);
					new_anim->rotation_track_insert_key(dtrack, 0, r);
					default_anim->rotation_track_interpolate(j, to, &r);
					new_anim->rotation_track_insert_key(dtrack, to - from, r);
				} else if (default_anim->track_get_type(j) == Animation::TYPE_SCALE_3D) {
					Vector3 s;
					default_anim->scale_track_interpolate(j, from, &s);
					new_anim->scale_track_insert_key(dtrack, 0, s);
					default_anim->scale_track_interpolate(j, to, &s);
					new_anim->scale_track_insert_key(dtrack, to - from, s);
				} else if (default_anim->track_get_type(j) == Animation::TYPE_VALUE) {
					Variant var = default_anim->value_track_interpolate(j, from);
					new_anim->track_insert_key(dtrack, 0, var);
					Variant to_var = default_anim->value_track_interpolate(j, to);
					new_anim->track_insert_key(dtrack, to - from, to_var);
				} else if (default_anim->track_get_type(j) == Animation::TYPE_BLEND_SHAPE) {
					float interp;
					default_anim->blend_shape_track_interpolate(j, from, &interp);
					new_anim->blend_shape_track_insert_key(dtrack, 0, interp);
					default_anim->blend_shape_track_interpolate(j, to, &interp);
					new_anim->blend_shape_track_insert_key(dtrack, to - from, interp);
				}
			}
		}

		new_anim->set_loop_mode(loop_mode);
		new_anim->set_length(to - from);
		anim->add_animation(name, new_anim);

		Ref<Animation> saved_anim = _save_animation_to_file(new_anim, save_to_file, save_to_path, keep_current);
		if (saved_anim != new_anim) {
			anim->add_animation(name, saved_anim);
		}
	}

	anim->remove_animation("default"); //remove default (no longer needed)
}

void ResourceImporterScene::_optimize_animations(AnimationPlayer *anim, float p_max_lin_error, float p_max_ang_error, float p_max_angle) {
	List<StringName> anim_names;
	anim->get_animation_list(&anim_names);
	for (const StringName &E : anim_names) {
		Ref<Animation> a = anim->get_animation(E);
		a->optimize(p_max_lin_error, p_max_ang_error, Math::deg2rad(p_max_angle));
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

			// Decomposition
			Mesh::ConvexDecompositionSettings decomposition_default;
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "decomposition/advanced", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/precision", PROPERTY_HINT_RANGE, "1,10,1"), 5));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "decomposition/max_concavity", PROPERTY_HINT_RANGE, "0.0,1.0,0.001", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.max_concavity));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "decomposition/symmetry_planes_clipping_bias", PROPERTY_HINT_RANGE, "0.0,1.0,0.001", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.symmetry_planes_clipping_bias));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "decomposition/revolution_axes_clipping_bias", PROPERTY_HINT_RANGE, "0.0,1.0,0.001", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.revolution_axes_clipping_bias));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "decomposition/min_volume_per_convex_hull", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.min_volume_per_convex_hull));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/resolution", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.resolution));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/max_num_vertices_per_convex_hull", PROPERTY_HINT_RANGE, "5,512,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.max_num_vertices_per_convex_hull));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/plane_downsampling", PROPERTY_HINT_RANGE, "1,16,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.plane_downsampling));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/convexhull_downsampling", PROPERTY_HINT_RANGE, "1,16,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.convexhull_downsampling));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "decomposition/normalize_mesh", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.normalize_mesh));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/mode", PROPERTY_HINT_ENUM, "Voxel,Tetrahedron", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), static_cast<int>(decomposition_default.mode)));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "decomposition/convexhull_approximation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.convexhull_approximation));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "decomposition/max_convex_hulls", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.max_convex_hulls));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "decomposition/project_hull_vertices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), decomposition_default.project_hull_vertices));

			// Primitives: Box, Sphere, Cylinder, Capsule.
			r_options->push_back(ImportOption(PropertyInfo(Variant::VECTOR3, "primitive/size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), Vector3(2.0, 2.0, 2.0)));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "primitive/height", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 1.0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "primitive/radius", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 1.0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::VECTOR3, "primitive/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), Vector3()));
			r_options->push_back(ImportOption(PropertyInfo(Variant::VECTOR3, "primitive/rotation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), Vector3()));
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
		} break;
		case INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "import/skip_import", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "optimizer/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), true));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "optimizer/max_linear_error"), 0.05));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "optimizer/max_angular_error"), 0.01));
			r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "optimizer/max_angle"), 22));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compression/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compression/page_size", PROPERTY_HINT_RANGE, "4,512,1,suffix:kb"), 8));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "import_tracks/position", PROPERTY_HINT_ENUM, "IfPresent,IfPresentForAll,Never"), 1));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "import_tracks/rotation", PROPERTY_HINT_ENUM, "IfPresent,IfPresentForAll,Never"), 1));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "import_tracks/scale", PROPERTY_HINT_ENUM, "IfPresent,IfPresentForAll,Never"), 1));
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
		default: {
		}
	}

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		post_importer_plugins.write[i]->get_internal_import_options(EditorScenePostImportPlugin::InternalImportCategory(p_category), r_options);
	}
}

bool ResourceImporterScene::get_internal_option_visibility(InternalImportCategory p_category, const String &p_option, const Map<StringName, Variant> &p_options) const {
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

			if (
					p_option == "physics/body_type" ||
					p_option == "physics/shape_type") {
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
		} break;
		case INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE: {
			if (p_option.begins_with("optimizer/") && p_option != "optimizer/enabled" && !bool(p_options["optimizer/enabled"])) {
				return false;
			}
			if (p_option.begins_with("compression/") && p_option != "compression/enabled" && !bool(p_options["compression/enabled"])) {
				return false;
			}

			if (p_option.begins_with("slice_")) {
				int max_slice = p_options["slices/amount"];
				int slice = p_option.get_slice("_", 1).to_int() - 1;
				if (slice >= max_slice) {
					return false;
				}
			}
		} break;
		default: {
		}
	}

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		Variant ret = post_importer_plugins.write[i]->get_internal_option_visibility(EditorScenePostImportPlugin::InternalImportCategory(p_category), p_option, p_options);
		if (ret.get_type() == Variant::BOOL) {
			return ret;
		}
	}

	return true;
}

bool ResourceImporterScene::get_internal_option_update_view_required(InternalImportCategory p_category, const String &p_option, const Map<StringName, Variant> &p_options) const {
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
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_type", PROPERTY_HINT_TYPE_STRING, "Node"), "Node3D"));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_name"), "Scene Root"));

	List<String> script_extentions;
	ResourceLoader::get_recognized_extensions_for_type("Script", &script_extentions);

	String script_ext_hint;

	for (const String &E : script_extentions) {
		if (!script_ext_hint.is_empty()) {
			script_ext_hint += ",";
		}
		script_ext_hint += "*." + E;
	}

	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "nodes/root_scale", PROPERTY_HINT_RANGE, "0.001,1000,0.001"), 1.0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/ensure_tangents"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/generate_lods"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/create_shadow_meshes"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "meshes/light_baking", PROPERTY_HINT_ENUM, "Disabled,Static (VoxelGI/SDFGI),Static Lightmaps (VoxelGI/SDFGI/LightmapGI),Dynamic (VoxelGI only)", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "meshes/lightmap_texel_size", PROPERTY_HINT_RANGE, "0.001,100,0.001"), 0.1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "skins/use_named_skins"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/import"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "animation/fps", PROPERTY_HINT_RANGE, "1,120,1"), 30));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "import_script/path", PROPERTY_HINT_FILE, script_ext_hint), ""));

	r_options->push_back(ImportOption(PropertyInfo(Variant::DICTIONARY, "_subresources", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), Dictionary()));

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		post_importer_plugins.write[i]->get_import_options(p_path, r_options);
	}

	for (Ref<EditorSceneFormatImporter> importer : importers) {
		importer->get_import_options(p_path, r_options);
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

void ResourceImporterScene::_generate_meshes(Node *p_node, const Dictionary &p_mesh_data, bool p_generate_lods, bool p_create_shadow_meshes, LightBakeMode p_light_bake_mode, float p_lightmap_texel_size, const Vector<uint8_t> &p_src_lightmap_cache, Vector<Vector<uint8_t>> &r_lightmap_caches) {
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

				String mesh_id;

				if (src_mesh_node->get_mesh()->has_meta("import_id")) {
					mesh_id = src_mesh_node->get_mesh()->get_meta("import_id");
				} else {
					mesh_id = src_mesh_node->get_mesh()->get_name();
				}

				if (!mesh_id.is_empty() && p_mesh_data.has(mesh_id)) {
					Dictionary mesh_settings = p_mesh_data[mesh_id];

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

				if (generate_lods) {
					src_mesh_node->get_mesh()->generate_lods(merge_angle, split_angle);
				}

				if (create_shadow_meshes) {
					src_mesh_node->get_mesh()->create_shadow_mesh();
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

				if (!save_to_file.is_empty()) {
					Ref<Mesh> existing = Ref<Resource>(ResourceCache::get(save_to_file));
					if (existing.is_valid()) {
						//if somehow an existing one is useful, create
						existing->reset_state();
					}
					mesh = src_mesh_node->get_mesh()->get_mesh(existing);

					ResourceSaver::save(save_to_file, mesh); //override

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

		p_node->replace_by(mesh_node);
		memdelete(p_node);
		p_node = mesh_node;
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_generate_meshes(p_node->get_child(i), p_mesh_data, p_generate_lods, p_create_shadow_meshes, p_light_bake_mode, p_lightmap_texel_size, p_src_lightmap_cache, r_lightmap_caches);
	}
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
	ERR_FAIL_COND(parent == nullptr);
	OrderedHashMap<NodePath, uint32_t> used_tracks[TRACK_CHANNEL_MAX];
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

			for (OrderedHashMap<NodePath, uint32_t>::Element J = used_tracks[j].front(); J; J = J.next()) {
				if (J.get() == pass) {
					continue;
				}

				NodePath path = J.key();
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

Node *ResourceImporterScene::pre_import(const String &p_source_file) {
	Ref<EditorSceneFormatImporter> importer;
	String ext = p_source_file.get_extension().to_lower();

	EditorProgress progress("pre-import", TTR("Pre-Import Scene"), 0);
	progress.step(TTR("Importing Scene..."), 0);

	for (Set<Ref<EditorSceneFormatImporter>>::Element *E = importers.front(); E; E = E->next()) {
		List<String> extensions;
		E->get()->get_extensions(&extensions);

		for (const String &F : extensions) {
			if (F.to_lower() == ext) {
				importer = E->get();
				break;
			}
		}

		if (importer.is_valid()) {
			break;
		}
	}

	ERR_FAIL_COND_V(!importer.is_valid(), nullptr);

	Error err = OK;
	Node *scene = importer->import_scene(p_source_file, EditorSceneFormatImporter::IMPORT_ANIMATION | EditorSceneFormatImporter::IMPORT_GENERATE_TANGENT_ARRAYS, Map<StringName, Variant>(), 15, nullptr, &err);
	if (!scene || err != OK) {
		return nullptr;
	}

	Map<Ref<ImporterMesh>, Vector<Ref<Shape3D>>> collision_map;
	List<Pair<NodePath, Node *>> node_renames;
	_pre_fix_node(scene, scene, collision_map, node_renames);

	return scene;
}

Error ResourceImporterScene::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	const String &src_path = p_source_file;

	Ref<EditorSceneFormatImporter> importer;
	String ext = src_path.get_extension().to_lower();

	EditorProgress progress("import", TTR("Import Scene"), 104);
	progress.step(TTR("Importing Scene..."), 0);

	for (Set<Ref<EditorSceneFormatImporter>>::Element *E = importers.front(); E; E = E->next()) {
		List<String> extensions;
		E->get()->get_extensions(&extensions);

		for (const String &F : extensions) {
			if (F.to_lower() == ext) {
				importer = E->get();
				break;
			}
		}

		if (importer.is_valid()) {
			break;
		}
	}

	ERR_FAIL_COND_V(!importer.is_valid(), ERR_FILE_UNRECOGNIZED);

	float fps = p_options["animation/fps"];

	int import_flags = 0;

	if (bool(p_options["animation/import"])) {
		import_flags |= EditorSceneFormatImporter::IMPORT_ANIMATION;
	}

	if (bool(p_options["skins/use_named_skins"])) {
		import_flags |= EditorSceneFormatImporter::IMPORT_USE_NAMED_SKIN_BINDS;
	}

	bool ensure_tangents = p_options["meshes/ensure_tangents"];
	if (ensure_tangents) {
		import_flags |= EditorSceneFormatImporter::IMPORT_GENERATE_TANGENT_ARRAYS;
	}

	Error err = OK;
	List<String> missing_deps; // for now, not much will be done with this
	Node *scene = importer->import_scene(src_path, import_flags, p_options, fps, &missing_deps, &err);
	if (!scene || err != OK) {
		return err;
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

	Set<Ref<ImporterMesh>> scanned_meshes;
	Map<Ref<ImporterMesh>, Vector<Ref<Shape3D>>> collision_map;
	List<Pair<NodePath, Node *>> node_renames;

	_pre_fix_node(scene, scene, collision_map, node_renames);

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		post_importer_plugins.write[i]->pre_process(scene, p_options);
	}

	_post_fix_node(scene, scene, collision_map, scanned_meshes, node_data, material_data, animation_data, fps);

	String root_type = p_options["nodes/root_type"];
	root_type = root_type.split(" ")[0]; // full root_type is "ClassName (filename.gd)" for a script global class.

	Ref<Script> root_script = nullptr;
	if (ScriptServer::is_global_class(root_type)) {
		root_script = ResourceLoader::load(ScriptServer::get_global_class_path(root_type));
		root_type = ScriptServer::get_global_class_base(root_type);
	}

	if (root_type != "Node3D") {
		Node *base_node = Object::cast_to<Node>(ClassDB::instantiate(root_type));

		if (base_node) {
			scene->replace_by(base_node);
			memdelete(scene);
			scene = base_node;
		}
	}

	if (root_script.is_valid()) {
		scene->set_script(Variant(root_script));
	}

	float root_scale = 1.0;
	if (Object::cast_to<Node3D>(scene)) {
		root_scale = p_options["nodes/root_scale"];
		Object::cast_to<Node3D>(scene)->scale(Vector3(root_scale, root_scale, root_scale));
	}

	if (p_options["nodes/root_name"] != "Scene Root") {
		scene->set_name(p_options["nodes/root_name"]);
	} else {
		scene->set_name(p_save_path.get_file().get_basename());
	}

	bool gen_lods = bool(p_options["meshes/generate_lods"]);
	bool create_shadow_meshes = bool(p_options["meshes/create_shadow_meshes"]);
	int light_bake_mode = p_options["meshes/light_baking"];
	float texel_size = p_options["meshes/lightmap_texel_size"];
	float lightmap_texel_size = MAX(0.001, texel_size);

	Vector<uint8_t> src_lightmap_cache;
	Vector<Vector<uint8_t>> mesh_lightmap_caches;

	{
		src_lightmap_cache = FileAccess::get_file_as_array(p_source_file + ".unwrap_cache", &err);
		if (err != OK) {
			src_lightmap_cache.clear();
		}
	}

	Dictionary mesh_data;
	if (subresources.has("meshes")) {
		mesh_data = subresources["meshes"];
	}
	_generate_meshes(scene, mesh_data, gen_lods, create_shadow_meshes, LightBakeMode(light_bake_mode), lightmap_texel_size, src_lightmap_cache, mesh_lightmap_caches);

	if (mesh_lightmap_caches.size()) {
		FileAccessRef f = FileAccess::open(p_source_file + ".unwrap_cache", FileAccess::WRITE);
		if (f) {
			f->store_32(mesh_lightmap_caches.size());
			for (int i = 0; i < mesh_lightmap_caches.size(); i++) {
				String md5 = String::md5(mesh_lightmap_caches[i].ptr());
				f->store_buffer(mesh_lightmap_caches[i].ptr(), mesh_lightmap_caches[i].size());
			}
			f->close();
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

	Ref<PackedScene> packer = memnew(PackedScene);
	packer->pack(scene);
	print_verbose("Saving scene to: " + p_save_path + ".scn");
	err = ResourceSaver::save(p_save_path + ".scn", packer); //do not take over, let the changed files reload themselves
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save scene to file '" + p_save_path + ".scn'.");

	memdelete(scene);

	//this is not the time to reimport, wait until import process is done, import file is saved, etc.
	//EditorNode::get_singleton()->reload_scene(p_source_file);

	return OK;
}

ResourceImporterScene *ResourceImporterScene::singleton = nullptr;

bool ResourceImporterScene::ResourceImporterScene::has_advanced_options() const {
	return true;
}
void ResourceImporterScene::ResourceImporterScene::show_advanced_options(const String &p_path) {
	SceneImportSettings::get_singleton()->open_settings(p_path);
}

ResourceImporterScene::ResourceImporterScene() {
	singleton = this;
}

///////////////////////////////////////

uint32_t EditorSceneFormatImporterESCN::get_import_flags() const {
	return IMPORT_SCENE;
}

void EditorSceneFormatImporterESCN::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("escn");
}

Node *EditorSceneFormatImporterESCN::import_scene(const String &p_path, uint32_t p_flags, const Map<StringName, Variant> &p_options, int p_bake_fps, List<String> *r_missing_deps, Error *r_err) {
	Error error;
	Ref<PackedScene> ps = ResourceFormatLoaderText::singleton->load(p_path, p_path, &error);
	ERR_FAIL_COND_V_MSG(!ps.is_valid(), nullptr, "Cannot load scene as text resource from path '" + p_path + "'.");

	Node *scene = ps->instantiate();
	ERR_FAIL_COND_V(!scene, nullptr);

	return scene;
}

Ref<Animation> EditorSceneFormatImporterESCN::import_animation(const String &p_path, uint32_t p_flags, const Map<StringName, Variant> &p_options, int p_bake_fps) {
	ERR_FAIL_V(Ref<Animation>());
}
