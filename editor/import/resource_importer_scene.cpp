/*************************************************************************/
/*  resource_importer_scene.cpp                                          */
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

#include "resource_importer_scene.h"

#include "core/io/resource_saver.h"
#include "editor/editor_node.h"
#include "editor/import/scene_import_settings.h"
#include "editor/import/scene_importer_mesh_node_3d.h"
#include "scene/3d/area_3d.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/navigation_region_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/3d/vehicle_body_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/box_shape_3d.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/ray_shape_3d.h"
#include "scene/resources/resource_format_text.h"
#include "scene/resources/sphere_shape_3d.h"
#include "scene/resources/surface_tool.h"
#include "scene/resources/world_margin_shape_3d.h"

uint32_t EditorSceneImporter::get_import_flags() const {
	if (get_script_instance()) {
		return get_script_instance()->call("_get_import_flags");
	}

	ERR_FAIL_V(0);
}

void EditorSceneImporter::get_extensions(List<String> *r_extensions) const {
	if (get_script_instance()) {
		Array arr = get_script_instance()->call("_get_extensions");
		for (int i = 0; i < arr.size(); i++) {
			r_extensions->push_back(arr[i]);
		}
		return;
	}

	ERR_FAIL();
}

Node *EditorSceneImporter::import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err) {
	if (get_script_instance()) {
		return get_script_instance()->call("_import_scene", p_path, p_flags, p_bake_fps);
	}

	ERR_FAIL_V(nullptr);
}

Ref<Animation> EditorSceneImporter::import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps) {
	if (get_script_instance()) {
		return get_script_instance()->call("_import_animation", p_path, p_flags);
	}

	ERR_FAIL_V(nullptr);
}

//for documenters, these functions are useful when an importer calls an external conversion helper (like, fbx2gltf),
//and you want to load the resulting file

Node *EditorSceneImporter::import_scene_from_other_importer(const String &p_path, uint32_t p_flags, int p_bake_fps) {
	return ResourceImporterScene::get_singleton()->import_scene_from_other_importer(this, p_path, p_flags, p_bake_fps);
}

Ref<Animation> EditorSceneImporter::import_animation_from_other_importer(const String &p_path, uint32_t p_flags, int p_bake_fps) {
	return ResourceImporterScene::get_singleton()->import_animation_from_other_importer(this, p_path, p_flags, p_bake_fps);
}

void EditorSceneImporter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("import_scene_from_other_importer", "path", "flags", "bake_fps"), &EditorSceneImporter::import_scene_from_other_importer);
	ClassDB::bind_method(D_METHOD("import_animation_from_other_importer", "path", "flags", "bake_fps"), &EditorSceneImporter::import_animation_from_other_importer);

	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_import_flags"));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_extensions"));

	MethodInfo mi = MethodInfo(Variant::OBJECT, "_import_scene", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::INT, "flags"), PropertyInfo(Variant::INT, "bake_fps"));
	mi.return_val.class_name = "Node";
	BIND_VMETHOD(mi);
	mi = MethodInfo(Variant::OBJECT, "_import_animation", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::INT, "flags"), PropertyInfo(Variant::INT, "bake_fps"));
	mi.return_val.class_name = "Animation";
	BIND_VMETHOD(mi);

	BIND_CONSTANT(IMPORT_SCENE);
	BIND_CONSTANT(IMPORT_ANIMATION);
	BIND_CONSTANT(IMPORT_FAIL_ON_MISSING_DEPENDENCIES);
	BIND_CONSTANT(IMPORT_GENERATE_TANGENT_ARRAYS);
	BIND_CONSTANT(IMPORT_USE_NAMED_SKIN_BINDS);
}

/////////////////////////////////
void EditorScenePostImport::_bind_methods() {
	BIND_VMETHOD(MethodInfo(Variant::OBJECT, "post_import", PropertyInfo(Variant::OBJECT, "scene")));
	ClassDB::bind_method(D_METHOD("get_source_file"), &EditorScenePostImport::get_source_file);
}

Node *EditorScenePostImport::post_import(Node *p_scene) {
	if (get_script_instance()) {
		return get_script_instance()->call("post_import", p_scene);
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

String ResourceImporterScene::get_importer_name() const {
	return "scene";
}

String ResourceImporterScene::get_visible_name() const {
	return "Scene";
}

void ResourceImporterScene::get_recognized_extensions(List<String> *p_extensions) const {
	for (Set<Ref<EditorSceneImporter>>::Element *E = importers.front(); E; E = E->next()) {
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

bool ResourceImporterScene::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {
	if (p_option.begins_with("animation/")) {
		if (p_option != "animation/import" && !bool(p_options["animation/import"])) {
			return false;
		}
	}

	if (p_option == "meshes/lightmap_texel_size" && int(p_options["meshes/light_baking"]) < 3) {
		return false;
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

static void _gen_shape_list(const Ref<Mesh> &mesh, List<Ref<Shape3D>> &r_shape_list, bool p_convex) {
	ERR_FAIL_NULL_MSG(mesh, "Cannot generate shape list with null mesh value");
	if (!p_convex) {
		Ref<Shape3D> shape = mesh->create_trimesh_shape();
		r_shape_list.push_back(shape);
	} else {
		Vector<Ref<Shape3D>> cd = mesh->convex_decompose();
		if (cd.size()) {
			for (int i = 0; i < cd.size(); i++) {
				r_shape_list.push_back(cd[i]);
			}
		}
	}
}

static void _pre_gen_shape_list(const Ref<EditorSceneImporterMesh> &mesh, List<Ref<Shape3D>> &r_shape_list, bool p_convex) {
	ERR_FAIL_NULL_MSG(mesh, "Cannot generate shape list with null mesh value");
	if (!p_convex) {
		Ref<Shape3D> shape = mesh->create_trimesh_shape();
		r_shape_list.push_back(shape);
	} else {
		Vector<Ref<Shape3D>> cd = mesh->convex_decompose();
		if (cd.size()) {
			for (int i = 0; i < cd.size(); i++) {
				r_shape_list.push_back(cd[i]);
			}
		}
	}
}

Node *ResourceImporterScene::_pre_fix_node(Node *p_node, Node *p_root, Map<Ref<EditorSceneImporterMesh>, List<Ref<Shape3D>>> &collision_map) {
	// children first
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *r = _pre_fix_node(p_node->get_child(i), p_root, collision_map);
		if (!r) {
			i--; //was erased
		}
	}

	String name = p_node->get_name();

	bool isroot = p_node == p_root;

	if (!isroot && _teststr(name, "noimp")) {
		memdelete(p_node);
		return nullptr;
	}

	if (Object::cast_to<EditorSceneImporterMeshNode3D>(p_node)) {
		EditorSceneImporterMeshNode3D *mi = Object::cast_to<EditorSceneImporterMeshNode3D>(p_node);

		Ref<EditorSceneImporterMesh> m = mi->get_mesh();

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
		//remove animations referencing non-importable nodes
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(p_node);

		List<StringName> anims;
		ap->get_animation_list(&anims);
		for (List<StringName>::Element *E = anims.front(); E; E = E->next()) {
			Ref<Animation> anim = ap->get_animation(E->get());
			ERR_CONTINUE(anim.is_null());
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

			String animname = E->get();
			const int loop_string_count = 3;
			static const char *loop_strings[loop_string_count] = { "loops", "loop", "cycle" };
			for (int i = 0; i < loop_string_count; i++) {
				if (_teststr(animname, loop_strings[i])) {
					anim->set_loop(true);
					animname = _fixstr(animname, loop_strings[i]);
					ap->rename_animation(E->get(), animname);
				}
			}
		}
	}

	if (_teststr(name, "colonly") || _teststr(name, "convcolonly")) {
		if (isroot) {
			return p_node;
		}
		EditorSceneImporterMeshNode3D *mi = Object::cast_to<EditorSceneImporterMeshNode3D>(p_node);
		if (mi) {
			Ref<EditorSceneImporterMesh> mesh = mi->get_mesh();

			if (mesh.is_valid()) {
				List<Ref<Shape3D>> shapes;
				String fixed_name;
				if (collision_map.has(mesh)) {
					shapes = collision_map[mesh];
				} else if (_teststr(name, "colonly")) {
					_pre_gen_shape_list(mesh, shapes, false);
					collision_map[mesh] = shapes;
				} else if (_teststr(name, "convcolonly")) {
					_pre_gen_shape_list(mesh, shapes, true);
					collision_map[mesh] = shapes;
				}

				if (_teststr(name, "colonly")) {
					fixed_name = _fixstr(name, "colonly");
				} else if (_teststr(name, "convcolonly")) {
					fixed_name = _fixstr(name, "convcolonly");
				}

				ERR_FAIL_COND_V(fixed_name == String(), nullptr);

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
			sb->set_name(_fixstr(name, "colonly"));
			Object::cast_to<Node3D>(sb)->set_transform(Object::cast_to<Node3D>(p_node)->get_transform());
			p_node->replace_by(sb);
			memdelete(p_node);
			p_node = nullptr;
			CollisionShape3D *colshape = memnew(CollisionShape3D);
			if (empty_draw_type == "CUBE") {
				BoxShape3D *boxShape = memnew(BoxShape3D);
				boxShape->set_size(Vector3(2, 2, 2));
				colshape->set_shape(boxShape);
			} else if (empty_draw_type == "SINGLE_ARROW") {
				RayShape3D *rayShape = memnew(RayShape3D);
				rayShape->set_length(1);
				colshape->set_shape(rayShape);
				Object::cast_to<Node3D>(sb)->rotate_x(Math_PI / 2);
			} else if (empty_draw_type == "IMAGE") {
				WorldMarginShape3D *world_margin_shape = memnew(WorldMarginShape3D);
				colshape->set_shape(world_margin_shape);
			} else {
				SphereShape3D *sphereShape = memnew(SphereShape3D);
				sphereShape->set_radius(1);
				colshape->set_shape(sphereShape);
			}
			sb->add_child(colshape);
			colshape->set_owner(sb->get_owner());
		}

	} else if (_teststr(name, "rigid") && Object::cast_to<EditorSceneImporterMeshNode3D>(p_node)) {
		if (isroot) {
			return p_node;
		}

		EditorSceneImporterMeshNode3D *mi = Object::cast_to<EditorSceneImporterMeshNode3D>(p_node);
		Ref<EditorSceneImporterMesh> mesh = mi->get_mesh();

		if (mesh.is_valid()) {
			List<Ref<Shape3D>> shapes;
			if (collision_map.has(mesh)) {
				shapes = collision_map[mesh];
			} else {
				_gen_shape_list(mesh, shapes, true);
			}

			RigidBody3D *rigid_body = memnew(RigidBody3D);
			rigid_body->set_name(_fixstr(name, "rigid"));
			p_node->replace_by(rigid_body);
			rigid_body->set_transform(mi->get_transform());
			p_node = rigid_body;
			mi->set_transform(Transform());
			rigid_body->add_child(mi);
			mi->set_owner(rigid_body->get_owner());

			_add_shapes(rigid_body, shapes);
		}

	} else if ((_teststr(name, "col") || (_teststr(name, "convcol"))) && Object::cast_to<EditorSceneImporterMeshNode3D>(p_node)) {
		EditorSceneImporterMeshNode3D *mi = Object::cast_to<EditorSceneImporterMeshNode3D>(p_node);

		Ref<EditorSceneImporterMesh> mesh = mi->get_mesh();

		if (mesh.is_valid()) {
			List<Ref<Shape3D>> shapes;
			String fixed_name;
			if (collision_map.has(mesh)) {
				shapes = collision_map[mesh];
			} else if (_teststr(name, "col")) {
				_gen_shape_list(mesh, shapes, false);
				collision_map[mesh] = shapes;
			} else if (_teststr(name, "convcol")) {
				_gen_shape_list(mesh, shapes, true);
				collision_map[mesh] = shapes;
			}

			if (_teststr(name, "col")) {
				fixed_name = _fixstr(name, "col");
			} else if (_teststr(name, "convcol")) {
				fixed_name = _fixstr(name, "convcol");
			}

			if (fixed_name != String()) {
				if (mi->get_parent() && !mi->get_parent()->has_node(fixed_name)) {
					mi->set_name(fixed_name);
				}
			}

			if (shapes.size()) {
				StaticBody3D *col = memnew(StaticBody3D);
				mi->add_child(col);
				col->set_owner(mi->get_owner());

				_add_shapes(col, shapes);
			}
		}

	} else if (_teststr(name, "navmesh") && Object::cast_to<EditorSceneImporterMeshNode3D>(p_node)) {
		if (isroot) {
			return p_node;
		}

		EditorSceneImporterMeshNode3D *mi = Object::cast_to<EditorSceneImporterMeshNode3D>(p_node);

		Ref<EditorSceneImporterMesh> mesh = mi->get_mesh();
		ERR_FAIL_COND_V(mesh.is_null(), nullptr);
		NavigationRegion3D *nmi = memnew(NavigationRegion3D);

		nmi->set_name(_fixstr(name, "navmesh"));
		Ref<NavigationMesh> nmesh = mesh->create_navigation_mesh();
		nmi->set_navigation_mesh(nmesh);
		Object::cast_to<Node3D>(nmi)->set_transform(mi->get_transform());
		p_node->replace_by(nmi);
		memdelete(p_node);
		p_node = nmi;

	} else if (Object::cast_to<EditorSceneImporterMeshNode3D>(p_node)) {
		//last attempt, maybe collision inside the mesh data

		EditorSceneImporterMeshNode3D *mi = Object::cast_to<EditorSceneImporterMeshNode3D>(p_node);

		Ref<EditorSceneImporterMesh> mesh = mi->get_mesh();
		if (!mesh.is_null()) {
			List<Ref<Shape3D>> shapes;
			if (collision_map.has(mesh)) {
				shapes = collision_map[mesh];
			} else if (_teststr(mesh->get_name(), "col")) {
				_gen_shape_list(mesh, shapes, false);
				collision_map[mesh] = shapes;
				mesh->set_name(_fixstr(mesh->get_name(), "col"));
			} else if (_teststr(mesh->get_name(), "convcol")) {
				_gen_shape_list(mesh, shapes, true);
				collision_map[mesh] = shapes;
				mesh->set_name(_fixstr(mesh->get_name(), "convcol"));
			}

			if (shapes.size()) {
				StaticBody3D *col = memnew(StaticBody3D);
				p_node->add_child(col);
				col->set_owner(p_node->get_owner());

				_add_shapes(col, shapes);
			}
		}
	}

	return p_node;
}

Node *ResourceImporterScene::_post_fix_node(Node *p_node, Node *p_root, Map<Ref<EditorSceneImporterMesh>, List<Ref<Shape3D>>> &collision_map, Set<Ref<EditorSceneImporterMesh>> &r_scanned_meshes, const Dictionary &p_node_data, const Dictionary &p_material_data, const Dictionary &p_animation_data, float p_animation_fps) {
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

	if (Object::cast_to<EditorSceneImporterMeshNode3D>(p_node)) {
		EditorSceneImporterMeshNode3D *mi = Object::cast_to<EditorSceneImporterMeshNode3D>(p_node);

		Ref<EditorSceneImporterMesh> m = mi->get_mesh();

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

						if (mat_id != String() && p_material_data.has(mat_id)) {
							Dictionary matdata = p_material_data[mat_id];
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
				int mesh_physics_mode = node_settings["generate/physics"];

				if (mesh_physics_mode != MESH_PHYSICS_DISABLED) {
					List<Ref<Shape3D>> shapes;

					if (collision_map.has(m)) {
						shapes = collision_map[m];
					} else {
						switch (mesh_physics_mode) {
							case MESH_PHYSICS_MESH_AND_STATIC_COLLIDER: {
								_pre_gen_shape_list(m, shapes, false);
							} break;
							case MESH_PHYSICS_RIGID_BODY_AND_MESH: {
								_pre_gen_shape_list(m, shapes, true);
							} break;
							case MESH_PHYSICS_STATIC_COLLIDER_ONLY: {
								_pre_gen_shape_list(m, shapes, false);
							} break;
							case MESH_PHYSICS_AREA_ONLY: {
								_pre_gen_shape_list(m, shapes, true);
							} break;
						}
					}

					if (shapes.size()) {
						CollisionObject3D *base = nullptr;
						switch (mesh_physics_mode) {
							case MESH_PHYSICS_MESH_AND_STATIC_COLLIDER: {
								StaticBody3D *col = memnew(StaticBody3D);
								p_node->add_child(col);
								base = col;
							} break;
							case MESH_PHYSICS_RIGID_BODY_AND_MESH: {
								RigidBody3D *rigid_body = memnew(RigidBody3D);
								rigid_body->set_name(p_node->get_name());
								p_node->replace_by(rigid_body);
								rigid_body->set_transform(mi->get_transform());
								p_node = rigid_body;
								mi->set_transform(Transform());
								rigid_body->add_child(mi);
								mi->set_owner(rigid_body->get_owner());
								base = rigid_body;
							} break;
							case MESH_PHYSICS_STATIC_COLLIDER_ONLY: {
								StaticBody3D *col = memnew(StaticBody3D);
								col->set_transform(mi->get_transform());
								col->set_name(p_node->get_name());
								p_node->replace_by(col);
								memdelete(p_node);
								p_node = col;
								base = col;
							} break;
							case MESH_PHYSICS_AREA_ONLY: {
								Area3D *area = memnew(Area3D);
								area->set_transform(mi->get_transform());
								area->set_name(p_node->get_name());
								p_node->replace_by(area);
								memdelete(p_node);
								p_node = area;
								base = area;

							} break;
						}

						int idx = 0;
						for (List<Ref<Shape3D>>::Element *E = shapes.front(); E; E = E->next()) {
							CollisionShape3D *cshape = memnew(CollisionShape3D);
							cshape->set_shape(E->get());
							base->add_child(cshape);

							cshape->set_owner(base->get_owner());
							idx++;
						}
					}
				}
			}
		}
	}

	//navmesh (node may have changed type above)
	if (Object::cast_to<EditorSceneImporterMeshNode3D>(p_node)) {
		EditorSceneImporterMeshNode3D *mi = Object::cast_to<EditorSceneImporterMeshNode3D>(p_node);

		Ref<EditorSceneImporterMesh> m = mi->get_mesh();

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
						mi->add_child(nmi);
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
			for (List<ImportOption>::Element *E = iopts.front(); E; E = E->next()) {
				if (!node_settings.has(E->get().option.name)) {
					node_settings[E->get().option.name] = E->get().default_value;
				}
			}
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
				bool loop = node_settings["clip_" + itos(i + 1) + "/loops"];
				bool save_to_file = node_settings["clip_" + itos(i + 1) + "/save_to_file/enabled"];
				bool save_to_path = node_settings["clip_" + itos(i + 1) + "/save_to_file/path"];
				bool save_to_file_keep_custom = node_settings["clip_" + itos(i + 1) + "/save_to_file/keep_custom_tracks"];

				animation_clips.push_back(name);
				animation_clips.push_back(from_frame / p_animation_fps);
				animation_clips.push_back(end_frame / p_animation_fps);
				animation_clips.push_back(loop);
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
			for (List<StringName>::Element *E = anims.front(); E; E = E->next()) {
				String name = E->get();
				Ref<Animation> anim = ap->get_animation(name);
				if (p_animation_data.has(name)) {
					Dictionary anim_settings = p_animation_data[name];
					{
						//fill with default values
						List<ImportOption> iopts;
						get_internal_import_options(INTERNAL_IMPORT_CATEGORY_ANIMATION, &iopts);
						for (List<ImportOption>::Element *F = iopts.front(); F; F = F->next()) {
							if (!anim_settings.has(F->get().option.name)) {
								anim_settings[F->get().option.name] = F->get().default_value;
							}
						}
					}

					anim->set_loop(anim_settings["settings/loops"]);
					bool save = anim_settings["save_to_file/enabled"];
					String path = anim_settings["save_to_file/path"];
					bool keep_custom = anim_settings["save_to_file/keep_custom_tracks"];

					Ref<Animation> saved_anim = _save_animation_to_file(anim, save, path, keep_custom);

					if (saved_anim != anim) {
						ap->add_animation(name, saved_anim); //replace
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
			anim->set_loop(old_anim->has_loop());
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
		return;
	}

	Ref<Animation> default_anim = anim->get_animation("default");

	for (int i = 0; i < p_clips.size(); i += 7) {
		String name = p_clips[i];
		float from = p_clips[i + 1];
		float to = p_clips[i + 2];
		bool loop = p_clips[i + 3];
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
							if (default_anim->track_get_type(j) == Animation::TYPE_TRANSFORM) {
								Quat q;
								Vector3 p;
								Vector3 s;
								default_anim->transform_track_interpolate(j, from, &p, &q, &s);
								new_anim->transform_track_insert_key(dtrack, 0, p, q, s);
							}
							if (default_anim->track_get_type(j) == Animation::TYPE_VALUE) {
								Variant var = default_anim->value_track_interpolate(j, from);
								new_anim->track_insert_key(dtrack, 0, var);
							}
						}
					}

					if (default_anim->track_get_type(j) == Animation::TYPE_TRANSFORM) {
						Quat q;
						Vector3 p;
						Vector3 s;
						default_anim->transform_track_get_key(j, k, &p, &q, &s);
						new_anim->transform_track_insert_key(dtrack, kt - from, p, q, s);
					}
					if (default_anim->track_get_type(j) == Animation::TYPE_VALUE) {
						Variant var = default_anim->track_get_key_value(j, k);
						new_anim->track_insert_key(dtrack, kt - from, var);
					}
				}

				if (dtrack != -1 && kt >= to) {
					if (default_anim->track_get_type(j) == Animation::TYPE_TRANSFORM) {
						Quat q;
						Vector3 p;
						Vector3 s;
						default_anim->transform_track_interpolate(j, to, &p, &q, &s);
						new_anim->transform_track_insert_key(dtrack, to - from, p, q, s);
					}
					if (default_anim->track_get_type(j) == Animation::TYPE_VALUE) {
						Variant var = default_anim->value_track_interpolate(j, to);
						new_anim->track_insert_key(dtrack, to - from, var);
					}
				}
			}

			if (dtrack == -1 && p_bake_all) {
				new_anim->add_track(default_anim->track_get_type(j));
				dtrack = new_anim->get_track_count() - 1;
				new_anim->track_set_path(dtrack, default_anim->track_get_path(j));
				if (default_anim->track_get_type(j) == Animation::TYPE_TRANSFORM) {
					Quat q;
					Vector3 p;
					Vector3 s;
					default_anim->transform_track_interpolate(j, from, &p, &q, &s);
					new_anim->transform_track_insert_key(dtrack, 0, p, q, s);
					default_anim->transform_track_interpolate(j, to, &p, &q, &s);
					new_anim->transform_track_insert_key(dtrack, to - from, p, q, s);
				}
				if (default_anim->track_get_type(j) == Animation::TYPE_VALUE) {
					Variant var = default_anim->value_track_interpolate(j, from);
					new_anim->track_insert_key(dtrack, 0, var);
					Variant to_var = default_anim->value_track_interpolate(j, to);
					new_anim->track_insert_key(dtrack, to - from, to_var);
				}
			}
		}

		new_anim->set_loop(loop);
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
	for (List<StringName>::Element *E = anim_names.front(); E; E = E->next()) {
		Ref<Animation> a = anim->get_animation(E->get());
		a->optimize(p_max_lin_error, p_max_ang_error, Math::deg2rad(p_max_angle));
	}
}

void ResourceImporterScene::get_internal_import_options(InternalImportCategory p_category, List<ImportOption> *r_options) const {
	switch (p_category) {
		case INTERNAL_IMPORT_CATEGORY_NODE: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "import/skip_import", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
		} break;
		case INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "import/skip_import", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/physics", PROPERTY_HINT_ENUM, "Disabled,Mesh + Static Collider,Rigid Body + Mesh,Static Collider Only,Area Only"), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/navmesh", PROPERTY_HINT_ENUM, "Disabled,Mesh + NavMesh,NavMesh Only"), 0));
		} break;
		case INTERNAL_IMPORT_CATEGORY_MESH: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "save_to_file/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "save_to_file/path", PROPERTY_HINT_SAVE_FILE, "*.res,*.tres"), ""));
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "save_to_file/make_streamable"), ""));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/shadow_meshes", PROPERTY_HINT_ENUM, "Default,Enable,Disable"), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/lightmap_uv", PROPERTY_HINT_ENUM, "Default,Enable,Disable"), 0));
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "generate/lods", PROPERTY_HINT_ENUM, "Default,Enable,Disable"), 0));
		} break;
		case INTERNAL_IMPORT_CATEGORY_MATERIAL: {
			r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "use_external/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
			r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "use_external/path", PROPERTY_HINT_FILE, "*.material,*.res,*.tres"), ""));
		} break;
		case INTERNAL_IMPORT_CATEGORY_ANIMATION: {
			r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "settings/loops"), false));
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
			r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "slices/amount", PROPERTY_HINT_RANGE, "0,256,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));

			for (int i = 0; i < 256; i++) {
				r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "slice_" + itos(i + 1) + "/name"), ""));
				r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "slice_" + itos(i + 1) + "/start_frame"), 0));
				r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "slice_" + itos(i + 1) + "/end_frame"), 0));
				r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "slice_" + itos(i + 1) + "/loops"), false));
				r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "slice_" + itos(i + 1) + "/save_to_file/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
				r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "slice_" + itos(i + 1) + "/save_to_file/path", PROPERTY_HINT_SAVE_FILE, ".res,*.tres"), ""));
				r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "slice_" + itos(i + 1) + "/save_to_file/keep_custom_tracks"), false));
			}
		} break;
		default: {
		}
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
			if (p_option.begins_with("animation/optimizer/") && p_option != "animation/optimizer/enabled" && !bool(p_options["animation/optimizer/enabled"])) {
				return false;
			}

			if (p_option.begins_with("animation/slice_")) {
				int max_slice = p_options["animation/slices/amount"];
				int slice = p_option.get_slice("/", 1).get_slice("_", 1).to_int() - 1;
				if (slice >= max_slice) {
					return false;
				}
			}
		} break;
		default: {
		}
	}

	return true;
}

void ResourceImporterScene::get_import_options(List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_type", PROPERTY_HINT_TYPE_STRING, "Node"), "Node3D"));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_name"), "Scene Root"));

	List<String> script_extentions;
	ResourceLoader::get_recognized_extensions_for_type("Script", &script_extentions);

	String script_ext_hint;

	for (List<String>::Element *E = script_extentions.front(); E; E = E->next()) {
		if (script_ext_hint != "") {
			script_ext_hint += ",";
		}
		script_ext_hint += "*." + E->get();
	}

	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "nodes/root_scale", PROPERTY_HINT_RANGE, "0.001,1000,0.001"), 1.0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/ensure_tangents"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/generate_lods"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/create_shadow_meshes"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "meshes/light_baking", PROPERTY_HINT_ENUM, "Disabled,Dynamic,Static,Static Lightmaps", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 2));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "meshes/lightmap_texel_size", PROPERTY_HINT_RANGE, "0.001,100,0.001"), 0.1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "skins/use_named_skins"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/import"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "animation/fps", PROPERTY_HINT_RANGE, "1,120,1"), 15));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "import_script/path", PROPERTY_HINT_FILE, script_ext_hint), ""));

	r_options->push_back(ImportOption(PropertyInfo(Variant::DICTIONARY, "_subresources", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), Dictionary()));
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

Node *ResourceImporterScene::import_scene_from_other_importer(EditorSceneImporter *p_exception, const String &p_path, uint32_t p_flags, int p_bake_fps) {
	Ref<EditorSceneImporter> importer;
	String ext = p_path.get_extension().to_lower();

	for (Set<Ref<EditorSceneImporter>>::Element *E = importers.front(); E; E = E->next()) {
		if (E->get().ptr() == p_exception) {
			continue;
		}
		List<String> extensions;
		E->get()->get_extensions(&extensions);

		for (List<String>::Element *F = extensions.front(); F; F = F->next()) {
			if (F->get().to_lower() == ext) {
				importer = E->get();
				break;
			}
		}

		if (importer.is_valid()) {
			break;
		}
	}

	ERR_FAIL_COND_V(!importer.is_valid(), nullptr);

	List<String> missing;
	Error err;
	return importer->import_scene(p_path, p_flags, p_bake_fps, &missing, &err);
}

Ref<Animation> ResourceImporterScene::import_animation_from_other_importer(EditorSceneImporter *p_exception, const String &p_path, uint32_t p_flags, int p_bake_fps) {
	Ref<EditorSceneImporter> importer;
	String ext = p_path.get_extension().to_lower();

	for (Set<Ref<EditorSceneImporter>>::Element *E = importers.front(); E; E = E->next()) {
		if (E->get().ptr() == p_exception) {
			continue;
		}
		List<String> extensions;
		E->get()->get_extensions(&extensions);

		for (List<String>::Element *F = extensions.front(); F; F = F->next()) {
			if (F->get().to_lower() == ext) {
				importer = E->get();
				break;
			}
		}

		if (importer.is_valid()) {
			break;
		}
	}

	ERR_FAIL_COND_V(!importer.is_valid(), nullptr);

	return importer->import_animation(p_path, p_flags, p_bake_fps);
}

void ResourceImporterScene::_generate_meshes(Node *p_node, const Dictionary &p_mesh_data, bool p_generate_lods, bool p_create_shadow_meshes, LightBakeMode p_light_bake_mode, float p_lightmap_texel_size, const Vector<uint8_t> &p_src_lightmap_cache, Vector<Vector<uint8_t>> &r_lightmap_caches) {
	EditorSceneImporterMeshNode3D *src_mesh_node = Object::cast_to<EditorSceneImporterMeshNode3D>(p_node);
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
				bool create_shadow_meshes = p_create_shadow_meshes;
				bool bake_lightmaps = p_light_bake_mode == LIGHT_BAKE_STATIC_LIGHTMAPS;
				String save_to_file;

				String mesh_id;

				if (src_mesh_node->get_mesh()->has_meta("import_id")) {
					mesh_id = src_mesh_node->get_mesh()->get_meta("import_id");
				} else {
					mesh_id = src_mesh_node->get_mesh()->get_name();
				}

				if (mesh_id != String() && p_mesh_data.has(mesh_id)) {
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

					if (mesh_settings.has("save_to_file/enabled") && bool(mesh_settings["save_to_file/enabled"]) && mesh_settings.has("save_to_file/path")) {
						save_to_file = mesh_settings["save_to_file/path"];
						if (!save_to_file.is_resource_file()) {
							save_to_file = "";
						}
					}
				}

				if (generate_lods) {
					src_mesh_node->get_mesh()->generate_lods();
				}
				if (create_shadow_meshes) {
					src_mesh_node->get_mesh()->create_shadow_mesh();
				}

				if (bake_lightmaps) {
					Transform xf;
					Node3D *n = src_mesh_node;
					while (n) {
						xf = n->get_transform() * xf;
						n = n->get_parent_spatial();
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

				if (save_to_file != String()) {
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
				mesh_node->set_gi_mode(GeometryInstance3D::GI_MODE_BAKED);
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

void ResourceImporterScene::_add_shapes(Node *p_node, const List<Ref<Shape3D>> &p_shapes) {
	for (const List<Ref<Shape3D>>::Element *E = p_shapes.front(); E; E = E->next()) {
		CollisionShape3D *cshape = memnew(CollisionShape3D);
		cshape->set_shape(E->get());
		p_node->add_child(cshape);

		cshape->set_owner(p_node->get_owner());
	}
}

Node *ResourceImporterScene::pre_import(const String &p_source_file) {
	Ref<EditorSceneImporter> importer;
	String ext = p_source_file.get_extension().to_lower();

	EditorProgress progress("pre-import", TTR("Pre-Import Scene"), 0);
	progress.step(TTR("Importing Scene..."), 0);

	for (Set<Ref<EditorSceneImporter>>::Element *E = importers.front(); E; E = E->next()) {
		List<String> extensions;
		E->get()->get_extensions(&extensions);

		for (List<String>::Element *F = extensions.front(); F; F = F->next()) {
			if (F->get().to_lower() == ext) {
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
	Node *scene = importer->import_scene(p_source_file, EditorSceneImporter::IMPORT_ANIMATION | EditorSceneImporter::IMPORT_GENERATE_TANGENT_ARRAYS, 15, nullptr, &err);
	if (!scene || err != OK) {
		return nullptr;
	}

	Map<Ref<EditorSceneImporterMesh>, List<Ref<Shape3D>>> collision_map;

	_pre_fix_node(scene, scene, collision_map);

	return scene;
}

Error ResourceImporterScene::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	const String &src_path = p_source_file;

	Ref<EditorSceneImporter> importer;
	String ext = src_path.get_extension().to_lower();

	EditorProgress progress("import", TTR("Import Scene"), 104);
	progress.step(TTR("Importing Scene..."), 0);

	for (Set<Ref<EditorSceneImporter>>::Element *E = importers.front(); E; E = E->next()) {
		List<String> extensions;
		E->get()->get_extensions(&extensions);

		for (List<String>::Element *F = extensions.front(); F; F = F->next()) {
			if (F->get().to_lower() == ext) {
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
		import_flags |= EditorSceneImporter::IMPORT_ANIMATION;
	}

	if (bool(p_options["skins/use_named_skins"])) {
		import_flags |= EditorSceneImporter::IMPORT_USE_NAMED_SKIN_BINDS;
	}

	bool ensure_tangents = p_options["meshes/ensure_tangents"];
	if (ensure_tangents) {
		import_flags |= EditorSceneImporter::IMPORT_GENERATE_TANGENT_ARRAYS;
	}

	Error err = OK;
	List<String> missing_deps; // for now, not much will be done with this
	Node *scene = importer->import_scene(src_path, import_flags, fps, &missing_deps, &err);
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

	Set<Ref<EditorSceneImporterMesh>> scanned_meshes;
	Map<Ref<EditorSceneImporterMesh>, List<Ref<Shape3D>>> collision_map;

	_pre_fix_node(scene, scene, collision_map);
	_post_fix_node(scene, scene, collision_map, scanned_meshes, node_data, material_data, animation_data, fps);

	String root_type = p_options["nodes/root_type"];
	root_type = root_type.split(" ")[0]; // full root_type is "ClassName (filename.gd)" for a script global class.

	Ref<Script> root_script = nullptr;
	if (ScriptServer::is_global_class(root_type)) {
		root_script = ResourceLoader::load(ScriptServer::get_global_class_path(root_type));
		root_type = ScriptServer::get_global_class_base(root_type);
	}

	if (root_type != "Node3D") {
		Node *base_node = Object::cast_to<Node>(ClassDB::instance(root_type));

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

	if (post_import_script_path != "") {
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
					TTR("Did you return a Node-derived object in the `post_import()` method?"));
			return err;
		}
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

uint32_t EditorSceneImporterESCN::get_import_flags() const {
	return IMPORT_SCENE;
}

void EditorSceneImporterESCN::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("escn");
}

Node *EditorSceneImporterESCN::import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err) {
	Error error;
	Ref<PackedScene> ps = ResourceFormatLoaderText::singleton->load(p_path, p_path, &error);
	ERR_FAIL_COND_V_MSG(!ps.is_valid(), nullptr, "Cannot load scene as text resource from path '" + p_path + "'.");

	Node *scene = ps->instance();
	ERR_FAIL_COND_V(!scene, nullptr);

	return scene;
}

Ref<Animation> EditorSceneImporterESCN::import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps) {
	ERR_FAIL_V(Ref<Animation>());
}
