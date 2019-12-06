/*************************************************************************/
/*  resource_importer_scene.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "resource_saver_scene.h"

#include "core/io/resource_saver.h"
#include "editor/editor_node.h"
#include "scene/resources/packed_scene.h"

#include "scene/3d/collision_shape.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/navigation.h"
#include "scene/3d/physics_body.h"
#include "scene/3d/portal.h"
#include "scene/3d/room_instance.h"
#include "scene/3d/vehicle_body.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/plane_shape.h"
#include "scene/resources/ray_shape.h"
#include "scene/resources/resource_format_text.h"
#include "scene/resources/sphere_shape.h"

uint32_t EditorSceneExporter::get_save_flags() const {

	if (get_script_instance()) {
		return get_script_instance()->call("_get_import_flags");
	}

	ERR_FAIL_V(0);
}
void EditorSceneExporter::get_exporter_extensions(List<String> *r_extensions) const {

	if (get_script_instance()) {
		Array arr = get_script_instance()->call("_get_extensions");
		for (int i = 0; i < arr.size(); i++) {
			r_extensions->push_back(arr[i]);
		}
		return;
	}

	ERR_FAIL();
}
void EditorSceneExporter::save_scene(Node *p_node, const String &p_path, const String &p_src_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err) {
	Error err = Error::FAILED;
	if (get_script_instance()) {
		get_script_instance()->call("_save_scene", p_node, p_path, p_src_path, p_flags, p_bake_fps);
		err = Error::OK;
	}
	if (r_err) {
		*r_err = err;
	}
}

void EditorSceneExporter::save_animation(Node *p_node, const String &p_path, const String &p_src_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err) {
	Error err = Error::FAILED;
	if (get_script_instance()) {
		get_script_instance()->call("_save_animation", p_node, p_path, p_flags);
		err = OK;
	}
	if (r_err) {
		*r_err = err;
	}
}

void EditorSceneExporter::_bind_methods() {
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_import_flags"));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_extensions"));

	MethodInfo mi = MethodInfo(Variant::OBJECT, "_import_scene", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::INT, "flags"), PropertyInfo(Variant::INT, "bake_fps"));
	mi.return_val.class_name = "Node";
	BIND_VMETHOD(mi);
	mi = MethodInfo(Variant::OBJECT, "_import_animation", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::INT, "flags"), PropertyInfo(Variant::INT, "bake_fps"));
	mi.return_val.class_name = "Animation";
	BIND_VMETHOD(mi);
}

String ResourceExporterScene::get_exporter_name() const {

	return "scene";
}

String ResourceExporterScene::get_visible_name() const {

	return "Scene";
}

void ResourceExporterScene::get_recognized_extensions(List<String> *p_extensions) const {
	for (Set<Ref<EditorSceneExporter> >::Element *E = exporters.front(); E; E = E->next()) {
		E->get()->get_exporter_extensions(p_extensions);
	}
}

String ResourceExporterScene::get_save_extension() const {
	return "scn";
}

String ResourceExporterScene::get_resource_type() const {

	return "PackedScene";
}

bool ResourceExporterScene::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

	if (p_option.begins_with("animation/")) {
		if (p_option != "animation/import" && !bool(p_options["animation/import"]))
			return false;

		if (p_option == "animation/keep_custom_tracks" && int(p_options["animation/storage"]) == 0)
			return false;

		if (p_option.begins_with("animation/optimizer/") && p_option != "animation/optimizer/enabled" && !bool(p_options["animation/optimizer/enabled"]))
			return false;

		if (p_option.begins_with("animation/clip_")) {
			int max_clip = p_options["animation/clips/amount"];
			int clip = p_option.get_slice("/", 1).get_slice("_", 1).to_int() - 1;
			if (clip >= max_clip)
				return false;
		}
	}

	if (p_option == "materials/keep_on_reimport" && int(p_options["materials/storage"]) == 0) {
		return false;
	}

	if (p_option == "meshes/lightmap_texel_size" && int(p_options["meshes/light_baking"]) < 2) {
		return false;
	}

	return true;
}

int ResourceExporterScene::get_preset_count() const {
	return PRESET_MAX;
}
String ResourceExporterScene::get_preset_name(int p_idx) const {

	//switch (p_idx) {
	//case PRESET_SINGLE_SCENE: return TTR("Import as Single Scene");
	//case PRESET_SEPARATE_ANIMATIONS: return TTR("Import with Separate Animations");
	//case PRESET_SEPARATE_MATERIALS: return TTR("Import with Separate Materials");
	//case PRESET_SEPARATE_MESHES: return TTR("Import with Separate Objects");
	//case PRESET_SEPARATE_MESHES_AND_MATERIALS: return TTR("Import with Separate Objects+Materials");
	//case PRESET_SEPARATE_MESHES_AND_ANIMATIONS: return TTR("Import with Separate Objects+Animations");
	//case PRESET_SEPARATE_MATERIALS_AND_ANIMATIONS: return TTR("Import with Separate Materials+Animations");
	//case PRESET_SEPARATE_MESHES_MATERIALS_AND_ANIMATIONS: return TTR("Import with Separate Objects+Materials+Animations");
	//case PRESET_MULTIPLE_SCENES: return TTR("Import as Multiple Scenes");
	//case PRESET_MULTIPLE_SCENES_AND_MATERIALS: return TTR("Import as Multiple Scenes+Materials");
	//}

	return "";
}

void ResourceExporterScene::_create_clips(Node *scene, const Array &p_clips, bool p_bake_all) {

	if (!scene->has_node(String("AnimationPlayer")))
		return;

	Node *n = scene->get_node(String("AnimationPlayer"));
	ERR_FAIL_COND(!n);
	AnimationPlayer *anim = Object::cast_to<AnimationPlayer>(n);
	ERR_FAIL_COND(!anim);

	if (!anim->has_animation("default"))
		return;

	Ref<Animation> default_anim = anim->get_animation("default");

	for (int i = 0; i < p_clips.size(); i += 4) {

		String name = p_clips[i];
		float from = p_clips[i + 1];
		float to = p_clips[i + 2];
		bool loop = p_clips[i + 3];
		if (from >= to)
			continue;

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
	}

	anim->remove_animation("default"); //remove default (no longer needed)
}

void ResourceExporterScene::_filter_anim_tracks(Ref<Animation> anim, Set<String> &keep) {

	Ref<Animation> a = anim;
	ERR_FAIL_COND(!a.is_valid());

	for (int j = 0; j < a->get_track_count(); j++) {

		String path = a->track_get_path(j);

		if (!keep.has(path)) {
			a->remove_track(j);
			j--;
		}
	}
}

void ResourceExporterScene::_filter_tracks(Node *scene, const String &p_text) {

	if (!scene->has_node(String("AnimationPlayer")))
		return;
	Node *n = scene->get_node(String("AnimationPlayer"));
	ERR_FAIL_COND(!n);
	AnimationPlayer *anim = Object::cast_to<AnimationPlayer>(n);
	ERR_FAIL_COND(!anim);

	Vector<String> strings = p_text.split("\n");
	for (int i = 0; i < strings.size(); i++) {

		strings.write[i] = strings[i].strip_edges();
	}

	List<StringName> anim_names;
	anim->get_animation_list(&anim_names);
	for (List<StringName>::Element *E = anim_names.front(); E; E = E->next()) {

		String name = E->get();
		bool valid_for_this = false;
		bool valid = false;

		Set<String> keep;
		Set<String> keep_local;

		for (int i = 0; i < strings.size(); i++) {

			if (strings[i].begins_with("@")) {

				valid_for_this = false;
				for (Set<String>::Element *F = keep_local.front(); F; F = F->next()) {
					keep.insert(F->get());
				}
				keep_local.clear();

				Vector<String> filters = strings[i].substr(1, strings[i].length()).split(",");
				for (int j = 0; j < filters.size(); j++) {

					String fname = filters[j].strip_edges();
					if (fname == "")
						continue;
					int fc = fname[0];
					bool plus;
					if (fc == '+')
						plus = true;
					else if (fc == '-')
						plus = false;
					else
						continue;

					String filter = fname.substr(1, fname.length()).strip_edges();

					if (!name.matchn(filter))
						continue;
					valid_for_this = plus;
				}

				if (valid_for_this)
					valid = true;

			} else if (valid_for_this) {

				Ref<Animation> a = anim->get_animation(name);
				if (!a.is_valid())
					continue;

				for (int j = 0; j < a->get_track_count(); j++) {

					String path = a->track_get_path(j);

					String tname = strings[i];
					if (tname == "")
						continue;
					int fc = tname[0];
					bool plus;
					if (fc == '+')
						plus = true;
					else if (fc == '-')
						plus = false;
					else
						continue;

					String filter = tname.substr(1, tname.length()).strip_edges();

					if (!path.matchn(filter))
						continue;

					if (plus)
						keep_local.insert(path);
					else if (!keep.has(path)) {
						keep_local.erase(path);
					}
				}
			}
		}

		if (valid) {
			for (Set<String>::Element *F = keep_local.front(); F; F = F->next()) {
				keep.insert(F->get());
			}
			_filter_anim_tracks(anim->get_animation(name), keep);
		}
	}
}

void ResourceExporterScene::_optimize_animations(Node *scene, float p_max_lin_error, float p_max_ang_error, float p_max_angle) {

	if (!scene->has_node(String("AnimationPlayer")))
		return;
	Node *n = scene->get_node(String("AnimationPlayer"));
	ERR_FAIL_COND(!n);
	AnimationPlayer *anim = Object::cast_to<AnimationPlayer>(n);
	ERR_FAIL_COND(!anim);

	List<StringName> anim_names;
	anim->get_animation_list(&anim_names);
	for (List<StringName>::Element *E = anim_names.front(); E; E = E->next()) {

		Ref<Animation> a = anim->get_animation(E->get());
		a->optimize(p_max_lin_error, p_max_ang_error, Math::deg2rad(p_max_angle));
	}
}

static String _make_extname(const String &p_str) {

	String ext_name = p_str.replace(".", "_");
	ext_name = ext_name.replace(":", "_");
	ext_name = ext_name.replace("\"", "_");
	ext_name = ext_name.replace("<", "_");
	ext_name = ext_name.replace(">", "_");
	ext_name = ext_name.replace("/", "_");
	ext_name = ext_name.replace("|", "_");
	ext_name = ext_name.replace("\\", "_");
	ext_name = ext_name.replace("?", "_");
	ext_name = ext_name.replace("*", "_");

	return ext_name;
}

void ResourceExporterScene::_find_meshes(Node *p_node, Map<Ref<ArrayMesh>, Transform> &meshes) {

	List<PropertyInfo> pi;
	p_node->get_property_list(&pi);

	MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);

	if (mi) {

		Ref<ArrayMesh> mesh = mi->get_mesh();

		if (mesh.is_valid() && !meshes.has(mesh)) {
			Spatial *s = mi;
			Transform transform;
			while (s) {
				transform = transform * s->get_transform();
				s = s->get_parent_spatial();
			}

			meshes[mesh] = transform;
		}
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {

		_find_meshes(p_node->get_child(i), meshes);
	}
}

void ResourceExporterScene::_make_external_resources(Node *p_node, const String &p_base_path, bool p_make_animations, bool p_animations_as_text, bool p_keep_animations, bool p_make_materials, bool p_materials_as_text, bool p_keep_materials, bool p_make_meshes, bool p_meshes_as_text, Map<Ref<Animation>, Ref<Animation> > &p_animations, Map<Ref<Material>, Ref<Material> > &p_materials, Map<Ref<ArrayMesh>, Ref<ArrayMesh> > &p_meshes) {

	List<PropertyInfo> pi;

	if (p_make_animations) {
		if (Object::cast_to<AnimationPlayer>(p_node)) {
			AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(p_node);

			List<StringName> anims;
			ap->get_animation_list(&anims);
			for (List<StringName>::Element *E = anims.front(); E; E = E->next()) {

				Ref<Animation> anim = ap->get_animation(E->get());
				ERR_CONTINUE(anim.is_null());

				if (!p_animations.has(anim)) {

					//mark what comes from the file first, this helps eventually keep user data
					for (int i = 0; i < anim->get_track_count(); i++) {
						anim->track_set_imported(i, true);
					}

					String ext_name;

					if (p_animations_as_text) {
						ext_name = p_base_path.plus_file(_make_extname(E->get()) + ".tres");
					} else {
						ext_name = p_base_path.plus_file(_make_extname(E->get()) + ".anim");
					}

					if (FileAccess::exists(ext_name) && p_keep_animations) {
						//try to keep custom animation tracks
						Ref<Animation> old_anim = ResourceLoader::load(ext_name, "Animation", true);
						if (old_anim.is_valid()) {
							//meergeee
							for (int i = 0; i < old_anim->get_track_count(); i++) {
								if (!old_anim->track_is_imported(i)) {
									old_anim->copy_track(i, anim);
								}
							}
							anim->set_loop(old_anim->has_loop());
						}
					}

					anim->set_path(ext_name, true); //if not set, then its never saved externally
					ResourceSaver::save(ext_name, anim, ResourceSaver::FLAG_CHANGE_PATH);
					p_animations[anim] = anim;
				}
			}
		}
	}

	p_node->get_property_list(&pi);

	for (List<PropertyInfo>::Element *E = pi.front(); E; E = E->next()) {

		if (E->get().type == Variant::OBJECT) {

			Ref<Material> mat = p_node->get(E->get().name);

			if (p_make_materials && mat.is_valid() && mat->get_name() != "") {

				if (!p_materials.has(mat)) {

					String ext_name;

					if (p_materials_as_text) {
						ext_name = p_base_path.plus_file(_make_extname(mat->get_name()) + ".tres");
					} else {
						ext_name = p_base_path.plus_file(_make_extname(mat->get_name()) + ".material");
					}

					if (p_keep_materials && FileAccess::exists(ext_name)) {
						//if exists, use it
						p_materials[mat] = ResourceLoader::load(ext_name);
					} else {

						ResourceSaver::save(ext_name, mat, ResourceSaver::FLAG_CHANGE_PATH);
						p_materials[mat] = ResourceLoader::load(ext_name, "", true); // disable loading from the cache.
					}
				}

				if (p_materials[mat] != mat) {

					p_node->set(E->get().name, p_materials[mat]);
				}
			} else {

				Ref<ArrayMesh> mesh = p_node->get(E->get().name);

				if (mesh.is_valid()) {

					bool mesh_just_added = false;

					if (p_make_meshes) {

						if (!p_meshes.has(mesh)) {

							//meshes are always overwritten, keeping them is not practical
							String ext_name;

							if (p_meshes_as_text) {
								ext_name = p_base_path.plus_file(_make_extname(mesh->get_name()) + ".tres");
							} else {
								ext_name = p_base_path.plus_file(_make_extname(mesh->get_name()) + ".mesh");
							}

							ResourceSaver::save(ext_name, mesh, ResourceSaver::FLAG_CHANGE_PATH);
							p_meshes[mesh] = ResourceLoader::load(ext_name);
							p_node->set(E->get().name, p_meshes[mesh]);
							mesh_just_added = true;
						}
					}

					if (p_make_materials) {

						if (mesh_just_added || !p_meshes.has(mesh)) {

							for (int i = 0; i < mesh->get_surface_count(); i++) {
								mat = mesh->surface_get_material(i);

								if (!mat.is_valid())
									continue;
								if (mat->get_name() == "")
									continue;

								if (!p_materials.has(mat)) {
									String ext_name;

									if (p_materials_as_text) {
										ext_name = p_base_path.plus_file(_make_extname(mat->get_name()) + ".tres");
									} else {
										ext_name = p_base_path.plus_file(_make_extname(mat->get_name()) + ".material");
									}

									if (p_keep_materials && FileAccess::exists(ext_name)) {
										//if exists, use it
										p_materials[mat] = ResourceLoader::load(ext_name);
									} else {

										ResourceSaver::save(ext_name, mat, ResourceSaver::FLAG_CHANGE_PATH);
										p_materials[mat] = ResourceLoader::load(ext_name, "", true); // disable loading from the cache.
									}
								}

								if (p_materials[mat] != mat) {

									mesh->surface_set_material(i, p_materials[mat]);

									//re-save the mesh since a material is now assigned
									if (p_make_meshes) {

										String ext_name;

										if (p_meshes_as_text) {
											ext_name = p_base_path.plus_file(_make_extname(mesh->get_name()) + ".tres");
										} else {
											ext_name = p_base_path.plus_file(_make_extname(mesh->get_name()) + ".mesh");
										}

										ResourceSaver::save(ext_name, mesh, ResourceSaver::FLAG_CHANGE_PATH);
										p_meshes[mesh] = ResourceLoader::load(ext_name);
									}
								}
							}

							if (!p_make_meshes) {
								p_meshes[mesh] = Ref<ArrayMesh>(); //save it anyway, so it won't be checked again
							}
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {

		_make_external_resources(p_node->get_child(i), p_base_path, p_make_animations, p_animations_as_text, p_keep_animations, p_make_materials, p_materials_as_text, p_keep_materials, p_make_meshes, p_meshes_as_text, p_animations, p_materials, p_meshes);
	}
}

void ResourceExporterScene::get_export_options(List<ExportOption> *r_options, int p_preset) const {

	// r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_type", PROPERTY_HINT_TYPE_STRING, "Node"), "Spatial"));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_name"), "Scene Root"));

	List<String> script_extentions;
	ResourceLoader::get_recognized_extensions_for_type("Script", &script_extentions);

	String script_ext_hint;

	for (List<String>::Element *E = script_extentions.front(); E; E = E->next()) {
		if (script_ext_hint != "")
			script_ext_hint += ",";
		script_ext_hint += "*." + E->get();
	}

	//bool materials_out = p_preset == PRESET_SEPARATE_MATERIALS || p_preset == PRESET_SEPARATE_MESHES_AND_MATERIALS || p_preset == PRESET_MULTIPLE_SCENES_AND_MATERIALS || p_preset == PRESET_SEPARATE_MATERIALS_AND_ANIMATIONS || p_preset == PRESET_SEPARATE_MESHES_MATERIALS_AND_ANIMATIONS;
	//bool meshes_out = p_preset == PRESET_SEPARATE_MESHES || p_preset == PRESET_SEPARATE_MESHES_AND_MATERIALS || p_preset == PRESET_SEPARATE_MESHES_AND_ANIMATIONS || p_preset == PRESET_SEPARATE_MESHES_MATERIALS_AND_ANIMATIONS;
	//bool scenes_out = p_preset == PRESET_MULTIPLE_SCENES || p_preset == PRESET_MULTIPLE_SCENES_AND_MATERIALS;
	//bool animations_out = p_preset == PRESET_SEPARATE_ANIMATIONS || p_preset == PRESET_SEPARATE_MESHES_AND_ANIMATIONS || p_preset == PRESET_SEPARATE_MATERIALS_AND_ANIMATIONS || p_preset == PRESET_SEPARATE_MESHES_MATERIALS_AND_ANIMATIONS;

	// r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "nodes/root_scale", PROPERTY_HINT_RANGE, "0.001,1000,0.001"), 1.0));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/custom_script", PROPERTY_HINT_FILE, script_ext_hint), ""));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "nodes/storage", PROPERTY_HINT_ENUM, "Single Scene,Instanced Sub-Scenes"), scenes_out ? 1 : 0));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "materials/location", PROPERTY_HINT_ENUM, "Node,Mesh"), (meshes_out || materials_out) ? 1 : 0));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "materials/storage", PROPERTY_HINT_ENUM, "Built-In,Files (.material),Files (.tres)", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), materials_out ? 1 : 0));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "materials/keep_on_reimport"), materials_out));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/compress"), true));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/ensure_tangents"), true));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "meshes/storage", PROPERTY_HINT_ENUM, "Built-In,Files (.mesh),Files (.tres)"), meshes_out ? 1 : 0));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "meshes/light_baking", PROPERTY_HINT_ENUM, "Disabled,Enable,Gen Lightmaps", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "meshes/lightmap_texel_size", PROPERTY_HINT_RANGE, "0.001,100,0.001"), 0.1));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "external_files/store_in_subdir"), false));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/import", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), true));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "animation/fps", PROPERTY_HINT_RANGE, "1,120,1"), 15));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "animation/filter_script", PROPERTY_HINT_MULTILINE_TEXT), ""));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "animation/storage", PROPERTY_HINT_ENUM, "Built-In,Files (.anim),Files (.tres)", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), animations_out));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/keep_custom_tracks"), animations_out));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/optimizer/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), true));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "animation/optimizer/max_linear_error"), 0.05));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "animation/optimizer/max_angular_error"), 0.01));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "animation/optimizer/max_angle"), 22));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/optimizer/remove_unused_tracks"), true));
	// r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "animation/clips/amount", PROPERTY_HINT_RANGE, "0,256,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));
	// for (int i = 0; i < 256; i++) {
	// 	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "animation/clip_" + itos(i + 1) + "/name"), ""));
	// 	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "animation/clip_" + itos(i + 1) + "/start_frame"), 0));
	// 	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "animation/clip_" + itos(i + 1) + "/end_frame"), 0));
	// 	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/clip_" + itos(i + 1) + "/loops"), false));
	// }
}

void ResourceExporterScene::_replace_owner(Node *p_node, Node *p_scene, Node *p_new_owner) {

	if (p_node != p_new_owner && p_node->get_owner() == p_scene) {
		p_node->set_owner(p_new_owner);
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *n = p_node->get_child(i);
		_replace_owner(n, p_scene, p_new_owner);
	}
}

Error ResourceExporterScene::export_(Node *p_node, const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	const String &src_path = p_source_file;

	Ref<EditorSceneExporter> exporter;
	String ext = src_path.get_extension().to_lower();

	EditorProgress progress("export", TTR("Export Scene"), 104);
	progress.step(TTR("Exporting Scene..."), 0);

	for (Set<Ref<EditorSceneExporter> >::Element *E = exporters.front(); E; E = E->next()) {

		List<String> extensions;
		E->get()->get_exporter_extensions(&extensions);

		for (List<String>::Element *F = extensions.front(); F; F = F->next()) {

			if (F->get().to_lower() == ext) {

				exporter = E->get();
				break;
			}
		}

		if (exporter.is_valid())
			break;
	}

	ERR_FAIL_COND_V(!exporter.is_valid(), ERR_FILE_UNRECOGNIZED);

	float fps = p_options["animation/fps"];

	int import_flags = EditorSceneImporter::IMPORT_ANIMATION_DETECT_LOOP;
	if (!bool(p_options["animation/optimizer/remove_unused_tracks"]))
		import_flags |= EditorSceneImporter::IMPORT_ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS;

	if (bool(p_options["animation/import"]))
		import_flags |= EditorSceneImporter::IMPORT_ANIMATION;

	if (int(p_options["meshes/compress"]))
		import_flags |= EditorSceneImporter::IMPORT_USE_COMPRESSION;

	if (bool(p_options["meshes/ensure_tangents"]))
		import_flags |= EditorSceneImporter::IMPORT_GENERATE_TANGENT_ARRAYS;

	if (int(p_options["materials/location"]) == 0)
		import_flags |= EditorSceneImporter::IMPORT_MATERIALS_IN_INSTANCES;

	Error err = OK;
	exporter->save_scene(p_node, p_save_path, src_path, import_flags, fps, r_gen_files, &err);
	if (err != OK) {
		return err;
	}

	progress.step(TTR("Saving..."), 104);

	return OK;
}

ResourceExporterScene *ResourceExporterScene::singleton = NULL;

ResourceExporterScene::ResourceExporterScene() {
	singleton = this;
}
