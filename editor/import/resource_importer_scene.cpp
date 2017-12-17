/*************************************************************************/
/*  resource_importer_scene.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor/editor_node.h"
#include "io/resource_saver.h"
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
#include "scene/resources/sphere_shape.h"

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

	ERR_FAIL_V(NULL);
}

Ref<Animation> EditorSceneImporter::import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps) {

	if (get_script_instance()) {
		return get_script_instance()->call("_import_animation", p_path, p_flags);
	}

	ERR_FAIL_V(NULL);
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
	BIND_CONSTANT(IMPORT_ANIMATION_DETECT_LOOP);
	BIND_CONSTANT(IMPORT_ANIMATION_OPTIMIZE);
	BIND_CONSTANT(IMPORT_ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS);
	BIND_CONSTANT(IMPORT_ANIMATION_KEEP_VALUE_TRACKS);
	BIND_CONSTANT(IMPORT_GENERATE_TANGENT_ARRAYS);
	BIND_CONSTANT(IMPORT_FAIL_ON_MISSING_DEPENDENCIES);
	BIND_CONSTANT(IMPORT_MATERIALS_IN_INSTANCES);
	BIND_CONSTANT(IMPORT_USE_COMPRESSION);
}

/////////////////////////////////
void EditorScenePostImport::_bind_methods() {

	BIND_VMETHOD(MethodInfo("post_import", PropertyInfo(Variant::OBJECT, "scene")));
}

Node *EditorScenePostImport::post_import(Node *p_scene) {

	if (get_script_instance())
		return get_script_instance()->call("post_import", p_scene);

	return p_scene;
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

	for (Set<Ref<EditorSceneImporter> >::Element *E = importers.front(); E; E = E->next()) {
		E->get()->get_extensions(p_extensions);
	}
}

String ResourceImporterScene::get_save_extension() const {
	return "scn";
}

String ResourceImporterScene::get_resource_type() const {

	return "PackedScene";
}

bool ResourceImporterScene::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

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

int ResourceImporterScene::get_preset_count() const {
	return PRESET_MAX;
}
String ResourceImporterScene::get_preset_name(int p_idx) const {

	switch (p_idx) {
		case PRESET_SINGLE_SCENE: return TTR("Import as Single Scene");
		case PRESET_SEPARATE_ANIMATIONS: return TTR("Import with Separate Animations");
		case PRESET_SEPARATE_MATERIALS: return TTR("Import with Separate Materials");
		case PRESET_SEPARATE_MESHES: return TTR("Import with Separate Objects");
		case PRESET_SEPARATE_MESHES_AND_MATERIALS: return TTR("Import with Separate Objects+Materials");
		case PRESET_SEPARATE_MESHES_AND_ANIMATIONS: return TTR("Import with Separate Objects+Animations");
		case PRESET_SEPARATE_MATERIALS_AND_ANIMATIONS: return TTR("Import with Separate Materials+Animations");
		case PRESET_SEPARATE_MESHES_MATERIALS_AND_ANIMATIONS: return TTR("Import with Separate Objects+Materials+Animations");
		case PRESET_MULTIPLE_SCENES: return TTR("Import as Multiple Scenes");
		case PRESET_MULTIPLE_SCENES_AND_MATERIALS: return TTR("Import as Multiple Scenes+Materials");
	}

	return "";
}

static bool _teststr(const String &p_what, const String &p_str) {

	if (p_what.findn("$" + p_str) != -1) //blender and other stuff
		return true;
	if (p_what.to_lower().ends_with("-" + p_str)) //collada only supports "_" and "-" besides letters
		return true;
	if (p_what.to_lower().ends_with("_" + p_str)) //collada only supports "_" and "-" besides letters
		return true;
	return false;
}

static String _fixstr(const String &p_what, const String &p_str) {

	if (p_what.findn("$" + p_str) != -1) //blender and other stuff
		return p_what.replace("$" + p_str, "");
	if (p_what.to_lower().ends_with("-" + p_str)) //collada only supports "_" and "-" besides letters
		return p_what.substr(0, p_what.length() - (p_str.length() + 1));
	if (p_what.to_lower().ends_with("_" + p_str)) //collada only supports "_" and "-" besides letters
		return p_what.substr(0, p_what.length() - (p_str.length() + 1));
	return p_what;
}

Node *ResourceImporterScene::_fix_node(Node *p_node, Node *p_root, Map<Ref<ArrayMesh>, Ref<Shape> > &collision_map, LightBakeMode p_light_bake_mode) {

	// children first..
	for (int i = 0; i < p_node->get_child_count(); i++) {

		Node *r = _fix_node(p_node->get_child(i), p_root, collision_map, p_light_bake_mode);
		if (!r) {
			print_line("was erased..");
			i--; //was erased
		}
	}

	String name = p_node->get_name();

	bool isroot = p_node == p_root;

	if (!isroot && _teststr(name, "noimp")) {

		memdelete(p_node);
		return NULL;
	}

	if (Object::cast_to<MeshInstance>(p_node)) {

		MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);

		Ref<ArrayMesh> m = mi->get_mesh();

		if (m.is_valid()) {

			for (int i = 0; i < m->get_surface_count(); i++) {

				Ref<SpatialMaterial> mat = m->surface_get_material(i);
				if (!mat.is_valid())
					continue;

				if (_teststr(mat->get_name(), "alpha")) {

					mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
					mat->set_name(_fixstr(mat->get_name(), "alpha"));
				}
				if (_teststr(mat->get_name(), "vcol")) {

					mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
					mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
					mat->set_name(_fixstr(mat->get_name(), "vcol"));
				}
			}
		}

		if (p_light_bake_mode != LIGHT_BAKE_DISABLED) {

			mi->set_flag(GeometryInstance::FLAG_USE_BAKED_LIGHT, true);
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
		}
	}

	if (_teststr(name, "colonly") || _teststr(name, "convcolonly")) {

		if (isroot)
			return p_node;
		MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);
		if (mi) {
			Node *col;

			if (_teststr(name, "colonly")) {
				col = mi->create_trimesh_collision_node();
				ERR_FAIL_COND_V(!col, NULL);

				col->set_name(_fixstr(name, "colonly"));
			} else {
				col = mi->create_convex_collision_node();
				ERR_FAIL_COND_V(!col, NULL);

				col->set_name(_fixstr(name, "convcolonly"));
			}

			Object::cast_to<Spatial>(col)->set_transform(mi->get_transform());
			p_node->replace_by(col);
			memdelete(p_node);
			p_node = col;

			StaticBody *sb = Object::cast_to<StaticBody>(col);
			CollisionShape *colshape = Object::cast_to<CollisionShape>(sb->get_child(0));
			colshape->set_name("shape");
			colshape->set_owner(p_node->get_owner());
		} else if (p_node->has_meta("empty_draw_type")) {
			String empty_draw_type = String(p_node->get_meta("empty_draw_type"));
			print_line(empty_draw_type);
			StaticBody *sb = memnew(StaticBody);
			sb->set_name(_fixstr(name, "colonly"));
			Object::cast_to<Spatial>(sb)->set_transform(Object::cast_to<Spatial>(p_node)->get_transform());
			p_node->replace_by(sb);
			memdelete(p_node);
			CollisionShape *colshape = memnew(CollisionShape);
			if (empty_draw_type == "CUBE") {
				BoxShape *boxShape = memnew(BoxShape);
				boxShape->set_extents(Vector3(1, 1, 1));
				colshape->set_shape(boxShape);
				colshape->set_name("BoxShape");
			} else if (empty_draw_type == "SINGLE_ARROW") {
				RayShape *rayShape = memnew(RayShape);
				rayShape->set_length(1);
				colshape->set_shape(rayShape);
				colshape->set_name("RayShape");
				Object::cast_to<Spatial>(sb)->rotate_x(Math_PI / 2);
			} else if (empty_draw_type == "IMAGE") {
				PlaneShape *planeShape = memnew(PlaneShape);
				colshape->set_shape(planeShape);
				colshape->set_name("PlaneShape");
			} else {
				SphereShape *sphereShape = memnew(SphereShape);
				sphereShape->set_radius(1);
				colshape->set_shape(sphereShape);
				colshape->set_name("SphereShape");
			}
			sb->add_child(colshape);
			colshape->set_owner(sb->get_owner());
		}

	} else if (_teststr(name, "rigid") && Object::cast_to<MeshInstance>(p_node)) {

		if (isroot)
			return p_node;

		// get mesh instance and bounding box
		MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);
		AABB aabb = mi->get_aabb();

		// create a new rigid body collision node
		RigidBody *rigid_body = memnew(RigidBody);
		Node *col = rigid_body;
		ERR_FAIL_COND_V(!col, NULL);

		// remove node name postfix
		col->set_name(_fixstr(name, "rigid"));
		// get mesh instance xform matrix to the rigid body collision node
		Object::cast_to<Spatial>(col)->set_transform(mi->get_transform());
		// save original node by duplicating it into a new instance and correcting the name
		Node *mesh = p_node->duplicate();
		mesh->set_name(_fixstr(name, "rigid"));
		// reset the xform matrix of the duplicated node so it can inherit parent node xform
		Object::cast_to<Spatial>(mesh)->set_transform(Transform(Basis()));
		// reparent the new mesh node to the rigid body collision node
		p_node->add_child(mesh);
		mesh->set_owner(p_node->get_owner());
		// replace the original node with the rigid body collision node
		p_node->replace_by(col);
		memdelete(p_node);
		p_node = col;

		// create an alias for the rigid body collision node
		RigidBody *rb = Object::cast_to<RigidBody>(col);
		// create a new Box collision shape and set the right extents
		Ref<BoxShape> shape = memnew(BoxShape);
		shape->set_extents(aabb.get_size() * 0.5);
		CollisionShape *colshape = memnew(CollisionShape);
		colshape->set_name("shape");
		colshape->set_shape(shape);
		// reparent the new collision shape to the rigid body collision node
		rb->add_child(colshape);
		colshape->set_owner(p_node->get_owner());

	} else if ((_teststr(name, "col") || (_teststr(name, "convcol"))) && Object::cast_to<MeshInstance>(p_node)) {

		MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);
		Node *col;

		if (_teststr(name, "col")) {
			mi->set_name(_fixstr(name, "col"));
			col = mi->create_trimesh_collision_node();
			ERR_FAIL_COND_V(!col, NULL);

			col->set_name("col");
		} else {
			mi->set_name(_fixstr(name, "convcol"));
			col = mi->create_convex_collision_node();
			ERR_FAIL_COND_V(!col, NULL);

			col->set_name("convcol");
		}

		p_node->add_child(col);

		StaticBody *sb = Object::cast_to<StaticBody>(col);
		CollisionShape *colshape = Object::cast_to<CollisionShape>(sb->get_child(0));
		colshape->set_name("shape");
		col->add_child(colshape);
		colshape->set_owner(p_node->get_owner());
		sb->set_owner(p_node->get_owner());

	} else if (_teststr(name, "navmesh") && Object::cast_to<MeshInstance>(p_node)) {

		if (isroot)
			return p_node;

		MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);

		Ref<ArrayMesh> mesh = mi->get_mesh();
		ERR_FAIL_COND_V(mesh.is_null(), NULL);
		NavigationMeshInstance *nmi = memnew(NavigationMeshInstance);

		nmi->set_name(_fixstr(name, "navmesh"));
		Ref<NavigationMesh> nmesh = memnew(NavigationMesh);
		nmesh->create_from_mesh(mesh);
		nmi->set_navigation_mesh(nmesh);
		Object::cast_to<Spatial>(nmi)->set_transform(mi->get_transform());
		p_node->replace_by(nmi);
		memdelete(p_node);
		p_node = nmi;
	} else if (_teststr(name, "vehicle")) {

		if (isroot)
			return p_node;

		Node *owner = p_node->get_owner();
		Spatial *s = Object::cast_to<Spatial>(p_node);
		VehicleBody *bv = memnew(VehicleBody);
		String n = _fixstr(p_node->get_name(), "vehicle");
		bv->set_name(n);
		p_node->replace_by(bv);
		p_node->set_name(n);
		bv->add_child(p_node);
		bv->set_owner(owner);
		p_node->set_owner(owner);
		bv->set_transform(s->get_transform());
		s->set_transform(Transform());

		p_node = bv;

	} else if (_teststr(name, "wheel")) {

		if (isroot)
			return p_node;

		Node *owner = p_node->get_owner();
		Spatial *s = Object::cast_to<Spatial>(p_node);
		VehicleWheel *bv = memnew(VehicleWheel);
		String n = _fixstr(p_node->get_name(), "wheel");
		bv->set_name(n);
		p_node->replace_by(bv);
		p_node->set_name(n);
		bv->add_child(p_node);
		bv->set_owner(owner);
		p_node->set_owner(owner);
		bv->set_transform(s->get_transform());
		s->set_transform(Transform());

		p_node = bv;

	} else if (Object::cast_to<MeshInstance>(p_node)) {

		//last attempt, maybe collision inside the mesh data

		MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);

		Ref<ArrayMesh> mesh = mi->get_mesh();
		if (!mesh.is_null()) {

			if (_teststr(mesh->get_name(), "col") || _teststr(mesh->get_name(), "convcol")) {
				Ref<Shape> shape;
				if (_teststr(mesh->get_name(), "col")) {
					mesh->set_name(_fixstr(mesh->get_name(), "col"));

					if (collision_map.has(mesh)) {
						shape = collision_map[mesh];

					} else {

						shape = mesh->create_trimesh_shape();
						if (!shape.is_null())
							collision_map[mesh] = shape;
					}
				} else if (_teststr(mesh->get_name(), "convcol")) {
					mesh->set_name(_fixstr(mesh->get_name(), "convcol"));

					if (collision_map.has(mesh)) {
						shape = collision_map[mesh];

					} else {

						shape = mesh->create_convex_shape();
						if (!shape.is_null())
							collision_map[mesh] = shape;
					}
				}

				if (!shape.is_null()) {
					StaticBody *col = memnew(StaticBody);
					CollisionShape *cshape = memnew(CollisionShape);
					cshape->set_shape(shape);
					col->add_child(cshape);

					col->set_transform(mi->get_transform());
					col->set_name(mi->get_name());
					p_node->replace_by(col);
					memdelete(p_node);
					p_node = col;

					cshape->set_name("shape");
					cshape->set_owner(p_node->get_owner());
				}
			}
		}
	}

	return p_node;
}

void ResourceImporterScene::_create_clips(Node *scene, const Array &p_clips, bool p_bake_all) {

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
						}
					}

					if (default_anim->track_get_type(j) == Animation::TYPE_TRANSFORM) {
						Quat q;
						Vector3 p;
						Vector3 s;
						default_anim->transform_track_get_key(j, k, &p, &q, &s);
						new_anim->transform_track_insert_key(dtrack, kt - from, p, q, s);
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
			}
		}

		new_anim->set_loop(loop);
		new_anim->set_length(to - from);
		anim->add_animation(name, new_anim);
	}

	anim->remove_animation("default"); //remove default (no longer needed)
}

void ResourceImporterScene::_filter_anim_tracks(Ref<Animation> anim, Set<String> &keep) {

	Ref<Animation> a = anim;
	ERR_FAIL_COND(!a.is_valid());

	print_line("From Anim " + anim->get_name() + ":");

	for (int j = 0; j < a->get_track_count(); j++) {

		String path = a->track_get_path(j);

		if (!keep.has(path)) {

			print_line("Remove: " + path);
			a->remove_track(j);
			j--;
		}
	}
}

void ResourceImporterScene::_filter_tracks(Node *scene, const String &p_text) {

	if (!scene->has_node(String("AnimationPlayer")))
		return;
	Node *n = scene->get_node(String("AnimationPlayer"));
	ERR_FAIL_COND(!n);
	AnimationPlayer *anim = Object::cast_to<AnimationPlayer>(n);
	ERR_FAIL_COND(!anim);

	Vector<String> strings = p_text.split("\n");
	for (int i = 0; i < strings.size(); i++) {

		strings[i] = strings[i].strip_edges();
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
		} else {
		}
	}
}

void ResourceImporterScene::_optimize_animations(Node *scene, float p_max_lin_error, float p_max_ang_error, float p_max_angle) {

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

void ResourceImporterScene::_find_meshes(Node *p_node, Map<Ref<ArrayMesh>, Transform> &meshes) {

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

			print_line("mesh transform: " + meshes[mesh]);
		}
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {

		_find_meshes(p_node->get_child(i), meshes);
	}
}

void ResourceImporterScene::_make_external_resources(Node *p_node, const String &p_base_path, bool p_make_animations, bool p_keep_animations, bool p_make_materials, bool p_keep_materials, bool p_make_meshes, Map<Ref<Animation>, Ref<Animation> > &p_animations, Map<Ref<Material>, Ref<Material> > &p_materials, Map<Ref<ArrayMesh>, Ref<ArrayMesh> > &p_meshes) {

	List<PropertyInfo> pi;

	print_line("node: " + String(p_node->get_name()));

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

					String ext_name = p_base_path.plus_file(_make_extname(E->get()) + ".anim");

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
						}
					}

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

					String ext_name = p_base_path.plus_file(_make_extname(mat->get_name()) + ".material");
					if (p_keep_materials && FileAccess::exists(ext_name)) {
						//if exists, use it
						p_materials[mat] = ResourceLoader::load(ext_name);
					} else {

						ResourceSaver::save(ext_name, mat, ResourceSaver::FLAG_CHANGE_PATH);
						p_materials[mat] = ResourceLoader::load(ext_name);
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
							String ext_name = p_base_path.plus_file(_make_extname(mesh->get_name()) + ".mesh");

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

									String ext_name = p_base_path.plus_file(_make_extname(mat->get_name()) + ".material");
									;
									if (FileAccess::exists(ext_name)) {
										//if exists, use it
										p_materials[mat] = ResourceLoader::load(ext_name);
									} else {

										ResourceSaver::save(ext_name, mat, ResourceSaver::FLAG_CHANGE_PATH);
										p_materials[mat] = ResourceLoader::load(ext_name);
									}
								}

								if (p_materials[mat] != mat) {

									mesh->surface_set_material(i, p_materials[mat]);

									//re-save the mesh since a material is now assigned
									if (p_make_meshes) {
										String ext_name = p_base_path.plus_file(_make_extname(mesh->get_name()) + ".mesh");
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

		_make_external_resources(p_node->get_child(i), p_base_path, p_make_animations, p_keep_animations, p_make_materials, p_keep_materials, p_make_meshes, p_animations, p_materials, p_meshes);
	}
}

void ResourceImporterScene::get_import_options(List<ImportOption> *r_options, int p_preset) const {

	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_type", PROPERTY_HINT_TYPE_STRING, "Node"), "Spatial"));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_name"), "Scene Root"));

	List<String> script_extentions;
	ResourceLoader::get_recognized_extensions_for_type("Script", &script_extentions);

	String script_ext_hint;

	for (List<String>::Element *E = script_extentions.front(); E; E = E->next()) {
		if (script_ext_hint != "")
			script_ext_hint += ",";
		script_ext_hint += "*." + E->get();
	}

	bool materials_out = p_preset == PRESET_SEPARATE_MATERIALS || p_preset == PRESET_SEPARATE_MESHES_AND_MATERIALS || p_preset == PRESET_MULTIPLE_SCENES_AND_MATERIALS || p_preset == PRESET_SEPARATE_MATERIALS_AND_ANIMATIONS || p_preset == PRESET_SEPARATE_MESHES_MATERIALS_AND_ANIMATIONS;
	bool meshes_out = p_preset == PRESET_SEPARATE_MESHES || p_preset == PRESET_SEPARATE_MESHES_AND_MATERIALS || p_preset == PRESET_SEPARATE_MESHES_AND_ANIMATIONS || p_preset == PRESET_SEPARATE_MESHES_MATERIALS_AND_ANIMATIONS;
	bool scenes_out = p_preset == PRESET_MULTIPLE_SCENES || p_preset == PRESET_MULTIPLE_SCENES_AND_MATERIALS;
	bool animations_out = p_preset == PRESET_SEPARATE_ANIMATIONS || p_preset == PRESET_SEPARATE_MESHES_AND_ANIMATIONS || p_preset == PRESET_SEPARATE_MATERIALS_AND_ANIMATIONS || p_preset == PRESET_SEPARATE_MESHES_MATERIALS_AND_ANIMATIONS;

	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "nodes/root_scale", PROPERTY_HINT_RANGE, "0.001,1000,0.001"), 1.0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/custom_script", PROPERTY_HINT_FILE, script_ext_hint), ""));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "nodes/storage", PROPERTY_HINT_ENUM, "Single Scene,Instanced Sub-Scenes"), scenes_out ? 1 : 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "materials/location", PROPERTY_HINT_ENUM, "Node,Mesh"), (meshes_out || materials_out) ? 1 : 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "materials/storage", PROPERTY_HINT_ENUM, "Built-In,Files", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), materials_out ? 1 : 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "materials/keep_on_reimport"), materials_out ? true : false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/compress"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/ensure_tangents"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "meshes/storage", PROPERTY_HINT_ENUM, "Built-In,Files"), meshes_out ? 1 : 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "meshes/light_baking", PROPERTY_HINT_ENUM, "Disabled,Enable,Gen Lightmaps", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "meshes/lightmap_texel_size", PROPERTY_HINT_RANGE, "0.001,100,0.001"), 0.1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "external_files/store_in_subdir"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/import", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "animation/fps", PROPERTY_HINT_RANGE, "1,120,1"), 15));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "animation/filter_script", PROPERTY_HINT_MULTILINE_TEXT), ""));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "animation/storage", PROPERTY_HINT_ENUM, "Built-In,Files"), animations_out ? true : false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/keep_custom_tracks"), animations_out ? true : false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/optimizer/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "animation/optimizer/max_linear_error"), 0.05));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "animation/optimizer/max_angular_error"), 0.01));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "animation/optimizer/max_angle"), 22));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/optimizer/remove_unused_tracks"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "animation/clips/amount", PROPERTY_HINT_RANGE, "0,256,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));
	for (int i = 0; i < 256; i++) {
		r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "animation/clip_" + itos(i + 1) + "/name"), ""));
		r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "animation/clip_" + itos(i + 1) + "/start_frame"), 0));
		r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "animation/clip_" + itos(i + 1) + "/end_frame"), 0));
		r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/clip_" + itos(i + 1) + "/loops"), false));
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

Node *ResourceImporterScene::import_scene_from_other_importer(EditorSceneImporter *p_exception, const String &p_path, uint32_t p_flags, int p_bake_fps) {

	Ref<EditorSceneImporter> importer;
	String ext = p_path.get_extension().to_lower();

	for (Set<Ref<EditorSceneImporter> >::Element *E = importers.front(); E; E = E->next()) {

		if (E->get().ptr() == p_exception)
			continue;
		List<String> extensions;
		E->get()->get_extensions(&extensions);

		for (List<String>::Element *F = extensions.front(); F; F = F->next()) {

			if (F->get().to_lower() == ext) {

				importer = E->get();
				break;
			}
		}

		if (importer.is_valid())
			break;
	}

	ERR_FAIL_COND_V(!importer.is_valid(), NULL);

	List<String> missing;
	Error err;
	return importer->import_scene(p_path, p_flags, p_bake_fps, &missing, &err);
}

Ref<Animation> ResourceImporterScene::import_animation_from_other_importer(EditorSceneImporter *p_exception, const String &p_path, uint32_t p_flags, int p_bake_fps) {

	Ref<EditorSceneImporter> importer;
	String ext = p_path.get_extension().to_lower();

	for (Set<Ref<EditorSceneImporter> >::Element *E = importers.front(); E; E = E->next()) {

		if (E->get().ptr() == p_exception)
			continue;
		List<String> extensions;
		E->get()->get_extensions(&extensions);

		for (List<String>::Element *F = extensions.front(); F; F = F->next()) {

			if (F->get().to_lower() == ext) {

				importer = E->get();
				break;
			}
		}

		if (importer.is_valid())
			break;
	}

	ERR_FAIL_COND_V(!importer.is_valid(), NULL);

	return importer->import_animation(p_path, p_flags, p_bake_fps);
}

Error ResourceImporterScene::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files) {

	String src_path = p_source_file;

	Ref<EditorSceneImporter> importer;
	String ext = src_path.get_extension().to_lower();

	EditorProgress progress("import", TTR("Import Scene"), 104);
	progress.step(TTR("Importing Scene.."), 0);

	for (Set<Ref<EditorSceneImporter> >::Element *E = importers.front(); E; E = E->next()) {

		List<String> extensions;
		E->get()->get_extensions(&extensions);

		for (List<String>::Element *F = extensions.front(); F; F = F->next()) {

			if (F->get().to_lower() == ext) {

				importer = E->get();
				break;
			}
		}

		if (importer.is_valid())
			break;
	}

	ERR_FAIL_COND_V(!importer.is_valid(), ERR_FILE_UNRECOGNIZED);

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
	List<String> missing_deps; // for now, not much will be done with this
	Node *scene = importer->import_scene(src_path, import_flags, fps, &missing_deps, &err);
	if (!scene || err != OK) {
		return err;
	}

	String root_type = p_options["nodes/root_type"];

	if (scene->get_class() != root_type) {
		Node *base_node = Object::cast_to<Node>(ClassDB::instance(root_type));

		if (base_node) {

			scene->replace_by(base_node);
			memdelete(scene);
			scene = base_node;
		}
	}

	if (Object::cast_to<Spatial>(scene)) {
		float root_scale = p_options["nodes/root_scale"];
		Object::cast_to<Spatial>(scene)->scale(Vector3(root_scale, root_scale, root_scale));
	}

	scene->set_name(p_options["nodes/root_name"]);

	err = OK;

	String animation_filter = String(p_options["animation/filter_script"]).strip_edges();

	bool use_optimizer = p_options["animation/optimizer/enabled"];
	float anim_optimizer_linerr = p_options["animation/optimizer/max_linear_error"];
	float anim_optimizer_angerr = p_options["animation/optimizer/max_angular_error"];
	float anim_optimizer_maxang = p_options["animation/optimizer/max_angle"];
	int light_bake_mode = p_options["meshes/light_baking"];

	Map<Ref<ArrayMesh>, Ref<Shape> > collision_map;

	scene = _fix_node(scene, scene, collision_map, LightBakeMode(light_bake_mode));

	if (use_optimizer) {
		_optimize_animations(scene, anim_optimizer_linerr, anim_optimizer_angerr, anim_optimizer_maxang);
	}

	Array animation_clips;
	{

		int clip_count = p_options["animation/clips/amount"];

		for (int i = 0; i < clip_count; i++) {
			String name = p_options["animation/clip_" + itos(i + 1) + "/name"];
			int from_frame = p_options["animation/clip_" + itos(i + 1) + "/start_frame"];
			int end_frame = p_options["animation/clip_" + itos(i + 1) + "/end_frame"];
			bool loop = p_options["animation/clip_" + itos(i + 1) + "/loops"];

			animation_clips.push_back(name);
			animation_clips.push_back(from_frame / fps);
			animation_clips.push_back(end_frame / fps);
			animation_clips.push_back(loop);
		}
	}
	if (animation_clips.size()) {
		_create_clips(scene, animation_clips, !bool(p_options["animation/optimizer/remove_unused_tracks"]));
	}

	if (animation_filter != "") {
		_filter_tracks(scene, animation_filter);
	}

	bool external_animations = int(p_options["animation/storage"]) == 1;
	bool keep_custom_tracks = p_options["animation/keep_custom_tracks"];
	bool external_materials = p_options["materials/storage"];
	bool external_meshes = p_options["meshes/storage"];
	bool external_scenes = int(p_options["nodes/storage"]) == 1;

	String base_path = p_source_file.get_base_dir();

	if (external_animations || external_materials || external_meshes || external_scenes) {

		if (bool(p_options["external_files/store_in_subdir"])) {
			String subdir_name = p_source_file.get_file().get_basename();
			DirAccess *da = DirAccess::open(base_path);
			print_line("at path " + da->get_current_dir() + " making " + subdir_name);
			Error err = da->make_dir(subdir_name);
			memdelete(da);
			ERR_FAIL_COND_V(err != OK && err != ERR_ALREADY_EXISTS, err);
			base_path = base_path.plus_file(subdir_name);
		}
	}

	if (light_bake_mode == 2 /* || generate LOD */) {

		Map<Ref<ArrayMesh>, Transform> meshes;
		_find_meshes(scene, meshes);

		if (light_bake_mode == 2) {

			float texel_size = p_options["meshes/lightmap_texel_size"];
			texel_size = MAX(0.001, texel_size);

			EditorProgress progress("gen_lightmaps", TTR("Generating Lightmaps"), meshes.size());
			int step = 0;
			for (Map<Ref<ArrayMesh>, Transform>::Element *E = meshes.front(); E; E = E->next()) {

				Ref<ArrayMesh> mesh = E->key();
				String name = mesh->get_name();
				if (name == "") { //should not happen but..
					name = "Mesh " + itos(step);
				}

				progress.step(TTR("Generating for Mesh: ") + name + " (" + itos(step) + "/" + itos(meshes.size()) + ")", step);

				Error err = mesh->lightmap_unwrap(E->get(), texel_size);
				if (err != OK) {
					EditorNode::add_io_error("Mesh '" + name + "' failed lightmap generation. Please fix geometry.");
				}
				step++;
			}
		}
	}

	if (external_animations || external_materials || external_meshes) {
		Map<Ref<Animation>, Ref<Animation> > anim_map;
		Map<Ref<Material>, Ref<Material> > mat_map;
		Map<Ref<ArrayMesh>, Ref<ArrayMesh> > mesh_map;

		bool keep_materials = bool(p_options["materials/keep_on_reimport"]);

		_make_external_resources(scene, base_path, external_animations, keep_custom_tracks, external_materials, keep_materials, external_meshes, anim_map, mat_map, mesh_map);
	}

	progress.step(TTR("Running Custom Script.."), 2);

	String post_import_script_path = p_options["nodes/custom_script"];
	Ref<EditorScenePostImport> post_import_script;

	if (post_import_script_path != "") {
		Ref<Script> scr = ResourceLoader::load(post_import_script_path);
		if (!scr.is_valid()) {
			EditorNode::add_io_error(TTR("Couldn't load post-import script:") + " " + post_import_script_path);
		} else {

			post_import_script = Ref<EditorScenePostImport>(memnew(EditorScenePostImport));
			post_import_script->set_script(scr.get_ref_ptr());
			if (!post_import_script->get_script_instance()) {
				EditorNode::add_io_error(TTR("Invalid/broken script for post-import (check console):") + " " + post_import_script_path);
				post_import_script.unref();
				return ERR_CANT_CREATE;
			}
		}
	}

	if (post_import_script.is_valid()) {
		scene = post_import_script->post_import(scene);
		if (!scene) {
			EditorNode::add_io_error(TTR("Error running post-import script:") + " " + post_import_script_path);
			return err;
		}
	}

	progress.step(TTR("Saving.."), 104);

	if (external_scenes) {
		//save sub-scenes as instances!
		for (int i = 0; i < scene->get_child_count(); i++) {
			Node *child = scene->get_child(i);
			if (child->get_owner() != scene)
				continue; //not a real child probably created by scene type (ig, a scrollbar)
			_replace_owner(child, scene, child);

			String cn = String(child->get_name()).strip_edges().replace(".", "_").replace(":", "_");
			if (cn == String()) {
				cn = "ChildNode" + itos(i);
			}
			String path = base_path.plus_file(cn + ".scn");
			child->set_filename(path);

			Ref<PackedScene> packer = memnew(PackedScene);
			packer->pack(child);
			err = ResourceSaver::save(path, packer); //do not take over, let the changed files reload themselves
			ERR_FAIL_COND_V(err != OK, err);
		}
	}

	Ref<PackedScene> packer = memnew(PackedScene);
	packer->pack(scene);
	print_line("SAVING TO: " + p_save_path + ".scn");
	err = ResourceSaver::save(p_save_path + ".scn", packer); //do not take over, let the changed files reload themselves
	ERR_FAIL_COND_V(err != OK, err);

	memdelete(scene);

	//this is not the time to reimport, wait until import process is done, import file is saved, etc.
	//EditorNode::get_singleton()->reload_scene(p_source_file);

	return OK;
}

ResourceImporterScene *ResourceImporterScene::singleton = NULL;

ResourceImporterScene::ResourceImporterScene() {
	singleton = this;
}
