/*************************************************************************/
/*  resource_importer_scene.h                                            */
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

#ifndef RESOURCEIMPORTERSCENE_H
#define RESOURCEIMPORTERSCENE_H

#include "core/io/resource_importer.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/animation.h"
#include "scene/resources/mesh.h"
#include "scene/resources/shape_3d.h"
#include "scene/resources/skin.h"

class Material;
class AnimationPlayer;

class EditorSceneImporterMesh;
class EditorSceneImporter : public Reference {
	GDCLASS(EditorSceneImporter, Reference);

protected:
	static void _bind_methods();

	Node *import_scene_from_other_importer(const String &p_path, uint32_t p_flags, int p_bake_fps);
	Ref<Animation> import_animation_from_other_importer(const String &p_path, uint32_t p_flags, int p_bake_fps);

public:
	enum ImportFlags {
		IMPORT_SCENE = 1,
		IMPORT_ANIMATION = 2,
		IMPORT_FAIL_ON_MISSING_DEPENDENCIES = 4,
		IMPORT_GENERATE_TANGENT_ARRAYS = 8,
		IMPORT_USE_NAMED_SKIN_BINDS = 16,

	};

	virtual uint32_t get_import_flags() const;
	virtual void get_extensions(List<String> *r_extensions) const;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err = nullptr);
	virtual Ref<Animation> import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps);

	EditorSceneImporter() {}
};

class EditorScenePostImport : public Reference {
	GDCLASS(EditorScenePostImport, Reference);

	String source_file;

protected:
	static void _bind_methods();

public:
	String get_source_file() const;
	virtual Node *post_import(Node *p_scene);
	virtual void init(const String &p_source_file);
	EditorScenePostImport();
};

class ResourceImporterScene : public ResourceImporter {
	GDCLASS(ResourceImporterScene, ResourceImporter);

	Set<Ref<EditorSceneImporter>> importers;

	static ResourceImporterScene *singleton;

	enum LightBakeMode {
		LIGHT_BAKE_DISABLED,
		LIGHT_BAKE_DYNAMIC,
		LIGHT_BAKE_STATIC,
		LIGHT_BAKE_STATIC_LIGHTMAPS
	};

	enum MeshPhysicsMode {
		MESH_PHYSICS_DISABLED,
		MESH_PHYSICS_MESH_AND_STATIC_COLLIDER,
		MESH_PHYSICS_RIGID_BODY_AND_MESH,
		MESH_PHYSICS_STATIC_COLLIDER_ONLY,
		MESH_PHYSICS_AREA_ONLY,
	};

	enum NavMeshMode {
		NAVMESH_DISABLED,
		NAVMESH_MESH_AND_NAVMESH,
		NAVMESH_NAVMESH_ONLY,
	};

	enum MeshOverride {
		MESH_OVERRIDE_DEFAULT,
		MESH_OVERRIDE_ENABLE,
		MESH_OVERRIDE_DISABLE,
	};

	void _replace_owner(Node *p_node, Node *p_scene, Node *p_new_owner);
	void _generate_meshes(Node *p_node, const Dictionary &p_mesh_data, bool p_generate_lods, bool p_create_shadow_meshes, LightBakeMode p_light_bake_mode, float p_lightmap_texel_size, const Vector<uint8_t> &p_src_lightmap_cache, Vector<Vector<uint8_t>> &r_lightmap_caches);
	void _add_shapes(Node *p_node, const List<Ref<Shape3D>> &p_shapes);

public:
	static ResourceImporterScene *get_singleton() { return singleton; }

	const Set<Ref<EditorSceneImporter>> &get_importers() const { return importers; }

	void add_importer(Ref<EditorSceneImporter> p_importer) { importers.insert(p_importer); }
	void remove_importer(Ref<EditorSceneImporter> p_importer) { importers.erase(p_importer); }

	virtual String get_importer_name() const override;
	virtual String get_visible_name() const override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual String get_save_extension() const override;
	virtual String get_resource_type() const override;
	virtual int get_format_version() const override;

	virtual int get_preset_count() const override;
	virtual String get_preset_name(int p_idx) const override;

	enum InternalImportCategory {
		INTERNAL_IMPORT_CATEGORY_NODE,
		INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE,
		INTERNAL_IMPORT_CATEGORY_MESH,
		INTERNAL_IMPORT_CATEGORY_MATERIAL,
		INTERNAL_IMPORT_CATEGORY_ANIMATION,
		INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE,
		INTERNAL_IMPORT_CATEGORY_MAX
	};

	void get_internal_import_options(InternalImportCategory p_category, List<ImportOption> *r_options) const;
	bool get_internal_option_visibility(InternalImportCategory p_category, const String &p_option, const Map<StringName, Variant> &p_options) const;

	virtual void get_import_options(List<ImportOption> *r_options, int p_preset = 0) const override;
	virtual bool get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const override;
	virtual int get_import_order() const override { return 100; } //after everything

	Node *_pre_fix_node(Node *p_node, Node *p_root, Map<Ref<EditorSceneImporterMesh>, List<Ref<Shape3D>>> &collision_map);
	Node *_post_fix_node(Node *p_node, Node *p_root, Map<Ref<EditorSceneImporterMesh>, List<Ref<Shape3D>>> &collision_map, Set<Ref<EditorSceneImporterMesh>> &r_scanned_meshes, const Dictionary &p_node_data, const Dictionary &p_material_data, const Dictionary &p_animation_data, float p_animation_fps);

	Ref<Animation> _save_animation_to_file(Ref<Animation> anim, bool p_save_to_file, String p_save_to_path, bool p_keep_custom_tracks);
	void _create_clips(AnimationPlayer *anim, const Array &p_clips, bool p_bake_all);
	void _optimize_animations(AnimationPlayer *anim, float p_max_lin_error, float p_max_ang_error, float p_max_angle);

	Node *pre_import(const String &p_source_file);
	virtual Error import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = nullptr, Variant *r_metadata = nullptr) override;

	Node *import_scene_from_other_importer(EditorSceneImporter *p_exception, const String &p_path, uint32_t p_flags, int p_bake_fps);
	Ref<Animation> import_animation_from_other_importer(EditorSceneImporter *p_exception, const String &p_path, uint32_t p_flags, int p_bake_fps);

	virtual bool has_advanced_options() const override;
	virtual void show_advanced_options(const String &p_path) override;

	virtual bool can_import_threaded() const override { return false; }

	ResourceImporterScene();
};

class EditorSceneImporterESCN : public EditorSceneImporter {
	GDCLASS(EditorSceneImporterESCN, EditorSceneImporter);

public:
	virtual uint32_t get_import_flags() const override;
	virtual void get_extensions(List<String> *r_extensions) const override;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err = nullptr) override;
	virtual Ref<Animation> import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps) override;
};

#endif // RESOURCEIMPORTERSCENE_H
