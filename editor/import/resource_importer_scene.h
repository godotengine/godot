/**************************************************************************/
/*  resource_importer_scene.h                                             */
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

#ifndef RESOURCE_IMPORTER_SCENE_H
#define RESOURCE_IMPORTER_SCENE_H

#include "core/io/resource_importer.h"
#include "scene/resources/animation.h"
#include "scene/resources/mesh.h"
#include "scene/resources/shape.h"

class Material;
class EditorSceneImporter : public Reference {
	GDCLASS(EditorSceneImporter, Reference);

protected:
	static void _bind_methods();

	Node *import_scene_from_other_importer(const String &p_path, uint32_t p_flags, int p_bake_fps, uint32_t p_compress_flags);
	Ref<Animation> import_animation_from_other_importer(const String &p_path, uint32_t p_flags, int p_bake_fps);

public:
	enum ImportFlags {
		IMPORT_SCENE = 1,
		IMPORT_ANIMATION = 2,
		IMPORT_ANIMATION_DETECT_LOOP = 4,
		IMPORT_ANIMATION_OPTIMIZE = 8,
		IMPORT_ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS = 16,
		IMPORT_ANIMATION_KEEP_VALUE_TRACKS = 32,
		IMPORT_GENERATE_TANGENT_ARRAYS = 256,
		IMPORT_FAIL_ON_MISSING_DEPENDENCIES = 512,
		IMPORT_MATERIALS_IN_INSTANCES = 1024,
		IMPORT_USE_COMPRESSION = 2048,
		IMPORT_USE_NAMED_SKIN_BINDS = 4096,
		IMPORT_USE_LEGACY_NAMES = 8192,

	};

	virtual uint32_t get_import_flags() const;
	virtual void get_extensions(List<String> *r_extensions) const;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, uint32_t p_compress_flags, List<String> *r_missing_deps, Error *r_err = nullptr);
	virtual Ref<Animation> import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps);

	EditorSceneImporter() {}
};

class EditorScenePostImport : public Reference {
	GDCLASS(EditorScenePostImport, Reference);

	String source_folder;
	String source_file;

protected:
	static void _bind_methods();

public:
	String get_source_folder() const;
	String get_source_file() const;
	virtual Node *post_import(Node *p_scene);
	virtual void init(const String &p_source_folder, const String &p_source_file);
	EditorScenePostImport();
};

class ResourceImporterScene : public ResourceImporter {
	GDCLASS(ResourceImporterScene, ResourceImporter);

	Set<Ref<EditorSceneImporter>> importers;

	static ResourceImporterScene *singleton;

	enum Presets {
		PRESET_SEPARATE_MATERIALS,
		PRESET_SEPARATE_MESHES,
		PRESET_SEPARATE_ANIMATIONS,

		PRESET_SINGLE_SCENE,

		PRESET_SEPARATE_MESHES_AND_MATERIALS,
		PRESET_SEPARATE_MESHES_AND_ANIMATIONS,
		PRESET_SEPARATE_MATERIALS_AND_ANIMATIONS,
		PRESET_SEPARATE_MESHES_MATERIALS_AND_ANIMATIONS,

		PRESET_MULTIPLE_SCENES,
		PRESET_MULTIPLE_SCENES_AND_MATERIALS,
		PRESET_MAX
	};

	enum LightBakeMode {
		LIGHT_BAKE_DISABLED,
		LIGHT_BAKE_ENABLE,
		LIGHT_BAKE_LIGHTMAPS
	};

	void _replace_owner(Node *p_node, Node *p_scene, Node *p_new_owner);
	void _add_shapes(Node *p_node, const List<Ref<Shape>> &p_shapes);

public:
	static ResourceImporterScene *get_singleton() { return singleton; }

	const Set<Ref<EditorSceneImporter>> &get_importers() const { return importers; }

	void add_importer(Ref<EditorSceneImporter> p_importer) { importers.insert(p_importer); }
	void remove_importer(Ref<EditorSceneImporter> p_importer) { importers.erase(p_importer); }

	virtual String get_importer_name() const;
	virtual String get_visible_name() const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual String get_save_extension() const;
	virtual String get_resource_type() const;

	virtual int get_preset_count() const;
	virtual String get_preset_name(int p_idx) const;

	virtual void get_import_options(List<ImportOption> *r_options, int p_preset = 0) const;
	virtual bool get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const;
	// Import scenes *after* everything else (such as textures).
	virtual int get_import_order() const { return ResourceImporter::IMPORT_ORDER_SCENE; }

	void _find_meshes(Node *p_node, Map<Ref<ArrayMesh>, Transform> &meshes);

	void _make_external_resources(Node *p_node, const String &p_base_path, bool p_make_animations, bool p_animations_as_text, bool p_keep_animations, bool p_make_materials, bool p_materials_as_text, bool p_keep_materials, bool p_make_meshes, bool p_meshes_as_text, Map<Ref<Animation>, Ref<Animation>> &p_animations, Map<Ref<Material>, Ref<Material>> &p_materials, Map<Ref<ArrayMesh>, Ref<ArrayMesh>> &p_meshes);

	Node *_fix_node(Node *p_node, Node *p_root, Map<Ref<Mesh>, List<Ref<Shape>>> &collision_map, LightBakeMode p_light_bake_mode, List<Pair<NodePath, Node *>> &r_node_renames);

	void _create_clips(Node *scene, const Array &p_clips, bool p_bake_all);
	void _filter_anim_tracks(Ref<Animation> anim, Set<String> &keep);
	void _filter_tracks(Node *scene, const String &p_text);
	void _optimize_animations(Node *scene, float p_max_lin_error, float p_max_ang_error, float p_max_angle);

	virtual Error import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = nullptr, Variant *r_metadata = nullptr);

	Node *import_scene_from_other_importer(EditorSceneImporter *p_exception, const String &p_path, uint32_t p_flags, int p_bake_fps, uint32_t p_compress_flags);
	Ref<Animation> import_animation_from_other_importer(EditorSceneImporter *p_exception, const String &p_path, uint32_t p_flags, int p_bake_fps);

	ResourceImporterScene();
};

class EditorSceneImporterESCN : public EditorSceneImporter {
	GDCLASS(EditorSceneImporterESCN, EditorSceneImporter);

public:
	virtual uint32_t get_import_flags() const;
	virtual void get_extensions(List<String> *r_extensions) const;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, uint32_t p_compress_flags, List<String> *r_missing_deps, Error *r_err = nullptr);
	virtual Ref<Animation> import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps);
};

#endif // RESOURCE_IMPORTER_SCENE_H
