/*************************************************************************/
/*  resource_importer_scene.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef RESOURCEIMPORTERSCENE_H
#define RESOURCEIMPORTERSCENE_H

#include "io/resource_import.h"
#include "scene/resources/animation.h"
#include "scene/resources/shape.h"

class Material;

class EditorSceneImporter : public Reference {

	GDCLASS(EditorSceneImporter, Reference);

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
		IMPORT_MATERIALS_IN_INSTANCES = 1024

	};

	virtual uint32_t get_import_flags() const = 0;
	virtual void get_extensions(List<String> *r_extensions) const = 0;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err = NULL) = 0;
	virtual Ref<Animation> import_animation(const String &p_path, uint32_t p_flags) = 0;

	EditorSceneImporter() {}
};

class EditorScenePostImport : public Reference {

	GDCLASS(EditorScenePostImport, Reference);

protected:
	static void _bind_methods();

public:
	virtual Node *post_import(Node *p_scene);
	EditorScenePostImport();
};

class ResourceImporterScene : public ResourceImporter {
	GDCLASS(ResourceImporterScene, ResourceImporter)

	Set<Ref<EditorSceneImporter> > importers;

	static ResourceImporterScene *singleton;

public:
	static ResourceImporterScene *get_singleton() { return singleton; }

	const Set<Ref<EditorSceneImporter> > &get_importers() const { return importers; }

	void add_importer(Ref<EditorSceneImporter> p_importer) { importers.insert(p_importer); }

	virtual String get_importer_name() const;
	virtual String get_visible_name() const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual String get_save_extension() const;
	virtual String get_resource_type() const;

	virtual int get_preset_count() const;
	virtual String get_preset_name(int p_idx) const;

	virtual void get_import_options(List<ImportOption> *r_options, int p_preset = 0) const;
	virtual bool get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const;

	void _make_external_resources(Node *p_node, const String &p_base_path, bool p_make_materials, bool p_make_meshes, Map<Ref<Material>, Ref<Material> > &p_materials, Map<Ref<Mesh>, Ref<Mesh> > &p_meshes);

	Node *_fix_node(Node *p_node, Node *p_root, Map<Ref<Mesh>, Ref<Shape> > &collision_map);

	void _create_clips(Node *scene, const Array &p_clips, bool p_bake_all);
	void _filter_anim_tracks(Ref<Animation> anim, Set<String> &keep);
	void _filter_tracks(Node *scene, const String &p_text);
	void _optimize_animations(Node *scene, float p_max_lin_error, float p_max_ang_error, float p_max_angle);

	virtual Error import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = NULL);

	ResourceImporterScene();
};

#endif // RESOURCEIMPORTERSCENE_H
