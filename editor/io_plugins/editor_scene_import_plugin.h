/*************************************************************************/
/*  editor_scene_import_plugin.h                                         */
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
#ifndef EDITOR_SCENE_IMPORT_PLUGIN_H
#define EDITOR_SCENE_IMPORT_PLUGIN_H
#if 0
#include "editor/editor_dir_dialog.h"
#include "editor/editor_file_system.h"
#include "editor/editor_import_export.h"
#include "editor/io_plugins/editor_texture_import_plugin.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/tree.h"
#include "scene/resources/animation.h"
#include "scene/resources/mesh.h"


class EditorNode;
class EditorSceneImportDialog;

class EditorSceneImporter : public Reference {

	GDCLASS(EditorSceneImporter,Reference );
public:

	enum ImportFlags {
		IMPORT_SCENE=1,
		IMPORT_ANIMATION=2,
		IMPORT_ANIMATION_DETECT_LOOP=4,
		IMPORT_ANIMATION_OPTIMIZE=8,
		IMPORT_ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS=16,
		IMPORT_ANIMATION_KEEP_VALUE_TRACKS=32,
		IMPORT_GENERATE_TANGENT_ARRAYS=256,
		IMPORT_FAIL_ON_MISSING_DEPENDENCIES=512

	};

	virtual uint32_t get_import_flags() const=0;
	virtual void get_extensions(List<String> *r_extensions) const=0;
	virtual Node* import_scene(const String& p_path,uint32_t p_flags,int p_bake_fps,List<String> *r_missing_deps,Error* r_err=NULL)=0;
	virtual Ref<Animation> import_animation(const String& p_path,uint32_t p_flags)=0;



	EditorSceneImporter();
};

/////////////////////////////////////////


//Plugin for post processing scenes or images

class EditorScenePostImport : public Reference {

	GDCLASS(EditorScenePostImport,Reference );
protected:

	static void _bind_methods();
public:

	virtual Node* post_import(Node* p_scene);
	EditorScenePostImport();
};


class EditorSceneImportPlugin : public EditorImportPlugin {

	GDCLASS(EditorSceneImportPlugin,EditorImportPlugin);

	EditorSceneImportDialog *dialog;

	Vector<Ref<EditorSceneImporter> > importers;

	enum TextureRole {
		TEXTURE_ROLE_DEFAULT,
		TEXTURE_ROLE_DIFFUSE,
		TEXTURE_ROLE_NORMALMAP
	};

	void _find_resources(const Variant& p_var,Map<Ref<ImageTexture>,TextureRole >& image_map,int p_flags);
	Node* _fix_node(Node *p_node,Node *p_root,Map<Ref<Mesh>,Ref<Shape> > &collision_map,uint32_t p_flags,Map<Ref<ImageTexture>,TextureRole >& image_map);
	void _create_clips(Node *scene, const Array& p_clips, bool p_bake_all);
	void _filter_anim_tracks(Ref<Animation> anim,Set<String> &keep);
	void _filter_tracks(Node *scene, const String& p_text);
	void _optimize_animations(Node *scene, float p_max_lin_error,float p_max_ang_error,float p_max_angle);

	void _tag_import_paths(Node *p_scene,Node *p_node);

	void _find_resources_to_merge(Node *scene, Node *node, bool p_merge_material, Map<String,Ref<Material> >&materials, bool p_merge_anims, Map<String,Ref<Animation> >& merged_anims, Set<Ref<Mesh> > &tested_meshes);
	void _merge_found_resources(Node *scene, Node *node, bool p_merge_material, const Map<String, Ref<Material> > &materials, bool p_merge_anims, const Map<String,Ref<Animation> >& merged_anims, Set<Ref<Mesh> > &tested_meshes);


public:

	enum SceneFlags {

		SCENE_FLAG_CREATE_COLLISIONS=1<<0,
		SCENE_FLAG_CREATE_PORTALS=1<<1,
		SCENE_FLAG_CREATE_ROOMS=1<<2,
		SCENE_FLAG_SIMPLIFY_ROOMS=1<<3,
		SCENE_FLAG_CREATE_BILLBOARDS=1<<4,
		SCENE_FLAG_CREATE_IMPOSTORS=1<<5,
		SCENE_FLAG_CREATE_LODS=1<<6,
		SCENE_FLAG_CREATE_CARS=1<<8,
		SCENE_FLAG_CREATE_WHEELS=1<<9,
		SCENE_FLAG_DETECT_ALPHA=1<<15,
		SCENE_FLAG_DETECT_VCOLOR=1<<16,
		SCENE_FLAG_CREATE_NAVMESH=1<<17,
		SCENE_FLAG_DETECT_LIGHTMAP_LAYER=1<<18,

		SCENE_FLAG_MERGE_KEEP_MATERIALS=1<<20,
		SCENE_FLAG_MERGE_KEEP_EXTRA_ANIM_TRACKS=1<<21,

		SCENE_FLAG_REMOVE_NOIMP=1<<24,
		SCENE_FLAG_IMPORT_ANIMATIONS=1<<25,
		SCENE_FLAG_COMPRESS_GEOMETRY=1<<26,
		SCENE_FLAG_GENERATE_TANGENT_ARRAYS=1<<27,
		SCENE_FLAG_LINEARIZE_DIFFUSE_TEXTURES=1<<28,
		SCENE_FLAG_SET_LIGHTMAP_TO_UV2_IF_EXISTS=1<<29,
		SCENE_FLAG_CONVERT_NORMALMAPS_TO_XY=1<<30,
	};



	virtual String get_name() const;
	virtual String get_visible_name() const;
	virtual void import_dialog(const String& p_from="");
	virtual Error import(const String& p_path, const Ref<ResourceImportMetadata>& p_from);

	Error import1(const Ref<ResourceImportMetadata>& p_from,Node**r_node,List<String> *r_missing=NULL);
	Error import2(Node* p_scene,const String& p_path, const Ref<ResourceImportMetadata>& p_from);

	void add_importer(const Ref<EditorSceneImporter>& p_importer);
	const Vector<Ref<EditorSceneImporter> >& get_importers() { return importers; }

	virtual void import_from_drop(const Vector<String>& p_drop,const String& p_dest_path);

	EditorSceneImportPlugin(EditorNode* p_editor=NULL);


};


class EditorSceneAnimationImportPlugin : public EditorImportPlugin {

	GDCLASS(EditorSceneAnimationImportPlugin,EditorImportPlugin);
public:


	enum AnimationFlags {

		ANIMATION_DETECT_LOOP=1,
		ANIMATION_KEEP_VALUE_TRACKS=2,
		ANIMATION_OPTIMIZE=4,
		ANIMATION_FORCE_ALL_TRACKS_IN_ALL_CLIPS=8
	};

	virtual String get_name() const;
	virtual String get_visible_name() const;
	virtual void import_dialog(const String& p_from="");
	virtual Error import(const String& p_path, const Ref<ResourceImportMetadata>& p_from);

	EditorSceneAnimationImportPlugin(EditorNode* p_editor=NULL);


};

#endif
#endif // EDITOR_SCENE_IMPORT_PLUGIN_H
