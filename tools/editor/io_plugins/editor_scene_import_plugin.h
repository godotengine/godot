/*************************************************************************/
/*  editor_scene_import_plugin.h                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

#include "scene/gui/dialogs.h"
#include "scene/gui/tree.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/resources/mesh.h"
#include "tools/editor/editor_file_system.h"
#include "tools/editor/editor_dir_dialog.h"
#include "tools/editor/editor_import_export.h"
#include "tools/editor/io_plugins/editor_texture_import_plugin.h"
#include "scene/resources/animation.h"

class EditorNode;
class EditorSceneImportDialog;

class EditorSceneImporter : public Reference {

	OBJ_TYPE(EditorSceneImporter,Reference );
public:

	enum ImportFlags {
		IMPORT_SCENE=1,
		IMPORT_ANIMATION=2,
		IMPORT_ANIMATION_DETECT_LOOP=4,
		IMPORT_ANIMATION_OPTIMIZE=8,
		IMPORT_GENERATE_TANGENT_ARRAYS=16,
		IMPORT_FAIL_ON_MISSING_DEPENDENCIES=128

	};

	virtual uint32_t get_import_flags() const=0;
	virtual void get_extensions(List<String> *r_extensions) const=0;
	virtual Node* import_scene(const String& p_path,uint32_t p_flags,Error* r_err=NULL)=0;
	virtual Ref<Animation> import_animation(const String& p_path,uint32_t p_flags)=0;



	EditorSceneImporter();
};

/////////////////////////////////////////


//Plugin for post processing scenes or images

class EditorScenePostImport : public Reference {

	OBJ_TYPE(EditorScenePostImport,Reference );
protected:

	static void _bind_methods();
public:

	virtual Error post_import(Node* p_scene);
	EditorScenePostImport();
};


class EditorSceneImportPlugin : public EditorImportPlugin {

	OBJ_TYPE(EditorSceneImportPlugin,EditorImportPlugin);

	EditorSceneImportDialog *dialog;

	Vector<Ref<EditorSceneImporter> > importers;

	void _find_resources(const Variant& p_var,Set<Ref<ImageTexture> >& image_map);
	Node* _fix_node(Node *p_node,Node *p_root,Map<Ref<Mesh>,Ref<Shape> > &collision_map,uint32_t p_flags,Set<Ref<ImageTexture> >& image_map);
	void _merge_node(Node *p_node,Node*p_root,Node *p_existing,Set<Ref<Resource> >& checked_resources);
	void _merge_scenes(Node *p_existing,Node *p_new);


public:

	enum SceneFlags {

		SCENE_FLAG_CREATE_COLLISIONS=1,
		SCENE_FLAG_CREATE_PORTALS=2,
		SCENE_FLAG_CREATE_ROOMS=4,
		SCENE_FLAG_SIMPLIFY_ROOMS=8,
		SCENE_FLAG_CREATE_BILLBOARDS=16,
		SCENE_FLAG_CREATE_IMPOSTORS=32,
		SCENE_FLAG_CREATE_LODS=64,
		SCENE_FLAG_REMOVE_NOIMP=128,
		SCENE_FLAG_IMPORT_ANIMATIONS=256,
		SCENE_FLAG_COMPRESS_GEOMETRY=512,
		SCENE_FLAG_FAIL_ON_MISSING_IMAGES=1024,
		SCENE_FLAG_GENERATE_TANGENT_ARRAYS=2048,
		SCENE_FLAG_DONT_SAVE_TO_DB=8192
	};


	virtual String get_name() const;
	virtual String get_visible_name() const;
	virtual void import_dialog(const String& p_from="");
	virtual Error import(const String& p_path, const Ref<ResourceImportMetadata>& p_from);

	void add_importer(const Ref<EditorSceneImporter>& p_importer);
	const Vector<Ref<EditorSceneImporter> >& get_importers() { return importers; }

	EditorSceneImportPlugin(EditorNode* p_editor=NULL);


};


class EditorSceneAnimationImportPlugin : public EditorImportPlugin {

	OBJ_TYPE(EditorSceneAnimationImportPlugin,EditorImportPlugin);
public:


	enum AnimationFlags {

		ANIMATION_DETECT_LOOP=1,
		ANIMATION_KEEP_VALUE_TRACKS=2,
		ANIMATION_OPTIMIZE=4
	};

	virtual String get_name() const;
	virtual String get_visible_name() const;
	virtual void import_dialog(const String& p_from="");
	virtual Error import(const String& p_path, const Ref<ResourceImportMetadata>& p_from);

	EditorSceneAnimationImportPlugin(EditorNode* p_editor=NULL);


};



#endif // EDITOR_SCENE_IMPORT_PLUGIN_H
