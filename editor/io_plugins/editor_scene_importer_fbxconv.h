/*************************************************************************/
/*  editor_scene_importer_fbxconv.h                                      */
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
#ifndef EDITOR_SCENE_IMPORTER_FBXCONV_H
#define EDITOR_SCENE_IMPORTER_FBXCONV_H

#include "editor/io_plugins/editor_scene_import_plugin.h"
#include "scene/3d/skeleton.h"

#if 0

class EditorSceneImporterFBXConv : public EditorSceneImporter {

	GDCLASS(EditorSceneImporterFBXConv,EditorSceneImporter );


	struct BoneInfo {

		Skeleton *skeleton;
		Transform rest;
		int index;
		bool has_anim_chan;
		bool has_rest;
		Dictionary node;
		BoneInfo() {
			has_rest=false;
			skeleton=NULL;
			index=-1;
			has_anim_chan=false;
		}
	};

	struct SurfaceInfo {
		Array array;
		Mesh::PrimitiveType primitive;
	};

	struct State {

		Node *scene;
		Array meshes;
		Array materials;
		Array nodes;
		Array animations;
		Map<String,BoneInfo > bones;
		Map<String,Skeleton*> skeletons;
		Map<String,Ref<Mesh> > mesh_cache;
		Map<String,SurfaceInfo> surface_cache;
		Map<String,Ref<Material> > material_cache;
		Map<String,Ref<Texture> > texture_cache;
		List<String> *missing_deps;
		String base_path;
		bool import_animations;
	};

	String _id(const String& p_id) const;

	Transform _get_transform_mixed(const Dictionary& d, const Dictionary& dbase);
	Transform _get_transform(const Dictionary& d);
	Color _get_color(const Array& a);
	void _detect_bones_in_nodes(State& state,const Array& p_nodes);
	void _detect_bones(State& state);

	Error _parse_bones(State& state,const Array &p_bones,Skeleton* p_skeleton);
	void _parse_skeletons(const String& p_name,State& state, const Array &p_nodes, Skeleton*p_skeleton=NULL, int p_parent=-1);

	void _add_surface(State& state,Ref<Mesh>& m,const Dictionary &part);
	Error _parse_nodes(State& state,const Array &p_nodes,Node* p_base);
	Error _parse_animations(State& state);
	void _parse_materials(State& state);
	void _parse_surfaces(State& state);
	Error _parse_json(State& state,const String& p_path);
	Error _parse_fbx(State &state, const String &p_path);

public:

	virtual uint32_t get_import_flags() const;
	virtual void get_extensions(List<String> *r_extensions) const;
	virtual Node* import_scene(const String& p_path,uint32_t p_flags,List<String> *r_missing_deps=NULL,Error* r_err=NULL);
	virtual Ref<Animation> import_animation(const String& p_path,uint32_t p_flags);

	EditorSceneImporterFBXConv();
};

#endif // EDITOR_SCENE_IMPORTER_FBXCONV_H
#endif
