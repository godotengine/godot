#ifndef EDITOR_SCENE_IMPORTER_FBXCONV_H
#define EDITOR_SCENE_IMPORTER_FBXCONV_H

#include "tools/editor/io_plugins/editor_scene_import_plugin.h"
#include "scene/3d/skeleton.h"


class EditorSceneImporterFBXConv : public EditorSceneImporter {

	OBJ_TYPE(EditorSceneImporterFBXConv,EditorSceneImporter );


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
