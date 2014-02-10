/*************************************************************************/
/*  scene_loader.cpp                                                     */
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
#include "scene_loader.h"
#include "globals.h"
#include "path_remap.h"

#ifdef OLD_SCENE_FORMAT_ENABLED

SceneFormatLoader *SceneLoader::loader[MAX_LOADERS];

int SceneLoader::loader_count=0;


void SceneInteractiveLoader::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_scene"),&SceneInteractiveLoader::get_scene);
	ObjectTypeDB::bind_method(_MD("poll"),&SceneInteractiveLoader::poll);
	ObjectTypeDB::bind_method(_MD("get_stage"),&SceneInteractiveLoader::get_stage);
	ObjectTypeDB::bind_method(_MD("get_stage_count"),&SceneInteractiveLoader::get_stage_count);
}

class SceneInteractiveLoaderDefault : public SceneInteractiveLoader {

	OBJ_TYPE( SceneInteractiveLoaderDefault, SceneInteractiveLoader );
public:
	Node *scene;

	virtual void set_local_path(const String& p_local_path) { scene->set_filename(p_local_path); }
	virtual Node *get_scene() { return scene; }
	virtual Error poll() { return ERR_FILE_EOF; }
	virtual int get_stage() const { return 1; }
	virtual int get_stage_count() const { return 1; }

	SceneInteractiveLoaderDefault() {}
};


Ref<SceneInteractiveLoader> SceneFormatLoader::load_interactive(const String &p_path,bool p_root_scene_hint) {

	Node *scene = load(p_path,p_root_scene_hint);
	if (!scene)
		return Ref<SceneInteractiveLoader>();
	Ref<SceneInteractiveLoaderDefault> sil = Ref<SceneInteractiveLoaderDefault>( memnew( SceneInteractiveLoaderDefault ));
	sil->scene=scene;
	return sil;
}



bool SceneFormatLoader::recognize(const String& p_extension) const {
	
	
	List<String> extensions;
	get_recognized_extensions(&extensions);
	for (List<String>::Element *E=extensions.front();E;E=E->next()) {
		
		if (E->get().nocasecmp_to(p_extension.extension())==0)
			return true;
	}
	
	return false;
}

Ref<SceneInteractiveLoader> SceneLoader::load_interactive(const String &p_path,bool p_save_root_state) {

	String local_path=Globals::get_singleton()->localize_path(p_path);

	String remapped_path = PathRemap::get_singleton()->get_remap(local_path);
	String extension=remapped_path.extension();

	for (int i=0;i<loader_count;i++) {

		if (!loader[i]->recognize(extension))
			continue;
		Ref<SceneInteractiveLoader> il = loader[i]->load_interactive(remapped_path,p_save_root_state);

		if (il.is_null() && remapped_path!=local_path)
			il = loader[i]->load_interactive(local_path,p_save_root_state);

		ERR_EXPLAIN("Error loading scene: "+local_path);
		ERR_FAIL_COND_V(il.is_null(),Ref<SceneInteractiveLoader>());
		il->set_local_path(local_path);

		return il;
	}

	ERR_EXPLAIN("No loader found for scene: "+p_path);
	ERR_FAIL_V(Ref<SceneInteractiveLoader>());
	return Ref<SceneInteractiveLoader>();
}

Node* SceneLoader::load(const String &p_path,bool p_root_scene_hint) {

	String local_path=Globals::get_singleton()->localize_path(p_path);

	String remapped_path = PathRemap::get_singleton()->get_remap(local_path);
	String extension=remapped_path.extension();

	for (int i=0;i<loader_count;i++) {
		
		if (!loader[i]->recognize(extension))
			continue;
		Node*node = loader[i]->load(remapped_path,p_root_scene_hint);

		if (!node && remapped_path!=local_path)
			node = loader[i]->load(local_path,p_root_scene_hint);

		ERR_EXPLAIN("Error loading scene: "+local_path);
		ERR_FAIL_COND_V(!node,NULL);
		node->set_filename(local_path);

		return node;
	}

	ERR_EXPLAIN("No loader found for scene: "+p_path);
	ERR_FAIL_V(NULL);
}

void SceneLoader::get_recognized_extensions(List<String> *p_extensions) {
	
	for (int i=0;i<loader_count;i++) {
		
		loader[i]->get_recognized_extensions(p_extensions);
	}
	
}

void SceneLoader::add_scene_format_loader(SceneFormatLoader *p_format_loader) {
	
	ERR_FAIL_COND( loader_count >= MAX_LOADERS );
	loader[loader_count++]=p_format_loader;
}


#endif
