/*************************************************************************/
/*  scene_loader.h                                                       */
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
#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include "scene/main/node.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#ifdef OLD_SCENE_FORMAT_ENABLED

class SceneInteractiveLoader : public Reference {

	OBJ_TYPE(SceneInteractiveLoader,Reference);
protected:

	static void _bind_methods();
public:

	virtual void set_local_path(const String& p_local_path)=0;
	virtual Node *get_scene()=0;
	virtual Error poll()=0;
	virtual int get_stage() const=0;
	virtual int get_stage_count() const=0;


	SceneInteractiveLoader() {}
};

class SceneFormatLoader {
public:
	
	virtual Ref<SceneInteractiveLoader> load_interactive(const String &p_path,bool p_root_scene_hint=false);
	virtual Node* load(const String &p_path,bool p_root_scene_hint=false)=0;
	virtual void get_recognized_extensions(List<String> *p_extensions) const=0;
	bool recognize(const String& p_extension) const;
		
	virtual ~SceneFormatLoader() {}
};

class SceneLoader {	
	
	enum {
		MAX_LOADERS=64
	};
	
	static SceneFormatLoader *loader[MAX_LOADERS];
	static int loader_count;
	
public:
	
	static Ref<SceneInteractiveLoader> load_interactive(const String &p_path,bool p_save_root_state=false);
	static Node* load(const String &p_path,bool p_save_root_state=false);
	static void add_scene_format_loader(SceneFormatLoader *p_format_loader);
	static void get_recognized_extensions(List<String> *p_extensions);
	
	
};

#endif

#endif
