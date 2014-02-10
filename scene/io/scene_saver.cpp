/*************************************************************************/
/*  scene_saver.cpp                                                      */
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
#include "scene_saver.h"
#include "print_string.h"

#ifdef OLD_SCENE_FORMAT_ENABLED
SceneFormatSaver *SceneSaver::saver[MAX_SAVERS];

int SceneSaver::saver_count=0;

bool SceneFormatSaver::recognize(const String& p_extension) const {
	
	
	List<String> extensions;
	get_recognized_extensions(&extensions);
	for (List<String>::Element *E=extensions.front();E;E=E->next()) {
		

		if (E->get().nocasecmp_to(p_extension.extension())==0)
			return true;
	}
	
	return false;
}

Error SceneSaver::save(const String &p_path,const Node* p_scene,uint32_t p_flags,const Ref<OptimizedSaver> &p_optimizer) {
	
	String extension=p_path.extension();
	Error err=ERR_FILE_UNRECOGNIZED;
	bool recognized=false;
	
	for (int i=0;i<saver_count;i++) {
		
		if (!saver[i]->recognize(extension))
			continue;
		recognized=true;
		err = saver[i]->save(p_path,p_scene,p_flags,p_optimizer);
		if (err == OK )
			return OK;
	}

	if (err) {
		if (!recognized) {
			ERR_EXPLAIN("No saver format found for scene: "+p_path);
		} else {
			ERR_EXPLAIN("Couldn't save scene: "+p_path);
		}
		ERR_FAIL_V(err);
	}
	
	return err;
}

void SceneSaver::get_recognized_extensions(List<String> *p_extensions) {
	
	for (int i=0;i<saver_count;i++) {
		
		saver[i]->get_recognized_extensions(p_extensions);
	}		
}

void SceneSaver::add_scene_format_saver(SceneFormatSaver *p_format_saver) {
	
	ERR_FAIL_COND( saver_count >= MAX_SAVERS );
	saver[saver_count++]=p_format_saver;
}

#endif
