/*************************************************************************/
/*  scene_saver.h                                                        */
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
#ifndef SCENE_SAVER_H
#define SCENE_SAVER_H

#include "scene/main/node.h"
#include "io/object_saver.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


#ifdef OLD_SCENE_FORMAT_ENABLED

class SceneFormatSaver {
public:
	
	virtual Error save(const String &p_path,const Node* p_scen,uint32_t p_flags=0,const Ref<OptimizedSaver> &p_optimizer=Ref<OptimizedSaver>())=0;
	virtual void get_recognized_extensions(List<String> *p_extensions) const=0;
	bool recognize(const String& p_extension) const;
	virtual ~SceneFormatSaver() {}
};




class SceneSaver {	
	
	enum {
		MAX_SAVERS=64
	};
	
	static SceneFormatSaver *saver[MAX_SAVERS];
	static int saver_count;
	
public:
	enum SaverFlags {

		FLAG_RELATIVE_PATHS=1,
		FLAG_BUNDLE_RESOURCES=2,
		FLAG_BUNDLE_INSTANCED_SCENES=4,
		FLAG_OMIT_EDITOR_PROPERTIES=8,
		FLAG_SAVE_BIG_ENDIAN=16
	};

	static Error save(const String &p_path,const Node* p_scenezz,uint32_t p_flags=0,const Ref<OptimizedSaver> &p_optimizer=Ref<OptimizedSaver>());
	static void add_scene_format_saver(SceneFormatSaver *p_format_saver);
	static void get_recognized_extensions(List<String> *p_extensions);
};



#endif
#endif
