/*************************************************************************/
/*  scene_binds.h                                                        */
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
#ifndef SCENE_BINDS_H
#define SCENE_BINDS_H

#include "scene/io/scene_loader.h"
#include "scene/io/scene_saver.h"

#ifdef OLD_SCENE_FORMAT_ENABLED

class SceneIO : public Object {

	OBJ_TYPE( SceneIO, Object );
protected:

	static void _bind_methods();
public:

	enum SaveFlags {

		SAVE_FLAG_RELATIVE_PATHS=SceneSaver::FLAG_RELATIVE_PATHS,
		SAVE_FLAG_BUNDLE_RESOURCES=SceneSaver::FLAG_BUNDLE_RESOURCES,
		SAVE_FLAG_BUNDLE_INSTANCED_SCENES=SceneSaver::FLAG_BUNDLE_INSTANCED_SCENES,
		SAVE_FLAG_OMIT_EDITOR_PROPERTIES=SceneSaver::FLAG_OMIT_EDITOR_PROPERTIES,
		SAVE_FLAG_SAVE_BIG_ENDIAN=SceneSaver::FLAG_SAVE_BIG_ENDIAN
	};

	Node* load(const String& p_scene);
	Error save(const String& p_path, Node *p_scene,int p_flags=0,const Ref<OptimizedSaver> &p_optimizer=Ref<OptimizedSaver>());
	Ref<SceneInteractiveLoader> load_interactive(const String& p_scene);

	SceneIO() {}
};

#endif
#endif // SCENE_BINDS_H
