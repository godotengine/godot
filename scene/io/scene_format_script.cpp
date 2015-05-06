/*************************************************************************/
/*  scene_format_script.cpp                                              */
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
#include "scene_format_script.h"
#if 0
Node* SceneFormatLoaderScript::load(const String &p_path,bool p_save_instance_state) {

	Ref<Script> script = ResourceLoader::load(p_path);
	ERR_EXPLAIN("Can't load script-based scene: "+p_path);
	ERR_FAIL_COND_V(script.is_null(),NULL);
	ERR_EXPLAIN("Script does not instance in a node: "+p_path);
	ERR_FAIL_COND_V(script->get_node_type()=="",NULL);
	String node_type=script->get_node_type();
	Object *obj = ObjectTypeDB::instance(node_type);
	ERR_EXPLAIN("Unknown node type for instancing '"+node_type+"' in script: "+p_path);
	ERR_FAIL_COND_V(!obj,NULL);
	Node *node = obj->cast_to<Node>();
	if (!node)
		memdelete(obj);
	ERR_EXPLAIN("Node type '"+node_type+"' not of type 'Node'' in script: "+p_path);
	ERR_FAIL_COND_V(!node,NULL);

	node->set_script(script.get_ref_ptr());

	return node;
}

void SceneFormatLoaderScript::get_recognized_extensions(List<String> *p_extensions) const {

	for (int i=0;i<ScriptServer::get_language_count();i++) {

		ScriptServer::get_language(i)->get_recognized_extensions(p_extensions);
	}
}


SceneFormatLoaderScript::SceneFormatLoaderScript()
{
}
#endif
