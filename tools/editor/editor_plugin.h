/*************************************************************************/
/*  editor_plugin.h                                                      */
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
#ifndef EDITOR_PLUGIN_H
#define EDITOR_PLUGIN_H

#include "scene/main/node.h"
#include "scene/resources/texture.h"
#include "undo_redo.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class EditorNode;
class Spatial;
class Camera;

class EditorPlugin : public Node {
	
	OBJ_TYPE( EditorPlugin, Node );
friend class EditorData;
	UndoRedo *undo_redo;

	UndoRedo* _get_undo_redo() { return undo_redo; }

protected:

	static void _bind_methods();
	UndoRedo& get_undo_redo() { return *undo_redo; }

	void add_custom_type(const String& p_type, const String& p_base,const Ref<Script>& p_script, const Ref<Texture>& p_icon);
	void remove_custom_type(const String& p_type);


public:

	enum CustomControlContainer {
		CONTAINER_TOOLBAR,
		CONTAINER_SPATIAL_EDITOR_MENU,
		CONTAINER_SPATIAL_EDITOR_SIDE,
		CONTAINER_SPATIAL_EDITOR_BOTTOM,
		CONTAINER_CANVAS_EDITOR_MENU,
		CONTAINER_CANVAS_EDITOR_SIDE,
		CONTAINER_CANVAS_EDITOR_BOTTOM
	};

	//TODO: send a resoucre for editing to the editor node?

	void add_custom_control(CustomControlContainer p_location,Control *p_control);

	virtual bool create_spatial_gizmo(Spatial* p_spatial);
	virtual bool forward_input_event(const InputEvent& p_event);
	virtual bool forward_spatial_input_event(Camera* p_camera,const InputEvent& p_event);
	virtual String get_name() const;
	virtual bool has_main_screen() const;
	virtual void make_visible(bool p_visible);
	virtual void selected_notify() {}//notify that it was raised by the user, not the editor
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_node) const;
	virtual Dictionary get_state() const; //save editor state so it can't be reloaded when reloading scene
	virtual void set_state(const Dictionary& p_state)  ; //restore editor state (likely was saved with the scene)
	virtual void clear() ; // clear any temporary data in te editor, reset it (likely new scene or load another scene)
	virtual void save_external_data() ; // if editor references external resources/scenes, save them
	virtual void apply_changes() ; // if changes are pending in editor, apply them
	virtual void get_breakpoints(List<String> *p_breakpoints);
	virtual bool get_remove_list(List<Node*> *p_list);

	virtual void restore_global_state();
	virtual void save_global_state();

	EditorPlugin();
	virtual ~EditorPlugin();

};

VARIANT_ENUM_CAST( EditorPlugin::CustomControlContainer );


typedef EditorPlugin* (*EditorPluginCreateFunc)(EditorNode *);

class EditorPlugins {

	enum {
		MAX_CREATE_FUNCS=64
	};

	static EditorPluginCreateFunc creation_funcs[MAX_CREATE_FUNCS];
	static int creation_func_count;

	template<class T>
	static EditorPlugin *creator(EditorNode *p_node) {
		return memnew( T(p_node) );
	}

public:

	static int get_plugin_count() { return creation_func_count; }
	static EditorPlugin* create(int p_idx,EditorNode* p_editor)	 { ERR_FAIL_INDEX_V(p_idx,creation_func_count,NULL); return creation_funcs[p_idx](p_editor); }

	template<class T>
	static void add_by_type() {
		add_create_func(creator<T>);
	}

	static void add_create_func(EditorPluginCreateFunc p_func) {

		ERR_FAIL_COND(creation_func_count>=MAX_CREATE_FUNCS);
		creation_funcs[creation_func_count++]=p_func;
	}

};








#endif
