/*************************************************************************/
/*  baked_light_editor_plugin.h                                          */
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
#ifndef BAKED_LIGHT_EDITOR_PLUGIN_H
#define BAKED_LIGHT_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/plugins/baked_light_baker.h"
#include "scene/gui/spin_box.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#if 0

class MeshInstance;

class BakedLightEditor : public Control {

	GDCLASS(BakedLightEditor, Control );


	float update_timeout;
	PoolVector<uint8_t> octree_texture;
	PoolVector<uint8_t> light_texture;
	PoolVector<int> octree_sampler;

	BakedLightBaker *baker;
	AcceptDialog *err_dialog;

	HBoxContainer *bake_hbox;
	Button *button_bake;
	Button *button_reset;
	Button *button_make_lightmaps;
	Label *bake_info;

	uint64_t last_rays_time;



	BakedLightInstance *node;

	enum Menu {

		MENU_OPTION_BAKE,
		MENU_OPTION_CLEAR
	};

	void _bake_lightmaps();

	void _bake_pressed();
	void _clear_pressed();

	void _end_baking();
	void _menu_option(int);

friend class BakedLightEditorPlugin;
protected:
	void _node_removed(Node *p_node);
	static void _bind_methods();
	void _notification(int p_what);
public:

	void edit(BakedLightInstance *p_baked_light);
	BakedLightEditor();
	~BakedLightEditor();
};

class BakedLightEditorPlugin : public EditorPlugin {

	GDCLASS( BakedLightEditorPlugin, EditorPlugin );

	BakedLightEditor *baked_light_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "BakedLight"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	BakedLightEditorPlugin(EditorNode *p_node);
	~BakedLightEditorPlugin();

};

#endif // MULTIMESH_EDITOR_PLUGIN_H
#endif
