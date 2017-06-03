/*************************************************************************/
/*  sample_player_editor_plugin.h                                        */
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
#ifndef SAMPLE_PLAYER_EDITOR_PLUGIN_H
#define SAMPLE_PLAYER_EDITOR_PLUGIN_H

#if 0

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/spatial_sample_player.h"
#include "scene/audio/sample_player.h"
#include "scene/gui/option_button.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class SamplePlayerEditor : public Control {

	GDCLASS(SamplePlayerEditor, Control );

	Panel *panel;
	Button * play;
	Button * stop;
	OptionButton *samples;
	Node *node;


	void _update_sample_library();
	void _play();
	void _stop();

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();
public:

	void edit(Node *p_sample_player);
	SamplePlayerEditor();
};

class SamplePlayerEditorPlugin : public EditorPlugin {

	GDCLASS( SamplePlayerEditorPlugin, EditorPlugin );

	SamplePlayerEditor *sample_player_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "SamplePlayer"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	SamplePlayerEditorPlugin(EditorNode *p_node);
	~SamplePlayerEditorPlugin();

};

#endif
#endif // SAMPLE_PLAYER_EDITOR_PLUGIN_H
