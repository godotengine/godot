/*************************************************************************/
/*  sample_editor_plugin.h                                               */
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
#ifndef SAMPLE_EDITOR_PLUGIN_H
#define SAMPLE_EDITOR_PLUGIN_H

#if 0
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/audio/sample_player.h"
#include "scene/resources/sample.h"
#include "scene/resources/sample_library.h"


class SampleEditor : public Panel {

	GDCLASS(SampleEditor, Panel );


	SamplePlayer *player;
	Label *info_label;
	Ref<ImageTexture> peakdisplay;
	Ref<Sample> sample;
	Ref<SampleLibrary> library;
	TextureRect *sample_texframe;
	Button *stop;
	Button *play;

	void _play_pressed();
	void _stop_pressed();
	void _update_sample();

protected:
	void _notification(int p_what);
	void _gui_input(InputEvent p_event);
	static void _bind_methods();
public:

	static void generate_preview_texture(const Ref<Sample>& p_sample,Ref<ImageTexture> &p_texture);
	void edit(Ref<Sample> p_sample);
	SampleEditor();
};


class SampleEditorPlugin : public EditorPlugin {

	GDCLASS( SampleEditorPlugin, EditorPlugin );

	SampleEditor *sample_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "Sample"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	SampleEditorPlugin(EditorNode *p_node);
	~SampleEditorPlugin();

};

#endif

#endif // SAMPLE_EDITOR_PLUGIN_H
