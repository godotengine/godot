/*************************************************************************/
/*  editor_playback_plugin.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef EDITOR_PREVIEW_PLUGIN_H
#define EDITOR_PREVIEW_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/immediate_geometry.h"
#include "scene/3d/light.h"
#include "scene/3d/visual_instance.h"
#include "scene/gui/panel_container.h"

class Camera;

class EditorPlaybackViewport : public Control {

	GDCLASS(EditorPlaybackViewport, Control);
	friend class EditorPlayback;

private:
	String name;
	Size2 prev_size;

	EditorNode *editor;
	EditorData *editor_data;

	ViewportContainer *viewport_container;

	Control *surface;
	Viewport *viewport;
	Camera *camera;
	Camera *camera_pointer;

	String last_message;
	String message;
	float message_time;

	void set_message(String p_message, float p_time = 5);

	//
	void _draw();

	void _smouseenter();
	void _smouseexit();
	void _sinput(const Ref<InputEvent> &p_event);
	EditorPlayback *editor_playback;

protected:
	void _current_camera_changed(Object *p_camera);
	void _update_camera_state();
	void _notification(int p_what);
	static void _bind_methods();

public:
	void reset();

	void focus_selection();

	Viewport *get_viewport_node() { return viewport; }

	EditorPlaybackViewport(EditorPlayback *p_editor_playback, EditorNode *p_editor);
	~EditorPlaybackViewport();
};

class EditorPlaybackViewportContainer : public Container {

	GDCLASS(EditorPlaybackViewportContainer, Container)
private:
	bool mouseover;
	float ratio_h;
	float ratio_v;

	void _gui_input(const Ref<InputEvent> &p_event);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	EditorPlaybackViewportContainer();
};

class EditorPlayback : public VBoxContainer {

	GDCLASS(EditorPlayback, VBoxContainer);

private:
	static const unsigned int VIEWPORTS_COUNT = 1;

	EditorNode *editor;
	EditorSelection *editor_selection;

	EditorPlaybackViewportContainer *viewport_base;
	EditorPlaybackViewport *viewport;

	//
	//

	HBoxContainer *hbc_menu;

	void _toggle_maximize_view(Object *p_viewport);

	static EditorPlayback *singleton;

	void _node_removed(Node *p_node);
	EditorPlayback();

protected:
	void _notification(int p_what);
	//void _gui_input(InputEvent p_event);
	void _unhandled_key_input(Ref<InputEvent> p_event);

	static void _bind_methods();

public:
	static EditorPlayback *get_singleton() { return singleton; }

	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	EditorPlaybackViewport *get_editor_viewport() {
		return viewport;
	}

	Camera *get_camera() { return NULL; }
	void clear();

	EditorPlayback(EditorNode *p_editor);
	~EditorPlayback();
};

class EditorPlaybackPlugin : public EditorPlugin {

	GDCLASS(EditorPlaybackPlugin, EditorPlugin);

	EditorPlayback *editor_playback;
	EditorNode *editor;

protected:
	static void _bind_methods();

public:
	EditorPlayback *get_editor_playback() { return editor_playback; }
	virtual String get_name() const { return "Game"; }
	bool has_main_screen() const { return true; }
	virtual void make_visible(bool p_visible);
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;

	virtual Dictionary get_state() const;
	virtual void set_state(const Dictionary &p_state);
	virtual void clear() { editor_playback->clear(); }

	EditorPlaybackPlugin(EditorNode *p_node);
	~EditorPlaybackPlugin();
};

#endif
