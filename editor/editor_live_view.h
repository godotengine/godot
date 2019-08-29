/*************************************************************************/
/*  editor_live_view.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef EDITOR_LIVE_VIEW_H
#define EDITOR_LIVE_VIEW_H

#include "scene/gui/box_container.h"

class Label;
class SharedMemory;
class TabContainer;
class TextureRect;

class EditorLiveView : public VBoxContainer {

	GDCLASS(EditorLiveView, VBoxContainer);

	TabContainer *tab_container;

	TextureRect *texture_rect;
	Label *label;

	bool stop_while_unfocused;

	bool visible;
	bool editor_focused;
	bool editor_recovered_focus;

	bool running;
	bool can_request;
	SharedMemory *fb_data;
	Ref<Image> fb_img;
	uint32_t min_frame_interval;
	uint32_t last_frame_time;

	void _update_processing_state();

protected:
	void _notification(int p_what);

public:
	void start();
	void stop();
	void refresh();

	explicit EditorLiveView();
	~EditorLiveView();
};

class LiveViewDebugHelper {

	SharedMemory *fb_data;

public:
	void handle_request_framebuffer(int p_target_w, int p_target_h);

	explicit LiveViewDebugHelper();
	~LiveViewDebugHelper();
};

#endif // EDITOR_LIVE_VIEW_H
