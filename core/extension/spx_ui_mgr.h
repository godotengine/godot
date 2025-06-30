/**************************************************************************/
/*  spx_ui_mgr.h                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef SPX_UI_MGR_H
#define SPX_UI_MGR_H

#include "gdextension_spx_ext.h"
#include "spx_engine.h"
#include "spx_ui.h"
class Node;
class SpxUiMgr : SpxBaseMgr {
	SPXCLASS(SpxUIMgr, SpxBaseMgr)

public:
	virtual ~SpxUiMgr() = default; // Added virtual destructor to fix -Werror=non-virtual-dtor

private:
	RBMap<GdObj, SpxUi *> id_objects;

	Control* create_control(GdString path);

public:
	void on_awake() override;
	void on_node_destroy(SpxUi *node);
	SpxUi* on_create_node(Control *control,GdInt type,bool is_attach = true);
	SpxUi *get_node(GdObj obj);

	static ESpxUiType get_node_type(Node* obj);
	void on_click(ISpxUi *node);
public:

	GdObj bind_node(GdObj obj, GdString rel_path);

	GdObj create_node(GdString path);
	GdObj create_button(GdString path,GdString text);
	GdObj create_label(GdString path, GdString text);
	GdObj create_image(GdString path);
	GdObj create_toggle(GdString path, GdBool value);
	GdObj create_slider(GdString path, GdFloat value);
	GdObj create_input(GdString path, GdString text);
	GdBool destroy_node(GdObj obj);

	GdInt get_type(GdObj obj);
	void set_text(GdObj obj, GdString text);
	GdString get_text(GdObj obj);
	void set_texture(GdObj obj, GdString path);
	GdString get_texture(GdObj obj);
	void set_color(GdObj obj, GdColor color);
	GdColor get_color(GdObj obj);
	void set_font_size(GdObj obj, GdInt size);
	GdInt get_font_size(GdObj obj);
	void set_visible(GdObj obj, GdBool visible);
	GdBool get_visible(GdObj obj);
	void set_interactable(GdObj obj, GdBool interactable);
	GdBool get_interactable(GdObj obj);
	void set_rect(GdObj obj, GdRect2 rect);
	GdRect2 get_rect(GdObj obj);


	GdInt get_layout_direction(GdObj obj);
	void set_layout_direction(GdObj obj,GdInt value);
	GdInt get_layout_mode(GdObj obj);
	void set_layout_mode(GdObj obj,GdInt value);
	GdInt get_anchors_preset(GdObj obj);
	void set_anchors_preset(GdObj obj,GdInt value);
	GdVec2 get_scale(GdObj obj);
	void set_scale(GdObj obj,GdVec2 value);
	GdVec2 get_position(GdObj obj);
	void set_position(GdObj obj,GdVec2 value);
	GdVec2 get_size(GdObj obj);
	void set_size(GdObj obj,GdVec2 value);
	GdVec2 get_global_position(GdObj obj);
	void set_global_position(GdObj obj,GdVec2 value);
	GdFloat get_rotation(GdObj obj);
	void set_rotation(GdObj obj,GdFloat value);

	GdBool get_flip(GdObj obj,GdBool horizontal);
	void set_flip(GdObj obj,GdBool horizontal, GdBool is_flip);
};

#endif // SPX_UI_MGR_H
