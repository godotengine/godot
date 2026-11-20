/**************************************************************************/
/*  shared_controls.cpp                                                   */
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

#include "shared_controls.h"

#include "editor/editor_node.h"
#include "scene/gui/label.h"
#include "scene/resources/style_box_flat.h"

SpanningHeader::SpanningHeader(const String &p_text) {
	Ref<StyleBoxFlat> title_sbf;
	title_sbf.instantiate();
	title_sbf->set_bg_color(EditorNode::get_singleton()->get_editor_theme()->get_color("dark_color_3", "Editor"));
	add_theme_style_override(SceneStringName(panel), title_sbf);
	set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	Label *title = memnew(Label(p_text));
	add_child(title);
	title->set_horizontal_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
	title->set_vertical_alignment(VerticalAlignment::VERTICAL_ALIGNMENT_CENTER);
}

DarkPanelContainer::DarkPanelContainer() {
	set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	set_v_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	Ref<StyleBoxFlat> content_wrapper_sbf;
	content_wrapper_sbf.instantiate();
	content_wrapper_sbf->set_bg_color(EditorNode::get_singleton()->get_editor_theme()->get_color("dark_color_2", "Editor"));
	add_theme_style_override(SceneStringName(panel), content_wrapper_sbf);
}
