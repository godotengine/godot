/*************************************************************************/
/*  editor_log.cpp                                                       */
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
#include "editor_log.h"

#include "editor_node.h"
#include "scene/gui/center_container.h"
#include "version.h"

void EditorLog::_error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, ErrorHandlerType p_type) {

	EditorLog *self = (EditorLog *)p_self;
	if (self->current != Thread::get_caller_ID())
		return;

	String err_str;
	if (p_errorexp && p_errorexp[0]) {
		err_str = p_errorexp;
	} else {
		err_str = String(p_file) + ":" + itos(p_line) + " - " + String(p_error);
	}

	/*
	if (!self->is_visible_in_tree())
		self->emit_signal("show_request");
	*/

	err_str = " " + err_str;
	self->log->add_newline();

	Ref<Texture> icon;

	switch (p_type) {
		case ERR_HANDLER_ERROR: {

			icon = self->get_icon("Error", "EditorIcons");
			return; // these are confusing
		} break;
		case ERR_HANDLER_WARNING: {

			icon = self->get_icon("Error", "EditorIcons");

		} break;
		case ERR_HANDLER_SCRIPT: {

			icon = self->get_icon("ScriptError", "EditorIcons");
		} break;
		case ERR_HANDLER_SHADER: {

			icon = self->get_icon("Shader", "EditorIcons");
		} break;
	}

	self->add_message(err_str, true);
}

void EditorLog::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		log->add_color_override("default_color", get_color("font_color", "Tree"));
		//button->set_icon(get_icon("Console","EditorIcons"));
	}

	/*if (p_what==NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();
		get_stylebox("panel","PopupMenu")->draw(ci,Rect2(Point2(),get_size()));
		int top_ofs = 20;
		int border_ofs=4;
		Ref<StyleBox> style = get_stylebox("normal","TextEdit");

		style->draw(ci,Rect2( Point2(border_ofs,top_ofs),get_size()-Size2(border_ofs*2,top_ofs+border_ofs)));
	}*/
}

void EditorLog::_clear_request() {

	log->clear();
}

void EditorLog::clear() {
	_clear_request();
}

void EditorLog::add_message(const String &p_msg, bool p_error) {

	if (p_error) {
		Ref<Texture> icon = get_icon("Error", "EditorIcons");
		log->add_image(icon);
		//button->set_icon(icon);
		log->push_color(get_color("fg_error", "Editor"));
	} else {
		//button->set_icon(Ref<Texture>());
	}

	log->add_newline();
	log->add_text(p_msg);
	//button->set_text(p_msg);

	if (p_error)
		log->pop();
}

/*
void EditorLog::_dragged(const Point2& p_ofs) {

	int ofs = ec->get_minsize().height;
	ofs = ofs-p_ofs.y;
	if (ofs<50)
		ofs=50;
	if (ofs>300)
		ofs=300;
	ec->set_minsize(Size2(ec->get_minsize().width,ofs));
	minimum_size_changed();

}
*/

void EditorLog::_undo_redo_cbk(void *p_self, const String &p_name) {

	EditorLog *self = (EditorLog *)p_self;
	self->add_message(p_name);
}

void EditorLog::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_clear_request"), &EditorLog::_clear_request);
	ClassDB::bind_method("_override_logger_styles", &EditorLog::_override_logger_styles);
	//ClassDB::bind_method(D_METHOD("_dragged"),&EditorLog::_dragged );
	ADD_SIGNAL(MethodInfo("clear_request"));
}

EditorLog::EditorLog() {

	VBoxContainer *vb = this;
	add_constant_override("separation", get_constant("separation", "VBoxContainer"));

	HBoxContainer *hb = memnew(HBoxContainer);
	vb->add_child(hb);
	title = memnew(Label);
	title->set_text(TTR(" Output:"));
	title->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(title);

	//pd = memnew( PaneDrag );
	//hb->add_child(pd);
	//pd->connect("dragged",this,"_dragged");
	//pd->set_default_cursor_shape(Control::CURSOR_MOVE);

	clearbutton = memnew(Button);
	hb->add_child(clearbutton);
	clearbutton->set_text(TTR("Clear"));
	clearbutton->connect("pressed", this, "_clear_request");

	ec = memnew(Control);
	vb->add_child(ec);
	ec->set_custom_minimum_size(Size2(0, 180));
	ec->set_v_size_flags(SIZE_EXPAND_FILL);

	pc = memnew(PanelContainer);
	ec->add_child(pc);
	pc->set_area_as_parent_rect();
	pc->connect("tree_entered", this, "_override_logger_styles");

	log = memnew(RichTextLabel);
	log->set_scroll_follow(true);
	log->set_selection_enabled(true);
	log->set_focus_mode(FOCUS_CLICK);
	pc->add_child(log);
	add_message(VERSION_FULL_NAME " (c) 2008-2017 Juan Linietsky, Ariel Manzur.");
	//log->add_text("Initialization Complete.\n"); //because it looks cool.

	eh.errfunc = _error_handler;
	eh.userdata = this;
	add_error_handler(&eh);

	current = Thread::get_caller_ID();

	EditorNode::get_undo_redo()->set_commit_notify_callback(_undo_redo_cbk, this);
}

void EditorLog::deinit() {

	remove_error_handler(&eh);
}

void EditorLog::_override_logger_styles() {

	pc->add_style_override("panel", get_stylebox("normal", "TextEdit"));
}

EditorLog::~EditorLog() {
}
