/*************************************************************************/
/*  texture_editor_plugin.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "resource_preview_dock.h"






void ResourcePreview::toggle_preview( bool hidden ) {
	int child_count = get_child_count();

	if ( !hidden ) {
		for (int i=1; i<child_count; i++) {
			Control *control = get_child(i)->cast_to<Control>();
			control->show();
		}
		collapse_button->set_normal_texture( get_icon("DockCollapse","EditorIcons") );
	}
	else {
		for (int i=1; i<child_count; i++) {
			Control *control = get_child(i)->cast_to<Control>();
			control->hide();
		}
		collapse_button->set_normal_texture( get_icon("DockExpand","EditorIcons") );
	}
	collapse_button->set_modulate( Color(0,0,0,0) );
}


void ResourcePreview::_notification(int p_what) {
	if (p_what==NOTIFICATION_ENTER_TREE) {
		collapse_button->set_normal_texture( get_icon("DockCollapse","EditorIcons") );
		collapse_button->connect( "toggled", this, "toggle_preview" );
		collapse_button->connect( "mouse_entered", collapse_button, "set_modulate", varray( Color(1,1,1,1) ) );
		collapse_button->connect( "mouse_exited", collapse_button, "set_modulate", varray( Color(0,0,0,0) ) );
	}
}

ResourcePreview::ResourcePreview() {
	collapse_button = memnew( TextureButton );
	collapse_button->set_margin( MARGIN_LEFT, 10);
	collapse_button->set_anchor( MARGIN_RIGHT, Control::ANCHOR_END, 10);
	collapse_button->set_margin( MARGIN_BOTTOM, 20);
	collapse_button->set_modulate( Color(0,0,0,0) );
	collapse_button->set_toggle_mode(true);

	add_child(collapse_button);
}


void ResourcePreview::_bind_methods() {

	ClassDB::bind_method("toggle_preview",&ResourcePreview::toggle_preview);

}

ResourcePreview::~ResourcePreview()
{
}
