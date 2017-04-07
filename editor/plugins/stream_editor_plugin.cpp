/*************************************************************************/
/*  stream_editor_plugin.cpp                                             */
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
#include "stream_editor_plugin.h"

#if 0

void StreamEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {
		play->set_icon( get_icon("Play","EditorIcons") );
		stop->set_icon( get_icon("Stop","EditorIcons") );
	}

}
void StreamEditor::_node_removed(Node *p_node) {

	if(p_node==node) {
		node=NULL;
		hide();
	}

}

void StreamEditor::_play() {

	node->call("play");
}

void StreamEditor::_stop() {

	node->call("stop");
}

void StreamEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_play"),&StreamEditor::_play);
	ClassDB::bind_method(D_METHOD("_stop"),&StreamEditor::_stop);

}

void StreamEditor::edit(Node *p_stream) {

	node=p_stream;

}
StreamEditor::StreamEditor() {

	play = memnew( Button );


	play->set_anchor_and_margin(MARGIN_LEFT,Control::ANCHOR_END,60);
	play->set_anchor_and_margin(MARGIN_RIGHT,Control::ANCHOR_END,40);
	play->set_anchor_and_margin(MARGIN_TOP,Control::ANCHOR_BEGIN,0);
	play->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_BEGIN,0);

	add_child(play);

	stop = memnew( Button );

	stop->set_pos(Point2( 35, 5 ));
	stop->set_anchor_and_margin(MARGIN_LEFT,Control::ANCHOR_END,30);
	stop->set_anchor_and_margin(MARGIN_RIGHT,Control::ANCHOR_END,10);
	stop->set_anchor_and_margin(MARGIN_TOP,Control::ANCHOR_BEGIN,0);
	stop->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_BEGIN,0);
	add_child(stop);


	play->connect("pressed", this,"_play");
	stop->connect("pressed", this,"_stop");

}


void StreamEditorPlugin::edit(Object *p_object) {

	stream_editor->edit(p_object->cast_to<Node>());
}

bool StreamEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("StreamPlayer") || p_object->is_class("SpatialStreamPlayer");
}

void StreamEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		stream_editor->show();
		stream_editor->set_fixed_process(true);
	} else {

		stream_editor->hide();
		stream_editor->set_fixed_process(false);
		stream_editor->edit(NULL);
	}

}

StreamEditorPlugin::StreamEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	stream_editor = memnew( StreamEditor );
	editor->get_viewport()->add_child(stream_editor);

	stream_editor->set_anchor(MARGIN_LEFT,Control::ANCHOR_END);
	stream_editor->set_anchor(MARGIN_RIGHT,Control::ANCHOR_END);
	stream_editor->set_margin(MARGIN_LEFT,60);
	stream_editor->set_margin(MARGIN_RIGHT,0);
	stream_editor->set_margin(MARGIN_TOP,0);
	stream_editor->set_margin(MARGIN_BOTTOM,10);


	stream_editor->hide();



}


StreamEditorPlugin::~StreamEditorPlugin()
{
}

#endif
