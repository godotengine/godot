/*************************************************************************/
/*  sample_player_editor_plugin.cpp                                      */
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

#if 0
#include "sample_player_editor_plugin.h"

#include "scene/resources/sample_library.h"


void SamplePlayerEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {
		play->set_icon( get_icon("Play","EditorIcons") );
		stop->set_icon( get_icon("Stop","EditorIcons") );
	}

}

void SamplePlayerEditor::_node_removed(Node *p_node) {

	if(p_node==node) {
		node=NULL;
		hide();
	}

}

void SamplePlayerEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_play"),&SamplePlayerEditor::_play);
	ClassDB::bind_method(D_METHOD("_stop"),&SamplePlayerEditor::_stop);

}


void SamplePlayerEditor::_play() {

	if (!node)
		return;
	if (samples->get_item_count()<=0)
		return;

	node->call("play",samples->get_item_text( samples->get_selected() ));
	stop->set_pressed(false);
	play->set_pressed(true);
}

void SamplePlayerEditor::_stop() {

	if (!node)
		return;
	if (samples->get_item_count()<=0)
		return;

	node->call("stop_all");
	print_line("STOP ALL!!");
	stop->set_pressed(true);
	play->set_pressed(false);

}


void SamplePlayerEditor::_update_sample_library() {

	samples->clear();
	Ref<SampleLibrary> sl = node->call("get_sample_library");
	if (sl.is_null()) {
		samples->add_item("<NO SAMPLE LIBRARY>");
		return; //no sample library;
	}

	List<StringName> samplenames;
	sl->get_sample_list(&samplenames);
	samplenames.sort_custom<StringName::AlphCompare>();
	for(List<StringName>::Element *E=samplenames.front();E;E=E->next()) {
		samples->add_item(E->get());
	}

}

void SamplePlayerEditor::edit(Node *p_sample_player) {

	node=p_sample_player;
	if (node) {
		_update_sample_library();
	}

}
SamplePlayerEditor::SamplePlayerEditor() {


	play = memnew( Button );

	play->set_pos(Point2( 5, 5 ));
	play->set_toggle_mode(true);
	play->set_anchor_and_margin(MARGIN_LEFT,Control::ANCHOR_END,250);
	play->set_anchor_and_margin(MARGIN_RIGHT,Control::ANCHOR_END,230);
	play->set_anchor_and_margin(MARGIN_TOP,Control::ANCHOR_BEGIN,0);
	play->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_BEGIN,0);

	add_child(play);

	stop = memnew( Button );

	stop->set_pos(Point2( 35, 5 ));
	stop->set_toggle_mode(true);
	stop->set_anchor_and_margin(MARGIN_LEFT,Control::ANCHOR_END,220);
	stop->set_anchor_and_margin(MARGIN_RIGHT,Control::ANCHOR_END,200);
	stop->set_anchor_and_margin(MARGIN_TOP,Control::ANCHOR_BEGIN,0);
	stop->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_BEGIN,0);
	add_child(stop);

	samples = memnew( OptionButton );
	samples->set_anchor_and_margin(MARGIN_LEFT,Control::ANCHOR_END,190);
	samples->set_anchor_and_margin(MARGIN_RIGHT,Control::ANCHOR_END,5);
	samples->set_anchor_and_margin(MARGIN_TOP,Control::ANCHOR_BEGIN,0);
	samples->set_anchor_and_margin(MARGIN_BOTTOM,Control::ANCHOR_BEGIN,0);
	add_child(samples);

	play->connect("pressed", this,"_play");
	stop->connect("pressed", this,"_stop");

}


void SamplePlayerEditorPlugin::edit(Object *p_object) {

	sample_player_editor->edit(p_object->cast_to<Node>());
}

bool SamplePlayerEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("SamplePlayer2D") || p_object->is_class("SamplePlayer") || p_object->is_class("SpatialSamplePlayer");
}

void SamplePlayerEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		sample_player_editor->show();
		sample_player_editor->set_fixed_process(true);
	} else {

		sample_player_editor->hide();
		sample_player_editor->set_fixed_process(false);
		sample_player_editor->edit(NULL);
	}

}

SamplePlayerEditorPlugin::SamplePlayerEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	sample_player_editor = memnew( SamplePlayerEditor );
	editor->get_viewport()->add_child(sample_player_editor);

	sample_player_editor->set_anchor(MARGIN_LEFT,Control::ANCHOR_END);
	sample_player_editor->set_anchor(MARGIN_RIGHT,Control::ANCHOR_END);
	sample_player_editor->set_margin(MARGIN_LEFT,250);
	sample_player_editor->set_margin(MARGIN_RIGHT,0);
	sample_player_editor->set_margin(MARGIN_TOP,0);
	sample_player_editor->set_margin(MARGIN_BOTTOM,10);


	sample_player_editor->hide();



}


SamplePlayerEditorPlugin::~SamplePlayerEditorPlugin()
{
}

#endif
