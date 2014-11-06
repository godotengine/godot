/*************************************************************************/
/*  sample_library_editor_plugin.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "sample_library_editor_plugin.h"

#include "io/resource_loader.h"
#include "globals.h"
#include "tools/editor/editor_settings.h"
#include "scene/main/viewport.h"
#include "sample_editor_plugin.h"



void SampleLibraryEditor::_input_event(InputEvent p_event) {


}

void SampleLibraryEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_FIXED_PROCESS) {

	}

	if (p_what==NOTIFICATION_ENTER_TREE) {
		play->set_icon( get_icon("Play","EditorIcons") );
		stop->set_icon( get_icon("Stop","EditorIcons") );
		load->set_icon( get_icon("Folder","EditorIcons") );
		_delete->set_icon( get_icon("Del","EditorIcons") );
	}

	if (p_what==NOTIFICATION_READY) {

//		NodePath("/root")->connect("node_removed", this,"_node_removed",Vector<Variant>(),true);
	}

	if (p_what==NOTIFICATION_DRAW) {

	}
}

void SampleLibraryEditor::_play_pressed() {

	if (!tree->get_selected())
		return;

	String to_play = tree->get_selected()->get_text(0);

	player->play(to_play,true);
	play->set_pressed(false);
	stop->set_pressed(false);
}
void SampleLibraryEditor::_stop_pressed() {

	player->stop_all();
	play->set_pressed(false);
}

void SampleLibraryEditor::_file_load_request(const DVector<String>& p_path) {


	for(int i=0;i<p_path.size();i++) {

		String path = p_path[i];
		Ref<Sample> sample = ResourceLoader::load(path,"Sample");
		if (sample.is_null()) {
			dialog->set_text("ERROR: Couldn't load sample!");
			dialog->set_title("Error!");
			//dialog->get_cancel()->set_text("Close");
			dialog->get_ok()->set_text("Close");
			dialog->popup_centered(Size2(300,60));
			return; ///beh should show an error i guess
		}
		String basename = path.get_file().basename();
		String name=basename;
		int counter=0;
		while(sample_library->has_sample(name)) {
			counter++;
			name=basename+"_"+itos(counter);
		}

		undo_redo->create_action("Add Sample");
		undo_redo->add_do_method(sample_library.operator->(),"add_sample",name,sample);
		undo_redo->add_undo_method(sample_library.operator->(),"remove_sample",name);
		undo_redo->add_do_method(this,"_update_library");
		undo_redo->add_undo_method(this,"_update_library");
		undo_redo->commit_action();
	}
}

void SampleLibraryEditor::_load_pressed() {

	file->popup_centered_ratio();

}

void SampleLibraryEditor::_button_pressed(Object *p_item,int p_column, int p_id) {

	TreeItem *ti=p_item->cast_to<TreeItem>();
	String name = ti->get_text(0);

	if (p_column==0) {

		player->play(name,true);
	} else if (p_column==1) {

		get_tree()->get_root()->get_child(0)->call("_resource_selected",sample_library->get_sample(name));

	}


}





void SampleLibraryEditor::_item_edited() {

	if (!tree->get_selected())
		return;

	TreeItem *s = tree->get_selected();

	if (tree->get_selected_column()==0) {
		// renamed
		String old_name=s->get_metadata(0);
		String new_name=s->get_text(0);
		if (old_name==new_name)
			return;

		if (new_name=="" || new_name.find("\\")!=-1 || new_name.find("/")!=-1 || sample_library->has_sample(new_name)) {

			s->set_text(0,old_name);
			return;
		}

		Ref<Sample> samp = sample_library->get_sample(old_name);
		undo_redo->create_action("Rename Sample");
		undo_redo->add_do_method(sample_library.operator->(),"remove_sample",old_name);
		undo_redo->add_do_method(sample_library.operator->(),"add_sample",new_name,samp);
		undo_redo->add_undo_method(sample_library.operator->(),"remove_sample",new_name);
		undo_redo->add_undo_method(sample_library.operator->(),"add_sample",old_name,samp);
		undo_redo->add_do_method(this,"_update_library");
		undo_redo->add_undo_method(this,"_update_library");
		undo_redo->commit_action();

	} else if (tree->get_selected_column()==3) {

		StringName n = s->get_text(0);
		sample_library->sample_set_volume_db(n,s->get_range(3));

	} else if (tree->get_selected_column()==4) {

		StringName n = s->get_text(0);
		sample_library->sample_set_pitch_scale(n,s->get_range(4));

	} else if (tree->get_selected_column()==5) {

		//edit

		Ref<Sample> samp = sample_library->get_sample(tree->get_selected()->get_metadata(0));

		get_tree()->get_root()->get_child(0)->call("_resource_selected",samp);
	}


}

void SampleLibraryEditor::_delete_confirm_pressed() {

	if (!tree->get_selected())
		return;

	String to_remove = tree->get_selected()->get_text(0);
	undo_redo->create_action("Delete Sample");
	undo_redo->add_do_method(sample_library.operator->(),"remove_sample",to_remove);
	undo_redo->add_undo_method(sample_library.operator->(),"add_sample",to_remove,sample_library->get_sample(to_remove));
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();
}


void SampleLibraryEditor::_delete_pressed() {


	if (!tree->get_selected())
		return;

	_delete_confirm_pressed(); //it has undo.. why bother with a dialog..
	/*
	dialog->set_title("Confirm...");
	dialog->set_text("Remove Sample '"+tree->get_selected()->get_text(0)+"' ?");
	//dialog->get_cancel()->set_text("Cancel");
	//dialog->get_ok()->show();
	dialog->get_ok()->set_text("Remove");
	dialog->popup_centered(Size2(300,60));*/

}


void SampleLibraryEditor::_update_library() {

	player->stop_all();

	tree->clear();
	tree->set_hide_root(true);
	TreeItem *root = tree->create_item(NULL);

	List<StringName> names;
	sample_library->get_sample_list(&names);

	for(List<StringName>::Element *E=names.front();E;E=E->next()) {

		TreeItem *ti = tree->create_item(root);
		ti->set_cell_mode(0,TreeItem::CELL_MODE_STRING);
		ti->set_editable(0,true);
		ti->set_selectable(0,true);
		ti->set_text(0,E->get());
		ti->set_metadata(0,E->get());
		ti->add_button(0,get_icon("Play","EditorIcons"),0);
		ti->add_button(1,get_icon("Edit","EditorIcons"),1);

		Ref<Sample> smp = sample_library->get_sample(E->get());

		Ref<ImageTexture> preview( memnew( ImageTexture ));
		preview->create(128,16,Image::FORMAT_RGB);
		SampleEditor::generate_preview_texture(smp,preview);
		ti->set_cell_mode(1,TreeItem::CELL_MODE_ICON);
		ti->set_selectable(1,false);
		ti->set_editable(1,false);
		ti->set_icon(1,preview);


		ti->set_cell_mode(2,TreeItem::CELL_MODE_STRING);
		ti->set_editable(2,false);
		ti->set_selectable(2,false);
		Ref<Sample> s = sample_library->get_sample(E->get());
		ti->set_text(2,String()+/*itos(s->get_length())+" frames ("+String::num(s->get_length()/(float)s->get_mix_rate(),2)+" s), "+*/(s->get_format()==Sample::FORMAT_PCM16?"16 Bits, ":(s->get_format()==Sample::FORMAT_PCM8?"8 bits, ":"IMA-ADPCM,"))+(s->is_stereo()?"Stereo":"Mono"));

		ti->set_cell_mode(3,TreeItem::CELL_MODE_RANGE);
		ti->set_range_config(3,-60,24,0.01);
		ti->set_selectable(3,true);
		ti->set_editable(3,true);
		ti->set_range(3,sample_library->sample_get_volume_db(E->get()));

		ti->set_cell_mode(4,TreeItem::CELL_MODE_RANGE);
		ti->set_range_config(4,0.01,100,0.01);
		ti->set_selectable(4,true);
		ti->set_editable(4,true);
		ti->set_range(4,sample_library->sample_get_pitch_scale(E->get()));

		//ti->set_cell_mode(5,TreeItem::CELL_MODE_CUSTOM);
		//ti->set_text(5,"Edit..");
		//ti->set_selectable(5,true);
		//ti->set_editable(5,true);
	}

	//player->add_sample("default",sample);
}



void SampleLibraryEditor::edit(Ref<SampleLibrary> p_sample_library) {

	sample_library=p_sample_library;


	if (!sample_library.is_null()) {
		player->set_sample_library(sample_library);
		_update_library();
	} else {

		hide();
		set_fixed_process(false);
	}

}



void SampleLibraryEditor::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_input_event"),&SampleLibraryEditor::_input_event);
	ObjectTypeDB::bind_method(_MD("_play_pressed"),&SampleLibraryEditor::_play_pressed);
	ObjectTypeDB::bind_method(_MD("_stop_pressed"),&SampleLibraryEditor::_stop_pressed);
	ObjectTypeDB::bind_method(_MD("_load_pressed"),&SampleLibraryEditor::_load_pressed);
	ObjectTypeDB::bind_method(_MD("_item_edited"),&SampleLibraryEditor::_item_edited);
	ObjectTypeDB::bind_method(_MD("_delete_pressed"),&SampleLibraryEditor::_delete_pressed);
	ObjectTypeDB::bind_method(_MD("_delete_confirm_pressed"),&SampleLibraryEditor::_delete_confirm_pressed);
	ObjectTypeDB::bind_method(_MD("_file_load_request"),&SampleLibraryEditor::_file_load_request);
	ObjectTypeDB::bind_method(_MD("_update_library"),&SampleLibraryEditor::_update_library);
	ObjectTypeDB::bind_method(_MD("_button_pressed"),&SampleLibraryEditor::_button_pressed);
}

SampleLibraryEditor::SampleLibraryEditor() {

	player = memnew(SamplePlayer);
	add_child(player);
	add_style_override("panel", get_stylebox("panel","Panel"));


	play = memnew( Button );

	play->set_pos(Point2( 5, 5 ));
	play->set_size( Size2(1,1 ) );
	play->set_toggle_mode(true);
	//add_child(play);

	stop = memnew( Button );

	stop->set_pos(Point2( 5, 5 ));
	stop->set_size( Size2(1,1 ) );
	//stop->set_toggle_mode(true);
	add_child(stop);

	load = memnew( Button );

	load->set_pos(Point2( 35, 5 ));
	load->set_size( Size2(1,1 ) );
	add_child(load);

	_delete = memnew( Button );

	file = memnew( FileDialog );
	add_child(file);
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Sample",&extensions);
	for(int i=0;i<extensions.size();i++)
		file->add_filter("*."+extensions[i]);
	file->set_mode(FileDialog::MODE_OPEN_FILES);

	_delete->set_pos(Point2( 65, 5 ));
	_delete->set_size( Size2(1,1 ) );
	add_child(_delete);

	tree = memnew( Tree );
	tree->set_columns(5);
	add_child(tree);
	tree->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,5);
	tree->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,5);
	tree->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,30);
	tree->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,5);
	tree->set_column_titles_visible(true);
	tree->set_column_title(0,"Name");

	tree->set_column_title(1,"Preview");
	tree->set_column_title(2,"Format");
	tree->set_column_title(3,"dB");
	tree->set_column_title(4,"PScale");
	tree->set_column_min_width(1,150);
	tree->set_column_min_width(2,100);
	tree->set_column_min_width(3,50);
	tree->set_column_min_width(4,50);
	tree->set_column_expand(1,false);
	tree->set_column_expand(2,false);
	tree->set_column_expand(3,false);
	tree->set_column_expand(4,false);

	dialog = memnew( ConfirmationDialog );
	add_child( dialog );

	tree->connect("button_pressed",this,"_button_pressed");
	play->connect("pressed", this,"_play_pressed");
	stop->connect("pressed", this,"_stop_pressed");
	load->connect("pressed", this,"_load_pressed");
	_delete->connect("pressed", this,"_delete_pressed");
	file->connect("files_selected", this,"_file_load_request");
	//dialog->connect("confirmed", this,"_delete_confirm_pressed");
	tree->connect("item_edited", this,"_item_edited");


}


void SampleLibraryEditorPlugin::edit(Object *p_object) {

	sample_library_editor->set_undo_redo(&get_undo_redo());
	SampleLibrary * s = p_object->cast_to<SampleLibrary>();
	if (!s)
		return;

	sample_library_editor->edit(Ref<SampleLibrary>(s));
}

bool SampleLibraryEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("SampleLibrary");
}

void SampleLibraryEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		sample_library_editor->show();
//		sample_library_editor->set_process(true);
	} else {

		sample_library_editor->hide();
//		sample_library_editor->set_process(false);
	}

}

SampleLibraryEditorPlugin::SampleLibraryEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	sample_library_editor = memnew( SampleLibraryEditor );
	editor->get_viewport()->add_child(sample_library_editor);
	sample_library_editor->set_area_as_parent_rect();
//	sample_library_editor->set_anchor( MARGIN_TOP, Control::ANCHOR_END);
//	sample_library_editor->set_margin( MARGIN_TOP, 120 );
	sample_library_editor->hide();



}


SampleLibraryEditorPlugin::~SampleLibraryEditorPlugin()
{
}


