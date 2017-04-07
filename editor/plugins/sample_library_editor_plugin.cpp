/*************************************************************************/
/*  sample_library_editor_plugin.cpp                                     */
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
#include "sample_library_editor_plugin.h"

#include "editor/editor_settings.h"
#include "global_config.h"
#include "io/resource_loader.h"
#include "sample_editor_plugin.h"
#include "scene/main/viewport.h"


void SampleLibraryEditor::_gui_input(InputEvent p_event) {


}

void SampleLibraryEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_PROCESS) {
		if (is_playing && !player->is_active()) {
			TreeItem *tl=last_sample_playing->cast_to<TreeItem>();
			tl->set_button(0,0,get_icon("Play","EditorIcons"));
			is_playing = false;
			set_process(false);
		}
	}

	if (p_what==NOTIFICATION_ENTER_TREE) {
		load->set_icon( get_icon("Folder","EditorIcons") );
		load->set_tooltip(TTR("Open Sample File(s)"));
	}

	if (p_what==NOTIFICATION_READY) {

		//NodePath("/root")->connect("node_removed", this,"_node_removed",Vector<Variant>(),true);
	}

	if (p_what==NOTIFICATION_DRAW) {

	}
}

void SampleLibraryEditor::_file_load_request(const PoolVector<String>& p_path) {


	for(int i=0;i<p_path.size();i++) {

		String path = p_path[i];
		Ref<Sample> sample = ResourceLoader::load(path,"Sample");
		if (sample.is_null()) {
			dialog->set_text(TTR("ERROR: Couldn't load sample!"));
			dialog->set_title(TTR("Error!"));
			//dialog->get_cancel()->set_text("Close");
			dialog->get_ok()->set_text(TTR("Close"));
			dialog->popup_centered_minsize();
			return; ///beh should show an error i guess
		}
		String basename = path.get_file().get_basename();
		String name=basename;
		int counter=0;
		while(sample_library->has_sample(name)) {
			counter++;
			name=basename+"_"+itos(counter);
		}

		undo_redo->create_action(TTR("Add Sample"));
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

	if (p_column==0) { // Play/Stop

		String btn_type;
		if(!is_playing) {
			is_playing = true;
			btn_type = TTR("Stop");
			player->play(name,true);
			last_sample_playing = p_item;
			set_process(true);
		} else {
			player->stop_all();
			if(last_sample_playing != p_item){
				TreeItem *tl=last_sample_playing->cast_to<TreeItem>();
				tl->set_button(p_column,0,get_icon("Play","EditorIcons"));
				btn_type = TTR("Stop");
				player->play(name,true);
				last_sample_playing = p_item;
			} else {
				btn_type = TTR("Play");
				is_playing = false;
			}
		}
		ti->set_button(p_column,0,get_icon(btn_type,"EditorIcons"));
	} else if (p_column==1) { // Edit

		get_tree()->get_root()->get_child(0)->call("_resource_selected",sample_library->get_sample(name));
	} else if (p_column==5) { // Delete

		ti->select(0);
		_delete_pressed();
	}


}





void SampleLibraryEditor::_item_edited() {

	if (!tree->get_selected())
		return;

	TreeItem *s = tree->get_selected();

	if (tree->get_selected_column()==0) { // Name
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
		undo_redo->create_action(TTR("Rename Sample"));
		undo_redo->add_do_method(sample_library.operator->(),"remove_sample",old_name);
		undo_redo->add_do_method(sample_library.operator->(),"add_sample",new_name,samp);
		undo_redo->add_undo_method(sample_library.operator->(),"remove_sample",new_name);
		undo_redo->add_undo_method(sample_library.operator->(),"add_sample",old_name,samp);
		undo_redo->add_do_method(this,"_update_library");
		undo_redo->add_undo_method(this,"_update_library");
		undo_redo->commit_action();

	} else if (tree->get_selected_column()==3) { // Volume dB

		StringName n = s->get_text(0);
		sample_library->sample_set_volume_db(n,s->get_range(3));

	} else if (tree->get_selected_column()==4) { // Pitch scale

		StringName n = s->get_text(0);
		sample_library->sample_set_pitch_scale(n,s->get_range(4));

	}


}

void SampleLibraryEditor::_delete_pressed() {

	if (!tree->get_selected())
		return;

	String to_remove = tree->get_selected()->get_text(0);
	undo_redo->create_action(TTR("Delete Sample"));
	undo_redo->add_do_method(sample_library.operator->(),"remove_sample",to_remove);
	undo_redo->add_undo_method(sample_library.operator->(),"add_sample",to_remove,sample_library->get_sample(to_remove));
	undo_redo->add_do_method(this,"_update_library");
	undo_redo->add_undo_method(this,"_update_library");
	undo_redo->commit_action();
}


void SampleLibraryEditor::_update_library() {

	player->stop_all();

	tree->clear();
	tree->set_hide_root(true);
	TreeItem *root = tree->create_item(NULL);

	List<StringName> names;
	sample_library->get_sample_list(&names);
	names.sort_custom<StringName::AlphCompare>();

	for(List<StringName>::Element *E=names.front();E;E=E->next()) {

		TreeItem *ti = tree->create_item(root);

		// Name + Play/Stop
		ti->set_cell_mode(0,TreeItem::CELL_MODE_STRING);
		ti->set_editable(0,true);
		ti->set_selectable(0,true);
		ti->set_text(0,E->get());
		ti->set_metadata(0,E->get());
		ti->add_button(0,get_icon("Play","EditorIcons"));

		Ref<Sample> smp = sample_library->get_sample(E->get());

		// Preview/edit
		Ref<ImageTexture> preview( memnew( ImageTexture ));
		preview->create(128,16,Image::FORMAT_RGB8);
		SampleEditor::generate_preview_texture(smp,preview);
		ti->set_cell_mode(1,TreeItem::CELL_MODE_ICON);
		ti->set_selectable(1,false);
		ti->set_editable(1,false);
		ti->set_icon(1,preview);
		ti->add_button(1,get_icon("Edit","EditorIcons"));

		// Format
		ti->set_cell_mode(2,TreeItem::CELL_MODE_STRING);
		ti->set_editable(2,false);
		ti->set_selectable(2,false);
		ti->set_text(2,String()+(smp->get_format()==Sample::FORMAT_PCM16?TTR("16 Bits")+", ":(smp->get_format()==Sample::FORMAT_PCM8?TTR("8 Bits")+", ":"IMA-ADPCM,"))+(smp->is_stereo()?TTR("Stereo"):TTR("Mono")));

		// Volume dB
		ti->set_cell_mode(3,TreeItem::CELL_MODE_RANGE);
		ti->set_range_config(3,-60,24,0.01);
		ti->set_selectable(3,true);
		ti->set_editable(3,true);
		ti->set_range(3,sample_library->sample_get_volume_db(E->get()));

		// Pitch scale
		ti->set_cell_mode(4,TreeItem::CELL_MODE_RANGE);
		ti->set_range_config(4,0.01,100,0.01);
		ti->set_selectable(4,true);
		ti->set_editable(4,true);
		ti->set_range(4,sample_library->sample_get_pitch_scale(E->get()));

		// Delete
		ti->set_cell_mode(5,TreeItem::CELL_MODE_STRING);
		ti->add_button(5,get_icon("Remove","EditorIcons"));

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
	}

}

Variant SampleLibraryEditor::get_drag_data_fw(const Point2& p_point,Control* p_from) {

	TreeItem*ti =tree->get_item_at_pos(p_point);
	if (!ti)
		return Variant();

	String name = ti->get_metadata(0);

	RES res = sample_library->get_sample(name);
	if (!res.is_valid())
		return Variant();

	return EditorNode::get_singleton()->drag_resource(res,p_from);


}

bool SampleLibraryEditor::can_drop_data_fw(const Point2& p_point,const Variant& p_data,Control* p_from) const {



	Dictionary d = p_data;

	if (!d.has("type"))
		return false;

	if (d.has("from") && (Object*)(d["from"])==tree)
		return false;

	if (String(d["type"])=="resource" && d.has("resource")) {
		RES r=d["resource"];

		Ref<Sample> sample = r;

		if (sample.is_valid()) {

			return true;
		}
	}


	if (String(d["type"])=="files") {

		Vector<String> files = d["files"];

		if (files.size()==0)
			return false;

		for(int i=0;i<files.size();i++) {
			String file = files[0];
			String ftype = EditorFileSystem::get_singleton()->get_file_type(file);

			if (ftype!="Sample") {
				return false;
			}

		}

		return true;

	}
	return false;
}

void SampleLibraryEditor::drop_data_fw(const Point2& p_point,const Variant& p_data,Control* p_from) {

	if (!can_drop_data_fw(p_point,p_data,p_from))
		return;

	Dictionary d = p_data;

	if (!d.has("type"))
		return;


	if (String(d["type"])=="resource" && d.has("resource")) {
		RES r=d["resource"];

		Ref<Sample> sample = r;

		if (sample.is_valid()) {

			String basename;
			if (sample->get_name()!="") {
				basename=sample->get_name();
			} else if (sample->get_path().is_resource_file()) {
				basename = sample->get_path().get_basename();
			} else {
				basename="Sample";
			}

			String name=basename;
			int counter=0;
			while(sample_library->has_sample(name)) {
				counter++;
				name=basename+"_"+itos(counter);
			}

			undo_redo->create_action(TTR("Add Sample"));
			undo_redo->add_do_method(sample_library.operator->(),"add_sample",name,sample);
			undo_redo->add_undo_method(sample_library.operator->(),"remove_sample",name);
			undo_redo->add_do_method(this,"_update_library");
			undo_redo->add_undo_method(this,"_update_library");
			undo_redo->commit_action();
		}
	}


	if (String(d["type"])=="files") {

		PoolVector<String> files = d["files"];

		_file_load_request(files);

	}

}


void SampleLibraryEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"),&SampleLibraryEditor::_gui_input);
	ClassDB::bind_method(D_METHOD("_load_pressed"),&SampleLibraryEditor::_load_pressed);
	ClassDB::bind_method(D_METHOD("_item_edited"),&SampleLibraryEditor::_item_edited);
	ClassDB::bind_method(D_METHOD("_delete_pressed"),&SampleLibraryEditor::_delete_pressed);
	ClassDB::bind_method(D_METHOD("_file_load_request"),&SampleLibraryEditor::_file_load_request);
	ClassDB::bind_method(D_METHOD("_update_library"),&SampleLibraryEditor::_update_library);
	ClassDB::bind_method(D_METHOD("_button_pressed"),&SampleLibraryEditor::_button_pressed);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &SampleLibraryEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &SampleLibraryEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &SampleLibraryEditor::drop_data_fw);

}

SampleLibraryEditor::SampleLibraryEditor() {

	player = memnew(SamplePlayer);
	add_child(player);
	add_style_override("panel", get_stylebox("panel","Panel"));


	load = memnew( Button );
	load->set_pos(Point2( 5, 5 ));
	load->set_size( Size2(1,1 ) );
	add_child(load);

	file = memnew( EditorFileDialog );
	add_child(file);
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Sample",&extensions);
	for(int i=0;i<extensions.size();i++)
		file->add_filter("*."+extensions[i]);
	file->set_mode(EditorFileDialog::MODE_OPEN_FILES);

	tree = memnew( Tree );
	tree->set_columns(6);
	add_child(tree);
	tree->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,5);
	tree->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,5);
	tree->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,30);
	tree->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,5);
	tree->set_column_titles_visible(true);
	tree->set_column_title(0,TTR("Name"));
	tree->set_column_title(1,TTR("Preview"));
	tree->set_column_title(2,TTR("Format"));
	tree->set_column_title(3,"dB");
	tree->set_column_title(4,TTR("Pitch"));
	tree->set_column_title(5,"");

	tree->set_column_min_width(1,150);
	tree->set_column_min_width(2,100);
	tree->set_column_min_width(3,50);
	tree->set_column_min_width(4,50);
	tree->set_column_min_width(5,32);
	tree->set_column_expand(1,false);
	tree->set_column_expand(2,false);
	tree->set_column_expand(3,false);
	tree->set_column_expand(4,false);
	tree->set_column_expand(5,false);

	tree->set_drag_forwarding(this);

	dialog = memnew( ConfirmationDialog );
	add_child( dialog );

	tree->connect("button_pressed",this,"_button_pressed");
	load->connect("pressed", this,"_load_pressed");
	file->connect("files_selected", this,"_file_load_request");
	tree->connect("item_edited", this,"_item_edited");

	is_playing = false;
}


void SampleLibraryEditorPlugin::edit(Object *p_object) {

	sample_library_editor->set_undo_redo(&get_undo_redo());
	SampleLibrary * s = p_object->cast_to<SampleLibrary>();
	if (!s)
		return;

	sample_library_editor->edit(Ref<SampleLibrary>(s));
}

bool SampleLibraryEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("SampleLibrary");
}

void SampleLibraryEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		//sample_library_editor->show();
		button->show();
		editor->make_bottom_panel_item_visible(sample_library_editor);
		//sample_library_editor->set_process(true);
	} else {

		if (sample_library_editor->is_visible_in_tree())
			editor->hide_bottom_panel();
		button->hide();

		//sample_library_editor->set_process(false);
	}

}

SampleLibraryEditorPlugin::SampleLibraryEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	sample_library_editor = memnew( SampleLibraryEditor );

	//editor->get_viewport()->add_child(sample_library_editor);
	sample_library_editor->set_custom_minimum_size(Size2(0,250));
	button=p_node->add_bottom_panel_item("SampleLibrary",sample_library_editor);
	button->hide();

	//sample_library_editor->set_area_as_parent_rect();
	//sample_library_editor->set_anchor( MARGIN_TOP, Control::ANCHOR_END);
	//sample_library_editor->set_margin( MARGIN_TOP, 120 );
	//sample_library_editor->hide();



}


SampleLibraryEditorPlugin::~SampleLibraryEditorPlugin()
{
}
#endif
