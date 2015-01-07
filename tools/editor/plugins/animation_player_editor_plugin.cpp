/*************************************************************************/
/*  animation_player_editor_plugin.cpp                                   */
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
#include "animation_player_editor_plugin.h"
#include "io/resource_loader.h"


void AnimationPlayerEditor::_node_removed(Node *p_node) {

	if (player && player == p_node) {
		player=NULL;
		hide();
		set_process(false);
		if (edit_anim->is_pressed()) {

			editor->get_animation_editor()->set_animation(Ref<Animation>());
			editor->get_animation_editor()->set_root(NULL);
			editor->animation_editor_make_visible(false);
			edit_anim->set_pressed(false);
		}
	}
}

void AnimationPlayerEditor::_input_event(InputEvent p_event) {


}

void AnimationPlayerEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_PROCESS) {

		if (!player)
			return;

		updating = true;

		if (player->is_playing()) {

			{
				String animname=player->get_current_animation();

				if (player->has_animation(animname)) {
					Ref<Animation> anim = player->get_animation(animname);
					if (!anim.is_null()) {

						seek->set_max(anim->get_length());
					}
				}
			}
			seek->set_val(player->get_current_animation_pos());
			if (edit_anim->is_pressed())
				editor->get_animation_editor()->set_anim_pos(player->get_current_animation_pos());
			EditorNode::get_singleton()->get_property_editor()->refresh();

		} else if (last_active) {
			//need the last frame after it stopped

			seek->set_val(player->get_current_animation_pos());
		}

		last_active=player->is_playing();
		//seek->set_val(player->get_pos());
		updating = false;
	}

	if (p_what==NOTIFICATION_ENTER_TREE) {

		editor->connect("hide_animation_player_editors",this,"_hide_anim_editors");
		add_anim->set_icon( get_icon("New","EditorIcons") );
		rename_anim->set_icon( get_icon("Rename","EditorIcons") );
		duplicate_anim->set_icon( get_icon("Duplicate","EditorIcons") );
		autoplay->set_icon( get_icon("AutoPlay","EditorIcons") );
		load_anim->set_icon( get_icon("Folder","EditorIcons") );
		remove_anim->set_icon( get_icon("Del","EditorIcons") );
		edit_anim->set_icon( get_icon("Edit","EditorIcons") );
		blend_anim->set_icon( get_icon("Blend","EditorIcons") );
		play->set_icon( get_icon("Play","EditorIcons") );
		autoplay_icon=get_icon("AutoPlay","EditorIcons");
		stop->set_icon( get_icon("Stop","EditorIcons") );
		resource_edit_anim->set_icon( get_icon("EditResource","EditorIcons") );
		pin->set_normal_texture(get_icon("Pin","EditorIcons") );
		pin->set_pressed_texture( get_icon("PinPressed","EditorIcons") );

		blend_editor.next->connect("text_changed",this,"_blend_editor_next_changed");

/*
		anim_editor_load->set_normal_texture( get_icon("AnimGet","EditorIcons"));
		anim_editor_store->set_normal_texture( get_icon("AnimSet","EditorIcons"));
		anim_editor_load->set_pressed_texture( get_icon("AnimGet","EditorIcons"));
		anim_editor_store->set_pressed_texture( get_icon("AnimSet","EditorIcons"));
		anim_editor_load->set_hover_texture( get_icon("AnimGetHl","EditorIcons"));
		anim_editor_store->set_hover_texture( get_icon("AnimSetHl","EditorIcons"));
*/
	}

	if (p_what==NOTIFICATION_READY) {

		get_tree()->connect("node_removed",this,"_node_removed");
	}

	if (p_what==NOTIFICATION_DRAW) {

	}
}

void AnimationPlayerEditor::_autoplay_pressed() {

	if (updating)
		return;
	if (animation->get_item_count()==0) {
		return;
	}

	String current = animation->get_item_text( animation->get_selected() );
	if (player->get_autoplay()==current) {
		//unset
		undo_redo->create_action("Toggle Autoplay");
		undo_redo->add_do_method(player,"set_autoplay","");
		undo_redo->add_undo_method(player,"set_autoplay",player->get_autoplay());
		undo_redo->add_do_method(this,"_animation_player_changed",player);
		undo_redo->add_undo_method(this,"_animation_player_changed",player);
		undo_redo->commit_action();


	} else {
		//set
		undo_redo->create_action("Toggle Autoplay");
		undo_redo->add_do_method(player,"set_autoplay",current);
		undo_redo->add_undo_method(player,"set_autoplay",player->get_autoplay());
		undo_redo->add_do_method(this,"_animation_player_changed",player);
		undo_redo->add_undo_method(this,"_animation_player_changed",player);
		undo_redo->commit_action();
	}

}

void AnimationPlayerEditor::_play_pressed() {

	String current;
	if (animation->get_selected()>=0 && animation->get_selected()<animation->get_item_count()) {

		current = animation->get_item_text( animation->get_selected() );
	}

	if (current!="") {

		if (current==player->get_current_animation())
			player->stop(); //so it wont blend with itself
		player->play(current );
	}

	//unstop
	stop->set_pressed(false);
	//unpause
	//pause->set_pressed(false);
}
void AnimationPlayerEditor::_stop_pressed() {

	player->stop();
	play->set_pressed(false);
	stop->set_pressed(true);
	//pause->set_pressed(false);
	//player->set_pause(false);
}

void AnimationPlayerEditor::_pause_pressed() {

	//player->set_pause( pause->is_pressed() );
}
void AnimationPlayerEditor::_animation_selected(int p_which) {

	if (updating)
		return;
	// when selecting an animation, the idea is that the only interesting behavior
	// ui-wise is that it should play/blend the next one if currently playing
	String current;
	if (animation->get_selected()>=0 && animation->get_selected()<animation->get_item_count()) {

		current = animation->get_item_text( animation->get_selected() );
	}

	if (current!="") {


		player->set_current_animation( current );

		Ref<Animation> anim =  player->get_animation(current);
		if (edit_anim->is_pressed()) {
			Ref<Animation> anim =  player->get_animation(current);
			editor->get_animation_editor()->set_animation(anim);
			Node *root = player->get_node(player->get_root());
			if (root) {
				editor->get_animation_editor()->set_root(root);
			}
		}
		seek->set_max(anim->get_length());


	} else {
		if (edit_anim->is_pressed()) {
			editor->get_animation_editor()->set_animation(Ref<Animation>());
			editor->get_animation_editor()->set_root(NULL);
		}

	}


	autoplay->set_pressed(current==player->get_autoplay());
}

void AnimationPlayerEditor::_animation_new() {

	renaming=false;
	name_title->set_text("New Animation Name:");

	int count=1;
	String base="New Anim";
	while(true) {
		String attempt  = base;
		if (count>1)
			attempt+=" ("+itos(count)+")";
		if (player->has_animation(attempt)) {
			count++;
			continue;
		}
		base=attempt;
		break;
	}

	name->set_text(base);
	name_dialog->popup_centered(Size2(300,90));
	name->select_all();
	name->grab_focus();
}
void AnimationPlayerEditor::_animation_rename() {

	if (animation->get_item_count()==0)
		return;
	int selected = animation->get_selected();
	String selected_name = animation->get_item_text(selected);

	name_title->set_text("Change Animation Name:");
	name->set_text(selected_name);
	renaming=true;
	name_dialog->popup_centered(Size2(300,90));
	name->select_all();
	name->grab_focus();

}
void AnimationPlayerEditor::_animation_load() {
	ERR_FAIL_COND(!player);
	file->set_mode( FileDialog::MODE_OPEN_FILE );
	file->clear_filters();
	List<String> extensions;

	ResourceLoader::get_recognized_extensions_for_type("Animation",&extensions);
	for (List<String>::Element *E=extensions.front();E;E=E->next()) {

		file->add_filter("*."+E->get()+" ; "+E->get().to_upper() );

	}

	file->popup_centered_ratio();


}
void AnimationPlayerEditor::_animation_remove() {

	if (animation->get_item_count()==0)
		return;

	String current = animation->get_item_text(animation->get_selected());
	Ref<Animation> anim =  player->get_animation(current);


	undo_redo->create_action("Remove Animation");
	undo_redo->add_do_method(player,"remove_animation",current);
	undo_redo->add_undo_method(player,"add_animation",current,anim);
	undo_redo->add_do_method(this,"_animation_player_changed",player);
	undo_redo->add_undo_method(this,"_animation_player_changed",player);
	undo_redo->commit_action();

}

void AnimationPlayerEditor::_select_anim_by_name(const String& p_anim) {

	int idx=-1;
	for(int i=0;i<animation->get_item_count();i++) {

		if (animation->get_item_text(i)==p_anim) {

			idx=i;
			break;
		}
	}

	ERR_FAIL_COND(idx==-1);


	animation->select(idx);

	_animation_selected(idx);

}

void AnimationPlayerEditor::_animation_name_edited() {

	player->stop();

	String new_name = name->get_text();
	if (new_name=="" || new_name.find(":")!=-1 || new_name.find("/")!=-1) {
		error_dialog->set_text("ERROR: Invalid animation name!");
		error_dialog->popup_centered(Size2(300,70));
		return;
	}

	if (renaming && animation->get_item_count()>0 && animation->get_item_text(animation->get_selected())==new_name) {
		name_dialog->hide();
		return;
	}

	if (player->has_animation(new_name)) {
		error_dialog->set_text("ERROR: Animation Name Already Exists!");
		error_dialog->popup_centered(Size2(300,70));
		return;
	}

	if (renaming) {
		String current = animation->get_item_text(animation->get_selected());
		Ref<Animation> anim =  player->get_animation(current);

		undo_redo->create_action("Rename Animation");
		undo_redo->add_do_method(player,"rename_animation",current,new_name);
		undo_redo->add_do_method(anim.ptr(),"set_name",new_name);
		undo_redo->add_undo_method(player,"rename_animation",new_name,current);
		undo_redo->add_undo_method(anim.ptr(),"set_name",current);
		undo_redo->add_do_method(this,"_animation_player_changed",player);
		undo_redo->add_undo_method(this,"_animation_player_changed",player);
		undo_redo->commit_action();

		_select_anim_by_name(new_name);

	} else {

		Ref<Animation> new_anim = Ref<Animation>(memnew( Animation ));
		new_anim->set_name(new_name);

		undo_redo->create_action("Add Animation");
		undo_redo->add_do_method(player,"add_animation",new_name,new_anim);
		undo_redo->add_undo_method(player,"remove_animation",new_name);
		undo_redo->add_do_method(this,"_animation_player_changed",player);
		undo_redo->add_undo_method(this,"_animation_player_changed",player);
		undo_redo->commit_action();

		_select_anim_by_name(new_name);

	}

	name_dialog->hide();
}


void AnimationPlayerEditor::_blend_editor_next_changed(const String& p_string) {

	if (animation->get_item_count()==0)
		return;

	String current = animation->get_item_text(animation->get_selected());
	player->animation_set_next(current,p_string);

}

void AnimationPlayerEditor::_animation_blend() {

	if (updating_blends)
		return;

	blend_editor.tree->clear();

	if (animation->get_item_count()==0)
		return;

	String current = animation->get_item_text(animation->get_selected());

	blend_editor.dialog->popup_centered(Size2(400,400));

	blend_editor.tree->set_hide_root(true);
	blend_editor.tree->set_column_min_width(0,10);
	blend_editor.tree->set_column_min_width(1,3);

	List<StringName> anims;
	player->get_animation_list(&anims);
	TreeItem *root = blend_editor.tree->create_item();
	updating_blends=true;

	for(List<StringName>::Element *E=anims.front();E;E=E->next()) {

		String to=E->get();
		TreeItem *blend=blend_editor.tree->create_item(root);
		blend->set_editable(0,false);
		blend->set_editable(1,true);
		blend->set_text(0,to);
		blend->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
		blend->set_range_config(1,0,3600,0.001);
		blend->set_range(1,player->get_blend_time(current,to));
	}

	blend_editor.next->set_text( player->animation_get_next(current) );

	updating_blends=false;
}

void AnimationPlayerEditor::_blend_edited() {

	if (updating_blends)
		return;

	if (animation->get_item_count()==0)
		return;

	String current = animation->get_item_text(animation->get_selected());

	TreeItem *selected = blend_editor.tree->get_edited();
	if (!selected)
		return;

	updating_blends=true;
	String to=selected->get_text(0);
	float blend_time = selected->get_range(1);
	float prev_blend_time = player->get_blend_time(current,to);

	undo_redo->create_action("Change Blend Time");
	undo_redo->add_do_method(player,"set_blend_time",current,to,blend_time);
	undo_redo->add_undo_method(player,"set_blend_time",current,to,prev_blend_time);
	undo_redo->add_do_method(this,"_animation_player_changed",player);
	undo_redo->add_undo_method(this,"_animation_player_changed",player);
	undo_redo->commit_action();
	updating_blends=false;
}

void AnimationPlayerEditor::ensure_visibility() {

	_animation_edit();
}

void AnimationPlayerEditor::_animation_resource_edit() {

	if (animation->get_item_count()) {
		String current = animation->get_item_text(animation->get_selected());
		Ref<Animation> anim =  player->get_animation(current);
		editor->edit_resource(anim);
	}

}

void AnimationPlayerEditor::_animation_edit() {

//	if (animation->get_item_count()==0)
//		return;

	if (edit_anim->is_pressed()) {
		editor->animation_editor_make_visible(true);

		//editor->get_animation_editor()->set_root(player->get_roo); - get root pending
		if (animation->get_item_count()) {
			String current = animation->get_item_text(animation->get_selected());
			Ref<Animation> anim =  player->get_animation(current);
			editor->get_animation_editor()->set_animation(anim);
			Node *root = player->get_node(player->get_root());
			if (root) {
				editor->get_animation_editor()->set_root(root);
			}

		} else {

			editor->get_animation_editor()->set_animation(Ref<Animation>());
			editor->get_animation_editor()->set_root(NULL);

		}
	} else {
		editor->animation_editor_make_visible(false);
		editor->get_animation_editor()->set_animation(Ref<Animation>());
		editor->get_animation_editor()->set_root(NULL);
	}

	//get_scene()->get_root_node()->call("_resource_selected",anim,"");

}
void AnimationPlayerEditor::_file_selected(String p_file) {

	ERR_FAIL_COND(!player);

	Ref<Resource> res = ResourceLoader::load(p_file,"Animation");
	ERR_FAIL_COND(res.is_null());
	ERR_FAIL_COND( !res->is_type("Animation") );
	if (p_file.find_last("/")!=-1) {

		p_file=p_file.substr( p_file.find_last("/")+1, p_file.length() );

	}
	if (p_file.find_last("\\")!=-1) {

		p_file=p_file.substr( p_file.find_last("\\")+1, p_file.length() );

	}

	if (p_file.find(".")!=-1)
		p_file=p_file.substr(0,p_file.find("."));

	undo_redo->create_action("Load Animation");
	undo_redo->add_do_method(player,"add_animation",p_file,res);
	undo_redo->add_undo_method(player,"remove_animation",p_file);
	if (player->has_animation(p_file)) {
		undo_redo->add_undo_method(player,"add_animation",p_file,player->get_animation(p_file));

	}
	undo_redo->add_do_method(this,"_animation_player_changed",player);
	undo_redo->add_undo_method(this,"_animation_player_changed",player);
	undo_redo->commit_action();

}

void AnimationPlayerEditor::_scale_changed(const String& p_scale) {

	player->set_speed(p_scale.to_double());
}

void AnimationPlayerEditor::_update_animation() {

	// the purpose of _update_animation is to reflect the current state
	// of the animation player in the current editor..

	updating=true;


	if (player->is_playing()) {

		play->set_pressed(true);
		stop->set_pressed(false);

	} else {

		play->set_pressed(false);
		stop->set_pressed(true);
	}

	scale->set_text( String::num(player->get_speed(),2) );
	String current=player->get_current_animation();

	for (int i=0;i<animation->get_item_count();i++) {

		if (animation->get_item_text(i)==current) {
			animation->select(i);
			break;
		}
	}

	updating=false;
}

void AnimationPlayerEditor::_update_player() {

	if (!player)
		return;

	updating=true;
	List<StringName> animlist;
	player->get_animation_list(&animlist);

	animation->clear();
	nodename->set_text(player->get_name());

	stop->set_disabled(animlist.size()==0);
	play->set_disabled(animlist.size()==0);
	autoplay->set_disabled(animlist.size()==0);
	duplicate_anim->set_disabled(animlist.size()==0);
	rename_anim->set_disabled(animlist.size()==0);
	blend_anim->set_disabled(animlist.size()==0);
	remove_anim->set_disabled(animlist.size()==0);
	resource_edit_anim->set_disabled(animlist.size()==0);

	int active_idx=-1;
	for (List<StringName>::Element *E=animlist.front();E;E=E->next()) {

		if (player->get_autoplay()==E->get())
			animation->add_icon_item(autoplay_icon,E->get());
		else
			animation->add_item(E->get());

		if (player->get_current_animation()==E->get())
			active_idx=animation->get_item_count()-1;

	}

	updating=false;
	if (active_idx!=-1) {
		animation->select(active_idx);
		autoplay->set_pressed(animation->get_item_text(active_idx)==player->get_autoplay());
		_animation_selected(active_idx);

	} else if (animation->get_item_count()>0){

		animation->select(0);
		autoplay->set_pressed(animation->get_item_text(0)==player->get_autoplay());
		_animation_selected(0);
	}

	//pause->set_pressed(player->is_paused());

	if (edit_anim->is_pressed()) {

		if (animation->get_item_count()) {
			String current = animation->get_item_text(animation->get_selected());
			Ref<Animation> anim =  player->get_animation(current);
			editor->get_animation_editor()->set_animation(anim);
			Node *root = player->get_node(player->get_root());
			if (root) {
				editor->get_animation_editor()->set_root(root);
			}

		}

	}

	_update_animation();
}



void AnimationPlayerEditor::edit(AnimationPlayer *p_player) {


	if (player && pin->is_pressed())
		return; //ignore, pinned
	player=p_player;

	if (player)
		_update_player();
	else {

//		hide();

	}

}


void AnimationPlayerEditor::_animation_duplicate() {


	if (!animation->get_item_count())
		return;

	String current = animation->get_item_text(animation->get_selected());
	Ref<Animation> anim =  player->get_animation(current);
	if (!anim.is_valid())
		return;

	Ref<Animation> new_anim = memnew( Animation );
	List<PropertyInfo> plist;
	anim->get_property_list(&plist);
	for (List<PropertyInfo>::Element *E=plist.front();E;E=E->next()) {

		if (E->get().usage&PROPERTY_USAGE_STORAGE) {

			new_anim->set(E->get().name, anim->get(E->get().name));
		}
	}
	new_anim->set_path("");

	String new_name = current;
	while(player->has_animation(new_name)) {

		new_name=new_name+" (copy)";
	}


	undo_redo->create_action("Duplicate Animation");
	undo_redo->add_do_method(player,"add_animation",new_name,new_anim);
	undo_redo->add_undo_method(player,"remove_animation",new_name);
	undo_redo->add_do_method(this,"_animation_player_changed",player);
	undo_redo->add_undo_method(this,"_animation_player_changed",player);
	undo_redo->commit_action();


	for(int i=0;i<animation->get_item_count();i++) {

		if (animation->get_item_text(i)==new_name) {

			animation->select(i);
			_animation_selected(i);
			return;
		}
	}

}

void AnimationPlayerEditor::_seek_value_changed(float p_value) {

	if (updating || !player || player->is_playing()) {
		return;
	};


	updating=true;
	String current=player->get_current_animation(); //animation->get_item_text( animation->get_selected() );
	if (current == "" || !player->has_animation(current)) {
		updating=false;
		current="";
		return;
	};

	Ref<Animation> anim;
	anim=player->get_animation(current);

	float pos = anim->get_length() * (p_value / seek->get_max());

	if (player->is_valid()) {
		float cpos = player->get_current_animation_pos();

		player->seek_delta(pos,pos-cpos);
	} else {
		player->seek(pos,true);
	}

	if (edit_anim->is_pressed())
		editor->get_animation_editor()->set_anim_pos(pos);

	updating=true;
};

void AnimationPlayerEditor::_animation_player_changed(Object *p_pl) {

	if (player==p_pl && is_visible()) {

		_update_player();
		if (blend_editor.dialog->is_visible())
			_animation_blend(); //update
	}
}



void AnimationPlayerEditor::_list_changed() {

	if(is_visible())
		_update_player();
}
#if 0
void AnimationPlayerEditor::_editor_store() {

	if (animation->get_item_count()==0)
		return;
	String current = animation->get_item_text(animation->get_selected());
	Ref<Animation> anim =  player->get_animation(current);

	if (editor->get_animation_editor()->get_current_animation()==anim)
		return; //already there


	undo_redo->create_action("Store anim in editor");
	undo_redo->add_do_method(editor->get_animation_editor(),"set_animation",anim);
	undo_redo->add_undo_method(editor->get_animation_editor(),"remove_animation",anim);
	undo_redo->commit_action();
}

void AnimationPlayerEditor::_editor_load(){

	Ref<Animation> anim = editor->get_animation_editor()->get_current_animation();
	if (anim.is_null())
		return;

	String existing = player->find_animation(anim);
	if (existing!="") {
		_select_anim_by_name(existing);
		return; //already has
	}

	int count=1;
	String base=anim->get_name();
	bool noname=false;
	if (base=="") {
		base="New Anim";
		noname=true;
	}

	while(true) {
		String attempt  = base;
		if (count>1)
			attempt+=" ("+itos(count)+")";
		if (player->has_animation(attempt)) {
			count++;
			continue;
		}
		base=attempt;
		break;
	}

	if (noname)
		anim->set_name(base);

	undo_redo->create_action("Add Animation From Editor");
	undo_redo->add_do_method(player,"add_animation",base,anim);
	undo_redo->add_undo_method(player,"remove_animation",base);
	undo_redo->add_do_method(this,"_animation_player_changed",player);
	undo_redo->add_undo_method(this,"_animation_player_changed",player);
	undo_redo->commit_action();

	_select_anim_by_name(base);


}
#endif

void AnimationPlayerEditor::_animation_key_editor_anim_len_changed(float p_len) {

	seek->set_max(p_len);

}


void AnimationPlayerEditor::_animation_key_editor_seek(float p_pos) {

	if (!is_visible())
		return;
	if (!player)
		return;

	if (player->is_playing()	)
		return;

	seek->set_val(p_pos);
	EditorNode::get_singleton()->get_property_editor()->refresh();



	//seekit
}

void AnimationPlayerEditor::_hide_anim_editors() {

	player=NULL;
	hide();
	set_process(false);
	if (edit_anim->is_pressed()) {

		editor->get_animation_editor()->set_animation(Ref<Animation>());
		editor->get_animation_editor()->set_root(NULL);
		editor->animation_editor_make_visible(false);
		edit_anim->set_pressed(false);
	}
}

void AnimationPlayerEditor::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_input_event"),&AnimationPlayerEditor::_input_event);
	ObjectTypeDB::bind_method(_MD("_node_removed"),&AnimationPlayerEditor::_node_removed);
	ObjectTypeDB::bind_method(_MD("_play_pressed"),&AnimationPlayerEditor::_play_pressed);
	ObjectTypeDB::bind_method(_MD("_stop_pressed"),&AnimationPlayerEditor::_stop_pressed);
	ObjectTypeDB::bind_method(_MD("_autoplay_pressed"),&AnimationPlayerEditor::_autoplay_pressed);
	ObjectTypeDB::bind_method(_MD("_pause_pressed"),&AnimationPlayerEditor::_pause_pressed);
	ObjectTypeDB::bind_method(_MD("_animation_selected"),&AnimationPlayerEditor::_animation_selected);
	ObjectTypeDB::bind_method(_MD("_animation_name_edited"),&AnimationPlayerEditor::_animation_name_edited);
	ObjectTypeDB::bind_method(_MD("_animation_new"),&AnimationPlayerEditor::_animation_new);
	ObjectTypeDB::bind_method(_MD("_animation_rename"),&AnimationPlayerEditor::_animation_rename);
	ObjectTypeDB::bind_method(_MD("_animation_load"),&AnimationPlayerEditor::_animation_load);
	ObjectTypeDB::bind_method(_MD("_animation_remove"),&AnimationPlayerEditor::_animation_remove);
	ObjectTypeDB::bind_method(_MD("_animation_blend"),&AnimationPlayerEditor::_animation_blend);
	ObjectTypeDB::bind_method(_MD("_animation_edit"),&AnimationPlayerEditor::_animation_edit);
	ObjectTypeDB::bind_method(_MD("_animation_resource_edit"),&AnimationPlayerEditor::_animation_resource_edit);
	ObjectTypeDB::bind_method(_MD("_file_selected"),&AnimationPlayerEditor::_file_selected);
	ObjectTypeDB::bind_method(_MD("_seek_value_changed"),&AnimationPlayerEditor::_seek_value_changed);
	ObjectTypeDB::bind_method(_MD("_animation_player_changed"),&AnimationPlayerEditor::_animation_player_changed);
	ObjectTypeDB::bind_method(_MD("_blend_edited"),&AnimationPlayerEditor::_blend_edited);
//	ObjectTypeDB::bind_method(_MD("_seek_frame_changed"),&AnimationPlayerEditor::_seek_frame_changed);
	ObjectTypeDB::bind_method(_MD("_scale_changed"),&AnimationPlayerEditor::_scale_changed);
	//ObjectTypeDB::bind_method(_MD("_editor_store_all"),&AnimationPlayerEditor::_editor_store_all);
	///jectTypeDB::bind_method(_MD("_editor_load_all"),&AnimationPlayerEditor::_editor_load_all);
	ObjectTypeDB::bind_method(_MD("_list_changed"),&AnimationPlayerEditor::_list_changed);
	ObjectTypeDB::bind_method(_MD("_animation_key_editor_seek"),&AnimationPlayerEditor::_animation_key_editor_seek);
	ObjectTypeDB::bind_method(_MD("_animation_key_editor_anim_len_changed"),&AnimationPlayerEditor::_animation_key_editor_anim_len_changed);
	ObjectTypeDB::bind_method(_MD("_hide_anim_editors"),&AnimationPlayerEditor::_hide_anim_editors);
	ObjectTypeDB::bind_method(_MD("_animation_duplicate"),&AnimationPlayerEditor::_animation_duplicate);
	ObjectTypeDB::bind_method(_MD("_blend_editor_next_changed"),&AnimationPlayerEditor::_blend_editor_next_changed);




}

AnimationPlayerEditor::AnimationPlayerEditor(EditorNode *p_editor) {
	editor=p_editor;

	updating=false;

	set_focus_mode(FOCUS_ALL);

	player=NULL;
	add_style_override("panel", get_stylebox("panel","Panel"));


	Label * l;

	/*l= memnew( Label );
	l->set_text("Animation Player:");
	add_child(l);*/

	HBoxContainer *hb = memnew( HBoxContainer );
	add_child(hb);


	add_anim = memnew( Button );
	add_anim->set_tooltip("Create new animation in player.");

	hb->add_child(add_anim);



	load_anim = memnew( Button );
	load_anim->set_tooltip("Load an animation from disk.");
	hb->add_child(load_anim);

	duplicate_anim = memnew( Button );
	hb->add_child(duplicate_anim);
	duplicate_anim->set_tooltip("Duplicate Animation");

	animation = memnew( OptionButton );
	hb->add_child(animation);
	animation->set_h_size_flags(SIZE_EXPAND_FILL);
	animation->set_tooltip("Display list of animations in player.");

	autoplay = memnew( Button );
	hb->add_child(autoplay);
	autoplay->set_tooltip("Autoplay On Load");


	rename_anim = memnew( Button );
	hb->add_child(rename_anim);
	rename_anim->set_tooltip("Rename Animation");

	remove_anim = memnew( Button );

	hb->add_child(remove_anim);
	remove_anim->set_tooltip("Remove Animation");

	blend_anim = memnew( Button );
	hb->add_child(blend_anim);
	blend_anim->set_tooltip("Edit Target Blend Times");



	edit_anim = memnew( Button );
	edit_anim->set_toggle_mode(true);
	hb->add_child(edit_anim);
	edit_anim->set_tooltip("Open animation editor.\nProperty editor will displays all editable keys too.");


	hb = memnew (HBoxContainer);
	add_child(hb);

	play = memnew( Button );
	play->set_tooltip("Play selected animation.");

	hb->add_child(play);

	stop = memnew( Button );
	stop->set_toggle_mode(true);
	hb->add_child(stop);
	play->set_tooltip("Stop animation playback.");

	//pause = memnew( Button );
	//pause->set_toggle_mode(true);
	//hb->add_child(pause);

	seek = memnew( HSlider );
	seek->set_val(0);
	seek->set_step(0.01);
	hb->add_child(seek);
	seek->set_h_size_flags(SIZE_EXPAND_FILL);
	seek->set_stretch_ratio(8);
	seek->set_tooltip("Seek animation (when stopped).");

	frame = memnew( SpinBox );
	hb->add_child(frame);
	frame->set_h_size_flags(SIZE_EXPAND_FILL);
	frame->set_stretch_ratio(2);
	frame->set_tooltip("Animation position (in seconds).");
	seek->share(frame);



	scale = memnew( LineEdit );
	hb->add_child(scale);
	scale->set_h_size_flags(SIZE_EXPAND_FILL);
	scale->set_stretch_ratio(1);
	scale->set_tooltip("Scale animation playback globally for the node.");
	scale->hide();

	resource_edit_anim= memnew( Button );
	hb->add_child(resource_edit_anim);


	file = memnew(FileDialog);
	add_child(file);

	name_dialog = memnew( ConfirmationDialog );
	name_dialog->set_hide_on_ok(false);
	add_child(name_dialog);
	name = memnew( LineEdit );
	name_dialog->add_child(name);
	name->set_pos(Point2(18,30));
	name->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,10);
	name_dialog->register_text_enter(name);


	l = memnew( Label );
	l->set_text("Animation Name:");
	l->set_pos( Point2(10,10) );

	name_dialog->add_child(l);
	name_title=l;

	error_dialog = memnew( ConfirmationDialog );
	error_dialog->get_ok()->set_text("Close");
	//error_dialog->get_cancel()->set_text("Close");
	error_dialog->set_text("Error!");
	add_child(error_dialog);

	name_dialog->connect("confirmed", this,"_animation_name_edited");
	
	blend_editor.dialog = memnew( AcceptDialog );
	add_child(blend_editor.dialog);
	blend_editor.dialog->get_ok()->set_text("Close");
	blend_editor.dialog->set_hide_on_ok(true);
	VBoxContainer *blend_vb = memnew( VBoxContainer);
	blend_editor.dialog->add_child(blend_vb);
	blend_editor.dialog->set_child_rect(blend_vb);
	blend_editor.tree = memnew( Tree );
	blend_editor.tree->set_columns(2);
	blend_vb->add_margin_child("Blend Times: ",blend_editor.tree,true);
	blend_editor.next = memnew( LineEdit );
	blend_vb->add_margin_child("Next (Auto Queue):",blend_editor.next);
	blend_editor.dialog->set_title("Cross-Animation Blend Times");
	updating_blends=false;

	blend_editor.tree->connect("item_edited",this,"_blend_edited");
	

	autoplay->connect("pressed", this,"_autoplay_pressed");
	autoplay->set_toggle_mode(true);
	play->connect("pressed", this,"_play_pressed");	
	stop->connect("pressed", this,"_stop_pressed");
	//pause->connect("pressed", this,"_pause_pressed");
	add_anim->connect("pressed", this,"_animation_new");
	rename_anim->connect("pressed", this,"_animation_rename");
	load_anim->connect("pressed", this,"_animation_load");
	duplicate_anim->connect("pressed", this,"_animation_duplicate");
	//frame->connect("text_entered", this,"_seek_frame_changed");
	edit_anim->connect("pressed", this,"_animation_edit");
	blend_anim->connect("pressed", this,"_animation_blend");
	remove_anim->connect("pressed", this,"_animation_remove");
	animation->connect("item_selected", this,"_animation_selected",Vector<Variant>(),true);
	resource_edit_anim->connect("pressed", this,"_animation_resource_edit");
	file->connect("file_selected", this,"_file_selected");
	 seek->connect("value_changed", this, "_seek_value_changed",Vector<Variant>(),true);
	 scale->connect("text_entered", this, "_scale_changed",Vector<Variant>(),true);
	 editor->get_animation_editor()->connect("timeline_changed",this,"_animation_key_editor_seek");
	 editor->get_animation_editor()->connect("animation_len_changed",this,"_animation_key_editor_anim_len_changed");

	 HBoxContainer *ahb = editor->get_animation_panel_hb();
	 nodename = memnew( Label );
	 ahb->add_child(nodename);
	 nodename->set_h_size_flags(SIZE_EXPAND_FILL);
	 nodename->set_opacity(0.5);
	 pin = memnew( TextureButton );
	 pin->set_toggle_mode(true);
	 ahb->add_child(pin);

	renaming=false;
	last_active=false;
}


void AnimationPlayerEditorPlugin::edit(Object *p_object) {

	anim_editor->set_undo_redo(&get_undo_redo());
	if (!p_object)
		return;
	anim_editor->edit(p_object->cast_to<AnimationPlayer>());
}

bool AnimationPlayerEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("AnimationPlayer");
}

void AnimationPlayerEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		anim_editor->show();
		anim_editor->set_process(true);
		anim_editor->ensure_visibility();
		editor->animation_panel_make_visible(true);
	} else {

//		anim_editor->hide();
//		anim_editor->set_idle_process(false);
	}

}

AnimationPlayerEditorPlugin::AnimationPlayerEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	anim_editor = memnew( AnimationPlayerEditor(editor) );
	editor->get_animation_panel()->add_child(anim_editor);
	/*
	editor->get_viewport()->add_child(anim_editor);
	anim_editor->set_area_as_parent_rect();
	anim_editor->set_anchor( MARGIN_TOP, Control::ANCHOR_END);
	anim_editor->set_margin( MARGIN_TOP, 75 );
	anim_editor->set_anchor( MARGIN_RIGHT, Control::ANCHOR_END);
	anim_editor->set_margin( MARGIN_RIGHT, 0 );*/
	anim_editor->hide();



}


AnimationPlayerEditorPlugin::~AnimationPlayerEditorPlugin()
{
}


