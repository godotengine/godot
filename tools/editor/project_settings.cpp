/*************************************************************************/
/*  project_settings.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "project_settings.h"
#include "scene/gui/tab_container.h"
#include "globals.h"
#include "os/keyboard.h"
#include "editor_node.h"
#include "scene/gui/margin_container.h"
#include "translation.h"
#include "global_constants.h"

ProjectSettings *ProjectSettings::singleton=NULL;

static const char* _button_names[JOY_BUTTON_MAX]={
"PS X, XBox A, Nintendo B",
"PS Circle, XBox B, Nintendo A",
"PS Square, XBox X, Nintendo Y",
"PS Triangle, XBox Y, Nintendo X",
"L, L1",
"R, R1",
"L2",
"R2",
"L3",
"R3",
"Select, Nintendo -",
"Start, Nintendo +",
"D-Pad Up",
"D-Pad Down",
"D-Pad Left",
"D-Pad Right"
};

void ProjectSettings::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		globals_editor->edit(Globals::get_singleton());

		search_button->set_icon(get_icon("Zoom","EditorIcons"));
		clear_button->set_icon(get_icon("Close","EditorIcons"));

		translation_list->connect("button_pressed",this,"_translation_delete");
		_update_actions();
		popup_add->add_icon_item(get_icon("Keyboard","EditorIcons"),"Key",InputEvent::KEY);
		popup_add->add_icon_item(get_icon("JoyButton","EditorIcons"),"Joy Button",InputEvent::JOYSTICK_BUTTON);
		popup_add->add_icon_item(get_icon("JoyAxis","EditorIcons"),"Joy Axis",InputEvent::JOYSTICK_MOTION);
		popup_add->add_icon_item(get_icon("Mouse","EditorIcons"),"Mouse Button",InputEvent::MOUSE_BUTTON);

		List<String> tfn;
		ResourceLoader::get_recognized_extensions_for_type("Translation",&tfn);
		for (List<String>::Element *E=tfn.front();E;E=E->next()) {

			translation_file_open->add_filter("*."+E->get());
		}

		List<String> rfn;
		ResourceLoader::get_recognized_extensions_for_type("Resource",&rfn);
		for (List<String>::Element *E=rfn.front();E;E=E->next()) {

			translation_res_file_open->add_filter("*."+E->get());
			translation_res_option_file_open->add_filter("*."+E->get());
		}

		List<String> afn;
		ResourceLoader::get_recognized_extensions_for_type("Script",&afn);
		ResourceLoader::get_recognized_extensions_for_type("PackedScene",&afn);

		for (List<String>::Element *E=afn.front();E;E=E->next()) {

			autoload_file_open->add_filter("*."+E->get());
		}

	}
}

void ProjectSettings::_action_selected() {

	TreeItem *ti=input_editor->get_selected();
	if (!ti || !ti->is_editable(0))
		return;

	add_at="input/"+ti->get_text(0);
}

void ProjectSettings::_action_edited() {

	TreeItem *ti=input_editor->get_selected();
	if (!ti)
		return;

	String new_name=ti->get_text(0);
	String old_name=add_at.substr(add_at.find("/")+1,add_at.length());

	if (new_name==old_name)
		return;

	if (new_name.find("/")!=-1 || new_name.find(":")!=-1 || new_name=="") {

		ti->set_text(0,old_name);
		add_at="input/"+old_name;

		message->set_text("Invalid Action (Anything goes but / or :).");
		message->popup_centered(Size2(300,100));
		return;
	}

	String action_prop="input/"+new_name;

	if (Globals::get_singleton()->has(action_prop)) {

		ti->set_text(0,old_name);
		add_at="input/"+old_name;

		message->set_text("Action '"+new_name+"' already exists!.");
		message->popup_centered(Size2(300,100));
		return;
	}

	int order = Globals::get_singleton()->get_order(add_at);
	Array va = Globals::get_singleton()->get(add_at);
	bool persisting = Globals::get_singleton()->is_persisting(add_at);

	setting=true;
	undo_redo->create_action("Rename Input Action Event");
	undo_redo->add_do_method(Globals::get_singleton(),"clear",add_at);
	undo_redo->add_do_method(Globals::get_singleton(),"set",action_prop,va);
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting",action_prop,persisting);
	undo_redo->add_do_method(Globals::get_singleton(),"set_order",action_prop,order);
	undo_redo->add_undo_method(Globals::get_singleton(),"clear",action_prop);
	undo_redo->add_undo_method(Globals::get_singleton(),"set",add_at,va);
	undo_redo->add_undo_method(Globals::get_singleton(),"set_persisting",add_at,persisting);
	undo_redo->add_undo_method(Globals::get_singleton(),"set_order",add_at,order);
	undo_redo->add_do_method(this,"_update_actions");
	undo_redo->add_undo_method(this,"_update_actions");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();
	setting=false;

	add_at=action_prop;

}


void ProjectSettings::_device_input_add() {




	InputEvent ie;
	String name=add_at;
	Variant old_val = Globals::get_singleton()->get(name);
	Array arr=old_val;
	ie.device=device_id->get_val();

	ie.type=add_type;

	switch(add_type) {

		case InputEvent::MOUSE_BUTTON: {


				ie.mouse_button.button_index=device_index->get_selected()+1;

				for(int i=0;i<arr.size();i++) {

					InputEvent aie=arr[i];
					if (aie.device == ie.device && aie.type==InputEvent::MOUSE_BUTTON && aie.mouse_button.button_index==ie.mouse_button.button_index) {
						return;
					}
				}

		} break;
		case InputEvent::JOYSTICK_MOTION: {

				ie.joy_motion.axis = device_index->get_selected()>>1;
				ie.joy_motion.axis_value = device_index->get_selected()&1?1:-1;


				for(int i=0;i<arr.size();i++) {

					InputEvent aie=arr[i];
					if (aie.device == ie.device && aie.type==InputEvent::JOYSTICK_MOTION && aie.joy_motion.axis==ie.joy_motion.axis && aie.joy_motion.axis_value==ie.joy_motion.axis_value) {
						return;
					}
				}

		} break;
		case InputEvent::JOYSTICK_BUTTON: {

				ie.joy_button.button_index=device_index->get_selected();

				for(int i=0;i<arr.size();i++) {

					InputEvent aie=arr[i];
					if (aie.device == ie.device && aie.type==InputEvent::JOYSTICK_BUTTON && aie.joy_button.button_index==ie.joy_button.button_index) {
						return;
					}
				}

		} break;
		default:{}
	}


	arr.push_back(ie);

	undo_redo->create_action("Add Input Action Event");
	undo_redo->add_do_method(Globals::get_singleton(),"set",name,arr);
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting",name,true);
	undo_redo->add_undo_method(Globals::get_singleton(),"set",name,old_val);
	undo_redo->add_do_method(this,"_update_actions");
	undo_redo->add_undo_method(this,"_update_actions");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();


}


void ProjectSettings::_press_a_key_confirm() {

	if (last_wait_for_key.type!=InputEvent::KEY)
		return;

	InputEvent ie;
	ie.type=InputEvent::KEY;
	ie.key.scancode=last_wait_for_key.key.scancode;
	ie.key.mod=last_wait_for_key.key.mod;
	String name=add_at;

	Variant old_val = Globals::get_singleton()->get(name);
	Array arr=old_val;

	for(int i=0;i<arr.size();i++) {

		InputEvent aie=arr[i];
		if (aie.type==InputEvent::KEY && aie.key.scancode==ie.key.scancode && aie.key.mod==ie.key.mod) {
			return;
		}
	}

	arr.push_back(ie);

	undo_redo->create_action("Add Input Action Event");
	undo_redo->add_do_method(Globals::get_singleton(),"set",name,arr);
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting",name,true);
	undo_redo->add_undo_method(Globals::get_singleton(),"set",name,old_val);
	undo_redo->add_do_method(this,"_update_actions");
	undo_redo->add_undo_method(this,"_update_actions");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();


}

void ProjectSettings::_wait_for_key(const InputEvent& p_event) {


	if (p_event.type==InputEvent::KEY && p_event.key.pressed && p_event.key.scancode!=0) {

		last_wait_for_key=p_event;
		String str=keycode_get_string(p_event.key.scancode).capitalize();
		if (p_event.key.mod.meta)
			str="Meta+"+str;
		if (p_event.key.mod.shift)
			str="Shift+"+str;
		if (p_event.key.mod.alt)
			str="Alt+"+str;
		if (p_event.key.mod.control)
			str="Control+"+str;


		press_a_key_label->set_text(str);
		press_a_key->accept_event();

	}
}


void ProjectSettings::_add_item(int p_item){

	add_type=InputEvent::Type(p_item);

	switch(add_type) {

		case InputEvent::KEY: {

			press_a_key_label->set_text("Press a Key..");
			last_wait_for_key=InputEvent();
			press_a_key->popup_centered(Size2(250,80));
			press_a_key->grab_focus();
		} break;
		case InputEvent::MOUSE_BUTTON: {

			device_id->set_val(0);
			device_index_label->set_text("Mouse Button Index:");
			device_index->clear();
			device_index->add_item("Left Button");
			device_index->add_item("Right Button");
			device_index->add_item("Middle Button");
			device_index->add_item("Wheel Up Button");
			device_index->add_item("Wheel Down Button");
			device_index->add_item("Button 6");
			device_index->add_item("Button 7");
			device_index->add_item("Button 8");
			device_index->add_item("Button 9");
			device_input->popup_centered(Size2(350,95));
		} break;
		case InputEvent::JOYSTICK_MOTION: {

			device_id->set_val(0);
			device_index_label->set_text("Joystick Axis Index:");
			device_index->clear();
			for(int i=0;i<JOY_AXIS_MAX*2;i++) {

				String desc;

				int ax=i/2;
				if (ax==0 || ax==1)
					desc=" (Left Stick)";
				else if (ax==2 || ax==3)
					desc=" (Right Stick)";
				else if (ax==6)
					desc=" (L2)";
				else if (ax==7)
					desc=" (R2)";


				device_index->add_item("Axis "+itos(i/2)+" "+(i&1?"+":"-")+desc);
			}
			device_input->popup_centered(Size2(350,95));

		} break;
		case InputEvent::JOYSTICK_BUTTON: {

			device_id->set_val(0);
			device_index_label->set_text("Joystick Button Index:");
			device_index->clear();

			for(int i=0;i<JOY_BUTTON_MAX;i++) {

				device_index->add_item(itos(i)+": "+String(_button_names[i]));
			}
			device_input->popup_centered(Size2(350,95));

		} break;
		default:{}
	}
}




void ProjectSettings::_action_button_pressed(Object* p_obj, int p_column,int p_id) {

	TreeItem *ti=p_obj->cast_to<TreeItem>();

	ERR_FAIL_COND(!ti);

	if (p_id==1) {
		Point2 ofs = input_editor->get_global_pos();
		Rect2 ir=input_editor->get_item_rect(ti);
		ir.pos.y-=input_editor->get_scroll().y;
		ofs+=ir.pos+ir.size;
		ofs.x-=100;
		popup_add->set_pos(ofs);
		popup_add->popup();
		add_at="input/"+ti->get_text(0);

	} else if (p_id==2) {
		//remove

		if (ti->get_parent()==input_editor->get_root()) {

			//remove main thing

			String name="input/"+ti->get_text(0);
			Variant old_val = Globals::get_singleton()->get(name);
			int order=Globals::get_singleton()->get_order(name);

			undo_redo->create_action("Add Input Action");
			undo_redo->add_do_method(Globals::get_singleton(),"clear",name);
			undo_redo->add_undo_method(Globals::get_singleton(),"set",name,old_val);
			undo_redo->add_undo_method(Globals::get_singleton(),"set_order",name,order);
			undo_redo->add_undo_method(Globals::get_singleton(),"set_persisting",name,Globals::get_singleton()->is_persisting(name));
			undo_redo->add_do_method(this,"_update_actions");
			undo_redo->add_undo_method(this,"_update_actions");
			undo_redo->add_do_method(this,"_settings_changed");
			undo_redo->add_undo_method(this,"_settings_changed");
			undo_redo->commit_action();

		} else {
			//remove action
			String name="input/"+ti->get_parent()->get_text(0);
			Variant old_val = Globals::get_singleton()->get(name);
			int idx = ti->get_metadata(0);

			Array va = old_val;

			ERR_FAIL_INDEX(idx,va.size());

			for(int i=idx;i<va.size()-1;i++) {

				va[i]=va[i+1];
			}

			va.resize(va.size()-1);


			undo_redo->create_action("Erase Input Action Event");
			undo_redo->add_do_method(Globals::get_singleton(),"set",name,va);
			undo_redo->add_undo_method(Globals::get_singleton(),"set",name,old_val);
			undo_redo->add_do_method(this,"_update_actions");
			undo_redo->add_undo_method(this,"_update_actions");
			undo_redo->add_do_method(this,"_settings_changed");
			undo_redo->add_undo_method(this,"_settings_changed");
			undo_redo->commit_action();

		}
	}

}


void ProjectSettings::_update_actions() {

	if (setting)
		return;

	input_editor->clear();
	TreeItem *root = input_editor->create_item();
	input_editor->set_hide_root(true);

	List<PropertyInfo> props;
	Globals::get_singleton()->get_property_list(&props);

	for(List<PropertyInfo>::Element *E=props.front();E;E=E->next()) {

		const PropertyInfo &pi=E->get();
		if (!pi.name.begins_with("input/"))
			continue;

		String name = pi.name.get_slice("/",1);
		if (name=="")
			continue;

		TreeItem *item=input_editor->create_item(root);
		//item->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		item->set_text(0,name);
		item->add_button(0,get_icon("Add","EditorIcons"),1);
		if (!Globals::get_singleton()->get_input_presets().find(pi.name)) {
			item->add_button(0,get_icon("Remove","EditorIcons"),2);
			item->set_editable(0,true);
		}
		item->set_custom_bg_color(0,get_color("prop_subsection","Editor"));
		//item->set_checked(0,pi.usage&PROPERTY_USAGE_CHECKED);


		Array actions=Globals::get_singleton()->get(pi.name);

		for(int i=0;i<actions.size();i++) {

			if (actions[i].get_type()!=Variant::INPUT_EVENT)
				continue;
			InputEvent ie = actions[i];

			TreeItem *action = input_editor->create_item(item);

			switch(ie.type) {

				case InputEvent::KEY: {

					String str=keycode_get_string(ie.key.scancode).capitalize();
					if (ie.key.mod.meta)
						str="Meta+"+str;
					if (ie.key.mod.shift)
						str="Shift+"+str;
					if (ie.key.mod.alt)
						str="Alt+"+str;
					if (ie.key.mod.control)
						str="Control+"+str;

					action->set_text(0,str);
					action->set_icon(0,get_icon("Keyboard","EditorIcons"));

				} break;
				case InputEvent::JOYSTICK_BUTTON: {

					String str = "Device "+itos(ie.device)+", Button "+itos(ie.joy_button.button_index);
					if (ie.joy_button.button_index>=0 && ie.joy_button.button_index<JOY_BUTTON_MAX)
						str+=String()+" ("+_button_names[ie.joy_button.button_index]+").";
					else
						str+=".";

					action->set_text(0,str);
					action->set_icon(0,get_icon("JoyButton","EditorIcons"));
				} break;
				case InputEvent::MOUSE_BUTTON: {

					String str = "Device "+itos(ie.device)+", ";
					switch (ie.mouse_button.button_index) {
						case BUTTON_LEFT: str+="Left Button."; break;
						case BUTTON_RIGHT: str+="Right Button."; break;
						case BUTTON_MIDDLE: str+="Middle Button."; break;
						case BUTTON_WHEEL_UP: str+="Wheel Up."; break;
						case BUTTON_WHEEL_DOWN: str+="Wheel Down."; break;
						default: str+="Button "+itos(ie.mouse_button.button_index)+".";
					}

					action->set_text(0,str);
					action->set_icon(0,get_icon("Mouse","EditorIcons"));
				} break;
				case InputEvent::JOYSTICK_MOTION: {

					String desc;
					int ax = ie.joy_motion.axis;

					if (ax==0 || ax==1)
						desc=" (Left Stick).";
					else if (ax==2 || ax==3)
						desc=" (Right Stick).";
					else if (ax==6)
						desc=" (L2).";
					else if (ax==7)
						desc=" (R2).";
					String str = "Device "+itos(ie.device)+", Axis "+itos(ie.joy_motion.axis)+" "+(ie.joy_motion.axis_value<0?"-":"+")+desc;
					action->set_text(0,str);
					action->set_icon(0,get_icon("JoyAxis","EditorIcons"));
				} break;
			}
			action->add_button(0,get_icon("Remove","EditorIcons"),2);
			action->set_metadata(0,i);
		}
	}
}


void ProjectSettings::popup_project_settings() {

	//popup_centered(Size2(500,400));
	popup_centered_ratio();
	globals_editor->update_category_list();
	_update_translations();
	_update_autoload();
	plugin_settings->update_plugins();
}


void ProjectSettings::_item_selected() {


	TreeItem *ti = globals_editor->get_property_editor()->get_scene_tree()->get_selected();
	if (!ti)
		return;
	if (!ti->get_parent())
		return;
	category->set_text(globals_editor->get_current_section());
	property->set_text(ti->get_text(0));
	popup_platform->set_disabled(false);


}


void ProjectSettings::_item_adds(String) {

	_item_add();
}

void ProjectSettings::_item_add() {

	Variant value;
	switch(type->get_selected()) {
		case 0: value=false; break;
		case 1: value=0; break;
		case 2: value=0.0; break;
		case 3: value=""; break;
	}

	String catname = category->get_text().strip_edges();
	/*if (!catname.is_valid_identifier()) {
		message->set_text("Invalid Category.\nValid characters: a-z,A-Z,0-9 or _");
		message->popup_centered(Size2(300,100));
		return;
	}*/

	String propname = property->get_text().strip_edges();
	/*if (!propname.is_valid_identifier()) {
		message->set_text("Invalid Property.\nValid characters: a-z,A-Z,0-9 or _");
		message->popup_centered(Size2(300,100));
		return;
	}*/

	String name = catname!="" ? catname+"/"+propname : propname;

	Globals::get_singleton()->set(name,value);

	globals_editor->set_current_section(catname);
	globals_editor->update_category_list();

	_settings_changed();
}

void ProjectSettings::_item_del() {

	String catname = category->get_text().strip_edges();
	//ERR_FAIL_COND(!catname.is_valid_identifier());
	String propname = property->get_text().strip_edges();
	//ERR_FAIL_COND(!propname.is_valid_identifier());

	String name = catname!="" ? catname+"/"+propname : propname;

	Globals::get_singleton()->set(name,Variant());

	globals_editor->set_current_section(catname);
	globals_editor->update_category_list();

	_settings_changed();
}

void ProjectSettings::_action_adds(String) {

	_action_add();
}

void ProjectSettings::_action_add() {

	String action = action_name->get_text();
	if (action.find("/")!=-1 || action.find(":")!=-1 || action=="") {
		message->set_text("Invalid Action (Anything goes but / or :).");
		message->popup_centered(Size2(300,100));
		return;
	}

	if (Globals::get_singleton()->has("input/"+action)) {
		message->set_text("Action '"+action+"' already exists!.");
		message->popup_centered(Size2(300,100));
		return;
	}

	Array va;
	String name = "input/"+action;
	undo_redo->create_action("Add Input Action Event");
	undo_redo->add_do_method(Globals::get_singleton(),"set",name,va);
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting",name,true);
	undo_redo->add_undo_method(Globals::get_singleton(),"clear",name);
	undo_redo->add_do_method(this,"_update_actions");
	undo_redo->add_undo_method(this,"_update_actions");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();

	TreeItem *r = input_editor->get_root();

	if (!r)
		return;
	r=r->get_children();
	if (!r)
		return;
	while(r->get_next())
		r=r->get_next();

	if (!r)
		return;
	r->select(0);
	input_editor->ensure_cursor_is_visible();

}


void ProjectSettings::_item_checked(const String& p_item, bool p_check) {

	undo_redo->create_action("Toggle Persisting");
	String full_item = globals_editor->get_full_item_path(p_item);

	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting",full_item,p_check);
	undo_redo->add_undo_method(Globals::get_singleton(),"set_persisting",full_item,!p_check);
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->add_do_method(globals_editor->get_property_editor(),"update_tree");
	undo_redo->add_undo_method(globals_editor->get_property_editor(),"update_tree");
	undo_redo->commit_action();

}


void ProjectSettings::_save() {

	Error err = Globals::get_singleton()->save();
	message->set_text(err!=OK?"Error saving settings.":"Settings Saved OK.");
	message->popup_centered(Size2(300,100));
}



void ProjectSettings::_settings_prop_edited(const String& p_name) {

	String full_item = globals_editor->get_full_item_path(p_name);

	if (!Globals::get_singleton()->is_persisting(full_item)) {
		Globals::get_singleton()->set_persisting(full_item,true);

		{
			//small usability workaround, if anything related to resolution scaling or size is modified, change all of them together
			if (full_item=="display/width" || full_item=="display/height" || full_item=="display/stretch_mode") {
				Globals::get_singleton()->set_persisting("display/height",true);
				Globals::get_singleton()->set_persisting("display/width",true);
			}
		}


//		globals_editor->update_property(p_name);
		globals_editor->get_property_editor()->update_tree();
	}
	_settings_changed();
}

void ProjectSettings::_settings_changed() {

	timer->start();
}

void ProjectSettings::queue_save() {
	_settings_changed();
}


void ProjectSettings::_copy_to_platform(int p_which) {

	String catname = category->get_text();
	if (!catname.is_valid_identifier()) {
		message->set_text("Invalid Category.\nValid characters: a-z,A-Z,0-9 or _");
		message->popup_centered(Size2(300,100));
		return;
	}


	String propname = property->get_text();
	if (!propname.is_valid_identifier()) {
		message->set_text("Invalid Property.\nValid characters: a-z,A-Z,0-9 or _");
		message->popup_centered(Size2(300,100));
		return;
	}

	String name = catname+"/"+propname;
	Variant value=Globals::get_singleton()->get(name);

	catname+="."+popup_platform->get_popup()->get_item_text(p_which);;
	name = catname+"/"+propname;

	Globals::get_singleton()->set(name,value);
	globals_editor->get_property_editor()->update_tree();

}


void ProjectSettings::add_translation(const String& p_translation) {

	_translation_add(p_translation);
}

void ProjectSettings::_translation_add(const String& p_path) {

	StringArray translations = Globals::get_singleton()->get("locale/translations");


	for(int i=0;i<translations.size();i++) {

		if (translations[i]==p_path)
			return; //exists
	}

	translations.push_back(p_path);
	undo_redo->create_action("Add Translation");
	undo_redo->add_do_property(Globals::get_singleton(),"locale/translations",translations);
	undo_redo->add_undo_property(Globals::get_singleton(),"locale/translations",Globals::get_singleton()->get("locale/translations"));
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting","locale/translations",true);
	undo_redo->add_do_method(this,"_update_translations");
	undo_redo->add_undo_method(this,"_update_translations");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();

}

void ProjectSettings::_translation_file_open() {

	translation_file_open->popup_centered_ratio();
}


void ProjectSettings::_autoload_file_callback(const String& p_path) {

	autoload_add_path->set_text(p_path);
	//if (autoload_add_name->get_text().strip_edges()==String()) {

		autoload_add_name->set_text( p_path.get_file().basename() );
	//}

	//_translation_add(p_translation);
}

void ProjectSettings::_autoload_file_open() {

	autoload_file_open->popup_centered_ratio();
}

void ProjectSettings::_autoload_edited() {

	if (updating_autoload)
		return;

	TreeItem *ti = autoload_list->get_edited();
	if (!ti || autoload_list->get_edited_column()!=2)
		return;

	updating_autoload=true;
	bool checked=ti->is_checked(2);

	String base="autoload/"+ti->get_text(0);

	String path = Globals::get_singleton()->get(base);
	int order = Globals::get_singleton()->get_order(base);

	if (path.begins_with("*"))
		path=path.substr(1,path.length());

	if (checked)
		path="*"+path;

	undo_redo->create_action("Toggle Autoload GlobalVar");
	undo_redo->add_do_property(Globals::get_singleton(),base,path);
	undo_redo->add_undo_property(Globals::get_singleton(),base,Globals::get_singleton()->get(base));
	undo_redo->add_do_method(Globals::get_singleton(),"set_order",base,order); // keep order, as config order matters for these
	undo_redo->add_undo_method(Globals::get_singleton(),"set_order",base,order);
	undo_redo->add_do_method(this,"_update_autoload");
	undo_redo->add_undo_method(this,"_update_autoload");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();
	updating_autoload=false;

}

void ProjectSettings::_autoload_add() {

	String name = autoload_add_name->get_text();
	if (!name.is_valid_identifier()) {
		message->set_text("Invalid Name.\nValid characters: a-z,A-Z,0-9 or _");
		message->popup_centered(Size2(300,100));
		return;

	}

	if (ObjectTypeDB::type_exists(name)) {

		message->set_text("Invalid Name.Must not collide with an existing engine class name.");
		message->popup_centered(Size2(300,100));
		return;

	}

	for(int i=0;i<Variant::VARIANT_MAX;i++) {
		if (Variant::get_type_name(Variant::Type(i))==name) {

			message->set_text("Invalid Name.Must not collide with an existing buit-in type name.");
			message->popup_centered(Size2(300,100));
			return;

		}
	}

	for(int i=0;i<GlobalConstants::get_global_constant_count();i++) {

		if (GlobalConstants::get_global_constant_name(i)==name) {

			message->set_text("Invalid Name.Must not collide with an existing global constant name.");
			message->popup_centered(Size2(300,100));
			return;
		}

	}

	String path = autoload_add_path->get_text();
	if (!FileAccess::exists(path)) {
		message->set_text("Invalid Path.\nFile does not exist.");
		message->popup_centered(Size2(300,100));
		return;

	}
	if (!path.begins_with("res://")) {
		message->set_text("Invalid Path.\nNot in resource path.");
		message->popup_centered(Size2(300,100));
		return;

	}

	undo_redo->create_action("Add Autoload");
	name = "autoload/"+name;
	undo_redo->add_do_property(Globals::get_singleton(),name,"*"+path);
	if (Globals::get_singleton()->has(name))
		undo_redo->add_undo_property(Globals::get_singleton(),name,Globals::get_singleton()->get(name));
	else
		undo_redo->add_undo_property(Globals::get_singleton(),name,Variant());

	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting",name,true);
	undo_redo->add_do_method(this,"_update_autoload");
	undo_redo->add_undo_method(this,"_update_autoload");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();

	autoload_add_path->set_text("");
	autoload_add_name->set_text("");

	//autoload_file_open->popup_centered_ratio();
}

void ProjectSettings::_autoload_delete(Object *p_item,int p_column, int p_button) {


	TreeItem *ti=p_item->cast_to<TreeItem>();
	String name = "autoload/"+ti->get_text(0);

	if (p_button==0) {
		//delete
		int order = Globals::get_singleton()->get_order(name);
		undo_redo->create_action("Remove Autoload");
		undo_redo->add_do_property(Globals::get_singleton(),name,Variant());
		undo_redo->add_undo_property(Globals::get_singleton(),name,Globals::get_singleton()->get(name));
		undo_redo->add_undo_method(Globals::get_singleton(),"set_persisting",name,true);
		undo_redo->add_undo_method(Globals::get_singleton(),"set_order",name,order);
		undo_redo->add_do_method(this,"_update_autoload");
		undo_redo->add_undo_method(this,"_update_autoload");
		undo_redo->add_do_method(this,"_settings_changed");
		undo_redo->add_undo_method(this,"_settings_changed");
		undo_redo->commit_action();
	} else {

		TreeItem *swap;

		if (p_button==1) {
			swap=ti->get_prev();
		} else if (p_button==2) {
			swap=ti->get_next();
		}
		if (!swap)
			return;

		String swap_name= "autoload/"+swap->get_text(0);

		int order = Globals::get_singleton()->get_order(name);
		int swap_order = Globals::get_singleton()->get_order(swap_name);

		undo_redo->create_action("Move Autoload");
		undo_redo->add_do_method(Globals::get_singleton(),"set_order",swap_name,order);
		undo_redo->add_do_method(Globals::get_singleton(),"set_order",name,swap_order);
		undo_redo->add_undo_method(Globals::get_singleton(),"set_order",swap_name,swap_order);
		undo_redo->add_undo_method(Globals::get_singleton(),"set_order",name,order);
		undo_redo->add_do_method(this,"_update_autoload");
		undo_redo->add_undo_method(this,"_update_autoload");
		undo_redo->add_do_method(this,"_settings_changed");
		undo_redo->add_undo_method(this,"_settings_changed");
		undo_redo->commit_action();

	}

}


void ProjectSettings::_translation_delete(Object *p_item,int p_column, int p_button) {

	TreeItem *ti = p_item->cast_to<TreeItem>();
	ERR_FAIL_COND(!ti);

	int idx=ti->get_metadata(0);

	StringArray translations = Globals::get_singleton()->get("locale/translations");

	ERR_FAIL_INDEX(idx,translations.size());

	translations.remove(idx);

	undo_redo->create_action("Remove Translation");
	undo_redo->add_do_property(Globals::get_singleton(),"locale/translations",translations);
	undo_redo->add_undo_property(Globals::get_singleton(),"locale/translations",Globals::get_singleton()->get("locale/translations"));
	undo_redo->add_do_method(this,"_update_translations");
	undo_redo->add_undo_method(this,"_update_translations");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();


}

void ProjectSettings::_translation_res_file_open() {

	translation_res_file_open->popup_centered_ratio();

}

void ProjectSettings::_translation_res_add(const String& p_path){

	Variant prev;
	Dictionary remaps;

	if (Globals::get_singleton()->has("locale/translation_remaps")) {
		remaps = Globals::get_singleton()->get("locale/translation_remaps");
		prev=remaps;
	}

	if (remaps.has(p_path))
		return; //pointless already has it

	remaps[p_path]=StringArray();

	undo_redo->create_action("Add Remapped Path");
	undo_redo->add_do_property(Globals::get_singleton(),"locale/translation_remaps",remaps);
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting","locale/translation_remaps",true);
	undo_redo->add_undo_property(Globals::get_singleton(),"locale/translation_remaps",prev);
	undo_redo->add_do_method(this,"_update_translations");
	undo_redo->add_undo_method(this,"_update_translations");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();

}

void ProjectSettings::_translation_res_option_file_open(){

	translation_res_option_file_open->popup_centered_ratio();

}
void ProjectSettings::_translation_res_option_add(const String& p_path) {

	ERR_FAIL_COND(!Globals::get_singleton()->has("locale/translation_remaps"));

	Dictionary remaps = Globals::get_singleton()->get("locale/translation_remaps");

	TreeItem *k = translation_remap->get_selected();
	ERR_FAIL_COND(!k);

	String key = k->get_metadata(0);

	ERR_FAIL_COND(!remaps.has(key));
	StringArray r = remaps[key];
	r.push_back(p_path+":"+"en");
	remaps[key]=r;


	undo_redo->create_action("Resource Remap Add Remap");
	undo_redo->add_do_property(Globals::get_singleton(),"locale/translation_remaps",remaps);
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting","locale/translation_remaps",true);
	undo_redo->add_undo_property(Globals::get_singleton(),"locale/translation_remaps",Globals::get_singleton()->get("locale/translation_remaps"));
	undo_redo->add_do_method(this,"_update_translations");
	undo_redo->add_undo_method(this,"_update_translations");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();

}



void ProjectSettings::_translation_res_select() {

	if (updating_translations)
		return;


	call_deferred("_update_translations");
}

void ProjectSettings::_translation_res_option_changed() {

	if (updating_translations)
		return;

	if (!Globals::get_singleton()->has("locale/translation_remaps"))
		return;

	Dictionary remaps = Globals::get_singleton()->get("locale/translation_remaps");

	TreeItem *k = translation_remap->get_selected();
	ERR_FAIL_COND(!k);
	TreeItem *ed = translation_remap_options->get_edited();
	ERR_FAIL_COND(!ed);

	String key = k->get_metadata(0);
	int idx = ed->get_metadata(0);
	String path = ed->get_metadata(1);
	int which = ed->get_range(1);

	Vector<String> langs = TranslationServer::get_all_locales();

	ERR_FAIL_INDEX(which,langs.size());


	ERR_FAIL_COND(!remaps.has(key));
	StringArray r = remaps[key];
	ERR_FAIL_INDEX(idx,r.size());
	r.set(idx,path+":"+langs[which]);
	remaps[key]=r;

	updating_translations=true;
	undo_redo->create_action("Change Resource Remap Language");
	undo_redo->add_do_property(Globals::get_singleton(),"locale/translation_remaps",remaps);
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting","locale/translation_remaps",true);
	undo_redo->add_undo_property(Globals::get_singleton(),"locale/translation_remaps",Globals::get_singleton()->get("locale/translation_remaps"));
	undo_redo->add_do_method(this,"_update_translations");
	undo_redo->add_undo_method(this,"_update_translations");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();
	updating_translations=false;

}


void ProjectSettings::_translation_res_delete(Object *p_item,int p_column, int p_button) {


	if (updating_translations)
		return;

	if (!Globals::get_singleton()->has("locale/translation_remaps"))
		return;

	Dictionary remaps = Globals::get_singleton()->get("locale/translation_remaps");

	TreeItem *k = p_item->cast_to<TreeItem>();

	String key = k->get_metadata(0);
	ERR_FAIL_COND(!remaps.has(key));

	remaps.erase(key);

	undo_redo->create_action("Remove Resource Remap");
	undo_redo->add_do_property(Globals::get_singleton(),"locale/translation_remaps",remaps);
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting","locale/translation_remaps",true);
	undo_redo->add_undo_property(Globals::get_singleton(),"locale/translation_remaps",Globals::get_singleton()->get("locale/translation_remaps"));
	undo_redo->add_do_method(this,"_update_translations");
	undo_redo->add_undo_method(this,"_update_translations");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();
}

void ProjectSettings::_translation_res_option_delete(Object *p_item,int p_column, int p_button) {

	if (updating_translations)
		return;

	if (!Globals::get_singleton()->has("locale/translation_remaps"))
		return;

	Dictionary remaps = Globals::get_singleton()->get("locale/translation_remaps");

	TreeItem *k = translation_remap->get_selected();
	ERR_FAIL_COND(!k);
	TreeItem *ed = p_item->cast_to<TreeItem>();
	ERR_FAIL_COND(!ed);

	String key = k->get_metadata(0);
	int idx = ed->get_metadata(0);

	ERR_FAIL_COND(!remaps.has(key));
	StringArray r = remaps[key];
	ERR_FAIL_INDEX(idx,remaps.size());
	r.remove(idx);
	remaps[key]=r;


	undo_redo->create_action("Remove Resource Remap Option");
	undo_redo->add_do_property(Globals::get_singleton(),"locale/translation_remaps",remaps);
	undo_redo->add_do_method(Globals::get_singleton(),"set_persisting","locale/translation_remaps",true);
	undo_redo->add_undo_property(Globals::get_singleton(),"locale/translation_remaps",Globals::get_singleton()->get("locale/translation_remaps"));
	undo_redo->add_do_method(this,"_update_translations");
	undo_redo->add_undo_method(this,"_update_translations");
	undo_redo->add_do_method(this,"_settings_changed");
	undo_redo->add_undo_method(this,"_settings_changed");
	undo_redo->commit_action();

}

void ProjectSettings::_update_translations() {

	//update translations

	if (updating_translations)
		return;

	updating_translations=true;

	translation_list->clear();
	TreeItem *root = translation_list->create_item(NULL);
	translation_list->set_hide_root(true);
	if (Globals::get_singleton()->has("locale/translations")) {

		StringArray translations = Globals::get_singleton()->get("locale/translations");
		for(int i=0;i<translations.size();i++) {

			TreeItem *t = translation_list->create_item(root);
			t->set_editable(0,false);
			t->set_text(0,translations[i].replace_first("res://",""));
			t->set_tooltip(0,translations[i]);
			t->set_metadata(0,i);
			t->add_button(0,get_icon("Del","EditorIcons"),0);
		}
	}


	//update translation remaps

	String remap_selected;
	if (translation_remap->get_selected()) {
		remap_selected = translation_remap->get_selected()->get_metadata(0);
	}

	translation_remap->clear();
	translation_remap_options->clear();
	root = translation_remap->create_item(NULL);
	TreeItem *root2 = translation_remap_options->create_item(NULL);
	translation_remap->set_hide_root(true);
	translation_remap_options->set_hide_root(true);
	translation_res_option_add_button->set_disabled(true);

	Vector<String> langs = TranslationServer::get_all_locales();
	Vector<String> names = TranslationServer::get_all_locale_names();
	String langnames;
	for(int i=0;i<names.size();i++) {
		if (i>0)
			langnames+=",";
		langnames+=names[i];
	}

	if (Globals::get_singleton()->has("locale/translation_remaps")) {

		Dictionary remaps = Globals::get_singleton()->get("locale/translation_remaps");
		List<Variant> rk;
		remaps.get_key_list(&rk);
		Vector<String> keys;
		for(List<Variant>::Element *E=rk.front();E;E=E->next()) {
			keys.push_back(E->get());
		}
		keys.sort();

		for(int i=0;i<keys.size();i++) {

			TreeItem *t = translation_remap->create_item(root);
			t->set_editable(0,false);
			t->set_text(0,keys[i].replace_first("res://",""));
			t->set_tooltip(0,keys[i]);
			t->set_metadata(0,keys[i]);
			t->add_button(0,get_icon("Del","EditorIcons"),0);
			if (keys[i]==remap_selected) {
				t->select(0);
				translation_res_option_add_button->set_disabled(false);

				StringArray selected = remaps[keys[i]];
				for(int j=0;j<selected.size();j++) {

					String s = selected[j];
					int qp = s.find_last(":");
					String path = s.substr(0,qp);
					String locale = s.substr(qp+1,s.length());

					TreeItem *t2 = translation_remap_options->create_item(root2);
					t2->set_editable(0,false);
					t2->set_text(0,path.replace_first("res://",""));
					t2->set_tooltip(0,path);
					t2->set_metadata(0,j);
					t2->add_button(0,get_icon("Del","EditorIcons"),0);
					t2->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
					t2->set_text(1,langnames);
					t2->set_editable(1,true);
					t2->set_metadata(1,path);
					int idx = langs.find(locale);
					print_line("find "+locale+" at "+itos(idx));
					if (idx<0)
						idx=0;

					t2->set_range(1,idx);
				}
			}


		}
	}


	updating_translations=false;

}

void ProjectSettings::_update_autoload() {

	if (updating_autoload)
		return;

	updating_autoload=true;

	autoload_list->clear();
	TreeItem *root = autoload_list->create_item();
	autoload_list->set_hide_root(true);

	List<PropertyInfo> props;
	Globals::get_singleton()->get_property_list(&props);

	for(List<PropertyInfo>::Element *E=props.front();E;E=E->next()) {

		const PropertyInfo &pi=E->get();
		if (!pi.name.begins_with("autoload/"))
			continue;

		String name = pi.name.get_slice("/",1);
		String path = Globals::get_singleton()->get(pi.name);

		if (name=="")
			continue;
		bool global=false;
		if (path.begins_with("*")) {
			path=path.substr(1,path.length());
			global=true;
		}
		TreeItem *t = autoload_list->create_item(root);
		t->set_text(0,name);
		t->set_text(1,path);
		t->set_cell_mode(2,TreeItem::CELL_MODE_CHECK);
		t->set_editable(2,true);
		t->set_text(2,"Enable");
		t->set_checked(2,global);
		t->add_button(3,get_icon("MoveUp","EditorIcons"),1);
		t->add_button(3,get_icon("MoveDown","EditorIcons"),2);
		t->add_button(3,get_icon("Del","EditorIcons"),0);


	}

	updating_autoload=false;

}

void ProjectSettings::_toggle_search_bar(bool p_pressed) {

	globals_editor->get_property_editor()->set_use_filter(p_pressed);

	if (p_pressed) {

		search_bar->show();
		add_prop_bar->hide();
		search_box->grab_focus();
		search_box->select_all();
	} else {

		search_bar->hide();
		add_prop_bar->show();
	}
}

void ProjectSettings::_clear_search_box() {

	if (search_box->get_text()=="")
		return;

	search_box->clear();
	globals_editor->get_property_editor()->update_tree();
}

void ProjectSettings::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_item_selected"),&ProjectSettings::_item_selected);
	ObjectTypeDB::bind_method(_MD("_item_add"),&ProjectSettings::_item_add);
	ObjectTypeDB::bind_method(_MD("_item_adds"),&ProjectSettings::_item_adds);
	ObjectTypeDB::bind_method(_MD("_item_del"),&ProjectSettings::_item_del);
	ObjectTypeDB::bind_method(_MD("_item_checked"),&ProjectSettings::_item_checked);
	ObjectTypeDB::bind_method(_MD("_save"),&ProjectSettings::_save);
	ObjectTypeDB::bind_method(_MD("_action_add"),&ProjectSettings::_action_add);
	ObjectTypeDB::bind_method(_MD("_action_adds"),&ProjectSettings::_action_adds);
	ObjectTypeDB::bind_method(_MD("_action_selected"),&ProjectSettings::_action_selected);
	ObjectTypeDB::bind_method(_MD("_action_edited"),&ProjectSettings::_action_edited);
	ObjectTypeDB::bind_method(_MD("_action_button_pressed"),&ProjectSettings::_action_button_pressed);
	ObjectTypeDB::bind_method(_MD("_update_actions"),&ProjectSettings::_update_actions);
	ObjectTypeDB::bind_method(_MD("_wait_for_key"),&ProjectSettings::_wait_for_key);
	ObjectTypeDB::bind_method(_MD("_add_item"),&ProjectSettings::_add_item);
	ObjectTypeDB::bind_method(_MD("_device_input_add"),&ProjectSettings::_device_input_add);
	ObjectTypeDB::bind_method(_MD("_press_a_key_confirm"),&ProjectSettings::_press_a_key_confirm);
	ObjectTypeDB::bind_method(_MD("_settings_prop_edited"),&ProjectSettings::_settings_prop_edited);
	ObjectTypeDB::bind_method(_MD("_copy_to_platform"),&ProjectSettings::_copy_to_platform);
	ObjectTypeDB::bind_method(_MD("_update_translations"),&ProjectSettings::_update_translations);
	ObjectTypeDB::bind_method(_MD("_translation_delete"),&ProjectSettings::_translation_delete);
	ObjectTypeDB::bind_method(_MD("_settings_changed"),&ProjectSettings::_settings_changed);
	ObjectTypeDB::bind_method(_MD("_translation_add"),&ProjectSettings::_translation_add);
	ObjectTypeDB::bind_method(_MD("_translation_file_open"),&ProjectSettings::_translation_file_open);

	ObjectTypeDB::bind_method(_MD("_translation_res_add"),&ProjectSettings::_translation_res_add);
	ObjectTypeDB::bind_method(_MD("_translation_res_file_open"),&ProjectSettings::_translation_res_file_open);
	ObjectTypeDB::bind_method(_MD("_translation_res_option_add"),&ProjectSettings::_translation_res_option_add);
	ObjectTypeDB::bind_method(_MD("_translation_res_option_file_open"),&ProjectSettings::_translation_res_option_file_open);
	ObjectTypeDB::bind_method(_MD("_translation_res_select"),&ProjectSettings::_translation_res_select);
	ObjectTypeDB::bind_method(_MD("_translation_res_option_changed"),&ProjectSettings::_translation_res_option_changed);
	ObjectTypeDB::bind_method(_MD("_translation_res_delete"),&ProjectSettings::_translation_res_delete);
	ObjectTypeDB::bind_method(_MD("_translation_res_option_delete"),&ProjectSettings::_translation_res_option_delete);

	ObjectTypeDB::bind_method(_MD("_autoload_add"),&ProjectSettings::_autoload_add);
	ObjectTypeDB::bind_method(_MD("_autoload_file_open"),&ProjectSettings::_autoload_file_open);
	ObjectTypeDB::bind_method(_MD("_autoload_file_callback"),&ProjectSettings::_autoload_file_callback);
	ObjectTypeDB::bind_method(_MD("_update_autoload"),&ProjectSettings::_update_autoload);
	ObjectTypeDB::bind_method(_MD("_autoload_delete"),&ProjectSettings::_autoload_delete);
	ObjectTypeDB::bind_method(_MD("_autoload_edited"),&ProjectSettings::_autoload_edited);

	ObjectTypeDB::bind_method(_MD("_clear_search_box"),&ProjectSettings::_clear_search_box);
	ObjectTypeDB::bind_method(_MD("_toggle_search_bar"),&ProjectSettings::_toggle_search_bar);

}

ProjectSettings::ProjectSettings(EditorData *p_data) {


	singleton=this;
	set_title("Project Settings (engine.cfg)");
	undo_redo=&p_data->get_undo_redo();
	data=p_data;


	TabContainer *tab_container = memnew( TabContainer );
	add_child(tab_container);
	set_child_rect(tab_container);

	//tab_container->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN, 15 );
	//tab_container->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END, 15 );
	//tab_container->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN, 15 );
	//tab_container->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END, 35 );

	VBoxContainer *props_base = memnew( VBoxContainer );
	props_base->set_alignment(BoxContainer::ALIGN_BEGIN);
	props_base->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tab_container->add_child(props_base);
	props_base->set_name("General");

	HBoxContainer *hbc = memnew( HBoxContainer );
	hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	props_base->add_child(hbc);

	search_button = memnew( ToolButton );
	search_button->set_toggle_mode(true);
	search_button->set_pressed(false);
	search_button->set_text("Search");
	hbc->add_child(search_button);
	search_button->connect("toggled",this,"_toggle_search_bar");

	hbc->add_child( memnew( VSeparator ) );

	add_prop_bar = memnew( HBoxContainer );
	add_prop_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->add_child(add_prop_bar);

	Label *l = memnew( Label );
	add_prop_bar->add_child(l);
	l->set_text("Category:");

	category = memnew( LineEdit );
	category->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_prop_bar->add_child(category);
	category->connect("text_entered",this,"_item_adds");

	l = memnew( Label );
	add_prop_bar->add_child(l);
	l->set_text("Property:");

	property = memnew( LineEdit );
	property->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_prop_bar->add_child(property);
	property->connect("text_entered",this,"_item_adds");

	l = memnew( Label );
	add_prop_bar->add_child(l);
	l->set_text("Type:");

	type = memnew( OptionButton );
	type->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_prop_bar->add_child(type);
	type->add_item("bool");
	type->add_item("int");
	type->add_item("float");
	type->add_item("string");

	Button *add = memnew( Button );
	add_prop_bar->add_child(add);
	add->set_text("Add");
	add->connect("pressed",this,"_item_add");

	Button *del = memnew( Button );
	add_prop_bar->add_child(del);
	del->set_text("Del");
	del->connect("pressed",this,"_item_del");

	search_bar = memnew( HBoxContainer );
	search_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->add_child(search_bar);
	search_bar->hide();

	search_box = memnew( LineEdit );
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_bar->add_child(search_box);

	clear_button = memnew( ToolButton );
	search_bar->add_child(clear_button);
	clear_button->connect("pressed",this,"_clear_search_box");

	globals_editor = memnew( SectionedPropertyEditor );
	props_base->add_child(globals_editor);
	//globals_editor->hide_top_label();
	globals_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	globals_editor->get_property_editor()->register_text_enter(search_box);
	globals_editor->get_property_editor()->set_capitalize_paths(false);
	globals_editor->get_property_editor()->get_scene_tree()->connect("cell_selected",this,"_item_selected");
	globals_editor->get_property_editor()->connect("property_toggled",this,"_item_checked",varray(),CONNECT_DEFERRED);
	globals_editor->get_property_editor()->connect("property_edited",this,"_settings_prop_edited");

/*
	Button *save = memnew( Button );
	props_base->add_child(save);

	save->set_anchor(MARGIN_LEFT,ANCHOR_END);
	save->set_anchor(MARGIN_RIGHT,ANCHOR_END);
	save->set_anchor(MARGIN_TOP,ANCHOR_END);
	save->set_anchor(MARGIN_BOTTOM,ANCHOR_END);
	save->set_begin( Point2(80,28) );
	save->set_end( Point2(10,20) );
	save->set_text("Save");
	save->connect("pressed",this,"_save");
*/

	hbc = memnew( HBoxContainer );
	props_base->add_child(hbc);

	popup_platform = memnew( MenuButton );
	popup_platform->set_text("Copy To Platform..");
	popup_platform->set_disabled(true);
	hbc->add_child(popup_platform);

	hbc->add_spacer();

	List<StringName> ep;
	EditorImportExport::get_singleton()->get_export_platforms(&ep);
	ep.sort_custom<StringName::AlphCompare>();

	for(List<StringName>::Element *E=ep.front();E;E=E->next()) {

		popup_platform->get_popup()->add_item( E->get() );

	}

	popup_platform->get_popup()->connect("item_pressed",this,"_copy_to_platform");
	get_ok()->set_text("Close");
	set_hide_on_ok(true);

	message = memnew( ConfirmationDialog );
	add_child(message);
//	message->get_cancel()->hide();
	message->set_hide_on_ok(true);

	Control *input_base = memnew( Control );
	input_base->set_name("Input Map");
	input_base->set_area_as_parent_rect();;
	tab_container->add_child(input_base);

	l = memnew( Label );
	input_base->add_child(l);
	l->set_pos(Point2(6,5));
	l->set_text("Action:");

	action_name = memnew( LineEdit );
	action_name->set_anchor(MARGIN_RIGHT,ANCHOR_RATIO);
	action_name->set_begin( Point2(5,25) );
	action_name->set_end( Point2(0.85,26) );
	input_base->add_child(action_name);
	action_name->connect("text_entered",this,"_action_adds");

	add = memnew( Button );
	input_base->add_child(add);
	add->set_anchor(MARGIN_LEFT,ANCHOR_RATIO);
	add->set_begin( Point2(0.86,25) );
	add->set_anchor(MARGIN_RIGHT,ANCHOR_END);
	add->set_end( Point2(5,26) );
	add->set_text("Add");
	add->connect("pressed",this,"_action_add");

	input_editor = memnew( Tree );
	input_base->add_child(input_editor);
	input_editor->set_area_as_parent_rect();
	input_editor->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN, 55 );
	input_editor->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END, 35 );
	input_editor->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN, 5 );
	input_editor->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END, 5 );
	input_editor->connect("item_edited",this,"_action_edited");
	input_editor->connect("cell_selected",this,"_action_selected");
	input_editor->connect("button_pressed",this,"_action_button_pressed");
	popup_add = memnew( PopupMenu );
	add_child(popup_add);
	popup_add->connect("item_pressed",this,"_add_item");

	press_a_key = memnew( ConfirmationDialog );
	press_a_key->set_focus_mode(FOCUS_ALL);
	add_child(press_a_key);

	l = memnew( Label );
	l->set_text("Press a Key..");
	l->set_area_as_parent_rect();
	l->set_align(Label::ALIGN_CENTER);
	l->set_margin(MARGIN_TOP,20);
	l->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,30);
	press_a_key_label=l;
	press_a_key->add_child(l);
	press_a_key->connect("input_event",this,"_wait_for_key");
	press_a_key->connect("confirmed",this,"_press_a_key_confirm");


	device_input=memnew( ConfirmationDialog );
	add_child(device_input);
	device_input->get_ok()->set_text("Add");
	device_input->connect("confirmed",this,"_device_input_add");

	l = memnew( Label );
	l->set_text("Device:");
	l->set_pos(Point2(15,10));
	device_input->add_child(l);

	l = memnew( Label );
	l->set_text("Index:");
	l->set_pos(Point2(90,10));
	device_input->add_child(l);
	device_index_label=l;

	device_id = memnew( SpinBox );
	device_id->set_pos(Point2(20,30));
	device_id->set_size(Size2(70,10));
	device_id->set_val(0);

	device_input->add_child(device_id);

	device_index = memnew( OptionButton );
	device_index->set_pos(Point2(95,30));
	device_index->set_size(Size2(300,10));
	device_index->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,10);

	device_input->add_child(device_index);

	/*
	save = memnew( Button );
	input_base->add_child(save);
	save->set_anchor(MARGIN_LEFT,ANCHOR_END);
	save->set_anchor(MARGIN_RIGHT,ANCHOR_END);
	save->set_anchor(MARGIN_TOP,ANCHOR_END);
	save->set_anchor(MARGIN_BOTTOM,ANCHOR_END);
	save->set_begin( Point2(80,28) );
	save->set_end( Point2(10,20) );
	save->set_text("Save");
	save->connect("pressed",this,"_save");
*/
	setting=false;

	//translations
	TabContainer *translations = memnew( TabContainer );
	translations->set_name("Localization");
	tab_container->add_child(translations);

	{

		VBoxContainer *tvb = memnew( VBoxContainer );
		translations->add_child(tvb);
		tvb->set_name("Translations");
		HBoxContainer *thb = memnew( HBoxContainer);
		tvb->add_child(thb);
		thb->add_child( memnew( Label("Translations:")));
		thb->add_spacer();
		Button *addtr = memnew( Button("Add..") );
		addtr->connect("pressed",this,"_translation_file_open");
		thb->add_child(addtr);
		MarginContainer *tmc = memnew( MarginContainer );
		tvb->add_child(tmc);
		tmc->set_v_size_flags(SIZE_EXPAND_FILL);
		translation_list = memnew( Tree );
		translation_list->set_v_size_flags(SIZE_EXPAND_FILL);
		tmc->add_child(translation_list);

		translation_file_open=memnew( EditorFileDialog );
		add_child(translation_file_open);
		translation_file_open->set_mode(EditorFileDialog::MODE_OPEN_FILE);
		translation_file_open->connect("file_selected",this,"_translation_add");

	}

	{
		VBoxContainer *tvb = memnew( VBoxContainer );
		translations->add_child(tvb);
		tvb->set_name("Remaps");
		HBoxContainer *thb = memnew( HBoxContainer);
		tvb->add_child(thb);
		thb->add_child( memnew( Label("Resources:")));
		thb->add_spacer();
		Button *addtr = memnew( Button("Add..") );
		addtr->connect("pressed",this,"_translation_res_file_open");
		thb->add_child(addtr);
		MarginContainer *tmc = memnew( MarginContainer );
		tvb->add_child(tmc);
		tmc->set_v_size_flags(SIZE_EXPAND_FILL);
		translation_remap = memnew( Tree );
		translation_remap->set_v_size_flags(SIZE_EXPAND_FILL);
		translation_remap->connect("cell_selected",this,"_translation_res_select");
		tmc->add_child(translation_remap);
		translation_remap->connect("button_pressed",this,"_translation_res_delete");

		translation_res_file_open=memnew( EditorFileDialog );
		add_child(translation_res_file_open);
		translation_res_file_open->set_mode(EditorFileDialog::MODE_OPEN_FILE);
		translation_res_file_open->connect("file_selected",this,"_translation_res_add");

		thb = memnew( HBoxContainer);
		tvb->add_child(thb);
		thb->add_child( memnew( Label("Remaps by Locale:")));
		thb->add_spacer();
		addtr = memnew( Button("Add..") );
		addtr->connect("pressed",this,"_translation_res_option_file_open");
		translation_res_option_add_button=addtr;
		thb->add_child(addtr);
		tmc = memnew( MarginContainer );
		tvb->add_child(tmc);
		tmc->set_v_size_flags(SIZE_EXPAND_FILL);
		translation_remap_options = memnew( Tree );
		translation_remap_options->set_v_size_flags(SIZE_EXPAND_FILL);
		tmc->add_child(translation_remap_options);

		translation_remap_options->set_columns(2);
		translation_remap_options->set_column_title(0,"Path");
		translation_remap_options->set_column_title(1,"Locale");
		translation_remap_options->set_column_titles_visible(true);
		translation_remap_options->set_column_expand(0,true);
		translation_remap_options->set_column_expand(1,false);
		translation_remap_options->set_column_min_width(1,200);
		translation_remap_options->connect("item_edited",this,"_translation_res_option_changed");
		translation_remap_options->connect("button_pressed",this,"_translation_res_option_delete");

		translation_res_option_file_open=memnew( EditorFileDialog );
		add_child(translation_res_option_file_open);
		translation_res_option_file_open->set_mode(EditorFileDialog::MODE_OPEN_FILE);
		translation_res_option_file_open->connect("file_selected",this,"_translation_res_option_add");

	}


	{
		VBoxContainer *avb = memnew( VBoxContainer );
		tab_container->add_child(avb);
		avb->set_name("AutoLoad");
		HBoxContainer *ahb = memnew( HBoxContainer);
		avb->add_child(ahb);


		VBoxContainer *avb_path = memnew( VBoxContainer );
		avb_path->set_h_size_flags(SIZE_EXPAND_FILL);
		HBoxContainer *ahb_path = memnew( HBoxContainer );
		autoload_add_path = memnew(LineEdit);
		autoload_add_path->set_h_size_flags(SIZE_EXPAND_FILL);
		ahb_path->add_child(autoload_add_path);
		Button *browseaa = memnew( Button("..") );
		ahb_path->add_child(browseaa);
		browseaa->connect("pressed",this,"_autoload_file_open");

		avb_path->add_margin_child("Path:",ahb_path);
		ahb->add_child(avb_path);

		VBoxContainer *avb_name = memnew( VBoxContainer );
		avb_name->set_h_size_flags(SIZE_EXPAND_FILL);

		HBoxContainer *ahb_name = memnew( HBoxContainer );
		autoload_add_name = memnew(LineEdit);
		autoload_add_name->set_h_size_flags(SIZE_EXPAND_FILL);
		ahb_name->add_child(autoload_add_name);
		avb_name->add_margin_child("Node Name:",ahb_name);
		Button *addaa = memnew( Button("Add") );
		ahb_name->add_child(addaa);
		addaa->connect("pressed",this,"_autoload_add");

		ahb->add_child(avb_name);

		autoload_list = memnew( Tree );
		autoload_list->set_v_size_flags(SIZE_EXPAND_FILL);
		avb->add_margin_child("List:",autoload_list,true);

		autoload_file_open=memnew( EditorFileDialog );
		add_child(autoload_file_open);
		autoload_file_open->set_mode(EditorFileDialog::MODE_OPEN_FILE);
		autoload_file_open->connect("file_selected",this,"_autoload_file_callback");

		autoload_list->set_columns(4);
		autoload_list->set_column_titles_visible(true);
		autoload_list->set_column_title(0,"Name");
		autoload_list->set_column_expand(0,true);
		autoload_list->set_column_min_width(0,100);
		autoload_list->set_column_title(1,"Path");
		autoload_list->set_column_expand(1,true);
		autoload_list->set_column_min_width(1,100);
		autoload_list->set_column_title(2,"Singleton");
		autoload_list->set_column_expand(2,false);
		autoload_list->set_column_min_width(2,80);
		autoload_list->set_column_expand(3,false);
		autoload_list->set_column_min_width(3,80);

		autoload_list->connect("button_pressed",this,"_autoload_delete");
		autoload_list->connect("item_edited",this,"_autoload_edited");

		updating_autoload=false;

	}

	{

		plugin_settings = memnew( EditorPluginSettings );
		plugin_settings->set_name("Plugins");
		tab_container->add_child(plugin_settings);
	}

	timer = memnew( Timer );
	timer->set_wait_time(1.5);
	timer->connect("timeout",Globals::get_singleton(),"save");
	timer->set_one_shot(true);
	add_child(timer);

	updating_translations=false;


	/*
	Control * es = memnew( Control );
	es->set_name("Export");
	tab_container->add_child(es);
	export_settings = memnew( ProjectExportSettings );
	es->add_child(export_settings);
	export_settings->set_area_as_parent_rect();
	export_settings->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END, 35 );
*/
}
