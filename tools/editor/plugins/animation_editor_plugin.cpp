/*************************************************************************/
/*  animation_editor_plugin.cpp                                          */
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
#include "animation_editor_plugin.h"
#include "io/resource_loader.h"


class AnimationEditor_TrackEditor : public Object {

	OBJ_TYPE(AnimationEditor_TrackEditor,Object);

protected:

	bool _set(const StringName& p_name, const Variant& p_value) {

		if (anim.is_null())
			return false;
		String name=p_name;

		if (name=="track/interpolation") {

			anim_editor->_internal_set_interpolation_type(track,Animation::InterpolationType(p_value.operator int()));

		} else if (name.begins_with("keys/")) {

			int key = name.get_slice("/",1).to_int();
			ERR_FAIL_INDEX_V( key,anim->track_get_key_count(track), false );
			String what = name.get_slice("/",2);
			float time = anim->track_get_key_time(track,key);
			float transition = anim->track_get_key_transition(track,key);

			if (what=="time") {
				Variant v = anim->track_get_key_value(track,key);
				anim_editor->_internal_set_key(track,time,transition,v);
				return true;
			}
			if (what=="transition") {
				transition=p_value;
				Variant v = anim->track_get_key_value(track,key);
				anim_editor->_internal_set_key(track,time,transition,v);
				return true;
			}

			switch(anim->track_get_type(track)) {

				case Animation::TYPE_TRANSFORM: {

					Vector3 scale,loc;
					Quat rot;
					anim->transform_track_get_key(track,key,&loc,&rot,&scale);


					if (what=="loc") {
						loc=p_value;
					} else if (what=="scale") {
						scale=p_value;
					} else if (what=="rot") {
						rot=p_value;
					} else {
						return false; //meh
					}

					Dictionary k;
					k["rot"]=rot;
					k["loc"]=loc;
					k["scale"]=scale;
					anim_editor->_internal_set_key(track,time,transition,k);

				} break;
				case Animation::TYPE_METHOD: {

				} break;
				case Animation::TYPE_VALUE: {

					if (what=="value")
						anim_editor->_internal_set_key(track,time,transition,p_value);
				} break;
				default: { return false; }

			}

		} else
			return false;

		return true;
	}

	bool _get(const StringName& p_name,Variant &r_ret) const {

		if (anim.is_null())
			return false;
		String name=p_name;

		if (name=="track/interpolation") {

			r_ret=anim->track_get_interpolation_type(track);

		} else if (name.begins_with("keys/")) {

			int key = name.get_slice("/",1).to_int();
			ERR_FAIL_INDEX_V( key,anim->track_get_key_count(track), Variant() );
			String what = name.get_slice("/",2);

			if (what=="time") {
				r_ret=anim->track_get_key_time(track,key);
				return true;
			}

			if (what=="transition") {
				r_ret=anim->track_get_key_transition(track,key);
				return true;
			}

			switch(anim->track_get_type(track)) {

				case Animation::TYPE_TRANSFORM: {

					Vector3 scale,loc;
					Quat rot;
					anim->transform_track_get_key(track,key,&loc,&rot,&scale);


					if (what=="loc") {
						r_ret= loc;
					} else if (what=="scale") {
						r_ret= scale;
					} else if (what=="rot") {
						r_ret= rot;
					}

				} break;
				case Animation::TYPE_METHOD: {

				} break;
				case Animation::TYPE_VALUE: {

					if (what=="value")
						r_ret= anim->track_get_key_value(track,key);
				} break;
				default: { return false; }

			}
		} else
			return false;

		return true;
	}
	void _get_property_list( List<PropertyInfo> *p_list) const {

		p_list->push_back(PropertyInfo(Variant::INT,"track/interpolation",PROPERTY_HINT_ENUM,"Nearest,Linear,Cubic") );

		if (anim.is_null())
			return;

		int keycount = anim->track_get_key_count(track);

		for(int i=0;i<keycount;i++) {

			p_list->push_back(PropertyInfo(Variant::REAL,"keys/"+itos(i)+"/time",PROPERTY_HINT_RANGE,"0,3600,0.001") );
			p_list->push_back(PropertyInfo(Variant::REAL,"keys/"+itos(i)+"/transition",PROPERTY_HINT_EXP_EASING) );
			switch(anim->track_get_type(track)) {

				case Animation::TYPE_TRANSFORM: {

					p_list->push_back(PropertyInfo(Variant::VECTOR3,"keys/"+itos(i)+"/loc" ) );
					p_list->push_back(PropertyInfo(Variant::QUAT,"keys/"+itos(i)+"/rot" ) );
					p_list->push_back(PropertyInfo(Variant::VECTOR3,"keys/"+itos(i)+"/scale" ) );

				} break;
				case Animation::TYPE_METHOD: {

				} break;
				case Animation::TYPE_VALUE: {


					Variant v = anim->track_get_key_value(track,i);

					PropertyHint hint=PROPERTY_HINT_NONE;
					String hint_string;
					if (v.get_type()==Variant::INT) {
						hint=PROPERTY_HINT_RANGE;
						hint_string="-16384,16384,1";
					} else if (v.get_type()==Variant::REAL) {
						hint=PROPERTY_HINT_RANGE;
						hint_string="-16384,16384,0.001";
					} else if (v.get_type()==Variant::OBJECT) {
						hint=PROPERTY_HINT_RESOURCE_TYPE;
						hint_string="Resource";
					}



					p_list->push_back(PropertyInfo(v.get_type(),"keys/"+itos(i)+"/value",hint,hint_string ) );



				} break;

			}
		}

	}

public:

	AnimationEditor *anim_editor;

	Ref<Animation> anim;
	int track;
	AnimationEditor_TrackEditor() { }

};


void AnimationEditor::update_anim() {

	tracks->clear();
	key_editor->edit(NULL);
	TreeItem *root = tracks->create_item(NULL);

	TreeItem *sel=NULL;
	int selected_track=-1;
	if (animation->has_meta("_anim_editor_selected_track_"))
		selected_track=animation->get_meta("_anim_editor_selected_track_");

	for(int i=0;i<animation->get_track_count();i++) {

		String path = animation->track_get_path(i);
		TreeItem *track = tracks->create_item(root);
		track->set_text(0,itos(i));
		track->set_editable(0,false);
		track->set_text(1,path);
		track->set_editable(1,true);
		track->set_metadata(0,i);
		if (selected_track==i)
			sel=track;

		switch(animation->track_get_type(i)) {

			case Animation::TYPE_TRANSFORM: {

				track->set_icon(0,get_icon("Matrix","EditorIcons"));

			} break;
			case Animation::TYPE_METHOD: {
				track->set_icon(0,get_icon("TrackMethod","EditorIcons"));
			} break;
			case Animation::TYPE_VALUE: {

				track->set_icon(0,get_icon("TrackValue","EditorIcons"));
			} break;


		}

	}

	if (sel) {
		sel->select(1);
		_update_track_keys();
	} else {
		selected_track=-1;
	}
}

void AnimationEditor::_update_track_keys()  {
	if (selected_track<0 || selected_track>=animation->get_track_count())
		return;
	track_editor->anim=animation;
	track_editor->track=selected_track;
	key_editor->edit(NULL);
	key_editor->edit(track_editor);

	if (animation->track_get_type(selected_track)==Animation::TYPE_VALUE) {
		key_time->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,160);
		key_type->show();
	} else {
		key_time->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,60);
		key_type->hide();
	}


}

void AnimationEditor::_track_path_changed() {

	TreeItem *ti=tracks->get_edited();
	int track=ti->get_metadata(0);
	String path=ti->get_text(1);
	if (track<0 || track>=animation->get_track_count())
		return;
	undo_redo->create_action("Create Anim Track");
	undo_redo->add_do_method(animation.ptr(),"track_set_path",track,path);
	undo_redo->add_undo_method(animation.ptr(),"track_set_path",track,animation->track_get_path(track));
	undo_redo->add_do_method(this,"_internal_check_update",animation);
	undo_redo->add_undo_method(this,"_internal_check_update",animation);
	undo_redo->commit_action();

}

void AnimationEditor::_track_selected() {

	TreeItem *ti=tracks->get_selected();
	if(!ti)
		return;
	selected_track=ti->get_metadata(0);
	animation->set_meta("_anim_editor_selected_track_",selected_track);
	_update_track_keys();
}

void AnimationEditor::_track_added() {

	undo_redo->create_action("Create Anim Track");
	undo_redo->add_do_method(animation.ptr(),"add_track",track_type->get_selected(),-1);
	undo_redo->add_undo_method(animation.ptr(),"remove_track",animation->get_track_count());
	undo_redo->add_do_method(this,"_internal_set_selected_track",animation->get_track_count(),animation);
	undo_redo->add_undo_method(this,"_internal_set_selected_track",selected_track,animation);
	undo_redo->add_do_method(this,"_internal_check_update",animation);
	undo_redo->add_undo_method(this,"_internal_check_update",animation);
	undo_redo->commit_action();
	update_anim();
}

void AnimationEditor::_track_removed() {

	if (selected_track<0 || selected_track>=animation->get_track_count())
		return;

	undo_redo->create_action("Remove Anim Track");
	undo_redo->add_do_method(animation.ptr(),"remove_track",selected_track);
	undo_redo->add_undo_method(animation.ptr(),"add_track",animation->track_get_type(selected_track),selected_track);
	undo_redo->add_undo_method(animation.ptr(),"track_set_path",selected_track,animation->track_get_path(selected_track));
	//todo interpolation
	for(int i=0;i<animation->track_get_key_count(selected_track);i++) {

		Variant v = animation->track_get_key_value(selected_track,i);
		float time =  animation->track_get_key_time(selected_track,i);

		undo_redo->add_undo_method(animation.ptr(),"track_insert_key",selected_track,time,v);

	}

	int old_track=selected_track;
	if (selected_track>0)
		selected_track--;
	if (selected_track<0 || selected_track>=(animation->get_track_count()-1))
		selected_track=-1;

	undo_redo->add_do_method(this,"_internal_set_selected_track",selected_track,animation);
	undo_redo->add_undo_method(this,"_internal_set_selected_track",old_track,animation);

	undo_redo->add_do_method(this,"_internal_check_update",animation);
	undo_redo->add_undo_method(this,"_internal_check_update",animation);
	undo_redo->commit_action();

}

void AnimationEditor::_internal_set_interpolation_type(int p_track,Animation::InterpolationType p_type) {

	undo_redo->create_action("Set Interpolation");
	undo_redo->add_do_method(animation.ptr(),"track_set_interpolation_type",p_track,p_type);
	undo_redo->add_undo_method(animation.ptr(),"track_set_interpolation_type",p_track,animation->track_get_interpolation_type(p_track));
	undo_redo->add_do_method(this,"_internal_set_selected_track",selected_track,animation);
	undo_redo->add_undo_method(this,"_internal_set_selected_track",selected_track,animation);
	undo_redo->add_do_method(this,"_internal_check_update",animation);
	undo_redo->add_undo_method(this,"_internal_check_update",animation);
	undo_redo->commit_action();

}

void AnimationEditor::_internal_set_selected_track(int p_which,const Ref<Animation>& p_anim) {

	if (is_visible() && animation==p_anim) {
		selected_track=p_which;
		animation->set_meta("_anim_editor_selected_track_",selected_track);
	}
}

void AnimationEditor::_track_moved_up() {


	if (selected_track<0 || selected_track>=animation->get_track_count())
		return;

	if (selected_track<(animation->get_track_count()-1)) {
		undo_redo->create_action("Move Up Track");
		undo_redo->add_do_method(animation.ptr(),"track_move_up",selected_track);
		undo_redo->add_undo_method(animation.ptr(),"track_move_down",selected_track+1);
		undo_redo->add_do_method(this,"_internal_set_selected_track",selected_track+1,animation);
		undo_redo->add_undo_method(this,"_internal_set_selected_track",selected_track,animation);
		undo_redo->add_do_method(this,"_internal_check_update",animation);
		undo_redo->add_undo_method(this,"_internal_check_update",animation);
		undo_redo->commit_action();
	}
}

void AnimationEditor::_track_moved_down() {



	if (selected_track<0 || selected_track>=animation->get_track_count())
		return;
	if (selected_track>0) {
		undo_redo->create_action("Move Down Track");
		undo_redo->add_do_method(animation.ptr(),"track_move_down",selected_track);
		undo_redo->add_undo_method(animation.ptr(),"track_move_up",selected_track-1);
		undo_redo->add_do_method(this,"_internal_set_selected_track",selected_track-1,animation);
		undo_redo->add_undo_method(this,"_internal_set_selected_track",selected_track,animation);
		undo_redo->add_do_method(this,"_internal_check_update",animation);
		undo_redo->add_undo_method(this,"_internal_check_update",animation);
		undo_redo->commit_action();
	}


}

void AnimationEditor::_key_added() {

	if (selected_track<0 || selected_track>=animation->get_track_count())
		return;

	bool need_variant= animation->track_get_type(selected_track)==Animation::TYPE_VALUE;

	Variant v;

	if (need_variant) {

		switch(key_type->get_selected()) {

			case Variant::NIL: v=Variant(); break;
			case Variant::BOOL: v=false; break;
			case Variant::INT: v=0; break;
			case Variant::REAL: v=0.0; break;
			case Variant::STRING: v=""; break;
			case Variant::VECTOR2: v=Vector2(); break;		// 5
			case Variant::RECT2: v=Rect2(); break;
			case Variant::VECTOR3: v=Vector3(); break;
			case Variant::PLANE: v=Plane(); break;
			case Variant::QUAT: v=Quat(); break;
			case Variant::_AABB: v=AABB(); break; //sorry naming convention fail :( not like it's used often // 10
			case Variant::MATRIX3: v=Matrix3(); break;
			case Variant::TRANSFORM: v=Transform(); break;
			case Variant::COLOR: v=Color(); break;
			case Variant::IMAGE: v=Image(); break;
			case Variant::NODE_PATH: v=NodePath(); break;		// 15
			case Variant::_RID: v=RID(); break;
			case Variant::OBJECT: v=Variant(); break;
			case Variant::INPUT_EVENT: v=InputEvent(); break;
			case Variant::DICTIONARY: v=Dictionary(); break;		// 20
			case Variant::ARRAY: v=Array(); break;
			case Variant::RAW_ARRAY: v=DVector<uint8_t>(); break;
			case Variant::INT_ARRAY: v=DVector<int>(); break;
			case Variant::REAL_ARRAY: v=DVector<real_t>(); break;
			case Variant::STRING_ARRAY: v=DVector<String>(); break;	//25
			case Variant::VECTOR3_ARRAY: v=DVector<Vector3>(); break;
			case Variant::COLOR_ARRAY: v=DVector<Color>(); break;
			default: v=Variant(); break;
		}
	}

	float time = key_time->get_text().to_double();

	switch(animation->track_get_type(selected_track)) {
		case Animation::TYPE_TRANSFORM: {

			Dictionary d;
			d["loc"]=Vector3();
			d["rot"]=Quat();
			d["scale"]=Vector3(1,1,1);
			v=d;

		} break;
		case Animation::TYPE_VALUE: {
			//v=v
		} break;
		case Animation::TYPE_METHOD: {

			return; //not do anything yet
		} break;
	}

	_internal_set_key(selected_track,time,1.0,v);

	_update_track_keys();
}



void AnimationEditor::_internal_check_update(Ref<Animation> p_anim) {

	if (is_visible() && p_anim==animation)	{
		update_anim();
	}
}

void AnimationEditor::_internal_set_key(int p_track, float p_time, float p_transition,const Variant& p_value) {

	int prev = animation->track_find_key(p_track,p_time);
	bool existing = (prev>=0) && (animation->track_get_key_time(p_track,prev)==p_time);

	undo_redo->create_action("Insert Key");

	undo_redo->add_do_method(animation.ptr(),"track_insert_key",p_track,p_time,p_value,p_transition);
	if (existing)
		undo_redo->add_undo_method(animation.ptr(),"track_insert_key",p_track,p_time,animation->track_get_key_value(p_track,existing),animation->track_get_key_transition(p_track,existing));
	else
		undo_redo->add_undo_method(animation.ptr(),"track_remove_key",p_track,prev+1);
	undo_redo->add_do_method(this,"_internal_check_update",animation);
	undo_redo->add_undo_method(this,"_internal_check_update",animation);

	undo_redo->commit_action();
}

void AnimationEditor::_key_removed() {

	if (selected_track<0 || selected_track>=animation->get_track_count())
		return;

	String sel=key_editor->get_selected_path();
	if (sel.get_slice_count("/")<2)
		return;
	if (sel.get_slice("/",0)!="keys")
		return;
	int key = sel.get_slice("/",1).to_int();
	if (key<0 || key>=animation->track_get_key_count(selected_track))
		return;


	undo_redo->create_action("Remove Key");

	Variant data = animation->track_get_key_value(selected_track,key);
	float time = animation->track_get_key_time(selected_track,key);
	undo_redo->add_do_method(animation.ptr(),"track_remove_key",selected_track,key);
	undo_redo->add_undo_method(animation.ptr(),"track_insert_key",selected_track,time,data);
	undo_redo->add_do_method(this,"_internal_check_update",animation);
	undo_redo->add_undo_method(this,"_internal_check_update",animation);
	undo_redo->commit_action();

	_update_track_keys();
}



void AnimationEditor::edit(const Ref<Animation>& p_animation) {


	animation=p_animation;
	if (!animation.is_null())
		update_anim();


}



void AnimationEditor::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_track_selected"),&AnimationEditor::_track_selected);
	ObjectTypeDB::bind_method(_MD("_track_added"),&AnimationEditor::_track_added);
	ObjectTypeDB::bind_method(_MD("_track_removed"),&AnimationEditor::_track_removed);
	ObjectTypeDB::bind_method(_MD("_track_moved_up"),&AnimationEditor::_track_moved_up);
	ObjectTypeDB::bind_method(_MD("_track_moved_down"),&AnimationEditor::_track_moved_down);
	ObjectTypeDB::bind_method(_MD("_track_path_changed"),&AnimationEditor::_track_path_changed);
	ObjectTypeDB::bind_method(_MD("_key_added"),&AnimationEditor::_key_added);
	ObjectTypeDB::bind_method(_MD("_key_removed"),&AnimationEditor::_key_removed);
	ObjectTypeDB::bind_method(_MD("_internal_check_update"),&AnimationEditor::_internal_check_update);
	ObjectTypeDB::bind_method(_MD("_internal_set_selected_track"),&AnimationEditor::_internal_set_selected_track);

}

void AnimationEditor::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_SCENE: {

			add_track->set_icon( get_icon("Add","EditorIcons") );
			remove_track->set_icon( get_icon("Del","EditorIcons") );
			move_up_track->set_icon( get_icon("Up","EditorIcons") );
			move_down_track->set_icon( get_icon("Down","EditorIcons") );
			time_icon->set_texture( get_icon("Time","EditorIcons") );

			add_key->set_icon( get_icon("Add","EditorIcons") );
			remove_key->set_icon( get_icon("Del","EditorIcons") );

		} break;
	}
}

AnimationEditor::AnimationEditor() {

	panel = memnew( Panel );
	add_child(panel);
	panel->set_area_as_parent_rect();

	Control *left_pane = memnew( Control );
	panel->add_child(left_pane);
	left_pane->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_RATIO,0.5);
	left_pane->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,0);

	Label *l = memnew( Label );
	l->set_text("Track List:");
	l->set_pos(Point2(5,5));
	left_pane->add_child(l);

	/*
	track_name = memnew( LineEdit );
	track_name->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,10);
	track_name->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	track_name->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,80);
	track_name->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	left_pane->add_child(track_name);
*/

	track_type = memnew( OptionButton );
	track_type->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,10);
	track_type->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	track_type->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,115);
	track_type->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	left_pane->add_child(track_type);
	track_type->add_item("Transform",Animation::TYPE_TRANSFORM);
	track_type->add_item("Value",Animation::TYPE_VALUE);
	track_type->add_item("Method",Animation::TYPE_METHOD);


	add_track = memnew( Button );
	add_track->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,110);
	add_track->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	add_track->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,90);
	add_track->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	left_pane->add_child(add_track);

	remove_track = memnew( Button );
	remove_track->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,85);
	remove_track->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	remove_track->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,60);
	remove_track->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	left_pane->add_child(remove_track);

	move_up_track = memnew( Button );
	move_up_track->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,55);
	move_up_track->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	move_up_track->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,30);
	move_up_track->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	left_pane->add_child(move_up_track);

	move_down_track = memnew( Button );
	move_down_track->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,25);
	move_down_track->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	move_down_track->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,0);
	move_down_track->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	left_pane->add_child(move_down_track);

	tracks = memnew(Tree);
	tracks->set_columns(2);
	tracks->set_column_expand(0,false);
	tracks->set_column_min_width(0,55);
	tracks->set_column_expand(1,true);
	tracks->set_column_min_width(1,100);
	tracks->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,55);
	tracks->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,0);
	tracks->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,10);
	tracks->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,5);
	tracks->set_hide_root(true);
	left_pane->add_child(tracks);


	Control *right_pane = memnew( Control );
	panel->add_child(right_pane);
	right_pane->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_RATIO,0.5);
	right_pane->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,5);
	right_pane->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,0);

	l = memnew( Label );
	l->set_text("Track Keys:");
	l->set_pos(Point2(5,5));
	right_pane->add_child(l);

	time_icon = memnew(  TextureFrame );
	time_icon->set_pos(Point2(8,28));
	right_pane->add_child(time_icon);

	key_time = memnew( LineEdit );
	key_time->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,24);
	key_time->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,160);
	key_time->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	key_time->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	key_time->set_text("0.0");
	right_pane->add_child(key_time);

	key_type = memnew( OptionButton );
	key_type->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,160);
	key_type->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,60);
	key_type->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	key_type->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	right_pane->add_child(key_type);

	for(int i=0;i<Variant::VARIANT_MAX;i++) {

		key_type->add_item(Variant::get_type_name(Variant::Type(i)));
	}

	add_key = memnew( Button );
	add_key->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,55);
	add_key->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,30);
	add_key->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	add_key->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	right_pane->add_child(add_key);

	remove_key = memnew( Button );
	remove_key->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_END,25);
	remove_key->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,0);
	remove_key->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,25);
	remove_key->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_BEGIN,45);
	right_pane->add_child(remove_key);

	key_editor = memnew(PropertyEditor);
	key_editor->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,55);
	key_editor->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,0);
	key_editor->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,10);
	key_editor->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,5);
	key_editor->hide_top_label();

	right_pane->add_child(key_editor);

	track_editor = memnew( AnimationEditor_TrackEditor );
	track_editor->anim_editor=this;
	selected_track=-1;

	add_track->connect("pressed", this,"_track_added");
	remove_track->connect("pressed", this,"_track_removed");
	move_up_track->connect("pressed", this,"_track_moved_up");
	move_down_track->connect("pressed", this,"_track_moved_down");
	tracks->connect("cell_selected", this,"_track_selected");
	tracks->connect("item_edited", this,"_track_path_changed");
	add_key->connect("pressed", this,"_key_added");
	remove_key->connect("pressed", this,"_key_removed");
}

AnimationEditor::~AnimationEditor() {

	memdelete(track_editor);
}

void AnimationEditorPlugin::edit(Object *p_node) {

	animation_editor->set_undo_redo(&get_undo_redo());
	if (p_node && p_node->cast_to<Animation>()) {
		animation_editor->edit( p_node->cast_to<Animation>() );
		animation_editor->show();
	} else
		animation_editor->hide();
}

bool AnimationEditorPlugin::handles(Object *p_node) const{

	return p_node->is_type("Animation");
}

void AnimationEditorPlugin::make_visible(bool p_visible){

	if (p_visible)
		animation_editor->show();
	else
		animation_editor->hide();
}

AnimationEditorPlugin::AnimationEditorPlugin(EditorNode *p_node) {

	animation_editor = memnew( AnimationEditor );

	p_node->get_viewport()->add_child(animation_editor);
	animation_editor->set_area_as_parent_rect();
	animation_editor->hide();




}


