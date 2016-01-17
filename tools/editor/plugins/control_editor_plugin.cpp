/*************************************************************************/
/*  control_editor_plugin.cpp                                            */
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
#if 0
#include "control_editor_plugin.h"
#include "print_string.h"
#include "editor_node.h"
#include "os/keyboard.h"
#include "scene/main/viewport.h"

void ControlEditor::_add_control(Control *p_control,const EditInfo& p_info) {

	if (controls.has(p_control))
		return;

	controls.insert(p_control,p_info);
	p_control->call_deferred("connect","visibility_changed",this,"_visibility_changed",varray(p_control->get_instance_ID()));
}

void ControlEditor::_remove_control(Control *p_control) {

	p_control->call_deferred("disconnect","visibility_changed",this,"_visibility_changed");
	controls.erase(p_control);
}
void ControlEditor::_clear_controls(){

	while(controls.size())
		_remove_control(controls.front()->key());
}

void ControlEditor::_visibility_changed(ObjectID p_control) {

	Object *c = ObjectDB::get_instance(p_control);
	if (!c)
		return;
	Control *ct = c->cast_to<Control>();
	if (!ct)
		return;

	_remove_control(ct);
}


void ControlEditor::_node_removed(Node *p_node) {

	Control *control = (Control*)p_node; //not a good cast, but safe
	if (controls.has(control))
		_remove_control(control);

	if (current_window==p_node) {
		_clear_controls();
	}
	update();
}

// slow as hell
Control* ControlEditor::_select_control_at_pos(const Point2& p_pos,Node* p_node) {
	
	for (int i=p_node->get_child_count()-1;i>=0;i--) {
	
		Control *r=_select_control_at_pos(p_pos,p_node->get_child(i));
		if (r)
			return r;
	}
	
	Control *c=p_node->cast_to<Control>();
	
	if (c) {
		Rect2 rect = c->get_window_rect();
		if (c->get_window()==current_window) {
			rect.pos=transform.xform(rect.pos).floor();
		}
		if (rect.has_point(p_pos))
			return c;
	}

	return NULL;
}


void ControlEditor::_key_move(const Vector2& p_dir, bool p_snap) {

	if (drag!=DRAG_NONE)
		return;

	Vector2 motion=p_dir;
	if (p_snap)
		motion*=snap_val->get_text().to_double();

	undo_redo->create_action("Edit Control");
	for(ControlMap::Element *E=controls.front();E;E=E->next()) {
		Control *control = E->key();
		undo_redo->add_do_method(control,"set_pos",control->get_pos()+motion);
		undo_redo->add_undo_method(control,"set_pos",control->get_pos());
	}
	undo_redo->commit_action();
}


void ControlEditor::_input_event(InputEvent p_event) {
	
	if (p_event.type==InputEvent::MOUSE_BUTTON) {
		
		const InputEventMouseButton &b=p_event.mouse_button;

		if (b.button_index==BUTTON_RIGHT) {

			if (controls.size() && drag!=DRAG_NONE) {
				//cancel drag
				for(ControlMap::Element *E=controls.front();E;E=E->next()) {
					Control *control = E->key();
					control->set_pos(E->get().drag_pos);
					control->set_size(E->get().drag_size);
				}

			} else if (b.pressed) {
				popup->set_pos(Point2(b.x,b.y));
				popup->popup();
			}
			return;
		}
		//if (!controls.size())
		//	return;
		
		if (b.button_index!=BUTTON_LEFT)
			return;

		if (!b.pressed) {
			
			if (drag!=DRAG_NONE) {

				if (undo_redo) {

					undo_redo->create_action("Edit Control");
					for(ControlMap::Element *E=controls.front();E;E=E->next()) {
						Control *control = E->key();
						undo_redo->add_do_method(control,"set_pos",control->get_pos());
						undo_redo->add_do_method(control,"set_size",control->get_size());
						undo_redo->add_undo_method(control,"set_pos",E->get().drag_pos);
						undo_redo->add_undo_method(control,"set_size",E->get().drag_size);
					}
					undo_redo->commit_action();
				}

				drag=DRAG_NONE;

			}
			return;
		}


		if (controls.size()==1) {
			//try single control edit
			Control *control = controls.front()->key();
			ERR_FAIL_COND(!current_window);

			Rect2 rect=control->get_window_rect();
			Point2 ofs=Point2();//get_global_pos();
			Rect2 draw_rect=Rect2(rect.pos-ofs,rect.size);
			Point2 click=Point2(b.x,b.y);
			click = transform.affine_inverse().xform(click);
			Size2 handle_size=Size2(handle_len,handle_len);

			drag = DRAG_NONE;

			if (Rect2(draw_rect.pos-handle_size,handle_size).has_point(click))
				drag=DRAG_TOP_LEFT;
			else if (Rect2(draw_rect.pos+draw_rect.size,handle_size).has_point(click))
				drag=DRAG_BOTTOM_RIGHT;
			else if(Rect2(draw_rect.pos+Point2(draw_rect.size.width,-handle_size.y),handle_size).has_point(click))
				drag=DRAG_TOP_RIGHT;
			else if (Rect2(draw_rect.pos+Point2(-handle_size.x,draw_rect.size.height),handle_size).has_point(click))
				drag=DRAG_BOTTOM_LEFT;
			else if (Rect2(draw_rect.pos+Point2(Math::floor((draw_rect.size.width-handle_size.x)/2.0),-handle_size.height),handle_size).has_point(click))
				drag=DRAG_TOP;
			else if( Rect2(draw_rect.pos+Point2(-handle_size.width,Math::floor((draw_rect.size.height-handle_size.y)/2.0)),handle_size).has_point(click))
				drag=DRAG_LEFT;
			else if ( Rect2(draw_rect.pos+Point2(Math::floor((draw_rect.size.width-handle_size.x)/2.0),draw_rect.size.height),handle_size).has_point(click))
				drag=DRAG_BOTTOM;
			else if( Rect2(draw_rect.pos+Point2(draw_rect.size.width,Math::floor((draw_rect.size.height-handle_size.y)/2.0)),handle_size).has_point(click))
				drag=DRAG_RIGHT;

			if (drag!=DRAG_NONE) {
				drag_from=click;
				controls[control].drag_pos=control->get_pos();
				controls[control].drag_size=control->get_size();
				controls[control].drag_limit=drag_from+controls[control].drag_size-control->get_minimum_size();
				return;
			}


		}

		//multi control edit

		Point2 click=Point2(b.x,b.y);
		Node* scene = get_scene()->get_root_node()->cast_to<EditorNode>()->get_edited_scene();
		if (!scene)
			return;
		/*
		if (current_window) {
			//no window.... ?
			click-=current_window->get_scroll();
		}*/
		Control *c=_select_control_at_pos(click, scene);

		Node* n = c;
		while ((n && n != scene && n->get_owner() != scene) || (n && !n->is_type("Control"))) {
			n = n->get_parent();
		};
		c = n->cast_to<Control>();


		if (b.mod.control) { //additive selection

			if (!c)
				return; //nothing to add

			if (current_window && controls.size() && c->get_window()!=current_window)
				return; //cant multiple select from multiple windows

			if (!controls.size())
				current_window=c->get_window();

			if (controls.has(c)) {
				//already in here, erase it
				_remove_control(c);
				update();
				return;
			}

			//check parents!
			Control *parent = c->get_parent()->cast_to<Control>();

			while(parent) {

				if (controls.has(parent))
					return; //a parent is already selected, so this is pointless
				parent=parent->get_parent()->cast_to<Control>();
			}

			//check childrens of everything!
			List<Control*> to_erase;

			for(ControlMap::Element *E=controls.front();E;E=E->next()) {
				parent = E->key()->get_parent()->cast_to<Control>();
				while(parent) {
					if (parent==c) {
						to_erase.push_back(E->key());
						break;
					}
					parent=parent->get_parent()->cast_to<Control>();
				}
			}

			while(to_erase.size()) {
				_remove_control(to_erase.front()->get());
				to_erase.pop_front();
			}

			_add_control(c,EditInfo());
			update();
		} else {
			//regular selection
			if (!c) {
				_clear_controls();
				update();
				return;
			}

			if (!controls.has(c)) {
				_clear_controls();
				current_window=c->get_window();
				_add_control(c,EditInfo());
				//reselect
				if (get_scene()->is_editor_hint()) {
					get_scene()->get_root_node()->call("edit_node",c);
				}

			}



			for(ControlMap::Element *E=controls.front();E;E=E->next()) {

				EditInfo &ei=E->get();
				Control *control=E->key();
				ei.drag_pos=control->get_pos();
				ei.drag_size=control->get_size();
				ei.drag_limit=drag_from+ei.drag_size-control->get_minimum_size();
			}

			drag=DRAG_ALL;
			drag_from=click;
			update();
		}

	}
	
	if (p_event.type==InputEvent::MOUSE_MOTION) {
		
		const InputEventMouseMotion &m=p_event.mouse_motion;
		
		if (drag==DRAG_NONE || !current_window)
			return;

		for(ControlMap::Element *E=controls.front();E;E=E->next()) {

			Control *control = E->key();
			Point2 control_drag_pos=E->get().drag_pos;
			Point2 control_drag_size=E->get().drag_size;
			Point2 control_drag_limit=E->get().drag_limit;

			Point2 pos=Point2(m.x,m.y);
			pos = transform.affine_inverse().xform(pos);

			switch(drag) {
				case DRAG_ALL: {

					control->set_pos( snapify(control_drag_pos+(pos-drag_from)) );
				} break;
				case DRAG_RIGHT: {

					control->set_size( snapify(Size2(control_drag_size.width+(pos-drag_from).x,control_drag_size.height)) );
				} break;
				case DRAG_BOTTOM: {

					control->set_size( snapify(Size2(control_drag_size.width,control_drag_size.height+(pos-drag_from).y)) );
				} break;
				case DRAG_BOTTOM_RIGHT: {

					control->set_size( snapify(control_drag_size+(pos-drag_from)) );
				} break;
				case DRAG_TOP_LEFT: {

					if(pos.x>control_drag_limit.x)
						pos.x=control_drag_limit.x;
					if(pos.y>control_drag_limit.y)
						pos.y=control_drag_limit.y;

					Point2 old_size = control->get_size();
					Point2 new_pos = snapify(control_drag_pos+(pos-drag_from));
					Point2 new_size = old_size + (control->get_pos() - new_pos);

					control->set_pos( new_pos );
					control->set_size( new_size );
				} break;
				case DRAG_TOP: {

					if(pos.y>control_drag_limit.y)
						pos.y=control_drag_limit.y;

					Point2 old_size = control->get_size();
					Point2 new_pos = snapify(control_drag_pos+Point2(0,pos.y-drag_from.y));
					Point2 new_size = old_size + (control->get_pos() - new_pos);

					control->set_pos( new_pos );
					control->set_size( new_size );
				} break;
				case DRAG_LEFT: {

					if(pos.x>control_drag_limit.x)
						pos.x=control_drag_limit.x;

					Point2 old_size = control->get_size();
					Point2 new_pos = snapify(control_drag_pos+Point2(pos.x-drag_from.x,0));
					Point2 new_size = old_size + (control->get_pos() - new_pos);

					control->set_pos( new_pos );
					control->set_size( new_size );

				} break;
				case DRAG_TOP_RIGHT: {

					if(pos.y>control_drag_limit.y)
						pos.y=control_drag_limit.y;

					Point2 old_size = control->get_size();
					Point2 new_pos = snapify(control_drag_pos+Point2(0,pos.y-drag_from.y));

					float new_size_y = Point2( old_size + (control->get_pos() - new_pos)).y;
					float new_size_x = snapify(control_drag_size+Point2(pos.x-drag_from.x,0)).x;

					control->set_pos( new_pos );
					control->set_size( Point2(new_size_x, new_size_y) );
				} break;
				case DRAG_BOTTOM_LEFT: {

					if(pos.x>control_drag_limit.x)
						pos.x=control_drag_limit.x;

					Point2 old_size = control->get_size();
					Point2 new_pos = snapify(control_drag_pos+Point2(pos.x-drag_from.x,0));

					float new_size_y = snapify(control_drag_size+Point2(0,pos.y-drag_from.y)).y;
					float new_size_x = Point2( old_size + (control->get_pos() - new_pos)).x;

					control->set_pos( new_pos );
					control->set_size( Point2(new_size_x, new_size_y) );


				} break;

			default:{}
			}
		}
	}

	if (p_event.type==InputEvent::KEY) {

		const InputEventKey &k=p_event.key;

		if (k.pressed) {

			if (k.scancode==KEY_UP)
				_key_move(Vector2(0,-1),k.mod.shift);
			else if (k.scancode==KEY_DOWN)
				_key_move(Vector2(0,1),k.mod.shift);
			else if (k.scancode==KEY_LEFT)
				_key_move(Vector2(-1,0),k.mod.shift);
			else if (k.scancode==KEY_RIGHT)
				_key_move(Vector2(1,0),k.mod.shift);
		}

	}

		
}


bool ControlEditor::get_remove_list(List<Node*> *p_list) {

	for(ControlMap::Element *E=controls.front();E;E=E->next()) {

		p_list->push_back(E->key());
	}

	return !p_list->empty();
}

void ControlEditor::_update_scroll(float) {

	if (updating_scroll)
		return;

	if (!current_window)
		return;

	Point2 ofs;
	ofs.x=h_scroll->get_val();
	ofs.y=v_scroll->get_val();

//	current_window->set_scroll(-ofs);

	transform=Matrix32();

	transform.scale_basis(Size2(zoom,zoom));
	transform.elements[2]=-ofs*zoom;


	RID viewport = editor->get_scene_root()->get_viewport();

	VisualServer::get_singleton()->viewport_set_global_canvas_transform(viewport,transform);

	update();

}

void ControlEditor::_notification(int p_what) {
	
	if (p_what==NOTIFICATION_PROCESS) {
		
		for(ControlMap::Element *E=controls.front();E;E=E->next()) {

			Control *control = E->key();
			Rect2 r=control->get_window_rect();
			if (r != E->get().last_rect ) {
				update();
				E->get().last_rect=r;
			}
		}
		
	}
	
	if (p_what==NOTIFICATION_CHILDREN_CONFIGURED) {
		
		get_scene()->connect("node_removed",this,"_node_removed");
	}
	
	if (p_what==NOTIFICATION_DRAW) {
		
		// TODO fetch the viewport?
		/*
		if (!control) {
			h_scroll->hide();
			v_scroll->hide();
			return;
		}
		*/
		_update_scrollbars();

		if (!current_window)
			return;

		for(ControlMap::Element *E=controls.front();E;E=E->next()) {

			Control *control = E->key();

			Rect2 rect=control->get_window_rect();
			RID ci=get_canvas_item();
			VisualServer::get_singleton()->canvas_item_set_clip(ci,true);
			Point2 ofs=Point2();//get_global_pos();
			Rect2 draw_rect=Rect2(rect.pos-ofs,rect.size);
			draw_rect.pos = transform.xform(draw_rect.pos);
			Color light_edit_color=Color(1.0,0.8,0.8);
			Color dark_edit_color=Color(0.4,0.1,0.1);
			Size2 handle_size=Size2(handle_len,handle_len);

#define DRAW_RECT( m_rect, m_color )\
VisualServer::get_singleton()->canvas_item_add_rect(ci,m_rect,m_color);

#define DRAW_EMPTY_RECT( m_rect, m_color )\
	DRAW_RECT( Rect2(m_rect.pos,Size2(m_rect.size.width,1)), m_color );\
	DRAW_RECT(Rect2(Point2(m_rect.pos.x,m_rect.pos.y+m_rect.size.height-1),Size2(m_rect.size.width,1)), m_color);\
	DRAW_RECT(Rect2(m_rect.pos,Size2(1,m_rect.size.height)), m_color);\
	DRAW_RECT(Rect2(Point2(m_rect.pos.x+m_rect.size.width-1,m_rect.pos.y),Size2(1,m_rect.size.height)), m_color);

#define DRAW_BORDER_RECT( m_rect, m_border_color,m_color )\
	DRAW_RECT( m_rect, m_color );\
	DRAW_EMPTY_RECT( m_rect, m_border_color );

			DRAW_EMPTY_RECT( draw_rect.grow(2), light_edit_color );
			DRAW_EMPTY_RECT( draw_rect.grow(1), dark_edit_color );

			if (controls.size()==1) {
				DRAW_BORDER_RECT( Rect2(draw_rect.pos-handle_size,handle_size), light_edit_color,dark_edit_color );
				DRAW_BORDER_RECT( Rect2(draw_rect.pos+draw_rect.size,handle_size), light_edit_color,dark_edit_color );
				DRAW_BORDER_RECT( Rect2(draw_rect.pos+Point2(draw_rect.size.width,-handle_size.y),handle_size), light_edit_color,dark_edit_color );
				DRAW_BORDER_RECT( Rect2(draw_rect.pos+Point2(-handle_size.x,draw_rect.size.height),handle_size), light_edit_color,dark_edit_color );

				DRAW_BORDER_RECT( Rect2(draw_rect.pos+Point2(Math::floor((draw_rect.size.width-handle_size.x)/2.0),-handle_size.height),handle_size), light_edit_color,dark_edit_color );
				DRAW_BORDER_RECT( Rect2(draw_rect.pos+Point2(-handle_size.width,Math::floor((draw_rect.size.height-handle_size.y)/2.0)),handle_size), light_edit_color,dark_edit_color );
				DRAW_BORDER_RECT( Rect2(draw_rect.pos+Point2(Math::floor((draw_rect.size.width-handle_size.x)/2.0),draw_rect.size.height),handle_size), light_edit_color,dark_edit_color );
				DRAW_BORDER_RECT( Rect2(draw_rect.pos+Point2(draw_rect.size.width,Math::floor((draw_rect.size.height-handle_size.y)/2.0)),handle_size), light_edit_color,dark_edit_color );
			}

			//DRAW_EMPTY_RECT( Rect2( current_window->get_scroll()-Point2(1,1), get_size()+Size2(2,2)), Color(0.8,0.8,1.0,0.8) );
			E->get().last_rect = rect;
		}
	}	
}

void ControlEditor::edit(Control *p_control) {
	
	drag=DRAG_NONE;

	_clear_controls();
	_add_control(p_control,EditInfo());
	current_window=p_control->get_window();
	update();

}


void ControlEditor::_find_controls_span(Node *p_node, Rect2& r_rect) {

	if (!editor->get_scene())
		return;

	if (p_node!=editor->get_edited_scene() && p_node->get_owner()!=editor->get_edited_scene())
		return;

	if (p_node->cast_to<Control>()) {
		Control *c = p_node->cast_to<Control>();
		if (c->get_viewport() != editor->get_viewport()->get_viewport())
			return; //bye, it's in another viewport

		if (!c->get_parent_control()) {

			Rect2 span = c->get_subtree_span_rect();
			r_rect.merge(span);
		}
	}

	for(int i=0;i<p_node->get_child_count();i++) {

		_find_controls_span(p_node->get_child(i),r_rect);
	}
}

void ControlEditor::_update_scrollbars() {


	if (!editor->get_scene()) {
		h_scroll->hide();
		v_scroll->hide();
		return;
	}

	updating_scroll=true;


	Size2 size = get_size();
	Size2 hmin = h_scroll->get_minimum_size();
	Size2 vmin = v_scroll->get_minimum_size();

	v_scroll->set_begin( Point2(size.width - vmin.width, 0) );
	v_scroll->set_end( Point2(size.width, size.height) );

	h_scroll->set_begin( Point2( 0, size.height - hmin.height) );
	h_scroll->set_end( Point2(size.width-vmin.width, size.height) );


	Rect2 local_rect = Rect2(Point2(),get_size()-Size2(vmin.width,hmin.height));

	Rect2 control_rect=local_rect;
	if (editor->get_edited_scene())
		_find_controls_span(editor->get_edited_scene(),control_rect);
	control_rect.pos*=zoom;
	control_rect.size*=zoom;

	/*
	for(ControlMap::Element *E=controls.front();E;E=E->next()) {

		Control *control = E->key();
		Rect2 r = control->get_window()->get_subtree_span_rect();
		if (E==controls.front()) {
			control_rect = r.merge(local_rect);
		} else {
			control_rect = control_rect.merge(r);
		}
	}

	*/
	Point2 ofs;


	if (control_rect.size.height <= local_rect.size.height) {

		v_scroll->hide();
		ofs.y=0;
	} else {

		v_scroll->show();
		v_scroll->set_min(control_rect.pos.y);
		v_scroll->set_max(control_rect.pos.y+control_rect.size.y);
		v_scroll->set_page(local_rect.size.y);
		ofs.y=-v_scroll->get_val();
	}

	if (control_rect.size.width <= local_rect.size.width) {

		h_scroll->hide();
		ofs.x=0;
	} else {

		h_scroll->show();
		h_scroll->set_min(control_rect.pos.x);
		h_scroll->set_max(control_rect.pos.x+control_rect.size.x);
		h_scroll->set_page(local_rect.size.x);
		ofs.x=-h_scroll->get_val();
	}

//	transform=Matrix32();
	transform.elements[2]=ofs*zoom;
	RID viewport = editor->get_scene_root()->get_viewport();
	VisualServer::get_singleton()->viewport_set_global_canvas_transform(viewport,transform);

//	transform.scale_basis(Vector2(zoom,zoom));
	updating_scroll=false;

}


Point2i ControlEditor::snapify(const Point2i& p_pos) const {

	bool active=popup->is_item_checked(0);
	int snap = snap_val->get_text().to_int();

	if (!active || snap<1)
		return p_pos;

	Point2i pos=p_pos;
	pos.x-=pos.x%snap;
	pos.y-=pos.y%snap;
	return pos;


}
void ControlEditor::_popup_callback(int p_op) {

	switch(p_op) {

		case SNAP_USE: {

			popup->set_item_checked(0,!popup->is_item_checked(0));
		} break;
		case SNAP_CONFIGURE: {
			snap_dialog->popup_centered(Size2(200,85));
		} break;
	}
}

void ControlEditor::_bind_methods() {
	
	ObjectTypeDB::bind_method("_input_event",&ControlEditor::_input_event);
	ObjectTypeDB::bind_method("_node_removed",&ControlEditor::_node_removed);		
	ObjectTypeDB::bind_method("_update_scroll",&ControlEditor::_update_scroll);
	ObjectTypeDB::bind_method("_popup_callback",&ControlEditor::_popup_callback);
	ObjectTypeDB::bind_method("_visibility_changed",&ControlEditor::_visibility_changed);
}

ControlEditor::ControlEditor(EditorNode *p_editor) {

	editor=p_editor;
	h_scroll = memnew( HScrollBar );
	v_scroll = memnew( VScrollBar );

	add_child(h_scroll);
	add_child(v_scroll);
	h_scroll->connect("value_changed", this,"_update_scroll",Vector<Variant>(),true);
	v_scroll->connect("value_changed", this,"_update_scroll",Vector<Variant>(),true);


	updating_scroll=false;
	set_focus_mode(FOCUS_ALL);
	handle_len=10;

	popup=memnew( PopupMenu );
	popup->add_check_item("Use Snap");
	popup->add_item("Configure Snap..");
	add_child(popup);

	snap_dialog = memnew( ConfirmationDialog );
	snap_dialog->get_ok()->hide();
	snap_dialog->get_cancel()->set_text("Close");
	add_child(snap_dialog);

	Label *l = memnew(Label);
	l->set_text("Snap:");
	l->set_pos(Point2(5,5));
	snap_dialog->add_child(l);

	snap_val=memnew(LineEdit);
	snap_val->set_text("5");
	snap_val->set_anchor(MARGIN_RIGHT,ANCHOR_END);
	snap_val->set_begin(Point2(15,25));
	snap_val->set_end(Point2(10,25));
	snap_dialog->add_child(snap_val);

	popup->connect("item_pressed", this,"_popup_callback");
	current_window=NULL;

	zoom=0.5;
}


void ControlEditorPlugin::edit(Object *p_object) {
	
	control_editor->set_undo_redo(&get_undo_redo());
	control_editor->edit(p_object->cast_to<Control>());
}

bool ControlEditorPlugin::handles(Object *p_object) const {
	
	return p_object->is_type("Control");
}

void ControlEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		control_editor->show();
		control_editor->set_process(true);
	} else {
	
		control_editor->hide();
		control_editor->set_process(false);
	}

}

ControlEditorPlugin::ControlEditorPlugin(EditorNode *p_node) {
	
	editor=p_node;
	control_editor = memnew( ControlEditor(editor) );
	editor->get_viewport()->add_child(control_editor);
	control_editor->set_area_as_parent_rect();
	control_editor->hide();

	

}


ControlEditorPlugin::~ControlEditorPlugin()
{
}


#endif
