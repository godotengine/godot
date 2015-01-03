/*************************************************************************/
/*  control.cpp                                                          */
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
#include "control.h"
#include "servers/visual_server.h"
#include "scene/main/viewport.h"
#include "scene/main/canvas_layer.h"
#include "globals.h"

#include "print_string.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "message_queue.h"
#include "scene/scene_string_names.h"
#include "scene/gui/panel.h"
#include "scene/gui/label.h"
#include <stdio.h>


class TooltipPanel : public Panel {

	OBJ_TYPE(TooltipPanel,Panel)
public:
	TooltipPanel() {};

};

class TooltipLabel : public Label {

	OBJ_TYPE(TooltipLabel,Label)
public:
	TooltipLabel() {};

};

Control::Window::Window() {


	mouse_focus=NULL;
	mouse_focus_button=-1;
	key_focus=NULL;
	mouse_over=NULL;
	disable_input=false;

	cancelled_input_ID=0;
	tooltip=NULL;
	tooltip_popup=NULL;
	tooltip_label=NULL;
	subwindow_order_dirty=false;
}


Variant Control::edit_get_state() const {

	return get_rect();

}
void Control::edit_set_state(const Variant& p_state) {

	Rect2 state=p_state;
	set_pos(state.pos);
	set_size(state.size);
}

void Control::set_custom_minimum_size(const Size2& p_custom) {

	if (p_custom==data.custom_minimum_size)
		return;
	data.custom_minimum_size=p_custom;
	minimum_size_changed();
}

Size2 Control::get_custom_minimum_size() const{

	return data.custom_minimum_size;
}

Size2 Control::get_combined_minimum_size() const {

	Size2 minsize = get_minimum_size();
	minsize.x = MAX(minsize.x,data.custom_minimum_size.x);
	minsize.y = MAX(minsize.y,data.custom_minimum_size.y);
	return minsize;
}

Size2 Control::edit_get_minimum_size() const {

	return get_combined_minimum_size();
}

void Control::edit_set_rect(const Rect2& p_edit_rect) {


	Rect2 new_rect=get_rect();

	new_rect.pos+=p_edit_rect.pos.snapped(Vector2(1,1));
	new_rect.size=p_edit_rect.size.snapped(Vector2(1,1));

	set_pos(new_rect.pos);
	set_size(new_rect.size);

}

bool Control::_set(const StringName& p_name, const Variant& p_value) {


	String name= p_name;
	if (!name.begins_with("custom"))
		return false;

	if (p_value.get_type()==Variant::NIL) {

		if (name.begins_with("custom_icons/")) {
			String dname = name.get_slice("/",1);
			data.icon_override.erase(dname);
			update();
		} else if (name.begins_with("custom_styles/")) {
			String dname = name.get_slice("/",1);
			data.style_override.erase(dname);
			update();
		} else if (name.begins_with("custom_fonts/")) {
			String dname = name.get_slice("/",1);
			data.font_override.erase(dname);
			update();
		} else if (name.begins_with("custom_colors/")) {
			String dname = name.get_slice("/",1);
			data.color_override.erase(dname);
			update();
		} else if (name.begins_with("custom_constants/")) {
			String dname = name.get_slice("/",1);
			data.constant_override.erase(dname);
			update();
		} else
			return false;

	} else {
		if (name.begins_with("custom_icons/")) {
			String dname = name.get_slice("/",1);
			add_icon_override(dname,p_value);
		} else if (name.begins_with("custom_styles/")) {
			String dname = name.get_slice("/",1);
			add_style_override(dname,p_value);
		} else if (name.begins_with("custom_fonts/")) {
			String dname = name.get_slice("/",1);
			add_font_override(dname,p_value);
		} else if (name.begins_with("custom_colors/")) {
			String dname = name.get_slice("/",1);
			add_color_override(dname,p_value);
		} else if (name.begins_with("custom_constants/")) {
			String dname = name.get_slice("/",1);
			add_constant_override(dname,p_value);
		} else
			return false;
	}
	return true;

}

void Control::_update_minimum_size() {

	if (!is_inside_tree())
		return;

	data.pending_min_size_update=false;
	Size2 minsize = get_combined_minimum_size();
	if (minsize.x > data.size_cache.x ||
	    minsize.y > data.size_cache.y
	    ) {
		_size_changed();
	}

	emit_signal(SceneStringNames::get_singleton()->minimum_size_changed);

}

bool Control::_get(const StringName& p_name,Variant &r_ret) const {


	String sname=p_name;

	if (!sname.begins_with("custom"))
		return false;

	if (sname.begins_with("custom_icons/")) {
		String name = sname.get_slice("/",1);

		r_ret= data.icon_override.has(name)?Variant(data.icon_override[name]):Variant();
	} else if (sname.begins_with("custom_styles/")) {
		String name = sname.get_slice("/",1);

		r_ret= data.style_override.has(name)?Variant(data.style_override[name]):Variant();
	} else if (sname.begins_with("custom_fonts/")) {
		String name = sname.get_slice("/",1);

		r_ret= data.font_override.has(name)?Variant(data.font_override[name]):Variant();
	} else if (sname.begins_with("custom_colors/")) {
		String name = sname.get_slice("/",1);
		r_ret= data.color_override.has(name)?Variant(data.color_override[name]):Variant();
	} else if (sname.begins_with("custom_constants/")) {
		String name = sname.get_slice("/",1);

		r_ret= data.constant_override.has(name)?Variant(data.constant_override[name]):Variant();
	} else
		return false;


	
	return true;
	

}
void Control::_get_property_list( List<PropertyInfo> *p_list) const {

	Ref<Theme> theme;
	if (data.theme.is_valid()) {

		theme=data.theme;
	} else {
		theme=Theme::get_default();
	}


	{
		List<StringName> names;
		theme->get_icon_list(get_type_name(),&names);
		for(List<StringName>::Element *E=names.front();E;E=E->next()) {

			uint32_t hint= PROPERTY_USAGE_EDITOR|PROPERTY_USAGE_CHECKABLE;
			if (data.icon_override.has(E->get()))
				hint|=PROPERTY_USAGE_STORAGE|PROPERTY_USAGE_CHECKED;

			p_list->push_back( PropertyInfo(Variant::OBJECT,"custom_icons/"+E->get(),PROPERTY_HINT_RESOURCE_TYPE, "Texture",hint) );
		}
	}
	{
		List<StringName> names;
		theme->get_stylebox_list(get_type_name(),&names);
		for(List<StringName>::Element *E=names.front();E;E=E->next()) {

			uint32_t hint= PROPERTY_USAGE_EDITOR|PROPERTY_USAGE_CHECKABLE;
			if (data.style_override.has(E->get()))
				hint|=PROPERTY_USAGE_STORAGE|PROPERTY_USAGE_CHECKED;

			p_list->push_back( PropertyInfo(Variant::OBJECT,"custom_styles/"+E->get(),PROPERTY_HINT_RESOURCE_TYPE, "StyleBox",hint) );
		}
	}
	{
		List<StringName> names;
		theme->get_font_list(get_type_name(),&names);
		for(List<StringName>::Element *E=names.front();E;E=E->next()) {

			uint32_t hint= PROPERTY_USAGE_EDITOR|PROPERTY_USAGE_CHECKABLE;
			if (data.font_override.has(E->get()))
				hint|=PROPERTY_USAGE_STORAGE|PROPERTY_USAGE_CHECKED;

			p_list->push_back( PropertyInfo(Variant::OBJECT,"custom_fonts/"+E->get(),PROPERTY_HINT_RESOURCE_TYPE, "Font",hint) );
		}
	}
	{
		List<StringName> names;
		theme->get_color_list(get_type_name(),&names);
		for(List<StringName>::Element *E=names.front();E;E=E->next()) {

			uint32_t hint= PROPERTY_USAGE_EDITOR|PROPERTY_USAGE_CHECKABLE;
			if (data.color_override.has(E->get()))
				hint|=PROPERTY_USAGE_STORAGE|PROPERTY_USAGE_CHECKED;

			p_list->push_back( PropertyInfo(Variant::COLOR,"custom_colors/"+E->get(),PROPERTY_HINT_NONE, "",hint) );
		}
	}
	{
		List<StringName> names;
		theme->get_constant_list(get_type_name(),&names);
		for(List<StringName>::Element *E=names.front();E;E=E->next()) {

			uint32_t hint= PROPERTY_USAGE_EDITOR|PROPERTY_USAGE_CHECKABLE;
			if (data.constant_override.has(E->get()))
				hint|=PROPERTY_USAGE_STORAGE|PROPERTY_USAGE_CHECKED;

			p_list->push_back( PropertyInfo(Variant::INT,"custom_constants/"+E->get(),PROPERTY_HINT_RANGE, "-16384,16384",hint) );
		}
	}


}


Control *Control::get_parent_control() const {

	return data.parent;
}

void Control::_input_text(const String& p_text) {

	if (!window)
		return;
	if (window->key_focus)
		window->key_focus->call("set_text",p_text);

}

void Control::_gui_input(const InputEvent& p_event) {

	_window_input_event(p_event);
}

void Control::_resize(const Size2& p_size) {
	
	_size_changed();
}



void Control::_notification(int p_notification) {


	switch(p_notification) {

		case NOTIFICATION_ENTER_TREE: {

			if (data.window==this) {

				window = memnew( Window );
				add_to_group("_vp_gui_input"+itos(get_viewport()->get_instance_ID()));
				add_to_group("windows");

				window->tooltip_timer = memnew( Timer );
				add_child(window->tooltip_timer);
				window->tooltip_timer->force_parent_owned();
				window->tooltip_timer->set_wait_time( GLOBAL_DEF("display/tooltip_delay",0.7));
				window->tooltip_timer->connect("timeout",this,"_window_show_tooltip");
				window->tooltip=NULL;
				window->tooltip_popup = memnew( TooltipPanel );
				add_child(window->tooltip_popup);
				window->tooltip_popup->force_parent_owned();
				window->tooltip_label = memnew( TooltipLabel );
				window->tooltip_popup->add_child(window->tooltip_label);
				window->tooltip_popup->set_as_toplevel(true);
				window->tooltip_popup->hide();
				window->drag_attempted=false;
				window->drag_preview=NULL;

				if (get_tree()->is_editor_hint()) {

					Node *n = this;
					while(n) {

						if (n->has_meta("_editor_disable_input")) {
							window->disable_input=true;
							break;
						}
						n=n->get_parent();
					}
				}

			} else {
				window=NULL;
			}

			_size_changed();

		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (data.window) {

				if (data.window->window->mouse_focus == this)
					data.window->window->mouse_focus=NULL;
				if (data.window->window->key_focus == this)
					data.window->window->key_focus=NULL;
				if (data.window->window->mouse_over == this)
					data.window->window->mouse_over=NULL;
				if (data.window->window->tooltip == this)
					data.window->window->tooltip=NULL;
			}

			if (window) {

				remove_from_group("_vp_gui_input"+itos(get_viewport()->get_instance_ID()));
				remove_from_group("windows");
				if (window->tooltip_timer)
					memdelete(window->tooltip_timer);
				window->tooltip_timer=NULL;
				window->tooltip=NULL;
				if (window->tooltip_popup)
					memdelete(window->tooltip_popup);
				window->tooltip_popup=NULL;

				memdelete(window);
				window=NULL;

			}



		} break;


		case NOTIFICATION_ENTER_CANVAS: {

			data.window=NULL;
			data.viewport=NULL;
			data.parent=NULL;

			Control *_window=this;
			bool gap=false;
			bool gap_valid=true;
			bool window_found=false;

			Node *parent=_window->get_parent();
			if (parent && parent->cast_to<Control>()) {

				data.parent=parent->cast_to<Control>();
			}

			Viewport *viewport=NULL;

			parent=this; //meh

			while(parent) {

				Control *c=parent->cast_to<Control>();

				if (!window_found && c) {
					if (!gap && c!=this) {
						gap_valid=false;
					}

					_window = c;
				}

				CanvasItem *ci =parent->cast_to<CanvasItem>();

				if ((ci && ci->is_set_as_toplevel()) || !ci) {
					gap=true;
				}

				if (parent->cast_to<CanvasLayer>()) {
					window_found=true; //don't go beyond canvas layer
				}

				viewport =parent->cast_to<Viewport>();
				if (viewport) {
					break; //no go beyond viewport either
				}

				parent=parent->get_parent();
			}

			data.window=_window;
			data.viewport=viewport;
			data.parent_canvas_item=get_parent_item();

			if (data.parent_canvas_item) {

				data.parent_canvas_item->connect("item_rect_changed",this,"_size_changed");
			} else if (data.viewport) {

				//connect viewport
				data.viewport->connect("size_changed",this,"_size_changed");
			} else {

			}


			if (gap && gap_valid && data.window!=this) {
				//is a subwindow, conditions to meet subwindow status are quite complex..
				data.SI = data.window->window->subwindows.push_back(this);
				data.window->window->subwindow_order_dirty=true;

			}


		} break;
		case NOTIFICATION_EXIT_CANVAS: {


			if (data.parent_canvas_item) {

				data.parent_canvas_item->disconnect("item_rect_changed",this,"_size_changed");
				data.parent_canvas_item=NULL;
			} else if (data.viewport) {

				//disconnect viewport
				data.viewport->disconnect("size_changed",this,"_size_changed");
			} else {

			}

			if (data.MI) {

				data.window->window->modal_stack.erase(data.MI);
				data.MI=NULL;
			}

			if (data.SI) {
				//erase from subwindows
				data.window->window->subwindows.erase(data.SI);
				data.SI=NULL;
			}

			data.viewport=NULL;
			data.window=NULL;
			data.parent=NULL;

		} break;


		case NOTIFICATION_PARENTED: {

			Control * parent = get_parent()->cast_to<Control>();

			//make children reference them theme
			if (parent && data.theme.is_null() && parent->data.theme_owner)
				_propagate_theme_changed(parent->data.theme_owner);

		} break;
		case NOTIFICATION_UNPARENTED: {

			//make children unreference the theme
			if (data.theme.is_null() && data.theme_owner)
				_propagate_theme_changed(NULL);

		} break;
		 case NOTIFICATION_MOVED_IN_PARENT: {
			 // some parents need to know the order of the childrens to draw (like TabContainer)
			 // update if necesary
			 if (data.parent)
				 data.parent->update();
			 update();

			 if (data.SI && data.window) {
				 data.window->window->subwindow_order_dirty=true;
			 }

		 } break;
		case NOTIFICATION_RESIZED: {

			emit_signal(SceneStringNames::get_singleton()->resized);
		} break;
		case NOTIFICATION_DRAW: {

			Matrix32 xform;
			xform.set_origin(get_pos());
			VisualServer::get_singleton()->canvas_item_set_transform(get_canvas_item(),xform);
			VisualServer::get_singleton()->canvas_item_set_custom_rect( get_canvas_item(),true, Rect2(Point2(),get_size()));
			//emit_signal(SceneStringNames::get_singleton()->draw);

		} break;
		case NOTIFICATION_MOUSE_ENTER: {

			emit_signal(SceneStringNames::get_singleton()->mouse_enter);
		} break;
		case NOTIFICATION_MOUSE_EXIT: {

			emit_signal(SceneStringNames::get_singleton()->mouse_exit);
		} break;
		case NOTIFICATION_FOCUS_ENTER: {

			emit_signal(SceneStringNames::get_singleton()->focus_enter);
			update();
		} break;
		case NOTIFICATION_FOCUS_EXIT: {

			emit_signal(SceneStringNames::get_singleton()->focus_exit);
			update();

		} break;
		case NOTIFICATION_THEME_CHANGED: {

			update();
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {

			if (!is_visible()) {

				if (data.window->window->mouse_focus == this) {
					data.window->window->mouse_focus=NULL;
				}
				if (data.window==this) {
					window->drag_data=Variant();
					if (window->drag_preview) {
						memdelete( window->drag_preview);
						window->drag_preview=NULL;
					}
				}

				if (data.window->window->key_focus == this)
					data.window->window->key_focus=NULL;
				if (data.window->window->mouse_over == this)
					data.window->window->mouse_over=NULL;
				if (data.window->window->tooltip == this)
					data.window->window->tooltip=NULL;
				if (data.window->window->tooltip == this)
					data.window->window->tooltip=NULL;

				_modal_stack_remove();
				minimum_size_changed();

				//remove key focus
				//remove modalness
			} else {

				_size_changed();
			}

		} break;
		case SceneTree::NOTIFICATION_WM_UNFOCUS_REQUEST: {

			if (!window)
				return;
			if (window->key_focus)
				window->key_focus->release_focus();

		} break;



	}
}


bool Control::clips_input() const {

	return false;
}
bool Control::has_point(const Point2& p_point) const {

	if (get_script_instance()) {
		Variant v=p_point;
		const Variant *p=&v;
		Variant::CallError ce;
		Variant ret = get_script_instance()->call(SceneStringNames::get_singleton()->has_point,&p,1,ce);
		if (ce.error==Variant::CallError::CALL_OK) {
			return ret;
		}
	}
	/*if (has_stylebox("mask")) {
		Ref<StyleBox> mask = get_stylebox("mask");
		return mask->test_mask(p_point,Rect2(Point2(),get_size()));
	}*/
	return Rect2( Point2(), get_size() ).has_point(p_point);
}

Variant Control::get_drag_data(const Point2& p_point) {

	if (get_script_instance()) {
		Variant v=p_point;
		const Variant *p=&v;
		Variant::CallError ce;
		Variant ret = get_script_instance()->call(SceneStringNames::get_singleton()->get_drag_data,&p,1,ce);
		if (ce.error==Variant::CallError::CALL_OK)
			return ret;
	}

	return Variant();
}


bool Control::can_drop_data(const Point2& p_point,const Variant& p_data) const {

	if (get_script_instance()) {
		Variant v=p_point;
		const Variant *p[2]={&v,&p_data};
		Variant::CallError ce;
		Variant ret = get_script_instance()->call(SceneStringNames::get_singleton()->can_drop_data,p,2,ce);
		if (ce.error==Variant::CallError::CALL_OK)
			return ret;
	}

	return Variant();

}
void Control::drop_data(const Point2& p_point,const Variant& p_data){

	if (get_script_instance()) {
		Variant v=p_point;
		const Variant *p[2]={&v,&p_data};
		Variant::CallError ce;
		Variant ret = get_script_instance()->call(SceneStringNames::get_singleton()->drop_data,p,2,ce);
		if (ce.error==Variant::CallError::CALL_OK)
			return;
	}
}

void Control::force_drag(const Variant& p_data,Control *p_control) {

	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!data.window);
	ERR_FAIL_COND(p_data.get_type()==Variant::NIL);



	data.window->window->drag_data=p_data;
	data.window->window->mouse_focus=NULL;

	if (p_control) {
		data.window->set_drag_preview(p_control);
	}
}

void Control::set_drag_preview(Control *p_control) {

	ERR_FAIL_NULL(p_control);
	ERR_FAIL_COND( !((Object*)p_control)->cast_to<Control>());
	ERR_FAIL_COND(!is_inside_tree() || !data.window);
	ERR_FAIL_COND(p_control->is_inside_tree());
	ERR_FAIL_COND(p_control->get_parent()!=NULL);

	if (data.window->window->drag_preview) {
		memdelete(data.window->window->drag_preview);
	}
	p_control->set_as_toplevel(true);
	p_control->set_pos(data.window->window->last_mouse_pos);
	data.window->add_child(p_control);
	if (data.window->window->drag_preview) {
		memdelete( data.window->window->drag_preview );
	}

	data.window->window->drag_preview=p_control;

}


Control* Control::_find_next_visible_control_at_pos(Node* p_node,const Point2& p_global,Matrix32& r_xform) const {

	return NULL;
}

Control* Control::_find_control_at_pos(CanvasItem* p_node,const Point2& p_global,const Matrix32& p_xform,Matrix32& r_inv_xform)  {

	if (p_node->cast_to<Viewport>())
		return NULL;

	Control *c=p_node->cast_to<Control>();

	if (c) {
	//	print_line("at "+String(c->get_path())+" POS "+c->get_pos()+" bt "+p_xform);
	}

	if (c==data.window) {
		//try subwindows first!!

		c->_window_sort_subwindows(); // sort them

		for (List<Control*>::Element *E=c->window->subwindows.back();E;E=E->prev()) {

			Control *sw = E->get();
			if (!sw->is_visible())
				continue;

			Matrix32 xform;
			CanvasItem *pci = sw->get_parent_item();
			if (pci)
				xform=pci->get_global_transform();

			Control *ret = _find_control_at_pos(sw,p_global,xform,r_inv_xform);
			if (ret)
				return ret;

		}
	}

	if (p_node->is_hidden()) {
		//return _find_next_visible_control_at_pos(p_node,p_global,r_inv_xform);
		return NULL; //canvas item hidden, discard
	}

	Matrix32 matrix = p_xform * p_node->get_transform();

	if (!c || !c->clips_input() || c->has_point(matrix.affine_inverse().xform(p_global))) {

		for(int i=p_node->get_child_count()-1;i>=0;i--) {

			if (p_node==data.window->window->tooltip_popup)
				continue;

			CanvasItem *ci = p_node->get_child(i)->cast_to<CanvasItem>();
			if (!ci || ci->is_set_as_toplevel())
				continue;

			Control *ret=_find_control_at_pos(ci,p_global,matrix,r_inv_xform);;
			if (ret)
				return ret;
		}
	}

	if (!c)
		return NULL;

	matrix.affine_invert();

	//conditions for considering this as a valid control for return
	if (!c->data.ignore_mouse && c->has_point(matrix.xform(p_global)) && (!window->drag_preview || (c!=window->drag_preview && !window->drag_preview->is_a_parent_of(c)))) {
		r_inv_xform=matrix;
		return c;
	} else
		return NULL;
}
		
void Control::_window_cancel_input_ID(int p_input) {

	window->cancelled_input_ID=(unsigned int)p_input;
}
		
void Control::_window_remove_focus() {
	
	if (window->key_focus) {
		
		Node *f=window->key_focus;
		window->key_focus=NULL;
		f->notification( NOTIFICATION_FOCUS_EXIT,true );

	}		
}

bool Control::window_has_modal_stack() const {


	if (!data.window)
		return false;
	return data.window->window->modal_stack.size();
}

void Control::_window_cancel_tooltip() {

	window->tooltip=NULL;
	if (window->tooltip_timer)
		window->tooltip_timer->stop();
	if (window->tooltip_popup)
		window->tooltip_popup->hide();

}

void Control::_window_show_tooltip() {

	if (!window->tooltip) {
		return;
	}

	String tooltip = window->tooltip->get_tooltip( window->tooltip->get_global_transform().xform_inv(window->tooltip_pos) );
	if (tooltip.length()==0)
		return; // bye


	if (!window->tooltip_label) {
		return;
	}
	Ref<StyleBox> ttp = get_stylebox("panel","TooltipPanel");

	window->tooltip_label->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,ttp->get_margin(MARGIN_LEFT));
	window->tooltip_label->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,ttp->get_margin(MARGIN_TOP));
	window->tooltip_label->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,ttp->get_margin(MARGIN_RIGHT));
	window->tooltip_label->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,ttp->get_margin(MARGIN_BOTTOM));
	window->tooltip_label->set_text(tooltip);
	Rect2 r(window->tooltip_pos+Point2(10,10),window->tooltip_label->get_combined_minimum_size()+ttp->get_minimum_size());
	Rect2 vr = get_viewport_rect();
	if (r.size.x+r.pos.x>vr.size.x)
		r.pos.x=vr.size.x-r.size.x;
	else if (r.pos.x<0)
		r.pos.x=0;

	if (r.size.y+r.pos.y>vr.size.y)
		r.pos.y=vr.size.y-r.size.y;
	else if (r.pos.y<0)
		r.pos.y=0;

	window->tooltip_popup->set_pos(r.pos);
	window->tooltip_popup->set_size(r.size);

	window->tooltip_popup->raise();

	window->tooltip_popup->show();
}


void Control::_window_call_input(Control *p_control,const InputEvent& p_input) {


	while(p_control) {

		p_control->call_multilevel(SceneStringNames::get_singleton()->_input_event,p_input);
		if (window->key_event_accepted)
			break;
		p_control->emit_signal(SceneStringNames::get_singleton()->input_event,p_input);
		if (p_control->is_set_as_toplevel()) {
			break;
		}
		if (window->key_event_accepted)
			break;
		if (p_control->data.stop_mouse && (p_input.type==InputEvent::MOUSE_BUTTON || p_input.type==InputEvent::MOUSE_MOTION))
			break;
		p_control=p_control->data.parent;
	}
}

void Control::_window_input_event(InputEvent p_event) {



	if (!window)
		return;

	if (window->disable_input)
		return;

	if (p_event.ID==window->cancelled_input_ID) {
		return;
	}
	if (!is_visible()) {
		return; //simple and plain
	}
	switch(p_event.type) {
	
		case InputEvent::MOUSE_BUTTON: {


		window->key_event_accepted=false;

		Point2 mpos =(get_canvas_transform()).affine_inverse().xform(Point2(p_event.mouse_button.x,p_event.mouse_button.y));
		if (p_event.mouse_button.pressed) {



			Size2 pos = mpos;
			if (window->mouse_focus && p_event.mouse_button.button_index!=window->mouse_focus_button) {

				//do not steal mouse focus and stuff

			} else {


				_window_sort_modal_stack();
				while (!window->modal_stack.empty()) {

					Control *top = window->modal_stack.back()->get();
					if (!top->has_point(top->get_global_transform().affine_inverse().xform(pos))) {

						if (top->data.modal_exclusive) {
							//cancel event, sorry, modal exclusive EATS UP ALL
							get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_cancel_input_ID",p_event.ID);
							get_tree()->set_input_as_handled();
							return; // no one gets the event if exclusive NO ONE
						}

						top->notification(NOTIFICATION_MODAL_CLOSE);
						top->_modal_stack_remove();
						top->hide();
					} else {
						break;
					}
				}



				Matrix32 parent_xform;

				if (data.parent_canvas_item)
					parent_xform=data.parent_canvas_item->get_global_transform();



				window->mouse_focus = _find_control_at_pos(this,pos,parent_xform,window->focus_inv_xform);
				//print_line("has mf "+itos(window->mouse_focus!=NULL));
				window->mouse_focus_button=p_event.mouse_button.button_index;

				if (!window->mouse_focus) {
					break;
				}

				if (p_event.mouse_button.button_index==BUTTON_LEFT) {
					window->drag_accum=Vector2();
					window->drag_attempted=false;
					window->drag_data=Variant();
				}


			}

				p_event.mouse_button.global_x = pos.x;
				p_event.mouse_button.global_y = pos.y;

				pos = window->focus_inv_xform.xform(pos);
				p_event.mouse_button.x = pos.x;
				p_event.mouse_button.y = pos.y;

#ifdef DEBUG_ENABLED
				if (ScriptDebugger::get_singleton()) {

					Array arr;
					arr.push_back(window->mouse_focus->get_path());
					arr.push_back(window->mouse_focus->get_type());
					ScriptDebugger::get_singleton()->send_message("click_ctrl",arr);
				}

				/*if (bool(GLOBAL_DEF("debug/print_clicked_control",false))) {

					print_line(String(window->mouse_focus->get_path())+" - "+pos);
				}*/
#endif

				if (window->mouse_focus->get_focus_mode()!=FOCUS_NONE && window->mouse_focus!=window->key_focus && p_event.mouse_button.button_index==BUTTON_LEFT) {
					// also get keyboard focus
					window->mouse_focus->grab_focus();
				}


				if (window->mouse_focus->can_process()) {
					_window_call_input(window->mouse_focus,p_event);
				}
				
				get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_cancel_input_ID",p_event.ID);
				get_tree()->set_input_as_handled();

			} else {

				if (window->drag_preview && p_event.mouse_button.button_index==BUTTON_LEFT) {
					memdelete( window->drag_preview );
					window->drag_preview=NULL;
				}

				if (!window->mouse_focus) {

					if (window->mouse_over && window->drag_data.get_type()!=Variant::NIL && p_event.mouse_button.button_index==BUTTON_LEFT) {

						Size2 pos = mpos;
						pos = window->focus_inv_xform.xform(pos);
						window->mouse_over->drop_data(pos,window->drag_data);
						window->drag_data=Variant();
						//change mouse accordingly
					}

					break;
				}

				Size2 pos = mpos;
				p_event.mouse_button.global_x = pos.x;
				p_event.mouse_button.global_y = pos.y;
				pos = window->focus_inv_xform.xform(pos);
				p_event.mouse_button.x = pos.x;
				p_event.mouse_button.y = pos.y;

				if (window->mouse_focus->can_process()) {
					_window_call_input(window->mouse_focus,p_event);
				}

				if (p_event.mouse_button.button_index==window->mouse_focus_button) {
					window->mouse_focus=NULL;
					window->mouse_focus_button=-1;
				}

				if (window->drag_data.get_type()!=Variant::NIL && p_event.mouse_button.button_index==BUTTON_LEFT) {
					window->drag_data=Variant(); //always clear
				}


				get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_cancel_input_ID",p_event.ID);
				get_tree()->set_input_as_handled();

			}
		} break;
		case InputEvent::MOUSE_MOTION: {

			window->key_event_accepted=false;

			Matrix32 localizer = (get_canvas_transform()).affine_inverse();
			Size2 pos = localizer.xform(Size2(p_event.mouse_motion.x,p_event.mouse_motion.y));
			Vector2 speed = localizer.basis_xform(Point2(p_event.mouse_motion.speed_x,p_event.mouse_motion.speed_y));
			Vector2 rel = localizer.basis_xform(Point2(p_event.mouse_motion.relative_x,p_event.mouse_motion.relative_y));

			window->last_mouse_pos=pos;
			
			Control *over = NULL;

			Matrix32 parent_xform;
			if (data.parent_canvas_item)
				parent_xform=data.parent_canvas_item->get_global_transform();

			// D&D
			if (!window->drag_attempted && window->mouse_focus && p_event.mouse_motion.button_mask&BUTTON_MASK_LEFT) {

				window->drag_accum+=rel;
				float len = window->drag_accum.length();
				if (len>10) {
					window->drag_data=window->mouse_focus->get_drag_data(window->focus_inv_xform.xform(pos)-window->drag_accum);
					if (window->drag_data.get_type()!=Variant::NIL) {

						window->mouse_focus=NULL;
					}
					window->drag_attempted=true;
				}
			}


			if (window->mouse_focus) {
				over=window->mouse_focus;
				//recompute focus_inv_xform again here

			} else {

				over = _find_control_at_pos(this,pos,parent_xform,window->focus_inv_xform);
			}


			if (window->drag_data.get_type()==Variant::NIL && over && !window->modal_stack.empty()) {

				Control *top = window->modal_stack.back()->get();
				if (over!=top && !top->is_a_parent_of(over)) {

					break; // don't send motion event to anything below modal stack top
				}
			}

			if (over!=window->mouse_over) {
			
				if (window->mouse_over)
					window->mouse_over->notification(NOTIFICATION_MOUSE_EXIT);
					
				if (over)
					over->notification(NOTIFICATION_MOUSE_ENTER);
					
			}
			
			window->mouse_over=over;

			get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_cancel_tooltip");

			if (window->drag_preview) {
				window->drag_preview->set_pos(pos);
			}

			if (!over) {
				OS::get_singleton()->set_cursor_shape(OS::CURSOR_ARROW);
			 	break;
			}
			 	
			p_event.mouse_motion.global_x = pos.x;
			p_event.mouse_motion.global_y = pos.y;
			p_event.mouse_motion.speed_x=speed.x;
			p_event.mouse_motion.speed_y=speed.y;
			p_event.mouse_motion.relative_x=rel.x;
			p_event.mouse_motion.relative_y=rel.y;

			if (p_event.mouse_motion.button_mask==0 && window->tooltip_timer) {
				//nothing pressed

				bool can_tooltip=true;

				if (!window->modal_stack.empty()) {
					if (window->modal_stack.back()->get()!=over && !window->modal_stack.back()->get()->is_a_parent_of(over))
						can_tooltip=false;

				}


				if (can_tooltip) {

					window->tooltip=over;
					window->tooltip_pos=(parent_xform * get_transform()).affine_inverse().xform(pos);
					window->tooltip_timer->start();
				}
			}


			pos = window->focus_inv_xform.xform(pos);


			p_event.mouse_motion.x = pos.x;
			p_event.mouse_motion.y = pos.y;


			CursorShape cursor_shape = over->get_cursor_shape(pos);
			OS::get_singleton()->set_cursor_shape( (OS::CursorShape)cursor_shape );


			if (over->can_process()) {
				_window_call_input(over,p_event);
			}


			
			get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_cancel_input_ID",p_event.ID);
			get_tree()->set_input_as_handled();


			if (window->drag_data.get_type()!=Variant::NIL && p_event.mouse_motion.button_mask&BUTTON_MASK_LEFT) {

				/*bool can_drop =*/ over->can_drop_data(pos,window->drag_data);
				//change mouse accordingly i guess
			}

		} break;
		case InputEvent::ACTION:
		case InputEvent::JOYSTICK_BUTTON:
		case InputEvent::KEY: {
	
			if (window->key_focus) {
			
				window->key_event_accepted=false;
				if (window->key_focus->can_process()) {
					window->key_focus->call_multilevel("_input_event",p_event);
					if (window->key_focus) //maybe lost it
						window->key_focus->emit_signal(SceneStringNames::get_singleton()->input_event,p_event);
				}


				if (window->key_event_accepted) {

					get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_cancel_input_ID",p_event.ID);
					break;
				}
			}


			if (p_event.is_pressed() && p_event.is_action("ui_cancel") && !window->modal_stack.empty()) {

				_window_sort_modal_stack();
				Control *top = window->modal_stack.back()->get();
				if (!top->data.modal_exclusive) {

					top->notification(NOTIFICATION_MODAL_CLOSE);
					top->_modal_stack_remove();
					top->hide();
				}
			}


			Control * from = window->key_focus ? window->key_focus : NULL; //hmm

			//keyboard focus
			//if (from && p_event.key.pressed && !p_event.key.mod.alt && !p_event.key.mod.meta && !p_event.key.mod.command) {

			if (from && p_event.is_pressed()) {
				Control * next=NULL;

				if (p_event.is_action("ui_focus_next")) {

					next = from->find_next_valid_focus();
				}

				if (p_event.is_action("ui_focus_prev")) {

					next = from->find_prev_valid_focus();
				}

				if (p_event.is_action("ui_up")) {

					next = from->_get_focus_neighbour(MARGIN_TOP);
				}

				if (p_event.is_action("ui_left")) {

					next = from->_get_focus_neighbour(MARGIN_LEFT);
				}

				if (p_event.is_action("ui_right")) {

					next = from->_get_focus_neighbour(MARGIN_RIGHT);
				}

				if (p_event.is_action("ui_down")) {

					next = from->_get_focus_neighbour(MARGIN_BOTTOM);
				}


				if (next) {
					next->grab_focus();
					get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_cancel_input_ID",p_event.ID);
				}
			}

		} break;
	}
}

Control *Control::get_window() const {
	
	return data.window;
}

bool Control::is_window() const {

	return (is_inside_tree() && window);
}


Size2 Control::get_minimum_size() const {
	
	ScriptInstance *si = const_cast<Control*>(this)->get_script_instance();
	if (si) {

		Variant::CallError ce;
		Variant s = si->call(SceneStringNames::get_singleton()->get_minimum_size,NULL,0,ce);
		if (ce.error==Variant::CallError::CALL_OK)
			return s;
	}
	return Size2();
}


Ref<Texture> Control::get_icon(const StringName& p_name,const StringName& p_type) const {
	
	if (p_type==StringName()) {

		const Ref<Texture>* tex = data.icon_override.getptr(p_name);
		if (tex)
			return *tex;
	}

	StringName type = p_type?p_type:get_type_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_icon(p_name, type ) )
			return data.theme_owner->data.theme->get_icon(p_name, type );
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;

	}

	return Theme::get_default()->get_icon( p_name, type );

}

Ref<StyleBox> Control::get_stylebox(const StringName& p_name,const StringName& p_type) const {
		
	if (p_type==StringName()) {
		const Ref<StyleBox>* style = data.style_override.getptr(p_name);
		if (style)
			return *style;
	}

	StringName type = p_type?p_type:get_type_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_stylebox(p_name, type ) )
			return data.theme_owner->data.theme->get_stylebox(p_name, type );
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;
	}

	return Theme::get_default()->get_stylebox( p_name, type );

}
Ref<Font> Control::get_font(const StringName& p_name,const StringName& p_type) const {

	if (p_type==StringName()) {
		const Ref<Font>* font = data.font_override.getptr(p_name);
		if (font)
			return *font;
	}

	StringName type = p_type?p_type:get_type_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_font(p_name, type ) )
			return data.theme_owner->data.theme->get_font(p_name, type );
		if (theme_owner->data.theme->get_default_theme_font().is_valid())
			return theme_owner->data.theme->get_default_theme_font();
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;

	}

	return Theme::get_default()->get_font( p_name, type );

}
Color Control::get_color(const StringName& p_name,const StringName& p_type) const {

	if (p_type==StringName()) {
		const Color* color = data.color_override.getptr(p_name);
		if (color)
			return *color;
	}

	StringName type = p_type?p_type:get_type_name();
	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_color(p_name, type ) )
			return data.theme_owner->data.theme->get_color(p_name, type );
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;

	}

	return Theme::get_default()->get_color( p_name, type );

}

int Control::get_constant(const StringName& p_name,const StringName& p_type) const {

	if (p_type==StringName()) {
		const int* constant = data.constant_override.getptr(p_name);
		if (constant)
			return *constant;
	}

	StringName type = p_type?p_type:get_type_name();
		// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_constant(p_name, type ) )
			return data.theme_owner->data.theme->get_constant(p_name, type );
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;

	}

	return Theme::get_default()->get_constant( p_name, type );

	
}


bool Control::has_icon(const StringName& p_name,const StringName& p_type) const {
	
	if (p_type==StringName()) {
		const Ref<Texture>* tex = data.icon_override.getptr(p_name);
		if (tex)
			return true;
	}

	StringName type = p_type?p_type:get_type_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_icon(p_name, type ) )
			return true;
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;

	}

	return Theme::get_default()->has_icon( p_name, type );

}
bool Control::has_stylebox(const StringName& p_name,const StringName& p_type) const {
		
	if (p_type==StringName()) {
		const Ref<StyleBox>* style = data.style_override.getptr(p_name);

		if (style)
			return true;
	}

	StringName type = p_type?p_type:get_type_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_stylebox(p_name, type ) )
			return true;
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;

	}

	return Theme::get_default()->has_stylebox( p_name, type );

}
bool Control::has_font(const StringName& p_name,const StringName& p_type) const {
	
	if (p_type==StringName()) {
		const Ref<Font>* font = data.font_override.getptr(p_name);
		if (font)
			return true;
	}


	StringName type = p_type?p_type:get_type_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_font(p_name, type ) )
			return true;
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;

	}

	return Theme::get_default()->has_font( p_name, type );

}
bool Control::has_color(const StringName& p_name,const StringName& p_type) const {
	
	if (p_type==StringName()) {
		const Color* color = data.color_override.getptr(p_name);
		if (color)
			return true;
	}

	StringName type = p_type?p_type:get_type_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_color(p_name, type ) )
			return true;
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;

	}

	return Theme::get_default()->has_color( p_name, type );

}

bool Control::has_constant(const StringName& p_name,const StringName& p_type) const {

	if (p_type==StringName()) {

		const int* constant = data.constant_override.getptr(p_name);
		if (constant)
			return true;
	}


	StringName type = p_type?p_type:get_type_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while(theme_owner) {

		if (theme_owner->data.theme->has_constant(p_name, type ) )
			return true;
		Control *parent = theme_owner->get_parent()?theme_owner->get_parent()->cast_to<Control>():NULL;

		if (parent)
			theme_owner=parent->data.theme_owner;
		else
			theme_owner=NULL;

	}

	return Theme::get_default()->has_constant( p_name, type );
}

Size2 Control::get_parent_area_size() const {

	ERR_FAIL_COND_V(!is_inside_tree(),Size2());

	Size2 parent_size;

	if (data.parent_canvas_item) {

		parent_size=data.parent_canvas_item->get_item_rect().size;
	} else if (data.viewport) {

		parent_size=data.viewport->get_visible_rect().size;
	} 
	return parent_size;

}

void Control::_size_changed() {

	if (!is_inside_tree())
		return;

	Size2 parent_size = get_parent_area_size();

	float margin_pos[4];

	for(int i=0;i<4;i++) {

		float area = parent_size[i&1];
		switch(data.anchor[i]) {

			case ANCHOR_BEGIN: {

				margin_pos[i]=data.margin[i];
			} break;
			case ANCHOR_END: {

				margin_pos[i]=area-data.margin[i];
			} break;
			case ANCHOR_RATIO: {

				margin_pos[i]=area*data.margin[i];
			} break;
            case ANCHOR_CENTER: {

                margin_pos[i]=(area/2)-data.margin[i];
            } break;
		}
	}

	Point2 new_pos_cache=Point2(margin_pos[0],margin_pos[1]).floor();
	Size2 new_size_cache=Point2(margin_pos[2],margin_pos[3]).floor()-new_pos_cache;
	Size2 minimum_size=get_combined_minimum_size();

	new_size_cache.x = MAX( minimum_size.x, new_size_cache.x );
	new_size_cache.y = MAX( minimum_size.y, new_size_cache.y );


	if (new_pos_cache == data.pos_cache && new_size_cache == data.size_cache)
		return; // did not change, don't emit signal

	data.pos_cache=new_pos_cache;
	data.size_cache=new_size_cache;

	notification(NOTIFICATION_RESIZED);
	item_rect_changed();
	_change_notify_margins();
	_notify_transform();
}

float Control::_get_parent_range(int p_idx) const {
	
	if (!is_inside_tree()) {
	
		return 1.0;
		
	} if (data.parent_canvas_item) {

		return data.parent_canvas_item->get_item_rect().size[p_idx&1];
	} else if (data.viewport) {
		return data.viewport->get_visible_rect().size[p_idx&1];
	}

	return 1.0;
}


float Control::_get_range(int p_idx) const {
	
	p_idx&=1;

	float parent_range = _get_parent_range( p_idx );
	float from = _a2s( data.margin[p_idx], data.anchor[p_idx], parent_range );
	float to = _a2s( data.margin[p_idx+2], data.anchor[p_idx+2], parent_range );
	
	return to-from;
}

float Control::_s2a(float p_val, AnchorType p_anchor,float p_range) const {
	
	switch(p_anchor) {
		
		case ANCHOR_BEGIN: {			
			return p_val;
		} break;
		case ANCHOR_END: {
			return p_range-p_val;
		} break;
		case ANCHOR_RATIO: {
			return p_val/p_range;
		} break;			
        case ANCHOR_CENTER: {
            return (p_range/2)-p_val;
        } break;
	}	
	
	return 0;
}


float Control::_a2s(float p_val, AnchorType p_anchor,float p_range) const {
	
	switch(p_anchor) {
		
		case ANCHOR_BEGIN: {			
			return Math::floor(p_val);
		} break;
		case ANCHOR_END: {
			return Math::floor(p_range-p_val);
		} break;
		case ANCHOR_RATIO: {
			return Math::floor(p_range*p_val);
		} break;			
		case ANCHOR_CENTER: {
		    return Math::floor((p_range/2)-p_val);
		} break;
	}
	return 0;
}


void Control::set_anchor(Margin p_margin,AnchorType p_anchor) {
	
	if (!is_inside_tree()) {
		
		data.anchor[p_margin]=p_anchor;
	} else {
		float pr = _get_parent_range(p_margin);
		float s = _a2s( data.margin[p_margin], data.anchor[p_margin], pr );
		data.anchor[p_margin]=p_anchor;
		data.margin[p_margin] = _s2a( s, p_anchor, pr );
	}
	_change_notify();
}

void Control::set_anchor_and_margin(Margin p_margin,AnchorType p_anchor, float p_pos) {

	set_anchor(p_margin,p_anchor);
	set_margin(p_margin,p_pos);
}


Control::AnchorType Control::get_anchor(Margin p_margin) const {
	
	return data.anchor[p_margin];	
}





void Control::_change_notify_margins() {

	// this avoids sending the whole object data again on a change
	_change_notify("margin/left");
	_change_notify("margin/top");
	_change_notify("margin/right");
	_change_notify("margin/bottom");
	_change_notify("rect/pos");
	_change_notify("rect/size");

}


void Control::set_margin(Margin p_margin,float p_value) {

	data.margin[p_margin]=p_value;
	_size_changed();

}

void Control::set_begin(const Size2& p_point) {
	
	data.margin[0]=p_point.x;
	data.margin[1]=p_point.y;
	_size_changed();
}

void Control::set_end(const Size2& p_point) {
	
	data.margin[2]=p_point.x;
	data.margin[3]=p_point.y;
	_size_changed();
}

float Control::get_margin(Margin p_margin) const {
	
	return data.margin[p_margin];
}

Size2 Control::get_begin() const {
	
	return Size2( data.margin[0], data.margin[1] );
}
Size2 Control::get_end() const {
	
	return Size2( data.margin[2], data.margin[3] );
}

Point2 Control::get_global_pos() const {
	
	return get_global_transform().get_origin();
}

void Control::set_global_pos(const Point2& p_point) {
	
	Matrix32 inv;

	if (data.parent_canvas_item) {

		inv = data.parent_canvas_item->get_global_transform().affine_inverse();
	}

	set_pos(inv.xform(p_point));
}

void Control::set_pos(const Size2& p_point) {

	float pw = _get_parent_range(0);
	float ph = _get_parent_range(1);

	float x = _a2s( data.margin[0], data.anchor[0], pw );
	float y = _a2s( data.margin[1], data.anchor[1], ph );
	float x2 = _a2s( data.margin[2], data.anchor[2], pw );
	float y2 = _a2s( data.margin[3], data.anchor[3], ph );

	Size2 ret = Size2(x2-x,y2-y);
	Size2 min = get_combined_minimum_size();

	Size2 size = Size2(MAX( min.width, ret.width),MAX( min.height, ret.height));
	float w=size.x;
	float h=size.y;
	
	x=p_point.x;
	y=p_point.y;
	
	data.margin[0] = _s2a( x, data.anchor[0], pw );
	data.margin[1] = _s2a( y, data.anchor[1], ph );
	data.margin[2] = _s2a( x+w, data.anchor[2], pw );
	data.margin[3] = _s2a( y+h, data.anchor[3], ph );

	_size_changed();
}

void Control::set_size(const Size2& p_size) {
		
	Size2 new_size=p_size;
	Size2 min=get_combined_minimum_size();
	if (new_size.x<min.x)
		new_size.x=min.x;
	if (new_size.y<min.y)
		new_size.y=min.y;
	
	float pw = _get_parent_range(0);
	float ph = _get_parent_range(1);
	
	float x = _a2s( data.margin[0], data.anchor[0], pw );
	float y = _a2s( data.margin[1], data.anchor[1], ph );
	
	float w=new_size.width;
	float h=new_size.height;
	
	data.margin[2] = _s2a( x+w, data.anchor[2], pw );
	data.margin[3] = _s2a( y+h, data.anchor[3], ph );
	
	_size_changed();
}


Size2 Control::get_pos() const {

	return data.pos_cache;
}

Size2 Control::get_size() const {
	
	return data.size_cache;
}

Rect2 Control::get_global_rect() const {
						    
	return Rect2( get_global_pos(), get_size() );				    
}

Rect2 Control::get_window_rect() const {

	Rect2 gr = get_global_rect();
	if (data.viewport)
		gr.pos+=data.viewport->get_visible_rect().pos;
	return gr;
}


Rect2 Control::get_rect() const {
	
	return Rect2(get_pos(),get_size());
}

Rect2 Control::get_item_rect() const {

	return Rect2(Point2(),get_size());
}

void Control::set_area_as_parent_rect(int p_margin) {
	
	data.anchor[MARGIN_LEFT]=ANCHOR_BEGIN;
	data.anchor[MARGIN_TOP]=ANCHOR_BEGIN;
	data.anchor[MARGIN_RIGHT]=ANCHOR_END;
	data.anchor[MARGIN_BOTTOM]=ANCHOR_END;
	for(int i=0;i<4;i++)
		data.margin[i]=p_margin;

	_size_changed();
	
}

void Control::add_icon_override(const StringName& p_name, const Ref<Texture>& p_icon) {

	ERR_FAIL_COND(p_icon.is_null());
	data.icon_override[p_name]=p_icon;
	notification(NOTIFICATION_THEME_CHANGED);
	update();

}
void Control::add_style_override(const StringName& p_name, const Ref<StyleBox>& p_style) {

	ERR_FAIL_COND(p_style.is_null());
	data.style_override[p_name]=p_style;
	notification(NOTIFICATION_THEME_CHANGED);
	update();
}


void Control::add_font_override(const StringName& p_name, const Ref<Font>& p_font) {

	ERR_FAIL_COND(p_font.is_null());
	data.font_override[p_name]=p_font;
	notification(NOTIFICATION_THEME_CHANGED);
	update();
}
void Control::add_color_override(const StringName& p_name, const Color& p_color) {

	data.color_override[p_name]=p_color;
	notification(NOTIFICATION_THEME_CHANGED);
	update();
}
void Control::add_constant_override(const StringName& p_name, int p_constant) {

	data.constant_override[p_name]=p_constant;
	notification(NOTIFICATION_THEME_CHANGED);
	update();
}

void Control::set_focus_mode(FocusMode p_focus_mode) {

	if (is_inside_tree() && p_focus_mode == FOCUS_NONE && data.focus_mode!=FOCUS_NONE && has_focus())
		release_focus();

	data.focus_mode=p_focus_mode;

}

static Control *_next_control(Control *p_from) {

	if (p_from->is_set_as_toplevel())
		return NULL; // can't go above

	Control *parent = p_from->get_parent()?p_from->get_parent()->cast_to<Control>():NULL;	

	if (!parent) {

		return NULL;
	}


	int next = p_from->get_position_in_parent();
	ERR_FAIL_INDEX_V(next,parent->get_child_count(),NULL);
	for(int i=(next+1);i<parent->get_child_count();i++) {

		Control *c = parent->get_child(i)->cast_to<Control>();
		if (!c || !c->is_visible() || c->is_set_as_toplevel())
			continue;

		return c;
	}

	//no next in parent, try the same in parent
	return _next_control(parent);
}

Control *Control::find_next_valid_focus() const {

	Control *from = const_cast<Control*>(this);
	
	while(true) {
	
		
		// find next child

		Control *next_child=NULL;


		for(int i=0;i<from->get_child_count();i++) {

			Control *c = from->get_child(i)->cast_to<Control>();
			if (!c || !c->is_visible() || c->is_set_as_toplevel()) {
				continue;
			}

			next_child=c;
			break;
		}

		if (next_child) {

			from = next_child;
		} else {

			next_child=_next_control(from);
			if (!next_child) { //nothing else.. go up and find either window or subwindow
				next_child=const_cast<Control*>(this);
				while(next_child && !next_child->is_set_as_toplevel()) {
					if (next_child->get_parent()) {
						next_child=next_child->get_parent()->cast_to<Control>();
					} else
						next_child=NULL;

				}

				if (!next_child) {
					next_child=get_window();
				}
			}

		}


		if (next_child==this) // no next control->
			return (get_focus_mode()==FOCUS_ALL)?next_child:NULL;

		if (next_child->get_focus_mode()==FOCUS_ALL)
			return next_child;

		from = next_child;
	}
	
	return NULL;


}




static Control *_prev_control(Control *p_from) {


	Control *child=NULL;
	for(int i=p_from->get_child_count()-1;i>=0;i--) {

		Control *c = p_from->get_child(i)->cast_to<Control>();
		if (!c || !c->is_visible() || c->is_set_as_toplevel())
			continue;

		child=c;
		break;
	}

	if (!child)
		return p_from;

	//no prev in parent, try the same in parent
	return _prev_control(child);
}

Control *Control::find_prev_valid_focus() const {
	Control *from = const_cast<Control*>(this);

	while(true) {


		// find prev child


			Control *prev_child = NULL;



			if ( from->is_set_as_toplevel() || !from->get_parent() || !from->get_parent()->cast_to<Control>()) {

				//find last of the childs

				prev_child=_prev_control(from);

			} else {

				for(int i=(from->get_position_in_parent()-1);i>=0;i--) {


					Control *c = from->get_parent()->get_child(i)->cast_to<Control>();

					if (!c || !c->is_visible() || c->is_set_as_toplevel()) {
						continue;
					}

					prev_child=c;
					break;

				}

				if (!prev_child) {



					prev_child = from->get_parent()->cast_to<Control>();
				} else {



					prev_child = _prev_control(prev_child);
				}
			}




			if (prev_child==this) // no prev control->
				return (get_focus_mode()==FOCUS_ALL)?prev_child:NULL;

			if (prev_child->get_focus_mode()==FOCUS_ALL)
				return prev_child;

			from = prev_child;

		}

		return NULL;

	return NULL;
}


Control::FocusMode Control::get_focus_mode() const {

	return data.focus_mode;
}
bool Control::has_focus() const {

	return (data.window && data.window->window->key_focus==this);
}

void Control::grab_focus() {

	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!data.window);
	
	if (data.focus_mode==FOCUS_NONE)
		return;
	
	//no need for change
	if (data.window->window->key_focus && data.window->window->key_focus==this)
		return;
		
	get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_window_remove_focus");	
	data.window->window->key_focus=this;
	notification(NOTIFICATION_FOCUS_ENTER);
#ifdef DEBUG_ENABLED
	if (GLOBAL_DEF("debug/print_clicked_control", false)) {
		print_line(String(get_path())+" - focus");
	};
#endif
	update();

}	

void Control::release_focus() {

	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!data.window);

	if (!has_focus())
		return;

	get_tree()->call_group(SceneTree::GROUP_CALL_REALTIME,"windows","_window_remove_focus");
	//data.window->window->key_focus=this;
	//notification(NOTIFICATION_FOCUS_ENTER);
	update();

}

bool Control::is_toplevel_control() const {

	return is_inside_tree() && (!data.parent_canvas_item && !window && is_set_as_toplevel());
}

void Control::show_modal(bool p_exclusive) {
	
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!data.SI && data.window!=this);
	ERR_FAIL_COND(!data.window);

	if (is_visible())
		hide();

	ERR_FAIL_COND( data.MI );
	show();
	raise();

	data.window->window->modal_stack.push_back(this);
	data.MI = data.window->window->modal_stack.back();
	data.modal_exclusive=p_exclusive;
	if (data.window->window->key_focus)
		data.modal_prev_focus_owner = data.window->window->key_focus->get_instance_ID();
	else
		data.modal_prev_focus_owner=0;
	
}

void Control::_window_sort_subwindows() {

	if (!window->subwindow_order_dirty)
		return;

	window->modal_stack.sort_custom<CComparator>();
	window->subwindows.sort_custom<CComparator>();
	window->subwindow_order_dirty=false;

}

void Control::_window_sort_modal_stack() {

	window->modal_stack.sort_custom<CComparator>();
}

void Control::_modal_stack_remove() {


	List<Control*>::Element *next=NULL; //transfer the focus stack to the next


	if (data.window && data.MI) {

		next = data.MI->next();


		data.window->window->modal_stack.erase(data.MI);
		data.MI=NULL;
	}

	if (data.modal_prev_focus_owner) {

		if (!next) { //top of stack

			Object *pfo = ObjectDB::get_instance(data.modal_prev_focus_owner);
			Control *pfoc = pfo->cast_to<Control>();
			if (!pfoc)
				return;

			if (!pfoc->is_inside_tree() || !pfoc->is_visible())
				return;
			pfoc->grab_focus();
		} else {

			next->get()->data.modal_prev_focus_owner=data.modal_prev_focus_owner;
		}

		data.modal_prev_focus_owner=0;
	}
}

void Control::_propagate_theme_changed(Control *p_owner) {

	for(int i=0;i<get_child_count();i++) {

		Control *child = get_child(i)->cast_to<Control>();
		if (child && child->data.theme.is_null()) //has no theme, propagate
			child->_propagate_theme_changed(p_owner);
	}

	data.theme_owner=p_owner;
	_notification(NOTIFICATION_THEME_CHANGED);
	update();
}

void Control::set_theme(const Ref<Theme>& p_theme) {

	data.theme=p_theme;
	if (!p_theme.is_null()) {

		_propagate_theme_changed(this);
	} else {

		Control *parent = get_parent()?get_parent()->cast_to<Control>():NULL;
		if (parent && parent->data.theme_owner) {
			_propagate_theme_changed(parent->data.theme_owner);
		} else {

			_propagate_theme_changed(NULL);
		}

	}


}

void Control::_window_accept_event() {

	window->key_event_accepted=true;
	if (is_inside_tree())
		get_tree()->set_input_as_handled();

}
void Control::accept_event() {

	if (is_inside_tree() && get_window())
		get_window()->_window_accept_event();

}

Ref<Theme> Control::get_theme() const {

	return data.theme;
}

void Control::set_tooltip(const String& p_tooltip) {

	data.tooltip=p_tooltip;
}
String Control::get_tooltip(const Point2& p_pos) const {

	return data.tooltip;
}

void Control::set_default_cursor_shape(CursorShape p_shape) {

	data.default_cursor=p_shape;
}

Control::CursorShape Control::get_default_cursor_shape() const {

	return data.default_cursor;
}
Control::CursorShape Control::get_cursor_shape(const Point2& p_pos) const {

	return data.default_cursor;
}

Matrix32 Control::get_transform() const {

	Matrix32 xf;
	xf.set_origin(get_pos());
	return xf;
}

String Control::_get_tooltip() const {

	return data.tooltip;
}

void Control::set_focus_neighbour(Margin p_margin, const NodePath &p_neighbour) {

	ERR_FAIL_INDEX(p_margin,4);
	data.focus_neighbour[p_margin]=p_neighbour;
}

NodePath Control::get_focus_neighbour(Margin p_margin) const {

	ERR_FAIL_INDEX_V(p_margin,4,NodePath());
	return data.focus_neighbour[p_margin];
}

#define MAX_NEIGHBOUR_SEARCH_COUNT 512

Control *Control::_get_focus_neighbour(Margin p_margin,int p_count) {

	if (p_count >= MAX_NEIGHBOUR_SEARCH_COUNT)
		return NULL;
	if (!data.focus_neighbour[p_margin].is_empty()) {

		Control *c=NULL;
		Node * n = get_node(data.focus_neighbour[p_margin]);
		if (n) {
			c=n->cast_to<Control>();

			if (!c) {

				ERR_EXPLAIN("Next focus node is not a control: "+n->get_name());
				ERR_FAIL_V(NULL);
			}
		} else {
			return NULL;
		}
		bool valid=true;
		if (c->is_hidden())
			valid=false;
		if (c->get_focus_mode()==FOCUS_NONE)
			valid=false;
		if (valid)
			return c;

		c=c->_get_focus_neighbour(p_margin,p_count+1);
		return c;
	}


	float dist=1e7;
	Control * result=NULL;

	Point2 points[4];

	Matrix32 xform = get_global_transform();
	Rect2 rect = get_item_rect();

	points[0]=xform.xform(rect.pos);
	points[1]=xform.xform(rect.pos + Point2(rect.size.x, 0));
	points[2]=xform.xform(rect.pos + rect.size);
	points[3]=xform.xform(rect.pos + Point2(0, rect.size.y));

	const Vector2 dir[4]={
		Vector2(-1,0),
		Vector2(0,-1),
		Vector2(1,0),
		Vector2(0,1)
	};

	Vector2 vdir=dir[p_margin];

	float maxd=-1e7;

	for(int i=0;i<4;i++) {

		float d = vdir.dot(points[i]);
		if (d>maxd)
			maxd=d;
	}

	Node *base=this;

	while (base) {

		Control *c = base->cast_to<Control>();
		if (c) {
			if (c->data.SI)
				break;
			if (c==data.window)
				break;
		}
		base=base->get_parent();
	}

	if (!base)
		return NULL;

	_window_find_focus_neighbour(vdir,base,points,maxd,dist,&result);

	return result;

}

void Control::_window_find_focus_neighbour(const Vector2& p_dir, Node *p_at,const Point2* p_points,float p_min ,float &r_closest_dist,Control **r_closest) {

	if (p_at->cast_to<Viewport>())
		return; //bye

	Control *c = p_at->cast_to<Control>();

	if (c && c !=this && c->get_focus_mode()==FOCUS_ALL && c->is_visible()) {

		Point2 points[4];

		Matrix32 xform = c->get_global_transform();
		Rect2 rect = c->get_item_rect();

		points[0]=xform.xform(rect.pos);
		points[1]=xform.xform(rect.pos + Point2(rect.size.x, 0));
		points[2]=xform.xform(rect.pos + rect.size);
		points[3]=xform.xform(rect.pos + Point2(0, rect.size.y));


		float min=1e7;

		for(int i=0;i<4;i++) {

			float d = p_dir.dot(points[i]);
			if (d < min)
				min =d;
		}


		if (min>(p_min-CMP_EPSILON)) {

			for(int i=0;i<4;i++) {

				Vector2 la=p_points[i];
				Vector2 lb=p_points[(i+1)%4];

				for(int j=0;j<4;j++) {

					Vector2 fa=points[j];
					Vector2 fb=points[(j+1)%4];

					Vector2 pa,pb;
					float d=Geometry::get_closest_points_between_segments(la,lb,fa,fb,pa,pb);
					//float d = Geometry::get_closest_distance_between_segments(Vector3(la.x,la.y,0),Vector3(lb.x,lb.y,0),Vector3(fa.x,fa.y,0),Vector3(fb.x,fb.y,0));
					if (d<r_closest_dist) {
						r_closest_dist=d;
						*r_closest=c;
					}
				}
			}
		}

	}

	for(int i=0;i<p_at->get_child_count();i++) {

		Node *child=p_at->get_child(i);
		Control *childc = child->cast_to<Control>();
		if (childc && childc->data.SI)
			continue; //subwindow, ignore
		_window_find_focus_neighbour(p_dir,p_at->get_child(i),p_points,p_min,r_closest_dist,r_closest);
	}
}

void Control::set_h_size_flags(int p_flags) {

	if (data.h_size_flags==p_flags)
		return;
	data.h_size_flags=p_flags;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

int Control::get_h_size_flags() const{
	return data.h_size_flags;
}
void Control::set_v_size_flags(int p_flags) {

	if (data.v_size_flags==p_flags)
		return;
	data.v_size_flags=p_flags;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}


void Control::set_stretch_ratio(float p_ratio) {

	if (data.expand==p_ratio)
		return;

	data.expand=p_ratio;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

float Control::get_stretch_ratio() const {

	return data.expand;
}


void Control::grab_click_focus() {

	ERR_FAIL_COND(!is_inside_tree());

	if (data.window && data.window->window->mouse_focus) {

		Window *w=data.window->window;
		if (w->mouse_focus==this)
			return;
		InputEvent ie;
		ie.type=InputEvent::MOUSE_BUTTON;
		InputEventMouseButton &mb=ie.mouse_button;

		//send unclic

		Point2 click =w->mouse_focus->get_global_transform().affine_inverse().xform(w->last_mouse_pos);
		mb.x=click.x;
		mb.y=click.y;
		mb.button_index=w->mouse_focus_button;
		mb.pressed=false;
		w->mouse_focus->call_deferred("_input_event",ie);


		w->mouse_focus=this;
		w->focus_inv_xform=w->mouse_focus->get_global_transform().affine_inverse();
		click =w->mouse_focus->get_global_transform().affine_inverse().xform(w->last_mouse_pos);
		mb.x=click.x;
		mb.y=click.y;
		mb.button_index=w->mouse_focus_button;
		mb.pressed=true;
		w->mouse_focus->call_deferred("_input_event",ie);

	}
}

void Control::minimum_size_changed() {

	if (!is_inside_tree())
		return;

	if (data.pending_min_size_update)
		return;


	data.pending_min_size_update=true;
	MessageQueue::get_singleton()->push_call(this,"_update_minimum_size");

	if (!is_toplevel_control()) {
		Control *pc = get_parent_control();
		if (pc)
			pc->minimum_size_changed();
	}
}

int Control::get_v_size_flags() const{
	return data.v_size_flags;
}

void Control::set_ignore_mouse(bool p_ignore) {

	data.ignore_mouse=p_ignore;
}

bool Control::is_ignoring_mouse() const {

	return data.ignore_mouse;
}

void Control::set_stop_mouse(bool p_stop) {

	data.stop_mouse=p_stop;
}

bool Control::is_stopping_mouse() const {

	return data.stop_mouse;
}

Control *Control::get_focus_owner() const {

	ERR_FAIL_COND_V(!is_inside_tree(),NULL);
	ERR_FAIL_COND_V(!data.window,NULL);
	return data.window->window->key_focus;
}

void Control::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_window_input_event"),&Control::_window_input_event);
	ObjectTypeDB::bind_method(_MD("_gui_input"),&Control::_gui_input);
	ObjectTypeDB::bind_method(_MD("_input_text"),&Control::_input_text);
//	ObjectTypeDB::bind_method(_MD("_window_resize_event"),&Control::_window_resize_event);
	ObjectTypeDB::bind_method(_MD("_window_remove_focus"),&Control::_window_remove_focus);
	ObjectTypeDB::bind_method(_MD("_cancel_input_ID"),&Control::_window_cancel_input_ID);
	ObjectTypeDB::bind_method(_MD("_cancel_tooltip"),&Control::_window_cancel_tooltip);
	ObjectTypeDB::bind_method(_MD("_window_show_tooltip"),&Control::_window_show_tooltip);
	ObjectTypeDB::bind_method(_MD("_size_changed"),&Control::_size_changed);
	ObjectTypeDB::bind_method(_MD("_update_minimum_size"),&Control::_update_minimum_size);

	ObjectTypeDB::bind_method(_MD("accept_event"),&Control::accept_event);
	ObjectTypeDB::bind_method(_MD("get_minimum_size"),&Control::get_minimum_size);
	ObjectTypeDB::bind_method(_MD("get_combined_minimum_size"),&Control::get_combined_minimum_size);
	ObjectTypeDB::bind_method(_MD("is_window"),&Control::is_window);
	ObjectTypeDB::bind_method(_MD("get_window"),&Control::get_window);
	ObjectTypeDB::bind_method(_MD("set_anchor","margin","anchor_mode"),&Control::set_anchor);
	ObjectTypeDB::bind_method(_MD("get_anchor","margin"),&Control::get_anchor);
	ObjectTypeDB::bind_method(_MD("set_margin","margin","offset"),&Control::set_margin);
	ObjectTypeDB::bind_method(_MD("set_anchor_and_margin","margin","anchor_mode","offset"),&Control::set_anchor_and_margin);
	ObjectTypeDB::bind_method(_MD("set_begin","pos"),&Control::set_begin);
	ObjectTypeDB::bind_method(_MD("set_end","pos"),&Control::set_end);
	ObjectTypeDB::bind_method(_MD("set_pos","pos"),&Control::set_pos);
	ObjectTypeDB::bind_method(_MD("set_size","size"),&Control::set_size);
	ObjectTypeDB::bind_method(_MD("set_custom_minimum_size","size"),&Control::set_custom_minimum_size);
	ObjectTypeDB::bind_method(_MD("set_global_pos","pos"),&Control::set_global_pos);
	ObjectTypeDB::bind_method(_MD("get_margin","margin"),&Control::get_margin);
	ObjectTypeDB::bind_method(_MD("get_begin"),&Control::get_begin);
	ObjectTypeDB::bind_method(_MD("get_end"),&Control::get_end);
	ObjectTypeDB::bind_method(_MD("get_pos"),&Control::get_pos);
	ObjectTypeDB::bind_method(_MD("get_size"),&Control::get_size);
	ObjectTypeDB::bind_method(_MD("get_custom_minimum_size"),&Control::get_custom_minimum_size);
	ObjectTypeDB::bind_method(_MD("get_parent_area_size"),&Control::get_size);
	ObjectTypeDB::bind_method(_MD("get_global_pos"),&Control::get_global_pos);
	ObjectTypeDB::bind_method(_MD("get_rect"),&Control::get_rect);
	ObjectTypeDB::bind_method(_MD("get_global_rect"),&Control::get_global_rect);
	ObjectTypeDB::bind_method(_MD("set_area_as_parent_rect","margin"),&Control::set_area_as_parent_rect,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("show_modal","exclusive"),&Control::show_modal,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("set_focus_mode","mode"),&Control::set_focus_mode);
	ObjectTypeDB::bind_method(_MD("has_focus"),&Control::has_focus);
	ObjectTypeDB::bind_method(_MD("grab_focus"),&Control::grab_focus);
	ObjectTypeDB::bind_method(_MD("release_focus"),&Control::release_focus);
	ObjectTypeDB::bind_method(_MD("get_focus_owner:Control"),&Control::get_focus_owner);

	ObjectTypeDB::bind_method(_MD("set_h_size_flags","flags"),&Control::set_h_size_flags);
	ObjectTypeDB::bind_method(_MD("get_h_size_flags"),&Control::get_h_size_flags);

	ObjectTypeDB::bind_method(_MD("set_stretch_ratio","ratio"),&Control::set_stretch_ratio);
	ObjectTypeDB::bind_method(_MD("get_stretch_ratio"),&Control::get_stretch_ratio);

	ObjectTypeDB::bind_method(_MD("set_v_size_flags","flags"),&Control::set_v_size_flags);
	ObjectTypeDB::bind_method(_MD("get_v_size_flags"),&Control::get_v_size_flags);

	ObjectTypeDB::bind_method(_MD("set_theme","theme:Theme"),&Control::set_theme);
	ObjectTypeDB::bind_method(_MD("get_theme:Theme"),&Control::get_theme);

	ObjectTypeDB::bind_method(_MD("add_icon_override","name","texture:Texture"),&Control::add_icon_override);
	ObjectTypeDB::bind_method(_MD("add_style_override","name","stylebox:StyleBox"),&Control::add_style_override);
	ObjectTypeDB::bind_method(_MD("add_font_override","name","font:Font"),&Control::add_font_override);
	ObjectTypeDB::bind_method(_MD("add_color_override","name","color"),&Control::add_color_override);
	ObjectTypeDB::bind_method(_MD("add_constant_override","name","constant"),&Control::add_constant_override);

	ObjectTypeDB::bind_method(_MD("get_icon:Texture","name","type"),&Control::get_icon,DEFVAL(""));
	ObjectTypeDB::bind_method(_MD("get_stylebox:StyleBox","name","type"),&Control::get_stylebox,DEFVAL(""));
	ObjectTypeDB::bind_method(_MD("get_font:Font","name","type"),&Control::get_font,DEFVAL(""));
	ObjectTypeDB::bind_method(_MD("get_color","name","type"),&Control::get_color,DEFVAL(""));
	ObjectTypeDB::bind_method(_MD("get_constant","name","type"),&Control::get_constant,DEFVAL(""));


	ObjectTypeDB::bind_method(_MD("get_parent_control:Control"),&Control::get_parent_control);	

	ObjectTypeDB::bind_method(_MD("set_tooltip","tooltip"),&Control::set_tooltip);
	ObjectTypeDB::bind_method(_MD("get_tooltip","atpos"),&Control::get_tooltip,DEFVAL(Point2()));
	ObjectTypeDB::bind_method(_MD("_get_tooltip"),&Control::_get_tooltip);

	ObjectTypeDB::bind_method(_MD("set_default_cursor_shape","shape"),&Control::set_default_cursor_shape);
	ObjectTypeDB::bind_method(_MD("get_default_cursor_shape"),&Control::get_default_cursor_shape);
	ObjectTypeDB::bind_method(_MD("get_cursor_shape","pos"),&Control::get_cursor_shape,DEFVAL(Point2()));

	ObjectTypeDB::bind_method(_MD("set_focus_neighbour","margin","neighbour"),&Control::set_focus_neighbour);
	ObjectTypeDB::bind_method(_MD("get_focus_neighbour","margin"),&Control::get_focus_neighbour);

	ObjectTypeDB::bind_method(_MD("set_ignore_mouse","ignore"),&Control::set_ignore_mouse);
	ObjectTypeDB::bind_method(_MD("is_ignoring_mouse"),&Control::is_ignoring_mouse);

	ObjectTypeDB::bind_method(_MD("force_drag","data","preview"),&Control::force_drag);

	ObjectTypeDB::bind_method(_MD("set_stop_mouse","stop"),&Control::set_stop_mouse);
	ObjectTypeDB::bind_method(_MD("is_stopping_mouse"),&Control::is_stopping_mouse);

	ObjectTypeDB::bind_method(_MD("grab_click_focus"),&Control::grab_click_focus);

	ObjectTypeDB::bind_method(_MD("set_drag_preview","control:Control"),&Control::set_drag_preview);

	BIND_VMETHOD(MethodInfo("_input_event",PropertyInfo(Variant::INPUT_EVENT,"event")));
	BIND_VMETHOD(MethodInfo(Variant::VECTOR2,"get_minimum_size"));
	BIND_VMETHOD(MethodInfo(Variant::OBJECT,"get_drag_data",PropertyInfo(Variant::VECTOR2,"pos")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL,"can_drop_data",PropertyInfo(Variant::VECTOR2,"pos"),PropertyInfo(Variant::NIL,"data")));
	BIND_VMETHOD(MethodInfo("drop_data",PropertyInfo(Variant::VECTOR2,"pos"),PropertyInfo(Variant::NIL,"data")));

	ADD_PROPERTYINZ( PropertyInfo(Variant::INT,"anchor/left", PROPERTY_HINT_ENUM, "Begin,End,Ratio,Center"), _SCS("set_anchor"),_SCS("get_anchor"), MARGIN_LEFT );
	ADD_PROPERTYINZ( PropertyInfo(Variant::INT,"anchor/top", PROPERTY_HINT_ENUM, "Begin,End,Ratio,Center"), _SCS("set_anchor"),_SCS("get_anchor"), MARGIN_TOP );
	ADD_PROPERTYINZ( PropertyInfo(Variant::INT,"anchor/right", PROPERTY_HINT_ENUM, "Begin,End,Ratio,Center"), _SCS("set_anchor"),_SCS("get_anchor"), MARGIN_RIGHT );
	ADD_PROPERTYINZ( PropertyInfo(Variant::INT,"anchor/bottom", PROPERTY_HINT_ENUM, "Begin,End,Ratio,Center"), _SCS("set_anchor"),_SCS("get_anchor"), MARGIN_BOTTOM );

	ADD_PROPERTYINZ( PropertyInfo(Variant::INT,"margin/left", PROPERTY_HINT_RANGE, "-4096,4096"), _SCS("set_margin"),_SCS("get_margin"), MARGIN_LEFT );
	ADD_PROPERTYINZ( PropertyInfo(Variant::INT,"margin/top", PROPERTY_HINT_RANGE, "-4096,4096"), _SCS("set_margin"),_SCS("get_margin"), MARGIN_TOP );
	ADD_PROPERTYINZ( PropertyInfo(Variant::INT,"margin/right", PROPERTY_HINT_RANGE, "-4096,4096"), _SCS("set_margin"),_SCS("get_margin"), MARGIN_RIGHT );
	ADD_PROPERTYINZ( PropertyInfo(Variant::INT,"margin/bottom", PROPERTY_HINT_RANGE, "-4096,4096"), _SCS("set_margin"),_SCS("get_margin"), MARGIN_BOTTOM );

	ADD_PROPERTYNZ( PropertyInfo(Variant::VECTOR2,"rect/pos", PROPERTY_HINT_NONE, "",PROPERTY_USAGE_EDITOR), _SCS("set_pos"),_SCS("get_pos") );
	ADD_PROPERTYNZ( PropertyInfo(Variant::VECTOR2,"rect/size", PROPERTY_HINT_NONE, "",PROPERTY_USAGE_EDITOR), _SCS("set_size"),_SCS("get_size") );
	ADD_PROPERTYNZ( PropertyInfo(Variant::VECTOR2,"rect/min_size"), _SCS("set_custom_minimum_size"),_SCS("get_custom_minimum_size") );
	ADD_PROPERTYNZ( PropertyInfo(Variant::STRING,"hint/tooltip", PROPERTY_HINT_MULTILINE_TEXT), _SCS("set_tooltip"),_SCS("_get_tooltip") );
	ADD_PROPERTYI( PropertyInfo(Variant::NODE_PATH,"focus_neighbour/left" ), _SCS("set_focus_neighbour"),_SCS("get_focus_neighbour"),MARGIN_LEFT );
	ADD_PROPERTYI( PropertyInfo(Variant::NODE_PATH,"focus_neighbour/top" ), _SCS("set_focus_neighbour"),_SCS("get_focus_neighbour"),MARGIN_TOP );
	ADD_PROPERTYI( PropertyInfo(Variant::NODE_PATH,"focus_neighbour/right" ), _SCS("set_focus_neighbour"),_SCS("get_focus_neighbour"),MARGIN_RIGHT );
	ADD_PROPERTYI( PropertyInfo(Variant::NODE_PATH,"focus_neighbour/bottom" ), _SCS("set_focus_neighbour"),_SCS("get_focus_neighbour"),MARGIN_BOTTOM );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"focus/ignore_mouse"), _SCS("set_ignore_mouse"),_SCS("is_ignoring_mouse") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"focus/stop_mouse"), _SCS("set_stop_mouse"),_SCS("is_stopping_mouse") );

	ADD_PROPERTYNZ( PropertyInfo(Variant::INT,"size_flags/horizontal", PROPERTY_HINT_FLAGS, "Expand,Fill"), _SCS("set_h_size_flags"),_SCS("get_h_size_flags") );
	ADD_PROPERTYNZ( PropertyInfo(Variant::INT,"size_flags/vertical", PROPERTY_HINT_FLAGS, "Expand,Fill"), _SCS("set_v_size_flags"),_SCS("get_v_size_flags") );
	ADD_PROPERTY( PropertyInfo(Variant::INT,"size_flags/stretch_ratio", PROPERTY_HINT_RANGE, "1,128,0.01"), _SCS("set_stretch_ratio"),_SCS("get_stretch_ratio") );
	ADD_PROPERTYNZ( PropertyInfo(Variant::OBJECT,"theme/theme", PROPERTY_HINT_RESOURCE_TYPE, "Theme"), _SCS("set_theme"),_SCS("get_theme") );

	BIND_CONSTANT( ANCHOR_BEGIN );
	BIND_CONSTANT( ANCHOR_END );
	BIND_CONSTANT( ANCHOR_RATIO );	
    BIND_CONSTANT( ANCHOR_CENTER );
	BIND_CONSTANT( FOCUS_NONE );
	BIND_CONSTANT( FOCUS_CLICK );
	BIND_CONSTANT( FOCUS_ALL );


	BIND_CONSTANT( NOTIFICATION_RESIZED );
	BIND_CONSTANT( NOTIFICATION_MOUSE_ENTER );
	BIND_CONSTANT( NOTIFICATION_MOUSE_EXIT );
	BIND_CONSTANT( NOTIFICATION_FOCUS_ENTER );
	BIND_CONSTANT( NOTIFICATION_FOCUS_EXIT );
	BIND_CONSTANT( NOTIFICATION_THEME_CHANGED );
	BIND_CONSTANT( NOTIFICATION_MODAL_CLOSE );

	BIND_CONSTANT( CURSOR_ARROW );
	BIND_CONSTANT( CURSOR_IBEAM );
	BIND_CONSTANT( CURSOR_POINTING_HAND );
	BIND_CONSTANT( CURSOR_CROSS );
	BIND_CONSTANT( CURSOR_WAIT );
	BIND_CONSTANT( CURSOR_BUSY );
	BIND_CONSTANT( CURSOR_DRAG );
	BIND_CONSTANT( CURSOR_CAN_DROP );
	BIND_CONSTANT( CURSOR_FORBIDDEN );
	BIND_CONSTANT( CURSOR_VSIZE );
	BIND_CONSTANT( CURSOR_HSIZE );
	BIND_CONSTANT( CURSOR_BDIAGSIZE );
	BIND_CONSTANT( CURSOR_FDIAGSIZE );
	BIND_CONSTANT( CURSOR_MOVE );
	BIND_CONSTANT( CURSOR_VSPLIT );
	BIND_CONSTANT( CURSOR_HSPLIT );
	BIND_CONSTANT( CURSOR_HELP );

	BIND_CONSTANT( SIZE_EXPAND );
	BIND_CONSTANT( SIZE_FILL );
	BIND_CONSTANT( SIZE_EXPAND_FILL );

	ADD_SIGNAL( MethodInfo("resized") );
	ADD_SIGNAL( MethodInfo("input_event") );
	ADD_SIGNAL( MethodInfo("mouse_enter") );
	ADD_SIGNAL( MethodInfo("mouse_exit") );
	ADD_SIGNAL( MethodInfo("focus_enter") );
	ADD_SIGNAL( MethodInfo("focus_exit") );
	ADD_SIGNAL( MethodInfo("size_flags_changed") );
	ADD_SIGNAL( MethodInfo("minimum_size_changed") );

	
}
Control::Control() {
	
	data.parent=NULL;
	data.window=NULL;
	data.viewport=NULL;
	data.ignore_mouse=false;
	data.stop_mouse=true;
	window=NULL;

	data.SI=NULL;
	data.MI=NULL;
	data.modal=false;
	data.theme_owner=NULL;
	data.modal_exclusive=false;
	data.default_cursor = CURSOR_ARROW;
	data.h_size_flags=SIZE_FILL;
	data.v_size_flags=SIZE_FILL;
	data.expand=1;
	data.pending_min_size_update=false;


	for (int i=0;i<4;i++) {
		data.anchor[i]=ANCHOR_BEGIN;
		data.margin[i]=0;
	}
	data.focus_mode=FOCUS_NONE;
	data.modal_prev_focus_owner=0;




			
}


Control::~Control()
{
}


