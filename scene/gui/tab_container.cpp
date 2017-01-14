/*************************************************************************/
/*  tab_container.cpp                                                    */
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
#include "tab_container.h"

#include "message_queue.h"



int TabContainer::_get_top_margin() const {

	Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
	Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
	Ref<Font> font = get_font("font");

	int h = MAX( tab_bg->get_minimum_size().height,tab_fg->get_minimum_size().height);

	int ch = font->get_height();
	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (!c->has_meta("_tab_icon"))
			continue;

		Ref<Texture> tex = c->get_meta("_tab_icon");
		if (!tex.is_valid())
			continue;
		ch = MAX( ch, tex->get_size().height );
	}

	h+=ch;

	return h;

}



void TabContainer::_gui_input(const InputEvent& p_event) {

	if (p_event.type==InputEvent::MOUSE_BUTTON &&
	    p_event.mouse_button.pressed &&
	    p_event.mouse_button.button_index==BUTTON_LEFT) {

		// clicks
		Point2 pos( p_event.mouse_button.x, p_event.mouse_button.y );

		int top_margin = _get_top_margin();
		if (pos.y>top_margin)
			return; // no click (too far down)

		if (pos.x<tabs_ofs_cache)
			return; // no click (too far left)

		Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
		Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
		Ref<Font> font = get_font("font");
		Ref<Texture> incr = get_icon("increment");
		Ref<Texture> decr = get_icon("decrement");
		Ref<Texture> menu = get_icon("menu");
		Ref<Texture> menu_hl = get_icon("menu_hl");

		if (popup && pos.x>get_size().width-menu->get_width()) {


			emit_signal("pre_popup_pressed");
			Vector2 pp_pos = get_global_pos();
			pp_pos.x+=get_size().width;
			pp_pos.x-=popup->get_size().width;
			pp_pos.y+=menu->get_height();

			popup->set_global_pos( pp_pos );
			popup->popup();
			return;
		}
		pos.x-=tabs_ofs_cache;

		int idx=0;
		int found=-1;
		bool rightroom=false;

		for(int i=0;i<get_child_count();i++) {

			Control *c = get_child(i)->cast_to<Control>();
			if (!c)
				continue;
			if (c->is_set_as_toplevel())
				continue;

			if (idx<tab_display_ofs) {
				idx++;
				continue;
			}

			if (idx>last_tab_cache) {
				rightroom=true;
				break;
			}

			String s = c->has_meta("_tab_name")?String(XL_MESSAGE(String(c->get_meta("_tab_name")))):String(c->get_name());
			int tab_width=font->get_string_size(s).width;

			if (c->has_meta("_tab_icon")) {
				Ref<Texture> icon = c->get_meta("_tab_icon");
				if (icon.is_valid()) {
					tab_width+=icon->get_width();
					if (s!="")
						tab_width+=get_constant("hseparation");

				}
			}

			if (idx==current) {

				tab_width+=tab_fg->get_minimum_size().width;
			} else {
				tab_width+=tab_bg->get_minimum_size().width;
			}

			if (pos.x < tab_width) {

				found=idx;
				break;
			}

			pos.x-=tab_width;
			idx++;
		}

		if (buttons_visible_cache) {

			if (p_event.mouse_button.x>get_size().width-incr->get_width()) {
				if (rightroom) {
					tab_display_ofs+=1;
					update();
				}
			} else if (p_event.mouse_button.x>get_size().width-incr->get_width()-decr->get_width()) {

				if (tab_display_ofs>0) {
					tab_display_ofs-=1;
					update();
				}

			}
		}


		if (found!=-1) {

			set_current_tab(found);
		}
	}

}

void TabContainer::_notification(int p_what) {


	switch(p_what) {


		case NOTIFICATION_DRAW: {

			RID ci = get_canvas_item();
			Ref<StyleBox> panel = get_stylebox("panel");
			Size2 size = get_size();

			if (!tabs_visible) {

				panel->draw(ci, Rect2( 0, 0, size.width, size.height));
				return;
			}



			Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
			Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
			Ref<Texture> incr = get_icon("increment");
			Ref<Texture> decr = get_icon("decrement");
			Ref<Texture> menu = get_icon("menu");
			Ref<Texture> menu_hl = get_icon("menu_hl");
			Ref<Font> font = get_font("font");
			Color color_fg = get_color("font_color_fg");
			Color color_bg = get_color("font_color_bg");

			int side_margin = get_constant("side_margin");
			int top_margin = _get_top_margin();


			Size2 top_size = Size2( size.width, top_margin );



			int w=0;
			int idx=0;
			Vector<int> offsets;
			Vector<Control*> controls;
			int from=0;
			int limit=get_size().width;
			if (popup) {
				top_size.width-=menu->get_width();
				limit-=menu->get_width();
			}

			bool notdone=false;
			last_tab_cache=-1;

			for(int i=0;i<get_child_count();i++) {

				Control *c = get_child(i)->cast_to<Control>();
				if (!c)
					continue;
				if (c->is_set_as_toplevel())
					continue;
				if (idx<tab_display_ofs) {
					idx++;
					from=idx;
					continue;
				}

				if (w>=get_size().width) {
					buttons_visible_cache=true;
					notdone=true;
					break;
				}

				offsets.push_back(w);
				controls.push_back(c);

				String s = c->has_meta("_tab_name")?String(XL_MESSAGE(String(c->get_meta("_tab_name")))):String(c->get_name());
				w+=font->get_string_size(s).width;
				if (c->has_meta("_tab_icon")) {
					Ref<Texture> icon = c->get_meta("_tab_icon");
					if (icon.is_valid()) {
						w+=icon->get_width();
						if (s!="")
						     w+=get_constant("hseparation");

					}
				}

				if (idx==current) {

					w+=tab_fg->get_minimum_size().width;
				} else {
					w+=tab_bg->get_minimum_size().width;
				}

				if (idx<tab_display_ofs) {

				}
				last_tab_cache=idx;

				idx++;
			}


			int ofs;

			switch(align) {

				case ALIGN_LEFT: ofs = side_margin; break;
				case ALIGN_CENTER: ofs = (int(limit) - w)/2; break;
				case ALIGN_RIGHT: ofs = int(limit) - w - side_margin; break;
			};

			tab_display_ofs=0;


			tabs_ofs_cache=ofs;
			idx=0;



			for(int i=0;i<controls.size();i++) {

				idx=i+from;
				if (current>=from && current<from+controls.size()-1) {
					//current is visible! draw it last.
					if (i==controls.size()-1) {
						idx=current;
					} else if (idx>=current) {
						idx+=1;
					}
				}

				Control *c = controls[idx-from];

				String s = c->has_meta("_tab_name")?String(c->get_meta("_tab_name")):String(c->get_name());
				int w=font->get_string_size(s).width;
				Ref<Texture> icon;
				if (c->has_meta("_tab_icon")) {
					icon = c->get_meta("_tab_icon");
					if (icon.is_valid()) {

						w+=icon->get_width();
						if (s!="")
							w+=get_constant("hseparation");

					}
				}


				Ref<StyleBox> sb;
				Color col;

				if (idx==current) {

					sb=tab_fg;
					col=color_fg;
				} else {
					sb=tab_bg;
					col=color_bg;
				}

				int lofs = ofs + offsets[idx-from];

				Size2i sb_ms = sb->get_minimum_size();
				Rect2 sb_rect = Rect2( lofs, 0, w+sb_ms.width, top_margin);


				sb->draw(ci, sb_rect );

				Point2i lpos = sb_rect.pos;
				lpos.x+=sb->get_margin(MARGIN_LEFT);
				if (icon.is_valid()) {

					icon->draw(ci, Point2i( lpos.x, sb->get_margin(MARGIN_TOP)+((sb_rect.size.y-sb_ms.y)-icon->get_height())/2 ) );
					if (s!="")
						lpos.x+=icon->get_width()+get_constant("hseparation");

				}

				font->draw(ci, Point2i( lpos.x, sb->get_margin(MARGIN_TOP)+((sb_rect.size.y-sb_ms.y)-font->get_height())/2+font->get_ascent() ), s, col );

				idx++;
			}


			if (buttons_visible_cache) {

				int vofs = (top_margin-incr->get_height())/2;
				decr->draw(ci,Point2(limit,vofs),Color(1,1,1,tab_display_ofs==0?0.5:1.0));
				incr->draw(ci,Point2(limit+incr->get_width(),vofs),Color(1,1,1,notdone?1.0:0.5));
			}

			if (popup) {
				int from = get_size().width-menu->get_width();

				if (mouse_x_cache > from)
					menu_hl->draw(get_canvas_item(),Size2(from,0));
				else
					menu->draw(get_canvas_item(),Size2(from,0));
			}

			panel->draw(ci, Rect2( 0, top_size.height, size.width, size.height-top_size.height));

		} break;
		case NOTIFICATION_THEME_CHANGED: {
			if (get_tab_count() > 0) {
				call_deferred("set_current_tab",get_current_tab()); //wait until all changed theme
			}
		} break;
	}
}

void TabContainer::_child_renamed_callback() {

	update();
}

void TabContainer::add_child_notify(Node *p_child) {

	Control::add_child_notify(p_child);

	Control *c = p_child->cast_to<Control>();
	if (!c)
		return;
	if (c->is_set_as_toplevel())
		return;

	bool first=false;

	if (get_tab_count()!=1)
		c->hide();
	else {
		c->show();
		//call_deferred("set_current_tab",0);
		first=true;
		current=0;
	}
	c->set_area_as_parent_rect();
	if (tabs_visible)
		c->set_margin(MARGIN_TOP,_get_top_margin());
	Ref<StyleBox> sb = get_stylebox("panel");
	for(int i=0;i<4;i++)
		c->set_margin(Margin(i),c->get_margin(Margin(i))+sb->get_margin(Margin(i)));


	update();
	p_child->connect("renamed", this,"_child_renamed_callback");
	if(first)
		emit_signal("tab_changed",current);
}

int TabContainer::get_tab_count() const {

	int count=0;

	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		count++;
	}

	return count;
}


void TabContainer::set_current_tab(int p_current) {

	ERR_FAIL_INDEX( p_current, get_tab_count() );

	current=p_current;

	int idx=0;

	Ref<StyleBox> sb=get_stylebox("panel");
	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (idx==current) {
			c->show();
			c->set_area_as_parent_rect();
			if (tabs_visible)
				c->set_margin(MARGIN_TOP,_get_top_margin());
			for(int i=0;i<4;i++)
				c->set_margin(Margin(i),c->get_margin(Margin(i))+sb->get_margin(Margin(i)));


		} else
			c->hide();
		idx++;
	}

	_change_notify("current_tab");
	emit_signal("tab_changed",current);
	update();
}

int TabContainer::get_current_tab() const {

	return current;
}

Control* TabContainer::get_tab_control(int p_idx) const {

	int idx=0;


	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (idx==p_idx) {
			return c;

		}
		idx++;
	}

	return NULL;
}
Control* TabContainer::get_current_tab_control() const {

	int idx=0;


	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (idx==current) {
			return c;

		}
		idx++;
	}

	return NULL;
}

void TabContainer::remove_child_notify(Node *p_child) {

	Control::remove_child_notify(p_child);

	int tc = get_tab_count();
	if (current==tc-1) {
		current--;
		if (current<0)
			current=0;
		else {
			call_deferred("set_current_tab",current);
		}
	}

	p_child->disconnect("renamed", this,"_child_renamed_callback");

	update();
}

void TabContainer::set_tab_align(TabAlign p_align) {

	ERR_FAIL_INDEX(p_align,3);
	align=p_align;
	update();

	_change_notify("tab_align");
}
TabContainer::TabAlign TabContainer::get_tab_align() const {

	return align;
}

void TabContainer::set_tabs_visible(bool p_visibe) {

	if (p_visibe==tabs_visible)
		return;

	tabs_visible=p_visibe;

	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (p_visibe)
			c->set_margin(MARGIN_TOP,_get_top_margin());
		else
			c->set_margin(MARGIN_TOP,0);

	}
	update();
}

bool TabContainer::are_tabs_visible() const {

	return tabs_visible;

}


Control *TabContainer::_get_tab(int p_idx) const {

	int idx=0;

	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (idx==p_idx)
			return c;
		idx++;

	}
	return NULL;

}

void TabContainer::set_tab_title(int p_tab,const String& p_title) {

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND(!child);
	child->set_meta("_tab_name",p_title);

}

String TabContainer::get_tab_title(int p_tab) const{

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND_V(!child,"");
	if (child->has_meta("_tab_name"))
		return child->get_meta("_tab_name");
	else
		return child->get_name();

}

void TabContainer::set_tab_icon(int p_tab,const Ref<Texture>& p_icon){

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND(!child);
	child->set_meta("_tab_icon",p_icon);

}
Ref<Texture> TabContainer::get_tab_icon(int p_tab) const{

	Control *child = _get_tab(p_tab);
	ERR_FAIL_COND_V(!child,Ref<Texture>());
	if (child->has_meta("_tab_icon"))
		return child->get_meta("_tab_icon");
	else
		return Ref<Texture>();
}

void TabContainer::get_translatable_strings(List<String> *p_strings) const {

	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;

		if (!c->has_meta("_tab_name"))
			continue;

		String name = c->get_meta("_tab_name");

		if (name!="")
			p_strings->push_back(name);
	}
}


Size2 TabContainer::get_minimum_size() const {

	Size2 ms;

	for(int i=0;i<get_child_count();i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;

		if (!c->is_visible_in_tree())
			continue;

		Size2 cms = c->get_combined_minimum_size();
		ms.x=MAX(ms.x,cms.x);
		ms.y=MAX(ms.y,cms.y);
	}

	Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
	Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
	Ref<Font> font = get_font("font");

	ms.y+=MAX(tab_bg->get_minimum_size().y,tab_fg->get_minimum_size().y);
	ms.y+=font->get_height();

	Ref<StyleBox> sb = get_stylebox("panel");
	ms+=sb->get_minimum_size();

	return ms;
}

void TabContainer::set_popup(Node *p_popup) {
	ERR_FAIL_NULL(p_popup);
	popup=p_popup->cast_to<Popup>();
	update();
}

Popup* TabContainer::get_popup() const {
	return popup;
}


void TabContainer::_bind_methods() {

	ClassDB::bind_method(_MD("_gui_input"),&TabContainer::_gui_input);
	ClassDB::bind_method(_MD("get_tab_count"),&TabContainer::get_tab_count);
	ClassDB::bind_method(_MD("set_current_tab","tab_idx"),&TabContainer::set_current_tab);
	ClassDB::bind_method(_MD("get_current_tab"),&TabContainer::get_current_tab);
	ClassDB::bind_method(_MD("get_current_tab_control:Control"),&TabContainer::get_current_tab_control);
	ClassDB::bind_method(_MD("get_tab_control:Control","idx"),&TabContainer::get_tab_control);
	ClassDB::bind_method(_MD("set_tab_align","align"),&TabContainer::set_tab_align);
	ClassDB::bind_method(_MD("get_tab_align"),&TabContainer::get_tab_align);
	ClassDB::bind_method(_MD("set_tabs_visible","visible"),&TabContainer::set_tabs_visible);
	ClassDB::bind_method(_MD("are_tabs_visible"),&TabContainer::are_tabs_visible);
	ClassDB::bind_method(_MD("set_tab_title","tab_idx","title"),&TabContainer::set_tab_title);
	ClassDB::bind_method(_MD("get_tab_title","tab_idx"),&TabContainer::get_tab_title);
	ClassDB::bind_method(_MD("set_tab_icon","tab_idx","icon:Texture"),&TabContainer::set_tab_icon);
	ClassDB::bind_method(_MD("get_tab_icon:Texture","tab_idx"),&TabContainer::get_tab_icon);
	ClassDB::bind_method(_MD("set_popup","popup:Popup"),&TabContainer::set_popup);
	ClassDB::bind_method(_MD("get_popup:Popup"),&TabContainer::get_popup);

	ClassDB::bind_method(_MD("_child_renamed_callback"),&TabContainer::_child_renamed_callback);

	ADD_SIGNAL(MethodInfo("tab_changed",PropertyInfo(Variant::INT,"tab")));
	ADD_SIGNAL(MethodInfo("pre_popup_pressed"));

	ADD_PROPERTY( PropertyInfo(Variant::INT, "tab_align", PROPERTY_HINT_ENUM,"Left,Center,Right"), _SCS("set_tab_align"), _SCS("get_tab_align") );
	ADD_PROPERTY( PropertyInfo(Variant::INT, "current_tab", PROPERTY_HINT_RANGE,"-1,4096,1",PROPERTY_USAGE_EDITOR), _SCS("set_current_tab"), _SCS("get_current_tab") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "tabs_visible"), _SCS("set_tabs_visible"), _SCS("are_tabs_visible") );

}

TabContainer::TabContainer() {

	tab_display_ofs=0;
	buttons_visible_cache=false;
	tabs_ofs_cache=0;
	current=0;
	mouse_x_cache=0;
	align=ALIGN_CENTER;
	tabs_visible=true;
	popup=NULL;

}
