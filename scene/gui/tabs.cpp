/*************************************************************************/
/*  tabs.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#include "tabs.h"

#include "message_queue.h"

Size2 Tabs::get_minimum_size() const {


	Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
	Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
	Ref<Font> font = get_font("font");

	Size2 ms(0, MAX( tab_bg->get_minimum_size().height,tab_fg->get_minimum_size().height)+font->get_height() );

//	h+=MIN( get_constant("label_valign_fg"), get_constant("label_valign_bg") );

	for(int i=0;i<tabs.size();i++) {

		Ref<Texture> tex = tabs[i].icon;
		if (tex.is_valid()) {
			ms.height = MAX( ms.height, tex->get_size().height );
			if (tabs[i].text!="")
				ms.width+=get_constant("hseparation");

		}
		ms.width+=font->get_string_size(tabs[i].text).width;
		if (current==i)
			ms.width+=tab_fg->get_minimum_size().width;
		else
			ms.width+=tab_bg->get_minimum_size().width;

		if (tabs[i].right_button.is_valid()) {
			Ref<Texture> rb=tabs[i].right_button;
			Size2 bms = rb->get_size()+get_stylebox("button")->get_minimum_size();
			bms.width+=get_constant("hseparation");

			ms.width+=bms.width;
			ms.height=MAX(bms.height+tab_bg->get_minimum_size().height,ms.height);
		}

		if (tabs[i].close_button.is_valid()) {
			Ref<Texture> cb=tabs[i].close_button;
			Size2 bms = cb->get_size()+get_stylebox("button")->get_minimum_size();
			bms.width+=get_constant("hseparation");

			ms.width+=bms.width;
			ms.height=MAX(bms.height+tab_bg->get_minimum_size().height,ms.height);
		}
	}

	return ms;
}



void Tabs::_input_event(const InputEvent& p_event) {

	if (p_event.type==InputEvent::MOUSE_MOTION) {

		Point2 pos( p_event.mouse_motion.x, p_event.mouse_motion.y );

		int hover_buttons=-1;
		hover=-1;
		for(int i=0;i<tabs.size();i++) {

			// test hovering tab to display close button if policy says so
			if (cb_displaypolicy == SHOW_HOVER) {
				int ofs=tabs[i].ofs_cache;
				int size = tabs[i].ofs_cache;
				if (pos.x >=tabs[i].ofs_cache && pos.x<tabs[i].ofs_cache+tabs[i].size_cache) {
					hover=i;
				}
			}


			// test hovering right button and close button
			if (tabs[i].rb_rect.has_point(pos)) {
				rb_hover=i;
				hover_buttons = i;
				break;
			}
			else if (tabs[i].cb_rect.has_point(pos)) {
				cb_hover=i;
				hover_buttons = i;
				break;
			}



		}

		if (hover_buttons == -1) {	// no hover
			rb_hover= hover_buttons;
			cb_hover= hover_buttons;
		}
		update();

		return;
	}




	if (rb_pressing && p_event.type==InputEvent::MOUSE_BUTTON &&
	    !p_event.mouse_button.pressed &&
	    p_event.mouse_button.button_index==BUTTON_LEFT) {

		if (rb_hover!=-1) {
			//pressed
			emit_signal("right_button_pressed",rb_hover);
		}

		rb_pressing=false;
		update();
	}

	if (cb_pressing && p_event.type==InputEvent::MOUSE_BUTTON &&
		!p_event.mouse_button.pressed &&
		p_event.mouse_button.button_index==BUTTON_LEFT) {

		if (cb_hover!=-1) {
			//pressed
			emit_signal("tab_close",cb_hover);
		}

		cb_pressing=false;
		update();
	}


	if (p_event.type==InputEvent::MOUSE_BUTTON &&
	    p_event.mouse_button.pressed &&
	    p_event.mouse_button.button_index==BUTTON_LEFT) {

		// clicks
		Point2 pos( p_event.mouse_button.x, p_event.mouse_button.y );

		int found=-1;
		for(int i=0;i<tabs.size();i++) {

			if (tabs[i].rb_rect.has_point(pos)) {
				rb_pressing=true;
				update();
				return;
			}

			if (tabs[i].cb_rect.has_point(pos)) {
				cb_pressing=true;
				update();
				return;
			}

			int ofs=tabs[i].ofs_cache;
			int size = tabs[i].ofs_cache;
			if (pos.x >=tabs[i].ofs_cache && pos.x<tabs[i].ofs_cache+tabs[i].size_cache) {

				found=i;
				break;
			}
		}


		if (found!=-1) {

			set_current_tab(found);
			emit_signal("tab_changed",found);
		}
	}

}

void Tabs::_notification(int p_what) {


	switch(p_what) {

		case NOTIFICATION_MOUSE_EXIT: {
			rb_hover=-1;
			cb_hover=-1;
			hover=-1;
			update();
		} break;
		case NOTIFICATION_DRAW: {

			RID ci = get_canvas_item();

			Ref<StyleBox> tab_bg = get_stylebox("tab_bg");
			Ref<StyleBox> tab_fg = get_stylebox("tab_fg");
			Ref<Font> font = get_font("font");
			Color color_fg = get_color("font_color_fg");
			Color color_bg = get_color("font_color_bg");

			int h = get_size().height;

			int label_valign_fg = get_constant("label_valign_fg");
			int label_valign_bg = get_constant("label_valign_bg");

			int w=0;

			int mw = get_minimum_size().width;

			if (tab_align==ALIGN_CENTER) {
				w=(get_size().width-mw)/2;
			} else if (tab_align==ALIGN_RIGHT) {
				w=get_size().width-mw;

			}

			if (w<0) {
				w=0;
			}

			for(int i=0;i<tabs.size();i++) {

				tabs[i].ofs_cache=w;

				String s = tabs[i].text;
				int lsize=0;
				int slen=font->get_string_size(s).width;
				lsize+=slen;

				Ref<Texture> icon;
				if (tabs[i].icon.is_valid()) {
					icon = tabs[i].icon;
					if (icon.is_valid()) {
						lsize+=icon->get_width();
						if (s!="")
							lsize+=get_constant("hseparation");

					}
				}

				if (tabs[i].right_button.is_valid()) {
					Ref<StyleBox> style = get_stylebox("button");
					Ref<Texture> rb=tabs[i].right_button;

					lsize+=get_constant("hseparation");
					lsize+=style->get_margin(MARGIN_LEFT);
					lsize+=rb->get_width();
					lsize+=style->get_margin(MARGIN_RIGHT);

				}

				// Close button
				switch (cb_displaypolicy) {
				case SHOW_ALWAYS: {
					if (tabs[i].close_button.is_valid()) {
						Ref<StyleBox> style = get_stylebox("button");
						Ref<Texture> rb=tabs[i].close_button;

						lsize+=get_constant("hseparation");
						lsize+=style->get_margin(MARGIN_LEFT);
						lsize+=rb->get_width();
						lsize+=style->get_margin(MARGIN_RIGHT);

					}
				} break;
				case SHOW_ACTIVE_ONLY: {
					if (i==current) {
						if (tabs[i].close_button.is_valid()) {
							Ref<StyleBox> style = get_stylebox("button");
							Ref<Texture> rb=tabs[i].close_button;

							lsize+=get_constant("hseparation");
							lsize+=style->get_margin(MARGIN_LEFT);
							lsize+=rb->get_width();
							lsize+=style->get_margin(MARGIN_RIGHT);

						}
					}
				} break;
				case SHOW_HOVER: {
					if (i==current || i==hover) {
						if (tabs[i].close_button.is_valid()) {
							Ref<StyleBox> style = get_stylebox("button");
							Ref<Texture> rb=tabs[i].close_button;

							lsize+=get_constant("hseparation");
							lsize+=style->get_margin(MARGIN_LEFT);
							lsize+=rb->get_width();
							lsize+=style->get_margin(MARGIN_RIGHT);

						}
					}
				} break;
				case SHOW_NEVER:	// by default, never show close button
				default: {
					// do nothing
				} break;

				}


				Ref<StyleBox> sb;
				int va;
				Color col;

				if (i==current) {

					sb=tab_fg;
					va=label_valign_fg;
					col=color_fg;
				} else {
					sb=tab_bg;
					va=label_valign_bg;
					col=color_bg;
				}


				Size2i sb_ms = sb->get_minimum_size();
				Rect2 sb_rect = Rect2( w, 0, lsize+sb_ms.width, h);
				sb->draw(ci, sb_rect );

				w+=sb->get_margin(MARGIN_LEFT);

				if (icon.is_valid()) {

					icon->draw(ci, Point2i( w, sb->get_margin(MARGIN_TOP)+((sb_rect.size.y-sb_ms.y)-icon->get_height())/2 ) );
					if (s!="")
						w+=icon->get_width()+get_constant("hseparation");

				}

				font->draw(ci, Point2i( w, sb->get_margin(MARGIN_TOP)+((sb_rect.size.y-sb_ms.y)-font->get_height())/2+font->get_ascent() ), s, col );

				w+=slen;

				if (tabs[i].right_button.is_valid()) {
					Ref<StyleBox> style = get_stylebox("button");
					Ref<Texture> rb=tabs[i].right_button;

					w+=get_constant("hseparation");

					Rect2 rb_rect;
					rb_rect.size=style->get_minimum_size()+rb->get_size();
					rb_rect.pos.x=w;
					rb_rect.pos.y=sb->get_margin(MARGIN_TOP)+((sb_rect.size.y-sb_ms.y)-(rb_rect.size.y))/2;

					if (rb_hover==i) {
						if (rb_pressing)
							get_stylebox("button_pressed")->draw(ci,rb_rect);
						else
							style->draw(ci,rb_rect);
					}

					w+=style->get_margin(MARGIN_LEFT);

					rb->draw(ci,Point2i( w,rb_rect.pos.y+style->get_margin(MARGIN_TOP) ));
					w+=rb->get_width();
					w+=style->get_margin(MARGIN_RIGHT);
					tabs[i].rb_rect=rb_rect;


				}




				// Close button
				switch (cb_displaypolicy) {
				case SHOW_ALWAYS: {
					if (tabs[i].close_button.is_valid()) {
						Ref<StyleBox> style = get_stylebox("button");
						Ref<Texture> cb=tabs[i].close_button;

						w+=get_constant("hseparation");

						Rect2 cb_rect;
						cb_rect.size=style->get_minimum_size()+cb->get_size();
						cb_rect.pos.x=w;
						cb_rect.pos.y=sb->get_margin(MARGIN_TOP)+((sb_rect.size.y-sb_ms.y)-(cb_rect.size.y))/2;

						if (cb_hover==i) {
							if (cb_pressing)
								get_stylebox("button_pressed")->draw(ci,cb_rect);
							else
								style->draw(ci,cb_rect);
						}

						w+=style->get_margin(MARGIN_LEFT);

						cb->draw(ci,Point2i( w,cb_rect.pos.y+style->get_margin(MARGIN_TOP) ));
						w+=cb->get_width();
						w+=style->get_margin(MARGIN_RIGHT);
						tabs[i].cb_rect=cb_rect;
					}
				} break;
				case SHOW_ACTIVE_ONLY: {
					if (current==i) {
						if (tabs[i].close_button.is_valid()) {
							Ref<StyleBox> style = get_stylebox("button");
							Ref<Texture> cb=tabs[i].close_button;

							w+=get_constant("hseparation");

							Rect2 cb_rect;
							cb_rect.size=style->get_minimum_size()+cb->get_size();
							cb_rect.pos.x=w;
							cb_rect.pos.y=sb->get_margin(MARGIN_TOP)+((sb_rect.size.y-sb_ms.y)-(cb_rect.size.y))/2;

							if (cb_hover==i) {
								if (cb_pressing)
									get_stylebox("button_pressed")->draw(ci,cb_rect);
								else
									style->draw(ci,cb_rect);
							}

							w+=style->get_margin(MARGIN_LEFT);

							cb->draw(ci,Point2i( w,cb_rect.pos.y+style->get_margin(MARGIN_TOP) ));
							w+=cb->get_width();
							w+=style->get_margin(MARGIN_RIGHT);
							tabs[i].cb_rect=cb_rect;
						}
					}
				} break;
				case SHOW_HOVER: {
					if (current==i || hover==i) {
						if (tabs[i].close_button.is_valid()) {
							Ref<StyleBox> style = get_stylebox("button");
							Ref<Texture> cb=tabs[i].close_button;

							w+=get_constant("hseparation");

							Rect2 cb_rect;
							cb_rect.size=style->get_minimum_size()+cb->get_size();
							cb_rect.pos.x=w;
							cb_rect.pos.y=sb->get_margin(MARGIN_TOP)+((sb_rect.size.y-sb_ms.y)-(cb_rect.size.y))/2;

							if (cb_hover==i) {
								if (cb_pressing)
									get_stylebox("button_pressed")->draw(ci,cb_rect);
								else
									style->draw(ci,cb_rect);
							}

							w+=style->get_margin(MARGIN_LEFT);

							cb->draw(ci,Point2i( w,cb_rect.pos.y+style->get_margin(MARGIN_TOP) ));
							w+=cb->get_width();
							w+=style->get_margin(MARGIN_RIGHT);
							tabs[i].cb_rect=cb_rect;
						}
					}
				} break;
				case SHOW_NEVER:
				default: {
					// show nothing
				} break;

				}

				w+=sb->get_margin(MARGIN_RIGHT);

				tabs[i].size_cache=w-tabs[i].ofs_cache;

			}


		} break;
	}
}

int Tabs::get_tab_count() const {


	return tabs.size();
}


void Tabs::set_current_tab(int p_current) {

	ERR_FAIL_INDEX( p_current, get_tab_count() );

    //printf("DEBUG %p: set_current_tab to %i\n", this, p_current);
	current=p_current;	

	_change_notify("current_tab");
	//emit_signal("tab_changed",current);
	update();
}

int Tabs::get_current_tab() const {

	return current;
}


void Tabs::set_tab_title(int p_tab,const String& p_title) {

	ERR_FAIL_INDEX(p_tab,tabs.size());
	tabs[p_tab].text=p_title;
	update();
	minimum_size_changed();

}

String Tabs::get_tab_title(int p_tab) const{

	ERR_FAIL_INDEX_V(p_tab,tabs.size(),"");
	return tabs[p_tab].text;


}

void Tabs::set_tab_icon(int p_tab,const Ref<Texture>& p_icon){

	ERR_FAIL_INDEX(p_tab,tabs.size());
	tabs[p_tab].icon=p_icon;
	update();
	minimum_size_changed();

}
Ref<Texture> Tabs::get_tab_icon(int p_tab) const{

	ERR_FAIL_INDEX_V(p_tab,tabs.size(),Ref<Texture>());
	return tabs[p_tab].icon;

}



void Tabs::set_tab_right_button(int p_tab,const Ref<Texture>& p_right_button){

	ERR_FAIL_INDEX(p_tab,tabs.size());
	tabs[p_tab].right_button=p_right_button;
	update();
	minimum_size_changed();

}
Ref<Texture> Tabs::get_tab_right_button(int p_tab) const{

	ERR_FAIL_INDEX_V(p_tab,tabs.size(),Ref<Texture>());
	return tabs[p_tab].right_button;

}

void Tabs::set_tab_close_button(int p_tab, const Ref<Texture>& p_close_button) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs[p_tab].close_button=p_close_button;
	update();
	minimum_size_changed();
}


Ref<Texture> Tabs::get_tab_close_button(int p_tab) const{

	ERR_FAIL_INDEX_V(p_tab,tabs.size(),Ref<Texture>());
	return tabs[p_tab].close_button;

}

void Tabs::add_tab(const String& p_str,const Ref<Texture>& p_icon) {

	Tab t;
	t.text=p_str;
	t.icon=p_icon;

	t.close_button = get_icon("Close","EditorIcons");

	tabs.push_back(t);

	update();
	minimum_size_changed();

}

void Tabs::clear_tabs() {
	tabs.clear();
	current=0;
	update();
}

void Tabs::remove_tab(int p_idx) {

	ERR_FAIL_INDEX(p_idx,tabs.size());
	tabs.remove(p_idx);
	if (current>=p_idx)
		current--;
	update();
	minimum_size_changed();

	if (current<0)
		current=0;
	if (current>=tabs.size())
		current=tabs.size()-1;

	//emit_signal("tab_changed",current);

}

void Tabs::set_tab_close_display_policy(CloseButtonDisplayPolicy p_cb_displaypolicy) {
	cb_displaypolicy = p_cb_displaypolicy;
}


void Tabs::set_tab_align(TabAlign p_align) {

	tab_align=p_align;
	update();
}

Tabs::TabAlign Tabs::get_tab_align() const {

	return tab_align;
}


void Tabs::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_input_event"),&Tabs::_input_event);
	ObjectTypeDB::bind_method(_MD("get_tab_count"),&Tabs::get_tab_count);
	ObjectTypeDB::bind_method(_MD("set_current_tab","tab_idx"),&Tabs::set_current_tab);
	ObjectTypeDB::bind_method(_MD("get_current_tab"),&Tabs::get_current_tab);
	ObjectTypeDB::bind_method(_MD("set_tab_title","tab_idx","title"),&Tabs::set_tab_title);
	ObjectTypeDB::bind_method(_MD("get_tab_title","tab_idx"),&Tabs::get_tab_title);
	ObjectTypeDB::bind_method(_MD("set_tab_icon","tab_idx","icon:Texture"),&Tabs::set_tab_icon);
	ObjectTypeDB::bind_method(_MD("get_tab_icon:Texture","tab_idx"),&Tabs::get_tab_icon);
	ObjectTypeDB::bind_method(_MD("remove_tab","tab_idx"),&Tabs::remove_tab);
	ObjectTypeDB::bind_method(_MD("add_tab","title","icon:Texture"),&Tabs::add_tab);
	ObjectTypeDB::bind_method(_MD("set_tab_align","align"),&Tabs::set_tab_align);
	ObjectTypeDB::bind_method(_MD("get_tab_align"),&Tabs::get_tab_align);

	ADD_SIGNAL(MethodInfo("tab_changed",PropertyInfo(Variant::INT,"tab")));
	ADD_SIGNAL(MethodInfo("right_button_pressed",PropertyInfo(Variant::INT,"tab")));
	ADD_SIGNAL(MethodInfo("tab_close",PropertyInfo(Variant::INT,"tab")));


	ADD_PROPERTY( PropertyInfo(Variant::INT, "current_tab", PROPERTY_HINT_RANGE,"-1,4096,1",PROPERTY_USAGE_EDITOR), _SCS("set_current_tab"), _SCS("get_current_tab") );

	BIND_CONSTANT( ALIGN_LEFT );
	BIND_CONSTANT( ALIGN_CENTER );
	BIND_CONSTANT( ALIGN_RIGHT );

	BIND_CONSTANT( SHOW_ACTIVE_ONLY );
	BIND_CONSTANT( SHOW_ALWAYS );
	BIND_CONSTANT( SHOW_HOVER );
	BIND_CONSTANT( SHOW_NEVER );
}


Tabs::Tabs() {

	current=0;
	tab_align=ALIGN_CENTER;
	rb_hover=-1;
	rb_pressing=false;

	cb_hover=-1;
	cb_pressing=false;
	cb_displaypolicy = SHOW_NEVER; // Default : no close button
}
