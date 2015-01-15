/*************************************************/
/*  default_theme.cpp                            */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#include "default_theme.h"

#include "scene/resources/theme.h"

#include "theme_data.h"
#include "os/os.h"


#include "normal_font.inc"
#include "bold_font.inc"
#include "mono_font.inc"

#include "font_normal.inc"
#include "font_source.inc"
#include "font_large.inc"

typedef Map<const void*,Ref<ImageTexture> > TexCacheMap;

static  TexCacheMap *tex_cache;

template<class T>
static Ref<StyleBoxTexture> make_stylebox(T p_src,float p_left, float p_top, float p_right, float p_botton,float p_margin_left=-1, float p_margin_top=-1, float p_margin_right=-1, float p_margin_botton=-1, bool p_draw_center=true) {
	
	Ref<ImageTexture> texture;


	if (tex_cache->has(p_src)) {
		texture=(*tex_cache)[p_src];
	} else {

		texture = Ref<ImageTexture>( memnew( ImageTexture ) );
		texture->create_from_image( Image(p_src),ImageTexture::FLAG_FILTER );
		(*tex_cache)[p_src]=texture;
	}


	Ref<StyleBoxTexture> style( memnew( StyleBoxTexture ) );
	style->set_texture(texture);
	style->set_margin_size( MARGIN_LEFT, p_left );
	style->set_margin_size( MARGIN_RIGHT, p_right );
	style->set_margin_size( MARGIN_BOTTOM, p_botton );
	style->set_margin_size( MARGIN_TOP, p_top );
	style->set_default_margin( MARGIN_LEFT, p_margin_left );
	style->set_default_margin( MARGIN_RIGHT, p_margin_right );
	style->set_default_margin( MARGIN_BOTTOM, p_margin_botton );
	style->set_default_margin( MARGIN_TOP, p_margin_top );
	style->set_draw_center(p_draw_center);
	
	return style;
}

template<class T>
static Ref<Texture> make_icon(T p_src) {
	
	
	Ref<ImageTexture> texture( memnew( ImageTexture ) );
	texture->create_from_image( Image(p_src),ImageTexture::FLAG_FILTER );
		
	return texture;
}

static Ref<Font> make_font(int p_height,int p_ascent, int p_valign, int p_charcount, const int *p_chars,const Ref<Texture> &p_texture) {


	Ref<Font> font( memnew( Font ) );
	font->add_texture( p_texture );
	
	for (int i=0;i<p_charcount;i++) {

		const int *c = &p_chars[i*8];

		int chr=c[0];
		Rect2 frect;
		frect.pos.x=c[1];
		frect.pos.y=c[2];
		frect.size.x=c[3];
		frect.size.y=c[4];
		Point2 align( c[5], c[6]+p_valign);
		int advance=c[7];


		font->add_char( chr, 0, frect, align,advance );
		
	}
	
	font->set_height( p_height );
	font->set_ascent( p_ascent );

	return font;
}

static Ref<Font> make_font2(int p_height,int p_ascent, int p_charcount, const int *p_char_rects,int p_kerning_count,const int *p_kernings,int p_w, int p_h, const unsigned char *p_img) {


	Ref<Font> font( memnew( Font ) );

	DVector<uint8_t> img;
	img.resize(p_w*p_h*2);
	{
		DVector<uint8_t>::Write w = img.write();
		for(int i=0;i<(p_w*p_h*2);i++) {
			w[i]=p_img[i];
		}
	}

	Image image(p_w,p_h,0,Image::FORMAT_GRAYSCALE_ALPHA,img);
	Ref<ImageTexture> tex = memnew( ImageTexture );
	tex->create_from_image(image);

	font->add_texture( tex );

	for (int i=0;i<p_charcount;i++) {

		const int *c = &p_char_rects[i*8];

		int chr=c[0];
		Rect2 frect;
		frect.pos.x=c[1];
		frect.pos.y=c[2];
		frect.size.x=c[3];
		frect.size.y=c[4];
		Point2 align( c[6], c[5]);
		int advance=c[7];

		font->add_char( chr, 0, frect, align,advance );

	}

	for(int i=0;i<p_kerning_count;i++) {

		font->add_kerning_pair(p_kernings[i*3+0],p_kernings[i*3+1],p_kernings[i*3+2]);
	}

	font->set_height( p_height );
	font->set_ascent( p_ascent );

	return font;
}


static Ref<StyleBox> make_empty_stylebox(float p_margin_left=-1, float p_margin_top=-1, float p_margin_right=-1, float p_margin_botton=-1) {

	Ref<StyleBox> style( memnew( StyleBoxEmpty) );
	
	style->set_default_margin( MARGIN_LEFT, p_margin_left );
	style->set_default_margin( MARGIN_RIGHT, p_margin_right );
	style->set_default_margin( MARGIN_BOTTOM, p_margin_botton );
	style->set_default_margin( MARGIN_TOP, p_margin_top );
	
	return style;
}

#ifndef DEFAULT_THEME_DISABLED

void make_default_theme() {


	tex_cache = memnew( TexCacheMap );

	Ref<Theme> t( memnew( Theme ) );

	//Ref<Font> default_font = make_font(_bi_font_normal_height,_bi_font_normal_ascent,_bi_font_normal_valign,_bi_font_normal_charcount,_bi_font_normal_characters,make_icon(font_normal_png));
	Ref<Font> default_font=make_font2(_builtin_normal_font_height,_builtin_normal_font_ascent,_builtin_normal_font_charcount,&_builtin_normal_font_charrects[0][0],_builtin_normal_font_kerning_pair_count,&_builtin_normal_font_kerning_pairs[0][0],_builtin_normal_font_img_width,_builtin_normal_font_img_height,_builtin_normal_font_img_data);
	Ref<Font> source_font=make_font2(_builtin_source_font_height,_builtin_source_font_ascent,_builtin_source_font_charcount,&_builtin_source_font_charrects[0][0],_builtin_source_font_kerning_pair_count,&_builtin_source_font_kerning_pairs[0][0],_builtin_source_font_img_width,_builtin_source_font_img_height,_builtin_source_font_img_data);
	Ref<Font> large_font=make_font2(_builtin_large_font_height,_builtin_large_font_ascent,_builtin_large_font_charcount,&_builtin_large_font_charrects[0][0],_builtin_large_font_kerning_pair_count,&_builtin_large_font_kerning_pairs[0][0],_builtin_large_font_img_width,_builtin_large_font_img_height,_builtin_large_font_img_data);

	// Font Colors

	Color control_font_color = Color::html("e0e0e0");
	Color control_font_color_low = Color::html("b0b0b0");
	Color control_font_color_hover = Color::html("f0f0f0");
	Color control_font_color_disabled = Color(0.9,0.9,0.9,0.2);
	Color control_font_color_pressed = Color::html("ffffff");
	Color font_color_selection = Color::html("7d7d7d");


	// Panel

	t->set_stylebox("panel","Panel", make_stylebox( panel_bg_png,0,0,0,0) );



	// Focus

	Ref<StyleBoxTexture> focus = make_stylebox( focus_png,5,5,5,5);
	for(int i=0;i<4;i++) {
		focus->set_expand_margin_size(Margin(i),2);
	}



	// Button

	t->set_stylebox("normal","Button", make_stylebox( button_normal_png,4,4,4,4,8,4,8,4) );
	t->set_stylebox("pressed","Button", make_stylebox( button_pressed_png,4,4,4,4) );
	t->set_stylebox("hover","Button", make_stylebox( button_hover_png,4,4,4,4) );
	t->set_stylebox("disabled","Button", make_stylebox( button_disabled_png,4,4,4,4) );
	t->set_stylebox("focus","Button", focus );

	t->set_font("font","Button", default_font );

	t->set_color("font_color","Button", control_font_color );
	t->set_color("font_color_pressed","Button", control_font_color_pressed );
	t->set_color("font_color_hover","Button", control_font_color_hover );
	t->set_color("font_color_disabled","Button", control_font_color_disabled );

	t->set_constant("hseparation","Button", 2);



	// ColorPickerButton

	t->set_stylebox("normal","ColorPickerButton", make_stylebox( button_normal_png,4,4,4,4) );
	t->set_stylebox("pressed","ColorPickerButton", make_stylebox( button_pressed_png,4,4,4,4) );
	t->set_stylebox("hover","ColorPickerButton", make_stylebox( button_hover_png,4,4,4,4) );
	t->set_stylebox("disabled","ColorPickerButton", make_stylebox( button_disabled_png,4,4,4,4) );
	t->set_stylebox("focus","ColorPickerButton", focus );

	t->set_font("font","ColorPickerButton", default_font );

	t->set_color("font_color","ColorPickerButton", Color(1,1,1,1) );
	t->set_color("font_color_pressed","ColorPickerButton", Color(0.8,0.8,0.8,1) );
	t->set_color("font_color_hover","ColorPickerButton", Color(1,1,1,1) );
	t->set_color("font_color_disabled","ColorPickerButton", Color(0.9,0.9,0.9,0.3) );

	t->set_constant("hseparation","ColorPickerButton", 2 );


	// ToolButton

	Ref<StyleBox> tb_empty = memnew( StyleBoxEmpty );
	tb_empty->set_default_margin(MARGIN_LEFT,8);
	tb_empty->set_default_margin(MARGIN_RIGHT,8);
	tb_empty->set_default_margin(MARGIN_TOP,4);
	tb_empty->set_default_margin(MARGIN_BOTTOM,4);

	t->set_stylebox("normal","ToolButton", tb_empty);
	t->set_stylebox("pressed","ToolButton", make_stylebox( button_pressed_png,4,4,4,4) );
	t->set_stylebox("hover","ToolButton", make_stylebox( button_normal_png,4,4,4,4) );
	t->set_stylebox("disabled","ToolButton", make_empty_stylebox(4,4,4,4) );
	t->set_stylebox("focus","ToolButton", focus );

	t->set_font("font","ToolButton", default_font );

	t->set_color("font_color","ToolButton", control_font_color );
	t->set_color("font_color_pressed","ToolButton", control_font_color_pressed );
	t->set_color("font_color_hover","ToolButton", control_font_color_hover );
	t->set_color("font_color_disabled","ToolButton", Color(0.9,0.95,1,0.3) );

	t->set_constant("hseparation","ToolButton", 0 );



	// OptionButton

	t->set_stylebox("normal","OptionButton", make_stylebox( option_button_normal_png,5,5,21,5,8,4,8,4) );
	t->set_stylebox("pressed","OptionButton", make_stylebox( option_button_pressed_png,5,5,21,5) );
	t->set_stylebox("hover","OptionButton", make_stylebox( option_button_hover_png,5,5,21,5) );
	t->set_stylebox("disabled","OptionButton", make_stylebox( option_button_disabled_png,5,5,21,5) );
	t->set_stylebox("focus","OptionButton", focus );

	t->set_icon("arrow","OptionButton", make_icon( option_arrow_png ) );

	t->set_font("font","OptionButton", default_font );

	t->set_color("font_color","OptionButton", control_font_color );
	t->set_color("font_color_pressed","OptionButton", control_font_color_pressed );
	t->set_color("font_color_hover","OptionButton", control_font_color_hover );
	t->set_color("font_color_disabled","OptionButton", control_font_color_disabled );

	t->set_constant("hseparation","OptionButton", 2 );
	t->set_constant("arrow_margin","OptionButton", 2 );



	// MenuButton

	t->set_stylebox("normal","MenuButton", make_stylebox( button_normal_png,4,4,4,4,8,4,8,4) );
	t->set_stylebox("pressed","MenuButton", make_stylebox( tool_button_pressed_png ,4,4,4,4) );
	t->set_stylebox("hover","MenuButton", make_stylebox( button_normal_png,4,4,4,4) );
	t->set_stylebox("disabled","MenuButton", make_empty_stylebox(0,0,0,0) );

	t->set_font("font","MenuButton", default_font );

	t->set_color("font_color","MenuButton", control_font_color );
	t->set_color("font_color_pressed","MenuButton", control_font_color_pressed );
	t->set_color("font_color_hover","MenuButton", control_font_color_hover );
	t->set_color("font_color_disabled","MenuButton", Color(1,1,1,0.3) );

	t->set_constant("hseparation","MenuButton", 0 );



	// CheckButton
	
	Ref<StyleBox> cb_empty = memnew( StyleBoxEmpty );
	cb_empty->set_default_margin(MARGIN_LEFT,6);
	cb_empty->set_default_margin(MARGIN_RIGHT,70);
	cb_empty->set_default_margin(MARGIN_TOP,4);
	cb_empty->set_default_margin(MARGIN_BOTTOM,4);

	t->set_stylebox("normal","CheckButton", cb_empty );
	t->set_stylebox("pressed","CheckButton", cb_empty );
	t->set_stylebox("disabled","CheckButton", cb_empty );
	t->set_stylebox("hover","CheckButton", cb_empty );
	t->set_stylebox("focus","CheckButton", focus );

	t->set_icon("on","CheckButton", make_icon(toggle_on_png) );
	t->set_icon("off","CheckButton", make_icon(toggle_off_png));

	t->set_font("font","CheckButton", default_font );

	t->set_color("font_color","CheckButton", control_font_color );
	t->set_color("font_color_pressed","CheckButton", control_font_color_pressed );
	t->set_color("font_color_hover","CheckButton", control_font_color_hover );
	t->set_color("font_color_disabled","CheckButton", control_font_color_disabled );

	t->set_constant("hseparation","CheckButton",4);
	t->set_constant("check_vadjust","CheckButton",0);



	// Label
	            
	t->set_font("font","Label", default_font );

	t->set_color("font_color","Label", Color(1,1,1) );
	t->set_color("font_color_shadow","Label", Color(0,0,0,0) );

	t->set_constant("shadow_offset_x","Label", 1 );
	t->set_constant("shadow_offset_y","Label", 1 );
	t->set_constant("shadow_as_outline","Label", 0 );



	// LineEdit

	t->set_stylebox("normal","LineEdit", make_stylebox( line_edit_png,5,5,5,5) );
	t->set_stylebox("focus","LineEdit", focus );
	t->set_stylebox("read_only","LineEdit", make_stylebox( line_edit_disabled_png,6,6,6,6) );

	t->set_font("font","LineEdit", default_font );

	t->set_color("font_color","LineEdit", control_font_color );
	t->set_color("font_color_selected","LineEdit", Color(0,0,0) );
	t->set_color("cursor_color","LineEdit", control_font_color_hover );
	t->set_color("selection_color","LineEdit", font_color_selection );

	t->set_constant("minimum_spaces","LineEdit", 12 );



	// ProgressBar

	t->set_stylebox("bg","ProgressBar", make_stylebox( progress_bar_png,4,4,4,4,0,0,0,0) );
	t->set_stylebox("fg","ProgressBar", make_stylebox( progress_fill_png,6,6,6,6,2,1,2,1) );

	t->set_font("font","ProgressBar", default_font );

	t->set_color("font_color","ProgressBar", control_font_color_hover );
	t->set_color("font_color_shadow","ProgressBar", Color(0,0,0) );



	// TextEdit

	t->set_stylebox("normal","TextEdit", make_stylebox( tree_bg_png,3,3,3,3) );
	t->set_stylebox("focus","TextEdit", focus );
	t->set_stylebox("completion","TextEdit", make_stylebox( tree_bg_png,3,3,3,3) );

	t->set_icon("tab","TextEdit", make_icon( tab_png) );

	t->set_font("font","TextEdit", default_font );

	t->set_color("completion_scroll_color","TextEdit", control_font_color_pressed );
	t->set_color("completion_existing","TextEdit", control_font_color );
	t->set_color("font_color","TextEdit", control_font_color );
	t->set_color("font_color_selected","TextEdit", Color(0,0,0) );
	t->set_color("selection_color","TextEdit", font_color_selection );
	t->set_color("mark_color","TextEdit", Color(1.0,0.4,0.4,0.4) );
	t->set_color("breakpoint_color","TextEdit", Color(0.8,0.8,0.4,0.2) );
	t->set_color("current_line_color","TextEdit", Color(0.25,0.25,0.26,0.8) );
	t->set_color("cursor_color","TextEdit", control_font_color );
	t->set_color("symbol_color","TextEdit", control_font_color_hover );
	t->set_color("brace_mismatch_color","TextEdit", Color(1,0.2,0.2) );

	t->set_constant("completion_lines","TextEdit", 7 );
	t->set_constant("completion_max_width","TextEdit", 50 );
	t->set_constant("completion_scroll_width","TextEdit", 3 );
	t->set_constant("line_spacing","TextEdit",4 );


	Ref<Texture> empty_icon = memnew( ImageTexture );

	// HScrollBar

	t->set_stylebox("scroll","HScrollBar", make_stylebox( scroll_bg_png,3,3,3,3,0,0,0,0) );
	t->set_stylebox("scroll_focus","HScrollBar", make_stylebox( scroll_bg_png,3,3,3,3,0,0,0,0) );
	t->set_stylebox("grabber","HScrollBar", make_stylebox( scroll_grabber_png,3,3,3,3,2,2,2,2) );
	t->set_stylebox("grabber_hilite","HScrollBar", make_stylebox( scroll_grabber_hl_png,3,3,3,3,2,2,2,2) );

	t->set_icon("increment","HScrollBar",empty_icon);
	t->set_icon("increment_hilite","HScrollBar",empty_icon);
	t->set_icon("decrement","HScrollBar",empty_icon);
	t->set_icon("decrement_hilite","HScrollBar",empty_icon);



	// VScrollBar

	t->set_stylebox("scroll","VScrollBar", make_stylebox( scroll_bg_png,3,3,3,3,0,0,0,0) );
	t->set_stylebox("scroll_focus","VScrollBar", make_stylebox( scroll_bg_png,3,3,3,3,0,0,0,0) );
	t->set_stylebox("grabber","VScrollBar", make_stylebox( scroll_grabber_png,3,3,3,3,2,2,2,2) );
	t->set_stylebox("grabber_hilite","VScrollBar", make_stylebox( scroll_grabber_hl_png,3,3,3,3,2,2,2,2) );

	t->set_icon("increment","VScrollBar",empty_icon);
	t->set_icon("increment_hilite","VScrollBar",empty_icon);
	t->set_icon("decrement","VScrollBar",empty_icon);
	t->set_icon("decrement_hilite","VScrollBar",empty_icon);



	// HSlider

	t->set_stylebox("slider","HSlider", make_stylebox( hslider_bg_png,4,4,4,4) );
	t->set_stylebox("grabber_hilite","HSlider", make_stylebox( hslider_grabber_hl_png,6,6,6,6) );
	t->set_stylebox("focus","HSlider", focus );

	t->set_icon("grabber","HSlider", make_icon( hslider_grabber_png ) );
	t->set_icon("grabber_hilite","HSlider", make_icon( hslider_grabber_hl_png ) );
	t->set_icon("tick","HSlider", make_icon( hslider_tick_png ) );




	// VSlider

	t->set_stylebox("slider","VSlider", make_stylebox( vslider_bg_png,4,4,4,4) );
	t->set_stylebox("grabber_hilite","VSlider", make_stylebox( vslider_grabber_hl_png,6,6,6,6) );
	t->set_stylebox("focus","HSlider", focus );

	t->set_icon("grabber","VSlider", make_icon( vslider_grabber_png)  );
	t->set_icon("grabber_hilite","VSlider", make_icon( vslider_grabber_hl_png ) );
	t->set_icon("tick","VSlider", make_icon( vslider_tick_png ) );



	// SpinBox

	t->set_icon("updown","SpinBox",make_icon(spinbox_updown_png));



	// WindowDialog

	Ref<StyleBoxTexture> style_pp_win = make_stylebox( popup_window_png,6,28,6,7);
	for(int i=0;i<4;i++)
		style_pp_win->set_expand_margin_size((Margin)i,3);
	style_pp_win->set_expand_margin_size(MARGIN_TOP,26);

	t->set_stylebox("panel","WindowDialog", style_pp_win );

	t->set_icon("close","WindowDialog", make_icon( close_png ) );
	t->set_icon("close_hilite","WindowDialog", make_icon( close_hl_png ) );

	t->set_font("title_font","WindowDialog", large_font );

	t->set_color("title_color","WindowDialog", Color(0,0,0) );

	t->set_constant("close_h_ofs","WindowDialog", 22 );
	t->set_constant("close_v_ofs","WindowDialog", 20 );
	t->set_constant("titlebar_height","WindowDialog", 18 );
	t->set_constant("title_height","WindowDialog", 20 );



	// Popup

	Ref<StyleBoxTexture> style_pp = make_stylebox( popup_bg_png,4,4,4,4,8,8,8,8);

	Ref<StyleBoxTexture> selected = make_stylebox( selection_png,6,6,6,6);
	for(int i=0;i<4;i++) {
		selected->set_expand_margin_size(Margin(i),2);
	}

	t->set_stylebox("panel","PopupPanel", style_pp );




	// PopupMenu

	t->set_stylebox("panel","PopupMenu", make_stylebox( popup_bg_png,4,4,4,4,10,10,10,10) );
	t->set_stylebox("panel_disabled","PopupMenu", make_stylebox( popup_bg_disabled_png,4,4,4,4) );
	t->set_stylebox("hover","PopupMenu", selected );
	t->set_stylebox("separator","PopupMenu", make_stylebox( vseparator_png,3,3,3,3) );

	t->set_icon("checked","PopupMenu", make_icon(checked_png) );
	t->set_icon("unchecked","PopupMenu", make_icon(unchecked_png) );
	t->set_icon("submenu","PopupMenu", make_icon(submenu_png) );

	t->set_font("font","PopupMenu", default_font );

	t->set_color("font_color","PopupMenu", control_font_color );
	t->set_color("font_color_accel","PopupMenu", Color(0.7,0.7,0.7,0.8) );
	t->set_color("font_color_disabled","PopupMenu", Color(0.4,0.4,0.4,0.8) );
	t->set_color("font_color_hover","PopupMenu", control_font_color );

	t->set_constant("hseparation","PopupMenu",4);
	t->set_constant("vseparation","PopupMenu",4);


	// GraphNode

	Ref<StyleBoxTexture> graphsb = make_stylebox(graph_node_png,6,24,6,5,16,24,16,5);
	//graphsb->set_expand_margin_size(MARGIN_LEFT,10);
	//graphsb->set_expand_margin_size(MARGIN_RIGHT,10);
	t->set_stylebox("frame","GraphNode", graphsb );
	t->set_constant("separation","GraphNode", 1 );
	t->set_icon("port","GraphNode", make_icon( graph_port_png ) );
	t->set_icon("close","GraphNode", make_icon( graph_node_close_png ) );
	t->set_font("title_font","GraphNode", default_font );
	t->set_color("title_color","GraphNode", Color(0,0,0,1));
	t->set_constant("title_offset","GraphNode", 18);
	t->set_constant("close_offset","GraphNode", 18);
	t->set_constant("port_offset","GraphNode", 3);


	// Tree

	Ref<StyleBoxTexture> tree_selected = make_stylebox( selection_png,4,4,4,4,8,0,8,0);
	Ref<StyleBoxTexture> tree_selected_oof = make_stylebox( selection_oof_png,4,4,4,4,8,0,8,0);

	t->set_stylebox("bg","Tree", make_stylebox( tree_bg_png,4,4,4,5) );
	t->set_stylebox("bg_focus","Tree", focus );
	t->set_stylebox("selected","Tree", tree_selected_oof );
	t->set_stylebox("selected_focus","Tree", tree_selected );
	t->set_stylebox("cursor","Tree", focus );
	t->set_stylebox("cursor_unfocused","Tree", focus );
	t->set_stylebox("button_pressed","Tree",make_stylebox( button_pressed_png,4,4,4,4));
	t->set_stylebox("title_button_normal","Tree", make_stylebox( tree_title_png,4,4,4,4) );
	t->set_stylebox("title_button_pressed","Tree", make_stylebox( tree_title_pressed_png,4,4,4,4) );
	t->set_stylebox("title_button_hover","Tree", make_stylebox( tree_title_png,4,4,4,4) );

	t->set_icon("checked","Tree",make_icon(checked_png));
	t->set_icon("unchecked","Tree",make_icon(unchecked_png));
	t->set_icon("updown","Tree",make_icon(updown_png));
	t->set_icon("select_arrow","Tree",make_icon(dropdown_png));
	t->set_icon("arrow","Tree",make_icon(arrow_down_png));
	t->set_icon("arrow_collapsed","Tree",make_icon(arrow_right_png));

	t->set_font("title_button_font","Tree", default_font );
	t->set_font("font","Tree", default_font );

	t->set_color("title_button_color","Tree", control_font_color );
	t->set_color("font_color","Tree", control_font_color_low );
	t->set_color("font_color_selected","Tree", control_font_color_pressed );
	t->set_color("selection_color","Tree", Color(0.1,0.1,1,0.8) );
	t->set_color("cursor_color","Tree", Color(0,0,0) );
	t->set_color("guide_color","Tree", Color(0,0,0,0.1) );

	t->set_constant("hseparation","Tree",4);
	t->set_constant("vseparation","Tree",2);
	t->set_constant("guide_width","Tree",2);
	t->set_constant("item_margin","Tree",12);
	t->set_constant("button_margin","Tree",4);



	// TextEdit

	t->set_stylebox("completion_selected","TextEdit", tree_selected );



	// TabContainer

	Ref<StyleBoxTexture> tc_sb = make_stylebox( tab_container_bg_png,4,4,4,4);
	for(int i=0;i<4;i++) {
		tc_sb->set_default_margin(Margin(i),4);
		tc_sb->set_expand_margin_size(Margin(i),2);
	}
	tc_sb->set_expand_margin_size(MARGIN_TOP,2);
	tc_sb->set_default_margin(MARGIN_TOP,8);

	t->set_stylebox("tab_fg","TabContainer", make_stylebox( tab_current_png,4,4,4,4,16,4,16,4) );
	t->set_stylebox("tab_bg","TabContainer", make_stylebox( tab_behind_png,4,4,4,4,16,6,16,4) );
	t->set_stylebox("panel","TabContainer", tc_sb );

	t->set_icon("increment","TabContainer",make_icon( scroll_button_right_png));
	t->set_icon("increment_hilite","TabContainer",make_icon( scroll_button_right_hl_png));
	t->set_icon("decrement","TabContainer",make_icon( scroll_button_left_png));
	t->set_icon("decrement_hilite","TabContainer",make_icon( scroll_button_left_hl_png));

	t->set_font("font","TabContainer", default_font );

	t->set_color("font_color_fg","TabContainer", control_font_color_hover );
	t->set_color("font_color_bg","TabContainer", control_font_color_low );

	t->set_constant("side_margin","TabContainer", 8 );
	t->set_constant("top_margin","TabContainer", 24);
	t->set_constant("label_valign_fg","TabContainer", 0);
	t->set_constant("label_valign_bg","TabContainer", 2);
	t->set_constant("hseparation","TabContainer", 4);



	// Tabs

	t->set_stylebox("tab_fg","Tabs", make_stylebox( tab_current_png,4,4,4,4,16,4,16,4) );
	t->set_stylebox("tab_bg","Tabs", make_stylebox( tab_behind_png,4,4,4,4,16,6,16,4) );
	t->set_stylebox("panel","Tabs", make_stylebox( tab_container_bg_png,4,4,4,4) );

	t->set_font("font","Tabs", default_font );

	t->set_color("font_color_fg","Tabs", control_font_color_hover );
	t->set_color("font_color_bg","Tabs", control_font_color_low );

	t->set_constant("top_margin","Tabs", 24);
	t->set_constant("label_valign_fg","Tabs", 0);
	t->set_constant("label_valign_bg","Tabs", 2);
	t->set_constant("hseparation","Tabs", 4);



	// Separators

	t->set_stylebox("separator","HSeparator", make_stylebox( vseparator_png,3,3,3,3) );
	t->set_stylebox("separator","VSeparator", make_stylebox( hseparator_png,3,3,3,3) );

	t->set_icon("close","Icons", make_icon(icon_close_png));
	t->set_font("source","Fonts", source_font);
	t->set_font("normal","Fonts", default_font );
	t->set_font("large","Fonts", large_font );

	t->set_constant("separation","HSeparator", 4);
	t->set_constant("separation","VSeparator", 4);


	// Dialogs

	t->set_constant("margin","Dialogs",8);
	t->set_constant("button_margin","Dialogs",32);



	// FileDialog
	
	t->set_icon("folder","FileDialog",make_icon(icon_folder_png));

	t->set_color("files_disabled","FileDialog",Color(0,0,0,0.7));



	// colorPicker

	t->set_constant("value_height","ColorPicker", 23 );
	t->set_constant("value_width","ColorPicker", 50);
	t->set_constant("color_width","ColorPicker", 100);
	t->set_constant("label_width","ColorPicker", 20);
	t->set_constant("hseparator","ColorPicker", 4);



	// TooltipPanel

	Ref<StyleBoxTexture> style_tt = make_stylebox( tooltip_bg_png,4,4,4,4);
	for(int i=0;i<4;i++)
		style_tt->set_expand_margin_size((Margin)i,4);

	t->set_stylebox("panel","TooltipPanel", style_tt );

	t->set_font("font","TooltipLabel", default_font );

	t->set_color("font_color","TooltipLabel", Color(0,0,0) );
	t->set_color("font_color_shadow","TooltipLabel", Color(0,0,0,0.1) );

	t->set_constant("shadow_offset_x","TooltipLabel", 1 );
	t->set_constant("shadow_offset_y","TooltipLabel", 1 );



	// RichTextLabel

	t->set_stylebox("focus","RichTextLabel", focus );

	t->set_font("default_font","RichTextLabel", default_font );

	t->set_color("default_color","RichTextLabel", control_font_color );
	t->set_color("font_color_selected","RichTextLabel", font_color_selection );
	t->set_color("selection_color","RichTextLabel", Color(0.1,0.1,1,0.8) );

	t->set_constant("line_separation","RichTextLabel", 1 );



	// Containers

	t->set_stylebox("bg","VSplitContainer", make_stylebox( vsplit_bg_png,1,1,1,1) );
	t->set_stylebox("bg","HSplitContainer", make_stylebox( hsplit_bg_png,1,1,1,1) );

	t->set_icon("grabber","VSplitContainer",make_icon(vsplitter_png));
	t->set_icon("grabber","HSplitContainer",make_icon(hsplitter_png));

	t->set_constant("separation","HBoxContainer",4);
	t->set_constant("separation","VBoxContainer",4);
	t->set_constant("margin","MarginContainer",8);
	t->set_constant("separation","GridContainer",4);
	t->set_constant("separation","HSplitContainer",12);
	t->set_constant("separation","VSplitContainer",12);
	t->set_constant("autohide","HSplitContainer",1);
	t->set_constant("autohide","VSplitContainer",1);



	// HButtonArray

	t->set_stylebox("normal","HButtonArray", make_stylebox( button_normal_png,4,4,4,4,0,4,22,4) );
	t->set_stylebox("selected","HButtonArray", make_stylebox( button_pressed_png,4,4,4,4,0,4,22,4) );
	t->set_stylebox("hover","HButtonArray", make_stylebox( button_hover_png,4,4,4,4) );

	t->set_font("font","HButtonArray", default_font);
	t->set_font("font_selected","HButtonArray", default_font);

	t->set_color("font_color","HButtonArray", control_font_color_low );
	t->set_color("font_color_selected","HButtonArray", control_font_color_hover );

	t->set_constant("icon_separator","HButtonArray", 4 );
	t->set_constant("button_separator","HButtonArray", 8 );

	t->set_stylebox("focus","HButtonArray", focus );


	// VButtonArray

	t->set_stylebox("normal","VButtonArray", make_stylebox( button_normal_png,4,4,4,4,0,4,22,4) );
	t->set_stylebox("selected","VButtonArray", make_stylebox( button_pressed_png,4,4,4,4,0,4,22,4) );
	t->set_stylebox("hover","VButtonArray", make_stylebox( button_hover_png,4,4,4,4) );

	t->set_font("font","VButtonArray", default_font);
	t->set_font("font_selected","VButtonArray", default_font);

	t->set_color("font_color","VButtonArray", control_font_color_low );
	t->set_color("font_color_selected","VButtonArray", control_font_color_hover );

	t->set_constant("icon_separator","VButtonArray", 4);
	t->set_constant("button_separator","VButtonArray", 8);

	t->set_stylebox("focus","VButtonArray", focus );


	// ReferenceFrame

	Ref<StyleBoxTexture> ttnc = make_stylebox( full_panel_bg_png,8,8,8,8);
	ttnc->set_draw_center(false);

	t->set_stylebox("border","ReferenceFrame", make_stylebox( reference_border_png,4,4,4,4) );
	t->set_stylebox("panelnc","Panel", ttnc );
	t->set_stylebox("panelf","Panel", tc_sb );
	t->set_stylebox("panel","PanelContainer", tc_sb );

	t->set_icon( "logo","Icons", make_icon(logo_png) );



	// Theme

	Theme::set_default( t );
	Theme::set_default_icon( make_icon(error_icon_png) );
	Theme::set_default_style( make_stylebox( error_icon_png,2,2,2,2) );
	Theme::set_default_font( default_font );

	memdelete( tex_cache );

}

#else

#include "error_icon.xpm"

void make_default_theme() {

	Ref<Theme> t( memnew( Theme ) );


	Image error_img(error_icon_xpm);
	Ref<Texture> texture( memnew( Texture ) );
	texture->create_from_image( error_img );

	Ref<StyleBoxTexture> style( memnew( StyleBoxTexture ) );
	style->set_texture(texture);

	for(int i=0;i<4;i++) {
		style->set_margin_size(  Margin(),8);
		style->set_default_margin(  Margin(),8);
	}

	Ref<Font> f = make_default_font();
	Theme::set_default( t );
	Theme::set_default_icon( texture );
	Theme::set_default_style( style );
	Theme::set_default_font( f );

}

#endif
void clear_default_theme() {
	
	Theme::set_default( Ref<Theme>() );
	Theme::set_default_icon( Ref< Texture >() );
	Theme::set_default_style( Ref< StyleBox >() );	
	Theme::set_default_font( Ref< Font >() );

}



