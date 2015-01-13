/*************************************************************************/
/*  text_edit.cpp                                                        */
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
/*****f********************************************/
/*  text_edit.cpp                                */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#include "text_edit.h"
#include "os/keyboard.h"
#include "os/os.h"

#include "globals.h"
#include "message_queue.h"

#define TAB_PIXELS

static bool _is_text_char(CharType c) {
	
	return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_';
}

static bool _is_symbol(CharType c) {
	
	return c!='_' && ((c>='!' && c<='/') || (c>=':' && c<='@') || (c>='[' && c<='`') || (c>='{' && c<='~') || c=='\t');
}

static bool _is_pair_right_symbol(CharType c) {
	return
			c == '"'  ||
			c == '\'' ||
			c == ')'  ||
			c == ']'  ||
			c == '}';
}

static bool _is_pair_left_symbol(CharType c) {
	return
			c == '"'  ||
			c == '\'' ||
			c == '('  ||
			c == '['  ||
			c == '{';
}

static bool _is_pair_symbol(CharType c) {
	return _is_pair_left_symbol(c) || _is_pair_right_symbol(c);
}

static CharType _get_right_pair_symbol(CharType c) {
	if(c == '"')
		return '"';
	if(c == '\'')
		return '\'';
	if(c == '(')
		return ')';
	if(c == '[')
		return ']';
	if(c == '{')
		return '}';
	return 0;
}

void TextEdit::Text::set_font(const Ref<Font>& p_font) {
	
	font=p_font;
}

void TextEdit::Text::set_tab_size(int p_tab_size) {
	
	tab_size=p_tab_size;
}

void TextEdit::Text::_update_line_cache(int p_line) const {
	
	int w =0;
	int tab_w=font->get_char_size(' ').width;
	
	int len = text[p_line].data.length();
	const CharType *str = text[p_line].data.c_str();
	
	//update width
	
	for(int i=0;i<len;i++) {
		if (str[i]=='\t') {
			
			int left = w%tab_w;
			if (left==0)
				w+=tab_w;
			else
				w+=tab_w-w%tab_w; // is right...
			
		} else {
			
			w+=font->get_char_size(str[i],str[i+1]).width;
		}
	}
	
	
	text[p_line].width_cache=w;
	
	//update regions
	
	text[p_line].region_info.clear();
	
	for(int i=0;i<len;i++) {
		
		if (!_is_symbol(str[i]))
			continue;
		if (str[i]=='\\') {
			i++; //skip quoted anything
			continue;
		}
		
		int left=len-i;
		
		for(int j=0;j<color_regions->size();j++) {
			
			const ColorRegion& cr=color_regions->operator [](j);
			
			/* BEGIN */
			
			int lr=cr.begin_key.length();
			if (lr==0 || lr>left)
				continue;
			
			const CharType* kc = cr.begin_key.c_str();
			
			bool match=true;
			
			for(int k=0;k<lr;k++) {
				if (kc[k]!=str[i+k]) {
					match=false;
					break;
				}
			}
			
			if (match) {
				
				ColorRegionInfo cri;
				cri.end=false;
				cri.region=j;
				text[p_line].region_info[i]=cri;
				i+=lr-1;
				break;
			}
			
			/* END */
			
			lr=cr.end_key.length();
			if (lr==0 || lr>left)
				continue;
			
			kc = cr.end_key.c_str();
			
			match=true;
			
			for(int k=0;k<lr;k++) {
				if (kc[k]!=str[i+k]) {
					match=false;
					break;
				}
			}
			
			if (match) {
				
				ColorRegionInfo cri;
				cri.end=true;
				cri.region=j;
				text[p_line].region_info[i]=cri;
				i+=lr-1;
				break;
			}
			
		}
	}
	
	
}

const Map<int,TextEdit::Text::ColorRegionInfo>& TextEdit::Text::get_color_region_info(int p_line) {
	
	Map<int,ColorRegionInfo> *cri=NULL;
	ERR_FAIL_INDEX_V(p_line,text.size(),*cri); //enjoy your crash
	
	if (text[p_line].width_cache==-1) {
		_update_line_cache(p_line);
	}
	
	return text[p_line].region_info;
}

int TextEdit::Text::get_line_width(int p_line) const {
	
	ERR_FAIL_INDEX_V(p_line,text.size(),-1);
	
	if (text[p_line].width_cache==-1) {
		_update_line_cache(p_line);
	}
	
	return text[p_line].width_cache;
}

void TextEdit::Text::clear_caches() {
	
	for(int i=0;i<text.size();i++)
		text[i].width_cache=-1;
	
}

void TextEdit::Text::clear() {
	
	
	text.clear();;
	insert(0,"");
}

int TextEdit::Text::get_max_width() const {
	//quite some work.. but should be fast enough.
	
	int max = 0;
	
	for(int i=0;i<text.size();i++)
		max=MAX(max,get_line_width(i));
	return max;
	
}

void TextEdit::Text::set(int p_line,const String& p_text) {
	
	ERR_FAIL_INDEX(p_line,text.size());
	
	text[p_line].width_cache=-1;
	text[p_line].data=p_text;
}


void TextEdit::Text::insert(int p_at,const String& p_text) {
	
	Line line;
	line.marked=false;
	line.breakpoint=false;
	line.width_cache=-1;
	line.data=p_text;
	text.insert(p_at,line);
}
void TextEdit::Text::remove(int p_at) {
	
	text.remove(p_at);
}

void TextEdit::_update_scrollbars() {
	
	
	Size2 size = get_size();
	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();
	
	
	
	v_scroll->set_begin( Point2(size.width - vmin.width, cache.style_normal->get_margin(MARGIN_TOP)) );
	v_scroll->set_end( Point2(size.width, size.height - cache.style_normal->get_margin(MARGIN_TOP) - cache.style_normal->get_margin(MARGIN_BOTTOM)) );
	
	h_scroll->set_begin( Point2( 0, size.height - hmin.height) );
	h_scroll->set_end( Point2(size.width-vmin.width, size.height) );
	
	
	int hscroll_rows = ((hmin.height-1)/get_row_height())+1;
	int visible_rows = get_visible_rows();
	int total_rows = text.size();
	
	int vscroll_pixels = v_scroll->get_combined_minimum_size().width;
	int visible_width = size.width - cache.style_normal->get_minimum_size().width;
	int total_width = text.get_max_width();
	
	bool use_hscroll=true;
	bool use_vscroll=true;
	
	if (total_rows <= visible_rows && total_width <= visible_width) {
		//thanks yessopie for this clever bit of logic
		use_hscroll=false;
		use_vscroll=false;
		
	} else {
		
		if (total_rows > visible_rows && total_width <= visible_width - vscroll_pixels) {
			//thanks yessopie for this clever bit of logic
			use_hscroll=false;
		}
		
		if (total_rows <= visible_rows - hscroll_rows && total_width > visible_width) {
			//thanks yessopie for this clever bit of logic
			use_vscroll=false;
		}
	}
	
	updating_scrolls=true;
	
	if (use_vscroll) {
		
		v_scroll->show();
		v_scroll->set_max(total_rows);
		v_scroll->set_page(visible_rows);
		
		v_scroll->set_val(cursor.line_ofs);
		
	}  else {
		cursor.line_ofs = 0;
		v_scroll->hide();
	}
	
	if (use_hscroll) {
		
		h_scroll->show();
		h_scroll->set_max(total_width);
		h_scroll->set_page(visible_width);
		h_scroll->set_val(cursor.x_ofs);
	} else {
		
		h_scroll->hide();
	}
	
	
	
	updating_scrolls=false;
}


void TextEdit::_notification(int p_what) {
	
	switch(p_what) {
		case NOTIFICATION_ENTER_TREE: {
			
			_update_caches();
			if (cursor_changed_dirty)
				MessageQueue::get_singleton()->push_call(this,"_cursor_changed_emit");
			if (text_changed_dirty)
				MessageQueue::get_singleton()->push_call(this,"_text_changed_emit");
			
		} break;
		case NOTIFICATION_RESIZED: {
			
			cache.size=get_size();
			adjust_viewport_to_cursor();
			
			
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			
			_update_caches();
		};
		case NOTIFICATION_DRAW: {
			
			int line_number_char_count=0;
			
			{
				int lc=text.size()+1;
				cache.line_number_w=0;
				while(lc) {
					cache.line_number_w+=1;
					lc/=10;
				};
				
				if (line_numbers) {
					
					line_number_char_count=cache.line_number_w;
					cache.line_number_w=(cache.line_number_w+1)*cache.font->get_char_size('0').width;
				} else {
					cache.line_number_w=0;
				}
				
				
			}
			_update_scrollbars();
			
			
			RID ci = get_canvas_item();
			int xmargin_beg=cache.style_normal->get_margin(MARGIN_LEFT)+cache.line_number_w;
			int xmargin_end=cache.size.width-cache.style_normal->get_margin(MARGIN_RIGHT);
			//let's do it easy for now:
			cache.style_normal->draw(ci,Rect2(Point2(),cache.size));
			if (has_focus())
				cache.style_focus->draw(ci,Rect2(Point2(),cache.size));
			
			
			int ascent=cache.font->get_ascent();
			
			int visible_rows = get_visible_rows();
			
			int tab_w = cache.font->get_char_size(' ').width*tab_size;
			
			Color color = cache.font_color;
			int in_region=-1;
			
			if (syntax_coloring) {
				
				if (custom_bg_color.a>0.01) {
					
					Point2i ofs = Point2i(cache.style_normal->get_offset())/2.0;
					VisualServer::get_singleton()->canvas_item_add_rect(ci,Rect2(ofs, get_size()-cache.style_normal->get_minimum_size()+ofs),custom_bg_color);
				}
				//compute actual region to start (may be inside say, a comment).
				//slow in very large documments :( but ok for source!
				
				for(int i=0;i<cursor.line_ofs;i++) {
					
					const Map<int,Text::ColorRegionInfo>& cri_map=text.get_color_region_info(i);
					
					if (in_region>=0 && color_regions[in_region].line_only) {
						in_region=-1; //reset regions that end at end of line
					}
					
					for( const Map<int,Text::ColorRegionInfo>::Element* E= cri_map.front();E;E=E->next() ) {
						
						const Text::ColorRegionInfo &cri=E->get();
						
						if (in_region==-1) {
							
							if (!cri.end) {
								
								in_region=cri.region;
							}
						} else if (in_region==cri.region && !color_regions[cri.region].line_only) { //ignore otherwise
							
							if (cri.end || color_regions[cri.region].eq) {
								
								in_region=-1;
							}
						}
					}
				}
			}
			
			int brace_open_match_line=-1;
			int brace_open_match_column=-1;
			bool brace_open_matching=false;
			bool brace_open_mismatch=false;
			int brace_close_match_line=-1;
			int brace_close_match_column=-1;
			bool brace_close_matching=false;
			bool brace_close_mismatch=false;
			
			
			if (brace_matching_enabled) {
				
				if (cursor.column<text[cursor.line].length()) {
					//check for open
					CharType c = text[cursor.line][cursor.column];
					CharType closec=0;
					
					if (c=='[') {
						closec=']';
					} else if (c=='{') {
						closec='}';
					} else if (c=='(') {
						closec=')';
					}
					
					if (closec!=0) {
						
						int stack=1;
						
						
						for(int i=cursor.line;i<text.size();i++) {
							
							int from = i==cursor.line?cursor.column+1:0;
							for(int j=from;j<text[i].length();j++) {
								
								CharType cc = text[i][j];
								if (cc==c)
									stack++;
								else if (cc==closec)
									stack--;
								
								if (stack==0) {
									brace_open_match_line=i;
									brace_open_match_column=j;
									brace_open_matching=true;
									
									break;
								}
							}
							if (brace_open_match_line!=-1)
								break;
						}
						
						if (!brace_open_matching)
							brace_open_mismatch=true;
						
						
					}
				}
				
				if (cursor.column>0) {
					CharType c = text[cursor.line][cursor.column-1];
					CharType closec=0;
					
					
					
					if (c==']') {
						closec='[';
					} else if (c=='}') {
						closec='{';
					} else if (c==')') {
						closec='(';
					}
					
					if (closec!=0) {
						
						int stack=1;
						
						
						for(int i=cursor.line;i>=0;i--) {
							
							int from = i==cursor.line?cursor.column-2:text[i].length()-1;
							for(int j=from;j>=0;j--) {
								
								CharType cc = text[i][j];
								if (cc==c)
									stack++;
								else if (cc==closec)
									stack--;
								
								if (stack==0) {
									brace_close_match_line=i;
									brace_close_match_column=j;
									brace_close_matching=true;
									
									break;
								}
							}
							if (brace_close_match_line!=-1)
								break;
						}
						
						if (!brace_close_matching)
							brace_close_mismatch=true;
						
						
					}
					
					
				}
			}
			
			
			int deregion=0; //force it to clear inrgion
			Point2 cursor_pos;
			
			for (int i=0;i<visible_rows;i++) {
				
				int line=i+cursor.line_ofs;
				
				if (line<0 || line>=(int)text.size())
					continue;
				
				const String &str=text[line];
				
				int char_margin=xmargin_beg-cursor.x_ofs;
				int char_ofs=0;
				int ofs_y=i*get_row_height()+cache.line_spacing/2;
				bool prev_is_char=false;
				bool in_keyword=false;
				Color keyword_color;
				
				if (cache.line_number_w) {
					Color fcol = cache.font_color;
					fcol.a*=0.4;
					String fc = String::num(line+1);
					while (fc.length() < line_number_char_count) {
						fc="0"+fc;
					}
					
					cache.font->draw(ci,Point2(cache.style_normal->get_margin(MARGIN_LEFT),ofs_y+cache.font->get_ascent()),fc,fcol);
				}
				
				const Map<int,Text::ColorRegionInfo>& cri_map=text.get_color_region_info(line);
				
				
				if (text.is_marked(line)) {
					
					VisualServer::get_singleton()->canvas_item_add_rect(ci,Rect2(xmargin_beg, ofs_y,xmargin_end-xmargin_beg,get_row_height()),cache.mark_color);
				}
				
				if (text.is_breakpoint(line)) {
					
					VisualServer::get_singleton()->canvas_item_add_rect(ci,Rect2(xmargin_beg, ofs_y,xmargin_end-xmargin_beg,get_row_height()),cache.breakpoint_color);
				}
				
				
				if (line==cursor.line) {
					
					VisualServer::get_singleton()->canvas_item_add_rect(ci,Rect2(xmargin_beg, ofs_y,xmargin_end-xmargin_beg,get_row_height()),cache.current_line_color);
					
				}
				for (int j=0;j<str.length();j++) {
					
					//look for keyword
					
					if (deregion>0) {
						deregion--;
						if (deregion==0)
							in_region=-1;
					}
					if (syntax_coloring && deregion==0) {
						
						
						color = cache.font_color; //reset
						//find keyword
						bool is_char = _is_text_char(str[j]);
						bool is_symbol=_is_symbol(str[j]);
						
						if (j==0 && in_region>=0 && color_regions[in_region].line_only) {
							in_region=-1; //reset regions that end at end of line
						}
						
						if (is_symbol && cri_map.has(j)) {
							
							
							const Text::ColorRegionInfo &cri=cri_map[j];
							
							if (in_region==-1) {
								
								if (!cri.end) {
									
									in_region=cri.region;
								}
							} else if (in_region==cri.region && !color_regions[cri.region].line_only) { //ignore otherwise
								
								if (cri.end || color_regions[cri.region].eq) {
									
									deregion=color_regions[cri.region].eq?color_regions[cri.region].begin_key.length():color_regions[cri.region].end_key.length();
								}
							}
						}
						
						if (!is_char)
							in_keyword=false;
						
						if (in_region==-1 && !in_keyword && is_char && !prev_is_char) {
							
							int to=j;
							while(_is_text_char(str[to]) && to<str.length())
								to++;
							
							uint32_t hash = String::hash(&str[j],to-j);
							StrRange range(&str[j],to-j);
							
							const Color *col=keywords.custom_getptr(range,hash);
							
							if (col) {
								
								in_keyword=true;
								keyword_color=*col;
							}
						}
						
						
						if (in_region>=0)
							color=color_regions[in_region].color;
						else if (in_keyword)
							color=keyword_color;
						else if (is_symbol)
							color=symbol_color;
						
						prev_is_char=is_char;
						
					}
					int char_w;
					
					//handle tabulator
					
					
					if (str[j]=='\t') {
						int left = char_ofs%tab_w;
						if (left==0)
							char_w=tab_w;
						else
							char_w=tab_w-char_ofs%tab_w; // is right...
						
					} else {
						char_w=cache.font->get_char_size(str[j],str[j+1]).width;
					}
					
					if ( (char_ofs+char_margin)<xmargin_beg) {
						char_ofs+=char_w;
						continue;
					}
					
					if ( (char_ofs+char_margin+char_w)>=xmargin_end) {
						if (syntax_coloring)
							continue;
						else
							break;
					}
					
					bool in_selection = (selection.active && line>=selection.from_line && line<=selection.to_line && (line>selection.from_line || j>=selection.from_column) && (line<selection.to_line || j<selection.to_column));
					
					
					if (in_selection) {
						//inside selection!
						VisualServer::get_singleton()->canvas_item_add_rect(ci,Rect2(Point2i( char_ofs+char_margin, ofs_y ), Size2i(char_w,get_row_height())),cache.selection_color);
					}
					
					
					if (brace_matching_enabled) {
						if ( (brace_open_match_line==line && brace_open_match_column==j) ||
								(cursor.column==j && cursor.line==line && (brace_open_matching||brace_open_mismatch))) {
							
							if (brace_open_mismatch)
								color=cache.brace_mismatch_color;
							cache.font->draw_char(ci,Point2i( char_ofs+char_margin, ofs_y+ascent),'_',str[j+1],in_selection?cache.font_selected_color:color);
							
						}
						
						if (
								(brace_close_match_line==line && brace_close_match_column==j) ||
								(cursor.column==j+1 && cursor.line==line && (brace_close_matching||brace_close_mismatch))) {
							
							
							if (brace_close_mismatch)
								color=cache.brace_mismatch_color;
							cache.font->draw_char(ci,Point2i( char_ofs+char_margin, ofs_y+ascent),'_',str[j+1],in_selection?cache.font_selected_color:color);
							
						}
					}
					
					
					if (str[j]>=32)
						cache.font->draw_char(ci,Point2i( char_ofs+char_margin, ofs_y+ascent),str[j],str[j+1],in_selection?cache.font_selected_color:color);
					
					else if (draw_tabs && str[j]=='\t') {
						int yofs= (get_row_height() - cache.tab_icon->get_height())/2;
						cache.tab_icon->draw(ci, Point2(char_ofs+char_margin,ofs_y+yofs),in_selection?cache.font_selected_color:color);
					}
					
					
					if (cursor.column==j && cursor.line==line) {
						
						cursor_pos = Point2i( char_ofs+char_margin, ofs_y );
						VisualServer::get_singleton()->canvas_item_add_rect(ci,Rect2(cursor_pos, Size2i(1,get_row_height())),cache.font_color);
						
						
					}
					char_ofs+=char_w;
					
				}
				
				if (cursor.column==str.length() && cursor.line==line) {
					
					cursor_pos=Point2i( char_ofs+char_margin, ofs_y );
					VisualServer::get_singleton()->canvas_item_add_rect(ci,Rect2(cursor_pos, Size2i(1,get_row_height())),cache.font_color);
					
				}
			}
			
			
			if (completion_active) {
				// code completion box
				Ref<StyleBox> csb = get_stylebox("completion");
				Ref<StyleBox> csel = get_stylebox("completion_selected");
				int maxlines = get_constant("completion_lines");
				int cmax_width = get_constant("completion_max_width")*cache.font->get_char_size('x').x;
				Color existing = get_color("completion_existing");
				existing.a=0.2;
				int scrollw = get_constant("completion_scroll_width");
				Color scrollc = get_color("completion_scroll_color");
				
				
				
				int lines = MIN(completion_options.size(),maxlines);
				int w=0;
				int h=lines*get_row_height();
				int nofs = cache.font->get_string_size(completion_base).width;
				
				
				if (completion_options.size() < 50) {
					for(int i=0;i<completion_options.size();i++) {
						int w2=MIN(cache.font->get_string_size(completion_options[i]).x,cmax_width);
						if (w2>w)
							w=w2;
					}
				} else {
					w=cmax_width;
				}
				
				int th = h + csb->get_minimum_size().y;
				if (cursor_pos.y+get_row_height()+th > get_size().height) {
					completion_rect.pos.y=cursor_pos.y-th;
				} else {
					completion_rect.pos.y=cursor_pos.y+get_row_height()+csb->get_offset().y;
					
				}
				
				if (cursor_pos.x-nofs+w+scrollw  > get_size().width) {
					completion_rect.pos.x=get_size().width-w-scrollw;
				} else {
					completion_rect.pos.x=cursor_pos.x-nofs;
				}
				
				completion_rect.size.width=w+2;
				completion_rect.size.height=h;
				if (completion_options.size()<=maxlines)
					scrollw=0;
				
				draw_style_box(csb,Rect2(completion_rect.pos-csb->get_offset(),completion_rect.size+csb->get_minimum_size()+Size2(scrollw,0)));
				
				
				int line_from = CLAMP(completion_index - lines/2, 0, completion_options.size() - lines);
				draw_style_box(csel,Rect2(Point2(completion_rect.pos.x,completion_rect.pos.y+(completion_index-line_from)*get_row_height()),Size2(completion_rect.size.width,get_row_height())));
				
				draw_rect(Rect2(completion_rect.pos,Size2(nofs,completion_rect.size.height)),existing);
				
				
				
				
				for(int i=0;i<lines;i++) {
					
					int l = line_from + i;
					ERR_CONTINUE( l < 0 || l>= completion_options.size());
					Color text_color = cache.font_color;
					for(int j=0;j<color_regions.size();j++) {
						if (completion_options[l].begins_with(color_regions[j].begin_key)) {
							text_color=color_regions[j].color;
						}
					}
					draw_string(cache.font,Point2(completion_rect.pos.x,completion_rect.pos.y+i*get_row_height()+cache.font->get_ascent()),completion_options[l],text_color,completion_rect.size.width);
				}
				
				if (scrollw) {
					//draw a small scroll rectangle to show a position in the options
					float r = maxlines / (float)completion_options.size();
					float o = line_from / (float)completion_options.size();
					draw_rect(Rect2(completion_rect.pos.x+completion_rect.size.width,completion_rect.pos.y+o*completion_rect.size.y,scrollw,completion_rect.size.y*r),scrollc);
				}
				
				completion_line_ofs=line_from;
				
			}
			
			if (completion_hint!="") {
				
				Ref<StyleBox> sb = get_stylebox("panel","TooltipPanel");
				Ref<Font> font = cache.font;
				Color font_color = get_color("font_color","TooltipLabel");
				
				
				int max_w=0;
				int sc = completion_hint.get_slice_count("\n");
				int offset=0;
				int spacing=0;
				for(int i=0;i<sc;i++) {
					
					String l = completion_hint.get_slice("\n",i);
					int len  = font->get_string_size(l).x;
					max_w = MAX(len,max_w);
					if (i==0) {
						offset = font->get_string_size(l.substr(0,l.find(String::chr(0xFFFF)))).x;
					} else {
						spacing+=cache.line_spacing;
					}
					
					
				}
				
				
				
				Size2 size = Size2(max_w,sc*font->get_height()+spacing);
				Size2 minsize = size+sb->get_minimum_size();
				
				
				if (completion_hint_offset==-0xFFFF) {
					completion_hint_offset=cursor_pos.x-offset;
				}
				
				
				Point2 hint_ofs = Vector2(completion_hint_offset,cursor_pos.y-minsize.y);
				draw_style_box(sb,Rect2(hint_ofs,minsize));
				
				spacing=0;
				for(int i=0;i<sc;i++) {
					int begin=0;
					int end=0;
					String l = completion_hint.get_slice("\n",i);
					
					if (l.find(String::chr(0xFFFF))!=-1) {
						begin = font->get_string_size(l.substr(0,l.find(String::chr(0xFFFF)))).x;
						end = font->get_string_size(l.substr(0,l.rfind(String::chr(0xFFFF)))).x;
					}
					
					draw_string(font,hint_ofs+sb->get_offset()+Vector2(0,font->get_ascent()+font->get_height()*i+spacing),l.replace(String::chr(0xFFFF),""),font_color);
					if (end>0) {
						Vector2 b = hint_ofs+sb->get_offset()+Vector2(begin,font->get_height()+font->get_height()*i+spacing-1);
						draw_line(b,b+Vector2(end-begin,0),font_color);
					}
					spacing+=cache.line_spacing;
				}
			}
			
			
		} break;
		case NOTIFICATION_FOCUS_ENTER: {
			
			if (OS::get_singleton()->has_virtual_keyboard())
				OS::get_singleton()->show_virtual_keyboard(get_text(),get_global_rect());
			
		} break;
		case NOTIFICATION_FOCUS_EXIT: {
			
			if (OS::get_singleton()->has_virtual_keyboard())
				OS::get_singleton()->hide_virtual_keyboard();
			
		} break;
			
	}
}

void TextEdit::_consume_pair_symbol(CharType ch) {
	
	int cursor_position_to_move = cursor_get_column() + 1;
	
	CharType ch_single[2] = {ch, 0};
	CharType ch_single_pair[2] = {_get_right_pair_symbol(ch), 0};
	CharType ch_pair[3] = {ch, _get_right_pair_symbol(ch), 0};
	
	if(is_selection_active()) {
		
		int new_column,new_line;
		
		_begin_compex_operation();
		_insert_text(get_selection_from_line(), get_selection_from_column(),
			     ch_single,
			     &new_line, &new_column);
		
		int to_col_offset = 0;
		if(get_selection_from_line() == get_selection_to_line())
			to_col_offset = 1;
		
		_insert_text(get_selection_to_line(),
			     get_selection_to_column() + to_col_offset,
			     ch_single_pair,
			     &new_line,&new_column);
		_end_compex_operation();
		
		cursor_set_line(get_selection_to_line());
		cursor_set_column(get_selection_to_column() + to_col_offset);
		
		deselect();
		update();
		return;
	}
	
	if( (ch == '\'' || ch == '"') &&
			cursor_get_column() > 0 &&
			_is_text_char(text[cursor.line][cursor_get_column() - 1])
			) {
		insert_text_at_cursor(ch_single);
		cursor_set_column(cursor_position_to_move);
		return;
	}
	
	if(cursor_get_column() < text[cursor.line].length()) {
		if(_is_text_char(text[cursor.line][cursor_get_column()])) {
			insert_text_at_cursor(ch_single);
			cursor_set_column(cursor_position_to_move);
			return;
		}
		if(	_is_pair_right_symbol(ch) &&
				text[cursor.line][cursor_get_column()] == ch
				) {
			cursor_set_column(cursor_position_to_move);
			return;
		}
	}
	
	
	insert_text_at_cursor(ch_pair);
	cursor_set_column(cursor_position_to_move);
	return;
	
}

void TextEdit::_consume_backspace_for_pair_symbol(int prev_line, int prev_column) {
	
	bool remove_right_symbol = false;
	
	if(cursor.column < text[cursor.line].length() && cursor.column > 0) {
		
		CharType left_char = text[cursor.line][cursor.column - 1];
		CharType right_char = text[cursor.line][cursor.column];
		
		if(right_char == _get_right_pair_symbol(left_char)) {
			remove_right_symbol = true;
		}
		
	}
	if(remove_right_symbol) {
		_remove_text(prev_line,prev_column,cursor.line,cursor.column + 1);
	} else {
		_remove_text(prev_line,prev_column,cursor.line,cursor.column);
	}
	
}

void TextEdit::backspace_at_cursor() {
	if (readonly)
		return;

	if (cursor.column==0 && cursor.line==0)
		return;
	
	int prev_line = cursor.column?cursor.line:cursor.line-1;
	int prev_column = cursor.column?(cursor.column-1):(text[cursor.line-1].length());
	if(auto_brace_completion_enabled &&
			cursor.column > 0 &&
			_is_pair_left_symbol(text[cursor.line][cursor.column - 1])) {
		_consume_backspace_for_pair_symbol(prev_line, prev_column);
	} else {
		_remove_text(prev_line,prev_column,cursor.line,cursor.column);
	}
	
	cursor_set_line(prev_line);
	cursor_set_column(prev_column);
	
}


bool TextEdit::_get_mouse_pos(const Point2i& p_mouse, int &r_row, int &r_col) const {
	
	int row=p_mouse.y;
	row-=cache.style_normal->get_margin(MARGIN_TOP);
	row/=get_row_height();
	
	if (row<0 || row>=get_visible_rows())
		return false;
	
	row+=cursor.line_ofs;
	int col=0;
	
	if (row>=text.size()) {
		
		row=text.size()-1;
		col=text[row].size();
	} else {
		
		col=p_mouse.x-(cache.style_normal->get_margin(MARGIN_LEFT)+cache.line_number_w);
		col+=cursor.x_ofs;
		col=get_char_pos_for( col, get_line(row) );
	}
	
	r_row=row;
	r_col=col;
	return true;
}

void TextEdit::_input_event(const InputEvent& p_input_event) {
	
	switch(p_input_event.type) {
		
		case InputEvent::MOUSE_BUTTON: {
			
			const InputEventMouseButton &mb=p_input_event.mouse_button;
			
			if (completion_active && completion_rect.has_point(Point2(mb.x,mb.y))) {
				
				if (!mb.pressed)
					return;
				
				if (mb.button_index==BUTTON_WHEEL_UP) {
					if (completion_index>0) {
						completion_index--;
						completion_current=completion_options[completion_index];
						update();
					}
					
				}
				if (mb.button_index==BUTTON_WHEEL_DOWN) {
					
					if (completion_index<completion_options.size()-1) {
						completion_index++;
						completion_current=completion_options[completion_index];
						update();
					}
				}
				
				if (mb.button_index==BUTTON_LEFT) {
					
					completion_index=CLAMP(completion_line_ofs+(mb.y-completion_rect.pos.y)/get_row_height(),0,completion_options.size()-1);
					
					completion_current=completion_options[completion_index];
					update();
					if (mb.doubleclick)
						_confirm_completion();
				}
				return;
			} else {
				_cancel_completion();
				_cancel_code_hint();
			}
			
			if (mb.pressed) {
				if (mb.button_index==BUTTON_WHEEL_UP) {
					v_scroll->set_val( v_scroll->get_val() -3 );
				}
				if (mb.button_index==BUTTON_WHEEL_DOWN) {
					v_scroll->set_val( v_scroll->get_val() +3 );
				}
				if (mb.button_index==BUTTON_LEFT) {
					
					int row,col;
					if (!_get_mouse_pos(Point2i(mb.x,mb.y), row,col))
						return;
					
					int prev_col=cursor.column;
					int prev_line=cursor.line;
					
					
					
					cursor_set_line( row );
					cursor_set_column( col );
					
					if (mb.mod.shift && (cursor.column!=prev_col || cursor.line!=prev_line)) {

						if (!selection.active) {
							selection.active=true;
							selection.selecting_mode=Selection::MODE_POINTER;
							selection.from_column=prev_col;
							selection.from_line=prev_line;
							selection.to_column=cursor.column;
							selection.to_line=cursor.line;

							if (selection.from_line>selection.to_line || (selection.from_line==selection.to_line && selection.from_column>selection.to_column)) {
								SWAP(selection.from_column,selection.to_column);
								SWAP(selection.from_line,selection.to_line);
								selection.shiftclick_left=false;
							} else {
								selection.shiftclick_left=true;
							}
							selection.selecting_line=prev_line;
							selection.selecting_column=prev_col;
							update();
						} else {

							if (cursor.line<selection.from_line || (cursor.line==selection.from_line && cursor.column<=selection.from_column)) {
								selection.from_column=cursor.column;
								selection.from_line=cursor.line;
							} else if (cursor.line>selection.to_line || (cursor.line==selection.to_line && cursor.column>=selection.to_column)) {
								selection.to_column=cursor.column;
								selection.to_line=cursor.line;

							} else if (!selection.shiftclick_left) {

								selection.from_column=cursor.column;
								selection.from_line=cursor.line;
							} else {

								selection.to_column=cursor.column;
								selection.to_line=cursor.line;
							}

							if (selection.from_line>selection.to_line || (selection.from_line==selection.to_line && selection.from_column>selection.to_column)) {
								SWAP(selection.from_column,selection.to_column);
								SWAP(selection.from_line,selection.to_line);
							}
							update();
						}






						
					} else {
						
						//if sel active and dblick last time < something
						
						//else
						selection.active=false;
						selection.selecting_mode=Selection::MODE_POINTER;
						selection.selecting_line=row;
						selection.selecting_column=col;
					}
					
					
					if (!mb.doubleclick && (OS::get_singleton()->get_ticks_msec()-last_dblclk)<600) {
						//tripleclick select line
						select(cursor.line,0,cursor.line,text[cursor.line].length());
						last_dblclk=0;
						
					} else if (mb.doubleclick && text[cursor.line].length()) {
						
						//doubleclick select world
						String s = text[cursor.line];
						int beg=CLAMP(cursor.column,0,s.length());
						int end=beg;
						
						if (s[beg]>32 || beg==s.length()) {
							
							bool symbol = beg < s.length() &&  _is_symbol(s[beg]); //not sure if right but most editors behave like this
							
							while(beg>0 && s[beg-1]>32 && (symbol==_is_symbol(s[beg-1]))) {
								beg--;
							}
							while(end<s.length() && s[end+1]>32 && (symbol==_is_symbol(s[end+1]))) {
								end++;
							}
							
							if (end<s.length())
								end+=1;
							
							select(cursor.line,beg,cursor.line,end);
						}
						
						last_dblclk = OS::get_singleton()->get_ticks_msec();
						
					}
					
					update();
				}
			} else {
				
				selection.selecting_mode=Selection::MODE_NONE;
				// notify to show soft keyboard
				notification(NOTIFICATION_FOCUS_ENTER);
			}
			
		} break;
		case InputEvent::MOUSE_MOTION: {
			
			const InputEventMouseMotion &mm=p_input_event.mouse_motion;
			
			if (mm.button_mask&BUTTON_MASK_LEFT) {
				
				int row,col;
				if (!_get_mouse_pos(Point2i(mm.x,mm.y), row,col))
					return;
				
				if (selection.selecting_mode==Selection::MODE_POINTER) {
					
					select(selection.selecting_line,selection.selecting_column,row,col);
					
					cursor_set_line( row );
					cursor_set_column( col );
					update();
					
				}
				
			}
			
		} break;
			
		case InputEvent::KEY: {
			
			InputEventKey k=p_input_event.key;
			
			if (!k.pressed)
				return;
			
			if (completion_active) {
				if (readonly)
					break;

				bool valid=true;
				if (k.mod.command || k.mod.meta)
					valid=false;
				
				if (valid) {
					
					if (!k.mod.alt) {
						if (k.scancode==KEY_UP) {

							if (completion_index>0) {
								completion_index--;
								completion_current=completion_options[completion_index];
								update();
							}
							accept_event();
							return;
						}


						if (k.scancode==KEY_DOWN) {

							if (completion_index<completion_options.size()-1) {
								completion_index++;
								completion_current=completion_options[completion_index];
								update();
							}
							accept_event();
							return;
						}

						if (k.scancode==KEY_PAGEUP) {

							completion_index-=get_constant("completion_lines");
							if (completion_index<0)
								completion_index=0;
							completion_current=completion_options[completion_index];
							update();
							accept_event();
							return;
						}


						if (k.scancode==KEY_PAGEDOWN) {

							completion_index+=get_constant("completion_lines");
							if (completion_index>=completion_options.size())
								completion_index=completion_options.size()-1;
							completion_current=completion_options[completion_index];
							update();
							accept_event();
							return;
						}

						if (k.scancode==KEY_HOME) {

							completion_index=0;
							completion_current=completion_options[completion_index];
							update();
							accept_event();
							return;
						}

						if (k.scancode==KEY_END) {

							completion_index=completion_options.size()-1;
							completion_current=completion_options[completion_index];
							update();
							accept_event();
							return;
						}


						if (k.scancode==KEY_DOWN) {

							if (completion_index<completion_options.size()-1) {
								completion_index++;
								completion_current=completion_options[completion_index];
								update();
							}
							accept_event();
							return;
						}

						if (k.scancode==KEY_RETURN || k.scancode==KEY_TAB) {

							_confirm_completion();
							accept_event();
							return;
						}

						if (k.scancode==KEY_BACKSPACE) {

							backspace_at_cursor();
							_update_completion_candidates();
							accept_event();
							return;
						}


						if (k.scancode==KEY_SHIFT) {
							accept_event();
							return;
						}
					}
					
					if (k.unicode>32) {
						
						if (cursor.column<text[cursor.line].length() && text[cursor.line][cursor.column]==k.unicode) {
							//same char, move ahead
							cursor_set_column(cursor.column+1);
							
						} else {
							//different char, go back
							const CharType chr[2] = {k.unicode, 0};
							if(auto_brace_completion_enabled && _is_pair_symbol(chr[0])) {
								_consume_pair_symbol(chr[0]);
							} else {
								_insert_text_at_cursor(chr);
							}
						}
						
						_update_completion_candidates();
						accept_event();
						
						return;
					}
				}
				
				_cancel_completion();
				
			}
			
			/* TEST CONTROL FIRST!! */
			
			// some remaps for duplicate functions..
			if (k.mod.command && !k.mod.shift && !k.mod.alt && !k.mod.meta && k.scancode==KEY_INSERT) {
				
				k.scancode=KEY_C;
			}
			if (!k.mod.command && k.mod.shift && !k.mod.alt && !k.mod.meta && k.scancode==KEY_INSERT) {
				
				k.scancode=KEY_V;
				k.mod.command=true;
				k.mod.shift=false;
			}
			
			// stuff to do when selection is active..
			
			if (selection.active) {

				if (readonly)
					break;

				bool clear=false;
				bool unselect=false;
				bool dobreak=false;
				
				switch(k.scancode) {
					
					case KEY_TAB: {
						
						String txt = _base_get_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
						String prev_txt=txt;
						
						if (k.mod.shift) {
							
							for(int i=0;i<txt.length();i++) {
								if (((i>0 && txt[i-1]=='\n') || (i==0 /*&& selection.from_column==0*/)) && (txt[i]=='\t' || txt[i]==' ')) {
									txt.remove(i);
									//i--;
								}
							}
						} else {
							
							for(int i=0;i<txt.length();i++) {
								
								if (((i>0 && txt[i-1]=='\n') || (i==0 /*&& selection.from_column==0*/))) {
									txt=txt.insert(i,"\t");
									//i--;
								}
							}
						}
						
						if (txt!=prev_txt) {
							
							int sel_line=selection.from_line;
							int sel_column=selection.from_column;
							
							cursor_set_line(selection.from_line);
							cursor_set_column(selection.from_column);
							_begin_compex_operation();
							_remove_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
							_insert_text_at_cursor(txt);
							_end_compex_operation();
							selection.active=true;
							selection.from_column=sel_column;
							selection.from_line=sel_line;
							selection.to_column=cursor.column;
							selection.to_line=cursor.line;
							update();
						}
						
						dobreak=true;
						accept_event();
						
					} break;
					case KEY_X:
					case KEY_C:
						//special keys often used with control, wait...
						clear=(!k.mod.command || k.mod.shift || k.mod.alt );
						break;
					case KEY_DELETE:
					case KEY_BACKSPACE:
						accept_event();
						clear=true; dobreak=true;
						break;
					case KEY_LEFT:
					case KEY_RIGHT:
					case KEY_UP:
					case KEY_DOWN:
					case KEY_PAGEUP:
					case KEY_PAGEDOWN:
					case KEY_HOME:
					case KEY_END:
						// ignore arrows if any modifiers are held (shift = selecting, others may be used for editor hotkeys)
						if (k.mod.command || k.mod.shift || k.mod.alt || k.mod.command)
							break;
						unselect=true;
						break;
						
					default:
						if (k.unicode>=32 && !k.mod.command && !k.mod.alt && !k.mod.meta)
							clear=true;
						if (auto_brace_completion_enabled && _is_pair_left_symbol(k.unicode))
							clear=false;
				}
				
				if (unselect) {
					selection.active=false;
					selection.selecting_mode=Selection::MODE_NONE;
					update();
				}
				if (clear) {
					
					selection.active=false;
					update();
					_remove_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
					cursor_set_line(selection.from_line);
					cursor_set_column(selection.from_column);
					update();
				}
				if (dobreak)
					break;
			}
			
			selection.selecting_test=false;
			
			bool scancode_handled=true;
			
			// special scancode test...
			
			switch (k.scancode) {
				
				case KEY_ENTER:
				case KEY_RETURN: {

					if (readonly)
						break;

					String ins="\n";
					
					//keep indentation
					for(int i=0;i<text[cursor.line].length();i++) {
						if (text[cursor.line][i]=='\t')
							ins+="\t";
						else
							break;
					}
					
					_insert_text_at_cursor(ins);
					_push_current_op();
					
				} break;
				case KEY_ESCAPE: {
					if (completion_hint!="") {
						completion_hint="";
						update();
						
					}
				} break;
				case KEY_TAB: {
					
					if (readonly)
						break;
					
					if (selection.active) {
						
						
					} else {
						if (k.mod.shift) {
							
							int cc = cursor.column;
							if (cc>0 && cc<=text[cursor.line].length() && text[cursor.line][cursor.column-1]=='\t') {
								//simple unindent
								
								backspace_at_cursor();
							}
						} else {
							//simple indent
							_insert_text_at_cursor("\t");
						}
					}
					
				} break;
				case KEY_BACKSPACE: {
					if (readonly)
						break;
					backspace_at_cursor();
					
				} break;
				case KEY_LEFT: {
					
					if (k.mod.shift)
						_pre_shift_selection();
					
#ifdef APPLE_STYLE_KEYS
					if (k.mod.command) {
						cursor_set_column(0);
					} else if (k.mod.alt) {
						
#else
					if (k.mod.alt) {
						scancode_handled=false;
						break;
					} else if (k.mod.command) {
#endif
						bool prev_char=false;
						int cc=cursor.column;
						while (cc>0) {
							
							bool ischar=_is_text_char(text[cursor.line][cc-1]);
							
							if (prev_char && !ischar)
								break;
							
							prev_char=ischar;
							cc--;
							
						}
						
						cursor_set_column(cc);
						
					} else if (cursor.column==0) {
						
						if (cursor.line>0) {
							cursor_set_line(cursor.line-1);
							cursor_set_column(text[cursor.line].length());
						}
					} else {
						cursor_set_column(cursor_get_column()-1);
					}
					
					if (k.mod.shift)
						_post_shift_selection();
					
				} break;
				case KEY_RIGHT: {
					
					if (k.mod.shift)
						_pre_shift_selection();
					
#ifdef APPLE_STYLE_KEYS
					if (k.mod.command) {
						cursor_set_column(text[cursor.line].length());
					} else if (k.mod.alt) {
#else
					if (k.mod.alt) {
						scancode_handled=false;
						break;
					} else if (k.mod.command) {
#endif
						bool prev_char=false;
						int cc=cursor.column;
						while (cc<text[cursor.line].length()) {
							
							bool ischar=_is_text_char(text[cursor.line][cc]);
							
							if (prev_char && !ischar)
								break;
							prev_char=ischar;
							cc++;
						}
						
						cursor_set_column(cc);
						
					} else if (cursor.column==text[cursor.line].length()) {
						
						if (cursor.line<text.size()-1) {
							cursor_set_line(cursor.line+1);
							cursor_set_column(0);
						}
					} else {
						cursor_set_column(cursor_get_column()+1);
					}
					
					if (k.mod.shift)
						_post_shift_selection();
					
				} break;
				case KEY_UP: {
					
					if (k.mod.shift)
						_pre_shift_selection();
					if (k.mod.alt) {
						scancode_handled=false;
						break;
					}
#ifdef APPLE_STYLE_KEYS
					if (k.mod.command)
						cursor_set_line(0);
					else
#endif
						cursor_set_line(cursor_get_line()-1);
					
					if (k.mod.shift)
						_post_shift_selection();
					_cancel_code_hint();
					
				} break;
				case KEY_DOWN: {
					
					if (k.mod.shift)
						_pre_shift_selection();
					if (k.mod.alt) {
						scancode_handled=false;
						break;
					}
#ifdef APPLE_STYLE_KEYS
					if (k.mod.command)
						cursor_set_line(text.size()-1);
					else
#endif
						cursor_set_line(cursor_get_line()+1);
					
					if (k.mod.shift)
						_post_shift_selection();
					_cancel_code_hint();
					
				} break;
					
				case KEY_DELETE: {
					
					if (readonly)
						break;
					int curline_len = text[cursor.line].length();
					
					if (cursor.line==text.size()-1 && cursor.column==curline_len)
						break; //nothing to do
					
					int next_line = cursor.column<curline_len?cursor.line:cursor.line+1;
					int next_column = cursor.column<curline_len?(cursor.column+1):0;
					_remove_text(cursor.line,cursor.column,next_line,next_column);
					update();
				} break;
#ifdef APPLE_STYLE_KEYS
				case KEY_HOME: {
					
					
					if (k.mod.shift)
						_pre_shift_selection();
					
					cursor_set_line(0);
					
					if (k.mod.shift)
						_post_shift_selection();
					
				} break;
				case KEY_END: {
					
					if (k.mod.shift)
						_pre_shift_selection();
					
					cursor_set_line(text.size()-1);
					
					if (k.mod.shift)
						_post_shift_selection();
					
				} break;
					
#else
				case KEY_HOME: {
					
					
					if (k.mod.shift)
						_pre_shift_selection();
					
					// compute whitespace symbols seq length
					int current_line_whitespace_len = 0;
					while(current_line_whitespace_len < text[cursor.line].length()) {
						CharType c = text[cursor.line][current_line_whitespace_len];
						if(c != '\t' && c != ' ')
							break;
						current_line_whitespace_len++;
					}
					
					if(cursor_get_column() == current_line_whitespace_len)
						cursor_set_column(0);
					else
						cursor_set_column(current_line_whitespace_len);
					
					if (k.mod.command)
						cursor_set_line(0);
					
					if (k.mod.shift)
						_post_shift_selection();
					
				} break;
				case KEY_END: {
					
					if (k.mod.shift)
						_pre_shift_selection();
					
					if (k.mod.command)
						cursor_set_line(text.size()-1);
					cursor_set_column(text[cursor.line].length());
					
					if (k.mod.shift)
						_post_shift_selection();
					
				} break;
#endif
				case KEY_PAGEUP: {
					
					if (k.mod.shift)
						_pre_shift_selection();
					
					cursor_set_line(cursor_get_line()-get_visible_rows());
					
					if (k.mod.shift)
						_post_shift_selection();
					
				} break;
				case KEY_PAGEDOWN: {
					
					if (k.mod.shift)
						_pre_shift_selection();
					
					cursor_set_line(cursor_get_line()+get_visible_rows());
					
					if (k.mod.shift)
						_post_shift_selection();
					
				} break;
				case KEY_A: {
					
					if (!k.mod.command || k.mod.shift || k.mod.alt) {
						scancode_handled=false;
						break;
					}
					
					if (text.size()==1 && text[0].length()==0)
						break;
					selection.active=true;
					selection.from_line=0;
					selection.from_column=0;
					selection.to_line=text.size()-1;
					selection.to_column=text[selection.to_line].size();
					selection.selecting_mode=Selection::MODE_NONE;
					update();
					
				} break;
				case KEY_X: {
					
					if (!k.mod.command || k.mod.shift || k.mod.alt) {
						scancode_handled=false;
						break;
					}
					
					if (!selection.active){
						
						String clipboard = text[cursor.line];
						OS::get_singleton()->set_clipboard(clipboard);
						cursor_set_line(cursor.line);
						cursor_set_column(0);
						_remove_text(cursor.line,0,cursor.line,text[cursor.line].length());
						
						backspace_at_cursor();
						update();
						cursor_set_line(cursor.line+1);
						cut_copy_line = true;
						
					}
					else
					{
						
						String clipboard = _base_get_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
						OS::get_singleton()->set_clipboard(clipboard);
						
						cursor_set_line(selection.from_line);
						cursor_set_column(selection.from_column);
						
						_remove_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
						selection.active=false;
						selection.selecting_mode=Selection::MODE_NONE;
						update();
						cut_copy_line = false;
					}
					
				} break;
				case KEY_C: {
					
					if (!k.mod.command || k.mod.shift || k.mod.alt) {
						scancode_handled=false;
						break;
					}
					
					if (!selection.active){
						String clipboard = _base_get_text(cursor.line,0,cursor.line,text[cursor.line].length());
						OS::get_singleton()->set_clipboard(clipboard);
						cut_copy_line = true;
					}
					else{
						String clipboard = _base_get_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
						OS::get_singleton()->set_clipboard(clipboard);
						cut_copy_line = false;
					}
				} break;
				case KEY_Z: {
					
					if (!k.mod.command) {
						scancode_handled=false;
						break;
					}
					
					if (k.mod.shift)
						redo();
					else
						undo();
				} break;
				case KEY_V: {
					
					if (!k.mod.command || k.mod.shift || k.mod.alt) {
						scancode_handled=false;
						break;
					}
					
					String clipboard = OS::get_singleton()->get_clipboard();
					
					if (selection.active) {
						selection.active=false;
						_remove_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
						cursor_set_line(selection.from_line);
						cursor_set_column(selection.from_column);
						
					}
					else if (cut_copy_line)
					{
						cursor_set_column(0);
						String ins="\n";
						clipboard += ins;
					}
					
					_insert_text_at_cursor(clipboard);
					
					update();
				} break;
				case KEY_SPACE: {
#ifdef OSX_ENABLED
					if (completion_enabled && k.mod.meta) { //cmd-space is spotlight shortcut in OSX
#else
					if (completion_enabled && k.mod.command) {
#endif
						
						query_code_comple();
						scancode_handled=true;
					} else {
						scancode_handled=false;
					}
					
				} break;
					
				case KEY_U:{
					if (!k.mod.command || k.mod.shift) {
						scancode_handled=false;
						break;
					}
					else {
						if (selection.active) {
							int ini = selection.from_line;
							int end = selection.to_line;
							for (int i=ini; i<= end; i++)
							{
								if (text[i][0] == '#')
									_remove_text(i,0,i,1);
							}
						}
						else{
							if (text[cursor.line][0] == '#')
								_remove_text(cursor.line,0,cursor.line,1);
						}
						update();
					}
					break;}
					
				default: {
					
					scancode_handled=false;
				} break;
					
			}
			
			if (scancode_handled)
				accept_event();
			/*
	    if (!scancode_handled && !k.mod.command && !k.mod.alt) {
	    
		if (k.unicode>=32) {
		
		    if (readonly)
			break;
			
		    accept_event();
		} else {
		
		    break;
		}
	    }
*/
			if (!scancode_handled && !k.mod.command) { //for german kbds
				
				if (k.unicode>=32) {
					
					if (readonly)
						break;
					
					const CharType chr[2] = {k.unicode, 0};
					
					if(auto_brace_completion_enabled && _is_pair_symbol(chr[0])) {
						_consume_pair_symbol(chr[0]);
					} else {
						_insert_text_at_cursor(chr);
					}
					
					accept_event();
				} else {
					
					break;
				}
			}
			
			
			if (!selection.selecting_test) {
				
				selection.selecting_mode=Selection::MODE_NONE;
			}
			
			return;
		} break;
			
	}
	
}


void TextEdit::_pre_shift_selection() {
	
	
	if (!selection.active || selection.selecting_mode!=Selection::MODE_SHIFT) {
		
		selection.selecting_line=cursor.line;
		selection.selecting_column=cursor.column;
		selection.active=true;
		selection.selecting_mode=Selection::MODE_SHIFT;
	}
}

void TextEdit::_post_shift_selection() {
	
	
	if (selection.active && selection.selecting_mode==Selection::MODE_SHIFT) {
		
		select(selection.selecting_line,selection.selecting_column,cursor.line,cursor.column);
		update();
	}
	
	
	selection.selecting_test=true;
}

/**** TEXT EDIT CORE API ****/

void TextEdit::_base_insert_text(int p_line, int p_char,const String& p_text,int &r_end_line,int &r_end_column) {
	
	//save for undo...
	ERR_FAIL_INDEX(p_line,text.size());
	ERR_FAIL_COND(p_char<0);
	
	/* STEP 1 add spaces if the char is greater than the end of the line */
	while(p_char>text[p_line].length()) {
		
		text.set(p_line,text[p_line]+String::chr(' '));
	}
	
	/* STEP 2 separate dest string in pre and post text */
	
	String preinsert_text = text[p_line].substr(0,p_char);
	String postinsert_text = text[p_line].substr(p_char,text[p_line].size());
	
	/* STEP 3 remove \r from source text and separate in substrings */
	
	//buh bye \r and split
	Vector<String> substrings = p_text.replace("\r","").split("\n");
	
	
	for(int i=0;i<substrings.size();i++) {
		//insert the substrings
		
		if (i==0) {
			
			text.set(p_line,preinsert_text+substrings[i]);
		} else {
			
			text.insert(p_line+i,substrings[i]);
		}
		
		if (i==substrings.size()-1){
			
			text.set(p_line+i,text[p_line+i]+postinsert_text);
		}
	}
	
	r_end_line=p_line+substrings.size()-1;
	r_end_column=text[r_end_line].length()-postinsert_text.length();
	
	if (!text_changed_dirty && !setting_text) {
		if (is_inside_tree())
			MessageQueue::get_singleton()->push_call(this,"_text_changed_emit");
		text_changed_dirty=true;
	}
	
}

String TextEdit::_base_get_text(int p_from_line, int p_from_column,int p_to_line,int p_to_column) const {
	
	ERR_FAIL_INDEX_V(p_from_line,text.size(),String());
	ERR_FAIL_INDEX_V(p_from_column,text[p_from_line].length()+1,String());
	ERR_FAIL_INDEX_V(p_to_line,text.size(),String());
	ERR_FAIL_INDEX_V(p_to_column,text[p_to_line].length()+1,String());
	ERR_FAIL_COND_V(p_to_line < p_from_line ,String()); // from > to
	ERR_FAIL_COND_V(p_to_line == p_from_line && p_to_column<p_from_column,String()); // from > to
	
	String ret;
	
	for(int i=p_from_line;i<=p_to_line;i++) {
		
		int begin = (i==p_from_line)?p_from_column:0;
		int end = (i==p_to_line)?p_to_column:text[i].length();
		
		if (i>p_from_line)
			ret+="\n";
		ret+=text[i].substr(begin,end-begin);
	}
	
	return ret;
}

void TextEdit::_base_remove_text(int p_from_line, int p_from_column,int p_to_line,int p_to_column) {
	
	ERR_FAIL_INDEX(p_from_line,text.size());
	ERR_FAIL_INDEX(p_from_column,text[p_from_line].length()+1);
	ERR_FAIL_INDEX(p_to_line,text.size());
	ERR_FAIL_INDEX(p_to_column,text[p_to_line].length()+1);
	ERR_FAIL_COND(p_to_line < p_from_line ); // from > to
	ERR_FAIL_COND(p_to_line == p_from_line && p_to_column<p_from_column); // from > to
	
	
	String pre_text = text[p_from_line].substr(0,p_from_column);
	String post_text = text[p_to_line].substr(p_to_column,text[p_to_line].length());
	
	for(int i=p_from_line;i<p_to_line;i++) {
		
		text.remove(p_from_line+1);
	}
	
	text.set(p_from_line,pre_text+post_text);
	
	if (!text_changed_dirty && !setting_text) {
		if (is_inside_tree())
			MessageQueue::get_singleton()->push_call(this,"_text_changed_emit");
		text_changed_dirty=true;
	}
}

void TextEdit::_insert_text(int p_line, int p_char,const String& p_text,int *r_end_line,int *r_end_column) {
	
	if (!setting_text)
		idle_detect->start();
	
	if (undo_enabled) {
		_clear_redo();
	}
	
	int retline,retchar;
	_base_insert_text(p_line,p_char,p_text,retline,retchar);
	if (r_end_line)
		*r_end_line=retline;
	if (r_end_column)
		*r_end_column=retchar;
	
	if (!undo_enabled)
		return;
	
	/* UNDO!! */
	TextOperation op;
	op.type=TextOperation::TYPE_INSERT;
	op.from_line=p_line;
	op.from_column=p_char;
	op.to_line=retline;
	op.to_column=retchar;
	op.text=p_text;
	op.version=++version;
	op.chain_forward=false;
	op.chain_backward=false;
	
	//see if it shold just be set as current op
	if (current_op.type!=op.type) {
		_push_current_op();
		current_op=op;
		
		return; //set as current op, return
	}
	//see if it can be merged
	if (current_op.to_line!=p_line || current_op.to_column!=p_char) {
		_push_current_op();
		current_op=op;
		return; //set as current op, return
	}
	//merge current op
	
	current_op.text+=p_text;
	current_op.to_column=retchar;
	current_op.to_line=retline;
	current_op.version=op.version;
	
}

void TextEdit::_remove_text(int p_from_line, int p_from_column,int p_to_line,int p_to_column) {
	
	if (!setting_text)
		idle_detect->start();
	
	String text;
	if (undo_enabled) {
		_clear_redo();
		text=_base_get_text(p_from_line,p_from_column,p_to_line,p_to_column);
	}
	
	_base_remove_text(p_from_line,p_from_column,p_to_line,p_to_column);
	
	if (!undo_enabled)
		return;
	
	/* UNDO!! */
	TextOperation op;
	op.type=TextOperation::TYPE_REMOVE;
	op.from_line=p_from_line;
	op.from_column=p_from_column;
	op.to_line=p_to_line;
	op.to_column=p_to_column;
	op.text=text;
	op.version=++version;
	op.chain_forward=false;
	op.chain_backward=false;
	
	//see if it shold just be set as current op
	if (current_op.type!=op.type) {
		_push_current_op();
		current_op=op;
		return; //set as current op, return
	}
	//see if it can be merged
	if (current_op.from_line==p_to_line && current_op.from_column==p_to_column) {
		//basckace or similar
		current_op.text=text+current_op.text;
		current_op.from_line=p_from_line;
		current_op.from_column=p_from_column;
		return; //update current op
	}
	if (current_op.from_line==p_from_line && current_op.from_column==p_from_column) {
		
		//current_op.text=text+current_op.text;
		//current_op.from_line=p_from_line;
		//current_op.from_column=p_from_column;
		//return; //update current op
	}
	
	_push_current_op();
	current_op=op;
	
}


void TextEdit::_insert_text_at_cursor(const String& p_text) {
	
	int new_column,new_line;
	_insert_text(cursor.line,cursor.column,p_text,&new_line,&new_column);
	cursor_set_line(new_line);
	cursor_set_column(new_column);
	
	update();
}




int TextEdit::get_char_count() {
	
	int totalsize=0;
	
	for (int i=0;i<text.size();i++) {
		
		if (i>0)
			totalsize++; // incliude \n
		totalsize+=text[i].length();
	}
	
	return totalsize; // omit last \n
}

Size2 TextEdit::get_minimum_size() {
	
	return cache.style_normal->get_minimum_size();
}
int TextEdit::get_visible_rows() const {
	
	int total=cache.size.height;
	total-=cache.style_normal->get_minimum_size().height;
	total/=get_row_height();
	return total;
}
void TextEdit::adjust_viewport_to_cursor() {
	
	if (cursor.line_ofs>cursor.line)
		cursor.line_ofs=cursor.line;
	
	int visible_width=cache.size.width-cache.style_normal->get_minimum_size().width-cache.line_number_w;
	if (v_scroll->is_visible())
		visible_width-=v_scroll->get_combined_minimum_size().width;
	visible_width-=20; // give it a little more space
	
	
	//printf("rowofs %i, visrows %i, cursor.line %i\n",cursor.line_ofs,get_visible_rows(),cursor.line);
	
	int visible_rows = get_visible_rows();
	if (h_scroll->is_visible())
		visible_rows-=((h_scroll->get_combined_minimum_size().height-1)/get_row_height());
	
	if (cursor.line>=(cursor.line_ofs+visible_rows))
		cursor.line_ofs=cursor.line-visible_rows+1;
	if (cursor.line<cursor.line_ofs)
		cursor.line_ofs=cursor.line;
	
	int cursor_x = get_column_x_offset( cursor.column, text[cursor.line] );
	
	if (cursor_x>(cursor.x_ofs+visible_width))
		cursor.x_ofs=cursor_x-visible_width+1;
	
	if (cursor_x < cursor.x_ofs)
		cursor.x_ofs=cursor_x;
	
	update();
	/*
    get_range()->set_max(text.size());
    
    get_range()->set_page(get_visible_rows());
    
    get_range()->set((int)cursor.line_ofs);
*/
	
	
}

void TextEdit::cursor_set_column(int p_col) {
	
	if (p_col<0)
		p_col=0;
	
	cursor.column=p_col;
	if (cursor.column > get_line( cursor.line ).length())
		cursor.column=get_line( cursor.line ).length();
	
	cursor.last_fit_x=get_column_x_offset(cursor.column,get_line(cursor.line));
	
	adjust_viewport_to_cursor();
	
	if (!cursor_changed_dirty) {
		if (is_inside_tree())
			MessageQueue::get_singleton()->push_call(this,"_cursor_changed_emit");
		cursor_changed_dirty=true;
	}
	
}


void TextEdit::cursor_set_line(int p_row) {
	
	if (setting_row)
		return;
	
	setting_row=true;
	if (p_row<0)
		p_row=0;
	
	
	if (p_row>=(int)text.size())
		p_row=(int)text.size()-1;
	
	cursor.line=p_row;
	cursor.column=get_char_pos_for( cursor.last_fit_x, get_line( cursor.line) );
	
	
	adjust_viewport_to_cursor();
	
	setting_row=false;
	
	
	if (!cursor_changed_dirty) {
		if (is_inside_tree())
			MessageQueue::get_singleton()->push_call(this,"_cursor_changed_emit");
		cursor_changed_dirty=true;
	}
	
}


int TextEdit::cursor_get_column() const {
	
	return cursor.column;
}


int TextEdit::cursor_get_line() const {
	
	return cursor.line;
}



void TextEdit::_scroll_moved(double p_to_val) {
	
	if (updating_scrolls)
		return;
	
	if (h_scroll->is_visible())
		cursor.x_ofs=h_scroll->get_val();
	if (v_scroll->is_visible())
		cursor.line_ofs=v_scroll->get_val();
	update();
}





int TextEdit::get_row_height() const {
	
	return cache.font->get_height()+cache.line_spacing;
}

int TextEdit::get_char_pos_for(int p_px,String p_str) const {
	
	int px=0;
	int c=0;
	
	int tab_w = cache.font->get_char_size(' ').width*tab_size;
	
	while (c<p_str.length()) {
		
		int w=0;
		
		if (p_str[c]=='\t') {
			
			int left = px%tab_w;
			if (left==0)
				w=tab_w;
			else
				w=tab_w-px%tab_w; // is right...
			
		} else {
			
			w=cache.font->get_char_size(p_str[c],p_str[c+1]).width;
		}
		
		if (p_px<(px+w/2))
			break;
		px+=w;
		c++;
	}
	
	return c;
}

int TextEdit::get_column_x_offset(int p_char,String p_str) {
	
	int px=0;
	
	int tab_w = cache.font->get_char_size(' ').width*tab_size;
	
	for (int i=0;i<p_char;i++) {
		
		if (i>=p_str.length())
			break;
		
		if (p_str[i]=='\t') {
			
			int left = px%tab_w;
			if (left==0)
				px+=tab_w;
			else
				px+=tab_w-px%tab_w; // is right...
			
		} else {
			px+=cache.font->get_char_size(p_str[i],p_str[i+1]).width;
		}
	}
	
	return px;
	
}

void TextEdit::insert_text_at_cursor(const String& p_text) {
	
	if (selection.active) {
		
		cursor_set_line(selection.from_line);
		cursor_set_column(selection.from_column);
		
		_remove_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
		selection.active=false;
		selection.selecting_mode=Selection::MODE_NONE;
		
	}
	
	_insert_text_at_cursor(p_text);
	update();
	
}

Control::CursorShape TextEdit::get_cursor_shape(const Point2& p_pos) const {
	if(completion_active && completion_rect.has_point(p_pos)) {
		return CURSOR_ARROW;
	}
	return CURSOR_IBEAM;
}


void TextEdit::set_text(String p_text){
	
	setting_text=true;
	_clear();
	_insert_text_at_cursor(p_text);
	
	cursor.column=0;
	cursor.line=0;
	cursor.x_ofs=0;
	cursor.line_ofs=0;
	cursor.last_fit_x=0;
	cursor_set_line(0);
	cursor_set_column(0);
	update();
	setting_text=false;
	
	//get_range()->set(0);
};

String TextEdit::get_text() {
	String longthing;
	int len = text.size();
	for (int i=0;i<len;i++) {
		
		
		longthing+=text[i];
		if (i!=len-1)
			longthing+="\n";
	}
	
	return longthing;
	
};

String TextEdit::get_text_for_completion() {
	
	String longthing;
	int len = text.size();
	for (int i=0;i<len;i++) {
		
		if (i==cursor.line) {
			longthing+=text[i].substr(0,cursor.column);
			longthing+=String::chr(0xFFFF); //not unicode, represents the cursor
			longthing+=text[i].substr(cursor.column,text[i].size());
		} else {
			
			longthing+=text[i];
		}
		
		
		if (i!=len-1)
			longthing+="\n";
	}
	
	return longthing;
	
};


String TextEdit::get_line(int line) const {
	
	if (line<0 || line>=text.size())
		return "";
	
	return 	text[line];
	
};

void TextEdit::_clear() {
	
	clear_undo_history();
	text.clear();
	cursor.column=0;
	cursor.line=0;
	cursor.x_ofs=0;
	cursor.line_ofs=0;
	cursor.last_fit_x=0;
}



void TextEdit::clear() {
	
	setting_text=true;
	_clear();
	setting_text=false;
	
};

void TextEdit::set_readonly(bool p_readonly) {
	
	
	readonly=p_readonly;
}

void TextEdit::set_wrap(bool p_wrap) {
	
	wrap=p_wrap;
}

void TextEdit::set_max_chars(int p_max_chars) {
	
	max_chars=p_max_chars;
}

void TextEdit::_update_caches() {
	
	cache.style_normal=get_stylebox("normal");
	cache.style_focus=get_stylebox("focus");
	cache.font=get_font("font");
	cache.font_color=get_color("font_color");
	cache.font_selected_color=get_color("font_selected_color");
	cache.keyword_color=get_color("keyword_color");
	cache.selection_color=get_color("selection_color");
	cache.mark_color=get_color("mark_color");
	cache.current_line_color=get_color("current_line_color");
	cache.breakpoint_color=get_color("breakpoint_color");
	cache.brace_mismatch_color=get_color("brace_mismatch_color");
	cache.line_spacing=get_constant("line_spacing");
	cache.row_height = cache.font->get_height() + cache.line_spacing;
	cache.tab_icon=get_icon("tab");
	text.set_font(cache.font);
	
}


void TextEdit::clear_colors() {
	
	keywords.clear();
	color_regions.clear();;
	text.clear_caches();
	custom_bg_color=Color(0,0,0,0);
}

void TextEdit::set_custom_bg_color(const Color& p_color) {
	
	custom_bg_color=p_color;
	update();
}

void TextEdit::add_keyword_color(const String& p_keyword,const Color& p_color) {
	
	keywords[p_keyword]=p_color;
	update();
	
}

void TextEdit::add_color_region(const String& p_begin_key,const String& p_end_key,const Color &p_color,bool p_line_only) {
	
	color_regions.push_back(ColorRegion(p_begin_key,p_end_key,p_color,p_line_only));
	text.clear_caches();
	update();
	
}

void TextEdit::set_symbol_color(const Color& p_color) {
	
	symbol_color=p_color;
	update();
}

void TextEdit::set_syntax_coloring(bool p_enabled) {
	
	syntax_coloring=p_enabled;
	update();
}

bool TextEdit::is_syntax_coloring_enabled() const {
	
	return syntax_coloring;
}

void TextEdit::cut() {
	
	if (!selection.active)
		return;
	
	String clipboard = _base_get_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
	OS::get_singleton()->set_clipboard(clipboard);
	
	cursor_set_line(selection.from_line);
	cursor_set_column(selection.from_column);
	
	_remove_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
	selection.active=false;
	selection.selecting_mode=Selection::MODE_NONE;
	update();
	
}

void TextEdit::copy() {
	
	if (!selection.active)
		return;
	
	String clipboard = _base_get_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
	OS::get_singleton()->set_clipboard(clipboard);
	
}
void TextEdit::paste() {
	
	if (selection.active) {
		
		cursor_set_line(selection.from_line);
		cursor_set_column(selection.from_column);
		
		_remove_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
		selection.active=false;
		selection.selecting_mode=Selection::MODE_NONE;
		
	}
	
	String clipboard = OS::get_singleton()->get_clipboard();
	_insert_text_at_cursor(clipboard);
	update();
	
}

void TextEdit::select_all() {
	
	if (text.size()==1 && text[0].length()==0)
		return;
	selection.active=true;
	selection.from_line=0;
	selection.from_column=0;
	selection.to_line=text.size()-1;
	selection.to_column=text[selection.to_line].size();
	selection.selecting_mode=Selection::MODE_NONE;
	update();
	
}


void TextEdit::deselect() {
	
	selection.active=false;
	update();
}

void TextEdit::select(int p_from_line,int p_from_column,int p_to_line,int p_to_column) {
	
	if (p_from_line>=text.size())
		p_from_line=text.size()-1;
	if (p_from_column>=text[p_from_line].length())
		p_from_column=text[p_from_line].length();
	
	if (p_to_line>=text.size())
		p_to_line=text.size()-1;
	if (p_to_column>=text[p_to_line].length())
		p_to_column=text[p_to_line].length();
	
	selection.from_line=p_from_line;
	selection.from_column=p_from_column;
	selection.to_line=p_to_line;
	selection.to_column=p_to_column;
	
	selection.active=true;
	
	if (selection.from_line==selection.to_line) {
		
		if (selection.from_column==selection.to_column) {
			
			selection.active=false;
			
		} else if (selection.from_column>selection.to_column) {
			
			SWAP( selection.from_column, selection.to_column );
		}
	} else if (selection.from_line>selection.to_line) {
		
		SWAP( selection.from_line, selection.to_line );
		SWAP( selection.from_column, selection.to_column );
	}
	
	
	update();
}

bool TextEdit::is_selection_active() const {
	
	return selection.active;
}
int TextEdit::get_selection_from_line() const {
	
	ERR_FAIL_COND_V(!selection.active,-1);
	return selection.from_line;
	
}
int TextEdit::get_selection_from_column() const {
	
	ERR_FAIL_COND_V(!selection.active,-1);
	return selection.from_column;
	
}
int TextEdit::get_selection_to_line() const {
	
	ERR_FAIL_COND_V(!selection.active,-1);
	return selection.to_line;
	
}
int TextEdit::get_selection_to_column() const {
	
	ERR_FAIL_COND_V(!selection.active,-1);
	return selection.to_column;
	
}

String TextEdit::get_selection_text() const {
	
	if (!selection.active)
		return "";
	
	return _base_get_text(selection.from_line,selection.from_column,selection.to_line,selection.to_column);
	
}

String TextEdit::get_word_under_cursor() const {
	
	int prev_cc = cursor.column;
	while(prev_cc >0) {
		bool is_char = _is_text_char(text[cursor.line][prev_cc-1]);
		if (!is_char)
			break;
		--prev_cc;
	}
	
	int next_cc = cursor.column;
	while(next_cc<text[cursor.line].length()) {
		bool is_char = _is_text_char(text[cursor.line][next_cc]);
		if(!is_char)
			break;
		++ next_cc;
	}
	if (prev_cc == cursor.column || next_cc == cursor.column)
		return "";
	return text[cursor.line].substr(prev_cc, next_cc-prev_cc);
}

DVector<int> TextEdit::_search_bind(const String &p_key,uint32_t p_search_flags, int p_from_line,int p_from_column) const {
	
	int col,line;
	if (search(p_key,p_search_flags,p_from_line,p_from_column,col,line)) {
		DVector<int> result;
		result.resize(2);
		result.set(0,line);
		result.set(1,col);
		return result;
		
	} else {
		
		return DVector<int>();
	}
}

bool TextEdit::search(const String &p_key,uint32_t p_search_flags, int p_from_line, int p_from_column,int &r_line,int &r_column) const {
	
	if (p_key.length()==0)
		return false;
	ERR_FAIL_INDEX_V(p_from_line,text.size(),false);
	ERR_FAIL_INDEX_V(p_from_column,text[p_from_line].length()+1,false);
	
	//search through the whole documment, but start by current line
	
	int line=-1;
	int pos=-1;
	
	line=p_from_line;
	
	for(int i=0;i<text.size()+1;i++) {
		//backwards is broken...
		//int idx=(p_search_flags&SEARCH_BACKWARDS)?(text.size()-i):i; //do backwards seearch
		
		
		if (line<0) {
			line=text.size()-1;
		}
		if (line==text.size()) {
			line=0;
		}
		
		String text_line = text[line];
		int from_column=0;
		if (line==p_from_line) {
			
			if (i==text.size()) {
				//wrapped
				
				if (p_search_flags&SEARCH_BACKWARDS) {
					text_line=text_line.substr(from_column,text_line.length());
					from_column=text_line.length();
				} else {
					text_line=text_line.substr(0,from_column);
					from_column=0;
				}
				
			} else {
				
				from_column=p_from_column;
			}
			
			
		} else {
			//text_line=text_line.substr(0,p_from_column); //wrap around for missing begining.
			if (p_search_flags&SEARCH_BACKWARDS)
				from_column=text_line.length()-1;
			else
				from_column=0;
		}
		
		pos=-1;
		
		if (!(p_search_flags&SEARCH_BACKWARDS)) {
			
			pos = (p_search_flags&SEARCH_MATCH_CASE)?text_line.find(p_key,from_column):text_line.findn(p_key,from_column);
		} else {
			
			pos = (p_search_flags&SEARCH_MATCH_CASE)?text_line.rfind(p_key,from_column):text_line.rfindn(p_key,from_column);
		}
		
		if (pos!=-1 && (p_search_flags&SEARCH_WHOLE_WORDS)) {
			//validate for whole words
			if (pos>0 && _is_text_char(text_line[pos-1]))
				pos=-1;
			else if (_is_text_char(text_line[pos+p_key.length()]))
				pos=-1;
		}
		
		if (pos!=-1)
			break;
		
		if (p_search_flags&SEARCH_BACKWARDS)
			line--;
		else
			line++;
		
	}
	
	if (pos==-1) {
		r_line=-1;
		r_column=-1;
		return false;
	}
	
	r_line=line;
	r_column=pos;
	
	
	return true;
}

void TextEdit::_cursor_changed_emit() {
	
	emit_signal("cursor_changed");
	cursor_changed_dirty=false;
}

void TextEdit::_text_changed_emit() {
	
	emit_signal("text_changed");
	text_changed_dirty=false;
}

void TextEdit::set_line_as_marked(int p_line,bool p_marked) {
	
	ERR_FAIL_INDEX(p_line,text.size());
	text.set_marked(p_line,p_marked);
	update();
}

bool TextEdit::is_line_set_as_breakpoint(int p_line) const {
	
	ERR_FAIL_INDEX_V(p_line,text.size(),false);
	return text.is_breakpoint(p_line);
	
}

void TextEdit::set_line_as_breakpoint(int p_line,bool p_breakpoint) {
	
	
	ERR_FAIL_INDEX(p_line,text.size());
	text.set_breakpoint(p_line,p_breakpoint);
	update();
}

void TextEdit::get_breakpoints(List<int> *p_breakpoints) const {
	
	for(int i=0;i<text.size();i++) {
		if (text.is_breakpoint(i))
			p_breakpoints->push_back(i);
	}
}

int TextEdit::get_line_count() const {
	
	return text.size();
}

void TextEdit::_do_text_op(const TextOperation& p_op, bool p_reverse) {
	
	ERR_FAIL_COND(p_op.type==TextOperation::TYPE_NONE);
	
	bool insert = p_op.type==TextOperation::TYPE_INSERT;
	if (p_reverse)
		insert=!insert;
	
	if (insert) {
		
		int check_line;
		int check_column;
		_base_insert_text(p_op.from_line,p_op.from_column,p_op.text,check_line,check_column);
		ERR_FAIL_COND( check_line != p_op.to_line ); // BUG
		ERR_FAIL_COND( check_column != p_op.to_column ); // BUG
	} else {
		
		_base_remove_text(p_op.from_line,p_op.from_column,p_op.to_line,p_op.to_column);
	}
	
}

void TextEdit::_clear_redo() {
	
	if (undo_stack_pos==NULL)
		return; //nothing to clear
	
	_push_current_op();
	
	while (undo_stack_pos)	{
		List<TextOperation>::Element *elem = undo_stack_pos;
		undo_stack_pos=undo_stack_pos->next();
		undo_stack.erase(elem);
	}
}


void TextEdit::undo() {
	
	_push_current_op();
	
	if (undo_stack_pos==NULL) {
		
		if (!undo_stack.size())
			return; //nothing to undo
		
		undo_stack_pos=undo_stack.back();
		
	} else if (undo_stack_pos==undo_stack.front())
		return; // at the bottom of the undo stack
	else
		undo_stack_pos=undo_stack_pos->prev();
	
	_do_text_op( undo_stack_pos->get(),true);
	if(undo_stack_pos->get().chain_backward) {
		do {
			undo_stack_pos = undo_stack_pos->prev();
			_do_text_op(undo_stack_pos->get(), true);
		} while(!undo_stack_pos->get().chain_forward);
	}
	
	cursor_set_line(undo_stack_pos->get().from_line);
	cursor_set_column(undo_stack_pos->get().from_column);
	update();
}

void TextEdit::redo() {
	
	_push_current_op();
	
	if (undo_stack_pos==NULL)
		return; //nothing to do.
	
	_do_text_op(undo_stack_pos->get(), false);
	if(undo_stack_pos->get().chain_forward) {
		do {
			undo_stack_pos=undo_stack_pos->next();
			_do_text_op(undo_stack_pos->get(), false);
		} while(!undo_stack_pos->get().chain_backward);
	}
	cursor_set_line(undo_stack_pos->get().from_line);
	cursor_set_column(undo_stack_pos->get().from_column);
	undo_stack_pos=undo_stack_pos->next();
	update();
}

void TextEdit::clear_undo_history() {
	
	saved_version=0;
	current_op.type=TextOperation::TYPE_NONE;
	undo_stack_pos=NULL;
	undo_stack.clear();
	
}

void TextEdit::_begin_compex_operation() {
	_push_current_op();
	next_operation_is_complex=true;
}

void TextEdit::_end_compex_operation() {
	
	_push_current_op();
	ERR_FAIL_COND(undo_stack.size() == 0);
	
	if(undo_stack.back()->get().chain_forward) {
		undo_stack.back()->get().chain_forward=false;
		return;
	}
	
	undo_stack.back()->get().chain_backward=true;
}

void TextEdit::_push_current_op() {
	
	if (current_op.type==TextOperation::TYPE_NONE)
		return; // do nothing
	
	if(next_operation_is_complex) {
		current_op.chain_forward=true;
		next_operation_is_complex=false;
	}
	
	undo_stack.push_back(current_op);
	current_op.type=TextOperation::TYPE_NONE;
	current_op.text="";
	current_op.chain_forward=false;
	
}

void TextEdit::set_draw_tabs(bool p_draw) {
	
	draw_tabs=p_draw;
}

bool TextEdit::is_drawing_tabs() const{
	
	return draw_tabs;
}

uint32_t TextEdit::get_version() const {
	return current_op.version;
}
uint32_t TextEdit::get_saved_version() const {
	
	return saved_version;
}
void TextEdit::tag_saved_version() {
	
	saved_version=get_version();
}

int TextEdit::get_v_scroll() const {
	
	return v_scroll->get_val();
}
void TextEdit::set_v_scroll(int p_scroll) {
	
	v_scroll->set_val(p_scroll);
	cursor.line_ofs=p_scroll;
}

int TextEdit::get_h_scroll() const {
	
	return h_scroll->get_val();
}
void TextEdit::set_h_scroll(int p_scroll) {
	
	h_scroll->set_val(p_scroll);
}

void TextEdit::set_completion(bool p_enabled,const Vector<String>& p_prefixes) {
	
	completion_prefixes.clear();
	completion_enabled=p_enabled;
	for(int i=0;i<p_prefixes.size();i++)
		completion_prefixes.insert(p_prefixes[i]);
}

void TextEdit::_confirm_completion() {
	
	String remaining=completion_current.substr(completion_base.length(),completion_current.length()-completion_base.length());
	String l = text[cursor.line];
	bool same=true;
	//if what is going to be inserted is the same as what it is, don't change it
	for(int i=0;i<remaining.length();i++) {
		int c=i+cursor.column;
		if (c>=l.length() || l[c]!=remaining[i]) {
			same=false;
			break;
		}
	}
	
	if (same)
		cursor_set_column(cursor.column+remaining.length());
	else {
		insert_text_at_cursor(remaining);
		if (remaining.ends_with("(") && auto_brace_completion_enabled) {
			insert_text_at_cursor(")");
			cursor.column--;
		}
	}
	
	_cancel_completion();
}


void TextEdit::_cancel_code_hint() {
	completion_hint="";
	update();
}

void TextEdit::_cancel_completion() {
	
	if (!completion_active)
		return;
	
	completion_active=false;
	update();
	
}

static bool _is_completable(CharType c) {
	
	return !_is_symbol(c) || c=='"' || c=='\'';
}


void TextEdit::_update_completion_candidates() {
	
	String l = text[cursor.line];
	int cofs = CLAMP(cursor.column,0,l.length());
	
	
	String s;

	//look for keywords first

	bool pre_keyword=false;

	if (cofs>0 && l[cofs-1]==' ') {
		int kofs=cofs-1;
		String kw;
		while (kofs>=0 && l[kofs]==' ')
			kofs--;

		while(kofs>=0 && l[kofs]>32 && _is_completable(l[kofs])) {
			kw=String::chr(l[kofs])+kw;
			kofs--;
		}

		pre_keyword=keywords.has(kw);
		print_line("KW "+kw+"? "+itos(pre_keyword));

	} else {


		while(cofs>0 && l[cofs-1]>32 && _is_completable(l[cofs-1])) {
			s=String::chr(l[cofs-1])+s;
			if (l[cofs-1]=='\'' || l[cofs-1]=='"')
				break;

			cofs--;
		}
	}

	
	update();
	
	if (!pre_keyword && s=="" && (cofs==0 || !completion_prefixes.has(String::chr(l[cofs-1])))) {
		//none to complete, cancel
		_cancel_completion();
		return;
	}
	
	completion_options.clear();
	completion_index=0;
	completion_base=s;
	int ci_match=0;
	for(int i=0;i<completion_strings.size();i++) {
		if (completion_strings[i].begins_with(s)) {
			completion_options.push_back(completion_strings[i]);
			int m=0;
			int max=MIN(completion_current.length(),completion_strings[i].length());
			if (max<ci_match)
				continue;
			for(int j=0;j<max;j++) {
				
				if (j>=completion_strings[i].length())
					break;
				if (completion_current[j]!=completion_strings[i][j])
					break;
				m++;
			}
			if (m>ci_match) {
				ci_match=m;
				completion_index=completion_options.size()-1;
			}
			
		}
	}
	
	
	
	if (completion_options.size()==0) {
		//no options to complete, cancel
		_cancel_completion();
		return;
		
	}
	
	completion_current=completion_options[completion_index];
	
#if 0	// even there's only one option, user still get the chance to choose using it or not
	if (completion_options.size()==1) {
		//one option to complete, just complete it automagically
		_confirm_completion();
		//		insert_text_at_cursor(completion_options[0].substr(s.length(),completion_options[0].length()-s.length()));
		_cancel_completion();
		return;
		
	}
#endif
	if (completion_options.size()==1 && s==completion_options[0])
		_cancel_completion();
	
	completion_enabled=true;
}



void TextEdit::query_code_comple() {
	
	String l = text[cursor.line];
	int ofs = CLAMP(cursor.column,0,l.length());
	
	if (ofs>0 && (_is_completable(l[ofs-1]) || completion_prefixes.has(String::chr(l[ofs-1]))))
		emit_signal("request_completion");
	
}


void TextEdit::set_code_hint(const String& p_hint) {
	
	completion_hint=p_hint;
	completion_hint_offset=-0xFFFF;
	update();
}

void TextEdit::code_complete(const Vector<String> &p_strings) {
	
	
	completion_strings=p_strings;
	completion_active=true;
	completion_current="";
	completion_index=0;
	_update_completion_candidates();
	//
}


String TextEdit::get_tooltip(const Point2& p_pos) const {
	
	if (!tooltip_obj)
		return Control::get_tooltip(p_pos);
	int row,col;
	if (!_get_mouse_pos(p_pos, row,col)) {
		return Control::get_tooltip(p_pos);
	}
	
	String s = text[row];
	if (s.length()==0)
		return Control::get_tooltip(p_pos);
	int beg=CLAMP(col,0,s.length());
	int end=beg;
	
	
	if (s[beg]>32 || beg==s.length()) {
		
		bool symbol = beg < s.length() &&  _is_symbol(s[beg]); //not sure if right but most editors behave like this
		
		while(beg>0 && s[beg-1]>32 && (symbol==_is_symbol(s[beg-1]))) {
			beg--;
		}
		while(end<s.length() && s[end+1]>32 && (symbol==_is_symbol(s[end+1]))) {
			end++;
		}
		
		if (end<s.length())
			end+=1;
		
		String tt = tooltip_obj->call(tooltip_func,s.substr(beg,end-beg),tooltip_ud);
		
		return tt;
		
	}
	
	return Control::get_tooltip(p_pos);
	
}

void TextEdit::set_tooltip_request_func(Object *p_obj, const StringName& p_function,const Variant& p_udata) {
	
	tooltip_obj=p_obj;
	tooltip_func=p_function;
	tooltip_ud=p_udata;
}

void TextEdit::set_line(int line, String new_text)
{
	if (line < 0 || line > text.size())
		return;
	_remove_text(line, 0, line, text[line].length());
	_insert_text(line, 0, new_text);
}

void TextEdit::insert_at(const String &p_text, int at)
{
	cursor_set_column(0);
	cursor_set_line(at);
	_insert_text(at, 0, p_text+"\n");
}

void TextEdit::set_show_line_numbers(bool p_show) {
	
	line_numbers=p_show;
	update();
}


void TextEdit::_bind_methods() {
	
	
	ObjectTypeDB::bind_method(_MD("_input_event"),&TextEdit::_input_event);
	ObjectTypeDB::bind_method(_MD("_scroll_moved"),&TextEdit::_scroll_moved);
	ObjectTypeDB::bind_method(_MD("_cursor_changed_emit"),&TextEdit::_cursor_changed_emit);
	ObjectTypeDB::bind_method(_MD("_text_changed_emit"),&TextEdit::_text_changed_emit);
	ObjectTypeDB::bind_method(_MD("_push_current_op"),&TextEdit::_push_current_op);
	
	BIND_CONSTANT( SEARCH_MATCH_CASE );
	BIND_CONSTANT( SEARCH_WHOLE_WORDS );
	BIND_CONSTANT( SEARCH_BACKWARDS );
	
	/*
    ObjectTypeDB::bind_method(_MD("delete_char"),&TextEdit::delete_char);
    ObjectTypeDB::bind_method(_MD("delete_line"),&TextEdit::delete_line);
*/
	
	ObjectTypeDB::bind_method(_MD("set_text","text"),&TextEdit::set_text);
	ObjectTypeDB::bind_method(_MD("insert_text_at_cursor","text"),&TextEdit::insert_text_at_cursor);
	
	ObjectTypeDB::bind_method(_MD("get_line_count"),&TextEdit::get_line_count);
	ObjectTypeDB::bind_method(_MD("get_text"),&TextEdit::get_text);
	ObjectTypeDB::bind_method(_MD("get_line"),&TextEdit::get_line);
	
	ObjectTypeDB::bind_method(_MD("cursor_set_column","column"),&TextEdit::cursor_set_column);
	ObjectTypeDB::bind_method(_MD("cursor_set_line","line"),&TextEdit::cursor_set_line);
	
	ObjectTypeDB::bind_method(_MD("cursor_get_column"),&TextEdit::cursor_get_column);
	ObjectTypeDB::bind_method(_MD("cursor_get_line"),&TextEdit::cursor_get_line);
	
	
	ObjectTypeDB::bind_method(_MD("set_readonly","enable"),&TextEdit::set_readonly);
	ObjectTypeDB::bind_method(_MD("set_wrap","enable"),&TextEdit::set_wrap);
	ObjectTypeDB::bind_method(_MD("set_max_chars","amount"),&TextEdit::set_max_chars);
	
	ObjectTypeDB::bind_method(_MD("cut"),&TextEdit::cut);
	ObjectTypeDB::bind_method(_MD("copy"),&TextEdit::copy);
	ObjectTypeDB::bind_method(_MD("paste"),&TextEdit::paste);
	ObjectTypeDB::bind_method(_MD("select_all"),&TextEdit::select_all);
	ObjectTypeDB::bind_method(_MD("select","from_line","from_column","to_line","to_column"),&TextEdit::select);
	
	ObjectTypeDB::bind_method(_MD("is_selection_active"),&TextEdit::is_selection_active);
	ObjectTypeDB::bind_method(_MD("get_selection_from_line"),&TextEdit::get_selection_from_line);
	ObjectTypeDB::bind_method(_MD("get_selection_from_column"),&TextEdit::get_selection_from_column);
	ObjectTypeDB::bind_method(_MD("get_selection_to_line"),&TextEdit::get_selection_to_line);
	ObjectTypeDB::bind_method(_MD("get_selection_to_column"),&TextEdit::get_selection_to_column);
	ObjectTypeDB::bind_method(_MD("get_selection_text"),&TextEdit::get_selection_text);
	ObjectTypeDB::bind_method(_MD("get_word_under_cursor"),&TextEdit::get_word_under_cursor);
	ObjectTypeDB::bind_method(_MD("search","flags","from_line","from_column","to_line","to_column"),&TextEdit::_search_bind);
	
	ObjectTypeDB::bind_method(_MD("undo"),&TextEdit::undo);
	ObjectTypeDB::bind_method(_MD("redo"),&TextEdit::redo);
	ObjectTypeDB::bind_method(_MD("clear_undo_history"),&TextEdit::clear_undo_history);
	
	ObjectTypeDB::bind_method(_MD("set_syntax_coloring","enable"),&TextEdit::set_syntax_coloring);
	ObjectTypeDB::bind_method(_MD("is_syntax_coloring_enabled"),&TextEdit::is_syntax_coloring_enabled);
	
	
	ObjectTypeDB::bind_method(_MD("add_keyword_color","keyword","color"),&TextEdit::add_keyword_color);
	ObjectTypeDB::bind_method(_MD("add_color_region","begin_key","end_key","color","line_only"),&TextEdit::add_color_region,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("set_symbol_color","color"),&TextEdit::set_symbol_color);
	ObjectTypeDB::bind_method(_MD("set_custom_bg_color","color"),&TextEdit::set_custom_bg_color);
	ObjectTypeDB::bind_method(_MD("clear_colors"),&TextEdit::clear_colors);
	
	
	ADD_SIGNAL(MethodInfo("cursor_changed"));
	ADD_SIGNAL(MethodInfo("text_changed"));
	ADD_SIGNAL(MethodInfo("request_completion"));
	
}

TextEdit::TextEdit()  {
	
	readonly=false;
	setting_row=false;
	draw_tabs=false;
	max_chars=0;
	clear();
	wrap=false;
	set_focus_mode(FOCUS_ALL);
	_update_caches();
	cache.size=Size2(1,1);
	tab_size=4;
	text.set_tab_size(tab_size);
	text.clear();
	//	text.insert(1,"Mongolia..");
	//	text.insert(2,"PAIS GENEROSO!!");
	text.set_color_regions(&color_regions);
	
	h_scroll = memnew( HScrollBar );
	v_scroll = memnew( VScrollBar );
	
	add_child(h_scroll);
	add_child(v_scroll);
	
	updating_scrolls=false;
	selection.active=false;
	
	h_scroll->connect("value_changed", this,"_scroll_moved");
	v_scroll->connect("value_changed", this,"_scroll_moved");
	
	cursor_changed_dirty=false;
	text_changed_dirty=false;
	
	selection.selecting_mode=Selection::MODE_NONE;
	selection.selecting_line=0;
	selection.selecting_column=0;
	selection.selecting_test=false;
	selection.active=false;
	syntax_coloring=false;
	
	custom_bg_color=Color(0,0,0,0);
	idle_detect = memnew( Timer );
	add_child(idle_detect);
	idle_detect->set_one_shot(true);
	idle_detect->set_wait_time(GLOBAL_DEF("display/text_edit_idle_detect_sec",3));
	idle_detect->connect("timeout", this,"_push_current_op");
	
#if 0
	syntax_coloring=true;
	keywords["void"]=Color(0.3,0.0,0.1);
	keywords["int"]=Color(0.3,0.0,0.1);
	keywords["function"]=Color(0.3,0.0,0.1);
	keywords["class"]=Color(0.3,0.0,0.1);
	keywords["extends"]=Color(0.3,0.0,0.1);
	keywords["constructor"]=Color(0.3,0.0,0.1);
	symbol_color=Color(0.1,0.0,0.3,1.0);
	
	color_regions.push_back(ColorRegion("/*","*/",Color(0.4,0.6,0,4)));
	color_regions.push_back(ColorRegion("//","",Color(0.6,0.6,0.4)));
	color_regions.push_back(ColorRegion("\"","\"",Color(0.4,0.7,0.7)));
	color_regions.push_back(ColorRegion("'","'",Color(0.4,0.8,0.8)));
	color_regions.push_back(ColorRegion("#","",Color(0.2,1.0,0.2)));
	
#endif
	
	current_op.type=TextOperation::TYPE_NONE;
	undo_enabled=true;
	undo_stack_pos=NULL;
	setting_text=false;
	last_dblclk=0;
	current_op.version=0;
	version=0;
	saved_version=0;
	
	completion_enabled=false;
	completion_active=false;
	completion_line_ofs=0;
	tooltip_obj=NULL;
	line_numbers=false;
	next_operation_is_complex=false;
	auto_brace_completion_enabled=false;
	brace_matching_enabled=false;
	
}

TextEdit::~TextEdit()
{
}
