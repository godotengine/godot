/*************************************************************************/
/*  rich_text_label.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "rich_text_label.h"

#include "core/math/math_defs.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "scene/scene_string_names.h"
#include "servers/display_server.h"

#include "modules/modules_enabled.gen.h"
#ifdef MODULE_REGEX_ENABLED
#include "modules/regex/regex.h"
#endif

RichTextLabel::Item *RichTextLabel::_get_next_item(Item *p_item, bool p_free) const {
	if (p_free) {
		if (p_item->subitems.size()) {
			return p_item->subitems.front()->get();
		} else if (!p_item->parent) {
			return nullptr;
		} else if (p_item->E->next()) {
			return p_item->E->next()->get();
		} else {
			//go up until something with a next is found
			while (p_item->parent && !p_item->E->next()) {
				p_item = p_item->parent;
			}

			if (p_item->parent) {
				return p_item->E->next()->get();
			} else {
				return nullptr;
			}
		}

	} else {
		if (p_item->subitems.size() && p_item->type != ITEM_TABLE) {
			return p_item->subitems.front()->get();
		} else if (p_item->type == ITEM_FRAME) {
			return nullptr;
		} else if (p_item->E->next()) {
			return p_item->E->next()->get();
		} else {
			//go up until something with a next is found
			while (p_item->type != ITEM_FRAME && !p_item->E->next()) {
				p_item = p_item->parent;
			}

			if (p_item->type != ITEM_FRAME) {
				return p_item->E->next()->get();
			} else {
				return nullptr;
			}
		}
	}

	return nullptr;
}

RichTextLabel::Item *RichTextLabel::_get_prev_item(Item *p_item, bool p_free) const {
	if (p_free) {
		if (p_item->subitems.size()) {
			return p_item->subitems.back()->get();
		} else if (!p_item->parent) {
			return nullptr;
		} else if (p_item->E->prev()) {
			return p_item->E->prev()->get();
		} else {
			//go back until something with a prev is found
			while (p_item->parent && !p_item->E->prev()) {
				p_item = p_item->parent;
			}

			if (p_item->parent) {
				return p_item->E->prev()->get();
			} else {
				return nullptr;
			}
		}

	} else {
		if (p_item->subitems.size() && p_item->type != ITEM_TABLE) {
			return p_item->subitems.back()->get();
		} else if (p_item->type == ITEM_FRAME) {
			return nullptr;
		} else if (p_item->E->prev()) {
			return p_item->E->prev()->get();
		} else {
			//go back until something with a prev is found
			while (p_item->type != ITEM_FRAME && !p_item->E->prev()) {
				p_item = p_item->parent;
			}

			if (p_item->type != ITEM_FRAME) {
				return p_item->E->prev()->get();
			} else {
				return nullptr;
			}
		}
	}

	return nullptr;
}

Rect2 RichTextLabel::_get_text_rect() {
	Ref<StyleBox> style = get_theme_stylebox(SNAME("normal"));
	return Rect2(style->get_offset(), get_size() - style->get_minimum_size());
}

RichTextLabel::Item *RichTextLabel::_get_item_at_pos(RichTextLabel::Item *p_item_from, RichTextLabel::Item *p_item_to, int p_position) {
	int offset = 0;
	for (Item *it = p_item_from; it && it != p_item_to; it = _get_next_item(it)) {
		switch (it->type) {
			case ITEM_TEXT: {
				ItemText *t = (ItemText *)it;
				offset += t->text.length();
				if (offset > p_position) {
					return it;
				}
			} break;
			case ITEM_NEWLINE:
			case ITEM_IMAGE:
			case ITEM_TABLE: {
				offset += 1;
			} break;
			default:
				break;
		}
	}
	return p_item_from;
}

String RichTextLabel::_roman(int p_num, bool p_capitalize) const {
	if (p_num > 3999) {
		return "ERR";
	};
	String s;
	if (p_capitalize) {
		String M[] = { "", "M", "MM", "MMM" };
		String C[] = { "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" };
		String X[] = { "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" };
		String I[] = { "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" };
		s = M[p_num / 1000] + C[(p_num % 1000) / 100] + X[(p_num % 100) / 10] + I[p_num % 10];
	} else {
		String M[] = { "", "m", "mm", "mmm" };
		String C[] = { "", "c", "cc", "ccc", "cd", "d", "dc", "dcc", "dccc", "cm" };
		String X[] = { "", "x", "xx", "xxx", "xl", "l", "lx", "lxx", "lxxx", "xc" };
		String I[] = { "", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix" };
		s = M[p_num / 1000] + C[(p_num % 1000) / 100] + X[(p_num % 100) / 10] + I[p_num % 10];
	}
	return s;
}

String RichTextLabel::_letters(int p_num, bool p_capitalize) const {
	int64_t n = p_num;

	int chars = 0;
	do {
		n /= 24;
		chars++;
	} while (n);

	String s;
	s.resize(chars + 1);
	char32_t *c = s.ptrw();
	c[chars] = 0;
	n = p_num;
	do {
		int mod = ABS(n % 24);
		char a = (p_capitalize ? 'A' : 'a');
		c[--chars] = a + mod - 1;

		n /= 24;
	} while (n);

	return s;
}

void RichTextLabel::_resize_line(ItemFrame *p_frame, int p_line, const Ref<Font> &p_base_font, int p_base_font_size, int p_width) {
	ERR_FAIL_COND(p_frame == nullptr);
	ERR_FAIL_COND(p_line < 0 || p_line >= p_frame->lines.size());

	Line &l = p_frame->lines.write[p_line];

	l.offset.x = _find_margin(l.from, p_base_font, p_base_font_size);
	l.text_buf->set_width(p_width - l.offset.x);

	if (tab_size > 0) { // Align inline tabs.
		Vector<float> tabs;
		tabs.push_back(tab_size * p_base_font->get_char_size(' ', 0, p_base_font_size).width);
		l.text_buf->tab_align(tabs);
	}

	Item *it_to = (p_line + 1 < p_frame->lines.size()) ? p_frame->lines[p_line + 1].from : nullptr;
	for (Item *it = l.from; it && it != it_to; it = _get_next_item(it)) {
		switch (it->type) {
			case ITEM_TABLE: {
				ItemTable *table = static_cast<ItemTable *>(it);
				int hseparation = get_theme_constant(SNAME("table_hseparation"));
				int vseparation = get_theme_constant(SNAME("table_vseparation"));
				int col_count = table->columns.size();

				for (int i = 0; i < col_count; i++) {
					table->columns.write[i].width = 0;
				}

				int idx = 0;
				for (Item *E : table->subitems) {
					ERR_CONTINUE(E->type != ITEM_FRAME); // Children should all be frames.
					ItemFrame *frame = static_cast<ItemFrame *>(E);
					for (int i = 0; i < frame->lines.size(); i++) {
						_resize_line(frame, i, p_base_font, p_base_font_size, 1);
					}
					idx++;
				}

				// Compute minimum width for each cell.
				const int available_width = p_width - hseparation * (col_count - 1);

				// Compute available width and total ratio (for expanders).
				int total_ratio = 0;
				int remaining_width = available_width;
				table->total_width = hseparation;

				for (int i = 0; i < col_count; i++) {
					remaining_width -= table->columns[i].min_width;
					if (table->columns[i].max_width > table->columns[i].min_width) {
						table->columns.write[i].expand = true;
					}
					if (table->columns[i].expand) {
						total_ratio += table->columns[i].expand_ratio;
					}
				}

				// Assign actual widths.
				for (int i = 0; i < col_count; i++) {
					table->columns.write[i].width = table->columns[i].min_width;
					if (table->columns[i].expand && total_ratio > 0) {
						table->columns.write[i].width += table->columns[i].expand_ratio * remaining_width / total_ratio;
					}
					table->total_width += table->columns[i].width + hseparation;
				}

				// Resize to max_width if needed and distribute the remaining space.
				bool table_need_fit = true;
				while (table_need_fit) {
					table_need_fit = false;
					// Fit slim.
					for (int i = 0; i < col_count; i++) {
						if (!table->columns[i].expand) {
							continue;
						}
						int dif = table->columns[i].width - table->columns[i].max_width;
						if (dif > 0) {
							table_need_fit = true;
							table->columns.write[i].width = table->columns[i].max_width;
							table->total_width -= dif;
							total_ratio -= table->columns[i].expand_ratio;
						}
					}
					// Grow.
					remaining_width = available_width - table->total_width;
					if (remaining_width > 0 && total_ratio > 0) {
						for (int i = 0; i < col_count; i++) {
							if (table->columns[i].expand) {
								int dif = table->columns[i].max_width - table->columns[i].width;
								if (dif > 0) {
									int slice = table->columns[i].expand_ratio * remaining_width / total_ratio;
									int incr = MIN(dif, slice);
									table->columns.write[i].width += incr;
									table->total_width += incr;
								}
							}
						}
					}
				}

				// Update line width and get total height.
				idx = 0;
				table->total_height = 0;
				table->rows.clear();

				Vector2 offset;
				float row_height = 0.0;

				for (Item *E : table->subitems) {
					ERR_CONTINUE(E->type != ITEM_FRAME); // Children should all be frames.
					ItemFrame *frame = static_cast<ItemFrame *>(E);

					int column = idx % col_count;

					offset.x += frame->padding.position.x;
					float yofs = frame->padding.position.y;
					for (int i = 0; i < frame->lines.size(); i++) {
						frame->lines.write[i].text_buf->set_width(table->columns[column].width);
						table->columns.write[column].width = MAX(table->columns.write[column].width, ceil(frame->lines[i].text_buf->get_size().x));

						if (i > 0) {
							frame->lines.write[i].offset.y = frame->lines[i - 1].offset.y + frame->lines[i - 1].text_buf->get_size().y;
						} else {
							frame->lines.write[i].offset.y = 0;
						}
						frame->lines.write[i].offset += offset;

						float h = frame->lines[i].text_buf->get_size().y;
						if (frame->min_size_over.y > 0) {
							h = MAX(h, frame->min_size_over.y);
						}
						if (frame->max_size_over.y > 0) {
							h = MIN(h, frame->max_size_over.y);
						}
						yofs += h;
					}
					yofs += frame->padding.size.y;
					offset.x += table->columns[column].width + hseparation + frame->padding.size.x;

					row_height = MAX(yofs, row_height);
					if (column == col_count - 1) {
						offset.x = 0;
						row_height += vseparation;
						table->total_height += row_height;
						offset.y += row_height;
						table->rows.push_back(row_height);
						row_height = 0;
					}
					idx++;
				}
				l.text_buf->resize_object((uint64_t)it, Size2(table->total_width, table->total_height), table->inline_align);
			} break;
			default:
				break;
		}
	}

	if (p_line > 0) {
		l.offset.y = p_frame->lines[p_line - 1].offset.y + p_frame->lines[p_line - 1].text_buf->get_size().y + get_theme_constant(SNAME("line_separation"));
	} else {
		l.offset.y = 0;
	}
}

void RichTextLabel::_shape_line(ItemFrame *p_frame, int p_line, const Ref<Font> &p_base_font, int p_base_font_size, int p_width, int *r_char_offset) {
	ERR_FAIL_COND(p_frame == nullptr);
	ERR_FAIL_COND(p_line < 0 || p_line >= p_frame->lines.size());

	Line &l = p_frame->lines.write[p_line];

	// Clear cache.
	l.text_buf->clear();
	l.text_buf->set_flags(TextServer::BREAK_MANDATORY | TextServer::BREAK_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA | TextServer::JUSTIFICATION_WORD_BOUND | TextServer::JUSTIFICATION_TRIM_EDGE_SPACES);
	l.char_offset = *r_char_offset;
	l.char_count = 0;

	// Add indent.
	l.offset.x = _find_margin(l.from, p_base_font, p_base_font_size);
	l.text_buf->set_width(p_width - l.offset.x);
	l.text_buf->set_align((HAlign)_find_align(l.from));
	l.text_buf->set_direction(_find_direction(l.from));

	if (tab_size > 0) { // Align inline tabs.
		Vector<float> tabs;
		tabs.push_back(tab_size * p_base_font->get_char_size(' ', 0, p_base_font_size).width);
		l.text_buf->tab_align(tabs);
	}

	// Shape current paragraph.
	String text;
	Item *it_to = (p_line + 1 < p_frame->lines.size()) ? p_frame->lines[p_line + 1].from : nullptr;
	int remaining_characters = visible_characters - l.char_offset;
	for (Item *it = l.from; it && it != it_to; it = _get_next_item(it)) {
		if (visible_characters >= 0 && remaining_characters <= 0) {
			break;
		}
		switch (it->type) {
			case ITEM_DROPCAP: {
				// Add dropcap.
				const ItemDropcap *dc = (ItemDropcap *)it;
				if (dc != nullptr) {
					l.text_buf->set_dropcap(dc->text, dc->font, dc->font_size, dc->dropcap_margins);
					l.dc_color = dc->color;
					l.dc_ol_size = dc->ol_size;
					l.dc_ol_color = dc->ol_color;
				} else {
					l.text_buf->clear_dropcap();
				}
			} break;
			case ITEM_NEWLINE: {
				Ref<Font> font = _find_font(it);
				if (font.is_null()) {
					font = p_base_font;
				}
				int font_size = _find_font_size(it);
				if (font_size == -1) {
					font_size = p_base_font_size;
				}
				l.text_buf->add_string("\n", font, font_size, Dictionary(), "");
				text += "\n";
				l.char_count++;
				remaining_characters--;
			} break;
			case ITEM_TEXT: {
				ItemText *t = (ItemText *)it;
				Ref<Font> font = _find_font(it);
				if (font.is_null()) {
					font = p_base_font;
				}
				int font_size = _find_font_size(it);
				if (font_size == -1) {
					font_size = p_base_font_size;
				}
				Dictionary font_ftr = _find_font_features(it);
				String lang = _find_language(it);
				String tx = t->text;
				if (visible_characters >= 0 && remaining_characters >= 0) {
					tx = tx.substr(0, remaining_characters);
				}
				remaining_characters -= tx.length();

				l.text_buf->add_string(tx, font, font_size, font_ftr, lang);
				text += tx;
				l.char_count += tx.length();
			} break;
			case ITEM_IMAGE: {
				ItemImage *img = (ItemImage *)it;
				l.text_buf->add_object((uint64_t)it, img->image->get_size(), img->inline_align, 1);
				text += String::chr(0xfffc);
				l.char_count++;
				remaining_characters--;
			} break;
			case ITEM_TABLE: {
				ItemTable *table = static_cast<ItemTable *>(it);
				int hseparation = get_theme_constant(SNAME("table_hseparation"));
				int vseparation = get_theme_constant(SNAME("table_vseparation"));
				int col_count = table->columns.size();
				int t_char_count = 0;
				// Set minimums to zero.
				for (int i = 0; i < col_count; i++) {
					table->columns.write[i].min_width = 0;
					table->columns.write[i].max_width = 0;
					table->columns.write[i].width = 0;
				}
				// Compute minimum width for each cell.
				const int available_width = p_width - hseparation * (col_count - 1);

				int idx = 0;
				for (Item *E : table->subitems) {
					ERR_CONTINUE(E->type != ITEM_FRAME); // Children should all be frames.
					ItemFrame *frame = static_cast<ItemFrame *>(E);

					int column = idx % col_count;
					for (int i = 0; i < frame->lines.size(); i++) {
						int char_offset = l.char_offset + l.char_count;
						_shape_line(frame, i, p_base_font, p_base_font_size, 1, &char_offset);
						int cell_ch = (char_offset - (l.char_offset + l.char_count));
						l.char_count += cell_ch;
						t_char_count += cell_ch;
						remaining_characters -= cell_ch;

						table->columns.write[column].min_width = MAX(table->columns[column].min_width, ceil(frame->lines[i].text_buf->get_size().x));
						table->columns.write[column].max_width = MAX(table->columns[column].max_width, ceil(frame->lines[i].text_buf->get_non_wrapped_size().x));
					}
					idx++;
				}

				// Compute available width and total ratio (for expanders).
				int total_ratio = 0;
				int remaining_width = available_width;
				table->total_width = hseparation;

				for (int i = 0; i < col_count; i++) {
					remaining_width -= table->columns[i].min_width;
					if (table->columns[i].max_width > table->columns[i].min_width) {
						table->columns.write[i].expand = true;
					}
					if (table->columns[i].expand) {
						total_ratio += table->columns[i].expand_ratio;
					}
				}

				// Assign actual widths.
				for (int i = 0; i < col_count; i++) {
					table->columns.write[i].width = table->columns[i].min_width;
					if (table->columns[i].expand && total_ratio > 0) {
						table->columns.write[i].width += table->columns[i].expand_ratio * remaining_width / total_ratio;
					}
					table->total_width += table->columns[i].width + hseparation;
				}

				// Resize to max_width if needed and distribute the remaining space.
				bool table_need_fit = true;
				while (table_need_fit) {
					table_need_fit = false;
					// Fit slim.
					for (int i = 0; i < col_count; i++) {
						if (!table->columns[i].expand) {
							continue;
						}
						int dif = table->columns[i].width - table->columns[i].max_width;
						if (dif > 0) {
							table_need_fit = true;
							table->columns.write[i].width = table->columns[i].max_width;
							table->total_width -= dif;
							total_ratio -= table->columns[i].expand_ratio;
						}
					}
					// Grow.
					remaining_width = available_width - table->total_width;
					if (remaining_width > 0 && total_ratio > 0) {
						for (int i = 0; i < col_count; i++) {
							if (table->columns[i].expand) {
								int dif = table->columns[i].max_width - table->columns[i].width;
								if (dif > 0) {
									int slice = table->columns[i].expand_ratio * remaining_width / total_ratio;
									int incr = MIN(dif, slice);
									table->columns.write[i].width += incr;
									table->total_width += incr;
								}
							}
						}
					}
				}

				// Update line width and get total height.
				idx = 0;
				table->total_height = 0;
				table->rows.clear();

				Vector2 offset;
				float row_height = 0.0;

				for (const List<Item *>::Element *E = table->subitems.front(); E; E = E->next()) {
					ERR_CONTINUE(E->get()->type != ITEM_FRAME); // Children should all be frames.
					ItemFrame *frame = static_cast<ItemFrame *>(E->get());

					int column = idx % col_count;

					offset.x += frame->padding.position.x;
					float yofs = frame->padding.position.y;
					for (int i = 0; i < frame->lines.size(); i++) {
						frame->lines.write[i].text_buf->set_width(table->columns[column].width);
						table->columns.write[column].width = MAX(table->columns.write[column].width, ceil(frame->lines[i].text_buf->get_size().x));

						if (i > 0) {
							frame->lines.write[i].offset.y = frame->lines[i - 1].offset.y + frame->lines[i - 1].text_buf->get_size().y;
						} else {
							frame->lines.write[i].offset.y = 0;
						}
						frame->lines.write[i].offset += offset;

						float h = frame->lines[i].text_buf->get_size().y;
						if (frame->min_size_over.y > 0) {
							h = MAX(h, frame->min_size_over.y);
						}
						if (frame->max_size_over.y > 0) {
							h = MIN(h, frame->max_size_over.y);
						}
						yofs += h;
					}
					yofs += frame->padding.size.y;
					offset.x += table->columns[column].width + hseparation + frame->padding.size.x;

					row_height = MAX(yofs, row_height);
					// Add row height after last column of the row or last cell of the table.
					if (column == col_count - 1 || E->next() == nullptr) {
						offset.x = 0;
						row_height += vseparation;
						table->total_height += row_height;
						offset.y += row_height;
						table->rows.push_back(row_height);
						row_height = 0;
					}
					idx++;
				}

				l.text_buf->add_object((uint64_t)it, Size2(table->total_width, table->total_height), table->inline_align, t_char_count);
				text += String::chr(0xfffc).repeat(t_char_count);
			} break;
			default:
				break;
		}
	}

	//Apply BiDi override.
	l.text_buf->set_bidi_override(structured_text_parser(_find_stt(l.from), st_args, text));

	*r_char_offset = l.char_offset + l.char_count;

	if (p_line > 0) {
		l.offset.y = p_frame->lines[p_line - 1].offset.y + p_frame->lines[p_line - 1].text_buf->get_size().y + get_theme_constant(SNAME("line_separation"));
	} else {
		l.offset.y = 0;
	}
}

int RichTextLabel::_draw_line(ItemFrame *p_frame, int p_line, const Vector2 &p_ofs, int p_width, const Color &p_base_color, int p_outline_size, const Color &p_outline_color, const Color &p_font_shadow_color, int p_shadow_outline_size, const Point2 &p_shadow_ofs) {
	Vector2 off;

	ERR_FAIL_COND_V(p_frame == nullptr, 0);
	ERR_FAIL_COND_V(p_line < 0 || p_line >= p_frame->lines.size(), 0);

	Line &l = p_frame->lines.write[p_line];

	Item *it_from = l.from;
	Item *it_to = (p_line + 1 < p_frame->lines.size()) ? p_frame->lines[p_line + 1].from : nullptr;
	Variant meta;

	if (it_from == nullptr) {
		return 0;
	}

	RID ci = get_canvas_item();
	bool rtl = (l.text_buf->get_direction() == TextServer::DIRECTION_RTL);
	bool lrtl = is_layout_rtl();

	Vector<int> list_index;
	Vector<ItemList *> list_items;
	_find_list(l.from, list_index, list_items);

	String prefix;
	for (int i = 0; i < list_index.size(); i++) {
		if (rtl) {
			prefix = prefix + ".";
		} else {
			prefix = "." + prefix;
		}
		String segment;
		if (list_items[i]->list_type == LIST_DOTS) {
			static const char32_t _prefix[2] = { 0x25CF, 0 };
			prefix = _prefix;
			break;
		} else if (list_items[i]->list_type == LIST_NUMBERS) {
			segment = TS->format_number(itos(list_index[i]), _find_language(l.from));
		} else if (list_items[i]->list_type == LIST_LETTERS) {
			segment = _letters(list_index[i], list_items[i]->capitalize);
		} else if (list_items[i]->list_type == LIST_ROMAN) {
			segment = _roman(list_index[i], list_items[i]->capitalize);
		}
		if (rtl) {
			prefix = prefix + segment;
		} else {
			prefix = segment + prefix;
		}
	}
	if (prefix != "") {
		Ref<Font> font = _find_font(l.from);
		if (font.is_null()) {
			font = get_theme_font(SNAME("normal_font"));
		}
		int font_size = _find_font_size(l.from);
		if (font_size == -1) {
			font_size = get_theme_font_size(SNAME("normal_font_size"));
		}
		if (rtl) {
			float offx = 0.0f;
			if (!lrtl && p_frame == main) { // Skip Scrollbar.
				offx -= scroll_w;
			}
			font->draw_string(ci, p_ofs + Vector2(p_width - l.offset.x + offx, l.text_buf->get_line_ascent(0)), " " + prefix, HALIGN_LEFT, l.offset.x, font_size, _find_color(l.from, p_base_color));
		} else {
			float offx = 0.0f;
			if (lrtl && p_frame == main) { // Skip Scrollbar.
				offx += scroll_w;
			}
			font->draw_string(ci, p_ofs + Vector2(offx, l.text_buf->get_line_ascent(0)), prefix + " ", HALIGN_RIGHT, l.offset.x, font_size, _find_color(l.from, p_base_color));
		}
	}

	// Draw dropcap.
	int dc_lines = l.text_buf->get_dropcap_lines();
	float h_off = l.text_buf->get_dropcap_size().x;
	if (l.dc_ol_size > 0) {
		l.text_buf->draw_dropcap_outline(ci, p_ofs + ((rtl) ? Vector2() : Vector2(l.offset.x, 0)), l.dc_ol_size, l.dc_ol_color);
	}
	l.text_buf->draw_dropcap(ci, p_ofs + ((rtl) ? Vector2() : Vector2(l.offset.x, 0)), l.dc_color);

	int line_count = 0;
	Size2 ctrl_size = get_size();
	// Draw text.
	for (int line = 0; line < l.text_buf->get_line_count(); line++) {
		RID rid = l.text_buf->get_line_rid(line);
		if (p_ofs.y + off.y >= ctrl_size.height) {
			break;
		}
		if (p_ofs.y + off.y + TS->shaped_text_get_size(rid).y <= 0) {
			off.y += TS->shaped_text_get_size(rid).y;
			continue;
		}

		float width = l.text_buf->get_width();
		float length = TS->shaped_text_get_width(rid);

		// Draw line.
		line_count++;

		if (rtl) {
			off.x = p_width - l.offset.x - width;
			if (!lrtl && p_frame == main) { // Skip Scrollbar.
				off.x -= scroll_w;
			}
		} else {
			off.x = l.offset.x;
			if (lrtl && p_frame == main) { // Skip Scrollbar.
				off.x += scroll_w;
			}
		}

		// Draw text.
		switch (l.text_buf->get_align()) {
			case HALIGN_FILL:
			case HALIGN_LEFT: {
				if (rtl) {
					off.x += width - length;
				}
			} break;
			case HALIGN_CENTER: {
				off.x += Math::floor((width - length) / 2.0);
			} break;
			case HALIGN_RIGHT: {
				if (!rtl) {
					off.x += width - length;
				}
			} break;
		}

		if (line <= dc_lines) {
			if (rtl) {
				off.x -= h_off;
			} else {
				off.x += h_off;
			}
		}

		//draw_rect(Rect2(p_ofs + off, TS->shaped_text_get_size(rid)), Color(1,0,0), false, 2); //DEBUG_RECTS

		off.y += TS->shaped_text_get_ascent(rid) + l.text_buf->get_spacing_top();
		// Draw inlined objects.
		Array objects = TS->shaped_text_get_objects(rid);
		for (int i = 0; i < objects.size(); i++) {
			Item *it = (Item *)(uint64_t)objects[i];
			if (it != nullptr) {
				Rect2 rect = TS->shaped_text_get_object_rect(rid, objects[i]);
				//draw_rect(rect, Color(1,0,0), false, 2); //DEBUG_RECTS
				switch (it->type) {
					case ITEM_IMAGE: {
						ItemImage *img = static_cast<ItemImage *>(it);
						img->image->draw_rect(ci, Rect2(p_ofs + rect.position + off, rect.size), false, img->color);
					} break;
					case ITEM_TABLE: {
						ItemTable *table = static_cast<ItemTable *>(it);
						Color odd_row_bg = get_theme_color(SNAME("table_odd_row_bg"));
						Color even_row_bg = get_theme_color(SNAME("table_even_row_bg"));
						Color border = get_theme_color(SNAME("table_border"));
						int hseparation = get_theme_constant(SNAME("table_hseparation"));
						int col_count = table->columns.size();
						int row_count = table->rows.size();

						int idx = 0;
						for (Item *E : table->subitems) {
							ItemFrame *frame = static_cast<ItemFrame *>(E);

							int col = idx % col_count;
							int row = idx / col_count;

							if (frame->lines.size() != 0 && row < row_count) {
								Vector2 coff = frame->lines[0].offset;
								if (rtl) {
									coff.x = rect.size.width - table->columns[col].width - coff.x;
								}
								if (row % 2 == 0) {
									draw_rect(Rect2(p_ofs + rect.position + off + coff - frame->padding.position, Size2(table->columns[col].width + hseparation + frame->padding.position.x + frame->padding.size.x, table->rows[row])), (frame->odd_row_bg != Color(0, 0, 0, 0) ? frame->odd_row_bg : odd_row_bg), true);
								} else {
									draw_rect(Rect2(p_ofs + rect.position + off + coff - frame->padding.position, Size2(table->columns[col].width + hseparation + frame->padding.position.x + frame->padding.size.x, table->rows[row])), (frame->even_row_bg != Color(0, 0, 0, 0) ? frame->even_row_bg : even_row_bg), true);
								}
								draw_rect(Rect2(p_ofs + rect.position + off + coff - frame->padding.position, Size2(table->columns[col].width + hseparation + frame->padding.position.x + frame->padding.size.x, table->rows[row])), (frame->border != Color(0, 0, 0, 0) ? frame->border : border), false);
							}

							for (int j = 0; j < frame->lines.size(); j++) {
								_draw_line(frame, j, p_ofs + rect.position + off + Vector2(0, frame->lines[j].offset.y), rect.size.x, p_base_color, p_outline_size, p_outline_color, p_font_shadow_color, p_shadow_outline_size, p_shadow_ofs);
							}
							idx++;
						}
					} break;
					default:
						break;
				}
			}
		}

		const Glyph *glyphs = TS->shaped_text_get_glyphs(rid);
		int gl_size = TS->shaped_text_get_glyph_count(rid);

		Vector2 gloff = off;
		// Draw oulines and shadow.
		for (int i = 0; i < gl_size; i++) {
			Item *it = _get_item_at_pos(it_from, it_to, glyphs[i].start);
			int size = _find_outline_size(it, p_outline_size);
			Color font_color = _find_outline_color(it, p_outline_color);
			Color font_shadow_color = p_font_shadow_color;
			if ((size <= 0 || font_color.a == 0) && (font_shadow_color.a == 0)) {
				gloff.x += glyphs[i].advance;
				continue;
			}

			// Get FX.
			ItemFade *fade = nullptr;
			Item *fade_item = it;
			while (fade_item) {
				if (fade_item->type == ITEM_FADE) {
					fade = static_cast<ItemFade *>(fade_item);
					break;
				}
				fade_item = fade_item->parent;
			}

			Vector<ItemFX *> fx_stack;
			_fetch_item_fx_stack(it, fx_stack);
			bool custom_fx_ok = true;

			Point2 fx_offset = Vector2(glyphs[i].x_off, glyphs[i].y_off);
			RID frid = glyphs[i].font_rid;
			uint32_t gl = glyphs[i].index;
			uint16_t gl_fl = glyphs[i].flags;
			uint8_t gl_cn = glyphs[i].count;
			bool cprev = false;
			if (gl_cn == 0) { // Parts of the same cluster, always connected.
				cprev = true;
			}
			if (gl_fl & TextServer::GRAPHEME_IS_RTL) { // Check if previous grapheme cluster is connected.
				if (i > 0 && (glyphs[i - 1].flags & TextServer::GRAPHEME_IS_CONNECTED)) {
					cprev = true;
				}
			} else {
				if (glyphs[i].flags & TextServer::GRAPHEME_IS_CONNECTED) {
					cprev = true;
				}
			}

			//Apply fx.
			float faded_visibility = 1.0f;
			if (fade) {
				if (glyphs[i].start >= fade->starting_index) {
					faded_visibility -= (float)(glyphs[i].start - fade->starting_index) / (float)fade->length;
					faded_visibility = faded_visibility < 0.0f ? 0.0f : faded_visibility;
				}
				font_color.a = faded_visibility;
				font_shadow_color.a = faded_visibility;
			}

			bool visible = (font_color.a != 0) || (font_shadow_color.a != 0);

			for (int j = 0; j < fx_stack.size(); j++) {
				ItemFX *item_fx = fx_stack[j];
				if (item_fx->type == ITEM_CUSTOMFX && custom_fx_ok) {
					ItemCustomFX *item_custom = static_cast<ItemCustomFX *>(item_fx);

					Ref<CharFXTransform> charfx = item_custom->char_fx_transform;
					Ref<RichTextEffect> custom_effect = item_custom->custom_effect;

					if (!custom_effect.is_null()) {
						charfx->elapsed_time = item_custom->elapsed_time;
						charfx->range = Vector2i(l.char_offset + glyphs[i].start, l.char_offset + glyphs[i].end);
						charfx->visibility = visible;
						charfx->outline = true;
						charfx->font = frid;
						charfx->glyph_index = gl;
						charfx->glyph_flags = gl_fl;
						charfx->glyph_count = gl_cn;
						charfx->offset = fx_offset;
						charfx->color = font_color;

						bool effect_status = custom_effect->_process_effect_impl(charfx);
						custom_fx_ok = effect_status;

						fx_offset += charfx->offset;
						font_color = charfx->color;
						frid = charfx->font;
						gl = charfx->glyph_index;
						visible &= charfx->visibility;
					}
				} else if (item_fx->type == ITEM_SHAKE) {
					ItemShake *item_shake = static_cast<ItemShake *>(item_fx);

					if (!cprev) {
						uint64_t char_current_rand = item_shake->offset_random(glyphs[i].start);
						uint64_t char_previous_rand = item_shake->offset_previous_random(glyphs[i].start);
						uint64_t max_rand = 2147483647;
						double current_offset = Math::range_lerp(char_current_rand % max_rand, 0, max_rand, 0.0f, 2.f * (float)Math_PI);
						double previous_offset = Math::range_lerp(char_previous_rand % max_rand, 0, max_rand, 0.0f, 2.f * (float)Math_PI);
						double n_time = (double)(item_shake->elapsed_time / (0.5f / item_shake->rate));
						n_time = (n_time > 1.0) ? 1.0 : n_time;
						item_shake->prev_off = Point2(Math::lerp(Math::sin(previous_offset), Math::sin(current_offset), n_time), Math::lerp(Math::cos(previous_offset), Math::cos(current_offset), n_time)) * (float)item_shake->strength / 10.0f;
					}
					fx_offset += item_shake->prev_off;
				} else if (item_fx->type == ITEM_WAVE) {
					ItemWave *item_wave = static_cast<ItemWave *>(item_fx);

					if (!cprev) {
						double value = Math::sin(item_wave->frequency * item_wave->elapsed_time + ((p_ofs.x + gloff.x) / 50)) * (item_wave->amplitude / 10.0f);
						item_wave->prev_off = Point2(0, 1) * value;
					}
					fx_offset += item_wave->prev_off;
				} else if (item_fx->type == ITEM_TORNADO) {
					ItemTornado *item_tornado = static_cast<ItemTornado *>(item_fx);

					if (!cprev) {
						double torn_x = Math::sin(item_tornado->frequency * item_tornado->elapsed_time + ((p_ofs.x + gloff.x) / 50)) * (item_tornado->radius);
						double torn_y = Math::cos(item_tornado->frequency * item_tornado->elapsed_time + ((p_ofs.x + gloff.x) / 50)) * (item_tornado->radius);
						item_tornado->prev_off = Point2(torn_x, torn_y);
					}
					fx_offset += item_tornado->prev_off;
				} else if (item_fx->type == ITEM_RAINBOW) {
					ItemRainbow *item_rainbow = static_cast<ItemRainbow *>(item_fx);

					font_color = font_color.from_hsv(item_rainbow->frequency * (item_rainbow->elapsed_time + ((p_ofs.x + gloff.x) / 50)), item_rainbow->saturation, item_rainbow->value, font_color.a);
				}
			}

			// Draw glyph outlines.
			for (int j = 0; j < glyphs[i].repeat; j++) {
				if (visible) {
					if (frid != RID()) {
						if (font_shadow_color.a > 0) {
							TS->font_draw_glyph(frid, ci, glyphs[i].font_size, p_ofs + fx_offset + gloff + p_shadow_ofs, gl, font_shadow_color);
						}
						if (font_shadow_color.a > 0 && p_shadow_outline_size > 0) {
							TS->font_draw_glyph_outline(frid, ci, glyphs[i].font_size, p_shadow_outline_size, p_ofs + fx_offset + gloff + p_shadow_ofs, gl, font_shadow_color);
						}
						if (font_color.a != 0.0 && size > 0) {
							TS->font_draw_glyph_outline(frid, ci, glyphs[i].font_size, size, p_ofs + fx_offset + gloff, gl, font_color);
						}
					}
				}
				gloff.x += glyphs[i].advance;
			}
		}

		Vector2 fbg_line_off = off + p_ofs;
		// Draw background color box
		Vector2i chr_range = TS->shaped_text_get_range(rid);
		_draw_fbg_boxes(ci, rid, fbg_line_off, it_from, it_to, chr_range.x, chr_range.y, 0);

		// Draw main text.
		Color selection_fg = get_theme_color(SNAME("font_selected_color"));
		Color selection_bg = get_theme_color(SNAME("selection_color"));

		int sel_start = -1;
		int sel_end = -1;

		if (selection.active && (selection.from_frame->lines[selection.from_line].char_offset + selection.from_char) <= (l.char_offset + TS->shaped_text_get_range(rid).y) && (selection.to_frame->lines[selection.to_line].char_offset + selection.to_char) >= (l.char_offset + TS->shaped_text_get_range(rid).x)) {
			sel_start = MAX(TS->shaped_text_get_range(rid).x, (selection.from_frame->lines[selection.from_line].char_offset + selection.from_char) - l.char_offset);
			sel_end = MIN(TS->shaped_text_get_range(rid).y, (selection.to_frame->lines[selection.to_line].char_offset + selection.to_char) - l.char_offset);

			Vector<Vector2> sel = TS->shaped_text_get_selection(rid, sel_start, sel_end);
			for (int i = 0; i < sel.size(); i++) {
				Rect2 rect = Rect2(sel[i].x + p_ofs.x + off.x, p_ofs.y + off.y - TS->shaped_text_get_ascent(rid), sel[i].y - sel[i].x, TS->shaped_text_get_size(rid).y);
				RenderingServer::get_singleton()->canvas_item_add_rect(ci, rect, selection_bg);
			}
		}

		for (int i = 0; i < gl_size; i++) {
			bool selected = selection.active && (sel_start != -1) && (glyphs[i].start >= sel_start) && (glyphs[i].end <= sel_end);
			Item *it = _get_item_at_pos(it_from, it_to, glyphs[i].start);
			Color font_color = _find_color(it, p_base_color);
			if (_find_underline(it) || (_find_meta(it, &meta) && underline_meta)) {
				Color uc = font_color;
				uc.a *= 0.5;
				float y_off = TS->shaped_text_get_underline_position(rid);
				float underline_width = TS->shaped_text_get_underline_thickness(rid) * get_theme_default_base_scale();
				draw_line(p_ofs + Vector2(off.x, off.y + y_off), p_ofs + Vector2(off.x + glyphs[i].advance * glyphs[i].repeat, off.y + y_off), uc, underline_width);
			} else if (_find_strikethrough(it)) {
				Color uc = font_color;
				uc.a *= 0.5;
				float y_off = -TS->shaped_text_get_ascent(rid) + TS->shaped_text_get_size(rid).y / 2;
				float underline_width = TS->shaped_text_get_underline_thickness(rid) * get_theme_default_base_scale();
				draw_line(p_ofs + Vector2(off.x, off.y + y_off), p_ofs + Vector2(off.x + glyphs[i].advance * glyphs[i].repeat, off.y + y_off), uc, underline_width);
			}

			// Get FX.
			ItemFade *fade = nullptr;
			Item *fade_item = it;
			while (fade_item) {
				if (fade_item->type == ITEM_FADE) {
					fade = static_cast<ItemFade *>(fade_item);
					break;
				}
				fade_item = fade_item->parent;
			}

			Vector<ItemFX *> fx_stack;
			_fetch_item_fx_stack(it, fx_stack);
			bool custom_fx_ok = true;

			Point2 fx_offset = Vector2(glyphs[i].x_off, glyphs[i].y_off);
			RID frid = glyphs[i].font_rid;
			uint32_t gl = glyphs[i].index;
			uint16_t gl_fl = glyphs[i].flags;
			uint8_t gl_cn = glyphs[i].count;
			bool cprev = false;
			if (gl_cn == 0) { // Parts of the same grapheme cluster, always connected.
				cprev = true;
			}
			if (gl_fl & TextServer::GRAPHEME_IS_RTL) { // Check if previous grapheme cluster is connected.
				if (i > 0 && (glyphs[i - 1].flags & TextServer::GRAPHEME_IS_CONNECTED)) {
					cprev = true;
				}
			} else {
				if (glyphs[i].flags & TextServer::GRAPHEME_IS_CONNECTED) {
					cprev = true;
				}
			}

			//Apply fx.
			float faded_visibility = 1.0f;
			if (fade) {
				if (glyphs[i].start >= fade->starting_index) {
					faded_visibility -= (float)(glyphs[i].start - fade->starting_index) / (float)fade->length;
					faded_visibility = faded_visibility < 0.0f ? 0.0f : faded_visibility;
				}
				font_color.a = faded_visibility;
			}

			bool visible = (font_color.a != 0);

			for (int j = 0; j < fx_stack.size(); j++) {
				ItemFX *item_fx = fx_stack[j];
				if (item_fx->type == ITEM_CUSTOMFX && custom_fx_ok) {
					ItemCustomFX *item_custom = static_cast<ItemCustomFX *>(item_fx);

					Ref<CharFXTransform> charfx = item_custom->char_fx_transform;
					Ref<RichTextEffect> custom_effect = item_custom->custom_effect;

					if (!custom_effect.is_null()) {
						charfx->elapsed_time = item_custom->elapsed_time;
						charfx->range = Vector2i(l.char_offset + glyphs[i].start, l.char_offset + glyphs[i].end);
						charfx->visibility = visible;
						charfx->outline = false;
						charfx->font = frid;
						charfx->glyph_index = gl;
						charfx->glyph_flags = gl_fl;
						charfx->glyph_count = gl_cn;
						charfx->offset = fx_offset;
						charfx->color = font_color;

						bool effect_status = custom_effect->_process_effect_impl(charfx);
						custom_fx_ok = effect_status;

						fx_offset += charfx->offset;
						font_color = charfx->color;
						frid = charfx->font;
						gl = charfx->glyph_index;
						visible &= charfx->visibility;
					}
				} else if (item_fx->type == ITEM_SHAKE) {
					ItemShake *item_shake = static_cast<ItemShake *>(item_fx);

					if (!cprev) {
						uint64_t char_current_rand = item_shake->offset_random(glyphs[i].start);
						uint64_t char_previous_rand = item_shake->offset_previous_random(glyphs[i].start);
						uint64_t max_rand = 2147483647;
						double current_offset = Math::range_lerp(char_current_rand % max_rand, 0, max_rand, 0.0f, 2.f * (float)Math_PI);
						double previous_offset = Math::range_lerp(char_previous_rand % max_rand, 0, max_rand, 0.0f, 2.f * (float)Math_PI);
						double n_time = (double)(item_shake->elapsed_time / (0.5f / item_shake->rate));
						n_time = (n_time > 1.0) ? 1.0 : n_time;
						item_shake->prev_off = Point2(Math::lerp(Math::sin(previous_offset), Math::sin(current_offset), n_time), Math::lerp(Math::cos(previous_offset), Math::cos(current_offset), n_time)) * (float)item_shake->strength / 10.0f;
					}
					fx_offset += item_shake->prev_off;
				} else if (item_fx->type == ITEM_WAVE) {
					ItemWave *item_wave = static_cast<ItemWave *>(item_fx);

					if (!cprev) {
						double value = Math::sin(item_wave->frequency * item_wave->elapsed_time + ((p_ofs.x + off.x) / 50)) * (item_wave->amplitude / 10.0f);
						item_wave->prev_off = Point2(0, 1) * value;
					}
					fx_offset += item_wave->prev_off;
				} else if (item_fx->type == ITEM_TORNADO) {
					ItemTornado *item_tornado = static_cast<ItemTornado *>(item_fx);

					if (!cprev) {
						double torn_x = Math::sin(item_tornado->frequency * item_tornado->elapsed_time + ((p_ofs.x + off.x) / 50)) * (item_tornado->radius);
						double torn_y = Math::cos(item_tornado->frequency * item_tornado->elapsed_time + ((p_ofs.x + off.x) / 50)) * (item_tornado->radius);
						item_tornado->prev_off = Point2(torn_x, torn_y);
					}
					fx_offset += item_tornado->prev_off;
				} else if (item_fx->type == ITEM_RAINBOW) {
					ItemRainbow *item_rainbow = static_cast<ItemRainbow *>(item_fx);

					font_color = font_color.from_hsv(item_rainbow->frequency * (item_rainbow->elapsed_time + ((p_ofs.x + off.x) / 50)), item_rainbow->saturation, item_rainbow->value, font_color.a);
				}
			}

			if (selected) {
				font_color = override_selected_font_color ? selection_fg : font_color;
			}

			// Draw glyphs.
			for (int j = 0; j < glyphs[i].repeat; j++) {
				if (visible) {
					if (frid != RID()) {
						TS->font_draw_glyph(frid, ci, glyphs[i].font_size, p_ofs + fx_offset + off, gl, selected ? selection_fg : font_color);
					} else if ((glyphs[i].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
						TS->draw_hex_code_box(ci, glyphs[i].font_size, p_ofs + fx_offset + off, gl, selected ? selection_fg : font_color);
					}
				}
				off.x += glyphs[i].advance;
			}
		}
		// Draw foreground color box
		_draw_fbg_boxes(ci, rid, fbg_line_off, it_from, it_to, chr_range.x, chr_range.y, 1);

		off.y += TS->shaped_text_get_descent(rid) + l.text_buf->get_spacing_bottom();
	}

	return line_count;
}

void RichTextLabel::_find_click(ItemFrame *p_frame, const Point2i &p_click, ItemFrame **r_click_frame, int *r_click_line, Item **r_click_item, int *r_click_char, bool *r_outside) {
	if (r_click_item) {
		*r_click_item = nullptr;
	}
	if (r_click_char != nullptr) {
		*r_click_char = 0;
	}
	if (r_outside != nullptr) {
		*r_outside = true;
	}

	Size2 size = get_size();
	Rect2 text_rect = _get_text_rect();

	int vofs = vscroll->get_value();

	// Search for the first line.
	int from_line = 0;

	//TODO, change to binary search ?
	while (from_line < main->lines.size()) {
		if (main->lines[from_line].offset.y + main->lines[from_line].text_buf->get_size().y >= vofs) {
			break;
		}
		from_line++;
	}

	if (from_line >= main->lines.size()) {
		return;
	}

	Point2 ofs = text_rect.get_position() + Vector2(0, main->lines[from_line].offset.y - vofs);
	while (ofs.y < size.height && from_line < main->lines.size()) {
		_find_click_in_line(p_frame, from_line, ofs, text_rect.size.x, p_click, r_click_frame, r_click_line, r_click_item, r_click_char);
		ofs.y += main->lines[from_line].text_buf->get_size().y + get_theme_constant(SNAME("line_separation"));
		if (((r_click_item != nullptr) && ((*r_click_item) != nullptr)) || ((r_click_frame != nullptr) && ((*r_click_frame) != nullptr))) {
			if (r_outside != nullptr) {
				*r_outside = false;
			}
			return;
		}
		from_line++;
	}
}

float RichTextLabel::_find_click_in_line(ItemFrame *p_frame, int p_line, const Vector2 &p_ofs, int p_width, const Point2i &p_click, ItemFrame **r_click_frame, int *r_click_line, Item **r_click_item, int *r_click_char) {
	Vector2 off;

	int char_pos = -1;
	Line &l = p_frame->lines.write[p_line];
	bool rtl = (l.text_buf->get_direction() == TextServer::DIRECTION_RTL);
	bool lrtl = is_layout_rtl();
	bool table_hit = false;

	for (int line = 0; line < l.text_buf->get_line_count(); line++) {
		RID rid = l.text_buf->get_line_rid(line);

		float width = l.text_buf->get_width();
		float length = TS->shaped_text_get_width(rid);

		if (rtl) {
			off.x = p_width - l.offset.x - width;
			if (!lrtl && p_frame == main) { // Skip Scrollbar.
				off.x -= scroll_w;
			}
		} else {
			off.x = l.offset.x;
			if (lrtl && p_frame == main) { // Skip Scrollbar.
				off.x += scroll_w;
			}
		}

		switch (l.text_buf->get_align()) {
			case HALIGN_FILL:
			case HALIGN_LEFT: {
				if (rtl) {
					off.x += width - length;
				}
			} break;
			case HALIGN_CENTER: {
				off.x += Math::floor((width - length) / 2.0);
			} break;
			case HALIGN_RIGHT: {
				if (!rtl) {
					off.x += width - length;
				}
			} break;
		}

		off.y += TS->shaped_text_get_ascent(rid) + l.text_buf->get_spacing_top();

		Array objects = TS->shaped_text_get_objects(rid);
		for (int i = 0; i < objects.size(); i++) {
			Item *it = (Item *)(uint64_t)objects[i];
			if (it != nullptr) {
				Rect2 rect = TS->shaped_text_get_object_rect(rid, objects[i]);
				if (rect.has_point(p_click - p_ofs - off)) {
					switch (it->type) {
						case ITEM_TABLE: {
							int hseparation = get_theme_constant(SNAME("table_hseparation"));
							int vseparation = get_theme_constant(SNAME("table_vseparation"));

							ItemTable *table = static_cast<ItemTable *>(it);

							table_hit = true;

							int idx = 0;
							int col_count = table->columns.size();
							int row_count = table->rows.size();

							for (Item *E : table->subitems) {
								ItemFrame *frame = static_cast<ItemFrame *>(E);

								int col = idx % col_count;
								int row = idx / col_count;

								if (frame->lines.size() != 0 && row < row_count) {
									Vector2 coff = frame->lines[0].offset;
									if (rtl) {
										coff.x = rect.size.width - table->columns[col].width - coff.x;
									}
									Rect2 crect = Rect2(p_ofs + off + rect.position + coff - frame->padding.position, Size2(table->columns[col].width + hseparation, table->rows[row] + vseparation) + frame->padding.position + frame->padding.size);
									if (col == col_count - 1) {
										if (rtl) {
											crect.size.x = crect.position.x + crect.size.x;
											crect.position.x = 0;
										} else {
											crect.size.x = get_size().x;
										}
									}
									if (crect.has_point(p_click)) {
										for (int j = 0; j < frame->lines.size(); j++) {
											_find_click_in_line(frame, j, p_ofs + off + rect.position + Vector2(0, frame->lines[j].offset.y), rect.size.x, p_click, r_click_frame, r_click_line, r_click_item, r_click_char);
											if (((r_click_item != nullptr) && ((*r_click_item) != nullptr)) || ((r_click_frame != nullptr) && ((*r_click_frame) != nullptr))) {
												return off.y;
											}
										}
									}
								}
								idx++;
							}
						} break;
						default:
							break;
					}
				}
			}
		}
		Rect2 rect = Rect2(p_ofs + off - Vector2(0, TS->shaped_text_get_ascent(rid)), Size2(get_size().x, TS->shaped_text_get_size(rid).y));

		if (rect.has_point(p_click) && !table_hit) {
			char_pos = TS->shaped_text_hit_test_position(rid, p_click.x - rect.position.x);
		}
		off.y += TS->shaped_text_get_descent(rid) + l.text_buf->get_spacing_bottom();
	}

	if (char_pos >= 0) {
		// Find item.
		if (r_click_item != nullptr) {
			Item *it = p_frame->lines[p_line].from;
			Item *it_to = (p_line + 1 < p_frame->lines.size()) ? p_frame->lines[p_line + 1].from : nullptr;
			if (char_pos == p_frame->lines[p_line].char_count) {
				// Selection after the end of line, select last item.
				if (it_to != nullptr) {
					*r_click_item = _get_prev_item(it_to);
				} else {
					for (Item *i = it; i && i != it_to; i = _get_next_item(i)) {
						*r_click_item = i;
					}
				}
			} else {
				// Selection in the line.
				*r_click_item = _get_item_at_pos(it, it_to, char_pos);
			}
		}

		if (r_click_frame != nullptr) {
			*r_click_frame = p_frame;
		}

		if (r_click_line != nullptr) {
			*r_click_line = p_line;
		}

		if (r_click_char != nullptr) {
			*r_click_char = char_pos;
		}
	}

	return off.y;
}

void RichTextLabel::_scroll_changed(double) {
	if (updating_scroll) {
		return;
	}

	if (scroll_follow && vscroll->get_value() >= (vscroll->get_max() - vscroll->get_page())) {
		scroll_following = true;
	} else {
		scroll_following = false;
	}

	scroll_updated = true;

	update();
}

void RichTextLabel::_update_scroll() {
	int total_height = get_content_height();

	bool exceeds = total_height > get_size().height && scroll_active;

	if (exceeds != scroll_visible) {
		if (exceeds) {
			scroll_visible = true;
			scroll_w = vscroll->get_combined_minimum_size().width;
			vscroll->show();
			vscroll->set_anchor_and_offset(SIDE_LEFT, ANCHOR_END, -scroll_w);
		} else {
			scroll_visible = false;
			scroll_w = 0;
			vscroll->hide();
		}

		main->first_resized_line = 0; //invalidate ALL
		_validate_line_caches(main);
	}
}

void RichTextLabel::_update_fx(RichTextLabel::ItemFrame *p_frame, double p_delta_time) {
	Item *it = p_frame;
	while (it) {
		ItemFX *ifx = nullptr;

		if (it->type == ITEM_CUSTOMFX || it->type == ITEM_SHAKE || it->type == ITEM_WAVE || it->type == ITEM_TORNADO || it->type == ITEM_RAINBOW) {
			ifx = static_cast<ItemFX *>(it);
		}

		if (!ifx) {
			it = _get_next_item(it, true);
			continue;
		}

		ifx->elapsed_time += p_delta_time;

		ItemShake *shake = nullptr;

		if (it->type == ITEM_SHAKE) {
			shake = static_cast<ItemShake *>(it);
		}

		if (shake) {
			bool cycle = (shake->elapsed_time > (1.0f / shake->rate));
			if (cycle) {
				shake->elapsed_time -= (1.0f / shake->rate);
				shake->reroll_random();
			}
		}

		it = _get_next_item(it, true);
	}
}

void RichTextLabel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_MOUSE_EXIT: {
			if (meta_hovering) {
				meta_hovering = nullptr;
				emit_signal(SNAME("meta_hover_ended"), current_meta);
				current_meta = false;
				update();
			}
		} break;
		case NOTIFICATION_RESIZED: {
			main->first_resized_line = 0; //invalidate ALL
			update();

		} break;
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			if (text != "") {
				set_text(text);
			}

			main->first_invalid_line = 0; //invalidate ALL
			update();
		} break;
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			main->first_invalid_line = 0; //invalidate ALL
			update();
		} break;
		case NOTIFICATION_DRAW: {
			_validate_line_caches(main);
			_update_scroll();

			RID ci = get_canvas_item();

			Size2 size = get_size();
			Rect2 text_rect = _get_text_rect();

			draw_style_box(get_theme_stylebox(SNAME("normal")), Rect2(Point2(), size));

			if (has_focus()) {
				RenderingServer::get_singleton()->canvas_item_add_clip_ignore(ci, true);
				draw_style_box(get_theme_stylebox(SNAME("focus")), Rect2(Point2(), size));
				RenderingServer::get_singleton()->canvas_item_add_clip_ignore(ci, false);
			}

			float vofs = vscroll->get_value();

			// Search for the first line.
			int from_line = 0;

			//TODO, change to binary search ?
			while (from_line < main->lines.size()) {
				if (main->lines[from_line].offset.y + main->lines[from_line].text_buf->get_size().y >= vofs) {
					break;
				}
				from_line++;
			}

			if (from_line >= main->lines.size()) {
				break; //nothing to draw
			}
			Ref<Font> base_font = get_theme_font(SNAME("normal_font"));
			Color base_color = get_theme_color(SNAME("default_color"));
			Color outline_color = get_theme_color(SNAME("font_outline_color"));
			int outline_size = get_theme_constant(SNAME("outline_size"));
			Color font_shadow_color = get_theme_color(SNAME("font_shadow_color"));
			int shadow_outline_size = get_theme_constant(SNAME("shadow_outline_size"));
			Point2 shadow_ofs(get_theme_constant(SNAME("shadow_offset_x")), get_theme_constant(SNAME("shadow_offset_y")));

			visible_paragraph_count = 0;
			visible_line_count = 0;

			// New cache draw.
			Point2 ofs = text_rect.get_position() + Vector2(0, main->lines[from_line].offset.y - vofs);
			while (ofs.y < size.height && from_line < main->lines.size()) {
				visible_paragraph_count++;
				visible_line_count += _draw_line(main, from_line, ofs, text_rect.size.x, base_color, outline_size, outline_color, font_shadow_color, shadow_outline_size, shadow_ofs);
				ofs.y += main->lines[from_line].text_buf->get_size().y + get_theme_constant(SNAME("line_separation"));
				from_line++;
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (is_visible_in_tree()) {
				double dt = get_process_delta_time();
				_update_fx(main, dt);
				update();
			}
		} break;
		case NOTIFICATION_FOCUS_EXIT: {
			if (deselect_on_focus_loss_enabled) {
				selection.active = false;
				update();
			}
		} break;
	}
}

Control::CursorShape RichTextLabel::get_cursor_shape(const Point2 &p_pos) const {
	if (!underline_meta) {
		return get_default_cursor_shape();
	}

	if (selection.click_item) {
		return CURSOR_IBEAM;
	}

	if (main->first_invalid_line < main->lines.size()) {
		return get_default_cursor_shape(); //invalid
	}

	if (main->first_resized_line < main->lines.size()) {
		return get_default_cursor_shape(); //invalid
	}

	Item *item = nullptr;
	bool outside = true;
	((RichTextLabel *)(this))->_find_click(main, p_pos, nullptr, nullptr, &item, nullptr, &outside);

	if (item && !outside && ((RichTextLabel *)(this))->_find_meta(item, nullptr)) {
		return CURSOR_POINTING_HAND;
	}

	return get_default_cursor_shape();
}

void RichTextLabel::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		if (main->first_invalid_line < main->lines.size()) {
			return;
		}
		if (main->first_resized_line < main->lines.size()) {
			return;
		}

		if (b->get_button_index() == MOUSE_BUTTON_LEFT) {
			if (b->is_pressed() && !b->is_double_click()) {
				scroll_updated = false;
				ItemFrame *c_frame = nullptr;
				int c_line = 0;
				Item *c_item = nullptr;
				int c_index = 0;
				bool outside;

				_find_click(main, b->get_position(), &c_frame, &c_line, &c_item, &c_index, &outside);
				if (c_item != nullptr) {
					if (selection.enabled) {
						selection.click_frame = c_frame;
						selection.click_item = c_item;
						selection.click_line = c_line;
						selection.click_char = c_index;

						// Erase previous selection.
						if (selection.active) {
							selection.from_frame = nullptr;
							selection.from_line = 0;
							selection.from_item = nullptr;
							selection.from_char = 0;
							selection.to_frame = nullptr;
							selection.to_line = 0;
							selection.to_item = nullptr;
							selection.to_char = 0;
							selection.active = false;

							update();
						}
					}
				}
			} else if (b->is_pressed() && b->is_double_click() && selection.enabled) {
				//double_click: select word

				ItemFrame *c_frame = nullptr;
				int c_line = 0;
				Item *c_item = nullptr;
				int c_index = 0;
				bool outside;

				_find_click(main, b->get_position(), &c_frame, &c_line, &c_item, &c_index, &outside);

				if (c_frame) {
					const Line &l = c_frame->lines[c_line];
					PackedInt32Array words = TS->shaped_text_get_word_breaks(l.text_buf->get_rid());
					for (int i = 0; i < words.size(); i = i + 2) {
						if (c_index >= words[i] && c_index < words[i + 1]) {
							selection.from_frame = c_frame;
							selection.from_line = c_line;
							selection.from_item = c_item;
							selection.from_char = words[i];

							selection.to_frame = c_frame;
							selection.to_line = c_line;
							selection.to_item = c_item;
							selection.to_char = words[i + 1];

							selection.active = true;
							if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CLIPBOARD_PRIMARY)) {
								DisplayServer::get_singleton()->clipboard_set_primary(get_selected_text());
							}
							update();
							break;
						}
					}
				}
			} else if (!b->is_pressed()) {
				if (selection.enabled && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CLIPBOARD_PRIMARY)) {
					DisplayServer::get_singleton()->clipboard_set_primary(get_selected_text());
				}
				selection.click_item = nullptr;

				if (!b->is_double_click() && !scroll_updated) {
					Item *c_item = nullptr;

					bool outside = true;
					_find_click(main, b->get_position(), nullptr, nullptr, &c_item, nullptr, &outside);

					if (c_item) {
						Variant meta;
						if (!outside && _find_meta(c_item, &meta)) {
							//meta clicked
							emit_signal(SNAME("meta_clicked"), meta);
						}
					}
				}
			}
		}

		if (b->get_button_index() == MOUSE_BUTTON_WHEEL_UP) {
			if (scroll_active) {
				vscroll->set_value(vscroll->get_value() - vscroll->get_page() * b->get_factor() * 0.5 / 8);
			}
		}
		if (b->get_button_index() == MOUSE_BUTTON_WHEEL_DOWN) {
			if (scroll_active) {
				vscroll->set_value(vscroll->get_value() + vscroll->get_page() * b->get_factor() * 0.5 / 8);
			}
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {
		if (scroll_active) {
			vscroll->set_value(vscroll->get_value() + vscroll->get_page() * pan_gesture->get_delta().y * 0.5 / 8);
		}

		return;
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (k->is_pressed()) {
			bool handled = false;

			if (k->is_action("ui_page_up") && vscroll->is_visible_in_tree()) {
				vscroll->set_value(vscroll->get_value() - vscroll->get_page());
				handled = true;
			}
			if (k->is_action("ui_page_down") && vscroll->is_visible_in_tree()) {
				vscroll->set_value(vscroll->get_value() + vscroll->get_page());
				handled = true;
			}
			if (k->is_action("ui_up") && vscroll->is_visible_in_tree()) {
				vscroll->set_value(vscroll->get_value() - get_theme_font(SNAME("normal_font"))->get_height(get_theme_font_size(SNAME("normal_font_size"))));
				handled = true;
			}
			if (k->is_action("ui_down") && vscroll->is_visible_in_tree()) {
				vscroll->set_value(vscroll->get_value() + get_theme_font(SNAME("normal_font"))->get_height(get_theme_font_size(SNAME("normal_font_size"))));
				handled = true;
			}
			if (k->is_action("ui_home") && vscroll->is_visible_in_tree()) {
				vscroll->set_value(0);
				handled = true;
			}
			if (k->is_action("ui_end") && vscroll->is_visible_in_tree()) {
				vscroll->set_value(vscroll->get_max());
				handled = true;
			}
			if (k->is_action("ui_copy")) {
				selection_copy();
				handled = true;
			}

			if (handled) {
				accept_event();
			}
		}
	}

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {
		if (main->first_invalid_line < main->lines.size()) {
			return;
		}
		if (main->first_resized_line < main->lines.size()) {
			return;
		}

		ItemFrame *c_frame = nullptr;
		int c_line = 0;
		Item *c_item = nullptr;
		int c_index = 0;
		bool outside;

		_find_click(main, m->get_position(), &c_frame, &c_line, &c_item, &c_index, &outside);
		if (selection.click_item && c_item) {
			selection.from_frame = selection.click_frame;
			selection.from_line = selection.click_line;
			selection.from_item = selection.click_item;
			selection.from_char = selection.click_char;

			selection.to_frame = c_frame;
			selection.to_line = c_line;
			selection.to_item = c_item;
			selection.to_char = c_index;

			bool swap = false;
			if (selection.from_item->index > selection.to_item->index) {
				swap = true;
			} else if (selection.from_item->index == selection.to_item->index) {
				if (selection.from_char > selection.to_char) {
					swap = true;
				} else if (selection.from_char == selection.to_char) {
					selection.active = false;
					update();
					return;
				}
			}

			if (swap) {
				SWAP(selection.from_frame, selection.to_frame);
				SWAP(selection.from_line, selection.to_line);
				SWAP(selection.from_item, selection.to_item);
				SWAP(selection.from_char, selection.to_char);
			}

			selection.active = true;
			update();
		}

		Variant meta;
		ItemMeta *item_meta;
		if (c_item && !outside && _find_meta(c_item, &meta, &item_meta)) {
			if (meta_hovering != item_meta) {
				if (meta_hovering) {
					emit_signal(SNAME("meta_hover_ended"), current_meta);
				}
				meta_hovering = item_meta;
				current_meta = meta;
				emit_signal(SNAME("meta_hover_started"), meta);
			}
		} else if (meta_hovering) {
			meta_hovering = nullptr;
			emit_signal(SNAME("meta_hover_ended"), current_meta);
			current_meta = false;
		}
	}
}

void RichTextLabel::_find_frame(Item *p_item, ItemFrame **r_frame, int *r_line) {
	if (r_frame != nullptr) {
		*r_frame = nullptr;
	}
	if (r_line != nullptr) {
		*r_line = 0;
	}

	Item *item = p_item;

	while (item) {
		if (item->parent != nullptr && item->parent->type == ITEM_FRAME) {
			if (r_frame != nullptr) {
				*r_frame = (ItemFrame *)item->parent;
			}
			if (r_line != nullptr) {
				*r_line = item->line;
			}
			return;
		}

		item = item->parent;
	}
}

Ref<Font> RichTextLabel::_find_font(Item *p_item) {
	Item *fontitem = p_item;

	while (fontitem) {
		if (fontitem->type == ITEM_FONT) {
			ItemFont *fi = static_cast<ItemFont *>(fontitem);
			return fi->font;
		}

		fontitem = fontitem->parent;
	}

	return Ref<Font>();
}

int RichTextLabel::_find_font_size(Item *p_item) {
	Item *sizeitem = p_item;

	while (sizeitem) {
		if (sizeitem->type == ITEM_FONT_SIZE) {
			ItemFontSize *fi = static_cast<ItemFontSize *>(sizeitem);
			return fi->font_size;
		}

		sizeitem = sizeitem->parent;
	}

	return -1;
}

int RichTextLabel::_find_outline_size(Item *p_item, int p_default) {
	Item *sizeitem = p_item;

	while (sizeitem) {
		if (sizeitem->type == ITEM_OUTLINE_SIZE) {
			ItemOutlineSize *fi = static_cast<ItemOutlineSize *>(sizeitem);
			return fi->outline_size;
		}

		sizeitem = sizeitem->parent;
	}

	return p_default;
}

Dictionary RichTextLabel::_find_font_features(Item *p_item) {
	Item *ffitem = p_item;

	while (ffitem) {
		if (ffitem->type == ITEM_FONT_FEATURES) {
			ItemFontFeatures *fi = static_cast<ItemFontFeatures *>(ffitem);
			return fi->opentype_features;
		}

		ffitem = ffitem->parent;
	}

	return Dictionary();
}

RichTextLabel::ItemDropcap *RichTextLabel::_find_dc_item(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_DROPCAP) {
			return static_cast<ItemDropcap *>(item);
		}
		item = item->parent;
	}

	return nullptr;
}

RichTextLabel::ItemList *RichTextLabel::_find_list_item(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_LIST) {
			return static_cast<ItemList *>(item);
		}
		item = item->parent;
	}

	return nullptr;
}

int RichTextLabel::_find_list(Item *p_item, Vector<int> &r_index, Vector<ItemList *> &r_list) {
	Item *item = p_item;
	Item *prev_item = p_item;

	int level = 0;

	while (item) {
		if (item->type == ITEM_LIST) {
			ItemList *list = static_cast<ItemList *>(item);

			ItemFrame *frame = nullptr;
			int line = -1;
			_find_frame(list, &frame, &line);

			int index = 1;
			if (frame != nullptr) {
				for (int i = list->line + 1; i <= prev_item->line; i++) {
					if (_find_list_item(frame->lines[i].from) == list) {
						index++;
					}
				}
			}

			r_index.push_back(index);
			r_list.push_back(list);

			prev_item = item;
		}
		level++;
		item = item->parent;
	}

	return level;
}

int RichTextLabel::_find_margin(Item *p_item, const Ref<Font> &p_base_font, int p_base_font_size) {
	Item *item = p_item;

	float margin = 0.0;

	while (item) {
		if (item->type == ITEM_INDENT) {
			Ref<Font> font = _find_font(item);
			if (font.is_null()) {
				font = p_base_font;
			}
			int font_size = _find_font_size(item);
			if (font_size == -1) {
				font_size = p_base_font_size;
			}
			margin += tab_size * font->get_char_size(' ', 0, font_size).width;

		} else if (item->type == ITEM_LIST) {
			Ref<Font> font = _find_font(item);
			if (font.is_null()) {
				font = p_base_font;
			}
			int font_size = _find_font_size(item);
			if (font_size == -1) {
				font_size = p_base_font_size;
			}
			margin += tab_size * font->get_char_size(' ', 0, font_size).width;
		}

		item = item->parent;
	}

	return margin;
}

RichTextLabel::Align RichTextLabel::_find_align(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_PARAGRAPH) {
			ItemParagraph *p = static_cast<ItemParagraph *>(item);
			return p->align;
		}

		item = item->parent;
	}

	return default_align;
}

TextServer::Direction RichTextLabel::_find_direction(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_PARAGRAPH) {
			ItemParagraph *p = static_cast<ItemParagraph *>(item);
			if (p->direction != Control::TEXT_DIRECTION_INHERITED) {
				return (TextServer::Direction)p->direction;
			}
		}

		item = item->parent;
	}

	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		return is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR;
	} else {
		return (TextServer::Direction)text_direction;
	}
}

Control::StructuredTextParser RichTextLabel::_find_stt(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_PARAGRAPH) {
			ItemParagraph *p = static_cast<ItemParagraph *>(item);
			return p->st_parser;
		}

		item = item->parent;
	}

	return st_parser;
}

String RichTextLabel::_find_language(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_PARAGRAPH) {
			ItemParagraph *p = static_cast<ItemParagraph *>(item);
			return p->language;
		}

		item = item->parent;
	}

	return language;
}

Color RichTextLabel::_find_color(Item *p_item, const Color &p_default_color) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_COLOR) {
			ItemColor *color = static_cast<ItemColor *>(item);
			return color->color;
		}

		item = item->parent;
	}

	return p_default_color;
}

Color RichTextLabel::_find_outline_color(Item *p_item, const Color &p_default_color) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_OUTLINE_COLOR) {
			ItemOutlineColor *color = static_cast<ItemOutlineColor *>(item);
			return color->color;
		}

		item = item->parent;
	}

	return p_default_color;
}

bool RichTextLabel::_find_underline(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_UNDERLINE) {
			return true;
		}

		item = item->parent;
	}

	return false;
}

bool RichTextLabel::_find_strikethrough(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_STRIKETHROUGH) {
			return true;
		}

		item = item->parent;
	}

	return false;
}

void RichTextLabel::_fetch_item_fx_stack(Item *p_item, Vector<ItemFX *> &r_stack) {
	Item *item = p_item;
	while (item) {
		if (item->type == ITEM_CUSTOMFX || item->type == ITEM_SHAKE || item->type == ITEM_WAVE || item->type == ITEM_TORNADO || item->type == ITEM_RAINBOW) {
			r_stack.push_back(static_cast<ItemFX *>(item));
		}

		item = item->parent;
	}
}

bool RichTextLabel::_find_meta(Item *p_item, Variant *r_meta, ItemMeta **r_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_META) {
			ItemMeta *meta = static_cast<ItemMeta *>(item);
			if (r_meta) {
				*r_meta = meta->meta;
			}
			if (r_item) {
				*r_item = meta;
			}
			return true;
		}

		item = item->parent;
	}

	return false;
}

Color RichTextLabel::_find_bgcolor(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_BGCOLOR) {
			ItemBGColor *color = static_cast<ItemBGColor *>(item);
			return color->color;
		}

		item = item->parent;
	}

	return Color(0, 0, 0, 0);
}

Color RichTextLabel::_find_fgcolor(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ITEM_FGCOLOR) {
			ItemFGColor *color = static_cast<ItemFGColor *>(item);
			return color->color;
		}

		item = item->parent;
	}

	return Color(0, 0, 0, 0);
}

bool RichTextLabel::_find_layout_subitem(Item *from, Item *to) {
	if (from && from != to) {
		if (from->type != ITEM_FONT && from->type != ITEM_COLOR && from->type != ITEM_UNDERLINE && from->type != ITEM_STRIKETHROUGH) {
			return true;
		}

		for (Item *E : from->subitems) {
			bool layout = _find_layout_subitem(E, to);

			if (layout) {
				return true;
			}
		}
	}

	return false;
}

void RichTextLabel::_validate_line_caches(ItemFrame *p_frame) {
	if (p_frame->first_invalid_line == p_frame->lines.size()) {
		if (p_frame->first_resized_line == p_frame->lines.size()) {
			return;
		}

		// Resize lines without reshaping.
		Size2 size = get_size();
		if (fixed_width != -1) {
			size.width = fixed_width;
		}
		Rect2 text_rect = _get_text_rect();

		Ref<Font> base_font = get_theme_font(SNAME("normal_font"));
		int base_font_size = get_theme_font_size(SNAME("normal_font_size"));

		for (int i = p_frame->first_resized_line; i < p_frame->lines.size(); i++) {
			_resize_line(p_frame, i, base_font, base_font_size, text_rect.get_size().width - scroll_w);
		}

		int total_height = 0;
		if (p_frame->lines.size()) {
			total_height = p_frame->lines[p_frame->lines.size() - 1].offset.y + p_frame->lines[p_frame->lines.size() - 1].text_buf->get_size().y;
		}

		p_frame->first_resized_line = p_frame->lines.size();

		updating_scroll = true;
		vscroll->set_max(total_height);
		vscroll->set_page(text_rect.size.height);
		if (scroll_follow && scroll_following) {
			vscroll->set_value(total_height - size.height);
		}
		updating_scroll = false;

		if (fit_content_height) {
			minimum_size_changed();
		}
		return;
	}

	// Shape invalid lines.
	Size2 size = get_size();
	if (fixed_width != -1) {
		size.width = fixed_width;
	}
	Rect2 text_rect = _get_text_rect();

	Ref<Font> base_font = get_theme_font(SNAME("normal_font"));
	int base_font_size = get_theme_font_size(SNAME("normal_font_size"));

	int total_chars = (p_frame->first_invalid_line == 0) ? 0 : (p_frame->lines[p_frame->first_invalid_line].char_offset + p_frame->lines[p_frame->first_invalid_line].char_count);
	for (int i = p_frame->first_invalid_line; i < p_frame->lines.size(); i++) {
		_shape_line(p_frame, i, base_font, base_font_size, text_rect.get_size().width - scroll_w, &total_chars);
	}

	int total_height = 0;
	if (p_frame->lines.size()) {
		total_height = p_frame->lines[p_frame->lines.size() - 1].offset.y + p_frame->lines[p_frame->lines.size() - 1].text_buf->get_size().y;
	}

	p_frame->first_invalid_line = p_frame->lines.size();
	p_frame->first_resized_line = p_frame->lines.size();

	updating_scroll = true;
	vscroll->set_max(total_height);
	vscroll->set_page(text_rect.size.height);
	if (scroll_follow && scroll_following) {
		vscroll->set_value(total_height - size.height);
	}
	updating_scroll = false;

	if (fit_content_height) {
		minimum_size_changed();
	}
}

void RichTextLabel::_invalidate_current_line(ItemFrame *p_frame) {
	if (p_frame->lines.size() - 1 <= p_frame->first_invalid_line) {
		p_frame->first_invalid_line = p_frame->lines.size() - 1;
		update();
	}
}

void RichTextLabel::add_text(const String &p_text) {
	if (current->type == ITEM_TABLE) {
		return; //can't add anything here
	}

	int pos = 0;

	while (pos < p_text.length()) {
		int end = p_text.find("\n", pos);
		String line;
		bool eol = false;
		if (end == -1) {
			end = p_text.length();
		} else {
			eol = true;
		}

		if (pos == 0 && end == p_text.length()) {
			line = p_text;
		} else {
			line = p_text.substr(pos, end - pos);
		}

		if (line.length() > 0) {
			if (current->subitems.size() && current->subitems.back()->get()->type == ITEM_TEXT) {
				//append text condition!
				ItemText *ti = static_cast<ItemText *>(current->subitems.back()->get());
				ti->text += line;
				_invalidate_current_line(main);

			} else {
				//append item condition
				ItemText *item = memnew(ItemText);
				item->text = line;
				_add_item(item, false);
			}
		}

		if (eol) {
			ItemNewline *item = memnew(ItemNewline);
			item->line = current_frame->lines.size();
			_add_item(item, false);
			current_frame->lines.resize(current_frame->lines.size() + 1);
			if (item->type != ITEM_NEWLINE) {
				current_frame->lines.write[current_frame->lines.size() - 1].from = item;
			}
			_invalidate_current_line(current_frame);
		}

		pos = end + 1;
	}
}

void RichTextLabel::_add_item(Item *p_item, bool p_enter, bool p_ensure_newline) {
	p_item->parent = current;
	p_item->E = current->subitems.push_back(p_item);
	p_item->index = current_idx++;
	p_item->char_ofs = current_char_ofs;
	if (p_item->type == ITEM_TEXT) {
		ItemText *t = (ItemText *)p_item;
		current_char_ofs += t->text.length();
	} else if (p_item->type == ITEM_IMAGE) {
		current_char_ofs++;
	}

	if (p_enter) {
		current = p_item;
	}

	if (p_ensure_newline) {
		Item *from = current_frame->lines[current_frame->lines.size() - 1].from;
		// only create a new line for Item types that generate content/layout, ignore those that represent formatting/styling
		if (_find_layout_subitem(from, p_item)) {
			_invalidate_current_line(current_frame);
			current_frame->lines.resize(current_frame->lines.size() + 1);
		}
	}

	if (current_frame->lines[current_frame->lines.size() - 1].from == nullptr) {
		current_frame->lines.write[current_frame->lines.size() - 1].from = p_item;
	}
	p_item->line = current_frame->lines.size() - 1;

	_invalidate_current_line(current_frame);

	if (fixed_width != -1) {
		minimum_size_changed();
	}
}

void RichTextLabel::_remove_item(Item *p_item, const int p_line, const int p_subitem_line) {
	int size = p_item->subitems.size();
	if (size == 0) {
		p_item->parent->subitems.erase(p_item);
		// If a newline was erased, all lines AFTER the newline need to be decremented.
		if (p_item->type == ITEM_NEWLINE) {
			current_frame->lines.remove(p_line);
			for (int i = 0; i < current->subitems.size(); i++) {
				if (current->subitems[i]->line > p_subitem_line) {
					current->subitems[i]->line--;
				}
			}
		}
	} else {
		// First, remove all child items for the provided item.
		for (int i = 0; i < size; i++) {
			_remove_item(p_item->subitems.front()->get(), p_line, p_subitem_line);
		}
		// Then remove the provided item itself.
		p_item->parent->subitems.erase(p_item);
	}
}

void RichTextLabel::add_image(const Ref<Texture2D> &p_image, const int p_width, const int p_height, const Color &p_color, InlineAlign p_align) {
	if (current->type == ITEM_TABLE) {
		return;
	}

	ERR_FAIL_COND(p_image.is_null());
	ERR_FAIL_COND(p_image->get_width() == 0);
	ERR_FAIL_COND(p_image->get_height() == 0);
	ItemImage *item = memnew(ItemImage);

	item->image = p_image;
	item->color = p_color;
	item->inline_align = p_align;

	if (p_width > 0) {
		// custom width
		item->size.width = p_width;
		if (p_height > 0) {
			// custom height
			item->size.height = p_height;
		} else {
			// calculate height to keep aspect ratio
			item->size.height = p_image->get_height() * p_width / p_image->get_width();
		}
	} else {
		if (p_height > 0) {
			// custom height
			item->size.height = p_height;
			// calculate width to keep aspect ratio
			item->size.width = p_image->get_width() * p_height / p_image->get_height();
		} else {
			// keep original width and height
			item->size = p_image->get_size();
		}
	}

	_add_item(item, false);
}

void RichTextLabel::add_newline() {
	if (current->type == ITEM_TABLE) {
		return;
	}
	ItemNewline *item = memnew(ItemNewline);
	item->line = current_frame->lines.size();
	_add_item(item, false);
	current_frame->lines.resize(current_frame->lines.size() + 1);
	_invalidate_current_line(current_frame);
}

bool RichTextLabel::remove_line(const int p_line) {
	if (p_line >= current_frame->lines.size() || p_line < 0) {
		return false;
	}

	// Remove all subitems with the same line as that provided.
	Vector<int> subitem_indices_to_remove;
	for (int i = 0; i < current->subitems.size(); i++) {
		if (current->subitems[i]->line == p_line) {
			subitem_indices_to_remove.push_back(i);
		}
	}

	bool had_newline = false;
	// Reverse for loop to remove items from the end first.
	for (int i = subitem_indices_to_remove.size() - 1; i >= 0; i--) {
		int subitem_idx = subitem_indices_to_remove[i];
		had_newline = had_newline || current->subitems[subitem_idx]->type == ITEM_NEWLINE;
		_remove_item(current->subitems[subitem_idx], current->subitems[subitem_idx]->line, p_line);
	}

	if (!had_newline) {
		current_frame->lines.remove(p_line);
		if (current_frame->lines.size() == 0) {
			current_frame->lines.resize(1);
		}
	}

	if (p_line == 0 && current->subitems.size() > 0) {
		main->lines.write[0].from = main;
	}

	main->first_invalid_line = 0; // p_line ???
	update();

	return true;
}

void RichTextLabel::push_dropcap(const String &p_string, const Ref<Font> &p_font, int p_size, const Rect2 &p_dropcap_margins, const Color &p_color, int p_ol_size, const Color &p_ol_color) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ERR_FAIL_COND(p_string.is_empty());
	ERR_FAIL_COND(p_font.is_null());
	ERR_FAIL_COND(p_size <= 0);

	ItemDropcap *item = memnew(ItemDropcap);

	item->text = p_string;
	item->font = p_font;
	item->font_size = p_size;
	item->color = p_color;
	item->ol_size = p_ol_size;
	item->ol_color = p_ol_color;
	item->dropcap_margins = p_dropcap_margins;
	_add_item(item, false);
}

void RichTextLabel::push_font(const Ref<Font> &p_font) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ERR_FAIL_COND(p_font.is_null());
	ItemFont *item = memnew(ItemFont);

	item->font = p_font;
	_add_item(item, true);
}

void RichTextLabel::push_normal() {
	Ref<Font> normal_font = get_theme_font(SNAME("normal_font"));
	ERR_FAIL_COND(normal_font.is_null());

	push_font(normal_font);
}

void RichTextLabel::push_bold() {
	Ref<Font> bold_font = get_theme_font(SNAME("bold_font"));
	ERR_FAIL_COND(bold_font.is_null());

	push_font(bold_font);
}

void RichTextLabel::push_bold_italics() {
	Ref<Font> bold_italics_font = get_theme_font(SNAME("bold_italics_font"));
	ERR_FAIL_COND(bold_italics_font.is_null());

	push_font(bold_italics_font);
}

void RichTextLabel::push_italics() {
	Ref<Font> italics_font = get_theme_font(SNAME("italics_font"));
	ERR_FAIL_COND(italics_font.is_null());

	push_font(italics_font);
}

void RichTextLabel::push_mono() {
	Ref<Font> mono_font = get_theme_font(SNAME("mono_font"));
	ERR_FAIL_COND(mono_font.is_null());

	push_font(mono_font);
}

void RichTextLabel::push_font_size(int p_font_size) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemFontSize *item = memnew(ItemFontSize);

	item->font_size = p_font_size;
	_add_item(item, true);
}

void RichTextLabel::push_font_features(const Dictionary &p_features) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemFontFeatures *item = memnew(ItemFontFeatures);

	item->opentype_features = p_features;
	_add_item(item, true);
}

void RichTextLabel::push_outline_size(int p_font_size) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemOutlineSize *item = memnew(ItemOutlineSize);

	item->outline_size = p_font_size;
	_add_item(item, true);
}

void RichTextLabel::push_color(const Color &p_color) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemColor *item = memnew(ItemColor);

	item->color = p_color;
	_add_item(item, true);
}

void RichTextLabel::push_outline_color(const Color &p_color) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemOutlineColor *item = memnew(ItemOutlineColor);

	item->color = p_color;
	_add_item(item, true);
}

void RichTextLabel::push_underline() {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemUnderline *item = memnew(ItemUnderline);

	_add_item(item, true);
}

void RichTextLabel::push_strikethrough() {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemStrikethrough *item = memnew(ItemStrikethrough);

	_add_item(item, true);
}

void RichTextLabel::push_paragraph(Align p_align, Control::TextDirection p_direction, const String &p_language, Control::StructuredTextParser p_st_parser) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);

	ItemParagraph *item = memnew(ItemParagraph);
	item->align = p_align;
	item->direction = p_direction;
	item->language = p_language;
	item->st_parser = p_st_parser;
	_add_item(item, true, true);
}

void RichTextLabel::push_indent(int p_level) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ERR_FAIL_COND(p_level < 0);

	ItemIndent *item = memnew(ItemIndent);
	item->level = p_level;
	_add_item(item, true, true);
}

void RichTextLabel::push_list(int p_level, ListType p_list, bool p_capitalize) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ERR_FAIL_COND(p_level < 0);

	ItemList *item = memnew(ItemList);

	item->list_type = p_list;
	item->level = p_level;
	item->capitalize = p_capitalize;
	_add_item(item, true, true);
}

void RichTextLabel::push_meta(const Variant &p_meta) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemMeta *item = memnew(ItemMeta);

	item->meta = p_meta;
	_add_item(item, true);
}

void RichTextLabel::push_table(int p_columns, InlineAlign p_align) {
	ERR_FAIL_COND(p_columns < 1);
	ItemTable *item = memnew(ItemTable);

	item->columns.resize(p_columns);
	item->total_width = 0;
	item->inline_align = p_align;
	for (int i = 0; i < item->columns.size(); i++) {
		item->columns.write[i].expand = false;
		item->columns.write[i].expand_ratio = 1;
	}
	_add_item(item, true, false);
}

void RichTextLabel::push_fade(int p_start_index, int p_length) {
	ItemFade *item = memnew(ItemFade);
	item->starting_index = p_start_index;
	item->length = p_length;
	_add_item(item, true);
}

void RichTextLabel::push_shake(int p_strength = 10, float p_rate = 24.0f) {
	ItemShake *item = memnew(ItemShake);
	item->strength = p_strength;
	item->rate = p_rate;
	_add_item(item, true);
}

void RichTextLabel::push_wave(float p_frequency = 1.0f, float p_amplitude = 10.0f) {
	ItemWave *item = memnew(ItemWave);
	item->frequency = p_frequency;
	item->amplitude = p_amplitude;
	_add_item(item, true);
}

void RichTextLabel::push_tornado(float p_frequency = 1.0f, float p_radius = 10.0f) {
	ItemTornado *item = memnew(ItemTornado);
	item->frequency = p_frequency;
	item->radius = p_radius;
	_add_item(item, true);
}

void RichTextLabel::push_rainbow(float p_saturation, float p_value, float p_frequency) {
	ItemRainbow *item = memnew(ItemRainbow);
	item->frequency = p_frequency;
	item->saturation = p_saturation;
	item->value = p_value;
	_add_item(item, true);
}

void RichTextLabel::push_bgcolor(const Color &p_color) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemBGColor *item = memnew(ItemBGColor);

	item->color = p_color;
	_add_item(item, true);
}

void RichTextLabel::push_fgcolor(const Color &p_color) {
	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemFGColor *item = memnew(ItemFGColor);

	item->color = p_color;
	_add_item(item, true);
}

void RichTextLabel::push_customfx(Ref<RichTextEffect> p_custom_effect, Dictionary p_environment) {
	ItemCustomFX *item = memnew(ItemCustomFX);
	item->custom_effect = p_custom_effect;
	item->char_fx_transform->environment = p_environment;
	_add_item(item, true);
}

void RichTextLabel::set_table_column_expand(int p_column, bool p_expand, int p_ratio) {
	ERR_FAIL_COND(current->type != ITEM_TABLE);
	ItemTable *table = static_cast<ItemTable *>(current);
	ERR_FAIL_INDEX(p_column, table->columns.size());
	table->columns.write[p_column].expand = p_expand;
	table->columns.write[p_column].expand_ratio = p_ratio;
}

void RichTextLabel::set_cell_row_background_color(const Color &p_odd_row_bg, const Color &p_even_row_bg) {
	ERR_FAIL_COND(current->type != ITEM_FRAME);
	ItemFrame *cell = static_cast<ItemFrame *>(current);
	ERR_FAIL_COND(!cell->cell);
	cell->odd_row_bg = p_odd_row_bg;
	cell->even_row_bg = p_even_row_bg;
}

void RichTextLabel::set_cell_border_color(const Color &p_color) {
	ERR_FAIL_COND(current->type != ITEM_FRAME);
	ItemFrame *cell = static_cast<ItemFrame *>(current);
	ERR_FAIL_COND(!cell->cell);
	cell->border = p_color;
}

void RichTextLabel::set_cell_size_override(const Size2 &p_min_size, const Size2 &p_max_size) {
	ERR_FAIL_COND(current->type != ITEM_FRAME);
	ItemFrame *cell = static_cast<ItemFrame *>(current);
	ERR_FAIL_COND(!cell->cell);
	cell->min_size_over = p_min_size;
	cell->max_size_over = p_max_size;
}

void RichTextLabel::set_cell_padding(const Rect2 &p_padding) {
	ERR_FAIL_COND(current->type != ITEM_FRAME);
	ItemFrame *cell = static_cast<ItemFrame *>(current);
	ERR_FAIL_COND(!cell->cell);
	cell->padding = p_padding;
}

void RichTextLabel::push_cell() {
	ERR_FAIL_COND(current->type != ITEM_TABLE);

	ItemFrame *item = memnew(ItemFrame);
	item->parent_frame = current_frame;
	_add_item(item, true);
	current_frame = item;
	item->cell = true;
	item->lines.resize(1);
	item->lines.write[0].from = nullptr;
	item->first_invalid_line = 0; // parent frame last line ???
}

int RichTextLabel::get_current_table_column() const {
	ERR_FAIL_COND_V(current->type != ITEM_TABLE, -1);

	ItemTable *table = static_cast<ItemTable *>(current);

	return table->subitems.size() % table->columns.size();
}

void RichTextLabel::pop() {
	ERR_FAIL_COND(!current->parent);
	if (current->type == ITEM_FRAME) {
		current_frame = static_cast<ItemFrame *>(current)->parent_frame;
	}
	current = current->parent;
}

void RichTextLabel::clear() {
	main->_clear_children();
	current = main;
	current_frame = main;
	main->lines.clear();
	main->lines.resize(1);
	main->first_invalid_line = 0;
	update();

	selection.click_frame = nullptr;
	selection.click_item = nullptr;
	selection.active = false;

	current_idx = 1;
	current_char_ofs = 0;
	if (scroll_follow) {
		scroll_following = true;
	}

	if (fixed_width != -1) {
		minimum_size_changed();
	}
}

void RichTextLabel::set_tab_size(int p_spaces) {
	tab_size = p_spaces;
	main->first_resized_line = 0;
	update();
}

int RichTextLabel::get_tab_size() const {
	return tab_size;
}

void RichTextLabel::set_fit_content_height(bool p_enabled) {
	if (p_enabled != fit_content_height) {
		fit_content_height = p_enabled;
		minimum_size_changed();
	}
}

bool RichTextLabel::is_fit_content_height_enabled() const {
	return fit_content_height;
}

void RichTextLabel::set_meta_underline(bool p_underline) {
	underline_meta = p_underline;
	update();
}

bool RichTextLabel::is_meta_underlined() const {
	return underline_meta;
}

void RichTextLabel::set_override_selected_font_color(bool p_override_selected_font_color) {
	override_selected_font_color = p_override_selected_font_color;
}

bool RichTextLabel::is_overriding_selected_font_color() const {
	return override_selected_font_color;
}

void RichTextLabel::set_offset(int p_pixel) {
	vscroll->set_value(p_pixel);
}

void RichTextLabel::set_scroll_active(bool p_active) {
	if (scroll_active == p_active) {
		return;
	}

	scroll_active = p_active;
	vscroll->set_drag_node_enabled(p_active);
	update();
}

bool RichTextLabel::is_scroll_active() const {
	return scroll_active;
}

void RichTextLabel::set_scroll_follow(bool p_follow) {
	scroll_follow = p_follow;
	if (!vscroll->is_visible_in_tree() || vscroll->get_value() >= (vscroll->get_max() - vscroll->get_page())) {
		scroll_following = true;
	}
}

bool RichTextLabel::is_scroll_following() const {
	return scroll_follow;
}

void RichTextLabel::parse_bbcode(const String &p_bbcode) {
	clear();
	append_text(p_bbcode);
}

void RichTextLabel::append_text(const String &p_bbcode) {
	int pos = 0;

	List<String> tag_stack;
	Ref<Font> normal_font = get_theme_font(SNAME("normal_font"));
	Ref<Font> bold_font = get_theme_font(SNAME("bold_font"));
	Ref<Font> italics_font = get_theme_font(SNAME("italics_font"));
	Ref<Font> bold_italics_font = get_theme_font(SNAME("bold_italics_font"));
	Ref<Font> mono_font = get_theme_font(SNAME("mono_font"));

	Color base_color = get_theme_color(SNAME("default_color"));

	int indent_level = 0;

	bool in_bold = false;
	bool in_italics = false;

	set_process_internal(false);

	while (pos < p_bbcode.length()) {
		int brk_pos = p_bbcode.find("[", pos);

		if (brk_pos < 0) {
			brk_pos = p_bbcode.length();
		}

		if (brk_pos > pos) {
			add_text(p_bbcode.substr(pos, brk_pos - pos));
		}

		if (brk_pos == p_bbcode.length()) {
			break; //nothing else to add
		}

		int brk_end = p_bbcode.find("]", brk_pos + 1);

		if (brk_end == -1) {
			//no close, add the rest
			add_text(p_bbcode.substr(brk_pos, p_bbcode.length() - brk_pos));
			break;
		}

		String tag = p_bbcode.substr(brk_pos + 1, brk_end - brk_pos - 1);
		Vector<String> split_tag_block = tag.split(" ", false);

		// Find optional parameters.
		String bbcode_name;
		typedef Map<String, String> OptionMap;
		OptionMap bbcode_options;
		if (!split_tag_block.is_empty()) {
			bbcode_name = split_tag_block[0];
			for (int i = 1; i < split_tag_block.size(); i++) {
				const String &expr = split_tag_block[i];
				int value_pos = expr.find("=");
				if (value_pos > -1) {
					bbcode_options[expr.substr(0, value_pos)] = expr.substr(value_pos + 1);
				}
			}
		} else {
			bbcode_name = tag;
		}

		// Find main parameter.
		String bbcode_value;
		int main_value_pos = bbcode_name.find("=");
		if (main_value_pos > -1) {
			bbcode_value = bbcode_name.substr(main_value_pos + 1);
			bbcode_name = bbcode_name.substr(0, main_value_pos);
		}

		if (tag.begins_with("/") && tag_stack.size()) {
			bool tag_ok = tag_stack.size() && tag_stack.front()->get() == tag.substr(1, tag.length());

			if (tag_stack.front()->get() == "b") {
				in_bold = false;
			}
			if (tag_stack.front()->get() == "i") {
				in_italics = false;
			}
			if ((tag_stack.front()->get() == "indent") || (tag_stack.front()->get() == "ol") || (tag_stack.front()->get() == "ul")) {
				indent_level--;
			}

			if (!tag_ok) {
				add_text("[" + tag);
				pos = brk_end;
				continue;
			}

			tag_stack.pop_front();
			pos = brk_end + 1;
			if (tag != "/img" && tag != "/dropcap") {
				pop();
			}

		} else if (tag == "b") {
			//use bold font
			in_bold = true;
			if (in_italics) {
				push_font(bold_italics_font);
			} else {
				push_font(bold_font);
			}
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "i") {
			//use italics font
			in_italics = true;
			if (in_bold) {
				push_font(bold_italics_font);
			} else {
				push_font(italics_font);
			}
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "code") {
			//use monospace font
			push_font(mono_font);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("table=")) {
			Vector<String> subtag = tag.substr(6, tag.length()).split(",");
			int columns = subtag[0].to_int();
			if (columns < 1) {
				columns = 1;
			}

			int align = INLINE_ALIGN_TOP;
			if (subtag.size() > 2) {
				if (subtag[1] == "top" || subtag[1] == "t") {
					align = INLINE_ALIGN_TOP_TO;
				} else if (subtag[1] == "center" || subtag[1] == "c") {
					align = INLINE_ALIGN_CENTER_TO;
				} else if (subtag[1] == "bottom" || subtag[1] == "b") {
					align = INLINE_ALIGN_BOTTOM_TO;
				}
				if (subtag[2] == "top" || subtag[2] == "t") {
					align |= INLINE_ALIGN_TO_TOP;
				} else if (subtag[2] == "center" || subtag[2] == "c") {
					align |= INLINE_ALIGN_TO_CENTER;
				} else if (subtag[2] == "baseline" || subtag[2] == "l") {
					align |= INLINE_ALIGN_TO_BASELINE;
				} else if (subtag[2] == "bottom" || subtag[2] == "b") {
					align |= INLINE_ALIGN_TO_BOTTOM;
				}
			} else if (subtag.size() > 1) {
				if (subtag[1] == "top" || subtag[1] == "t") {
					align = INLINE_ALIGN_TOP;
				} else if (subtag[1] == "center" || subtag[1] == "c") {
					align = INLINE_ALIGN_CENTER;
				} else if (subtag[1] == "bottom" || subtag[1] == "b") {
					align = INLINE_ALIGN_BOTTOM;
				}
			}

			push_table(columns, (InlineAlign)align);
			pos = brk_end + 1;
			tag_stack.push_front("table");
		} else if (tag == "cell") {
			push_cell();
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("cell=")) {
			int ratio = tag.substr(5, tag.length()).to_int();
			if (ratio < 1) {
				ratio = 1;
			}

			set_table_column_expand(get_current_table_column(), true, ratio);
			push_cell();

			pos = brk_end + 1;
			tag_stack.push_front("cell");
		} else if (tag.begins_with("cell ")) {
			Vector<String> subtag = tag.substr(5, tag.length()).split(" ");

			for (int i = 0; i < subtag.size(); i++) {
				Vector<String> subtag_a = subtag[i].split("=");
				if (subtag_a.size() == 2) {
					if (subtag_a[0] == "expand") {
						int ratio = subtag_a[1].to_int();
						if (ratio < 1) {
							ratio = 1;
						}
						set_table_column_expand(get_current_table_column(), true, ratio);
					}
				}
			}
			push_cell();
			const Color fallback_color = Color(0, 0, 0, 0);
			for (int i = 0; i < subtag.size(); i++) {
				Vector<String> subtag_a = subtag[i].split("=");
				if (subtag_a.size() == 2) {
					if (subtag_a[0] == "border") {
						Color color = Color::from_string(subtag_a[1], fallback_color);
						set_cell_border_color(color);
					} else if (subtag_a[0] == "bg") {
						Vector<String> subtag_b = subtag_a[1].split(",");
						if (subtag_b.size() == 2) {
							Color color1 = Color::from_string(subtag_b[0], fallback_color);
							Color color2 = Color::from_string(subtag_b[1], fallback_color);
							set_cell_row_background_color(color1, color2);
						}
					}
				}
			}

			pos = brk_end + 1;
			tag_stack.push_front("cell");
		} else if (tag == "u") {
			//use underline
			push_underline();
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "s") {
			//use strikethrough
			push_strikethrough();
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "lrm") {
			add_text(String::chr(0x200E));
			pos = brk_end + 1;
		} else if (tag == "rlm") {
			add_text(String::chr(0x200F));
			pos = brk_end + 1;
		} else if (tag == "lre") {
			add_text(String::chr(0x202A));
			pos = brk_end + 1;
		} else if (tag == "rle") {
			add_text(String::chr(0x202B));
			pos = brk_end + 1;
		} else if (tag == "lro") {
			add_text(String::chr(0x202D));
			pos = brk_end + 1;
		} else if (tag == "rlo") {
			add_text(String::chr(0x202E));
			pos = brk_end + 1;
		} else if (tag == "pdf") {
			add_text(String::chr(0x202C));
			pos = brk_end + 1;
		} else if (tag == "alm") {
			add_text(String::chr(0x061c));
			pos = brk_end + 1;
		} else if (tag == "lri") {
			add_text(String::chr(0x2066));
			pos = brk_end + 1;
		} else if (tag == "rli") {
			add_text(String::chr(0x2027));
			pos = brk_end + 1;
		} else if (tag == "fsi") {
			add_text(String::chr(0x2068));
			pos = brk_end + 1;
		} else if (tag == "pdi") {
			add_text(String::chr(0x2069));
			pos = brk_end + 1;
		} else if (tag == "zwj") {
			add_text(String::chr(0x200D));
			pos = brk_end + 1;
		} else if (tag == "zwnj") {
			add_text(String::chr(0x200C));
			pos = brk_end + 1;
		} else if (tag == "wj") {
			add_text(String::chr(0x2060));
			pos = brk_end + 1;
		} else if (tag == "shy") {
			add_text(String::chr(0x00AD));
			pos = brk_end + 1;
		} else if (tag == "center") {
			push_paragraph(ALIGN_CENTER);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "fill") {
			push_paragraph(ALIGN_FILL);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "right") {
			push_paragraph(ALIGN_RIGHT);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "ul") {
			indent_level++;
			push_list(indent_level, LIST_DOTS, false);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if ((tag == "ol") || (tag == "ol type=1")) {
			indent_level++;
			push_list(indent_level, LIST_NUMBERS, false);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "ol type=a") {
			indent_level++;
			push_list(indent_level, LIST_LETTERS, false);
			pos = brk_end + 1;
			tag_stack.push_front("ol");
		} else if (tag == "ol type=A") {
			indent_level++;
			push_list(indent_level, LIST_LETTERS, true);
			pos = brk_end + 1;
			tag_stack.push_front("ol");
		} else if (tag == "ol type=i") {
			indent_level++;
			push_list(indent_level, LIST_ROMAN, false);
			pos = brk_end + 1;
			tag_stack.push_front("ol");
		} else if (tag == "ol type=I") {
			indent_level++;
			push_list(indent_level, LIST_ROMAN, true);
			pos = brk_end + 1;
			tag_stack.push_front("ol");
		} else if (tag == "indent") {
			indent_level++;
			push_indent(indent_level);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "p") {
			push_paragraph(ALIGN_LEFT);
			pos = brk_end + 1;
			tag_stack.push_front("p");
		} else if (tag.begins_with("p ")) {
			Vector<String> subtag = tag.substr(2, tag.length()).split(" ");
			Align align = ALIGN_LEFT;
			Control::TextDirection dir = Control::TEXT_DIRECTION_INHERITED;
			String lang;
			Control::StructuredTextParser st_parser = STRUCTURED_TEXT_DEFAULT;
			for (int i = 0; i < subtag.size(); i++) {
				Vector<String> subtag_a = subtag[i].split("=");
				if (subtag_a.size() == 2) {
					if (subtag_a[0] == "align") {
						if (subtag_a[1] == "l" || subtag_a[1] == "left") {
							align = ALIGN_LEFT;
						} else if (subtag_a[1] == "c" || subtag_a[1] == "center") {
							align = ALIGN_CENTER;
						} else if (subtag_a[1] == "r" || subtag_a[1] == "right") {
							align = ALIGN_RIGHT;
						} else if (subtag_a[1] == "f" || subtag_a[1] == "fill") {
							align = ALIGN_FILL;
						}
					} else if (subtag_a[0] == "dir" || subtag_a[0] == "direction") {
						if (subtag_a[1] == "a" || subtag_a[1] == "auto") {
							dir = Control::TEXT_DIRECTION_AUTO;
						} else if (subtag_a[1] == "l" || subtag_a[1] == "ltr") {
							dir = Control::TEXT_DIRECTION_LTR;
						} else if (subtag_a[1] == "r" || subtag_a[1] == "rtl") {
							dir = Control::TEXT_DIRECTION_RTL;
						}
					} else if (subtag_a[0] == "lang" || subtag_a[0] == "language") {
						lang = subtag_a[1];
					} else if (subtag_a[0] == "st" || subtag_a[0] == "bidi_override") {
						if (subtag_a[1] == "d" || subtag_a[1] == "default") {
							st_parser = STRUCTURED_TEXT_DEFAULT;
						} else if (subtag_a[1] == "u" || subtag_a[1] == "uri") {
							st_parser = STRUCTURED_TEXT_URI;
						} else if (subtag_a[1] == "f" || subtag_a[1] == "file") {
							st_parser = STRUCTURED_TEXT_FILE;
						} else if (subtag_a[1] == "e" || subtag_a[1] == "email") {
							st_parser = STRUCTURED_TEXT_EMAIL;
						} else if (subtag_a[1] == "l" || subtag_a[1] == "list") {
							st_parser = STRUCTURED_TEXT_LIST;
						} else if (subtag_a[1] == "n" || subtag_a[1] == "none") {
							st_parser = STRUCTURED_TEXT_NONE;
						} else if (subtag_a[1] == "c" || subtag_a[1] == "custom") {
							st_parser = STRUCTURED_TEXT_CUSTOM;
						}
					}
				}
			}
			push_paragraph(align, dir, lang, st_parser);
			pos = brk_end + 1;
			tag_stack.push_front("p");
		} else if (tag == "url") {
			int end = p_bbcode.find("[", brk_end);
			if (end == -1) {
				end = p_bbcode.length();
			}
			String url = p_bbcode.substr(brk_end + 1, end - brk_end - 1);
			push_meta(url);

			pos = brk_end + 1;
			tag_stack.push_front(tag);

		} else if (tag.begins_with("url=")) {
			String url = tag.substr(4, tag.length());
			push_meta(url);
			pos = brk_end + 1;
			tag_stack.push_front("url");
		} else if (tag.begins_with("dropcap")) {
			Vector<String> subtag = tag.substr(5, tag.length()).split(" ");
			Ref<Font> f = get_theme_font(SNAME("normal_font"));
			int fs = get_theme_font_size(SNAME("normal_font_size")) * 3;
			Color color = get_theme_color(SNAME("default_color"));
			Color outline_color = get_theme_color(SNAME("outline_color"));
			int outline_size = get_theme_constant(SNAME("outline_size"));
			Rect2 dropcap_margins = Rect2();

			for (int i = 0; i < subtag.size(); i++) {
				Vector<String> subtag_a = subtag[i].split("=");
				if (subtag_a.size() == 2) {
					if (subtag_a[0] == "font" || subtag_a[0] == "f") {
						String fnt = subtag_a[1];
						Ref<Font> font = ResourceLoader::load(fnt, "Font");
						if (font.is_valid()) {
							f = font;
						}
					} else if (subtag_a[0] == "font_size") {
						fs = subtag_a[1].to_int();
					} else if (subtag_a[0] == "margins") {
						Vector<String> subtag_b = subtag_a[1].split(",");
						if (subtag_b.size() == 4) {
							dropcap_margins.position.x = subtag_b[0].to_float();
							dropcap_margins.position.y = subtag_b[1].to_float();
							dropcap_margins.size.x = subtag_b[2].to_float();
							dropcap_margins.size.y = subtag_b[3].to_float();
						}
					} else if (subtag_a[0] == "outline_size") {
						outline_size = subtag_a[1].to_int();
					} else if (subtag_a[0] == "color") {
						color = Color::from_string(subtag_a[1], color);
					} else if (subtag_a[0] == "outline_color") {
						outline_color = Color::from_string(subtag_a[1], outline_color);
					}
				}
			}
			int end = p_bbcode.find("[", brk_end);
			if (end == -1) {
				end = p_bbcode.length();
			}

			String txt = p_bbcode.substr(brk_end + 1, end - brk_end - 1);

			push_dropcap(txt, f, fs, dropcap_margins, color, outline_size, outline_color);

			pos = end;
			tag_stack.push_front(bbcode_name);
		} else if (tag.begins_with("img")) {
			int align = INLINE_ALIGN_CENTER;
			if (tag.begins_with("img=")) {
				Vector<String> subtag = tag.substr(4, tag.length()).split(",");
				if (subtag.size() > 1) {
					if (subtag[0] == "top" || subtag[0] == "t") {
						align = INLINE_ALIGN_TOP_TO;
					} else if (subtag[0] == "center" || subtag[0] == "c") {
						align = INLINE_ALIGN_CENTER_TO;
					} else if (subtag[0] == "bottom" || subtag[0] == "b") {
						align = INLINE_ALIGN_BOTTOM_TO;
					}
					if (subtag[1] == "top" || subtag[1] == "t") {
						align |= INLINE_ALIGN_TO_TOP;
					} else if (subtag[1] == "center" || subtag[1] == "c") {
						align |= INLINE_ALIGN_TO_CENTER;
					} else if (subtag[1] == "baseline" || subtag[1] == "l") {
						align |= INLINE_ALIGN_TO_BASELINE;
					} else if (subtag[1] == "bottom" || subtag[1] == "b") {
						align |= INLINE_ALIGN_TO_BOTTOM;
					}
				} else if (subtag.size() > 0) {
					if (subtag[0] == "top" || subtag[0] == "t") {
						align = INLINE_ALIGN_TOP;
					} else if (subtag[0] == "center" || subtag[0] == "c") {
						align = INLINE_ALIGN_CENTER;
					} else if (subtag[0] == "bottom" || subtag[0] == "b") {
						align = INLINE_ALIGN_BOTTOM;
					}
				}
			}

			int end = p_bbcode.find("[", brk_end);
			if (end == -1) {
				end = p_bbcode.length();
			}

			String image = p_bbcode.substr(brk_end + 1, end - brk_end - 1);

			Ref<Texture2D> texture = ResourceLoader::load(image, "Texture2D");
			if (texture.is_valid()) {
				Color color = Color(1.0, 1.0, 1.0);
				OptionMap::Element *color_option = bbcode_options.find("color");
				if (color_option) {
					color = Color::from_string(color_option->value(), color);
				}

				int width = 0;
				int height = 0;
				if (!bbcode_value.is_empty()) {
					int sep = bbcode_value.find("x");
					if (sep == -1) {
						width = bbcode_value.to_int();
					} else {
						width = bbcode_value.substr(0, sep).to_int();
						height = bbcode_value.substr(sep + 1).to_int();
					}
				} else {
					OptionMap::Element *width_option = bbcode_options.find("width");
					if (width_option) {
						width = width_option->value().to_int();
					}

					OptionMap::Element *height_option = bbcode_options.find("height");
					if (height_option) {
						height = height_option->value().to_int();
					}
				}

				add_image(texture, width, height, color, (InlineAlign)align);
			}

			pos = end;
			tag_stack.push_front(bbcode_name);
		} else if (tag.begins_with("color=")) {
			String color_str = tag.substr(6, tag.length());
			Color color = Color::from_string(color_str, base_color);
			push_color(color);
			pos = brk_end + 1;
			tag_stack.push_front("color");

		} else if (tag.begins_with("outline_color=")) {
			String color_str = tag.substr(14, tag.length());
			Color color = Color::from_string(color_str, base_color);
			push_outline_color(color);
			pos = brk_end + 1;
			tag_stack.push_front("outline_color");

		} else if (tag.begins_with("font=")) {
			String fnt = tag.substr(5, tag.length());

			Ref<Font> font = ResourceLoader::load(fnt, "Font");
			if (font.is_valid()) {
				push_font(font);
			} else {
				push_font(normal_font);
			}

			pos = brk_end + 1;
			tag_stack.push_front("font");
		} else if (tag.begins_with("font_size=")) {
			int fnt_size = tag.substr(10, tag.length()).to_int();
			push_font_size(fnt_size);
			pos = brk_end + 1;
			tag_stack.push_front("font_size");
		} else if (tag.begins_with("opentype_features=")) {
			String fnt_ftr = tag.substr(18, tag.length());
			Vector<String> subtag = fnt_ftr.split(",");
			Dictionary ftrs;
			for (int i = 0; i < subtag.size(); i++) {
				Vector<String> subtag_a = subtag[i].split("=");
				if (subtag_a.size() == 2) {
					ftrs[TS->name_to_tag(subtag_a[0])] = subtag_a[1].to_int();
				} else if (subtag_a.size() == 1) {
					ftrs[TS->name_to_tag(subtag_a[0])] = 1;
				}
			}
			push_font_features(ftrs);
			pos = brk_end + 1;
			tag_stack.push_front("opentype_features");
		} else if (tag.begins_with("font ")) {
			Vector<String> subtag = tag.substr(2, tag.length()).split(" ");

			for (int i = 1; i < subtag.size(); i++) {
				Vector<String> subtag_a = subtag[i].split("=", true, 2);
				if (subtag_a.size() == 2) {
					if (subtag_a[0] == "name" || subtag_a[0] == "n") {
						String fnt = subtag_a[1];
						Ref<Font> font = ResourceLoader::load(fnt, "Font");
						if (font.is_valid()) {
							push_font(font);
						} else {
							push_font(normal_font);
						}
					} else if (subtag_a[0] == "size" || subtag_a[0] == "s") {
						int fnt_size = subtag_a[1].to_int();
						push_font_size(fnt_size);
					}
				}
			}

			pos = brk_end + 1;
			tag_stack.push_front("font");
		} else if (tag.begins_with("outline_size=")) {
			int fnt_size = tag.substr(13, tag.length()).to_int();
			push_outline_size(fnt_size);
			pos = brk_end + 1;
			tag_stack.push_front("outline_size");

		} else if (bbcode_name == "fade") {
			int start_index = 0;
			OptionMap::Element *start_option = bbcode_options.find("start");
			if (start_option) {
				start_index = start_option->value().to_int();
			}

			int length = 10;
			OptionMap::Element *length_option = bbcode_options.find("length");
			if (length_option) {
				length = length_option->value().to_int();
			}

			push_fade(start_index, length);
			pos = brk_end + 1;
			tag_stack.push_front("fade");
		} else if (bbcode_name == "shake") {
			int strength = 5;
			OptionMap::Element *strength_option = bbcode_options.find("level");
			if (strength_option) {
				strength = strength_option->value().to_int();
			}

			float rate = 20.0f;
			OptionMap::Element *rate_option = bbcode_options.find("rate");
			if (rate_option) {
				rate = rate_option->value().to_float();
			}

			push_shake(strength, rate);
			pos = brk_end + 1;
			tag_stack.push_front("shake");
			set_process_internal(true);
		} else if (bbcode_name == "wave") {
			float amplitude = 20.0f;
			OptionMap::Element *amplitude_option = bbcode_options.find("amp");
			if (amplitude_option) {
				amplitude = amplitude_option->value().to_float();
			}

			float period = 5.0f;
			OptionMap::Element *period_option = bbcode_options.find("freq");
			if (period_option) {
				period = period_option->value().to_float();
			}

			push_wave(period, amplitude);
			pos = brk_end + 1;
			tag_stack.push_front("wave");
			set_process_internal(true);
		} else if (bbcode_name == "tornado") {
			float radius = 10.0f;
			OptionMap::Element *radius_option = bbcode_options.find("radius");
			if (radius_option) {
				radius = radius_option->value().to_float();
			}

			float frequency = 1.0f;
			OptionMap::Element *frequency_option = bbcode_options.find("freq");
			if (frequency_option) {
				frequency = frequency_option->value().to_float();
			}

			push_tornado(frequency, radius);
			pos = brk_end + 1;
			tag_stack.push_front("tornado");
			set_process_internal(true);
		} else if (bbcode_name == "rainbow") {
			float saturation = 0.8f;
			OptionMap::Element *saturation_option = bbcode_options.find("sat");
			if (saturation_option) {
				saturation = saturation_option->value().to_float();
			}

			float value = 0.8f;
			OptionMap::Element *value_option = bbcode_options.find("val");
			if (value_option) {
				value = value_option->value().to_float();
			}

			float frequency = 1.0f;
			OptionMap::Element *frequency_option = bbcode_options.find("freq");
			if (frequency_option) {
				frequency = frequency_option->value().to_float();
			}

			push_rainbow(saturation, value, frequency);
			pos = brk_end + 1;
			tag_stack.push_front("rainbow");
			set_process_internal(true);

		} else if (tag.begins_with("bgcolor=")) {
			String color_str = tag.substr(8, tag.length());
			Color color = Color::from_string(color_str, base_color);

			push_bgcolor(color);
			pos = brk_end + 1;
			tag_stack.push_front("bgcolor");

		} else if (tag.begins_with("fgcolor=")) {
			String color_str = tag.substr(8, tag.length());
			Color color = Color::from_string(color_str, base_color);

			push_fgcolor(color);
			pos = brk_end + 1;
			tag_stack.push_front("fgcolor");

		} else {
			Vector<String> &expr = split_tag_block;
			if (expr.size() < 1) {
				add_text("[");
				pos = brk_pos + 1;
			} else {
				String identifier = expr[0];
				expr.remove(0);
				Dictionary properties = parse_expressions_for_values(expr);
				Ref<RichTextEffect> effect = _get_custom_effect_by_code(identifier);

				if (!effect.is_null()) {
					push_customfx(effect, properties);
					pos = brk_end + 1;
					tag_stack.push_front(identifier);
					set_process_internal(true);
				} else {
					add_text("["); //ignore
					pos = brk_pos + 1;
				}
			}
		}
	}

	Vector<ItemFX *> fx_items;
	for (Item *E : main->subitems) {
		Item *subitem = static_cast<Item *>(E);
		_fetch_item_fx_stack(subitem, fx_items);

		if (fx_items.size()) {
			set_process_internal(true);
			break;
		}
	}
}

void RichTextLabel::scroll_to_paragraph(int p_paragraph) {
	ERR_FAIL_INDEX(p_paragraph, main->lines.size());
	_validate_line_caches(main);
	vscroll->set_value(main->lines[p_paragraph].offset.y);
}

int RichTextLabel::get_paragraph_count() const {
	return current_frame->lines.size();
}

int RichTextLabel::get_visible_paragraph_count() const {
	if (!is_visible()) {
		return 0;
	}
	return visible_paragraph_count;
}

void RichTextLabel::scroll_to_line(int p_line) {
	_validate_line_caches(main);

	int line_count = 0;
	for (int i = 0; i < main->lines.size(); i++) {
		if ((line_count <= p_line) && (line_count + main->lines[i].text_buf->get_line_count() >= p_line)) {
			float line_offset = 0.f;
			for (int j = 0; j < p_line - line_count; j++) {
				line_offset += main->lines[i].text_buf->get_line_size(j).y;
			}
			vscroll->set_value(main->lines[i].offset.y + line_offset);
			return;
		}
		line_count += main->lines[i].text_buf->get_line_count();
	}
}

int RichTextLabel::get_line_count() const {
	int line_count = 0;
	for (int i = 0; i < main->lines.size(); i++) {
		line_count += main->lines[i].text_buf->get_line_count();
	}
	return line_count;
}

int RichTextLabel::get_visible_line_count() const {
	if (!is_visible()) {
		return 0;
	}
	return visible_line_count;
}

void RichTextLabel::set_selection_enabled(bool p_enabled) {
	selection.enabled = p_enabled;
	if (!p_enabled) {
		if (selection.active) {
			selection.active = false;
			update();
		}
		set_focus_mode(FOCUS_NONE);
	} else {
		set_focus_mode(FOCUS_ALL);
	}
}

void RichTextLabel::set_deselect_on_focus_loss_enabled(const bool p_enabled) {
	deselect_on_focus_loss_enabled = p_enabled;
	if (p_enabled && selection.active && !has_focus()) {
		selection.active = false;
		update();
	}
}

bool RichTextLabel::_search_table(ItemTable *p_table, List<Item *>::Element *p_from, const String &p_string, bool p_reverse_search) {
	List<Item *>::Element *E = p_from;
	while (E != nullptr) {
		ERR_CONTINUE(E->get()->type != ITEM_FRAME); // Children should all be frames.
		ItemFrame *frame = static_cast<ItemFrame *>(E->get());
		if (p_reverse_search) {
			for (int i = frame->lines.size() - 1; i >= 0; i--) {
				if (_search_line(frame, i, p_string, -1, p_reverse_search)) {
					return true;
				}
			}
		} else {
			for (int i = 0; i < frame->lines.size(); i++) {
				if (_search_line(frame, i, p_string, 0, p_reverse_search)) {
					return true;
				}
			}
		}
		E = p_reverse_search ? E->prev() : E->next();
	}
	return false;
}

bool RichTextLabel::_search_line(ItemFrame *p_frame, int p_line, const String &p_string, int p_char_idx, bool p_reverse_search) {
	ERR_FAIL_COND_V(p_frame == nullptr, false);
	ERR_FAIL_COND_V(p_line < 0 || p_line >= p_frame->lines.size(), false);

	Line &l = p_frame->lines.write[p_line];

	String text;
	Item *it_to = (p_line + 1 < p_frame->lines.size()) ? p_frame->lines[p_line + 1].from : nullptr;
	for (Item *it = l.from; it && it != it_to; it = _get_next_item(it)) {
		switch (it->type) {
			case ITEM_NEWLINE: {
				text += "\n";
			} break;
			case ITEM_TEXT: {
				ItemText *t = (ItemText *)it;
				text += t->text;
			} break;
			case ITEM_IMAGE: {
				text += " ";
			} break;
			case ITEM_TABLE: {
				ItemTable *table = static_cast<ItemTable *>(it);
				List<Item *>::Element *E = p_reverse_search ? table->subitems.back() : table->subitems.front();
				if (_search_table(table, E, p_string, p_reverse_search)) {
					return true;
				}
			} break;
			default:
				break;
		}
	}

	int sp = -1;
	if (p_reverse_search) {
		sp = text.rfindn(p_string, p_char_idx);
	} else {
		sp = text.findn(p_string, p_char_idx);
	}

	if (sp != -1) {
		selection.from_frame = p_frame;
		selection.from_line = p_line;
		selection.from_item = _get_item_at_pos(l.from, it_to, sp);
		selection.from_char = sp;
		selection.to_frame = p_frame;
		selection.to_line = p_line;
		selection.to_item = _get_item_at_pos(l.from, it_to, sp + p_string.length());
		selection.to_char = sp + p_string.length();
		selection.active = true;
		return true;
	}

	return false;
}

bool RichTextLabel::search(const String &p_string, bool p_from_selection, bool p_search_previous) {
	ERR_FAIL_COND_V(!selection.enabled, false);

	if (p_string.size() == 0) {
		selection.active = false;
		return false;
	}

	int char_idx = p_search_previous ? -1 : 0;
	int current_line = 0;
	int ending_line = main->lines.size() - 1;
	if (p_from_selection && selection.active) {
		// First check to see if other results exist in current line
		char_idx = p_search_previous ? selection.from_char - 1 : selection.to_char;
		if (!(p_search_previous && char_idx < 0) &&
				_search_line(selection.from_frame, selection.from_line, p_string, char_idx, p_search_previous)) {
			scroll_to_line(selection.from_frame->line + selection.from_line);
			update();
			return true;
		}
		char_idx = p_search_previous ? -1 : 0;

		// Next, check to see if the current search result is in a table
		if (selection.from_frame->parent != nullptr && selection.from_frame->parent->type == ITEM_TABLE) {
			// Find last search result in table
			ItemTable *parent_table = static_cast<ItemTable *>(selection.from_frame->parent);
			List<Item *>::Element *parent_element = p_search_previous ? parent_table->subitems.back() : parent_table->subitems.front();

			while (parent_element->get() != selection.from_frame) {
				parent_element = p_search_previous ? parent_element->prev() : parent_element->next();
				ERR_FAIL_COND_V(parent_element == nullptr, false);
			}

			// Search remainder of table
			if (!(p_search_previous && parent_element == parent_table->subitems.front()) &&
					parent_element != parent_table->subitems.back()) {
				parent_element = p_search_previous ? parent_element->prev() : parent_element->next(); // Don't want to search current item
				ERR_FAIL_COND_V(parent_element == nullptr, false);

				// Search for next element
				if (_search_table(parent_table, parent_element, p_string, p_search_previous)) {
					scroll_to_line(selection.from_frame->line + selection.from_line);
					update();
					return true;
				}
			}
		}

		ending_line = selection.from_frame->line + selection.from_line;
		current_line = p_search_previous ? ending_line - 1 : ending_line + 1;
	} else if (p_search_previous) {
		current_line = ending_line;
		ending_line = 0;
	}

	// Search remainder of the file
	while (current_line != ending_line) {
		// Wrap around
		if (current_line < 0) {
			current_line = main->lines.size() - 1;
		} else if (current_line >= main->lines.size()) {
			current_line = 0;
		}

		if (_search_line(main, current_line, p_string, char_idx, p_search_previous)) {
			scroll_to_line(current_line);
			update();
			return true;
		}
		p_search_previous ? current_line-- : current_line++;
	}

	if (p_from_selection && selection.active) {
		// Check contents of selection
		return _search_line(main, current_line, p_string, char_idx, p_search_previous);
	} else {
		return false;
	}
}

String RichTextLabel::_get_line_text(ItemFrame *p_frame, int p_line, Selection p_selection) const {
	String text;
	ERR_FAIL_COND_V(p_frame == nullptr, text);
	ERR_FAIL_COND_V(p_line < 0 || p_line >= p_frame->lines.size(), text);

	Line &l = p_frame->lines.write[p_line];

	Item *it_to = (p_line + 1 < p_frame->lines.size()) ? p_frame->lines[p_line + 1].from : nullptr;
	int end_idx = 0;
	if (it_to != nullptr) {
		end_idx = it_to->index;
	} else {
		for (Item *it = l.from; it && it != it_to; it = _get_next_item(it)) {
			end_idx = it->index + 1;
		}
	}
	for (Item *it = l.from; it && it != it_to; it = _get_next_item(it)) {
		if (it->type == ITEM_TABLE) {
			ItemTable *table = static_cast<ItemTable *>(it);
			for (Item *E : table->subitems) {
				ERR_CONTINUE(E->type != ITEM_FRAME); // Children should all be frames.
				ItemFrame *frame = static_cast<ItemFrame *>(E);
				for (int i = 0; i < frame->lines.size(); i++) {
					text += _get_line_text(frame, i, p_selection);
				}
			}
		}
		if ((p_selection.to_item != nullptr) && (p_selection.to_item->index < l.from->index)) {
			continue;
		}
		if ((p_selection.from_item != nullptr) && (p_selection.from_item->index >= end_idx)) {
			continue;
		}
		if (it->type == ITEM_DROPCAP) {
			const ItemDropcap *dc = static_cast<ItemDropcap *>(it);
			text += dc->text;
		} else if (it->type == ITEM_TEXT) {
			const ItemText *t = static_cast<ItemText *>(it);
			text += t->text;
		} else if (it->type == ITEM_NEWLINE) {
			text += "\n";
		} else if (it->type == ITEM_IMAGE) {
			text += " ";
		}
	}
	if ((l.from != nullptr) && (p_frame == p_selection.to_frame) && (p_selection.to_item != nullptr) && (p_selection.to_item->index >= l.from->index) && (p_selection.to_item->index < end_idx)) {
		text = text.substr(0, p_selection.to_char);
	}
	if ((l.from != nullptr) && (p_frame == p_selection.from_frame) && (p_selection.from_item != nullptr) && (p_selection.from_item->index >= l.from->index) && (p_selection.from_item->index < end_idx)) {
		text = text.substr(p_selection.from_char, -1);
	}
	return text;
}

String RichTextLabel::get_selected_text() const {
	if (!selection.active || !selection.enabled) {
		return "";
	}

	String text;
	for (int i = 0; i < main->lines.size(); i++) {
		text += _get_line_text(main, i, selection);
	}
	return text;
}

void RichTextLabel::selection_copy() {
	String text = get_selected_text();

	if (text != "") {
		DisplayServer::get_singleton()->clipboard_set(text);
	}
}

bool RichTextLabel::is_selection_enabled() const {
	return selection.enabled;
}

bool RichTextLabel::is_deselect_on_focus_loss_enabled() const {
	return deselect_on_focus_loss_enabled;
}

int RichTextLabel::get_selection_from() const {
	if (!selection.active || !selection.enabled) {
		return -1;
	}

	return selection.from_frame->lines[selection.from_line].char_offset + selection.from_char;
}

int RichTextLabel::get_selection_to() const {
	if (!selection.active || !selection.enabled) {
		return -1;
	}

	return selection.to_frame->lines[selection.to_line].char_offset + selection.to_char - 1;
}

void RichTextLabel::set_text(const String &p_bbcode) {
	text = p_bbcode;
	if (is_inside_tree() && use_bbcode) {
		parse_bbcode(p_bbcode);
	} else { // raw text
		clear();
		add_text(p_bbcode);
	}
}

String RichTextLabel::get_text() const {
	return text;
}

void RichTextLabel::set_use_bbcode(bool p_enable) {
	if (use_bbcode == p_enable) {
		return;
	}
	use_bbcode = p_enable;
	notify_property_list_changed();
	set_text(text);
}

bool RichTextLabel::is_using_bbcode() const {
	return use_bbcode;
}

String RichTextLabel::get_parsed_text() const {
	String text = "";
	Item *it = main;
	while (it) {
		if (it->type == ITEM_DROPCAP) {
			const ItemDropcap *dc = (ItemDropcap *)it;
			if (dc != nullptr) {
				text += dc->text;
			}
		} else if (it->type == ITEM_TEXT) {
			ItemText *t = static_cast<ItemText *>(it);
			text += t->text;
		} else if (it->type == ITEM_NEWLINE) {
			text += "\n";
		} else if (it->type == ITEM_IMAGE) {
			text += " ";
		} else if (it->type == ITEM_INDENT || it->type == ITEM_LIST) {
			text += "\t";
		}
		it = _get_next_item(it, true);
	}
	return text;
}

void RichTextLabel::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		main->first_invalid_line = 0; //invalidate ALL
		_validate_line_caches(main);
		update();
	}
}

void RichTextLabel::set_structured_text_bidi_override(Control::StructuredTextParser p_parser) {
	if (st_parser != p_parser) {
		st_parser = p_parser;
		main->first_invalid_line = 0; //invalidate ALL
		_validate_line_caches(main);
		update();
	}
}

Control::StructuredTextParser RichTextLabel::get_structured_text_bidi_override() const {
	return st_parser;
}

void RichTextLabel::set_structured_text_bidi_override_options(Array p_args) {
	st_args = p_args;
	main->first_invalid_line = 0; //invalidate ALL
	_validate_line_caches(main);
	update();
}

Array RichTextLabel::get_structured_text_bidi_override_options() const {
	return st_args;
}

Control::TextDirection RichTextLabel::get_text_direction() const {
	return text_direction;
}

void RichTextLabel::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		main->first_invalid_line = 0; //invalidate ALL
		_validate_line_caches(main);
		update();
	}
}

String RichTextLabel::get_language() const {
	return language;
}

void RichTextLabel::set_percent_visible(float p_percent) {
	if (percent_visible != p_percent) {
		if (p_percent < 0 || p_percent >= 1) {
			visible_characters = -1;
			percent_visible = 1;
		} else {
			visible_characters = get_total_character_count() * p_percent;
			percent_visible = p_percent;
		}
		main->first_invalid_line = 0; //invalidate ALL
		_validate_line_caches(main);
		update();
	}
}

float RichTextLabel::get_percent_visible() const {
	return percent_visible;
}

void RichTextLabel::set_effects(Array p_effects) {
	custom_effects = p_effects;
	if ((text != "") && use_bbcode) {
		parse_bbcode(text);
	}
}

Array RichTextLabel::get_effects() {
	return custom_effects;
}

void RichTextLabel::install_effect(const Variant effect) {
	Ref<RichTextEffect> rteffect;
	rteffect = effect;

	if (rteffect.is_valid()) {
		custom_effects.push_back(effect);
		if ((text != "") && use_bbcode) {
			parse_bbcode(text);
		}
	}
}

int RichTextLabel::get_content_height() const {
	int total_height = 0;
	if (main->lines.size()) {
		total_height = main->lines[main->lines.size() - 1].offset.y + main->lines[main->lines.size() - 1].text_buf->get_size().y;
	}
	return total_height;
}

#ifndef DISABLE_DEPRECATED
// People will be very angry, if their texts get erased, because of #39148. (3.x -> 4.0)
// Altough some people may not used bbcode_text, so we only overwrite, if bbcode_text is not empty
bool RichTextLabel::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "bbcode_text" && !((String)p_value).is_empty()) {
		set_text(p_value);
		return true;
	}
	return false;
}
#endif

void RichTextLabel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_parsed_text"), &RichTextLabel::get_parsed_text);
	ClassDB::bind_method(D_METHOD("add_text", "text"), &RichTextLabel::add_text);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &RichTextLabel::set_text);
	ClassDB::bind_method(D_METHOD("add_image", "image", "width", "height", "color", "inline_align"), &RichTextLabel::add_image, DEFVAL(0), DEFVAL(0), DEFVAL(Color(1.0, 1.0, 1.0)), DEFVAL(INLINE_ALIGN_CENTER));
	ClassDB::bind_method(D_METHOD("newline"), &RichTextLabel::add_newline);
	ClassDB::bind_method(D_METHOD("remove_line", "line"), &RichTextLabel::remove_line);
	ClassDB::bind_method(D_METHOD("push_font", "font"), &RichTextLabel::push_font);
	ClassDB::bind_method(D_METHOD("push_font_size", "font_size"), &RichTextLabel::push_font_size);
	ClassDB::bind_method(D_METHOD("push_font_features", "opentype_features"), &RichTextLabel::push_font_features);
	ClassDB::bind_method(D_METHOD("push_normal"), &RichTextLabel::push_normal);
	ClassDB::bind_method(D_METHOD("push_bold"), &RichTextLabel::push_bold);
	ClassDB::bind_method(D_METHOD("push_bold_italics"), &RichTextLabel::push_bold_italics);
	ClassDB::bind_method(D_METHOD("push_italics"), &RichTextLabel::push_italics);
	ClassDB::bind_method(D_METHOD("push_mono"), &RichTextLabel::push_mono);
	ClassDB::bind_method(D_METHOD("push_color", "color"), &RichTextLabel::push_color);
	ClassDB::bind_method(D_METHOD("push_outline_size", "outline_size"), &RichTextLabel::push_outline_size);
	ClassDB::bind_method(D_METHOD("push_outline_color", "color"), &RichTextLabel::push_outline_color);
	ClassDB::bind_method(D_METHOD("push_paragraph", "align", "base_direction", "language", "st_parser"), &RichTextLabel::push_paragraph, DEFVAL(TextServer::DIRECTION_AUTO), DEFVAL(""), DEFVAL(STRUCTURED_TEXT_DEFAULT));
	ClassDB::bind_method(D_METHOD("push_indent", "level"), &RichTextLabel::push_indent);
	ClassDB::bind_method(D_METHOD("push_list", "level", "type", "capitalize"), &RichTextLabel::push_list);
	ClassDB::bind_method(D_METHOD("push_meta", "data"), &RichTextLabel::push_meta);
	ClassDB::bind_method(D_METHOD("push_underline"), &RichTextLabel::push_underline);
	ClassDB::bind_method(D_METHOD("push_strikethrough"), &RichTextLabel::push_strikethrough);
	ClassDB::bind_method(D_METHOD("push_table", "columns", "inline_align"), &RichTextLabel::push_table, DEFVAL(INLINE_ALIGN_TOP));
	ClassDB::bind_method(D_METHOD("push_dropcap", "string", "font", "size", "dropcap_margins", "color", "outline_size", "outline_color"), &RichTextLabel::push_dropcap, DEFVAL(Rect2()), DEFVAL(Color(1, 1, 1)), DEFVAL(0), DEFVAL(Color(0, 0, 0, 0)));
	ClassDB::bind_method(D_METHOD("set_table_column_expand", "column", "expand", "ratio"), &RichTextLabel::set_table_column_expand);
	ClassDB::bind_method(D_METHOD("set_cell_row_background_color", "odd_row_bg", "even_row_bg"), &RichTextLabel::set_cell_row_background_color);
	ClassDB::bind_method(D_METHOD("set_cell_border_color", "color"), &RichTextLabel::set_cell_border_color);
	ClassDB::bind_method(D_METHOD("set_cell_size_override", "min_size", "max_size"), &RichTextLabel::set_cell_size_override);
	ClassDB::bind_method(D_METHOD("set_cell_padding", "padding"), &RichTextLabel::set_cell_padding);
	ClassDB::bind_method(D_METHOD("push_cell"), &RichTextLabel::push_cell);
	ClassDB::bind_method(D_METHOD("push_fgcolor", "fgcolor"), &RichTextLabel::push_fgcolor);
	ClassDB::bind_method(D_METHOD("push_bgcolor", "bgcolor"), &RichTextLabel::push_bgcolor);
	ClassDB::bind_method(D_METHOD("pop"), &RichTextLabel::pop);

	ClassDB::bind_method(D_METHOD("clear"), &RichTextLabel::clear);

	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override", "parser"), &RichTextLabel::set_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override"), &RichTextLabel::get_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override_options", "args"), &RichTextLabel::set_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override_options"), &RichTextLabel::get_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &RichTextLabel::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &RichTextLabel::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &RichTextLabel::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &RichTextLabel::get_language);

	ClassDB::bind_method(D_METHOD("set_meta_underline", "enable"), &RichTextLabel::set_meta_underline);
	ClassDB::bind_method(D_METHOD("is_meta_underlined"), &RichTextLabel::is_meta_underlined);

	ClassDB::bind_method(D_METHOD("set_override_selected_font_color", "override"), &RichTextLabel::set_override_selected_font_color);
	ClassDB::bind_method(D_METHOD("is_overriding_selected_font_color"), &RichTextLabel::is_overriding_selected_font_color);

	ClassDB::bind_method(D_METHOD("set_scroll_active", "active"), &RichTextLabel::set_scroll_active);
	ClassDB::bind_method(D_METHOD("is_scroll_active"), &RichTextLabel::is_scroll_active);

	ClassDB::bind_method(D_METHOD("set_scroll_follow", "follow"), &RichTextLabel::set_scroll_follow);
	ClassDB::bind_method(D_METHOD("is_scroll_following"), &RichTextLabel::is_scroll_following);

	ClassDB::bind_method(D_METHOD("get_v_scroll"), &RichTextLabel::get_v_scroll);

	ClassDB::bind_method(D_METHOD("scroll_to_line", "line"), &RichTextLabel::scroll_to_line);
	ClassDB::bind_method(D_METHOD("scroll_to_paragraph", "paragraph"), &RichTextLabel::scroll_to_paragraph);

	ClassDB::bind_method(D_METHOD("set_tab_size", "spaces"), &RichTextLabel::set_tab_size);
	ClassDB::bind_method(D_METHOD("get_tab_size"), &RichTextLabel::get_tab_size);

	ClassDB::bind_method(D_METHOD("set_fit_content_height", "enabled"), &RichTextLabel::set_fit_content_height);
	ClassDB::bind_method(D_METHOD("is_fit_content_height_enabled"), &RichTextLabel::is_fit_content_height_enabled);

	ClassDB::bind_method(D_METHOD("set_selection_enabled", "enabled"), &RichTextLabel::set_selection_enabled);
	ClassDB::bind_method(D_METHOD("is_selection_enabled"), &RichTextLabel::is_selection_enabled);

	ClassDB::bind_method(D_METHOD("set_deselect_on_focus_loss_enabled", "enable"), &RichTextLabel::set_deselect_on_focus_loss_enabled);
	ClassDB::bind_method(D_METHOD("is_deselect_on_focus_loss_enabled"), &RichTextLabel::is_deselect_on_focus_loss_enabled);

	ClassDB::bind_method(D_METHOD("get_selection_from"), &RichTextLabel::get_selection_from);
	ClassDB::bind_method(D_METHOD("get_selection_to"), &RichTextLabel::get_selection_to);

	ClassDB::bind_method(D_METHOD("get_selected_text"), &RichTextLabel::get_selected_text);

	ClassDB::bind_method(D_METHOD("parse_bbcode", "bbcode"), &RichTextLabel::parse_bbcode);
	ClassDB::bind_method(D_METHOD("append_text", "bbcode"), &RichTextLabel::append_text);

	ClassDB::bind_method(D_METHOD("get_text"), &RichTextLabel::get_text);

	ClassDB::bind_method(D_METHOD("set_visible_characters", "amount"), &RichTextLabel::set_visible_characters);
	ClassDB::bind_method(D_METHOD("get_visible_characters"), &RichTextLabel::get_visible_characters);

	ClassDB::bind_method(D_METHOD("set_percent_visible", "percent_visible"), &RichTextLabel::set_percent_visible);
	ClassDB::bind_method(D_METHOD("get_percent_visible"), &RichTextLabel::get_percent_visible);

	ClassDB::bind_method(D_METHOD("get_total_character_count"), &RichTextLabel::get_total_character_count);

	ClassDB::bind_method(D_METHOD("set_use_bbcode", "enable"), &RichTextLabel::set_use_bbcode);
	ClassDB::bind_method(D_METHOD("is_using_bbcode"), &RichTextLabel::is_using_bbcode);

	ClassDB::bind_method(D_METHOD("get_line_count"), &RichTextLabel::get_line_count);
	ClassDB::bind_method(D_METHOD("get_visible_line_count"), &RichTextLabel::get_visible_line_count);

	ClassDB::bind_method(D_METHOD("get_paragraph_count"), &RichTextLabel::get_paragraph_count);
	ClassDB::bind_method(D_METHOD("get_visible_paragraph_count"), &RichTextLabel::get_visible_paragraph_count);

	ClassDB::bind_method(D_METHOD("get_content_height"), &RichTextLabel::get_content_height);

	ClassDB::bind_method(D_METHOD("parse_expressions_for_values", "expressions"), &RichTextLabel::parse_expressions_for_values);

	ClassDB::bind_method(D_METHOD("set_effects", "effects"), &RichTextLabel::set_effects);
	ClassDB::bind_method(D_METHOD("get_effects"), &RichTextLabel::get_effects);
	ClassDB::bind_method(D_METHOD("install_effect", "effect"), &RichTextLabel::install_effect);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters", PROPERTY_HINT_RANGE, "-1,128000,1"), "set_visible_characters", "get_visible_characters");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "percent_visible", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_percent_visible", "get_percent_visible");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "meta_underlined"), "set_meta_underline", "is_meta_underlined");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_size", PROPERTY_HINT_RANGE, "0,24,1"), "set_tab_size", "get_tab_size");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bbcode_enabled"), "set_use_bbcode", "is_using_bbcode");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fit_content_height"), "set_fit_content_height", "is_fit_content_height_enabled");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_active"), "set_scroll_active", "is_scroll_active");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_following"), "set_scroll_follow", "is_scroll_following");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "selection_enabled"), "set_selection_enabled", "is_selection_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_selected_font_color"), "set_override_selected_font_color", "is_overriding_selected_font_color");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deselect_on_focus_loss_enabled"), "set_deselect_on_focus_loss_enabled", "is_deselect_on_focus_loss_enabled");

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "custom_effects", PROPERTY_HINT_ARRAY_TYPE, vformat("%s/%s:%s", Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, "RichTextEffect"), (PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SCRIPT_VARIABLE)), "set_effects", "get_effects");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");

	ADD_GROUP("Structured Text", "structured_text_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "structured_text_bidi_override", PROPERTY_HINT_ENUM, "Default,URI,File,Email,List,None,Custom"), "set_structured_text_bidi_override", "get_structured_text_bidi_override");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "structured_text_bidi_override_options"), "set_structured_text_bidi_override_options", "get_structured_text_bidi_override_options");

	ADD_SIGNAL(MethodInfo("meta_clicked", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("meta_hover_started", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("meta_hover_ended", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
	BIND_ENUM_CONSTANT(ALIGN_FILL);

	BIND_ENUM_CONSTANT(LIST_NUMBERS);
	BIND_ENUM_CONSTANT(LIST_LETTERS);
	BIND_ENUM_CONSTANT(LIST_ROMAN);
	BIND_ENUM_CONSTANT(LIST_DOTS);

	BIND_ENUM_CONSTANT(ITEM_FRAME);
	BIND_ENUM_CONSTANT(ITEM_TEXT);
	BIND_ENUM_CONSTANT(ITEM_IMAGE);
	BIND_ENUM_CONSTANT(ITEM_NEWLINE);
	BIND_ENUM_CONSTANT(ITEM_FONT);
	BIND_ENUM_CONSTANT(ITEM_FONT_SIZE);
	BIND_ENUM_CONSTANT(ITEM_FONT_FEATURES);
	BIND_ENUM_CONSTANT(ITEM_COLOR);
	BIND_ENUM_CONSTANT(ITEM_OUTLINE_SIZE);
	BIND_ENUM_CONSTANT(ITEM_OUTLINE_COLOR);
	BIND_ENUM_CONSTANT(ITEM_UNDERLINE);
	BIND_ENUM_CONSTANT(ITEM_STRIKETHROUGH);
	BIND_ENUM_CONSTANT(ITEM_PARAGRAPH);
	BIND_ENUM_CONSTANT(ITEM_INDENT);
	BIND_ENUM_CONSTANT(ITEM_LIST);
	BIND_ENUM_CONSTANT(ITEM_TABLE);
	BIND_ENUM_CONSTANT(ITEM_FADE);
	BIND_ENUM_CONSTANT(ITEM_SHAKE);
	BIND_ENUM_CONSTANT(ITEM_WAVE);
	BIND_ENUM_CONSTANT(ITEM_TORNADO);
	BIND_ENUM_CONSTANT(ITEM_RAINBOW);
	BIND_ENUM_CONSTANT(ITEM_BGCOLOR);
	BIND_ENUM_CONSTANT(ITEM_FGCOLOR);
	BIND_ENUM_CONSTANT(ITEM_META);
	BIND_ENUM_CONSTANT(ITEM_DROPCAP);
	BIND_ENUM_CONSTANT(ITEM_CUSTOMFX);
}

void RichTextLabel::set_visible_characters(int p_visible) {
	if (visible_characters != p_visible) {
		visible_characters = p_visible;
		if (p_visible == -1) {
			percent_visible = 1;
		} else {
			int total_char_count = get_total_character_count();
			if (total_char_count > 0) {
				percent_visible = (float)p_visible / (float)total_char_count;
			}
		}
		main->first_invalid_line = 0; //invalidate ALL
		_validate_line_caches(main);
		update();
	}
}

int RichTextLabel::get_visible_characters() const {
	return visible_characters;
}

int RichTextLabel::get_total_character_count() const {
	// Note: Do not use line buffer "char_count", it includes only visible characters.
	int tc = 0;
	Item *it = main;
	while (it) {
		if (it->type == ITEM_TEXT) {
			ItemText *t = static_cast<ItemText *>(it);
			tc += t->text.length();
		} else if (it->type == ITEM_NEWLINE) {
			tc++;
		} else if (it->type == ITEM_IMAGE) {
			tc++;
		}
		it = _get_next_item(it, true);
	}

	return tc;
}

void RichTextLabel::set_fixed_size_to_width(int p_width) {
	fixed_width = p_width;
	minimum_size_changed();
}

Size2 RichTextLabel::get_minimum_size() const {
	Ref<StyleBox> style = get_theme_stylebox(SNAME("normal"));
	Size2 size = style->get_minimum_size();

	if (fixed_width != -1) {
		size.x += fixed_width;
	}

	if (fixed_width != -1 || fit_content_height) {
		const_cast<RichTextLabel *>(this)->_validate_line_caches(main);
		size.y += get_content_height();
	}

	return size;
}

void RichTextLabel::_draw_fbg_boxes(RID p_ci, RID p_rid, Vector2 line_off, Item *it_from, Item *it_to, int start, int end, int fbg_flag) {
	Vector2i fbg_index = Vector2i(end, start);
	Color last_color = Color(0, 0, 0, 0);
	bool draw_box = false;
	// Draw a box based on color tags associated with glyphs
	for (int i = start; i < end; i++) {
		Item *it = _get_item_at_pos(it_from, it_to, i);
		Color color = Color(0, 0, 0, 0);

		if (fbg_flag == 0) {
			color = _find_bgcolor(it);
		} else {
			color = _find_fgcolor(it);
		}

		bool change_to_color = ((color.a > 0) && ((last_color.a - 0.0) < 0.01));
		bool change_from_color = (((color.a - 0.0) < 0.01) && (last_color.a > 0.0));
		bool change_color = (((color.a > 0) == (last_color.a > 0)) && (color != last_color));

		if (change_to_color) {
			fbg_index.x = MIN(i, fbg_index.x);
			fbg_index.y = MAX(i, fbg_index.y);
		}

		if (change_from_color || change_color) {
			fbg_index.x = MIN(i, fbg_index.x);
			fbg_index.y = MAX(i, fbg_index.y);
			draw_box = true;
		}

		if (draw_box) {
			Vector<Vector2> sel = TS->shaped_text_get_selection(p_rid, fbg_index.x, fbg_index.y);
			for (int j = 0; j < sel.size(); j++) {
				Vector2 rect_off = line_off + Vector2(sel[j].x, -TS->shaped_text_get_ascent(p_rid));
				Vector2 rect_size = Vector2(sel[j].y - sel[j].x, TS->shaped_text_get_size(p_rid).y);
				RenderingServer::get_singleton()->canvas_item_add_rect(p_ci, Rect2(rect_off, rect_size), last_color);
			}
			fbg_index = Vector2i(end, start);
			draw_box = false;
		}

		if (change_color) {
			fbg_index.x = MIN(i, fbg_index.x);
			fbg_index.y = MAX(i, fbg_index.y);
		}

		last_color = color;
	}

	if (last_color.a > 0) {
		Vector<Vector2> sel = TS->shaped_text_get_selection(p_rid, fbg_index.x, end);
		for (int i = 0; i < sel.size(); i++) {
			Vector2 rect_off = line_off + Vector2(sel[i].x, -TS->shaped_text_get_ascent(p_rid));
			Vector2 rect_size = Vector2(sel[i].y - sel[i].x, TS->shaped_text_get_size(p_rid).y);
			RenderingServer::get_singleton()->canvas_item_add_rect(p_ci, Rect2(rect_off, rect_size), last_color);
		}
	}
}

Ref<RichTextEffect> RichTextLabel::_get_custom_effect_by_code(String p_bbcode_identifier) {
	for (int i = 0; i < custom_effects.size(); i++) {
		Ref<RichTextEffect> effect = custom_effects[i];
		if (!effect.is_valid()) {
			continue;
		}

		if (effect->get_bbcode() == p_bbcode_identifier) {
			return effect;
		}
	}

	return Ref<RichTextEffect>();
}

Dictionary RichTextLabel::parse_expressions_for_values(Vector<String> p_expressions) {
	Dictionary d = Dictionary();
	for (int i = 0; i < p_expressions.size(); i++) {
		String expression = p_expressions[i];

		Array a = Array();
		Vector<String> parts = expression.split("=", true);
		String key = parts[0];
		if (parts.size() != 2) {
			return d;
		}

		Vector<String> values = parts[1].split(",", false);

#ifdef MODULE_REGEX_ENABLED
		RegEx color = RegEx();
		color.compile("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$");
		RegEx nodepath = RegEx();
		nodepath.compile("^\\$");
		RegEx boolean = RegEx();
		boolean.compile("^(true|false)$");
		RegEx decimal = RegEx();
		decimal.compile("^-?^.?\\d+(\\.\\d+?)?$");
		RegEx numerical = RegEx();
		numerical.compile("^\\d+$");

		for (int j = 0; j < values.size(); j++) {
			if (!color.search(values[j]).is_null()) {
				a.append(Color::html(values[j]));
			} else if (!nodepath.search(values[j]).is_null()) {
				if (values[j].begins_with("$")) {
					String v = values[j].substr(1, values[j].length());
					a.append(NodePath(v));
				}
			} else if (!boolean.search(values[j]).is_null()) {
				if (values[j] == "true") {
					a.append(true);
				} else if (values[j] == "false") {
					a.append(false);
				}
			} else if (!decimal.search(values[j]).is_null()) {
				a.append(values[j].to_float());
			} else if (!numerical.search(values[j]).is_null()) {
				a.append(values[j].to_int());
			} else {
				a.append(values[j]);
			}
		}
#endif

		if (values.size() > 1) {
			d[key] = a;
		} else if (values.size() == 1) {
			d[key] = a[0];
		}
	}
	return d;
}

RichTextLabel::RichTextLabel() {
	main = memnew(ItemFrame);
	main->index = 0;
	current = main;
	main->lines.resize(1);
	main->lines.write[0].from = main;
	main->first_invalid_line = 0;
	main->first_resized_line = 0;
	current_frame = main;

	vscroll = memnew(VScrollBar);
	add_child(vscroll, false, INTERNAL_MODE_FRONT);
	vscroll->set_drag_node(String(".."));
	vscroll->set_step(1);
	vscroll->set_anchor_and_offset(SIDE_TOP, ANCHOR_BEGIN, 0);
	vscroll->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0);
	vscroll->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0);
	vscroll->connect("value_changed", callable_mp(this, &RichTextLabel::_scroll_changed));
	vscroll->set_step(1);
	vscroll->hide();

	set_clip_contents(true);
}

RichTextLabel::~RichTextLabel() {
	memdelete(main);
}
