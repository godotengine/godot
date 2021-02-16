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
#include "modules/regex/regex.h"
#include "scene/scene_string_names.h"
#ifdef TOOLS_ENABLED
#include "editor/editor_scale.h"
#endif

RichTextLabel::Item *RichTextLabel::_get_next_item(Item *p_item, bool p_free) {

	if (p_free) {

		if (p_item->subitems.size()) {

			return p_item->subitems.front()->get();
		} else if (!p_item->parent) {
			return NULL;
		} else if (p_item->E->next()) {

			return p_item->E->next()->get();
		} else {
			//go up until something with a next is found
			while (p_item->parent && !p_item->E->next()) {
				p_item = p_item->parent;
			}

			if (p_item->parent)
				return p_item->E->next()->get();
			else
				return NULL;
		}

	} else {
		if (p_item->subitems.size() && p_item->type != ITEM_TABLE) {

			return p_item->subitems.front()->get();
		} else if (p_item->type == ITEM_FRAME) {
			return NULL;
		} else if (p_item->E->next()) {

			return p_item->E->next()->get();
		} else {
			//go up until something with a next is found
			while (p_item->type != ITEM_FRAME && !p_item->E->next()) {
				p_item = p_item->parent;
			}

			if (p_item->type != ITEM_FRAME)
				return p_item->E->next()->get();
			else
				return NULL;
		}
	}

	return NULL;
}

RichTextLabel::Item *RichTextLabel::_get_prev_item(Item *p_item, bool p_free) {
	if (p_free) {

		if (p_item->subitems.size()) {

			return p_item->subitems.back()->get();
		} else if (!p_item->parent) {
			return NULL;
		} else if (p_item->E->prev()) {

			return p_item->E->prev()->get();
		} else {
			//go back until something with a prev is found
			while (p_item->parent && !p_item->E->prev()) {
				p_item = p_item->parent;
			}

			if (p_item->parent)
				return p_item->E->prev()->get();
			else
				return NULL;
		}

	} else {
		if (p_item->subitems.size() && p_item->type != ITEM_TABLE) {

			return p_item->subitems.back()->get();
		} else if (p_item->type == ITEM_FRAME) {
			return NULL;
		} else if (p_item->E->prev()) {

			return p_item->E->prev()->get();
		} else {
			//go back until something with a prev is found
			while (p_item->type != ITEM_FRAME && !p_item->E->prev()) {
				p_item = p_item->parent;
			}

			if (p_item->type != ITEM_FRAME)
				return p_item->E->prev()->get();
			else
				return NULL;
		}
	}

	return NULL;
}

Rect2 RichTextLabel::_get_text_rect() {
	Ref<StyleBox> style = get_stylebox("normal");
	return Rect2(style->get_offset(), get_size() - style->get_minimum_size());
}

int RichTextLabel::_process_line(ItemFrame *p_frame, const Vector2 &p_ofs, int &y, int p_width, int p_line, ProcessMode p_mode, const Ref<Font> &p_base_font, const Color &p_base_color, const Color &p_font_color_shadow, bool p_shadow_as_outline, const Point2 &shadow_ofs, const Point2i &p_click_pos, Item **r_click_item, int *r_click_char, bool *r_outside, int p_char_count) {

	ERR_FAIL_INDEX_V((int)p_mode, 3, 0);

	RID ci;
	if (r_outside)
		*r_outside = false;
	if (p_mode == PROCESS_DRAW) {
		ci = get_canvas_item();

		if (r_click_item)
			*r_click_item = NULL;
	}
	Line &l = p_frame->lines.write[p_line];
	Item *it = l.from;

	int line_ofs = 0;
	int margin = _find_margin(it, p_base_font);
	Align align = _find_align(it);
	int line = 0;
	int spaces = 0;

	int height = get_size().y;

	if (p_mode != PROCESS_CACHE) {

		ERR_FAIL_INDEX_V(line, l.offset_caches.size(), 0);
		line_ofs = l.offset_caches[line];
	}

	if (p_mode == PROCESS_CACHE) {
		l.offset_caches.clear();
		l.height_caches.clear();
		l.ascent_caches.clear();
		l.descent_caches.clear();
		l.char_count = 0;
		l.minimum_width = 0;
		l.maximum_width = 0;
	}

	int wofs = margin;
	int spaces_size = 0;
	int align_ofs = 0;

	if (p_mode != PROCESS_CACHE && align != ALIGN_FILL)
		wofs += line_ofs;

	int begin = margin;

	Ref<Font> cfont = _find_font(it);
	if (cfont.is_null())
		cfont = p_base_font;

	//line height should be the font height for the first time, this ensures that an empty line will never have zero height and successive newlines are displayed
	int line_height = cfont->get_height();
	int line_ascent = cfont->get_ascent();
	int line_descent = cfont->get_descent();

	int backtrack = 0; // for dynamic hidden content.

	int nonblank_line_count = 0; //number of nonblank lines as counted during PROCESS_DRAW

	Variant meta;

#define RETURN return nonblank_line_count

#define NEW_LINE                                                                                                                                                \
	{                                                                                                                                                           \
		if (p_mode != PROCESS_CACHE) {                                                                                                                          \
			line++;                                                                                                                                             \
			backtrack = 0;                                                                                                                                      \
			if (!line_is_blank) {                                                                                                                               \
				nonblank_line_count++;                                                                                                                          \
			}                                                                                                                                                   \
			line_is_blank = true;                                                                                                                               \
			if (line < l.offset_caches.size())                                                                                                                  \
				line_ofs = l.offset_caches[line];                                                                                                               \
			wofs = margin;                                                                                                                                      \
			if (align != ALIGN_FILL)                                                                                                                            \
				wofs += line_ofs;                                                                                                                               \
		} else {                                                                                                                                                \
			int used = wofs - margin;                                                                                                                           \
			switch (align) {                                                                                                                                    \
				case ALIGN_LEFT: l.offset_caches.push_back(0); break;                                                                                           \
				case ALIGN_CENTER: l.offset_caches.push_back(((p_width - margin) - used) / 2); break;                                                           \
				case ALIGN_RIGHT: l.offset_caches.push_back(((p_width - margin) - used)); break;                                                                \
				case ALIGN_FILL: l.offset_caches.push_back(line_wrapped ? ((p_width - margin) - used) : 0); break;                                              \
			}                                                                                                                                                   \
			l.height_caches.push_back(line_height);                                                                                                             \
			l.ascent_caches.push_back(line_ascent);                                                                                                             \
			l.descent_caches.push_back(line_descent);                                                                                                           \
			l.space_caches.push_back(spaces);                                                                                                                   \
		}                                                                                                                                                       \
		line_wrapped = false;                                                                                                                                   \
		y += line_height + get_constant(SceneStringNames::get_singleton()->line_separation);                                                                    \
		line_height = 0;                                                                                                                                        \
		line_ascent = 0;                                                                                                                                        \
		line_descent = 0;                                                                                                                                       \
		spaces = 0;                                                                                                                                             \
		spaces_size = 0;                                                                                                                                        \
		wofs = begin;                                                                                                                                           \
		align_ofs = 0;                                                                                                                                          \
		if (p_mode != PROCESS_CACHE) {                                                                                                                          \
			lh = line < l.height_caches.size() ? l.height_caches[line] : 1;                                                                                     \
			line_ascent = line < l.ascent_caches.size() ? l.ascent_caches[line] : 1;                                                                            \
			line_descent = line < l.descent_caches.size() ? l.descent_caches[line] : 1;                                                                         \
			if (align != ALIGN_FILL) {                                                                                                                          \
				if (line < l.offset_caches.size()) {                                                                                                            \
					wofs = l.offset_caches[line];                                                                                                               \
				}                                                                                                                                               \
			}                                                                                                                                                   \
		}                                                                                                                                                       \
		if (p_mode == PROCESS_POINTER && r_click_item && p_click_pos.y >= p_ofs.y + y && p_click_pos.y <= p_ofs.y + y + lh && p_click_pos.x < p_ofs.x + wofs) { \
			if (r_outside) *r_outside = true;                                                                                                                   \
			*r_click_item = it;                                                                                                                                 \
			*r_click_char = rchar;                                                                                                                              \
			RETURN;                                                                                                                                             \
		}                                                                                                                                                       \
	}

#define ENSURE_WIDTH(m_width)                                                                                                               \
	if (p_mode == PROCESS_CACHE) {                                                                                                          \
		l.maximum_width = MAX(l.maximum_width, MIN(p_width, wofs + m_width));                                                               \
		l.minimum_width = MAX(l.minimum_width, m_width);                                                                                    \
	}                                                                                                                                       \
	if (wofs - backtrack + m_width > p_width) {                                                                                             \
		line_wrapped = true;                                                                                                                \
		if (p_mode == PROCESS_CACHE) {                                                                                                      \
			if (spaces > 0)                                                                                                                 \
				spaces -= 1;                                                                                                                \
		}                                                                                                                                   \
		const bool x_in_range = (p_click_pos.x > p_ofs.x + wofs) && (!p_frame->cell || p_click_pos.x < p_ofs.x + p_width);                  \
		if (p_mode == PROCESS_POINTER && r_click_item && p_click_pos.y >= p_ofs.y + y && p_click_pos.y <= p_ofs.y + y + lh && x_in_range) { \
			if (r_outside) *r_outside = true;                                                                                               \
			*r_click_item = it;                                                                                                             \
			*r_click_char = rchar;                                                                                                          \
			RETURN;                                                                                                                         \
		}                                                                                                                                   \
		NEW_LINE                                                                                                                            \
	}

#define ADVANCE(m_width)                                                                                                                                                                                     \
	{                                                                                                                                                                                                        \
		if (p_mode == PROCESS_POINTER && r_click_item && p_click_pos.y >= p_ofs.y + y && p_click_pos.y <= p_ofs.y + y + lh && p_click_pos.x >= p_ofs.x + wofs && p_click_pos.x < p_ofs.x + wofs + m_width) { \
			if (r_outside) *r_outside = false;                                                                                                                                                               \
			*r_click_item = it;                                                                                                                                                                              \
			*r_click_char = rchar;                                                                                                                                                                           \
			RETURN;                                                                                                                                                                                          \
		}                                                                                                                                                                                                    \
		wofs += m_width;                                                                                                                                                                                     \
	}

#define CHECK_HEIGHT(m_height)    \
	if (m_height > line_height) { \
		line_height = m_height;   \
	}

#define YRANGE_VISIBLE(m_top, m_height) \
	(m_height > 0 && ((m_top >= 0 && m_top < height) || ((m_top + m_height - 1) >= 0 && (m_top + m_height - 1) < height)))

	Color selection_fg;
	Color selection_bg;

	if (p_mode == PROCESS_DRAW) {
		selection_fg = get_color("font_color_selected");
		selection_bg = get_color("selection_color");
	}

	int rchar = 0;
	int lh = 0;
	bool line_is_blank = true;
	bool line_wrapped = false;
	int fh = 0;

	while (it) {

		switch (it->type) {

			case ITEM_ALIGN: {

				ItemAlign *align_it = static_cast<ItemAlign *>(it);

				align = align_it->align;

			} break;
			case ITEM_INDENT: {

				if (it != l.from) {
					ItemIndent *indent_it = static_cast<ItemIndent *>(it);

					int indent = indent_it->level * tab_size * cfont->get_char_size(' ').width;
					margin += indent;
					begin += indent;
					wofs += indent;
				}

			} break;
			case ITEM_TEXT: {

				ItemText *text = static_cast<ItemText *>(it);

				Ref<Font> font = _find_font(it);
				if (font.is_null())
					font = p_base_font;

				const CharType *c = text->text.c_str();
				const CharType *cf = c;
				int ascent = font->get_ascent();
				int descent = font->get_descent();

				Color color;
				Color font_color_shadow;
				bool underline = false;
				bool strikethrough = false;
				ItemFade *fade = NULL;
				int it_char_start = p_char_count;

				Vector<ItemFX *> fx_stack = Vector<ItemFX *>();
				_fetch_item_fx_stack(text, fx_stack);
				bool custom_fx_ok = true;

				if (p_mode == PROCESS_DRAW) {
					color = _find_color(text, p_base_color);
					font_color_shadow = _find_color(text, p_font_color_shadow);
					if (_find_underline(text) || (_find_meta(text, &meta) && underline_meta)) {
						underline = true;
					} else if (_find_strikethrough(text)) {
						strikethrough = true;
					}

					Item *fade_item = it;
					while (fade_item) {
						if (fade_item->type == ITEM_FADE) {
							fade = static_cast<ItemFade *>(fade_item);
							break;
						}
						fade_item = fade_item->parent;
					}

				} else if (p_mode == PROCESS_CACHE) {
					l.char_count += text->text.length();
				}

				rchar = 0;
				FontDrawer drawer(font, Color(1, 1, 1));
				while (*c) {

					int end = 0;
					int w = 0;
					int fw = 0;

					lh = 0;

					if (p_mode != PROCESS_CACHE) {
						lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
						line_ascent = line < l.ascent_caches.size() ? l.ascent_caches[line] : 1;
						line_descent = line < l.descent_caches.size() ? l.descent_caches[line] : 1;
					}
					while (c[end] != 0 && !(end && c[end - 1] == ' ' && c[end] != ' ')) {

						int cw = font->get_char_size(c[end], c[end + 1]).width;
						if (c[end] == '\t') {
							cw = tab_size * font->get_char_size(' ').width;
						}

						if (end > 0 && w + cw + begin > p_width) {
							break; //don't allow lines longer than assigned width
						}

						w += cw;
						fw += cw;

						end++;
					}
					CHECK_HEIGHT(fh);
					ENSURE_WIDTH(w);

					line_ascent = MAX(line_ascent, ascent);
					line_descent = MAX(line_descent, descent);
					fh = line_ascent + line_descent;

					if (end && c[end - 1] == ' ') {
						if (p_mode == PROCESS_CACHE) {
							spaces_size += font->get_char_size(' ').width;
						} else if (align == ALIGN_FILL) {
							int ln = MIN(l.offset_caches.size() - 1, line);
							if (l.space_caches[ln]) {
								align_ofs = spaces * l.offset_caches[ln] / l.space_caches[ln];
							}
						}
						spaces++;
					}

					{

						int ofs = 0 - backtrack;

						for (int i = 0; i < end; i++) {
							int pofs = wofs + ofs;

							if (p_mode == PROCESS_POINTER && r_click_char && p_click_pos.y >= p_ofs.y + y && p_click_pos.y <= p_ofs.y + y + lh) {

								int cw = font->get_char_size(c[i], c[i + 1]).x;

								if (c[i] == '\t') {
									cw = tab_size * font->get_char_size(' ').width;
								}

								if (p_click_pos.x - cw / 2 > p_ofs.x + align_ofs + pofs) {

									rchar = int((&c[i]) - cf);
								}

								ofs += cw;
							} else if (p_mode == PROCESS_DRAW) {
								bool selected = false;
								Color fx_color = Color(color);
								Point2 fx_offset;
								CharType fx_char = c[i];

								if (selection.active) {

									int cofs = (&c[i]) - cf;
									if ((text->index > selection.from->index || (text->index == selection.from->index && cofs >= selection.from_char)) && (text->index < selection.to->index || (text->index == selection.to->index && cofs <= selection.to_char))) {
										selected = true;
									}
								}

								int cw = 0;
								int c_item_offset = p_char_count - it_char_start;

								float faded_visibility = 1.0f;
								if (fade) {
									if (c_item_offset >= fade->starting_index) {
										faded_visibility -= (float)(c_item_offset - fade->starting_index) / (float)fade->length;
										faded_visibility = faded_visibility < 0.0f ? 0.0f : faded_visibility;
									}
									fx_color.a = faded_visibility;
								}

								bool visible = visible_characters < 0 || ((p_char_count < visible_characters && YRANGE_VISIBLE(y + lh - line_descent - line_ascent, line_ascent + line_descent)) &&
																				 faded_visibility > 0.0f);

								const bool previously_visible = visible;

								for (int j = 0; j < fx_stack.size(); j++) {

									ItemFX *item_fx = fx_stack[j];

									if (item_fx->type == ITEM_CUSTOMFX && custom_fx_ok) {
										ItemCustomFX *item_custom = static_cast<ItemCustomFX *>(item_fx);

										Ref<CharFXTransform> charfx = item_custom->char_fx_transform;
										Ref<RichTextEffect> custom_effect = item_custom->custom_effect;

										if (!custom_effect.is_null()) {
											charfx->elapsed_time = item_custom->elapsed_time;
											charfx->relative_index = c_item_offset;
											charfx->absolute_index = p_char_count;
											charfx->visibility = visible;
											charfx->offset = fx_offset;
											charfx->color = fx_color;
											charfx->character = fx_char;

											bool effect_status = custom_effect->_process_effect_impl(charfx);
											custom_fx_ok = effect_status;

											fx_offset += charfx->offset;
											fx_color = charfx->color;
											visible &= charfx->visibility;
											fx_char = charfx->character;
										}
									} else if (item_fx->type == ITEM_SHAKE) {
										ItemShake *item_shake = static_cast<ItemShake *>(item_fx);

										uint64_t char_current_rand = item_shake->offset_random(c_item_offset);
										uint64_t char_previous_rand = item_shake->offset_previous_random(c_item_offset);
										uint64_t max_rand = 2147483647;
										double current_offset = Math::range_lerp(char_current_rand % max_rand, 0, max_rand, 0.0f, 2.f * (float)Math_PI);
										double previous_offset = Math::range_lerp(char_previous_rand % max_rand, 0, max_rand, 0.0f, 2.f * (float)Math_PI);
										double n_time = (double)(item_shake->elapsed_time / (0.5f / item_shake->rate));
										n_time = (n_time > 1.0) ? 1.0 : n_time;
										fx_offset += Point2(Math::lerp(Math::sin(previous_offset),
																	Math::sin(current_offset),
																	n_time),
															 Math::lerp(Math::cos(previous_offset),
																	 Math::cos(current_offset),
																	 n_time)) *
													 (float)item_shake->strength / 10.0f;
									} else if (item_fx->type == ITEM_WAVE) {
										ItemWave *item_wave = static_cast<ItemWave *>(item_fx);

										double value = Math::sin(item_wave->frequency * item_wave->elapsed_time + ((p_ofs.x + pofs) / 50)) * (item_wave->amplitude / 10.0f);
										fx_offset += Point2(0, 1) * value;
									} else if (item_fx->type == ITEM_TORNADO) {
										ItemTornado *item_tornado = static_cast<ItemTornado *>(item_fx);

										double torn_x = Math::sin(item_tornado->frequency * item_tornado->elapsed_time + ((p_ofs.x + pofs) / 50)) * (item_tornado->radius);
										double torn_y = Math::cos(item_tornado->frequency * item_tornado->elapsed_time + ((p_ofs.x + pofs) / 50)) * (item_tornado->radius);
										fx_offset += Point2(torn_x, torn_y);
									} else if (item_fx->type == ITEM_RAINBOW) {
										ItemRainbow *item_rainbow = static_cast<ItemRainbow *>(item_fx);

										fx_color = fx_color.from_hsv(item_rainbow->frequency * (item_rainbow->elapsed_time + ((p_ofs.x + pofs) / 50)),
												item_rainbow->saturation,
												item_rainbow->value,
												fx_color.a);
									}
								}

								if (visible)
									line_is_blank = false;

								if (c[i] == '\t')
									visible = false;

								if (visible) {

									if (selected) {
										cw = font->get_char_size(fx_char, c[i + 1]).x;
										draw_rect(Rect2(p_ofs.x + pofs, p_ofs.y + y, cw, lh), selection_bg);
									}

									if (p_font_color_shadow.a > 0) {
										float x_ofs_shadow = align_ofs + pofs;
										float y_ofs_shadow = y + lh - line_descent;
										font->draw_char(ci, Point2(x_ofs_shadow, y_ofs_shadow) + shadow_ofs + fx_offset, fx_char, c[i + 1], p_font_color_shadow);

										if (p_shadow_as_outline) {
											font->draw_char(ci, Point2(x_ofs_shadow, y_ofs_shadow) + Vector2(-shadow_ofs.x, shadow_ofs.y) + fx_offset, fx_char, c[i + 1], p_font_color_shadow);
											font->draw_char(ci, Point2(x_ofs_shadow, y_ofs_shadow) + Vector2(shadow_ofs.x, -shadow_ofs.y) + fx_offset, fx_char, c[i + 1], p_font_color_shadow);
											font->draw_char(ci, Point2(x_ofs_shadow, y_ofs_shadow) + Vector2(-shadow_ofs.x, -shadow_ofs.y) + fx_offset, fx_char, c[i + 1], p_font_color_shadow);
										}
									}

									if (selected) {
										drawer.draw_char(ci, p_ofs + Point2(align_ofs + pofs, y + lh - line_descent), fx_char, c[i + 1], override_selected_font_color ? selection_fg : fx_color);
									} else {
										cw = drawer.draw_char(ci, p_ofs + Point2(align_ofs + pofs, y + lh - line_descent) + fx_offset, fx_char, c[i + 1], fx_color);
									}
								} else if (previously_visible && c[i] != '\t') {
									backtrack += font->get_char_size(fx_char, c[i + 1]).x;
								}

								p_char_count++;
								if (c[i] == '\t') {
									cw = tab_size * font->get_char_size(' ').width;
									backtrack = MAX(0, backtrack - cw);
								}

								ofs += cw;
							}
						}

						if (underline) {
							Color uc = color;
							uc.a *= 0.5;
							int uy = y + lh - line_descent + 2;
							float underline_width = 1.0;
#ifdef TOOLS_ENABLED
							underline_width *= EDSCALE;
#endif
							VS::get_singleton()->canvas_item_add_line(ci, p_ofs + Point2(align_ofs + wofs, uy), p_ofs + Point2(align_ofs + wofs + w, uy), uc, underline_width);
						} else if (strikethrough) {
							Color uc = color;
							uc.a *= 0.5;
							int uy = y + lh - (line_ascent + line_descent) / 2;
							float strikethrough_width = 1.0;
#ifdef TOOLS_ENABLED
							strikethrough_width *= EDSCALE;
#endif
							VS::get_singleton()->canvas_item_add_line(ci, p_ofs + Point2(align_ofs + wofs, uy), p_ofs + Point2(align_ofs + wofs + w, uy), uc, strikethrough_width);
						}
					}

					ADVANCE(fw);
					CHECK_HEIGHT(fh); //must be done somewhere
					c = &c[end];
				}

			} break;
			case ITEM_IMAGE: {

				lh = 0;
				if (p_mode != PROCESS_CACHE)
					lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
				else
					l.char_count += 1; //images count as chars too

				ItemImage *img = static_cast<ItemImage *>(it);

				Ref<Font> font = _find_font(it);
				if (font.is_null())
					font = p_base_font;

				if (p_mode == PROCESS_POINTER && r_click_char)
					*r_click_char = 0;

				ENSURE_WIDTH(img->size.width);

				bool visible = visible_characters < 0 || (p_char_count < visible_characters && YRANGE_VISIBLE(y + lh - font->get_descent() - img->size.height, img->size.height));
				if (visible)
					line_is_blank = false;

				if (p_mode == PROCESS_DRAW && visible) {
					img->image->draw_rect(ci, Rect2(p_ofs + Point2(align_ofs + wofs, y + lh - font->get_descent() - img->size.height), img->size));
				}
				p_char_count++;

				ADVANCE(img->size.width);
				CHECK_HEIGHT((img->size.height + font->get_descent()));

			} break;
			case ITEM_NEWLINE: {

				lh = 0;

				if (p_mode != PROCESS_CACHE) {
					lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
					line_is_blank = true;
				}

			} break;
			case ITEM_TABLE: {

				lh = 0;
				ItemTable *table = static_cast<ItemTable *>(it);
				int hseparation = get_constant("table_hseparation");
				int vseparation = get_constant("table_vseparation");
				Color ccolor = _find_color(table, p_base_color);
				Vector2 draw_ofs = Point2(wofs, y);
				Color font_color_shadow = get_color("font_color_shadow");
				bool use_outline = get_constant("shadow_as_outline");
				Point2 shadow_ofs2(get_constant("shadow_offset_x"), get_constant("shadow_offset_y"));

				if (p_mode == PROCESS_CACHE) {

					int idx = 0;
					//set minimums to zero
					for (int i = 0; i < table->columns.size(); i++) {
						table->columns.write[i].min_width = 0;
						table->columns.write[i].max_width = 0;
						table->columns.write[i].width = 0;
					}
					//compute minimum width for each cell
					const int available_width = p_width - hseparation * (table->columns.size() - 1) - wofs;

					for (List<Item *>::Element *E = table->subitems.front(); E; E = E->next()) {
						ERR_CONTINUE(E->get()->type != ITEM_FRAME); //children should all be frames
						ItemFrame *frame = static_cast<ItemFrame *>(E->get());

						int column = idx % table->columns.size();

						int ly = 0;

						for (int i = 0; i < frame->lines.size(); i++) {

							_process_line(frame, Point2(), ly, available_width, i, PROCESS_CACHE, cfont, Color(), font_color_shadow, use_outline, shadow_ofs2);
							table->columns.write[column].min_width = MAX(table->columns[column].min_width, frame->lines[i].minimum_width);
							table->columns.write[column].max_width = MAX(table->columns[column].max_width, frame->lines[i].maximum_width);
						}
						idx++;
					}

					//compute available width and total ratio (for expanders)

					int total_ratio = 0;
					int remaining_width = available_width;
					table->total_width = hseparation;

					for (int i = 0; i < table->columns.size(); i++) {
						remaining_width -= table->columns[i].min_width;
						if (table->columns[i].max_width > table->columns[i].min_width)
							table->columns.write[i].expand = true;
						if (table->columns[i].expand)
							total_ratio += table->columns[i].expand_ratio;
					}

					//assign actual widths
					for (int i = 0; i < table->columns.size(); i++) {
						table->columns.write[i].width = table->columns[i].min_width;
						if (table->columns[i].expand && total_ratio > 0)
							table->columns.write[i].width += table->columns[i].expand_ratio * remaining_width / total_ratio;
						table->total_width += table->columns[i].width + hseparation;
					}

					//resize to max_width if needed and distribute the remaining space
					bool table_need_fit = true;
					while (table_need_fit) {
						table_need_fit = false;
						//fit slim
						for (int i = 0; i < table->columns.size(); i++) {
							if (!table->columns[i].expand)
								continue;
							int dif = table->columns[i].width - table->columns[i].max_width;
							if (dif > 0) {
								table_need_fit = true;
								table->columns.write[i].width = table->columns[i].max_width;
								table->total_width -= dif;
								total_ratio -= table->columns[i].expand_ratio;
							}
						}
						//grow
						remaining_width = available_width - table->total_width;
						if (remaining_width > 0 && total_ratio > 0) {
							for (int i = 0; i < table->columns.size(); i++) {
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

					//compute caches properly again with the right width
					idx = 0;
					for (List<Item *>::Element *E = table->subitems.front(); E; E = E->next()) {
						ERR_CONTINUE(E->get()->type != ITEM_FRAME); //children should all be frames
						ItemFrame *frame = static_cast<ItemFrame *>(E->get());

						int column = idx % table->columns.size();

						for (int i = 0; i < frame->lines.size(); i++) {

							int ly = 0;
							_process_line(frame, Point2(), ly, table->columns[column].width, i, PROCESS_CACHE, cfont, Color(), font_color_shadow, use_outline, shadow_ofs2);
							frame->lines.write[i].height_cache = ly; //actual height
							frame->lines.write[i].height_accum_cache = ly; //actual height
						}
						idx++;
					}
				}

				Point2 offset(align_ofs + hseparation, vseparation);

				int row_height = 0;
				//draw using computed caches
				int idx = 0;
				for (List<Item *>::Element *E = table->subitems.front(); E; E = E->next()) {
					ERR_CONTINUE(E->get()->type != ITEM_FRAME); //children should all be frames
					ItemFrame *frame = static_cast<ItemFrame *>(E->get());

					int column = idx % table->columns.size();

					int ly = 0;
					int yofs = 0;

					int lines_h = frame->lines[frame->lines.size() - 1].height_accum_cache - (frame->lines[0].height_accum_cache - frame->lines[0].height_cache);
					int lines_ofs = p_ofs.y + offset.y + draw_ofs.y;

					bool visible = lines_ofs < get_size().height && lines_ofs + lines_h >= 0;
					if (visible)
						line_is_blank = false;

					for (int i = 0; i < frame->lines.size(); i++) {

						if (visible) {
							if (p_mode == PROCESS_DRAW) {
								nonblank_line_count += _process_line(frame, p_ofs + offset + draw_ofs + Vector2(0, yofs), ly, table->columns[column].width, i, PROCESS_DRAW, cfont, ccolor, font_color_shadow, use_outline, shadow_ofs2);
							} else if (p_mode == PROCESS_POINTER) {
								_process_line(frame, p_ofs + offset + draw_ofs + Vector2(0, yofs), ly, table->columns[column].width, i, PROCESS_POINTER, cfont, ccolor, font_color_shadow, use_outline, shadow_ofs2, p_click_pos, r_click_item, r_click_char, r_outside);
								if (r_click_item && *r_click_item) {
									RETURN; // exit early
								}
							}
						}

						yofs += frame->lines[i].height_cache;
						if (p_mode == PROCESS_CACHE) {
							frame->lines.write[i].height_accum_cache = offset.y + draw_ofs.y + frame->lines[i].height_cache;
						}
					}

					row_height = MAX(yofs, row_height);
					offset.x += table->columns[column].width + hseparation;

					if (column == table->columns.size() - 1) {

						offset.y += row_height + vseparation;
						offset.x = hseparation;
						row_height = 0;
					}
					idx++;
				}

				int total_height = offset.y;
				if (row_height) {
					total_height = row_height + vseparation;
				}

				ADVANCE(table->total_width);
				CHECK_HEIGHT(total_height);

			} break;

			default: {
			}
		}

		Item *itp = it;

		it = _get_next_item(it);

		if (it && (p_line + 1 < p_frame->lines.size()) && p_frame->lines[p_line + 1].from == it) {

			if (p_mode == PROCESS_POINTER && r_click_item && p_click_pos.y >= p_ofs.y + y && p_click_pos.y <= p_ofs.y + y + lh) {
				//went to next line, but pointer was on the previous one
				if (r_outside) *r_outside = true;
				*r_click_item = itp;
				*r_click_char = rchar;
				RETURN;
			}

			break;
		}
	}
	NEW_LINE;

	RETURN;

#undef RETURN
#undef NEW_LINE
#undef ENSURE_WIDTH
#undef ADVANCE
#undef CHECK_HEIGHT
}

void RichTextLabel::_scroll_changed(double) {

	if (updating_scroll)
		return;

	if (scroll_follow && vscroll->get_value() >= (vscroll->get_max() - vscroll->get_page()))
		scroll_following = true;
	else
		scroll_following = false;

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
			vscroll->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_END, -scroll_w);
		} else {
			scroll_visible = false;
			scroll_w = 0;
			vscroll->hide();
		}

		main->first_invalid_line = 0; //invalidate ALL
		_validate_line_caches(main);
	}
}

void RichTextLabel::_update_fx(RichTextLabel::ItemFrame *p_frame, float p_delta_time) {
	Item *it = p_frame;
	while (it) {
		ItemFX *ifx = NULL;

		if (it->type == ITEM_CUSTOMFX || it->type == ITEM_SHAKE || it->type == ITEM_WAVE || it->type == ITEM_TORNADO || it->type == ITEM_RAINBOW) {
			ifx = static_cast<ItemFX *>(it);
		}

		if (!ifx) {
			it = _get_next_item(it, true);
			continue;
		}

		ifx->elapsed_time += p_delta_time;

		ItemShake *shake = NULL;

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
				meta_hovering = NULL;
				emit_signal("meta_hover_ended", current_meta);
				current_meta = false;
				update();
			}
		} break;
		case NOTIFICATION_RESIZED: {

			main->first_invalid_line = 0; //invalidate ALL
			update();

		} break;
		case NOTIFICATION_ENTER_TREE: {

			if (bbcode != "")
				set_bbcode(bbcode);

			main->first_invalid_line = 0; //invalidate ALL
			update();

		} break;
		case NOTIFICATION_THEME_CHANGED: {

			update();

		} break;
		case NOTIFICATION_DRAW: {

			_validate_line_caches(main);
			_update_scroll();

			RID ci = get_canvas_item();

			Size2 size = get_size();
			Rect2 text_rect = _get_text_rect();

			draw_style_box(get_stylebox("normal"), Rect2(Point2(), size));

			if (has_focus()) {
				VisualServer::get_singleton()->canvas_item_add_clip_ignore(ci, true);
				draw_style_box(get_stylebox("focus"), Rect2(Point2(), size));
				VisualServer::get_singleton()->canvas_item_add_clip_ignore(ci, false);
			}

			int ofs = vscroll->get_value();

			//todo, change to binary search

			int from_line = 0;
			int total_chars = 0;
			while (from_line < main->lines.size()) {

				if (main->lines[from_line].height_accum_cache + _get_text_rect().get_position().y >= ofs)
					break;
				total_chars += main->lines[from_line].char_count;
				from_line++;
			}

			if (from_line >= main->lines.size())
				break; //nothing to draw
			int y = (main->lines[from_line].height_accum_cache - main->lines[from_line].height_cache) - ofs;
			Ref<Font> base_font = get_font("normal_font");
			Color base_color = get_color("default_color");
			Color font_color_shadow = get_color("font_color_shadow");
			bool use_outline = get_constant("shadow_as_outline");
			Point2 shadow_ofs(get_constant("shadow_offset_x"), get_constant("shadow_offset_y"));

			visible_line_count = 0;
			while (y < size.height && from_line < main->lines.size()) {

				visible_line_count += _process_line(main, text_rect.get_position(), y, text_rect.get_size().width - scroll_w, from_line, PROCESS_DRAW, base_font, base_color, font_color_shadow, use_outline, shadow_ofs, Point2i(), NULL, NULL, NULL, total_chars);
				total_chars += main->lines[from_line].char_count;

				from_line++;
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			float dt = get_process_delta_time();

			_update_fx(main, dt);
			update();
		}
	}
}

void RichTextLabel::_find_click(ItemFrame *p_frame, const Point2i &p_click, Item **r_click_item, int *r_click_char, bool *r_outside) {

	if (r_click_item)
		*r_click_item = NULL;

	Rect2 text_rect = _get_text_rect();
	int ofs = vscroll->get_value();
	Color font_color_shadow = get_color("font_color_shadow");
	bool use_outline = get_constant("shadow_as_outline");
	Point2 shadow_ofs(get_constant("shadow_offset_x"), get_constant("shadow_offset_y"));

	//todo, change to binary search
	int from_line = 0;

	while (from_line < p_frame->lines.size()) {

		if (p_frame->lines[from_line].height_accum_cache >= ofs)
			break;
		from_line++;
	}

	if (from_line >= p_frame->lines.size())
		return;

	int y = (p_frame->lines[from_line].height_accum_cache - p_frame->lines[from_line].height_cache) - ofs;
	Ref<Font> base_font = get_font("normal_font");
	Color base_color = get_color("default_color");

	while (y < text_rect.get_size().height && from_line < p_frame->lines.size()) {

		_process_line(p_frame, text_rect.get_position(), y, text_rect.get_size().width - scroll_w, from_line, PROCESS_POINTER, base_font, base_color, font_color_shadow, use_outline, shadow_ofs, p_click, r_click_item, r_click_char, r_outside);
		if (r_click_item && *r_click_item)
			return;
		from_line++;
	}
}

Control::CursorShape RichTextLabel::get_cursor_shape(const Point2 &p_pos) const {

	if (!underline_meta)
		return CURSOR_ARROW;

	if (selection.click)
		return CURSOR_IBEAM;

	if (main->first_invalid_line < main->lines.size())
		return CURSOR_ARROW; //invalid

	int line = 0;
	Item *item = NULL;
	bool outside;
	((RichTextLabel *)(this))->_find_click(main, p_pos, &item, &line, &outside);

	if (item && !outside && ((RichTextLabel *)(this))->_find_meta(item, NULL))
		return CURSOR_POINTING_HAND;

	return CURSOR_ARROW;
}

void RichTextLabel::_gui_input(Ref<InputEvent> p_event) {

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		if (main->first_invalid_line < main->lines.size())
			return;

		if (b->get_button_index() == BUTTON_LEFT) {
			if (b->is_pressed() && !b->is_doubleclick()) {
				scroll_updated = false;
				int line = 0;
				Item *item = NULL;

				bool outside;
				_find_click(main, b->get_position(), &item, &line, &outside);

				if (item) {

					if (selection.enabled) {

						selection.click = item;
						selection.click_char = line;

						// Erase previous selection.
						if (selection.active) {
							selection.from = NULL;
							selection.from_char = '\0';
							selection.to = NULL;
							selection.to_char = '\0';
							selection.active = false;

							update();
						}
					}
				}
			} else if (b->is_pressed() && b->is_doubleclick() && selection.enabled) {

				//doubleclick: select word
				int line = 0;
				Item *item = NULL;
				bool outside;

				_find_click(main, b->get_position(), &item, &line, &outside);

				while (item && item->type != ITEM_TEXT) {

					item = _get_next_item(item, true);
				}

				if (item && item->type == ITEM_TEXT) {

					String itext = static_cast<ItemText *>(item)->text;

					int beg, end;
					if (select_word(itext, line, beg, end)) {

						selection.from = item;
						selection.to = item;
						selection.from_char = beg;
						selection.to_char = end - 1;
						selection.active = true;
						update();
					}
				}
			} else if (!b->is_pressed()) {

				selection.click = NULL;

				if (!b->is_doubleclick() && !scroll_updated) {
					int line = 0;
					Item *item = NULL;

					bool outside;
					_find_click(main, b->get_position(), &item, &line, &outside);

					if (item) {

						Variant meta;
						if (!outside && _find_meta(item, &meta)) {
							//meta clicked

							emit_signal("meta_clicked", meta);
						}
					}
				}
			}
		}

		if (b->get_button_index() == BUTTON_WHEEL_UP) {
			if (scroll_active)
				vscroll->set_value(vscroll->get_value() - vscroll->get_page() * b->get_factor() * 0.5 / 8);
		}
		if (b->get_button_index() == BUTTON_WHEEL_DOWN) {
			if (scroll_active)
				vscroll->set_value(vscroll->get_value() + vscroll->get_page() * b->get_factor() * 0.5 / 8);
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {

		if (scroll_active)

			vscroll->set_value(vscroll->get_value() + vscroll->get_page() * pan_gesture->get_delta().y * 0.5 / 8);

		return;
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (k->is_pressed() && !k->get_alt() && !k->get_shift()) {
			bool handled = false;
			switch (k->get_scancode()) {
				case KEY_PAGEUP: {

					if (vscroll->is_visible_in_tree()) {
						vscroll->set_value(vscroll->get_value() - vscroll->get_page());
						handled = true;
					}
				} break;
				case KEY_PAGEDOWN: {

					if (vscroll->is_visible_in_tree()) {
						vscroll->set_value(vscroll->get_value() + vscroll->get_page());
						handled = true;
					}
				} break;
				case KEY_UP: {

					if (vscroll->is_visible_in_tree()) {
						vscroll->set_value(vscroll->get_value() - get_font("normal_font")->get_height());
						handled = true;
					}
				} break;
				case KEY_DOWN: {

					if (vscroll->is_visible_in_tree()) {
						vscroll->set_value(vscroll->get_value() + get_font("normal_font")->get_height());
						handled = true;
					}
				} break;
				case KEY_HOME: {

					if (vscroll->is_visible_in_tree()) {
						vscroll->set_value(0);
						handled = true;
					}
				} break;
				case KEY_END: {

					if (vscroll->is_visible_in_tree()) {
						vscroll->set_value(vscroll->get_max());
						handled = true;
					}
				} break;
				case KEY_INSERT:
				case KEY_C: {

					if (k->get_command()) {
						selection_copy();
						handled = true;
					}

				} break;
			}

			if (handled)
				accept_event();
		}
	}

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {
		if (main->first_invalid_line < main->lines.size())
			return;

		int line = 0;
		Item *item = NULL;
		bool outside;
		_find_click(main, m->get_position(), &item, &line, &outside);

		if (selection.click) {

			if (!item)
				return; // do not update

			selection.from = selection.click;
			selection.from_char = selection.click_char;

			selection.to = item;
			selection.to_char = line;

			bool swap = false;
			if (selection.from->index > selection.to->index)
				swap = true;
			else if (selection.from->index == selection.to->index) {
				if (selection.from_char > selection.to_char)
					swap = true;
				else if (selection.from_char == selection.to_char) {

					selection.active = false;
					return;
				}
			}

			if (swap) {
				SWAP(selection.from, selection.to);
				SWAP(selection.from_char, selection.to_char);
			}

			selection.active = true;
			update();
		}

		Variant meta;
		ItemMeta *item_meta;
		if (item && !outside && _find_meta(item, &meta, &item_meta)) {
			if (meta_hovering != item_meta) {
				if (meta_hovering) {
					emit_signal("meta_hover_ended", current_meta);
				}
				meta_hovering = item_meta;
				current_meta = meta;
				emit_signal("meta_hover_started", meta);
			}
		} else if (meta_hovering) {
			meta_hovering = NULL;
			emit_signal("meta_hover_ended", current_meta);
			current_meta = false;
		}
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

int RichTextLabel::_find_margin(Item *p_item, const Ref<Font> &p_base_font) {

	Item *item = p_item;

	int margin = 0;

	while (item) {

		if (item->type == ITEM_INDENT) {

			Ref<Font> font = _find_font(item);
			if (font.is_null())
				font = p_base_font;

			ItemIndent *indent = static_cast<ItemIndent *>(item);

			margin += indent->level * tab_size * font->get_char_size(' ').width;

		} else if (item->type == ITEM_LIST) {

			Ref<Font> font = _find_font(item);
			if (font.is_null())
				font = p_base_font;
		}

		item = item->parent;
	}

	return margin;
}

RichTextLabel::Align RichTextLabel::_find_align(Item *p_item) {

	Item *item = p_item;

	while (item) {

		if (item->type == ITEM_ALIGN) {

			ItemAlign *align = static_cast<ItemAlign *>(item);
			return align->align;
		}

		item = item->parent;
	}

	return default_align;
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

bool RichTextLabel::_find_by_type(Item *p_item, ItemType p_type) {

	ERR_FAIL_INDEX_V((int)p_type, 19, false);

	Item *item = p_item;

	while (item) {
		if (item->type == p_type) {
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
			if (r_meta)
				*r_meta = meta->meta;
			if (r_item)
				*r_item = meta;
			return true;
		}

		item = item->parent;
	}

	return false;
}

bool RichTextLabel::_find_layout_subitem(Item *from, Item *to) {

	if (from && from != to) {
		if (from->type != ITEM_FONT && from->type != ITEM_COLOR && from->type != ITEM_UNDERLINE && from->type != ITEM_STRIKETHROUGH)
			return true;

		for (List<Item *>::Element *E = from->subitems.front(); E; E = E->next()) {
			bool layout = _find_layout_subitem(E->get(), to);

			if (layout)
				return true;
		}
	}

	return false;
}

void RichTextLabel::_validate_line_caches(ItemFrame *p_frame) {

	if (p_frame->first_invalid_line == p_frame->lines.size())
		return;

	//validate invalid lines
	Size2 size = get_size();
	if (fixed_width != -1) {
		size.width = fixed_width;
	}
	Rect2 text_rect = _get_text_rect();
	Color font_color_shadow = get_color("font_color_shadow");
	bool use_outline = get_constant("shadow_as_outline");
	Point2 shadow_ofs(get_constant("shadow_offset_x"), get_constant("shadow_offset_y"));

	Ref<Font> base_font = get_font("normal_font");

	for (int i = p_frame->first_invalid_line; i < p_frame->lines.size(); i++) {

		int y = 0;
		_process_line(p_frame, text_rect.get_position(), y, text_rect.get_size().width - scroll_w, i, PROCESS_CACHE, base_font, Color(), font_color_shadow, use_outline, shadow_ofs);
		p_frame->lines.write[i].height_cache = y;
		p_frame->lines.write[i].height_accum_cache = y;

		if (i > 0)
			p_frame->lines.write[i].height_accum_cache += p_frame->lines[i - 1].height_accum_cache;
	}

	int total_height = 0;
	if (p_frame->lines.size())
		total_height = p_frame->lines[p_frame->lines.size() - 1].height_accum_cache + get_stylebox("normal")->get_minimum_size().height;

	main->first_invalid_line = p_frame->lines.size();

	updating_scroll = true;
	vscroll->set_max(total_height);
	vscroll->set_page(size.height);
	if (scroll_follow && scroll_following)
		vscroll->set_value(total_height - size.height);

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

	if (current->type == ITEM_TABLE)
		return; //can't add anything here

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

		if (pos == 0 && end == p_text.length())
			line = p_text;
		else
			line = p_text.substr(pos, end - pos);

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
			if (item->type != ITEM_NEWLINE)
				current_frame->lines.write[current_frame->lines.size() - 1].from = item;
			_invalidate_current_line(current_frame);
		}

		pos = end + 1;
	}
}

void RichTextLabel::_add_item(Item *p_item, bool p_enter, bool p_ensure_newline) {

	p_item->parent = current;
	p_item->E = current->subitems.push_back(p_item);
	p_item->index = current_idx++;

	if (p_enter)
		current = p_item;

	if (p_ensure_newline) {
		Item *from = current_frame->lines[current_frame->lines.size() - 1].from;
		// only create a new line for Item types that generate content/layout, ignore those that represent formatting/styling
		if (_find_layout_subitem(from, p_item)) {
			_invalidate_current_line(current_frame);
			current_frame->lines.resize(current_frame->lines.size() + 1);
		}
	}

	if (current_frame->lines[current_frame->lines.size() - 1].from == NULL) {
		current_frame->lines.write[current_frame->lines.size() - 1].from = p_item;
	}
	p_item->line = current_frame->lines.size() - 1;

	_invalidate_current_line(current_frame);
}

void RichTextLabel::_remove_item(Item *p_item, const int p_line, const int p_subitem_line) {

	int size = p_item->subitems.size();
	if (size == 0) {
		p_item->parent->subitems.erase(p_item);
		if (p_item->type == ITEM_NEWLINE) {
			current_frame->lines.remove(p_line);
			for (int i = p_subitem_line; i < current->subitems.size(); i++) {
				if (current->subitems[i]->line > 0)
					current->subitems[i]->line--;
			}
		}
	} else {
		for (int i = 0; i < size; i++) {
			_remove_item(p_item->subitems.front()->get(), p_line, p_subitem_line);
		}
	}
}

void RichTextLabel::add_image(const Ref<Texture> &p_image, const int p_width, const int p_height) {

	if (current->type == ITEM_TABLE)
		return;

	ERR_FAIL_COND(p_image.is_null());
	ERR_FAIL_COND(p_image->get_width() == 0);
	ERR_FAIL_COND(p_image->get_height() == 0);
	ItemImage *item = memnew(ItemImage);

	item->image = p_image;

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
			item->size.height = p_image->get_height();
			item->size.width = p_image->get_width();
		}
	}

	_add_item(item, false);
}

void RichTextLabel::add_newline() {

	if (current->type == ITEM_TABLE)
		return;
	ItemNewline *item = memnew(ItemNewline);
	item->line = current_frame->lines.size();
	_add_item(item, false);
	current_frame->lines.resize(current_frame->lines.size() + 1);
	_invalidate_current_line(current_frame);
}

bool RichTextLabel::remove_line(const int p_line) {

	if (p_line >= current_frame->lines.size() || p_line < 0)
		return false;

	int i = 0;
	while (i < current->subitems.size() && current->subitems[i]->line < p_line) {
		i++;
	}

	bool was_newline = false;
	while (i < current->subitems.size()) {
		was_newline = current->subitems[i]->type == ITEM_NEWLINE;
		_remove_item(current->subitems[i], current->subitems[i]->line, p_line);
		if (was_newline)
			break;
	}

	if (!was_newline) {
		current_frame->lines.remove(p_line);
		if (current_frame->lines.size() == 0) {
			current_frame->lines.resize(1);
		}
	}

	if (p_line == 0 && current->subitems.size() > 0)
		main->lines.write[0].from = main;

	main->first_invalid_line = 0;

	return true;
}

void RichTextLabel::push_font(const Ref<Font> &p_font) {

	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ERR_FAIL_COND(p_font.is_null());
	ItemFont *item = memnew(ItemFont);

	item->font = p_font;
	_add_item(item, true);
}

void RichTextLabel::push_normal() {
	Ref<Font> normal_font = get_font("normal_font");
	ERR_FAIL_COND(normal_font.is_null());

	push_font(normal_font);
}

void RichTextLabel::push_bold() {
	Ref<Font> bold_font = get_font("bold_font");
	ERR_FAIL_COND(bold_font.is_null());

	push_font(bold_font);
}

void RichTextLabel::push_bold_italics() {
	Ref<Font> bold_italics_font = get_font("bold_italics_font");
	ERR_FAIL_COND(bold_italics_font.is_null());

	push_font(bold_italics_font);
}

void RichTextLabel::push_italics() {
	Ref<Font> italics_font = get_font("italics_font");
	ERR_FAIL_COND(italics_font.is_null());

	push_font(italics_font);
}

void RichTextLabel::push_mono() {
	Ref<Font> mono_font = get_font("mono_font");
	ERR_FAIL_COND(mono_font.is_null());

	push_font(mono_font);
}

void RichTextLabel::push_color(const Color &p_color) {

	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemColor *item = memnew(ItemColor);

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

void RichTextLabel::push_align(Align p_align) {

	ERR_FAIL_COND(current->type == ITEM_TABLE);

	ItemAlign *item = memnew(ItemAlign);
	item->align = p_align;
	_add_item(item, true, true);
}

void RichTextLabel::push_indent(int p_level) {

	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ERR_FAIL_COND(p_level < 0);

	ItemIndent *item = memnew(ItemIndent);
	item->level = p_level;
	_add_item(item, true, true);
}

void RichTextLabel::push_list(ListType p_list) {

	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ERR_FAIL_INDEX(p_list, 3);

	ItemList *item = memnew(ItemList);

	item->list_type = p_list;
	_add_item(item, true, true);
}

void RichTextLabel::push_meta(const Variant &p_meta) {

	ERR_FAIL_COND(current->type == ITEM_TABLE);
	ItemMeta *item = memnew(ItemMeta);

	item->meta = p_meta;
	_add_item(item, true);
}

void RichTextLabel::push_table(int p_columns) {

	ERR_FAIL_COND(p_columns < 1);
	ItemTable *item = memnew(ItemTable);

	item->columns.resize(p_columns);
	item->total_width = 0;
	for (int i = 0; i < item->columns.size(); i++) {
		item->columns.write[i].expand = false;
		item->columns.write[i].expand_ratio = 1;
	}
	_add_item(item, true, true);
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

void RichTextLabel::push_cell() {

	ERR_FAIL_COND(current->type != ITEM_TABLE);

	ItemFrame *item = memnew(ItemFrame);
	item->parent_frame = current_frame;
	_add_item(item, true);
	current_frame = item;
	item->cell = true;
	item->parent_line = item->parent_frame->lines.size() - 1;
	item->lines.resize(1);
	item->lines.write[0].from = NULL;
	item->first_invalid_line = 0;
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
	selection.click = NULL;
	selection.active = false;
	current_idx = 1;
	if (scroll_follow) {
		scroll_following = true;
	}
}

void RichTextLabel::set_tab_size(int p_spaces) {

	tab_size = p_spaces;
	main->first_invalid_line = 0;
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

	if (scroll_active == p_active)
		return;

	scroll_active = p_active;
	vscroll->set_drag_node_enabled(p_active);
	update();
}

bool RichTextLabel::is_scroll_active() const {

	return scroll_active;
}

void RichTextLabel::set_scroll_follow(bool p_follow) {

	scroll_follow = p_follow;
	if (!vscroll->is_visible_in_tree() || vscroll->get_value() >= (vscroll->get_max() - vscroll->get_page()))
		scroll_following = true;
}

bool RichTextLabel::is_scroll_following() const {

	return scroll_follow;
}

Error RichTextLabel::parse_bbcode(const String &p_bbcode) {

	clear();
	return append_bbcode(p_bbcode);
}

Error RichTextLabel::append_bbcode(const String &p_bbcode) {

	int pos = 0;

	List<String> tag_stack;
	Ref<Font> normal_font = get_font("normal_font");
	Ref<Font> bold_font = get_font("bold_font");
	Ref<Font> italics_font = get_font("italics_font");
	Ref<Font> bold_italics_font = get_font("bold_italics_font");
	Ref<Font> mono_font = get_font("mono_font");

	Color base_color = get_color("default_color");

	int indent_level = 0;

	bool in_bold = false;
	bool in_italics = false;

	set_process_internal(false);

	while (pos < p_bbcode.length()) {

		int brk_pos = p_bbcode.find("[", pos);

		if (brk_pos < 0)
			brk_pos = p_bbcode.length();

		if (brk_pos > pos) {
			add_text(p_bbcode.substr(pos, brk_pos - pos));
		}

		if (brk_pos == p_bbcode.length())
			break; //nothing else to add

		int brk_end = p_bbcode.find("]", brk_pos + 1);

		if (brk_end == -1) {
			//no close, add the rest
			add_text(p_bbcode.substr(brk_pos, p_bbcode.length() - brk_pos));
			break;
		}

		String tag = p_bbcode.substr(brk_pos + 1, brk_end - brk_pos - 1);
		Vector<String> split_tag_block = tag.split(" ", false);
		String bbcode = !split_tag_block.empty() ? split_tag_block[0] : "";
		if (tag.begins_with("/") && tag_stack.size()) {

			bool tag_ok = tag_stack.size() && tag_stack.front()->get() == tag.substr(1, tag.length());

			if (tag_stack.front()->get() == "b")
				in_bold = false;
			if (tag_stack.front()->get() == "i")
				in_italics = false;
			if (tag_stack.front()->get() == "indent")
				indent_level--;

			if (!tag_ok) {
				add_text("[" + tag);
				pos = brk_end;
				continue;
			}

			tag_stack.pop_front();
			pos = brk_end + 1;
			if (tag != "/img")
				pop();

		} else if (tag == "b") {

			//use bold font
			in_bold = true;
			if (in_italics)
				push_font(bold_italics_font);
			else
				push_font(bold_font);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "i") {

			//use italics font
			in_italics = true;
			if (in_bold)
				push_font(bold_italics_font);
			else
				push_font(italics_font);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "code") {

			//use monospace font
			push_font(mono_font);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("table=")) {

			int columns = tag.substr(6, tag.length()).to_int();
			if (columns < 1)
				columns = 1;

			push_table(columns);
			pos = brk_end + 1;
			tag_stack.push_front("table");
		} else if (tag == "cell") {

			push_cell();
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("cell=")) {

			int ratio = tag.substr(5, tag.length()).to_int();
			if (ratio < 1)
				ratio = 1;

			set_table_column_expand(get_current_table_column(), true, ratio);
			push_cell();
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
		} else if (tag == "center") {

			push_align(ALIGN_CENTER);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "fill") {

			push_align(ALIGN_FILL);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "right") {

			push_align(ALIGN_RIGHT);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "ul") {

			push_list(LIST_DOTS);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "ol") {

			push_list(LIST_NUMBERS);
			pos = brk_end + 1;
			tag_stack.push_front(tag);
		} else if (tag == "indent") {

			indent_level++;
			push_indent(indent_level);
			pos = brk_end + 1;
			tag_stack.push_front(tag);

		} else if (tag == "url") {

			int end = p_bbcode.find("[", brk_end);
			if (end == -1)
				end = p_bbcode.length();
			String url = p_bbcode.substr(brk_end + 1, end - brk_end - 1);
			push_meta(url);

			pos = brk_end + 1;
			tag_stack.push_front(tag);

		} else if (tag.begins_with("url=")) {

			String url = tag.substr(4, tag.length());
			push_meta(url);
			pos = brk_end + 1;
			tag_stack.push_front("url");
		} else if (tag == "img") {

			int end = p_bbcode.find("[", brk_end);
			if (end == -1)
				end = p_bbcode.length();

			String image = p_bbcode.substr(brk_end + 1, end - brk_end - 1);

			Ref<Texture> texture = ResourceLoader::load(image, "Texture");
			if (texture.is_valid())
				add_image(texture);

			pos = end;
			tag_stack.push_front(tag);
		} else if (tag.begins_with("img=")) {

			int width = 0;
			int height = 0;

			String params = tag.substr(4, tag.length());
			int sep = params.find("x");
			if (sep == -1) {
				width = params.to_int();
			} else {
				width = params.substr(0, sep).to_int();
				height = params.substr(sep + 1, params.length()).to_int();
			}

			int end = p_bbcode.find("[", brk_end);
			if (end == -1)
				end = p_bbcode.length();

			String image = p_bbcode.substr(brk_end + 1, end - brk_end - 1);

			Ref<Texture> texture = ResourceLoader::load(image, "Texture");
			if (texture.is_valid())
				add_image(texture, width, height);

			pos = end;
			tag_stack.push_front("img");
		} else if (tag.begins_with("color=")) {

			String col = tag.substr(6, tag.length());
			Color color;

			if (col.begins_with("#"))
				color = Color::html(col);
			else if (col == "aqua")
				color = Color(0, 1, 1);
			else if (col == "black")
				color = Color(0, 0, 0);
			else if (col == "blue")
				color = Color(0, 0, 1);
			else if (col == "fuchsia")
				color = Color(1, 0, 1);
			else if (col == "gray" || col == "grey")
				color = Color(0.5, 0.5, 0.5);
			else if (col == "green")
				color = Color(0, 0.5, 0);
			else if (col == "lime")
				color = Color(0, 1, 0);
			else if (col == "maroon")
				color = Color(0.5, 0, 0);
			else if (col == "navy")
				color = Color(0, 0, 0.5);
			else if (col == "olive")
				color = Color(0.5, 0.5, 0);
			else if (col == "purple")
				color = Color(0.5, 0, 0.5);
			else if (col == "red")
				color = Color(1, 0, 0);
			else if (col == "silver")
				color = Color(0.75, 0.75, 0.75);
			else if (col == "teal")
				color = Color(0, 0.5, 0.5);
			else if (col == "white")
				color = Color(1, 1, 1);
			else if (col == "yellow")
				color = Color(1, 1, 0);
			else
				color = base_color;

			push_color(color);
			pos = brk_end + 1;
			tag_stack.push_front("color");

		} else if (tag.begins_with("font=")) {

			String fnt = tag.substr(5, tag.length());

			Ref<Font> font = ResourceLoader::load(fnt, "Font");
			if (font.is_valid())
				push_font(font);
			else
				push_font(normal_font);

			pos = brk_end + 1;
			tag_stack.push_front("font");

		} else if (bbcode == "fade") {
			int startIndex = 0;
			int length = 10;

			if (split_tag_block.size() > 1) {
				split_tag_block.remove(0);
				for (int i = 0; i < split_tag_block.size(); i++) {
					String expr = split_tag_block[i];
					if (expr.begins_with("start=")) {
						String start_str = expr.substr(6, expr.length());
						startIndex = start_str.to_int();
					} else if (expr.begins_with("length=")) {
						String end_str = expr.substr(7, expr.length());
						length = end_str.to_int();
					}
				}
			}

			push_fade(startIndex, length);
			pos = brk_end + 1;
			tag_stack.push_front("fade");
		} else if (bbcode == "shake") {
			int strength = 5;
			float rate = 20.0f;

			if (split_tag_block.size() > 1) {
				split_tag_block.remove(0);
				for (int i = 0; i < split_tag_block.size(); i++) {
					String expr = split_tag_block[i];
					if (expr.begins_with("level=")) {
						String str_str = expr.substr(6, expr.length());
						strength = str_str.to_int();
					} else if (expr.begins_with("rate=")) {
						String rate_str = expr.substr(5, expr.length());
						rate = rate_str.to_float();
					}
				}
			}

			push_shake(strength, rate);
			pos = brk_end + 1;
			tag_stack.push_front("shake");
			set_process_internal(true);
		} else if (bbcode == "wave") {
			float amplitude = 20.0f;
			float period = 5.0f;

			if (split_tag_block.size() > 1) {
				split_tag_block.remove(0);
				for (int i = 0; i < split_tag_block.size(); i++) {
					String expr = split_tag_block[i];
					if (expr.begins_with("amp=")) {
						String amp_str = expr.substr(4, expr.length());
						amplitude = amp_str.to_float();
					} else if (expr.begins_with("freq=")) {
						String period_str = expr.substr(5, expr.length());
						period = period_str.to_float();
					}
				}
			}

			push_wave(period, amplitude);
			pos = brk_end + 1;
			tag_stack.push_front("wave");
			set_process_internal(true);
		} else if (bbcode == "tornado") {
			float radius = 10.0f;
			float frequency = 1.0f;

			if (split_tag_block.size() > 1) {
				split_tag_block.remove(0);
				for (int i = 0; i < split_tag_block.size(); i++) {
					String expr = split_tag_block[i];
					if (expr.begins_with("radius=")) {
						String amp_str = expr.substr(7, expr.length());
						radius = amp_str.to_float();
					} else if (expr.begins_with("freq=")) {
						String period_str = expr.substr(5, expr.length());
						frequency = period_str.to_float();
					}
				}
			}

			push_tornado(frequency, radius);
			pos = brk_end + 1;
			tag_stack.push_front("tornado");
			set_process_internal(true);
		} else if (bbcode == "rainbow") {
			float saturation = 0.8f;
			float value = 0.8f;
			float frequency = 1.0f;

			if (split_tag_block.size() > 1) {
				split_tag_block.remove(0);
				for (int i = 0; i < split_tag_block.size(); i++) {
					String expr = split_tag_block[i];
					if (expr.begins_with("sat=")) {
						String sat_str = expr.substr(4, expr.length());
						saturation = sat_str.to_float();
					} else if (expr.begins_with("val=")) {
						String val_str = expr.substr(4, expr.length());
						value = val_str.to_float();
					} else if (expr.begins_with("freq=")) {
						String freq_str = expr.substr(5, expr.length());
						frequency = freq_str.to_float();
					}
				}
			}

			push_rainbow(saturation, value, frequency);
			pos = brk_end + 1;
			tag_stack.push_front("rainbow");
			set_process_internal(true);
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
	for (List<Item *>::Element *E = main->subitems.front(); E; E = E->next()) {
		Item *subitem = static_cast<Item *>(E->get());
		_fetch_item_fx_stack(subitem, fx_items);

		if (fx_items.size()) {
			set_process_internal(true);
			break;
		}
	}

	return OK;
}

void RichTextLabel::scroll_to_line(int p_line) {

	ERR_FAIL_INDEX(p_line, main->lines.size());
	_validate_line_caches(main);
	vscroll->set_value(main->lines[p_line].height_accum_cache - main->lines[p_line].height_cache);
}

int RichTextLabel::get_line_count() const {

	return current_frame->lines.size();
}

int RichTextLabel::get_visible_line_count() const {
	if (!is_visible())
		return 0;
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

bool RichTextLabel::search(const String &p_string, bool p_from_selection, bool p_search_previous) {

	ERR_FAIL_COND_V(!selection.enabled, false);
	Item *it = main;
	int charidx = 0;

	if (p_from_selection && selection.active) {
		it = selection.to;
		charidx = selection.to_char + 1;
	}

	while (it) {

		if (it->type == ITEM_TEXT) {

			ItemText *t = static_cast<ItemText *>(it);
			int sp = t->text.findn(p_string, charidx);
			if (sp != -1) {
				selection.from = it;
				selection.from_char = sp;
				selection.to = it;
				selection.to_char = sp + p_string.length() - 1;
				selection.active = true;
				update();

				_validate_line_caches(main);

				int fh = _find_font(t).is_valid() ? _find_font(t)->get_height() : get_font("normal_font")->get_height();

				float offset = 0;

				int line = t->line;
				Item *item = t;
				while (item) {
					if (item->type == ITEM_FRAME) {
						ItemFrame *frame = static_cast<ItemFrame *>(item);
						if (line >= 0 && line < frame->lines.size()) {
							offset += frame->lines[line].height_accum_cache - frame->lines[line].height_cache;
							line = frame->line;
						}
					}
					item = item->parent;
				}
				vscroll->set_value(offset - fh);

				return true;
			}
		}

		if (p_search_previous)
			it = _get_prev_item(it, true);
		else
			it = _get_next_item(it, true);
		charidx = 0;
	}

	return false;
}

String RichTextLabel::get_selected_text() {
	if (!selection.active || !selection.enabled) {
		return "";
	}

	String text;

	RichTextLabel::Item *item = selection.from;

	while (item) {

		if (item->type == ITEM_TEXT) {

			String itext = static_cast<ItemText *>(item)->text;
			if (item == selection.from && item == selection.to) {
				text += itext.substr(selection.from_char, selection.to_char - selection.from_char + 1);
			} else if (item == selection.from) {
				text += itext.substr(selection.from_char, itext.size());
			} else if (item == selection.to) {
				text += itext.substr(0, selection.to_char + 1);
			} else {
				text += itext;
			}

		} else if (item->type == ITEM_NEWLINE) {
			text += "\n";
		}
		if (item == selection.to)
			break;

		item = _get_next_item(item, true);
	}

	return text;
}

void RichTextLabel::selection_copy() {
	String text = get_selected_text();

	if (text != "") {
		OS::get_singleton()->set_clipboard(text);
	}
}

bool RichTextLabel::is_selection_enabled() const {

	return selection.enabled;
}

void RichTextLabel::set_bbcode(const String &p_bbcode) {
	bbcode = p_bbcode;
	if (is_inside_tree() && use_bbcode)
		parse_bbcode(p_bbcode);
	else { // raw text
		clear();
		add_text(p_bbcode);
	}
}

String RichTextLabel::get_bbcode() const {

	return bbcode;
}

void RichTextLabel::set_use_bbcode(bool p_enable) {
	if (use_bbcode == p_enable)
		return;
	use_bbcode = p_enable;
	set_bbcode(bbcode);
}

bool RichTextLabel::is_using_bbcode() const {

	return use_bbcode;
}

String RichTextLabel::get_text() {
	String text = "";
	Item *it = main;
	while (it) {
		if (it->type == ITEM_TEXT) {
			ItemText *t = static_cast<ItemText *>(it);
			text += t->text;
		} else if (it->type == ITEM_NEWLINE) {
			text += "\n";
		} else if (it->type == ITEM_INDENT) {
			text += "\t";
		}
		it = _get_next_item(it, true);
	}
	return text;
}

void RichTextLabel::set_text(const String &p_string) {
	clear();
	add_text(p_string);
}

void RichTextLabel::set_percent_visible(float p_percent) {

	if (p_percent < 0 || p_percent >= 1) {

		visible_characters = -1;
		percent_visible = 1;

	} else {

		visible_characters = get_total_character_count() * p_percent;
		percent_visible = p_percent;
	}
	_change_notify("visible_characters");
	update();
}

float RichTextLabel::get_percent_visible() const {
	return percent_visible;
}

void RichTextLabel::set_effects(const Vector<Variant> &effects) {
	custom_effects.clear();
	for (int i = 0; i < effects.size(); i++) {
		Ref<RichTextEffect> effect = Ref<RichTextEffect>(effects[i]);
		custom_effects.push_back(effect);
	}

	parse_bbcode(bbcode);
}

Vector<Variant> RichTextLabel::get_effects() {
	Vector<Variant> r;
	for (int i = 0; i < custom_effects.size(); i++) {
		r.push_back(custom_effects[i].get_ref_ptr());
	}
	return r;
}

void RichTextLabel::install_effect(const Variant effect) {
	Ref<RichTextEffect> rteffect;
	rteffect = effect;

	if (rteffect.is_valid()) {
		custom_effects.push_back(effect);
		parse_bbcode(bbcode);
	}
}

int RichTextLabel::get_content_height() const {
	int total_height = 0;
	if (main->lines.size())
		total_height = main->lines[main->lines.size() - 1].height_accum_cache + get_stylebox("normal")->get_minimum_size().height;
	return total_height;
}

void RichTextLabel::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &RichTextLabel::_gui_input);
	ClassDB::bind_method(D_METHOD("_scroll_changed"), &RichTextLabel::_scroll_changed);
	ClassDB::bind_method(D_METHOD("get_text"), &RichTextLabel::get_text);
	ClassDB::bind_method(D_METHOD("add_text", "text"), &RichTextLabel::add_text);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &RichTextLabel::set_text);
	ClassDB::bind_method(D_METHOD("add_image", "image", "width", "height"), &RichTextLabel::add_image, DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("newline"), &RichTextLabel::add_newline);
	ClassDB::bind_method(D_METHOD("remove_line", "line"), &RichTextLabel::remove_line);
	ClassDB::bind_method(D_METHOD("push_font", "font"), &RichTextLabel::push_font);
	ClassDB::bind_method(D_METHOD("push_normal"), &RichTextLabel::push_normal);
	ClassDB::bind_method(D_METHOD("push_bold"), &RichTextLabel::push_bold);
	ClassDB::bind_method(D_METHOD("push_bold_italics"), &RichTextLabel::push_bold_italics);
	ClassDB::bind_method(D_METHOD("push_italics"), &RichTextLabel::push_italics);
	ClassDB::bind_method(D_METHOD("push_mono"), &RichTextLabel::push_mono);
	ClassDB::bind_method(D_METHOD("push_color", "color"), &RichTextLabel::push_color);
	ClassDB::bind_method(D_METHOD("push_align", "align"), &RichTextLabel::push_align);
	ClassDB::bind_method(D_METHOD("push_indent", "level"), &RichTextLabel::push_indent);
	ClassDB::bind_method(D_METHOD("push_list", "type"), &RichTextLabel::push_list);
	ClassDB::bind_method(D_METHOD("push_meta", "data"), &RichTextLabel::push_meta);
	ClassDB::bind_method(D_METHOD("push_underline"), &RichTextLabel::push_underline);
	ClassDB::bind_method(D_METHOD("push_strikethrough"), &RichTextLabel::push_strikethrough);
	ClassDB::bind_method(D_METHOD("push_table", "columns"), &RichTextLabel::push_table);
	ClassDB::bind_method(D_METHOD("set_table_column_expand", "column", "expand", "ratio"), &RichTextLabel::set_table_column_expand);
	ClassDB::bind_method(D_METHOD("push_cell"), &RichTextLabel::push_cell);
	ClassDB::bind_method(D_METHOD("pop"), &RichTextLabel::pop);

	ClassDB::bind_method(D_METHOD("clear"), &RichTextLabel::clear);

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

	ClassDB::bind_method(D_METHOD("set_tab_size", "spaces"), &RichTextLabel::set_tab_size);
	ClassDB::bind_method(D_METHOD("get_tab_size"), &RichTextLabel::get_tab_size);

	ClassDB::bind_method(D_METHOD("set_fit_content_height", "enabled"), &RichTextLabel::set_fit_content_height);
	ClassDB::bind_method(D_METHOD("is_fit_content_height_enabled"), &RichTextLabel::is_fit_content_height_enabled);

	ClassDB::bind_method(D_METHOD("set_selection_enabled", "enabled"), &RichTextLabel::set_selection_enabled);
	ClassDB::bind_method(D_METHOD("is_selection_enabled"), &RichTextLabel::is_selection_enabled);

	ClassDB::bind_method(D_METHOD("parse_bbcode", "bbcode"), &RichTextLabel::parse_bbcode);
	ClassDB::bind_method(D_METHOD("append_bbcode", "bbcode"), &RichTextLabel::append_bbcode);

	ClassDB::bind_method(D_METHOD("set_bbcode", "text"), &RichTextLabel::set_bbcode);
	ClassDB::bind_method(D_METHOD("get_bbcode"), &RichTextLabel::get_bbcode);

	ClassDB::bind_method(D_METHOD("set_visible_characters", "amount"), &RichTextLabel::set_visible_characters);
	ClassDB::bind_method(D_METHOD("get_visible_characters"), &RichTextLabel::get_visible_characters);

	ClassDB::bind_method(D_METHOD("set_percent_visible", "percent_visible"), &RichTextLabel::set_percent_visible);
	ClassDB::bind_method(D_METHOD("get_percent_visible"), &RichTextLabel::get_percent_visible);

	ClassDB::bind_method(D_METHOD("get_total_character_count"), &RichTextLabel::get_total_character_count);

	ClassDB::bind_method(D_METHOD("set_use_bbcode", "enable"), &RichTextLabel::set_use_bbcode);
	ClassDB::bind_method(D_METHOD("is_using_bbcode"), &RichTextLabel::is_using_bbcode);

	ClassDB::bind_method(D_METHOD("get_line_count"), &RichTextLabel::get_line_count);
	ClassDB::bind_method(D_METHOD("get_visible_line_count"), &RichTextLabel::get_visible_line_count);

	ClassDB::bind_method(D_METHOD("get_content_height"), &RichTextLabel::get_content_height);

	ClassDB::bind_method(D_METHOD("parse_expressions_for_values", "expressions"), &RichTextLabel::parse_expressions_for_values);

	ClassDB::bind_method(D_METHOD("set_effects", "effects"), &RichTextLabel::set_effects);
	ClassDB::bind_method(D_METHOD("get_effects"), &RichTextLabel::get_effects);
	ClassDB::bind_method(D_METHOD("install_effect", "effect"), &RichTextLabel::install_effect);

	ADD_GROUP("BBCode", "bbcode_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bbcode_enabled"), "set_use_bbcode", "is_using_bbcode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bbcode_text", PROPERTY_HINT_MULTILINE_TEXT), "set_bbcode", "get_bbcode");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_characters", PROPERTY_HINT_RANGE, "-1,128000,1"), "set_visible_characters", "get_visible_characters");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "percent_visible", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_percent_visible", "get_percent_visible");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "meta_underlined"), "set_meta_underline", "is_meta_underlined");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_size", PROPERTY_HINT_RANGE, "0,24,1"), "set_tab_size", "get_tab_size");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT), "set_text", "get_text");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fit_content_height"), "set_fit_content_height", "is_fit_content_height_enabled");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_active"), "set_scroll_active", "is_scroll_active");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_following"), "set_scroll_follow", "is_scroll_following");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "selection_enabled"), "set_selection_enabled", "is_selection_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_selected_font_color"), "set_override_selected_font_color", "is_overriding_selected_font_color");

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "custom_effects", PROPERTY_HINT_RESOURCE_TYPE, "17/17:RichTextEffect", (PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SCRIPT_VARIABLE), "RichTextEffect"), "set_effects", "get_effects");

	ADD_SIGNAL(MethodInfo("meta_clicked", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("meta_hover_started", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("meta_hover_ended", PropertyInfo(Variant::NIL, "meta", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
	BIND_ENUM_CONSTANT(ALIGN_FILL);

	BIND_ENUM_CONSTANT(LIST_NUMBERS);
	BIND_ENUM_CONSTANT(LIST_LETTERS);
	BIND_ENUM_CONSTANT(LIST_DOTS);

	BIND_ENUM_CONSTANT(ITEM_FRAME);
	BIND_ENUM_CONSTANT(ITEM_TEXT);
	BIND_ENUM_CONSTANT(ITEM_IMAGE);
	BIND_ENUM_CONSTANT(ITEM_NEWLINE);
	BIND_ENUM_CONSTANT(ITEM_FONT);
	BIND_ENUM_CONSTANT(ITEM_COLOR);
	BIND_ENUM_CONSTANT(ITEM_UNDERLINE);
	BIND_ENUM_CONSTANT(ITEM_STRIKETHROUGH);
	BIND_ENUM_CONSTANT(ITEM_ALIGN);
	BIND_ENUM_CONSTANT(ITEM_INDENT);
	BIND_ENUM_CONSTANT(ITEM_LIST);
	BIND_ENUM_CONSTANT(ITEM_TABLE);
	BIND_ENUM_CONSTANT(ITEM_FADE);
	BIND_ENUM_CONSTANT(ITEM_SHAKE);
	BIND_ENUM_CONSTANT(ITEM_WAVE);
	BIND_ENUM_CONSTANT(ITEM_TORNADO);
	BIND_ENUM_CONSTANT(ITEM_RAINBOW);
	BIND_ENUM_CONSTANT(ITEM_CUSTOMFX);
	BIND_ENUM_CONSTANT(ITEM_META);
}

void RichTextLabel::set_visible_characters(int p_visible) {
	visible_characters = p_visible;
	if (p_visible == -1) {
		percent_visible = 1;
	} else {
		int total_char_count = get_total_character_count();
		if (total_char_count > 0) {
			percent_visible = (float)p_visible / (float)total_char_count;
		}
	}
	_change_notify("percent_visible");
	update();
}

int RichTextLabel::get_visible_characters() const {
	return visible_characters;
}
int RichTextLabel::get_total_character_count() const {

	int tc = 0;
	for (int i = 0; i < current_frame->lines.size(); i++)
		tc += current_frame->lines[i].char_count;

	return tc;
}

void RichTextLabel::set_fixed_size_to_width(int p_width) {
	fixed_width = p_width;
	minimum_size_changed();
}

Size2 RichTextLabel::get_minimum_size() const {

	Size2 size(0, 0);
	if (fixed_width != -1) {
		size.x = fixed_width;
	}

	if (fixed_width != -1 || fit_content_height) {
		const_cast<RichTextLabel *>(this)->_validate_line_caches(main);
		size.y = get_content_height();
	}

	return size;
}

Ref<RichTextEffect> RichTextLabel::_get_custom_effect_by_code(String p_bbcode_identifier) {
	for (int i = 0; i < custom_effects.size(); i++) {
		if (!custom_effects[i].is_valid())
			continue;

		if (custom_effects[i]->get_bbcode() == p_bbcode_identifier) {
			return custom_effects[i];
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

		RegEx color;
		color.compile("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$");
		RegEx nodepath;
		nodepath.compile("^\\$");
		RegEx boolean;
		boolean.compile("^(true|false)$");
		RegEx decimal;
		decimal.compile("^-?^.?\\d+(\\.\\d+?)?$");
		RegEx numerical;
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
				a.append(values[j].to_double());
			} else if (!numerical.search(values[j]).is_null()) {
				a.append(values[j].to_int());
			} else {
				a.append(values[j]);
			}
		}

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
	current_frame = main;
	tab_size = 4;
	default_align = ALIGN_LEFT;
	underline_meta = true;
	meta_hovering = NULL;
	override_selected_font_color = false;

	scroll_visible = false;
	scroll_follow = false;
	scroll_following = false;
	updating_scroll = false;
	scroll_active = true;
	scroll_w = 0;
	scroll_updated = false;

	vscroll = memnew(VScrollBar);
	add_child(vscroll);
	vscroll->set_drag_node(String(".."));
	vscroll->set_step(1);
	vscroll->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 0);
	vscroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);
	vscroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
	vscroll->connect("value_changed", this, "_scroll_changed");
	vscroll->set_step(1);
	vscroll->hide();
	current_idx = 1;
	use_bbcode = false;

	selection.click = NULL;
	selection.active = false;
	selection.enabled = false;

	visible_characters = -1;
	percent_visible = 1;
	visible_line_count = 0;

	fixed_width = -1;
	fit_content_height = false;

	set_clip_contents(true);
}

RichTextLabel::~RichTextLabel() {
	memdelete(main);
}
