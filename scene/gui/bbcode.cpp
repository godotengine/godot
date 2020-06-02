
#include "scene/gui/bbcode.h"
#include "core/math/math_defs.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "scene/scene_string_names.h"
#include "servers/display_server.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_scale.h"
#endif

BbCodeParser::BbCodeParser() {}

BbCodeParser::~BbCodeParser() {}

bool BbCodeParser::_new_line() {
	if (p_mode != ProcessMode::PROCESS_CACHE) {
		line++;
		backtrack = 0;
		if (!line_is_blank) {
			nonblank_line_count++;
		}
		line_is_blank = true;
		if (line < l.offset_caches.size())
			line_ofs = l.offset_caches[line];
		wofs = margin;
		if (align != Align::ALIGN_FILL)
			wofs += line_ofs;
	} else {
		int used = wofs - margin;
		switch (align) {
			case Align::ALIGN_LEFT:
				l.offset_caches.push_back(0);
				break;
			case Align::ALIGN_CENTER:
				l.offset_caches.push_back(((p_width - margin) - used) / 2);
				break;
			case Align::ALIGN_RIGHT:
				l.offset_caches.push_back(((p_width - margin) - used));
				break;
			case Align::ALIGN_FILL:
				l.offset_caches.push_back(line_wrapped ? ((p_width - margin) - used) : 0);
				break;
		}
		l.height_caches.push_back(line_height);
		l.ascent_caches.push_back(line_ascent);
		l.descent_caches.push_back(line_descent);
		l.space_caches.push_back(spaces);
	}
	line_wrapped = false;
	p_height += line_height + get_theme_constant(SceneStringNames::get_singleton()->line_separation);
	line_height = 0;
	line_ascent = 0;
	line_descent = 0;
	spaces = 0;
	spaces_size = 0;
	wofs = begin;
	align_ofs = 0;
	if (p_mode != ProcessMode::PROCESS_CACHE) {
		lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
		line_ascent = line < l.ascent_caches.size() ? l.ascent_caches[line] : 1;
		line_descent = line < l.descent_caches.size() ? l.descent_caches[line] : 1;
	}
	if (p_mode == ProcessMode::PROCESS_POINTER && r_click_item && p_click_pos.y >= p_ofs.y + p_height && p_click_pos.y <= p_ofs.y + p_height + lh && p_click_pos.x < p_ofs.x + wofs) {
		if (r_outside)
			*r_outside = true;
		*r_click_item = it;
		*r_click_char = rchar;
		return true;
	}
	return false;
}

bool BbCodeParser::_ensure_width(int m_width) {
	if (p_mode == ProcessMode::PROCESS_CACHE) {
		l.maximum_width = MAX(l.maximum_width, MIN(p_width, wofs + m_width));
		l.minimum_width = MAX(l.minimum_width, m_width);
	}
	if (wofs - backtrack + m_width > p_width) {
		line_wrapped = true;
		if (p_mode == ProcessMode::PROCESS_CACHE) {
			if (spaces > 0)
				spaces -= 1;
		}
		const bool x_in_range = (p_click_pos.x > p_ofs.x + wofs) && (!p_frame->cell || p_click_pos.x < p_ofs.x + p_width);
		if (p_mode == ProcessMode::PROCESS_POINTER && r_click_item && p_click_pos.y >= p_ofs.y + p_height && p_click_pos.y <= p_ofs.y + p_height + lh && x_in_range) {
			if (r_outside)
				*r_outside = true;
			*r_click_item = it;
			*r_click_char = rchar;
			return true;
		}
		return _new_line();
	}
	return false;
}

bool BbCodeParser::_advance(int m_width) {
	if (p_mode == ProcessMode::PROCESS_POINTER && r_click_item && p_click_pos.y >= p_ofs.y + p_height && p_click_pos.y <= p_ofs.y + p_height + lh && p_click_pos.x >= p_ofs.x + wofs && p_click_pos.x < p_ofs.x + wofs + m_width) {
		if (r_outside)
			*r_outside = false;
		*r_click_item = it;
		*r_click_char = rchar;
		return true;
	}
	wofs += m_width;
	return false;
}

void BbCodeParser::_check_height(int m_height) {
	if (m_height > line_height) {
		line_height = m_height;
	}
}

bool BbCodeParser::_y_range_visible(int m_top, int m_height) {
	return (m_height > 0 && ((m_top >= 0 && m_top < height) || ((m_top + m_height - 1) >= 0 && (m_top + m_height - 1) < height)));
}

BbCodeParser::Item *BbCodeParser::_get_next_item(Item *p_item, bool p_free) const {
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
		if (p_item->subitems.size() && p_item->type != ItemType::ITEM_TABLE) {
			return p_item->subitems.front()->get();
		} else if (p_item->type == ItemType::ITEM_FRAME) {
			return nullptr;
		} else if (p_item->E->next()) {
			return p_item->E->next()->get();
		} else {
			//go up until something with a next is found
			while (p_item->type != ItemType::ITEM_FRAME && !p_item->E->next()) {
				p_item = p_item->parent;
			}

			if (p_item->type != ItemType::ITEM_FRAME) {
				return p_item->E->next()->get();
			} else {
				return nullptr;
			}
		}
	}

	return nullptr;
}

Ref<Font> BbCodeParser::_find_font(Item *p_item) {
	Item *fontitem = p_item;

	while (fontitem) {
		if (fontitem->type == ItemType::ITEM_FONT) {
			ItemFont *fi = static_cast<ItemFont *>(fontitem);
			return fi->font;
		}

		fontitem = fontitem->parent;
	}

	return Ref<Font>();
}

int BbCodeParser::_find_margin(Item *p_item, const Ref<Font> &p_base_font) {
	Item *item = p_item;

	int margin = 0;

	while (item) {
		if (item->type == ItemType::ITEM_INDENT) {
			Ref<Font> font = _find_font(item);
			if (font.is_null()) {
				font = p_base_font;
			}

			ItemIndent *indent = static_cast<ItemIndent *>(item);

			margin += indent->level * tab_size * font->get_char_size(' ').width;

		} else if (item->type == ItemType::ITEM_LIST) {
			Ref<Font> font = _find_font(item);
			if (font.is_null()) {
				font = p_base_font;
			}
		}

		item = item->parent;
	}

	return margin;
}

BbCodeParser::Align BbCodeParser::_find_align(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ItemType::ITEM_ALIGN) {
			ItemAlign *align = static_cast<ItemAlign *>(item);
			return align->align;
		}

		item = item->parent;
	}

	return default_align;
}

Color BbCodeParser::_find_color(Item *p_item, const Color &p_default_color) {
	Item *item = p_item;

	while (item) {
		if (item->type == ItemType::ITEM_COLOR) {
			ItemColor *color = static_cast<ItemColor *>(item);
			return color->color;
		}

		item = item->parent;
	}

	return p_default_color;
}

bool BbCodeParser::_find_underline(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ItemType::ITEM_UNDERLINE) {
			return true;
		}

		item = item->parent;
	}

	return false;
}

bool BbCodeParser::_find_strikethrough(Item *p_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ItemType::ITEM_STRIKETHROUGH) {
			return true;
		}

		item = item->parent;
	}

	return false;
}

bool BbCodeParser::_find_meta(Item *p_item, Variant *r_meta, ItemMeta **r_item) {
	Item *item = p_item;

	while (item) {
		if (item->type == ItemType::ITEM_META) {
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

void BbCodeParser::_fetch_item_fx_stack(Item *p_item, Vector<ItemFX *> &r_stack) {
	Item *item = p_item;
	while (item) {
		if (item->type == ItemType::ITEM_CUSTOMFX || item->type == ItemType::ITEM_SHAKE || item->type == ItemType::ITEM_WAVE || item->type == ItemType::ITEM_TORNADO || item->type == ItemType::ITEM_RAINBOW) {
			r_stack.push_back(static_cast<ItemFX *>(item));
		}

		item = item->parent;
	}
}

int BbCodeParser::_process_line(ItemFrame *p_frame, const Vector2 &p_ofs, int &y, int p_width, int p_line, ProcessMode p_mode, const Ref<Font> &p_base_font, const Color &p_base_color, const Color &p_font_color_shadow, bool p_shadow_as_outline, const Point2 &shadow_ofs, const Point2i &p_click_pos, Item **r_click_item, int *r_click_char, bool *r_outside, int p_char_count) {
	ERR_FAIL_INDEX_V((int)p_mode, 3, 0);

	RID ci;
	if (r_outside) {
		*r_outside = false;
	}
	if (p_mode == ProcessMode::PROCESS_DRAW) {
		ci = get_canvas_item();

		if (r_click_item) {
			*r_click_item = nullptr;
		}
	}
	Line &l = p_frame->lines.write[p_line];
	Item *it = l.from;

	int line_ofs = 0;
	int margin = _find_margin(it, p_base_font);
	RichTextLabel::Align align = _find_align(it);
	int line = 0;
	int spaces = 0;

	int height = get_size().y;

	if (p_mode != ProcessMode::PROCESS_CACHE) {
		ERR_FAIL_INDEX_V(line, l.offset_caches.size(), 0);
		line_ofs = l.offset_caches[line];
	}

	if (p_mode == ProcessMode::PROCESS_CACHE) {
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

	if (p_mode != ProcessMode::PROCESS_CACHE && align != Align::ALIGN_FILL) {
		wofs += line_ofs;
	}

	int begin = wofs;

	Ref<Font> cfont = _find_font(it);
	if (cfont.is_null()) {
		cfont = p_base_font;
	}

	//line height should be the font height for the first time, this ensures that an empty line will never have zero height and successive newlines are displayed
	int line_height = cfont->get_height();
	int line_ascent = cfont->get_ascent();
	int line_descent = cfont->get_descent();

	int backtrack = 0; // for dynamic hidden content.

	int nonblank_line_count = 0; //number of nonblank lines as counted during ProcessMode::PROCESS_DRAW

	Variant meta;

	Color selection_fg;
	Color selection_bg;

	if (p_mode == RichTextLabel::ProcessMode::PROCESS_DRAW) {
		selection_fg = get_theme_color("font_color_selected");
		selection_bg = get_theme_color("selection_color");
	}

	int rchar = 0;
	int lh = 0;
	bool line_is_blank = true;
	bool line_wrapped = false;
	int fh = 0;

	while (it) {
		switch (it->type) {
			case ItemType::ITEM_ALIGN: {
				ItemAlign *align_it = static_cast<ItemAlign *>(it);

				align = align_it->align;

			} break;
			case ItemType::ITEM_INDENT: {
				if (it != l.from) {
					ItemIndent *indent_it = static_cast<ItemIndent *>(it);

					int indent = indent_it->level * tab_size * cfont->get_char_size(' ').width;
					margin += indent;
					begin += indent;
					wofs += indent;
				}

			} break;
			case ItemType::ITEM_TEXT: {
				ItemText *text = static_cast<ItemText *>(it);

				Ref<Font> font = _find_font(it);
				if (font.is_null()) {
					font = p_base_font;
				}

				const CharType *c = text->text.c_str();
				const CharType *cf = c;
				int ascent = font->get_ascent();
				int descent = font->get_descent();

				Color color;
				Color font_color_shadow;
				bool underline = false;
				bool strikethrough = false;
				ItemFade *fade = nullptr;
				int it_char_start = p_char_count;

				Vector<ItemFX *> fx_stack = Vector<ItemFX *>();
				_fetch_item_fx_stack(text, fx_stack);
				bool custom_fx_ok = true;

				if (p_mode == ProcessMode::PROCESS_DRAW) {
					color = _find_color(text, p_base_color);
					font_color_shadow = _find_color(text, p_font_color_shadow);
					if (_find_underline(text) || (_find_meta(text, &meta) && underline_meta)) {
						underline = true;
					} else if (_find_strikethrough(text)) {
						strikethrough = true;
					}

					Item *fade_item = it;
					while (fade_item) {
						if (fade_item->type == ItemType::ITEM_FADE) {
							fade = static_cast<ItemFade *>(fade_item);
							break;
						}
						fade_item = fade_item->parent;
					}

				} else if (p_mode == ProcessMode::PROCESS_CACHE) {
					l.char_count += text->text.length();
				}

				rchar = 0;
				FontDrawer drawer(font, Color(1, 1, 1));
				while (*c) {
					int end = 0;
					int w = 0;
					int fw = 0;

					lh = 0;

					if (p_mode != ProcessMode::PROCESS_CACHE) {
						lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
						line_ascent = line < l.ascent_caches.size() ? l.ascent_caches[line] : 1;
						line_descent = line < l.descent_caches.size() ? l.descent_caches[line] : 1;
					}
					while (c[end] != 0 && !(end && c[end - 1] == ' ' && c[end] != ' ')) {
						int cw = font->get_char_size(c[end], c[end + 1]).width;
						if (c[end] == '\t') {
							cw = tab_size * font->get_char_size(' ').width;
						}

						if (end > 0 && fw + cw + begin > p_width) {
							break; //don't allow lines longer than assigned width
						}

						fw += cw;

						end++;
					}
					_check_height(fh);
					if (_ensure_width(fw)) {
						return nonblank_line_count;
					}

					line_ascent = MAX(line_ascent, ascent);
					line_descent = MAX(line_descent, descent);
					fh = line_ascent + line_descent;

					if (end && c[end - 1] == ' ') {
						if (p_mode == ProcessMode::PROCESS_CACHE) {
							spaces_size += font->get_char_size(' ').width;
						} else if (align == Align::ALIGN_FILL) {
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

							if (p_mode == ProcessMode::PROCESS_POINTER && r_click_char && p_click_pos.y >= p_ofs.y + y && p_click_pos.y <= p_ofs.y + y + lh) {
								int cw = font->get_char_size(c[i], c[i + 1]).x;

								if (c[i] == '\t') {
									cw = tab_size * font->get_char_size(' ').width;
								}

								if (p_click_pos.x - cw / 2 > p_ofs.x + align_ofs + pofs) {
									rchar = int((&c[i]) - cf);
								}

								ofs += cw;
							} else if (p_mode == ProcessMode::PROCESS_DRAW) {
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

								bool visible = visible_characters < 0 || ((p_char_count < visible_characters && _y_range_visible(y + lh - line_descent - line_ascent, line_ascent + line_descent)) &&
																				 faded_visibility > 0.0f);

								const bool previously_visible = visible;

								for (int j = 0; j < fx_stack.size(); j++) {
									ItemFX *item_fx = fx_stack[j];

									if (item_fx->type == ItemType::ITEM_CUSTOMFX && custom_fx_ok) {
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
									} else if (item_fx->type == ItemType::ITEM_SHAKE) {
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
									} else if (item_fx->type == ItemType::ITEM_WAVE) {
										ItemWave *item_wave = static_cast<ItemWave *>(item_fx);

										double value = Math::sin(item_wave->frequency * item_wave->elapsed_time + ((p_ofs.x + pofs) / 50)) * (item_wave->amplitude / 10.0f);
										fx_offset += Point2(0, 1) * value;
									} else if (item_fx->type == ItemType::ITEM_TORNADO) {
										ItemTornado *item_tornado = static_cast<ItemTornado *>(item_fx);

										double torn_x = Math::sin(item_tornado->frequency * item_tornado->elapsed_time + ((p_ofs.x + pofs) / 50)) * (item_tornado->radius);
										double torn_y = Math::cos(item_tornado->frequency * item_tornado->elapsed_time + ((p_ofs.x + pofs) / 50)) * (item_tornado->radius);
										fx_offset += Point2(torn_x, torn_y);
									} else if (item_fx->type == ItemType::ITEM_RAINBOW) {
										ItemRainbow *item_rainbow = static_cast<ItemRainbow *>(item_fx);

										fx_color = fx_color.from_hsv(item_rainbow->frequency * (item_rainbow->elapsed_time + ((p_ofs.x + pofs) / 50)),
												item_rainbow->saturation,
												item_rainbow->value,
												fx_color.a);
									}
								}

								if (visible) {
									line_is_blank = false;
									w += font->get_char_size(c[i], c[i + 1]).x;
								}

								if (c[i] == '\t') {
									visible = false;
								}

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
							int uy = y + lh - line_descent + font->get_underline_position();
							float underline_width = font->get_underline_thickness();
#ifdef TOOLS_ENABLED
							underline_width *= EDSCALE;
#endif
							RS::get_singleton()->canvas_item_add_line(ci, p_ofs + Point2(align_ofs + wofs, uy), p_ofs + Point2(align_ofs + wofs + w, uy), uc, underline_width);
						} else if (strikethrough) {
							Color uc = color;
							uc.a *= 0.5;
							int uy = y + lh - (line_ascent + line_descent) / 2;
							float strikethrough_width = font->get_underline_thickness();
#ifdef TOOLS_ENABLED
							strikethrough_width *= EDSCALE;
#endif
							RS::get_singleton()->canvas_item_add_line(ci, p_ofs + Point2(align_ofs + wofs, uy), p_ofs + Point2(align_ofs + wofs + w, uy), uc, strikethrough_width);
						}
					}

					if (_advance(fw)) {
						return nonblank_line_count;
					}
					_check_height(fh); //must be done somewhere
					c = &c[end];
				}

			} break;
			case ItemType::ITEM_IMAGE: {
				lh = 0;
				if (p_mode != ProcessMode::PROCESS_CACHE) {
					lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
				} else {
					l.char_count += 1; //images count as chars too
				}

				ItemImage *img = static_cast<ItemImage *>(it);

				Ref<Font> font = _find_font(it);
				if (font.is_null()) {
					font = p_base_font;
				}

				if (p_mode == ProcessMode::PROCESS_POINTER && r_click_char) {
					*r_click_char = 0;
				}

				if (_ensure_width(img->size.width)) {
					return nonblank_line_count;
				}

				bool visible = visible_characters < 0 || (p_char_count < visible_characters && _y_range_visible(y + lh - font->get_descent() - img->size.height, img->size.height));
				if (visible) {
					line_is_blank = false;
				}

				if (p_mode == ProcessMode::PROCESS_DRAW && visible) {
					img->image->draw_rect(ci, Rect2(p_ofs + Point2(align_ofs + wofs, y + lh - font->get_descent() - img->size.height), img->size));
				}
				p_char_count++;

				if (_advance(img->size.width)) {
					return nonblank_line_count;
				}

				_check_height(img->size.height + font->get_descent());

			} break;
			case ItemType::ITEM_NEWLINE: {
				lh = 0;

				if (p_mode != ProcessMode::PROCESS_CACHE) {
					lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
					line_is_blank = true;
				}

			} break;
			case ItemType::ITEM_TABLE: {
				lh = 0;
				ItemTable *table = static_cast<ItemTable *>(it);
				int hseparation = get_theme_constant("table_hseparation");
				int vseparation = get_theme_constant("table_vseparation");
				Color ccolor = _find_color(table, p_base_color);
				Vector2 draw_ofs = Point2(wofs, y);
				Color font_color_shadow = get_theme_color("font_color_shadow");
				bool use_outline = get_theme_constant("shadow_as_outline");
				Point2 shadow_ofs2(get_theme_constant("shadow_offset_x"), get_theme_constant("shadow_offset_y"));

				if (p_mode == ProcessMode::PROCESS_CACHE) {
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
						ERR_CONTINUE(E->get()->type != ItemType::ITEM_FRAME); //children should all be frames
						ItemFrame *frame = static_cast<ItemFrame *>(E->get());

						int column = idx % table->columns.size();

						int ly = 0;

						for (int i = 0; i < frame->lines.size(); i++) {
							_process_line(frame, Point2(), ly, available_width, i, ProcessMode::PROCESS_CACHE, cfont, Color(), font_color_shadow, use_outline, shadow_ofs2);
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
						if (table->columns[i].max_width > table->columns[i].min_width) {
							table->columns.write[i].expand = true;
						}
						if (table->columns[i].expand) {
							total_ratio += table->columns[i].expand_ratio;
						}
					}

					//assign actual widths
					for (int i = 0; i < table->columns.size(); i++) {
						table->columns.write[i].width = table->columns[i].min_width;
						if (table->columns[i].expand && total_ratio > 0) {
							table->columns.write[i].width += table->columns[i].expand_ratio * remaining_width / total_ratio;
						}
						table->total_width += table->columns[i].width + hseparation;
					}

					//resize to max_width if needed and distribute the remaining space
					bool table_need_fit = true;
					while (table_need_fit) {
						table_need_fit = false;
						//fit slim
						for (int i = 0; i < table->columns.size(); i++) {
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
						ERR_CONTINUE(E->get()->type != ItemType::ITEM_FRAME); //children should all be frames
						ItemFrame *frame = static_cast<ItemFrame *>(E->get());

						int column = idx % table->columns.size();

						for (int i = 0; i < frame->lines.size(); i++) {
							int ly = 0;
							_process_line(frame, Point2(), ly, table->columns[column].width, i, ProcessMode::PROCESS_CACHE, cfont, Color(), font_color_shadow, use_outline, shadow_ofs2);
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
					ERR_CONTINUE(E->get()->type != ItemType::ITEM_FRAME); //children should all be frames
					ItemFrame *frame = static_cast<ItemFrame *>(E->get());

					int column = idx % table->columns.size();

					int ly = 0;
					int yofs = 0;

					int lines_h = frame->lines[frame->lines.size() - 1].height_accum_cache - (frame->lines[0].height_accum_cache - frame->lines[0].height_cache);
					int lines_ofs = p_ofs.y + offset.y + draw_ofs.y;

					bool visible = lines_ofs < get_size().height && lines_ofs + lines_h >= 0;
					if (visible) {
						line_is_blank = false;
					}

					for (int i = 0; i < frame->lines.size(); i++) {
						if (visible) {
							if (p_mode == ProcessMode::PROCESS_DRAW) {
								nonblank_line_count += _process_line(frame, p_ofs + offset + draw_ofs + Vector2(0, yofs), ly, table->columns[column].width, i, ProcessMode::PROCESS_DRAW, cfont, ccolor, font_color_shadow, use_outline, shadow_ofs2);
							} else if (p_mode == ProcessMode::PROCESS_POINTER) {
								_process_line(frame, p_ofs + offset + draw_ofs + Vector2(0, yofs), ly, table->columns[column].width, i, ProcessMode::PROCESS_POINTER, cfont, ccolor, font_color_shadow, use_outline, shadow_ofs2, p_click_pos, r_click_item, r_click_char, r_outside);
								if (r_click_item && *r_click_item) {
									return nonblank_line_count; // exit early
								}
							}
						}

						yofs += frame->lines[i].height_cache;
						if (p_mode == ProcessMode::PROCESS_CACHE) {
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
				if (_advance(table->total_width)) {
					return nonblank_line_count;
				}
				_check_height(total_height);

			} break;

			default: {
			}
		}

		Item *itp = it;

		it = _get_next_item(it);

		if (it && (p_line + 1 < p_frame->lines.size()) && p_frame->lines[p_line + 1].from == it) {
			if (p_mode == ProcessMode::PROCESS_POINTER && r_click_item && p_click_pos.y >= p_ofs.y + y && p_click_pos.y <= p_ofs.y + y + lh) {
				//went to next line, but pointer was on the previous one
				if (r_outside) {
					*r_outside = true;
				}
				*r_click_item = itp;
				*r_click_char = rchar;
				return nonblank_line_count;
			}

			break;
		}
	}
	_new_line();

	return nonblank_line_count;
}

void BbCodeParser::start_process(ItemFrame *_p_frame, const Vector2 &_p_ofs, const int p_line_height, const int _p_width, const int _p_line, const Ref<Font> &p_base_font) {
	p_frame = _p_frame;
	p_ofs = _p_ofs;
	p_height = p_line_height;
	p_width = _p_width;
	p_line = _p_line;

	l = p_frame->lines.write[p_line];
	it = l.from;

	margin = _find_margin(it, p_base_font);
	align = _find_align(it);
	height = get_size().y;

	wofs = margin;

	/*if (p_mode != ProcessMode::PROCESS_CACHE && align != Align::ALIGN_FILL) {
		wofs += line_ofs;
	}*/

	begin = wofs;

	cfont = _find_font(it);
	if (cfont.is_null()) {
		cfont = p_base_font;
	}

	//line height should be the font height for the first time, this ensures that an empty line will never have zero height and successive newlines are displayed
	line_height = cfont->get_height();
	line_ascent = cfont->get_ascent();
	line_descent = cfont->get_descent();
}

void BbCodeParser::process_cache() {
	ERR_FAIL_INDEX(line, l.offset_caches.size());
	line_ofs = l.offset_caches[line];

	l.offset_caches.clear();
	l.height_caches.clear();
	l.ascent_caches.clear();
	l.descent_caches.clear();
	l.char_count = 0;
	l.minimum_width = 0;
	l.maximum_width = 0;
}

void BbCodeParser::process_draw() {
}
