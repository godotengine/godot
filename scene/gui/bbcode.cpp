
#include "scene/gui/bbcode.h"
#include "core/math/math_defs.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "scene/scene_string_names.h"
#include "servers/display_server.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_scale.h"
#endif

BbCodeParser::BbCodeParser(ItemFrame *_p_frame, const Vector2 &_p_ofs, int &_p_height, int _p_width, int _p_line,
		const Ref<Font> &_p_base_font, const Color &_p_base_color, const Color &_p_font_color_shadow,
		bool _p_shadow_as_outline, const Point2 &_shadow_ofs, RichTextLabel &_p_ci) :
		l{ _p_frame->lines.write[_p_line] },
		ci{ _p_ci.get_canvas_item() },
		p_frame{ _p_frame },
		p_ofs{ _p_ofs },
		p_base_font{ _p_base_font },
		p_base_color{ _p_base_color },
		p_font_color_shadow{ _p_font_color_shadow },
		shadow_ofs{ _shadow_ofs },
		p_ci{ _p_ci },
		p_height{ _p_height },
		p_width{ _p_width },
		p_line{ _p_line },
		p_shadow_as_outline{ _p_shadow_as_outline } {
	it = l.from;
	line_ofs = 0;
	margin = _find_margin(it, p_base_font);
	align = _find_align(it);
	line = 0;
	spaces = 0;

	wofs = margin;
	spaces_size = 0;
	align_ofs = 0;

	cfont = _find_font(it);
	if (cfont.is_null()) {
		cfont = p_base_font;
	}

	//line height should be the font height for the first time, this ensures that an empty line will never have zero height and successive newlines are displayed
	line_height = cfont->get_height();
	line_ascent = cfont->get_ascent();
	line_descent = cfont->get_descent();

	backtrack = 0; // for dynamic hidden content.
	nonblank_line_count = 0; //number of nonblank lines as counted during ProcessMode::PROCESS_DRAW

	rchar = 0;
	lh = 0;
	line_is_blank = true;
	line_wrapped = false;
	fh = 0;

	tab_size = p_ci.tab_size;
	default_align = p_ci.default_align;
	underline_meta = p_ci.underline_meta;
	override_selected_font_color = p_ci.override_selected_font_color;
	selection = p_ci.selection;
	visible_characters = p_ci.visible_characters;
}

BbCodeParser::~BbCodeParser() {}

void BbCodeParser::_bind_methods() {
}

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
	p_height += line_height + p_ci.get_theme_constant(SceneStringNames::get_singleton()->line_separation);
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
		if (p_mode == ProcessMode::PROCESS_CACHE && spaces > 0) {
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
	return (m_height > 0 && ((m_top >= 0 && m_top < p_ci.get_size().y) || ((m_top + m_height - 1) >= 0 && (m_top + m_height - 1) < p_ci.get_size().y)));
}

bool BbCodeParser::_parse_text(ItemText *text) {
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
			return true;
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

				if (p_mode == ProcessMode::PROCESS_POINTER && r_click_char && p_click_pos.y >= p_ofs.y + p_height && p_click_pos.y <= p_ofs.y + p_height + lh) {
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

					bool visible = visible_characters < 0 ||
								   ((p_char_count < visible_characters &&
											_y_range_visible(p_height + lh - line_descent - line_ascent, line_ascent + line_descent)) &&
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
							p_ci.draw_rect(Rect2(p_ofs.x + pofs, p_ofs.y + p_height, cw, lh), selection_bg);
						}

						if (p_font_color_shadow.a > 0) {
							float x_ofs_shadow = align_ofs + pofs;
							float y_ofs_shadow = p_height + lh - line_descent;
							font->draw_char(ci, Point2(x_ofs_shadow, y_ofs_shadow) + shadow_ofs + fx_offset, fx_char, c[i + 1], p_font_color_shadow);

							if (p_shadow_as_outline) {
								font->draw_char(ci, Point2(x_ofs_shadow, y_ofs_shadow) + Vector2(-shadow_ofs.x, shadow_ofs.y) + fx_offset, fx_char, c[i + 1], p_font_color_shadow);
								font->draw_char(ci, Point2(x_ofs_shadow, y_ofs_shadow) + Vector2(shadow_ofs.x, -shadow_ofs.y) + fx_offset, fx_char, c[i + 1], p_font_color_shadow);
								font->draw_char(ci, Point2(x_ofs_shadow, y_ofs_shadow) + Vector2(-shadow_ofs.x, -shadow_ofs.y) + fx_offset, fx_char, c[i + 1], p_font_color_shadow);
							}
						}

						if (selected) {
							drawer.draw_char(ci, p_ofs + Point2(align_ofs + pofs, p_height + lh - line_descent), fx_char, c[i + 1], override_selected_font_color ? selection_fg : fx_color);
						} else {
							cw = drawer.draw_char(ci, p_ofs + Point2(align_ofs + pofs, p_height + lh - line_descent) + fx_offset, fx_char, c[i + 1], fx_color);
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
				int uy = p_height + lh - line_descent + font->get_underline_position();
				float underline_width = font->get_underline_thickness();
#ifdef TOOLS_ENABLED
				underline_width *= EDSCALE;
#endif
				RS::get_singleton()->canvas_item_add_line(ci, p_ofs + Point2(align_ofs + wofs, uy), p_ofs + Point2(align_ofs + wofs + w, uy), uc, underline_width);
			} else if (strikethrough) {
				Color uc = color;
				uc.a *= 0.5;
				int uy = p_height + lh - (line_ascent + line_descent) / 2;
				float strikethrough_width = font->get_underline_thickness();
#ifdef TOOLS_ENABLED
				strikethrough_width *= EDSCALE;
#endif
				RS::get_singleton()->canvas_item_add_line(ci, p_ofs + Point2(align_ofs + wofs, uy), p_ofs + Point2(align_ofs + wofs + w, uy), uc, strikethrough_width);
			}
		}

		if (_advance(fw)) {
			return true;
		}
		_check_height(fh); //must be done somewhere
		c = &c[end];
	}

	return false;
}

bool BbCodeParser::_parse_table(ItemTable *table) {
	lh = 0;
	int hseparation = p_ci.get_theme_constant("table_hseparation");
	int vseparation = p_ci.get_theme_constant("table_vseparation");
	Color ccolor = _find_color(table, p_base_color);
	Vector2 draw_ofs = Point2(wofs, p_height);
	Color font_color_shadow = p_ci.get_theme_color("font_color_shadow");
	bool use_outline = p_ci.get_theme_constant("shadow_as_outline");
	Point2 shadow_ofs2(p_ci.get_theme_constant("shadow_offset_x"), p_ci.get_theme_constant("shadow_offset_y"));

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
				BbCodeParser(frame, Point2(), ly, available_width, i, cfont, Color(), font_color_shadow, use_outline, shadow_ofs2, p_ci).process_cache();
				//_process_line(frame, Point2(), ly, available_width, i, ProcessMode::PROCESS_CACHE, cfont, Color(), font_color_shadow, use_outline, shadow_ofs2);
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
				BbCodeParser(frame, Point2(), ly, table->columns[column].width, i, cfont, Color(), font_color_shadow, use_outline, shadow_ofs2, p_ci).process_cache();
				//_process_line(frame, Point2(), ly, table->columns[column].width, i, ProcessMode::PROCESS_CACHE, cfont, Color(), font_color_shadow, use_outline, shadow_ofs2);
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

		bool visible = lines_ofs < p_ci.get_size().height && lines_ofs + lines_h >= 0;
		if (visible) {
			line_is_blank = false;
		}

		for (int i = 0; i < frame->lines.size(); i++) {
			if (visible) {
				if (p_mode == ProcessMode::PROCESS_DRAW) {
					nonblank_line_count += BbCodeParser(frame, p_ofs + offset + draw_ofs + Vector2(0, yofs), ly, table->columns[column].width, i, cfont, ccolor, font_color_shadow, use_outline, shadow_ofs2, p_ci).process_draw();
					//nonblank_line_count += _process_line(frame, p_ofs + offset + draw_ofs + Vector2(0, yofs), ly, table->columns[column].width, i, ProcessMode::PROCESS_DRAW, cfont, ccolor, font_color_shadow, use_outline, shadow_ofs2);
				} else if (p_mode == ProcessMode::PROCESS_POINTER) {
					BbCodeParser(frame, p_ofs + offset + draw_ofs + Vector2(0, yofs), ly, table->columns[column].width, i, cfont, ccolor, font_color_shadow, use_outline, shadow_ofs2, p_ci).process_pointer(p_click_pos, r_click_item, r_click_char, r_outside);
					//_process_line(frame, p_ofs + offset + draw_ofs + Vector2(0, yofs), ly, table->columns[column].width, i, ProcessMode::PROCESS_POINTER, cfont, ccolor, font_color_shadow, use_outline, shadow_ofs2, p_click_pos, r_click_item, r_click_char, r_outside);
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
		return true;
	}
	_check_height(total_height);
	return false;
}

bool BbCodeParser::_parse_image(ItemImage *img) {
	lh = 0;
	if (p_mode != ProcessMode::PROCESS_CACHE) {
		lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
	} else {
		l.char_count += 1; //images count as chars too
	}

	Ref<Font> font = _find_font(it);
	if (font.is_null()) {
		font = p_base_font;
	}

	if (p_mode == ProcessMode::PROCESS_POINTER && r_click_char) {
		*r_click_char = 0;
	}

	if (_ensure_width(img->size.width)) {
		return true;
	}

	bool visible = visible_characters < 0 || (p_char_count < visible_characters && _y_range_visible(p_height + lh - font->get_descent() - img->size.height, img->size.height));
	if (visible) {
		line_is_blank = false;
	}

	if (p_mode == ProcessMode::PROCESS_DRAW && visible) {
		img->image->draw_rect(ci, Rect2(p_ofs + Point2(align_ofs + wofs, p_height + lh - font->get_descent() - img->size.height), img->size), false, img->color);
	}
	p_char_count++;

	if (_advance(img->size.width)) {
		return true;
	}

	_check_height(img->size.height + font->get_descent());

	return false;
}

bool BbCodeParser::_parse_detect_click(Item *previous_item) {
	if (p_mode == ProcessMode::PROCESS_POINTER && r_click_item && p_click_pos.y >= p_ofs.y + p_height && p_click_pos.y <= p_ofs.y + p_height + lh) {
		//went to next line, but pointer was on the previous one
		if (r_outside) {
			*r_outside = true;
		}
		*r_click_item = previous_item;
		*r_click_char = rchar;
		return true;
	}
	return false;
}

void BbCodeParser::_common_initalize_process() {
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

/*
int BbCodeParser::_process_line(ItemFrame *p_frame, const Vector2 &p_ofs, int &y, int p_width, int p_line, ProcessMode p_mode, const Ref<Font> &p_base_font, const Color &p_base_color, const Color &p_font_color_shadow, bool p_shadow_as_outline, const Point2 &shadow_ofs, const Point2i &p_click_pos, Item **r_click_item, int *r_click_char, bool *r_outside, int p_char_count) {
	ERR_FAIL_INDEX_V((int)p_mode, 3, 0);

	if (r_outside) {
		*r_outside = false;
	}
	if (p_mode == ProcessMode::PROCESS_DRAW) {
		ci = get_canvas_item();

		if (r_click_item) {
			*r_click_item = nullptr;
		}
	}
	l = p_frame->lines.write[p_line];
	it = l.from;

	line_ofs = 0;
	margin = _find_margin(it, p_base_font);
	align = _find_align(it);
	line = 0;
	spaces = 0;

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

	wofs = margin;
	spaces_size = 0;
	align_ofs = 0;

	if (p_mode != ProcessMode::PROCESS_CACHE && align != Align::ALIGN_FILL) {
		wofs += line_ofs;
	}

	begin = wofs;

	cfont = _find_font(it);
	if (cfont.is_null()) {
		cfont = p_base_font;
	}

	//line height should be the font height for the first time, this ensures that an empty line will never have zero height and successive newlines are displayed
	line_height = cfont->get_height();
	line_ascent = cfont->get_ascent();
	line_descent = cfont->get_descent();

	backtrack = 0; // for dynamic hidden content.

	nonblank_line_count = 0; //number of nonblank lines as counted during ProcessMode::PROCESS_DRAW

	if (p_mode == RichTextLabel::ProcessMode::PROCESS_DRAW) {
		selection_fg = get_theme_color("font_color_selected");
		selection_bg = get_theme_color("selection_color");
	}

	rchar = 0;
	lh = 0;
	line_is_blank = true;
	line_wrapped = false;
	fh = 0;

	while (it) {
		switch (it->type) {
			case ItemType::ITEM_ALIGN: {
				ItemAlign *align_it = static_cast<ItemAlign *>(it);
				align = align_it->align;
			} break;
			case ItemType::ITEM_INDENT: {
				if (it == l.from) {
					break;
				}
				ItemIndent *indent_it = static_cast<ItemIndent *>(it);
				int indent = indent_it->level * tab_size * cfont->get_char_size(' ').width;
				margin += indent;
				begin += indent;
				wofs += indent;
			} break;
			case ItemType::ITEM_TEXT: {
				if (_parse_text(static_cast<ItemText *>(it))) {
					return nonblank_line_count;
				}
			} break;
			case ItemType::ITEM_IMAGE: {
				if (_parse_image(static_cast<ItemImage *>(it))) {
					return nonblank_line_count;
				}
			} break;
			case ItemType::ITEM_NEWLINE: {
				lh = 0;

				if (p_mode != ProcessMode::PROCESS_CACHE) {
					lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
					line_is_blank = true;
				}

			} break;
			case ItemType::ITEM_TABLE: {
				if (_parse_table(static_cast<ItemTable *>(it))) {
					return nonblank_line_count;
				}

			} break;

			default: {
			}
		}

		Item *itp = it;

		it = _get_next_item(it);

		if (it && (p_line + 1 < p_frame->lines.size()) && p_frame->lines[p_line + 1].from == it) {
			if (_parse_detect_click(itp)) {
				return nonblank_line_count;
			}

			break;
		}
	}
	if (_new_line()) {
		//Safe is safe. If somebody does something below _new_line we have a bug
		return nonblank_line_count;
	}

	return nonblank_line_count;
}
*/

int BbCodeParser::process_cache() {
	p_mode = ProcessMode::PROCESS_CACHE;

	_common_initalize_process();

	l.offset_caches.clear();
	l.height_caches.clear();
	l.ascent_caches.clear();
	l.descent_caches.clear();
	l.char_count = 0;
	l.minimum_width = 0;
	l.maximum_width = 0;

	begin = wofs;

	while (it) {
		switch (it->type) {
			case ItemType::ITEM_ALIGN: {
				ItemAlign *align_it = static_cast<ItemAlign *>(it);
				align = align_it->align;
			} break;
			case ItemType::ITEM_INDENT: {
				if (it == l.from) {
					break;
				}
				ItemIndent *indent_it = static_cast<ItemIndent *>(it);
				int indent = indent_it->level * tab_size * cfont->get_char_size(' ').width;
				margin += indent;
				begin += indent;
				wofs += indent;
			} break;
			case ItemType::ITEM_TEXT: {
				if (_parse_text(static_cast<ItemText *>(it))) {
					return p_height;
				}
			} break;
			case ItemType::ITEM_IMAGE: {
				if (_parse_image(static_cast<ItemImage *>(it))) {
					return p_height;
				}
			} break;
			case ItemType::ITEM_TABLE: {
				if (_parse_table(static_cast<ItemTable *>(it))) {
					return p_height;
				}

			} break;

			default: {
			}
		}

		it = _get_next_item(it);
		if (it && (p_line + 1 < p_frame->lines.size()) && p_frame->lines[p_line + 1].from == it) {
			break;
		}
	}

	if (_new_line()) {
		//Safe is safe. If somebody does something below _new_line we have a bug
		return p_height;
	}
	return p_height;
}

int BbCodeParser::process_draw(int _p_char_count) {
	p_mode = ProcessMode::PROCESS_DRAW;
	p_char_count = _p_char_count;

	_common_initalize_process();

	ERR_FAIL_INDEX_V(line, l.offset_caches.size(), 0);
	line_ofs = l.offset_caches[line];

	if (align != Align::ALIGN_FILL) {
		wofs += line_ofs;
	}

	begin = wofs;
	selection_fg = p_ci.get_theme_color("font_color_selected");
	selection_bg = p_ci.get_theme_color("selection_color");

	while (it) {
		switch (it->type) {
			case ItemType::ITEM_ALIGN: {
				ItemAlign *align_it = static_cast<ItemAlign *>(it);
				align = align_it->align;
			} break;
			case ItemType::ITEM_INDENT: {
				if (it == l.from) {
					break;
				}
				ItemIndent *indent_it = static_cast<ItemIndent *>(it);
				int indent = indent_it->level * tab_size * cfont->get_char_size(' ').width;
				margin += indent;
				begin += indent;
				wofs += indent;
			} break;
			case ItemType::ITEM_TEXT: {
				if (_parse_text(static_cast<ItemText *>(it))) {
					return nonblank_line_count;
				}
			} break;
			case ItemType::ITEM_IMAGE: {
				if (_parse_image(static_cast<ItemImage *>(it))) {
					return nonblank_line_count;
				}
			} break;
			case ItemType::ITEM_NEWLINE: {
				lh = 0;
				lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
				line_is_blank = true;

			} break;
			case ItemType::ITEM_TABLE: {
				if (_parse_table(static_cast<ItemTable *>(it))) {
					return nonblank_line_count;
				}

			} break;

			default: {
			}
		}

		it = _get_next_item(it);
		if (it && (p_line + 1 < p_frame->lines.size()) && p_frame->lines[p_line + 1].from == it) {
			break;
		}
	}
	if (_new_line()) {
		//Safe is safe. If somebody does something below _new_line we have a bug
		return nonblank_line_count;
	}

	return nonblank_line_count;
}

void BbCodeParser::process_pointer(const Point2i &_p_click_pos, Item **_r_click_item, int *_r_click_char, bool *_r_outside) {
	p_mode = ProcessMode::PROCESS_POINTER;

	p_click_pos = _p_click_pos;
	r_click_item = _r_click_item;
	r_click_char = _r_click_char;
	r_outside = _r_outside;

	if (r_outside) {
		*r_outside = false;
	}

	_common_initalize_process();

	ERR_FAIL_INDEX(line, l.offset_caches.size());
	line_ofs = l.offset_caches[line];

	if (align != Align::ALIGN_FILL) {
		wofs += line_ofs;
	}

	begin = wofs;

	while (it) {
		switch (it->type) {
			case ItemType::ITEM_ALIGN: {
				ItemAlign *align_it = static_cast<ItemAlign *>(it);
				align = align_it->align;
			} break;
			case ItemType::ITEM_INDENT: {
				if (it == l.from) {
					break;
				}
				ItemIndent *indent_it = static_cast<ItemIndent *>(it);
				int indent = indent_it->level * tab_size * cfont->get_char_size(' ').width;
				margin += indent;
				begin += indent;
				wofs += indent;
			} break;
			case ItemType::ITEM_TEXT: {
				if (_parse_text(static_cast<ItemText *>(it))) {
					return;
				}
			} break;
			case ItemType::ITEM_IMAGE: {
				if (_parse_image(static_cast<ItemImage *>(it))) {
					return;
				}
			} break;
			case ItemType::ITEM_NEWLINE: {
				lh = 0;
				lh = line < l.height_caches.size() ? l.height_caches[line] : 1;
				line_is_blank = true;
			} break;
			case ItemType::ITEM_TABLE: {
				if (_parse_table(static_cast<ItemTable *>(it))) {
					return;
				}

			} break;

			default: {
			}
		}

		Item *itp = it;
		it = _get_next_item(it);

		if (it && (p_line + 1 < p_frame->lines.size()) && p_frame->lines[p_line + 1].from == it) {
			if (_parse_detect_click(itp)) {
				return;
			}
			break;
		}
	}

	if (_new_line()) {
		//Safe is safe. If somebody does something below _new_line we have a bug
		return;
	}
}
