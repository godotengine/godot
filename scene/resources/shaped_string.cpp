/*************************************************************************/
/*  shaped_string.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "scene/resources/shaped_string.h"
#include "core/translation.h"

/*************************************************************************/
/*  ShapedString::Glyph                                                  */
/*************************************************************************/

ShapedString::Glyph::Glyph() {

	codepoint = 0;
}

ShapedString::Glyph::Glyph(uint32_t p_codepoint, Point2 p_offset, Point2 p_advance) {

	codepoint = p_codepoint;

	offset = p_offset;
	advance = p_advance;
}

/*************************************************************************/
/*  ShapedString::Cluster                                                */
/*************************************************************************/

ShapedString::Cluster::Cluster() {

	start = -1;
	end = -1;
	fallback_depth = -3;
	ascent = 0.0f;
	descent = 0.0f;
	width = 0.0f;
	valid = false;
}

/*************************************************************************/
/*  ShapedString::ScriptIterator                                         */
/*************************************************************************/
#ifdef USE_TEXT_SHAPING

bool ShapedString::ScriptIterator::same_script(int32_t p_script_one, int32_t p_script_two) {

	return p_script_one <= USCRIPT_INHERITED || p_script_two <= USCRIPT_INHERITED || p_script_one == p_script_two;
}

bool ShapedString::ScriptIterator::next() {

	if (is_rtl) {
		cur--;
	} else {
		cur++;
	}
	return (cur >= 0) && (cur < script_ranges.size());
}

int32_t ShapedString::ScriptIterator::get_start() const {

	if ((cur >= 0) && (cur < script_ranges.size())) {
		return script_ranges[cur].start;
	} else {
		return -1;
	}
}

int32_t ShapedString::ScriptIterator::get_end() const {

	if ((cur >= 0) && (cur < script_ranges.size())) {
		return script_ranges[cur].end;
	} else {
		return -1;
	}
}

hb_script_t ShapedString::ScriptIterator::get_script() const {

	if ((cur >= 0) && (cur < script_ranges.size())) {
		return script_ranges[cur].script;
	} else {
		return HB_SCRIPT_INVALID;
	}
}

void ShapedString::ScriptIterator::reset(hb_direction_t p_run_direction) {

	if (p_run_direction == HB_DIRECTION_LTR) {
		cur = -1;
		is_rtl = false;
	} else {
		cur = script_ranges.size();
		is_rtl = true;
	}
}

ShapedString::ScriptIterator::ScriptIterator(const UChar *p_chars, int32_t p_start, int32_t p_length) {

	struct ParenStackEntry {
		int32_t pair_index;
		UScriptCode script_code;
	};

	ParenStackEntry paren_stack[128];

	int32_t script_start;
	int32_t script_end = p_start;
	UScriptCode script_code;
	int32_t paren_sp = -1;
	int32_t start_sp = paren_sp;
	UErrorCode err = U_ZERO_ERROR;

	cur = -1;

	do {
		script_code = USCRIPT_COMMON;
		for (script_start = script_end; script_end < p_length; script_end += 1) {
			UChar high = p_chars[script_end];

			UChar32 ch = high;

			if (high >= 0xD800 && high <= 0xDBFF && script_end < p_length - 1) {
				UChar low = p_chars[script_end + 1];

				if (low >= 0xDC00 && low <= 0xDFFF) {
					ch = (high - 0xD800) * 0x0400 + low - 0xDC00 + 0x10000;
					script_end += 1;
				}
			}

			UScriptCode sc = uscript_getScript(ch, &err);
			if (U_FAILURE(err)) {
				ERR_EXPLAIN(String(u_errorName(err)));
				ERR_FAIL_COND(true);
			}
			if (u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) != U_BPT_NONE) {
				if (u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) == U_BPT_OPEN) {
					paren_stack[++paren_sp].pair_index = ch;
					paren_stack[paren_sp].script_code = script_code;
				} else if (paren_sp >= 0) {
					UChar32 paired_ch = u_getBidiPairedBracket(ch);
					while (paren_sp >= 0 && paren_stack[paren_sp].pair_index != paired_ch)
						paren_sp -= 1;
					if (paren_sp < start_sp) start_sp = paren_sp;
					if (paren_sp >= 0) sc = paren_stack[paren_sp].script_code;
				}
			}

			if (same_script(script_code, sc)) {
				if (script_code <= USCRIPT_INHERITED && sc > USCRIPT_INHERITED) {
					script_code = sc;
					while (start_sp < paren_sp)
						paren_stack[++start_sp].script_code = script_code;
				}
				if ((u_getIntPropertyValue(ch, UCHAR_BIDI_PAIRED_BRACKET_TYPE) == U_BPT_CLOSE) && paren_sp >= 0) {
					paren_sp -= 1;
					start_sp -= 1;
				}
			} else {
				if (ch >= 0x10000) script_end -= 1;
				break;
			}
		}

		ScriptRange rng;
		rng.script = hb_icu_script_to_script(script_code);
		rng.start = script_start;
		rng.end = script_end;

		script_ranges.push_back(rng);
	} while (script_end < p_length);
}

#endif

/*************************************************************************/
/*  ShapedString (Base)                                                  */
/*************************************************************************/

void ShapedString::_clear_props() {

#ifdef USE_TEXT_SHAPING
	if (bidi_iter) {
		ubidi_close(bidi_iter);
		bidi_iter = NULL;
	}
	if (script_iter) {
		memdelete(script_iter);
		script_iter = NULL;
	}
#endif
	_clear_visual();
}

void ShapedString::_clear_visual() {

	valid = false;
	visual.clear();
	ascent = base_font.is_null() ? 0.0f : base_font->get_ascent();
	descent = base_font.is_null() ? 0.0f : base_font->get_descent();
	width = 0.0f;
}

#ifdef USE_TEXT_SHAPING
void ShapedString::_generate_kashida_justification_opportunies(int32_t p_start, int32_t p_end, /*out*/ Vector<JustificationOpportunity> &p_ops) const {

	int32_t kashida_pos = -1;
	int8_t priority = 100;
	int64_t i = p_start;

	uint32_t pc = 0;

	while ((p_end > p_start) && is_transparent(data[p_end - 1]))
		p_end--;

	while (i < p_end) {
		uint32_t c = data[i];

		if (c == 0x0640) {
			kashida_pos = i;
			priority = 0;
		}
		if (priority >= 1 && i < p_end - 1) {
			if (is_seen_sad(c) && (data[i + 1] != 0x200C)) {
				kashida_pos = i;
				priority = 1;
			}
		}
		if (priority >= 2 && i > p_start) {
			if (is_teh_marbuta(c) || is_dal(c) || (is_heh(c) && i == p_end - 1)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 2;
				}
			}
		}
		if (priority >= 3 && i > p_start) {
			if (is_alef(c) || ((is_lam(c) || is_tah(c) || is_kaf(c) || is_gaf(c)) && i == p_end - 1)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 3;
				}
			}
		}
		if (priority >= 4 && i > p_start && i < p_end - 1) {
			if (is_beh(c)) {
				if (is_reh(data[i + 1]) || is_yeh(data[i + 1])) {
					if (is_connected_to_prev(c, pc)) {
						kashida_pos = i - 1;
						priority = 4;
					}
				}
			}
		}
		if (priority >= 5 && i > p_start) {
			if (is_waw(c) || ((is_ain(c) || is_qaf(c) || is_feh(c)) && i == p_end - 1)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 5;
				}
			}
		}
		if (priority >= 6 && i > p_start) {
			if (is_reh(c)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 6;
				}
			}
		}
		if (!is_transparent(c)) pc = c;
		i++;
	}
	if (kashida_pos > -1) {
		JustificationOpportunity op;
		op.position = kashida_pos;
		op.kashida = true;
		p_ops.push_back(op);
	}
}
#endif

void ShapedString::_generate_justification_opportunies(int32_t p_start, int32_t p_end, const char *p_lang, /*out*/ Vector<JustificationOpportunity> &p_ops) const {

#ifdef USE_TEXT_SHAPING
	UErrorCode err = U_ZERO_ERROR;
	UBreakIterator *bi = ubrk_open(UBRK_WORD, p_lang, data + p_start, p_end - p_start, &err);
	if (U_FAILURE(err)) {
		//No data - use fallback
#endif
		_generate_justification_opportunies_fallback(p_start, p_end, p_ops);
#ifdef USE_TEXT_SHAPING
		return;
	}
	int64_t limit = 0;
	while (ubrk_next(bi) != UBRK_DONE) {
		if (ubrk_getRuleStatus(bi) != UBRK_WORD_NONE) {
			_generate_kashida_justification_opportunies(p_start + limit, p_start + ubrk_current(bi), p_ops);

			JustificationOpportunity op;
			op.position = p_start + ubrk_current(bi);
			op.kashida = false;
			p_ops.push_back(op);
			limit = ubrk_current(bi) + 1;
		}
	}
	ubrk_close(bi);
#endif
}

void ShapedString::_generate_justification_opportunies_fallback(int32_t p_start, int32_t p_end, /*out*/ Vector<JustificationOpportunity> &p_ops) const {

	int64_t limit = p_start;
	for (int64_t i = p_start; i < p_end; i++) {
#ifdef USE_TEXT_SHAPING
		if (u_isWhitespace(data[i])) {
			_generate_kashida_justification_opportunies(limit, i, p_ops);
#else
		if (is_ws(data[i])) {
#endif
			JustificationOpportunity op;
			op.position = i;
			op.kashida = false;
			p_ops.push_back(op);
			limit = i + 1;
		}
	}
#ifdef USE_TEXT_SHAPING
	_generate_kashida_justification_opportunies(limit, p_end, p_ops);
#endif
}

void ShapedString::_generate_break_opportunies(int32_t p_start, int32_t p_end, const char *p_lang, /*out*/ Vector<BreakOpportunity> &p_ops) const {

#ifdef USE_TEXT_SHAPING
	UErrorCode err = U_ZERO_ERROR;
	UBreakIterator *bi = ubrk_open(UBRK_LINE, p_lang, data + p_start, p_end - p_start, &err);
	if (U_FAILURE(err)) {
		//No data - use fallback
#endif
		_generate_break_opportunies_fallback(p_start, p_end, p_ops);
#ifdef USE_TEXT_SHAPING
		return;
	}
	while (ubrk_next(bi) != UBRK_DONE) {
		BreakOpportunity op;
		op.position = p_start + ubrk_current(bi);
		op.hard = (ubrk_getRuleStatus(bi) == UBRK_LINE_HARD);
		p_ops.push_back(op);
	}
	ubrk_close(bi);
#endif
}

void ShapedString::_generate_break_opportunies_fallback(int32_t p_start, int32_t p_end, /*out*/ Vector<BreakOpportunity> &p_ops) const {

	for (int64_t i = p_start; i < p_end; i++) {
		if (is_break(data[i])) {
			BreakOpportunity op;
			op.position = i + 1;
			op.hard = true;
			p_ops.push_back(op);
			//printf("      bopf: %d %s\n", op.position, op.hard ? "H" : "S");
#ifdef USE_TEXT_SHAPING
		} else if (u_isWhitespace(data[i])) {
#else
		} else if (is_ws(data[i])) {
#endif
			BreakOpportunity op;
			op.position = i + 1;
			op.hard = false;
			p_ops.push_back(op);
			//printf("      bopf: %d %s\n", op.position, op.hard ? "H" : "S");
		}
	}
}

void ShapedString::_shape_full_string() {

	//Already shaped, nothing to do
	if (valid) {
		return;
	}

	//Nothing to shape
	if (base_font.is_null())
		return;

	if (data_size == 0)
		return;

#ifdef USE_TEXT_SHAPING
	//Create BiDi iterator
	if (!bidi_iter) {
		UErrorCode err = U_ZERO_ERROR;
		bidi_iter = ubidi_openSized(data_size, 0, &err);
		if (U_FAILURE(err)) {
			ERR_EXPLAIN(String(u_errorName(err)));
			ERR_FAIL_COND(true);
		}
		switch (base_direction) {
			case TEXT_DIRECTION_LOCALE: {
				ubidi_setPara(bidi_iter, data, data_size, uloc_isRightToLeft(TranslationServer::get_singleton()->get_locale().ascii().get_data()) ? UBIDI_RTL : UBIDI_LTR, NULL, &err);
			} break;
			case TEXT_DIRECTION_LTR: {
				ubidi_setPara(bidi_iter, data, data_size, UBIDI_LTR, NULL, &err);
			} break;
			case TEXT_DIRECTION_RTL: {
				ubidi_setPara(bidi_iter, data, data_size, UBIDI_RTL, NULL, &err);
			} break;
			case TEXT_DIRECTION_AUTO: {
				UBiDiDirection direction = ubidi_getBaseDirection(data, data_size);
				if (direction != UBIDI_NEUTRAL) {
					ubidi_setPara(bidi_iter, data, data_size, direction, NULL, &err);
				} else {
					ubidi_setPara(bidi_iter, data, data_size, uloc_isRightToLeft(TranslationServer::get_singleton()->get_locale().ascii().get_data()) ? UBIDI_RTL : UBIDI_LTR, NULL, &err);
				}
			} break;
		}
		if (U_FAILURE(err)) {
			ERR_EXPLAIN(String(u_errorName(err)));
			ERR_FAIL_COND(true);
		}
	}

	//Create Script iterator
	if (!script_iter) {
		script_iter = memnew(ScriptIterator(data, 0, data_size));
	}

	//Find BiDi runs in visual order
	UErrorCode err = U_ZERO_ERROR;
	int bidi_run_count = ubidi_countRuns(bidi_iter, &err);
	if (U_FAILURE(err)) {
		ERR_EXPLAIN(String(u_errorName(err)));
		ERR_FAIL_COND(true);
	}

	for (int i = 0; i < bidi_run_count; i++) {
		int32_t bidi_run_start = 0;
		int32_t bidi_run_length = 0;
		hb_direction_t bidi_run_direction = (ubidi_getVisualRun(bidi_iter, i, &bidi_run_start, &bidi_run_length) == UBIDI_LTR) ? HB_DIRECTION_LTR : HB_DIRECTION_RTL;

		_shape_bidi_run(bidi_run_direction, bidi_run_start, bidi_run_start + bidi_run_length);
	}
#else
	//Single run if no real shaping happens
	_shape_bidi_run(0, 0, data_size);
#endif

	//Calculate ascent, descent and width
	if (visual.size() > 0) {
		float max_neg_offset = 0.0f;
		float max_pos_offset = 0.0f;
		float max_ascent = 0.0f;
		float max_descent = 0.0f;
		float offset = 0.0f;

		for (int64_t i = 0; i < visual.size(); i++) {
			//Calc max ascent / descent
			if (max_ascent < visual[i].ascent) {
				max_ascent = visual[i].ascent;
			}
			if (max_descent < visual[i].descent) {
				max_descent = visual[i].descent;
			}
			width += visual[i].width;
			for (int64_t j = 0; j < visual[i].glyphs.size(); j++) {
				//Calc max offsets for glyphs shifted from baseline and add glyphs width
				if (visual[i].glyphs[j].offset.y > max_pos_offset) {
					max_pos_offset = visual[i].glyphs[j].offset.y;
				}
				if (visual[i].glyphs[j].offset.y < max_neg_offset) {
					max_neg_offset = visual[i].glyphs[j].offset.y;
				}
			}
			visual.write[i].offset = offset;
			offset += visual[i].width;
		}
		ascent = MAX(max_ascent, -max_neg_offset);
		descent = MAX(max_descent, max_pos_offset);
	} else {
		if (base_font.is_valid()) {
			ascent = base_font->get_ascent();
			descent = base_font->get_descent();
		} else {
			ascent = 15.0f;
			descent = 5.0f;
		}
	}

	//Ready
	valid = true;
}

Ref<ShapedString> ShapedString::_shape_substring(int32_t p_start, int32_t p_end) const {

	Ref<ShapedString> ret;
	ret.instance();

	//Trim edge spaces
#ifdef USE_TEXT_SHAPING
	while ((p_end > p_start) && u_isWhitespace(data[p_end - 1])) {
#else
	while ((p_end > p_start) && is_ws(data[p_end - 1])) {
#endif
		p_end--;
	}

#ifdef USE_TEXT_SHAPING
	while ((p_end > p_start) && u_isWhitespace(data[p_start])) {
#else
	while ((p_end > p_start) && is_ws(data[p_start])) {
#endif
		p_start++;
	}

	//Copy substring data
#ifdef USE_TEXT_SHAPING
	ret->data = (UChar *)memalloc((p_end - p_start) * sizeof(UChar));
	memcpy(ret->data, data + p_start, (p_end - p_start) * sizeof(UChar));
	ret->data_size = (p_end - p_start);
	ret->char_size = u_countChar32(ret->data, ret->data_size);
#else
	ret->data = data.substr(p_start, p_end);
	ret->data_size = ret->data.length();
	ret->char_size = ret->data_size;
#endif
	ret->base_direction = base_direction;
	ret->base_font = base_font;
#ifdef USE_TEXT_SHAPING
	ret->language = language;
	ret->font_features = font_features;

	UErrorCode err = U_ZERO_ERROR;
	//Create temporary line bidi & shape
	ret->bidi_iter = ubidi_openSized((p_end - p_start), 0, &err);
	if (U_FAILURE(err)) {
		ret->bidi_iter = NULL;
		//Do not error out on failure - just continue with full reshapeing
	} else {
		ubidi_setLine(bidi_iter, p_start, p_end, ret->bidi_iter, &err);
		if (U_FAILURE(err)) {
			ubidi_close(ret->bidi_iter);
			ret->bidi_iter = NULL;
		}
	}
#endif
	ret->_shape_full_string();

#ifdef USE_TEXT_SHAPING
	//Close temorary line BiDi
	if (ret->bidi_iter) {
		ubidi_close(ret->bidi_iter);
		ret->bidi_iter = NULL;
	}
#endif

	return ret;
}

void ShapedString::_shape_bidi_run(hb_direction_t p_run_direction, int32_t p_run_start, int32_t p_run_end) {

#ifdef USE_TEXT_SHAPING
	//Find intersecting script runs in visual order
	script_iter->reset(p_run_direction);
	while (script_iter->next()) {
		int32_t script_run_start = script_iter->get_start();
		int32_t script_run_end = script_iter->get_end();
		hb_script_t script_run_script = script_iter->get_script();

		if ((script_run_start < p_run_end) && (script_run_end > p_run_start)) {
			_shape_bidi_script_run(p_run_direction, script_run_script, MAX(script_run_start, p_run_start), MIN(script_run_end, p_run_end), -1);
		}
	}
#else
	_shape_bidi_script_run(p_run_direction, 0, p_run_start, p_run_end, -1);
#endif
}

void ShapedString::_shape_bidi_script_run(hb_direction_t p_run_direction, hb_script_t p_run_script, int32_t p_run_start, int32_t p_run_end, int p_fallback_index) {

#ifdef USE_TEXT_SHAPING
	//Shape monotone run using HarfBuzz
	hb_font_t *hb_font = base_font->get_hb_font(p_fallback_index);
	if (!hb_font) {
		_shape_hex_run(p_run_direction, p_run_start, p_run_end);
		return;
	}
	hb_buffer_clear_contents(hb_buffer);
	hb_buffer_set_direction(hb_buffer, p_run_direction);
	hb_buffer_set_flags(hb_buffer, (hb_buffer_flags_t)(HB_BUFFER_FLAG_DEFAULT | (p_run_start == 0 ? HB_BUFFER_FLAG_BOT : 0) | (p_run_end == (int32_t)data_size ? HB_BUFFER_FLAG_EOT : 0)));
	hb_buffer_set_script(hb_buffer, p_run_script);

	if (language != HB_LANGUAGE_INVALID) hb_buffer_set_language(hb_buffer, language);

	hb_buffer_add_utf16(hb_buffer, (const uint16_t *)data, data_size, p_run_start, p_run_end - p_run_start);
	hb_shape(hb_font, hb_buffer, font_features.empty() ? NULL : font_features.ptr(), font_features.size());

	//Compose grapheme clusters
	Vector<Cluster> run_clusters;

	unsigned int glyph_count;
	hb_glyph_info_t *glyph_info = hb_buffer_get_glyph_infos(hb_buffer, &glyph_count);
	hb_glyph_position_t *glyph_pos = hb_buffer_get_glyph_positions(hb_buffer, &glyph_count);

	if (glyph_count > 0) {
		uint32_t last_cluster_id = -1;
		for (unsigned int i = 0; i < glyph_count; i++) {
			if (glyph_info[i].cluster >= data_size) {
				ERR_EXPLAIN("HarfBuzz return invalid cluster index");
				ERR_FAIL_COND(true);
			}
			if (last_cluster_id != glyph_info[i].cluster) {
				//Start new cluster
				Cluster new_cluster;

				new_cluster.fallback_depth = p_fallback_index;
				new_cluster.is_rtl = (p_run_direction == HB_DIRECTION_RTL);
				new_cluster.cl_type = _CLUSTER_TYPE_TEXT;

				new_cluster.glyphs.push_back(Glyph(glyph_info[i].codepoint, Point2((glyph_pos[i].x_offset) / 64, -(glyph_pos[i].y_offset / 64)), Point2((glyph_pos[i].x_advance) / 64, (glyph_pos[i].y_advance / 64))));
				new_cluster.valid = ((glyph_info[i].codepoint != 0) || !u_isgraph(data[glyph_info[i].cluster]));
				new_cluster.start = glyph_info[i].cluster;
				new_cluster.end = glyph_info[i].cluster;
				new_cluster.ascent = base_font->get_ascent();
				new_cluster.descent = base_font->get_descent();
				new_cluster.width += (glyph_pos[i].x_advance) / 64;

				//Set previous logical cluster end limit
				if (i != 0) {
					if (p_run_direction == HB_DIRECTION_LTR) {
						run_clusters.write[run_clusters.size() - 1].end = glyph_info[i].cluster - 1;
					} else {
						new_cluster.end = run_clusters[run_clusters.size() - 1].start - 1;
					}
				}

				run_clusters.push_back(new_cluster);

				last_cluster_id = glyph_info[i].cluster;
			} else {
				//Add glyphs to existing cluster
				run_clusters.write[run_clusters.size() - 1].glyphs.push_back(Glyph(glyph_info[i].codepoint, Point2((glyph_pos[i].x_offset) / 64, -(glyph_pos[i].y_offset / 64)), Point2((glyph_pos[i].x_advance) / 64, (glyph_pos[i].y_advance / 64))));
				run_clusters.write[run_clusters.size() - 1].valid &= ((glyph_info[i].codepoint != 0) || !u_isgraph(data[glyph_info[i].cluster]));
				run_clusters.write[run_clusters.size() - 1].width += (glyph_pos[i].x_advance) / 64;
			}
		}
		//Set last logical cluster end limit
		if (run_clusters.size() > 0) {
			if (p_run_direction == HB_DIRECTION_LTR) {
				run_clusters.write[run_clusters.size() - 1].end = p_run_end - 1;
			} else {
				run_clusters.write[0].end = p_run_end - 1;
			}
		}
	}

	//Reshape sub-runs with invalid clusters using fallback fonts
	if (run_clusters.size() > 0) {
		int32_t failed_subrun_start = p_run_end + 1;
		int32_t failed_subrun_end = p_run_start;
		for (uint16_t i = 0; i < run_clusters.size(); i++) {
			if (run_clusters[i].valid) {
				if (failed_subrun_start != p_run_end + 1) {
					if (p_fallback_index < base_font->get_fallback_count() - 1) {
						_shape_bidi_script_run(p_run_direction, p_run_script, failed_subrun_start, failed_subrun_end + 1, p_fallback_index + 1);
					} else {
						_shape_hex_run(p_run_direction, failed_subrun_start, failed_subrun_end + 1);
					}
					failed_subrun_start = p_run_end + 1;
					failed_subrun_end = p_run_start;
				}
				visual.push_back(run_clusters[i]);
			} else {
				if (failed_subrun_start >= run_clusters[i].start) failed_subrun_start = run_clusters[i].start;
				if (failed_subrun_end <= run_clusters[i].end) failed_subrun_end = run_clusters[i].end;
			}
		}
		if (failed_subrun_start != p_run_end + 1) {
			if (p_fallback_index < base_font->get_fallback_count() - 1) {
				_shape_bidi_script_run(p_run_direction, p_run_script, failed_subrun_start, failed_subrun_end + 1, p_fallback_index + 1);
			} else {
				_shape_hex_run(p_run_direction, failed_subrun_start, failed_subrun_end + 1);
			}
		}
	}
#else
	for (int i = p_run_start; i < p_run_end; i++) {
		Glyph glyph;
		glyph.codepoint = data[i];
		glyph.advance.x = base_font->get_char_size(data[i], (i < data_size - 1) ? data[i + 1] : 0).x;

		if (glyph.advance.x != 0) {
			Cluster new_cluster;
			new_cluster.fallback_depth = -1;
			new_cluster.is_rtl = (p_run_direction == HB_DIRECTION_RTL);
			new_cluster.cl_type = _CLUSTER_TYPE_TEXT;
			new_cluster.valid = true;
			new_cluster.start = i;
			new_cluster.end = i;
			new_cluster.ascent = base_font->get_ascent();
			new_cluster.descent = base_font->get_descent();
			new_cluster.width = glyph.advance.x;
			new_cluster.glyphs.push_back(glyph);
			visual.push_back(new_cluster);
		} else {
			_shape_hex_run(0, i, i);
		}
	}
#endif
}

void ShapedString::_shape_hex_run(hb_direction_t p_run_direction, int32_t p_run_start, int32_t p_run_end) {

#ifdef USE_TEXT_SHAPING
	//"Shape" monotone run using HexBox fallback
	if (p_run_direction == HB_DIRECTION_LTR) {
#endif
		for (int i = p_run_start; i < p_run_end; i++) {
			Cluster hex_cluster;
			hex_cluster.fallback_depth = -1;
			hex_cluster.cl_type = _CLUSTER_TYPE_HEX_BOX;
			hex_cluster.is_rtl = (p_run_direction == HB_DIRECTION_RTL);
			hex_cluster.valid = true;
			hex_cluster.start = i;
			hex_cluster.end = i;
			hex_cluster.ascent = 15;
			hex_cluster.descent = 5;
			hex_cluster.width = 20;
			hex_cluster.glyphs.push_back(Glyph(data[i], Point2(0, 0), Point2(20, 0)));
			visual.push_back(hex_cluster);
		}
#ifdef USE_TEXT_SHAPING
	} else {
		for (int i = p_run_end - 1; i >= p_run_start; i--) {
			Cluster hex_cluster;
			hex_cluster.fallback_depth = -1;
			hex_cluster.cl_type = _CLUSTER_TYPE_HEX_BOX;
			hex_cluster.is_rtl = (p_run_direction == HB_DIRECTION_RTL);
			hex_cluster.valid = true;
			hex_cluster.start = i;
			hex_cluster.end = i;
			hex_cluster.ascent = 15;
			hex_cluster.descent = 5;
			hex_cluster.width = 20;
			hex_cluster.glyphs.push_back(Glyph(data[i], Point2(0, 0), Point2(20, 0)));
			visual.push_back(hex_cluster);
		}
	}
#endif
}

TextDirection ShapedString::get_base_direction() const {

#ifdef USE_TEXT_SHAPING
	return base_direction;
#else
	return TEXT_DIRECTION_AUTO;
#endif
}

void ShapedString::set_base_direction(TextDirection p_base_direction) {

#ifdef USE_TEXT_SHAPING
	if (base_direction != p_base_direction) {
		base_direction = p_base_direction;
		_clear_props();
	}
#endif
}

String ShapedString::get_text() const {

#ifdef USE_TEXT_SHAPING
	String ret;
	if (data_size > 0) {
		wchar_t *_data = NULL;

		UErrorCode err = U_ZERO_ERROR;
		int32_t _length = 0;
		u_strToWCS(NULL, 0, &_length, data, data_size, &err);
		if (err != U_BUFFER_OVERFLOW_ERROR) {
			ERR_PRINTS(u_errorName(err));
			return String();
		} else {
			err = U_ZERO_ERROR;
			_data = (wchar_t *)memalloc((_length + 1) * sizeof(wchar_t));
			if (!_data)
				return String();
			memset(_data, 0x00, (_length + 1) * sizeof(wchar_t));
			u_strToWCS(_data, _length, &_length, data, data_size, &err);
			if (U_FAILURE(err)) {
				ERR_PRINTS(u_errorName(err));
				memfree(_data);
				return String();
			}
		}

		ret = String(_data);
		memfree(_data);
	}

	return ret;
#else
	return data;
#endif
}

PoolByteArray ShapedString::get_utf8() const {

	PoolByteArray ret;
#ifdef USE_TEXT_SHAPING
	if (data_size > 0) {
		UErrorCode err = U_ZERO_ERROR;
		int32_t _length = 0;
		int32_t _subs = 0;
		u_strToUTF8WithSub(NULL, 0, &_length, data, data_size, 0xFFFD, &_subs, &err);
		if (err != U_BUFFER_OVERFLOW_ERROR) {
			ERR_PRINTS(u_errorName(err));
			return ret;
		} else {
			err = U_ZERO_ERROR;
			ret.resize(_length);
			u_strToUTF8WithSub((char *)ret.write().ptr(), _length, &_length, data, data_size, 0xFFFD, &_subs, &err);
			if (U_FAILURE(err)) {
				ERR_PRINTS(u_errorName(err));
				ret.resize(0);
				return ret;
			}
		}
	}
#endif
	return ret;
}

PoolByteArray ShapedString::get_utf16() const {

	PoolByteArray ret;
#ifdef USE_TEXT_SHAPING
	if (data_size > 0) {
		ret.resize(data_size * sizeof(UChar));
		memcpy(ret.write().ptr(), data, data_size * sizeof(UChar));
	}
#endif
	return ret;
}

PoolByteArray ShapedString::get_utf32() const {

	PoolByteArray ret;
#ifdef USE_TEXT_SHAPING
	if (data_size > 0) {
		UErrorCode err = U_ZERO_ERROR;
		int32_t _length = 0;
		int32_t _subs = 0;
		u_strToUTF32WithSub(NULL, 0, &_length, data, data_size, 0xFFFD, &_subs, &err);
		if (err != U_BUFFER_OVERFLOW_ERROR) {
			ERR_PRINTS(u_errorName(err));
			return ret;
		} else {
			err = U_ZERO_ERROR;
			ret.resize(_length * sizeof(UChar32));
			u_strToUTF32WithSub((UChar32 *)ret.write().ptr(), _length, &_length, data, data_size, 0xFFFD, &_subs, &err);
			if (U_FAILURE(err)) {
				ERR_PRINTS(u_errorName(err));
				ret.resize(0);
				return ret;
			}
		}
	}
#endif
	return ret;
}

void ShapedString::set_text(const String &p_text) {

	_clear_props();

#ifdef USE_TEXT_SHAPING
	UErrorCode err = U_ZERO_ERROR;
	int64_t _length = p_text.length();
	int32_t _real_length = 0;

	const wchar_t *_data = p_text.c_str();

	//clear
	if (data) memfree(data);
	data = NULL;
	data_size = 0;
	char_size = 0;

	if (_length == 0)
		return;

	u_strFromWCS(NULL, 0, &_real_length, _data, _length, &err);
	if (err != U_BUFFER_OVERFLOW_ERROR) {
		ERR_PRINTS(u_errorName(err));
		return;
	} else {
		err = U_ZERO_ERROR;
		data = (UChar *)memalloc(_real_length * sizeof(UChar));
		u_strFromWCS(data, _real_length, &_real_length, _data, _length, &err);
		if (U_FAILURE(err)) {
			ERR_PRINTS(u_errorName(err));
			return;
		}
		data_size += _real_length;
		char_size = u_countChar32(data, data_size);
	}
#else
	data = p_text;
	data_size = p_text.length();
	char_size = data_size;
#endif
}

void ShapedString::set_utf8(const PoolByteArray p_text) {

#ifdef USE_TEXT_SHAPING
	_clear_props();

	UErrorCode err = U_ZERO_ERROR;
	int64_t _length = p_text.size();
	int32_t _real_length = 0;
	int32_t _subs = 0;

	//clear
	if (data) memfree(data);
	data = NULL;
	data_size = 0;
	char_size = 0;

	if (_length == 0)
		return;

	u_strFromUTF8WithSub(NULL, 0, &_real_length, (const char *)p_text.read().ptr(), _length, 0xFFFD, &_subs, &err);
	if (err != U_BUFFER_OVERFLOW_ERROR) {
		ERR_PRINTS(u_errorName(err));
		return;
	} else {
		err = U_ZERO_ERROR;
		data = (UChar *)memalloc(_real_length * sizeof(UChar));
		u_strFromUTF8WithSub(data, _real_length, &_real_length, (const char *)p_text.read().ptr(), _length, 0xFFFD, &_subs, &err);
		if (U_FAILURE(err)) {
			ERR_PRINTS(u_errorName(err));
			return;
		}
		data_size += _real_length;
		char_size = u_countChar32(data, data_size);
	}
#endif
}

void ShapedString::set_utf16(const PoolByteArray p_text) {

#ifdef USE_TEXT_SHAPING
	_clear_props();

	int64_t _length = p_text.size();
	int32_t _real_length = 0;

	//clear
	if (data) memfree(data);
	data = NULL;
	data_size = 0;
	char_size = 0;
	char_size = 0;

	if (_length == 0)
		return;

	_real_length = _length / sizeof(UChar);
	data = (UChar *)memalloc(_length * sizeof(UChar));
	if (!data)
		return;
	memcpy(data, p_text.read().ptr(), _length * sizeof(UChar));
	data_size += _real_length;
	char_size = u_countChar32(data, data_size);
#endif
}

void ShapedString::set_utf32(const PoolByteArray p_text) {

#ifdef USE_TEXT_SHAPING
	_clear_props();

	UErrorCode err = U_ZERO_ERROR;
	int64_t _length = p_text.size() / sizeof(UChar32);
	int32_t _real_length = 0;
	int32_t _subs = 0;

	//clear
	if (data) memfree(data);
	data = NULL;
	data_size = 0;
	char_size = 0;

	if (_length == 0)
		return;

	u_strFromUTF32WithSub(NULL, 0, &_real_length, (const UChar32 *)p_text.read().ptr(), _length, 0xFFFD, &_subs, &err);
	if (err != U_BUFFER_OVERFLOW_ERROR) {
		ERR_PRINTS(u_errorName(err));
		return;
	} else {
		err = U_ZERO_ERROR;
		data = (UChar *)memalloc(_real_length * sizeof(UChar));
		u_strFromUTF32WithSub(data, _real_length, &_real_length, (const UChar32 *)p_text.read().ptr(), _length, 0xFFFD, &_subs, &err);
		if (U_FAILURE(err)) {
			ERR_PRINTS(u_errorName(err));
			return;
		}
		data_size += _real_length;
		char_size = u_countChar32(data, data_size);
	}
#endif
}

void ShapedString::add_text(const String &p_text) {

	replace_text(data_size, data_size, p_text);
}

void ShapedString::add_utf8(const PoolByteArray p_text) {

	replace_utf8(data_size, data_size, p_text);
}

void ShapedString::add_utf16(const PoolByteArray p_text) {

	replace_utf16(data_size, data_size, p_text);
}

void ShapedString::add_utf32(const PoolByteArray p_text) {

	replace_utf32(data_size, data_size, p_text);
}

void ShapedString::replace_text(int32_t p_start, int32_t p_end, const String &p_text) {

#ifdef USE_TEXT_SHAPING
	if ((p_start > p_end) || (p_end > (int32_t)data_size)) {
		ERR_PRINTS("Invalid range");
		return;
	}

	_clear_props();

	UErrorCode err = U_ZERO_ERROR;
	int64_t _length = p_text.length();
	int32_t _real_length = 0;

	const wchar_t *_data = p_text.c_str();

	if (_length != 0) {
		u_strFromWCS(NULL, 0, &_real_length, _data, _length, &err);
		if (err != U_BUFFER_OVERFLOW_ERROR) {
			ERR_PRINTS(u_errorName(err));
			return;
		} else {
			err = U_ZERO_ERROR;
		}
	}

	UChar *new_data = (UChar *)memalloc((data_size - (p_end - p_start) + _real_length) * sizeof(UChar));
	if (!new_data)
		return;
	if (data) {
		memcpy(new_data, data, p_start * sizeof(UChar));
		memcpy(new_data + p_start + _real_length, data + p_end, (data_size - p_end) * sizeof(UChar));
		memfree(data);
	}
	data = new_data;

	u_strFromWCS(data + p_start, _real_length, &_real_length, _data, _length, &err);
	if (U_FAILURE(err)) {
		ERR_PRINTS(u_errorName(err));
		return;
	}
	data_size = data_size - (p_end - p_start) + _real_length;
	char_size = u_countChar32(data, data_size);
#else
	data = data.substr(0, p_start) + p_text + data.substr(p_end, data.length() - p_end);
	data_size = data.length();
	char_size = data_size;
#endif
}

void ShapedString::replace_utf8(int32_t p_start, int32_t p_end, const PoolByteArray p_text) {

#ifdef USE_TEXT_SHAPING
	if ((p_start > p_end) || (p_end > (int32_t)data_size)) {
		ERR_PRINTS("Invalid range");
		return;
	}

	_clear_props();

	UErrorCode err = U_ZERO_ERROR;
	int64_t _length = p_text.size();
	int32_t _real_length = 0;
	int32_t _subs = 0;

	if (_length != 0) {
		u_strFromUTF8WithSub(NULL, 0, &_real_length, (const char *)p_text.read().ptr(), _length, 0xFFFD, &_subs, &err);
		if (err != U_BUFFER_OVERFLOW_ERROR) {
			ERR_PRINTS(u_errorName(err));
			return;
		} else {
			err = U_ZERO_ERROR;
		}
	}

	UChar *new_data = (UChar *)memalloc((data_size - (p_end - p_start) + _real_length) * sizeof(UChar));
	if (!new_data)
		return;
	if (data) {
		memcpy(new_data, data, p_start * sizeof(UChar));
		memcpy(new_data + p_start + _real_length, data + p_end, (data_size - p_end) * sizeof(UChar));
		memfree(data);
	}
	data = new_data;

	u_strFromUTF8WithSub(data + p_start, _real_length, &_real_length, (const char *)p_text.read().ptr(), _length, 0xFFFD, &_subs, &err);
	if (U_FAILURE(err)) {
		ERR_PRINTS(u_errorName(err));
		return;
	}
	data_size = data_size - (p_end - p_start) + _real_length;
	char_size = u_countChar32(data, data_size);
#endif
}

void ShapedString::replace_utf16(int32_t p_start, int32_t p_end, const PoolByteArray p_text) {

#ifdef USE_TEXT_SHAPING
	if ((p_start > p_end) || (p_end > (int32_t)data_size)) {
		ERR_PRINTS("Invalid range");
		return;
	}

	_clear_props();

	int64_t _length = p_text.size();
	int32_t _real_length = 0;

	if (_length != 0) {
		_real_length = _length / sizeof(UChar);
	}

	UChar *new_data = (UChar *)memalloc((data_size - (p_end - p_start) + _real_length) * sizeof(UChar));
	if (!new_data)
		return;
	if (data) {
		memcpy(new_data, data, p_start * sizeof(UChar));
		memcpy(new_data + p_start + _real_length, data + p_end, (data_size - p_end) * sizeof(UChar));
		memfree(data);
	}
	data = new_data;

	memcpy(data + p_start, p_text.read().ptr(), _length * sizeof(UChar));

	data_size = data_size - (p_end - p_start) + _real_length;
	char_size = u_countChar32(data, data_size);
#endif
}

void ShapedString::replace_utf32(int32_t p_start, int32_t p_end, const PoolByteArray p_text) {

#ifdef USE_TEXT_SHAPING
	if ((p_start > p_end) || (p_end > (int32_t)data_size)) {
		ERR_PRINTS("Invalid range");
		return;
	}

	_clear_props();

	UErrorCode err = U_ZERO_ERROR;
	int64_t _length = p_text.size() / sizeof(UChar32);
	int32_t _real_length = 0;
	int32_t _subs = 0;

	if (_length != 0) {
		u_strFromUTF32WithSub(NULL, 0, &_real_length, (const UChar32 *)p_text.read().ptr(), _length, 0xFFFD, &_subs, &err);
		if (err != U_BUFFER_OVERFLOW_ERROR) {
			ERR_PRINTS(u_errorName(err));
			return;
		} else {
			err = U_ZERO_ERROR;
		}
	}

	UChar *new_data = (UChar *)memalloc((data_size - (p_end - p_start) + _real_length) * sizeof(UChar));
	if (!new_data)
		return;
	if (data) {
		memcpy(new_data, data, p_start * sizeof(UChar));
		memcpy(new_data + p_start + _real_length, data + p_end, (data_size - p_end) * sizeof(UChar));
		memfree(data);
	}
	data = new_data;

	u_strFromUTF32WithSub(data + p_start, _real_length, &_real_length, (const UChar32 *)p_text.read().ptr(), _length, 0xFFFD, &_subs, &err);
	if (U_FAILURE(err)) {
		ERR_PRINTS(u_errorName(err));
		return;
	}
	data_size = data_size - (p_end - p_start) + _real_length;
	char_size = u_countChar32(data, data_size);
#endif
}

Ref<Font> ShapedString::get_base_font() const {

	return base_font;
}

void ShapedString::set_base_font(const Ref<Font> &p_font) {

	if (base_font != p_font) {
		base_font = p_font;
		_clear_visual();
	}
}

String ShapedString::get_features() const {

#ifdef USE_TEXT_SHAPING
	String ret;
	char _feature[255];
	for (int i = 0; i < font_features.size(); i++) {
		hb_feature_to_string(const_cast<hb_feature_t *>(&font_features[i]), _feature, 255);
		ret += String(_feature);
		if (i != font_features.size() - 1) ret += String(",");
	}
	return ret;
#else
	return "[Not supported!]";
#endif
}

void ShapedString::set_features(const String &p_features) {

#ifdef USE_TEXT_SHAPING
	Vector<String> v_features = p_features.split(",");
	for (int i = 0; i < v_features.size(); i++) {
		hb_feature_t feature;
		if (hb_feature_from_string(v_features[i].ascii().get_data(), -1, &feature)) {
			feature.start = 0;
			feature.end = (unsigned int)-1;
			font_features.push_back(feature);
		}
	}
	_clear_visual();
#endif
}

String ShapedString::get_language() const {

#ifdef USE_TEXT_SHAPING
	return String(hb_language_to_string(language));
#else
	return "[Not supported]";
#endif
}

void ShapedString::set_language(const String &p_language) {

#ifdef USE_TEXT_SHAPING
	language = hb_language_from_string(p_language.ascii().get_data(), -1);
	_clear_visual();
#endif
}

bool ShapedString::is_valid() const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	return valid;
}

bool ShapedString::empty() const {

	return data_size == 0;
}

int ShapedString::length() const {

	return char_size;
}

float ShapedString::get_ascent() const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return base_font.is_null() ? 0.0f : base_font->get_ascent();

	return ascent;
}

float ShapedString::get_descent() const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return base_font.is_null() ? 0.0f : base_font->get_descent();

	return descent;
}

float ShapedString::get_width() const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return 0.0f;

	return width;
}

float ShapedString::get_height() const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid) {
		return base_font.is_null() ? 0.0f : base_font->get_height();
	}

	return ascent + descent;
}

Vector<int> ShapedString::break_lines(float p_width, TextBreak p_flags) const {

	Vector<int> ret;

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return ret;

	if (p_flags == TEXT_BREAK_NONE) {
		ret.push_back(_offset_to_codepoint(data_size));
		return ret;
	}

	//Find safe break points
	Vector<BreakOpportunity> brk_ops;
#ifdef USE_TEXT_SHAPING
	_generate_break_opportunies(0, data_size, hb_language_to_string(language), brk_ops);
#else
	_generate_break_opportunies(0, data_size, "", brk_ops);
#endif

	//Sort clusters in logical order
	Vector<Cluster> logical = visual;
	logical.sort_custom<ClusterCompare>();

	//Break lines
	float width = 0.0f;
	int line_start = 0;

	int last_safe_brk = -1;
	int last_safe_brk_cluster = -1;

	int b = 0;
	int i = 0;
	while (i < logical.size()) {
		if ((b < brk_ops.size()) && (brk_ops[b].position == logical[i].start)) {
			last_safe_brk = b;
			last_safe_brk_cluster = i;
			b++;
			if (brk_ops[last_safe_brk].hard) {
				ret.push_back(_offset_to_codepoint(logical[i].end));

				width = 0.0f;
				line_start = logical[i].end;
				last_safe_brk = -1;
				last_safe_brk_cluster = -1;
				i++;
				continue;
			}
		}
		width += logical[i].width;
		if (p_flags == TEXT_BREAK_MANDATORY_AND_WORD_BOUND) {
			if ((p_width > 0) && (width >= p_width) && (last_safe_brk != -1) && (brk_ops[last_safe_brk].position != line_start)) {
				ret.push_back(_offset_to_codepoint(brk_ops[last_safe_brk].position));

				width = 0.0f;
				i = last_safe_brk_cluster;
				line_start = brk_ops[last_safe_brk].position;
				last_safe_brk = -1;
				last_safe_brk_cluster = -1;
				continue;
			}
		} else if (p_flags == TEXT_BREAK_MANDATORY_AND_ANYWHERE) {
			if ((p_width > 0) && (width >= p_width)) {
				ret.push_back(_offset_to_codepoint(logical[i].end));

				width = 0.0f;
				line_start = logical[i].end;
				last_safe_brk = -1;
				last_safe_brk_cluster = -1;
				i++;
				continue;
			}
		}
		i++;
	}
	if (line_start < (int32_t)data_size) {
		//Shape clusters after last safe break
		ret.push_back(_offset_to_codepoint(data_size));
	}

	return ret;
}

Ref<ShapedString> ShapedString::substr(int p_start, int p_end) const {

	int _from = _codepoint_to_offset(p_start);
	int _to = _codepoint_to_offset(p_end);

	if ((data_size == 0) || (p_start < 0) || (p_end > (int32_t)data_size) || (p_start > p_end)) {
		Ref<ShapedString> ret;
		ret.instance();

		return ret;
	}

	return _shape_substring(_from, _to);
}

#ifdef USE_TEXT_SHAPING
void ShapedString::_shape_single_cluster(int32_t p_start, int32_t p_end, hb_direction_t p_run_direction, hb_script_t p_run_script, UChar32 p_codepoint, int p_fallback_index, /*out*/ Cluster &p_cluster) const {

	//Shape single cluster using HarfBuzz
	hb_font_t *hb_font = base_font->get_hb_font(p_fallback_index);
	if (!hb_font) {
		return;
	}
	hb_buffer_clear_contents(hb_buffer);
	hb_buffer_set_direction(hb_buffer, p_run_direction);
	hb_buffer_set_flags(hb_buffer, (hb_buffer_flags_t)(HB_BUFFER_FLAG_DEFAULT));
	hb_buffer_set_script(hb_buffer, p_run_script);

	if (language != HB_LANGUAGE_INVALID) hb_buffer_set_language(hb_buffer, language);

	hb_buffer_add_utf32(hb_buffer, (const uint32_t *)&p_codepoint, 1, 0, 1);
	hb_shape(hb_font, hb_buffer, font_features.empty() ? NULL : font_features.ptr(), font_features.size());

	unsigned int glyph_count;
	hb_glyph_info_t *glyph_info = hb_buffer_get_glyph_infos(hb_buffer, &glyph_count);
	hb_glyph_position_t *glyph_pos = hb_buffer_get_glyph_positions(hb_buffer, &glyph_count);

	p_cluster.glyphs.clear();
	p_cluster.fallback_depth = p_fallback_index;
	p_cluster.is_rtl = (p_run_direction == HB_DIRECTION_RTL);
	p_cluster.cl_type = _CLUSTER_TYPE_TEXT;
	p_cluster.valid = true;
	p_cluster.start = p_start;
	p_cluster.end = p_end;
	p_cluster.ascent = base_font->get_ascent();
	p_cluster.descent = base_font->get_descent();
	p_cluster.width = 0.0f;

	if (glyph_count > 0) {
		for (unsigned int i = 0; i < glyph_count; i++) {
			p_cluster.glyphs.push_back(Glyph(glyph_info[i].codepoint, Point2((glyph_pos[i].x_offset) / 64, -(glyph_pos[i].y_offset / 64)), Point2((glyph_pos[i].x_advance) / 64, (glyph_pos[i].y_advance / 64))));
			p_cluster.valid &= ((glyph_info[i].codepoint != 0) || !u_isgraph(p_codepoint));
			p_cluster.width += (glyph_pos[i].x_advance) / 64;
		}
	}
	if (!p_cluster.valid) {
		if (p_fallback_index < base_font->get_fallback_count() - 1) {
			_shape_single_cluster(p_start, p_end, p_run_direction, p_run_script, p_codepoint, p_fallback_index + 1, p_cluster);
		}
	}
}
#endif

float ShapedString::extend_to_width(float p_width, TextJustification p_flags) {

	if (!valid)
		_shape_full_string();

	if (!valid)
		return width;

	if (p_flags == TEXT_JUSTIFICATION_NONE)
		return width;

	//Nothing to do
	if (width >= p_width)
		return width;

	//Find safe justification points
	Vector<JustificationOpportunity> jst_ops;
#ifdef USE_TEXT_SHAPING
	_generate_justification_opportunies(0, data_size, hb_language_to_string(language), jst_ops);
#else
	_generate_justification_opportunies(0, data_size, "", jst_ops);
#endif

#ifdef USE_TEXT_SHAPING
	int ks_count = 0;
#endif
	int ws_count = 0;
	for (int i = 0; i < jst_ops.size(); i++) {
		if ((jst_ops[i].position <= 0) || jst_ops[i].position >= (int32_t)data_size - 1)
			continue;
#ifdef USE_TEXT_SHAPING
		if (jst_ops[i].kashida) {
			ks_count++;
		}
#endif
		if (!jst_ops[i].kashida) {
			ws_count++;
		}
	}

#ifdef USE_TEXT_SHAPING
	//Step 1: Kashida justification
	if ((p_flags == TEXT_JUSTIFICATION_KASHIDA_AND_WHITESPACE) || (p_flags == TEXT_JUSTIFICATION_KASHIDA_ONLY)) {
		Cluster ks_cluster;
		float ks_width_per_op = (p_width - width) / ks_count;
		for (int i = 0; i < jst_ops.size(); i++) {
			if ((jst_ops[i].position <= 0) || jst_ops[i].position >= (int32_t)data_size - 1)
				continue;
			if (jst_ops[i].kashida) {
				int j = 0;
				while (j < visual.size()) {
					if (visual[j].start == jst_ops[i].position) {
						_shape_single_cluster(visual[j].start, visual[j].start, HB_DIRECTION_RTL, HB_SCRIPT_ARABIC, 0x0640, -1, ks_cluster);

						//Add new kashda multiple times
						int ks_count_per_op = ks_width_per_op / ks_cluster.width;
						for (int k = 0; k < ks_count_per_op; k++) {
							visual.insert(j, ks_cluster);
							j++;
						}

						width += ks_count_per_op * ks_cluster.width;
					}
					j++;
				}
			}
		}
	}
#endif

	//Step 2: Whitespace justification
	if ((p_flags == TEXT_JUSTIFICATION_KASHIDA_AND_WHITESPACE) || (p_flags == TEXT_JUSTIFICATION_WHITESPACE_ONLY)) {
		float ws_width_per_op = (p_width - width) / ws_count;

		Cluster ws_cluster;
		ws_cluster.fallback_depth = -1;
		ws_cluster.cl_type = _CLUSTER_TYPE_TEXT;
		ws_cluster.valid = true;
		ws_cluster.ascent = base_font->get_ascent();
		ws_cluster.descent = base_font->get_descent();
		ws_cluster.width = ws_width_per_op;
		ws_cluster.glyphs.push_back(Glyph(0, Point2(), Point2(ws_width_per_op, 0)));

		for (int i = 0; i < jst_ops.size(); i++) {
			if ((jst_ops[i].position <= 0) || jst_ops[i].position >= (int32_t)data_size - 1)
				continue;
			if (!jst_ops[i].kashida) {
				int j = 0;
				while (j < visual.size()) {
					if (visual[j].start == jst_ops[i].position) {
#ifdef USE_TEXT_SHAPING
						if ((visual[j].glyphs.size() == 1) && u_isWhitespace(data[visual[j].start])) {
#else
						if ((visual[j].glyphs.size() == 1) && is_ws(data[visual[j].start])) {
#endif
							//Extend existing whitespace
							visual.write[j].glyphs.write[0].advance.x += ws_width_per_op;
						} else {
							//Add new whitespace
							ws_cluster.is_rtl = visual[j].is_rtl;
							if (visual[j].is_rtl) {
								ws_cluster.start = visual[j].end + 1;
								ws_cluster.end = visual[j].end + 1;

								j++;
								visual.insert(j, ws_cluster);
								j++;
							} else {
								ws_cluster.start = visual[j].start;
								ws_cluster.end = visual[j].start;

								visual.insert(j, ws_cluster);
								j++;
							}
						}
						width += ws_width_per_op;
					}
					j++;
				}
			}
		}
	}

	return width;
}

float ShapedString::collapse_to_width(float p_width, TextJustification p_flags) {

	if (!valid)
		_shape_full_string();

	if (!valid)
		return width;

	if (p_flags == TEXT_JUSTIFICATION_NONE)
		return width;

	//Nothing to do
	if (width <= p_width)
		return width;

		//Find existing spaces and kashidas
#ifdef USE_TEXT_SHAPING
	int ks_count = 0;
#endif
	int ws_count = 0;

	for (int i = 0; i < (int32_t)data_size; i++) {
#ifdef USE_TEXT_SHAPING
		if (u_isWhitespace(data[i])) {
			ws_count++;
		}
		if (data[i] == 0x0640) {
			ks_count++;
		}
#else
		if (is_ws(data[i])) {
			ws_count++;
		}
#endif
	}

#ifdef USE_TEXT_SHAPING
	//Step 1: Remove kasidas
	if ((p_flags == TEXT_JUSTIFICATION_KASHIDA_AND_WHITESPACE) || (p_flags == TEXT_JUSTIFICATION_KASHIDA_ONLY)) {
		float ks_width_per_op = (width - p_width) / ks_count;
		int j = 0;
		while (j < visual.size()) {
			if ((visual[j].glyphs.size() == 1) && data[visual[j].start] == 0x0640) {
				if (visual[j].width >= ks_width_per_op) {
					width -= visual[j].width;
					visual.remove(j);
				}
			}
			j++;
		}
	}
#endif

	//Step 2: Collapse spaces (no 0.2 of original width)
	if ((p_flags == TEXT_JUSTIFICATION_KASHIDA_AND_WHITESPACE) || (p_flags == TEXT_JUSTIFICATION_WHITESPACE_ONLY)) {
		float ws_width_per_op = (width - p_width) / ws_count;
		int j = 0;
		while (j < visual.size()) {
#ifdef USE_TEXT_SHAPING
			if ((visual[j].glyphs.size() == 1) && u_isWhitespace(data[visual[j].start])) {
#else
			if ((visual[j].glyphs.size() == 1) && is_ws(data[visual[j].start])) {
#endif
				if (visual[j].width > ws_width_per_op) {
					visual.write[j].glyphs.write[0].advance.x -= ws_width_per_op;
					visual.write[j].width -= ws_width_per_op;
					width -= ws_width_per_op;
				}
			}
			j++;
		}
	}

	return width;
}

int ShapedString::get_cluster_index(int p_position) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return -1;

	int _from = _codepoint_to_offset(p_position);

	for (int i = 0; i < visual.size(); i++) {
		//printf("$$$ [%d %d] %d\n", visual[i].start, visual[i].end, _from);
		if (visual[i].start == _from) {
			return i;
		}
	}
	return -1;
}

Rect2 ShapedString::get_cluster_rect(int p_index) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return Rect2();

	if ((p_index < 0) || (p_index >= visual.size()))
		return Rect2();

	return Rect2(Point2(visual[p_index].offset, -visual[p_index].ascent), Size2(visual[p_index].width, visual[p_index].ascent + visual[p_index].descent));
}

float ShapedString::get_cluster_leading_edge(int p_index) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return 0.0f;

	if ((p_index < 0) || (p_index >= visual.size()))
		return 0.0f;

	float ret = visual[p_index].offset;
	if (!visual[p_index].is_rtl) {
		ret += visual[p_index].width;
	}
	return ret;
}

float ShapedString::get_cluster_trailing_edge(int p_index) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return 0.0f;

	if ((p_index < 0) || (p_index >= visual.size()))
		return 0.0f;

	float ret = visual[p_index].offset;
	if (visual[p_index].is_rtl) {
		ret += visual[p_index].width;
	}
	return ret;
}

int ShapedString::get_cluster_start(int p_index) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return -1;

	if ((p_index < 0) || (p_index >= visual.size()))
		return -1;

	return _offset_to_codepoint(visual[p_index].start);
}

int ShapedString::get_cluster_end(int p_index) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return -1;

	if ((p_index < 0) || (p_index >= visual.size()))
		return -1;

	return _offset_to_codepoint(visual[p_index].end);
}

float ShapedString::get_cluster_ascent(int p_index) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return 0.0f;

	if ((p_index < 0) || (p_index >= visual.size()))
		return 0.0f;

	return visual[p_index].ascent;
}

float ShapedString::get_cluster_descent(int p_index) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return 0.0f;

	if ((p_index < 0) || (p_index >= visual.size()))
		return 0.0f;

	return visual[p_index].descent;
}

float ShapedString::get_cluster_width(int p_index) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return 0.0f;

	if ((p_index < 0) || (p_index >= visual.size()))
		return 0.0f;

	return visual[p_index].width;
}

float ShapedString::get_cluster_height(int p_index) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return 0.0f;

	if ((p_index < 0) || (p_index >= visual.size()))
		return 0.0f;

	return visual[p_index].ascent + visual[p_index].descent;
}

Vector<Rect2> ShapedString::get_highlight_shapes(int p_start, int p_end) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return Vector<Rect2>();

	Vector<Rect2> ret;

	//Find safe UTF-16 char bounds
	int _from = _codepoint_to_offset(p_start);
	int _to = _codepoint_to_offset(p_end);

	float prev = 0.0f;
	for (int i = 0; i < visual.size(); i++) {
		float width = 0.0f;

		width += visual[i].width;

		if ((visual[i].start >= _from) && (visual[i].end < _to)) {
			ret.push_back(Rect2(prev, descent, width, descent + ascent));
		} else if ((visual[i].start < _from) && (visual[i].end >= _to)) {
			float char_width = visual[i].width / (visual[i].end + 1 - visual[i].start);
			int pos_ofs_s = _from - visual[i].start;
			int pos_ofs_e = _to - visual[i].start;
			ret.push_back(Rect2(prev + pos_ofs_s * char_width, descent, (pos_ofs_e - pos_ofs_s) * char_width, descent + ascent));
		} else if ((visual[i].start < _from) && (visual[i].end >= _from)) {
			float char_width = visual[i].width / (visual[i].end + 1 - visual[i].start);
			int pos_ofs = _from - visual[i].start;
			if (visual[i].is_rtl) {
				ret.push_back(Rect2(prev, descent, pos_ofs * char_width, descent + ascent));
			} else {
				ret.push_back(Rect2(prev + pos_ofs * char_width, descent, width - pos_ofs * char_width, descent + ascent));
			}
		} else if ((visual[i].start < _to) && (visual[i].end >= _to)) {
			float char_width = visual[i].width / (visual[i].end + 1 - visual[i].start);
			int pos_ofs = _to - visual[i].start;
			if (visual[i].is_rtl) {
				ret.push_back(Rect2(prev + pos_ofs * char_width, descent, width - pos_ofs * char_width, descent + ascent));
			} else {
				ret.push_back(Rect2(prev, descent, pos_ofs * char_width, descent + ascent));
			}
		}
		prev += width;
	}

	//merge intersectiong ranges
	int i = 0;
	while (i < ret.size()) {
		int j = i + 1;
		while (j < ret.size()) {
			if (ret[i].position.x + ret[i].size.x == ret[j].position.x) {
				ret.write[i].size.x += ret[j].size.x;
				ret.remove(j);
				continue;
			}
			j++;
		}
		i++;
	}
	return ret;
}

Vector<float> ShapedString::get_cursor_positions(int p_position, TextDirection p_primary_dir) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return Vector<float>();

	Vector<float> ret;

	//Find safe UTF-16 char bounds
	int _from = _codepoint_to_offset(p_position);

	int leading_cluster = -1;
	int trailing_cluster = -1;
	int mid_cluster = -1;

	if ((_from == 0) && (data_size == 0)) {
		//no data - cusror at start
		ret.push_back(0.0f);
		return ret;
	}
	if (_from > (int32_t)data_size) {
		//cursor after last char
		ret.push_back(get_cluster_leading_edge(data_size - 1));
		return ret;
	}

	for (int i = 0; i < visual.size(); i++) {
		if (((visual[i].start >= _from) && (visual[i].end <= _from)) || (visual[i].start == _from)) {
			trailing_cluster = i;
		} else if (((visual[i].start >= _from - 1) && (visual[i].end <= _from - 1)) || (visual[i].end == _from - 1)) {
			leading_cluster = i;
		} else if ((visual[i].start <= _from) && (visual[i].end >= _from)) {
			mid_cluster = i;
		}
	}
	if ((leading_cluster != -1) && (trailing_cluster != -1)) {
		if ((p_primary_dir == TEXT_DIRECTION_RTL) && (visual[trailing_cluster].is_rtl)) {
			ret.push_back(get_cluster_leading_edge(leading_cluster));
			ret.push_back(get_cluster_trailing_edge(trailing_cluster));
		} else {
			ret.push_back(get_cluster_trailing_edge(trailing_cluster));
			ret.push_back(get_cluster_leading_edge(leading_cluster));
		}
	} else {
		if (leading_cluster != -1) ret.push_back(get_cluster_leading_edge(leading_cluster));
		if (trailing_cluster != -1) ret.push_back(get_cluster_trailing_edge(trailing_cluster));
	}

	if (mid_cluster != -1) {
		float char_width = visual[mid_cluster].width / (visual[mid_cluster].end + 1 - visual[mid_cluster].start);
		int pos_ofs = _from - visual[mid_cluster].start;
		if (visual[mid_cluster].is_rtl) {
			ret.push_back(get_cluster_trailing_edge(mid_cluster) - pos_ofs * char_width);
		} else {
			ret.push_back(get_cluster_trailing_edge(mid_cluster) + pos_ofs * char_width);
		}
	}

	return ret;
}

TextDirection ShapedString::get_char_direction(int p_position) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return TEXT_DIRECTION_LTR;

	//Find safe UTF-16 char bounds
	int _from = _codepoint_to_offset(p_position);

	for (int i = 0; i < visual.size(); i++) {
		if ((_from >= visual[i].start) && (_from <= visual[i].end)) {
			return visual[i].is_rtl ? TEXT_DIRECTION_RTL : TEXT_DIRECTION_LTR;
		}
	}
	return TEXT_DIRECTION_LTR;
}

int ShapedString::_codepoint_to_offset(int p_position) const {

#ifdef USE_TEXT_SHAPING
	int _from = 0;
	if ((sizeof(wchar_t) == 4) && (char_size != data_size)) {
		U16_FWD_N(data, _from, (int32_t)data_size, p_position);
	} else {
		_from = p_position;
	}
	return _from;
#else
	return p_position;
#endif
}

int ShapedString::_offset_to_codepoint(int p_position) const {

#ifdef USE_TEXT_SHAPING
	if ((sizeof(wchar_t) == 4) && (char_size != data_size)) {
		int i = 0, c = 0;
		while (i < (int32_t)data_size) {
			if (U16_IS_LEAD(*(data + i))) {
				i++;
			}
			if (i >= p_position)
				break;
			i++;
			c++;
		}
		return c;
	} else {
		return p_position;
	}
#else
	return p_position;
#endif
}

int ShapedString::hit_test(float p_position) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return -1;

	if (visual.size() == 0) {
		return -1;
	}

	if (p_position < 0) {
		//Hit before first cluster
		if (visual[0].is_rtl) {
			return _offset_to_codepoint(visual[0].end + 1);
		} else {
			return _offset_to_codepoint(visual[0].start);
		}
	}

	float offset = 0.0f;
	for (int i = 0; i < visual.size(); i++) {
		if ((p_position >= offset) && (p_position <= offset + visual[i].width)) {
			//Dircet hit
			float char_width = visual[i].width / (visual[i].end + 1 - visual[i].start);
			int pos_ofs = Math::round((p_position - offset) / char_width);
			if (visual[i].is_rtl) {
				return _offset_to_codepoint(visual[i].end + 1 - pos_ofs);
			} else {
				return _offset_to_codepoint(pos_ofs + visual[i].start);
			}
		}
		offset += visual[i].width;
	};

	//Hit after last cluster
	if (visual[visual.size() - 1].is_rtl) {
		return _offset_to_codepoint(visual[visual.size() - 1].start);
	} else {
		return _offset_to_codepoint(visual[visual.size() - 1].end + 1);
	}
}

int ShapedString::clusters() const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return 0;

	return visual.size();
}

Vector2 ShapedString::draw_cluster(RID p_canvas_item, const Point2 &p_position, int p_index, const Color &p_modulate, bool p_outline) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return Vector2();

	if ((p_index < 0) || (p_index >= visual.size()))
		return Vector2();

	Vector2 ofs;
	for (int i = 0; i < visual[p_index].glyphs.size(); i++) {
		if (visual[p_index].cl_type == (int)_CLUSTER_TYPE_HEX_BOX) {
			Font::draw_hex_box(p_canvas_item, p_position + ofs - Point2(0, visual[p_index].ascent), visual[p_index].glyphs[i].codepoint, p_modulate);
		} else if (visual[p_index].cl_type == (int)_CLUSTER_TYPE_TEXT) {
			base_font->draw_glyph(p_canvas_item, p_position + ofs, visual[p_index].glyphs[i].codepoint, visual[p_index].glyphs[i].offset, visual[p_index].ascent, p_modulate, p_outline, visual[p_index].fallback_depth);
		} else {
			WARN_PRINTS("Invalid cluster type")
		}
		ofs += visual[p_index].glyphs[i].advance;
	}

	return ofs;
}

void ShapedString::draw(RID p_canvas_item, const Point2 &p_position, const Color &p_modulate, bool p_outline) const {

	if (!valid)
		const_cast<ShapedString *>(this)->_shape_full_string();

	if (!valid)
		return;

#ifdef DEBUG_DISPLAY_METRICS
	VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + Point2(0, -ascent), p_position + Point2(width, -ascent), Color(1, 0, 0, 0.5), 1);
	VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + Point2(0, 0), p_position + Point2(width, 0), Color(1, 1, 0, 0.5), 1);
	VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + Point2(0, descent), p_position + Point2(width, descent), Color(0, 0, 1, 0.5), 1);
#endif

	Vector2 ofs;
	for (int i = 0; i < visual.size(); i++) {
		for (int j = 0; j < visual[i].glyphs.size(); j++) {
			if (visual[i].cl_type == (int)_CLUSTER_TYPE_HEX_BOX) {
				Font::draw_hex_box(p_canvas_item, p_position + ofs - Point2(0, visual[i].ascent), visual[i].glyphs[j].codepoint, p_modulate);
			} else if (visual[i].cl_type == (int)_CLUSTER_TYPE_TEXT) {
				base_font->draw_glyph(p_canvas_item, p_position + ofs, visual[i].glyphs[j].codepoint, visual[i].glyphs[j].offset, visual[i].ascent, p_modulate, p_outline, visual[i].fallback_depth);
			} else {
				WARN_PRINTS("Invalid cluster type")
			}
			ofs += visual[i].glyphs[j].advance;
		}
	}
}

Array ShapedString::_break_lines(float p_width, TextBreak p_flags) const {

	Array ret;

	Vector<int> lines = break_lines(p_width, p_flags);
	for (int i = 0; i < lines.size(); i++) {
		ret.push_back(lines[i]);
	}

	return ret;
}

Array ShapedString::_get_highlight_shapes(int p_start, int p_end) const {

	Array ret;
	Vector<Rect2> rects = get_highlight_shapes(p_start, p_end);
	for (int i = 0; i < rects.size(); i++) {
		ret.push_back(rects[i]);
	}

	return ret;
}

Array ShapedString::_get_cursor_positions(int p_position, TextDirection p_primary_dir) const {

	Array ret;
	Vector<float> cpos = get_cursor_positions(p_position, p_primary_dir);
	for (int i = 0; i < cpos.size(); i++) {
		ret.push_back(cpos[i]);
	}

	return ret;
}

void ShapedString::_bind_methods() {

	//Input
	ClassDB::bind_method(D_METHOD("set_base_direction", "direction"), &ShapedString::set_base_direction);
	ClassDB::bind_method(D_METHOD("get_base_direction"), &ShapedString::get_base_direction);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "base_direction", PROPERTY_HINT_ENUM, "LTR,RTL,Locale,Auto"), "set_base_direction", "get_base_direction");

	ClassDB::bind_method(D_METHOD("set_text", "text"), &ShapedString::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &ShapedString::get_text);
	ClassDB::bind_method(D_METHOD("add_text", "text"), &ShapedString::add_text);
	ClassDB::bind_method(D_METHOD("replace_text", "start", "end", "text"), &ShapedString::replace_text);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text"), "set_text", "get_text");

	ClassDB::bind_method(D_METHOD("get_utf8"), &ShapedString::get_utf8);
	ClassDB::bind_method(D_METHOD("set_utf8", "data"), &ShapedString::set_utf8);
	ClassDB::bind_method(D_METHOD("add_utf8", "text"), &ShapedString::add_utf8);
	ClassDB::bind_method(D_METHOD("replace_utf8", "start", "end", "text"), &ShapedString::replace_utf8);

	ClassDB::bind_method(D_METHOD("get_utf16"), &ShapedString::get_utf16);
	ClassDB::bind_method(D_METHOD("set_utf16", "data"), &ShapedString::set_utf16);
	ClassDB::bind_method(D_METHOD("add_utf16", "text"), &ShapedString::add_utf16);
	ClassDB::bind_method(D_METHOD("replace_utf16", "start", "end", "text"), &ShapedString::replace_utf16);

	ClassDB::bind_method(D_METHOD("get_utf32"), &ShapedString::get_utf32);
	ClassDB::bind_method(D_METHOD("set_utf32", "data"), &ShapedString::set_utf32);
	ClassDB::bind_method(D_METHOD("add_utf32", "text"), &ShapedString::add_utf32);
	ClassDB::bind_method(D_METHOD("replace_utf32", "start", "end", "text"), &ShapedString::replace_utf32);

	ClassDB::bind_method(D_METHOD("set_base_font", "font"), &ShapedString::set_base_font);
	ClassDB::bind_method(D_METHOD("get_base_font"), &ShapedString::get_base_font);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "base_font", PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_base_font", "get_base_font");

	ClassDB::bind_method(D_METHOD("set_features", "features"), &ShapedString::set_features);
	ClassDB::bind_method(D_METHOD("get_features"), &ShapedString::get_features);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "features"), "set_features", "get_features");

	ClassDB::bind_method(D_METHOD("set_language", "language"), &ShapedString::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &ShapedString::get_language);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");

	//Line data
	ClassDB::bind_method(D_METHOD("is_valid"), &ShapedString::is_valid);
	ClassDB::bind_method(D_METHOD("empty"), &ShapedString::empty);
	ClassDB::bind_method(D_METHOD("length"), &ShapedString::length);

	ClassDB::bind_method(D_METHOD("get_ascent"), &ShapedString::get_ascent);
	ClassDB::bind_method(D_METHOD("get_descent"), &ShapedString::get_descent);
	ClassDB::bind_method(D_METHOD("get_width"), &ShapedString::get_width);
	ClassDB::bind_method(D_METHOD("get_height"), &ShapedString::get_height);

	//Line modification
	ClassDB::bind_method(D_METHOD("break_lines", "width", "flags"), &ShapedString::_break_lines);
	ClassDB::bind_method(D_METHOD("substr", "start", "end"), &ShapedString::substr);
	ClassDB::bind_method(D_METHOD("extend_to_width", "width", "flags"), &ShapedString::extend_to_width);
	ClassDB::bind_method(D_METHOD("collapse_to_width", "width", "flags"), &ShapedString::collapse_to_width);

	//Cluster data
	ClassDB::bind_method(D_METHOD("clusters"), &ShapedString::clusters);
	ClassDB::bind_method(D_METHOD("get_cluster_index", "starrt_position"), &ShapedString::get_cluster_index);
	ClassDB::bind_method(D_METHOD("get_cluster_trailing_edge", "index"), &ShapedString::get_cluster_trailing_edge);
	ClassDB::bind_method(D_METHOD("get_cluster_leading_edge", "index"), &ShapedString::get_cluster_leading_edge);
	ClassDB::bind_method(D_METHOD("get_cluster_start", "index"), &ShapedString::get_cluster_start);
	ClassDB::bind_method(D_METHOD("get_cluster_end", "index"), &ShapedString::get_cluster_end);
	ClassDB::bind_method(D_METHOD("get_cluster_ascent", "index"), &ShapedString::get_cluster_ascent);
	ClassDB::bind_method(D_METHOD("get_cluster_descent", "index"), &ShapedString::get_cluster_descent);
	ClassDB::bind_method(D_METHOD("get_cluster_width", "index"), &ShapedString::get_cluster_width);
	ClassDB::bind_method(D_METHOD("get_cluster_height", "index"), &ShapedString::get_cluster_height);
	ClassDB::bind_method(D_METHOD("get_cluster_rect", "index"), &ShapedString::get_cluster_rect);

	//Output
	ClassDB::bind_method(D_METHOD("get_highlight_shapes", "start", "end"), &ShapedString::_get_highlight_shapes);
	ClassDB::bind_method(D_METHOD("get_cursor_positions", "position", "flags"), &ShapedString::_get_cursor_positions);
	ClassDB::bind_method(D_METHOD("get_char_direction", "position"), &ShapedString::get_char_direction);
	ClassDB::bind_method(D_METHOD("hit_test", "position"), &ShapedString::hit_test);

	ClassDB::bind_method(D_METHOD("draw_cluster", "canvas_item", "position", "index", "modulate", "outline"), &ShapedString::draw_cluster, DEFVAL(Color(1, 1, 1)), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw", "canvas_item", "position", "modulate", "outline"), &ShapedString::draw, DEFVAL(Color(1, 1, 1)), DEFVAL(false));
}

ShapedString::ShapedString() {

	valid = false;
#ifdef USE_TEXT_SHAPING
	data = NULL;
	bidi_iter = NULL;
	script_iter = NULL;
	hb_buffer = hb_buffer_create();
	language = hb_language_from_string(TranslationServer::get_singleton()->get_locale().ascii().get_data(), -1);
#endif
	ascent = 0.0f;
	descent = 0.0f;
	width = 0.0f;
	data_size = 0;
	char_size = 0;
	base_direction = TEXT_DIRECTION_AUTO;
}

ShapedString::~ShapedString() {

#ifdef USE_TEXT_SHAPING
	if (bidi_iter) {
		ubidi_close(bidi_iter);
		bidi_iter = NULL;
	}

	if (script_iter) {
		memdelete(script_iter);
		script_iter = NULL;
	}

	if (hb_buffer) {
		hb_buffer_destroy(hb_buffer);
		hb_buffer = NULL;
	}

	if (data) {
		memfree(data);
	}
#endif
}
