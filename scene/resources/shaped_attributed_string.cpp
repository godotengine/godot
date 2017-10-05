/*************************************************************************/
/*  shaped_attributed_string.cpp                                         */
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

#include "scene/resources/shaped_attributed_string.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_scale.h"
#endif

#define DEBUG_DISPLAY_METRICS 1

/*************************************************************************/
/*  ShapedAttributedString                                               */
/*************************************************************************/

Ref<ShapedString> ShapedAttributedString::_shape_substring(int32_t p_start, int32_t p_end) const {

	Ref<ShapedAttributedString> ret;
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
#else
	ret->data = data.substr(p_start, p_end);
	ret->data_size = ret->data.length();
#endif
	ret->base_direction = base_direction;
	ret->base_font = base_font;
#ifdef USE_TEXT_SHAPING
	ret->language = language;
	ret->font_features = font_features;
#endif

	//Copy attributes
	const Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find_closest(p_start);
	while ((attrib) && (attrib->key() < p_end)) {
		ret->attributes[MAX(0, attrib->key() - p_start)] = attrib->get();
		attrib = attrib->next();
	}

#ifdef USE_TEXT_SHAPING
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

#ifdef USE_TEXT_SHAPING
void ShapedAttributedString::_shape_single_cluster(int32_t p_start, int32_t p_end, hb_direction_t p_run_direction, hb_script_t p_run_script, UChar32 p_codepoint, int p_fallback_index, /*out*/ Cluster &p_cluster) const {

	const Map<int, Map<TextAttribute, Variant> >::Element *attrib_iter = (p_run_direction == HB_DIRECTION_LTR) ? attributes.find_closest(p_start) : attributes.find_closest(p_end - 1);
	if (!attrib_iter) {
		//Shape as plain string
		ShapedString::_shape_single_cluster(p_start, p_end, p_run_direction, p_run_script, p_codepoint, p_fallback_index, p_cluster);
		return;
	}

	//Shape single cluster using HarfBuzz
	Ref<Font> _font = base_font;
	if (attrib_iter->get().has(TEXT_ATTRIBUTE_FONT)) {
		Ref<Font> cluster_font = Ref<Font>(attrib_iter->get()[TEXT_ATTRIBUTE_FONT]);
		if (!cluster_font.is_null()) _font = cluster_font;
	}
	hb_font_t *hb_font = _font->get_hb_font(p_fallback_index);
	if (!hb_font) {
		return;
	}
	hb_buffer_clear_contents(hb_buffer);
	hb_buffer_set_direction(hb_buffer, p_run_direction);
	hb_buffer_set_flags(hb_buffer, (hb_buffer_flags_t)(HB_BUFFER_FLAG_DEFAULT));
	hb_buffer_set_script(hb_buffer, p_run_script);

	if (attrib_iter->get().has(TEXT_ATTRIBUTE_LANGUAGE)) {
		String cluster_language = String(attrib_iter->get()[TEXT_ATTRIBUTE_LANGUAGE]);
		hb_language_t _language = hb_language_from_string(cluster_language.ascii().get_data(), -1);
		if (_language != HB_LANGUAGE_INVALID) hb_buffer_set_language(hb_buffer, _language);
	} else {
		if (language != HB_LANGUAGE_INVALID) hb_buffer_set_language(hb_buffer, language);
	}

	hb_buffer_add_utf32(hb_buffer, (const uint32_t *)&p_codepoint, 1, 0, 1);
	if (attrib_iter->get().has(TEXT_ATTRIBUTE_FONT_FEATURES)) {
		Vector<String> v_features = String(attrib_iter->get()[TEXT_ATTRIBUTE_FONT_FEATURES]).split(",");
		Vector<hb_feature_t> _font_features;
		for (int i = 0; i < v_features.size(); i++) {
			hb_feature_t feature;
			if (hb_feature_from_string(v_features[i].ascii().get_data(), -1, &feature)) {
				feature.start = 0;
				feature.end = (unsigned int)-1;
				_font_features.push_back(feature);
			}
		}
		hb_shape(hb_font, hb_buffer, _font_features.empty() ? NULL : _font_features.ptr(), _font_features.size());
	} else {
		hb_shape(hb_font, hb_buffer, font_features.empty() ? NULL : font_features.ptr(), font_features.size());
	}

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
	p_cluster.ascent = _font->get_ascent();
	p_cluster.descent = _font->get_descent();
	p_cluster.width = 0.0f;

	if (glyph_count > 0) {
		for (unsigned int i = 0; i < glyph_count; i++) {
			p_cluster.glyphs.push_back(Glyph(glyph_info[i].codepoint, Point2((glyph_pos[i].x_offset) / 64, -(glyph_pos[i].y_offset / 64)), Point2((glyph_pos[i].x_advance) / 64, (glyph_pos[i].y_advance / 64))));
			p_cluster.valid &= ((glyph_info[i].codepoint != 0) || !u_isgraph(p_codepoint));
			p_cluster.width += (glyph_pos[i].x_advance) / 64;
		}
	}
	if (!p_cluster.valid) {
		if (p_fallback_index < _font->get_fallback_count() - 1) {
			_shape_single_cluster(p_start, p_end, p_run_direction, p_run_script, p_codepoint, p_fallback_index + 1, p_cluster);
		}
	}
}
#endif

void ShapedAttributedString::_generate_justification_opportunies(int32_t p_start, int32_t p_end, const char *p_lang, /*out*/ Vector<JustificationOpportunity> &p_ops) const {

#ifdef USE_TEXT_SHAPING
	const Map<int, Map<TextAttribute, Variant> >::Element *attrib_iter = attributes.find_closest(p_start);
	if (!attrib_iter) {
#endif
		ShapedString::_generate_justification_opportunies(p_start, p_end, p_lang, p_ops);
#ifdef USE_TEXT_SHAPING
		return;
	}

	int sh_start = attrib_iter ? MAX(p_start, attrib_iter->key()) : p_start;
	int sh_end = attrib_iter->next() ? MIN(p_end, attrib_iter->next()->key()) : p_end;
	while (true) {

		if (attrib_iter->get().has(TEXT_ATTRIBUTE_LANGUAGE)) {
			ShapedString::_generate_justification_opportunies(sh_start, sh_end, String(attrib_iter->get()[TEXT_ATTRIBUTE_LANGUAGE]).ascii().get_data(), p_ops);
		} else {
			ShapedString::_generate_justification_opportunies(sh_start, sh_end, p_lang, p_ops);
		}

		if (attrib_iter->next() && (attrib_iter->next()->key() <= sh_end)) attrib_iter = attrib_iter->next();
		if (sh_end == p_end) break;
		sh_start = sh_end;
		sh_end = attrib_iter->next() ? MIN(attrib_iter->next()->key(), p_end) : p_end;
	}
#endif
}

void ShapedAttributedString::_generate_break_opportunies(int32_t p_start, int32_t p_end, const char *p_lang, /*out*/ Vector<BreakOpportunity> &p_ops) const {

	//printf("brkopt %d %d\n", p_start, p_end);

#ifdef USE_TEXT_SHAPING
	const Map<int, Map<TextAttribute, Variant> >::Element *attrib_iter = attributes.find_closest(p_start);
	if (!attrib_iter) {
#endif
		ShapedString::_generate_break_opportunies(p_start, p_end, p_lang, p_ops);
#ifdef USE_TEXT_SHAPING
		return;
	}

	int sh_start = attrib_iter ? MAX(p_start, attrib_iter->key()) : p_start;
	int sh_end = attrib_iter->next() ? MIN(p_end, attrib_iter->next()->key()) : p_end;
	while (true) {

		//printf("brkop %d %d\n", sh_start, sh_end);
		if (attrib_iter->get().has(TEXT_ATTRIBUTE_LANGUAGE)) {
			ShapedString::_generate_break_opportunies(sh_start, sh_end, String(attrib_iter->get()[TEXT_ATTRIBUTE_LANGUAGE]).ascii().get_data(), p_ops);
		} else {
			ShapedString::_generate_break_opportunies(sh_start, sh_end, p_lang, p_ops);
		}

		if (attrib_iter->next() && (attrib_iter->next()->key() <= sh_end)) attrib_iter = attrib_iter->next();
		if (sh_end == p_end) break;
		sh_start = sh_end;
		sh_end = attrib_iter->next() ? MIN(attrib_iter->next()->key(), p_end) : p_end;
	}
#endif
}

void ShapedAttributedString::_shape_bidi_script_run(hb_direction_t p_run_direction, hb_script_t p_run_script, int32_t p_run_start, int32_t p_run_end, int p_fallback_index) {

	const Map<int, Map<TextAttribute, Variant> >::Element *attrib_iter = (p_run_direction == HB_DIRECTION_LTR) ? attributes.find_closest(p_run_start) : attributes.find_closest(p_run_end - 1);
	if (!attrib_iter) {
		//Shape as plain string
		ShapedString::_shape_bidi_script_run(p_run_direction, p_run_script, p_run_start, p_run_end, p_fallback_index);
		return;
	}
	//Iter Attrib runs and call next bidi_script_attrib run
	int sh_start = attrib_iter ? MAX(p_run_start, attrib_iter->key()) : p_run_start;
	int sh_end = attrib_iter->next() ? MIN(p_run_end, attrib_iter->next()->key()) : p_run_end;
	while (true) {
		_shape_bidi_script_attrib_run(p_run_direction, p_run_script, attrib_iter->get(), sh_start, sh_end, p_fallback_index);
#ifdef USE_TEXT_SHAPING
		if (p_run_direction == HB_DIRECTION_LTR) {
#endif
			if (attrib_iter->next() && (attrib_iter->next()->key() <= sh_end)) attrib_iter = attrib_iter->next();
			if (sh_end == p_run_end) break;
			sh_start = sh_end;
			sh_end = attrib_iter->next() ? MIN(attrib_iter->next()->key(), p_run_end) : p_run_end;
#ifdef USE_TEXT_SHAPING
		} else {
			if (attrib_iter->prev() && (attrib_iter->key() >= sh_start)) attrib_iter = attrib_iter->prev();
			if (sh_start == p_run_start) break;
			sh_end = sh_start;
			sh_start = attrib_iter->prev() ? MAX(attrib_iter->key(), p_run_start) : p_run_start;
		}
#endif
	}
}

void ShapedAttributedString::_shape_rect_run(hb_direction_t p_run_direction, const Size2 &p_size, VAlign p_align, int32_t p_run_start, int32_t p_run_end) {

#ifdef USE_TEXT_SHAPING
	//"Shape" monotone image run
	if (p_run_direction == HB_DIRECTION_LTR) {
#endif
		for (int i = p_run_start; i < p_run_end; i++) {
			Cluster new_cluster;
			new_cluster.fallback_depth = -1;
			new_cluster.is_rtl = (p_run_direction == HB_DIRECTION_RTL);
			new_cluster.cl_type = _CLUSTER_TYPE_RECT;
			new_cluster.valid = true;
			new_cluster.start = i;
			new_cluster.end = i;
			switch (p_align) {
				case VALIGN_TOP: {
					new_cluster.ascent = p_size.height;
					new_cluster.descent = 0;
				} break;
				case VALIGN_CENTER: {
					new_cluster.ascent = p_size.height / 2;
					new_cluster.descent = p_size.height / 2;
				} break;
				case VALIGN_BOTTOM: {
					new_cluster.ascent = 0;
					new_cluster.descent = p_size.height;
				} break;
			}
			new_cluster.width = p_size.width;
			visual.push_back(new_cluster);
		}
#ifdef USE_TEXT_SHAPING
	} else {
		for (int i = p_run_end - 1; i >= p_run_start; i--) {
			Cluster new_cluster;
			new_cluster.fallback_depth = -1;
			new_cluster.is_rtl = (p_run_direction == HB_DIRECTION_RTL);
			new_cluster.cl_type = _CLUSTER_TYPE_RECT;
			new_cluster.valid = true;
			new_cluster.start = i;
			new_cluster.end = i;
			switch (p_align) {
				case VALIGN_TOP: {
					new_cluster.ascent = p_size.height;
					new_cluster.descent = 0;
				} break;
				case VALIGN_CENTER: {
					new_cluster.ascent = p_size.height / 2;
					new_cluster.descent = p_size.height / 2;
				} break;
				case VALIGN_BOTTOM: {
					new_cluster.ascent = 0;
					new_cluster.descent = p_size.height;
				} break;
			}
			new_cluster.width = p_size.width;
			visual.push_back(new_cluster);
		}
	}
#endif
}

void ShapedAttributedString::_shape_image_run(hb_direction_t p_run_direction, const Ref<Texture> &p_image, VAlign p_align, int32_t p_run_start, int32_t p_run_end) {

#ifdef USE_TEXT_SHAPING
	//"Shape" monotone image run
	if (p_run_direction == HB_DIRECTION_LTR) {
#endif
		for (int i = p_run_start; i < p_run_end; i++) {
			Cluster new_cluster;
			new_cluster.fallback_depth = -1;
			new_cluster.is_rtl = (p_run_direction == HB_DIRECTION_RTL);
			new_cluster.cl_type = _CLUSTER_TYPE_IMAGE;
			new_cluster.valid = true;
			new_cluster.start = i;
			new_cluster.end = i;
			switch (p_align) {
				case VALIGN_TOP: {
					new_cluster.ascent = p_image->get_height();
					new_cluster.descent = 0;
				} break;
				case VALIGN_CENTER: {
					new_cluster.ascent = p_image->get_height() / 2;
					new_cluster.descent = p_image->get_height() / 2;
				} break;
				case VALIGN_BOTTOM: {
					new_cluster.ascent = 0;
					new_cluster.descent = p_image->get_height();
				} break;
			}
			new_cluster.width = p_image->get_width();
			visual.push_back(new_cluster);
		}
#ifdef USE_TEXT_SHAPING
	} else {
		for (int i = p_run_end - 1; i >= p_run_start; i--) {
			Cluster new_cluster;
			new_cluster.fallback_depth = -1;
			new_cluster.is_rtl = (p_run_direction == HB_DIRECTION_RTL);
			new_cluster.cl_type = _CLUSTER_TYPE_IMAGE;
			new_cluster.valid = true;
			new_cluster.start = i;
			new_cluster.end = i;
			switch (p_align) {
				case VALIGN_TOP: {
					new_cluster.ascent = p_image->get_height();
					new_cluster.descent = 0;
				} break;
				case VALIGN_CENTER: {
					new_cluster.ascent = p_image->get_height() / 2;
					new_cluster.descent = p_image->get_height() / 2;
				} break;
				case VALIGN_BOTTOM: {
					new_cluster.ascent = 0;
					new_cluster.descent = p_image->get_height();
				} break;
			}
			new_cluster.width = p_image->get_width();
			visual.push_back(new_cluster);
		}
	}
#endif
}

void ShapedAttributedString::_shape_bidi_script_attrib_run(hb_direction_t p_run_direction, hb_script_t p_run_script, const Map<TextAttribute, Variant> &p_attribs, int32_t p_run_start, int32_t p_run_end, int p_fallback_index) {

	//Handle rects for embedded custom objects
	if (p_attribs.has(TEXT_ATTRIBUTE_REPLACEMENT_RECT)) {
		Size2 rect = Size2(p_attribs[TEXT_ATTRIBUTE_REPLACEMENT_RECT]);
		if (rect != Size2()) {
			int align = VALIGN_CENTER;
			if (p_attribs.has(TEXT_ATTRIBUTE_REPLACEMENT_VALIGN)) {
				align = int(p_attribs[TEXT_ATTRIBUTE_REPLACEMENT_VALIGN]);
			}

			_shape_rect_run(p_run_direction, rect, (VAlign)align, p_run_start, p_run_end);
			return;
		}
	}

	//Handle image runs
	if (p_attribs.has(TEXT_ATTRIBUTE_REPLACEMENT_IMAGE)) {
		Ref<Texture> image = Ref<Texture>(p_attribs[TEXT_ATTRIBUTE_REPLACEMENT_IMAGE]);
		if (!image.is_null()) {
			int align = VALIGN_CENTER;
			if (p_attribs.has(TEXT_ATTRIBUTE_REPLACEMENT_VALIGN)) {
				align = int(p_attribs[TEXT_ATTRIBUTE_REPLACEMENT_VALIGN]);
			}

			_shape_image_run(p_run_direction, image, (VAlign)align, p_run_start, p_run_end);
			return;
		}
	}

#ifdef USE_TEXT_SHAPING
	//Shape monotone run using HarfBuzz
	Ref<Font> _font = base_font;
	if (p_attribs.has(TEXT_ATTRIBUTE_FONT)) {
		Ref<Font> cluster_font = Ref<Font>(p_attribs[TEXT_ATTRIBUTE_FONT]);
		if (!cluster_font.is_null()) _font = cluster_font;
	}
	hb_font_t *hb_font = _font->get_hb_font(p_fallback_index);
	if (!hb_font) {
		_shape_hex_run(p_run_direction, p_run_start, p_run_end);
		return;
	}
	hb_buffer_clear_contents(hb_buffer);
	hb_buffer_set_direction(hb_buffer, p_run_direction);
	hb_buffer_set_flags(hb_buffer, (hb_buffer_flags_t)(HB_BUFFER_FLAG_DEFAULT | (p_run_start == 0 ? HB_BUFFER_FLAG_BOT : 0) | (p_run_end == (int32_t)data_size ? HB_BUFFER_FLAG_EOT : 0)));
	hb_buffer_set_script(hb_buffer, p_run_script);

	if (p_attribs.has(TEXT_ATTRIBUTE_LANGUAGE)) {
		String cluster_language = String(p_attribs[TEXT_ATTRIBUTE_LANGUAGE]);
		hb_language_t _language = hb_language_from_string(cluster_language.ascii().get_data(), -1);
		if (_language != HB_LANGUAGE_INVALID) hb_buffer_set_language(hb_buffer, _language);
	} else {
		if (language != HB_LANGUAGE_INVALID) hb_buffer_set_language(hb_buffer, language);
	}

	hb_buffer_add_utf16(hb_buffer, (const uint16_t *)data, data_size, p_run_start, p_run_end - p_run_start);
	if (p_attribs.has(TEXT_ATTRIBUTE_FONT_FEATURES)) {
		Vector<String> v_features = String(p_attribs[TEXT_ATTRIBUTE_FONT_FEATURES]).split(",");
		Vector<hb_feature_t> _font_features;
		for (int i = 0; i < v_features.size(); i++) {
			hb_feature_t feature;
			if (hb_feature_from_string(v_features[i].ascii().get_data(), -1, &feature)) {
				feature.start = 0;
				feature.end = (unsigned int)-1;
				_font_features.push_back(feature);
			}
		}
		hb_shape(hb_font, hb_buffer, _font_features.empty() ? NULL : _font_features.ptr(), _font_features.size());
	} else {
		hb_shape(hb_font, hb_buffer, font_features.empty() ? NULL : font_features.ptr(), font_features.size());
	}

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
				new_cluster.ascent = _font->get_ascent();
				new_cluster.descent = _font->get_descent();
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
		for (int i = 0; i < run_clusters.size(); i++) {
			if (run_clusters[i].valid) {
				if (failed_subrun_start != p_run_end + 1) {
					if (p_fallback_index < _font->get_fallback_count() - 1) {
						_shape_bidi_script_attrib_run(p_run_direction, p_run_script, p_attribs, failed_subrun_start, failed_subrun_end + 1, p_fallback_index + 1);
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
			if (p_fallback_index < _font->get_fallback_count() - 1) {
				_shape_bidi_script_attrib_run(p_run_direction, p_run_script, p_attribs, failed_subrun_start, failed_subrun_end + 1, p_fallback_index + 1);
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

bool ShapedAttributedString::_compare_attributes(const Map<TextAttribute, Variant> &p_first, const Map<TextAttribute, Variant> &p_second) const {

	if (p_first.size() != p_second.size()) return false;
	for (const Map<TextAttribute, Variant>::Element *E = p_first.front(); E; E = E->next()) {
		const Map<TextAttribute, Variant>::Element *F = p_second.find(E->key());
		if (!F || (E->value() != F->value())) return false;
	}
	return true;
}

void ShapedAttributedString::_ensure_break(int p_key) {

	//Ensures there is a run break at offset.
	Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find_closest(p_key);
	attributes[p_key] = (attrib) ? attrib->get() : Map<TextAttribute, Variant>();
}

void ShapedAttributedString::_optimize_attributes() {

	Vector<int> erase_list;
	for (const Map<int, Map<TextAttribute, Variant> >::Element *E = attributes.front(); E; E = E->next()) {

		if (E->prev() && (_compare_attributes(E->value(), E->prev()->value()))) {
			erase_list.push_back(E->key());
		}
	}

	for (int i = 0; i < erase_list.size(); i++) {
		attributes.erase(erase_list[i]);
	}
}

void ShapedAttributedString::add_attribute(TextAttribute p_attribute, Variant p_value, int p_start, int p_end) {

	//Find safe UTF-16 char bounds
	int _from = _codepoint_to_offset(p_start);
	int _to = _codepoint_to_offset(p_end);

	//Adds an attribute to a subrange of the string.

	if (_to == -1) _to = (int32_t)data_size;
	if (_from < 0 || _to > (int32_t)data_size || _from > _to) {
		ERR_EXPLAIN("Invalid substring range [" + itos(_from) + " ..." + itos(_to) + "] / " + itos(data_size));
		ERR_FAIL_COND(true);
	}

	_ensure_break(_from);

	if (_to < (int32_t)data_size) _ensure_break(_to);

	Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find(_from);
	while ((attrib) && (attrib->key() < _to)) {
		attrib->get()[p_attribute] = p_value;
		attrib = attrib->next();
	}
	_optimize_attributes();
	_clear_visual();
}

void ShapedAttributedString::remove_attribute(TextAttribute p_attribute, int p_start, int p_end) {

	//Find safe UTF-16 char bounds
	int _from = _codepoint_to_offset(p_start);
	int _to = _codepoint_to_offset(p_end);

	//Remove an attribute.

	if (_to == -1) _to = (int32_t)data_size;
	if (_from < 0 || _to > (int32_t)data_size || _from > _to) {
		ERR_EXPLAIN("Invalid substring range [" + itos(_from) + " ..." + itos(_to) + "] / " + itos(data_size));
		ERR_FAIL_COND(true);
	}

	_ensure_break(_from);

	if (_to < (int32_t)data_size) _ensure_break(_to);

	Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find(_from);
	while ((attrib) && (attrib->key() < _to)) {
		attrib->get().erase(p_attribute);
		attrib = attrib->next();
	}
	_optimize_attributes();
	_clear_visual();
}

void ShapedAttributedString::remove_attributes(int p_start, int p_end) {

	//Find safe UTF-16 char bounds
	int _from = _codepoint_to_offset(p_start);
	int _to = _codepoint_to_offset(p_end);

	//Remove an attribute.

	if (_to == -1) _to = (int32_t)data_size;
	if (_from < 0 || _to > (int32_t)data_size || _from > _to) {
		ERR_EXPLAIN("Invalid substring range [" + itos(_from) + " ..." + itos(_to) + "] / " + itos(data_size));
		ERR_FAIL_COND(true);
	}

	_ensure_break(_from);

	if (_to < (int32_t)data_size) _ensure_break(_to);

	Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find(_from);
	while ((attrib) && (attrib->key() < _to)) {
		attrib->get().clear();
		attrib = attrib->next();
	}
	_optimize_attributes();
	_clear_visual();
}

void ShapedAttributedString::replace_text(int32_t p_start, int32_t p_end, const String &p_text) {

	int32_t _len = data_size;

	ShapedString::replace_text(p_start, p_end, p_text);

	_len = data_size - _len;

	Map<int, Map<TextAttribute, Variant> > new_attributes;
	for (const Map<int, Map<TextAttribute, Variant> >::Element *it = attributes.front(); it; it = it->next()) {
		if (it->key() <= p_start) {
			new_attributes[it->key()] = it->get();
		}
		if (it->key() >= p_end)
			new_attributes[it->key() + _len] = it->get();
	}
	attributes = new_attributes;

	_clear_props();
}

void ShapedAttributedString::replace_utf8(int32_t p_start, int32_t p_end, const PoolByteArray p_text) {

	int32_t _len = data_size;

	ShapedString::replace_utf8(p_start, p_end, p_text);

	_len = data_size - _len;

	Map<int, Map<TextAttribute, Variant> > new_attributes;
	for (const Map<int, Map<TextAttribute, Variant> >::Element *it = attributes.front(); it; it = it->next()) {
		if (it->key() <= p_start) {
			new_attributes[it->key()] = it->get();
		}
		if (it->key() >= p_end)
			new_attributes[it->key() + _len] = it->get();
	}
	attributes = new_attributes;

	_clear_props();
}

void ShapedAttributedString::replace_utf16(int32_t p_start, int32_t p_end, const PoolByteArray p_text) {

	int32_t _len = data_size;

	ShapedString::replace_utf16(p_start, p_end, p_text);

	_len = data_size - _len;

	Map<int, Map<TextAttribute, Variant> > new_attributes;
	for (const Map<int, Map<TextAttribute, Variant> >::Element *it = attributes.front(); it; it = it->next()) {
		if (it->key() <= p_start) {
			new_attributes[it->key()] = it->get();
		}
		if (it->key() >= p_end)
			new_attributes[it->key() + _len] = it->get();
	}
	attributes = new_attributes;

	_clear_props();
}

void ShapedAttributedString::replace_utf32(int32_t p_start, int32_t p_end, const PoolByteArray p_text) {

	int32_t _len = data_size;

	ShapedString::replace_utf32(p_start, p_end, p_text);

	_len = data_size - _len;

	Map<int, Map<TextAttribute, Variant> > new_attributes;
	for (const Map<int, Map<TextAttribute, Variant> >::Element *it = attributes.front(); it; it = it->next()) {
		if (it->key() <= p_start) {
			new_attributes[it->key()] = it->get();
		}
		if (it->key() >= p_end)
			new_attributes[it->key() + _len] = it->get();
	}
	attributes = new_attributes;

	_clear_props();
}

void ShapedAttributedString::clear_attributes() {

	attributes.clear();
	_clear_visual();
}

Vector2 ShapedAttributedString::draw_cluster(RID p_canvas_item, const Point2 &p_position, int p_index, const Color &p_modulate, bool p_outline) const {

	if (!valid)
		const_cast<ShapedAttributedString *>(this)->_shape_full_string();

	if (!valid)
		return Vector2();

	if ((p_index < 0) || (p_index >= visual.size()))
		return Vector2();

	Vector2 ofs;
	if (visual[p_index].cl_type == (int)_CLUSTER_TYPE_HEX_BOX) {
		for (int i = 0; i < visual[p_index].glyphs.size(); i++) {
			Font::draw_hex_box(p_canvas_item, p_position + ofs - Point2(0, visual[p_index].ascent), visual[p_index].glyphs[i].codepoint, p_modulate);
		}
	} else if (visual[p_index].cl_type == (int)_CLUSTER_TYPE_TEXT) {
		for (int i = 0; i < visual[p_index].glyphs.size(); i++) {
			Ref<Font> _font = base_font;
			Color _color = p_modulate;
			Color _outline_color = p_modulate;
			bool _outline = p_outline;

			const Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find_closest(visual[p_index].start);
			if (attrib && attrib->get().has(TEXT_ATTRIBUTE_FONT)) {
				Ref<Font> cluster_font = Ref<Font>(attrib->get()[TEXT_ATTRIBUTE_FONT]);
				if (!cluster_font.is_null()) _font = cluster_font;
			}
			if (attrib && attrib->get().has(TEXT_ATTRIBUTE_COLOR)) {
				_color = Color(attrib->get()[TEXT_ATTRIBUTE_COLOR]);
			}
			if (attrib && attrib->get().has(TEXT_ATTRIBUTE_OUTLINE_COLOR)) {
				_outline_color = Color(attrib->get()[TEXT_ATTRIBUTE_OUTLINE_COLOR]);
				_outline = true;
			}

			_font->draw_glyph(p_canvas_item, p_position + ofs, visual[i].glyphs[i].codepoint, visual[p_index].glyphs[i].offset, visual[p_index].ascent, _outline ? _outline_color : _color, _outline, visual[p_index].fallback_depth);
			if (_outline)
				_font->draw_glyph(p_canvas_item, p_position + ofs, visual[i].glyphs[i].codepoint, visual[p_index].glyphs[i].offset, visual[p_index].ascent, _color, false, visual[p_index].fallback_depth);

			if (attrib && attrib->get().has(TEXT_ATTRIBUTE_UNDERLINE_COLOR)) {
				Color _ln_color = Color(attrib->get()[TEXT_ATTRIBUTE_UNDERLINE_COLOR]);
				float _width = 1.0f;
				if (attrib && attrib->get().has(TEXT_ATTRIBUTE_UNDERLINE_WIDTH)) {
					_width = float(attrib->get()[TEXT_ATTRIBUTE_UNDERLINE_WIDTH]);
				}
#ifdef TOOLS_ENABLED
				_width *= EDSCALE;
#endif
				VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + ofs + Point2(0, visual[p_index].descent), p_position + ofs + Point2(visual[p_index].width, visual[p_index].descent), _ln_color, _width);
			}
			if (attrib && attrib->get().has(TEXT_ATTRIBUTE_STRIKETHROUGH_COLOR)) {
				Color _ln_color = Color(attrib->get()[TEXT_ATTRIBUTE_STRIKETHROUGH_COLOR]);
				float _width = 1.0f;
				if (attrib && attrib->get().has(TEXT_ATTRIBUTE_STRIKETHROUGH_WIDTH)) {
					_width = float(attrib->get()[TEXT_ATTRIBUTE_STRIKETHROUGH_WIDTH]);
				}
#ifdef TOOLS_ENABLED
				_width *= EDSCALE;
#endif
				VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + ofs, p_position + ofs + Point2(visual[p_index].width, 0), _ln_color, _width);
			}
			ofs += visual[p_index].glyphs[i].advance;
		}
	} else if (visual[p_index].cl_type == (int)_CLUSTER_TYPE_IMAGE) {
		const Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find_closest(visual[p_index].start);
		if (attrib && attrib->get().has(TEXT_ATTRIBUTE_REPLACEMENT_IMAGE)) {
			Ref<Texture> image = Ref<Texture>(attrib->get()[TEXT_ATTRIBUTE_REPLACEMENT_IMAGE]);
			if (!image.is_null()) {
				image->draw(p_canvas_item, p_position + ofs + Point2(0, -visual[p_index].ascent));
			}
			ofs += Vector2(visual[p_index].width, 0);
		}
	} else if (visual[p_index].cl_type == (int)_CLUSTER_TYPE_RECT) {
#ifdef DEBUG_DISPLAY_METRICS
		const Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find_closest(visual[p_index].start);
		if (attrib && attrib->get().has(TEXT_ATTRIBUTE_REPLACEMENT_RECT)) {
			Size2 rect = Size2(attrib->get()[TEXT_ATTRIBUTE_REPLACEMENT_RECT]);
			if (rect != Size2()) {
				VisualServer::get_singleton()->canvas_item_add_rect(p_canvas_item, Rect2(p_position + ofs + Point2(0, -visual[p_index].ascent), rect), p_modulate);
			}
		}
#endif
		ofs += Vector2(visual[p_index].width, 0);
	} else {
		WARN_PRINTS("Invalid cluster type")
	}

	return ofs;
}

void ShapedAttributedString::draw(RID p_canvas_item, const Point2 &p_position, const Color &p_modulate, bool p_outline) const {

	if (!valid)
		const_cast<ShapedAttributedString *>(this)->_shape_full_string();

	if (!valid)
		return;
#ifdef DEBUG_DISPLAY_METRICS
	VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + Point2(0, -ascent), p_position + Point2(width, -ascent), Color(1, 0, 0, 0.5), 1);
	VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + Point2(0, 0), p_position + Point2(width, 0), Color(1, 1, 0, 0.5), 1);
	VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + Point2(0, descent), p_position + Point2(width, descent), Color(0, 0, 1, 0.5), 1);
#endif

	Vector2 ofs;
	for (int i = 0; i < visual.size(); i++) {
#ifdef DEBUG_DISPLAY_METRICS
		VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + ofs + Point2(0, -visual[i].ascent), p_position + ofs + Point2(visual[i].width, -visual[i].ascent), Color(1, 0.5, 0.5, 0.2), 3);
#endif
		if (visual[i].cl_type == (int)_CLUSTER_TYPE_HEX_BOX) {
			for (int j = 0; j < visual[i].glyphs.size(); j++) {
				Font::draw_hex_box(p_canvas_item, p_position + ofs - Point2(0, visual[i].ascent), visual[i].glyphs[j].codepoint, p_modulate);
				ofs += visual[i].glyphs[j].advance;
			}
		} else if (visual[i].cl_type == (int)_CLUSTER_TYPE_TEXT) {
			for (int j = 0; j < visual[i].glyphs.size(); j++) {
				Ref<Font> _font = base_font;
				Color _color = p_modulate;
				Color _outline_color = p_modulate;
				bool _outline = p_outline;

				const Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find_closest(visual[i].start);
				if (attrib && attrib->get().has(TEXT_ATTRIBUTE_FONT)) {
					Ref<Font> cluster_font = Ref<Font>(attrib->get()[TEXT_ATTRIBUTE_FONT]);
					if (!cluster_font.is_null()) _font = cluster_font;
				}
				if (attrib && attrib->get().has(TEXT_ATTRIBUTE_COLOR)) {
					_color = Color(attrib->get()[TEXT_ATTRIBUTE_COLOR]);
				}
				if (attrib && attrib->get().has(TEXT_ATTRIBUTE_OUTLINE_COLOR)) {
					_outline_color = Color(attrib->get()[TEXT_ATTRIBUTE_OUTLINE_COLOR]);
					_outline = true;
				}

				_font->draw_glyph(p_canvas_item, p_position + ofs, visual[i].glyphs[j].codepoint, visual[i].glyphs[j].offset, visual[i].ascent, _outline ? _outline_color : _color, _outline, visual[i].fallback_depth);
				if (_outline)
					_font->draw_glyph(p_canvas_item, p_position + ofs, visual[i].glyphs[j].codepoint, visual[i].glyphs[j].offset, visual[i].ascent, _color, false, visual[i].fallback_depth);

				if (attrib && attrib->get().has(TEXT_ATTRIBUTE_UNDERLINE_COLOR)) {
					Color _ln_color = Color(attrib->get()[TEXT_ATTRIBUTE_UNDERLINE_COLOR]);
					float _width = 1.0f;
					if (attrib && attrib->get().has(TEXT_ATTRIBUTE_UNDERLINE_WIDTH)) {
						_width = float(attrib->get()[TEXT_ATTRIBUTE_UNDERLINE_WIDTH]);
					}
#ifdef TOOLS_ENABLED
					_width *= EDSCALE;
#endif
					VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + ofs + Point2(0, visual[i].descent), p_position + ofs + Point2(visual[i].width, visual[i].descent), _ln_color, _width);
				}
				if (attrib && attrib->get().has(TEXT_ATTRIBUTE_STRIKETHROUGH_COLOR)) {
					Color _ln_color = Color(attrib->get()[TEXT_ATTRIBUTE_STRIKETHROUGH_COLOR]);
					float _width = 1.0f;
					if (attrib && attrib->get().has(TEXT_ATTRIBUTE_STRIKETHROUGH_WIDTH)) {
						_width = float(attrib->get()[TEXT_ATTRIBUTE_STRIKETHROUGH_WIDTH]);
					}
#ifdef TOOLS_ENABLED
					_width *= EDSCALE;
#endif
					VisualServer::get_singleton()->canvas_item_add_line(p_canvas_item, p_position + ofs, p_position + ofs + Point2(visual[i].width, 0), _ln_color, _width);
				}
				ofs += visual[i].glyphs[j].advance;
			}
		} else if (visual[i].cl_type == (int)_CLUSTER_TYPE_IMAGE) {
			const Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find_closest(visual[i].start);
			if (attrib && attrib->get().has(TEXT_ATTRIBUTE_REPLACEMENT_IMAGE)) {
				Ref<Texture> image = Ref<Texture>(attrib->get()[TEXT_ATTRIBUTE_REPLACEMENT_IMAGE]);
				if (!image.is_null()) {
					image->draw(p_canvas_item, p_position + ofs + Point2(0, -visual[i].ascent));
				}
				ofs += Vector2(visual[i].width, 0);
			}
		} else if (visual[i].cl_type == (int)_CLUSTER_TYPE_RECT) {
#ifdef DEBUG_DISPLAY_METRICS
			const Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find_closest(visual[i].start);
			if (attrib && attrib->get().has(TEXT_ATTRIBUTE_REPLACEMENT_RECT)) {
				Size2 rect = Size2(attrib->get()[TEXT_ATTRIBUTE_REPLACEMENT_RECT]);
				if (rect != Size2()) {
					VisualServer::get_singleton()->canvas_item_add_rect(p_canvas_item, Rect2(p_position + ofs + Point2(0, -visual[i].ascent), rect), Color(0, 0.2, 1, 0.2));
				}
			}
#endif
			ofs += Vector2(visual[i].width, 0);
		} else {
			WARN_PRINTS("Invalid cluster type")
		}
	}
}

Variant ShapedAttributedString::get_attribute(TextAttribute p_attribute, int p_index) const {

	//Find safe UTF-16 char bounds
	int _from = _codepoint_to_offset(p_index);

	//Returns attribute.
	if (_from < 0 || _from > (int32_t)data_size) {
		ERR_EXPLAIN("Invalid substring range [" + itos(_from) + "] / " + itos(data_size));
		ERR_FAIL_COND_V(true, Variant());
	}
	const Map<int, Map<TextAttribute, Variant> >::Element *attrib = attributes.find_closest(_from);
	if (!attrib) {
		ERR_EXPLAIN("Attribute not set");
		ERR_FAIL_COND_V(true, Variant());
	}
	return attrib->get()[p_attribute];
}

void ShapedAttributedString::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_attribute", "attribute_key", "attribute_value", "start", "end"), &ShapedAttributedString::add_attribute);
	ClassDB::bind_method(D_METHOD("remove_attribute", "attribute_key", "start", "end"), &ShapedAttributedString::remove_attribute);
	ClassDB::bind_method(D_METHOD("get_attribute", "attribute_key", "index"), &ShapedAttributedString::get_attribute);
	ClassDB::bind_method(D_METHOD("remove_attributes", "start", "end"), &ShapedAttributedString::remove_attributes);
	ClassDB::bind_method(D_METHOD("clear_attributes"), &ShapedAttributedString::clear_attributes);

	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_FONT);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_FONT_FEATURES);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_LANGUAGE);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_COLOR);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_OUTLINE_COLOR);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_UNDERLINE_COLOR);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_UNDERLINE_WIDTH);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_STRIKETHROUGH_COLOR);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_STRIKETHROUGH_WIDTH);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_REPLACEMENT_IMAGE);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_REPLACEMENT_RECT);
	BIND_ENUM_CONSTANT(TEXT_ATTRIBUTE_REPLACEMENT_VALIGN);
}
