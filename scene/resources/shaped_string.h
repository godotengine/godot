/*************************************************************************/
/*  shaped_string.h                                                      */
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

#ifndef SHAPED_STRING_H
#define SHAPED_STRING_H

#include "core/map.h"
#include "core/resource.h"
#include "core/ustring.h"
#include "scene/resources/font.h"
#include "scene/resources/texture.h"

#ifdef USE_TEXT_SHAPING
//ICU
#include <unicode/ubidi.h>
#include <unicode/ubrk.h>
#include <unicode/uchar.h>
#include <unicode/uclean.h>
#include <unicode/udata.h>
#include <unicode/uiter.h>
#include <unicode/uloc.h>
#include <unicode/uscript.h>
#include <unicode/ustring.h>
#include <unicode/utypes.h>

//HarfBuzz
#include <hb-icu.h>
#include <hb.h>

//HarfBuzz BitmapFont support
#include "drivers/harfbuzz/src/hb-bitmap.h"
#else
typedef int hb_direction_t;
typedef int hb_script_t;

#define HB_DIRECTION_LTR 0
#define HB_DIRECTION_RTL 1

typedef wchar_t UChar32;
typedef wchar_t UChar;
#endif

/*************************************************************************/
/*  ShapedString                                                         */
/*************************************************************************/

class ShapedString : public Reference {
	GDCLASS(ShapedString, Reference);

protected:
	enum {
		_CLUSTER_TYPE_INVALID = 0,
		_CLUSTER_TYPE_HEX_BOX = 1, //Fallback hex box font codepoint
		_CLUSTER_TYPE_TEXT = 2 //Normal glyphs
	};

	struct Glyph {

		UChar32 codepoint;
		Point2 offset;
		Point2 advance;

		Glyph();
		Glyph(uint32_t p_codepoint, Point2 p_offset, Point2 p_advance);
	};

	struct Cluster {

		int cl_type;

		int32_t start;
		int32_t end;

		bool valid;
		bool is_rtl;

		int fallback_depth;

		float offset;
		float ascent;
		float descent;
		float width;

		Vector<Glyph> glyphs;

		Cluster();
	};

	struct ClusterCompare {

		bool operator()(const Cluster &p_a, const Cluster &p_b) const {

			return p_a.start < p_b.start;
		}
	};

#ifdef USE_TEXT_SHAPING
	class ScriptIterator {

		struct ScriptRange {

			int32_t start;
			int32_t end;
			hb_script_t script;
		};

		static bool same_script(int32_t p_script_one, int32_t p_script_two);

		int cur;
		bool is_rtl;
		Vector<ScriptRange> script_ranges;

	public:
		bool next();

		int32_t get_start() const;
		int32_t get_end() const;
		hb_script_t get_script() const;

		void reset(hb_direction_t p_run_direction);

		ScriptIterator(const UChar *p_chars, int32_t p_start, int32_t p_length);
	};
#endif
	struct BreakOpportunity {

		int32_t position;
		bool hard;
	};

	struct JustificationOpportunity {

		int32_t position;
		bool kashida;
	};

	bool valid;

	TextDirection base_direction;
	Ref<Font> base_font;

#ifdef USE_TEXT_SHAPING
	UChar *data;
	Vector<hb_feature_t> font_features;
	hb_language_t language;

	UBiDi *bidi_iter;
	ScriptIterator *script_iter;
	hb_buffer_t *hb_buffer;
#else
	String data;
#endif
	size_t data_size;
	size_t char_size;

	Vector<Cluster> visual;

	float ascent;
	float descent;
	float width;

#ifdef USE_TEXT_SHAPING
	_FORCE_INLINE_ bool is_ain(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_AIN; }
	_FORCE_INLINE_ bool is_alef(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_ALEF; }
	_FORCE_INLINE_ bool is_beh(uint32_t p_chr) const {
		int32_t prop = u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP);
		return (prop == U_JG_BEH) || (prop == U_JG_NOON) || (prop == U_JG_AFRICAN_NOON) || (prop == U_JG_NYA) || (prop == U_JG_YEH) || (prop == U_JG_FARSI_YEH);
	}
	_FORCE_INLINE_ bool is_dal(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_DAL; }
	_FORCE_INLINE_ bool is_feh(uint32_t p_chr) const { return (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_FEH) || (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_AFRICAN_FEH); }
	_FORCE_INLINE_ bool is_gaf(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_GAF; }
	_FORCE_INLINE_ bool is_heh(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_HEH; }
	_FORCE_INLINE_ bool is_kaf(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_KAF; }
	_FORCE_INLINE_ bool is_lam(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_LAM; }
	_FORCE_INLINE_ bool is_qaf(uint32_t p_chr) const { return (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_QAF) || (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_AFRICAN_QAF); }
	_FORCE_INLINE_ bool is_reh(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_REH; }
	_FORCE_INLINE_ bool is_seen_sad(uint32_t p_chr) const { return (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_SAD) || (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_SEEN); }
	_FORCE_INLINE_ bool is_tah(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_TAH; }
	_FORCE_INLINE_ bool is_teh_marbuta(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_TEH_MARBUTA; }
	_FORCE_INLINE_ bool is_yeh(uint32_t p_chr) const {
		int32_t prop = u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP);
		return (prop == U_JG_YEH) || (prop == U_JG_FARSI_YEH) || (prop == U_JG_YEH_BARREE) || (prop == U_JG_BURUSHASKI_YEH_BARREE) || (prop == U_JG_YEH_WITH_TAIL);
	}
	_FORCE_INLINE_ bool is_waw(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_WAW; }
	_FORCE_INLINE_ bool is_transparent(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_TYPE) == U_JT_TRANSPARENT; }
	_FORCE_INLINE_ bool is_ligature(uint32_t p_chr, uint32_t p_nchr) const { return (is_lam(p_chr) && is_alef(p_nchr)); }
	_FORCE_INLINE_ bool is_connected_to_prev(uint32_t p_chr, uint32_t p_pchr) const {
		int32_t prop = u_getIntPropertyValue(p_pchr, UCHAR_JOINING_TYPE);
		return (prop != U_JT_RIGHT_JOINING) && (prop != U_JT_NON_JOINING) ? !is_ligature(p_pchr, p_chr) : false;
	}
#else
	_FORCE_INLINE_ bool is_ws(UChar32 p_char) const {
		return (p_char == 0x000A) || (p_char == 0x000B) || (p_char == 0x000C) || (p_char == 0x000D) || (p_char == 0x0085) || (p_char == 0x2028) || (p_char == 0x2029) || (p_char == 0x0009) || (p_char == 0x0020) || (p_char == 0x1680) || (p_char == 0x205F) || (p_char == 0x3000) || (p_char == 0x180E) || (p_char == 0x2060) || ((p_char >= 0x2000) && (p_char <= 0x200D));
	}
#endif
	_FORCE_INLINE_ bool is_break(UChar32 p_char) const {
		return (p_char == 0x000A) || (p_char == 0x000B) || (p_char == 0x000C) || (p_char == 0x000D) || (p_char == 0x0085) || (p_char == 0x2028) || (p_char == 0x2029);
	}

	virtual int _offset_to_codepoint(int p_position) const;
	virtual int _codepoint_to_offset(int p_position) const;

	virtual void _clear_props();
	virtual void _clear_visual();

#ifdef USE_TEXT_SHAPING
	virtual void _generate_kashida_justification_opportunies(int32_t p_start, int32_t p_end, /*out*/ Vector<JustificationOpportunity> &p_ops) const;
#endif
	virtual void _generate_justification_opportunies(int32_t p_start, int32_t p_end, const char *p_lang, /*out*/ Vector<JustificationOpportunity> &p_ops) const;
	virtual void _generate_break_opportunies(int32_t p_start, int32_t p_end, const char *p_lang, /*out*/ Vector<BreakOpportunity> &p_ops) const;

	virtual void _generate_justification_opportunies_fallback(int32_t p_start, int32_t p_end, /*out*/ Vector<JustificationOpportunity> &p_ops) const;
	virtual void _generate_break_opportunies_fallback(int32_t p_start, int32_t p_end, /*out*/ Vector<BreakOpportunity> &p_ops) const;

	virtual Ref<ShapedString> _shape_substring(int32_t p_start, int32_t p_end) const;

#ifdef USE_TEXT_SHAPING
	virtual void _shape_single_cluster(int32_t p_start, int32_t p_end, hb_direction_t p_run_direction, hb_script_t p_run_script, UChar32 p_codepoint, int p_fallback_index, /*out*/ Cluster &p_cluster) const;
#endif
	virtual void _shape_bidi_script_run(hb_direction_t p_run_direction, hb_script_t p_run_script, int32_t p_run_start, int32_t p_run_end, int p_fallback_index);
	virtual void _shape_bidi_run(hb_direction_t p_run_direction, int32_t p_run_start, int32_t p_run_end);
	virtual void _shape_hex_run(hb_direction_t p_run_direction, int32_t p_run_start, int32_t p_run_end);
	virtual void _shape_full_string();

	static void _bind_methods();

public:
	//Input
	TextDirection get_base_direction() const;
	void set_base_direction(TextDirection p_base_direction);

	String get_text() const;
	void set_text(const String &p_text);
	virtual void add_text(const String &p_text);
	virtual void replace_text(int32_t p_start, int32_t p_end, const String &p_text);

	PoolByteArray get_utf8() const;
	void set_utf8(const PoolByteArray p_text);
	virtual void add_utf8(const PoolByteArray p_text);
	virtual void replace_utf8(int32_t p_start, int32_t p_end, const PoolByteArray p_text);

	PoolByteArray get_utf16() const;
	void set_utf16(const PoolByteArray p_text);
	virtual void add_utf16(const PoolByteArray p_text);
	virtual void replace_utf16(int32_t p_start, int32_t p_end, const PoolByteArray p_text);

	PoolByteArray get_utf32() const;
	void set_utf32(const PoolByteArray p_text);
	virtual void add_utf32(const PoolByteArray p_text);
	virtual void replace_utf32(int32_t p_start, int32_t p_end, const PoolByteArray p_text);

	Ref<Font> get_base_font() const;
	void set_base_font(const Ref<Font> &p_font);

	String get_features() const;
	void set_features(const String &p_features);

	String get_language() const;
	void set_language(const String &p_language);

	//Line data
	virtual bool is_valid() const;
	virtual bool empty() const;
	virtual int length() const;

	virtual float get_ascent() const;
	virtual float get_descent() const;
	virtual float get_width() const;
	virtual float get_height() const;

	//Line modification
	virtual Vector<int> break_lines(float p_width, TextBreak p_flags) const;
	virtual Ref<ShapedString> substr(int p_start, int p_end) const;
	virtual float extend_to_width(float p_width, TextJustification p_flags);
	virtual float collapse_to_width(float p_width, TextJustification p_flags);

	//Cluster data
	virtual int clusters() const;
	virtual int get_cluster_index(int p_position) const;
	virtual float get_cluster_trailing_edge(int p_index) const;
	virtual float get_cluster_leading_edge(int p_index) const;
	virtual int get_cluster_start(int p_index) const;
	virtual int get_cluster_end(int p_index) const;
	virtual float get_cluster_ascent(int p_index) const;
	virtual float get_cluster_descent(int p_index) const;
	virtual float get_cluster_width(int p_index) const;
	virtual float get_cluster_height(int p_index) const;
	virtual Rect2 get_cluster_rect(int p_index) const;

	//Output
	virtual Vector<Rect2> get_highlight_shapes(int p_start, int p_end) const;
	virtual Vector<float> get_cursor_positions(int p_position, TextDirection p_primary_dir) const;
	virtual TextDirection get_char_direction(int p_position) const;
	virtual int hit_test(float p_position) const;

	virtual Vector2 draw_cluster(RID p_canvas_item, const Point2 &p_position, int p_index, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false) const;
	virtual void draw(RID p_canvas_item, const Point2 &p_position, const Color &p_modulate = Color(1, 1, 1), bool p_outline = false) const;

	//GDScript wrappers
	Array _break_lines(float p_width, TextBreak p_flags) const;
	Array _get_highlight_shapes(int p_start, int p_end) const;
	Array _get_cursor_positions(int p_position, TextDirection p_primary_dir) const;

	ShapedString();
	~ShapedString();
};

#endif
