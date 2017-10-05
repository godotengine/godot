/*************************************************************************/
/*  text_layout.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef PARA_LAYOUT_H
#define PARA_LAYOUT_H

#include "font.h"
#include "map.h"
#include "project_settings.h"
#include "resource.h"
#include "texture.h"
#include "vector.h"

//ICU
#include <unicode/putil.h>
#include <unicode/ubidi.h>
#include <unicode/ubrk.h>
#include <unicode/uchar.h>
#include <unicode/uclean.h>
#include <unicode/uscript.h>
#include <unicode/ushape.h>
#include <unicode/ustring.h>
#include <unicode/utypes.h>

//HarfBuzz
#include <hb-icu.h>
#include <hb.h>

/*************************************************************************/
/*  Enums                                                                */
/*************************************************************************/

enum TextHAlign {

	H_ALIGN_LEFT,
	H_ALIGN_CENTER,
	H_ALIGN_RIGHT,
	H_ALIGN_START,
	H_ALIGN_END,
	H_ALIGN_FILL
};

enum TextVAlign {

	V_ALIGN_TOP,
	V_ALIGN_CENTER,
	V_ALIGN_BOTTOM,
	V_ALIGN_PERCENTAGE
};

enum TextDirection {

	DIR_LTR,
	DIR_RTL,
	DIR_AUTO
};

enum TextLayoutItemType {

	OBJECT_INVALID,
	OBJECT_TEXT,
	OBJECT_IMAGE,
	OBJECT_SPAN,
	OBJECT_TABLE
};

enum TextInvalidateType {

	INVALIDATE_NONE,
	INVALIDATE_LAYOUT,
	INVALIDATE_LINES,
	INVALIDATE_RUNS,
	INVALIDATE_ALL
};

/*************************************************************************/
/*  Forward declarations                                                 */
/*************************************************************************/

class TextLayout;

/*************************************************************************/
/*  TextHitInfo                                                          */
/*************************************************************************/

class TextHitInfo : public Reference {

	friend TextLayout;
	GDCLASS(TextHitInfo, Reference);

protected:
	static void _bind_methods();

private:
	int index_start;
	int index_end;
	int index_insertion;

	TextLayoutItemType type;

	Rect2 bounds;
	Point2 leading_edge;
	Point2 trailing_edge;

public:
	int get_index_start();
	int get_index_end();
	int get_index_insertion();

	Rect2 get_bounds();
	Point2 get_leading_edge();
	Point2 get_trailing_edge();

	TextLayoutItemType get_type();

	TextHitInfo() {
		index_start = -1;
		index_end = -1;
		index_insertion = -1;

		type = OBJECT_INVALID;
	};
};

/*************************************************************************/
/*  TextLayoutItem                                                       */
/*************************************************************************/

class TextLayoutItem : public Resource {

	friend TextLayout;
	GDCLASS(TextLayoutItem, Resource);

protected:
	static void _bind_methods();

	TextLayoutItemType type;
	Point2 offset;
	TextVAlign align;
	Rect2 margins;
	bool dirty;
	float pcnt;

	virtual void update();

public:
	void set_align(TextVAlign p_align);
	TextVAlign get_align();

	void set_align_percentage(float p_align_pcnt);
	float get_align_percentage();

	void set_margins(const Rect2 &p_margins);
	Rect2 get_margins();

	virtual void _invalidate();

	virtual bool has(const Ref<TextLayout> &p_layout);
	virtual Size2 get_size() = 0;
	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Rect2 &p_clip) = 0;

	TextLayoutItem() {

		type = OBJECT_INVALID;

		offset = Size2(0, 0);
		align = V_ALIGN_BOTTOM;
		margins = Rect2(0, 0, 0, 0);
		dirty = false;
		pcnt = 0.1;
	};

	~TextLayoutItem(){};
};

/*************************************************************************/
/*  TextLayoutItemText                                                   */
/*************************************************************************/

class TextLayoutItemText : public TextLayoutItem {

	friend TextLayout;
	GDCLASS(TextLayoutItemText, TextLayoutItem);

protected:
	static void _bind_methods();

private:
	String string;

	Color fore_color;
	Color fore_color_invis;
	Color back_color;

	Color underline_color;
	Color overline_color;
	Color strikeout_color;

	int underline_width;
	int overline_width;
	int strikeout_width;

	Ref<Font> font;
	String language;
	String features;

	float rise;

	Vector<hb_feature_t> h_features;
	hb_language_t h_language;

public:
	void set_string(const String &p_string);
	String get_string();

	void set_rise(float p_rise);
	float get_rise() const;

	void set_fore_color(const Color &p_color);
	Color get_fore_color() const;

	void set_invis_color(const Color &p_color);
	Color get_invis_color() const;

	void set_back_color(const Color &p_color);
	Color get_back_color() const;

	void set_underline_color(const Color &p_color);
	Color get_underline_color() const;

	void set_overline_color(const Color &p_color);
	Color get_overline_color() const;

	void set_strikeout_color(const Color &p_color);
	Color get_strikeout_color() const;

	void set_underline_width(int p_width);
	int get_underline_width() const;

	void set_overline_width(int p_width);
	int get_overline_width() const;

	void set_strikeout_width(int p_width);
	int get_strikeout_width() const;

	void set_font(Ref<Font> p_font);
	Ref<Font> get_font() const;

	void set_language(const String &p_language);
	String get_language() const;

	void set_features(const String &p_features);
	String get_features() const;

	Size2 get_size() { return Size2(0, 0); }
	void draw(RID p_canvas_item, const Point2 &p_pos, const Rect2 &p_clip){};

	TextLayoutItemText() {

		type = OBJECT_TEXT;

		string = "";

		font = Ref<Font>(NULL);
		language = "";
		features = "";

		h_language = HB_LANGUAGE_INVALID;
		h_features.clear();

		fore_color = Color(1, 1, 1, 1);
		fore_color_invis = Color(1, 1, 1, 0.5);
		back_color = Color(0, 0, 0, 0);

		underline_color = Color(0, 0, 0, 0);
		overline_color = Color(0, 0, 0, 0);
		strikeout_color = Color(0, 0, 0, 0);

		underline_width = 0;
		overline_width = 0;
		strikeout_width = 0;

		rise = 0;
	};

	TextLayoutItemText(const String &p_string, Ref<Font> p_font, const String &p_language, const String &p_features, const Color &p_fore_color = Color(1, 1, 1, 1), const Color &p_back_color = Color(0, 0, 0, 0), const Color &p_invis_color = Color(1, 1, 1, 0.5)) {

		type = OBJECT_TEXT;

		string = p_string;

		font = p_font;
		language = p_language;
		features = p_features;

		h_language = hb_language_from_string(p_language.ascii().get_data(), -1);

		h_features.clear();
		Vector<String> v_features = p_features.split(",");
		for (int i = 0; i < v_features.size(); i++) {
			hb_feature_t feature;
			if (hb_feature_from_string(v_features[i].ascii().get_data(), -1, &feature)) h_features.push_back(feature);
		}

		fore_color = p_fore_color;
		fore_color_invis = p_invis_color;
		back_color = p_back_color;

		underline_color = Color(0, 0, 0, 0);
		overline_color = Color(0, 0, 0, 0);
		strikeout_color = Color(0, 0, 0, 0);

		underline_width = 0;
		overline_width = 0;
		strikeout_width = 0;

		rise = 0;
	};

	~TextLayoutItemText(){};
};

/*************************************************************************/
/*  TextLayoutItemImage                                                  */
/*************************************************************************/

class TextLayoutItemImage : public TextLayoutItem {

	friend TextLayout;
	GDCLASS(TextLayoutItemImage, TextLayoutItem);

protected:
	static void _bind_methods();

private:
	Ref<Texture> image;
	Color modulate;

public:
	void set_image(const Ref<Texture> &p_image);
	Ref<Texture> get_image();

	void set_modulate(Color p_modulate);
	Color get_modulate();

	Size2 get_size();
	void draw(RID p_canvas_item, const Point2 &p_pos, const Rect2 &p_clip);

	TextLayoutItemImage() {
		type = OBJECT_IMAGE;

		modulate = Color(1, 1, 1, 1);
	};

	TextLayoutItemImage(const Ref<Texture> &p_image, TextVAlign p_align, const Color &p_modulate, const Rect2 &p_margins) {

		type = OBJECT_IMAGE;

		image = p_image;
		align = p_align;
		pcnt = 0.1;
		modulate = p_modulate;
		offset = Point2(0, 0);
		margins = p_margins;
	};

	~TextLayoutItemImage(){};
};

/*************************************************************************/
/*  TextLayoutItemSpan                                                   */
/*************************************************************************/

class TextLayoutItemSpan : public TextLayoutItem {

	friend TextLayout;
	GDCLASS(TextLayoutItemSpan, TextLayoutItem);

protected:
	static void _bind_methods();

private:
	Ref<TextLayout> layout;

public:
	void set_layout(const Ref<TextLayout> &p_layout);
	Ref<TextLayout> get_layout();

	bool has(const Ref<TextLayout> &p_layout);
	Size2 get_size();
	void draw(RID p_canvas_item, const Point2 &p_pos, const Rect2 &p_clip);

	TextLayoutItemSpan() {

		type = OBJECT_SPAN;
	};

	TextLayoutItemSpan(const Ref<TextLayout> &p_layout, TextVAlign p_align, const Rect2 &p_margins) {

		type = OBJECT_SPAN;

		layout = p_layout;
		align = p_align;
		pcnt = 0.1;
		offset = Point2(0, 0);
		margins = p_margins;
	};

	~TextLayoutItemSpan(){};
};

/*************************************************************************/
/*  TextLayoutItemTable                                                  */
/*************************************************************************/

class TextLayoutItemTable : public TextLayoutItem {

	friend TextLayout;
	GDCLASS(TextLayoutItemTable, TextLayoutItem);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	static void _bind_methods();

private:
	Vector<int> columns;
	Vector<int> rows;
	Vector<Ref<TextLayout> > cells;
	int vseparation;
	int hseparation;
	Size2 size;

	void update();

public:
	void set_column_count(int p_columns);
	int get_column_count();

	void set_vseparation(int p_vseparation);
	int get_vseparation();

	void set_hseparation(int p_hseparation);
	int get_hseparation();

	Ref<TextLayout> get_cell(int p_idx) const;
	int get_cell_count() const;
	void add_cell(const Ref<TextLayout> &p_layout);
	void set_cell(int p_idx, const Ref<TextLayout> &p_layout);
	void insert_cell(int p_idx, const Ref<TextLayout> &p_layout);
	void remove_cell(int p_idx);

	bool has(const Ref<TextLayout> &p_layout);
	Size2 get_size();
	void draw(RID p_canvas_item, const Point2 &p_pos, const Rect2 &p_clip);
	Ref<TextLayout> hit_test(const Point2 &p_point);

	TextLayoutItemTable() {

		type = OBJECT_TABLE;

		vseparation = 0;
		hseparation = 0;
	};

	TextLayoutItemTable(int p_columns, TextVAlign p_align, const Rect2 &p_margins) {

		type = OBJECT_TABLE;

		align = p_align;
		pcnt = 0.1;
		offset = Point2(0, 0);
		margins = p_margins;

		columns.resize(p_columns);
		for (int i = 0; i < columns.size(); i++)
			columns[i] = 0;

		size = Size2(0, 0);
		vseparation = 0;
		hseparation = 0;
		dirty = true;
	};

	~TextLayoutItemTable() {

		columns.clear();
		cells.clear();
	};
};

/*************************************************************************/
/*  TextLayout                                                           */
/*************************************************************************/

class TextLayout : public Resource {

	GDCLASS(TextLayout, Resource);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

private:
	struct TextRange {

		uint32_t start;
		uint32_t end;

		hb_script_t script;
		Ref<TextLayoutItem> item;

		TextRange() {
			start = 0;
			end = 0;

			script = HB_SCRIPT_INVALID;
		};

		TextRange(hb_script_t p_script, uint32_t p_start, uint32_t p_end) {
			start = p_start;
			end = p_end;

			script = p_script;
		};

		TextRange(const Ref<TextLayoutItem> &p_item, uint32_t p_start, uint32_t p_end) {
			start = p_start;
			end = p_end;

			script = HB_SCRIPT_INVALID;
			item = p_item;
		};
	};

	enum BreakType {

		BREAK_INVALID,
		BREAK_LINE,
		BREAK_HARD
	};

	struct BreakOpportunity {

		size_t position;
		BreakType break_type;

		BreakOpportunity() {
			position = 0;
			break_type = BREAK_INVALID;
		};

		BreakOpportunity(size_t p_position, BreakType p_break_type) {
			position = p_position;
			break_type = p_break_type;
		};
	};

	enum JustificationType {

		JUSTIFICATION_INVALID,
		JUSTIFICATION_WORD,
		JUSTIFICATION_KASHIDA
	};

	struct JustificationOpportunity {

		size_t position;
		JustificationType just_type;

		JustificationOpportunity() {
			position = 0;
			just_type = JUSTIFICATION_INVALID;
		};

		JustificationOpportunity(size_t p_position, JustificationType p_just_type) {
			position = p_position;
			just_type = p_just_type;
		};
	};

	struct Glyph {

		uint32_t codepoint;

		bool is_ws;
		bool is_gr;
		bool is_ctrl;
		bool is_tab;
		bool is_cr;

		Point2 offset;
		Point2 size;

		Glyph() {
			codepoint = 0;
			is_ws = false;
			is_gr = false;
			is_ctrl = false;
			is_tab = false;
			is_cr = false;
			offset = Point2(0, 0);
			size = Point2(0, 0);
		};

		Glyph(UChar p_charcode, uint32_t p_codepoint, Point2 p_offset, Point2 p_size) {

			codepoint = p_codepoint;
			if (p_charcode == 0) {
				is_ws = false;
				is_gr = false;
				is_ctrl = false;
				is_tab = false;
				is_cr = false;
			} else {
				is_ws = u_isWhitespace(p_charcode);
				is_gr = u_isgraph(p_charcode);
				is_ctrl = u_iscntrl(p_charcode);
				is_tab = (p_charcode == 0x0009);
				is_cr = (p_charcode == 0x000A) || (p_charcode == 0x000D);
			}
			offset = p_offset;
			size = p_size;
		};
	};

	struct Cluster {

		size_t start;
		size_t end;

		Ref<TextLayoutItem> item;
		size_t fallback_index;

		Vector<Glyph> glyphs;
		bool valid;

		Point2 size;
		Point2 offset;
		float ascent;
		float descent;

		int line_index;

		int elongation;
		int elongation_offset;

		Cluster *next_v;
		Cluster *prev_v;
		Cluster *next_l;
		Cluster *prev_l;

		hb_direction_t direction;

		_FORCE_INLINE_ Point2 leading_edge() {
			return (direction == HB_DIRECTION_LTR) ? offset + Point2(size.x, 0) : offset;
		};

		_FORCE_INLINE_ Point2 trailing_edge() {
			return (direction == HB_DIRECTION_LTR) ? offset : offset + Point2(size.x, 0);
		};

		Cluster() {
			start = -1;
			end = -1;
			fallback_index = 0;
			valid = false;
			ascent = 0;
			descent = 0;
			line_index = -1;
			elongation = 0;
			elongation_offset = 0;

			next_v = NULL;
			prev_v = NULL;
			next_l = NULL;
			prev_l = NULL;
		};

		~Cluster() {
			glyphs.clear();
		};
	};

	struct Run {

		size_t start;
		size_t end;

		bool temp;
		bool has_tabs;

		hb_direction_t direction;

		Vector<Cluster *> clusters;

		Point2 size;
		Point2 offset;
		float ascent;
		float descent;
		float leading;
		float max_neg_glyph_displacement;
		float max_pos_glyph_displacement;

		Run() {
			start = -1;
			end = -1;
			temp = false;
			size = Size2(0, 0);
			ascent = 0;
			descent = 0;
			leading = 0;
			max_neg_glyph_displacement = 0;
			max_pos_glyph_displacement = 0;
			has_tabs = false;
		};

		~Run() {
			for (int i = 0; i < clusters.size(); i++) {
				memdelete(clusters[i]);
			}
			clusters.clear();
		};
	};

	struct RunCompare {
		bool operator()(const Run *p_a, const Run *p_b) const {

			return p_a->start < p_b->start;
		}
	};

	struct Line {
		size_t start;
		size_t end;

		Vector<Run *> runs;
		Vector<Run *> runs_logical;

		Size2 size;
		Point2 offset;
		float ascent;
		float descent;
		float leading;
		float max_neg_glyph_displacement;
		float max_pos_glyph_displacement;

		bool hard_break;

		Line() {
			start = -1;
			end = -1;
			ascent = 0;
			descent = 0;
			leading = 0;
			max_neg_glyph_displacement = 0;
			max_pos_glyph_displacement = 0;
			size = Size2(0, 0);
			offset = Point2(0, 0);
			hard_break = false;
		}

		~Line() {
			for (int i = 0; i < runs.size(); i++) {
				if (runs[i]->temp && runs[i]) memdelete(runs[i]);
			};
			runs.clear();
			runs_logical.clear();
		};
	};

	class Script {
		struct ParenStackEntry {
			int32_t pair_index;
			UScriptCode script_code;
		};

		static bool same_script(int32_t p_script_one, int32_t p_script_two);

		int32_t char_start;
		int32_t char_limit;

		int32_t script_start;
		int32_t script_end;
		UScriptCode script_code;

		ParenStackEntry paren_stack[128];
		int32_t paren_sp;

		const UChar *char_array;

	public:
		bool next();

		int32_t get_start();
		int32_t get_end();
		hb_script_t get_script();

		void reset();

		Script(const Vector<UChar> &p_chars, int32_t p_start, int32_t p_length);
	};

	struct Metrics {
		float width;
		float ascent;
		float descent;
		float leading;
		float max_neg_glyph_displacement;
		float max_pos_glyph_displacement;

		Metrics() {
			width = 0;
			ascent = 0;
			descent = 0;
			leading = 0;
			max_neg_glyph_displacement = 0;
			max_pos_glyph_displacement = 0;
		};
	};

	bool shape_text(int p_paragraph, Vector<Run *> &p_runs, Vector<Run *> &p_runs_log, size_t p_line_start, size_t p_line_end, Metrics &p_metrics, bool p_temp);
	void shape_run(size_t p_start, size_t p_end, Ref<TextLayoutItem> p_text, hb_script_t p_script, Run *p_run, Metrics &p_metrics, int p_fallback_index = -1);

	_FORCE_INLINE_ bool is_ain(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_AIN; };
	_FORCE_INLINE_ bool is_alef(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_ALEF; };
	_FORCE_INLINE_ bool is_beh(uint32_t p_chr) const {
		int32_t prop = u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP);
		return (prop == U_JG_BEH) || (prop == U_JG_NOON) || (prop == U_JG_AFRICAN_NOON) || (prop == U_JG_NYA) || (prop == U_JG_YEH) || (prop == U_JG_FARSI_YEH);
	};
	_FORCE_INLINE_ bool is_dal(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_DAL; };
	_FORCE_INLINE_ bool is_feh(uint32_t p_chr) const { return (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_FEH) || (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_AFRICAN_FEH); };
	_FORCE_INLINE_ bool is_gaf(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_GAF; };
	_FORCE_INLINE_ bool is_heh(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_HEH; };
	_FORCE_INLINE_ bool is_kaf(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_KAF; };
	_FORCE_INLINE_ bool is_lam(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_LAM; };
	_FORCE_INLINE_ bool is_qaf(uint32_t p_chr) const { return (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_QAF) || (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_AFRICAN_QAF); };
	_FORCE_INLINE_ bool is_reh(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_REH; };
	_FORCE_INLINE_ bool is_seen_sad(uint32_t p_chr) const { return (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_SAD) || (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_SEEN); };
	_FORCE_INLINE_ bool is_tah(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_TAH; };
	_FORCE_INLINE_ bool is_teh_marbuta(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_TEH_MARBUTA; };
	_FORCE_INLINE_ bool is_yeh(uint32_t p_chr) const {
		int32_t prop = u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP);
		return (prop == U_JG_YEH) || (prop == U_JG_FARSI_YEH) || (prop == U_JG_YEH_BARREE) || (prop == U_JG_BURUSHASKI_YEH_BARREE) || (prop == U_JG_YEH_WITH_TAIL);
	};
	_FORCE_INLINE_ bool is_waw(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_WAW; };
	_FORCE_INLINE_ bool is_transparent(uint32_t p_chr) const { return u_getIntPropertyValue(p_chr, UCHAR_JOINING_TYPE) == U_JT_TRANSPARENT; };
	_FORCE_INLINE_ bool is_ligature(uint32_t p_chr, uint32_t p_nchr) const { return (is_lam(p_chr) && is_alef(p_nchr)); };
	_FORCE_INLINE_ bool is_connected_to_prev(uint32_t p_chr, uint32_t p_pchr) const {
		int32_t prop = u_getIntPropertyValue(p_pchr, UCHAR_JOINING_TYPE);
		return (prop != U_JT_RIGHT_JOINING) && (prop != U_JT_NON_JOINING) ? !is_ligature(p_pchr, p_chr) : false;
	};

	_FORCE_INLINE_ Rect2 clip_rect(const Rect2 &p_rect) const {
		if (_clip_rect != Rect2(0, 0, -1, -1)) {
			return p_rect.clip(_clip_rect);
		}
		return p_rect;
	};

	void invalidate(TextInvalidateType p_level = INVALIDATE_ALL);

	void generate_break_opportunies(int32_t p_start, int32_t p_end, const String &p_lang);
	void generate_justification_opportunies(int32_t p_start, int32_t p_end, const String &p_lang);
	void generate_kashida_justification_opportunies(size_t p_start, size_t p_end);

	void update_text_boundaries();
	void update_text_runs();
	void update_lines();
	void update_layout();

	void draw_line(RID p_canvas_item, Point2 p_pos, float p_width, const Color &p_color, int p_size) const;

	void clear_cache();

	TextDirection _base_direction;

	Vector<int> _tab_stop;
	TextHAlign _align;
	TextHAlign _align_last;

	TextVAlign _valign;

	TextVAlign _parent_valign;
	TextHAlign _parent_halign;

	Size2 _min_area;
	Size2 _max_area;

	bool _autowrap;
	bool _hard_breaks;
	float _kashida_to_space_ratio;

	int _line_spacing;

	hb_buffer_t *_hb_buffer;

	bool _shaping;
	bool _bidi;
	bool _fallback;

	bool _kashida;
	bool _inter_word;
	bool _inter_char;
	bool _break_anywhere;

	Rect2 _clip_rect;

	Rect2 _padding;

	Color border_left_color;
	Color border_top_color;
	Color border_right_color;
	Color border_bottom_color;

	int border_left_width;
	int border_top_width;
	int border_right_width;
	int border_bottom_width;

	Color back_color;

	bool _invisibles;
	bool _controls;
	String _invis;

	//cache
	Vector<UChar> utf16_text;
	Vector<TextRange> _style_runs;

	Size2 _size;
	float _offset;

	bool _dirty_text_boundaries;
	bool _dirty_text_runs;
	bool _dirty_lines;
	bool _dirty_layout;

	hb_direction_t _para_direction;

	//output
	Cluster *first_logical;
	Cluster *first_visual;
	Cluster *last_logical;
	Cluster *last_visual;

	//level 3
	Vector<Line *> _lines;

	//level 2
	Vector<Run *> _runs;
	Vector<Run *> _runs_logical;

	//level 1
	Vector<JustificationOpportunity> _justification_opportunies;
	Vector<BreakOpportunity> _break_opportunies;

	static bool show_no_icu_data_warning;

	Vector<Ref<TextLayoutItem> > items;

public:
	void _invalidate();

	void add_item(const Ref<TextLayoutItem> &p_data);
	void set_item(int p_idx, const Ref<TextLayoutItem> &p_data);
	void insert_item(int p_idx, const Ref<TextLayoutItem> &p_data);
	virtual int get_item_count() const;
	Ref<TextLayoutItem> get_item(int p_idx) const;
	void remove_item(int p_idx);

	void set_border_left_color(const Color &p_color);
	Color get_border_left_color() const;

	void set_border_top_color(const Color &p_color);
	Color get_border_top_color() const;

	void set_border_right_color(const Color &p_color);
	Color get_border_right_color() const;

	void set_border_bottom_color(const Color &p_color);
	Color get_border_bottom_color() const;

	void set_border_left_width(int p_width);
	int get_border_left_width() const;

	void set_border_top_width(int p_width);
	int get_border_top_width() const;

	void set_border_right_width(int p_width);
	int get_border_right_width() const;

	void set_border_bottom_width(int p_width);
	int get_border_bottom_width() const;

	void set_back_color(const Color &p_color);
	Color get_back_color() const;

	void set_base_direction(TextDirection p_base_direction);
	TextDirection get_base_direction() const;

	void set_clip_rect(const Rect2 &p_clip);
	Rect2 get_clip_rect() const;

	void set_padding(const Rect2 &p_padding);
	Rect2 get_padding() const;

	void set_min_area(const Size2 &p_area);
	Size2 get_min_area() const;

	void set_max_area(const Size2 &p_area);
	Size2 get_max_area() const;

	void set_line_spacing(int p_spacing);
	int get_line_spacing() const;

	void set_autowrap(bool p_aw);
	bool get_autowrap() const;

	void set_hard_breaks(bool p_aw);
	bool get_hard_breaks() const;

	void set_parent_valign(TextVAlign p_align);
	TextVAlign get_parent_valign() const;

	void set_parent_halign(TextHAlign p_align);
	TextHAlign get_parent_halign() const;

	void set_valign(TextVAlign p_align);
	TextVAlign get_valign() const;

	void set_align(TextHAlign p_align);
	TextHAlign get_align() const;

	void set_align_last(TextHAlign p_align);
	TextHAlign get_align_last() const;

	void set_tab_stops(const Vector<int> &p_tab_stops);
	Vector<int> get_tab_stops() const;

	void set_show_invisible_characters(bool p_enable);
	bool get_show_invisible_characters() const;

	void set_show_control_characters(bool p_enable);
	bool get_show_control_characters() const;

	void set_invisible_characters(const String &p_string);
	String get_invisible_characters();

	void set_kasida_to_space_ratio(float p_ratio);
	float get_kasida_to_space_ratio() const;

	void set_enable_kasida_justification(bool p_enable);
	bool get_enable_kasida_justification() const;

	void set_enable_interword_justification(bool p_enable);
	bool get_enable_interword_justification() const;

	void set_enable_intercluster_justification(bool p_enable);
	bool get_enable_intercluster_justification() const;

	void set_enable_fallback_line_break(bool p_enable);
	bool get_enable_fallback_line_break() const;

	void set_enable_shaping(bool p_enable);
	bool get_enable_shaping() const;

	void set_enable_bidi(bool p_enable);
	bool get_enable_bidi() const;

	void set_enable_fallback(bool p_enable);
	bool get_enable_fallback() const;

	Rect2 get_bounds();

	int get_line_count();
	Rect2 get_line_bounds(int p_line);
	int get_line_start(int p_line);
	int get_line_end(int p_line);

	Ref<TextHitInfo> hit_test(const Point2 &p_point);
	Ref<TextLayout> hit_test_layout(const Point2 &p_point);

	bool has(const Ref<TextLayout> &p_layout) const;

	Vector<Variant> highlight_shapes_hit(Ref<TextHitInfo> p_first_hit, Ref<TextHitInfo> p_second_hit, bool p_selection = false);
	Vector<Variant> highlight_shapes_char(int p_start, int p_end);
	Vector<Variant> caret_shapes_hit(Ref<TextHitInfo> p_hit);
	Vector<Variant> caret_shapes_char(int p_index);

	Rect2 draw(RID p_canvas_item, const Point2 &p_pos);

	static void initialize_icu();
	static void finish_icu();

	TextLayout();
	~TextLayout();
};

VARIANT_ENUM_CAST(TextHAlign);
VARIANT_ENUM_CAST(TextVAlign);
VARIANT_ENUM_CAST(TextDirection);
VARIANT_ENUM_CAST(TextLayoutItemType);

#endif
