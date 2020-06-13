/*************************************************************************/
/*  bbcode.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef BBCODE_H
#define BBCODE_H

#include "core/math/vector2.h"
#include "rich_text_effect.h"
#include "scene/gui/rich_text_label.h"

class RichTextLabel;

class BbCodeProcess {
	//We use the classes from RTL here, to have a quick dependency overview
	//Later, we should move them over here, if we no longer have dependencies in the RTL
	using Line = RichTextLabel::Line;
	using Item = RichTextLabel::Item;
	using ItemFrame = RichTextLabel::ItemFrame;
	using Align = RichTextLabel::Align;
	using ItemType = RichTextLabel::ItemType;
	using ItemAlign = RichTextLabel::ItemAlign;
	using ItemIndent = RichTextLabel::ItemIndent;
	using ItemFont = RichTextLabel::ItemFont;
	using ItemText = RichTextLabel::ItemText;
	using ItemFade = RichTextLabel::ItemFade;
	using ItemFX = RichTextLabel::ItemFX;
	using ItemMeta = RichTextLabel::ItemMeta;
	using ItemColor = RichTextLabel::ItemColor;
	using Selection = RichTextLabel::Selection;
	using ItemCustomFX = RichTextLabel::ItemCustomFX;
	using ItemShake = RichTextLabel::ItemShake;
	using ItemWave = RichTextLabel::ItemWave;
	using ItemTornado = RichTextLabel::ItemTornado;
	using ItemRainbow = RichTextLabel::ItemRainbow;
	using ItemImage = RichTextLabel::ItemImage;
	using ItemTable = RichTextLabel::ItemTable;

protected:
	static void _bind_methods();

private:
	enum ProcessMode {
		PROCESS_CACHE,
		PROCESS_DRAW,
		PROCESS_POINTER
	};

private:
	//localy declared vars in _process_line
	//Actual line processed by process line (determined using p_frame and p_line)
	Line &l;
	Item *it = nullptr;
	RID ci;
	int line_ofs = 0;
	int margin = 0;
	Align align = Align::ALIGN_LEFT;
	int line = 0;
	int spaces = 0;
	int wofs = 0;
	int spaces_size = 0;
	int align_ofs = 0;
	int begin = 0;
	Ref<Font> cfont;

	int line_height = 0;
	int line_ascent = 0;
	int line_descent = 0;

	int backtrack = 0; // for dynamic hidden content.
	int nonblank_line_count = 0; //number of nonblank lines as counted during PROCESS_DRAW
	Variant meta;

	Color selection_fg;
	Color selection_bg;

	int rchar = 0;
	int lh = 0;
	bool line_is_blank = true;
	bool line_wrapped = false;
	int fh = 0;

private:
	//Makros as functions
	bool _new_line();
	bool _ensure_width(int m_width);
	bool _advance(int m_width); //rename?
	void _check_height(int m_height); //rename?
	bool _y_range_visible(int m_top, int m_height);

private:
	//abstraction functions
	bool _parse_text(ItemText *text);
	bool _parse_table(ItemTable *table);
	bool _parse_image(ItemImage *img);
	bool _parse_detect_click(Item *previous_item);
	void _common_initalize_process();

private:
	//Copied functions
	Item *_get_next_item(Item *p_item, bool p_free = false) const;
	Ref<Font> _find_font(Item *p_item);
	int _find_margin(Item *p_item, const Ref<Font> &p_base_font);
	Align _find_align(Item *p_item);
	void _fetch_item_fx_stack(Item *p_item, Vector<ItemFX *> &r_stack);
	Color _find_color(Item *p_item, const Color &p_default_color);
	bool _find_underline(Item *p_item);
	bool _find_strikethrough(Item *p_item);
	bool _find_meta(Item *p_item, Variant *r_meta, ItemMeta **r_item = nullptr);

	//TODOS:
	// split process_modes into different functions
	// get rid of the Process Modes

public:
	//Returns height
	int process_cache();
	//Returns nonblank_lines
	int process_draw(int _p_char_count = 0);
	//Has to return the clicked item, if it was clicked, clicked char
	void process_pointer(const Point2i &_p_click_pos, Item **_r_click_item, int *_r_click_char, bool *_r_outside);

public:
	//Duplicated vars
	int tab_size = 4;
	RichTextLabel::Align default_align = RichTextLabel::Align::ALIGN_LEFT;
	bool underline_meta = false;
	bool override_selected_font_color = false;
	Selection selection;
	int visible_characters = -1;

public:
	ProcessMode p_mode;
	Point2i p_click_pos_default = Point2i();

	//LineParser arguments
	ItemFrame *p_frame = nullptr;
	const Vector2 &p_ofs;
	const Ref<Font> &p_base_font;
	const Color &p_base_color;
	const Color &p_font_color_shadow;
	const Point2 &shadow_ofs;
	RichTextLabel &p_ci;
	int &p_height; //is altered ONLY when a new line is added //was y
	int p_width = 0;
	int p_line = 0;
	bool p_shadow_as_outline = false;

	Point2i &p_click_pos = p_click_pos_default;
	RichTextLabel::Item **r_click_item = nullptr;
	int *r_click_char = nullptr;
	bool *r_outside = nullptr;
	int p_char_count = 0;

	BbCodeProcess(ItemFrame *_p_frame, const Vector2 &_p_ofs, int &_p_height, int p_width, int p_line, const Ref<Font> &p_base_font, const Color &p_base_color, const Color &p_font_color_shadow, bool p_shadow_as_outline, const Point2 &shadow_ofs, RichTextLabel &p_ci);
	~BbCodeProcess();
};

#endif // BBCODE_H
