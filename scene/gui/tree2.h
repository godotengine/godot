/*************************************************************************/
/*  tree.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TREE2_H
#define TREE2_H

#include "scene/gui/control.h"
#include "scene/resources/text_line.h" //

class HScrollBar;
class VScrollBar;
class TreeItem2;
class TextLine;

class Tree2 : public Control {
	GDCLASS(Tree2, Control);

public:
	enum SelectMode {
		SELECT_SINGLE,
		SELECT_ROW,
		SELECT_MULTI,
	};

	enum DropModeFlags {
		DROP_MODE_DISABLED = 0,
		DROP_MODE_ON_ITEM = 1,
		DROP_MODE_INBETWEEN = 2,
	};

private:
	friend class TreeItem2;
	friend class TreeItemCell;
	friend class TreeItemCellText;

	struct ColumnInfo {
		int custom_min_width = 0;
		int expand_ratio = 1;
		bool expand = true;
		bool clip_content = false;
		String title;
		Ref<TextLine> text_buf;
		Dictionary opentype_features;
		String language;
		Control::TextDirection text_direction = Control::TEXT_DIRECTION_INHERITED;
		ColumnInfo() {
			text_buf.instantiate();
		}
	};
	
	bool show_column_titles = false;

	struct Cache {
		Ref<Font> font;
		Ref<Font> tb_font;
		int font_size = 0;
		int tb_font_size = 0;
		Ref<StyleBox> bg;
		Ref<StyleBox> selected;
		Ref<StyleBox> selected_focus;
		Ref<StyleBox> cursor;
		Ref<StyleBox> cursor_unfocus;
		Ref<StyleBox> button_pressed;
		Ref<StyleBox> title_button;
		Ref<StyleBox> title_button_hover;
		Ref<StyleBox> title_button_pressed;
		Ref<StyleBox> custom_button;
		Ref<StyleBox> custom_button_hover;
		Ref<StyleBox> custom_button_pressed;

		Color title_button_color;

		Ref<Texture2D> checked;
		Ref<Texture2D> unchecked;
		Ref<Texture2D> indeterminate;
		Ref<Texture2D> arrow_collapsed;
		Ref<Texture2D> arrow;
		Ref<Texture2D> select_arrow;
		Ref<Texture2D> updown;

		Color font_color;
		Color font_selected_color;
		Color guide_color;
		Color drop_position_color;
		Color relationship_line_color;
		Color parent_hl_line_color;
		Color children_hl_line_color;
		Color custom_button_font_highlight;
		Color font_outline_color;

		float base_scale = 1.0;

		int hseparation = 0;
		int vseparation = 0;
		int item_margin = 0;
		int button_margin = 0;
		Point2 offset;
		int draw_relationship_lines = 0;
		int relationship_line_width = 0;
		int parent_hl_line_width = 0;
		int children_hl_line_width = 0;
		int parent_hl_line_margin = 0;
		int draw_guides = 0;
		int scroll_border = 0;
		int scroll_speed = 0;
		int font_outline_size = 0;

		enum ClickType {
			CLICK_NONE,
			CLICK_TITLE,
			CLICK_BUTTON,

		};

		ClickType click_type = Cache::CLICK_NONE;
		ClickType hover_type = Cache::CLICK_NONE;
		int click_index = -1;
		int click_id = -1;
		TreeItem2 *click_item = nullptr;
		int click_column = 0;
		int hover_index = -1;
		Point2 click_pos;

		TreeItem2 *hover_item = nullptr;
		int hover_cell = -1;

		Point2i text_editor_position;

		bool rtl = false;
	} cache;

	HScrollBar *h_scroll = nullptr;
	VScrollBar *v_scroll = nullptr;

	bool h_scroll_enabled = true;
	bool v_scroll_enabled = true;
	
	void update_cache();
	void update_scrollbars();
	int get_title_button_height() const;
	Size2 get_internal_min_size() const;

	TreeItem2 *root = nullptr;
	TreeItem2 *popup_edited_item = nullptr;
	TreeItem2 *selected_item = nullptr;
	TreeItem2 *edited_item = nullptr;

	int blocked = 0;

	bool hide_root = false;
	bool hide_folding = false;

	Vector<ColumnInfo> columns;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_columns(int p_columns);

	TreeItem2 *create_item(TreeItem2 *p_parent = nullptr, int p_idx = -1);
	int get_column_width(int p_column) const;
	int get_column_minimum_width(int p_column) const;

	Tree2();
};

#endif
