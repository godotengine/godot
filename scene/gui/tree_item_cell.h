/*************************************************************************/
/*  tree_item.h                                                          */
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

#ifndef TREE_ITEM_CELL_H
#define TREE_ITEM_CELL_H

#include "core/object/class_db.h"
#include "core/object/ref_counted.h"

class TextLine;
class Texture2D;
class Tree2;
class TreeItem2;

class TreeItemCell : public Object {
	GDCLASS(TreeItemCell, Object);

	friend class TreeItem2;

protected:
	TreeItem2 *tree_item = nullptr;
	Tree2 *tree = nullptr;

	struct CellButton {
		Ref<Texture2D> texture;
		int id = 0;
		bool disabled = false;
		Color color = Color(1, 1, 1, 1);
		String tooltip;
	};
	Vector<CellButton> buttons;
	int column = 0;

	bool dirty = true;
	bool selected = false;
	
	bool editable = false;
	bool expand_right = false;

	bool custom_bg_color = false;
	bool custom_bg_outline = false;
	Color bg_color;

	Ref<Texture2D> icon;
	Color icon_color;
	int icon_max_w = 0;
	Rect2 icon_region;

	void _changed_notify();

	static void _bind_methods();

public:
	virtual void update() = 0;
	virtual int get_height();
	virtual void draw(Point2i p_pos, int p_label_h, Point2 p_draw_ofs, int &r_skip, int &r_offset);

	void add_button(const Ref<Texture2D> &p_button, int p_id = -1, bool p_disabled = false, const String &p_tooltip = "");

	Size2 get_icon_size() const;
	void draw_icon(const RID &p_where, const Point2 &p_pos, const Size2 &p_size = Size2(), const Color &p_color = Color()) const;

	TreeItemCell(TreeItem2 *p_item, int p_column);
};

class TreeItemCellText : public TreeItemCell {
	GDCLASS(TreeItemCellText, TreeItemCell);

protected:
	String text;
	HorizontalAlignment text_alignment = HORIZONTAL_ALIGNMENT_LEFT;
	Ref<TextLine> text_buf;

	bool custom_text_color = false;
	Color text_color;

	void draw_item_rect(const Rect2i &p_rect, const Color &p_color, const Color &p_icon_color, int p_ol_size, const Color &p_ol_color);

protected:
	static void _bind_methods();

public:
	virtual void update() override;
	virtual int get_height() override;
	virtual void draw(Point2i p_pos, int p_label_h, Point2 p_draw_ofs, int &r_skip, int &r_offset) override;

	void set_text(const String &p_text);
	String get_text() const;

	TreeItemCellText(TreeItem2 *p_item, int p_column);
};

class TreeItemCellCheck : public TreeItemCellText {
	GDCLASS(TreeItemCellCheck, TreeItemCellText);

public:
	virtual void update() override {};
	virtual void draw(Point2i p_pos, int p_label_h, Point2 p_draw_ofs, int &r_skip, int &r_offset) override {};
};

class TreeItemCellRange : public TreeItemCell {
	GDCLASS(TreeItemCellRange, TreeItemCell);

public:
	virtual void update() override {};
	virtual void draw(Point2i p_pos, int p_label_h, Point2 p_draw_ofs, int &r_skip, int &r_offset) override {};
};

#endif
