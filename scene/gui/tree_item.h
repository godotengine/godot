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

#ifndef TREE_ITEM_H
#define TREE_ITEM_H

#include "core/object/class_db.h"

class Tree2;
class TreeItemCell;

class TreeItem2 : public Object {
	GDCLASS(TreeItem2, Object);

	Vector<TreeItemCell *> cells;

	TreeItem2 *parent = nullptr;
	TreeItem2 *prev = nullptr;
	TreeItem2 *next = nullptr;
	TreeItem2 *first_child = nullptr;

	Tree2 *tree = nullptr;
	bool is_root = false;
	Vector<TreeItem2 *> children_cache;

	bool visible = true;
	bool collapsed = false;
	int custom_min_height = 0;
	bool disable_folding = false;

	void refresh_cells();
	void create_children_cache();

	int get_height();

protected:
	static void _bind_methods();

public:
	friend class Tree2;
	friend class TreeItemCell;

	TreeItem2 *create_child(int p_idx = -1);
	int get_custom_minimum_height() const;

	int get_visible_child_count();

	TreeItemCell *get_cell(int p_idx) const;

	int draw(const Point2i &p_pos, const Point2 &p_draw_ofs, const Size2 &p_draw_size);

	TreeItem2(Tree2 *p_tree);
};

#endif
