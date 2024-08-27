/**************************************************************************/
/*  test_tree.h                                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef TEST_TREE_H
#define TEST_TREE_H

#include "scene/gui/tree.h"

#include "tests/test_macros.h"

namespace TestTree {

TEST_CASE("[SceneTree][Tree] Check Tree Setters and Getters") {
	Tree *test_tree = memnew(Tree);
	int columns = 5;
	test_tree->create_item();
	test_tree->set_columns(columns);

	SUBCASE("[Tree] Set visibility and tooltip visibility on all columns") {
		//CHECK DEFAULT
		for (int column = 0; column < columns; column++) {
			CHECK_EQ(test_tree->is_column_visible(column), true);
			CHECK_EQ(test_tree->get_root()->is_tooltip_visible(column), true);
		}

		for (int column = 0; column < columns; column++) {
			test_tree->set_column_visible(column, false);
			test_tree->get_root()->set_tooltip_visible(column, false);
			CHECK_EQ(test_tree->is_column_visible(column), false);
			CHECK_EQ(test_tree->get_root()->is_tooltip_visible(column), false);
		}

		for (int column = 0; column < columns; column++) {
			test_tree->set_column_visible(column, true);
			test_tree->get_root()->set_tooltip_visible(column, true);
			CHECK_EQ(test_tree->is_column_visible(column), true);
			CHECK_EQ(test_tree->get_root()->is_tooltip_visible(column), true);
		}
	}

	SUBCASE("[Tree] Set visibility and tooltip visibility on alternating columns") {
		//CHECK DEFAULT
		for (int column = 0; column < columns; column++) {
			CHECK_EQ(test_tree->is_column_visible(column), true);
			CHECK_EQ(test_tree->get_root()->is_tooltip_visible(column), true);
		}

		for (int column = 0; column < columns; column += 2) {
			test_tree->set_column_visible(column, false);
			test_tree->get_root()->set_tooltip_visible(column, false);
			CHECK_EQ(test_tree->is_column_visible(column), false);
			CHECK_EQ(test_tree->get_root()->is_tooltip_visible(column), false);
		}

		for (int column = 1; column < columns; column += 2) {
			CHECK_EQ(test_tree->is_column_visible(column), true);
			CHECK_EQ(test_tree->get_root()->is_tooltip_visible(column), true);
		}

		for (int column = 1; column < columns; column += 2) {
			test_tree->set_column_visible(column, false);
			test_tree->get_root()->set_tooltip_visible(column, false);
			CHECK_EQ(test_tree->is_column_visible(column), false);
			CHECK_EQ(test_tree->get_root()->is_tooltip_visible(column), false);
		}

		for (int column = 0; column < columns; column += 2) {
			CHECK_EQ(test_tree->is_column_visible(column), false);
			CHECK_EQ(test_tree->get_root()->is_tooltip_visible(column), false);
		}

		for (int column = 0; column < columns; column++) {
			test_tree->set_column_visible(column, true);
			test_tree->get_root()->set_tooltip_visible(column, true);
			CHECK_EQ(test_tree->is_column_visible(column), true);
			CHECK_EQ(test_tree->get_root()->is_tooltip_visible(column), true);
		}
	}

	memdelete(test_tree);
}

TEST_CASE("[SceneTree][Tree] Check Tree Column Moving and Swapping") {
	Tree *test_tree = memnew(Tree);
	int columns = 5;
	test_tree->create_item();
	test_tree->set_columns(columns);

	SUBCASE("[Tree] Move columns between all indexes where column position is lower than index") {
		for (int column = 0; column < columns; column++) {
			for (int index = column + 1; index < columns; index++) {
				//Reset text on columns for each test
				for (int cell = 0; cell < columns; cell++) {
					test_tree->get_root()->set_text(cell, itos(cell));
					CHECK_EQ(test_tree->get_root()->get_text(cell), itos(cell));
				}

				test_tree->move_column(column, index);
				CHECK_EQ(test_tree->get_root()->get_text(index), itos(column));

				for (int prev_index = index - 1; prev_index >= column; prev_index--) {
					CHECK_EQ(test_tree->get_root()->get_text(prev_index), itos(prev_index + 1));
				}
			}
		}
	}

	SUBCASE("[Tree] Move columns between all indexes where column position is greater than index") {
		for (int column = columns - 1; column >= 0; column--) {
			for (int index = column - 1; index >= 0; index--) {
				//Reset text on columns for each test
				for (int cell = 0; cell < columns; cell++) {
					test_tree->get_root()->set_text(cell, itos(cell));
					CHECK_EQ(test_tree->get_root()->get_text(cell), itos(cell));
				}

				test_tree->move_column(column, index);
				CHECK_EQ(test_tree->get_root()->get_text(index), itos(column));

				for (int prev_index = index + 1; prev_index <= column; prev_index++) {
					CHECK_EQ(test_tree->get_root()->get_text(prev_index), itos(prev_index - 1));
				}
			}
		}
	}

	SUBCASE("[Tree] Swap columns between all indexes") {
		for (int column1 = 0; column1 < columns; column1++) {
			for (int column2 = 0; column2 < columns; column2++) {
				if (column1 == column2) {
					continue;
				}

				//Reset text on columns for each test
				for (int cell = 0; cell < columns; cell++) {
					test_tree->get_root()->set_text(cell, itos(cell));
					CHECK_EQ(test_tree->get_root()->get_text(cell), itos(cell));
				}

				test_tree->swap_columns(column1, column2);
				CHECK_EQ(test_tree->get_root()->get_text(column2), itos(column1));
				CHECK_EQ(test_tree->get_root()->get_text(column1), itos(column2));

				for (int cell = 0; cell < columns; cell++) {
					if (cell == column1 || cell == column2) {
						continue;
					}

					CHECK_EQ(test_tree->get_root()->get_text(cell), itos(cell));
				}
			}
		}
	}

	memdelete(test_tree);
}

} // namespace TestTree

#endif // TEST_TREE_H
