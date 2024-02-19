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

TEST_CASE("[SceneTree][Tree] context menu detection") {
    Tree *tree = memnew(Tree);
    SceneTree::get_singleton()->get_root()->add_child(tree);
    tree->grab_focus();

    SUBCASE("[Tree] menu activation on empty") {
        SIGNAL_WATCH(tree, "item_mouse_selected");

        Ref<InputEventKey> event = memnew(InputEventKey);
        event->set_keycode(Key::MENU);
        event->set_pressed(true);
        Input::get_singleton()->parse_input_event(event);

        MessageQueue::get_singleton()->flush();

        Vector2 mouse_pos = tree->get_local_mouse_position();
        Array selection_args;
        selection_args.push_back(mouse_pos);
        selection_args.push_back(MouseButton::RIGHT);

        Array test_args;
        test_args.push_back(selection_args);

        SIGNAL_CHECK("item_mouse_selected", test_args);

        SIGNAL_UNWATCH(tree, "item_mouse_selected");
        memdelete(tree);
    }
}

} // namespace TestTree

#endif // TEST_TREE_H
