/**************************************************************************/
/*  test_scene_tree.cpp                                                   */
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

#include "scene/main/scene_tree.h"
#include "tests/signal_watcher.h"
#include "tests/test_macros.h"

TEST_FORCE_LINK(test_scene_tree)

namespace TestSceneTree {

TEST_CASE("[SceneTree] Suspended process_frame signal") {
	SceneTree *tree = SceneTree::get_singleton();
	REQUIRE(tree != nullptr);

	SIGNAL_WATCH(tree, SNAME("process_frame"));
	tree->set_suspend(true);
	tree->process(0.1);
	SIGNAL_CHECK_FALSE(SNAME("process_frame"));

	tree->set_suspend(false);
	tree->process(0.1);
	SIGNAL_CHECK(SNAME("process_frame"), Array({ {} }));
	SIGNAL_UNWATCH(tree, SNAME("process_frame"));
}

} // namespace TestSceneTree
