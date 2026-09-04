/**************************************************************************/
/*  test_viewport_bookmarks.cpp                                           */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_viewport_bookmarks)

#ifdef TOOLS_ENABLED

#include "editor/scene/viewport_bookmarks.h"
#include "scene/main/node.h"

namespace TestViewportBookmarks {

TEST_CASE("[Editor][ViewportBookmarks] Validate and separate bookmark collections") {
	Node root;

	Dictionary valid_2d;
	valid_2d["name"] = "Overview";
	valid_2d["offset"] = Vector2(10, 20);
	valid_2d["zoom"] = 2.0;

	Dictionary invalid_2d = valid_2d.duplicate();
	invalid_2d["name"] = "Invalid";
	invalid_2d["zoom"] = -1.0;

	Dictionary duplicate_2d = valid_2d.duplicate();

	Dictionary valid_3d;
	valid_3d["name"] = "Entrance";
	valid_3d["position"] = Vector3(1, 2, 3);
	valid_3d["x_rotation"] = 0.25;
	valid_3d["y_rotation"] = 0.5;
	valid_3d["distance"] = 12.0;
	valid_3d["orthogonal"] = false;
	valid_3d["view_type"] = 0;

	Array bookmarks_2d_source;
	bookmarks_2d_source.push_back(valid_2d);
	bookmarks_2d_source.push_back(invalid_2d);
	bookmarks_2d_source.push_back(duplicate_2d);
	Array bookmarks_3d_source;
	bookmarks_3d_source.push_back(valid_3d);

	Dictionary metadata;
	metadata["version"] = 1;
	metadata["2d"] = bookmarks_2d_source;
	metadata["3d"] = bookmarks_3d_source;
	root.set_meta(ViewportBookmarks::META_KEY, metadata);

	Array bookmarks_2d = ViewportBookmarks::get_bookmarks(&root, ViewportBookmarks::TYPE_2D);
	REQUIRE(bookmarks_2d.size() == 1);
	CHECK(Dictionary(bookmarks_2d[0])["name"] == "Overview");

	Array bookmarks_3d = ViewportBookmarks::get_bookmarks(&root, ViewportBookmarks::TYPE_3D);
	REQUIRE(bookmarks_3d.size() == 1);
	CHECK(Dictionary(bookmarks_3d[0])["name"] == "Entrance");
}

TEST_CASE("[Editor][ViewportBookmarks] Preserve the other collection when updating metadata") {
	Node root;

	Dictionary bookmark_3d;
	bookmark_3d["name"] = "Shared 3D";
	bookmark_3d["position"] = Vector3();
	bookmark_3d["x_rotation"] = 0.0;
	bookmark_3d["y_rotation"] = 0.0;
	bookmark_3d["distance"] = 4.0;
	bookmark_3d["orthogonal"] = true;
	bookmark_3d["view_type"] = 1;

	Array bookmarks_3d;
	bookmarks_3d.push_back(bookmark_3d);
	Dictionary metadata;
	metadata["version"] = 1;
	metadata["3d"] = bookmarks_3d;
	root.set_meta(ViewportBookmarks::META_KEY, metadata);

	Dictionary bookmark_2d;
	bookmark_2d["name"] = "Shared 2D";
	bookmark_2d["offset"] = Vector2();
	bookmark_2d["zoom"] = 1.0;

	Array bookmarks_2d;
	bookmarks_2d.push_back(bookmark_2d);
	Dictionary updated = ViewportBookmarks::make_metadata(&root, ViewportBookmarks::TYPE_2D, bookmarks_2d);
	CHECK(Array(updated["2d"]).size() == 1);
	CHECK(Array(updated["3d"]).size() == 1);
}

TEST_CASE("[Editor][ViewportBookmarks] Validate names") {
	Dictionary bookmark;
	bookmark["name"] = "Existing";
	Array bookmarks;
	bookmarks.push_back(bookmark);

	CHECK_FALSE(ViewportBookmarks::is_valid_name(bookmarks, ""));
	CHECK_FALSE(ViewportBookmarks::is_valid_name(bookmarks, " Existing "));
	CHECK(ViewportBookmarks::is_valid_name(bookmarks, "Existing", 0));
	CHECK(ViewportBookmarks::is_valid_name(bookmarks, "Other"));
}

} // namespace TestViewportBookmarks

#endif // TOOLS_ENABLED
