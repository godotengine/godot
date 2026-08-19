/**************************************************************************/
/*  test_texture_rect.cpp                                                 */
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

TEST_FORCE_LINK(test_texture_rect)

#include "core/io/image.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/atlas_texture.h"
#include "scene/resources/image_texture.h"

namespace TestTextureRect {

TEST_CASE("[SceneTree][TextureRect] Default properties") {
	TextureRect *texture_rect = memnew(TextureRect);

	CHECK(texture_rect->get_texture().is_null());
	CHECK(texture_rect->get_expand_mode() == TextureRect::EXPAND_KEEP_SIZE);
	CHECK(texture_rect->get_stretch_mode() == TextureRect::STRETCH_SCALE);
	CHECK_FALSE(texture_rect->is_flipped_h());
	CHECK_FALSE(texture_rect->is_flipped_v());

	// Set in the TextureRect constructor, unlike the Control default.
	CHECK(texture_rect->get_mouse_filter() == Control::MOUSE_FILTER_PASS);

	CHECK(texture_rect->get_combined_minimum_size() == Size2(0, 0));

	memdelete(texture_rect);
}

TEST_CASE("[SceneTree][TextureRect] Set/get texture") {
	TextureRect *texture_rect = memnew(TextureRect);

	Ref<Image> image = memnew(Image(8, 8, false, Image::FORMAT_RGB8));
	Ref<ImageTexture> tex = ImageTexture::create_from_image(image);

	texture_rect->set_texture(tex);
	CHECK(texture_rect->get_texture() == tex);

	texture_rect->set_texture(Ref<Texture2D>());
	CHECK(texture_rect->get_texture().is_null());

	memdelete(texture_rect);
}

TEST_CASE("[SceneTree][TextureRect] Expand mode") {
	TextureRect *texture_rect = memnew(TextureRect);

	texture_rect->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	CHECK(texture_rect->get_expand_mode() == TextureRect::EXPAND_IGNORE_SIZE);

	texture_rect->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH);
	CHECK(texture_rect->get_expand_mode() == TextureRect::EXPAND_FIT_WIDTH);

	texture_rect->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	CHECK(texture_rect->get_expand_mode() == TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);

	texture_rect->set_expand_mode(TextureRect::EXPAND_FIT_HEIGHT);
	CHECK(texture_rect->get_expand_mode() == TextureRect::EXPAND_FIT_HEIGHT);

	texture_rect->set_expand_mode(TextureRect::EXPAND_FIT_HEIGHT_PROPORTIONAL);
	CHECK(texture_rect->get_expand_mode() == TextureRect::EXPAND_FIT_HEIGHT_PROPORTIONAL);

	texture_rect->set_expand_mode(TextureRect::EXPAND_KEEP_SIZE);
	CHECK(texture_rect->get_expand_mode() == TextureRect::EXPAND_KEEP_SIZE);

	memdelete(texture_rect);
}

TEST_CASE("[SceneTree][TextureRect] Stretch mode") {
	TextureRect *texture_rect = memnew(TextureRect);

	texture_rect->set_stretch_mode(TextureRect::STRETCH_TILE);
	CHECK(texture_rect->get_stretch_mode() == TextureRect::STRETCH_TILE);

	texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP);
	CHECK(texture_rect->get_stretch_mode() == TextureRect::STRETCH_KEEP);

	texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	CHECK(texture_rect->get_stretch_mode() == TextureRect::STRETCH_KEEP_CENTERED);

	texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT);
	CHECK(texture_rect->get_stretch_mode() == TextureRect::STRETCH_KEEP_ASPECT);

	texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	CHECK(texture_rect->get_stretch_mode() == TextureRect::STRETCH_KEEP_ASPECT_CENTERED);

	texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_COVERED);
	CHECK(texture_rect->get_stretch_mode() == TextureRect::STRETCH_KEEP_ASPECT_COVERED);

	texture_rect->set_stretch_mode(TextureRect::STRETCH_SCALE);
	CHECK(texture_rect->get_stretch_mode() == TextureRect::STRETCH_SCALE);

	memdelete(texture_rect);
}

TEST_CASE("[SceneTree][TextureRect] Flip flags") {
	TextureRect *texture_rect = memnew(TextureRect);

	SUBCASE("flip_h can be toggled") {
		texture_rect->set_flip_h(true);
		CHECK(texture_rect->is_flipped_h());

		texture_rect->set_flip_h(false);
		CHECK_FALSE(texture_rect->is_flipped_h());
	}

	SUBCASE("flip_v can be toggled") {
		texture_rect->set_flip_v(true);
		CHECK(texture_rect->is_flipped_v());

		texture_rect->set_flip_v(false);
		CHECK_FALSE(texture_rect->is_flipped_v());
	}

	SUBCASE("flip_h and flip_v are independent") {
		texture_rect->set_flip_h(true);
		texture_rect->set_flip_v(false);

		CHECK(texture_rect->is_flipped_h());
		CHECK_FALSE(texture_rect->is_flipped_v());
	}

	memdelete(texture_rect);
}

TEST_CASE("[SceneTree][TextureRect] Minimum size") {
	TextureRect *texture_rect = memnew(TextureRect);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(texture_rect);

	Ref<Image> image = memnew(Image(64, 32, false, Image::FORMAT_RGB8));
	Ref<ImageTexture> tex = ImageTexture::create_from_image(image);

	SUBCASE("EXPAND_KEEP_SIZE reports the texture's pixel dimensions") {
		texture_rect->set_texture(tex);
		SceneTree::get_singleton()->process(0);

		CHECK(texture_rect->get_combined_minimum_size() == Size2(64, 32));
	}

	SUBCASE("EXPAND_IGNORE_SIZE reports zero even with a texture") {
		texture_rect->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
		texture_rect->set_texture(tex);
		SceneTree::get_singleton()->process(0);

		CHECK(texture_rect->get_combined_minimum_size() == Size2(0, 0));
	}

	SUBCASE("Clearing the texture resets minimum size to zero") {
		texture_rect->set_texture(tex);
		SceneTree::get_singleton()->process(0);

		texture_rect->set_texture(Ref<Texture2D>());
		SceneTree::get_singleton()->process(0);

		CHECK(texture_rect->get_combined_minimum_size() == Size2(0, 0));
	}

	memdelete(texture_rect);
}

TEST_CASE("[SceneTree][TextureRect] Configuration warnings") {
	TextureRect *texture_rect = memnew(TextureRect);

	SUBCASE("No warnings by default") {
		CHECK(texture_rect->get_configuration_warnings().is_empty());
	}

	SUBCASE("STRETCH_TILE with a plain texture produces no warning") {
		Ref<Image> image = memnew(Image(4, 4, false, Image::FORMAT_RGB8));
		texture_rect->set_texture(ImageTexture::create_from_image(image));
		texture_rect->set_stretch_mode(TextureRect::STRETCH_TILE);

		CHECK(texture_rect->get_configuration_warnings().is_empty());
	}

	SUBCASE("STRETCH_TILE with an AtlasTexture with non-zero margin produces a warning") {
		Ref<AtlasTexture> at;
		at.instantiate();
		at->set_margin(Rect2(1, 0, 0, 0));

		texture_rect->set_texture(at);
		texture_rect->set_stretch_mode(TextureRect::STRETCH_TILE);

		CHECK(texture_rect->get_configuration_warnings().size() == 1);
	}

	memdelete(texture_rect);
}

} // namespace TestTextureRect
