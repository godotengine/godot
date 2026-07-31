/**************************************************************************/
/*  test_texture_button.cpp                                               */
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

TEST_FORCE_LINK(test_texture_button)

#include "core/io/image.h"
#include "scene/gui/texture_button.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/bit_map.h"
#include "scene/resources/image_texture.h"

namespace TestTextureButton {

static Ref<ImageTexture> make_test_texture(int p_w, int p_h) {
	Ref<Image> image = memnew(Image(p_w, p_h, false, Image::FORMAT_RGB8));
	return ImageTexture::create_from_image(image);
}

TEST_CASE("[SceneTree][TextureButton] Default properties") {
	TextureButton *texture_button = memnew(TextureButton);

	CHECK(texture_button->get_texture_normal().is_null());
	CHECK(texture_button->get_texture_pressed().is_null());
	CHECK(texture_button->get_texture_hover().is_null());
	CHECK(texture_button->get_texture_disabled().is_null());
	CHECK(texture_button->get_texture_focused().is_null());
	CHECK(texture_button->get_click_mask().is_null());

	CHECK_FALSE(texture_button->get_ignore_texture_size());
	CHECK(texture_button->get_stretch_mode() == TextureButton::STRETCH_KEEP);
	CHECK_FALSE(texture_button->is_flipped_h());
	CHECK_FALSE(texture_button->is_flipped_v());

	CHECK(texture_button->get_mouse_filter() == Control::MOUSE_FILTER_STOP);

	CHECK(texture_button->get_combined_minimum_size() == Size2(0, 0));

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] Texture setters and getters") {
	TextureButton *texture_button = memnew(TextureButton);

	Ref<ImageTexture> tex_a = make_test_texture(8, 8);
	Ref<ImageTexture> tex_b = make_test_texture(12, 10);

	SUBCASE("texture_normal") {
		texture_button->set_texture_normal(tex_a);
		CHECK(texture_button->get_texture_normal() == tex_a);
		texture_button->set_texture_normal(Ref<Texture2D>());
		CHECK(texture_button->get_texture_normal().is_null());
	}

	SUBCASE("texture_pressed") {
		texture_button->set_texture_pressed(tex_a);
		CHECK(texture_button->get_texture_pressed() == tex_a);
		texture_button->set_texture_pressed(Ref<Texture2D>());
		CHECK(texture_button->get_texture_pressed().is_null());
	}

	SUBCASE("texture_hover") {
		texture_button->set_texture_hover(tex_a);
		CHECK(texture_button->get_texture_hover() == tex_a);
		texture_button->set_texture_hover(Ref<Texture2D>());
		CHECK(texture_button->get_texture_hover().is_null());
	}

	SUBCASE("texture_disabled") {
		texture_button->set_texture_disabled(tex_a);
		CHECK(texture_button->get_texture_disabled() == tex_a);
		texture_button->set_texture_disabled(Ref<Texture2D>());
		CHECK(texture_button->get_texture_disabled().is_null());
	}

	SUBCASE("texture_focused") {
		texture_button->set_texture_focused(tex_a);
		CHECK(texture_button->get_texture_focused() == tex_a);
		texture_button->set_texture_focused(Ref<Texture2D>());
		CHECK(texture_button->get_texture_focused().is_null());
	}

	SUBCASE("Independent texture slots") {
		texture_button->set_texture_normal(tex_a);
		texture_button->set_texture_pressed(tex_b);
		CHECK(texture_button->get_texture_normal() == tex_a);
		CHECK(texture_button->get_texture_pressed() == tex_b);
	}

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] Click mask set and get") {
	TextureButton *texture_button = memnew(TextureButton);

	Ref<BitMap> mask;
	mask.instantiate();
	mask->create(Size2i(4, 3));

	texture_button->set_click_mask(mask);
	CHECK(texture_button->get_click_mask() == mask);

	texture_button->set_click_mask(Ref<BitMap>());
	CHECK(texture_button->get_click_mask().is_null());

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] Combined minimum size priority") {
	TextureButton *texture_button = memnew(TextureButton);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(texture_button);

	Ref<ImageTexture> tex_n = make_test_texture(16, 8);
	Ref<ImageTexture> tex_p = make_test_texture(10, 20);
	Ref<ImageTexture> tex_h = make_test_texture(4, 4);

	SUBCASE("texture_normal wins when present") {
		texture_button->set_texture_normal(tex_n);
		texture_button->set_texture_pressed(tex_p);
		texture_button->set_texture_hover(tex_h);
		SceneTree::get_singleton()->process(0);
		CHECK(texture_button->get_combined_minimum_size() == Size2(16, 8));
	}

	SUBCASE("texture_pressed when normal is empty") {
		texture_button->set_texture_pressed(tex_p);
		texture_button->set_texture_hover(tex_h);
		SceneTree::get_singleton()->process(0);
		CHECK(texture_button->get_combined_minimum_size() == Size2(10, 20));
	}

	SUBCASE("texture_hover when normal and pressed are empty") {
		texture_button->set_texture_hover(tex_h);
		SceneTree::get_singleton()->process(0);
		CHECK(texture_button->get_combined_minimum_size() == Size2(4, 4));
	}

	SUBCASE("click_mask size when no textures") {
		Ref<BitMap> mask;
		mask.instantiate();
		mask->create(Size2i(7, 5));
		texture_button->set_click_mask(mask);
		SceneTree::get_singleton()->process(0);
		CHECK(texture_button->get_combined_minimum_size() == Size2(7, 5));
	}

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] ignore_texture_size") {
	TextureButton *texture_button = memnew(TextureButton);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(texture_button);

	Ref<ImageTexture> tex = make_test_texture(32, 16);
	texture_button->set_texture_normal(tex);
	texture_button->set_ignore_texture_size(true);
	SceneTree::get_singleton()->process(0);

	CHECK(texture_button->get_combined_minimum_size() == Size2(0, 0));

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] Stretch mode") {
	TextureButton *texture_button = memnew(TextureButton);

	texture_button->set_stretch_mode(TextureButton::STRETCH_SCALE);
	CHECK(texture_button->get_stretch_mode() == TextureButton::STRETCH_SCALE);

	texture_button->set_stretch_mode(TextureButton::STRETCH_TILE);
	CHECK(texture_button->get_stretch_mode() == TextureButton::STRETCH_TILE);

	texture_button->set_stretch_mode(TextureButton::STRETCH_KEEP);
	CHECK(texture_button->get_stretch_mode() == TextureButton::STRETCH_KEEP);

	texture_button->set_stretch_mode(TextureButton::STRETCH_KEEP_CENTERED);
	CHECK(texture_button->get_stretch_mode() == TextureButton::STRETCH_KEEP_CENTERED);

	texture_button->set_stretch_mode(TextureButton::STRETCH_KEEP_ASPECT);
	CHECK(texture_button->get_stretch_mode() == TextureButton::STRETCH_KEEP_ASPECT);

	texture_button->set_stretch_mode(TextureButton::STRETCH_KEEP_ASPECT_CENTERED);
	CHECK(texture_button->get_stretch_mode() == TextureButton::STRETCH_KEEP_ASPECT_CENTERED);

	texture_button->set_stretch_mode(TextureButton::STRETCH_KEEP_ASPECT_COVERED);
	CHECK(texture_button->get_stretch_mode() == TextureButton::STRETCH_KEEP_ASPECT_COVERED);

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] Flip flags") {
	TextureButton *texture_button = memnew(TextureButton);

	SUBCASE("flip_h can be toggled") {
		texture_button->set_flip_h(true);
		CHECK(texture_button->is_flipped_h());
		texture_button->set_flip_h(false);
		CHECK_FALSE(texture_button->is_flipped_h());
	}

	SUBCASE("flip_v can be toggled") {
		texture_button->set_flip_v(true);
		CHECK(texture_button->is_flipped_v());
		texture_button->set_flip_v(false);
		CHECK_FALSE(texture_button->is_flipped_v());
	}

	SUBCASE("flip_h and flip_v are independent") {
		texture_button->set_flip_h(true);
		texture_button->set_flip_v(false);
		CHECK(texture_button->is_flipped_h());
		CHECK_FALSE(texture_button->is_flipped_v());
	}

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] has_point without click mask") {
	TextureButton *texture_button = memnew(TextureButton);
	Control *control = texture_button;
	texture_button->set_size(Size2(20, 15));

	CHECK(control->has_point(Point2(10, 7)));
	CHECK_FALSE(control->has_point(Point2(25, 7)));

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] has_point with click mask (keep stretch)") {
	TextureButton *texture_button = memnew(TextureButton);

	Ref<BitMap> mask;
	mask.instantiate();
	mask->create(Size2i(4, 4));
	mask->set_bit_rect(Rect2i(0, 0, 4, 4), false);
	mask->set_bit(2, 2, true);

	texture_button->set_click_mask(mask);
	texture_button->set_size(Size2(4, 4));

	// `has_point` uses draw-time bookkeeping; refresh it without requiring a full render pass.
	texture_button->notification(Control::NOTIFICATION_DRAW);

	Control *control = texture_button;
	CHECK_FALSE(control->has_point(Point2(0.5f, 0.5f)));
	CHECK(control->has_point(Point2(2.5f, 2.5f)));

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] has_point with click mask and stretch scale") {
	TextureButton *texture_button = memnew(TextureButton);

	Ref<ImageTexture> tex = make_test_texture(10, 10);
	texture_button->set_texture_normal(tex);

	Ref<BitMap> mask;
	mask.instantiate();
	mask->create(Size2i(10, 10));
	mask->set_bit_rect(Rect2i(0, 0, 10, 10), true);

	texture_button->set_click_mask(mask);
	texture_button->set_stretch_mode(TextureButton::STRETCH_SCALE);
	texture_button->set_size(Size2(40, 40));
	texture_button->notification(Control::NOTIFICATION_DRAW);

	Control *control = texture_button;
	CHECK(control->has_point(Point2(20.0f, 20.0f)));

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] has_point with click mask and stretch tile") {
	TextureButton *texture_button = memnew(TextureButton);

	Ref<ImageTexture> tex = make_test_texture(4, 4);
	texture_button->set_texture_normal(tex);

	Ref<BitMap> mask;
	mask.instantiate();
	mask->create(Size2i(4, 4));
	mask->set_bit_rect(Rect2i(0, 0, 4, 4), false);
	mask->set_bit(2, 2, true);

	texture_button->set_click_mask(mask);
	texture_button->set_stretch_mode(TextureButton::STRETCH_TILE);
	texture_button->set_size(Size2(12, 12));
	texture_button->notification(Control::NOTIFICATION_DRAW);

	Control *control = texture_button;
	CHECK_FALSE(control->has_point(Point2(1.0f, 1.0f)));
	CHECK(control->has_point(Point2(2.5f, 2.5f)));
	// Second horizontal tile: local x becomes 2 after subtracting one mask width.
	CHECK(control->has_point(Point2(6.5f, 2.5f)));

	memdelete(texture_button);
}

TEST_CASE("[SceneTree][TextureButton] has_point with click mask and stretch keep aspect covered") {
	TextureButton *texture_button = memnew(TextureButton);

	Ref<ImageTexture> tex = make_test_texture(10, 10);
	texture_button->set_texture_normal(tex);

	Ref<BitMap> mask;
	mask.instantiate();
	mask->create(Size2i(10, 10));
	mask->set_bit_rect(Rect2i(0, 0, 10, 10), true);

	texture_button->set_click_mask(mask);
	texture_button->set_stretch_mode(TextureButton::STRETCH_KEEP_ASPECT_COVERED);
	texture_button->set_size(Size2(20, 20));
	texture_button->notification(Control::NOTIFICATION_DRAW);

	Control *control = texture_button;
	CHECK(control->has_point(Point2(10.0f, 10.0f)));

	memdelete(texture_button);
}

} // namespace TestTextureButton
