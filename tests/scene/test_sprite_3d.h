/**************************************************************************/
/*  test_sprite_3d.h                                                      */
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

#pragma once

#include "scene/3d/sprite_3d.h"
#include "scene/resources/image_texture.h"
#include "tests/test_macros.h"

namespace TestSprite3D {

TEST_CASE("[SceneTree][Sprite3D] Sprite Reuse") {
	constexpr double DELTA_TIME = 1.0 / 60.0;
	// Enable sprite reuse by setting the current rendering method to anything that's not gl_compatibility.
	const String default_rendering_method = OS::get_singleton()->get_current_rendering_method();
	OS::get_singleton()->set_current_rendering_method("forward_plus");
	Sprite3D *sprite_0 = memnew(Sprite3D);
	Sprite3D *sprite_1 = memnew(Sprite3D);
	Sprite3D *sprite_2 = memnew(Sprite3D);
	const Ref<Image> image = memnew(Image(16, 32, false, Image::FORMAT_RGB8));
	const Ref<ImageTexture> image_texture = ImageTexture::create_from_image(image);
	sprite_0->set_texture(image_texture);
	sprite_1->set_texture(image_texture);
	sprite_2->set_texture(image_texture);

	SceneTree::get_singleton()->get_root()->add_child(sprite_0);
	SceneTree::get_singleton()->get_root()->add_child(sprite_1);
	SceneTree::get_singleton()->get_root()->add_child(sprite_2);

	SUBCASE("[Sprite3D] New sprites using the same texture should be using the same mesh.") {
		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		Sprite3D *new_sprite = memnew(Sprite3D);
		new_sprite->set_texture(image_texture);
		SceneTree::get_singleton()->get_root()->add_child(new_sprite);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
		CHECK_EQ(sprite_2->get_base(), new_sprite->get_base());

		memdelete(new_sprite);
	}

	SUBCASE("[Sprite3D] Destroying sprites should not cause other sprites to have invalid meshes.") {
		memdelete(sprite_0);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(RS::get_singleton()->mesh_get_surface_count(sprite_1->get_base()), 0);
		CHECK_NE(RS::get_singleton()->mesh_get_surface_count(sprite_2->get_base()), 0);

		memdelete(sprite_1);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(RS::get_singleton()->mesh_get_surface_count(sprite_2->get_base()), 0);

		sprite_0 = memnew(Sprite3D);
		sprite_1 = memnew(Sprite3D);
		sprite_0->set_texture(image_texture);
		sprite_1->set_texture(image_texture);
		SceneTree::get_singleton()->get_root()->add_child(sprite_0);
		SceneTree::get_singleton()->get_root()->add_child(sprite_1);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different draw flags should not be using the same mesh.") {
		const bool default_flag_transparent = sprite_0->get_draw_flag(SpriteBase3D::FLAG_TRANSPARENT);
		const bool default_flag_shaded = sprite_0->get_draw_flag(SpriteBase3D::FLAG_SHADED);
		sprite_0->set_draw_flag(SpriteBase3D::FLAG_TRANSPARENT, !default_flag_transparent);
		sprite_0->set_draw_flag(SpriteBase3D::FLAG_SHADED, !default_flag_shaded);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_1->set_draw_flag(SpriteBase3D::FLAG_TRANSPARENT, !default_flag_transparent);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());
		CHECK_NE(sprite_0->get_base(), sprite_2->get_base());

		sprite_0->set_draw_flag(SpriteBase3D::FLAG_TRANSPARENT, default_flag_transparent);
		sprite_0->set_draw_flag(SpriteBase3D::FLAG_SHADED, default_flag_shaded);
		sprite_1->set_draw_flag(SpriteBase3D::FLAG_TRANSPARENT, default_flag_transparent);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different textures should not be using the same mesh.") {
		const Ref<Image> other_image = memnew(Image(image->get_width(), image->get_height(), false, Image::FORMAT_RGB8));
		const Ref<ImageTexture> other_image_texture = ImageTexture::create_from_image(other_image);
		sprite_2->set_texture(other_image_texture);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_texture(other_image_texture);
		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_2->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_texture(image_texture);
		sprite_2->set_texture(image_texture);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different modulate values should not be using the same mesh.") {
		const Color default_modulate = sprite_0->get_modulate();
		Color red(1, 0, 0);
		Color green(0, 1, 0);
		sprite_0->set_modulate(red);
		sprite_1->set_modulate(green);
		sprite_2->set_modulate(red);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_2->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_modulate(red);
		sprite_1->set_modulate(green);
		sprite_2->set_modulate(green);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_modulate(default_modulate);
		sprite_1->set_modulate(default_modulate);
		sprite_2->set_modulate(default_modulate);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites of different pixel sizes should not be using the same mesh.") {
		const real_t default_pixel_size = sprite_0->get_pixel_size();
		sprite_0->set_pixel_size(MAX(default_pixel_size, 0.01f) / 2);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_1->set_pixel_size(sprite_0->get_pixel_size());
		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_pixel_size(default_pixel_size);
		sprite_1->set_pixel_size(default_pixel_size);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different render priorities should not be using the same mesh.") {
		const int default_render_priority = sprite_0->get_render_priority();
		sprite_0->set_render_priority(default_render_priority + 1);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_1->set_render_priority(sprite_0->get_render_priority());
		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_render_priority(default_render_priority);
		sprite_1->set_render_priority(default_render_priority);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different axes should not be using the same mesh.") {
		const Vector3::Axis default_axis = sprite_0->get_axis();
		sprite_0->set_axis(Vector3::AXIS_X);
		sprite_1->set_axis(Vector3::AXIS_Y);
		sprite_2->set_axis(Vector3::AXIS_Y);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_FALSE(sprite_0->get_aabb().is_equal_approx(sprite_1->get_aabb()));
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
		CHECK(sprite_1->get_aabb().is_equal_approx(sprite_2->get_aabb()));

		sprite_0->set_axis(Vector3::AXIS_Z);
		sprite_1->set_axis(Vector3::AXIS_X);
		sprite_2->set_axis(Vector3::AXIS_Z);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_FALSE(sprite_0->get_aabb().is_equal_approx(sprite_1->get_aabb()));
		CHECK_EQ(sprite_0->get_base(), sprite_2->get_base());
		CHECK(sprite_0->get_aabb().is_equal_approx(sprite_2->get_aabb()));

		sprite_0->set_axis(default_axis);
		sprite_1->set_axis(default_axis);
		sprite_2->set_axis(default_axis);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK(sprite_0->get_aabb().is_equal_approx(sprite_1->get_aabb()));
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
		CHECK(sprite_1->get_aabb().is_equal_approx(sprite_2->get_aabb()));
	}

	SUBCASE("[Sprite3D] Sprites with different flip_h/flip_v values should not be using the same mesh.") {
		REQUIRE_FALSE(sprite_0->is_flipped_h());
		REQUIRE_FALSE(sprite_1->is_flipped_v());

		sprite_0->set_flip_h(true);
		sprite_1->set_flip_v(true);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());
		CHECK_NE(sprite_0->get_base(), sprite_2->get_base());

		sprite_2->set_flip_v(true);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_flip_v(true);
		sprite_1->set_flip_h(true);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_flip_h(false);
		sprite_0->set_flip_v(false);
		sprite_1->set_flip_h(false);
		sprite_1->set_flip_v(false);
		sprite_2->set_flip_v(false);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different alpha cut settings should not be using the same mesh.") {
		REQUIRE_EQ(sprite_0->get_alpha_cut_mode(), SpriteBase3D::ALPHA_CUT_DISABLED);
		const float default_alpha_hash_scale = sprite_0->get_alpha_hash_scale();
		const float default_alpha_scissor_threshold = sprite_0->get_alpha_scissor_threshold();

		sprite_0->set_alpha_cut_mode(SpriteBase3D::ALPHA_CUT_HASH);
		sprite_0->set_alpha_hash_scale(MAX(default_alpha_hash_scale, 0.5f) / 2);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_1->set_alpha_cut_mode(SpriteBase3D::ALPHA_CUT_HASH);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());
		CHECK_NE(sprite_0->get_base(), sprite_2->get_base());

		sprite_1->set_alpha_hash_scale(sprite_0->get_alpha_hash_scale());

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_alpha_scissor_threshold(MAX(default_alpha_scissor_threshold, 0.5f) / 2);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());
		CHECK_NE(sprite_0->get_base(), sprite_2->get_base());

		sprite_1->set_alpha_scissor_threshold(sprite_0->get_alpha_scissor_threshold());

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_alpha_cut_mode(SpriteBase3D::ALPHA_CUT_DISABLED);
		sprite_0->set_alpha_hash_scale(default_alpha_hash_scale);
		sprite_0->set_alpha_scissor_threshold(default_alpha_scissor_threshold);
		sprite_1->set_alpha_cut_mode(SpriteBase3D::ALPHA_CUT_DISABLED);
		sprite_1->set_alpha_hash_scale(default_alpha_hash_scale);
		sprite_1->set_alpha_scissor_threshold(default_alpha_scissor_threshold);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different alpha antialiasing settings should not be using the same mesh.") {
		REQUIRE_EQ(sprite_0->get_alpha_antialiasing(), BaseMaterial3D::ALPHA_ANTIALIASING_OFF);
		const float default_alpha_antialiasing_edge = sprite_0->get_alpha_antialiasing_edge();

		sprite_0->set_alpha_antialiasing(BaseMaterial3D::ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE);
		sprite_0->set_alpha_antialiasing_edge(MAX(default_alpha_antialiasing_edge, 0.5f) / 2);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_1->set_alpha_antialiasing(sprite_0->get_alpha_antialiasing());

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());
		CHECK_NE(sprite_0->get_base(), sprite_2->get_base());

		sprite_1->set_alpha_antialiasing_edge(sprite_0->get_alpha_antialiasing_edge());

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different billboard modes should not be using the same mesh.") {
		const BaseMaterial3D::BillboardMode default_billboard_mode = sprite_0->get_billboard_mode();

		sprite_0->set_billboard_mode(BaseMaterial3D::BILLBOARD_ENABLED);
		sprite_1->set_billboard_mode(BaseMaterial3D::BILLBOARD_ENABLED);
		sprite_2->set_billboard_mode(BaseMaterial3D::BILLBOARD_DISABLED);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_NE(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_billboard_mode(default_billboard_mode);
		sprite_1->set_billboard_mode(default_billboard_mode);
		sprite_2->set_billboard_mode(default_billboard_mode);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different texture filters should not be using the same mesh.") {
		const BaseMaterial3D::TextureFilter default_texture_filter = sprite_0->get_texture_filter();

		sprite_0->set_texture_filter(BaseMaterial3D::TEXTURE_FILTER_LINEAR);
		sprite_1->set_texture_filter(BaseMaterial3D::TEXTURE_FILTER_NEAREST);
		sprite_2->set_texture_filter(BaseMaterial3D::TEXTURE_FILTER_NEAREST);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_NE(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_texture_filter(default_texture_filter);
		sprite_1->set_texture_filter(default_texture_filter);
		sprite_2->set_texture_filter(default_texture_filter);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different material overrides should be allowed to share the same mesh.") {
		REQUIRE(sprite_0->get_material_override().is_null());

		Ref<PlaceholderMaterial> material_override;
		material_override.instantiate();
		sprite_0->set_material_override(material_override);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_material_override(nullptr);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	SUBCASE("[Sprite3D] Sprites with different material overlays should be allowed to share the same mesh.") {
		REQUIRE(sprite_0->get_material_overlay().is_null());

		Ref<PlaceholderMaterial> material_overlay;
		material_overlay.instantiate();
		sprite_0->set_material_overlay(material_overlay);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());

		sprite_0->set_material_overlay(nullptr);

		SceneTree::get_singleton()->process(DELTA_TIME);
		CHECK_EQ(sprite_0->get_base(), sprite_1->get_base());
		CHECK_EQ(sprite_1->get_base(), sprite_2->get_base());
	}

	memdelete(sprite_0);
	memdelete(sprite_1);
	memdelete(sprite_2);

	OS::get_singleton()->set_current_rendering_method(default_rendering_method);
}

} // namespace TestSprite3D
