/**************************************************************************/
/*  animated_texture.cpp                                                  */
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

#include "animated_texture.h"

#include "core/os/os.h"
#include "servers/rendering/rendering_server.h"

void AnimatedTexture::_update_proxy() {
	float delta;

	if (prev_ticks == 0) {
		delta = 0;
		prev_ticks = OS::get_singleton()->get_ticks_usec();
	} else {
		uint64_t ticks = OS::get_singleton()->get_ticks_usec();
		delta = float(double(ticks - prev_ticks) / 1000000.0);
		prev_ticks = ticks;
	}

	if (current_frame == frame_count - 1 && one_shot) {
		// Return early, we're at the end
		return;
	}

	float time_change = (delta * speed_scale);
	// Possible Optimization - time_change %= full_duration, to save iterations
	if (!pause) {
		time += time_change;
	}

	bool frame_update = false;
	while (time < 0 || time >= frames[current_frame].duration) {
		if (time < 0) {
			current_frame -= 1;
			if (current_frame < 0) {
				current_frame = frame_count - 1;
			}
			time += frames[current_frame].duration;
			frame_update = true;
		} else if (time >= frames[current_frame].duration) {
			time -= frames[current_frame].duration;
			current_frame += 1;
			if (current_frame >= frame_count) {
				current_frame = 0;
			}
			frame_update = true;
		}
	}

	if (frame_update) {
		_blit_frame(current_frame);
	}
}

void AnimatedTexture::_blit_frame(int p_frame) {
	if (frames[p_frame].texture.is_valid()) {
		// Rendering server expects textureParameters as a TypedArray[RID]
		Array textures;
		textures.push_back(proxy);

		Array src_textures;
		src_textures.push_back(frames[p_frame].texture);

		// Could Copy_Effects or Texture_Update_Partial be more effective?
		RenderingServer::get_singleton()->texture_drawable_blit_rect(textures, Rect2(0, 0, proxy_width, proxy_height), blit_material, Color(1, 1, 1, 1), src_textures, 0);
		// Maybe have to generate mipmaps here if we enable those
	}
}

void AnimatedTexture::set_frames(int p_frames) {
	ERR_FAIL_COND(p_frames < 1 || p_frames > MAX_FRAMES);
	frame_count = p_frames;
}

int AnimatedTexture::get_frames() const {
	return frame_count;
}

void AnimatedTexture::set_current_frame(int p_frame) {
	ERR_FAIL_COND(p_frame < 0 || p_frame >= frame_count);
	current_frame = p_frame;
	_blit_frame(current_frame);
	time = 0;
}

int AnimatedTexture::get_current_frame() const {
	return current_frame;
}

void AnimatedTexture::set_pause(bool p_pause) {
	pause = p_pause;
}

bool AnimatedTexture::get_pause() const {
	return pause;
}

void AnimatedTexture::set_one_shot(bool p_one_shot) {
	one_shot = p_one_shot;
}

bool AnimatedTexture::get_one_shot() const {
	return one_shot;
}

void AnimatedTexture::set_frame_texture(int p_frame, const Ref<Texture2D> &p_texture) {
	ERR_FAIL_COND(p_texture == this);
	ERR_FAIL_INDEX(p_frame, MAX_FRAMES);
	frames[p_frame].texture = p_texture;
	if (p_frame == 0 && p_texture.is_valid()) {
		// Possible Optimization: Don't rebuild if new Texture is same size
		if (proxy.is_valid()) {
			RID new_proxy = RenderingServer::get_singleton()->texture_drawable_create(p_texture->get_width(), p_texture->get_height(), RS::TEXTURE_DRAWABLE_FORMAT_RGBA8, Color(1, 1, 1, 0), false);
			RenderingServer::get_singleton()->texture_replace(proxy, new_proxy);
		} else {
			proxy = RenderingServer::get_singleton()->texture_drawable_create(p_texture->get_width(), p_texture->get_height(), RS::TEXTURE_DRAWABLE_FORMAT_RGBA8, Color(1, 1, 1, 0), false);
		}
		proxy_width = p_texture->get_width();
		proxy_height = p_texture->get_height();
		notify_property_list_changed();
		emit_changed();
	}
	_blit_frame(current_frame);
}

Ref<Texture2D> AnimatedTexture::get_frame_texture(int p_frame) const {
	ERR_FAIL_INDEX_V(p_frame, MAX_FRAMES, Ref<Texture2D>());
	return frames[p_frame].texture;
}

void AnimatedTexture::set_frame_duration(int p_frame, float p_duration) {
	ERR_FAIL_INDEX(p_frame, MAX_FRAMES);
	frames[p_frame].duration = p_duration;
}

float AnimatedTexture::get_frame_duration(int p_frame) const {
	ERR_FAIL_INDEX_V(p_frame, MAX_FRAMES, 0);
	return frames[p_frame].duration;
}

void AnimatedTexture::set_speed_scale(float p_scale) {
	ERR_FAIL_COND(p_scale < -1000 || p_scale >= 1000);
	speed_scale = p_scale;
}

float AnimatedTexture::get_speed_scale() const {
	return speed_scale;
}

int AnimatedTexture::get_width() const {
	return proxy_width;
}

int AnimatedTexture::get_height() const {
	return proxy_height;
}

RID AnimatedTexture::get_rid() const {
	return proxy;
}

bool AnimatedTexture::has_alpha() const {
	if (frames[current_frame].texture.is_null()) {
		return false;
	}

	return frames[current_frame].texture->has_alpha();
}

Ref<Image> AnimatedTexture::get_image() const {
	if (frames[current_frame].texture.is_null()) {
		return Ref<Image>();
	}

	return frames[current_frame].texture->get_image();
}

bool AnimatedTexture::is_pixel_opaque(int p_x, int p_y) const {
	if (frames[current_frame].texture.is_valid()) {
		return frames[current_frame].texture->is_pixel_opaque(p_x, p_y);
	}
	return true;
}

void AnimatedTexture::_validate_property(PropertyInfo &p_property) const {
	String prop = p_property.name;
	if (prop.begins_with("frame_")) {
		int frame = prop.get_slicec('/', 0).get_slicec('_', 1).to_int();
		if (frame >= frame_count) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

void AnimatedTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, get_size()), get_rid(), false, p_modulate, p_transpose);
	RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, 1.0, 0.0, 1.0, 0.0);
}

void AnimatedTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, get_rid(), p_tile, p_modulate, p_transpose);
	RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, 1.0, 0.0, 1.0, 0.0);
}

void AnimatedTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, get_rid(), p_src_rect, p_modulate, p_transpose, p_clip_uv);
	RenderingServer::get_singleton()->canvas_item_add_animation_slice(p_canvas_item, 1.0, 0.0, 1.0, 0.0);
}

void AnimatedTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_frames", "frames"), &AnimatedTexture::set_frames);
	ClassDB::bind_method(D_METHOD("get_frames"), &AnimatedTexture::get_frames);

	ClassDB::bind_method(D_METHOD("set_current_frame", "frame"), &AnimatedTexture::set_current_frame);
	ClassDB::bind_method(D_METHOD("get_current_frame"), &AnimatedTexture::get_current_frame);

	ClassDB::bind_method(D_METHOD("set_pause", "pause"), &AnimatedTexture::set_pause);
	ClassDB::bind_method(D_METHOD("get_pause"), &AnimatedTexture::get_pause);

	ClassDB::bind_method(D_METHOD("set_one_shot", "one_shot"), &AnimatedTexture::set_one_shot);
	ClassDB::bind_method(D_METHOD("get_one_shot"), &AnimatedTexture::get_one_shot);

	ClassDB::bind_method(D_METHOD("set_speed_scale", "scale"), &AnimatedTexture::set_speed_scale);
	ClassDB::bind_method(D_METHOD("get_speed_scale"), &AnimatedTexture::get_speed_scale);

	ClassDB::bind_method(D_METHOD("set_frame_texture", "frame", "texture"), &AnimatedTexture::set_frame_texture);
	ClassDB::bind_method(D_METHOD("get_frame_texture", "frame"), &AnimatedTexture::get_frame_texture);

	ClassDB::bind_method(D_METHOD("set_frame_duration", "frame", "duration"), &AnimatedTexture::set_frame_duration);
	ClassDB::bind_method(D_METHOD("get_frame_duration", "frame"), &AnimatedTexture::get_frame_duration);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "frames", PROPERTY_HINT_RANGE, "1," + itos(MAX_FRAMES), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_frames", "get_frames");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_frame", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_current_frame", "get_current_frame");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pause"), "set_pause", "get_pause");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "one_shot"), "set_one_shot", "get_one_shot");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed_scale", PROPERTY_HINT_RANGE, "-60,60,0.1,or_less,or_greater"), "set_speed_scale", "get_speed_scale");

	for (int i = 0; i < MAX_FRAMES; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "frame_" + itos(i) + "/texture", PROPERTY_HINT_RESOURCE_TYPE, Texture2D::get_class_static(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_frame_texture", "get_frame_texture", i);
		ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "frame_" + itos(i) + "/duration", PROPERTY_HINT_RANGE, "0.0,16.0,0.01,or_greater,suffix:s", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_frame_duration", "get_frame_duration", i);
	}

	BIND_CONSTANT(MAX_FRAMES);
}

void AnimatedTexture::_finish_non_thread_safe_setup() {
	RenderingServer::get_singleton()->connect("frame_pre_draw", callable_mp(this, &AnimatedTexture::_update_proxy));
}

AnimatedTexture::AnimatedTexture() {
	String code = R"(
// AnimatedTexture Blit Shader.

shader_type texture_blit;
render_mode blend_disabled;

uniform sampler2D source_texture0 : hint_blit_source0;

void blit() {
	// Copies from each whole source texture to a rect on each output texture.
	COLOR0 = texture(source_texture0, UV);
}
)";

	blit_shader = RS::get_singleton()->shader_create_from_code(code);
	blit_material = RS::get_singleton()->material_create_from_shader(RID(), 0, blit_shader);

	proxy = RenderingServer::get_singleton()->texture_drawable_create(proxy_width, proxy_height, RS::TEXTURE_DRAWABLE_FORMAT_RGBA8, Color(1, 1, 1, 0), false);
	MessageQueue::get_main_singleton()->push_callable(callable_mp(this, &AnimatedTexture::_finish_non_thread_safe_setup));
}

AnimatedTexture::~AnimatedTexture() {
	if (proxy.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RenderingServer::get_singleton()->free_rid(proxy);
	}
	RS::get_singleton()->free_rid(blit_material);
	RS::get_singleton()->free_rid(blit_shader);
}
