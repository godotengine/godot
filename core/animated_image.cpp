/*************************************************************************/
/*  animated_image.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "animated_image.h"

LoadAnimatedImageFunction AnimatedImage::_load_gif = NULL;

Error AnimatedImage::load_from_file(const String &p_path, int max_frames) {

	clear();
	String ext = p_path.get_extension().to_lower();
	if (ext == "gif") {

		Ref<AnimatedImage> animated_image = Ref<AnimatedImage>(this);
		return _load_gif(animated_image, p_path, max_frames);
	} else {

		ERR_PRINTS("Unrecognized image: " + p_path);
		return ERR_FILE_UNRECOGNIZED;
	}
}

Error AnimatedImage::load_from_buffer(const PoolByteArray &p_data, int max_frames) {

	clear();
	if (p_data[0] == 'G') {

		Ref<AnimatedImage> animated_image = Ref<AnimatedImage>(this);
		return _load_gif(animated_image, p_data, max_frames);
	} else {

		ERR_PRINTS("Unrecognized image.");
		return ERR_FILE_UNRECOGNIZED;
	}
}

Ref<AnimatedTexture> AnimatedImage::to_animated_texture(uint32_t p_flags, int max_frames) const {

	if (max_frames <= 0 || max_frames > AnimatedTexture::MAX_FRAMES)
		max_frames = AnimatedTexture::MAX_FRAMES;

	int frames_size = MIN(get_frames(), max_frames);

	Ref<AnimatedTexture> animated_texture = memnew(AnimatedTexture);
	animated_texture->set_fps(0); // Allow each frame to have a custom delay.

	for (int i = 0; i < frames_size; i++) {

		Ref<ImageTexture> texture = memnew(ImageTexture);
		texture->create_from_image(frames[i].image, p_flags);
		animated_texture->set_frame_texture(i, texture);
		animated_texture->set_frame_delay(i, frames[i].delay);
	}

	animated_texture->set_frames(frames_size);

	return animated_texture;
}

Ref<SpriteFrames> AnimatedImage::to_sprite_frames(uint32_t p_flags, int max_frames) const {

	if (max_frames <= 0)
		max_frames = get_frames();

	int frames_size = MIN(get_frames(), max_frames);

	Ref<SpriteFrames> sprite_frames = memnew(SpriteFrames);

	float time = frames[0].delay;
	for (int i = 0; i < frames_size; i++) {

		Ref<ImageTexture> texture = memnew(ImageTexture);
		texture->create_from_image(frames[i].image, p_flags);
		sprite_frames->add_frame("default", texture);
		time += frames[i].delay;
	}

	time /= max_frames; // Use the average time of all the frames.
	sprite_frames->set_animation_speed("default", 1.0 / time);

	return sprite_frames;
}

void AnimatedImage::add_frame(const Ref<Image> &p_image, float p_delay, int p_idx) {

	ERR_FAIL_COND(p_idx > get_frames() - 1);
	Frame frame;
	frame.image = p_image;
	frame.delay = p_delay;

	if (p_idx < 0)
		frames.push_back(frame);
	else
		frames.set(p_idx, frame);
}

void AnimatedImage::remove_frame(int p_idx) {

	ERR_FAIL_COND(p_idx < 0);
	ERR_FAIL_COND(p_idx > get_frames() - 1);
	frames.remove(p_idx);
}

void AnimatedImage::set_image(int p_idx, const Ref<Image> &p_image) {

	ERR_FAIL_COND(p_idx < 0);
	ERR_FAIL_COND(p_idx > get_frames() - 1);
	frames.write[p_idx].image = p_image;
}

Ref<Image> AnimatedImage::get_image(int p_idx) const {

	ERR_FAIL_COND_V(p_idx < 0, Ref<Image>());
	ERR_FAIL_COND_V(p_idx > get_frames() - 1, Ref<Image>());
	return frames[p_idx].image;
}

void AnimatedImage::set_delay(int p_idx, float p_delay) {

	ERR_FAIL_COND(p_idx < 0);
	ERR_FAIL_COND(p_idx > get_frames() - 1);
	frames.write[p_idx].delay = p_delay;
}

float AnimatedImage::get_delay(int p_idx) const {

	ERR_FAIL_COND_V(p_idx < 0, 0);
	ERR_FAIL_COND_V(p_idx > get_frames() - 1, 0);
	return frames[p_idx].delay;
}

int AnimatedImage::get_frames() const {

	return frames.size();
}

void AnimatedImage::clear() {

	frames.clear();
}

void AnimatedImage::_bind_methods() {

	ClassDB::bind_method(D_METHOD("load_from_file", "path", "max_frames"), &AnimatedImage::load_from_file, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("load_from_buffer", "data", "max_frames"), &AnimatedImage::load_from_buffer, DEFVAL(0));

	ClassDB::bind_method(D_METHOD("to_animated_texture", "flags", "max_frames"), &AnimatedImage::to_animated_texture, DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("to_sprite_frames", "flags", "max_frames"), &AnimatedImage::to_sprite_frames, DEFVAL(0), DEFVAL(0));

	ClassDB::bind_method(D_METHOD("add_frame", "image", "delay", "idx"), &AnimatedImage::add_frame, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_frame", "idx"), &AnimatedImage::remove_frame);

	ClassDB::bind_method(D_METHOD("set_image", "idx", "image"), &AnimatedImage::set_image);
	ClassDB::bind_method(D_METHOD("get_image", "idx"), &AnimatedImage::get_image);

	ClassDB::bind_method(D_METHOD("set_delay", "idx", "delay"), &AnimatedImage::set_delay);
	ClassDB::bind_method(D_METHOD("get_delay", "idx"), &AnimatedImage::get_delay);

	ClassDB::bind_method(D_METHOD("get_frames"), &AnimatedImage::get_frames);

	ClassDB::bind_method(D_METHOD("clear"), &AnimatedImage::clear);
}
