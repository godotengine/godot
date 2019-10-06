/*************************************************************************/
/*  animated_image.h                                                     */
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

#ifndef ANIMATED_IMAGE_H
#define ANIMATED_IMAGE_H

#include "core/image.h"
#include "scene/2d/animated_sprite.h"

class AnimatedImage;

typedef Error (*LoadAnimatedImageFunction)(Ref<AnimatedImage> &r_animated_image, const Variant &source, int max_frames);

class AnimatedImage : public Reference {
	GDCLASS(AnimatedImage, Reference);

	struct Frame {

		Ref<Image> image;
		float delay;
	};
	Vector<Frame> frames;

protected:
	static void _bind_methods();

public:
	enum SourceFormat {
		GIF
	};

	enum ImportType {
		ANIMATED_TEXTURE,
		SPRITE_FRAMES
	};

	enum SourceType {
		FILE,
		BUFFER
	};

	static LoadAnimatedImageFunction _load_gif;

	Error load_from_file(const String &p_path, int max_frames = 0);
	Error load_from_buffer(const PoolByteArray &p_data, int max_frames = 0);

	Ref<AnimatedTexture> to_animated_texture(uint32_t p_flags = 0, int max_frames = 0) const;
	Ref<SpriteFrames> to_sprite_frames(uint32_t p_flags = 0, int max_frames = 0) const;

	void add_frame(const Ref<Image> &p_image, float p_delay, int p_idx = -1);
	void remove_frame(int p_idx);

	void set_image(int p_idx, const Ref<Image> &p_image);
	Ref<Image> get_image(int p_idx) const;

	void set_delay(int p_idx, float p_delay);
	float get_delay(int p_idx) const;

	int get_frames() const;

	void clear();
};

#endif // ANIMATED_IMAGE_H
