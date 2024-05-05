/**************************************************************************/
/*  image_loader_svg.h                                                    */
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

#ifndef LOTTIE_SHEET_H
#define LOTTIE_SHEET_H

#include "core/io/json.h"
#include "core/io/resource.h"

#include <thorvg.h>

class LottieSheet : public Resource {
	GDCLASS(LottieSheet, Resource);

	std::unique_ptr<tvg::SwCanvas> sw_canvas = tvg::SwCanvas::gen();
	std::unique_ptr<tvg::Animation> animation = tvg::Animation::gen();
	tvg::Picture *picture = animation->picture();
	Ref<Image> image;
	uint32_t *buffer = nullptr;
	Ref<JSON> json = memnew(JSON);

	float scale;
	uint32_t width, height;

	void _load_data(String p_string, float p_scale);

protected:
	static void _bind_methods();

public:
	static Ref<LottieSheet> load_string(String p_string, float p_scale = 1);

	static Ref<LottieSheet> load_json(Ref<JSON> p_json, float p_scale = 1);

	Ref<JSON> get_json();
	void set_json(Ref<JSON> p_json);

	float get_scale();
	void set_scale(float p_scale);

	void update_image(int frame);

	Ref<Image> get_image() { return image; };

	Ref<Image> get_frame_image(int frame) {
		update_image(frame);
		return image;
	};

	Vector2i get_image_size() {
		return Vector2i(width, height);
	}

	float total_frame() { return animation->totalFrame(); };

	float duration() { return animation->duration(); };

	~LottieSheet();
};

#endif // LOTTIE_SHEET_H
