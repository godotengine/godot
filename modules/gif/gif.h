/*************************************************************************/
/*  gif.h                                                                */
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

#ifndef GIF_H
#define GIF_H

#include "core/io/animated_image_loader.h"

struct GifFileType;

class Gif {

	GifFileType *gif;

	Error parse_error(Error parse_error, const String &message);
	Error gif_error(int gif_error);

	Error _open(void *source, AnimatedImage::SourceType source_type);
	Error _load_frames(Ref<AnimatedImage> &r_animated_image, int max_frames = 0);
	Error _close();

public:
	Error load_from_file_access(Ref<AnimatedImage> &r_animated_image, FileAccess *f, int max_frames = 0);
	Error load_from_buffer(Ref<AnimatedImage> &r_animated_image, const PoolByteArray &p_data, int max_frames = 0);

	Gif();
};

class AnimatedImageLoaderGIF : public AnimatedImageFormatLoader {

public:
	virtual Error load_animated_image(Ref<AnimatedImage> &r_animated_image, FileAccess *f, int max_frames = 0) const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool recognize_format(AnimatedImage::SourceFormat p_format) const;

	AnimatedImageLoaderGIF();
};

#endif // GIF_H
