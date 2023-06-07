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

#ifndef IMAGE_LOADER_SVG_H
#define IMAGE_LOADER_SVG_H

#include "core/io/image_loader.h"
#include "core/ustring.h"

/**
	@author Daniel Ramirez <djrmuv@gmail.com>
*/

// Forward declare and include thirdparty headers in .cpp.
struct NSVGrasterizer;
struct NSVGimage;

class SVGRasterizer {
	NSVGrasterizer *rasterizer;

public:
	void rasterize(NSVGimage *p_image, float p_tx, float p_ty, float p_scale, unsigned char *p_dst, int p_w, int p_h, int p_stride);

	SVGRasterizer();
	~SVGRasterizer();
};

class ImageLoaderSVG : public ImageFormatLoader {
	static struct ReplaceColors {
		List<uint32_t> old_colors;
		List<uint32_t> new_colors;
	} replace_colors;
	static SVGRasterizer rasterizer;
	static void _convert_colors(NSVGimage *p_svg_image);

public:
	static void set_convert_colors(Dictionary *p_replace_color = nullptr);
	static Error create_image(Ref<Image> p_image, const PoolVector<uint8_t> *p_data, float p_scale, bool upsample, bool convert_colors = false);
	static Error create_image_from_string(Ref<Image> p_image, const char *p_svg_str, float p_scale, bool upsample, bool convert_colors = false);

	virtual Error load_image(Ref<Image> p_image, FileAccess *f, bool p_force_linear, float p_scale);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	ImageLoaderSVG();
};

#endif // IMAGE_LOADER_SVG_H
