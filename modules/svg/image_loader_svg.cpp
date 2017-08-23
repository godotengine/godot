/*************************************************************************/
/*  image_loader_svg.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "image_loader_svg.h"

#include "os/os.h"
#include "print_string.h"

#include <ustring.h>

void SVGRasterizer::rasterize(NSVGimage *p_image, float p_tx, float p_ty, float p_scale, unsigned char *p_dst, int p_w, int p_h, int p_stride) {
	nsvgRasterize(rasterizer, p_image, p_tx, p_ty, p_scale, p_dst, p_w, p_h, p_stride);
}

SVGRasterizer::SVGRasterizer() {
	rasterizer = nsvgCreateRasterizer();
}
SVGRasterizer::~SVGRasterizer() {
	nsvgDeleteRasterizer(rasterizer);
}

SVGRasterizer ImageLoaderSVG::rasterizer;

Error ImageLoaderSVG::_create_image(Ref<Image> p_image, const PoolVector<uint8_t> *p_data, float p_scale, bool upsample) {
	NSVGimage *svg_image;
	PoolVector<uint8_t>::Read src_r = p_data->read();
	svg_image = nsvgParse((char *)src_r.ptr(), "px", 96);
	if (svg_image == NULL) {
		ERR_PRINT("SVG Corrupted");
		return ERR_FILE_CORRUPT;
	}

	float upscale = upsample ? 2.0 : 1.0;

	int w = (int)(svg_image->width * p_scale * upscale);
	int h = (int)(svg_image->height * p_scale * upscale);

	PoolVector<uint8_t> dst_image;
	dst_image.resize(w * h * 4);

	PoolVector<uint8_t>::Write dw = dst_image.write();

	rasterizer.rasterize(svg_image, 0, 0, p_scale * upscale, (unsigned char *)dw.ptr(), w, h, w * 4);

	dw = PoolVector<uint8_t>::Write();
	p_image->create(w, h, false, Image::FORMAT_RGBA8, dst_image);
	if (upsample)
		p_image->shrink_x2();

	nsvgDelete(svg_image);

	return OK;
}

Error ImageLoaderSVG::create_image_from_string(Ref<Image> p_image, const char *svg_str, float p_scale, bool upsample) {

	size_t str_len = strlen(svg_str);
	PoolVector<uint8_t> src_data;
	src_data.resize(str_len + 1);
	PoolVector<uint8_t>::Write src_w = src_data.write();
	memcpy(src_w.ptr(), svg_str, str_len + 1);

	return _create_image(p_image, &src_data, p_scale, upsample);
}

Error ImageLoaderSVG::load_image(Ref<Image> p_image, FileAccess *f, bool p_force_linear, float p_scale) {

	uint32_t size = f->get_len();
	PoolVector<uint8_t> src_image;
	src_image.resize(size + 1);
	PoolVector<uint8_t>::Write src_w = src_image.write();
	f->get_buffer(src_w.ptr(), size);
	src_w.ptr()[size] = '\0';

	return _create_image(p_image, &src_image, p_scale, 1.0);
}

void ImageLoaderSVG::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("svg");
	p_extensions->push_back("svgz");
}

ImageLoaderSVG::ImageLoaderSVG() {
}
