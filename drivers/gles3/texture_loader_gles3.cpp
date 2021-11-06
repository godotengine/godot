/*************************************************************************/
/*  texture_loader_gles3.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "texture_loader_gles3.h"
#ifdef GLES3_BACKEND_ENABLED

#include "core/io/file_access.h"
#include "core/string/print_string.h"

#include <string.h>

RES ResourceFormatGLES2Texture::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	unsigned int width = 8;
	unsigned int height = 8;

	//We just use some format
	Image::Format fmt = Image::FORMAT_RGB8;
	int rowsize = 3 * width;

	Vector<uint8_t> dstbuff;

	dstbuff.resize(rowsize * height);

	uint8_t **row_p = memnew_arr(uint8_t *, height);

	for (unsigned int i = 0; i < height; i++) {
		row_p[i] = 0; //No colors any more, I want them to turn black
	}

	memdelete_arr(row_p);

	Ref<Image> img = memnew(Image(width, height, 0, fmt, dstbuff));

	Ref<ImageTexture> texture = memnew(ImageTexture);
	texture->create_from_image(img);

	if (r_error)
		*r_error = OK;

	return texture;
}

void ResourceFormatGLES2Texture::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("bmp");
	p_extensions->push_back("dds");
	p_extensions->push_back("exr");
	p_extensions->push_back("jpeg");
	p_extensions->push_back("jpg");
	p_extensions->push_back("hdr");
	p_extensions->push_back("pkm");
	p_extensions->push_back("png");
	p_extensions->push_back("pvr");
	p_extensions->push_back("svg");
	p_extensions->push_back("svgz");
	p_extensions->push_back("tga");
	p_extensions->push_back("webp");
}

bool ResourceFormatGLES2Texture::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "Texture2D");
}

String ResourceFormatGLES2Texture::get_resource_type(const String &p_path) const {
	String extension = p_path.get_extension().to_lower();
	if (
			extension == "bmp" ||
			extension == "dds" ||
			extension == "exr" ||
			extension == "jpeg" ||
			extension == "jpg" ||
			extension == "hdr" ||
			extension == "pkm" ||
			extension == "png" ||
			extension == "pvr" ||
			extension == "svg" ||
			extension == "svgz" ||
			extension == "tga" ||
			extension == "webp") {
		return "ImageTexture";
	}

	return "";
}

#endif
