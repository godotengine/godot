/*************************************************************************/
/*  image_loader_png.cpp                                                 */
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
#include "image_loader_png.h"

#include "os/os.h"
#include "print_string.h"

#include <string.h>

void ImageLoaderPNG::_read_png_data(png_structp png_ptr, png_bytep data, png_size_t p_length) {

	FileAccess *f = (FileAccess *)png_get_io_ptr(png_ptr);
	f->get_buffer((uint8_t *)data, p_length);
}

/*
png_structp png_ptr = png_create_read_struct_2
    (PNG_LIBPNG_VER_STRING, (png_voidp)user_error_ptr,
     user_error_fn, user_warning_fn, (png_voidp)
     user_mem_ptr, user_malloc_fn, user_free_fn);
*/
static png_voidp _png_malloc_fn(png_structp png_ptr, png_size_t size) {

	return memalloc(size);
}

static void _png_free_fn(png_structp png_ptr, png_voidp ptr) {

	memfree(ptr);
}

static void _png_error_function(png_structp, png_const_charp text) {

	ERR_PRINT(text);
}

static void _png_warn_function(png_structp, png_const_charp text) {

	WARN_PRINT(text);
}

typedef void(PNGAPI *png_error_ptr) PNGARG((png_structp, png_const_charp));

Error ImageLoaderPNG::_load_image(void *rf_up, png_rw_ptr p_func, Image *p_image) {

	png_structp png;
	png_infop info;

	//png = png_create_read_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL, NULL, NULL);

	png = png_create_read_struct_2(PNG_LIBPNG_VER_STRING, (png_voidp)NULL, _png_error_function, _png_warn_function, (png_voidp)NULL,
			_png_malloc_fn, _png_free_fn);

	ERR_FAIL_COND_V(!png, ERR_OUT_OF_MEMORY);

	info = png_create_info_struct(png);
	if (!info) {
		png_destroy_read_struct(&png, NULL, NULL);
		ERR_PRINT("Out of Memory");
		return ERR_OUT_OF_MEMORY;
	}

	if (setjmp(png_jmpbuf(png))) {

		png_destroy_read_struct(&png, NULL, NULL);
		ERR_PRINT("PNG Corrupted");
		return ERR_FILE_CORRUPT;
	}

	png_set_read_fn(png, (void *)rf_up, p_func);

	png_uint_32 width, height;
	int depth, color;

	png_read_info(png, info);
	png_get_IHDR(png, info, &width, &height, &depth, &color, NULL, NULL, NULL);

	//https://svn.gov.pt/projects/ccidadao/repository/middleware-offline/trunk/_src/eidmw/FreeImagePTEiD/Source/FreeImage/PluginPNG.cpp
	//png_get_text(png,info,)
	/*
	printf("Image width:%i\n", width);
	printf("Image Height:%i\n", height);
	printf("Bit depth:%i\n", depth);
	printf("Color type:%i\n", color);
	*/

	bool update_info = false;

	if (depth < 8) { //only bit dept 8 per channel is handled

		png_set_packing(png);
		update_info = true;
	};

	if (png_get_color_type(png, info) == PNG_COLOR_TYPE_PALETTE) {
		png_set_palette_to_rgb(png);
		update_info = true;
	}

	if (depth > 8) {
		png_set_strip_16(png);
		update_info = true;
	}

	if (png_get_valid(png, info, PNG_INFO_tRNS)) {
		//png_set_expand_gray_1_2_4_to_8(png);
		png_set_tRNS_to_alpha(png);
		update_info = true;
	}

	if (update_info) {
		png_read_update_info(png, info);
		png_get_IHDR(png, info, &width, &height, &depth, &color, NULL, NULL, NULL);
	}

	int components = 0;

	Image::Format fmt;
	switch (color) {

		case PNG_COLOR_TYPE_GRAY: {

			fmt = Image::FORMAT_L8;
			components = 1;
		} break;
		case PNG_COLOR_TYPE_GRAY_ALPHA: {

			fmt = Image::FORMAT_LA8;
			components = 2;
		} break;
		case PNG_COLOR_TYPE_RGB: {

			fmt = Image::FORMAT_RGB8;
			components = 3;
		} break;
		case PNG_COLOR_TYPE_RGB_ALPHA: {

			fmt = Image::FORMAT_RGBA8;
			components = 4;
		} break;
		default: {

			ERR_PRINT("INVALID PNG TYPE");
			png_destroy_read_struct(&png, &info, NULL);
			return ERR_UNAVAILABLE;
		} break;
	}

	//int rowsize = png_get_rowbytes(png, info);
	int rowsize = components * width;

	PoolVector<uint8_t> dstbuff;

	dstbuff.resize(rowsize * height);

	PoolVector<uint8_t>::Write dstbuff_write = dstbuff.write();

	uint8_t *data = dstbuff_write.ptr();

	uint8_t **row_p = memnew_arr(uint8_t *, height);

	for (unsigned int i = 0; i < height; i++) {
		row_p[i] = &data[components * width * i];
	}

	png_read_image(png, (png_bytep *)row_p);

	memdelete_arr(row_p);

	p_image->create(width, height, 0, fmt, dstbuff);

	png_destroy_read_struct(&png, &info, NULL);

	return OK;
}

Error ImageLoaderPNG::load_image(Image *p_image, FileAccess *f) {

	Error err = _load_image(f, _read_png_data, p_image);
	f->close();

	return err;
}

void ImageLoaderPNG::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("png");
}

struct PNGReadStatus {

	int offset;
	int size;
	const unsigned char *image;
};

static void user_read_data(png_structp png_ptr, png_bytep data, png_size_t p_length) {

	PNGReadStatus *rstatus;
	rstatus = (PNGReadStatus *)png_get_io_ptr(png_ptr);

	png_size_t to_read = p_length;
	if (rstatus->size >= 0) {
		to_read = MIN(p_length, rstatus->size - rstatus->offset);
	}
	memcpy(data, &rstatus->image[rstatus->offset], to_read);
	rstatus->offset += to_read;

	if (to_read < p_length) {
		memset(&data[to_read], 0, p_length - to_read);
	}
}

static Image _load_mem_png(const uint8_t *p_png, int p_size) {

	PNGReadStatus prs;
	prs.image = p_png;
	prs.offset = 0;
	prs.size = p_size;

	Image img;
	Error err = ImageLoaderPNG::_load_image(&prs, user_read_data, &img);
	ERR_FAIL_COND_V(err, Image());

	return img;
}

static Image _lossless_unpack_png(const PoolVector<uint8_t> &p_data) {

	int len = p_data.size();
	PoolVector<uint8_t>::Read r = p_data.read();
	ERR_FAIL_COND_V(r[0] != 'P' || r[1] != 'N' || r[2] != 'G' || r[3] != ' ', Image());
	return _load_mem_png(&r[4], len - 4);
}

static void _write_png_data(png_structp png_ptr, png_bytep data, png_size_t p_length) {

	PoolVector<uint8_t> &v = *(PoolVector<uint8_t> *)png_get_io_ptr(png_ptr);
	int vs = v.size();

	v.resize(vs + p_length);
	PoolVector<uint8_t>::Write w = v.write();
	copymem(&w[vs], data, p_length);
	//print_line("png write: "+itos(p_length));
}

static PoolVector<uint8_t> _lossless_pack_png(const Image &p_image) {

	Image img = p_image;
	if (img.is_compressed())
		img.decompress();

	ERR_FAIL_COND_V(img.is_compressed(), PoolVector<uint8_t>());

	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep *row_pointers;

	/* initialize stuff */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	ERR_FAIL_COND_V(!png_ptr, PoolVector<uint8_t>());

	info_ptr = png_create_info_struct(png_ptr);

	ERR_FAIL_COND_V(!info_ptr, PoolVector<uint8_t>());

	if (setjmp(png_jmpbuf(png_ptr))) {
		ERR_FAIL_V(PoolVector<uint8_t>());
	}
	PoolVector<uint8_t> ret;
	ret.push_back('P');
	ret.push_back('N');
	ret.push_back('G');
	ret.push_back(' ');

	png_set_write_fn(png_ptr, &ret, _write_png_data, NULL);

	/* write header */
	if (setjmp(png_jmpbuf(png_ptr))) {
		ERR_FAIL_V(PoolVector<uint8_t>());
	}

	int pngf = 0;
	int cs = 0;

	switch (img.get_format()) {

		case Image::FORMAT_L8: {

			pngf = PNG_COLOR_TYPE_GRAY;
			cs = 1;
		} break;
		case Image::FORMAT_LA8: {

			pngf = PNG_COLOR_TYPE_GRAY_ALPHA;
			cs = 2;
		} break;
		case Image::FORMAT_RGB8: {

			pngf = PNG_COLOR_TYPE_RGB;
			cs = 3;
		} break;
		case Image::FORMAT_RGBA8: {

			pngf = PNG_COLOR_TYPE_RGB_ALPHA;
			cs = 4;
		} break;
		default: {

			if (img.detect_alpha()) {

				img.convert(Image::FORMAT_RGBA8);
				pngf = PNG_COLOR_TYPE_RGB_ALPHA;
				cs = 4;
			} else {

				img.convert(Image::FORMAT_RGB8);
				pngf = PNG_COLOR_TYPE_RGB;
				cs = 3;
			}
		}
	}

	int w = img.get_width();
	int h = img.get_height();
	png_set_IHDR(png_ptr, info_ptr, w, h,
			8, pngf, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	/* write bytes */
	if (setjmp(png_jmpbuf(png_ptr))) {
		ERR_FAIL_V(PoolVector<uint8_t>());
	}

	PoolVector<uint8_t>::Read r = img.get_data().read();

	row_pointers = (png_bytep *)memalloc(sizeof(png_bytep) * h);
	for (int i = 0; i < h; i++) {

		row_pointers[i] = (png_bytep)&r[i * w * cs];
	}
	png_write_image(png_ptr, row_pointers);

	memfree(row_pointers);

	/* end write */
	if (setjmp(png_jmpbuf(png_ptr))) {

		ERR_FAIL_V(PoolVector<uint8_t>());
	}

	png_write_end(png_ptr, NULL);

	return ret;
}

ImageLoaderPNG::ImageLoaderPNG() {

	Image::_png_mem_loader_func = _load_mem_png;
	Image::lossless_unpacker = _lossless_unpack_png;
	Image::lossless_packer = _lossless_pack_png;
}
