/**************************************************************************/
/*  image_loader_libjpeg_turbo.cpp                                        */
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

#include "image_loader_libjpeg_turbo.h"

#include "core/io/file_access_memory.h"

#include <turbojpeg.h>

Error jpeg_turbo_load_image_from_buffer(Image *p_image, const uint8_t *p_buffer, int p_buffer_len) {
	tjhandle tj_instance = tj3Init(TJINIT_DECOMPRESS);
	if (tj_instance == NULL) {
		return FAILED;
	}

	if (tj3DecompressHeader(tj_instance, p_buffer, p_buffer_len) < 0) {
		tj3Destroy(tj_instance);
		return ERR_FILE_CORRUPT;
	}

	const unsigned int width = tj3Get(tj_instance, TJPARAM_JPEGWIDTH);
	const unsigned int height = tj3Get(tj_instance, TJPARAM_JPEGHEIGHT);
	const TJCS colorspace = (TJCS)tj3Get(tj_instance, TJPARAM_COLORSPACE);

	if (tj3Get(tj_instance, TJPARAM_PRECISION) > 8) {
		// Proceed anyway and convert to rgb8?
		tj3Destroy(tj_instance);
		return ERR_UNAVAILABLE;
	}

	TJPF tj_pixel_format;
	Image::Format gd_pixel_format;
	if (colorspace == TJCS_GRAY) {
		tj_pixel_format = TJPF_GRAY;
		gd_pixel_format = Image::FORMAT_L8;
	} else {
		// Force everything else (RGB, CMYK etc) into RGB8.
		tj_pixel_format = TJPF_RGB;
		gd_pixel_format = Image::FORMAT_RGB8;
	}

	Vector<uint8_t> data;
	data.resize(width * height * tjPixelSize[tj_pixel_format]);

	if (tj3Decompress8(tj_instance, p_buffer, p_buffer_len, data.ptrw(), 0, tj_pixel_format) < 0) {
		tj3Destroy(tj_instance);
		return ERR_FILE_CORRUPT;
	}

	tj3Destroy(tj_instance);
	p_image->set_data(width, height, false, gd_pixel_format, data);
	return OK;
}

static int _get_exif_orientation(Ref<FileAccess> p_file) {
	const uint64_t file_size = p_file->get_length();

	// JPEG APP1 segment.
	bool app1_found = false;
	bool prev_ff = false;
	while (!p_file->eof_reached()) {
		uint8_t cur = p_file->get_8();
		if (prev_ff && cur == 0xE1) {
			app1_found = true;
			break;
		}
		prev_ff = (cur == 0xFF);
	}
	if (!app1_found) {
		return -1;
	}

	// 2 bytes - Segment length.
	// 6 bytes - "Exif\0\0".
	// 2 bytes - TIFF Header: Endianness.
	// 2 bytes - TIFF Header: Magic 42.
	// 4 bytes - TIFF Header: Offset to IFD0.
	if (file_size - p_file->get_position() < 16) {
		return -1;
	}
	p_file->seek(p_file->get_position() + 2); // Skip length.
	if (memcmp(p_file->get_buffer(6).ptr(), "Exif\0\0", 6) != 0) {
		return -1;
	}
	const uint64_t tiff_header_start = p_file->get_position();
	switch (p_file->get_16()) {
		case 0x4949: { // "II" - Little endian.
			p_file->set_big_endian(false);
		} break;
		case 0x4D4D: { // "MM" - Big endian.
			p_file->set_big_endian(true);
		} break;
		default: {
			ERR_FAIL_V_MSG(-1, "Invalid Exif endianness.");
		} break;
	}
	ERR_FAIL_COND_V_MSG(p_file->get_16() != 0x002A, -1, "Invalid Exif TIFF magic.");

	const uint32_t offset = tiff_header_start + p_file->get_32();
	if (file_size - offset < 2) {
		return -1;
	}
	p_file->seek(offset);
	const uint16_t entry_count = p_file->get_16();

	for (unsigned int i = 0; i < entry_count; i++) {
		// 2 bytes - Tag ID
		// 2 bytes - Data type
		// 4 bytes - Number of items
		// 4 bytes - Value or value offset
		if (file_size - p_file->get_position() < 12) {
			return -1;
		}
		const uint16_t tag = p_file->get_16();
		const uint16_t type = p_file->get_16();
		const uint32_t count = p_file->get_32();

		// Orientation tag & SHORT type.
		if (tag != 0x0112 || type != 3 || count != 1) {
			p_file->seek(p_file->get_position() + 4); // Skip value/offset.
			continue;
		}
		int orientation = p_file->get_16(); // SHORT is left-justified within the 4 bytes.
		ERR_FAIL_COND_V_MSG(orientation <= 0 || orientation > 8, -1, "Invalid Exif orientation value.");
		return orientation;
	}

	return -1;
}

static void _fix_orientation(Ref<Image> p_image, Ref<FileAccess> p_file) {
	switch (_get_exif_orientation(p_file)) {
		case 2: {
			p_image->flip_x();
		} break;
		case 3: {
			p_image->rotate_180();
		} break;
		case 4: {
			p_image->flip_y();
		} break;
		case 5: {
			p_image->rotate_90(CLOCKWISE);
			p_image->flip_x();
		} break;
		case 6: {
			p_image->rotate_90(CLOCKWISE);
		} break;
		case 7: {
			p_image->rotate_90(COUNTERCLOCKWISE);
			p_image->flip_x();
		} break;
		case 8: {
			p_image->rotate_90(COUNTERCLOCKWISE);
		} break;
	}
}

Error ImageLoaderLibJPEGTurbo::load_image(Ref<Image> p_image, Ref<FileAccess> f, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale) {
	Vector<uint8_t> src_image;
	uint64_t src_image_len = f->get_length();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	src_image.resize(src_image_len);

	uint8_t *w = src_image.ptrw();

	f->get_buffer(&w[0], src_image_len);

	Error err = jpeg_turbo_load_image_from_buffer(p_image.ptr(), w, src_image_len);
	if (err == OK && p_flags.has_flag(FLAG_FIX_ORIENTATION)) {
		f->seek(0);
		_fix_orientation(p_image, f);
	}

	return err;
}

void ImageLoaderLibJPEGTurbo::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("jpg");
	p_extensions->push_back("jpeg");
}

static Ref<Image> _jpeg_turbo_mem_loader_func(const uint8_t *p_data, int p_size, bool p_fix_orientation) {
	Ref<Image> img;
	img.instantiate();
	Error err = jpeg_turbo_load_image_from_buffer(img.ptr(), p_data, p_size);
	ERR_FAIL_COND_V(err, Ref<Image>());

	if (p_fix_orientation) {
		Ref<FileAccessMemory> file;
		file.instantiate();
		if (file->open_custom(p_data, p_size) == OK) {
			_fix_orientation(img, file);
		}
	}

	return img;
}

static Vector<uint8_t> _jpeg_turbo_buffer_save_func(const Ref<Image> &p_img, float p_quality) {
	Vector<uint8_t> output;

	ERR_FAIL_COND_V(p_img.is_null() || p_img->is_empty(), output);

	Ref<Image> image = p_img->duplicate();
	if (image->is_compressed()) {
		Error error = image->decompress();
		ERR_FAIL_COND_V_MSG(error != OK, output, "Couldn't decompress image.");
	}

	if (image->get_format() != Image::FORMAT_RGB8) {
		// Allow grayscale L8?
		image = image->duplicate();
		image->convert(Image::FORMAT_RGB8);
	}

	tjhandle tj_instance = tj3Init(TJINIT_COMPRESS);
	ERR_FAIL_COND_V_MSG(tj_instance == NULL, output, "Couldn't create tjhandle");

	if (tj3Set(tj_instance, TJPARAM_QUALITY, (int)(p_quality * 100)) < 0) {
		tj3Destroy(tj_instance);
		ERR_FAIL_V_MSG(output, "Couldn't set jpg quality");
	}

	if (tj3Set(tj_instance, TJPARAM_PRECISION, 8) < 0) {
		tj3Destroy(tj_instance);
		ERR_FAIL_V_MSG(output, "Couldn't set jpg precision");
	}

	if (tj3Set(tj_instance, TJPARAM_SUBSAMP, TJSAMP_420) < 0) {
		tj3Destroy(tj_instance);
		ERR_FAIL_V_MSG(output, "Couldn't set jpg subsamples");
	}

	// If the godot image format is `Image::FORMAT_L8` we could set the appropriate
	// color space here rather than defaulting to RGB.

	unsigned char *jpeg_buff = NULL;
	size_t jpeg_size = 0;
	int code = tj3Compress8(
			tj_instance,
			image->get_data().ptr(),
			image->get_width(),
			0,
			image->get_height(),
			TJPF_RGB,
			&jpeg_buff,
			&jpeg_size);

	if (code < 0) {
		tj3Destroy(tj_instance);
		tj3Free(jpeg_buff);
		ERR_FAIL_V_MSG(output, "Couldn't compress jpg");
	}

	output.resize(jpeg_size);
	memcpy(output.ptrw(), jpeg_buff, jpeg_size);

	tj3Destroy(tj_instance);
	tj3Free(jpeg_buff);

	return output;
}

static Error _jpeg_turbo_save_func(const String &p_path, const Ref<Image> &p_img, float p_quality) {
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V_MSG(err, err, vformat("Can't save JPG at path: '%s'.", p_path));

	Vector<uint8_t> data = _jpeg_turbo_buffer_save_func(p_img, p_quality);
	ERR_FAIL_COND_V(data.size() == 0, FAILED);
	ERR_FAIL_COND_V_MSG(!file->store_buffer(data.ptr(), data.size()), FAILED, "Failed writing jpg to file");

	return OK;
}

ImageLoaderLibJPEGTurbo::ImageLoaderLibJPEGTurbo() {
	Image::_jpg_mem_loader_func = _jpeg_turbo_mem_loader_func;
	Image::save_jpg_func = _jpeg_turbo_save_func;
	Image::save_jpg_buffer_func = _jpeg_turbo_buffer_save_func;
}
