/**************************************************************************/
/*  image_loader_bmp.cpp                                                  */
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

#include "image_loader_bmp.h"

#include "core/io/file_access_memory.h"

static uint8_t get_mask_width(uint16_t mask) {
	// Returns number of ones in the binary value of the parameter: mask.
	// Uses a Simple pop_count.
	uint8_t c = 0u;
	for (; mask != 0u; mask &= mask - 1u) {
		c++;
	}
	return c;
}

Error ImageLoaderBMP::convert_to_image(Ref<Image> p_image,
		const uint8_t *p_buffer,
		const uint8_t *p_color_buffer,
		const uint32_t color_table_size,
		const bmp_header_s &p_header) {
	Error err = OK;

	if (p_buffer == nullptr) {
		err = FAILED;
	}

	if (err == OK) {
		size_t index = 0;
		size_t width = (size_t)p_header.bmp_info_header.bmp_width;
		size_t height = (size_t)p_header.bmp_info_header.bmp_height;
		size_t bits_per_pixel = (size_t)p_header.bmp_info_header.bmp_bit_count;

		// Check whether we can load it

		if (bits_per_pixel == 1) {
			// Requires bit unpacking...
			ERR_FAIL_COND_V_MSG(width % 8 != 0, ERR_UNAVAILABLE,
					vformat("1-bpp BMP images must have a width that is a multiple of 8, but the imported BMP is %d pixels wide.", int(width)));
			ERR_FAIL_COND_V_MSG(height % 8 != 0, ERR_UNAVAILABLE,
					vformat("1-bpp BMP images must have a height that is a multiple of 8, but the imported BMP is %d pixels tall.", int(height)));

		} else if (bits_per_pixel == 2) {
			// Requires bit unpacking...
			ERR_FAIL_COND_V_MSG(width % 4 != 0, ERR_UNAVAILABLE,
					vformat("2-bpp BMP images must have a width that is a multiple of 4, but the imported BMP is %d pixels wide.", int(width)));
			ERR_FAIL_COND_V_MSG(height % 4 != 0, ERR_UNAVAILABLE,
					vformat("2-bpp BMP images must have a height that is a multiple of 4, but the imported BMP is %d pixels tall.", int(height)));

		} else if (bits_per_pixel == 4) {
			// Requires bit unpacking...
			ERR_FAIL_COND_V_MSG(width % 2 != 0, ERR_UNAVAILABLE,
					vformat("4-bpp BMP images must have a width that is a multiple of 2, but the imported BMP is %d pixels wide.", int(width)));
			ERR_FAIL_COND_V_MSG(height % 2 != 0, ERR_UNAVAILABLE,
					vformat("4-bpp BMP images must have a height that is a multiple of 2, but the imported BMP is %d pixels tall.", int(height)));
		}

		// Image data (might be indexed)
		Vector<uint8_t> data;
		int data_len = 0;

		if (bits_per_pixel <= 8) { // indexed
			data_len = width * height;
		} else { // color
			data_len = width * height * 4;
		}
		ERR_FAIL_COND_V_MSG(data_len == 0, ERR_BUG, "Couldn't parse the BMP image data.");
		err = data.resize(data_len);

		uint8_t *data_w = data.ptrw();
		uint8_t *write_buffer = data_w;

		const uint32_t width_bytes = width * bits_per_pixel / 8;
		const uint32_t line_width = (width_bytes + 3) & ~3;

		// The actual data traversal is determined by
		// the data width in case of 8/4/2/1 bit images
		const uint32_t w = bits_per_pixel >= 16 ? width : width_bytes;
		const uint8_t *line = p_buffer + (line_width * (height - 1));
		const uint8_t *end_buffer = p_buffer + p_header.bmp_file_header.bmp_file_size - p_header.bmp_file_header.bmp_file_offset;

		for (uint64_t i = 0; i < height; i++) {
			const uint8_t *line_ptr = line;

			for (unsigned int j = 0; j < w; j++) {
				ERR_FAIL_COND_V(line_ptr >= end_buffer, ERR_FILE_CORRUPT);
				switch (bits_per_pixel) {
					case 1: {
						uint8_t color_index = *line_ptr;

						write_buffer[index + 0] = (color_index >> 7) & 1;
						write_buffer[index + 1] = (color_index >> 6) & 1;
						write_buffer[index + 2] = (color_index >> 5) & 1;
						write_buffer[index + 3] = (color_index >> 4) & 1;
						write_buffer[index + 4] = (color_index >> 3) & 1;
						write_buffer[index + 5] = (color_index >> 2) & 1;
						write_buffer[index + 6] = (color_index >> 1) & 1;
						write_buffer[index + 7] = (color_index >> 0) & 1;

						index += 8;
						line_ptr += 1;
					} break;
					case 2: {
						uint8_t color_index = *line_ptr;

						write_buffer[index + 0] = (color_index >> 6) & 3;
						write_buffer[index + 1] = (color_index >> 4) & 3;
						write_buffer[index + 2] = (color_index >> 2) & 3;
						write_buffer[index + 3] = color_index & 3;

						index += 4;
						line_ptr += 1;
					} break;
					case 4: {
						uint8_t color_index = *line_ptr;

						write_buffer[index + 0] = (color_index >> 4) & 0x0f;
						write_buffer[index + 1] = color_index & 0x0f;

						index += 2;
						line_ptr += 1;
					} break;
					case 8: {
						uint8_t color_index = *line_ptr;

						write_buffer[index] = color_index;

						index += 1;
						line_ptr += 1;
					} break;
					case 16: {
						uint16_t rgb = (static_cast<uint16_t>(line_ptr[1]) << 8) | line_ptr[0];
						// A1R5G5B5/X1R5G5B5 => uint16_t
						// [A/X]1R5G2 | G3B5 => uint8_t | uint8_t
						uint8_t ba = (rgb & p_header.bmp_bitfield.alpha_mask) >> p_header.bmp_bitfield.alpha_offset; // Alpha 0b 1000 ...
						uint8_t b0 = (rgb & p_header.bmp_bitfield.red_mask) >> p_header.bmp_bitfield.red_offset; // Red   0b 0111 1100 ...
						uint8_t b1 = (rgb & p_header.bmp_bitfield.green_mask) >> p_header.bmp_bitfield.green_offset; // Green 0b 0000 0011 1110 ...
						uint8_t b2 = (rgb & p_header.bmp_bitfield.blue_mask); // >> p_header.bmp_bitfield.blue_offset; // Blue  0b  ... 0001 1111

						// Next we apply some color scaling going from a variable value space to a 256 value space.
						// This may be simplified some but left as is for legibility.
						// float scaled_value = unscaled_value * byte_max_value / color_channel_maximum_value + rounding_offset;
						float f0 = b0 * 255.0f / static_cast<float>(p_header.bmp_bitfield.red_max) + 0.5f;
						float f1 = b1 * 255.0f / static_cast<float>(p_header.bmp_bitfield.green_max) + 0.5f;
						float f2 = b2 * 255.0f / static_cast<float>(p_header.bmp_bitfield.blue_max) + 0.5f;
						write_buffer[index + 0] = static_cast<uint8_t>(f0); // R
						write_buffer[index + 1] = static_cast<uint8_t>(f1); // G
						write_buffer[index + 2] = static_cast<uint8_t>(f2); // B

						if (p_header.bmp_bitfield.alpha_mask_width > 0) {
							write_buffer[index + 3] = ba * 0xFF; // Alpha value(Always true or false so no scaling)
						} else {
							write_buffer[index + 3] = 0xFF; // No Alpha channel, Show everything.
						}

						index += 4;
						line_ptr += 2;
					} break;
					case 24: {
						write_buffer[index + 2] = line_ptr[0];
						write_buffer[index + 1] = line_ptr[1];
						write_buffer[index + 0] = line_ptr[2];
						write_buffer[index + 3] = 0xff;

						index += 4;
						line_ptr += 3;
					} break;
					case 32: {
						write_buffer[index + 2] = line_ptr[0];
						write_buffer[index + 1] = line_ptr[1];
						write_buffer[index + 0] = line_ptr[2];
						write_buffer[index + 3] = line_ptr[3];

						index += 4;
						line_ptr += 4;
					} break;
				}
			}
			line -= line_width;
		}

		if (p_color_buffer == nullptr || color_table_size == 0) { // regular pixels

			p_image->set_data(width, height, false, Image::FORMAT_RGBA8, data);

		} else { // data is in indexed format, extend it

			// Palette data
			Vector<uint8_t> palette_data;
			palette_data.resize(color_table_size * 4);

			uint8_t *palette_data_w = palette_data.ptrw();
			uint8_t *pal = palette_data_w;

			const uint8_t *cb = p_color_buffer;

			for (unsigned int i = 0; i < color_table_size; ++i) {
				pal[i * 4 + 0] = cb[2];
				pal[i * 4 + 1] = cb[1];
				pal[i * 4 + 2] = cb[0];
				pal[i * 4 + 3] = 0xff;

				cb += 4;
			}
			// Extend palette to image
			Vector<uint8_t> extended_data;
			extended_data.resize(data.size() * 4);

			uint8_t *ex_w = extended_data.ptrw();
			uint8_t *dest = ex_w;

			const int num_pixels = width * height;

			for (int i = 0; i < num_pixels; i++) {
				dest[0] = pal[write_buffer[i] * 4 + 0];
				dest[1] = pal[write_buffer[i] * 4 + 1];
				dest[2] = pal[write_buffer[i] * 4 + 2];
				dest[3] = pal[write_buffer[i] * 4 + 3];

				dest += 4;
			}
			p_image->set_data(width, height, false, Image::FORMAT_RGBA8, extended_data);
		}
	}
	return err;
}

Error ImageLoaderBMP::load_image(Ref<Image> p_image, Ref<FileAccess> f, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale) {
	bmp_header_s bmp_header;
	Error err = ERR_INVALID_DATA;

	// A valid bmp file should always at least have a
	// file header and a minimal info header
	if (f->get_length() > BITMAP_FILE_HEADER_SIZE + BITMAP_INFO_HEADER_MIN_SIZE) {
		// File Header
		bmp_header.bmp_file_header.bmp_signature = f->get_16();
		if (bmp_header.bmp_file_header.bmp_signature == BITMAP_SIGNATURE) {
			bmp_header.bmp_file_header.bmp_file_size = f->get_32();
			bmp_header.bmp_file_header.bmp_file_padding = f->get_32();
			bmp_header.bmp_file_header.bmp_file_offset = f->get_32();

			// Info Header
			bmp_header.bmp_info_header.bmp_header_size = f->get_32();
			ERR_FAIL_COND_V_MSG(bmp_header.bmp_info_header.bmp_header_size < BITMAP_INFO_HEADER_MIN_SIZE, ERR_FILE_CORRUPT,
					vformat("Couldn't parse the BMP info header. The file is likely corrupt: %s", f->get_path()));

			bmp_header.bmp_info_header.bmp_width = f->get_32();
			bmp_header.bmp_info_header.bmp_height = f->get_32();

			bmp_header.bmp_info_header.bmp_planes = f->get_16();
			ERR_FAIL_COND_V_MSG(bmp_header.bmp_info_header.bmp_planes != 1, ERR_FILE_CORRUPT,
					vformat("Couldn't parse the BMP planes. The file is likely corrupt: %s", f->get_path()));

			bmp_header.bmp_info_header.bmp_bit_count = f->get_16();
			bmp_header.bmp_info_header.bmp_compression = f->get_32();
			bmp_header.bmp_info_header.bmp_size_image = f->get_32();
			bmp_header.bmp_info_header.bmp_pixels_per_meter_x = f->get_32();
			bmp_header.bmp_info_header.bmp_pixels_per_meter_y = f->get_32();
			bmp_header.bmp_info_header.bmp_colors_used = f->get_32();
			bmp_header.bmp_info_header.bmp_important_colors = f->get_32();

			switch (bmp_header.bmp_info_header.bmp_compression) {
				case BI_BITFIELDS: {
					bmp_header.bmp_bitfield.red_mask = f->get_32();
					bmp_header.bmp_bitfield.green_mask = f->get_32();
					bmp_header.bmp_bitfield.blue_mask = f->get_32();
					bmp_header.bmp_bitfield.alpha_mask = f->get_32();

					bmp_header.bmp_bitfield.red_mask_width = get_mask_width(bmp_header.bmp_bitfield.red_mask);
					bmp_header.bmp_bitfield.green_mask_width = get_mask_width(bmp_header.bmp_bitfield.green_mask);
					bmp_header.bmp_bitfield.blue_mask_width = get_mask_width(bmp_header.bmp_bitfield.blue_mask);
					bmp_header.bmp_bitfield.alpha_mask_width = get_mask_width(bmp_header.bmp_bitfield.alpha_mask);

					bmp_header.bmp_bitfield.alpha_offset = bmp_header.bmp_bitfield.red_mask_width + bmp_header.bmp_bitfield.green_mask_width + bmp_header.bmp_bitfield.blue_mask_width;
					bmp_header.bmp_bitfield.red_offset = bmp_header.bmp_bitfield.green_mask_width + bmp_header.bmp_bitfield.blue_mask_width;
					bmp_header.bmp_bitfield.green_offset = bmp_header.bmp_bitfield.blue_mask_width;

					bmp_header.bmp_bitfield.red_max = (1 << bmp_header.bmp_bitfield.red_mask_width) - 1;
					bmp_header.bmp_bitfield.green_max = (1 << bmp_header.bmp_bitfield.green_mask_width) - 1;
					bmp_header.bmp_bitfield.blue_max = (1 << bmp_header.bmp_bitfield.blue_mask_width) - 1;
				} break;
				case BI_RLE8:
				case BI_RLE4:
				case BI_CMYKRLE8:
				case BI_CMYKRLE4: {
					// Stop parsing.
					ERR_FAIL_V_MSG(ERR_UNAVAILABLE,
							vformat("RLE compressed BMP files are not yet supported: %s", f->get_path()));
				} break;
			}
			// Don't rely on sizeof(bmp_file_header) as structure padding
			// adds 2 bytes offset leading to misaligned color table reading
			uint32_t ct_offset = BITMAP_FILE_HEADER_SIZE + bmp_header.bmp_info_header.bmp_header_size;
			f->seek(ct_offset);

			uint32_t color_table_size = 0;

			// bmp_colors_used may report 0 despite having a color table
			// for 4 and 1 bit images, so don't rely on this information
			if (bmp_header.bmp_info_header.bmp_bit_count <= 8) {
				// Support 256 colors max
				color_table_size = 1 << bmp_header.bmp_info_header.bmp_bit_count;
				ERR_FAIL_COND_V_MSG(color_table_size == 0, ERR_BUG,
						vformat("Couldn't parse the BMP color table: %s", f->get_path()));
			}

			Vector<uint8_t> bmp_color_table;
			// Color table is usually 4 bytes per color -> [B][G][R][0]
			bmp_color_table.resize(color_table_size * 4);
			uint8_t *bmp_color_table_w = bmp_color_table.ptrw();
			f->get_buffer(bmp_color_table_w, color_table_size * 4);

			f->seek(bmp_header.bmp_file_header.bmp_file_offset);

			uint32_t bmp_buffer_size = (bmp_header.bmp_file_header.bmp_file_size - bmp_header.bmp_file_header.bmp_file_offset);

			Vector<uint8_t> bmp_buffer;
			err = bmp_buffer.resize(bmp_buffer_size);
			if (err == OK) {
				uint8_t *bmp_buffer_w = bmp_buffer.ptrw();
				f->get_buffer(bmp_buffer_w, bmp_buffer_size);

				const uint8_t *bmp_buffer_r = bmp_buffer.ptr();
				const uint8_t *bmp_color_table_r = bmp_color_table.ptr();
				err = convert_to_image(p_image, bmp_buffer_r,
						bmp_color_table_r, color_table_size, bmp_header);
			}
		}
	}
	return err;
}

void ImageLoaderBMP::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("bmp");
}

static Ref<Image> _bmp_mem_loader_func(const uint8_t *p_bmp, int p_size) {
	Ref<FileAccessMemory> memfile;
	memfile.instantiate();
	Error open_memfile_error = memfile->open_custom(p_bmp, p_size);
	ERR_FAIL_COND_V_MSG(open_memfile_error, Ref<Image>(), "Could not create memfile for BMP image buffer.");

	Ref<Image> img;
	img.instantiate();
	Error load_error = ImageLoaderBMP().load_image(img, memfile, false, 1.0f);
	ERR_FAIL_COND_V_MSG(load_error, Ref<Image>(), "Failed to load BMP image.");
	return img;
}

ImageLoaderBMP::ImageLoaderBMP() {
	Image::_bmp_mem_loader_func = _bmp_mem_loader_func;
}
