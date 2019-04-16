/*************************************************************************/
/*  image_loader_bmp.cpp                                                 */
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

#include "image_loader_bmp.h"

Error ImageLoaderBMP::convert_to_image(Ref<Image> p_image,
		const uint8_t *p_buffer,
		const uint8_t *p_color_buffer,
		const bmp_header_s &p_header) {

	Error err = OK;

	bool has_data = p_buffer != NULL;
	bool has_palette = p_color_buffer != NULL;

	if (!has_data)
		err = FAILED;

	if (err == OK) {
		size_t index = 0;
		size_t width = (size_t)p_header.bmp_info_header.bmp_width;
		size_t height = (size_t)p_header.bmp_info_header.bmp_height;
		size_t bits_per_pixel = (size_t)p_header.bmp_info_header.bmp_bit_count;

		if (p_header.bmp_info_header.bmp_compression != 0) {
			err = FAILED;
		}
		// Check whether we can load it
		switch (bits_per_pixel) {
			case 1:
				// Requires bit unpacking
				ERR_FAIL_COND_V(width % 8 != 0, ERR_UNAVAILABLE);
				ERR_FAIL_COND_V(height % 8 != 0, ERR_UNAVAILABLE);
			case 4:
				ERR_FAIL_COND_V(width % 2 != 0, ERR_UNAVAILABLE);
				ERR_FAIL_COND_V(height % 2 != 0, ERR_UNAVAILABLE);
			case 8:
			case 24:
			case 32:
				break;
			default: {
				ERR_FAIL_V(ERR_UNAVAILABLE);
			}
		}
		if (err == OK) {
			// Palette data
			PoolVector<uint8_t> palette_data;

			if (has_palette) {

				uint32_t color_table_size = 0;
				if (p_header.bmp_info_header.bmp_bit_count == 1)
					color_table_size = 2;
				else if (p_header.bmp_info_header.bmp_bit_count == 4)
					color_table_size = 16;
				else if (p_header.bmp_info_header.bmp_bit_count == 8)
					color_table_size = 256;

				uint32_t palette_size = p_header.bmp_info_header.bmp_colors_used;
				if (palette_size == 0)
					palette_size = color_table_size;

				palette_data.resize(palette_size * 4);

				PoolVector<uint8_t>::Write palette_data_w = palette_data.write();
				uint8_t *pal = palette_data_w.ptr();

				const uint8_t *cb = p_color_buffer;

				for (unsigned int i = 0; i < palette_size; ++i) {
					uint32_t color = *((uint32_t *)cb);

					pal[i * 4 + 0] = (color >> 16) & 0xff;
					pal[i * 4 + 1] = (color >> 8) & 0xff;
					pal[i * 4 + 2] = (color)&0xff;
					pal[i * 4 + 3] = 0xff;

					cb += 4;
				}
			}
			// Pixel data (or index data)
			PoolVector<uint8_t> image_data;
			int image_data_len = 0;

			switch (bits_per_pixel) {
				case 1:
				case 4:
				case 8: { // indexed
					image_data_len = width * height;
				} break;
				case 24:
				case 32: { // color
					image_data_len = width * height * 4;
				} break;
			}
			ERR_FAIL_COND_V(image_data_len == 0, ERR_BUG);
			err = image_data.resize(image_data_len);

			PoolVector<uint8_t>::Write image_data_w = image_data.write();
			uint8_t *write_buffer = image_data_w.ptr();

			const uint32_t width_bytes = width * bits_per_pixel / 8;
			const uint32_t line_width = (width_bytes + 3) & ~3;
			const uint32_t w = bits_per_pixel >= 24 ? width : width_bytes;

			const uint8_t *line = p_buffer + (line_width * (height - 1));

			for (unsigned int i = 0; i < height; i++) {
				const uint8_t *line_ptr = line;

				for (unsigned int j = 0; j < w; j++) {
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
						case 24: {
							uint32_t color = *((uint32_t *)line_ptr);

							write_buffer[index + 2] = color & 0xff;
							write_buffer[index + 1] = (color >> 8) & 0xff;
							write_buffer[index + 0] = (color >> 16) & 0xff;
							write_buffer[index + 3] = 0xff;

							index += 4;
							line_ptr += 3;
						} break;
						case 32: {
							uint32_t color = *((uint32_t *)line_ptr);

							write_buffer[index + 2] = color & 0xff;
							write_buffer[index + 1] = (color >> 8) & 0xff;
							write_buffer[index + 0] = (color >> 16) & 0xff;
							write_buffer[index + 3] = color >> 24;

							index += 4;
							line_ptr += 4;
						} break;
					}
				}
				line -= line_width;
			}

			if (bits_per_pixel > 8) {
				p_image->create(width, height, 0, Image::FORMAT_RGBA8, image_data);
			} else {
				p_image->create(width, height, 0, Image::FORMAT_RGBA8);
				p_image->create_palette(palette_data, image_data);
				p_image->apply_palette();
			}
		}
	}
	return err;
}

Error ImageLoaderBMP::load_image(Ref<Image> p_image, FileAccess *f,
		bool p_force_linear, float p_scale) {

	bmp_header_s bmp_header;
	Error err = ERR_INVALID_DATA;

	if (f->get_len() > BITMAP_FILE_HEADER_SIZE + BITMAP_INFO_HEADER_MIN_SIZE) {
		// File Header
		bmp_header.bmp_file_header.bmp_signature = f->get_16();
		if (bmp_header.bmp_file_header.bmp_signature == BITMAP_SIGNATURE) {
			bmp_header.bmp_file_header.bmp_file_size = f->get_32();
			bmp_header.bmp_file_header.bmp_file_padding = f->get_32();
			bmp_header.bmp_file_header.bmp_file_offset = f->get_32();

			// Info Header
			bmp_header.bmp_info_header.bmp_header_size = f->get_32();
			ERR_FAIL_COND_V(bmp_header.bmp_info_header.bmp_header_size < BITMAP_INFO_HEADER_MIN_SIZE, ERR_FILE_CORRUPT);

			bmp_header.bmp_info_header.bmp_width = f->get_32();
			bmp_header.bmp_info_header.bmp_height = f->get_32();

			bmp_header.bmp_info_header.bmp_planes = f->get_16();
			ERR_FAIL_COND_V(bmp_header.bmp_info_header.bmp_planes != 1, ERR_FILE_CORRUPT);

			bmp_header.bmp_info_header.bmp_bit_count = f->get_16();
			bmp_header.bmp_info_header.bmp_compression = f->get_32();
			bmp_header.bmp_info_header.bmp_size_image = f->get_32();
			bmp_header.bmp_info_header.bmp_pixels_per_meter_x = f->get_32();
			bmp_header.bmp_info_header.bmp_pixels_per_meter_y = f->get_32();
			bmp_header.bmp_info_header.bmp_colors_used = f->get_32();
			bmp_header.bmp_info_header.bmp_important_colors = f->get_32();

			bmp_header.bmp_info_header.bmp_red_mask = f->get_32();
			bmp_header.bmp_info_header.bmp_green_mask = f->get_32();
			bmp_header.bmp_info_header.bmp_blue_mask = f->get_32();
			bmp_header.bmp_info_header.bmp_alpha_mask = f->get_32();
			bmp_header.bmp_info_header.bmp_cs_type = f->get_32();
			for (int i = 0; i < 9; i++)
				bmp_header.bmp_info_header.bmp_endpoints[i] = f->get_32();

			bmp_header.bmp_info_header.bmp_gamma_red = f->get_32();
			bmp_header.bmp_info_header.bmp_gamma_green = f->get_32();
			bmp_header.bmp_info_header.bmp_gamma_blue = f->get_32();

			uint32_t ct_offset = BITMAP_FILE_HEADER_SIZE +
								 bmp_header.bmp_info_header.bmp_header_size;
			f->seek(ct_offset);

			uint32_t color_table_size = 0;
			if (bmp_header.bmp_info_header.bmp_bit_count == 1)
				color_table_size = 2;
			else if (bmp_header.bmp_info_header.bmp_bit_count == 4)
				color_table_size = 16;
			else if (bmp_header.bmp_info_header.bmp_bit_count == 8)
				color_table_size = 256;

			PoolVector<uint8_t> bmp_color_table;
			if (color_table_size > 0) {
				err = bmp_color_table.resize(color_table_size * 4);
				PoolVector<uint8_t>::Write bmp_color_table_w = bmp_color_table.write();
				f->get_buffer(bmp_color_table_w.ptr(), color_table_size * 4);
			}

			f->seek(bmp_header.bmp_file_header.bmp_file_offset);

			uint32_t bmp_buffer_size = (bmp_header.bmp_file_header.bmp_file_size -
										bmp_header.bmp_file_header.bmp_file_offset);

			PoolVector<uint8_t> bmp_buffer;
			err = bmp_buffer.resize(bmp_buffer_size);
			if (err == OK) {
				PoolVector<uint8_t>::Write bmp_buffer_w = bmp_buffer.write();
				f->get_buffer(bmp_buffer_w.ptr(), bmp_buffer_size);

				PoolVector<uint8_t>::Read bmp_buffer_r = bmp_buffer.read();
				PoolVector<uint8_t>::Read bmp_color_table_r = bmp_color_table.read();
				err = convert_to_image(p_image, bmp_buffer_r.ptr(),
						bmp_color_table_r.ptr(), bmp_header);
			}
			f->close();
		}
	}
	return err;
}

void ImageLoaderBMP::get_recognized_extensions(
		List<String> *p_extensions) const {

	p_extensions->push_back("bmp");
}

ImageLoaderBMP::ImageLoaderBMP() {}
