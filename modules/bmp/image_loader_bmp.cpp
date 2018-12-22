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

	if (p_buffer == NULL)
		err = FAILED;

	if (err == OK) {
		size_t index = 0;
		size_t width = (size_t)p_header.bmp_info_header.bmp_width;
		size_t height = (size_t)p_header.bmp_info_header.bmp_height;
		size_t bits_per_pixel = (size_t)p_header.bmp_info_header.bmp_bit_count;

		if (p_header.bmp_info_header.bmp_compression != BI_RGB) {
			err = FAILED;
		}

		if (!(bits_per_pixel == 8 || bits_per_pixel == 24 || bits_per_pixel == 32)) {
			err = FAILED;
		}

		if (err == OK) {

			uint32_t line_width = ((p_header.bmp_info_header.bmp_width *
										   p_header.bmp_info_header.bmp_bit_count / 8) +
										  3) &
								  ~3;

			PoolVector<uint8_t> image_data;
			err = image_data.resize(width * height * 4);

			PoolVector<uint8_t>::Write image_data_w = image_data.write();
			uint8_t *write_buffer = image_data_w.ptr();

			const uint32_t color_index_max = p_header.bmp_info_header.bmp_colors_used - 1;
			const uint8_t *line = p_buffer + (line_width * (height - 1));
			for (unsigned int i = 0; i < height; i++) {
				const uint8_t *line_ptr = line;
				for (unsigned int j = 0; j < width; j++) {
					switch (bits_per_pixel) {
						case 8: {
							uint8_t color_index = CLAMP(*line_ptr, 0, color_index_max);
							uint32_t color = 0x000000;

							if (p_color_buffer != NULL)
								color = ((uint32_t *)p_color_buffer)[color_index];

							write_buffer[index + 2] = color & 0xff;
							write_buffer[index + 1] = (color >> 8) & 0xff;
							write_buffer[index + 0] = (color >> 16) & 0xff;
							write_buffer[index + 3] = 0xff;
							index += 4;
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
			p_image->create(width, height, 0, Image::FORMAT_RGBA8, image_data);
		}
	}
	return err;
}

Error ImageLoaderBMP::load_image(Ref<Image> p_image, FileAccess *f,
		bool p_force_linear, float p_scale) {

	bmp_header_s bmp_header;
	Error err = ERR_INVALID_DATA;

	static const size_t FILE_HEADER_SIZE = 14;
	static const size_t INFO_HEADER_SIZE = 40;

	// A valid bmp file should always at least have a
	// file header and a minimal info header
	if (f->get_len() > FILE_HEADER_SIZE + INFO_HEADER_SIZE) {
		// File Header
		bmp_header.bmp_file_header.bmp_signature = f->get_16();
		if (bmp_header.bmp_file_header.bmp_signature == BITMAP_SIGNATURE) {
			bmp_header.bmp_file_header.bmp_file_size = f->get_32();
			bmp_header.bmp_file_header.bmp_file_padding = f->get_32();
			bmp_header.bmp_file_header.bmp_file_offset = f->get_32();

			// Info Header
			bmp_header.bmp_info_header.bmp_header_size = f->get_32();
			bmp_header.bmp_info_header.bmp_width = f->get_32();
			bmp_header.bmp_info_header.bmp_height = f->get_32();
			bmp_header.bmp_info_header.bmp_planes = f->get_16();
			bmp_header.bmp_info_header.bmp_bit_count = f->get_16();
			bmp_header.bmp_info_header.bmp_compression = f->get_32();
			bmp_header.bmp_info_header.bmp_size_image = f->get_32();
			bmp_header.bmp_info_header.bmp_pixels_per_meter_x = f->get_32();
			bmp_header.bmp_info_header.bmp_pixels_per_meter_y = f->get_32();
			bmp_header.bmp_info_header.bmp_colors_used = f->get_32();
			bmp_header.bmp_info_header.bmp_important_colors = f->get_32();

			// Compressed bitmaps not supported, stop parsing
			if (bmp_header.bmp_info_header.bmp_compression != BI_RGB) {
				ERR_EXPLAIN("Unsupported bmp file: " + f->get_path());
				f->close();
				ERR_FAIL_V(err)
			}

			f->seek(FILE_HEADER_SIZE +
					bmp_header.bmp_info_header.bmp_header_size);

			if (bmp_header.bmp_info_header.bmp_bit_count < 16 && bmp_header.bmp_info_header.bmp_colors_used == 0)
				bmp_header.bmp_info_header.bmp_colors_used = 1 << bmp_header.bmp_info_header.bmp_bit_count;

			// Color table is usually 4 bytes per color -> [B][G][R][0]
			uint32_t color_table_size = bmp_header.bmp_info_header.bmp_colors_used * 4;

			PoolVector<uint8_t> bmp_color_table;
			if (color_table_size > 0) {
				err = bmp_color_table.resize(color_table_size);
				PoolVector<uint8_t>::Write bmp_color_table_w = bmp_color_table.write();
				f->get_buffer(bmp_color_table_w.ptr(), color_table_size);
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
