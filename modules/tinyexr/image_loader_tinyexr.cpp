/*************************************************************************/
/*  image_loader_tinyexr.cpp                                             */
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

#include "image_loader_tinyexr.h"

#include "core/os/os.h"
#include "core/print_string.h"

#include "thirdparty/tinyexr/tinyexr.h"

Error ImageLoaderTinyEXR::load_image(Ref<Image> p_image, FileAccess *f, bool p_force_linear, float p_scale) {

	PoolVector<uint8_t> src_image;
	int src_image_len = f->get_len();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	src_image.resize(src_image_len);

	PoolVector<uint8_t>::Write w = src_image.write();

	f->get_buffer(&w[0], src_image_len);

	f->close();

	// Re-implementation of tinyexr's LoadEXRFromMemory using Godot types to store the Image data
	// and Godot's error codes.
	// When debugging after updating the thirdparty library, check that we're still in sync with
	// their API usage in LoadEXRFromMemory.

	EXRVersion exr_version;
	EXRImage exr_image;
	EXRHeader exr_header;
	const char *err = NULL;

	InitEXRHeader(&exr_header);

	int ret = ParseEXRVersionFromMemory(&exr_version, w.ptr(), src_image_len);
	if (ret != TINYEXR_SUCCESS) {

		return ERR_FILE_CORRUPT;
	}

	ret = ParseEXRHeaderFromMemory(&exr_header, &exr_version, w.ptr(), src_image_len, &err);
	if (ret != TINYEXR_SUCCESS) {
		if (err) {
			ERR_PRINTS(String(err));
		}
		return ERR_FILE_CORRUPT;
	}

	// Read HALF channel as FLOAT. (GH-13490)
	for (int i = 0; i < exr_header.num_channels; i++) {
		if (exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
			exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
		}
	}

	InitEXRImage(&exr_image);
	ret = LoadEXRImageFromMemory(&exr_image, &exr_header, w.ptr(), src_image_len, &err);
	if (ret != TINYEXR_SUCCESS) {
		if (err) {
			ERR_PRINTS(String(err));
		}
		return ERR_FILE_CORRUPT;
	}

	// RGBA
	int idxR = -1;
	int idxG = -1;
	int idxB = -1;
	int idxA = -1;
	for (int c = 0; c < exr_header.num_channels; c++) {
		if (strcmp(exr_header.channels[c].name, "R") == 0) {
			idxR = c;
		} else if (strcmp(exr_header.channels[c].name, "G") == 0) {
			idxG = c;
		} else if (strcmp(exr_header.channels[c].name, "B") == 0) {
			idxB = c;
		} else if (strcmp(exr_header.channels[c].name, "A") == 0) {
			idxA = c;
		}
	}

	if (exr_header.num_channels == 1) {
		// Grayscale channel only.
		idxR = 0;
		idxG = 0;
		idxB = 0;
		idxA = 0;
	} else {
		// Assume RGB(A)
		if (idxR == -1) {
			ERR_PRINT("TinyEXR: R channel not found.");
			// @todo { free exr_image }
			return ERR_FILE_CORRUPT;
		}

		if (idxG == -1) {
			ERR_PRINT("TinyEXR: G channel not found.")
			// @todo { free exr_image }
			return ERR_FILE_CORRUPT;
		}

		if (idxB == -1) {
			ERR_PRINT("TinyEXR: B channel not found.")
			// @todo { free exr_image }
			return ERR_FILE_CORRUPT;
		}
	}

	// EXR image data loaded, now parse it into Godot-friendly image data

	PoolVector<uint8_t> imgdata;
	Image::Format format;
	int output_channels = 0;

	if (idxA != -1) {

		imgdata.resize(exr_image.width * exr_image.height * 8); //RGBA16
		format = Image::FORMAT_RGBAH;
		output_channels = 4;
	} else {

		imgdata.resize(exr_image.width * exr_image.height * 6); //RGB16
		format = Image::FORMAT_RGBH;
		output_channels = 3;
	}

	EXRTile single_image_tile;
	int num_tiles;
	int tile_width = 0;
	int tile_height = 0;

	const EXRTile *exr_tiles;

	if (!exr_header.tiled) {
		single_image_tile.images = exr_image.images;
		single_image_tile.width = exr_image.width;
		single_image_tile.height = exr_image.height;
		single_image_tile.level_x = exr_image.width;
		single_image_tile.level_y = exr_image.height;
		single_image_tile.offset_x = 0;
		single_image_tile.offset_y = 0;

		exr_tiles = &single_image_tile;
		num_tiles = 1;
		tile_width = exr_image.width;
		tile_height = exr_image.height;
	} else {
		tile_width = exr_header.tile_size_x;
		tile_height = exr_header.tile_size_y;
		num_tiles = exr_image.num_tiles;
		exr_tiles = exr_image.tiles;
	}

	{
		PoolVector<uint8_t>::Write wd = imgdata.write();
		uint16_t *iw = (uint16_t *)wd.ptr();

		// Assume `out_rgba` have enough memory allocated.
		for (int tile_index = 0; tile_index < num_tiles; tile_index++) {

			const EXRTile &tile = exr_tiles[tile_index];

			int tw = tile.width;
			int th = tile.height;

			const float *r_channel_start = reinterpret_cast<const float *>(tile.images[idxR]);
			const float *g_channel_start = reinterpret_cast<const float *>(tile.images[idxG]);
			const float *b_channel_start = reinterpret_cast<const float *>(tile.images[idxB]);
			const float *a_channel_start = NULL;

			if (idxA != -1) {
				a_channel_start = reinterpret_cast<const float *>(tile.images[idxA]);
			}

			uint16_t *first_row_w = iw + (tile.offset_y * tile_height * exr_image.width + tile.offset_x * tile_width) * output_channels;

			for (int y = 0; y < th; y++) {
				const float *r_channel = r_channel_start + y * tile_width;
				const float *g_channel = g_channel_start + y * tile_width;
				const float *b_channel = b_channel_start + y * tile_width;
				const float *a_channel = NULL;

				if (a_channel_start) {
					a_channel = a_channel_start + y * tile_width;
				}

				uint16_t *row_w = first_row_w + (y * exr_image.width * output_channels);

				for (int x = 0; x < tw; x++) {

					Color color(*r_channel++, *g_channel++, *b_channel++);

					if (p_force_linear)
						color = color.to_linear();

					*row_w++ = Math::make_half_float(color.r);
					*row_w++ = Math::make_half_float(color.g);
					*row_w++ = Math::make_half_float(color.b);

					if (idxA != -1) {
						*row_w++ = Math::make_half_float(*a_channel++);
					}
				}
			}
		}
	}

	p_image->create(exr_image.width, exr_image.height, false, format, imgdata);

	w = PoolVector<uint8_t>::Write();

	FreeEXRHeader(&exr_header);
	FreeEXRImage(&exr_image);

	return OK;
}

void ImageLoaderTinyEXR::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("exr");
}

ImageLoaderTinyEXR::ImageLoaderTinyEXR() {
}
