/**************************************************************************/
/*  buffer_decoder.cpp                                                    */
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

#include "buffer_decoder.h"

#include "servers/camera/camera_feed.h"

#include <linux/videodev2.h>

BufferDecoder::BufferDecoder(CameraFeed *p_camera_feed) {
	camera_feed = p_camera_feed;
	width = camera_feed->get_format().width;
	height = camera_feed->get_format().height;
	image.instantiate();
}

AbstractYuyvBufferDecoder::AbstractYuyvBufferDecoder(CameraFeed *p_camera_feed) :
		BufferDecoder(p_camera_feed) {
	switch (camera_feed->get_format().pixel_format) {
		case V4L2_PIX_FMT_YYUV:
			component_indexes = new int[4]{ 0, 1, 2, 3 };
			break;
		case V4L2_PIX_FMT_YVYU:
			component_indexes = new int[4]{ 0, 2, 3, 1 };
			break;
		case V4L2_PIX_FMT_UYVY:
			component_indexes = new int[4]{ 1, 3, 0, 2 };
			break;
		case V4L2_PIX_FMT_VYUY:
			component_indexes = new int[4]{ 1, 3, 2, 0 };
			break;
		default:
			component_indexes = new int[4]{ 0, 2, 1, 3 };
	}
}

AbstractYuyvBufferDecoder::~AbstractYuyvBufferDecoder() {
	delete[] component_indexes;
}

SeparateYuyvBufferDecoder::SeparateYuyvBufferDecoder(CameraFeed *p_camera_feed) :
		AbstractYuyvBufferDecoder(p_camera_feed) {
	y_image_data.resize(width * height);
	cbcr_image_data.resize(width * height);
	y_image.instantiate();
	cbcr_image.instantiate();
}

void SeparateYuyvBufferDecoder::decode(StreamingBuffer p_buffer) {
	uint8_t *y_dst = (uint8_t *)y_image_data.ptrw();
	uint8_t *uv_dst = (uint8_t *)cbcr_image_data.ptrw();
	uint8_t *src = (uint8_t *)p_buffer.start;
	// YUYV-family is packed 4:2:2 (2 bytes per pixel). Honor driver-reported
	// pitch in case bytesperline is padded for row alignment.
	const int row_stride = p_buffer.pitch > 0 ? p_buffer.pitch : width * 2;

	for (int y = 0; y < height; y++) {
		uint8_t *row = src + (size_t)y * row_stride;
		uint8_t *y0_src = row + component_indexes[0];
		uint8_t *y1_src = row + component_indexes[1];
		uint8_t *u_src = row + component_indexes[2];
		uint8_t *v_src = row + component_indexes[3];

		for (int x = 0; x < width; x += 2) {
			*y_dst++ = *y0_src;
			*y_dst++ = *y1_src;
			*uv_dst++ = *u_src;
			*uv_dst++ = *v_src;

			y0_src += 4;
			y1_src += 4;
			u_src += 4;
			v_src += 4;
		}
	}

	if (y_image.is_valid()) {
		y_image->set_data(width, height, false, Image::FORMAT_L8, y_image_data);
	} else {
		y_image.instantiate(width, height, false, Image::FORMAT_RGB8, y_image_data);
	}
	if (cbcr_image.is_valid()) {
		cbcr_image->set_data(width, height, false, Image::FORMAT_L8, cbcr_image_data);
	} else {
		cbcr_image.instantiate(width, height, false, Image::FORMAT_RGB8, cbcr_image_data);
	}

	camera_feed->set_ycbcr_images(y_image, cbcr_image);
}

YuyvToGrayscaleBufferDecoder::YuyvToGrayscaleBufferDecoder(CameraFeed *p_camera_feed) :
		AbstractYuyvBufferDecoder(p_camera_feed) {
	image_data.resize(width * height);
}

void YuyvToGrayscaleBufferDecoder::decode(StreamingBuffer p_buffer) {
	uint8_t *dst = (uint8_t *)image_data.ptrw();
	uint8_t *src = (uint8_t *)p_buffer.start;
	const int row_stride = p_buffer.pitch > 0 ? p_buffer.pitch : width * 2;

	for (int y = 0; y < height; y++) {
		uint8_t *row = src + (size_t)y * row_stride;
		uint8_t *y0_src = row + component_indexes[0];
		uint8_t *y1_src = row + component_indexes[1];

		for (int x = 0; x < width; x += 2) {
			*dst++ = *y0_src;
			*dst++ = *y1_src;

			y0_src += 4;
			y1_src += 4;
		}
	}

	if (image.is_valid()) {
		image->set_data(width, height, false, Image::FORMAT_L8, image_data);
	} else {
		image.instantiate(width, height, false, Image::FORMAT_RGB8, image_data);
	}

	camera_feed->set_rgb_image(image);
}

YuyvToRgbBufferDecoder::YuyvToRgbBufferDecoder(CameraFeed *p_camera_feed) :
		AbstractYuyvBufferDecoder(p_camera_feed) {
	image_data.resize(width * height * 3);
}

void YuyvToRgbBufferDecoder::decode(StreamingBuffer p_buffer) {
	uint8_t *src = (uint8_t *)p_buffer.start;
	uint8_t *dst = (uint8_t *)image_data.ptrw();
	const int row_stride = p_buffer.pitch > 0 ? p_buffer.pitch : width * 2;

	for (int y = 0; y < height; y++) {
		uint8_t *row = src + (size_t)y * row_stride;
		uint8_t *y0_src = row + component_indexes[0];
		uint8_t *y1_src = row + component_indexes[1];
		uint8_t *u_src = row + component_indexes[2];
		uint8_t *v_src = row + component_indexes[3];

		for (int x = 0; x < width; x += 2) {
			int u = *u_src;
			int v = *v_src;
			int u1 = (((u - 128) << 7) + (u - 128)) >> 6;
			int rg = (((u - 128) << 1) + (u - 128) + ((v - 128) << 2) + ((v - 128) << 1)) >> 3;
			int v1 = (((v - 128) << 1) + (v - 128)) >> 1;

			*dst++ = CLAMP(*y0_src + v1, 0, 255);
			*dst++ = CLAMP(*y0_src - rg, 0, 255);
			*dst++ = CLAMP(*y0_src + u1, 0, 255);

			*dst++ = CLAMP(*y1_src + v1, 0, 255);
			*dst++ = CLAMP(*y1_src - rg, 0, 255);
			*dst++ = CLAMP(*y1_src + u1, 0, 255);

			y0_src += 4;
			y1_src += 4;
			u_src += 4;
			v_src += 4;
		}
	}

	if (image.is_valid()) {
		image->set_data(width, height, false, Image::FORMAT_RGB8, image_data);
	} else {
		image.instantiate(width, height, false, Image::FORMAT_RGB8, image_data);
	}

	camera_feed->set_rgb_image(image);
}

CopyBufferDecoder::CopyBufferDecoder(CameraFeed *p_camera_feed, bool p_rgba) :
		BufferDecoder(p_camera_feed) {
	rgba = p_rgba;
	image_data.resize(width * height * (rgba ? 4 : 2));
}

void CopyBufferDecoder::decode(StreamingBuffer p_buffer) {
	uint8_t *dst = (uint8_t *)image_data.ptrw();
	uint8_t *src = (uint8_t *)p_buffer.start;
	const int bpp = rgba ? 4 : 2;
	const int row_bytes = width * bpp;
	const int row_stride = p_buffer.pitch > 0 ? p_buffer.pitch : row_bytes;

	if (row_stride == row_bytes) {
		// No padding: single contiguous copy.
		memcpy(dst, src, (size_t)row_bytes * height);
	} else {
		// Strip row padding.
		for (int y = 0; y < height; y++) {
			memcpy(dst + (size_t)y * row_bytes, src + (size_t)y * row_stride, row_bytes);
		}
	}

	if (image.is_valid()) {
		image->set_data(width, height, false, rgba ? Image::FORMAT_RGBA8 : Image::FORMAT_LA8, image_data);
	} else {
		image.instantiate(width, height, false, rgba ? Image::FORMAT_RGBA8 : Image::FORMAT_LA8, image_data);
	}

	camera_feed->set_rgb_image(image);
}

Yuv420BufferDecoder::Yuv420BufferDecoder(CameraFeed *p_camera_feed, bool p_v_plane_first) :
		BufferDecoder(p_camera_feed) {
	v_plane_first = p_v_plane_first;
	const size_t chroma_size = (size_t)(width / 2) * (height / 2);
	y_buffer.resize((size_t)width * height);
	cbcr_buffer.resize(chroma_size * 2);
	y_image.instantiate();
	cbcr_image.instantiate();
}

void Yuv420BufferDecoder::decode(StreamingBuffer p_buffer) {
	const int chroma_width = width / 2;
	const int chroma_height = height / 2;

	// Y plane uses the driver-reported pitch (bytesperline); chroma planes
	// follow the V4L2 convention of half-stride. Fall back to tightly packed
	// when no pitch is provided (legacy producers, contiguous buffers).
	const int y_stride = p_buffer.pitch > 0 ? p_buffer.pitch : width;
	const int chroma_stride = y_stride / 2;

	const size_t y_plane_size = (size_t)y_stride * height;
	const size_t chroma_plane_size = (size_t)chroma_stride * chroma_height;
	const size_t expected_size = y_plane_size + chroma_plane_size * 2;

	if (p_buffer.length < expected_size) {
		return;
	}

	const uint8_t *src = (const uint8_t *)p_buffer.start;
	const uint8_t *plane2 = src + y_plane_size;
	const uint8_t *plane3 = plane2 + chroma_plane_size;
	const uint8_t *cb_plane = v_plane_first ? plane3 : plane2;
	const uint8_t *cr_plane = v_plane_first ? plane2 : plane3;

	// Copy Y plane row by row to strip stride padding.
	uint8_t *y_dst = y_buffer.ptrw();
	for (int y = 0; y < height; y++) {
		memcpy(y_dst + (size_t)y * width, src + (size_t)y * y_stride, width);
	}

	// Interleave Cb and Cr into RG8 (R=Cb, G=Cr) row by row.
	uint8_t *cbcr_dst = cbcr_buffer.ptrw();
	for (int y = 0; y < chroma_height; y++) {
		const uint8_t *cb_row = cb_plane + (size_t)y * chroma_stride;
		const uint8_t *cr_row = cr_plane + (size_t)y * chroma_stride;
		for (int x = 0; x < chroma_width; x++) {
			*cbcr_dst++ = cb_row[x];
			*cbcr_dst++ = cr_row[x];
		}
	}

	y_image->set_data(width, height, false, Image::FORMAT_L8, y_buffer);
	cbcr_image->set_data(chroma_width, chroma_height, false, Image::FORMAT_RG8, cbcr_buffer);
	camera_feed->set_ycbcr_images(y_image, cbcr_image);
}

JpegBufferDecoder::JpegBufferDecoder(CameraFeed *p_camera_feed) :
		BufferDecoder(p_camera_feed) {
}

void JpegBufferDecoder::decode(StreamingBuffer p_buffer) {
	image_data.resize(p_buffer.length);
	uint8_t *dst = (uint8_t *)image_data.ptrw();
	memcpy(dst, p_buffer.start, p_buffer.length);
	if (image->load_jpg_from_buffer(image_data) == OK) {
		camera_feed->set_rgb_image(image);
	}
}
