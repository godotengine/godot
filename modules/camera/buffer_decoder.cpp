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

#ifdef LINUXBSD_ENABLED
#include <linux/videodev2.h>
#endif

BufferDecoder::BufferDecoder(CameraFeed *p_camera_feed) {
	camera_feed = p_camera_feed;
	width = camera_feed->get_format().width;
	height = camera_feed->get_format().height;
}

AbstractYuyvBufferDecoder::AbstractYuyvBufferDecoder(CameraFeed *p_camera_feed) :
		BufferDecoder(p_camera_feed) {
#ifdef LINUXBSD_ENABLED
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
#else
	component_indexes = new int[4]{ 0, 2, 1, 3 };
#endif
}

AbstractYuyvBufferDecoder::~AbstractYuyvBufferDecoder() {
	delete[] component_indexes;
}

SeparateYuyvBufferDecoder::SeparateYuyvBufferDecoder(CameraFeed *p_camera_feed) :
		AbstractYuyvBufferDecoder(p_camera_feed) {
	y_image_data.resize(width * height);
	cbcr_image_data.resize(width * height);
}

void SeparateYuyvBufferDecoder::decode(StreamingBuffer p_buffer) {
	uint8_t *y_dst = (uint8_t *)y_image_data.ptrw();
	uint8_t *uv_dst = (uint8_t *)cbcr_image_data.ptrw();
	uint8_t *src = (uint8_t *)p_buffer.start;
	uint8_t *y0_src = src + component_indexes[0];
	uint8_t *y1_src = src + component_indexes[1];
	uint8_t *u_src = src + component_indexes[2];
	uint8_t *v_src = src + component_indexes[3];

	for (int i = 0; i < width * height; i += 2) {
		*y_dst++ = *y0_src;
		*y_dst++ = *y1_src;
		*uv_dst++ = *u_src;
		*uv_dst++ = *v_src;

		y0_src += 4;
		y1_src += 4;
		u_src += 4;
		v_src += 4;
	}

	// Defer to main thread to avoid race conditions with RenderingServer.
	y_image.instantiate();
	y_image->set_data(width, height, false, Image::FORMAT_L8, y_image_data);

	cbcr_image.instantiate();
	cbcr_image->set_data(width / 2, height, false, Image::FORMAT_RG8, cbcr_image_data);

	camera_feed->call_deferred("set_ycbcr_images", y_image, cbcr_image);
}

YuyvToGrayscaleBufferDecoder::YuyvToGrayscaleBufferDecoder(CameraFeed *p_camera_feed) :
		AbstractYuyvBufferDecoder(p_camera_feed) {
	image_data.resize(width * height);
}

void YuyvToGrayscaleBufferDecoder::decode(StreamingBuffer p_buffer) {
	uint8_t *dst = (uint8_t *)image_data.ptrw();
	uint8_t *src = (uint8_t *)p_buffer.start;
	uint8_t *y0_src = src + component_indexes[0];
	uint8_t *y1_src = src + component_indexes[1];

	for (int i = 0; i < width * height; i += 2) {
		*dst++ = *y0_src;
		*dst++ = *y1_src;

		y0_src += 4;
		y1_src += 4;
	}

	// Defer to main thread to avoid race conditions with RenderingServer.
	image.instantiate();
	image->set_data(width, height, false, Image::FORMAT_L8, image_data);

	camera_feed->call_deferred("set_rgb_image", image);
}

YuyvToRgbBufferDecoder::YuyvToRgbBufferDecoder(CameraFeed *p_camera_feed) :
		AbstractYuyvBufferDecoder(p_camera_feed) {
	image_data.resize(width * height * 3);
}

void YuyvToRgbBufferDecoder::decode(StreamingBuffer p_buffer) {
	uint8_t *src = (uint8_t *)p_buffer.start;
	uint8_t *y0_src = src + component_indexes[0];
	uint8_t *y1_src = src + component_indexes[1];
	uint8_t *u_src = src + component_indexes[2];
	uint8_t *v_src = src + component_indexes[3];
	uint8_t *dst = (uint8_t *)image_data.ptrw();

	for (int i = 0; i < width * height; i += 2) {
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

	// Defer to main thread to avoid race conditions with RenderingServer.
	image.instantiate();
	image->set_data(width, height, false, Image::FORMAT_RGB8, image_data);

	camera_feed->call_deferred("set_rgb_image", image);
}

CopyBufferDecoder::CopyBufferDecoder(CameraFeed *p_camera_feed, const CopyFormat &p_format) :
		BufferDecoder(p_camera_feed) {
	format = p_format.format;
	convert_bgr = p_format.convert_bgr;
	image_data.resize(width * height * p_format.stride);
}

void CopyBufferDecoder::decode(StreamingBuffer p_buffer) {
	uint8_t *src = (uint8_t *)p_buffer.start;
	uint8_t *dst = (uint8_t *)image_data.ptrw();

	if (convert_bgr) {
		// Windows RGB24 uses BGR byte order.
		// See: https://learn.microsoft.com/en-us/windows/win32/directshow/uncompressed-rgb-video-subtypes
		if (flip_vertical) {
			for (int y = 0; y < height; y++) {
				uint8_t *row_src = src + (height - 1 - y) * width * 3;
				for (int x = 0; x < width; x++) {
					dst[0] = row_src[2];
					dst[1] = row_src[1];
					dst[2] = row_src[0];
					dst += 3;
					row_src += 3;
				}
			}
		} else {
			for (int i = 0; i < width * height; i++) {
				dst[0] = src[2];
				dst[1] = src[1];
				dst[2] = src[0];
				dst += 3;
				src += 3;
			}
		}
	} else {
		if (flip_vertical) {
			int stride = p_buffer.length / height;
			for (int y = 0; y < height; y++) {
				uint8_t *row_src = src + (height - 1 - y) * stride;
				memcpy(dst, row_src, stride);
				dst += stride;
			}
		} else {
			memcpy(dst, src, p_buffer.length);
		}
	}

	// Defer to main thread to avoid race conditions with RenderingServer.
	image.instantiate();
	image->set_data(width, height, false, format, image_data);

	camera_feed->call_deferred("set_rgb_image", image);
}

JpegBufferDecoder::JpegBufferDecoder(CameraFeed *p_camera_feed) :
		BufferDecoder(p_camera_feed) {
}

void JpegBufferDecoder::decode(StreamingBuffer p_buffer) {
	image_data.resize(p_buffer.length);
	uint8_t *dst = (uint8_t *)image_data.ptrw();
	memcpy(dst, p_buffer.start, p_buffer.length);

	// Defer to main thread to avoid race conditions with RenderingServer.
	image.instantiate();
	if (image->load_jpg_from_buffer(image_data) == OK) {
		camera_feed->call_deferred("set_rgb_image", image);
	}
}

Nv12BufferDecoder::Nv12BufferDecoder(CameraFeed *p_camera_feed) :
		BufferDecoder(p_camera_feed) {
}

void Nv12BufferDecoder::decode(StreamingBuffer p_buffer) {
	// NV12 format: Y plane followed by interleaved UV plane.
	// Stride may be larger than width due to memory alignment.
	int stride = p_buffer.stride > 0 ? p_buffer.stride : width;
	int y_plane_size = stride * height;

	int y_size = width * height;
	int uv_size = width * height / 2;

	data_y.resize(y_size);
	data_uv.resize(uv_size);

	uint8_t *src = (uint8_t *)p_buffer.start;
	uint8_t *y_dst = (uint8_t *)data_y.ptrw();
	uint8_t *uv_dst = (uint8_t *)data_uv.ptrw();

	// Copy Y plane (row by row if stride differs from width).
	if (stride == width) {
		if (flip_vertical) {
			for (int y = 0; y < height; y++) {
				memcpy(y_dst + y * width, src + (height - 1 - y) * stride, width);
			}
		} else {
			memcpy(y_dst, src, y_size);
		}
	} else {
		for (int y = 0; y < height; y++) {
			int src_row = flip_vertical ? (height - 1 - y) : y;
			memcpy(y_dst + y * width, src + src_row * stride, width);
		}
	}

	// Copy UV plane (row by row if stride differs from width).
	uint8_t *uv_src = src + y_plane_size;
	int uv_height = height / 2;
	if (stride == width) {
		if (flip_vertical) {
			for (int y = 0; y < uv_height; y++) {
				memcpy(uv_dst + y * width, uv_src + (uv_height - 1 - y) * stride, width);
			}
		} else {
			memcpy(uv_dst, uv_src, uv_size);
		}
	} else {
		for (int y = 0; y < uv_height; y++) {
			int src_row = flip_vertical ? (uv_height - 1 - y) : y;
			memcpy(uv_dst + y * width, uv_src + src_row * stride, width);
		}
	}

	// Defer to main thread to avoid race conditions with RenderingServer.
	image_y.instantiate();
	image_y->set_data(width, height, false, Image::FORMAT_L8, data_y);

	image_uv.instantiate();
	image_uv->set_data(width / 2, height / 2, false, Image::FORMAT_RG8, data_uv);

	camera_feed->call_deferred("set_ycbcr_images", image_y, image_uv);
}

NullBufferDecoder::NullBufferDecoder(CameraFeed *p_camera_feed) :
		BufferDecoder(p_camera_feed) {
}

void NullBufferDecoder::decode(StreamingBuffer p_buffer) {
}
