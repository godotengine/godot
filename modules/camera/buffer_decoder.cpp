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
	uint8_t *y0_src = src + component_indexes[0];
	uint8_t *y1_src = src + component_indexes[1];

	for (int i = 0; i < width * height; i += 2) {
		*dst++ = *y0_src;
		*dst++ = *y1_src;

		y0_src += 4;
		y1_src += 4;
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
	memcpy(dst, p_buffer.start, p_buffer.length);

	if (image.is_valid()) {
		image->set_data(width, height, false, rgba ? Image::FORMAT_RGBA8 : Image::FORMAT_LA8, image_data);
	} else {
		image.instantiate(width, height, false, rgba ? Image::FORMAT_RGBA8 : Image::FORMAT_LA8, image_data);
	}

	camera_feed->set_rgb_image(image);
}

JpegBufferDecoder::JpegBufferDecoder(CameraFeed *p_camera_feed) :
		BufferDecoder(p_camera_feed) {
	tj_instance = tj3Init(TJINIT_DECOMPRESS);
	if (tj_instance) {
		// Allow partial/corrupt JPEG decoding for streaming sources like cameras.
		tj3Set(tj_instance, TJPARAM_STOPONWARNING, 0);
	}
	y_image.instantiate();
	cbcr_image.instantiate();
}

JpegBufferDecoder::~JpegBufferDecoder() {
	if (tj_instance) {
		tj3Destroy(tj_instance);
		tj_instance = nullptr;
	}
}

void JpegBufferDecoder::decode(StreamingBuffer p_buffer) {
	if (!tj_instance) {
		return;
	}

	size_t data_size = p_buffer.bytes_used > 0 ? p_buffer.bytes_used : p_buffer.length;
	uint8_t *src = (uint8_t *)p_buffer.start;

	// Verify JPEG SOI marker (FFD8).
	if (data_size < 2 || src[0] != 0xFF || src[1] != 0xD8) {
		return;
	}

	if (tj3DecompressHeader(tj_instance, src, data_size) < 0) {
		return;
	}

	const int jpeg_width = tj3Get(tj_instance, TJPARAM_JPEGWIDTH);
	const int jpeg_height = tj3Get(tj_instance, TJPARAM_JPEGHEIGHT);
	const TJCS colorspace = (TJCS)tj3Get(tj_instance, TJPARAM_COLORSPACE);
	const TJSAMP subsamp = (TJSAMP)tj3Get(tj_instance, TJPARAM_SUBSAMP);

	if (tj3Get(tj_instance, TJPARAM_PRECISION) > 8) {
		return;
	}

	// Grayscale images: decode directly.
	if (colorspace == TJCS_GRAY || subsamp == TJSAMP_GRAY) {
		// Resize buffer only if dimensions changed.
		if (jpeg_width != buffer_width || jpeg_height != buffer_height) {
			buffer_width = jpeg_width;
			buffer_height = jpeg_height;
			y_plane_buffer.resize(buffer_width * buffer_height);
		}

		if (tj3Decompress8(tj_instance, src, data_size, y_plane_buffer.ptrw(), 0, TJPF_GRAY) < 0) {
			return;
		}

		image->set_data(buffer_width, buffer_height, false, Image::FORMAT_L8, y_plane_buffer);
		camera_feed->set_rgb_image(image);
		return;
	}

	// Color images: decode to YUV planes and pass to GPU shader for conversion.
	const int y_plane_width = tj3YUVPlaneWidth(0, jpeg_width, subsamp);
	const int y_plane_height = tj3YUVPlaneHeight(0, jpeg_height, subsamp);
	const int cb_plane_width = tj3YUVPlaneWidth(1, jpeg_width, subsamp);
	const int cb_plane_height = tj3YUVPlaneHeight(1, jpeg_height, subsamp);
	const size_t y_size = (size_t)y_plane_width * y_plane_height;
	const size_t cb_size = (size_t)cb_plane_width * cb_plane_height;

	// Resize buffers only if dimensions changed.
	if (jpeg_width != buffer_width || jpeg_height != buffer_height) {
		buffer_width = jpeg_width;
		buffer_height = jpeg_height;

		y_plane_buffer.resize(y_size);
		cb_plane_buffer.resize(cb_size);
		cr_plane_buffer.resize(cb_size);
		// CbCr interleaved buffer (LA8 format: L=Cb, A=Cr).
		cbcr_buffer.resize(cb_size * 2);
	}

	// Set up YUV plane pointers (separate buffers, no extra copy needed for Y).
	uint8_t *y_ptr = y_plane_buffer.ptrw();
	uint8_t *cb_ptr = cb_plane_buffer.ptrw();
	uint8_t *cr_ptr = cr_plane_buffer.ptrw();

	unsigned char *planes[3] = { y_ptr, cb_ptr, cr_ptr };
	int strides[3] = { y_plane_width, cb_plane_width, cb_plane_width };

	if (tj3DecompressToYUVPlanes8(tj_instance, src, data_size, planes, strides) < 0) {
		return;
	}

	// Interleave Cb and Cr into RG8 format for GPU shader (R=Cb, G=Cr).
	uint8_t *cbcr_dst = cbcr_buffer.ptrw();
	for (size_t i = 0; i < cb_size; i++) {
		*cbcr_dst++ = cb_ptr[i]; // R channel = Cb (U)
		*cbcr_dst++ = cr_ptr[i]; // G channel = Cr (V)
	}

	y_image->set_data(y_plane_width, y_plane_height, false, Image::FORMAT_L8, y_plane_buffer);
	cbcr_image->set_data(cb_plane_width, cb_plane_height, false, Image::FORMAT_RG8, cbcr_buffer);
	camera_feed->set_ycbcr_images(y_image, cbcr_image);
}
