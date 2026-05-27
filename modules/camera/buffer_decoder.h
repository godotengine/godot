/**************************************************************************/
/*  buffer_decoder.h                                                      */
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

#pragma once

#include "core/io/image.h"
#include "core/templates/vector.h"

class CameraFeed;

struct StreamingBuffer {
	void *start = nullptr;
	size_t length = 0;
	int32_t pitch = 0; // Y plane row stride in bytes (0 = assume tightly packed: pitch == width).
};

class BufferDecoder {
protected:
	CameraFeed *camera_feed = nullptr;
	Ref<Image> image;
	int width = 0;
	int height = 0;

public:
	virtual void decode(StreamingBuffer p_buffer) = 0;

	BufferDecoder(CameraFeed *p_camera_feed);
	virtual ~BufferDecoder() {}
};

class AbstractYuyvBufferDecoder : public BufferDecoder {
protected:
	int *component_indexes = nullptr;

public:
	AbstractYuyvBufferDecoder(CameraFeed *p_camera_feed);
	~AbstractYuyvBufferDecoder();
};

class SeparateYuyvBufferDecoder : public AbstractYuyvBufferDecoder {
private:
	Vector<uint8_t> y_image_data;
	Vector<uint8_t> cbcr_image_data;
	Ref<Image> y_image;
	Ref<Image> cbcr_image;

public:
	SeparateYuyvBufferDecoder(CameraFeed *p_camera_feed);
	virtual void decode(StreamingBuffer p_buffer) override;
};

class YuyvToGrayscaleBufferDecoder : public AbstractYuyvBufferDecoder {
private:
	Vector<uint8_t> image_data;

public:
	YuyvToGrayscaleBufferDecoder(CameraFeed *p_camera_feed);
	virtual void decode(StreamingBuffer p_buffer) override;
};

class YuyvToRgbBufferDecoder : public AbstractYuyvBufferDecoder {
private:
	Vector<uint8_t> image_data;

public:
	YuyvToRgbBufferDecoder(CameraFeed *p_camera_feed);
	virtual void decode(StreamingBuffer p_buffer) override;
};

class CopyBufferDecoder : public BufferDecoder {
private:
	Vector<uint8_t> image_data;
	bool rgba = false;

public:
	CopyBufferDecoder(CameraFeed *p_camera_feed, bool p_rgba);
	virtual void decode(StreamingBuffer p_buffer) override;
};

// Decoder for V4L2_PIX_FMT_YUV420 (YU12 / I420) and V4L2_PIX_FMT_YVU420 (YV12).
// Planar 4:2:0: Y plane (w*h) followed by Cb and Cr planes (w/2 * h/2 each).
// V4L2_PIX_FMT_YVU420 swaps the order to Y, V, U.
class Yuv420BufferDecoder : public BufferDecoder {
private:
	bool v_plane_first = false; // True for V4L2_PIX_FMT_YVU420 (Y, V, U order).
	Vector<uint8_t> y_buffer;
	Vector<uint8_t> cbcr_buffer; // Interleaved CbCr (RG8 format: R=Cb, G=Cr).
	Ref<Image> y_image;
	Ref<Image> cbcr_image;

public:
	Yuv420BufferDecoder(CameraFeed *p_camera_feed, bool p_v_plane_first);
	virtual void decode(StreamingBuffer p_buffer) override;
};

class JpegBufferDecoder : public BufferDecoder {
private:
	Vector<uint8_t> image_data;

public:
	JpegBufferDecoder(CameraFeed *p_camera_feed);
	virtual void decode(StreamingBuffer p_buffer) override;
};
