/**************************************************************************/
/*  external_texture.h                                                    */
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

#include "core/os/mutex.h"
#include "scene/resources/texture.h"

// External textures as defined by OES_EGL_image_external (GLES) or VK_ANDROID_external_memory_android_hardware_buffer (Vulkan).
class ExternalTexture : public Texture2D {
	GDCLASS(ExternalTexture, Texture2D);

public:
	enum ColorModel {
		COLOR_MODEL_AUTOMATIC,
		COLOR_MODEL_RGB,
		COLOR_MODEL_BT601,
		COLOR_MODEL_BT709,
		COLOR_MODEL_BT2020,
	};

	enum ColorRange {
		COLOR_RANGE_AUTOMATIC,
		COLOR_RANGE_FULL,
		COLOR_RANGE_NARROW,
	};

	enum CompletionStatus {
		COMPLETION_COMPLETED,
		COMPLETION_DROPPED,
		COMPLETION_FAILED,
	};

private:
	mutable Mutex mutex;
	mutable RID texture;
	mutable bool using_placeholder = false;
	Size2 size = Size2(256, 256);
	uint64_t external_buffer = 0;
	Size2i resolved_size;
	bool resolved_alpha = false;
	bool using_hardware_buffer = false;

	void _ensure_created() const;
	void _hardware_buffer_changed(const Size2i &p_size, bool p_has_alpha);

protected:
	static void _bind_methods();

public:
	uint64_t get_external_texture_id() const;

	virtual Size2 get_size() const override;
	void set_size(const Size2 &p_size);

	void set_external_buffer_id(uint64_t p_external_buffer);
	Error submit_hardware_buffer(uint64_t p_buffer, ColorModel p_color_model = COLOR_MODEL_AUTOMATIC, ColorRange p_color_range = COLOR_RANGE_AUTOMATIC, const Rect2i &p_crop = Rect2i(), const Callable &p_completion = Callable());

	virtual int get_width() const override;
	virtual int get_height() const override;

	virtual RID get_rid() const override;
	virtual bool has_alpha() const override;
	virtual Ref<Image> get_image() const override;

	ExternalTexture();
	~ExternalTexture();
};

VARIANT_ENUM_CAST(ExternalTexture::ColorModel);
VARIANT_ENUM_CAST(ExternalTexture::ColorRange);
VARIANT_ENUM_CAST(ExternalTexture::CompletionStatus);
