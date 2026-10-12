/**************************************************************************/
/*  external_texture.cpp                                                  */
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

#include "external_texture.h"

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "servers/rendering/rendering_server.h"

static bool _uses_rendering_device() {
#ifdef RD_ENABLED
	return RenderingServer::get_singleton()->get_rendering_device() != nullptr;
#else
	return false;
#endif
}

void ExternalTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &ExternalTexture::set_size);
	ClassDB::bind_method(D_METHOD("get_external_texture_id"), &ExternalTexture::get_external_texture_id);
	ClassDB::bind_method(D_METHOD("set_external_buffer_id", "external_buffer_id"), &ExternalTexture::set_external_buffer_id);
	ClassDB::bind_method(D_METHOD("submit_hardware_buffer", "buffer", "color_model", "color_range", "crop", "completion"), &ExternalTexture::submit_hardware_buffer, DEFVAL(COLOR_MODEL_AUTOMATIC), DEFVAL(COLOR_RANGE_AUTOMATIC), DEFVAL(Rect2i()), DEFVAL(Callable()));

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");

	BIND_ENUM_CONSTANT(COLOR_MODEL_AUTOMATIC);
	BIND_ENUM_CONSTANT(COLOR_MODEL_RGB);
	BIND_ENUM_CONSTANT(COLOR_MODEL_BT601);
	BIND_ENUM_CONSTANT(COLOR_MODEL_BT709);
	BIND_ENUM_CONSTANT(COLOR_MODEL_BT2020);

	BIND_ENUM_CONSTANT(COLOR_RANGE_AUTOMATIC);
	BIND_ENUM_CONSTANT(COLOR_RANGE_FULL);
	BIND_ENUM_CONSTANT(COLOR_RANGE_NARROW);

	BIND_ENUM_CONSTANT(COMPLETION_COMPLETED);
	BIND_ENUM_CONSTANT(COMPLETION_DROPPED);
	BIND_ENUM_CONSTANT(COMPLETION_FAILED);
}

uint64_t ExternalTexture::get_external_texture_id() const {
	MutexLock lock(mutex);
	if (_uses_rendering_device()) {
		return 0;
	}
	_ensure_created();
	return RenderingServer::get_singleton()->texture_get_native_handle(texture);
}

void ExternalTexture::set_size(const Size2 &p_size) {
	MutexLock lock(mutex);
	if (p_size.width > 0 && p_size.height > 0 && p_size != size) {
		size = p_size;
		if (using_hardware_buffer) {
			return;
		}
		_ensure_created();
		RenderingServer::get_singleton()->texture_external_update(texture, size.width, size.height, external_buffer);
		emit_changed();
	}
}

Size2 ExternalTexture::get_size() const {
	MutexLock lock(mutex);
	return using_hardware_buffer ? Size2(resolved_size) : size;
}

void ExternalTexture::set_external_buffer_id(uint64_t p_external_buffer) {
	MutexLock lock(mutex);
	if (p_external_buffer != external_buffer) {
		external_buffer = p_external_buffer;
		_ensure_created();
		RenderingServer::get_singleton()->texture_external_update(texture, size.width, size.height, external_buffer);
	}
}

Error ExternalTexture::submit_hardware_buffer(uint64_t p_buffer, ColorModel p_color_model, ColorRange p_color_range, const Rect2i &p_crop, const Callable &p_completion) {
	ERR_FAIL_INDEX_V(p_color_model, COLOR_MODEL_BT2020 + 1, ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_color_range, COLOR_RANGE_NARROW + 1, ERR_INVALID_PARAMETER);
	MutexLock lock(mutex);
	_ensure_created();
	Error error = RenderingServer::get_singleton()->texture_external_submit(texture, p_buffer, RSE::ExternalTextureColorModel(p_color_model), RSE::ExternalTextureColorRange(p_color_range), p_crop, p_completion, callable_mp(this, &ExternalTexture::_hardware_buffer_changed));
	if (error == OK && !using_hardware_buffer) {
		resolved_size = Size2i(size);
		using_hardware_buffer = true;
	}
	return error;
}

int ExternalTexture::get_width() const {
	return get_size().width;
}

int ExternalTexture::get_height() const {
	return get_size().height;
}

bool ExternalTexture::has_alpha() const {
	MutexLock lock(mutex);
	return resolved_alpha;
}

RID ExternalTexture::get_rid() const {
	MutexLock lock(mutex);
	if (_uses_rendering_device()) {
		_ensure_created();
		return texture;
	}
	if (!texture.is_valid()) {
		texture = RenderingServer::get_singleton()->texture_2d_placeholder_create();
		using_placeholder = true;
	}
	return texture;
}

Ref<Image> ExternalTexture::get_image() const {
	if (!_uses_rendering_device()) {
		return Ref<Image>();
	}
	return RenderingServer::get_singleton()->texture_2d_get(get_rid());
}

void ExternalTexture::_ensure_created() const {
	if (texture.is_valid() && !using_placeholder) {
		return;
	}

	RID new_texture = RenderingServer::get_singleton()->texture_external_create(size.width, size.height);
	if (using_placeholder) {
		DEV_ASSERT(texture.is_valid());
		RenderingServer::get_singleton()->texture_replace(texture, new_texture);
		using_placeholder = false;
	} else {
		texture = new_texture;
	}
}

void ExternalTexture::_hardware_buffer_changed(const Size2i &p_size, bool p_has_alpha) {
	{
		MutexLock lock(mutex);
		resolved_size = p_size;
		resolved_alpha = p_has_alpha;
	}
	emit_changed();
}

ExternalTexture::ExternalTexture() {
}

ExternalTexture::~ExternalTexture() {
	if (texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RenderingServer::get_singleton()->free_rid(texture);
	}
}
