/**************************************************************************/
/*  test_external_texture.cpp                                             */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_external_texture)

#include "core/object/message_queue.h"
#include "scene/resources/external_texture.h"
#include "tests/signal_watcher.h"

namespace TestExternalTexture {

TEST_CASE("[SceneTree][ExternalTexture] Submissions reject invalid arguments and unsupported renderers") {
	Ref<ExternalTexture> texture = memnew(ExternalTexture);
	SIGNAL_WATCH(*texture, CoreStringName(changed));
	ERR_PRINT_OFF;
	CHECK(texture->submit_hardware_buffer(0) == ERR_INVALID_PARAMETER);
	CHECK(texture->submit_hardware_buffer(1, ExternalTexture::ColorModel(-1)) == ERR_INVALID_PARAMETER);
	CHECK(texture->submit_hardware_buffer(1, ExternalTexture::ColorModel(5)) == ERR_INVALID_PARAMETER);
	CHECK(texture->submit_hardware_buffer(1, ExternalTexture::COLOR_MODEL_AUTOMATIC, ExternalTexture::ColorRange(-1)) == ERR_INVALID_PARAMETER);
	CHECK(texture->submit_hardware_buffer(1, ExternalTexture::COLOR_MODEL_AUTOMATIC, ExternalTexture::ColorRange(3)) == ERR_INVALID_PARAMETER);
	CHECK(texture->submit_hardware_buffer(1, ExternalTexture::COLOR_MODEL_AUTOMATIC, ExternalTexture::COLOR_RANGE_AUTOMATIC, Rect2i(), Callable(*texture, "emit_changed").unbind(1)) == ERR_UNAVAILABLE);
	ERR_PRINT_ON;
	MessageQueue::get_singleton()->flush();
	SIGNAL_CHECK_FALSE(CoreStringName(changed));
	SIGNAL_UNWATCH(*texture, CoreStringName(changed));
	CHECK(texture->get_size() == Size2(256, 256));
	CHECK_FALSE(texture->has_alpha());
}

TEST_CASE("[SceneTree][ExternalTexture] Size and native texture access keep a stable RID") {
	Ref<ExternalTexture> texture = memnew(ExternalTexture);
	RID rid = texture->get_rid();
	REQUIRE(rid.is_valid());
	texture->set_size(Size2(640, 480));
	CHECK(texture->get_size() == Size2(640, 480));
	CHECK(texture->get_width() == 640);
	CHECK(texture->get_height() == 480);
	texture->set_size(Size2(0, 480));
	CHECK(texture->get_size() == Size2(640, 480));
	CHECK(texture->get_external_texture_id() == 0);
	texture->set_external_buffer_id(1);
	CHECK(texture->get_rid() == rid);
	CHECK(texture->get_image().is_null());
}

} // namespace TestExternalTexture
