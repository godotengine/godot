/**************************************************************************/
/*  viewport_upscaler.cpp                                                 */
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

#include "viewport_upscaler.h"

#include "servers/rendering/rendering_server.h"

void ViewportUpscaler::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_requires_motion_vectors", "enabled"), &ViewportUpscaler::set_requires_motion_vectors);
	ClassDB::bind_method(D_METHOD("get_requires_motion_vectors"), &ViewportUpscaler::get_requires_motion_vectors);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "requires_motion_vectors"), "set_requires_motion_vectors", "get_requires_motion_vectors");

	ClassDB::bind_method(D_METHOD("set_mipmap_bias", "mipmap_bias"), &ViewportUpscaler::set_mipmap_bias);
	ClassDB::bind_method(D_METHOD("get_mipmap_bias"), &ViewportUpscaler::get_mipmap_bias);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mipmap_bias"), "set_mipmap_bias", "get_mipmap_bias");

	ClassDB::bind_method(D_METHOD("set_jitter_phase_count", "jitter_phase_count"), &ViewportUpscaler::set_jitter_phase_count);
	ClassDB::bind_method(D_METHOD("get_jitter_phase_count"), &ViewportUpscaler::get_jitter_phase_count);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "jitter_phase_count"), "set_jitter_phase_count", "get_jitter_phase_count");

	GDVIRTUAL_BIND(_render_callback, "render_data");
}

void ViewportUpscaler::render_callback(const RenderData *p_render_data) {
	GDVIRTUAL_CALL(_render_callback, p_render_data);
}

void ViewportUpscaler::set_requires_motion_vectors(bool p_enabled) {
	requires_motion_vectors = p_enabled;
	if (rid.is_valid()) {
		RenderingServer *rs = RenderingServer::get_singleton();
		ERR_FAIL_NULL(rs);
		rs->viewport_upscaler_set_requires_motion_vectors(rid, requires_motion_vectors);
	}
}

bool ViewportUpscaler::get_requires_motion_vectors() const {
	return requires_motion_vectors;
}

void ViewportUpscaler::set_mipmap_bias(float p_mipmap_bias) {
	mipmap_bias = p_mipmap_bias;
	if (rid.is_valid()) {
		RenderingServer *rs = RenderingServer::get_singleton();
		ERR_FAIL_NULL(rs);
		rs->viewport_upscaler_set_mipmap_bias(rid, mipmap_bias);
	}
}

float ViewportUpscaler::get_mipmap_bias() const {
	return mipmap_bias;
}

void ViewportUpscaler::set_jitter_phase_count(uint32_t p_jitter_phase_count) {
	jitter_phase_count = p_jitter_phase_count;
}

uint32_t ViewportUpscaler::get_jitter_phase_count() const {
	return jitter_phase_count;
}

ViewportUpscaler::ViewportUpscaler() {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs != nullptr) {
		rid = rs->viewport_upscaler_create();
		rs->viewport_upscaler_set_render_callback(rid, callable_mp(this, &ViewportUpscaler::render_callback));
	}
}

ViewportUpscaler::~ViewportUpscaler() {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs != nullptr && rid.is_valid()) {
		rs->free_rid(rid);
	}
}
