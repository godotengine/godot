/**************************************************************************/
/*  rendering_effect.cpp                                                  */
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

#include "rendering_effect.h"
#include "servers/rendering_server.h"

void RenderingEffect::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_effect_callback_type", "effect_callback_type"), &RenderingEffect::set_effect_callback_type);
	ClassDB::bind_method(D_METHOD("get_effect_callback_type"), &RenderingEffect::get_effect_callback_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "effect_callback_type", PROPERTY_HINT_ENUM, "Pre opaque,Post opaque,Pre transparent,Post transparent"), "set_effect_callback_type", "get_effect_callback_type");

	ClassDB::bind_method(D_METHOD("set_access_resolved_color", "enable"), &RenderingEffect::set_access_resolved_color);
	ClassDB::bind_method(D_METHOD("get_access_resolved_color"), &RenderingEffect::get_access_resolved_color);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "access_resolved_color"), "set_access_resolved_color", "get_access_resolved_color");

	ClassDB::bind_method(D_METHOD("set_access_resolved_depth", "enable"), &RenderingEffect::set_access_resolved_depth);
	ClassDB::bind_method(D_METHOD("get_access_resolved_depth"), &RenderingEffect::get_access_resolved_depth);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "access_resolved_depth"), "set_access_resolved_depth", "get_access_resolved_depth");

	ClassDB::bind_method(D_METHOD("set_needs_motion_vectors", "enable"), &RenderingEffect::set_needs_motion_vectors);
	ClassDB::bind_method(D_METHOD("get_needs_motion_vectors"), &RenderingEffect::get_needs_motion_vectors);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "needs_motion_vectors"), "set_needs_motion_vectors", "get_needs_motion_vectors");

	BIND_ENUM_CONSTANT(EFFECT_CALLBACK_TYPE_PRE_OPAQUE)
	BIND_ENUM_CONSTANT(EFFECT_CALLBACK_TYPE_POST_OPAQUE)
	BIND_ENUM_CONSTANT(EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT)
	BIND_ENUM_CONSTANT(EFFECT_CALLBACK_TYPE_POST_TRANSPARENT)
	BIND_ENUM_CONSTANT(EFFECT_CALLBACK_TYPE_MAX)

	GDVIRTUAL_BIND(_render_callback, "effect_callback_type", "render_data");
}

void RenderingEffect::set_effect_callback_type(EffectCallbackType p_callback_type) {
	effect_callback_type = p_callback_type;
	if (rid.is_valid()) {
		RenderingServer *rs = RenderingServer::get_singleton();
		ERR_FAIL_NULL(rs);
		rs->rendering_effect_set_callback(rid, RenderingServer::RenderingEffectCallbackType(effect_callback_type), Callable(this, "_render_callback"));
	}
}

RenderingEffect::EffectCallbackType RenderingEffect::get_effect_callback_type() const {
	return effect_callback_type;
}

void RenderingEffect::set_access_resolved_color(bool p_enabled) {
	access_resolved_color = p_enabled;
	if (rid.is_valid()) {
		RenderingServer *rs = RenderingServer::get_singleton();
		ERR_FAIL_NULL(rs);
		rs->rendering_effect_set_flag(rid, RS::RenderingEffectFlags::RENDERING_EFFECT_FLAG_ACCESS_RESOLVED_COLOR, access_resolved_color);
	}
}

bool RenderingEffect::get_access_resolved_color() const {
	return access_resolved_color;
}

void RenderingEffect::set_access_resolved_depth(bool p_enabled) {
	access_resolved_depth = p_enabled;
	if (rid.is_valid()) {
		RenderingServer *rs = RenderingServer::get_singleton();
		ERR_FAIL_NULL(rs);
		rs->rendering_effect_set_flag(rid, RS::RenderingEffectFlags::RENDERING_EFFECT_FLAG_ACCESS_RESOLVED_DEPTH, access_resolved_depth);
	}
}

bool RenderingEffect::get_access_resolved_depth() const {
	return access_resolved_depth;
}

void RenderingEffect::set_needs_motion_vectors(bool p_enabled) {
	needs_motion_vectors = p_enabled;
	if (rid.is_valid()) {
		RenderingServer *rs = RenderingServer::get_singleton();
		ERR_FAIL_NULL(rs);
		rs->rendering_effect_set_flag(rid, RS::RenderingEffectFlags::RENDERING_EFFECT_FLAG_NEEDS_MOTION_VECTORS, needs_motion_vectors);
	}
}

bool RenderingEffect::get_needs_motion_vectors() const {
	return needs_motion_vectors;
}

RenderingEffect::RenderingEffect() {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs != nullptr) {
		rid = rs->rendering_effect_create();
		rs->rendering_effect_set_callback(rid, RenderingServer::RenderingEffectCallbackType(effect_callback_type), Callable(this, "_render_callback"));
	}
}

RenderingEffect::~RenderingEffect() {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs != nullptr && rid.is_valid()) {
		rs->free(rid);
	}
}
