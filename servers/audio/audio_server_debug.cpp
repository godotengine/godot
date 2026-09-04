/**************************************************************************/
/*  audio_server_debug.cpp                                                */
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

#ifdef DEBUG_ENABLED

#include "audio_server_debug.h"

#include "core/config/project_settings.h"
#include "servers/audio/audio_server.h"
#include "servers/rendering/rendering_server.h"

AudioServerDebug *AudioServerDebug::singleton = nullptr;

AudioServerDebug *AudioServerDebug::get_singleton() {
	return singleton;
}

AudioServerDebug::AudioServerDebug() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
	register_settings();
	generate_rids();
}

AudioServerDebug::~AudioServerDebug() {
	if (singleton == this) {
		singleton = nullptr;
	}
	clear_rids();
}

void AudioServerDebug::register_settings() {
	debug_audio_2d_visualization_mode = GLOBAL_DEF_BASIC(PropertyInfo(Variant::INT, "debug/shapes/audio/2d/audio_visualization_mode", PROPERTY_HINT_ENUM, "Disabled,Draw max range,Draw attenuation"), 2);
	debug_audio_2d_visualization_color = GLOBAL_DEF_BASIC("debug/shapes/audio/2d/audio_visualization_color", Color(0.0, 0.5, .9, 1.0));
	debug_audio_2d_visualization_ring_count = GLOBAL_DEF(PropertyInfo(Variant::INT, "debug/shapes/audio/2d/audio_visualization_ring_count", PROPERTY_HINT_RANGE, "8,24,1"), 12);
}

void AudioServerDebug::update_from_settings() {
	if (!ProjectSettings::get_singleton()->check_changed_settings_in_group("debug/shapes/audio/2d/")) {
		return;
	}
	set_debug_audio_2d_visualization_mode(GLOBAL_GET("debug/shapes/audio/2d/audio_visualization_mode"));
	set_debug_audio_2d_visualization_color(GLOBAL_GET("debug/shapes/audio/2d/audio_visualization_color"));
	set_debug_audio_2d_visualization_ring_count(GLOBAL_GET("debug/shapes/audio/2d/audio_visualization_ring_count"));
}

void AudioServerDebug::generate_rids() {
	RenderingServer *rs = RS::get_singleton();
	debug_audio_2d_visualization_circle_mesh_rid = rs->mesh_create();

	const int segments = 32;

	// center
	Vector<Vector2> circle_vertices;
	circle_vertices.resize(segments * 3);

	Vector2 *w = circle_vertices.ptrw();
	int ic = 0;
	for (int i = 0; i < segments; i++) {
		float phi0 = (Math::TAU / segments) * i;
		float phi1 = (Math::TAU / segments) * (i + 1);

		Vector2 p0 = Vector2(Math::cos(phi0), Math::sin(phi0));
		Vector2 p1 = Vector2(Math::cos(phi1), Math::sin(phi1));

		w[ic++] = Vector2(0, 0);
		w[ic++] = p0;
		w[ic++] = p1;
	}

	Array ring_mesh_array;
	ring_mesh_array.resize(RSE::ARRAY_MAX);
	ring_mesh_array[RSE::ARRAY_VERTEX] = circle_vertices;

	rs->mesh_add_surface_from_arrays(debug_audio_2d_visualization_circle_mesh_rid, RSE::PRIMITIVE_TRIANGLES, ring_mesh_array, Array(), Dictionary(), RSE::ARRAY_FLAG_USE_2D_VERTICES);

	// rings
	debug_audio_2d_visualization_rings_mesh_rids.resize(debug_audio_2d_visualization_ring_count);
	for (int r = 0; r < debug_audio_2d_visualization_ring_count; r++) {
		float inner_radius = (r * 1) + 0.5f;
		float outer_radius = ((r + 1) * 1) + 0.5f;

		RID r_mesh = rs->mesh_create();
		Vector<Vector2> ring_v;
		ring_v.resize((segments + 1) * 2);

		Vector2 *v_ptr = ring_v.ptrw();
		int idx = 0;
		for (int i = 0; i <= segments; i++) {
			float a = Math::TAU * float(i) / segments;
			float cos_a = Math::cos(a);
			float sin_a = Math::sin(a);

			v_ptr[idx++] = Vector2(cos_a * inner_radius, sin_a * inner_radius);
			v_ptr[idx++] = Vector2(cos_a * outer_radius, sin_a * outer_radius);
		}

		Array arrays_ring;
		arrays_ring.resize(RSE::ARRAY_MAX);
		arrays_ring[RSE::ARRAY_VERTEX] = ring_v;

		rs->mesh_add_surface_from_arrays(r_mesh, RSE::PRIMITIVE_TRIANGLE_STRIP, arrays_ring, Array(), Dictionary(), RSE::ARRAY_FLAG_USE_2D_VERTICES);
		debug_audio_2d_visualization_rings_mesh_rids.write[r] = r_mesh;
	}

	// outline
	debug_audio_2d_visualization_outline_mesh_rid = rs->mesh_create();

	Vector<Vector2> outline_vertices;
	outline_vertices.resize(segments + 1);
	for (int i = 0; i <= segments; i++) {
		float phi = (Math::TAU / segments) * i;
		outline_vertices.write[i] = Vector2(Math::cos(phi), Math::sin(phi));
	}

	Array outline_mesh_array;
	outline_mesh_array.resize(RSE::ARRAY_MAX);
	outline_mesh_array[RSE::ARRAY_VERTEX] = outline_vertices;

	rs->mesh_add_surface_from_arrays(debug_audio_2d_visualization_outline_mesh_rid, RSE::PRIMITIVE_LINE_STRIP, outline_mesh_array, Array(), Dictionary(), RSE::ARRAY_FLAG_USE_2D_VERTICES);
}

void AudioServerDebug::clear_rids() {
	RenderingServer *rs = RS::get_singleton();
	rs->free_rid(debug_audio_2d_visualization_circle_mesh_rid);
	for (int r = 0; r < debug_audio_2d_visualization_rings_mesh_rids.size(); r++) {
		rs->free_rid(debug_audio_2d_visualization_rings_mesh_rids[r]);
	}
	rs->free_rid(debug_audio_2d_visualization_outline_mesh_rid);
}

RID AudioServerDebug::get_debug_audio_2d_visualization_circle_mesh_rid() const {
	return debug_audio_2d_visualization_circle_mesh_rid;
}

Vector<RID> AudioServerDebug::get_debug_audio_2d_visualization_rings_mesh_rids() const {
	return debug_audio_2d_visualization_rings_mesh_rids;
}

RID AudioServerDebug::get_debug_audio_2d_visualization_outline_mesh_rid() const {
	return debug_audio_2d_visualization_outline_mesh_rid;
}

int AudioServerDebug::get_debug_audio_2d_visualization_ring_count() const {
	return debug_audio_2d_visualization_ring_count;
}

void AudioServerDebug::set_debug_audio_2d_visualization_ring_count(int p_count) {
	if (debug_audio_2d_visualization_ring_count == p_count) {
		return;
	}
	debug_audio_2d_visualization_ring_count = p_count;
	AudioServer::get_singleton()->emit_debug_audio_2d_visualization_changed_signal(true);
}

Color AudioServerDebug::get_debug_audio_2d_visualization_color() const {
	return debug_audio_2d_visualization_color;
}

void AudioServerDebug::set_debug_audio_2d_visualization_color(const Color &p_color) {
	if (debug_audio_2d_visualization_color == p_color) {
		return;
	}
	debug_audio_2d_visualization_color = p_color;
	AudioServer::get_singleton()->emit_debug_audio_2d_visualization_changed_signal(false);
}

int AudioServerDebug::get_debug_audio_2d_visualization_mode() const {
	return debug_audio_2d_visualization_mode;
}

void AudioServerDebug::set_debug_audio_2d_visualization_mode(int p_mode) {
	if (debug_audio_2d_visualization_mode == p_mode) {
		return;
	}
	debug_audio_2d_visualization_mode = p_mode;
	AudioServer::get_singleton()->emit_debug_audio_2d_visualization_changed_signal(false);
}

void AudioServerDebug::set_debug_audio_2d_visualization_enabled(bool p_enabled) {
	debug_audio_2d_visualization_enabled = p_enabled;
}

bool AudioServerDebug::get_debug_audio_2d_visualization_enabled() const {
	return debug_audio_2d_visualization_enabled;
}

#endif // DEBUG_ENABLED
