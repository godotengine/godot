/**************************************************************************/
/*  waveform_rect.cpp                                                     */
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

#include "waveform_rect.h"

void WaveformRect::update_waveform() {
	Rect2 rect = get_rect();
	Size2 size = rect.size;

	if (preview.is_null()) {
		return;
	}

	for (int i = 0; i <= size.x; i++) {
		float ofs = i * preview_len / size.x;
		float ofs_n = (i + 1) * preview_len / size.x;

		float maxi = preview->get_max(ofs, ofs_n) * 0.5 + 0.5;
		float mini = preview->get_min(ofs, ofs_n) * 0.5 + 0.5;

		draw_line(Vector2(i + 1, size.y * 0.05 + mini * size.y * 0.9), Vector2(i + 1, size.y * 0.05 + maxi * size.y * 0.9), get_color(), 1, false);
	}
}

void WaveformRect::set_stream(Ref<AudioStream> p_stream) {
	if (p_stream != stream) {
		stream = p_stream;
		preview = generator->generate_preview(stream);
		preview_len = float(preview->get_length());
		call_deferred("queue_redraw");
	}
}

void WaveformRect::set_color(const Color &p_color) {
	if (p_color != color) {
		color = p_color;
		call_deferred("queue_redraw");
	}
}

Ref<AudioStream> WaveformRect::get_stream() {
	return stream;
}

Color WaveformRect::get_color() const {
	return color;
}

void WaveformRect::_preview_updated(ObjectID p_id) {
	if (not loaded) {
		loaded = true;
		emit_signal(SNAME("waveform_updated"));
	}
}

void WaveformRect::_notification(int p_notification) {
	switch (p_notification) {
		case Control::NOTIFICATION_DRAW: {
			if (loaded) {
				update_waveform();
			}
		} break;
	}
}

void WaveformRect::_bind_methods() {
	ClassDB::bind_method(D_METHOD("update_waveform"), &WaveformRect::update_waveform);
	ClassDB::bind_method(D_METHOD("set_color", "color"), &WaveformRect::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &WaveformRect::get_color);
	ClassDB::bind_method(D_METHOD("set_stream", "stream"), &WaveformRect::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream"), &WaveformRect::get_stream);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"), "set_stream", "get_stream");

	ADD_SIGNAL(MethodInfo("waveform_updated"));
}

WaveformRect::WaveformRect() {
	color = Color(1.0, 1.0, 1.0); // default as white
	stream = nullptr;
	generator = memnew(AudioStreamPreviewGenerator);
	generator->connect(SNAME("preview_updated"), callable_mp(this, &WaveformRect::_preview_updated));
}
