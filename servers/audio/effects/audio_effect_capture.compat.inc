/**************************************************************************/
/*  audio_effect_capture.compat.inc                                       */
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

#ifndef DISABLE_DEPRECATED

bool AudioEffectCapture::_can_get_buffer_bind_compat_92532(int p_frames) const {
	return can_get_buffer(p_frames, 0);
}

PackedVector2Array AudioEffectCapture::_get_buffer_bind_compat_92532(int p_len) {
	return get_buffer(p_len, 0);
}

void AudioEffectCapture::_clear_buffer_bind_compat_92532() {
	clear_buffer(0);
}

int AudioEffectCapture::_get_frames_available_bind_compat_92532() const {
	return get_frames_available(0);
}

int64_t AudioEffectCapture::_get_discarded_frames_bind_compat_92532() const {
	return get_discarded_frames(0);
}

int AudioEffectCapture::_get_buffer_length_frames_bind_compat_92532() const {
	return get_buffer_length_frames(0);
}

int64_t AudioEffectCapture::_get_pushed_frames_bind_compat_92532() const {
	return get_pushed_frames(0);
}

void AudioEffectCapture::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("can_get_buffer", "frames"), &AudioEffectCapture::_can_get_buffer_bind_compat_92532);
	ClassDB::bind_compatibility_method(D_METHOD("get_buffer", "frames"), &AudioEffectCapture::_get_buffer_bind_compat_92532);
	ClassDB::bind_compatibility_method(D_METHOD("clear_buffer"), &AudioEffectCapture::_clear_buffer_bind_compat_92532);
	ClassDB::bind_compatibility_method(D_METHOD("get_frames_available"), &AudioEffectCapture::_get_frames_available_bind_compat_92532);
	ClassDB::bind_compatibility_method(D_METHOD("get_discarded_frames"), &AudioEffectCapture::_get_discarded_frames_bind_compat_92532);
	ClassDB::bind_compatibility_method(D_METHOD("get_buffer_length_frames"), &AudioEffectCapture::_get_buffer_length_frames_bind_compat_92532);
	ClassDB::bind_compatibility_method(D_METHOD("get_pushed_frames"), &AudioEffectCapture::_get_pushed_frames_bind_compat_92532);
}

#endif // DISABLE_DEPRECATED
