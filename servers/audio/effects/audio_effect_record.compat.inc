/**************************************************************************/
/*  audio_effect_record.compat.inc                                        */
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

void AudioEffectRecord::_set_recording_active_bind_compat_92532(bool p_record) {
	set_recording_active(p_record, 0);
}

bool AudioEffectRecord::_is_recording_active_bind_compat_92532() const {
	return is_recording_active(0);
}

Ref<AudioStreamWAV> AudioEffectRecord::_get_recording_bind_compat_92532() const {
	return get_recording(0);
}

void AudioEffectRecord::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("set_recording_active", "frames"), &AudioEffectRecord::_set_recording_active_bind_compat_92532);
	ClassDB::bind_compatibility_method(D_METHOD("is_recording_active"), &AudioEffectRecord::_is_recording_active_bind_compat_92532);
	ClassDB::bind_compatibility_method(D_METHOD("get_recording"), &AudioEffectRecord::_get_recording_bind_compat_92532);
}

#endif // DISABLE_DEPRECATED
