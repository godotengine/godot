/**************************************************************************/
/*  audio_stream_polyphonic.compat.inc                                    */
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

AudioStreamPlaybackPolyphonic::ID AudioStreamPlaybackPolyphonic::_play_stream_bind_compat_91382(const Ref<AudioStream> &p_stream, float p_from_offset, float p_volume_db, float p_pitch_scale) {
	return play_stream(p_stream, p_from_offset, p_volume_db, p_pitch_scale, AudioServer::PlaybackType::PLAYBACK_TYPE_DEFAULT, SceneStringName(Master));
}

void AudioStreamPlaybackPolyphonic::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("play_stream", "stream", "from_offset", "volume_db", "pitch_scale"), &AudioStreamPlaybackPolyphonic::_play_stream_bind_compat_91382, DEFVAL(0), DEFVAL(0), DEFVAL(1.0));
}

#endif
