/*************************************************************************/
/*  audio_effect_stream.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef AUDIO_EFFECT_STREAM_OPUS_H
#define AUDIO_EFFECT_STREAM_OPUS_H

#include "core/config/engine.h"
#include "servers/audio_server.h"

#include "ring_buffer_audio_frame.h"
#include "servers/audio/audio_effect.h"

class AudioEffectStream;

class AudioEffectStreamInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectStreamInstance, AudioEffectInstance);
	friend class AudioEffectStream;
	Ref<AudioEffectStream> base;

	bool is_streaming = false;

	Ref<RingBufferAudioFrame> output_ring_buffer;
	enum {
		IO_BUFFER_SIZE_MS = 1500
	};

public:
	void init();
	virtual void process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) override;
	virtual bool process_silence() const override;
	AudioEffectStreamInstance();
	~AudioEffectStreamInstance();
	void set_streaming(bool val);
	bool get_streaming() const;
};

class AudioEffectStream : public AudioEffect {
	GDCLASS(AudioEffectStream, AudioEffect)
	friend class AudioEffectStreamInstance;

	Ref<AudioEffectStreamInstance> current_instance;

protected:
	static void _bind_methods();

public:
	virtual Ref<AudioEffectInstance> instance() override;

	Ref<RingBufferAudioFrame> init(int32_t p_ring_buffer_max_size);
	bool is_streaming_active() const;
	AudioEffectStream();
	~AudioEffectStream();

private:
	bool buffering_active;
};

#endif // AUDIO_EFFECT_STREAM_OPUS_H
