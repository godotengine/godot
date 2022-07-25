/*************************************************************************/
/*  audio_effect_stereo_enhance.h                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef AUDIO_EFFECT_STEREO_ENHANCE_H
#define AUDIO_EFFECT_STEREO_ENHANCE_H

#include "servers/audio/audio_effect.h"

class AudioEffectStereoEnhance;

class AudioEffectStereoEnhanceInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectStereoEnhanceInstance, AudioEffectInstance);
	friend class AudioEffectStereoEnhance;
	Ref<AudioEffectStereoEnhance> base;

	enum {

		MAX_DELAY_MS = 50
	};

	float *delay_ringbuff;
	unsigned int ringbuff_pos;
	unsigned int ringbuff_mask;

public:
	virtual void process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count);

	~AudioEffectStereoEnhanceInstance();
};

class AudioEffectStereoEnhance : public AudioEffect {
	GDCLASS(AudioEffectStereoEnhance, AudioEffect);

	friend class AudioEffectStereoEnhanceInstance;
	float volume_db;

	float pan_pullout;
	float time_pullout;
	float surround;

protected:
	static void _bind_methods();

public:
	Ref<AudioEffectInstance> instance();

	void set_pan_pullout(float p_amount);
	float get_pan_pullout() const;

	void set_time_pullout(float p_amount);
	float get_time_pullout() const;

	void set_surround(float p_amount);
	float get_surround() const;

	AudioEffectStereoEnhance();
};

#endif // AUDIO_EFFECT_STEREO_ENHANCE_H
