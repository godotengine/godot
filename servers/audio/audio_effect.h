/**************************************************************************/
/*  audio_effect.h                                                        */
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

#ifndef AUDIO_EFFECT_H
#define AUDIO_EFFECT_H

#include "core/io/resource.h"
#include "core/math/audio_frame.h"
#include "core/object/gdvirtual.gen.inc"
#include "core/variant/native_ptr.h"

class AudioEffectInstance : public RefCounted {
	GDCLASS(AudioEffectInstance, RefCounted);

protected:
	GDVIRTUAL3_REQUIRED(_process, GDExtensionConstPtr<AudioFrame>, GDExtensionPtr<AudioFrame>, int)
	GDVIRTUAL0RC(bool, _process_silence)
	static void _bind_methods();

public:
	virtual void process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count);
	virtual bool process_silence() const;
};

class AudioEffect : public Resource {
	GDCLASS(AudioEffect, Resource);

protected:
	GDVIRTUAL0R_REQUIRED(Ref<AudioEffectInstance>, _instantiate)
	static void _bind_methods();

public:
	virtual Ref<AudioEffectInstance> instantiate();
	AudioEffect();
};

#endif // AUDIO_EFFECT_H
