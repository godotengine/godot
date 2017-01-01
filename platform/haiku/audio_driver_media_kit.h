/*************************************************************************/
/*  audio_driver_media_kit.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "servers/audio/audio_server_sw.h"

#ifdef MEDIA_KIT_ENABLED

#include "core/os/thread.h"
#include "core/os/mutex.h"

#include <kernel/image.h> // needed for image_id
#include <SoundPlayer.h>

class AudioDriverMediaKit : public AudioDriverSW {
	Mutex* mutex;

	BSoundPlayer* player;
	static int32_t* samples_in;

	static void PlayBuffer(void* cookie, void* buffer, size_t size, const media_raw_audio_format& format);

	unsigned int mix_rate;
	OutputFormat output_format;
	unsigned int buffer_size;
	int channels;

	bool active;

public:

	const char* get_name() const {
		return "MediaKit";
	};

	virtual Error init();
	virtual void start();
	virtual int get_mix_rate() const;
	virtual OutputFormat get_output_format() const;
	virtual void lock();
	virtual void unlock();
	virtual void finish();

	AudioDriverMediaKit();
	~AudioDriverMediaKit();
};

#endif
