/*************************************************************************/
/*  audio_driver_nacl.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef AUDIO_DRIVER_NACL_H
#define AUDIO_DRIVER_NACL_H

#include "servers/audio/audio_server_sw.h"

#include "ppapi/cpp/audio.h"

class AudioDriverNacl : public AudioDriverSW {

	static void output_callback(void* samples, uint32_t buffer_size, void* data);

	int32_t* samples_in;
	int sample_frame_count_;
	int sample_count;
	pp::Audio audio_;

public:

	virtual const char* get_name() const;

	virtual Error init();
	virtual void start();
	virtual int get_mix_rate() const ;
	virtual OutputFormat get_output_format() const;
	virtual void lock();
	virtual void unlock();
	virtual void finish();


	AudioDriverNacl();
	~AudioDriverNacl();
};

#endif // AUDIO_DRIVER_NACL_H

