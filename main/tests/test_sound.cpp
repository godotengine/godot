/*************************************************************************/
/*  test_sound.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "test_sound.h"
#include "math_funcs.h"
#include "os/main_loop.h"
#include "servers/visual_server.h"

#include "io/resource_loader.h"
#include "os/os.h"
#include "print_string.h"
#include "servers/audio_server.h"

#if 0
namespace TestSound {


class TestMainLoop : public MainLoop {

	bool quit;
	Ref<Sample> sample;

public:
	virtual void input_event(const InputEvent& p_event) {


	}
	virtual void request_quit() {

		quit=true;
	}

	virtual void init() {

		List<String> cmdline = OS::get_singleton()->get_cmdline_args();
		quit=false;
		if (cmdline.size()) {

			sample=ResourceLoader::load(cmdline.back()->get());
			ERR_FAIL_COND(sample.is_null());
			print_line("Sample loaded OK");
		}

		RID voice = AudioServer::get_singleton()->voice_create();
		AudioServer::get_singleton()->voice_play( voice, sample->get_rid() );


	}

	virtual bool idle(float p_time) {
		return false;
	}


	virtual bool iteration(float p_time) {

		return quit;
	}
	virtual void finish() {

	}

};


MainLoop* test() {

	return memnew( TestMainLoop );

}

}
#endif
