/*************************************************************************/
/*  visual_server_wrap_mt.cpp                                            */
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
#include "visual_server_wrap_mt.h"
#include "global_config.h"
#include "os/os.h"

void VisualServerWrapMT::thread_exit() {

	exit = true;
}

void VisualServerWrapMT::thread_draw() {

	draw_mutex->lock();

	draw_pending--;
	bool draw = (draw_pending == 0); // only draw when no more flushes are pending

	draw_mutex->unlock();

	if (draw) {

		visual_server->draw();
	}
}

void VisualServerWrapMT::thread_flush() {

	draw_mutex->lock();

	draw_pending--;

	draw_mutex->unlock();
}

void VisualServerWrapMT::_thread_callback(void *_instance) {

	VisualServerWrapMT *vsmt = reinterpret_cast<VisualServerWrapMT *>(_instance);

	vsmt->thread_loop();
}

void VisualServerWrapMT::thread_loop() {

	server_thread = Thread::get_caller_ID();

	OS::get_singleton()->make_rendering_thread();

	visual_server->init();

	exit = false;
	draw_thread_up = true;
	while (!exit) {
		// flush commands one by one, until exit is requested
		command_queue.wait_and_flush_one();
	}

	command_queue.flush_all(); // flush all

	visual_server->finish();
}

/* EVENT QUEUING */

void VisualServerWrapMT::sync() {

	if (create_thread) {

		/* TODO: sync with the thread */

		/*
		ERR_FAIL_COND(!draw_mutex);
		draw_mutex->lock();
		draw_pending++; //cambiar por un saferefcount
		draw_mutex->unlock();
		*/
		//command_queue.push( this, &VisualServerWrapMT::thread_flush);
	} else {

		command_queue.flush_all(); //flush all pending from other threads
	}
}

void VisualServerWrapMT::draw() {

	if (create_thread) {

		/* TODO: Make it draw
		ERR_FAIL_COND(!draw_mutex);
		draw_mutex->lock();
		draw_pending++; //cambiar por un saferefcount
		draw_mutex->unlock();

		command_queue.push( this, &VisualServerWrapMT::thread_draw);
		*/
	} else {

		visual_server->draw();
	}
}

void VisualServerWrapMT::init() {

	if (create_thread) {

		draw_mutex = Mutex::create();
		print_line("CREATING RENDER THREAD");
		OS::get_singleton()->release_rendering_thread();
		if (create_thread) {
			thread = Thread::create(_thread_callback, this);
			print_line("STARTING RENDER THREAD");
		}
		while (!draw_thread_up) {
			OS::get_singleton()->delay_usec(1000);
		}
		print_line("DONE RENDER THREAD");
	} else {

		visual_server->init();
	}
}

void VisualServerWrapMT::finish() {

	if (thread) {

		command_queue.push(this, &VisualServerWrapMT::thread_exit);
		Thread::wait_to_finish(thread);
		memdelete(thread);

		texture_free_cached_ids();
		//mesh_free_cached_ids();

		thread = NULL;
	} else {
		visual_server->finish();
	}

	if (draw_mutex)
		memdelete(draw_mutex);
}

VisualServerWrapMT::VisualServerWrapMT(VisualServer *p_contained, bool p_create_thread)
	: command_queue(p_create_thread) {

	visual_server = p_contained;
	create_thread = p_create_thread;
	thread = NULL;
	draw_mutex = NULL;
	draw_pending = 0;
	draw_thread_up = false;
	alloc_mutex = Mutex::create();
	pool_max_size = GLOBAL_DEF("memory/servers/thread_rid_prealloc_amount", 20);

	if (!p_create_thread) {
		server_thread = Thread::get_caller_ID();
	} else {
		server_thread = 0;
	}
}

VisualServerWrapMT::~VisualServerWrapMT() {

	memdelete(visual_server);
	memdelete(alloc_mutex);
	//finish();
}
