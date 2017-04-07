/*************************************************************************/
/*  command_queue_mt.cpp                                                 */
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
#include "command_queue_mt.h"

#include "os/os.h"

void CommandQueueMT::lock() {

	if (mutex)
		mutex->lock();
}

void CommandQueueMT::unlock() {

	if (mutex)
		mutex->unlock();
}

void CommandQueueMT::wait_for_flush() {

	// wait one millisecond for a flush to happen
	OS::get_singleton()->delay_usec(1000);
}

CommandQueueMT::SyncSemaphore *CommandQueueMT::_alloc_sync_sem() {

	int idx = -1;

	while (true) {

		for (int i = 0; i < SYNC_SEMAPHORES; i++) {

			if (!sync_sems[i].in_use) {
				sync_sems[i].in_use = true;
				idx = i;
				break;
			}
		}

		if (idx == -1) {
			wait_for_flush();
		} else {
			break;
		}
	}

	return &sync_sems[idx];
}

CommandQueueMT::CommandQueueMT(bool p_sync) {

	read_ptr = 0;
	write_ptr = 0;
	mutex = Mutex::create();

	for (int i = 0; i < SYNC_SEMAPHORES; i++) {

		sync_sems[i].sem = Semaphore::create();
		sync_sems[i].in_use = false;
	}
	if (p_sync)
		sync = Semaphore::create();
	else
		sync = NULL;
}

CommandQueueMT::~CommandQueueMT() {

	if (sync)
		memdelete(sync);
	memdelete(mutex);
	for (int i = 0; i < SYNC_SEMAPHORES; i++) {

		memdelete(sync_sems[i].sem);
	}
}
