/*************************************************************************/
/*  test_thread_array_processor.h                                        */
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

#ifndef TEST_THREAD_ARRAY_PROCESSOR_H
#define TEST_THREAD_ARRAY_PROCESSOR_H

#include <thirdparty/doctest/doctest.h>

#include "core/engine.h"
#include "core/os/threaded_array_processor.h"

class WaitTasks {
	void _process(uint32_t index, uint32_t *_create) {
        OS::get_singleton()->delay_usec(1000);
	}

public:
	void process() {
		int32_t count = 10000;
		Vector<uint32_t> numbers;
		numbers.resize(count);
		thread_process_array(count, this, &WaitTasks::_process, numbers.ptrw());
	}
};

TEST_CASE("[ThreadedProcessArray] Process") {
	WaitTasks random_numbers;

	OS::get_singleton()->print("\n\nTest 1: ThreadedProcessArray process\n");
	uint32_t time = OS::get_singleton()->get_system_time_msecs();
	random_numbers.process();
	time = OS::get_singleton()->get_system_time_msecs() - time;
	OS::get_singleton()->print("\tTime taken in: %d\n", time);
}

#endif
