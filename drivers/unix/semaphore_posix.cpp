/*************************************************************************/
/*  semaphore_posix.cpp                                                  */
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
#include "semaphore_posix.h"

#if defined(UNIX_ENABLED) || defined(PTHREAD_ENABLED)

#include "os/memory.h"
#include <errno.h>
#include <stdio.h>

Error SemaphorePosix::wait() {

	while (sem_wait(&sem)) {
		if (errno == EINTR) {
			errno = 0;
			continue;
		} else {
			perror("sem waiting");
			return ERR_BUSY;
		}
	}
	return OK;
}

Error SemaphorePosix::post() {

	return (sem_post(&sem) == 0) ? OK : ERR_BUSY;
}
int SemaphorePosix::get() const {

	int val;
	sem_getvalue(&sem, &val);

	return val;
}

Semaphore *SemaphorePosix::create_semaphore_posix() {

	return memnew(SemaphorePosix);
}

void SemaphorePosix::make_default() {

	create_func = create_semaphore_posix;
}

SemaphorePosix::SemaphorePosix() {

	int r = sem_init(&sem, 0, 0);
	if (r != 0)
		perror("sem creating");
}

SemaphorePosix::~SemaphorePosix() {

	sem_destroy(&sem);
}

#endif
