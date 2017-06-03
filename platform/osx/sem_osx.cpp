/*************************************************************************/
/*  sem_osx.cpp                                                          */
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
#include "sem_osx.h"

#include <fcntl.h>
#include <unistd.h>

void cgsem_init(cgsem_t *cgsem) {
	int flags, fd, i;

	pipe(cgsem->pipefd);

	/* Make the pipes FD_CLOEXEC to allow them to close should we call
	 * execv on restart. */
	for (i = 0; i < 2; i++) {
		fd = cgsem->pipefd[i];
		flags = fcntl(fd, F_GETFD, 0);
		flags |= FD_CLOEXEC;
		fcntl(fd, F_SETFD, flags);
	}
}

void cgsem_post(cgsem_t *cgsem) {
	const char buf = 1;

	write(cgsem->pipefd[1], &buf, 1);
}

void cgsem_wait(cgsem_t *cgsem) {
	char buf;

	read(cgsem->pipefd[0], &buf, 1);
}

void cgsem_destroy(cgsem_t *cgsem) {
	close(cgsem->pipefd[1]);
	close(cgsem->pipefd[0]);
}

#include "os/memory.h"
#include <errno.h>

Error SemaphoreOSX::wait() {

	cgsem_wait(&sem);
	return OK;
}

Error SemaphoreOSX::post() {

	cgsem_post(&sem);

	return OK;
}
int SemaphoreOSX::get() const {

	return 0;
}

Semaphore *SemaphoreOSX::create_semaphore_osx() {

	return memnew(SemaphoreOSX);
}

void SemaphoreOSX::make_default() {

	create_func = create_semaphore_osx;
}

SemaphoreOSX::SemaphoreOSX() {

	cgsem_init(&sem);
}

SemaphoreOSX::~SemaphoreOSX() {

	cgsem_destroy(&sem);
}
