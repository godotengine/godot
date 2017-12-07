/*
 * poll_windows: poll compatibility wrapper for Windows
 * Copyright © 2012-2013 RealVNC Ltd.
 * Copyright © 2009-2010 Pete Batard <pete@akeo.ie>
 * With contributions from Michael Plante, Orin Eman et al.
 * Parts of poll implementation from libusb-win32, by Stephan Meyer et al.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 *
 */

/*
 * poll() and pipe() Windows compatibility layer for libusb 1.0
 *
 * The way this layer works is by using OVERLAPPED with async I/O transfers, as
 * OVERLAPPED have an associated event which is flagged for I/O completion.
 *
 * For USB pollable async I/O, you would typically:
 * - obtain a Windows HANDLE to a file or device that has been opened in
 *   OVERLAPPED mode
 * - call usbi_create_fd with this handle to obtain a custom fd.
 *   Note that if you need simultaneous R/W access, you need to call create_fd
 *   twice, once in RW_READ and once in RW_WRITE mode to obtain 2 separate
 *   pollable fds
 * - leave the core functions call the poll routine and flag POLLIN/POLLOUT
 *
 * The pipe pollable synchronous I/O works using the overlapped event associated
 * with a fake pipe. The read/write functions are only meant to be used in that
 * context.
 */
#include <config.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "libusbi.h"

// Uncomment to debug the polling layer
//#define DEBUG_POLL_WINDOWS
#if defined(DEBUG_POLL_WINDOWS)
#define poll_dbg usbi_dbg
#else
// MSVC++ < 2005 cannot use a variadic argument and non MSVC
// compilers produce warnings if parenthesis are omitted.
#if defined(_MSC_VER) && (_MSC_VER < 1400)
#define poll_dbg
#else
#define poll_dbg(...)
#endif
#endif

#if defined(_PREFAST_)
#pragma warning(disable:28719)
#endif

#define CHECK_INIT_POLLING do {if(!is_polling_set) init_polling();} while(0)

// public fd data
const struct winfd INVALID_WINFD = {-1, INVALID_HANDLE_VALUE, NULL, NULL, NULL, RW_NONE};
struct winfd poll_fd[MAX_FDS];
// internal fd data
struct {
	CRITICAL_SECTION mutex; // lock for fds
	// Additional variables for XP CancelIoEx partial emulation
	HANDLE original_handle;
	DWORD thread_id;
} _poll_fd[MAX_FDS];

// globals
BOOLEAN is_polling_set = FALSE;
LONG pipe_number = 0;
static volatile LONG compat_spinlock = 0;

#if !defined(_WIN32_WCE)
// CancelIoEx, available on Vista and later only, provides the ability to cancel
// a single transfer (OVERLAPPED) when used. As it may not be part of any of the
// platform headers, we hook into the Kernel32 system DLL directly to seek it.
static BOOL (__stdcall *pCancelIoEx)(HANDLE, LPOVERLAPPED) = NULL;
#define Use_Duplicate_Handles (pCancelIoEx == NULL)

static inline void setup_cancel_io(void)
{
	HMODULE hKernel32 = GetModuleHandleA("KERNEL32");
	if (hKernel32 != NULL) {
		pCancelIoEx = (BOOL (__stdcall *)(HANDLE,LPOVERLAPPED))
			GetProcAddress(hKernel32, "CancelIoEx");
	}
	usbi_dbg("Will use CancelIo%s for I/O cancellation",
		Use_Duplicate_Handles?"":"Ex");
}

static inline BOOL cancel_io(int _index)
{
	if ((_index < 0) || (_index >= MAX_FDS)) {
		return FALSE;
	}

	if ( (poll_fd[_index].fd < 0) || (poll_fd[_index].handle == INVALID_HANDLE_VALUE)
	  || (poll_fd[_index].handle == 0) || (poll_fd[_index].overlapped == NULL) ) {
		return TRUE;
	}
	if (poll_fd[_index].itransfer && poll_fd[_index].cancel_fn) {
		// Cancel outstanding transfer via the specific callback
		(*poll_fd[_index].cancel_fn)(poll_fd[_index].itransfer);
		return TRUE;
	}
	if (pCancelIoEx != NULL) {
		return (*pCancelIoEx)(poll_fd[_index].handle, poll_fd[_index].overlapped);
	}
	if (_poll_fd[_index].thread_id == GetCurrentThreadId()) {
		return CancelIo(poll_fd[_index].handle);
	}
	usbi_warn(NULL, "Unable to cancel I/O that was started from another thread");
	return FALSE;
}
#else
#define Use_Duplicate_Handles FALSE

static __inline void setup_cancel_io()
{
	// No setup needed on WinCE
}

static __inline BOOL cancel_io(int _index)
{
	if ((_index < 0) || (_index >= MAX_FDS)) {
		return FALSE;
	}
	if ( (poll_fd[_index].fd < 0) || (poll_fd[_index].handle == INVALID_HANDLE_VALUE)
	  || (poll_fd[_index].handle == 0) || (poll_fd[_index].overlapped == NULL) ) {
		return TRUE;
	}
	if (poll_fd[_index].itransfer && poll_fd[_index].cancel_fn) {
		// Cancel outstanding transfer via the specific callback
		(*poll_fd[_index].cancel_fn)(poll_fd[_index].itransfer);
	}
	return TRUE;
}
#endif

// Init
void init_polling(void)
{
	int i;

	while (InterlockedExchange((LONG *)&compat_spinlock, 1) == 1) {
		SleepEx(0, TRUE);
	}
	if (!is_polling_set) {
		setup_cancel_io();
		for (i=0; i<MAX_FDS; i++) {
			poll_fd[i] = INVALID_WINFD;
			_poll_fd[i].original_handle = INVALID_HANDLE_VALUE;
			_poll_fd[i].thread_id = 0;
			InitializeCriticalSection(&_poll_fd[i].mutex);
		}
		is_polling_set = TRUE;
	}
	InterlockedExchange((LONG *)&compat_spinlock, 0);
}

// Internal function to retrieve the table index (and lock the fd mutex)
static int _fd_to_index_and_lock(int fd)
{
	int i;

	if (fd < 0)
		return -1;

	for (i=0; i<MAX_FDS; i++) {
		if (poll_fd[i].fd == fd) {
			EnterCriticalSection(&_poll_fd[i].mutex);
			// fd might have changed before we got to critical
			if (poll_fd[i].fd != fd) {
				LeaveCriticalSection(&_poll_fd[i].mutex);
				continue;
			}
			return i;
		}
	}
	return -1;
}

static OVERLAPPED *create_overlapped(void)
{
	OVERLAPPED *overlapped = (OVERLAPPED*) calloc(1, sizeof(OVERLAPPED));
	if (overlapped == NULL) {
		return NULL;
	}
	overlapped->hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	if(overlapped->hEvent == NULL) {
		free (overlapped);
		return NULL;
	}
	return overlapped;
}

static void free_overlapped(OVERLAPPED *overlapped)
{
	if (overlapped == NULL)
		return;

	if ( (overlapped->hEvent != 0)
	  && (overlapped->hEvent != INVALID_HANDLE_VALUE) ) {
		CloseHandle(overlapped->hEvent);
	}
	free(overlapped);
}

void exit_polling(void)
{
	int i;

	while (InterlockedExchange((LONG *)&compat_spinlock, 1) == 1) {
		SleepEx(0, TRUE);
	}
	if (is_polling_set) {
		is_polling_set = FALSE;

		for (i=0; i<MAX_FDS; i++) {
			// Cancel any async I/O (handle can be invalid)
			cancel_io(i);
			// If anything was pending on that I/O, it should be
			// terminating, and we should be able to access the fd
			// mutex lock before too long
			EnterCriticalSection(&_poll_fd[i].mutex);
			free_overlapped(poll_fd[i].overlapped);
			if (Use_Duplicate_Handles) {
				// Close duplicate handle
				if (_poll_fd[i].original_handle != INVALID_HANDLE_VALUE) {
					CloseHandle(poll_fd[i].handle);
				}
			}
			poll_fd[i] = INVALID_WINFD;
			LeaveCriticalSection(&_poll_fd[i].mutex);
			DeleteCriticalSection(&_poll_fd[i].mutex);
		}
	}
	InterlockedExchange((LONG *)&compat_spinlock, 0);
}

/*
 * Create a fake pipe.
 * As libusb only uses pipes for signaling, all we need from a pipe is an
 * event. To that extent, we create a single wfd and overlapped as a means
 * to access that event.
 */
int usbi_pipe(int filedes[2])
{
	int i;
	OVERLAPPED* overlapped;

	CHECK_INIT_POLLING;

	overlapped = create_overlapped();

	if (overlapped == NULL) {
		return -1;
	}
	// The overlapped must have status pending for signaling to work in poll
	overlapped->Internal = STATUS_PENDING;
	overlapped->InternalHigh = 0;

	for (i=0; i<MAX_FDS; i++) {
		if (poll_fd[i].fd < 0) {
			EnterCriticalSection(&_poll_fd[i].mutex);
			// fd might have been allocated before we got to critical
			if (poll_fd[i].fd >= 0) {
				LeaveCriticalSection(&_poll_fd[i].mutex);
				continue;
			}

			// Use index as the unique fd number
			poll_fd[i].fd = i;
			// Read end of the "pipe"
			filedes[0] = poll_fd[i].fd;
			// We can use the same handle for both ends
			filedes[1] = filedes[0];

			poll_fd[i].handle = DUMMY_HANDLE;
			poll_fd[i].overlapped = overlapped;
			// There's no polling on the write end, so we just use READ for our needs
			poll_fd[i].rw = RW_READ;
			_poll_fd[i].original_handle = INVALID_HANDLE_VALUE;
			LeaveCriticalSection(&_poll_fd[i].mutex);
			return 0;
		}
	}
	free_overlapped(overlapped);
	return -1;
}

/*
 * Create both an fd and an OVERLAPPED from an open Windows handle, so that
 * it can be used with our polling function
 * The handle MUST support overlapped transfers (usually requires CreateFile
 * with FILE_FLAG_OVERLAPPED)
 * Return a pollable file descriptor struct, or INVALID_WINFD on error
 *
 * Note that the fd returned by this function is a per-transfer fd, rather
 * than a per-session fd and cannot be used for anything else but our
 * custom functions (the fd itself points to the NUL: device)
 * if you plan to do R/W on the same handle, you MUST create 2 fds: one for
 * read and one for write. Using a single R/W fd is unsupported and will
 * produce unexpected results
 */
struct winfd usbi_create_fd(HANDLE handle, int access_mode, struct usbi_transfer *itransfer, cancel_transfer *cancel_fn)
{
	int i;
	struct winfd wfd = INVALID_WINFD;
	OVERLAPPED* overlapped = NULL;

	CHECK_INIT_POLLING;

	if ((handle == 0) || (handle == INVALID_HANDLE_VALUE)) {
		return INVALID_WINFD;
	}

	wfd.itransfer = itransfer;
	wfd.cancel_fn = cancel_fn;

	if ((access_mode != RW_READ) && (access_mode != RW_WRITE)) {
		usbi_warn(NULL, "only one of RW_READ or RW_WRITE are supported. "
			"If you want to poll for R/W simultaneously, create multiple fds from the same handle.");
		return INVALID_WINFD;
	}
	if (access_mode == RW_READ) {
		wfd.rw = RW_READ;
	} else {
		wfd.rw = RW_WRITE;
	}

	overlapped = create_overlapped();
	if(overlapped == NULL) {
		return INVALID_WINFD;
	}

	for (i=0; i<MAX_FDS; i++) {
		if (poll_fd[i].fd < 0) {
			EnterCriticalSection(&_poll_fd[i].mutex);
			// fd might have been removed before we got to critical
			if (poll_fd[i].fd >= 0) {
				LeaveCriticalSection(&_poll_fd[i].mutex);
				continue;
			}
			// Use index as the unique fd number
			wfd.fd = i;
			// Attempt to emulate some of the CancelIoEx behaviour on platforms
			// that don't have it
			if (Use_Duplicate_Handles) {
				_poll_fd[i].thread_id = GetCurrentThreadId();
				if (!DuplicateHandle(GetCurrentProcess(), handle, GetCurrentProcess(),
					&wfd.handle, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
					usbi_dbg("could not duplicate handle for CancelIo - using original one");
					wfd.handle = handle;
					// Make sure we won't close the original handle on fd deletion then
					_poll_fd[i].original_handle = INVALID_HANDLE_VALUE;
				} else {
					_poll_fd[i].original_handle = handle;
				}
			} else {
				wfd.handle = handle;
			}
			wfd.overlapped = overlapped;
			memcpy(&poll_fd[i], &wfd, sizeof(struct winfd));
			LeaveCriticalSection(&_poll_fd[i].mutex);
			return wfd;
		}
	}
	free_overlapped(overlapped);
	return INVALID_WINFD;
}

static void _free_index(int _index)
{
	// Cancel any async IO (Don't care about the validity of our handles for this)
	cancel_io(_index);
	// close the duplicate handle (if we have an actual duplicate)
	if (Use_Duplicate_Handles) {
		if (_poll_fd[_index].original_handle != INVALID_HANDLE_VALUE) {
			CloseHandle(poll_fd[_index].handle);
		}
		_poll_fd[_index].original_handle = INVALID_HANDLE_VALUE;
		_poll_fd[_index].thread_id = 0;
	}
	free_overlapped(poll_fd[_index].overlapped);
	poll_fd[_index] = INVALID_WINFD;
}

/*
 * Release a pollable file descriptor.
 *
 * Note that the associated Windows handle is not closed by this call
 */
void usbi_free_fd(struct winfd *wfd)
{
	int _index;

	CHECK_INIT_POLLING;

	_index = _fd_to_index_and_lock(wfd->fd);
	if (_index < 0) {
		return;
	}
	_free_index(_index);
	*wfd = INVALID_WINFD;
	LeaveCriticalSection(&_poll_fd[_index].mutex);
}

/*
 * The functions below perform various conversions between fd, handle and OVERLAPPED
 */
struct winfd fd_to_winfd(int fd)
{
	int i;
	struct winfd wfd;

	CHECK_INIT_POLLING;

	if (fd < 0)
		return INVALID_WINFD;

	for (i=0; i<MAX_FDS; i++) {
		if (poll_fd[i].fd == fd) {
			EnterCriticalSection(&_poll_fd[i].mutex);
			// fd might have been deleted before we got to critical
			if (poll_fd[i].fd != fd) {
				LeaveCriticalSection(&_poll_fd[i].mutex);
				continue;
			}
			memcpy(&wfd, &poll_fd[i], sizeof(struct winfd));
			LeaveCriticalSection(&_poll_fd[i].mutex);
			return wfd;
		}
	}
	return INVALID_WINFD;
}

struct winfd handle_to_winfd(HANDLE handle)
{
	int i;
	struct winfd wfd;

	CHECK_INIT_POLLING;

	if ((handle == 0) || (handle == INVALID_HANDLE_VALUE))
		return INVALID_WINFD;

	for (i=0; i<MAX_FDS; i++) {
		if (poll_fd[i].handle == handle) {
			EnterCriticalSection(&_poll_fd[i].mutex);
			// fd might have been deleted before we got to critical
			if (poll_fd[i].handle != handle) {
				LeaveCriticalSection(&_poll_fd[i].mutex);
				continue;
			}
			memcpy(&wfd, &poll_fd[i], sizeof(struct winfd));
			LeaveCriticalSection(&_poll_fd[i].mutex);
			return wfd;
		}
	}
	return INVALID_WINFD;
}

struct winfd overlapped_to_winfd(OVERLAPPED* overlapped)
{
	int i;
	struct winfd wfd;

	CHECK_INIT_POLLING;

	if (overlapped == NULL)
		return INVALID_WINFD;

	for (i=0; i<MAX_FDS; i++) {
		if (poll_fd[i].overlapped == overlapped) {
			EnterCriticalSection(&_poll_fd[i].mutex);
			// fd might have been deleted before we got to critical
			if (poll_fd[i].overlapped != overlapped) {
				LeaveCriticalSection(&_poll_fd[i].mutex);
				continue;
			}
			memcpy(&wfd, &poll_fd[i], sizeof(struct winfd));
			LeaveCriticalSection(&_poll_fd[i].mutex);
			return wfd;
		}
	}
	return INVALID_WINFD;
}

/*
 * POSIX poll equivalent, using Windows OVERLAPPED
 * Currently, this function only accepts one of POLLIN or POLLOUT per fd
 * (but you can create multiple fds from the same handle for read and write)
 */
int usbi_poll(struct pollfd *fds, unsigned int nfds, int timeout)
{
	unsigned i;
	int _index, object_index, triggered;
	HANDLE *handles_to_wait_on;
	int *handle_to_index;
	DWORD nb_handles_to_wait_on = 0;
	DWORD ret;

	CHECK_INIT_POLLING;

	triggered = 0;
	handles_to_wait_on = (HANDLE*) calloc(nfds+1, sizeof(HANDLE));	// +1 for fd_update
	handle_to_index = (int*) calloc(nfds, sizeof(int));
	if ((handles_to_wait_on == NULL) || (handle_to_index == NULL)) {
		errno = ENOMEM;
		triggered = -1;
		goto poll_exit;
	}

	for (i = 0; i < nfds; ++i) {
		fds[i].revents = 0;

		// Only one of POLLIN or POLLOUT can be selected with this version of poll (not both)
		if ((fds[i].events & ~POLLIN) && (!(fds[i].events & POLLOUT))) {
			fds[i].revents |= POLLERR;
			errno = EACCES;
			usbi_warn(NULL, "unsupported set of events");
			triggered = -1;
			goto poll_exit;
		}

		_index = _fd_to_index_and_lock(fds[i].fd);
		poll_dbg("fd[%d]=%d: (overlapped=%p) got events %04X", i, poll_fd[_index].fd, poll_fd[_index].overlapped, fds[i].events);

		if ( (_index < 0) || (poll_fd[_index].handle == INVALID_HANDLE_VALUE)
		  || (poll_fd[_index].handle == 0) || (poll_fd[_index].overlapped == NULL)) {
			fds[i].revents |= POLLNVAL | POLLERR;
			errno = EBADF;
			if (_index >= 0) {
				LeaveCriticalSection(&_poll_fd[_index].mutex);
			}
			usbi_warn(NULL, "invalid fd");
			triggered = -1;
			goto poll_exit;
		}

		// IN or OUT must match our fd direction
		if ((fds[i].events & POLLIN) && (poll_fd[_index].rw != RW_READ)) {
			fds[i].revents |= POLLNVAL | POLLERR;
			errno = EBADF;
			usbi_warn(NULL, "attempted POLLIN on fd without READ access");
			LeaveCriticalSection(&_poll_fd[_index].mutex);
			triggered = -1;
			goto poll_exit;
		}

		if ((fds[i].events & POLLOUT) && (poll_fd[_index].rw != RW_WRITE)) {
			fds[i].revents |= POLLNVAL | POLLERR;
			errno = EBADF;
			usbi_warn(NULL, "attempted POLLOUT on fd without WRITE access");
			LeaveCriticalSection(&_poll_fd[_index].mutex);
			triggered = -1;
			goto poll_exit;
		}

		// The following macro only works if overlapped I/O was reported pending
		if ( (HasOverlappedIoCompleted(poll_fd[_index].overlapped))
		  || (HasOverlappedIoCompletedSync(poll_fd[_index].overlapped)) ) {
			poll_dbg("  completed");
			// checks above should ensure this works:
			fds[i].revents = fds[i].events;
			triggered++;
		} else {
			handles_to_wait_on[nb_handles_to_wait_on] = poll_fd[_index].overlapped->hEvent;
			handle_to_index[nb_handles_to_wait_on] = i;
			nb_handles_to_wait_on++;
		}
		LeaveCriticalSection(&_poll_fd[_index].mutex);
	}

	// If nothing was triggered, wait on all fds that require it
	if ((timeout != 0) && (triggered == 0) && (nb_handles_to_wait_on != 0)) {
		if (timeout < 0) {
			poll_dbg("starting infinite wait for %u handles...", (unsigned int)nb_handles_to_wait_on);
		} else {
			poll_dbg("starting %d ms wait for %u handles...", timeout, (unsigned int)nb_handles_to_wait_on);
		}
		ret = WaitForMultipleObjects(nb_handles_to_wait_on, handles_to_wait_on,
			FALSE, (timeout<0)?INFINITE:(DWORD)timeout);
		object_index = ret-WAIT_OBJECT_0;
		if ((object_index >= 0) && ((DWORD)object_index < nb_handles_to_wait_on)) {
			poll_dbg("  completed after wait");
			i = handle_to_index[object_index];
			_index = _fd_to_index_and_lock(fds[i].fd);
			fds[i].revents = fds[i].events;
			triggered++;
			if (_index >= 0) {
				LeaveCriticalSection(&_poll_fd[_index].mutex);
			}
		} else if (ret == WAIT_TIMEOUT) {
			poll_dbg("  timed out");
			triggered = 0;	// 0 = timeout
		} else {
			errno = EIO;
			triggered = -1;	// error
		}
	}

poll_exit:
	if (handles_to_wait_on != NULL) {
		free(handles_to_wait_on);
	}
	if (handle_to_index != NULL) {
		free(handle_to_index);
	}
	return triggered;
}

/*
 * close a fake pipe fd
 */
int usbi_close(int fd)
{
	int _index;
	int r = -1;

	CHECK_INIT_POLLING;

	_index = _fd_to_index_and_lock(fd);

	if (_index < 0) {
		errno = EBADF;
	} else {
		free_overlapped(poll_fd[_index].overlapped);
		poll_fd[_index] = INVALID_WINFD;
		LeaveCriticalSection(&_poll_fd[_index].mutex);
	}
	return r;
}

/*
 * synchronous write for fake "pipe" signaling
 */
ssize_t usbi_write(int fd, const void *buf, size_t count)
{
	int _index;
	UNUSED(buf);

	CHECK_INIT_POLLING;

	if (count != sizeof(unsigned char)) {
		usbi_err(NULL, "this function should only used for signaling");
		return -1;
	}

	_index = _fd_to_index_and_lock(fd);

	if ( (_index < 0) || (poll_fd[_index].overlapped == NULL) ) {
		errno = EBADF;
		if (_index >= 0) {
			LeaveCriticalSection(&_poll_fd[_index].mutex);
		}
		return -1;
	}

	poll_dbg("set pipe event (fd = %d, thread = %08X)", _index, (unsigned int)GetCurrentThreadId());
	SetEvent(poll_fd[_index].overlapped->hEvent);
	poll_fd[_index].overlapped->Internal = STATUS_WAIT_0;
	// If two threads write on the pipe at the same time, we need to
	// process two separate reads => use the overlapped as a counter
	poll_fd[_index].overlapped->InternalHigh++;

	LeaveCriticalSection(&_poll_fd[_index].mutex);
	return sizeof(unsigned char);
}

/*
 * synchronous read for fake "pipe" signaling
 */
ssize_t usbi_read(int fd, void *buf, size_t count)
{
	int _index;
	ssize_t r = -1;
	UNUSED(buf);

	CHECK_INIT_POLLING;

	if (count != sizeof(unsigned char)) {
		usbi_err(NULL, "this function should only used for signaling");
		return -1;
	}

	_index = _fd_to_index_and_lock(fd);

	if (_index < 0) {
		errno = EBADF;
		return -1;
	}

	if (WaitForSingleObject(poll_fd[_index].overlapped->hEvent, INFINITE) != WAIT_OBJECT_0) {
		usbi_warn(NULL, "waiting for event failed: %u", (unsigned int)GetLastError());
		errno = EIO;
		goto out;
	}

	poll_dbg("clr pipe event (fd = %d, thread = %08X)", _index, (unsigned int)GetCurrentThreadId());
	poll_fd[_index].overlapped->InternalHigh--;
	// Don't reset unless we don't have any more events to process
	if (poll_fd[_index].overlapped->InternalHigh <= 0) {
		ResetEvent(poll_fd[_index].overlapped->hEvent);
		poll_fd[_index].overlapped->Internal = STATUS_PENDING;
	}

	r = sizeof(unsigned char);

out:
	LeaveCriticalSection(&_poll_fd[_index].mutex);
	return r;
}
