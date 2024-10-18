/**************************************************************************/
/*  util.cpp                                                              */
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

#include "wayland/wayland_thread.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// Read the content pointed by fd into a Vector<uint8_t>.
Vector<uint8_t> WaylandThread::_read_fd(int fd) {
	// This is pretty much an arbitrary size.
	uint32_t chunk_size = 2048;

	LocalVector<uint8_t> data;
	data.resize(chunk_size);

	uint32_t bytes_read = 0;

	while (true) {
		ssize_t last_bytes_read = read(fd, data.ptr() + bytes_read, chunk_size);
		if (last_bytes_read < 0) {
			ERR_PRINT(vformat("Read error %d.", errno));

			data.clear();
			break;
		}

		if (last_bytes_read == 0) {
			// We're done, we've reached the EOF.
			DEBUG_LOG_WAYLAND_THREAD(vformat("Done reading %d bytes.", bytes_read));

			close(fd);

			data.resize(bytes_read);
			break;
		}

		DEBUG_LOG_WAYLAND_THREAD(vformat("Read chunk of %d bytes.", last_bytes_read));

		bytes_read += last_bytes_read;

		// Increase the buffer size by one chunk in preparation of the next read.
		data.resize(bytes_read + chunk_size);
	}

	return data;
}

// Based on the wayland book's shared memory boilerplate (PD/CC0).
// See: https://wayland-book.com/surfaces/shared-memory.html
int WaylandThread::_allocate_shm_file(size_t size) {
	int retries = 100;

	do {
		// Generate a random name.
		char name[] = "/wl_shm-godot-XXXXXX";
		for (long unsigned int i = sizeof(name) - 7; i < sizeof(name) - 1; i++) {
			name[i] = Math::random('A', 'Z');
		}

		// Try to open a shared memory object with that name.
		int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fd >= 0) {
			// Success, unlink its name as we just need the file descriptor.
			shm_unlink(name);

			// Resize the file to the requested length.
			int ret;
			do {
				ret = ftruncate(fd, size);
			} while (ret < 0 && errno == EINTR);

			if (ret < 0) {
				close(fd);
				return -1;
			}

			return fd;
		}

		retries--;
	} while (retries > 0 && errno == EEXIST);

	return -1;
}

// Return the content of a wl_data_offer.
Vector<uint8_t> WaylandThread::_wl_data_offer_read(struct wl_display *p_display, const char *p_mime, struct wl_data_offer *p_offer) {
	if (!p_offer) {
		return Vector<uint8_t>();
	}

	int fds[2];
	if (pipe(fds) == 0) {
		wl_data_offer_receive(p_offer, p_mime, fds[1]);

		// Let the compositor know about the pipe.
		// NOTE: It's important to just flush and not roundtrip here as we would risk
		// running some cleanup event, like for example `wl_data_device::leave`. We're
		// going to wait for the message anyways as the read will probably block if
		// the compositor doesn't read from the other end of the pipe.
		wl_display_flush(p_display);

		// Close the write end of the pipe, which we don't need and would otherwise
		// just stall our next `read`s.
		close(fds[1]);

		return _read_fd(fds[0]);
	}

	return Vector<uint8_t>();
}

// Read the content of a wp_primary_selection_offer.
Vector<uint8_t> WaylandThread::_wp_primary_selection_offer_read(struct wl_display *p_display, const char *p_mime, struct zwp_primary_selection_offer_v1 *p_offer) {
	if (!p_offer) {
		return Vector<uint8_t>();
	}

	int fds[2];
	if (pipe(fds) == 0) {
		zwp_primary_selection_offer_v1_receive(p_offer, p_mime, fds[1]);

		// NOTE: It's important to just flush and not roundtrip here as we would risk
		// running some cleanup event, like for example `wl_data_device::leave`. We're
		// going to wait for the message anyways as the read will probably block if
		// the compositor doesn't read from the other end of the pipe.
		wl_display_flush(p_display);

		// Close the write end of the pipe, which we don't need and would otherwise
		// just stall our next `read`s.
		close(fds[1]);

		return _read_fd(fds[0]);
	}

	return Vector<uint8_t>();
}
