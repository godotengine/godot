/*************************************************************************/
/*  camera_x11.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CAMERAX11_H
#define CAMERAX11_H

#include <linux/videodev2.h>
#include <stdint.h>
#include <string>
#include <thread>
#include <vector>

#include "servers/camera/camera_feed.h"
#include "servers/camera_server.h"

enum IOType {
	TYPE_IO_NONE = 0, // not usable
	TYPE_IO_MMAP = 1, // using mmap buffers
	TYPE_IO_USRPTR = 2, // use other buffers
	TYPE_IO_READ = 3 // using read call
};

// struct that store the used v4l2 functions
// either from v4l2 direct or from libv4l2
struct v4l2_funcs {
	int (*open)(const char *file, int oflag, ...);
	int (*close)(int fd);
	int (*dup)(int fd);
	int (*ioctl)(int fd, unsigned long int request, ...);
	long int (*read)(int fd, void *buffer, size_t n);
	void *(*mmap)(void *start, size_t length, int prot, int flags, int fd, int64_t offset);
	int (*munmap)(void *_start, size_t length);
	bool libv4l2;
};

class V4l2_Device {
private:
	// buffers for mmap
	struct buffer {
		void *start;
		size_t length;
	} * buffers;
	unsigned int n_buffers;
	struct v4l2_buffer buf;
	struct v4l2_capability cap;
	struct v4l2_format fmt;
	// only for read and userp
	unsigned int buffer_size;

	// the file descriptor
	int fd = -1;
	// the v4l2 functions (either libv4l2 or normal v4l2)
	struct v4l2_funcs *funcs;
	// the image data
	Vector<uint8_t> img_data;
	// whether device is initialized
	// access type and used pixelformat
	IOType type = TYPE_IO_NONE;

	// Thread that set image to the feed
	std::thread stream_thread;
	// and the function for the thread
	void stream_image(Ref<CameraFeed> feed);
	void get_image(Ref<CameraFeed> feed, uint8_t *buffer);

	// ioctl with some signal tolerance
	int xioctl(int fd, unsigned long int request, void *arg);
	bool buffer_available = false;

public:
	bool use_libv4l2;
	bool opened = false;

	bool streaming = false;
	std::string dev_name;
	String name;

	unsigned int width = 0;
	unsigned int height = 0;

	V4l2_Device(std::string dev_name, struct v4l2_funcs *funcs);
	~V4l2_Device();

	bool check_device(bool print_debug = false);
	bool close();

	bool request_buffers();
	void cleanup_buffers();

	bool start_streaming(Ref<CameraFeed> feed);
	void stop_streaming();
};

class CameraFeedX11 : public CameraFeed {
private:
	V4l2_Device *device;

public:
	V4l2_Device *get_device() const;

	CameraFeedX11();
	~CameraFeedX11();

	void set_device(V4l2_Device *p_device);

	bool activate_feed();
	void deactivate_feed();
};

class CameraX11 : public CameraServer {
private:
	struct v4l2_funcs funcs;
	void *libv4l2;
	bool alive = false;
	std::thread hotplug_thread;
	void check_change();

public:
	CameraX11();
	~CameraX11();

	void update_feeds();
};

#endif /* CAMERAX11_H */
