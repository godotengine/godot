/*************************************************************************/
/*  camera_x11.cpp                                                       */
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

#include "camera_x11.h"
#include "servers/camera/camera_feed.h"

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <linux/videodev2.h>

#define CLEAR(x) memset(&(x), 0, sizeof(x))

// formats that can be used
// mainly necessary if libv4l2 is not available
std::vector<__u32> supported_formats{
	V4L2_PIX_FMT_RGB24
};

int V4l2_Device::xioctl(int fd, unsigned long int request, void *arg) {
	int r;
	do {
		r = funcs->ioctl(fd, request, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r;
};

V4l2_Device::V4l2_Device(std::string dev_name, struct v4l2_funcs *funcs) {
	this->dev_name = dev_name;
	this->funcs = funcs;
	this->use_libv4l2 = funcs->libv4l2;
};

bool V4l2_Device::check_device(bool print_debug) {
	// print_debug is used whether debug messages should be printed
	// (as the devices are checked every second, which resulted in
	// many useless print outs)
	int fd = -1;
	struct stat st;
	if (stat(dev_name.c_str(), &st) == -1 || (!S_ISCHR(st.st_mode))) {
#ifdef DEBUG_ENABLED
		if (print_debug)
			print_line("Device name " + String(dev_name.c_str()) + " not available.");
#endif
		return false;
	}

	// open device
	fd = funcs->open(dev_name.c_str(), O_RDWR | O_NONBLOCK, 0);

	if (fd == -1) {
#ifdef DEBUG_ENABLED
		if (print_debug)
			print_line("Cannot open device " + String(dev_name.c_str()) + ".");
#endif
		return false;
	}

	CLEAR(cap);
	if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
#ifdef DEBUG_ENABLED
		if (print_debug)
			print_line("Cannot query capabilities for " + String(dev_name.c_str()) + ".");
#endif
		funcs->close(fd);
		return false;
	}

	name = String((const char *)cap.card) + String(" (") + String(dev_name.c_str()) + String(")");

	// Check if it is a videocapture device
	if ((cap.device_caps & V4L2_CAP_VIDEO_CAPTURE) != V4L2_CAP_VIDEO_CAPTURE) {
#ifdef DEBUG_ENABLED
		if (print_debug)
			print_line(String(dev_name.c_str()) + " is no video capture device.");
#endif
		funcs->close(fd);
		return false;
	}

	// if its just a meta data device (unused)
	if (((cap.device_caps & V4L2_CAP_META_OUTPUT) == V4L2_CAP_META_OUTPUT) || ((cap.device_caps & V4L2_CAP_META_CAPTURE) == V4L2_CAP_META_CAPTURE)) {
#ifdef DEBUG_ENABLED
		if (print_debug)
			print_line(String(dev_name.c_str()) + " is just a meta information device.");
#endif
		funcs->close(fd);
		return false;
	}

	// How the image data can be accessed
	if ((cap.capabilities & V4L2_CAP_STREAMING) == V4L2_CAP_STREAMING) {
		type = TYPE_IO_MMAP;
	} else if ((cap.capabilities & V4L2_CAP_READWRITE) == V4L2_CAP_READWRITE) {
		type = TYPE_IO_READ;
	} else {
#ifdef DEBUG_ENABLED
		if (print_debug)
			print_line(String(dev_name.c_str()) + " has no capability to capture frames.");
#endif
		funcs->close(fd);
		return false;
	}

	// Check if device supports supported formats
	CLEAR(fmt);
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	// Get default size and format
	if (xioctl(fd, VIDIOC_G_FMT, &fmt) == -1) {
#ifdef DEBUG_ENABLED
		if (print_debug)
			print_line("Cannot grab default format from " + String(dev_name.c_str()) + ".");
#endif
		funcs->close(fd);
		return false;
	}
	fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
	bool found = false;
	for (unsigned int i = 0; i < supported_formats.size(); ++i) {
		fmt.fmt.pix.pixelformat = supported_formats[i];
		// This commented code could be used if VIDIOC_TRY_FMT is not implemented
		// in the driver, but it will interfere with current executed streams.
		// if(-1 == xioctl(fd, VIDIOC_S_FMT, &fmt)) {
		//     if (errno == EBUSY) {
		//     #ifdef DEBUG_ENABLED
		//         print_line(String(dev_name.c_str()) + " is busy. Check for format.");
		//     #endif
		//         // If device is busy, check if it can use the format
		//         // Device is still available even if it is busy
		//         if (-1 == xioctl(fd, VIDIOC_TRY_FMT, &fmt)) {
		//             continue;
		//         }
		//     } else {
		//         continue;
		//     }
		// }
		if (xioctl(fd, VIDIOC_TRY_FMT, &fmt) == -1) {
			continue;
		}
		found = true;
		break;
	}
	if (!found) {
#ifdef DEBUG_ENABLED
		if (print_debug)
			print_line(String(dev_name.c_str()) + " has no supported pixelformat.");
#endif
		funcs->close(fd);
		return false;
	}

	/* Buggy driver paranoia. */
	unsigned int min = fmt.fmt.pix.width * 2;
	unsigned int bpl = fmt.fmt.pix.bytesperline;
	if (bpl < min)
		bpl = min;
	min = bpl * fmt.fmt.pix.height;
	buffer_size = fmt.fmt.pix.sizeimage;
	if (buffer_size < min)
		buffer_size = min;

	funcs->close(fd);

	// Now the device can be opened...
	// To start streaming the fmt must be set and the buffers must be prepared.
	return true;
}

bool V4l2_Device::close() {
	if (buffer_available) {
		cleanup_buffers();
	}
	if (fd != -1) {
		funcs->close(fd);
		fd = -1;
		opened = false;
	}
	return true;
}

bool V4l2_Device::request_buffers() {
	// open device
	if (fd != -1) {
		funcs->close(fd);
		fd = -1;
	}
	fd = funcs->open(dev_name.c_str(), O_RDWR | O_NONBLOCK, 0);
	opened = true;

	if (fd == -1) {
#ifdef DEBUG_ENABLED
		print_line("Cannot open device " + String(dev_name.c_str()) + ".");
#endif
		return false;
	}

	if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
#ifdef DEBUG_ENABLED
		print_line("Cannot set format for " + String(dev_name.c_str()) + ".");
#endif
		return false;
	}

	width = 0;
	height = 0;

	struct v4l2_requestbuffers req;

	switch (type) {
		case TYPE_IO_READ: {
			buffers = (V4l2_Device::buffer *)calloc(1, sizeof(*buffers));

			if (!buffers) {
#ifdef DEBUG_ENABLED
				print_line(String(dev_name.c_str()) + ": Out of memory");
#endif
				return false;
			}

			buffers[0].length = buffer_size;
			buffers[0].start = malloc(buffer_size);

			if (!buffers[0].start) {
#ifdef DEBUG_ENABLED
				print_line(String(dev_name.c_str()) + ": Out of memory");
#endif
				return false;
			}
			break;
		}
		case TYPE_IO_MMAP: {
			CLEAR(req);

			req.count = 4;
			req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			req.memory = V4L2_MEMORY_MMAP;

			int r = xioctl(fd, VIDIOC_REQBUFS, &req);
			if (r == -1 && errno == EINVAL) {
				type = TYPE_IO_USRPTR;
				break;
				// the switch statement should go to the next level
			} else if (r == -1) {
				return false;
			} else {
				if (req.count < 2)
					// fprintf(stderr, "Insufficient buffer memory on %s\n", dev_name);
					return false;
				buffers = (V4l2_Device::buffer *)calloc(req.count, sizeof(*buffers));

				if (!buffers) {
					// fprintf(stderr, "Out of memory\n");
					return false;
				}

				for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
					CLEAR(buf);

					buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
					buf.memory = V4L2_MEMORY_MMAP;
					buf.index = n_buffers;

					if (xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1)
						return false;

					buffers[n_buffers].length = buf.length;
					buffers[n_buffers].start = funcs->mmap(
							NULL /* start anywhere */,
							buf.length,
							PROT_READ | PROT_WRITE /* required */,
							MAP_SHARED /* recommended */,
							fd, buf.m.offset);

					if (buffers[n_buffers].start == MAP_FAILED)
						return false;
				}
				break;
			}
		}
		default:
			break;
	}
	// must use two switch statements
	// as TYPE_IO_USRPTR can only be determined
	// during VIDIOC_REQBUFS
	switch (type) {
		case TYPE_IO_NONE:
			return false;
		case TYPE_IO_USRPTR: {
			CLEAR(req);

			req.count = 4;
			req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			req.memory = V4L2_MEMORY_USERPTR;

			if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
				return false;
			}

			buffers = (V4l2_Device::buffer *)calloc(4, sizeof(*buffers));

			if (!buffers) {
				//fprintf(stderr, "Out of memory\n");
				return false;
			}

			for (n_buffers = 0; n_buffers < 4; ++n_buffers) {
				buffers[n_buffers].length = buffer_size;
				buffers[n_buffers].start = malloc(buffer_size);

				if (!buffers[n_buffers].start) {
					// fprintf(stderr, "Out of memory\n");
					return false;
				}
			}
			break;
		}
		default:
			break;
	}
	buffer_available = true;
	return true;
}

void V4l2_Device::cleanup_buffers() {
	switch (type) {
		case TYPE_IO_READ: {
			free(buffers[0].start);
			break;
		}
		case TYPE_IO_MMAP: {
			for (unsigned int i = 0; i < n_buffers; ++i) {
				funcs->munmap(buffers[i].start, buffers[i].length);
			}
			break;
		}
		case TYPE_IO_USRPTR: {
			for (unsigned int i = 0; i < n_buffers; ++i)
				free(buffers[i].start);
			break;
		}
		default:
			break;
	}
	free(buffers);
	buffer_available = false;
}

V4l2_Device::~V4l2_Device() {
	if (this->opened) {
		this->close();
	}
}

bool V4l2_Device::start_streaming(Ref<CameraFeed> feed) {
	enum v4l2_buf_type b_type;

	// start streaming depending on type
	switch (type) {
		case TYPE_IO_MMAP: {
			for (unsigned int i = 0; i < n_buffers; ++i) {
				CLEAR(buf);
				buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				buf.memory = V4L2_MEMORY_MMAP;
				buf.index = i;

				if (xioctl(fd, VIDIOC_QBUF, &buf) == -1)
					return false;
			}
			b_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			if (xioctl(fd, VIDIOC_STREAMON, &b_type) == -1)
				return false;
			break;
		}
		case TYPE_IO_USRPTR: {
			for (unsigned int i = 0; i < n_buffers; ++i) {
				struct v4l2_buffer buf;

				CLEAR(buf);
				buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				buf.memory = V4L2_MEMORY_USERPTR;
				buf.index = i;
				buf.m.userptr = (unsigned long)buffers[i].start;
				buf.length = buffers[i].length;

				if (xioctl(fd, VIDIOC_QBUF, &buf) == -1)
					return false;
			}
			b_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			if (xioctl(fd, VIDIOC_STREAMON, &b_type) == -1)
				return false;
			break;
		}
		default:
			break;
	}

	streaming = true;
	stream_thread = std::thread(&V4l2_Device::stream_image, this, feed);
	return true;
};

void V4l2_Device::stop_streaming() {
	streaming = false;
	if (stream_thread.joinable())
		stream_thread.join();

	switch (type) {
		case TYPE_IO_MMAP:
		case TYPE_IO_USRPTR:
			enum v4l2_buf_type b_type;
			b_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			xioctl(fd, VIDIOC_STREAMOFF, &b_type);
			break;
		default:
			break;
	}
}

void V4l2_Device::stream_image(Ref<CameraFeed> feed) {
	fd_set fds;
	struct timeval tv;
	int r;
	while (this->streaming) {
		// check if new image available
		do {
			FD_ZERO(&fds);
			FD_SET(this->fd, &fds);

			/* Timeout. */
			tv.tv_sec = 2;
			tv.tv_usec = 0;

			r = select(fd + 1, &fds, NULL, NULL, &tv);
		} while (r == -1 && (errno == EINTR || errno == EAGAIN));
		if (r <= 0) {
			return;
		}

		// grab image depending on the type of image grabbing
		// currently only tested for TYPE_IO_MMAP
		switch (type) {
			case TYPE_IO_READ: {
				if (funcs->read(fd, buffers[0].start, buffers[0].length) == -1) {
					switch (errno) {
						case EAGAIN:
							continue;
						case EIO: // could be ignored somehow?
						default:
							return;
					}
				}

				get_image(feed, (uint8_t *)(buffers[0].start));
				break;
			}
			case TYPE_IO_MMAP: {
				CLEAR(buf);

				buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				buf.memory = V4L2_MEMORY_MMAP;

				if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
					switch (errno) {
						case EAGAIN:
							continue;
						case EIO: // could be ignored
						default:
							return;
					}
				}

				get_image(feed, (uint8_t *)(buffers[buf.index].start));

				if (xioctl(fd, VIDIOC_QBUF, &buf) == -1)
					return;
				break;
			}
			case TYPE_IO_USRPTR: {
				CLEAR(buf);

				buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				buf.memory = V4L2_MEMORY_USERPTR;

				if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
					switch (errno) {
						case EAGAIN:
							continue;
						case EIO: // could be ignored
						default:
							return;
					}
				}

				for (unsigned int i = 0; i < n_buffers; ++i)
					if (buf.m.userptr == (unsigned long)buffers[i].start && buf.length == buffers[i].length)
						break;

				get_image(feed, (uint8_t *)(buf.m.userptr));

				if (xioctl(fd, VIDIOC_QBUF, &buf) == -1)
					return;
				break;
			}
			default:
				break;
		}
	}
}

void V4l2_Device::get_image(Ref<CameraFeed> feed, uint8_t *buffer) {
	Ref<Image> img;
	img.instance();

	if (xioctl(fd, VIDIOC_G_FMT, &fmt) == -1) {
		return;
	}

	// other format implementations should be put here.
	// Mainly useful if libv4l2 is not installed
	switch (fmt.fmt.pix.pixelformat) {
		case V4L2_PIX_FMT_RGB24: {
			if (width != fmt.fmt.pix.width || height != fmt.fmt.pix.height) {
				width = fmt.fmt.pix.width;
				height = fmt.fmt.pix.height;
				img_data.resize(width * height * 3);
			}

			uint8_t *w = img_data.ptrw();
			// TODO: Buffer is 1024 Byte longer?
			memcpy(w, buffer, width * height * 3);

			img->create(width, height, 0, Image::FORMAT_RGB8, img_data);
			feed->set_RGB_img(img);
			break;
		}
		default:
			break;
	}
}

//////////////////////////////////////////////////////////////////////////
// CameraFeedX11 - Subclass for camera feeds in Linux

V4l2_Device *CameraFeedX11::get_device() const {
	return device;
};

CameraFeedX11::CameraFeedX11() {
	device = NULL;
};

void CameraFeedX11::set_device(V4l2_Device *p_device) {
	device = p_device;

	// get some info
	name = device->name;
	// I do not think that another position is possible on linux
	position = CameraFeed::FEED_UNSPECIFIED;
};

CameraFeedX11::~CameraFeedX11() {
	this->deactivate_feed();
	if (device != NULL) {
		memdelete(device);
		device = NULL;
	}
};

bool CameraFeedX11::activate_feed() {
	// activate streaming if not already
	if (!device->streaming) {
		if (!device->check_device()) {
			return false;
		}
		if (!device->request_buffers()) {
			device->close();
			return false;
		}
		if (!device->start_streaming(this)) {
			device->cleanup_buffers();
			device->close();
			return false;
		}
	};

	return true;
};

void CameraFeedX11::deactivate_feed() {
	// end camera capture if we have one
	if (device->streaming) {
		device->stop_streaming();
		device->cleanup_buffers();
		device->close();
	};
};

void CameraX11::update_feeds() {
	DIR *dir;
	struct dirent *ent;
	auto devs = std::vector<std::string>();
	// check the /dev/ directory for video entries
	if ((dir = opendir("/dev/")) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (strncmp(ent->d_name, "video", 5) == 0) {
				devs.push_back(std::string("/dev/") + std::string(ent->d_name));
			}
		}
		closedir(dir);
	}
	// sort so we start with video0
	std::sort(devs.begin(), devs.end());

	int j;
	unsigned int i;
	std::vector<std::string>::iterator it;

	// remove missing feeds
	for (j = 0; j < feeds.size(); ++j) {
		Ref<CameraFeedX11> feed = (Ref<CameraFeedX11>)feeds[j];
		std::string dev_name = feed->get_device()->dev_name;
		it = std::find(devs.begin(), devs.end(), dev_name);
		// also check if device is still openable
		if (it == devs.end() || !feed->get_device()->check_device(!alive)) {
			remove_feed(feed);
		}
	}

	for (i = 0; i < devs.size(); ++i) {
		// keep existing devices
		// currently only check by /dev/video* name
		bool found = false;
		for (j = 0; j < feeds.size(); ++j) {
			Ref<CameraFeedX11> feed = (Ref<CameraFeedX11>)feeds[j];
			if (devs[i] == feed->get_device()->dev_name) {
				found = true;
				break;
			}
		}
		if (!found) {
			// create new device and check if it is compatible
			V4l2_Device *dev = memnew(V4l2_Device(devs[i].c_str(), &this->funcs));
			if (dev->check_device(!alive)) {
				Ref<CameraFeedX11> newfeed;
				newfeed.instance();
				newfeed->set_device(dev);

				// assume display camera so inverse
				Transform2D transform = Transform2D(-1.0, 0.0, 0.0, -1.0, 1.0, 1.0);
				newfeed->set_transform(transform);

				add_feed(newfeed);
			} else {
				memdelete(dev);
			}
		}
	}
};

void CameraX11::check_change() {
	// simple hotplug functionality
	// checks for new cameras (or removed cameras)
	// every second.
	while (alive) {
		update_feeds();
		usleep(1000000);
	}
}

CameraX11::CameraX11() {
	// determine if libv4l2 is installed and
	// set functions appropriately
	libv4l2 = dlopen("libv4l2.so.0", RTLD_NOW);

	if (libv4l2 == NULL) {
		// the default v4l2 functions
		this->funcs.open = &open;
		this->funcs.close = &close;
		this->funcs.dup = &dup;
		this->funcs.ioctl = &ioctl;
		this->funcs.read = &read;
		this->funcs.mmap = &mmap;
		this->funcs.munmap = &munmap;
		this->funcs.libv4l2 = false;
#ifdef DEBUG_ENABLED
		print_line("libv4l2.so not found. Try standard v4l2 instead.");
#endif
	} else {
		// the libv4l2 functions
		this->funcs.open = (int (*)(const char *, int, ...))dlsym(libv4l2, "v4l2_open");
		this->funcs.close = (int (*)(int))dlsym(libv4l2, "v4l2_close");
		this->funcs.dup = (int (*)(int))dlsym(libv4l2, "v4l2_dup");
		this->funcs.ioctl = (int (*)(int, unsigned long int, ...))dlsym(libv4l2, "v4l2_ioctl");
		this->funcs.read = (long int (*)(int, void *, size_t))dlsym(libv4l2, "v4l2_read");
		this->funcs.mmap = (void *(*)(void *, size_t, int, int, int, int64_t))dlsym(libv4l2, "v4l2_mmap");
		this->funcs.munmap = (int (*)(void *, size_t))dlsym(libv4l2, "v4l2_munmap");
		this->funcs.libv4l2 = true;
#ifdef DEBUG_ENABLED
		print_line("libv4l2 found.");
#endif
	}

	// Find available cameras we have at this time
	update_feeds();

	// start the hotplug thread
	// alive is also used to trigger debug output
	// (as devices that are incompatible would always be checked
	// and therefore print always debug messages)
	alive = true;
	hotplug_thread = std::thread(&CameraX11::check_change, this);
};

CameraX11::~CameraX11() {
	// end the hotplug thread
	alive = false;
	if (hotplug_thread.joinable()) {
		hotplug_thread.join();
	}

	// close the library
	if (this->libv4l2 != NULL) {
		dlclose(this->libv4l2);
	}
};
