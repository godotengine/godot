/**************************************************************************/
/*  net_socket_android.h                                                  */
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

#ifndef NET_SOCKET_ANDROID_H
#define NET_SOCKET_ANDROID_H

#include "drivers/unix/net_socket_posix.h"

#include <jni.h>

/**
 * Specialized NetSocket implementation for Android.
 *
 * Some devices requires Android-specific code to acquire a MulticastLock
 * before sockets are allowed to receive broadcast and multicast packets.
 * This implementation calls into Java code and automatically acquire/release
 * the lock when broadcasting is enabled/disabled on a socket, or that socket
 * joins/leaves a multicast group.
 */
class NetSocketAndroid : public NetSocketPosix {
private:
	static jobject net_utils;
	static jclass cls;
	static jmethodID _multicast_lock_acquire;
	static jmethodID _multicast_lock_release;

	bool wants_broadcast;
	int multicast_groups;

	static void multicast_lock_acquire();
	static void multicast_lock_release();

protected:
	static NetSocket *_create_func();

public:
	static void make_default();
	static void setup(jobject p_net_utils);

	virtual void close();

	virtual Error set_broadcasting_enabled(bool p_enabled);
	virtual Error join_multicast_group(const IP_Address &p_multi_address, String p_if_name);
	virtual Error leave_multicast_group(const IP_Address &p_multi_address, String p_if_name);

	NetSocketAndroid();
	~NetSocketAndroid();
};

#endif // NET_SOCKET_ANDROID_H
