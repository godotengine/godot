/*************************************************************************/
/*  net_socket_android.cpp                                               */
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

#include "net_socket_android.h"

#include "thread_jandroid.h"

jobject NetSocketAndroid::net_utils = 0;
jclass NetSocketAndroid::cls = 0;
jmethodID NetSocketAndroid::_multicast_lock_acquire = 0;
jmethodID NetSocketAndroid::_multicast_lock_release = 0;

void NetSocketAndroid::setup(jobject p_net_utils) {
	JNIEnv *env = ThreadAndroid::get_env();

	net_utils = env->NewGlobalRef(p_net_utils);

	jclass c = env->GetObjectClass(net_utils);
	cls = (jclass)env->NewGlobalRef(c);

	_multicast_lock_acquire = env->GetMethodID(cls, "multicastLockAcquire", "()V");
	_multicast_lock_release = env->GetMethodID(cls, "multicastLockRelease", "()V");
}

void NetSocketAndroid::multicast_lock_acquire() {
	if (_multicast_lock_acquire) {
		JNIEnv *env = ThreadAndroid::get_env();
		env->CallVoidMethod(net_utils, _multicast_lock_acquire);
	}
}

void NetSocketAndroid::multicast_lock_release() {
	if (_multicast_lock_release) {
		JNIEnv *env = ThreadAndroid::get_env();
		env->CallVoidMethod(net_utils, _multicast_lock_release);
	}
}

NetSocket *NetSocketAndroid::_create_func() {
	return memnew(NetSocketAndroid);
}

void NetSocketAndroid::make_default() {
	_create = _create_func;
}

NetSocketAndroid::~NetSocketAndroid() {
	close();
}

void NetSocketAndroid::close() {
	NetSocketPosix::close();
	if (wants_broadcast)
		multicast_lock_release();
	if (multicast_groups)
		multicast_lock_release();
	wants_broadcast = false;
	multicast_groups = 0;
}

Error NetSocketAndroid::set_broadcasting_enabled(bool p_enabled) {
	Error err = NetSocketPosix::set_broadcasting_enabled(p_enabled);
	if (err != OK)
		return err;

	if (p_enabled != wants_broadcast) {
		if (p_enabled) {
			multicast_lock_acquire();
		} else {
			multicast_lock_release();
		}

		wants_broadcast = p_enabled;
	}

	return OK;
}

Error NetSocketAndroid::join_multicast_group(const IP_Address &p_multi_address, String p_if_name) {
	Error err = NetSocketPosix::join_multicast_group(p_multi_address, p_if_name);
	if (err != OK)
		return err;

	if (!multicast_groups)
		multicast_lock_acquire();
	multicast_groups++;

	return OK;
}

Error NetSocketAndroid::leave_multicast_group(const IP_Address &p_multi_address, String p_if_name) {
	Error err = NetSocketPosix::leave_multicast_group(p_multi_address, p_if_name);
	if (err != OK)
		return err;

	ERR_FAIL_COND_V(multicast_groups == 0, ERR_BUG);

	multicast_groups--;
	if (!multicast_groups)
		multicast_lock_release();

	return OK;
}
