/*************************************************************************/
/*  api.cpp                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "api.h"
#include "core/engine.h"
#ifdef ANDROID_ENABLED
#include "platform/android/java_godot_io_wrapper.h"
#include "platform/android/os_android.h"
#endif

#include "core/object.h"

class Android : public Object {
private:
	GDCLASS(Android, Object);

	static Android *singleton;

protected:
	static void _bind_methods();

public:
	void multicast_lock_acquire();
	void multicast_lock_release();

	static Android *get_singleton();
	Android();
	~Android();
};

static Android *android_api;

void register_android_api() {

	ClassDB::register_virtual_class<Android>();
	android_api = memnew(Android);
	Engine::get_singleton()->add_singleton(Engine::Singleton("Android", android_api));
}

void unregister_android_api() {

	memdelete(android_api);
}

Android *Android::singleton = NULL;

Android *Android::get_singleton() {

	return singleton;
}

Android::Android() {

	ERR_FAIL_COND_MSG(singleton != NULL, "Android singleton already exist.");
	singleton = this;
}

Android::~Android() {}

void Android::_bind_methods() {

	ClassDB::bind_method(D_METHOD("multicast_lock_acquire"), &Android::multicast_lock_acquire);
	ClassDB::bind_method(D_METHOD("multicast_lock_release"), &Android::multicast_lock_release);
}

void Android::multicast_lock_acquire() {
#ifdef ANDROID_ENABLED
	((OS_Android *)OS::get_singleton())->get_godot_io_java()->multicast_lock_acquire();
#endif
}

void Android::multicast_lock_release() {
#ifdef ANDROID_ENABLED
	((OS_Android *)OS::get_singleton())->get_godot_io_java()->multicast_lock_release();
#endif
}
