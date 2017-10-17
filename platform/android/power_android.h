/*************************************************************************/
/*  power_android.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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

#ifndef PLATFORM_ANDROID_POWER_ANDROID_H_
#define PLATFORM_ANDROID_POWER_ANDROID_H_

#include "os/os.h"
#include <android/native_window_jni.h>

class power_android {

	struct LocalReferenceHolder {
		JNIEnv *m_env;
		const char *m_func;
	};

private:
	static struct LocalReferenceHolder refs;
	static JNIEnv *env;
	static jmethodID mid;
	static jobject context;
	static jstring action;
	static jclass cls;
	static jobject filter;
	static jobject intent;
	static jstring iname;
	static jmethodID imid;
	static jstring bname;
	static jmethodID bmid;

	int nsecs_left;
	int percent_left;
	OS::PowerState power_state;

	bool GetPowerInfo_Android();
	bool UpdatePowerInfo();

public:
	static int s_active;

	power_android();
	virtual ~power_android();
	static bool LocalReferenceHolder_Init(struct LocalReferenceHolder *refholder, JNIEnv *env);
	static struct LocalReferenceHolder LocalReferenceHolder_Setup(const char *func);
	static void LocalReferenceHolder_Cleanup(struct LocalReferenceHolder *refholder);

	OS::PowerState get_power_state();
	int get_power_seconds_left();
	int get_power_percent_left();
};

#endif /* PLATFORM_ANDROID_POWER_ANDROID_H_ */
