/*************************************************************************/
/*  power_android.cpp                                                    */
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

/*
Adapted from corresponding SDL 2.0 code.
*/

/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2017 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "power_android.h"

#include "core/error_macros.h"

static void LocalReferenceHolder_Cleanup(struct LocalReferenceHolder *refholder) {
	if (refholder->m_env) {
		JNIEnv *env = refholder->m_env;
		(*env)->PopLocalFrame(env, NULL);
		--s_active;
	}
}

static struct LocalReferenceHolder LocalReferenceHolder_Setup(const char *func) {
	struct LocalReferenceHolder refholder;
	refholder.m_env = NULL;
	refholder.m_func = func;
	return refholder;
}

static bool LocalReferenceHolder_Init(struct LocalReferenceHolder *refholder, JNIEnv *env) {
	const int capacity = 16;
	if ((*env)->PushLocalFrame(env, capacity) < 0) {
		return false;
	}
	++s_active;
	refholder->m_env = env;
	return true;
}

static SDL_bool LocalReferenceHolder_IsActive(void) {
	return s_active > 0;
}

ANativeWindow *Android_JNI_GetNativeWindow(void) {
	ANativeWindow *anw;
	jobject s;
	JNIEnv *env = Android_JNI_GetEnv();

	s = (*env)->CallStaticObjectMethod(env, mActivityClass, midGetNativeSurface);
	anw = ANativeWindow_fromSurface(env, s);
	(*env)->DeleteLocalRef(env, s);

	return anw;
}

/* 
 *  CODE CHUNK IMPORTED FROM SDL 2.0
 * returns 0 on success or -1 on error (others undefined then)
 * returns truthy or falsy value in plugged, charged and battery
 * returns the value in seconds and percent or -1 if not available
 */
int Android_JNI_GetPowerInfo(int *plugged, int *charged, int *battery, int *seconds, int *percent) {
	env = Android_JNI_GetEnv();
	refs = LocalReferenceHolder_Setup(__FUNCTION__);

	if (!LocalReferenceHolder_Init(&refs, env)) {
		LocalReferenceHolder_Cleanup(&refs);
		return -1;
	}
	mid = (*env)->GetStaticMethodID(env, mActivityClass, "getContext", "()Landroid/content/Context;");
	context = (*env)->CallStaticObjectMethod(env, mActivityClass, mid);
	action = (*env)->NewStringUTF(env, "android.intent.action.BATTERY_CHANGED");
	cls = (*env)->FindClass(env, "android/content/IntentFilter");
	mid = (*env)->GetMethodID(env, cls, "<init>", "(Ljava/lang/String;)V");
	filter = (*env)->NewObject(env, cls, mid, action);
	(*env)->DeleteLocalRef(env, action);
	mid = (*env)->GetMethodID(env, mActivityClass, "registerReceiver", "(Landroid/content/BroadcastReceiver;Landroid/content/IntentFilter;)Landroid/content/Intent;");
	intent = (*env)->CallObjectMethod(env, context, mid, NULL, filter);
	(*env)->DeleteLocalRef(env, filter);
	cls = (*env)->GetObjectClass(env, intent);
	imid = (*env)->GetMethodID(env, cls, "getIntExtra", "(Ljava/lang/String;I)I");
// Watch out for C89 scoping rules because of the macro
#define GET_INT_EXTRA(var, key)                                \
	int var;                                                   \
	iname = (*env)->NewStringUTF(env, key);                    \
	var = (*env)->CallIntMethod(env, intent, imid, iname, -1); \
	(*env)->DeleteLocalRef(env, iname);
	bmid = (*env)->GetMethodID(env, cls, "getBooleanExtra", "(Ljava/lang/String;Z)Z");
// Watch out for C89 scoping rules because of the macro
#define GET_BOOL_EXTRA(var, key)                                          \
	int var;                                                              \
	bname = (*env)->NewStringUTF(env, key);                               \
	var = (*env)->CallBooleanMethod(env, intent, bmid, bname, JNI_FALSE); \
	(*env)->DeleteLocalRef(env, bname);
	if (plugged) {
		// Watch out for C89 scoping rules because of the macro
		GET_INT_EXTRA(plug, "plugged") // == BatteryManager.EXTRA_PLUGGED (API 5)
		if (plug == -1) {
			LocalReferenceHolder_Cleanup(&refs);
			return -1;
		}
		// 1 == BatteryManager.BATTERY_PLUGGED_AC
		// 2 == BatteryManager.BATTERY_PLUGGED_USB
		*plugged = (0 < plug) ? 1 : 0;
	}
	if (charged) {
		// Watch out for C89 scoping rules because of the macro
		GET_INT_EXTRA(status, "status") // == BatteryManager.EXTRA_STATUS (API 5)
		if (status == -1) {
			LocalReferenceHolder_Cleanup(&refs);
			return -1;
		}
		// 5 == BatteryManager.BATTERY_STATUS_FULL
		*charged = (status == 5) ? 1 : 0;
	}
	if (battery) {
		GET_BOOL_EXTRA(present, "present") // == BatteryManager.EXTRA_PRESENT (API 5)
		*battery = present ? 1 : 0;
	}
	if (seconds) {
		*seconds = -1; // not possible
	}
	if (percent) {
		int level;
		int scale;
		// Watch out for C89 scoping rules because of the macro
		{
			GET_INT_EXTRA(level_temp, "level") // == BatteryManager.EXTRA_LEVEL (API 5)
			level = level_temp;
		}
		// Watch out for C89 scoping rules because of the macro
		{
			GET_INT_EXTRA(scale_temp, "scale") // == BatteryManager.EXTRA_SCALE (API 5)
			scale = scale_temp;
		}
		if ((level == -1) || (scale == -1)) {
			LocalReferenceHolder_Cleanup(&refs);
			return -1;
		}
		*percent = level * 100 / scale;
	}
	(*env)->DeleteLocalRef(env, intent);
	LocalReferenceHolder_Cleanup(&refs);

	return 0;
}

bool power_android::GetPowerInfo_Android() {
	int battery;
	int plugged;
	int charged;

	if (Android_JNI_GetPowerInfo(&plugged, &charged, &battery, &this->nsecs_left, &this->percent_left) != -1) {
		if (plugged) {
			if (charged) {
				this->power_state = OS::POWERSTATE_CHARGED;
			} else if (battery) {
				this->power_state = OS::POWERSTATE_CHARGING;
			} else {
				this->power_state = OS::POWERSTATE_NO_BATTERY;
				this->nsecs_left = -1;
				this->percent_left = -1;
			}
		} else {
			this->power_state = OS::POWERSTATE_ON_BATTERY;
		}
	} else {
		this->power_state = OS::POWERSTATE_UNKNOWN;
		this->nsecs_left = -1;
		this->percent_left = -1;
	}

	return true;
}

OS::PowerState power_android::get_power_state() {
	if (GetPowerInfo_Android()) {
		return power_state;
	} else {
		WARN_PRINT("Power management is not implemented on this platform, defaulting to POWERSTATE_UNKNOWN");
		return OS::POWERSTATE_UNKNOWN;
	}
}

int power_android::get_power_seconds_left() {
	if (GetPowerInfo_Android()) {
		return nsecs_left;
	} else {
		WARN_PRINT("Power management is not implemented on this platform, defaulting to -1");
		return -1;
	}
}

int power_android::get_power_percent_left() {
	if (GetPowerInfo_Android()) {
		return percent_left;
	} else {
		WARN_PRINT("Power management is not implemented on this platform, defaulting to -1");
		return -1;
	}
}

power_android::power_android() :
		nsecs_left(-1),
		percent_left(-1),
		power_state(OS::POWERSTATE_UNKNOWN) {
}

power_android::~power_android() {
}
