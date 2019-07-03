/*************************************************************************/
/*  android_utils.cpp                                                    */
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

#include "android_utils.h"

#ifdef __ANDROID__

#include "platform/android/thread_jandroid.h"

namespace GDMonoUtils {
namespace Android {

String get_app_native_lib_dir() {
	JNIEnv *env = ThreadAndroid::get_env();

	jclass activityThreadClass = env->FindClass("android/app/ActivityThread");
	jmethodID currentActivityThread = env->GetStaticMethodID(activityThreadClass, "currentActivityThread", "()Landroid/app/ActivityThread;");
	jobject activityThread = env->CallStaticObjectMethod(activityThreadClass, currentActivityThread);
	jmethodID getApplication = env->GetMethodID(activityThreadClass, "getApplication", "()Landroid/app/Application;");
	jobject ctx = env->CallObjectMethod(activityThread, getApplication);

	jmethodID getApplicationInfo = env->GetMethodID(env->GetObjectClass(ctx), "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;");
	jobject applicationInfo = env->CallObjectMethod(ctx, getApplicationInfo);
	jfieldID nativeLibraryDirField = env->GetFieldID(env->GetObjectClass(applicationInfo), "nativeLibraryDir", "Ljava/lang/String;");
	jstring nativeLibraryDir = (jstring)env->GetObjectField(applicationInfo, nativeLibraryDirField);

	String result;

	const char *const nativeLibraryDir_utf8 = env->GetStringUTFChars(nativeLibraryDir, NULL);
	if (nativeLibraryDir_utf8) {
		result.parse_utf8(nativeLibraryDir_utf8);
		env->ReleaseStringUTFChars(nativeLibraryDir, nativeLibraryDir_utf8);
	}

	return result;
}

} // namespace Android
} // namespace GDMonoUtils

#endif // __ANDROID__
