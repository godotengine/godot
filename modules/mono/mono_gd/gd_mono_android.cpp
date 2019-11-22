/*************************************************************************/
/*  gd_mono_android.cpp                                                  */
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

#include "gd_mono_android.h"

#if defined(ANDROID_ENABLED)

#include <dlfcn.h> // dlopen, dlsym
#include <mono/utils/mono-dl-fallback.h>

#include "core/os/os.h"
#include "core/ustring.h"
#include "platform/android/thread_jandroid.h"

#include "../utils/path_utils.h"
#include "../utils/string_utils.h"

namespace GDMonoAndroid {

String app_native_lib_dir_cache;

String determine_app_native_lib_dir() {
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

String get_app_native_lib_dir() {
	if (app_native_lib_dir_cache.empty())
		app_native_lib_dir_cache = determine_app_native_lib_dir();
	return app_native_lib_dir_cache;
}

int gd_mono_convert_dl_flags(int flags) {
	// from mono's runtime-bootstrap.c

	int lflags = flags & MONO_DL_LOCAL ? 0 : RTLD_GLOBAL;

	if (flags & MONO_DL_LAZY)
		lflags |= RTLD_LAZY;
	else
		lflags |= RTLD_NOW;

	return lflags;
}

void *gd_mono_android_dlopen(const char *p_name, int p_flags, char **r_err, void *p_user_data) {
	String name = String::utf8(p_name);

	if (name.ends_with(".dll.so") || name.ends_with(".exe.so")) {
		String app_native_lib_dir = get_app_native_lib_dir();

		String orig_so_name = name.get_file();
		String so_name = "lib-aot-" + orig_so_name;
		String so_path = path::join(app_native_lib_dir, so_name);

		if (!FileAccess::exists(so_path)) {
			if (OS::get_singleton()->is_stdout_verbose())
				OS::get_singleton()->print("Cannot find shared library: '%s'\n", so_path.utf8().get_data());
			return NULL;
		}

		int lflags = gd_mono_convert_dl_flags(p_flags);

		void *handle = dlopen(so_path.utf8().get_data(), lflags);

		if (!handle) {
			if (OS::get_singleton()->is_stdout_verbose())
				OS::get_singleton()->print("Failed to open shared library: '%s'. Error: '%s'\n", so_path.utf8().get_data(), dlerror());
			return NULL;
		}

		if (OS::get_singleton()->is_stdout_verbose())
			OS::get_singleton()->print("Successfully loaded AOT shared library: '%s'\n", so_path.utf8().get_data());

		return handle;
	}

	return NULL;
}

void *gd_mono_android_dlsym(void *p_handle, const char *p_name, char **r_err, void *p_user_data) {
	void *sym_addr = dlsym(p_handle, p_name);

	if (sym_addr)
		return sym_addr;

	if (r_err)
		*r_err = str_format_new("%s\n", dlerror());

	return NULL;
}

void register_android_dl_fallback() {
	mono_dl_fallback_register(gd_mono_android_dlopen, gd_mono_android_dlsym, NULL, NULL);
}

} // namespace GDMonoAndroid

#endif
