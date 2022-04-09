/*************************************************************************/
/*  string_android.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef STRING_ANDROID_H
#define STRING_ANDROID_H
#include "core/string/ustring.h"
#include "thread_jandroid.h"
#include <jni.h>

/**
 * Converts JNI jstring to Godot String.
 * @param source Source JNI string. If null an empty string is returned.
 * @param env JNI environment instance. If null obtained by get_jni_env().
 * @return Godot string instance.
 */
static inline String jstring_to_string(jstring source, JNIEnv *env = nullptr) {
	String result;
	if (source) {
		if (!env) {
			env = get_jni_env();
		}
		const char *const source_utf8 = env->GetStringUTFChars(source, nullptr);
		if (source_utf8) {
			result.parse_utf8(source_utf8);
			env->ReleaseStringUTFChars(source, source_utf8);
		}
	}
	return result;
}

#endif // STRING_ANDROID_H
