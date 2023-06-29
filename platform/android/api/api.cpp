/**************************************************************************/
/*  api.cpp                                                               */
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

#include "api.h"

#include "java_class_wrapper.h"
#include "jni_singleton.h"

#ifdef ANDROID_ENABLED
#include "java_godot_wrapper.h"
#include "os_android.h"
#include "thread_jandroid.h"
#endif

#include "core/config/engine.h"
#include "core/extension/gdextension.h"

#if !defined(ANDROID_ENABLED)
static JavaClassWrapper *java_class_wrapper = nullptr;
#endif

static void *gdextension_android_get_env() {
#ifdef ANDROID_ENABLED
	return (void *)get_jni_env();
#else
	return nullptr;
#endif
}

static void *gdextension_android_get_activity() {
#ifdef ANDROID_ENABLED
	OS_Android *android = OS_Android::get_singleton();
	if (android) {
		GodotJavaWrapper *godot_java = android->get_godot_java();
		if (godot_java) {
			return (void *)godot_java->get_activity();
		}
	}
#endif

	return nullptr;
}

static void *gdextension_android_get_surface() {
#ifdef ANDROID_ENABLED
	OS_Android *android = OS_Android::get_singleton();
	if (android) {
		GodotJavaWrapper *godot_java = android->get_godot_java();
		if (godot_java) {
			return (void *)godot_java->get_surface();
		}
	}
#endif

	return nullptr;
}

static GDExtensionBool gdextension_android_is_activity_resumed() {
#ifdef ANDROID_ENABLED
	OS_Android *android = OS_Android::get_singleton();
	if (android) {
		GodotJavaWrapper *godot_java = android->get_godot_java();
		if (godot_java) {
			return godot_java->is_activity_resumed();
		}
	}
#endif

	return false;
}

void register_android_api() {
#if !defined(ANDROID_ENABLED)
	// On Android platforms, the `java_class_wrapper` instantiation and the
	// `JNISingleton` registration occurs in
	// `platform/android/java_godot_lib_jni.cpp#Java_org_godotengine_godot_GodotLib_setup`
	java_class_wrapper = memnew(JavaClassWrapper); // Dummy
	GDREGISTER_CLASS(JNISingleton);
#endif

	GDREGISTER_CLASS(JavaClass);
	GDREGISTER_CLASS(JavaClassWrapper);
	Engine::get_singleton()->add_singleton(Engine::Singleton("JavaClassWrapper", JavaClassWrapper::get_singleton()));

	GDExtension::register_interface_function("android_get_env", (GDExtensionInterfaceFunctionPtr)&gdextension_android_get_env);
	GDExtension::register_interface_function("android_get_activity", (GDExtensionInterfaceFunctionPtr)&gdextension_android_get_activity);
	GDExtension::register_interface_function("android_get_surface", (GDExtensionInterfaceFunctionPtr)&gdextension_android_get_surface);
	GDExtension::register_interface_function("android_is_activity_resumed", (GDExtensionInterfaceFunctionPtr)&gdextension_android_is_activity_resumed);
}

void unregister_android_api() {
#if !defined(ANDROID_ENABLED)
	memdelete(java_class_wrapper);
#endif
}

void JavaClassWrapper::_bind_methods() {
	ClassDB::bind_method(D_METHOD("wrap", "name"), &JavaClassWrapper::wrap);
}

#if !defined(ANDROID_ENABLED)

Variant JavaClass::callp(const StringName &, const Variant **, int, Callable::CallError &) {
	return Variant();
}

JavaClass::JavaClass() {
}

Variant JavaObject::callp(const StringName &, const Variant **, int, Callable::CallError &) {
	return Variant();
}

JavaClassWrapper *JavaClassWrapper::singleton = nullptr;

Ref<JavaClass> JavaClassWrapper::wrap(const String &) {
	return Ref<JavaClass>();
}

JavaClassWrapper::JavaClassWrapper() {
	singleton = this;
}

#endif
