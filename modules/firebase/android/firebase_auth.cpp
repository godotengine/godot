/*************************************************************************/
/*  firebase_auth.cpp                                                    */
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

#include "firebase_auth.h"
#include "core/engine.h"
#include "firebase/app.h"

#include "platform/android/thread_jandroid.h"

static Object *java_singleton = nullptr;

FirebaseAuth::FirebaseAuth() :
		state_listner(this),
		token_listner(this) {
	if (java_singleton == nullptr) {
		java_singleton = Engine::get_singleton()->get_singleton_object("GodotFirebaseAuth");
	}
}

void FirebaseAuth::initialize(const String &config) {
	if (!config.empty()) {
		print_line("ignore provided options. android using options from application resources");
	}
	JNIEnv *env = ThreadAndroid::get_env();
	jclass activityThread = env->FindClass("android/app/ActivityThread");
	jmethodID currentActivityThread = env->GetStaticMethodID(activityThread, "currentActivityThread", "()Landroid/app/ActivityThread;");
	jobject at = env->CallStaticObjectMethod(activityThread, currentActivityThread);
	jmethodID getApplication = env->GetMethodID(activityThread, "getApplication", "()Landroid/app/Application;");
	jobject context = env->CallObjectMethod(at, getApplication);
	jobject _godot_instance = env->NewGlobalRef(context);

	app = firebase::App::Create(env, _godot_instance);
	auth = Auth::GetAuth(app);
	auth->AddAuthStateListener(&state_listner);
	auth->AddIdTokenListener(&token_listner);
}

void FirebaseAuth::request_google_id_token() {
	java_singleton->call("requestGoogleIdToken", this->get_instance_id());
}

void FirebaseAuth::request_facebook_access_token() {
	java_singleton->call("requestFacebookAccessToken", this->get_instance_id());
}
