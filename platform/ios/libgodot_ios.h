/**************************************************************************/
/*  libgodot_ios.h                                                        */
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

#pragma once

// 2dog: entry points for iOS apps that link libgodot.a statically and own UIApplication. The host creates the
// engine, embeds Godot's view controller somewhere in its UIKit hierarchy and never pumps: GDTViewController's
// display link runs setup2, start and every frame, exactly as in Godot's own app template.

#include "core/extension/libgodot.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*libgodot_ios_callback)(void *p_userdata);

// Creates the OS and runs Main::setup with the given arguments (argv[0] is the program name; pass an absolute
// --main-pack). p_init_func may be null. Returns 0 on success, otherwise the Godot Error code; after an error no
// engine exists and the call may be repeated.
LIBGODOT_API int libgodot_ios_create(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func);

// Godot's UIViewController (GDTViewController), created on first use and retained for the process. Present it,
// or add it as a child view controller; Godot starts rendering once its view appears.
LIBGODOT_API void *libgodot_ios_view_controller(void);

// p_started runs on the main thread after the main loop initialized (the scene tree exists); p_quit_requested
// runs once when the main loop asks to quit. Either may be null.
LIBGODOT_API void libgodot_ios_set_callbacks(libgodot_ios_callback p_started, libgodot_ios_callback p_quit_requested, void *p_userdata);

LIBGODOT_API bool libgodot_ios_is_created(void);
LIBGODOT_API bool libgodot_ios_is_running(void);

// Like closing the window: the main window receives WINDOW_EVENT_CLOSE_REQUEST, so the scene tree sees
// NOTIFICATION_WM_CLOSE_REQUEST and quits unless the game handles it.
LIBGODOT_API void libgodot_ios_request_quit(void);

// UIApplication lifecycle. libgodot_ios_create registers NSNotificationCenter observers that call these, so hosts
// only call them for lifecycles the notifications do not cover.
LIBGODOT_API void libgodot_ios_on_focus_in(void);
LIBGODOT_API void libgodot_ios_on_focus_out(void);
LIBGODOT_API void libgodot_ios_on_enter_background(void);
LIBGODOT_API void libgodot_ios_on_exit_background(void);
LIBGODOT_API void libgodot_ios_on_memory_warning(void);

// Main::cleanup and OS teardown; also run from UIApplicationWillTerminateNotification.
LIBGODOT_API void libgodot_ios_terminate(void);

#ifdef __cplusplus
}
#endif
