/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#if defined(LINUXBSD_ENABLED)
#include "camera_linux.h"
#if defined(PIPEWIRE_ENABLED)
#include "camera_pipewire.h"
#endif
#endif
#if defined(WINDOWS_ENABLED)
#include "camera_win.h"
#endif
#if defined(MACOS_ENABLED)
#include "camera_macos.h"
#endif
#if defined(ANDROID_ENABLED)
#include "camera_android.h"
#endif

void initialize_camera_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

#if defined(LINUXBSD_ENABLED)
#if defined(PIPEWIRE_ENABLED)
#if defined(SOWRAP_ENABLED)
#if defined(DEBUG_ENABLED)
	int dylibloader_verbose = 1;
#else
	int dylibloader_verbose = 0;
#endif // defined(DEBUG_ENABLED)
	if (initialize_pipewire(dylibloader_verbose) == 0) {
		print_verbose("CameraServer: Using PipeWire driver.");
		CameraServer::make_default<CameraPipeWire>();
	} else {
		print_verbose("CameraServer: Using V4L2 driver.");
		CameraServer::make_default<CameraLinux>();
	}
#else
	print_verbose("CameraServer: Using PipeWire driver.");
	CameraServer::make_default<CameraPipeWire>();
#endif // defined(SOWRAP_ENABLED)
#else
	print_verbose("CameraServer: Using V4L2 driver.");
	CameraServer::make_default<CameraLinux>();
#endif // defined(PIPEWIRE_ENABLED)
#endif // defined(LINUXBSD_ENABLED)
#if defined(WINDOWS_ENABLED)
	CameraServer::make_default<CameraWindows>();
#endif
#if defined(MACOS_ENABLED)
	CameraServer::make_default<CameraMacOS>();
#endif
#if defined(ANDROID_ENABLED)
	CameraServer::make_default<CameraAndroid>();
#endif
}

void uninitialize_camera_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
