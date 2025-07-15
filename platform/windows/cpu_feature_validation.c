/**************************************************************************/
/*  cpu_feature_validation.c                                              */
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

#include <windows.h>

#ifdef WINDOWS_SUBSYSTEM_CONSOLE
extern int WINAPI mainCRTStartup();
#else
extern int WINAPI WinMainCRTStartup();
#endif

extern int WINAPI ShimMainCRTStartup() {
	if (IsProcessorFeaturePresent(PF_SSE4_2_INSTRUCTIONS_AVAILABLE)) {
#ifdef WINDOWS_SUBSYSTEM_CONSOLE
		return mainCRTStartup();
#else
		return WinMainCRTStartup();
#endif
	} else {
		MessageBoxW(NULL, L"A CPU with SSE4.2 instruction set support is required.", L"Godot Engine", MB_OK | MB_ICONEXCLAMATION | MB_TASKMODAL);
		return -1;
	}
}
