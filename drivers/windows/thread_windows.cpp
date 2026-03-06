/**************************************************************************/
/*  thread_windows.cpp                                                    */
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

#ifdef WINDOWS_ENABLED

#include "thread_windows.h"

#include "core/os/thread.h"
#include "core/string/ustring.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_AddressOfReturnAddress)
#endif

typedef HRESULT(WINAPI *SetThreadDescriptionPtr)(HANDLE p_thread, PCWSTR p_thread_description);
SetThreadDescriptionPtr w10_SetThreadDescription = nullptr;

static Error set_name(const String &p_name) {
	HANDLE hThread = GetCurrentThread();
	HRESULT res = E_FAIL;
	if (w10_SetThreadDescription) {
		res = w10_SetThreadDescription(hThread, (LPCWSTR)p_name.utf16().get_data()); // Windows 10 Redstone (1607) only.
	}
	return SUCCEEDED(res) ? OK : ERR_INVALID_PARAMETER;
}

static bool get_stack_limits(void **r_bottom, void **r_top, void **r_frame) {
	ULONG_PTR stack_lo = 0;
	ULONG_PTR stack_hi = 0;
	GetCurrentThreadStackLimits(&stack_lo, &stack_hi);
	SYSTEM_INFO sys_info;
	GetSystemInfo(&sys_info);

	if (stack_lo && stack_hi) {
		if (r_bottom) {
			*r_bottom = (uint8_t *)stack_hi;
		}
		if (r_top) {
			*r_top = (uint8_t *)stack_lo + sys_info.dwPageSize; // Add guard page size.
		}
		if (r_frame) {
#ifdef _MSC_VER
			*r_frame = _AddressOfReturnAddress();
#else
			*r_frame = __builtin_frame_address(0);
#endif
		}
		return true;
	} else {
		return false;
	}
}

void init_thread_win() {
	w10_SetThreadDescription = (SetThreadDescriptionPtr)(void *)GetProcAddress(LoadLibraryW(L"kernel32.dll"), "SetThreadDescription");

	Thread::_set_platform_functions({ set_name, get_stack_limits });
}

#endif // WINDOWS_ENABLED
