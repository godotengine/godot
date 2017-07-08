/*
 * Windows backend common header for libusb 1.0
 *
 * This file brings together header code common between
 * the desktop Windows and Windows CE backends.
 * Copyright © 2012-2013 RealVNC Ltd.
 * Copyright © 2009-2012 Pete Batard <pete@akeo.ie>
 * With contributions from Michael Plante, Orin Eman et al.
 * Parts of this code adapted from libusb-win32-v1 by Stephan Meyer
 * Major code testing contribution by Xiaofan Chen
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#pragma once

// Windows API default is uppercase - ugh!
#if !defined(bool)
#define bool BOOL
#endif
#if !defined(true)
#define true TRUE
#endif
#if !defined(false)
#define false FALSE
#endif

#define EPOCH_TIME	UINT64_C(116444736000000000)	// 1970.01.01 00:00:000 in MS Filetime

#if defined(__CYGWIN__ )
#define _stricmp strcasecmp
#define _strdup strdup
// _beginthreadex is MSVCRT => unavailable for cygwin. Fallback to using CreateThread
#define _beginthreadex(a, b, c, d, e, f) CreateThread(a, b, (LPTHREAD_START_ROUTINE)c, d, e, (LPDWORD)f)
#endif

#define safe_free(p) do {if (p != NULL) {free((void *)p); p = NULL;}} while (0)

#ifndef ARRAYSIZE
#define ARRAYSIZE(A) (sizeof(A)/sizeof((A)[0]))
#endif

#define ERR_BUFFER_SIZE	256

/*
 * API macros - leveraged from libusb-win32 1.x
 */
#ifndef _WIN32_WCE
#define DLL_STRINGIFY(s) #s
#define DLL_LOAD_LIBRARY(name) LoadLibraryA(DLL_STRINGIFY(name))
#else
#define DLL_STRINGIFY(s) L#s
#define DLL_LOAD_LIBRARY(name) LoadLibrary(DLL_STRINGIFY(name))
#endif

/*
 * Macros for handling DLL themselves
 */
#define DLL_DECLARE_HANDLE(name)				\
	static HMODULE __dll_##name##_handle = NULL

#define DLL_GET_HANDLE(name)					\
	do {							\
		__dll_##name##_handle = DLL_LOAD_LIBRARY(name);	\
		if (!__dll_##name##_handle)			\
			return LIBUSB_ERROR_OTHER;		\
	} while (0)

#define DLL_FREE_HANDLE(name)					\
	do {							\
		if (__dll_##name##_handle) {			\
			FreeLibrary(__dll_##name##_handle);	\
			__dll_##name##_handle = NULL;		\
		}						\
	} while(0)


/*
 * Macros for handling functions within a DLL
 */
#define DLL_DECLARE_FUNC_PREFIXNAME(api, ret, prefixname, name, args)	\
	typedef ret (api * __dll_##name##_func_t)args;			\
	static __dll_##name##_func_t prefixname = NULL

#define DLL_DECLARE_FUNC(api, ret, name, args)				\
	DLL_DECLARE_FUNC_PREFIXNAME(api, ret, name, name, args)
#define DLL_DECLARE_FUNC_PREFIXED(api, ret, prefix, name, args)		\
	DLL_DECLARE_FUNC_PREFIXNAME(api, ret, prefix##name, name, args)

#define DLL_LOAD_FUNC_PREFIXNAME(dll, prefixname, name, ret_on_failure)	\
	do {								\
		HMODULE h = __dll_##dll##_handle;			\
		prefixname = (__dll_##name##_func_t)GetProcAddress(h,	\
				DLL_STRINGIFY(name));			\
		if (prefixname)						\
			break;						\
		prefixname = (__dll_##name##_func_t)GetProcAddress(h,	\
				DLL_STRINGIFY(name) DLL_STRINGIFY(A));	\
		if (prefixname)						\
			break;						\
		prefixname = (__dll_##name##_func_t)GetProcAddress(h,	\
				DLL_STRINGIFY(name) DLL_STRINGIFY(W));	\
		if (prefixname)						\
			break;						\
		if (ret_on_failure)					\
			return LIBUSB_ERROR_NOT_FOUND;			\
	} while(0)

#define DLL_LOAD_FUNC(dll, name, ret_on_failure)			\
	DLL_LOAD_FUNC_PREFIXNAME(dll, name, name, ret_on_failure)
#define DLL_LOAD_FUNC_PREFIXED(dll, prefix, name, ret_on_failure)	\
	DLL_LOAD_FUNC_PREFIXNAME(dll, prefix##name, name, ret_on_failure)
