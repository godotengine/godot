/**************************************************************************/
/*  detect_prime_egl.h                                                    */
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

#ifndef DETECT_PRIME_EGL_H
#define DETECT_PRIME_EGL_H

#ifdef GLES3_ENABLED
#ifdef EGL_ENABLED

class DetectPrimeEGL {
private:
	struct Vendor {
		const char *glxvendor = nullptr;
		int priority = 0;
	};

	static constexpr Vendor vendor_map[] = {
		{ "Advanced Micro Devices, Inc.", 30 },
		{ "AMD", 30 },
		{ "NVIDIA Corporation", 30 },
		{ "X.Org", 30 },
		{ "Intel Open Source Technology Center", 20 },
		{ "Intel", 20 },
		{ "nouveau", 10 },
		{ "Mesa Project", 0 },
		{ nullptr, 0 }
	};

	static void create_context();

public:
	static int detect_prime();
};

#endif // GLES3_ENABLED
#endif // EGL_ENABLED

#endif // DETECT_PRIME_EGL_H
