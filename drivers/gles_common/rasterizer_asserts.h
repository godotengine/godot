/*************************************************************************/
/*  rasterizer_asserts.h                                                 */
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

#ifndef RASTERIZER_ASSERTS_H
#define RASTERIZER_ASSERTS_H

// For flow control checking, we want an easy way to apply asserts that occur in debug development builds only.
// This is enforced by outputting a warning which will fail CI checks if the define is set in a PR.
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
// only uncomment this define for error checking in development, not in the main repository
// as these checks will slow things down in debug builds.
//#define RASTERIZER_EXTRA_CHECKS
#endif

#ifdef RASTERIZER_EXTRA_CHECKS
#ifndef _MSC_VER
#warning do not define RASTERIZER_EXTRA_CHECKS in main repository builds
#endif
#define RAST_DEV_DEBUG_ASSERT(a) CRASH_COND(!(a))
#else
#define RAST_DEV_DEBUG_ASSERT(a)
#endif

// Also very useful, an assert check that only occurs in debug tools builds
#if defined(TOOLS_ENABLED) && defined(DEBUG_ENABLED)
#define RAST_DEBUG_ASSERT(a) CRASH_COND(!(a))
#else
#define RAST_DEBUG_ASSERT(a)
#endif

// Thin wrapper around ERR_FAIL_COND to allow us to make it debug only
#ifdef DEBUG_ENABLED
#define RAST_FAIL_COND(m_cond) ERR_FAIL_COND(m_cond)
#else
#define RAST_FAIL_COND(m_cond) \
	if (m_cond) {              \
	}
#endif

#endif // RASTERIZER_ASSERTS_H
