/**************************************************************************/
/*  version.h                                                             */
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

#include "core/version_generated.gen.h"

#include <stdint.h>

// Copied from typedefs.h to stay lean.
#ifndef _STR
#define _STR(m_x) #m_x
#define _MKSTR(m_x) _STR(m_x)
#endif

// Godot versions are of the form <major>.<minor> for the initial release,
// and then <major>.<minor>.<patch> for subsequent bugfix releases where <patch> != 0
// That's arbitrary, but we find it pretty and it's the current policy.

// Defines the main "branch" version. Patch versions in this branch should be
// forward-compatible.
// Example: "3.1"
#define GODOT_VERSION_BRANCH _MKSTR(GODOT_VERSION_MAJOR) "." _MKSTR(GODOT_VERSION_MINOR)
#if GODOT_VERSION_PATCH
// Example: "3.1.4"
#define GODOT_VERSION_NUMBER GODOT_VERSION_BRANCH "." _MKSTR(GODOT_VERSION_PATCH)
#else // patch is 0, we don't include it in the "pretty" version number.
// Example: "3.1" instead of "3.1.0"
#define GODOT_VERSION_NUMBER GODOT_VERSION_BRANCH
#endif // GODOT_VERSION_PATCH

// Version number encoded as hexadecimal int with one byte for each number,
// for easy comparison from code.
// Example: 3.1.4 will be 0x030104, making comparison easy from script.
#define GODOT_VERSION_HEX 0x10000 * GODOT_VERSION_MAJOR + 0x100 * GODOT_VERSION_MINOR + GODOT_VERSION_PATCH

// Describes the full configuration of that Godot version, including the version number,
// the status (beta, stable, etc.) and potential module-specific features (e.g. mono).
// Example: "3.1.4.stable.mono"
#define GODOT_VERSION_FULL_CONFIG GODOT_VERSION_NUMBER "." GODOT_VERSION_STATUS GODOT_VERSION_MODULE_CONFIG

// Similar to GODOT_VERSION_FULL_CONFIG, but also includes the (potentially custom) GODOT_VERSION_BUILD
// description (e.g. official, custom_build, etc.).
// Example: "3.1.4.stable.mono.official"
#define GODOT_VERSION_FULL_BUILD GODOT_VERSION_FULL_CONFIG "." GODOT_VERSION_BUILD

// Same as above, but prepended with Godot's name and a cosmetic "v" for "version".
// Example: "Godot v3.1.4.stable.official.mono"
#define GODOT_VERSION_FULL_NAME GODOT_VERSION_NAME " v" GODOT_VERSION_FULL_BUILD

// Git commit hash, generated at build time in `core/version_hash.gen.cpp`.
extern const char *const GODOT_VERSION_HASH;

// Git commit date UNIX timestamp (in seconds), generated at build time in `core/version_hash.gen.cpp`.
// Set to 0 if unknown.
extern const uint64_t GODOT_VERSION_TIMESTAMP;

#ifndef DISABLE_DEPRECATED
// Compatibility with pre-4.5 modules.
#define VERSION_SHORT_NAME GODOT_VERSION_SHORT_NAME
#define VERSION_NAME GODOT_VERSION_NAME
#define VERSION_MAJOR GODOT_VERSION_MAJOR
#define VERSION_MINOR GODOT_VERSION_MINOR
#define VERSION_PATCH GODOT_VERSION_PATCH
#define VERSION_STATUS GODOT_VERSION_STATUS
#define VERSION_BUILD GODOT_VERSION_BUILD
#define VERSION_MODULE_CONFIG GODOT_VERSION_MODULE_CONFIG
#define VERSION_WEBSITE GODOT_VERSION_WEBSITE
#define VERSION_DOCS_BRANCH GODOT_VERSION_DOCS_BRANCH
#define VERSION_DOCS_URL GODOT_VERSION_DOCS_URL
#define VERSION_BRANCH GODOT_VERSION_BRANCH
#define VERSION_NUMBER GODOT_VERSION_NUMBER
#define VERSION_HEX GODOT_VERSION_HEX
#define VERSION_FULL_CONFIG GODOT_VERSION_FULL_CONFIG
#define VERSION_FULL_BUILD GODOT_VERSION_FULL_BUILD
#define VERSION_FULL_NAME GODOT_VERSION_FULL_NAME
#define VERSION_HASH GODOT_VERSION_HASH
#define VERSION_TIMESTAMP GODOT_VERSION_TIMESTAMP
#endif // DISABLE_DEPRECATED
