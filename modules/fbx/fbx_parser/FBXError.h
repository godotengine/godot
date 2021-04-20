/*************************************************************************/
/*  FBXError.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

//
// Created by Gordon MacPherson on 20/04/2021.
//

#ifndef FBX_ERROR_H
#define FBX_ERROR_H

#include "core/string/print_string.h"

#define FBX_ERROR_DETECTED FBXError::IsCorrupt()
#define IF_FBX_IS_CORRUPT_RETURN \
	if (FBXError::IsCorrupt()) { \
		return;                  \
	}
#define IS_FBX_CORRUPT FBXError::IsCorrupt()
#define FBX_CORRUPT FBXError::SetCorrupt()

#define FBX_CORRUPT_ERROR_PTR \
	FBX_CORRUPT;              \
	return nullptr;

#define FBX_CORRUPT_ERROR_BOOL \
	if (IS_FBX_CORRUPT) {      \
		return false;          \
	}

struct FBXError {
	static bool IsCorrupt() {
		return FBXError::corrupt;
	}

	static void ClearCorrupt() {
		FBXError::corrupt = false;
	}

	static void SetCorrupt() {
		print_error("FBX Document was found to be corrupt, we have decided to stop all operations");
		FBXError::corrupt = true;
	}

	// NOTE: thread local should be required, but may crash since godot is not in the same static lib
	thread_local static bool corrupt;
};

#endif // FBX_ERROR_H
