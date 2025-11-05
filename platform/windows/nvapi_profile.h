/**************************************************************************/
/*  nvapi_profile.h                                                       */
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

#include "core/variant/dictionary.h"

const int OGL_THREAD_CONTROL_ID = 0x20C1221E;
const int OGL_THREAD_CONTROL_DISABLE = 0x00000002;
const int OGL_THREAD_CONTROL_ENABLE = 0x00000001;
const int VRR_MODE_ID = 0x1194F158;
const int VRR_MODE_FULLSCREEN_ONLY = 0x1;
const int OGL_CPL_PREFER_DXPRESENT_ID = 0x20D690F8;
const int OGL_CPL_PREFER_DXPRESENT_PREFER_DISABLED = 0x00000000;
const int OGL_CPL_PREFER_DXPRESENT_PREFER_ENABLED = 0x00000001;
const int OGL_CPL_PREFER_DXPRESENT_AUTO = 0x00000002;

bool nvapi_setup_profile(const Dictionary &p_props);
