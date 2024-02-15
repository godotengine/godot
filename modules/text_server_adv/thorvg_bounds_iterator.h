/**************************************************************************/
/*  thorvg_bounds_iterator.h                                              */
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

#ifndef THORVG_BOUNDS_ITERATOR_H
#define THORVG_BOUNDS_ITERATOR_H

#ifdef GDEXTENSION
// Headers for building as GDExtension plug-in.

#include <godot_cpp/core/mutex_lock.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

#elif defined(GODOT_MODULE)
// Headers for building as built-in module.

#include "core/typedefs.h"

#include "modules/modules_enabled.gen.h" // For svg.
#endif

#ifdef MODULE_SVG_ENABLED

#include <thorvg.h>

void tvg_get_bounds(tvg::Picture *p_picture, float &r_min_x, float &r_min_y, float &r_max_x, float &r_max_y);

#endif // MODULE_SVG_ENABLED

#endif // THORVG_BOUNDS_ITERATOR_H
