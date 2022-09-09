/*************************************************************************/
/*  godot_profiler.h                                                     */
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

#ifndef GODOT_PROFILER_H
#define GODOT_PROFILER_H

#ifdef ENABLE_PERFETTO

#include "perfetto.h"

void godot_profiler_init(void);

#define GODOT_PROFILER_INIT() godot_profiler_init()

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("godot_main").SetDescription("Godot Main"),
    perfetto::Category("godot_physics").SetDescription("Godot Physics"),
    perfetto::Category("godot_input").SetDescription("Godot Input"),
    perfetto::Category("godot_audio").SetDescription("Godot Audio"),
    perfetto::Category("godot_scene").SetDescription("Godot Scene"),
    perfetto::Category("godot_rendering").SetDescription("Godot Rendering"),
    perfetto::Category("app_input").SetDescription("App Input"),
    perfetto::Category("app_main").SetDescription("App Main"),
    perfetto::Category("app_scene").SetDescription("App Scene"),
    perfetto::Category("app_physics").SetDescription("App Physics")
);

#else

#define GODOT_PROFILER_INIT() ((void)0)

#define TRACE_EVENT_BEGIN(...) ((void)0)
#define TRACE_EVENT_END(...) ((void)0)

#define TRACE_EVENT(...) ((void)0)

#define TRACE_COUNTER(...) ((void)0)

#endif

#endif
