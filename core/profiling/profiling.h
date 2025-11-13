/**************************************************************************/
/*  profiling.h                                                           */
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

#include "core/typedefs.h"
#include "profiling.gen.h"

// This header provides profiling primitives (implemented as macros) for various backends.
// See the "No profiling" branch at the bottom for a short description of the functions.

// To configure / use the profiler, use the --profiler_path and other --profiler_* arguments
// when compiling Godot. You can also find details in the SCSub file (in this folder).

// Note: It is highly recommended to avoid including this header in other header files.
//       Prefer including it in .cpp files only. The reason is that we want to keep
//       the recompile cost of changing the profiler as low as possible.

#if defined(GODOT_USE_TRACY)
// Use the tracy profiler.

#include "char_intern.h"

#define TRACY_ENABLE
#include <tracy/Tracy.hpp>

// wanted for relaying gdscript source information to tracy
typedef tracy::SourceLocationData SourceLocationData;

// Define tracing macros.
#define GodotProfileFrameMark FrameMark
#define GodotProfileZone(m_zone_name) ZoneScopedN(m_zone_name)
#define GodotProfileZoneRename(m_zone_name) ZoneName(m_zone_name.ptr(), m_zone_name.size())
#define GodotProfileZoneGroupedFirst(m_group_name, m_zone_name) ZoneNamedN(__godot_tracy_zone_##m_group_name, m_zone_name, true)
#define GodotProfileZoneGroupedEndEarly(m_group_name, m_zone_name) __godot_tracy_zone_##m_group_name.~ScopedZone();
#ifndef TRACY_CALLSTACK
#define GodotProfileZoneGrouped(m_group_name, m_zone_name)                                                                                                       \
	GodotProfileZoneGroupedEndEarly(m_group_name, m_zone_name);                                                                                                  \
	static constexpr tracy::SourceLocationData TracyConcat(__tracy_source_location, TracyLine){ m_zone_name, TracyFunction, TracyFile, (uint32_t)TracyLine, 0 }; \
	new (&__godot_tracy_zone_##m_group_name) tracy::ScopedZone(&TracyConcat(__tracy_source_location, TracyLine), true)
#else
#define GodotProfileZoneGrouped(m_group_name, m_zone_name)                                                                                                       \
	GodotProfileZoneGroupedEndEarly(m_group_name, m_zone_name);                                                                                                  \
	static constexpr tracy::SourceLocationData TracyConcat(__tracy_source_location, TracyLine){ m_zone_name, TracyFunction, TracyFile, (uint32_t)TracyLine, 0 }; \
	new (&__godot_tracy_zone_##m_group_name) tracy::ScopedZone(&TracyConcat(__tracy_source_location, TracyLine), TRACY_CALLSTACK, true)
#endif
#define GodotProfileZoneGroupedRename(m_group_name, m_zone_name) ZoneNameV(__godot_tracy_zone_##m_group_name, m_zone_name.ptr(), m_zone_name.size())

// FIXME: I'm still trying to figure out whether I want to make another internment like thing for the source location
//   data or what.
#define GodotProfileZoneScript(m_varname, m_zone_name, m_function, m_file, m_line, m_color) \
	auto *__godot_tracy_sld_##m_varname = new SourceLocationData{                           \
		CharIntern::intern(m_zone_name),                                                    \
		CharIntern::intern(m_function),                                                     \
		CharIntern::intern(m_file),                                                         \
		static_cast<uint32_t>(m_line),                                                      \
		tracy::Color::DarkTurquoise                                                         \
	};                                                                                      \
	auto __godot_tracy_zone_##m_varname = tracy::ScopedZone(__godot_tracy_sld_##m_varname)

#define GodotProfileZoneDynamic(m_varname, m_zone_name) \
	GodotProfileZoneScript(                             \
			m_varname,                                  \
			CharIntern::intern(m_zone_name),            \
			CharIntern::intern(TracyFunction),          \
			CharIntern::intern(TracyFile),              \
			static_cast<uint32_t>(TracyLine),           \
			tracy::Color::DarkTurquoise)

void godot_init_profiler();

#elif defined(GODOT_USE_PERFETTO)
// Use the perfetto profiler.

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
		perfetto::Category("godot")
				.SetDescription("All Godot Events"), );

// See PERFETTO_INTERNAL_SCOPED_EVENT_FINALIZER
struct PerfettoGroupedEventEnder {
	_FORCE_INLINE_ void _end_now() {
		TRACE_EVENT_END("godot");
	}

	_FORCE_INLINE_ ~PerfettoGroupedEventEnder() {
		_end_now();
	}
};

#define GodotProfileFrameMark // TODO
#define GodotProfileZone(m_zone_name) TRACE_EVENT("godot", m_zone_name);
#define GodotProfileZoneRename(m_zone_name) // TODO
#define GodotProfileZoneGroupedFirst(m_group_name, m_zone_name) \
	TRACE_EVENT_BEGIN("godot", m_zone_name);                    \
	PerfettoGroupedEventEnder __godot_perfetto_zone_##m_group_name
#define GodotProfileZoneGroupedEndEarly(m_group_name, m_zone_name) __godot_perfetto_zone_##m_group_name.~PerfettoGroupedEventEnder()
#define GodotProfileZoneGrouped(m_group_name, m_zone_name) \
	__godot_perfetto_zone_##m_group_name._end_now();       \
	TRACE_EVENT_BEGIN("godot", m_zone_name);
#define GodotProfileZoneGroupedRename(m_group_name, m_zone_name) \\ TODO

#define GodotProfileZoneScript(m_varname, m_zone_name, m_function, m_file, m_line, m_color) \\ TODO
#define GodotProfileZoneDynamic(m_varname, m_zone_name) \\TODO

void godot_init_profiler();

#else
// No profiling; all macros are stubs.

void godot_init_profiler();

// Tell the profiling backend that a new frame has started.
#define GodotProfileFrameMark
// Defines a profile zone from here to the end of the scope.
#define GodotProfileZone(m_zone_name)
// Rename an existing zone
#define GodotProfileZoneRename(m_zone_name)
// Defines a profile zone group. The first profile zone starts immediately,
// and ends either when the next zone starts, or when the scope ends.
#define GodotProfileZoneGroupedFirst(m_group_name, m_zone_name)
// End the profile zone group's current profile zone now.
#define GodotProfileZoneGroupedEndEarly(m_group_name, m_zone_name)
// Replace the profile zone group's current profile zone.
// The new zone ends either when the next zone starts, or when the scope ends.
#define GodotProfileZoneGrouped(m_group_name, m_zone_name)
// Rename a grouped zone
#define GodotProfileZoneGroupedRename(m_group_name, m_zone_name)

// Define a zone with custom source information, for scripting, unique utf8
// strings will be copied and stored for the duration of the program.
#define GodotProfileZoneScript(m_varname, m_zone_name, m_function, m_file, m_line, m_color)
// Define a zone who's name changes dynamically, unique utf8 strings will be
// copied and stored for the duration of the program.
#define GodotProfileZoneDynamic(m_varname, m_zone_name)

#endif
