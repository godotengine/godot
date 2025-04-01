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
#include "modules/modules_enabled.gen.h"

#ifdef MODULE_PROFILER_ENABLED
#include "modules/profiler/profiling.gen.h"
#endif

#if defined(GODOT_USE_TRACY)
// Use the tracy profiler.

#define TRACY_ENABLE
#include <tracy/Tracy.hpp>

#ifndef TRACY_CALLSTACK
#define TRACY_CALLSTACK 0
#endif

// Define tracing macros.
#define GodotProfileFrameMark FrameMark
#define GodotProfileZone(m_zone_name) ZoneScopedN(m_zone_name)
#define GodotProfileZoneGroupedFirst(m_group_name, m_zone_name) ZoneNamedN(__godot_tracy_zone_##m_group_name, m_zone_name, true)
#define GodotProfileZoneGroupedEndEarly(m_group_name, m_zone_name) __godot_tracy_zone_##m_group_name.~ScopedZone();
#define GodotProfileZoneGrouped(m_group_name, m_zone_name)                                                                                                       \
	GodotProfileZoneGroupedEndEarly(m_group_name, m_zone_name);                                                                                                  \
	static constexpr tracy::SourceLocationData TracyConcat(__tracy_source_location, TracyLine){ m_zone_name, TracyFunction, TracyFile, (uint32_t)TracyLine, 0 }; \
	new (&__godot_tracy_zone_##m_group_name) tracy::ScopedZone(&TracyConcat(__tracy_source_location, TracyLine), TRACY_CALLSTACK, true)

void godot_init_profiler();

#elif defined(GODOT_USE_PERFETTO)
// Use the perfetto profiler.

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
		perfetto::Category("general")
				.SetDescription("All events"), );

// See PERFETTO_INTERNAL_SCOPED_EVENT_FINALIZER
struct PerfettoGroupedEventEnder {
	size_t category_idx;
	bool is_dynamic_category;

	template <size_t NC>
	_FORCE_INLINE_ explicit PerfettoGroupedEventEnder(const char (&category)[NC]) :
			category_idx(PERFETTO_GET_CATEGORY_INDEX(category)),
			is_dynamic_category(::PERFETTO_TRACK_EVENT_NAMESPACE::internal::IsDynamicCategory(category)) {
	}

	_FORCE_INLINE_ void _end_now() {
		// See TRACE_EVENT_END (perfetto v49.0)
		// The macro needed to be expanded so that we can pass category_idx and is_dynamic_category
		//  as dynamic values. The macro expects constexpr literals, which we do not have at this point.
		// When perfetto is updated, the internals may change, and this code may have to be revisited.
		::perfetto::internal::ValidateEventNameType<decltype(nullptr)>();
		namespace tns = PERFETTO_TRACK_EVENT_NAMESPACE;
		if (is_dynamic_category) {
			tns::TrackEvent::CallIfEnabled([&](uint32_t instances) PERFETTO_NO_THREAD_SAFETY_ANALYSIS {
				tns::TrackEvent::TraceForCategory(instances, this->category_idx,
						::perfetto::internal::DecayEventNameType(nullptr),
						::perfetto::protos::pbzero::TrackEvent::TYPE_SLICE_END);
			});
		} else {
			tns::TrackEvent::CallIfCategoryEnabled(this->category_idx,
					[&](uint32_t instances) PERFETTO_NO_THREAD_SAFETY_ANALYSIS {
						tns::TrackEvent::TraceForCategory(
								instances, this->category_idx,
								::perfetto::internal::DecayEventNameType(nullptr),
								::perfetto::protos::pbzero::TrackEvent::TYPE_SLICE_END);
					});
		}
	}

	_FORCE_INLINE_ ~PerfettoGroupedEventEnder() {
		_end_now();
	}

	template <size_t NC>
	_FORCE_INLINE_ void change(const char (&category)[NC]) {
		_end_now();
		this->category_idx = PERFETTO_GET_CATEGORY_INDEX(category);
		this->is_dynamic_category = ::PERFETTO_TRACK_EVENT_NAMESPACE::internal::IsDynamicCategory(category);
	}
};

#define GodotProfileFrameMark // TODO
#define GodotProfileZone(m_zone_name) TRACE_EVENT("general", m_zone_name);
#define GodotProfileZoneGroupedFirst(m_group_name, m_zone_name) \
	TRACE_EVENT_BEGIN("general", m_zone_name);                  \
	PerfettoGroupedEventEnder __godot_perfetto_zone_##m_group_name("general")
#define GodotProfileZoneGroupedEndEarly(m_group_name, m_zone_name) __godot_perfetto_zone_##m_group_name.~PerfettoGroupedEventEnder()
#define GodotProfileZoneGrouped(m_group_name, m_zone_name) \
	TRACE_EVENT_BEGIN("general", m_zone_name);             \
	__godot_perfetto_zone_##m_group_name.change("general")

void godot_init_profiler();

#else
// No profiling; all macros are stubs.

void godot_init_profiler();

#define GodotProfileFrameMark
#define GodotProfileZone(m_zone_name)
#define GodotProfileZoneGroupedFirst(m_group_name, m_zone_name)
#define GodotProfileZoneGroupedEndEarly(m_group_name, m_zone_name)
#define GodotProfileZoneGrouped(m_group_name, m_zone_name)

#endif
