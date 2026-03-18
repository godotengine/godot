/**************************************************************************/
/*  test_utils.h                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "core/os/os.h"
#include "core/string/print_string.h"

namespace TestGaussianSplatting {

// Performance measurement utility
class PerformanceTimer {
private:
	uint64_t start_time;
	String name;

public:
	PerformanceTimer(const String &p_name) : name(p_name) {
		start_time = OS::get_singleton()->get_ticks_usec();
	}

	float elapsed_ms() const {
		uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start_time;
		return elapsed / 1000.0f;
	}

	void print_elapsed() const {
		print_line(vformat("[%s] Elapsed: %.2f ms", name, elapsed_ms()));
	}
};

// GPU memory tracker utility
class GPUMemoryTracker {
private:
	RenderingDevice *rd;
	uint64_t initial_memory;

public:
	GPUMemoryTracker(RenderingDevice *p_rd) : rd(p_rd) {
		if (rd) {
			initial_memory = rd->get_memory_usage(RenderingDevice::MEMORY_TOTAL);
		}
	}

	float get_memory_usage_mb() const {
		if (!rd) return 0.0f;
		uint64_t current = rd->get_memory_usage(RenderingDevice::MEMORY_TOTAL);
		return (current - initial_memory) / (1024.0f * 1024.0f);
	}

	void print_usage() const {
		print_line(vformat("GPU Memory Usage: %.2f MB", get_memory_usage_mb()));
	}
};

} // namespace TestGaussianSplatting