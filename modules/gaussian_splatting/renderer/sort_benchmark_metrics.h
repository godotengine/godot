#ifndef GAUSSIAN_SORT_BENCHMARK_METRICS_H
#define GAUSSIAN_SORT_BENCHMARK_METRICS_H

#include <stdint.h>

namespace GaussianSplatting {

struct SortBenchmarkTimingMetrics {
	float submit_ms = 0.0f;
	float wait_ms = 0.0f;
	float gpu_ms = 0.0f;
	bool async_requested = false;
	bool waited_for_completion = false;
	bool used_async = false;
};

static inline SortBenchmarkTimingMetrics compute_sort_benchmark_timing(
		float p_submit_ms, float p_wait_ms, float p_reported_gpu_ms, uint64_t p_timeline_value) {
	constexpr float kAsyncWaitThresholdMs = 0.05f;

	SortBenchmarkTimingMetrics metrics;
	metrics.submit_ms = p_submit_ms > 0.0f ? p_submit_ms : 0.0f;
	metrics.wait_ms = p_wait_ms > 0.0f ? p_wait_ms : 0.0f;
	metrics.async_requested = p_timeline_value != 0;
	metrics.waited_for_completion = metrics.wait_ms > kAsyncWaitThresholdMs;
	metrics.used_async = metrics.async_requested && metrics.waited_for_completion;

	const float observed_total_ms = metrics.submit_ms + metrics.wait_ms;
	if (observed_total_ms > 0.0f) {
		metrics.gpu_ms = observed_total_ms;
	} else if (p_reported_gpu_ms > 0.0f) {
		metrics.gpu_ms = p_reported_gpu_ms;
	}

	return metrics;
}

} // namespace GaussianSplatting

#endif // GAUSSIAN_SORT_BENCHMARK_METRICS_H
