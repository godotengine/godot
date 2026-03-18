/**************************************************************************/
/*  test_sort_benchmark_metrics.h                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "../renderer/sort_benchmark_metrics.h"

#include "tests/test_macros.h"

namespace TestGaussianSplatting {

TEST_CASE("[GaussianSplatting][SortBenchmark] Timing metrics follow observed submit+wait latency") {
	const GaussianSplatting::SortBenchmarkTimingMetrics timing =
			GaussianSplatting::compute_sort_benchmark_timing(
					0.40f, 1.60f, 0.05f, 7);

	CHECK(timing.async_requested);
	CHECK(timing.waited_for_completion);
	CHECK(timing.used_async);
	CHECK_EQ(timing.submit_ms, doctest::Approx(0.40f));
	CHECK_EQ(timing.wait_ms, doctest::Approx(1.60f));
	CHECK_EQ(timing.gpu_ms, doctest::Approx(2.00f));
}

TEST_CASE("[GaussianSplatting][SortBenchmark] Async token without measured wait is not reported as used_async") {
	const GaussianSplatting::SortBenchmarkTimingMetrics timing =
			GaussianSplatting::compute_sort_benchmark_timing(
					2.50f, 0.01f, 0.50f, 3);

	CHECK(timing.async_requested);
	CHECK_FALSE(timing.waited_for_completion);
	CHECK_FALSE(timing.used_async);
	CHECK_EQ(timing.gpu_ms, doctest::Approx(2.51f));
}

TEST_CASE("[GaussianSplatting][SortBenchmark] Missing async token can still report wait cost") {
	const GaussianSplatting::SortBenchmarkTimingMetrics timing =
			GaussianSplatting::compute_sort_benchmark_timing(
					1.25f, 3.75f, 0.0f, 0);

	CHECK_FALSE(timing.async_requested);
	CHECK(timing.waited_for_completion);
	CHECK_FALSE(timing.used_async);
	CHECK_EQ(timing.gpu_ms, doctest::Approx(5.00f));
}

TEST_CASE("[GaussianSplatting][SortBenchmark] Reported sorter time is fallback when observed latency is unavailable") {
	const GaussianSplatting::SortBenchmarkTimingMetrics timing =
			GaussianSplatting::compute_sort_benchmark_timing(
					0.0f, 0.0f, 0.75f, 0);

	CHECK_FALSE(timing.async_requested);
	CHECK_FALSE(timing.waited_for_completion);
	CHECK_FALSE(timing.used_async);
	CHECK_EQ(timing.gpu_ms, doctest::Approx(0.75f));
}

} // namespace TestGaussianSplatting
