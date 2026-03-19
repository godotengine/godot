#include "test_macros.h"

#include "../interfaces/overflow_auto_tuner.h"
#include "../interfaces/rasterizer_interfaces.h"

TEST_CASE("[GaussianSplatting][OverflowAutoTuner] Discards stale async overflow snapshots after grace window") {
	OverflowAutoTuner tuner;
	OverflowAutoTuneConfig config = tuner.get_config();
	config.warmup_frames = 0;
	config.cooldown_frames = 0;
	config.ema_alpha = 1.0f;
	config.trigger_ratio = 0.01f;
	config.importance_step = 0.01f;
	config.tiny_step = 0.25f;
	tuner.set_config(config);
	tuner.set_baselines(0.0f, 0.5f);

	RasterOverflowStats stats;
	stats.overflow_tile_count = 8;
	stats.overflow_splats_clamped = 100;
	stats.overflow_splats_aggregated = 100;
	stats.frame_number = 42;

	CHECK(tuner.apply_feedback(stats, 100, 16).parameters_changed);
	tuner.apply_feedback(stats, 100, 16);
	tuner.apply_feedback(stats, 100, 16);

	const float importance_before_stale_drop = tuner.get_importance_threshold();
	const float tiny_before_stale_drop = tuner.get_tiny_splat_radius();

	const AutoTuneResult stale_result = tuner.apply_feedback(stats, 100, 16);
	CHECK_FALSE(stale_result.parameters_changed);
	CHECK_EQ(tuner.get_importance_threshold(), doctest::Approx(importance_before_stale_drop));
	CHECK_EQ(tuner.get_tiny_splat_radius(), doctest::Approx(tiny_before_stale_drop));

	stats.frame_number = 43;
	const AutoTuneResult fresh_result = tuner.apply_feedback(stats, 100, 16);
	CHECK(fresh_result.parameters_changed);
	CHECK(tuner.get_importance_threshold() > importance_before_stale_drop);
	CHECK(tuner.get_tiny_splat_radius() > tiny_before_stale_drop);
}
