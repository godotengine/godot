#pragma once

#include "test_macros.h"
#include "../animation/animation_state_machine.h"
#include "core/math/math_funcs.h"

TEST_CASE("[GaussianSplatting][Animation] linear keyframe interpolation") {
    using namespace GaussianSplatting;

    GaussianAnimationStateMachine state_machine;
    int clip_index = state_machine.add_clip("linear", 1.0f);
    state_machine.set_splat_count(1);

    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_OPACITY, 0.0f, 0.0f);
    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_OPACITY, 1.0f, 10.0f);
    state_machine.play(clip_index);

    CHECK(Math::is_equal_approx(state_machine.sample_opacity(0, 0.0f), 0.0f));
    CHECK(Math::is_equal_approx(state_machine.sample_opacity(0, 0.5f), 5.0f));
    CHECK(Math::is_equal_approx(state_machine.sample_opacity(0, 1.0f), 10.0f));

    // Test extrapolation (should clamp)
    CHECK(Math::is_equal_approx(state_machine.sample_opacity(0, -0.5f), 0.0f));
    CHECK(Math::is_equal_approx(state_machine.sample_opacity(0, 1.5f), 10.0f));
}

TEST_CASE("[GaussianSplatting][Animation] Vector3 keyframe interpolation") {
    using namespace GaussianSplatting;

    GaussianAnimationStateMachine state_machine;
    int clip_index = state_machine.add_clip("vector", 1.0f);
    state_machine.set_splat_count(1);

    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_POSITION, 0.0f, Vector3(0, 0, 0));
    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_POSITION, 1.0f, Vector3(10, 20, 30));
    state_machine.play(clip_index);

    Vector3 mid = state_machine.sample_position(0, 0.5f);
    CHECK(mid.is_equal_approx(Vector3(5, 10, 15)));
}

TEST_CASE("[GaussianSplatting][Animation] splat-indexed keyframes sample per-splat values") {
    using namespace GaussianSplatting;

    GaussianAnimationStateMachine state_machine;
    int clip_index = state_machine.add_clip("per_splat_position", 1.0f);
    state_machine.set_splat_count(3);

    PackedVector3Array start_positions;
    start_positions.push_back(Vector3(0, 0, 0));
    start_positions.push_back(Vector3(10, 0, 0));
    start_positions.push_back(Vector3(20, 0, 0));

    PackedVector3Array end_positions;
    end_positions.push_back(Vector3(10, 0, 0));
    end_positions.push_back(Vector3(20, 0, 0));
    end_positions.push_back(Vector3(30, 0, 0));

    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_POSITION, 0.0f, start_positions);
    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_POSITION, 1.0f, end_positions);
    state_machine.play(clip_index);

    CHECK(state_machine.sample_position(0, 0.5f).is_equal_approx(Vector3(5, 0, 0)));
    CHECK(state_machine.sample_position(1, 0.5f).is_equal_approx(Vector3(15, 0, 0)));
    CHECK(state_machine.sample_position(2, 0.5f).is_equal_approx(Vector3(25, 0, 0)));
}

TEST_CASE("[GaussianSplatting][Animation] mixed scalar and per-splat keyframes only reject missing indices in sampled window") {
    using namespace GaussianSplatting;

    GaussianAnimationStateMachine state_machine;
    int clip_index = state_machine.add_clip("mixed_window", 2.0f);
    state_machine.set_splat_count(2);

    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_POSITION, 0.0f, Vector3(0, 0, 0));
    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_POSITION, 1.0f, Vector3(10, 0, 0));

    PackedVector3Array per_splat_tail;
    per_splat_tail.push_back(Vector3(30, 0, 0)); // Only splat 0 provided.
    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_POSITION, 2.0f, per_splat_tail);
    state_machine.play(clip_index);

    // Sample window [0, 1] uses scalar keyframes; missing splat-1 entry at t=2 should not invalidate this.
    CHECK(state_machine.sample_position(1, 0.5f).is_equal_approx(Vector3(5, 0, 0)));
    CHECK(state_machine.sample_position(1, 1.0f).is_equal_approx(Vector3(10, 0, 0)));

    // Sample window [1, 2] includes a per-splat keyframe without splat-1 data and should fail gracefully.
    CHECK(state_machine.sample_position(1, 1.5f).is_equal_approx(Vector3()));
}

TEST_CASE("[GaussianSplatting][Animation] global track sampling stays index-agnostic in batch APIs") {
    using namespace GaussianSplatting;

    GaussianAnimationStateMachine state_machine;
    int clip_index = state_machine.add_clip("batch", 1.0f);
    state_machine.set_splat_count(4);

    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_OPACITY, 0.0f, 0.0f);
    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_OPACITY, 1.0f, 1.0f);
    state_machine.play(clip_index);

    const float sample_time = 0.25f;
    const float expected = state_machine.sample_opacity(0, sample_time);
    CHECK(Math::is_equal_approx(state_machine.sample_opacity(3, sample_time), expected));

    LocalVector<float> opacities;
    state_machine.sample_opacities_batch(opacities, sample_time);

    REQUIRE(opacities.size() == 4);
    for (uint32_t i = 0; i < opacities.size(); i++) {
        CHECK(Math::is_equal_approx(opacities[i], expected));
    }
}

TEST_CASE("[GaussianSplatting][Animation] integer opacity keyframes sample consistently across scalar and batch APIs") {
    using namespace GaussianSplatting;

    GaussianAnimationStateMachine state_machine;
    int clip_index = state_machine.add_clip("int_opacity", 1.0f);
    state_machine.set_splat_count(3);

    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_OPACITY, 0.0f, 0);
    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_OPACITY, 1.0f, 2);
    state_machine.play(clip_index);

    const float sample_time = 0.5f;
    const float expected = state_machine.sample_opacity(0, sample_time);
    CHECK(Math::is_equal_approx(expected, 1.0f));
    CHECK(Math::is_equal_approx(state_machine.sample_opacity(2, sample_time), expected));

    LocalVector<float> opacities;
    state_machine.sample_opacities_batch(opacities, sample_time);
    REQUIRE(opacities.size() == 3);
    for (uint32_t i = 0; i < opacities.size(); i++) {
        CHECK(Math::is_equal_approx(opacities[i], expected));
    }
}

TEST_CASE("[GaussianSplatting][Animation] mixed numeric opacity keyframes interpolate linearly") {
    using namespace GaussianSplatting;

    GaussianAnimationStateMachine state_machine;
    int clip_index = state_machine.add_clip("mixed_numeric_opacity", 1.0f);
    state_machine.set_splat_count(1);

    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_OPACITY, 0.0f, 0);
    state_machine.add_keyframe(clip_index, ANIMATION_PROPERTY_OPACITY, 1.0f, 2.0f);
    state_machine.play(clip_index);

    CHECK(Math::is_equal_approx(state_machine.sample_opacity(0, 0.5f), 1.0f));
}
TEST_CASE("[GaussianSplatting][Animation] state machine transitions") {
    using namespace GaussianSplatting;

    GaussianAnimationStateMachine state_machine;
    int clip_index = state_machine.add_clip("idle", 1.0f);
    state_machine.set_splat_count(1);

    CHECK_EQ(state_machine.get_state(), ANIMATION_STATE_STOPPED);

    state_machine.play(clip_index);
    CHECK_EQ(state_machine.get_state(), ANIMATION_STATE_PLAYING);

    state_machine.pause();
    CHECK_EQ(state_machine.get_state(), ANIMATION_STATE_PAUSED);

    state_machine.stop();
    CHECK_EQ(state_machine.get_state(), ANIMATION_STATE_STOPPED);

    state_machine.seek(0.25f);
    CHECK_EQ(state_machine.get_state(), ANIMATION_STATE_SEEKING);
}
