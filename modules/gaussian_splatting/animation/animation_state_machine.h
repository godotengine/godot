#ifndef GAUSSIAN_ANIMATION_STATE_MACHINE_H
#define GAUSSIAN_ANIMATION_STATE_MACHINE_H

#include "core/io/resource.h"
#include "core/variant/variant.h"
#include "core/math/vector3.h"
#include "core/math/color.h"
#include "core/math/quaternion.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "keyframe_interpolator.h"
#include "../persistence/incremental_saver.h"

namespace GaussianSplatting {

// Note: Underlying type specified to match forward declaration in incremental_saver.h
enum AnimationProperty : int {
    ANIMATION_PROPERTY_POSITION,
    ANIMATION_PROPERTY_COLOR,
    ANIMATION_PROPERTY_OPACITY,
    ANIMATION_PROPERTY_SCALE,
    ANIMATION_PROPERTY_ROTATION
};

enum AnimationState {
    ANIMATION_STATE_STOPPED,
    ANIMATION_STATE_PLAYING,
    ANIMATION_STATE_PAUSED,
    ANIMATION_STATE_SEEKING
};

class GaussianIncrementalSaver;

struct AnimationTrack {
    AnimationProperty property;
    LocalVector<Keyframe> keyframes;
    bool enabled = true;
    float weight = 1.0f;

    AnimationTrack() = default;
    AnimationTrack(AnimationProperty p_property) : property(p_property) {}
};

struct AnimationClip {
    String name;
    float duration = 1.0f;
    bool looping = false;
    LocalVector<AnimationTrack> tracks;

    AnimationClip() = default;
    AnimationClip(const String& p_name, float p_duration)
        : name(p_name), duration(p_duration) {}

    void add_track(AnimationProperty property);
    AnimationTrack* get_track(AnimationProperty property);
    const AnimationTrack* get_track(AnimationProperty property) const;
};

class GaussianAnimationStateMachine : public Resource {
    GDCLASS(GaussianAnimationStateMachine, Resource);

private:
    // Animation state
    AnimationState state = ANIMATION_STATE_STOPPED;
    float current_time = 0.0f;
    float playback_speed = 1.0f;

    // Animation clips
    LocalVector<AnimationClip> clips;
    HashMap<String, int> clip_name_to_index;
    int current_clip_index = -1;

    // Blending system
    struct BlendTarget {
        int clip_index;
        float target_weight;       // Desired final weight for this clip.
        float transition_time;     // Elapsed time since blend started.
        float transition_duration; // Total blend duration in seconds.
        float blend_progress;      // Normalized progress [0,1] = transition_time / transition_duration.
    };
    LocalVector<BlendTarget> blend_targets;

    // Splat data size for validation
    int splat_count = 0;

    // Interpolation cache for performance
    mutable KeyframeInterpolator interpolator;

    Ref<GaussianIncrementalSaver> incremental_saver;

    // Internal methods
    void _validate_clip_index(int p_index) const;
    void _update_blend_weights(float p_delta);
    Variant _sample_track_at_time(const AnimationTrack& track, float time, int splat_index) const;

protected:
    static void _bind_methods();

public:
    GaussianAnimationStateMachine();
    ~GaussianAnimationStateMachine();

    // Clip management
    int add_clip(const String& p_name, float p_duration = 1.0f);
    void remove_clip(int p_index);
    void remove_clip_by_name(const String& p_name);
    int get_clip_count() const { return clips.size(); }
    String get_clip_name(int p_index) const;
    float get_clip_duration(int p_index) const;
    void set_clip_duration(int p_index, float p_duration);
    bool get_clip_looping(int p_index) const;
    void set_clip_looping(int p_index, bool p_looping);

    // Track management
    void add_track_to_clip(int p_clip_index, AnimationProperty p_property);
    void remove_track_from_clip(int p_clip_index, AnimationProperty p_property);
    bool has_track(int p_clip_index, AnimationProperty p_property) const;

    // Keyframe management
    void add_keyframe(int p_clip_index, AnimationProperty p_property, float p_time, const Variant& p_value);
    void add_keyframe_bezier(int p_clip_index, AnimationProperty p_property, float p_time, const Variant& p_value, const Vector2& p_in_handle, const Vector2& p_out_handle);
    void remove_keyframe(int p_clip_index, AnimationProperty p_property, int p_keyframe_index);
    int get_keyframe_count(int p_clip_index, AnimationProperty p_property) const;
    float get_keyframe_time(int p_clip_index, AnimationProperty p_property, int p_keyframe_index) const;
    Variant get_keyframe_value(int p_clip_index, AnimationProperty p_property, int p_keyframe_index) const;

    // Playback control
    void play(int p_clip_index = -1);
    void pause();
    void stop();
    void seek(float p_time);
    void set_playback_speed(float p_speed) { playback_speed = p_speed; }
    float get_playback_speed() const { return playback_speed; }

    // State queries
    AnimationState get_state() const { return state; }
    float get_current_time() const { return current_time; }
    int get_current_clip() const { return current_clip_index; }
    bool is_playing() const { return state == ANIMATION_STATE_PLAYING; }

    // Blending system
    void blend_to_clip(int p_clip_index, float p_blend_duration = 0.3f);
    void set_clip_weight(int p_clip_index, float p_weight);
    float get_clip_weight(int p_clip_index) const;

    // Update and sampling
    void update(float p_delta);
    void set_splat_count(int p_count) { splat_count = p_count; }
    int get_splat_count() const { return splat_count; }

    // Sample animation values for specific splats
    Vector3 sample_position(int p_splat_index, float p_time = -1.0f) const;
    Color sample_color(int p_splat_index, float p_time = -1.0f) const;
    float sample_opacity(int p_splat_index, float p_time = -1.0f) const;
    Vector3 sample_scale(int p_splat_index, float p_time = -1.0f) const;
    Quaternion sample_rotation(int p_splat_index, float p_time = -1.0f) const;

    // Internal sampling helpers used by engine-side integration
    bool has_active_track(AnimationProperty p_property) const;
    bool try_sample_position(int p_splat_index, float p_time, Vector3 &r_out_position) const;
    bool try_sample_color(int p_splat_index, float p_time, Color &r_out_color) const;
    bool try_sample_opacity(int p_splat_index, float p_time, float &r_out_opacity) const;
    bool try_sample_scale(int p_splat_index, float p_time, Vector3 &r_out_scale) const;
    bool try_sample_rotation(int p_splat_index, float p_time, Quaternion &r_out_rotation) const;

    // Batch sampling for performance
    void sample_positions_batch(LocalVector<Vector3>& p_out_positions, float p_time = -1.0f) const;
    void sample_colors_batch(LocalVector<Color>& p_out_colors, float p_time = -1.0f) const;
    void sample_opacities_batch(LocalVector<float>& p_out_opacities, float p_time = -1.0f) const;

    // Serialization support
    Dictionary to_dict() const;
    void from_dict(const Dictionary& p_dict);

    void set_incremental_saver(const Ref<GaussianIncrementalSaver>& p_saver) { incremental_saver = p_saver; }
    Ref<GaussianIncrementalSaver> get_incremental_saver() const { return incremental_saver; }
};

} // namespace GaussianSplatting

VARIANT_ENUM_CAST(GaussianSplatting::AnimationProperty);
VARIANT_ENUM_CAST(GaussianSplatting::AnimationState);

#endif // GAUSSIAN_ANIMATION_STATE_MACHINE_H
