#include "animation_state_machine.h"
#include "core/object/class_db.h"
#include "core/string/ustring.h"
#include "../persistence/incremental_saver.h"
#include "../logger/gs_logger.h"

namespace GaussianSplatting {

namespace {

Dictionary _clip_to_dict(const AnimationClip &clip, int index) {
    Dictionary dict;
    dict["index"] = index;
    dict["name"] = clip.name;
    dict["duration"] = clip.duration;
    dict["looping"] = clip.looping;
    dict["track_count"] = clip.tracks.size();
    return dict;
}

Dictionary _track_to_dict(const AnimationTrack &track, int index) {
    Dictionary dict;
    dict["index"] = index;
    dict["property"] = (int)track.property;
    dict["enabled"] = track.enabled;
    dict["weight"] = track.weight;
    dict["keyframes"] = track.keyframes.size();
    return dict;
}

Dictionary _keyframe_to_dict(const Keyframe &keyframe, int index) {
    Dictionary dict;
    dict["index"] = index;
    dict["time"] = keyframe.time;
    dict["value"] = keyframe.value;
    dict["interpolation"] = (int)keyframe.interpolation;
    dict["in_handle"] = keyframe.in_handle;
    dict["out_handle"] = keyframe.out_handle;
    return dict;
}

bool _is_splat_container_type(Variant::Type p_type) {
    switch (p_type) {
        case Variant::ARRAY:
        case Variant::PACKED_FLOAT32_ARRAY:
        case Variant::PACKED_FLOAT64_ARRAY:
        case Variant::PACKED_INT32_ARRAY:
        case Variant::PACKED_INT64_ARRAY:
        case Variant::PACKED_VECTOR3_ARRAY:
        case Variant::PACKED_COLOR_ARRAY:
            return true;
        default:
            return false;
    }
}

bool _try_extract_splat_value(const Variant &p_value, int p_splat_index, Variant &r_value) {
    switch (p_value.get_type()) {
        case Variant::ARRAY: {
            Array values = p_value;
            if (p_splat_index < 0 || p_splat_index >= values.size()) {
                return false;
            }
            r_value = values[p_splat_index];
            return true;
        }
        case Variant::PACKED_FLOAT32_ARRAY: {
            PackedFloat32Array values = p_value;
            if (p_splat_index < 0 || p_splat_index >= values.size()) {
                return false;
            }
            r_value = values[p_splat_index];
            return true;
        }
        case Variant::PACKED_FLOAT64_ARRAY: {
            PackedFloat64Array values = p_value;
            if (p_splat_index < 0 || p_splat_index >= values.size()) {
                return false;
            }
            r_value = values[p_splat_index];
            return true;
        }
        case Variant::PACKED_INT32_ARRAY: {
            PackedInt32Array values = p_value;
            if (p_splat_index < 0 || p_splat_index >= values.size()) {
                return false;
            }
            r_value = values[p_splat_index];
            return true;
        }
        case Variant::PACKED_INT64_ARRAY: {
            PackedInt64Array values = p_value;
            if (p_splat_index < 0 || p_splat_index >= values.size()) {
                return false;
            }
            r_value = values[p_splat_index];
            return true;
        }
        case Variant::PACKED_VECTOR3_ARRAY: {
            PackedVector3Array values = p_value;
            if (p_splat_index < 0 || p_splat_index >= values.size()) {
                return false;
            }
            r_value = values[p_splat_index];
            return true;
        }
        case Variant::PACKED_COLOR_ARRAY: {
            PackedColorArray values = p_value;
            if (p_splat_index < 0 || p_splat_index >= values.size()) {
                return false;
            }
            r_value = values[p_splat_index];
            return true;
        }
        default:
            return false;
    }
}

void _find_keyframe_indices_for_sample(const LocalVector<Keyframe> &keyframes, float time, int &index_a, int &index_b) {
    const int size = keyframes.size();
    if (size == 0) {
        index_a = index_b = 0;
        return;
    }
    if (size == 1 || time <= keyframes[0].time) {
        index_a = index_b = 0;
        return;
    }
    if (time >= keyframes[size - 1].time) {
        index_a = index_b = size - 1;
        return;
    }

    int left = 0;
    int right = size - 1;
    while (left < right - 1) {
        const int mid = (left + right) / 2;
        if (keyframes[mid].time <= time) {
            left = mid;
        } else {
            right = mid;
        }
    }

    index_a = left;
    index_b = right;
}

bool _resolve_keyframe_sample_value(const Keyframe &p_keyframe, int p_splat_index, Variant &r_value, bool &r_extracted) {
    Variant extracted;
    if (_try_extract_splat_value(p_keyframe.value, p_splat_index, extracted)) {
        r_value = extracted;
        r_extracted = true;
        return true;
    }
    if (_is_splat_container_type(p_keyframe.value.get_type())) {
        r_extracted = false;
        return false;
    }
    r_value = p_keyframe.value;
    r_extracted = false;
    return true;
}

} // namespace

void AnimationClip::add_track(AnimationProperty property) {
    // Check if track already exists
    for (uint32_t i = 0; i < tracks.size(); i++) {
        if (tracks[i].property == property) {
            return; // Track already exists
        }
    }

    tracks.push_back(AnimationTrack(property));
}

AnimationTrack* AnimationClip::get_track(AnimationProperty property) {
    for (uint32_t i = 0; i < tracks.size(); i++) {
        if (tracks[i].property == property) {
            return &tracks[i];
        }
    }
    return nullptr;
}

const AnimationTrack* AnimationClip::get_track(AnimationProperty property) const {
    for (uint32_t i = 0; i < tracks.size(); i++) {
        if (tracks[i].property == property) {
            return &tracks[i];
        }
    }
    return nullptr;
}

void GaussianAnimationStateMachine::_bind_methods() {
    // Clip management
    ClassDB::bind_method(D_METHOD("add_clip", "name", "duration"), &GaussianAnimationStateMachine::add_clip, DEFVAL(1.0f));
    ClassDB::bind_method(D_METHOD("remove_clip", "index"), &GaussianAnimationStateMachine::remove_clip);
    ClassDB::bind_method(D_METHOD("remove_clip_by_name", "name"), &GaussianAnimationStateMachine::remove_clip_by_name);
    ClassDB::bind_method(D_METHOD("get_clip_count"), &GaussianAnimationStateMachine::get_clip_count);
    ClassDB::bind_method(D_METHOD("get_clip_name", "index"), &GaussianAnimationStateMachine::get_clip_name);
    ClassDB::bind_method(D_METHOD("get_clip_duration", "index"), &GaussianAnimationStateMachine::get_clip_duration);
    ClassDB::bind_method(D_METHOD("set_clip_duration", "index", "duration"), &GaussianAnimationStateMachine::set_clip_duration);
    ClassDB::bind_method(D_METHOD("get_clip_looping", "index"), &GaussianAnimationStateMachine::get_clip_looping);
    ClassDB::bind_method(D_METHOD("set_clip_looping", "index", "looping"), &GaussianAnimationStateMachine::set_clip_looping);

    // Track management
    ClassDB::bind_method(D_METHOD("add_track_to_clip", "clip_index", "property"), &GaussianAnimationStateMachine::add_track_to_clip);
    ClassDB::bind_method(D_METHOD("remove_track_from_clip", "clip_index", "property"), &GaussianAnimationStateMachine::remove_track_from_clip);
    ClassDB::bind_method(D_METHOD("has_track", "clip_index", "property"), &GaussianAnimationStateMachine::has_track);

    // Keyframe management
    ClassDB::bind_method(D_METHOD("add_keyframe", "clip_index", "property", "time", "value"), &GaussianAnimationStateMachine::add_keyframe);
    ClassDB::bind_method(D_METHOD("add_keyframe_bezier", "clip_index", "property", "time", "value", "in_handle", "out_handle"), &GaussianAnimationStateMachine::add_keyframe_bezier);
    ClassDB::bind_method(D_METHOD("remove_keyframe", "clip_index", "property", "keyframe_index"), &GaussianAnimationStateMachine::remove_keyframe);
    ClassDB::bind_method(D_METHOD("get_keyframe_count", "clip_index", "property"), &GaussianAnimationStateMachine::get_keyframe_count);
    ClassDB::bind_method(D_METHOD("get_keyframe_time", "clip_index", "property", "keyframe_index"), &GaussianAnimationStateMachine::get_keyframe_time);
    ClassDB::bind_method(D_METHOD("get_keyframe_value", "clip_index", "property", "keyframe_index"), &GaussianAnimationStateMachine::get_keyframe_value);

    // Playback control
    ClassDB::bind_method(D_METHOD("play", "clip_index"), &GaussianAnimationStateMachine::play, DEFVAL(-1));
    ClassDB::bind_method(D_METHOD("pause"), &GaussianAnimationStateMachine::pause);
    ClassDB::bind_method(D_METHOD("stop"), &GaussianAnimationStateMachine::stop);
    ClassDB::bind_method(D_METHOD("seek", "time"), &GaussianAnimationStateMachine::seek);
    ClassDB::bind_method(D_METHOD("set_playback_speed", "speed"), &GaussianAnimationStateMachine::set_playback_speed);
    ClassDB::bind_method(D_METHOD("get_playback_speed"), &GaussianAnimationStateMachine::get_playback_speed);

    // State queries
    ClassDB::bind_method(D_METHOD("get_state"), &GaussianAnimationStateMachine::get_state);
    ClassDB::bind_method(D_METHOD("get_current_time"), &GaussianAnimationStateMachine::get_current_time);
    ClassDB::bind_method(D_METHOD("get_current_clip"), &GaussianAnimationStateMachine::get_current_clip);
    ClassDB::bind_method(D_METHOD("is_playing"), &GaussianAnimationStateMachine::is_playing);

    // Blending
    ClassDB::bind_method(D_METHOD("blend_to_clip", "clip_index", "blend_duration"), &GaussianAnimationStateMachine::blend_to_clip, DEFVAL(0.3f));
    ClassDB::bind_method(D_METHOD("set_clip_weight", "clip_index", "weight"), &GaussianAnimationStateMachine::set_clip_weight);
    ClassDB::bind_method(D_METHOD("get_clip_weight", "clip_index"), &GaussianAnimationStateMachine::get_clip_weight);

    // Update and sampling
    ClassDB::bind_method(D_METHOD("update", "delta"), &GaussianAnimationStateMachine::update);
    ClassDB::bind_method(D_METHOD("set_splat_count", "count"), &GaussianAnimationStateMachine::set_splat_count);
    ClassDB::bind_method(D_METHOD("get_splat_count"), &GaussianAnimationStateMachine::get_splat_count);
    ClassDB::bind_method(D_METHOD("set_incremental_saver", "saver"), &GaussianAnimationStateMachine::set_incremental_saver);
    ClassDB::bind_method(D_METHOD("get_incremental_saver"), &GaussianAnimationStateMachine::get_incremental_saver);

    // Sampling methods
    ClassDB::bind_method(D_METHOD("sample_position", "splat_index", "time"), &GaussianAnimationStateMachine::sample_position, DEFVAL(-1.0f));
    ClassDB::bind_method(D_METHOD("sample_color", "splat_index", "time"), &GaussianAnimationStateMachine::sample_color, DEFVAL(-1.0f));
    ClassDB::bind_method(D_METHOD("sample_opacity", "splat_index", "time"), &GaussianAnimationStateMachine::sample_opacity, DEFVAL(-1.0f));
    ClassDB::bind_method(D_METHOD("sample_scale", "splat_index", "time"), &GaussianAnimationStateMachine::sample_scale, DEFVAL(-1.0f));
    ClassDB::bind_method(D_METHOD("sample_rotation", "splat_index", "time"), &GaussianAnimationStateMachine::sample_rotation, DEFVAL(-1.0f));

    // Enums
    BIND_ENUM_CONSTANT(ANIMATION_PROPERTY_POSITION);
    BIND_ENUM_CONSTANT(ANIMATION_PROPERTY_COLOR);
    BIND_ENUM_CONSTANT(ANIMATION_PROPERTY_OPACITY);
    BIND_ENUM_CONSTANT(ANIMATION_PROPERTY_SCALE);
    BIND_ENUM_CONSTANT(ANIMATION_PROPERTY_ROTATION);

    BIND_ENUM_CONSTANT(ANIMATION_STATE_STOPPED);
    BIND_ENUM_CONSTANT(ANIMATION_STATE_PLAYING);
    BIND_ENUM_CONSTANT(ANIMATION_STATE_PAUSED);
    BIND_ENUM_CONSTANT(ANIMATION_STATE_SEEKING);
}

GaussianAnimationStateMachine::GaussianAnimationStateMachine() {
}

GaussianAnimationStateMachine::~GaussianAnimationStateMachine() {
}

int GaussianAnimationStateMachine::add_clip(const String& p_name, float p_duration) {
    // Check for name conflicts
    if (clip_name_to_index.has(p_name)) {
        GS_LOG_ERROR_DEFAULT("Animation clip with name '" + p_name + "' already exists");
        return -1;
    }

    int index = clips.size();
    clips.push_back(AnimationClip(p_name, p_duration));
    clip_name_to_index[p_name] = index;

    if (incremental_saver.is_valid()) {
        incremental_saver->record_metadata_change(vformat("clip:%s", p_name), Variant(), _clip_to_dict(clips[index], index));
    }

    return index;
}

void GaussianAnimationStateMachine::remove_clip(int p_index) {
    _validate_clip_index(p_index);

    String clip_name = clips[p_index].name;
    Dictionary clip_before;
    if (incremental_saver.is_valid()) {
        clip_before = _clip_to_dict(clips[p_index], p_index);
    }
    clips.remove_at(p_index);
    clip_name_to_index.erase(clip_name);

    // Update indices in map
    for (auto& pair : clip_name_to_index) {
        if (pair.value > p_index) {
            pair.value--;
        }
    }

    // Update current clip if necessary
    if (current_clip_index == p_index) {
        current_clip_index = -1;
        state = ANIMATION_STATE_STOPPED;
    } else if (current_clip_index > p_index) {
        current_clip_index--;
    }

    if (incremental_saver.is_valid()) {
        incremental_saver->record_metadata_change(vformat("clip:%s", clip_name), clip_before, Variant());
    }
}

void GaussianAnimationStateMachine::remove_clip_by_name(const String& p_name) {
    if (clip_name_to_index.has(p_name)) {
        int index = clip_name_to_index[p_name];
        remove_clip(index);
    }
}

String GaussianAnimationStateMachine::get_clip_name(int p_index) const {
    _validate_clip_index(p_index);
    return clips[p_index].name;
}

float GaussianAnimationStateMachine::get_clip_duration(int p_index) const {
    _validate_clip_index(p_index);
    return clips[p_index].duration;
}

void GaussianAnimationStateMachine::set_clip_duration(int p_index, float p_duration) {
    _validate_clip_index(p_index);
    float previous = clips[p_index].duration;
    clips[p_index].duration = MAX(0.0f, p_duration);
    if (incremental_saver.is_valid()) {
        incremental_saver->record_metadata_change(vformat("clip:%s:duration", clips[p_index].name), previous, clips[p_index].duration);
    }
}

bool GaussianAnimationStateMachine::get_clip_looping(int p_index) const {
    _validate_clip_index(p_index);
    return clips[p_index].looping;
}

void GaussianAnimationStateMachine::set_clip_looping(int p_index, bool p_looping) {
    _validate_clip_index(p_index);
    bool previous = clips[p_index].looping;
    clips[p_index].looping = p_looping;
    if (incremental_saver.is_valid() && previous != p_looping) {
        incremental_saver->record_metadata_change(vformat("clip:%s:looping", clips[p_index].name), previous, p_looping);
    }
}

void GaussianAnimationStateMachine::add_track_to_clip(int p_clip_index, AnimationProperty p_property) {
    _validate_clip_index(p_clip_index);
    clips[p_clip_index].add_track(p_property);
    if (incremental_saver.is_valid()) {
        int track_index = clips[p_clip_index].tracks.size() - 1;
        Dictionary after;
        after["track"] = _track_to_dict(clips[p_clip_index].tracks[track_index], track_index);
        incremental_saver->record_animation_change(p_clip_index, p_property, Dictionary(), after);
    }
}

void GaussianAnimationStateMachine::remove_track_from_clip(int p_clip_index, AnimationProperty p_property) {
    _validate_clip_index(p_clip_index);

    for (uint32_t i = 0; i < clips[p_clip_index].tracks.size(); i++) {
        if (clips[p_clip_index].tracks[i].property == p_property) {
            Dictionary before;
            if (incremental_saver.is_valid()) {
                before["track"] = _track_to_dict(clips[p_clip_index].tracks[i], i);
            }
            clips[p_clip_index].tracks.remove_at(i);
            if (incremental_saver.is_valid()) {
                incremental_saver->record_animation_change(p_clip_index, p_property, before, Dictionary());
            }
            break;
        }
    }
}

bool GaussianAnimationStateMachine::has_track(int p_clip_index, AnimationProperty p_property) const {
    _validate_clip_index(p_clip_index);
    return clips[p_clip_index].get_track(p_property) != nullptr;
}

void GaussianAnimationStateMachine::add_keyframe(int p_clip_index, AnimationProperty p_property, float p_time, const Variant& p_value) {
    _validate_clip_index(p_clip_index);

    AnimationTrack* track = clips[p_clip_index].get_track(p_property);
    if (!track) {
        // Auto-create track if it doesn't exist
        clips[p_clip_index].add_track(p_property);
        track = clips[p_clip_index].get_track(p_property);
    }

    Keyframe keyframe(p_time, p_value);
    int inserted = KeyframeInterpolator::add_keyframe_sorted(track->keyframes, keyframe);
    if (incremental_saver.is_valid()) {
        Dictionary after;
        after["keyframe"] = _keyframe_to_dict(track->keyframes[inserted], inserted);
        incremental_saver->record_animation_change(p_clip_index, p_property, Dictionary(), after);
    }
}

void GaussianAnimationStateMachine::add_keyframe_bezier(int p_clip_index, AnimationProperty p_property, float p_time, const Variant& p_value, const Vector2& p_in_handle, const Vector2& p_out_handle) {
    _validate_clip_index(p_clip_index);

    AnimationTrack* track = clips[p_clip_index].get_track(p_property);
    if (!track) {
        clips[p_clip_index].add_track(p_property);
        track = clips[p_clip_index].get_track(p_property);
    }

    Keyframe keyframe(p_time, p_value, InterpolationType::CUBIC_BEZIER);
    keyframe.in_handle = p_in_handle;
    keyframe.out_handle = p_out_handle;
    int inserted = KeyframeInterpolator::add_keyframe_sorted(track->keyframes, keyframe);
    if (incremental_saver.is_valid()) {
        Dictionary after;
        after["keyframe"] = _keyframe_to_dict(track->keyframes[inserted], inserted);
        incremental_saver->record_animation_change(p_clip_index, p_property, Dictionary(), after);
    }
}

void GaussianAnimationStateMachine::remove_keyframe(int p_clip_index, AnimationProperty p_property, int p_keyframe_index) {
    _validate_clip_index(p_clip_index);

    AnimationTrack* track = clips[p_clip_index].get_track(p_property);
    if (track && p_keyframe_index >= 0 && static_cast<uint32_t>(p_keyframe_index) < track->keyframes.size()) {
        Dictionary before;
        if (incremental_saver.is_valid()) {
            before["keyframe"] = _keyframe_to_dict(track->keyframes[p_keyframe_index], p_keyframe_index);
        }
        track->keyframes.remove_at(p_keyframe_index);
        if (incremental_saver.is_valid()) {
            incremental_saver->record_animation_change(p_clip_index, p_property, before, Dictionary());
        }
    }
}

int GaussianAnimationStateMachine::get_keyframe_count(int p_clip_index, AnimationProperty p_property) const {
    _validate_clip_index(p_clip_index);

    const AnimationTrack* track = clips[p_clip_index].get_track(p_property);
    return track ? track->keyframes.size() : 0;
}

float GaussianAnimationStateMachine::get_keyframe_time(int p_clip_index, AnimationProperty p_property, int p_keyframe_index) const {
    _validate_clip_index(p_clip_index);

    const AnimationTrack* track = clips[p_clip_index].get_track(p_property);
    if (track && p_keyframe_index >= 0 && static_cast<uint32_t>(p_keyframe_index) < track->keyframes.size()) {
        return track->keyframes[p_keyframe_index].time;
    }
    return 0.0f;
}

Variant GaussianAnimationStateMachine::get_keyframe_value(int p_clip_index, AnimationProperty p_property, int p_keyframe_index) const {
    _validate_clip_index(p_clip_index);

    const AnimationTrack* track = clips[p_clip_index].get_track(p_property);
    if (track && p_keyframe_index >= 0 && static_cast<uint32_t>(p_keyframe_index) < track->keyframes.size()) {
        return track->keyframes[p_keyframe_index].value;
    }
    return Variant();
}

void GaussianAnimationStateMachine::play(int p_clip_index) {
    if (p_clip_index >= 0) {
        _validate_clip_index(p_clip_index);
        current_clip_index = p_clip_index;
    } else if (current_clip_index < 0 && !clips.is_empty()) {
        current_clip_index = 0;
    }

    if (current_clip_index >= 0) {
        state = ANIMATION_STATE_PLAYING;
    }
}

void GaussianAnimationStateMachine::pause() {
    if (state == ANIMATION_STATE_PLAYING) {
        state = ANIMATION_STATE_PAUSED;
    }
}

void GaussianAnimationStateMachine::stop() {
    state = ANIMATION_STATE_STOPPED;
    current_time = 0.0f;
}

void GaussianAnimationStateMachine::seek(float p_time) {
    current_time = MAX(0.0f, p_time);
    state = ANIMATION_STATE_SEEKING;
}

void GaussianAnimationStateMachine::blend_to_clip(int p_clip_index, float p_blend_duration) {
    _validate_clip_index(p_clip_index);

    // Self-reference guard: blending to the already-playing clip is a no-op.
    if (p_clip_index == current_clip_index) {
        return;
    }

    // Depth limit: prevent excessive blend chain accumulation.
    static constexpr uint32_t MAX_BLEND_CHAIN_DEPTH = 8;
    if (blend_targets.size() >= MAX_BLEND_CHAIN_DEPTH) {
        GS_LOG_WARN_DEFAULT("blend_to_clip: blend chain depth limit reached (" + itos(MAX_BLEND_CHAIN_DEPTH) + "), ignoring blend request");
        return;
    }

    BlendTarget target;
    target.clip_index = p_clip_index;
    target.target_weight = 1.0f;
    target.transition_time = 0.0f;
    target.transition_duration = MAX(0.0f, p_blend_duration);
    target.blend_progress = 0.0f;

    blend_targets.push_back(target);
}

void GaussianAnimationStateMachine::set_clip_weight(int p_clip_index, float p_weight) {
    _validate_clip_index(p_clip_index);

    // Find existing blend target or create new one
    for (uint32_t i = 0; i < blend_targets.size(); i++) {
        if (blend_targets[i].clip_index == p_clip_index) {
            blend_targets[i].target_weight = CLAMP(p_weight, 0.0f, 1.0f);
            return;
        }
    }

    // Create new blend target (immediate -- no transition)
    BlendTarget target;
    target.clip_index = p_clip_index;
    target.target_weight = CLAMP(p_weight, 0.0f, 1.0f);
    target.transition_time = 0.0f;
    target.transition_duration = 0.0f;
    target.blend_progress = 1.0f; // Immediately at full weight.
    blend_targets.push_back(target);
}

float GaussianAnimationStateMachine::get_clip_weight(int p_clip_index) const {
    _validate_clip_index(p_clip_index);

    for (uint32_t i = 0; i < blend_targets.size(); i++) {
        if (blend_targets[i].clip_index == p_clip_index) {
            // Return the current interpolated weight based on blend progress.
            return blend_targets[i].target_weight * blend_targets[i].blend_progress;
        }
    }

    return p_clip_index == current_clip_index ? 1.0f : 0.0f;
}

void GaussianAnimationStateMachine::update(float p_delta) {
    if (state != ANIMATION_STATE_PLAYING) {
        return;
    }

    if (current_clip_index < 0 || static_cast<uint32_t>(current_clip_index) >= clips.size()) {
        return;
    }

    // Update time
    current_time += p_delta * playback_speed;

    // Handle looping
    const AnimationClip& clip = clips[current_clip_index];
    if (clip.looping && current_time >= clip.duration) {
        current_time = fmod(current_time, clip.duration);
    } else if (current_time >= clip.duration) {
        current_time = clip.duration;
        state = ANIMATION_STATE_STOPPED;
    }

    // Update blend targets
    _update_blend_weights(p_delta);
}

bool GaussianAnimationStateMachine::has_active_track(AnimationProperty p_property) const {
    if (current_clip_index < 0 || static_cast<uint32_t>(current_clip_index) >= clips.size()) {
        return false;
    }

    const AnimationTrack* track = clips[current_clip_index].get_track(p_property);
    if (!track || !track->enabled || track->keyframes.is_empty()) {
        return false;
    }

    return true;
}

bool GaussianAnimationStateMachine::try_sample_position(int p_splat_index, float p_time, Vector3 &r_out_position) const {
    if (p_splat_index < 0 || p_splat_index >= splat_count) {
        return false;
    }

    if (current_clip_index < 0 || static_cast<uint32_t>(current_clip_index) >= clips.size()) {
        return false;
    }

    const AnimationTrack* track = clips[current_clip_index].get_track(ANIMATION_PROPERTY_POSITION);
    if (!track || !track->enabled || track->keyframes.is_empty()) {
        return false;
    }

    float sample_time = p_time >= 0.0f ? p_time : current_time;
    Variant result = _sample_track_at_time(*track, sample_time, p_splat_index);

    if (result.get_type() != Variant::VECTOR3) {
        return false;
    }

    r_out_position = result.operator Vector3();
    return true;
}

bool GaussianAnimationStateMachine::try_sample_color(int p_splat_index, float p_time, Color &r_out_color) const {
    if (p_splat_index < 0 || p_splat_index >= splat_count) {
        return false;
    }

    if (current_clip_index < 0 || static_cast<uint32_t>(current_clip_index) >= clips.size()) {
        return false;
    }

    const AnimationTrack* track = clips[current_clip_index].get_track(ANIMATION_PROPERTY_COLOR);
    if (!track || !track->enabled || track->keyframes.is_empty()) {
        return false;
    }

    float sample_time = p_time >= 0.0f ? p_time : current_time;
    Variant result = _sample_track_at_time(*track, sample_time, p_splat_index);

    if (result.get_type() != Variant::COLOR) {
        return false;
    }

    r_out_color = result.operator Color();
    return true;
}

bool GaussianAnimationStateMachine::try_sample_opacity(int p_splat_index, float p_time, float &r_out_opacity) const {
    if (p_splat_index < 0 || p_splat_index >= splat_count) {
        return false;
    }

    if (current_clip_index < 0 || static_cast<uint32_t>(current_clip_index) >= clips.size()) {
        return false;
    }

    const AnimationTrack* track = clips[current_clip_index].get_track(ANIMATION_PROPERTY_OPACITY);
    if (!track || !track->enabled || track->keyframes.is_empty()) {
        return false;
    }

    float sample_time = p_time >= 0.0f ? p_time : current_time;
    Variant result = _sample_track_at_time(*track, sample_time, p_splat_index);
    if (result.get_type() != Variant::FLOAT && result.get_type() != Variant::INT) {
        return false;
    }
    r_out_opacity = result.operator float();
    return true;
}

bool GaussianAnimationStateMachine::try_sample_scale(int p_splat_index, float p_time, Vector3 &r_out_scale) const {
    if (p_splat_index < 0 || p_splat_index >= splat_count) {
        return false;
    }

    if (current_clip_index < 0 || static_cast<uint32_t>(current_clip_index) >= clips.size()) {
        return false;
    }

    const AnimationTrack* track = clips[current_clip_index].get_track(ANIMATION_PROPERTY_SCALE);
    if (!track || !track->enabled || track->keyframes.is_empty()) {
        return false;
    }

    float sample_time = p_time >= 0.0f ? p_time : current_time;
    Variant result = _sample_track_at_time(*track, sample_time, p_splat_index);

    if (result.get_type() != Variant::VECTOR3) {
        return false;
    }

    r_out_scale = result.operator Vector3();
    return true;
}

bool GaussianAnimationStateMachine::try_sample_rotation(int p_splat_index, float p_time, Quaternion &r_out_rotation) const {
    if (p_splat_index < 0 || p_splat_index >= splat_count) {
        return false;
    }

    if (current_clip_index < 0 || static_cast<uint32_t>(current_clip_index) >= clips.size()) {
        return false;
    }

    const AnimationTrack* track = clips[current_clip_index].get_track(ANIMATION_PROPERTY_ROTATION);
    if (!track || !track->enabled || track->keyframes.is_empty()) {
        return false;
    }

    float sample_time = p_time >= 0.0f ? p_time : current_time;
    Variant result = _sample_track_at_time(*track, sample_time, p_splat_index);

    if (result.get_type() != Variant::QUATERNION) {
        return false;
    }

    r_out_rotation = result.operator Quaternion();
    return true;
}

Vector3 GaussianAnimationStateMachine::sample_position(int p_splat_index, float p_time) const {
    Vector3 position;
    if (try_sample_position(p_splat_index, p_time, position)) {
        return position;
    }
    return Vector3();
}

Color GaussianAnimationStateMachine::sample_color(int p_splat_index, float p_time) const {
    Color color;
    if (try_sample_color(p_splat_index, p_time, color)) {
        return color;
    }
    return Color();
}

float GaussianAnimationStateMachine::sample_opacity(int p_splat_index, float p_time) const {
    float opacity;
    if (try_sample_opacity(p_splat_index, p_time, opacity)) {
        return opacity;
    }
    return 1.0f;
}

Vector3 GaussianAnimationStateMachine::sample_scale(int p_splat_index, float p_time) const {
    Vector3 scale;
    if (try_sample_scale(p_splat_index, p_time, scale)) {
        return scale;
    }
    return Vector3(1, 1, 1);
}

Quaternion GaussianAnimationStateMachine::sample_rotation(int p_splat_index, float p_time) const {
    Quaternion rotation;
    if (try_sample_rotation(p_splat_index, p_time, rotation)) {
        return rotation;
    }
    return Quaternion();
}

void GaussianAnimationStateMachine::sample_positions_batch(LocalVector<Vector3>& p_out_positions, float p_time) const {
    p_out_positions.resize(splat_count);

    for (int i = 0; i < splat_count; i++) {
        p_out_positions[i] = sample_position(i, p_time);
    }
}

void GaussianAnimationStateMachine::sample_colors_batch(LocalVector<Color>& p_out_colors, float p_time) const {
    p_out_colors.resize(splat_count);

    for (int i = 0; i < splat_count; i++) {
        p_out_colors[i] = sample_color(i, p_time);
    }
}

void GaussianAnimationStateMachine::sample_opacities_batch(LocalVector<float>& p_out_opacities, float p_time) const {
    p_out_opacities.resize(splat_count);

    for (int i = 0; i < splat_count; i++) {
        p_out_opacities[i] = sample_opacity(i, p_time);
    }
}

Dictionary GaussianAnimationStateMachine::to_dict() const {
    Dictionary dict;

    // Save basic properties
    dict["current_time"] = current_time;
    dict["playback_speed"] = playback_speed;
    dict["current_clip_index"] = current_clip_index;
    dict["splat_count"] = splat_count;

    // Save clips
    Array clips_array;
    for (uint32_t i = 0; i < clips.size(); i++) {
        Dictionary clip_dict;
        clip_dict["name"] = clips[i].name;
        clip_dict["duration"] = clips[i].duration;
        clip_dict["looping"] = clips[i].looping;

        Array tracks_array;
        for (uint32_t j = 0; j < clips[i].tracks.size(); j++) {
            Dictionary track_dict;
            track_dict["property"] = (int)clips[i].tracks[j].property;
            track_dict["enabled"] = clips[i].tracks[j].enabled;
            track_dict["weight"] = clips[i].tracks[j].weight;

            Array keyframes_array;
            for (uint32_t k = 0; k < clips[i].tracks[j].keyframes.size(); k++) {
                Dictionary kf_dict;
                kf_dict["time"] = clips[i].tracks[j].keyframes[k].time;
                kf_dict["value"] = clips[i].tracks[j].keyframes[k].value;
                kf_dict["interpolation"] = (int)clips[i].tracks[j].keyframes[k].interpolation;
                kf_dict["in_handle"] = clips[i].tracks[j].keyframes[k].in_handle;
                kf_dict["out_handle"] = clips[i].tracks[j].keyframes[k].out_handle;
                keyframes_array.push_back(kf_dict);
            }
            track_dict["keyframes"] = keyframes_array;
            tracks_array.push_back(track_dict);
        }
        clip_dict["tracks"] = tracks_array;
        clips_array.push_back(clip_dict);
    }
    dict["clips"] = clips_array;

    return dict;
}

void GaussianAnimationStateMachine::from_dict(const Dictionary& p_dict) {
    // Clear existing data
    clips.clear();
    clip_name_to_index.clear();
    blend_targets.clear();

    // Load basic properties
    current_time = p_dict.get("current_time", 0.0f);
    playback_speed = p_dict.get("playback_speed", 1.0f);
    current_clip_index = p_dict.get("current_clip_index", -1);
    splat_count = p_dict.get("splat_count", 0);

    // Load clips
    Array clips_array = p_dict.get("clips", Array());
    for (int i = 0; i < clips_array.size(); i++) {
        Dictionary clip_dict = clips_array[i];

        AnimationClip clip;
        clip.name = clip_dict.get("name", "");
        clip.duration = clip_dict.get("duration", 1.0f);
        clip.looping = clip_dict.get("looping", false);

        Array tracks_array = clip_dict.get("tracks", Array());
        for (int j = 0; j < tracks_array.size(); j++) {
            Dictionary track_dict = tracks_array[j];

            AnimationTrack track;
            track.property = (AnimationProperty)(int)track_dict.get("property", 0);
            track.enabled = track_dict.get("enabled", true);
            track.weight = track_dict.get("weight", 1.0f);

            Array keyframes_array = track_dict.get("keyframes", Array());
            for (int k = 0; k < keyframes_array.size(); k++) {
                Dictionary kf_dict = keyframes_array[k];

                Keyframe keyframe;
                keyframe.time = kf_dict.get("time", 0.0f);
                keyframe.value = kf_dict.get("value", Variant());
                keyframe.interpolation = (InterpolationType)(int)kf_dict.get("interpolation", 0);
                keyframe.in_handle = kf_dict.get("in_handle", Vector2());
                keyframe.out_handle = kf_dict.get("out_handle", Vector2());

                track.keyframes.push_back(keyframe);
            }

            clip.tracks.push_back(track);
        }

        clips.push_back(clip);
        clip_name_to_index[clip.name] = i;
    }

    state = ANIMATION_STATE_STOPPED;
}

void GaussianAnimationStateMachine::_validate_clip_index(int p_index) const {
    ERR_FAIL_INDEX(p_index, static_cast<int>(clips.size()));
}

void GaussianAnimationStateMachine::_update_blend_weights(float p_delta) {
    for (int64_t i = static_cast<int64_t>(blend_targets.size()) - 1; i >= 0; i--) {
        BlendTarget& target = blend_targets[i];
        target.transition_time += p_delta;

        // Compute frame-rate independent normalized progress [0, 1].
        if (target.transition_duration <= 0.0f) {
            target.blend_progress = 1.0f;
        } else {
            target.blend_progress = CLAMP(target.transition_time / target.transition_duration, 0.0f, 1.0f);
        }

        if (target.blend_progress >= 1.0f) {
            // Transition complete -- commit the clip switch and remove the blend target.
            if (target.target_weight > 0.0f) {
                current_clip_index = target.clip_index;
            }
            blend_targets.remove_at(i);
        }
    }
}

Variant GaussianAnimationStateMachine::_sample_track_at_time(const AnimationTrack& track, float time, int splat_index) const {
    if (track.keyframes.is_empty()) {
        return Variant();
    }

    int index_a = 0;
    int index_b = 0;
    _find_keyframe_indices_for_sample(track.keyframes, time, index_a, index_b);
    if (index_a != index_b && track.keyframes[index_a].time == time) {
        // Exact keyframe hits should resolve to that keyframe only.
        index_b = index_a;
    }

    Variant sampled_a_value;
    bool extracted_a = false;
    if (!_resolve_keyframe_sample_value(track.keyframes[index_a], splat_index, sampled_a_value, extracted_a)) {
        return Variant();
    }

    if (index_a == index_b) {
        return sampled_a_value;
    }

    Variant sampled_b_value;
    bool extracted_b = false;
    if (!_resolve_keyframe_sample_value(track.keyframes[index_b], splat_index, sampled_b_value, extracted_b)) {
        return Variant();
    }

    if (!extracted_a && !extracted_b) {
        // Hot path: scalar tracks can interpolate directly without keyframe copies.
        return interpolator.interpolate(track.keyframes, time);
    }

    LocalVector<Keyframe> sampled_keyframes;
    sampled_keyframes.reserve(2);
    Keyframe sampled_a = track.keyframes[index_a];
    sampled_a.value = sampled_a_value;
    sampled_keyframes.push_back(sampled_a);

    Keyframe sampled_b = track.keyframes[index_b];
    sampled_b.value = sampled_b_value;
    sampled_keyframes.push_back(sampled_b);

    return interpolator.interpolate(sampled_keyframes, time);
}

} // namespace GaussianSplatting
