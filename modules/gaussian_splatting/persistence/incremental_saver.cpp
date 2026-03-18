#include "incremental_saver.h"
#include "gaussian_scene_serializer.h"
#include "../core/gaussian_data.h"
#include "../animation/animation_state_machine.h"

#include "core/error/error_macros.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/marshalls.h"
#include "core/object/class_db.h"
#include "core/math/math_funcs.h"
#include "core/math/vector2.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "core/templates/sort_array.h"
#include "core/templates/vector.h"

#include <cstring>

namespace GaussianSplatting {

namespace {

enum SplatPropertyBits {
    SPLAT_PROP_POSITION = 1 << 0,
    SPLAT_PROP_COLOR = 1 << 1,
    SPLAT_PROP_OPACITY = 1 << 2,
    SPLAT_PROP_SCALE = 1 << 3,
    SPLAT_PROP_ROTATION = 1 << 4,
};

bool _vec3_different(const Vector3 &a, const Vector3 &b) {
    return !a.is_equal_approx(b);
}

bool _quat_different(const Quaternion &a, const Quaternion &b) {
    return !a.is_equal_approx(b);
}

bool _color_different(const Color &a, const Color &b) {
    return !(Math::is_equal_approx(a.r, b.r) && Math::is_equal_approx(a.g, b.g) && Math::is_equal_approx(a.b, b.b) && Math::is_equal_approx(a.a, b.a));
}

uint64_t _now_usec() {
    if (Time *time = Time::get_singleton()) {
        return time->get_ticks_usec();
    }
    return (uint64_t)(OS::get_singleton()->get_ticks_usec());
}

static const uint32_t MAX_INCREMENTAL_CHANGE_COUNT = 1u << 20; // 1,048,576 entries
static const uint64_t MAX_INCREMENTAL_PAYLOAD_BYTES = 256ull * 1024ull * 1024ull; // 256 MiB safety cap
static const uint64_t INCREMENTAL_HEADER_DISK_SIZE =
        sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint16_t) +
        sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t);
static const uint64_t INCREMENTAL_ENTRY_DISK_SIZE =
        sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);

bool _safe_u64_add(uint64_t a, uint64_t b, uint64_t &r_out) {
    if (a > (uint64_t(-1) - b)) {
        return false;
    }
    r_out = a + b;
    return true;
}

bool _safe_u64_mul(uint64_t a, uint64_t b, uint64_t &r_out) {
    if (a == 0 || b == 0) {
        r_out = 0;
        return true;
    }
    if (a > (uint64_t(-1) / b)) {
        return false;
    }
    r_out = a * b;
    return true;
}

Error _ensure_file_write_ok(const Ref<FileAccess> &file, const char *context) {
    ERR_FAIL_COND_V_MSG(file.is_null(), ERR_INVALID_PARAMETER, "FileAccess is null while checking write status.");
    const Error io_error = file->get_error();
    if (io_error != OK) {
        ERR_PRINT(vformat("[GaussianIncrementalSaver] Write failure in %s (error=%d).",
                context ? context : "unknown",
                (int)io_error));
        return io_error;
    }
    return OK;
}

Error _encode_variant_to_bytes(const Variant &value, PackedByteArray &out_bytes) {
    int len = 0;
    Error err = encode_variant(value, nullptr, len, true);
    if (err != OK) {
        return err;
    }
    out_bytes.resize(len);
    return encode_variant(value, out_bytes.ptrw(), len, true);
}

Variant _decode_variant_from_bytes(const PackedByteArray &bytes) {
    Variant value;
    if (bytes.is_empty()) {
        return value;
    }
    Error err = decode_variant(value, bytes.ptr(), bytes.size(), nullptr, true);
    if (err != OK) {
        return Variant();
    }
    return value;
}

PackedByteArray _pack_change_data(const Dictionary &dict) {
    PackedByteArray bytes;
    if (_encode_variant_to_bytes(dict, bytes) != OK) {
        bytes.clear();
    }
    return bytes;
}

Dictionary _unpack_change_data(const PackedByteArray &bytes) {
    Variant value = _decode_variant_from_bytes(bytes);
    if (value.get_type() == Variant::DICTIONARY) {
        return value;
    }
    return Dictionary();
}

int _find_clip_index_by_name(GaussianAnimationStateMachine *p_animation, const String &p_clip_name) {
    ERR_FAIL_NULL_V(p_animation, -1);
    for (int i = 0; i < p_animation->get_clip_count(); i++) {
        if (p_animation->get_clip_name(i) == p_clip_name) {
            return i;
        }
    }
    return -1;
}

struct ClipMetadataEntry {
    String name;
    int recorded_index = -1;
    float duration = 1.0f;
    bool looping = false;
};

struct ClipFieldOverride {
    String name;
    String field;
    Variant value;
};

struct ClipMetadataRemovalComparator {
    _FORCE_INLINE_ bool operator()(const ClipMetadataEntry &a, const ClipMetadataEntry &b) const {
        if (a.recorded_index == b.recorded_index) {
            return a.name < b.name;
        }
        if (a.recorded_index < 0) {
            return false;
        }
        if (b.recorded_index < 0) {
            return true;
        }
        return a.recorded_index > b.recorded_index;
    }
};

struct ClipMetadataUpsertComparator {
    _FORCE_INLINE_ bool operator()(const ClipMetadataEntry &a, const ClipMetadataEntry &b) const {
        if (a.recorded_index == b.recorded_index) {
            return a.name < b.name;
        }
        if (a.recorded_index < 0) {
            return false;
        }
        if (b.recorded_index < 0) {
            return true;
        }
        return a.recorded_index < b.recorded_index;
    }
};

struct ClipFieldOverrideComparator {
    _FORCE_INLINE_ bool operator()(const ClipFieldOverride &a, const ClipFieldOverride &b) const {
        if (a.name == b.name) {
            return a.field < b.field;
        }
        return a.name < b.name;
    }
};

} // namespace

GaussianIncrementalSaver::GaussianIncrementalSaver() = default;
GaussianIncrementalSaver::~GaussianIncrementalSaver() = default;

void GaussianIncrementalSaver::_bind_methods() {
    ClassDB::bind_method(D_METHOD("start_tracking", "baseline_file"), &GaussianIncrementalSaver::start_tracking);
    ClassDB::bind_method(D_METHOD("stop_tracking"), &GaussianIncrementalSaver::stop_tracking);
    ClassDB::bind_method(D_METHOD("is_tracking_enabled"), &GaussianIncrementalSaver::is_tracking_enabled);
    ClassDB::bind_method(D_METHOD("record_animation_change", "clip_index", "property", "old_data", "new_data"), &GaussianIncrementalSaver::record_animation_change);
    ClassDB::bind_method(D_METHOD("record_metadata_change", "key", "old_value", "new_value"), &GaussianIncrementalSaver::record_metadata_change);
    ClassDB::bind_method(D_METHOD("save_changes", "incremental_file_path"), &GaussianIncrementalSaver::save_changes);
    ClassDB::bind_method(D_METHOD("load_and_apply_changes", "incremental_file_path", "gaussian_data", "animation"), &GaussianIncrementalSaver::load_and_apply_changes_bind, DEFVAL(Ref<GaussianAnimationStateMachine>()));
    ClassDB::bind_method(D_METHOD("merge_incremental_files", "files", "output"), &GaussianIncrementalSaver::merge_incremental_files);
    ClassDB::bind_method(D_METHOD("create_baseline", "baseline_file_path", "gaussian_data", "animation"), &GaussianIncrementalSaver::create_baseline_bind, DEFVAL(Ref<GaussianAnimationStateMachine>()));
    ClassDB::bind_method(D_METHOD("update_baseline", "new_baseline_file_path"), &GaussianIncrementalSaver::update_baseline);
    ClassDB::bind_method(D_METHOD("get_baseline_file"), &GaussianIncrementalSaver::get_baseline_file);
    ClassDB::bind_method(D_METHOD("get_change_count"), &GaussianIncrementalSaver::get_change_count);
    ClassDB::bind_method(D_METHOD("get_splat_change_count"), &GaussianIncrementalSaver::get_splat_change_count);
    ClassDB::bind_method(D_METHOD("get_animation_change_count"), &GaussianIncrementalSaver::get_animation_change_count);
    ClassDB::bind_method(D_METHOD("get_changed_splat_indices"), &GaussianIncrementalSaver::get_changed_splat_indices);
    ClassDB::bind_method(D_METHOD("get_change_statistics"), &GaussianIncrementalSaver::get_change_statistics);
    ClassDB::bind_method(D_METHOD("should_auto_save"), &GaussianIncrementalSaver::should_auto_save);
    ClassDB::bind_method(D_METHOD("should_create_full_save"), &GaussianIncrementalSaver::should_create_full_save);
    ClassDB::bind_method(D_METHOD("clear_changes"), &GaussianIncrementalSaver::clear_changes);
    ClassDB::bind_method(D_METHOD("validate_incremental_file", "file_path"), &GaussianIncrementalSaver::validate_incremental_file);
    ClassDB::bind_method(D_METHOD("get_incremental_file_info", "file_path"), &GaussianIncrementalSaver::get_incremental_file_info);
    ClassDB::bind_method(D_METHOD("estimate_save_size"), &GaussianIncrementalSaver::estimate_save_size);
    ClassDB::bind_static_method("GaussianIncrementalSaver", D_METHOD("is_incremental_file", "file_path"), &GaussianIncrementalSaver::is_incremental_file);
    ClassDB::bind_static_method("GaussianIncrementalSaver", D_METHOD("find_incremental_files", "directory", "basename"), &GaussianIncrementalSaver::find_incremental_files);
}

void GaussianIncrementalSaver::start_tracking(const String &baseline_file) {
    baseline_file_path = baseline_file;
    baseline_timestamp = 0;
    Ref<FileAccess> file = FileAccess::open(baseline_file, FileAccess::READ);
    if (file.is_valid()) {
        baseline_timestamp = file->get_modified_time(baseline_file);
    }
    GaussianSceneSerializer serializer;
    Dictionary info = serializer.get_file_info(baseline_file);
    if (info.has("valid") && info["valid"]) {
        baseline_splat_count = info.get("splat_count", 0);
    }
    splat_index_to_change.clear();
    clear_changes();
    is_tracking = true;
    last_save_time = _now_usec();
}

void GaussianIncrementalSaver::stop_tracking() {
    is_tracking = false;
    clear_changes();
}

void GaussianIncrementalSaver::record_splat_change(uint32_t index, const Gaussian &old_splat, const Gaussian &new_splat) {
    if (!is_tracking) {
        return;
    }
    _track_splat_change(index, old_splat, new_splat);
}

void GaussianIncrementalSaver::record_animation_change(int clip_index, AnimationProperty property, const Dictionary &old_data, const Dictionary &new_data) {
    if (!is_tracking) {
        return;
    }
    uint8_t change_type = 0;
    if (old_data.is_empty() && !new_data.is_empty()) {
        change_type = 1;
    } else if (!old_data.is_empty() && new_data.is_empty()) {
        change_type = 2;
    }
    Dictionary payload;
    payload["before"] = old_data;
    payload["after"] = new_data;
    _track_animation_change(clip_index, property, change_type, payload);
}

void GaussianIncrementalSaver::record_metadata_change(const String &key, const Variant &old_value, const Variant &new_value) {
    if (!is_tracking) {
        return;
    }
    MetadataDelta delta;
    if (metadata_changes.has(key)) {
        delta = metadata_changes[key];
    } else {
        delta.old_value = old_value;
    }
    delta.new_value = new_value;
    metadata_changes.insert(key, delta);
    accumulated_changes++;
}

void GaussianIncrementalSaver::_track_splat_change(uint32_t index, const Gaussian &old_splat, const Gaussian &new_splat) {
    uint8_t mask = 0;
    if (_vec3_different(old_splat.position, new_splat.position)) {
        mask |= SPLAT_PROP_POSITION;
    }
    if (_color_different(old_splat.sh_dc, new_splat.sh_dc)) {
        mask |= SPLAT_PROP_COLOR;
    }
    if (!Math::is_equal_approx(old_splat.opacity, new_splat.opacity)) {
        mask |= SPLAT_PROP_OPACITY;
    }
    if (_vec3_different(old_splat.scale, new_splat.scale)) {
        mask |= SPLAT_PROP_SCALE;
    }
    if (_quat_different(old_splat.rotation, new_splat.rotation)) {
        mask |= SPLAT_PROP_ROTATION;
    }

    if (mask == 0) {
        return;
    }

    SplatChange change;
    change.index = index;
    change.changed_properties = mask;
    change.position = new_splat.position;
    change.color = new_splat.sh_dc;
    change.opacity = new_splat.opacity;
    change.scale = new_splat.scale;
    change.rotation = new_splat.rotation;

    if (splat_index_to_change.has(index)) {
        int existing_index = splat_index_to_change[index];
        SplatChange &existing = splat_changes[existing_index];
        existing.changed_properties |= change.changed_properties;
        if (mask & SPLAT_PROP_POSITION) {
            existing.position = change.position;
        }
        if (mask & SPLAT_PROP_COLOR) {
            existing.color = change.color;
        }
        if (mask & SPLAT_PROP_OPACITY) {
            existing.opacity = change.opacity;
        }
        if (mask & SPLAT_PROP_SCALE) {
            existing.scale = change.scale;
        }
        if (mask & SPLAT_PROP_ROTATION) {
            existing.rotation = change.rotation;
        }
    } else {
        splat_index_to_change.insert(index, splat_changes.size());
        splat_changes.push_back(change);
    }

    accumulated_changes++;
}

void GaussianIncrementalSaver::_track_animation_change(int clip_index, AnimationProperty property, uint8_t change_type, const Dictionary &data) {
    AnimationChange change;
    change.clip_index = clip_index;
    change.property = property;
    change.change_type = change_type;
    change.data = data;
    animation_changes.push_back(change);
    accumulated_changes++;
}

Error GaussianIncrementalSaver::_write_change_entry(Ref<FileAccess> file, const ChangeEntry &entry) const {
    file->store_8((uint8_t)entry.type);
    file->store_32(entry.data_offset);
    file->store_32(entry.data_size);
    file->store_64(entry.timestamp);
    return _ensure_file_write_ok(file, "_write_change_entry");
}

Error GaussianIncrementalSaver::_read_change_entry(Ref<FileAccess> file, ChangeEntry &entry) const {
    if (file->eof_reached()) {
        return ERR_FILE_EOF;
    }
    entry.type = (ChangeType)file->get_8();
    entry.data_offset = file->get_32();
    entry.data_size = file->get_32();
    entry.timestamp = file->get_64();
    return OK;
}

Error GaussianIncrementalSaver::_apply_splat_changes(::GaussianData *gaussian_data) const {
    ERR_FAIL_NULL_V(gaussian_data, ERR_INVALID_PARAMETER);
    for (uint32_t i = 0; i < splat_changes.size(); i++) {
        const SplatChange &change = splat_changes[i];
        ERR_CONTINUE(change.index >= (uint32_t)gaussian_data->get_count());
        Gaussian g = gaussian_data->get_gaussian(change.index);
        if (change.changed_properties & SPLAT_PROP_POSITION) {
            g.position = change.position;
        }
        if (change.changed_properties & SPLAT_PROP_COLOR) {
            g.sh_dc = change.color;
        }
        if (change.changed_properties & SPLAT_PROP_OPACITY) {
            g.opacity = change.opacity;
        }
        if (change.changed_properties & SPLAT_PROP_SCALE) {
            g.scale = change.scale;
        }
        if (change.changed_properties & SPLAT_PROP_ROTATION) {
            g.rotation = change.rotation;
        }
        gaussian_data->set_gaussian(change.index, g);
    }
    return OK;
}

Error GaussianIncrementalSaver::_apply_metadata_changes(GaussianAnimationStateMachine *animation) const {
    if (!animation) {
        return OK;
    }

    Vector<ClipMetadataEntry> clip_removals;
    Vector<ClipMetadataEntry> clip_upserts;
    Vector<ClipFieldOverride> clip_field_overrides;

    for (const KeyValue<String, MetadataDelta> &E : metadata_changes) {
        PackedStringArray parts = E.key.split(":");
        if (parts.is_empty() || parts[0] != "clip") {
            continue;
        }

        if (parts.size() == 2) {
            const String clip_name = parts[1];
            if (clip_name.is_empty()) {
                continue;
            }

            const Variant &new_value = E.value.new_value;
            if (new_value.get_type() == Variant::NIL) {
                ClipMetadataEntry entry;
                entry.name = clip_name;
                if (E.value.old_value.get_type() == Variant::DICTIONARY) {
                    Dictionary old_clip_dict = E.value.old_value;
                    entry.recorded_index = (int)old_clip_dict.get("index", -1);
                }
                clip_removals.push_back(entry);
                continue;
            }

            if (new_value.get_type() != Variant::DICTIONARY) {
                continue;
            }

            Dictionary clip_dict = new_value;
            ClipMetadataEntry entry;
            entry.name = clip_name;
            entry.recorded_index = (int)clip_dict.get("index", -1);
            entry.duration = MAX((float)(double)clip_dict.get("duration", 1.0f), 0.0f);
            entry.looping = (bool)clip_dict.get("looping", false);
            clip_upserts.push_back(entry);
            continue;
        }

        if (parts.size() != 3) {
            continue;
        }

        const String clip_name = parts[1];
        const String field = parts[2];
        if (clip_name.is_empty()) {
            continue;
        }

        const Variant &new_value = E.value.new_value;
        if (new_value.get_type() == Variant::NIL) {
            continue;
        }

        ClipFieldOverride override_entry;
        override_entry.name = clip_name;
        override_entry.field = field;
        override_entry.value = new_value;
        clip_field_overrides.push_back(override_entry);
    }

    if (clip_removals.size() > 1) {
        SortArray<ClipMetadataEntry, ClipMetadataRemovalComparator> sorter;
        sorter.sort(clip_removals.ptrw(), clip_removals.size());
    }

    // Apply removals first to make index reconstruction deterministic for later additions.
    for (uint32_t i = 0; i < clip_removals.size(); i++) {
        int clip_index = _find_clip_index_by_name(animation, clip_removals[i].name);
        if (clip_index >= 0) {
            animation->remove_clip(clip_index);
        }
    }

    if (clip_upserts.size() > 1) {
        SortArray<ClipMetadataEntry, ClipMetadataUpsertComparator> sorter;
        sorter.sort(clip_upserts.ptrw(), clip_upserts.size());
    }

    for (uint32_t i = 0; i < clip_upserts.size(); i++) {
        const ClipMetadataEntry &entry = clip_upserts[i];
        int clip_index = _find_clip_index_by_name(animation, entry.name);
        if (clip_index < 0) {
            clip_index = animation->add_clip(entry.name, entry.duration);
        } else {
            animation->set_clip_duration(clip_index, entry.duration);
        }
        if (clip_index < 0) {
            continue;
        }
        animation->set_clip_looping(clip_index, entry.looping);
    }

    if (clip_field_overrides.size() > 1) {
        SortArray<ClipFieldOverride, ClipFieldOverrideComparator> sorter;
        sorter.sort(clip_field_overrides.ptrw(), clip_field_overrides.size());
    }

    for (uint32_t i = 0; i < clip_field_overrides.size(); i++) {
        const ClipFieldOverride &override_entry = clip_field_overrides[i];
        int clip_index = _find_clip_index_by_name(animation, override_entry.name);
        if (clip_index < 0) {
            float initial_duration = 1.0f;
            if (override_entry.field == "duration") {
                initial_duration = MAX((float)(double)override_entry.value, 0.0f);
            }
            clip_index = animation->add_clip(override_entry.name, initial_duration);
        }
        if (clip_index < 0) {
            continue;
        }

        if (override_entry.field == "duration") {
            animation->set_clip_duration(clip_index, MAX((float)(double)override_entry.value, 0.0f));
        } else if (override_entry.field == "looping") {
            animation->set_clip_looping(clip_index, (bool)override_entry.value);
        }
    }

    return OK;
}

Error GaussianIncrementalSaver::_apply_animation_changes(GaussianAnimationStateMachine *animation) const {
    if (!animation) {
        return OK;
    }

    // Clip metadata may be replayed in a different order than creation events were recorded.
    // Build a remap from recorded clip index -> runtime index by clip name so animation deltas
    // continue targeting the intended clip.
    HashMap<int, int> clip_index_remap;
    for (const KeyValue<String, MetadataDelta> &E : metadata_changes) {
        PackedStringArray parts = E.key.split(":");
        if (parts.size() != 2 || parts[0] != "clip") {
            continue;
        }
        if (parts[1].is_empty()) {
            continue;
        }
        if (E.value.new_value.get_type() != Variant::DICTIONARY) {
            continue;
        }

        Dictionary clip_dict = E.value.new_value;
        const int recorded_index = (int)clip_dict.get("index", -1);
        if (recorded_index < 0) {
            continue;
        }
        const int runtime_index = _find_clip_index_by_name(animation, parts[1]);
        if (runtime_index < 0 || runtime_index == recorded_index) {
            continue;
        }
        clip_index_remap.insert(recorded_index, runtime_index);
    }

    for (uint32_t i = 0; i < animation_changes.size(); i++) {
        const AnimationChange &change = animation_changes[i];
        int target_clip_index = change.clip_index;
        if (const int *mapped_index = clip_index_remap.getptr(change.clip_index)) {
            target_clip_index = *mapped_index;
        }
        ERR_CONTINUE(target_clip_index < 0 || target_clip_index >= animation->get_clip_count());
        Dictionary before = change.data.get("before", Dictionary());
        Dictionary after = change.data.get("after", Dictionary());
        Dictionary track_before = before.get("track", Dictionary());
        Dictionary track_after = after.get("track", Dictionary());
        if (!track_before.is_empty() || !track_after.is_empty()) {
            switch (change.change_type) {
                case 1: { // Track added
                    if (!track_after.is_empty() && !animation->has_track(target_clip_index, change.property)) {
                        animation->add_track_to_clip(target_clip_index, change.property);
                    }
                    break;
                }
                case 2: { // Track removed
                    if (!track_before.is_empty() && animation->has_track(target_clip_index, change.property)) {
                        animation->remove_track_from_clip(target_clip_index, change.property);
                    }
                    break;
                }
                default: {
                    // Track metadata changes are not yet tracked, but avoid treating them as keyframe edits.
                    break;
                }
            }
            continue;
        }

        Dictionary keyframe_before = before.get("keyframe", Dictionary());
        Dictionary keyframe_after = after.get("keyframe", Dictionary());
        switch (change.change_type) {
            case 1: // Added
                if (!keyframe_after.is_empty()) {
                    float time = keyframe_after.get("time", 0.0f);
                    Variant value = keyframe_after.get("value", Variant());
                    InterpolationType interpolation = (InterpolationType)(int)keyframe_after.get("interpolation", (int)InterpolationType::LINEAR);
                    if (interpolation == InterpolationType::CUBIC_BEZIER) {
                        Vector2 in_handle = keyframe_after.get("in_handle", Vector2());
                        Vector2 out_handle = keyframe_after.get("out_handle", Vector2());
                        animation->add_keyframe_bezier(target_clip_index, change.property, time, value, in_handle, out_handle);
                    } else {
                        animation->add_keyframe(target_clip_index, change.property, time, value);
                    }
                }
                break;
            case 2: // Removed
                if (!keyframe_before.is_empty()) {
                    int key_idx = keyframe_before.get("index", -1);
                    if (key_idx >= 0) {
                        animation->remove_keyframe(target_clip_index, change.property, key_idx);
                    }
                }
                break;
            default: // Modified
                if (!keyframe_after.is_empty()) {
                    int key_idx = keyframe_after.get("index", -1);
                    if (key_idx >= 0) {
                        animation->remove_keyframe(target_clip_index, change.property, key_idx);
                        float time = keyframe_after.get("time", 0.0f);
                        Variant value = keyframe_after.get("value", Variant());
                        InterpolationType interpolation = (InterpolationType)(int)keyframe_after.get("interpolation", (int)InterpolationType::LINEAR);
                        if (interpolation == InterpolationType::CUBIC_BEZIER) {
                            Vector2 in_handle = keyframe_after.get("in_handle", Vector2());
                            Vector2 out_handle = keyframe_after.get("out_handle", Vector2());
                            animation->add_keyframe_bezier(target_clip_index, change.property, time, value, in_handle, out_handle);
                        } else {
                            animation->add_keyframe(target_clip_index, change.property, time, value);
                        }
                    }
                }
                break;
        }
    }
    return OK;
}

void GaussianIncrementalSaver::clear_changes() {
    splat_changes.clear();
    animation_changes.clear();
    metadata_changes.clear();
    splat_index_to_change.clear();
    accumulated_changes = 0;
}

bool GaussianIncrementalSaver::should_auto_save() const {
    if (!is_tracking) {
        return false;
    }
    uint64_t now = _now_usec();
    return (now - last_save_time) > (uint64_t)(auto_save_interval * 1'000'000.0f);
}

bool GaussianIncrementalSaver::should_create_full_save() const {
    return accumulated_changes >= max_changes_before_full_save;
}

Array GaussianIncrementalSaver::get_changed_splat_indices() const {
    Array indices;
    indices.resize(splat_changes.size());
    for (uint32_t i = 0; i < splat_changes.size(); i++) {
        indices[i] = (int)splat_changes[i].index;
    }
    return indices;
}

Dictionary GaussianIncrementalSaver::get_change_statistics() const {
    Dictionary dict;
    dict["splat_changes"] = (int64_t)splat_changes.size();
    dict["animation_changes"] = (int64_t)animation_changes.size();
    dict["metadata_changes"] = (int64_t)metadata_changes.size();
    dict["accumulated"] = (int64_t)accumulated_changes;
    return dict;
}

Error GaussianIncrementalSaver::save_changes(const String &incremental_file_path) {
    ERR_FAIL_COND_V(!is_tracking, ERR_UNCONFIGURED);

    struct PendingChange {
        ChangeEntry entry;
        PackedByteArray data;
    };

    LocalVector<PendingChange> pending;
    pending.reserve(splat_changes.size() + animation_changes.size() + metadata_changes.size());

    uint64_t timestamp = _now_usec();

    for (uint32_t i = 0; i < splat_changes.size(); i++) {
        const SplatChange &change = splat_changes[i];
        Dictionary dict;
        dict["index"] = (int)change.index;
        dict["mask"] = (int)change.changed_properties;
        if (change.changed_properties & SPLAT_PROP_POSITION) {
            dict["position"] = change.position;
        }
        if (change.changed_properties & SPLAT_PROP_COLOR) {
            dict["color"] = change.color;
        }
        if (change.changed_properties & SPLAT_PROP_OPACITY) {
            dict["opacity"] = change.opacity;
        }
        if (change.changed_properties & SPLAT_PROP_SCALE) {
            dict["scale"] = change.scale;
        }
        if (change.changed_properties & SPLAT_PROP_ROTATION) {
            dict["rotation"] = change.rotation;
        }

        PendingChange pc;
        pc.entry.type = ChangeType::SPLAT_MODIFIED;
        pc.data = _pack_change_data(dict);
        pc.entry.data_size = pc.data.size();
        pc.entry.timestamp = timestamp;
        pending.push_back(pc);
    }

    for (uint32_t i = 0; i < animation_changes.size(); i++) {
        const AnimationChange &change = animation_changes[i];
        Dictionary dict;
        dict["clip_index"] = change.clip_index;
        dict["property"] = (int)change.property;
        dict["change_type"] = (int)change.change_type;
        dict["before"] = change.data.get("before", Dictionary());
        dict["after"] = change.data.get("after", Dictionary());

        PendingChange pc;
        pc.entry.type = ChangeType::ANIMATION_MODIFIED;
        pc.data = _pack_change_data(dict);
        pc.entry.data_size = pc.data.size();
        pc.entry.timestamp = timestamp;
        pending.push_back(pc);
    }

    for (const KeyValue<String, MetadataDelta> &E : metadata_changes) {
        Dictionary dict;
        dict["key"] = E.key;
        dict["old"] = E.value.old_value;
        dict["new"] = E.value.new_value;

        PendingChange pc;
        pc.entry.type = ChangeType::METADATA_MODIFIED;
        pc.data = _pack_change_data(dict);
        pc.entry.data_size = pc.data.size();
        pc.entry.timestamp = timestamp;
        pending.push_back(pc);
    }

    uint32_t offset = 0;
    for (uint32_t i = 0; i < pending.size(); i++) {
        pending[i].entry.data_offset = offset;
        offset += pending[i].entry.data_size;
    }

    Ref<FileAccess> file = FileAccess::open(incremental_file_path, FileAccess::WRITE);
    ERR_FAIL_COND_V(file.is_null(), ERR_CANT_CREATE);

    file->store_32(INCREMENTAL_MAGIC);
    file->store_16(INCREMENTAL_VERSION);
    file->store_16(INCREMENTAL_SAVER_LAYOUT_VERSION);
    file->store_64(timestamp);
    file->store_64(baseline_timestamp);
    file->store_32(baseline_splat_count);
    file->store_32(pending.size());
    Error err = _ensure_file_write_ok(file, "save_changes(header)");
    if (err != OK) {
        return err;
    }

    for (uint32_t i = 0; i < pending.size(); i++) {
        err = _write_change_entry(file, pending[i].entry);
        if (err != OK) {
            return err;
        }
    }

    for (uint32_t i = 0; i < pending.size(); i++) {
        file->store_buffer(pending[i].data);
        err = _ensure_file_write_ok(file, "save_changes(payload)");
        if (err != OK) {
            return err;
        }
    }

    err = _ensure_file_write_ok(file, "save_changes(final)");
    if (err != OK) {
        return err;
    }

    last_save_time = timestamp;
    clear_changes();
    return OK;
}

Error GaussianIncrementalSaver::load_and_apply_changes(const String &incremental_file_path, ::GaussianData *gaussian_data, GaussianAnimationStateMachine *animation) {
    Ref<FileAccess> file = FileAccess::open(incremental_file_path, FileAccess::READ);
    ERR_FAIL_COND_V(file.is_null(), ERR_FILE_NOT_FOUND);
    const uint64_t file_length = file->get_length();
    ERR_FAIL_COND_V(file_length < INCREMENTAL_HEADER_DISK_SIZE, ERR_FILE_CORRUPT);

    uint32_t magic = file->get_32();
    ERR_FAIL_COND_V_MSG(magic != INCREMENTAL_MAGIC, ERR_FILE_UNRECOGNIZED, "Invalid incremental file: " + incremental_file_path);
    uint16_t version = file->get_16();
    ERR_FAIL_COND_V(version > INCREMENTAL_VERSION, ERR_FILE_CORRUPT);
    uint16_t layout_version = file->get_16();
    ERR_FAIL_COND_V_MSG(layout_version != INCREMENTAL_SAVER_LAYOUT_VERSION, ERR_FILE_CORRUPT,
            vformat("Incremental file layout version mismatch: file=%d expected=%d. "
                    "The file was written with an incompatible struct layout. "
                    "Re-export the incremental file or revert to the full baseline.",
                    (int)layout_version, (int)INCREMENTAL_SAVER_LAYOUT_VERSION));
    uint64_t change_timestamp = file->get_64();
    baseline_timestamp = file->get_64();
    baseline_splat_count = file->get_32();
    uint32_t change_count = file->get_32();
    ERR_FAIL_COND_V(file->get_error() != OK, ERR_FILE_CORRUPT);
    ERR_FAIL_COND_V(change_count > MAX_INCREMENTAL_CHANGE_COUNT, ERR_FILE_CORRUPT);

    const uint64_t position_after_header = uint64_t(file->get_position());
    ERR_FAIL_COND_V(position_after_header > file_length, ERR_FILE_CORRUPT);
    const uint64_t bytes_after_header = file_length - position_after_header;
    uint64_t required_entry_bytes = 0;
    ERR_FAIL_COND_V(!_safe_u64_mul(uint64_t(change_count), INCREMENTAL_ENTRY_DISK_SIZE, required_entry_bytes), ERR_FILE_CORRUPT);
    ERR_FAIL_COND_V(required_entry_bytes > bytes_after_header, ERR_FILE_CORRUPT);

    LocalVector<ChangeEntry> entries;
    entries.resize(change_count);
    for (uint32_t i = 0; i < change_count; i++) {
        Error entry_err = _read_change_entry(file, entries[i]);
        ERR_FAIL_COND_V(entry_err != OK, ERR_FILE_CORRUPT);
        ERR_FAIL_COND_V(entries[i].type > ChangeType::METADATA_MODIFIED, ERR_FILE_CORRUPT);
    }

    const uint64_t payload_start = uint64_t(file->get_position());
    ERR_FAIL_COND_V(payload_start > file_length, ERR_FILE_CORRUPT);
    const uint64_t payload_available = file_length - payload_start;

    uint64_t max_offset = 0;
    for (uint32_t i = 0; i < entries.size(); i++) {
        const uint64_t offset = uint64_t(entries[i].data_offset);
        const uint64_t size = uint64_t(entries[i].data_size);
        ERR_FAIL_COND_V(offset > payload_available, ERR_FILE_CORRUPT);
        uint64_t end = 0;
        ERR_FAIL_COND_V(!_safe_u64_add(offset, size, end), ERR_FILE_CORRUPT);
        ERR_FAIL_COND_V(end > payload_available, ERR_FILE_CORRUPT);
        max_offset = MAX(max_offset, end);
    }
    ERR_FAIL_COND_V(max_offset > MAX_INCREMENTAL_PAYLOAD_BYTES, ERR_FILE_CORRUPT);
    uint64_t payload_end = 0;
    ERR_FAIL_COND_V(!_safe_u64_add(payload_start, max_offset, payload_end), ERR_FILE_CORRUPT);
    ERR_FAIL_COND_V(payload_end > file_length, ERR_FILE_CORRUPT);

    PackedByteArray data_blob = file->get_buffer(max_offset);
    ERR_FAIL_COND_V(uint64_t(data_blob.size()) != max_offset, ERR_FILE_CORRUPT);

    splat_changes.clear();
    animation_changes.clear();
    metadata_changes.clear();

    for (uint32_t i = 0; i < entries.size(); i++) {
        const ChangeEntry &entry = entries[i];
        const uint64_t entry_end = uint64_t(entry.data_offset) + uint64_t(entry.data_size);
        ERR_FAIL_COND_V(entry_end > uint64_t(data_blob.size()), ERR_FILE_CORRUPT);
        PackedByteArray payload;
        payload.resize(entry.data_size);
        if (entry.data_size > 0) {
            memcpy(payload.ptrw(), data_blob.ptr() + entry.data_offset, entry.data_size);
        }

        Dictionary dict = _unpack_change_data(payload);
        switch (entry.type) {
            case ChangeType::SPLAT_MODIFIED: {
                SplatChange change;
                change.index = dict.get("index", 0);
                change.changed_properties = dict.get("mask", 0);
                if (change.changed_properties & SPLAT_PROP_POSITION) {
                    change.position = dict.get("position", Vector3());
                }
                if (change.changed_properties & SPLAT_PROP_COLOR) {
                    change.color = dict.get("color", Color());
                }
                if (change.changed_properties & SPLAT_PROP_OPACITY) {
                    change.opacity = dict.get("opacity", 1.0f);
                }
                if (change.changed_properties & SPLAT_PROP_SCALE) {
                    change.scale = dict.get("scale", Vector3(1, 1, 1));
                }
                if (change.changed_properties & SPLAT_PROP_ROTATION) {
                    change.rotation = dict.get("rotation", Quaternion());
                }
                splat_changes.push_back(change);
                break;
            }
            case ChangeType::ANIMATION_MODIFIED: {
                AnimationChange change;
                change.clip_index = dict.get("clip_index", -1);
                change.property = (AnimationProperty)(int)dict.get("property", 0);
                change.change_type = dict.get("change_type", 0);
                Dictionary data;
                data["before"] = dict.get("before", Dictionary());
                data["after"] = dict.get("after", Dictionary());
                change.data = data;
                animation_changes.push_back(change);
                break;
            }
            case ChangeType::METADATA_MODIFIED: {
                MetadataDelta delta;
                String key = dict.get("key", "");
                delta.old_value = dict.get("old", Variant());
                delta.new_value = dict.get("new", Variant());
                metadata_changes.insert(key, delta);
                break;
            }
            default:
                break;
        }
    }

    accumulated_changes = entries.size();
    last_save_time = change_timestamp;

    if (gaussian_data) {
        Error err = _apply_splat_changes(gaussian_data);
        if (err != OK) {
            return err;
        }
    }
    if (animation) {
        Error err = _apply_metadata_changes(animation);
        if (err != OK) {
            return err;
        }
        err = _apply_animation_changes(animation);
        if (err != OK) {
            return err;
        }
    }

    return OK;
}

Error GaussianIncrementalSaver::load_and_apply_changes_bind(const String &incremental_file_path, const Ref<::GaussianData> &gaussian_data,
        const Ref<GaussianAnimationStateMachine> &animation) {
    ERR_FAIL_COND_V_MSG(gaussian_data.is_null(), ERR_INVALID_PARAMETER,
            "GaussianIncrementalSaver::load_and_apply_changes requires a valid GaussianData resource");

    ::GaussianData *data_ptr = gaussian_data.ptr();
    GaussianAnimationStateMachine *anim_ptr = animation.is_valid() ? animation.ptr() : nullptr;
    return load_and_apply_changes(incremental_file_path, data_ptr, anim_ptr);
}

Error GaussianIncrementalSaver::merge_incremental_files(const Array &incremental_files, const String &output_file) {
    ERR_FAIL_COND_V(incremental_files.is_empty(), ERR_INVALID_PARAMETER);
    LocalVector<SplatChange> aggregated_splats;
    LocalVector<AnimationChange> aggregated_anim;
    HashMap<String, MetadataDelta> aggregated_metadata;

    uint64_t merged_baseline_timestamp = baseline_timestamp;
    uint32_t merged_baseline_splats = baseline_splat_count;

    for (uint32_t i = 0; i < incremental_files.size(); i++) {
        String path = incremental_files[i];
        Error err = load_and_apply_changes(path, nullptr, nullptr);
        if (err != OK) {
            return err;
        }

        for (uint32_t j = 0; j < splat_changes.size(); j++) {
            aggregated_splats.push_back(splat_changes[j]);
        }
        for (uint32_t j = 0; j < animation_changes.size(); j++) {
            aggregated_anim.push_back(animation_changes[j]);
        }
        for (const KeyValue<String, MetadataDelta> &E : metadata_changes) {
            aggregated_metadata.insert(E.key, E.value);
        }
        merged_baseline_timestamp = baseline_timestamp;
        merged_baseline_splats = baseline_splat_count;
    }

    clear_changes();
    splat_changes = aggregated_splats;
    animation_changes = aggregated_anim;
    metadata_changes = aggregated_metadata;
    baseline_timestamp = merged_baseline_timestamp;
    baseline_splat_count = merged_baseline_splats;
    accumulated_changes = splat_changes.size() + animation_changes.size() + metadata_changes.size();
    return save_changes(output_file);
}

Error GaussianIncrementalSaver::create_baseline(const String &baseline_path, const ::GaussianData *gaussian_data, const GaussianAnimationStateMachine *animation) {
    ERR_FAIL_NULL_V(gaussian_data, ERR_INVALID_PARAMETER);
    GaussianSceneSerializer serializer;
    Error err = serializer.save_scene(baseline_path, gaussian_data, animation);
    if (err != OK) {
        return err;
    }
    baseline_file_path = baseline_path;
    Ref<FileAccess> file = FileAccess::open(baseline_path, FileAccess::READ);
    if (file.is_valid()) {
        baseline_timestamp = file->get_modified_time(baseline_path);
    }
    baseline_splat_count = gaussian_data->get_count();
    return OK;
}

Error GaussianIncrementalSaver::create_baseline_bind(const String &baseline_path, const Ref<::GaussianData> &gaussian_data,
        const Ref<GaussianAnimationStateMachine> &animation) {
    ERR_FAIL_COND_V_MSG(gaussian_data.is_null(), ERR_INVALID_PARAMETER,
            "GaussianIncrementalSaver::create_baseline requires a valid GaussianData resource");

    const ::GaussianData *data_ptr = gaussian_data.ptr();
    const GaussianAnimationStateMachine *anim_ptr = animation.is_valid() ? animation.ptr() : nullptr;
    return create_baseline(baseline_path, data_ptr, anim_ptr);
}

Error GaussianIncrementalSaver::update_baseline(const String &new_baseline_file_path) {
    baseline_file_path = new_baseline_file_path;
    Ref<FileAccess> file = FileAccess::open(new_baseline_file_path, FileAccess::READ);
    if (file.is_valid()) {
        baseline_timestamp = file->get_modified_time(new_baseline_file_path);
    }
    clear_changes();
    return OK;
}

Error GaussianIncrementalSaver::validate_incremental_file(const String &file_path) const {
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
    ERR_FAIL_COND_V(file.is_null(), ERR_FILE_NOT_FOUND);
    uint32_t magic = file->get_32();
    if (magic != INCREMENTAL_MAGIC) {
        return ERR_FILE_UNRECOGNIZED;
    }
    return OK;
}

Dictionary GaussianIncrementalSaver::get_incremental_file_info(const String &file_path) const {
    Dictionary info;
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
    if (file.is_null()) {
        info["valid"] = false;
        return info;
    }
    uint32_t magic = file->get_32();
    if (magic != INCREMENTAL_MAGIC) {
        info["valid"] = false;
        return info;
    }
    info["valid"] = true;
    info["version"] = file->get_16();
    info["layout_version"] = file->get_16();
    info["timestamp"] = (int64_t)file->get_64();
    info["baseline_timestamp"] = (int64_t)file->get_64();
    info["baseline_splats"] = (int64_t)file->get_32();
    info["change_count"] = (int64_t)file->get_32();
    return info;
}

uint64_t GaussianIncrementalSaver::estimate_save_size() const {
    uint64_t size = sizeof(uint32_t) + sizeof(uint16_t) * 2 + sizeof(uint64_t) * 2 + sizeof(uint32_t) * 2;
    for (uint32_t i = 0; i < splat_changes.size(); i++) {
        size += sizeof(ChangeEntry) + 128;
    }
    for (uint32_t i = 0; i < animation_changes.size(); i++) {
        size += sizeof(ChangeEntry) + 256;
    }
    size += metadata_changes.size() * (sizeof(ChangeEntry) + 64);
    return size;
}

bool GaussianIncrementalSaver::is_incremental_file(const String &file_path) {
    if (!file_path.ends_with("." + get_incremental_extension())) {
        return false;
    }
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
    if (file.is_null()) {
        return false;
    }
    return file->get_32() == INCREMENTAL_MAGIC;
}

Array GaussianIncrementalSaver::find_incremental_files(const String &directory, const String &basename) {
    Array result;
    Ref<DirAccess> dir = DirAccess::open(directory);
    if (dir.is_null()) {
        return result;
    }
    dir->list_dir_begin();
    while (true) {
        String file_name = dir->get_next();
        if (file_name.is_empty()) {
            break;
        }
        if (dir->current_is_dir()) {
            continue;
        }
        if (!file_name.ends_with("." + get_incremental_extension())) {
            continue;
        }
        if (!basename.is_empty() && !file_name.begins_with(basename)) {
            continue;
        }
        result.append(directory.path_join(file_name));
    }
    dir->list_dir_end();
    return result;
}

} // namespace GaussianSplatting
