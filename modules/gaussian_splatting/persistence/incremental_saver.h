#ifndef GAUSSIAN_INCREMENTAL_SAVER_H
#define GAUSSIAN_INCREMENTAL_SAVER_H

#include "core/io/file_access.h"
#include "core/io/resource.h"
#include "core/variant/variant.h"
#include "core/variant/dictionary.h"
#include "core/math/vector3.h"
#include "core/math/color.h"
#include "core/math/quaternion.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/string/ustring.h"

struct Gaussian;
class GaussianData;

namespace GaussianSplatting {

class GaussianAnimationStateMachine;
enum AnimationProperty : int;

// Change tracking structures
struct SplatChange {
    uint32_t index;
    uint8_t changed_properties; // Bitfield: position|color|opacity|scale|rotation
    Vector3 position;
    Color color;
    float opacity;
    Vector3 scale;
    Quaternion rotation;
};

struct AnimationChange {
    int clip_index;
    AnimationProperty property;
    uint8_t change_type; // 0=modified, 1=added, 2=removed
    Dictionary data; // Flexible storage for keyframes, tracks, etc.
};

// Incremental save file format
static const uint32_t INCREMENTAL_MAGIC = 0x47534946; // "GSIF" - Gaussian Scene Incremental File
static const uint16_t INCREMENTAL_VERSION = 1;

// Layout version tracks the on-disk struct layout of SplatChange / ChangeEntry.
// Bump this whenever the binary layout of serialised change entries changes
// (e.g. adding fields, reordering members, changing sizes).  The reader must
// reject files whose layout version differs from the compiled-in constant to
// prevent silent data corruption from mismatched struct packing.
static const uint16_t INCREMENTAL_SAVER_LAYOUT_VERSION = 1;

enum class ChangeType : uint8_t {
    SPLAT_MODIFIED = 0,
    SPLAT_ADDED = 1,
    SPLAT_REMOVED = 2,
    ANIMATION_MODIFIED = 3,
    METADATA_MODIFIED = 4
};

struct ChangeEntry {
    ChangeType type;
    uint32_t data_offset;
    uint32_t data_size;
    uint64_t timestamp;
};

class GaussianIncrementalSaver : public Resource {
    GDCLASS(GaussianIncrementalSaver, Resource);

private:
    // Change tracking
    LocalVector<SplatChange> splat_changes;
    LocalVector<AnimationChange> animation_changes;
    HashMap<uint32_t, int> splat_index_to_change;
    struct MetadataDelta {
        Variant old_value;
        Variant new_value;
    };
    HashMap<String, MetadataDelta> metadata_changes;

    // Baseline tracking
    String baseline_file_path;
    uint64_t baseline_timestamp = 0;
    uint32_t baseline_splat_count = 0;

    // Save settings
    float auto_save_interval = 30.0f; // seconds
    uint32_t max_changes_before_full_save = 10000;
    bool enable_change_compression = true;

    // Internal state
    bool is_tracking = false;
    uint64_t last_save_time = 0;
    uint32_t accumulated_changes = 0;

    // Internal methods
    void _track_splat_change(uint32_t index, const Gaussian& old_splat, const Gaussian& new_splat);
    void _track_animation_change(int clip_index, AnimationProperty property, uint8_t change_type, const Dictionary& data);
    Error _write_change_entry(Ref<FileAccess> file, const ChangeEntry& entry) const;
    Error _read_change_entry(Ref<FileAccess> file, ChangeEntry& entry) const;
    Error _apply_splat_changes(::GaussianData* gaussian_data) const;
    Error _apply_metadata_changes(GaussianAnimationStateMachine* animation) const;
    Error _apply_animation_changes(GaussianAnimationStateMachine* animation) const;

protected:
    static void _bind_methods();

public:
    GaussianIncrementalSaver();
    ~GaussianIncrementalSaver();

    // Change tracking control
    void start_tracking(const String& baseline_file);
    void stop_tracking();
    bool is_tracking_enabled() const { return is_tracking; }

    // Manual change recording
    void record_splat_change(uint32_t index, const Gaussian& old_splat, const Gaussian& new_splat);
    void record_animation_change(int clip_index, AnimationProperty property, const Dictionary& old_data, const Dictionary& new_data);
    void record_metadata_change(const String& key, const Variant& old_value, const Variant& new_value);

    // Incremental save operations
    Error save_changes(const String& incremental_file_path);
    Error load_and_apply_changes(const String& incremental_file_path, ::GaussianData* gaussian_data, GaussianAnimationStateMachine* animation = nullptr);
    Error load_and_apply_changes_bind(const String& incremental_file_path, const Ref<::GaussianData>& gaussian_data, const Ref<GaussianAnimationStateMachine>& animation = Ref<GaussianAnimationStateMachine>());
    Error merge_incremental_files(const Array& incremental_files, const String& output_file);

    // Baseline management
    Error create_baseline(const String& baseline_file_path, const ::GaussianData* gaussian_data, const GaussianAnimationStateMachine* animation = nullptr);
    Error create_baseline_bind(const String& baseline_file_path, const Ref<::GaussianData>& gaussian_data, const Ref<GaussianAnimationStateMachine>& animation = Ref<GaussianAnimationStateMachine>());
    Error update_baseline(const String& new_baseline_file_path);
    String get_baseline_file() const { return baseline_file_path; }

    // Change analysis
    uint32_t get_change_count() const { return splat_changes.size() + animation_changes.size(); }
    uint32_t get_splat_change_count() const { return splat_changes.size(); }
    uint32_t get_animation_change_count() const { return animation_changes.size(); }
    Array get_changed_splat_indices() const;
    Dictionary get_change_statistics() const;

    // Auto-save functionality
    void set_auto_save_interval(float seconds) { auto_save_interval = MAX(1.0f, seconds); }
    float get_auto_save_interval() const { return auto_save_interval; }

    void set_max_changes_before_full_save(uint32_t count) { max_changes_before_full_save = count; }
    uint32_t get_max_changes_before_full_save() const { return max_changes_before_full_save; }

    bool should_auto_save() const;
    bool should_create_full_save() const;

    // Change compression
    void set_enable_change_compression(bool enable) { enable_change_compression = enable; }
    bool get_enable_change_compression() const { return enable_change_compression; }

    // Utility methods
    void clear_changes();
    Error validate_incremental_file(const String& file_path) const;
    Dictionary get_incremental_file_info(const String& file_path) const;
    uint64_t estimate_save_size() const;

    // Static helpers
    static bool is_incremental_file(const String& file_path);
    static String get_incremental_extension() { return "gsif"; }
    static Array find_incremental_files(const String& directory, const String& basename);
};

} // namespace GaussianSplatting

#endif // GAUSSIAN_INCREMENTAL_SAVER_H
