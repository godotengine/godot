#ifndef GAUSSIAN_SCENE_SERIALIZER_H
#define GAUSSIAN_SCENE_SERIALIZER_H

#include "core/io/file_access.h"
#include "core/io/resource.h"
#include "core/variant/variant.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/string/ustring.h"
#include "../core/gaussian_data.h"
#include "../animation/animation_state_machine.h"

namespace GaussianSplatting {

// Binary format constants
static const uint32_t GAUSSIAN_SCENE_MAGIC = 0x47534346; // "GSCF" - Gaussian Scene File
static const uint16_t GAUSSIAN_SCENE_VERSION = 2;
// Minimum reader version that can open files written by this version.
// A v1 reader can still open v2 files by skipping unknown chunks.
static const uint16_t GAUSSIAN_SCENE_MIN_READER_VERSION = 1;

// Chunk types for extensible binary format
enum class ChunkType : uint32_t {
    HEADER = 0x48454144,        // "HEAD"
    GAUSSIAN_DATA = 0x47415553,  // "GAUS"
    ANIMATION_DATA = 0x414E494D, // "ANIM"
    METADATA = 0x4D455441,       // "META"
    ASSET_REFS = 0x41535354,     // "ASST"
    COMPRESSION = 0x434F4D50,    // "COMP"
    END_OF_FILE = 0x454F4600     // "EOF\0"
};

// Compression types
enum class CompressionType : uint8_t {
    NONE = 0,
    ZSTD = 1,
    LZ4 = 2
};

struct ChunkHeader {
    ChunkType type;
    uint32_t size;
    uint32_t checksum;
    uint32_t flags;
};

struct SceneHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t total_chunks;
    uint32_t splat_count;
    float bounds_min[3];
    float bounds_max[3];
    uint64_t creation_time;
    uint64_t modification_time;
    // v2+: minimum reader version required to open this file.  Readers whose
    // GAUSSIAN_SCENE_VERSION is at least this value may open the file and
    // gracefully skip any chunk types they do not recognize.
    uint16_t minimum_reader_version;
    uint16_t _reserved_v2; // Padding / reserved for future use.
};

// On-disk packed size of SceneHeader (may differ from sizeof(SceneHeader)
// because of compiler struct padding).
static const uint32_t SCENE_HEADER_PACKED_SIZE = 60;

struct AssetReference {
    String path;
    String type;
    uint64_t checksum;
    uint64_t file_size;
    uint64_t modification_time;
};

class GaussianSceneSerializer : public Resource {
    GDCLASS(GaussianSceneSerializer, Resource);

private:
    // Serialization settings
    CompressionType compression_type = CompressionType::ZSTD;
    int compression_level = 3;
    bool enable_checksum = true;
    bool incremental_mode = false;

    // Asset tracking
    LocalVector<AssetReference> asset_references;
    HashMap<String, int> asset_path_to_index;

    // Forward-compatibility: unknown chunks encountered during load are stored
    // here so they survive a round-trip (load -> re-save).  Each entry holds
    // the full raw chunk (ChunkHeader + payload) as a PackedByteArray.
    struct UnknownChunk {
        uint32_t type_raw; // Original ChunkType uint32_t value.
        uint32_t flags;
        uint32_t checksum;
        PackedByteArray payload;
    };
    Vector<UnknownChunk> unknown_chunks;

    // Cached checksum for the most recently written chunk. This keeps the
    // public API of `_write_chunk_header` simple while still allowing callers
    // to pre-compute the checksum once the payload has been assembled.
    mutable uint32_t pending_chunk_checksum = 0;

    // Internal serialization methods
    Error _write_chunk_header(Ref<FileAccess> file, ChunkType type, uint32_t size, uint32_t flags = 0);
    Error _write_scene_header(Ref<FileAccess> file, const SceneHeader& header);
    Error _write_gaussian_data_chunk(Ref<FileAccess> file, const ::GaussianData* gaussian_data);
    Error _write_animation_data_chunk(Ref<FileAccess> file, const GaussianAnimationStateMachine* animation);
    Error _write_metadata_chunk(Ref<FileAccess> file, const Dictionary& p_metadata);
    Error _write_asset_refs_chunk(Ref<FileAccess> file);

    Error _read_chunk_header(Ref<FileAccess> file, ChunkHeader& header) const;
    Error _read_scene_header(Ref<FileAccess> file, SceneHeader& header) const;
    Error _read_gaussian_data_chunk(Ref<FileAccess> file, const ChunkHeader& header, ::GaussianData* gaussian_data);
    Error _read_animation_data_chunk(Ref<FileAccess> file, const ChunkHeader& header, GaussianAnimationStateMachine* animation);
    Error _read_metadata_chunk(Ref<FileAccess> file, const ChunkHeader& header, Dictionary& r_metadata);
    Error _read_asset_refs_chunk(Ref<FileAccess> file, const ChunkHeader& header);

    // Compression helpers
    PackedByteArray _compress_data(const PackedByteArray& data, CompressionType type, bool& r_used_compression) const;
    PackedByteArray _decompress_data(const PackedByteArray& compressed_data, uint32_t original_size, CompressionType type) const;

    // Script bindings
    void set_compression_type_bind(int type);
    int get_compression_type_bind() const;

    // Checksum calculation
    uint32_t _calculate_checksum(const PackedByteArray& data) const;
    bool _verify_checksum(const PackedByteArray& data, uint32_t expected_checksum) const;

    // Asset management
    void _track_asset(const String& path, const String& type);
    bool _is_asset_modified(const AssetReference& ref) const;

protected:
    static void _bind_methods();

public:
    GaussianSceneSerializer();
    ~GaussianSceneSerializer();

    // Main serialization interface
    Error save_scene(const String& file_path, const ::GaussianData* gaussian_data,
                     const GaussianAnimationStateMachine* animation = nullptr,
                     const Dictionary& p_metadata = Dictionary());

    Error load_scene(const String& file_path, ::GaussianData* gaussian_data,
                     GaussianAnimationStateMachine* animation = nullptr,
                     Dictionary* r_metadata = nullptr);

    // Scripting helpers that work with Ref<> types.
    Error save_scene_bind(const String& file_path, const Ref<::GaussianData>& gaussian_data,
            const Ref<GaussianAnimationStateMachine>& animation, Dictionary p_metadata = Dictionary());
    Error load_scene_bind(const String& file_path, const Ref<::GaussianData>& gaussian_data,
            const Ref<GaussianAnimationStateMachine>& animation = Ref<GaussianAnimationStateMachine>(),
            Variant p_metadata = Variant());
    Error save_incremental_bind(const String& file_path, const String& base_file_path,
            const Ref<::GaussianData>& gaussian_data,
            const Ref<GaussianAnimationStateMachine>& animation);
    Error load_incremental_bind(const String& file_path, const String& base_file_path,
            const Ref<::GaussianData>& gaussian_data,
            const Ref<GaussianAnimationStateMachine>& animation);

    // Incremental save/load for large scenes
    Error save_incremental(const String& file_path, const String& base_file_path,
                          const ::GaussianData* gaussian_data,
                          const GaussianAnimationStateMachine* animation = nullptr);

    Error load_incremental(const String& file_path, const String& base_file_path,
                          ::GaussianData* gaussian_data,
                          GaussianAnimationStateMachine* animation = nullptr);

    // Asset management
    void add_asset_reference(const String& path, const String& type);
    void remove_asset_reference(const String& path);
    bool has_asset_reference(const String& path) const;
    Array get_asset_references() const;
    bool validate_assets() const;

    // Settings
    void set_compression_type(CompressionType type) { compression_type = type; }
    CompressionType get_compression_type() const { return compression_type; }

    void set_compression_level(int level) { compression_level = CLAMP(level, 1, 22); }
    int get_compression_level() const { return compression_level; }

    void set_enable_checksum(bool enable) { enable_checksum = enable; }
    bool get_enable_checksum() const { return enable_checksum; }

    void set_incremental_mode(bool enable) { incremental_mode = enable; }
    bool get_incremental_mode() const { return incremental_mode; }

    // Utility methods
    Error validate_file(const String& file_path) const;
    Dictionary get_file_info(const String& file_path) const;
    uint64_t get_file_size_estimate(const ::GaussianData* gaussian_data,
                                   const GaussianAnimationStateMachine* animation = nullptr) const;

    // Unknown chunk round-trip accessors
    int get_unknown_chunk_count() const { return unknown_chunks.size(); }
    void clear_unknown_chunks() { unknown_chunks.clear(); }

    // Migration registry stub — provides the hook point for future per-chunk
    // data migrations.  Currently returns ``data`` unchanged.
    static PackedByteArray _migrate_chunk_v1_to_v2(uint32_t chunk_type, const PackedByteArray& data);

    // Static helpers
    static bool is_gaussian_scene_file(const String& file_path);
    static String get_file_extension() { return "gsf"; } // Gaussian Scene File
    static Array get_supported_compression_types();
};

} // namespace GaussianSplatting

#endif // GAUSSIAN_SCENE_SERIALIZER_H
