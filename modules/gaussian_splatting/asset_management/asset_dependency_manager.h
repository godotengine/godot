#ifndef ASSET_DEPENDENCY_MANAGER_H
#define ASSET_DEPENDENCY_MANAGER_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/vector.h"
#include "core/string/ustring.h"
#include "core/crypto/crypto_core.h"
#include "core/templates/hashfuncs.h"
#include "../core/gaussian_splat_asset.h"

// Universal Asset ID with collision-resistant hashing
struct AssetID {
    uint64_t id_high;
    uint64_t id_low;

    AssetID() : id_high(0), id_low(0) {}
    AssetID(uint64_t high, uint64_t low) : id_high(high), id_low(low) {}

    bool operator==(const AssetID &other) const {
        return id_high == other.id_high && id_low == other.id_low;
    }

    bool operator!=(const AssetID &other) const {
        return !(*this == other);
    }

    String to_string() const;
    static AssetID from_string(const String &p_str);
    static AssetID generate_from_path(const String &p_path);
    static AssetID generate_unique();

    bool is_valid() const { return id_high != 0 || id_low != 0; }
};

static _FORCE_INLINE_ uint32_t hash(const AssetID &p_id) {
    return hash_murmur3_one_64(p_id.id_high) ^ hash_murmur3_one_64(p_id.id_low);
}

// Hash specialization for AssetID
struct AssetIDHasher {
    static _FORCE_INLINE_ uint32_t hash(const AssetID &p_id) { return ::hash(p_id); }
};

// Asset metadata and dependency information
struct AssetMetadata {
    AssetID asset_id;
    String file_path;
    String display_name;
    uint64_t file_size = 0;
    uint64_t file_timestamp = 0;
    uint32_t version = 1;
    String project_id;
    HashSet<AssetID, AssetIDHasher> dependencies;
    HashSet<AssetID, AssetIDHasher> dependents;
    HashMap<String, Variant> custom_properties;

    AssetMetadata() = default;
    AssetMetadata(const AssetID &p_id, const String &p_path) : asset_id(p_id), file_path(p_path) {}
};

// Dependency graph node for cycle detection and resolution ordering
struct DependencyNode {
    AssetID asset_id;
    HashSet<AssetID, AssetIDHasher> incoming_edges;  // Dependencies
    HashSet<AssetID, AssetIDHasher> outgoing_edges;  // Dependents
    int32_t visit_state = 0;  // 0=unvisited, 1=visiting, 2=visited (for cycle detection)

    DependencyNode() = default;
    DependencyNode(const AssetID &p_id) : asset_id(p_id) {}
};

class AssetDependencyManager : public RefCounted {
    GDCLASS(AssetDependencyManager, RefCounted);

private:
    static AssetDependencyManager *singleton;

    // Core storage
    HashMap<AssetID, AssetMetadata, AssetIDHasher> asset_registry;
    HashMap<String, AssetID> path_to_id_map;
    HashMap<AssetID, DependencyNode, AssetIDHasher> dependency_graph;

    // Cross-project support
    HashMap<String, HashMap<AssetID, AssetMetadata, AssetIDHasher>> project_registries;
    String current_project_id;

    // Performance optimization
    mutable bool dependency_cache_dirty = true;
    mutable HashMap<AssetID, Vector<AssetID>, AssetIDHasher> resolved_dependency_cache;

    // Internal methods
    Error _detect_cycles_recursive(const AssetID &p_asset_id, HashSet<AssetID, AssetIDHasher> &p_visiting) const;
    Vector<AssetID> _topological_sort_dependencies(const AssetID &p_asset_id) const;
    void _invalidate_dependency_cache();
    Error _validate_asset_file(const AssetMetadata &p_metadata) const;

protected:
    static void _bind_methods();

public:
    static AssetDependencyManager *get_singleton() { return singleton; }

    AssetDependencyManager();
    ~AssetDependencyManager();

    // Asset registration and metadata management
    Error register_asset(const String &p_file_path, const AssetID &p_custom_id = AssetID());
    Error unregister_asset(const AssetID &p_asset_id);
    Error update_asset_metadata(const AssetID &p_asset_id, const String &p_file_path = String());

    // Asset lookup and retrieval
    AssetID get_asset_id_from_path(const String &p_file_path) const;
    String get_asset_path_from_id(const AssetID &p_asset_id) const;
    AssetMetadata get_asset_metadata(const AssetID &p_asset_id) const;
    bool has_asset(const AssetID &p_asset_id) const;
    Vector<AssetID> get_all_assets() const;

    // Dependency management
    Error add_dependency(const AssetID &p_asset_id, const AssetID &p_dependency_id);
    Error remove_dependency(const AssetID &p_asset_id, const AssetID &p_dependency_id);
    Vector<AssetID> get_dependencies(const AssetID &p_asset_id, bool p_recursive = false) const;
    Vector<AssetID> get_dependents(const AssetID &p_asset_id, bool p_recursive = false) const;

    // Dependency resolution and validation
    Vector<AssetID> resolve_load_order(const AssetID &p_asset_id) const;
    Error detect_dependency_cycles() const;
    bool has_circular_dependency(const AssetID &p_asset_id) const;

    // Cross-project asset sharing
    Error set_current_project(const String &p_project_id);
    String get_current_project() const { return current_project_id; }
    Error export_asset_for_sharing(const AssetID &p_asset_id, const String &p_export_path);
    Error import_shared_asset(const String &p_import_path, const String &p_target_project = String());
    Vector<String> get_available_projects() const;

    // Asset versioning
    Error create_asset_version(const AssetID &p_asset_id, const String &p_version_notes = String());
    Vector<uint32_t> get_asset_versions(const AssetID &p_asset_id) const;
    Error revert_to_version(const AssetID &p_asset_id, uint32_t p_version);

    // Asset integrity and validation
    Error validate_asset(const AssetID &p_asset_id) const;
    Error validate_all_assets() const;
    Error repair_missing_dependencies(const AssetID &p_asset_id);

    // Performance and maintenance
    void optimize_dependency_graph();
    Error cleanup_orphaned_assets();
    Error rebuild_dependency_cache();

    // Statistics and diagnostics
    struct DependencyStats {
        uint32_t total_assets = 0;
        uint32_t total_dependencies = 0;
        uint32_t circular_dependencies = 0;
        uint32_t orphaned_assets = 0;
        uint32_t missing_files = 0;
        float average_dependency_depth = 0.0f;
    };

    DependencyStats get_dependency_statistics() const;
    void print_dependency_report() const;

    // Godot integration
    Array get_dependencies_array(const String &p_asset_path, bool p_recursive = false) const;
    Array get_dependents_array(const String &p_asset_path, bool p_recursive = false) const;
    String get_asset_id_string(const String &p_asset_path) const;

    // Script-friendly bindings that operate on string-based Asset IDs. These
    // helpers keep the public C++ API working with AssetID structures while the
    // scripting surface works with Variant-compatible types.
    Error register_asset_binding(const String &p_file_path, const String &p_custom_id = String());
    Error unregister_asset_binding(const String &p_asset_id);
    String get_asset_id_from_path_binding(const String &p_file_path) const;
    String get_asset_path_from_id_binding(const String &p_asset_id) const;
    bool has_asset_binding(const String &p_asset_id) const;
    Error add_dependency_binding(const String &p_asset_id, const String &p_dependency_id);
    Error remove_dependency_binding(const String &p_asset_id, const String &p_dependency_id);
    bool has_circular_dependency_binding(const String &p_asset_id) const;
    Error validate_asset_binding(const String &p_asset_id) const;

#ifdef TESTS_ENABLED
    bool is_dependency_cache_dirty_for_tests() const { return dependency_cache_dirty; }
    int32_t get_dependency_cache_entry_count_for_tests() const { return resolved_dependency_cache.size(); }
#endif
};

#endif // ASSET_DEPENDENCY_MANAGER_H
