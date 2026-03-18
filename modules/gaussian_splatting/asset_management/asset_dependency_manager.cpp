#include "asset_dependency_manager.h"

#include <functional>

#include "core/crypto/crypto.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/string/print_string.h"
#include "core/os/time.h"
#include "core/variant/variant_parser.h"
#include "../logger/gs_logger.h"

AssetDependencyManager *AssetDependencyManager::singleton = nullptr;

namespace {

static bool asset_id_from_string_checked(const String &p_asset_id, AssetID &r_out) {
    if (p_asset_id.is_empty()) {
        return false;
    }

    AssetID parsed = AssetID::from_string(p_asset_id);
    if (!parsed.is_valid()) {
        return false;
    }

    r_out = parsed;
    return true;
}

} // namespace

// AssetID implementation
String AssetID::to_string() const {
    return String::num_uint64(id_high, 16).pad_zeros(16) + String::num_uint64(id_low, 16).pad_zeros(16);
}

AssetID AssetID::from_string(const String &p_str) {
    if (p_str.length() != 32) {
        return AssetID();
    }

    String high_str = p_str.substr(0, 16);
    String low_str = p_str.substr(16, 16);

    if (!high_str.is_valid_hex_number(false) || !low_str.is_valid_hex_number(false)) {
        return AssetID();
    }

    uint64_t high = static_cast<uint64_t>(high_str.hex_to_int());
    uint64_t low = static_cast<uint64_t>(low_str.hex_to_int());

    return AssetID(high, low);
}

AssetID AssetID::generate_from_path(const String &p_path) {
    if (p_path.is_empty()) {
        return AssetID();
    }

    PackedByteArray path_bytes = p_path.to_utf8_buffer();
    unsigned char hash_bytes[32] = {};

    if (CryptoCore::sha256(path_bytes.ptr(), path_bytes.size(), hash_bytes) == OK) {
        uint64_t high = 0;
        uint64_t low = 0;

        for (int i = 0; i < 8; i++) {
            high = (high << 8) | hash_bytes[i];
            low = (low << 8) | hash_bytes[i + 8];
        }

        return AssetID(high, low);
    }

    // Fallback to simple hash-based generation if SHA-256 is unavailable.
    uint64_t simple_hash = p_path.hash64();
    return AssetID(simple_hash, ~simple_hash);
}

AssetID AssetID::generate_unique() {
    // Generate cryptographically secure unique ID
    Ref<Crypto> crypto = Crypto::create();
    if (crypto.is_valid()) {
        PackedByteArray random_bytes = crypto->generate_random_bytes(16);
        if (random_bytes.size() == 16) {
            uint64_t high = 0, low = 0;
            for (int i = 0; i < 8; i++) {
                high = (high << 8) | random_bytes[i];
                low = (low << 8) | random_bytes[i + 8];
            }
            return AssetID(high, low);
        }
    }

    // Fallback to time-based unique ID
    uint64_t timestamp = Time::get_singleton()->get_ticks_usec();
    static uint32_t counter = 0;
    counter++;

    return AssetID(timestamp, (uint64_t(counter) << 32) | (timestamp & 0xFFFFFFFF));
}

// AssetDependencyManager implementation
void AssetDependencyManager::_bind_methods() {
    ClassDB::bind_method(D_METHOD("register_asset", "file_path", "custom_id"), &AssetDependencyManager::register_asset_binding, DEFVAL(String()));
    ClassDB::bind_method(D_METHOD("unregister_asset", "asset_id"), &AssetDependencyManager::unregister_asset_binding);
    ClassDB::bind_method(D_METHOD("get_asset_id_from_path", "file_path"), &AssetDependencyManager::get_asset_id_from_path_binding);
    ClassDB::bind_method(D_METHOD("get_asset_path_from_id", "asset_id"), &AssetDependencyManager::get_asset_path_from_id_binding);
    ClassDB::bind_method(D_METHOD("has_asset", "asset_id"), &AssetDependencyManager::has_asset_binding);

    ClassDB::bind_method(D_METHOD("add_dependency", "asset_id", "dependency_id"), &AssetDependencyManager::add_dependency_binding);
    ClassDB::bind_method(D_METHOD("remove_dependency", "asset_id", "dependency_id"), &AssetDependencyManager::remove_dependency_binding);
    ClassDB::bind_method(D_METHOD("get_dependencies_array", "asset_path", "recursive"), &AssetDependencyManager::get_dependencies_array, DEFVAL(false));
    ClassDB::bind_method(D_METHOD("get_dependents_array", "asset_path", "recursive"), &AssetDependencyManager::get_dependents_array, DEFVAL(false));

    ClassDB::bind_method(D_METHOD("detect_dependency_cycles"), &AssetDependencyManager::detect_dependency_cycles);
    ClassDB::bind_method(D_METHOD("has_circular_dependency", "asset_id"), &AssetDependencyManager::has_circular_dependency_binding);
    ClassDB::bind_method(D_METHOD("validate_asset", "asset_id"), &AssetDependencyManager::validate_asset_binding);
    ClassDB::bind_method(D_METHOD("validate_all_assets"), &AssetDependencyManager::validate_all_assets);

    ClassDB::bind_method(D_METHOD("set_current_project", "project_id"), &AssetDependencyManager::set_current_project);
    ClassDB::bind_method(D_METHOD("get_current_project"), &AssetDependencyManager::get_current_project);

    ClassDB::bind_method(D_METHOD("get_asset_id_string", "asset_path"), &AssetDependencyManager::get_asset_id_string);
    ClassDB::bind_method(D_METHOD("print_dependency_report"), &AssetDependencyManager::print_dependency_report);
}

AssetDependencyManager::AssetDependencyManager() {
    if (singleton == nullptr) {
        singleton = this;
    }
    current_project_id = "default";
}

AssetDependencyManager::~AssetDependencyManager() {
    if (singleton == this) {
        singleton = nullptr;
    }
}

Error AssetDependencyManager::register_asset(const String &p_file_path, const AssetID &p_custom_id) {
    if (p_file_path.is_empty()) {
        return ERR_INVALID_PARAMETER;
    }

    // Check if file exists
    if (!FileAccess::exists(p_file_path)) {
        print_error("AssetDependencyManager: File does not exist: " + p_file_path);
        return ERR_FILE_NOT_FOUND;
    }

    // Generate or use custom asset ID
    AssetID asset_id = p_custom_id.is_valid() ? p_custom_id : AssetID::generate_from_path(p_file_path);

    // Check for conflicts
    if (asset_registry.has(asset_id)) {
        AssetMetadata existing = asset_registry[asset_id];
        if (existing.file_path != p_file_path) {
            print_error("AssetDependencyManager: Asset ID conflict for: " + p_file_path);
            return ERR_ALREADY_EXISTS;
        }
        // Update existing asset
        return update_asset_metadata(asset_id, p_file_path);
    }

    // Create new asset metadata
    AssetMetadata asset_metadata(asset_id, p_file_path);
    asset_metadata.display_name = p_file_path.get_file().get_basename();
    asset_metadata.project_id = current_project_id;

    // Get file information
    Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::READ);
    if (file.is_valid()) {
        asset_metadata.file_size = file->get_length();
        asset_metadata.file_timestamp = FileAccess::get_modified_time(p_file_path);
    }

    // Register the asset
    asset_registry[asset_id] = asset_metadata;
    path_to_id_map[p_file_path] = asset_id;

    // Create dependency graph node
    dependency_graph[asset_id] = DependencyNode(asset_id);

    _invalidate_dependency_cache();

    GS_LOG_DEBUG(gs_logger::Category::GENERAL, "AssetDependencyManager: Registered asset: " + p_file_path + " (ID: " + asset_id.to_string() + ")");
    return OK;
}

Error AssetDependencyManager::unregister_asset(const AssetID &p_asset_id) {
    if (!asset_registry.has(p_asset_id)) {
        return ERR_DOES_NOT_EXIST;
    }

    AssetMetadata asset_metadata = asset_registry[p_asset_id];

    // Remove all dependencies and dependents
    if (dependency_graph.has(p_asset_id)) {
        DependencyNode &node = dependency_graph[p_asset_id];

        // Remove this asset from all its dependencies' dependent lists
        for (const AssetID &dep_id : node.incoming_edges) {
            if (dependency_graph.has(dep_id)) {
                dependency_graph[dep_id].outgoing_edges.erase(p_asset_id);
            }
        }

        // Remove this asset from all its dependents' dependency lists
        for (const AssetID &dependent_id : node.outgoing_edges) {
            if (dependency_graph.has(dependent_id)) {
                dependency_graph[dependent_id].incoming_edges.erase(p_asset_id);
            }
        }

        dependency_graph.erase(p_asset_id);
    }

    // Remove from registries
    path_to_id_map.erase(asset_metadata.file_path);
    asset_registry.erase(p_asset_id);

    _invalidate_dependency_cache();

    GS_LOG_DEBUG(gs_logger::Category::GENERAL, "AssetDependencyManager: Unregistered asset: " + asset_metadata.file_path);
    return OK;
}

Error AssetDependencyManager::update_asset_metadata(const AssetID &p_asset_id, const String &p_file_path) {
    if (!asset_registry.has(p_asset_id)) {
        return ERR_DOES_NOT_EXIST;
    }

    AssetMetadata &asset_metadata = asset_registry[p_asset_id];

    // Update file path if provided
    if (!p_file_path.is_empty() && p_file_path != asset_metadata.file_path) {
        path_to_id_map.erase(asset_metadata.file_path);
        asset_metadata.file_path = p_file_path;
        path_to_id_map[p_file_path] = p_asset_id;
    }

    // Update file information
    if (FileAccess::exists(asset_metadata.file_path)) {
        Ref<FileAccess> file = FileAccess::open(asset_metadata.file_path, FileAccess::READ);
        if (file.is_valid()) {
            asset_metadata.file_size = file->get_length();
            asset_metadata.file_timestamp = FileAccess::get_modified_time(asset_metadata.file_path);
        }
    }

    return OK;
}

AssetID AssetDependencyManager::get_asset_id_from_path(const String &p_file_path) const {
    if (path_to_id_map.has(p_file_path)) {
        return path_to_id_map[p_file_path];
    }
    return AssetID();
}

String AssetDependencyManager::get_asset_path_from_id(const AssetID &p_asset_id) const {
    if (asset_registry.has(p_asset_id)) {
        return asset_registry[p_asset_id].file_path;
    }
    return String();
}

AssetMetadata AssetDependencyManager::get_asset_metadata(const AssetID &p_asset_id) const {
    if (asset_registry.has(p_asset_id)) {
        return asset_registry[p_asset_id];
    }
    return AssetMetadata();
}

bool AssetDependencyManager::has_asset(const AssetID &p_asset_id) const {
    return asset_registry.has(p_asset_id);
}

Vector<AssetID> AssetDependencyManager::get_all_assets() const {
    Vector<AssetID> assets;
    for (const KeyValue<AssetID, AssetMetadata> &entry : asset_registry) {
        assets.push_back(entry.key);
    }
    return assets;
}

Error AssetDependencyManager::add_dependency(const AssetID &p_asset_id, const AssetID &p_dependency_id) {
    if (!asset_registry.has(p_asset_id) || !asset_registry.has(p_dependency_id)) {
        return ERR_DOES_NOT_EXIST;
    }

    if (p_asset_id == p_dependency_id) {
        GS_LOG_WARN_DEFAULT("AssetDependencyManager::add_dependency: asset depends on itself (ID: " + p_asset_id.to_string() + "), ignoring self-reference");
        return ERR_INVALID_PARAMETER; // Self-dependency
    }

    // Add to dependency graph
    if (!dependency_graph.has(p_asset_id)) {
        dependency_graph[p_asset_id] = DependencyNode(p_asset_id);
    }
    if (!dependency_graph.has(p_dependency_id)) {
        dependency_graph[p_dependency_id] = DependencyNode(p_dependency_id);
    }

    dependency_graph[p_asset_id].incoming_edges.insert(p_dependency_id);
    dependency_graph[p_dependency_id].outgoing_edges.insert(p_asset_id);

    // Update metadata
    asset_registry[p_asset_id].dependencies.insert(p_dependency_id);
    asset_registry[p_dependency_id].dependents.insert(p_asset_id);

    // Check for cycles (optional immediate validation)
    if (has_circular_dependency(p_asset_id)) {
        // Remove the dependency that created the cycle
        remove_dependency(p_asset_id, p_dependency_id);
        return ERR_CYCLIC_LINK;
    }

    _invalidate_dependency_cache();
    return OK;
}

Error AssetDependencyManager::remove_dependency(const AssetID &p_asset_id, const AssetID &p_dependency_id) {
    if (!asset_registry.has(p_asset_id) || !asset_registry.has(p_dependency_id)) {
        return ERR_DOES_NOT_EXIST;
    }

    // Remove from dependency graph
    if (dependency_graph.has(p_asset_id)) {
        dependency_graph[p_asset_id].incoming_edges.erase(p_dependency_id);
    }
    if (dependency_graph.has(p_dependency_id)) {
        dependency_graph[p_dependency_id].outgoing_edges.erase(p_asset_id);
    }

    // Update metadata
    asset_registry[p_asset_id].dependencies.erase(p_dependency_id);
    asset_registry[p_dependency_id].dependents.erase(p_asset_id);

    _invalidate_dependency_cache();
    return OK;
}

Vector<AssetID> AssetDependencyManager::get_dependencies(const AssetID &p_asset_id, bool p_recursive) const {
    if (!asset_registry.has(p_asset_id)) {
        return Vector<AssetID>();
    }

    if (!p_recursive) {
        Vector<AssetID> deps;
        for (const AssetID &dep_id : asset_registry[p_asset_id].dependencies) {
            deps.push_back(dep_id);
        }
        return deps;
    }

    if (dependency_cache_dirty) {
        resolved_dependency_cache.clear();
        dependency_cache_dirty = false;
    }

    const Vector<AssetID> *cached = resolved_dependency_cache.getptr(p_asset_id);
    if (cached != nullptr) {
        return *cached;
    }
    Vector<AssetID> resolved = _topological_sort_dependencies(p_asset_id);
    resolved_dependency_cache.insert(p_asset_id, resolved);
    return resolved;
}

Vector<AssetID> AssetDependencyManager::get_dependents(const AssetID &p_asset_id, bool p_recursive) const {
    if (!asset_registry.has(p_asset_id)) {
        return Vector<AssetID>();
    }

    HashSet<AssetID, AssetIDHasher> visited;
    Vector<AssetID> result;

    std::function<void(const AssetID &)> collect_dependents = [&](const AssetID &current_id) {
        if (visited.has(current_id)) {
            return;
        }
        visited.insert(current_id);

        if (asset_registry.has(current_id)) {
            for (const AssetID &dependent_id : asset_registry[current_id].dependents) {
                result.push_back(dependent_id);
                if (p_recursive) {
                    collect_dependents(dependent_id);
                }
            }
        }
    };

    collect_dependents(p_asset_id);
    return result;
}

Vector<AssetID> AssetDependencyManager::resolve_load_order(const AssetID &p_asset_id) const {
    Vector<AssetID> load_order = get_dependencies(p_asset_id, true);
    if (asset_registry.has(p_asset_id)) {
        // Load the requested asset after all of its dependencies.
        load_order.push_back(p_asset_id);
    }
    return load_order;
}

Error AssetDependencyManager::detect_dependency_cycles() const {
    HashSet<AssetID, AssetIDHasher> visiting;

    for (const KeyValue<AssetID, AssetMetadata> &entry : asset_registry) {
        Error err = _detect_cycles_recursive(entry.key, visiting);
        if (err != OK) {
            return err;
        }
    }

    return OK;
}

bool AssetDependencyManager::has_circular_dependency(const AssetID &p_asset_id) const {
    HashSet<AssetID, AssetIDHasher> visiting;
    return _detect_cycles_recursive(p_asset_id, visiting) != OK;
}

Error AssetDependencyManager::_detect_cycles_recursive(const AssetID &p_asset_id, HashSet<AssetID, AssetIDHasher> &p_visiting) const {
    if (p_visiting.has(p_asset_id)) {
        return ERR_CYCLIC_LINK; // Cycle detected
    }

    if (!asset_registry.has(p_asset_id)) {
        return OK;
    }

    p_visiting.insert(p_asset_id);

    for (const AssetID &dep_id : asset_registry[p_asset_id].dependencies) {
        Error err = _detect_cycles_recursive(dep_id, p_visiting);
        if (err != OK) {
            return err;
        }
    }

    p_visiting.erase(p_asset_id);
    return OK;
}

Vector<AssetID> AssetDependencyManager::_topological_sort_dependencies(const AssetID &p_asset_id) const {
    Vector<AssetID> result;
    HashSet<AssetID, AssetIDHasher> visited;

    std::function<void(const AssetID &)> visit = [&](const AssetID &current_id) {
        if (visited.has(current_id) || !asset_registry.has(current_id)) {
            return;
        }

        visited.insert(current_id);

        // Visit all dependencies first (depth-first)
        for (const AssetID &dep_id : asset_registry[current_id].dependencies) {
            visit(dep_id);
        }

        // Add current asset after its dependencies
        result.push_back(current_id);
    };

    // Recursive dependency queries should only return dependencies,
    // not the queried asset itself.
    if (asset_registry.has(p_asset_id)) {
        for (const AssetID &dep_id : asset_registry[p_asset_id].dependencies) {
            visit(dep_id);
        }
    }
    return result;
}

void AssetDependencyManager::_invalidate_dependency_cache() {
    dependency_cache_dirty = true;
    resolved_dependency_cache.clear();
}

Error AssetDependencyManager::_validate_asset_file(const AssetMetadata &p_metadata) const {
    if (!FileAccess::exists(p_metadata.file_path)) {
        return ERR_FILE_NOT_FOUND;
    }

    uint64_t current_timestamp = FileAccess::get_modified_time(p_metadata.file_path);
    if (current_timestamp != p_metadata.file_timestamp) {
        return ERR_FILE_CORRUPT; // File has been modified
    }

    return OK;
}

// Godot integration methods
Array AssetDependencyManager::get_dependencies_array(const String &p_asset_path, bool p_recursive) const {
    AssetID asset_id = get_asset_id_from_path(p_asset_path);
    Vector<AssetID> deps = get_dependencies(asset_id, p_recursive);

    Array result;
    for (const AssetID &dep_id : deps) {
        String dep_path = get_asset_path_from_id(dep_id);
        if (!dep_path.is_empty()) {
            result.push_back(dep_path);
        }
    }
    return result;
}

Array AssetDependencyManager::get_dependents_array(const String &p_asset_path, bool p_recursive) const {
    AssetID asset_id = get_asset_id_from_path(p_asset_path);
    Vector<AssetID> dependents = get_dependents(asset_id, p_recursive);

    Array result;
    for (const AssetID &dependent_id : dependents) {
        String dependent_path = get_asset_path_from_id(dependent_id);
        if (!dependent_path.is_empty()) {
            result.push_back(dependent_path);
        }
    }
    return result;
}

String AssetDependencyManager::get_asset_id_string(const String &p_asset_path) const {
    AssetID asset_id = get_asset_id_from_path(p_asset_path);
    return asset_id.to_string();
}

AssetDependencyManager::DependencyStats AssetDependencyManager::get_dependency_statistics() const {
    DependencyStats stats;
    stats.total_assets = asset_registry.size();

    uint32_t total_dependency_count = 0;
    uint32_t missing_files = 0;

    for (const KeyValue<AssetID, AssetMetadata> &entry : asset_registry) {
        const AssetMetadata &asset_metadata = entry.value;
        total_dependency_count += asset_metadata.dependencies.size();

        if (!FileAccess::exists(asset_metadata.file_path)) {
            missing_files++;
        }
    }

    stats.total_dependencies = total_dependency_count;
    stats.missing_files = missing_files;
    stats.average_dependency_depth = stats.total_assets > 0 ? float(total_dependency_count) / float(stats.total_assets) : 0.0f;

    // Count circular dependencies
    for (const KeyValue<AssetID, AssetMetadata> &entry : asset_registry) {
        if (has_circular_dependency(entry.key)) {
            stats.circular_dependencies++;
        }
    }

    return stats;
}

void AssetDependencyManager::print_dependency_report() const {
    DependencyStats stats = get_dependency_statistics();

    GS_LOG_INFO_DEFAULT("=== Asset Dependency Report ===");
    GS_LOG_INFO_DEFAULT("Total Assets: " + itos(stats.total_assets));
    GS_LOG_INFO_DEFAULT("Total Dependencies: " + itos(stats.total_dependencies));
    GS_LOG_INFO_DEFAULT("Average Dependency Depth: " + rtos(stats.average_dependency_depth));
    GS_LOG_INFO_DEFAULT("Circular Dependencies: " + itos(stats.circular_dependencies));
    GS_LOG_INFO_DEFAULT("Missing Files: " + itos(stats.missing_files));
    GS_LOG_INFO_DEFAULT("Orphaned Assets: " + itos(stats.orphaned_assets));
    GS_LOG_INFO_DEFAULT("Current Project: " + current_project_id);
    GS_LOG_INFO_DEFAULT("==============================");
}

Error AssetDependencyManager::register_asset_binding(const String &p_file_path, const String &p_custom_id) {
    if (!p_custom_id.is_empty()) {
        AssetID custom_id = AssetID::from_string(p_custom_id);
        if (!custom_id.is_valid()) {
            return ERR_INVALID_PARAMETER;
        }
        return register_asset(p_file_path, custom_id);
    }

    return register_asset(p_file_path);
}

Error AssetDependencyManager::unregister_asset_binding(const String &p_asset_id) {
    AssetID parsed_id;
    if (!asset_id_from_string_checked(p_asset_id, parsed_id)) {
        return ERR_INVALID_PARAMETER;
    }

    return unregister_asset(parsed_id);
}

String AssetDependencyManager::get_asset_id_from_path_binding(const String &p_file_path) const {
    AssetID asset_id = get_asset_id_from_path(p_file_path);
    return asset_id.to_string();
}

String AssetDependencyManager::get_asset_path_from_id_binding(const String &p_asset_id) const {
    AssetID parsed_id;
    if (!asset_id_from_string_checked(p_asset_id, parsed_id)) {
        return String();
    }
    return get_asset_path_from_id(parsed_id);
}

bool AssetDependencyManager::has_asset_binding(const String &p_asset_id) const {
    AssetID parsed_id;
    if (!asset_id_from_string_checked(p_asset_id, parsed_id)) {
        return false;
    }
    return has_asset(parsed_id);
}

Error AssetDependencyManager::add_dependency_binding(const String &p_asset_id, const String &p_dependency_id) {
    AssetID asset_id;
    AssetID dependency_id;
    if (!asset_id_from_string_checked(p_asset_id, asset_id) || !asset_id_from_string_checked(p_dependency_id, dependency_id)) {
        return ERR_INVALID_PARAMETER;
    }

    return add_dependency(asset_id, dependency_id);
}

Error AssetDependencyManager::remove_dependency_binding(const String &p_asset_id, const String &p_dependency_id) {
    AssetID asset_id;
    AssetID dependency_id;
    if (!asset_id_from_string_checked(p_asset_id, asset_id) || !asset_id_from_string_checked(p_dependency_id, dependency_id)) {
        return ERR_INVALID_PARAMETER;
    }

    return remove_dependency(asset_id, dependency_id);
}

bool AssetDependencyManager::has_circular_dependency_binding(const String &p_asset_id) const {
    AssetID parsed_id;
    if (!asset_id_from_string_checked(p_asset_id, parsed_id)) {
        return false;
    }
    return has_circular_dependency(parsed_id);
}

Error AssetDependencyManager::validate_asset_binding(const String &p_asset_id) const {
    AssetID parsed_id;
    if (!asset_id_from_string_checked(p_asset_id, parsed_id)) {
        return ERR_INVALID_PARAMETER;
    }
    return validate_asset(parsed_id);
}

// Stub implementations for remaining methods (would be fully implemented in production)
Error AssetDependencyManager::set_current_project(const String &p_project_id) {
    current_project_id = p_project_id;
    return OK;
}

Error AssetDependencyManager::validate_asset(const AssetID &p_asset_id) const {
    if (!asset_registry.has(p_asset_id)) {
        return ERR_DOES_NOT_EXIST;
    }
    return _validate_asset_file(asset_registry[p_asset_id]);
}

Error AssetDependencyManager::validate_all_assets() const {
    Error first_error = OK;
    for (const KeyValue<AssetID, AssetMetadata> &entry : asset_registry) {
        Error err = _validate_asset_file(entry.value);
        if (err != OK) {
            print_error("Asset validation failed: " + entry.value.file_path);
            if (first_error == OK) {
                first_error = err;
            }
        }
    }
    return first_error;
}

Error AssetDependencyManager::export_asset_for_sharing(const AssetID &p_asset_id, const String &p_export_path) {
    if (!p_asset_id.is_valid() || p_export_path.is_empty()) {
        return ERR_INVALID_PARAMETER;
    }
    WARN_PRINT("AssetDependencyManager::export_asset_for_sharing is not implemented.");
    return ERR_UNAVAILABLE;
}

Error AssetDependencyManager::import_shared_asset(const String &p_import_path, const String &p_target_project) {
    if (p_import_path.is_empty()) {
        return ERR_INVALID_PARAMETER;
    }
    (void)p_target_project;
    WARN_PRINT("AssetDependencyManager::import_shared_asset is not implemented.");
    return ERR_UNAVAILABLE;
}

Vector<String> AssetDependencyManager::get_available_projects() const {
    Vector<String> projects;
    if (!current_project_id.is_empty()) {
        projects.push_back(current_project_id);
    }

    for (const KeyValue<String, HashMap<AssetID, AssetMetadata, AssetIDHasher>> &entry : project_registries) {
        if (entry.key != current_project_id) {
            projects.push_back(entry.key);
        }
    }

    return projects;
}

Error AssetDependencyManager::create_asset_version(const AssetID &p_asset_id, const String &p_version_notes) {
    if (!p_asset_id.is_valid()) {
        return ERR_INVALID_PARAMETER;
    }
    (void)p_version_notes;
    WARN_PRINT("AssetDependencyManager::create_asset_version is not implemented.");
    return ERR_UNAVAILABLE;
}

Vector<uint32_t> AssetDependencyManager::get_asset_versions(const AssetID &p_asset_id) const {
    Vector<uint32_t> versions;
    if (asset_registry.has(p_asset_id)) {
        versions.push_back(asset_registry[p_asset_id].version);
    }
    return versions;
}

Error AssetDependencyManager::revert_to_version(const AssetID &p_asset_id, uint32_t p_version) {
    if (!p_asset_id.is_valid() || p_version == 0) {
        return ERR_INVALID_PARAMETER;
    }
    WARN_PRINT("AssetDependencyManager::revert_to_version is not implemented.");
    return ERR_UNAVAILABLE;
}

Error AssetDependencyManager::repair_missing_dependencies(const AssetID &p_asset_id) {
    if (!p_asset_id.is_valid()) {
        return ERR_INVALID_PARAMETER;
    }
    WARN_PRINT("AssetDependencyManager::repair_missing_dependencies is not implemented.");
    return ERR_UNAVAILABLE;
}

void AssetDependencyManager::optimize_dependency_graph() {
    if (rebuild_dependency_cache() != OK) {
        WARN_PRINT("AssetDependencyManager::optimize_dependency_graph could not rebuild dependency cache.");
    }
}

Error AssetDependencyManager::cleanup_orphaned_assets() {
    WARN_PRINT("AssetDependencyManager::cleanup_orphaned_assets is not implemented.");
    return ERR_UNAVAILABLE;
}

Error AssetDependencyManager::rebuild_dependency_cache() {
    resolved_dependency_cache.clear();
    for (const KeyValue<AssetID, AssetMetadata> &entry : asset_registry) {
        resolved_dependency_cache[entry.key] = _topological_sort_dependencies(entry.key);
    }
    dependency_cache_dirty = false;
    return OK;
}
