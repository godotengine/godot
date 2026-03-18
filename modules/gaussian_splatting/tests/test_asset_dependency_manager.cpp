#include "test_macros.h"

#include "../asset_management/asset_dependency_manager.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"

#ifdef TESTS_ENABLED

namespace {

String _make_user_test_file_path(const String &p_prefix, const String &p_suffix = ".tmp") {
    const uint64_t ticks = OS::get_singleton() ? OS::get_singleton()->get_ticks_usec() : 0;
    return "user://" + p_prefix + "_" + itos(ticks) + p_suffix;
}

Error _write_test_file(const String &p_path, const String &p_contents = "test-data") {
    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
    if (file.is_null()) {
        return ERR_CANT_CREATE;
    }
    file->store_string(p_contents);
    file.unref();
    return OK;
}

void _remove_user_file(const String &p_user_path) {
    Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_USERDATA);
    if (dir.is_null()) {
        return;
    }
    dir->remove(p_user_path.get_file());
}

bool _string_vector_contains(const Vector<String> &p_values, const String &p_target) {
    for (int i = 0; i < p_values.size(); i++) {
        if (p_values[i] == p_target) {
            return true;
        }
    }
    return false;
}

bool _asset_id_vector_contains(const Vector<AssetID> &p_values, const AssetID &p_target) {
    for (int i = 0; i < p_values.size(); i++) {
        if (p_values[i] == p_target) {
            return true;
        }
    }
    return false;
}

} // namespace

namespace TestGaussianSplatting {

TEST_CASE("[GaussianSplatting][AssetDependencyManager] validate_all_assets returns non-OK on invalid asset file") {
    const String asset_path = _make_user_test_file_path("gs_asset_validation");
    _remove_user_file(asset_path);

    const Error write_err = _write_test_file(asset_path, "asset-validation-fixture");
    CHECK(write_err == OK);
    if (write_err != OK) {
        return;
    }

    Ref<AssetDependencyManager> manager;
    manager.instantiate();
    CHECK(manager.is_valid());
    if (!manager.is_valid()) {
        _remove_user_file(asset_path);
        return;
    }

    const Error register_err = manager->register_asset(asset_path);
    CHECK(register_err == OK);
    if (register_err != OK) {
        _remove_user_file(asset_path);
        return;
    }

    _remove_user_file(asset_path);

    const Error validate_err = manager->validate_all_assets();
    CHECK(validate_err == ERR_FILE_NOT_FOUND);
}

TEST_CASE("[GaussianSplatting][AssetDependencyManager] unimplemented APIs return ERR_UNAVAILABLE") {
    Ref<AssetDependencyManager> manager;
    manager.instantiate();
    CHECK(manager.is_valid());
    if (!manager.is_valid()) {
        return;
    }

    const AssetID asset_id = AssetID::generate_unique();

    CHECK(manager->export_asset_for_sharing(asset_id, "user://gs_export_fixture") == ERR_UNAVAILABLE);
    CHECK(manager->import_shared_asset("user://gs_import_fixture", "project_a") == ERR_UNAVAILABLE);
    CHECK(manager->create_asset_version(asset_id, "initial version") == ERR_UNAVAILABLE);
    CHECK(manager->revert_to_version(asset_id, 1) == ERR_UNAVAILABLE);
    CHECK(manager->repair_missing_dependencies(asset_id) == ERR_UNAVAILABLE);
    CHECK(manager->cleanup_orphaned_assets() == ERR_UNAVAILABLE);

    const Vector<String> projects = manager->get_available_projects();
    CHECK(_string_vector_contains(projects, manager->get_current_project()));
}

TEST_CASE("[GaussianSplatting][AssetDependencyManager] get_asset_versions reflects registered asset version") {
    const String asset_path = _make_user_test_file_path("gs_asset_versioning");
    _remove_user_file(asset_path);

    const Error write_err = _write_test_file(asset_path, "asset-version-fixture");
    CHECK(write_err == OK);
    if (write_err != OK) {
        return;
    }

    Ref<AssetDependencyManager> manager;
    manager.instantiate();
    CHECK(manager.is_valid());
    if (!manager.is_valid()) {
        _remove_user_file(asset_path);
        return;
    }

    const Error register_err = manager->register_asset(asset_path);
    CHECK(register_err == OK);
    if (register_err != OK) {
        _remove_user_file(asset_path);
        return;
    }

    const AssetID registered_id = manager->get_asset_id_from_path(asset_path);
    CHECK(registered_id.is_valid());
    if (!registered_id.is_valid()) {
        _remove_user_file(asset_path);
        return;
    }

    const Vector<uint32_t> versions = manager->get_asset_versions(registered_id);
    CHECK(versions.size() == 1);
    if (versions.size() == 1) {
        CHECK(versions[0] == 1);
    }

    _remove_user_file(asset_path);
}

TEST_CASE("[GaussianSplatting][AssetDependencyManager] recursive dependency cache populates and invalidates") {
    const String asset_a_path = _make_user_test_file_path("gs_asset_cache_a");
    const String asset_b_path = _make_user_test_file_path("gs_asset_cache_b");
    const String asset_c_path = _make_user_test_file_path("gs_asset_cache_c");

    _remove_user_file(asset_a_path);
    _remove_user_file(asset_b_path);
    _remove_user_file(asset_c_path);

    CHECK(_write_test_file(asset_a_path, "cache-a") == OK);
    CHECK(_write_test_file(asset_b_path, "cache-b") == OK);
    CHECK(_write_test_file(asset_c_path, "cache-c") == OK);

    Ref<AssetDependencyManager> manager;
    manager.instantiate();
    CHECK(manager.is_valid());
    if (!manager.is_valid()) {
        _remove_user_file(asset_a_path);
        _remove_user_file(asset_b_path);
        _remove_user_file(asset_c_path);
        return;
    }

    CHECK(manager->register_asset(asset_a_path) == OK);
    CHECK(manager->register_asset(asset_b_path) == OK);
    CHECK(manager->register_asset(asset_c_path) == OK);

    const AssetID asset_a_id = manager->get_asset_id_from_path(asset_a_path);
    const AssetID asset_b_id = manager->get_asset_id_from_path(asset_b_path);
    const AssetID asset_c_id = manager->get_asset_id_from_path(asset_c_path);
    CHECK(asset_a_id.is_valid());
    CHECK(asset_b_id.is_valid());
    CHECK(asset_c_id.is_valid());
    if (!asset_a_id.is_valid() || !asset_b_id.is_valid() || !asset_c_id.is_valid()) {
        _remove_user_file(asset_a_path);
        _remove_user_file(asset_b_path);
        _remove_user_file(asset_c_path);
        return;
    }

    CHECK(manager->is_dependency_cache_dirty_for_tests());
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 0);

    CHECK(manager->add_dependency(asset_a_id, asset_b_id) == OK);
    CHECK(manager->is_dependency_cache_dirty_for_tests());
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 0);

    const Vector<AssetID> deps_a_initial = manager->get_dependencies(asset_a_id, true);
    CHECK(!manager->is_dependency_cache_dirty_for_tests());
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 1);
    CHECK(_asset_id_vector_contains(deps_a_initial, asset_a_id));
    CHECK(_asset_id_vector_contains(deps_a_initial, asset_b_id));
    CHECK(deps_a_initial.size() == 2);

    const Vector<AssetID> deps_a_cached = manager->get_dependencies(asset_a_id, true);
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 1);
    CHECK(deps_a_cached.size() == deps_a_initial.size());

    const Vector<AssetID> deps_b_initial = manager->get_dependencies(asset_b_id, true);
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 2);
    CHECK(deps_b_initial.size() == 1);
    CHECK(_asset_id_vector_contains(deps_b_initial, asset_b_id));

    CHECK(manager->add_dependency(asset_a_id, asset_c_id) == OK);
    CHECK(manager->is_dependency_cache_dirty_for_tests());
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 0);

    const Vector<AssetID> deps_a_after_add = manager->get_dependencies(asset_a_id, true);
    CHECK(!manager->is_dependency_cache_dirty_for_tests());
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 1);
    CHECK(deps_a_after_add.size() == 3);
    CHECK(_asset_id_vector_contains(deps_a_after_add, asset_a_id));
    CHECK(_asset_id_vector_contains(deps_a_after_add, asset_b_id));
    CHECK(_asset_id_vector_contains(deps_a_after_add, asset_c_id));

    CHECK(manager->remove_dependency(asset_a_id, asset_b_id) == OK);
    CHECK(manager->is_dependency_cache_dirty_for_tests());
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 0);

    const Vector<AssetID> deps_a_after_remove = manager->get_dependencies(asset_a_id, true);
    CHECK(!manager->is_dependency_cache_dirty_for_tests());
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 1);
    CHECK(deps_a_after_remove.size() == 2);
    CHECK(_asset_id_vector_contains(deps_a_after_remove, asset_a_id));
    CHECK(_asset_id_vector_contains(deps_a_after_remove, asset_c_id));
    CHECK(!_asset_id_vector_contains(deps_a_after_remove, asset_b_id));

    CHECK(manager->rebuild_dependency_cache() == OK);
    CHECK(!manager->is_dependency_cache_dirty_for_tests());
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 3);

    const Vector<AssetID> deps_c_cached = manager->get_dependencies(asset_c_id, true);
    CHECK(manager->get_dependency_cache_entry_count_for_tests() == 3);
    CHECK(deps_c_cached.size() == 1);
    CHECK(_asset_id_vector_contains(deps_c_cached, asset_c_id));

    _remove_user_file(asset_a_path);
    _remove_user_file(asset_b_path);
    _remove_user_file(asset_c_path);
}

} // namespace TestGaussianSplatting

#endif // TESTS_ENABLED
