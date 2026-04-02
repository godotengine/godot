#pragma once

#include "test_macros.h"
#include "../io/ply_loader.h"
#include "../io/resource_importer_ply.h"
#include "../io/gaussian_splat_world_io.h"
#include "../core/gaussian_splat_world.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"

namespace {

// Minimal PLY file content for testing
const char *MINIMAL_PLY_CONTENT = R"(ply
format binary_little_endian 1.0
element vertex 2
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
)";

String _make_ply_fixture_path(const String &p_prefix) {
    const uint64_t ticks = OS::get_singleton() ? OS::get_singleton()->get_ticks_usec() : 0;
    const String base_temp = OS::get_singleton() ? OS::get_singleton()->get_temp_path() : ".";
    return base_temp.path_join("godotgs_ply_fixture_" + p_prefix + "_" + itos(ticks) + ".ply");
}

void _remove_ply_fixture(const String &p_path) {
    DirAccess::remove_absolute(p_path);
}

} // namespace

TEST_CASE("[GaussianSplatting][PLY] parse minimal binary PLY") {
    // Write test PLY to temp file
    const String path = _make_ply_fixture_path("minimal");

    // Create minimal PLY with header + binary data
    Ref<FileAccess> f = FileAccess::open(path, FileAccess::WRITE);
    CHECK_MESSAGE(f.is_valid(), "Should create test PLY file");
    if (!f.is_valid()) return;

    f->store_string(MINIMAL_PLY_CONTENT);

    // Write 2 vertices of binary data (14 floats each = 56 bytes per vertex)
    // Vertex 0: position (0,0,0), scale (1,1,1), rotation identity (w,x,y,z), opacity 1, dc (1,0,0)
    float v0[14] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    f->store_buffer((const uint8_t *)v0, sizeof(v0));

    // Vertex 1: position (1,0,0), scale (1,1,1), rotation identity (w,x,y,z), opacity 1, dc (0,1,0)
    float v1[14] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    f->store_buffer((const uint8_t *)v1, sizeof(v1));
    f.unref();

    // Load using PLYLoader
    PLYLoader loader;
    Error err = loader.load_file(path);

    CHECK_MESSAGE(err == OK, "PLY load should succeed");

    Ref<GaussianData> data = loader.get_gaussian_data();
    CHECK_MESSAGE(data.is_valid(), "Data should be valid");
    if (data.is_valid()) {
        CHECK_EQ(data->get_count(), 2);

        if (data->get_count() >= 2) {
            // Check first gaussian
            CHECK(data->get_gaussian(0).position.is_equal_approx(Vector3(0, 0, 0)));

            // Check second gaussian
            CHECK(data->get_gaussian(1).position.is_equal_approx(Vector3(1, 0, 0)));
        }
    }

    // Cleanup
    _remove_ply_fixture(path);
}

TEST_CASE("[GaussianSplatting][PLY] parse ASCII PLY") {
    const String path = _make_ply_fixture_path("ascii");

    const char *ascii_ply = R"(ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
0.5 0.5 0.5 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.8 0.5 0.5 0.5
)";

    Ref<FileAccess> f = FileAccess::open(path, FileAccess::WRITE);
    CHECK_MESSAGE(f.is_valid(), "Should create ASCII PLY file");
    if (!f.is_valid()) return;

    f->store_string(ascii_ply);
    f.unref();

    PLYLoader loader;
    Error err = loader.load_file(path);

    CHECK_MESSAGE(err == OK, "ASCII PLY load should succeed");

    Ref<GaussianData> data = loader.get_gaussian_data();
    CHECK_MESSAGE(data.is_valid(), "Data should be valid");
    if (data.is_valid()) {
        CHECK_EQ(data->get_count(), 1);

        if (data->get_count() >= 1) {
            CHECK(data->get_gaussian(0).position.is_equal_approx(Vector3(0.5f, 0.5f, 0.5f)));
        }
    }

    // Cleanup
    _remove_ply_fixture(path);
}

TEST_CASE("[GaussianSplatting][PLYLoader] Cache version mismatch forces re-parse") {
    // Write a minimal binary PLY fixture using the same pattern as other tests.
    const String ply_path = _make_ply_fixture_path("cache_version");

    {
        Ref<FileAccess> f = FileAccess::open(ply_path, FileAccess::WRITE);
        REQUIRE_MESSAGE(f.is_valid(), "Should create test PLY file");
        f->store_string(MINIMAL_PLY_CONTENT);
        float v0[14] = { 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0 };
        f->store_buffer((const uint8_t *)v0, sizeof(v0));
        float v1[14] = { 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0 };
        f->store_buffer((const uint8_t *)v1, sizeof(v1));
    }

    // First load: parses PLY, writes .gsplatcache.
    {
        PLYLoader loader;
        Error err = loader.load_file(ply_path);
        CHECK_MESSAGE(err == OK, "Initial PLY load should succeed");
        CHECK(loader.get_splat_count() == 2);
    }

    // Tamper with the cache version: load the .gsplatcache, change version, re-save.
    // Use the format loader/saver directly because .gsplatcache is not a globally
    // recognised extension (by design — it's internal to PLYLoader).
    const String cache_path = ply_path.get_basename() + ".gsplatcache";
    if (FileAccess::exists(cache_path)) {
        ResourceFormatLoaderGaussianSplatWorld format_loader;
        Error load_err = OK;
        Ref<Resource> resource = format_loader.load(cache_path, cache_path, &load_err);
        Ref<GaussianSplatWorld> world = resource;
        REQUIRE_MESSAGE(world.is_valid(), "Cache should be a valid GaussianSplatWorld");

        Dictionary metadata = world->get_metadata();
        metadata[StringName("cache_version")] = 9999; // Wrong version
        world->set_metadata(metadata);
        ResourceFormatSaverGaussianSplatWorld format_saver;
        format_saver.save(world, cache_path);

        // Second load: cache should be rejected because of version mismatch.
        PLYLoader loader;
        Error err = loader.load_file(ply_path);
        CHECK_MESSAGE(err == OK, "PLY load should still succeed (re-parse fallback)");
        CHECK(loader.get_splat_count() == 2);

        Dictionary stats = loader.get_load_statistics();
        if (stats.has("cache_hit")) {
            CHECK_MESSAGE(!(bool)stats["cache_hit"], "Version-mismatched cache should not be a cache hit");
        }
    } else {
        MESSAGE("Cache file not created (caching may be disabled); skipping version guard test");
    }

    // Cleanup.
    _remove_ply_fixture(ply_path);
    DirAccess::remove_absolute(cache_path);
}

TEST_CASE("[GaussianSplatting][PLYLoader] Legacy gsplatworld cache hits migrate to gsplatcache") {
    const String ply_path = _make_ply_fixture_path("legacy_cache_migration");

    {
        Ref<FileAccess> f = FileAccess::open(ply_path, FileAccess::WRITE);
        REQUIRE_MESSAGE(f.is_valid(), "Should create test PLY file");
        f->store_string(MINIMAL_PLY_CONTENT);
        float v0[14] = { 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0 };
        f->store_buffer((const uint8_t *)v0, sizeof(v0));
        float v1[14] = { 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0 };
        f->store_buffer((const uint8_t *)v1, sizeof(v1));
    }

    {
        PLYLoader loader;
        Error err = loader.load_file(ply_path);
        CHECK_MESSAGE(err == OK, "Initial PLY load should succeed");
        CHECK(loader.get_splat_count() == 2);
    }

    const String cache_path = ply_path.get_basename() + ".gsplatcache";
    const String legacy_cache_path = ply_path.get_basename() + ".gsplatworld";

    if (FileAccess::exists(cache_path)) {
        DirAccess::remove_absolute(legacy_cache_path);
        REQUIRE_MESSAGE(DirAccess::rename_absolute(cache_path, legacy_cache_path) == OK,
                "Renaming the cache to the legacy .gsplatworld path should succeed");
        CHECK_FALSE(FileAccess::exists(cache_path));
        CHECK(FileAccess::exists(legacy_cache_path));

        PLYLoader loader;
        Error err = loader.load_file(ply_path);
        CHECK_MESSAGE(err == OK, "PLY load through the legacy cache path should succeed");
        CHECK(loader.get_splat_count() == 2);

        Dictionary stats = loader.get_load_statistics();
        if (stats.has("cache_hit")) {
            CHECK_MESSAGE((bool)stats["cache_hit"], "Legacy cache fallback should still count as a cache hit");
        }

        CHECK_MESSAGE(FileAccess::exists(cache_path),
                "Legacy cache hits should rewrite the migrated .gsplatcache");
        CHECK_MESSAGE(!FileAccess::exists(legacy_cache_path),
                "Legacy cache hits should remove the migrated .gsplatworld cache");
    } else {
        MESSAGE("Cache file not created (caching may be disabled); skipping legacy cache migration test");
    }

    _remove_ply_fixture(ply_path);
    DirAccess::remove_absolute(cache_path);
    DirAccess::remove_absolute(legacy_cache_path);
}
