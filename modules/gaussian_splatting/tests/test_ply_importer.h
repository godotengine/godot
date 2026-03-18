#pragma once

#include "test_macros.h"
#include "../io/ply_loader.h"
#include "../io/resource_importer_ply.h"
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
