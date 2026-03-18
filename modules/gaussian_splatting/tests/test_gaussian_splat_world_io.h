#pragma once

#include "test_macros.h"

#include "../core/gaussian_splat_world.h"
#include "../io/gaussian_splat_world_io.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/os.h"


namespace {

Gaussian make_gaussian(const Vector3 &p_position, const Color &p_dc) {
    Gaussian g;
    g.position = p_position;
    g.scale = Vector3(1.0f, 1.0f, 1.0f);
    g.rotation = Quaternion();
    g.opacity = 1.0f;
    g.sh_dc = p_dc;
    g.sh_1[0] = Vector3(0.1f, 0.1f, 0.1f);
    g.sh_1[1] = Vector3();
    g.sh_1[2] = Vector3();
    g.normal = Vector3(0.0f, 1.0f, 0.0f);
    g.area = 1.0f;
    g.brush_axes = Vector2(1.0f, 1.0f);
    g.stroke_age = 0.0f;
    g.painterly_meta = gaussian_pack_painterly_meta(17, 300);
    return g;
}

Vector<Gaussian> build_gaussians() {
    Vector<Gaussian> gaussians;
    gaussians.resize(4);
    gaussians.write[0] = make_gaussian(Vector3(0.0f, 0.0f, 0.0f), Color(1.0f, 0.0f, 0.0f, 1.0f));
    gaussians.write[1] = make_gaussian(Vector3(1.0f, 0.0f, 0.0f), Color(0.0f, 1.0f, 0.0f, 1.0f));
    gaussians.write[2] = make_gaussian(Vector3(0.0f, 1.0f, 0.0f), Color(0.0f, 0.0f, 1.0f, 1.0f));
    gaussians.write[3] = make_gaussian(Vector3(1.0f, 1.0f, 0.0f), Color(1.0f, 1.0f, 1.0f, 1.0f));
    return gaussians;
}

Vector<GaussianSplatRenderer::StaticChunk> build_chunks() {
    Vector<GaussianSplatRenderer::StaticChunk> chunks;
    chunks.resize(2);

    GaussianSplatRenderer::StaticChunk first;
    first.bounds = AABB(Vector3(-0.5f, -0.5f, -0.5f), Vector3(1.5f, 1.5f, 1.0f));
    first.center = first.bounds.get_center();
    first.radius = 1.5f;
    first.indices.resize(2);
    first.indices.write[0] = 0;
    first.indices.write[1] = 1;
    chunks.write[0] = first;

    GaussianSplatRenderer::StaticChunk second;
    second.bounds = AABB(Vector3(-0.5f, 0.5f, -0.5f), Vector3(1.5f, 1.5f, 1.0f));
    second.center = second.bounds.get_center();
    second.radius = 1.5f;
    second.indices.resize(2);
    second.indices.write[0] = 2;
    second.indices.write[1] = 3;
    chunks.write[1] = second;

    return chunks;
}

struct GsplatWorldSaverGuard {
    Ref<ResourceFormatSaverGaussianSplatWorld> saver;

    GsplatWorldSaverGuard() {
        saver.instantiate();
        ResourceSaver::add_resource_format_saver(saver, true); // true = at_front for priority
    }

    ~GsplatWorldSaverGuard() {
        if (saver.is_valid()) {
            ResourceSaver::remove_resource_format_saver(saver);
        }
	}
};

String _make_world_io_fixture_path(const String &p_prefix) {
    const uint64_t ticks = OS::get_singleton() ? OS::get_singleton()->get_ticks_usec() : 0;
    const String base_temp = OS::get_singleton() ? OS::get_singleton()->get_temp_path() : ".";
    return base_temp.path_join("godotgs_world_io_" + p_prefix + "_" + itos(ticks) + ".gsplatworld");
}

void _remove_world_io_fixture(const String &p_path) {
    DirAccess::remove_absolute(p_path);
}

} // namespace

TEST_CASE("[GaussianSplatting][WorldIO] gsplatworld direct format saver/loader") {
    // This test bypasses ResourceLoader/ResourceSaver to test our format directly
    Ref<GaussianData> gaussian_data;
    gaussian_data.instantiate();
    Vector<Gaussian> gaussians = build_gaussians();
    gaussian_data->set_gaussians(gaussians);

    // Verify set_gaussians worked
    CHECK_MESSAGE(gaussian_data->get_count() == 4, "GaussianData should have 4 splats after set_gaussians");

    Ref<GaussianSplatWorld> world;
    world.instantiate();
    world->set_gaussian_data(gaussian_data);
    world->set_bounds(gaussian_data->get_aabb());
    world->set_static_chunks(build_chunks());

    Dictionary metadata;
    metadata[StringName("lod_levels")] = 2;
    world->set_metadata(metadata);

    // Verify world has data before save
    CHECK_MESSAGE(world->get_gaussian_data()->get_count() == 4, "World should have 4 splats before save");
    CHECK_MESSAGE(world->get_static_chunks().size() == 2, "World should have 2 chunks before save");

    const String path = _make_world_io_fixture_path("direct_test");

    // Use our format saver directly
    ResourceFormatSaverGaussianSplatWorld saver;
    const Error save_err = saver.save(world, path);
    CHECK_MESSAGE(save_err == OK, "Direct saver should succeed");
    if (save_err != OK) {
        return;
    }

    // Verify file was written
    CHECK_MESSAGE(FileAccess::exists(path), "File should exist after save");
    Ref<FileAccess> f = FileAccess::open(path, FileAccess::READ);
    CHECK_MESSAGE(f.is_valid(), "Should be able to open saved file");
    if (f.is_valid()) {
        const uint64_t file_size = f->get_length();
        // Expected: 104 header + 4*144 gaussian + 2*56 chunk_table + 4*4 indices + metadata
        MESSAGE("File size: ", file_size, " bytes");
        CHECK_MESSAGE(file_size > 104, "File should be larger than just header");
        f.unref();
    }

    // Use our format loader directly
    ResourceFormatLoaderGaussianSplatWorld loader;
    Error load_err = OK;
    Ref<Resource> loaded_res = loader.load(path, "", &load_err);
    CHECK_MESSAGE(load_err == OK, "Direct loader should succeed");
    CHECK_MESSAGE(loaded_res.is_valid(), "Loaded resource should be valid");

    Ref<GaussianSplatWorld> loaded = loaded_res;
    CHECK_MESSAGE(loaded.is_valid(), "Loaded resource should be GaussianSplatWorld");
    if (!loaded.is_valid()) {
        return;
    }

    Ref<GaussianData> loaded_data = loaded->get_gaussian_data();
    CHECK_MESSAGE(loaded_data.is_valid(), "Loaded world should have GaussianData");
    if (loaded_data.is_valid()) {
        MESSAGE("Loaded gaussian count: ", loaded_data->get_count());
        CHECK_EQ(loaded_data->get_count(), 4);
        const Gaussian g0 = loaded_data->get_gaussian(0);
        CHECK_EQ(gaussian_get_palette_id(g0.painterly_meta), 17);
        CHECK_EQ(gaussian_get_brush_override_id(g0.painterly_meta), 300);
    }

    const Vector<GaussianSplatRenderer::StaticChunk> &chunks = loaded->get_static_chunks();
    MESSAGE("Loaded chunk count: ", chunks.size());
    CHECK_EQ(chunks.size(), 2);

    // Cleanup
    _remove_world_io_fixture(path);
}

TEST_CASE("[GaussianSplatting][WorldIO] gsplatworld save/load round-trip") {
    // Check what resource type the loader returns for our extension
    String detected_type = ResourceLoader::get_resource_type("test.gsplatworld");
    MESSAGE("ResourceLoader detected type for .gsplatworld: '", detected_type, "'");

    // Register saver for test scope (workaround for test init order).
    GsplatWorldSaverGuard saver_guard;
    MESSAGE("Explicitly registered gsplatworld saver for test (at_front=true)");

    Ref<GaussianData> gaussian_data;
    gaussian_data.instantiate();
    Vector<Gaussian> gaussians = build_gaussians();
    gaussian_data->set_gaussians(gaussians);

    // Verify set_gaussians worked
    CHECK_MESSAGE(gaussian_data->get_count() == 4, "GaussianData should have 4 splats after set_gaussians");
    if (gaussian_data->get_count() != 4) {
        MESSAGE("CRITICAL: set_gaussians failed! Expected 4, got ", gaussian_data->get_count());
        return;
    }

    Ref<GaussianSplatWorld> world;
    world.instantiate();
    world->set_gaussian_data(gaussian_data);
    world->set_bounds(gaussian_data->get_aabb());
    world->set_static_chunks(build_chunks());

    Dictionary metadata;
    metadata[StringName("lod_levels")] = 2;
    metadata[StringName("author")] = String("test");
    world->set_metadata(metadata);

    const String path = _make_world_io_fixture_path("roundtrip");
    const Error save_err = ResourceSaver::save(world, path);
    CHECK_MESSAGE(save_err == OK, "Saving gsplatworld should succeed.");
    if (save_err != OK) {
        return;
    }

    // Check what ResourceSaver actually wrote
    {
        Ref<FileAccess> f = FileAccess::open(path, FileAccess::READ);
        if (f.is_valid()) {
            uint64_t fsize = f->get_length();
            MESSAGE("File saved via ResourceSaver, size: ", fsize, " bytes");
            if (fsize >= 4) {
                uint32_t magic = f->get_32();
                MESSAGE("File magic: 0x", String::num_int64(magic, 16), " (expected 0x57505347 for GSPW)");
                if (magic == 0x57505347) {
                    MESSAGE("File has correct GSPW magic - our saver was used!");
                    f->seek(12);
                    uint32_t splat_count = f->get_32();
                    MESSAGE("splat_count in file: ", splat_count);
                } else {
                    MESSAGE("File has WRONG magic - ResourceSaver used different saver!");
                    // Read first 50 bytes as text to see format
                    f->seek(0);
                    PackedByteArray first_bytes = f->get_buffer(MIN(fsize, (uint64_t)50));
                    String preview = String::utf8((const char*)first_bytes.ptr(), first_bytes.size());
                    MESSAGE("File preview: '", preview.substr(0, 50), "'");
                }
            }
            f.unref();
        }
    }

    Ref<GaussianSplatWorld> loaded = ResourceLoader::load(path);
    CHECK_MESSAGE(loaded.is_valid(), "Loading gsplatworld should succeed.");
    if (!loaded.is_valid()) {
        return;
    }

    Ref<GaussianData> loaded_data = loaded->get_gaussian_data();
    CHECK(loaded_data.is_valid());
    if (!loaded_data.is_valid()) {
        return;
    }

    MESSAGE("Loaded gaussian count via ResourceLoader: ", loaded_data->get_count());
    CHECK_EQ(loaded_data->get_count(), gaussians.size());
    if (loaded_data->get_count() > 0) {
        const Gaussian g0 = loaded_data->get_gaussian(0);
        CHECK(g0.position.is_equal_approx(gaussians[0].position));
        CHECK(g0.sh_dc.is_equal_approx(gaussians[0].sh_dc));
        CHECK_EQ(gaussian_get_palette_id(g0.painterly_meta), 17);
        CHECK_EQ(gaussian_get_brush_override_id(g0.painterly_meta), 300);
    }

    const Vector<GaussianSplatRenderer::StaticChunk> &chunks = loaded->get_static_chunks();
    MESSAGE("Loaded chunk count via ResourceLoader: ", chunks.size());
    CHECK_EQ(chunks.size(), 2);
    if (chunks.size() >= 2) {
        CHECK_EQ(chunks[0].indices.size(), 2);
        CHECK_EQ(chunks[0].indices[0], 0);
        CHECK_EQ(chunks[1].indices[1], 3);
    }

    Dictionary loaded_metadata = loaded->get_metadata();
    CHECK(loaded_metadata.has(StringName("lod_levels")));
    CHECK(int(loaded_metadata[StringName("lod_levels")]) == 2);

    // Cleanup
    _remove_world_io_fixture(path);
}
