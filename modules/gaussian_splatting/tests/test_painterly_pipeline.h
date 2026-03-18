/**************************************************************************/
/*  test_painterly_pipeline.h                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "tests/test_macros.h"
#include "test_macros.h"

#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/math/basis.h"
#include "core/math/math_funcs.h"
#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/os/os.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"
#include "core/math/random_number_generator.h"
#include "core/io/image.h"

#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"

#include "../core/gaussian_data.h"

#include <algorithm>
#include <vector>

#ifdef TESTS_ENABLED

namespace TestGaussianSplatting {

struct PainterlyShaderPermutation {
    String name;
    LocalVector<String> defines;
};

struct PainterlyCameraKey {
    double time = 0.0;
    Vector3 position = Vector3(0, 0, 6);
    Vector3 target = Vector3(0, 0, 0);
};

struct PainterlySceneDefinition {
    String name;
    String description;
    String density = "dense";
    int splat_count = 0;
    float radius = 0.25f;
    float fov_degrees = 60.0f;
    Vector<Color> palette;
    LocalVector<PainterlyCameraKey> camera_path;
    LocalVector<PainterlyShaderPermutation> permutations;

    bool is_dense() const { return density == "dense"; }
};

struct HeadlessRenderResult {
    Ref<Image> image;
    Vector<Color> pixels;
    float average_luminance = 0.0f;
    float coverage = 0.0f;
};

static Vector3 _parse_vector3(const Variant &value, const Vector3 &default_val = Vector3()) {
    if (value.get_type() == Variant::VECTOR3) {
        return value;
    }
    if (value.get_type() == Variant::ARRAY) {
        Array arr = value;
        if (arr.size() >= 3) {
            return Vector3(arr[0], arr[1], arr[2]);
        }
    }
    if (value.get_type() == Variant::PACKED_FLOAT32_ARRAY) {
        PackedFloat32Array arr = value;
        if (arr.size() >= 3) {
            return Vector3(arr[0], arr[1], arr[2]);
        }
    }
    return default_val;
}

static Dictionary _load_scene_json(const String &path) {
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    if (!file.is_valid()) {
        ERR_PRINT(vformat("Failed to open painterly scene definition: %s", path));
        return Dictionary();
    }

    String json_text = file->get_as_text();
    file.unref();

    Variant parsed = JSON::parse_string(json_text);
    if (parsed.get_type() != Variant::DICTIONARY) {
        ERR_PRINT(vformat("Painterly scene definition is not a dictionary: %s", path));
        return Dictionary();
    }

    return parsed;
}

static PainterlySceneDefinition _parse_scene_definition(const Dictionary &dict, const String &path) {
    PainterlySceneDefinition scene;

    scene.name = dict.get("name", path.get_file().get_basename());
    scene.description = dict.get("description", "");
    scene.density = dict.get("density", "dense");
    scene.splat_count = dict.get("splat_count", 1024);
    scene.radius = dict.get("radius", 0.25f);
    scene.fov_degrees = dict.get("fov_degrees", 60.0f);

    if (scene.splat_count <= 0) {
        scene.splat_count = 1;
    }
    if (scene.palette.is_empty()) {
        scene.palette.push_back(Color(0.8f, 0.7f, 0.6f, 1.0f));
        scene.palette.push_back(Color(0.3f, 0.35f, 0.4f, 1.0f));
        scene.palette.push_back(Color(0.9f, 0.5f, 0.35f, 1.0f));
    }

    if (dict.has("palette")) {
        Array palette_array = dict["palette"];
        scene.palette.clear();
        for (int i = 0; i < palette_array.size(); i++) {
            Variant entry = palette_array[i];
            Color color(0.75f, 0.75f, 0.75f, 0.85f);
            if (entry.get_type() == Variant::COLOR) {
                color = entry;
            } else if (entry.get_type() == Variant::ARRAY) {
                Array arr = entry;
                if (arr.size() >= 3) {
                    color = Color(arr[0], arr[1], arr[2], arr.size() > 3 ? (double)arr[3] : 0.85);
                }
            }
            scene.palette.push_back(color);
        }
        if (scene.palette.is_empty()) {
            scene.palette.push_back(Color(0.8f, 0.7f, 0.6f, 1.0f));
            scene.palette.push_back(Color(0.3f, 0.35f, 0.4f, 1.0f));
            scene.palette.push_back(Color(0.9f, 0.5f, 0.35f, 1.0f));
        }
    }

    if (dict.has("camera_path")) {
        Array camera_array = dict["camera_path"];
        for (int i = 0; i < camera_array.size(); i++) {
            Dictionary key_dict = camera_array[i];
            PainterlyCameraKey key;
            key.time = key_dict.get("time", (double)i);
            key.position = _parse_vector3(key_dict.get("position", Vector3(0, 0, 6)), Vector3(0, 0, 6));
            key.target = _parse_vector3(key_dict.get("target", Vector3(0, 0, 0)), Vector3());
            scene.camera_path.push_back(key);
        }
    }

    if (scene.camera_path.is_empty()) {
        PainterlyCameraKey key;
        key.position = Vector3(0, 0, 6);
        key.target = Vector3();
        scene.camera_path.push_back(key);
    }

    if (dict.has("shader_permutations")) {
        Array perm_array = dict["shader_permutations"];
        for (int i = 0; i < perm_array.size(); i++) {
            Dictionary perm_dict = perm_array[i];
            PainterlyShaderPermutation perm;
            perm.name = perm_dict.get("name", vformat("%s_perm_%d", scene.name, i));
            if (perm_dict.has("defines")) {
                Array defines_array = perm_dict["defines"];
                for (int j = 0; j < defines_array.size(); j++) {
                    perm.defines.push_back(String(defines_array[j]));
                }
            }
            scene.permutations.push_back(perm);
        }
    }

    if (scene.permutations.is_empty()) {
        PainterlyShaderPermutation fallback;
        fallback.name = scene.name + "_default";
        fallback.defines.push_back("PAINTERLY_STYLE_BRUSH");
        scene.permutations.push_back(fallback);
    }

    return scene;
}

static PainterlySceneDefinition load_scene_definition(const String &path) {
    Dictionary dict = _load_scene_json(path);
    return _parse_scene_definition(dict, path);
}

static LocalVector<Gaussian> generate_gaussians(const PainterlySceneDefinition &scene) {
    LocalVector<Gaussian> splats;
    splats.resize(scene.splat_count);

    RandomNumberGenerator rng;
    rng.set_seed(1337 + scene.name.hash());

    for (int i = 0; i < scene.splat_count; i++) {
        Gaussian &g = splats[i];

        Vector3 base_position;
        if (scene.is_dense()) {
            float angle = rng.randf_range(0.0f, static_cast<float>(Math::TAU));
            float ring = rng.randf_range(0.0f, 2.5f);
            base_position = Vector3(Math::cos(angle) * ring, rng.randf_range(-1.5f, 1.5f), Math::sin(angle) * ring);
        } else {
            base_position = Vector3(
                rng.randf_range(-3.5f, 3.5f),
                rng.randf_range(-2.5f, 2.5f),
                rng.randf_range(-1.5f, 1.5f)
            );
        }
        base_position.z += rng.randf_range(-0.8f, 0.8f);
        g.position = base_position;

        float scale_base = scene.radius;
        float scale_variation = scene.is_dense() ? rng.randf_range(scale_base * 0.4f, scale_base) : rng.randf_range(scale_base * 0.6f, scale_base * 1.4f);
        g.scale = Vector3(scale_variation, scale_variation, scale_variation * 0.75f);
        g.rotation = Quaternion();
        g.opacity = scene.is_dense() ? rng.randf_range(0.65f, 0.95f) : rng.randf_range(0.35f, 0.75f);

        const Color palette_color = scene.palette.is_empty() ? Color(0.8f, 0.8f, 0.8f, 1.0f) : scene.palette[i % scene.palette.size()];
        g.sh_dc = Color(palette_color.r, palette_color.g, palette_color.b, g.opacity);
        g.normal = Vector3(0, 0, -1);
        g.area = scale_variation * scale_variation * static_cast<float>(Math::PI);
    }

    return splats;
}

static HeadlessRenderResult render_scene_headless(const PainterlySceneDefinition &scene, const LocalVector<Gaussian> &splats, int camera_index = 0) {
    HeadlessRenderResult result;

    const int width = 128;
    const int height = 128;
    result.pixels.resize(width * height);
    {
        Color *write = result.pixels.ptrw();
        for (int i = 0; i < width * height; i++) {
            write[i] = Color(0, 0, 0, 0);
        }
    }

    const int camera_idx = CLAMP(camera_index, 0, static_cast<int>(scene.camera_path.size()) - 1);
    const PainterlyCameraKey &camera_key = scene.camera_path[camera_idx];

    Basis camera_basis = Basis().looking_at((camera_key.target - camera_key.position).normalized(), Vector3(0, 1, 0));
    Transform3D camera_transform(camera_basis, camera_key.position);
    Transform3D view = camera_transform.affine_inverse();

    float fov_radians = Math::deg_to_rad(scene.fov_degrees);
    if (!Math::is_finite(fov_radians) || fov_radians <= 0.0f) {
        fov_radians = static_cast<float>(Math::PI) / 3.0f;
    }
    float focal = 0.5f * width / Math::tan(fov_radians * 0.5f);

    std::vector<int> order(splats.size());
    for (size_t i = 0; i < order.size(); i++) {
        order[i] = static_cast<int>(i);
    }

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        Vector3 ca = view.xform(splats[a].position);
        Vector3 cb = view.xform(splats[b].position);
        float depth_a = -ca.z;
        float depth_b = -cb.z;
        return depth_a > depth_b;
    });

    Color *pixels = result.pixels.ptrw();

    for (int idx : order) {
        const Gaussian &g = splats[idx];
        Vector3 camera_space = view.xform(g.position);
        float depth = -camera_space.z;
        if (depth <= 0.01f) {
            continue;
        }

        float ndc_x = camera_space.x / depth;
        float ndc_y = camera_space.y / depth;
        float screen_x = width * 0.5f + ndc_x * focal;
        float screen_y = height * 0.5f - ndc_y * focal;

        float radius = MAX(g.scale.x, MAX(g.scale.y, g.scale.z));
        float pixel_radius = MAX(1.0f, radius * focal / depth);

        int min_x = MAX(0, static_cast<int>(Math::floor(screen_x - pixel_radius * 2.0f)));
        int max_x = MIN(width - 1, static_cast<int>(Math::ceil(screen_x + pixel_radius * 2.0f)));
        int min_y = MAX(0, static_cast<int>(Math::floor(screen_y - pixel_radius * 2.0f)));
        int max_y = MIN(height - 1, static_cast<int>(Math::ceil(screen_y + pixel_radius * 2.0f)));

        for (int y = min_y; y <= max_y; y++) {
            float dy = (static_cast<float>(y) + 0.5f - screen_y) / pixel_radius;
            float dy2 = dy * dy;
            for (int x = min_x; x <= max_x; x++) {
                float dx = (static_cast<float>(x) + 0.5f - screen_x) / pixel_radius;
                float dist_sq = dx * dx + dy2;
                if (dist_sq > 4.0f) {
                    continue;
                }
                float weight = Math::exp(-dist_sq * 1.5f);
                if (weight < 0.01f) {
                    continue;
                }

                float alpha = g.opacity * weight;
                Color src = Color(g.sh_dc.r, g.sh_dc.g, g.sh_dc.b, alpha);

                Color &dst = pixels[y * width + x];
                Color composed;
                composed.a = src.a + dst.a * (1.0f - src.a);
                if (composed.a <= 1e-5f) {
                    continue;
                }
                composed.r = (src.r * src.a + dst.r * dst.a * (1.0f - src.a)) / composed.a;
                composed.g = (src.g * src.a + dst.g * dst.a * (1.0f - src.a)) / composed.a;
                composed.b = (src.b * src.a + dst.b * dst.a * (1.0f - src.a)) / composed.a;
                dst = composed;
            }
        }
    }

    const Color *read = result.pixels.ptr();
    float luminance_sum = 0.0f;
    int covered_pixels = 0;
    const int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        luminance_sum += read[i].get_luminance();
        if (read[i].a > 0.01f) {
            covered_pixels++;
        }
    }

    result.coverage = static_cast<float>(covered_pixels) / static_cast<float>(total_pixels);
    result.average_luminance = luminance_sum / static_cast<float>(total_pixels);

    PackedByteArray bytes;
    bytes.resize(total_pixels * 4);
    uint8_t *byte_write = bytes.ptrw();
    for (int i = 0; i < total_pixels; i++) {
        Color c = read[i].clamp();
        byte_write[i * 4 + 0] = static_cast<uint8_t>(CLAMP(Math::round(c.r * 255.0f), 0, 255));
        byte_write[i * 4 + 1] = static_cast<uint8_t>(CLAMP(Math::round(c.g * 255.0f), 0, 255));
        byte_write[i * 4 + 2] = static_cast<uint8_t>(CLAMP(Math::round(c.b * 255.0f), 0, 255));
        byte_write[i * 4 + 3] = static_cast<uint8_t>(CLAMP(Math::round(c.a * 255.0f), 0, 255));
    }

    result.image.instantiate();
    result.image->set_data(width, height, false, Image::FORMAT_RGBA8, bytes);

    return result;
}

static float compute_image_difference(const HeadlessRenderResult &a, const HeadlessRenderResult &b) {
    ERR_FAIL_COND_V_MSG(a.pixels.size() != b.pixels.size(), 0.0f, "Image buffers must be the same size for comparison");
    const Color *aptr = a.pixels.ptr();
    const Color *bptr = b.pixels.ptr();
    float diff = 0.0f;
    const int total_pixels = a.pixels.size();
    for (int i = 0; i < total_pixels; i++) {
        diff += Math::abs(aptr[i].r - bptr[i].r);
        diff += Math::abs(aptr[i].g - bptr[i].g);
        diff += Math::abs(aptr[i].b - bptr[i].b);
    }
    return diff / (static_cast<float>(total_pixels) * 3.0f);
}

static String load_painterly_shader_source() {
    static String cached_source;
    if (!cached_source.is_empty()) {
        return cached_source;
    }

    const String shader_path = "res://modules/gaussian_splatting/shaders/painterly_resolve.glsl";
    Ref<FileAccess> file = FileAccess::open(shader_path, FileAccess::READ);
    if (!file.is_valid()) {
        ERR_PRINT(vformat("Unable to open painterly shader source: %s", shader_path));
        return String();
    }
    cached_source = file->get_as_text();
    return cached_source;
}

static bool compile_shader_permutation(RenderingDevice *rd, const PainterlyShaderPermutation &perm, String &error_out) {
    ERR_FAIL_COND_V(rd == nullptr, false);

    String shader_source = load_painterly_shader_source();
    if (shader_source.is_empty()) {
        error_out = "Painterly shader source is empty";
        return false;
    }

    String defines_block;
    for (uint32_t i = 0; i < perm.defines.size(); i++) {
        String define = perm.defines[i].strip_edges();
        if (define.is_empty()) {
            continue;
        }
        if (define.begins_with("#")) {
            defines_block += define + "\n";
        } else {
            defines_block += "#define " + define + "\n";
        }
    }

    int version_pos = shader_source.find("#version");
    String final_source = shader_source;
    if (version_pos >= 0) {
        int insert_pos = shader_source.find("\n", version_pos);
        if (insert_pos >= 0) {
            final_source = shader_source.substr(0, insert_pos + 1) + defines_block + shader_source.substr(insert_pos + 1);
        } else {
            final_source = shader_source + "\n" + defines_block;
        }
    } else {
        final_source = defines_block + shader_source;
    }

    Vector<uint8_t> spirv = rd->shader_compile_spirv_from_source(
        RD::SHADER_STAGE_COMPUTE,
        final_source,
        RD::SHADER_LANGUAGE_GLSL,
        &error_out
    );

    if (spirv.is_empty()) {
        return false;
    }

    Vector<RenderingDevice::ShaderStageSPIRVData> stages;
    RenderingDevice::ShaderStageSPIRVData stage_data;
    stage_data.shader_stage = RD::SHADER_STAGE_COMPUTE;
    stage_data.spirv = spirv;
    stages.push_back(stage_data);

    RID shader_rid = rd->shader_create_from_spirv(stages);
    if (!shader_rid.is_valid()) {
        error_out = "Failed to create shader RID";
        return false;
    }

    rd->free(shader_rid);
    return true;
}

static RenderingDevice *obtain_rendering_device() {
    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        RenderingServer *rs = RenderingServer::get_singleton();
        if (rs) {
            rd = rs->create_local_rendering_device();
        }
    }
    return rd;
}

TEST_SUITE("[GaussianSplatting][Painterly]") {
    TEST_CASE("[GaussianSplatting][Painterly] Shader permutations compile and headless render succeeds") {
        RenderingDevice *rd = obtain_rendering_device();
        if (!rd) {
            WARN_PRINT("RenderingDevice unavailable - skipping painterly shader compilation test");
            CHECK_MESSAGE(true, "Skipped painterly shader compilation due to missing RenderingDevice");
            return;
        }

        const String base_path = "res://modules/gaussian_splatting/tests/painterly_scenes/";
        Vector<String> scene_paths;
        scene_paths.push_back(base_path + "dense_atelier.json");
        scene_paths.push_back(base_path + "sparse_gallery.json");
        scene_paths.push_back(base_path + "animated_orbit.json");

        for (int i = 0; i < scene_paths.size(); i++) {
            const String &scene_path = scene_paths[i];
            INFO(("Painterly scene: " + scene_path).utf8().get_data());

            PainterlySceneDefinition scene = load_scene_definition(scene_path);
            LocalVector<Gaussian> splats = generate_gaussians(scene);
            CHECK(splats.size() == static_cast<uint32_t>(scene.splat_count));

            for (uint32_t perm_idx = 0; perm_idx < scene.permutations.size(); perm_idx++) {
                const PainterlyShaderPermutation &perm = scene.permutations[perm_idx];
                String error;
                CAPTURE(perm.name);
                bool compiled = compile_shader_permutation(rd, perm, error);
                CHECK_MESSAGE(compiled, (String("Failed to compile permutation ") + perm.name + ": " + error).utf8().get_data());
            }

            HeadlessRenderResult render = render_scene_headless(scene, splats, 0);
            CHECK(render.image.is_valid());
            CHECK(render.coverage > (scene.is_dense() ? 0.18f : 0.05f));
            CHECK(render.average_luminance > 0.01f);
        }
    }

    TEST_CASE("[GaussianSplatting][Painterly] Animated camera path produces varying frames") {
        RenderingDevice *rd = obtain_rendering_device();
        if (!rd) {
            WARN_PRINT("RenderingDevice unavailable - skipping painterly animation validation");
            CHECK_MESSAGE(true, "Skipped painterly animation validation due to missing RenderingDevice");
            return;
        }

        const String scene_path = "res://modules/gaussian_splatting/tests/painterly_scenes/animated_orbit.json";
        PainterlySceneDefinition scene = load_scene_definition(scene_path);
        LocalVector<Gaussian> splats = generate_gaussians(scene);

        HeadlessRenderResult start_frame = render_scene_headless(scene, splats, 0);
        HeadlessRenderResult end_frame = render_scene_headless(scene, splats, scene.camera_path.size() - 1);

        CHECK(start_frame.image.is_valid());
        CHECK(end_frame.image.is_valid());
        CHECK(start_frame.coverage > 0.15f);
        CHECK(end_frame.coverage > 0.15f);

        float delta = compute_image_difference(start_frame, end_frame);
        CHECK_MESSAGE(delta > 0.015f, "Animated orbit should produce visually distinct frames");
    }
}

} // namespace TestGaussianSplatting

#endif // TESTS_ENABLED
