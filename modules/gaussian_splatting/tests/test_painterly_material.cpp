/**************************************************************************/
/*  test_painterly_material.cpp                                          */
/*  Gaussian Splatting Painterly Shader Permutation Tests                 */
/**************************************************************************/

#include "test_macros.h"

#include "../painterly/painterly_material.h"

#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/object/ref_counted.h"
#include "core/string/print_string.h"
#include "scene/resources/image_texture.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_device_binds.h"
#include "servers/rendering_server.h"

#ifdef TESTS_ENABLED

namespace TestGaussianSplatting {
namespace {

static bool try_open_text(const String &p_path, String &r_resolved_path, String &r_text) {
    Vector<String> attempts;
    attempts.push_back(p_path);
    if (!p_path.begins_with("res://")) {
        attempts.push_back(String("res://") + p_path);
    } else {
        attempts.push_back(p_path.substr(6));
    }

    for (int i = 0; i < attempts.size(); i++) {
        const String &candidate = attempts[i];
        Ref<FileAccess> file = FileAccess::open(candidate, FileAccess::READ);
        if (file.is_valid()) {
            r_resolved_path = candidate;
            r_text = file->get_as_text();
            return true;
        }
    }

    return false;
}

static String load_shader_recursive(const String &p_path, Vector<String> &r_visited_paths) {
    String resolved_path;
    String source;
    if (!try_open_text(p_path, resolved_path, source)) {
        return String();
    }

    for (int i = 0; i < r_visited_paths.size(); i++) {
        if (r_visited_paths[i] == resolved_path) {
            return String();
        }
    }
    r_visited_paths.push_back(resolved_path);

    String base_dir = resolved_path.get_base_dir();
    PackedStringArray lines = source.split("\n", false);
    String result;

    for (int i = 0; i < lines.size(); i++) {
        const String &line = lines[i];
        String trimmed = line.strip_edges();
        if (trimmed.begins_with("#include")) {
            int first_quote = line.find(String("\""));
            int second_quote = line.find(String("\""), first_quote + 1);
            if (first_quote != -1 && second_quote != -1 && second_quote > first_quote) {
                String include_path = line.substr(first_quote + 1, second_quote - first_quote - 1);
                String absolute_path = base_dir.path_join(include_path);
                result += load_shader_recursive(absolute_path, r_visited_paths);
                if (!result.ends_with("\n")) {
                    result += "\n";
                }
                continue;
            }
        }

        result += line;
        if (i + 1 < lines.size()) {
            result += "\n";
        }
    }

    return result;
}

static String load_shader_source(const String &p_filename) {
    Vector<String> visited;
    String res_path = String("res://modules/gaussian_splatting/shaders/") + p_filename;
    String source = load_shader_recursive(res_path, visited);
    if (!source.is_empty()) {
        return source;
    }

    String local_path = String("modules/gaussian_splatting/shaders/") + p_filename;
    return load_shader_recursive(local_path, visited);
}

static String inject_defines(const String &p_source, const PackedStringArray &p_defines) {
    if (p_defines.is_empty()) {
        return p_source;
    }

    String defines_block;
    for (int i = 0; i < p_defines.size(); i++) {
        defines_block += String("#define ") + p_defines[i] + "\n";
    }

    int insert_pos = 0;
    int search_pos = 0;
    while (search_pos < p_source.length()) {
        int line_end = p_source.find(String("\n"), search_pos);
        if (line_end == -1) {
            line_end = p_source.length();
        }
        String line = p_source.substr(search_pos, line_end - search_pos);
        String trimmed = line.strip_edges();
        if (trimmed.begins_with("#version") || trimmed.begins_with("#extension")) {
            insert_pos = line_end + 1;
            search_pos = line_end + 1;
            continue;
        }
        break;
    }

    return p_source.substr(0, insert_pos) + defines_block + p_source.substr(insert_pos, p_source.length() - insert_pos);
}

struct PainterlyPermutation {
    const char *name;
    bool palette = false;
    bool brush = false;
    bool lighting = false;
};

static bool defines_contains(const PackedStringArray &p_defines, const String &p_define) {
    for (int i = 0; i < p_defines.size(); i++) {
        if (p_defines[i] == p_define) {
            return true;
        }
    }
    return false;
}

static bool compile_shader_permutation(RenderingDevice *p_rd, const String &p_vertex_code, const String &p_fragment_code,
        const String &p_name, String &r_error) {
    r_error = String();

    String vertex_error;
    Vector<uint8_t> vertex_spirv = p_rd->shader_compile_spirv_from_source(
            RD::SHADER_STAGE_VERTEX, p_vertex_code, RD::SHADER_LANGUAGE_GLSL, &vertex_error);
    if (vertex_spirv.is_empty()) {
        r_error = String("Vertex compile failed: ") + vertex_error;
        return false;
    }

    String fragment_error;
    Vector<uint8_t> fragment_spirv = p_rd->shader_compile_spirv_from_source(
            RD::SHADER_STAGE_FRAGMENT, p_fragment_code, RD::SHADER_LANGUAGE_GLSL, &fragment_error);
    if (fragment_spirv.is_empty()) {
        r_error = String("Fragment compile failed: ") + fragment_error;
        return false;
    }

    Vector<RenderingDevice::ShaderStageSPIRVData> stages;
    RenderingDevice::ShaderStageSPIRVData vertex_stage;
    vertex_stage.shader_stage = RD::SHADER_STAGE_VERTEX;
    vertex_stage.spirv = vertex_spirv;
    stages.push_back(vertex_stage);

    RenderingDevice::ShaderStageSPIRVData fragment_stage;
    fragment_stage.shader_stage = RD::SHADER_STAGE_FRAGMENT;
    fragment_stage.spirv = fragment_spirv;
    stages.push_back(fragment_stage);

    RID shader = p_rd->shader_create_from_spirv(stages, p_name);
    if (!shader.is_valid()) {
        r_error = "shader_create_from_spirv returned invalid RID";
        return false;
    }

    p_rd->free(shader);
    return true;
}

class PainterlyChangedCounter : public RefCounted {
    GDCLASS(PainterlyChangedCounter, RefCounted);

    int changed_count = 0;

public:
    void on_material_changed() {
        changed_count++;
    }

    int get_changed_count() const {
        return changed_count;
    }

    static void _bind_methods() {}
};

} // namespace

TEST_CASE("[GaussianSplatting] Painterly shader permutations compile") {
    REQUIRE_GPU_DEVICE();

    String vertex_source = load_shader_source("gaussian_splat.vert.glsl");
    String fragment_source = load_shader_source("gaussian_splat.frag.glsl");
    CHECK_MESSAGE(!vertex_source.is_empty(), "Failed to load gaussian_splat.vert.glsl");
    if (vertex_source.is_empty()) {
        return;
    }
    CHECK_MESSAGE(!fragment_source.is_empty(), "Failed to load gaussian_splat.frag.glsl");
    if (fragment_source.is_empty()) {
        return;
    }

    Ref<PainterlyMaterial> material;
    material.instantiate();

    const PainterlyPermutation permutations[] = {
        {"baseline", false, false, false},
        {"palette", true, false, false},
        {"brush", false, true, false},
        {"lighting", false, false, true},
        {"full", true, true, true},
    };

    for (const PainterlyPermutation &permutation : permutations) {
        material->set_palette_quantization_enabled(permutation.palette);
        material->set_brush_modulation_enabled(permutation.brush);
        material->set_lighting_stylization_enabled(permutation.lighting);

        PackedStringArray defines = material->get_shader_define_strings();

        String vertex_code = inject_defines(vertex_source, defines);
        String fragment_code = inject_defines(fragment_source, defines);

        CHECK_MESSAGE(!vertex_code.is_empty(), (String("Vertex code empty for permutation '") + permutation.name + "'").utf8().get_data());
        CHECK_MESSAGE(!fragment_code.is_empty(), (String("Fragment code empty for permutation '") + permutation.name + "'").utf8().get_data());

        const bool has_palette = defines_contains(defines, "PAINTERLY_ENABLE_PALETTE");
        const bool has_brush = defines_contains(defines, "PAINTERLY_ENABLE_BRUSH");
        const bool has_lighting = defines_contains(defines, "PAINTERLY_ENABLE_LIGHTING");
        CHECK_MESSAGE(has_palette == permutation.palette,
                (String("PAINTERLY_ENABLE_PALETTE mismatch in permutation '") + permutation.name + "'").utf8().get_data());
        CHECK_MESSAGE(has_brush == permutation.brush,
                (String("PAINTERLY_ENABLE_BRUSH mismatch in permutation '") + permutation.name + "'").utf8().get_data());
        CHECK_MESSAGE(has_lighting == permutation.lighting,
                (String("PAINTERLY_ENABLE_LIGHTING mismatch in permutation '") + permutation.name + "'").utf8().get_data());

        String compile_error;
        const bool compiled = compile_shader_permutation(rd, vertex_code, fragment_code,
                String("PainterlyPermutation_") + permutation.name, compile_error);
        CHECK_MESSAGE(compiled,
                (String("Shader permutation compile failed for '") + permutation.name + "': " + compile_error).utf8().get_data());
    }
}

TEST_CASE("[GaussianSplatting] Painterly material validation contract") {
    Ref<PainterlyMaterial> material;
    material.instantiate();

    CHECK(!material->has_required_resources());

    Vector<String> initial_missing = material->get_missing_resources();
    CHECK(initial_missing.find("palette_textures") != -1);
    CHECK(initial_missing.find("noise_luts") != -1);
    CHECK(initial_missing.find("stroke_density_curve") != -1);

    Ref<Image> image = Image::create_empty(1, 1, false, Image::FORMAT_RGBA8);
    REQUIRE(image.is_valid());
    image->fill(Color(1.0f, 1.0f, 1.0f, 1.0f));
    Ref<ImageTexture> texture = ImageTexture::create_from_image(image);
    REQUIRE(texture.is_valid());

    TypedArray<Texture2D> palettes;
    palettes.push_back(texture);
    material->set_palette_textures(palettes);

    TypedArray<Texture2D> noise_luts;
    noise_luts.push_back(texture);
    material->set_noise_luts(noise_luts);

    Ref<Curve> curve;
    curve.instantiate();
    material->set_stroke_density_curve(curve);

    CHECK(material->has_required_resources());
    CHECK(material->get_missing_resources().is_empty());

    material->set_palette_quantization_enabled(true);
    material->set_brush_modulation_enabled(true);
    material->set_lighting_stylization_enabled(true);

    material->set_cel_band_count(999);
    CHECK(material->get_cel_band_count() == 16);
    material->set_cel_band_count(-5);
    CHECK(material->get_cel_band_count() == 1);

    material->set_temporal_stability(-1.0f);
    CHECK(material->get_temporal_stability() == 0.0f);
    material->set_temporal_stability(2.0f);
    CHECK(material->get_temporal_stability() == 1.0f);

    Dictionary serialized = material->serialize();
    Ref<PainterlyMaterial> restored;
    restored.instantiate();
    restored->deserialize(serialized);

    CHECK(restored->has_required_resources());
    CHECK(restored->is_palette_quantization_enabled());
    CHECK(restored->is_brush_modulation_enabled());
    CHECK(restored->is_lighting_stylization_enabled());
    CHECK(restored->get_shader_define_strings() == material->get_shader_define_strings());
    CHECK(restored->get_cel_band_count() == material->get_cel_band_count());
    CHECK(restored->get_temporal_stability() == material->get_temporal_stability());
}

TEST_CASE("[GaussianSplatting] Painterly deserialize emits a single changed signal") {
    Ref<PainterlyMaterial> source;
    source.instantiate();

    Ref<Image> image = Image::create_empty(1, 1, false, Image::FORMAT_RGBA8);
    REQUIRE(image.is_valid());
    image->fill(Color(1.0f, 1.0f, 1.0f, 1.0f));
    Ref<ImageTexture> texture = ImageTexture::create_from_image(image);
    REQUIRE(texture.is_valid());

    TypedArray<Texture2D> palettes;
    palettes.push_back(texture);
    source->set_palette_textures(palettes);

    TypedArray<Texture2D> noise_luts;
    noise_luts.push_back(texture);
    source->set_noise_luts(noise_luts);

    Ref<Curve> curve;
    curve.instantiate();
    source->set_stroke_density_curve(curve);
    source->set_palette_quantization_enabled(true);
    source->set_brush_modulation_enabled(true);
    source->set_lighting_stylization_enabled(true);
    source->set_stroke_density_strength(0.9f);
    source->set_stroke_density_resolution(128);
    source->set_temporal_stability(0.75f);

    const Dictionary serialized = source->serialize();

    Ref<PainterlyMaterial> restored;
    restored.instantiate();
    Ref<PainterlyChangedCounter> counter;
    counter.instantiate();

    const Error connect_err = restored->connect("changed", callable_mp(counter.ptr(), &PainterlyChangedCounter::on_material_changed));
    CHECK(connect_err == OK);
    if (connect_err != OK) {
        return;
    }

    restored->deserialize(serialized);
    CHECK_MESSAGE(counter->get_changed_count() == 1,
            "deserialize should coalesce setter updates and emit exactly one changed signal");
}

} // namespace TestGaussianSplatting

#endif // TESTS_ENABLED
