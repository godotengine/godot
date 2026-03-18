#ifndef PAINTERLY_MATERIAL_H
#define PAINTERLY_MATERIAL_H

#include "core/io/resource.h"
#include "core/math/color.h"
#include "core/math/vector3.h"
#include "core/object/callable_method_pointer.h"
#include "core/templates/vector.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"
#include "scene/resources/curve.h"
#include "scene/resources/texture.h"

class PainterlyMaterial : public Resource {
    GDCLASS(PainterlyMaterial, Resource);

public:
    enum ShadingStyle {
        SHADING_STYLE_REALISTIC = 0,
        SHADING_STYLE_CEL,
        SHADING_STYLE_PAINTERLY,
        SHADING_STYLE_GOOCH,
    };

    enum StylePreset {
        STYLE_PRESET_CUSTOM = 0,
        STYLE_PRESET_REALISTIC,
        STYLE_PRESET_TOON,
        STYLE_PRESET_PAINTERLY,
        STYLE_PRESET_TECHNICAL,
    };

private:
    Vector<Ref<Texture2D>> palette_textures;
    Vector<Ref<Texture2D>> noise_luts;
    Ref<Curve> stroke_density_curve;
    int stroke_density_resolution = 256;
    float stroke_density_strength = 1.0f;

    bool palette_quantization_enabled = false;
    bool brush_modulation_enabled = false;
    bool lighting_stylization_enabled = false;

    PackedColorArray palette_colors;
    float palette_blend_strength = 0.75f;
    float palette_noise_strength = 0.15f;
    bool palette_preserve_luminance = true;

    int shading_style = SHADING_STYLE_REALISTIC;
    int style_preset = STYLE_PRESET_CUSTOM;
    int cel_band_count = 4;
    float cel_smoothness = 0.35f;
    float painterly_mix_strength = 0.6f;
    float brush_texture_influence = 0.5f;

    float brush_scale = 1.6f;
    float brush_softness = 1.4f;
    float brush_anisotropy = 1.0f;
    float brush_rotation_jitter = 0.15f;
    float brush_shape_noise = 0.25f;

    Color light_color = Color(1.0f, 0.95f, 0.9f, 1.0f);
    Color ambient_color = Color(0.12f, 0.12f, 0.12f, 1.0f);
    Vector3 light_direction = Vector3(0.25f, 0.5f, -1.0f).normalized();
    float diffuse_strength = 0.85f;
    float specular_strength = 0.2f;
    float rim_strength = 0.25f;
    float specular_power = 24.0f;
    float rim_power = 2.5f;
    Color rim_color = Color(1.0f, 0.92f, 0.78f, 1.0f);
    Color shadow_color = Color(0.22f, 0.19f, 0.24f, 0.5f);
    Color highlight_color = Color(1.05f, 0.98f, 0.88f, 0.7f);
    float color_temperature = 6500.0f;
    float color_temperature_strength = 0.2f;
    float temporal_stability = 0.5f;
    float gooch_cool_mix = 0.45f;
    float gooch_warm_mix = 0.55f;
    float lighting_intensity = 1.0f;

    mutable PackedFloat32Array cached_stroke_density_lut;
    mutable bool stroke_density_dirty = true;
    bool defer_changed_notifications = false;
    bool changed_notification_pending = false;

    // Bitmask tracking which property groups changed during a deferred update.
    enum DirtyFlag : uint32_t {
        DIRTY_PALETTE = 1 << 0,
        DIRTY_BRUSH = 1 << 1,
        DIRTY_LIGHTING = 1 << 2,
        DIRTY_SHADING_STYLE = 1 << 3,
        DIRTY_STROKE_DENSITY = 1 << 4,
        DIRTY_FEATURE_TOGGLE = 1 << 5,
        DIRTY_OTHER = 1 << 6,
    };
    uint32_t deferred_dirty_flags = 0;

    void _disconnect_curve();
    void _connect_curve();
    void _on_stroke_density_curve_changed();
    void _mark_stroke_density_dirty();
    void _emit_changed(uint32_t p_dirty_flag = DIRTY_OTHER);
    void _begin_bulk_update();
    void _end_bulk_update();
    void _clamp_palette();
    void _apply_style_preset(StylePreset p_preset);
    static void _bind_properties();

protected:
    static void _bind_methods();

public:
    static const int MAX_PALETTE_COLORS = 8;

    PainterlyMaterial();

    void set_palette_textures(const TypedArray<Texture2D> &p_textures);
    TypedArray<Texture2D> get_palette_textures() const;
    void add_palette_texture(const Ref<Texture2D> &p_texture);
    void remove_palette_texture(int p_index);
    int get_palette_texture_count() const;

    void set_noise_luts(const TypedArray<Texture2D> &p_textures);
    TypedArray<Texture2D> get_noise_luts() const;
    void add_noise_lut(const Ref<Texture2D> &p_texture);
    void remove_noise_lut(int p_index);
    int get_noise_lut_count() const;

    void set_stroke_density_curve(const Ref<Curve> &p_curve);
    Ref<Curve> get_stroke_density_curve() const { return stroke_density_curve; }

    void set_stroke_density_resolution(int p_resolution);
    int get_stroke_density_resolution() const { return stroke_density_resolution; }

    void set_stroke_density_strength(float p_strength);
    float get_stroke_density_strength() const { return stroke_density_strength; }

    PackedFloat32Array get_stroke_density_lut() const;

    void set_palette_quantization_enabled(bool p_enabled);
    bool is_palette_quantization_enabled() const;

    void set_brush_modulation_enabled(bool p_enabled);
    bool is_brush_modulation_enabled() const;

    void set_lighting_stylization_enabled(bool p_enabled);
    bool is_lighting_stylization_enabled() const;

    void set_palette_colors(const PackedColorArray &p_colors);
    PackedColorArray get_palette_colors() const;

    void set_palette_blend_strength(float p_strength);
    float get_palette_blend_strength() const;

    void set_palette_noise_strength(float p_strength);
    float get_palette_noise_strength() const;

    void set_palette_preserve_luminance(bool p_preserve);
    bool get_palette_preserve_luminance() const;

    void set_shading_style(int p_style);
    int get_shading_style() const;

    void set_style_preset(int p_preset);
    int get_style_preset() const;

    void set_cel_band_count(int p_bands);
    int get_cel_band_count() const;

    void set_cel_smoothness(float p_smoothness);
    float get_cel_smoothness() const;

    void set_painterly_mix_strength(float p_strength);
    float get_painterly_mix_strength() const;

    void set_brush_texture_influence(float p_influence);
    float get_brush_texture_influence() const;

    void set_brush_scale(float p_scale);
    float get_brush_scale() const;

    void set_brush_softness(float p_softness);
    float get_brush_softness() const;

    void set_brush_anisotropy(float p_anisotropy);
    float get_brush_anisotropy() const;

    void set_brush_rotation_jitter(float p_jitter);
    float get_brush_rotation_jitter() const;

    void set_brush_shape_noise(float p_noise);
    float get_brush_shape_noise() const;

    void set_light_color(const Color &p_color);
    Color get_light_color() const;

    void set_ambient_color(const Color &p_color);
    Color get_ambient_color() const;

    void set_light_direction(const Vector3 &p_direction);
    Vector3 get_light_direction() const;

    void set_diffuse_strength(float p_strength);
    float get_diffuse_strength() const;

    void set_specular_strength(float p_strength);
    float get_specular_strength() const;

    void set_rim_strength(float p_strength);
    float get_rim_strength() const;

    void set_specular_power(float p_power);
    float get_specular_power() const;

    void set_rim_power(float p_power);
    float get_rim_power() const;

    void set_rim_color(const Color &p_color);
    Color get_rim_color() const;

    void set_shadow_color(const Color &p_color);
    Color get_shadow_color() const;

    void set_highlight_color(const Color &p_color);
    Color get_highlight_color() const;

    void set_color_temperature(float p_temperature);
    float get_color_temperature() const;

    void set_color_temperature_strength(float p_strength);
    float get_color_temperature_strength() const;

    void set_temporal_stability(float p_value);
    float get_temporal_stability() const;

    void set_gooch_cool_mix(float p_value);
    float get_gooch_cool_mix() const;

    void set_gooch_warm_mix(float p_value);
    float get_gooch_warm_mix() const;

    void set_lighting_intensity(float p_intensity);
    float get_lighting_intensity() const;

    PackedStringArray get_shader_define_strings() const;

    bool has_required_resources() const;
    Vector<String> get_missing_resources() const;

    Dictionary serialize() const;
    void deserialize(const Dictionary &p_data);
};

VARIANT_ENUM_CAST(PainterlyMaterial::ShadingStyle);
VARIANT_ENUM_CAST(PainterlyMaterial::StylePreset);

#endif // PAINTERLY_MATERIAL_H
