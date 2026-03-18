#include "painterly_material.h"

#include "core/math/math_funcs.h"
#include "core/object/class_db.h"
#include <algorithm>

PainterlyMaterial::PainterlyMaterial() {
    palette_colors.push_back(Color(0.984f, 0.964f, 0.902f));
    palette_colors.push_back(Color(0.902f, 0.545f, 0.235f));
    palette_colors.push_back(Color(0.349f, 0.392f, 0.655f));
    palette_colors.push_back(Color(0.235f, 0.286f, 0.231f));
    _clamp_palette();
}

void PainterlyMaterial::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_palette_textures", "textures"), &PainterlyMaterial::set_palette_textures);
    ClassDB::bind_method(D_METHOD("get_palette_textures"), &PainterlyMaterial::get_palette_textures);
    ClassDB::bind_method(D_METHOD("add_palette_texture", "texture"), &PainterlyMaterial::add_palette_texture);
    ClassDB::bind_method(D_METHOD("remove_palette_texture", "index"), &PainterlyMaterial::remove_palette_texture);
    ClassDB::bind_method(D_METHOD("get_palette_texture_count"), &PainterlyMaterial::get_palette_texture_count);

    ClassDB::bind_method(D_METHOD("set_noise_luts", "textures"), &PainterlyMaterial::set_noise_luts);
    ClassDB::bind_method(D_METHOD("get_noise_luts"), &PainterlyMaterial::get_noise_luts);
    ClassDB::bind_method(D_METHOD("add_noise_lut", "texture"), &PainterlyMaterial::add_noise_lut);
    ClassDB::bind_method(D_METHOD("remove_noise_lut", "index"), &PainterlyMaterial::remove_noise_lut);
    ClassDB::bind_method(D_METHOD("get_noise_lut_count"), &PainterlyMaterial::get_noise_lut_count);

    ClassDB::bind_method(D_METHOD("set_stroke_density_curve", "curve"), &PainterlyMaterial::set_stroke_density_curve);
    ClassDB::bind_method(D_METHOD("get_stroke_density_curve"), &PainterlyMaterial::get_stroke_density_curve);

    ClassDB::bind_method(D_METHOD("set_stroke_density_resolution", "resolution"), &PainterlyMaterial::set_stroke_density_resolution);
    ClassDB::bind_method(D_METHOD("get_stroke_density_resolution"), &PainterlyMaterial::get_stroke_density_resolution);

    ClassDB::bind_method(D_METHOD("set_stroke_density_strength", "strength"), &PainterlyMaterial::set_stroke_density_strength);
    ClassDB::bind_method(D_METHOD("get_stroke_density_strength"), &PainterlyMaterial::get_stroke_density_strength);

    ClassDB::bind_method(D_METHOD("get_stroke_density_lut"), &PainterlyMaterial::get_stroke_density_lut);

    ClassDB::bind_method(D_METHOD("set_palette_quantization_enabled", "enabled"), &PainterlyMaterial::set_palette_quantization_enabled);
    ClassDB::bind_method(D_METHOD("is_palette_quantization_enabled"), &PainterlyMaterial::is_palette_quantization_enabled);

    ClassDB::bind_method(D_METHOD("set_brush_modulation_enabled", "enabled"), &PainterlyMaterial::set_brush_modulation_enabled);
    ClassDB::bind_method(D_METHOD("is_brush_modulation_enabled"), &PainterlyMaterial::is_brush_modulation_enabled);

    ClassDB::bind_method(D_METHOD("set_lighting_stylization_enabled", "enabled"), &PainterlyMaterial::set_lighting_stylization_enabled);
    ClassDB::bind_method(D_METHOD("is_lighting_stylization_enabled"), &PainterlyMaterial::is_lighting_stylization_enabled);

    ClassDB::bind_method(D_METHOD("set_palette_colors", "colors"), &PainterlyMaterial::set_palette_colors);
    ClassDB::bind_method(D_METHOD("get_palette_colors"), &PainterlyMaterial::get_palette_colors);

    ClassDB::bind_method(D_METHOD("set_palette_blend_strength", "strength"), &PainterlyMaterial::set_palette_blend_strength);
    ClassDB::bind_method(D_METHOD("get_palette_blend_strength"), &PainterlyMaterial::get_palette_blend_strength);

    ClassDB::bind_method(D_METHOD("set_palette_noise_strength", "strength"), &PainterlyMaterial::set_palette_noise_strength);
    ClassDB::bind_method(D_METHOD("get_palette_noise_strength"), &PainterlyMaterial::get_palette_noise_strength);

    ClassDB::bind_method(D_METHOD("set_palette_preserve_luminance", "enabled"), &PainterlyMaterial::set_palette_preserve_luminance);
    ClassDB::bind_method(D_METHOD("get_palette_preserve_luminance"), &PainterlyMaterial::get_palette_preserve_luminance);

    ClassDB::bind_method(D_METHOD("set_shading_style", "style"), &PainterlyMaterial::set_shading_style);
    ClassDB::bind_method(D_METHOD("get_shading_style"), &PainterlyMaterial::get_shading_style);
    ClassDB::bind_method(D_METHOD("set_style_preset", "preset"), &PainterlyMaterial::set_style_preset);
    ClassDB::bind_method(D_METHOD("get_style_preset"), &PainterlyMaterial::get_style_preset);
    ClassDB::bind_method(D_METHOD("set_cel_band_count", "bands"), &PainterlyMaterial::set_cel_band_count);
    ClassDB::bind_method(D_METHOD("get_cel_band_count"), &PainterlyMaterial::get_cel_band_count);
    ClassDB::bind_method(D_METHOD("set_cel_smoothness", "smoothness"), &PainterlyMaterial::set_cel_smoothness);
    ClassDB::bind_method(D_METHOD("get_cel_smoothness"), &PainterlyMaterial::get_cel_smoothness);
    ClassDB::bind_method(D_METHOD("set_painterly_mix_strength", "strength"), &PainterlyMaterial::set_painterly_mix_strength);
    ClassDB::bind_method(D_METHOD("get_painterly_mix_strength"), &PainterlyMaterial::get_painterly_mix_strength);
    ClassDB::bind_method(D_METHOD("set_brush_texture_influence", "influence"), &PainterlyMaterial::set_brush_texture_influence);
    ClassDB::bind_method(D_METHOD("get_brush_texture_influence"), &PainterlyMaterial::get_brush_texture_influence);

    ClassDB::bind_method(D_METHOD("set_brush_scale", "scale"), &PainterlyMaterial::set_brush_scale);
    ClassDB::bind_method(D_METHOD("get_brush_scale"), &PainterlyMaterial::get_brush_scale);

    ClassDB::bind_method(D_METHOD("set_brush_softness", "softness"), &PainterlyMaterial::set_brush_softness);
    ClassDB::bind_method(D_METHOD("get_brush_softness"), &PainterlyMaterial::get_brush_softness);

    ClassDB::bind_method(D_METHOD("set_brush_anisotropy", "anisotropy"), &PainterlyMaterial::set_brush_anisotropy);
    ClassDB::bind_method(D_METHOD("get_brush_anisotropy"), &PainterlyMaterial::get_brush_anisotropy);

    ClassDB::bind_method(D_METHOD("set_brush_rotation_jitter", "jitter"), &PainterlyMaterial::set_brush_rotation_jitter);
    ClassDB::bind_method(D_METHOD("get_brush_rotation_jitter"), &PainterlyMaterial::get_brush_rotation_jitter);

    ClassDB::bind_method(D_METHOD("set_brush_shape_noise", "noise"), &PainterlyMaterial::set_brush_shape_noise);
    ClassDB::bind_method(D_METHOD("get_brush_shape_noise"), &PainterlyMaterial::get_brush_shape_noise);

    ClassDB::bind_method(D_METHOD("set_light_color", "color"), &PainterlyMaterial::set_light_color);
    ClassDB::bind_method(D_METHOD("get_light_color"), &PainterlyMaterial::get_light_color);

    ClassDB::bind_method(D_METHOD("set_ambient_color", "color"), &PainterlyMaterial::set_ambient_color);
    ClassDB::bind_method(D_METHOD("get_ambient_color"), &PainterlyMaterial::get_ambient_color);

    ClassDB::bind_method(D_METHOD("set_light_direction", "direction"), &PainterlyMaterial::set_light_direction);
    ClassDB::bind_method(D_METHOD("get_light_direction"), &PainterlyMaterial::get_light_direction);

    ClassDB::bind_method(D_METHOD("set_diffuse_strength", "strength"), &PainterlyMaterial::set_diffuse_strength);
    ClassDB::bind_method(D_METHOD("get_diffuse_strength"), &PainterlyMaterial::get_diffuse_strength);

    ClassDB::bind_method(D_METHOD("set_specular_strength", "strength"), &PainterlyMaterial::set_specular_strength);
    ClassDB::bind_method(D_METHOD("get_specular_strength"), &PainterlyMaterial::get_specular_strength);

    ClassDB::bind_method(D_METHOD("set_rim_strength", "strength"), &PainterlyMaterial::set_rim_strength);
    ClassDB::bind_method(D_METHOD("get_rim_strength"), &PainterlyMaterial::get_rim_strength);

    ClassDB::bind_method(D_METHOD("set_specular_power", "power"), &PainterlyMaterial::set_specular_power);
    ClassDB::bind_method(D_METHOD("get_specular_power"), &PainterlyMaterial::get_specular_power);

    ClassDB::bind_method(D_METHOD("set_rim_power", "power"), &PainterlyMaterial::set_rim_power);
    ClassDB::bind_method(D_METHOD("get_rim_power"), &PainterlyMaterial::get_rim_power);

    ClassDB::bind_method(D_METHOD("set_rim_color", "color"), &PainterlyMaterial::set_rim_color);
    ClassDB::bind_method(D_METHOD("get_rim_color"), &PainterlyMaterial::get_rim_color);
    ClassDB::bind_method(D_METHOD("set_shadow_color", "color"), &PainterlyMaterial::set_shadow_color);
    ClassDB::bind_method(D_METHOD("get_shadow_color"), &PainterlyMaterial::get_shadow_color);
    ClassDB::bind_method(D_METHOD("set_highlight_color", "color"), &PainterlyMaterial::set_highlight_color);
    ClassDB::bind_method(D_METHOD("get_highlight_color"), &PainterlyMaterial::get_highlight_color);
    ClassDB::bind_method(D_METHOD("set_color_temperature", "temperature"), &PainterlyMaterial::set_color_temperature);
    ClassDB::bind_method(D_METHOD("get_color_temperature"), &PainterlyMaterial::get_color_temperature);
    ClassDB::bind_method(D_METHOD("set_color_temperature_strength", "strength"), &PainterlyMaterial::set_color_temperature_strength);
    ClassDB::bind_method(D_METHOD("get_color_temperature_strength"), &PainterlyMaterial::get_color_temperature_strength);
    ClassDB::bind_method(D_METHOD("set_temporal_stability", "stability"), &PainterlyMaterial::set_temporal_stability);
    ClassDB::bind_method(D_METHOD("get_temporal_stability"), &PainterlyMaterial::get_temporal_stability);
    ClassDB::bind_method(D_METHOD("set_gooch_cool_mix", "amount"), &PainterlyMaterial::set_gooch_cool_mix);
    ClassDB::bind_method(D_METHOD("get_gooch_cool_mix"), &PainterlyMaterial::get_gooch_cool_mix);
    ClassDB::bind_method(D_METHOD("set_gooch_warm_mix", "amount"), &PainterlyMaterial::set_gooch_warm_mix);
    ClassDB::bind_method(D_METHOD("get_gooch_warm_mix"), &PainterlyMaterial::get_gooch_warm_mix);
    ClassDB::bind_method(D_METHOD("set_lighting_intensity", "intensity"), &PainterlyMaterial::set_lighting_intensity);
    ClassDB::bind_method(D_METHOD("get_lighting_intensity"), &PainterlyMaterial::get_lighting_intensity);

    ClassDB::bind_method(D_METHOD("get_shader_define_strings"), &PainterlyMaterial::get_shader_define_strings);

    ClassDB::bind_method(D_METHOD("has_required_resources"), &PainterlyMaterial::has_required_resources);
    ClassDB::bind_method(D_METHOD("get_missing_resources"), &PainterlyMaterial::get_missing_resources);

    ClassDB::bind_method(D_METHOD("serialize"), &PainterlyMaterial::serialize);
    ClassDB::bind_method(D_METHOD("deserialize", "data"), &PainterlyMaterial::deserialize);

    _bind_properties();
}

void PainterlyMaterial::_bind_properties() {
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "palette_textures", PROPERTY_HINT_ARRAY_TYPE, "Texture2D"), "set_palette_textures", "get_palette_textures");
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "noise_luts", PROPERTY_HINT_ARRAY_TYPE, "Texture2D"), "set_noise_luts", "get_noise_luts");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stroke_density_curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_stroke_density_curve", "get_stroke_density_curve");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "stroke_density_resolution", PROPERTY_HINT_RANGE, "8,2048,1"), "set_stroke_density_resolution", "get_stroke_density_resolution");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stroke_density_strength", PROPERTY_HINT_RANGE, "0.0,10.0,0.01"), "set_stroke_density_strength", "get_stroke_density_strength");

    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "palette_quantization_enabled"), "set_palette_quantization_enabled", "is_palette_quantization_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "brush_modulation_enabled"), "set_brush_modulation_enabled", "is_brush_modulation_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lighting_stylization_enabled"), "set_lighting_stylization_enabled", "is_lighting_stylization_enabled");

    ADD_PROPERTY(PropertyInfo(Variant::PACKED_COLOR_ARRAY, "palette_colors"), "set_palette_colors", "get_palette_colors");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "palette_blend_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_palette_blend_strength", "get_palette_blend_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "palette_noise_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_palette_noise_strength", "get_palette_noise_strength");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "palette_preserve_luminance"), "set_palette_preserve_luminance", "get_palette_preserve_luminance");

    ADD_PROPERTY(PropertyInfo(Variant::INT, "shading_style", PROPERTY_HINT_ENUM, "Realistic,Cel,Painterly,Gooch"), "set_shading_style", "get_shading_style");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "style_preset", PROPERTY_HINT_ENUM, "Custom,Realistic,Toon,Painterly,Technical"), "set_style_preset", "get_style_preset");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "cel_band_count", PROPERTY_HINT_RANGE, "1,16,1"), "set_cel_band_count", "get_cel_band_count");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cel_smoothness", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_cel_smoothness", "get_cel_smoothness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly_mix_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_painterly_mix_strength", "get_painterly_mix_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brush_texture_influence", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_brush_texture_influence", "get_brush_texture_influence");

    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brush_scale", PROPERTY_HINT_RANGE, "0.1,8,0.01"), "set_brush_scale", "get_brush_scale");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brush_softness", PROPERTY_HINT_RANGE, "0.1,4,0.01"), "set_brush_softness", "get_brush_softness");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brush_anisotropy", PROPERTY_HINT_RANGE, "0.1,4,0.01"), "set_brush_anisotropy", "get_brush_anisotropy");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brush_rotation_jitter", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_brush_rotation_jitter", "get_brush_rotation_jitter");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "brush_shape_noise", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_brush_shape_noise", "get_brush_shape_noise");

    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "light_color"), "set_light_color", "get_light_color");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ambient_color"), "set_ambient_color", "get_ambient_color");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "light_direction"), "set_light_direction", "get_light_direction");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "diffuse_strength", PROPERTY_HINT_RANGE, "0,2,0.01"), "set_diffuse_strength", "get_diffuse_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "specular_strength", PROPERTY_HINT_RANGE, "0,2,0.01"), "set_specular_strength", "get_specular_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rim_strength", PROPERTY_HINT_RANGE, "0,2,0.01"), "set_rim_strength", "get_rim_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "specular_power", PROPERTY_HINT_RANGE, "1,128,0.1"), "set_specular_power", "get_specular_power");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rim_power", PROPERTY_HINT_RANGE, "0.1,16,0.1"), "set_rim_power", "get_rim_power");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "rim_color"), "set_rim_color", "get_rim_color");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "shadow_color"), "set_shadow_color", "get_shadow_color");
    ADD_PROPERTY(PropertyInfo(Variant::COLOR, "highlight_color"), "set_highlight_color", "get_highlight_color");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "color_temperature", PROPERTY_HINT_RANGE, "1000,20000,1"), "set_color_temperature", "get_color_temperature");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "color_temperature_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_color_temperature_strength", "get_color_temperature_strength");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temporal_stability", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_temporal_stability", "get_temporal_stability");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gooch_cool_mix", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_gooch_cool_mix", "get_gooch_cool_mix");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gooch_warm_mix", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_gooch_warm_mix", "get_gooch_warm_mix");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lighting_intensity", PROPERTY_HINT_RANGE, "0,8,0.01"), "set_lighting_intensity", "get_lighting_intensity");

    BIND_ENUM_CONSTANT(SHADING_STYLE_REALISTIC);
    BIND_ENUM_CONSTANT(SHADING_STYLE_CEL);
    BIND_ENUM_CONSTANT(SHADING_STYLE_PAINTERLY);
    BIND_ENUM_CONSTANT(SHADING_STYLE_GOOCH);

    BIND_ENUM_CONSTANT(STYLE_PRESET_CUSTOM);
    BIND_ENUM_CONSTANT(STYLE_PRESET_REALISTIC);
    BIND_ENUM_CONSTANT(STYLE_PRESET_TOON);
    BIND_ENUM_CONSTANT(STYLE_PRESET_PAINTERLY);
    BIND_ENUM_CONSTANT(STYLE_PRESET_TECHNICAL);
}

void PainterlyMaterial::_disconnect_curve() {
    if (stroke_density_curve.is_valid() && stroke_density_curve->is_connected("changed", callable_mp(this, &PainterlyMaterial::_on_stroke_density_curve_changed))) {
        stroke_density_curve->disconnect("changed", callable_mp(this, &PainterlyMaterial::_on_stroke_density_curve_changed));
    }
}

void PainterlyMaterial::_connect_curve() {
    if (stroke_density_curve.is_valid() && !stroke_density_curve->is_connected("changed", callable_mp(this, &PainterlyMaterial::_on_stroke_density_curve_changed))) {
        stroke_density_curve->connect("changed", callable_mp(this, &PainterlyMaterial::_on_stroke_density_curve_changed));
    }
}

void PainterlyMaterial::_on_stroke_density_curve_changed() {
    _mark_stroke_density_dirty();
    _emit_changed(DIRTY_STROKE_DENSITY);
}

void PainterlyMaterial::_mark_stroke_density_dirty() {
    stroke_density_dirty = true;
}

void PainterlyMaterial::_emit_changed(uint32_t p_dirty_flag) {
    if (defer_changed_notifications) {
        changed_notification_pending = true;
        deferred_dirty_flags |= p_dirty_flag;
        return;
    }
    Resource::emit_changed();
}

void PainterlyMaterial::_begin_bulk_update() {
    defer_changed_notifications = true;
    changed_notification_pending = false;
    deferred_dirty_flags = 0;
}

void PainterlyMaterial::_end_bulk_update() {
    const bool should_emit = changed_notification_pending;
    const uint32_t dirty = deferred_dirty_flags;
    defer_changed_notifications = false;
    changed_notification_pending = false;
    deferred_dirty_flags = 0;
    if (should_emit) {
        // Emit a single consolidated notification. Consumers can query
        // which groups changed via get_deferred_dirty_flags() if needed,
        // but for now the bitmask is logged for diagnostics and a single
        // emit_changed() is fired (matching Godot Resource semantics).
        (void)dirty; // Available for future per-group invalidation.
        Resource::emit_changed();
    }
}

void PainterlyMaterial::_clamp_palette() {
    if (palette_colors.size() > MAX_PALETTE_COLORS) {
        PackedColorArray trimmed;
        trimmed.resize(MAX_PALETTE_COLORS);
        for (int i = 0; i < MAX_PALETTE_COLORS; i++) {
            trimmed.set(i, palette_colors[i]);
        }
        palette_colors = trimmed;
    }
}

void PainterlyMaterial::_apply_style_preset(StylePreset p_preset) {
    switch (p_preset) {
        case STYLE_PRESET_REALISTIC: {
            shading_style = SHADING_STYLE_REALISTIC;
            cel_band_count = 4;
            cel_smoothness = 0.2f;
            painterly_mix_strength = 0.25f;
            brush_texture_influence = 0.2f;
            diffuse_strength = 0.85f;
            specular_strength = 0.25f;
            rim_strength = 0.3f;
            rim_power = 2.5f;
            rim_color = Color(1.0f, 0.95f, 0.9f, 1.0f);
            shadow_color = Color(0.18f, 0.17f, 0.20f, 0.4f);
            highlight_color = Color(1.0f, 0.97f, 0.90f, 0.55f);
            color_temperature = 6500.0f;
            color_temperature_strength = 0.1f;
            temporal_stability = 0.3f;
            gooch_cool_mix = 0.3f;
            gooch_warm_mix = 0.6f;
            lighting_intensity = 1.0f;
            break;
        }
        case STYLE_PRESET_TOON: {
            shading_style = SHADING_STYLE_CEL;
            cel_band_count = 4;
            cel_smoothness = 0.35f;
            painterly_mix_strength = 0.35f;
            brush_texture_influence = 0.15f;
            diffuse_strength = 0.95f;
            specular_strength = 0.15f;
            rim_strength = 0.45f;
            rim_power = 2.0f;
            rim_color = Color(1.0f, 0.9f, 0.7f, 1.0f);
            shadow_color = Color(0.2f, 0.22f, 0.32f, 0.6f);
            highlight_color = Color(1.1f, 1.05f, 0.95f, 0.85f);
            color_temperature = 6500.0f;
            color_temperature_strength = 0.2f;
            temporal_stability = 0.5f;
            gooch_cool_mix = 0.4f;
            gooch_warm_mix = 0.6f;
            lighting_intensity = 1.15f;
            break;
        }
        case STYLE_PRESET_PAINTERLY: {
            shading_style = SHADING_STYLE_PAINTERLY;
            cel_band_count = 3;
            cel_smoothness = 0.55f;
            painterly_mix_strength = 0.75f;
            brush_texture_influence = 0.65f;
            diffuse_strength = 0.9f;
            specular_strength = 0.25f;
            rim_strength = 0.35f;
            rim_power = 2.8f;
            rim_color = Color(1.08f, 0.96f, 0.85f, 1.0f);
            shadow_color = Color(0.25f, 0.2f, 0.22f, 0.55f);
            highlight_color = Color(1.15f, 1.05f, 0.9f, 0.9f);
            color_temperature = 5800.0f;
            color_temperature_strength = 0.35f;
            temporal_stability = 0.65f;
            gooch_cool_mix = 0.45f;
            gooch_warm_mix = 0.55f;
            lighting_intensity = 1.2f;
            break;
        }
        case STYLE_PRESET_TECHNICAL: {
            shading_style = SHADING_STYLE_GOOCH;
            cel_band_count = 3;
            cel_smoothness = 0.25f;
            painterly_mix_strength = 0.4f;
            brush_texture_influence = 0.25f;
            diffuse_strength = 0.8f;
            specular_strength = 0.35f;
            rim_strength = 0.2f;
            rim_power = 3.5f;
            rim_color = Color(0.95f, 0.95f, 1.0f, 1.0f);
            shadow_color = Color(0.2f, 0.3f, 0.45f, 0.6f);
            highlight_color = Color(1.05f, 1.0f, 0.95f, 0.65f);
            color_temperature = 7200.0f;
            color_temperature_strength = 0.25f;
            temporal_stability = 0.45f;
            gooch_cool_mix = 0.55f;
            gooch_warm_mix = 0.45f;
            lighting_intensity = 1.1f;
            break;
        }
        case STYLE_PRESET_CUSTOM:
        default:
            break;
    }

    cel_band_count = CLAMP(cel_band_count, 1, 16);
    cel_smoothness = CLAMP(cel_smoothness, 0.0f, 1.0f);
    painterly_mix_strength = CLAMP(painterly_mix_strength, 0.0f, 1.0f);
    brush_texture_influence = CLAMP(brush_texture_influence, 0.0f, 1.0f);
    diffuse_strength = MAX(diffuse_strength, 0.0f);
    specular_strength = MAX(specular_strength, 0.0f);
    rim_strength = MAX(rim_strength, 0.0f);
    rim_power = MAX(rim_power, 0.1f);
    color_temperature = CLAMP(color_temperature, 1000.0f, 20000.0f);
    color_temperature_strength = CLAMP(color_temperature_strength, 0.0f, 1.0f);
    temporal_stability = CLAMP(temporal_stability, 0.0f, 1.0f);
    gooch_cool_mix = CLAMP(gooch_cool_mix, 0.0f, 1.0f);
    gooch_warm_mix = CLAMP(gooch_warm_mix, 0.0f, 1.0f);
    lighting_intensity = MAX(lighting_intensity, 0.0f);
}

void PainterlyMaterial::set_palette_textures(const TypedArray<Texture2D> &p_textures) {
    palette_textures.clear();
    for (int i = 0; i < p_textures.size(); i++) {
        Ref<Texture2D> tex = p_textures[i];
        palette_textures.push_back(tex);
    }
    _emit_changed();
}

TypedArray<Texture2D> PainterlyMaterial::get_palette_textures() const {
    TypedArray<Texture2D> result;
    result.resize(palette_textures.size());
    for (int i = 0; i < palette_textures.size(); i++) {
        result[i] = palette_textures[i];
    }
    return result;
}

void PainterlyMaterial::add_palette_texture(const Ref<Texture2D> &p_texture) {
    if (p_texture.is_null()) {
        return;
    }
    palette_textures.push_back(p_texture);
    _emit_changed();
}

void PainterlyMaterial::remove_palette_texture(int p_index) {
    ERR_FAIL_INDEX(p_index, palette_textures.size());
    palette_textures.remove_at(p_index);
    _emit_changed();
}

int PainterlyMaterial::get_palette_texture_count() const {
    return palette_textures.size();
}

void PainterlyMaterial::set_noise_luts(const TypedArray<Texture2D> &p_textures) {
    noise_luts.clear();
    for (int i = 0; i < p_textures.size(); i++) {
        Ref<Texture2D> tex = p_textures[i];
        noise_luts.push_back(tex);
    }
    _emit_changed();
}

TypedArray<Texture2D> PainterlyMaterial::get_noise_luts() const {
    TypedArray<Texture2D> result;
    result.resize(noise_luts.size());
    for (int i = 0; i < noise_luts.size(); i++) {
        result[i] = noise_luts[i];
    }
    return result;
}

void PainterlyMaterial::add_noise_lut(const Ref<Texture2D> &p_texture) {
    if (p_texture.is_null()) {
        return;
    }
    noise_luts.push_back(p_texture);
    _emit_changed();
}

void PainterlyMaterial::remove_noise_lut(int p_index) {
    ERR_FAIL_INDEX(p_index, noise_luts.size());
    noise_luts.remove_at(p_index);
    _emit_changed();
}

int PainterlyMaterial::get_noise_lut_count() const {
    return noise_luts.size();
}

void PainterlyMaterial::set_stroke_density_curve(const Ref<Curve> &p_curve) {
    if (stroke_density_curve == p_curve) {
        return;
    }

    _disconnect_curve();
    stroke_density_curve = p_curve;
    _connect_curve();
    _mark_stroke_density_dirty();
    _emit_changed(DIRTY_STROKE_DENSITY);
}

void PainterlyMaterial::set_stroke_density_resolution(int p_resolution) {
    int resolution = MAX(8, p_resolution);
    if (resolution == stroke_density_resolution) {
        return;
    }
    stroke_density_resolution = resolution;
    _mark_stroke_density_dirty();
    _emit_changed(DIRTY_STROKE_DENSITY);
}

void PainterlyMaterial::set_stroke_density_strength(float p_strength) {
    float clamped = MAX(p_strength, 0.0f);
    if (Math::is_equal_approx(clamped, stroke_density_strength)) {
        return;
    }
    stroke_density_strength = clamped;
    _mark_stroke_density_dirty();
    _emit_changed(DIRTY_STROKE_DENSITY);
}

PackedFloat32Array PainterlyMaterial::get_stroke_density_lut() const {
    if (!stroke_density_dirty && cached_stroke_density_lut.size() == stroke_density_resolution) {
        return cached_stroke_density_lut;
    }

    cached_stroke_density_lut.resize(stroke_density_resolution);

    if (stroke_density_curve.is_valid()) {
        for (int i = 0; i < stroke_density_resolution; i++) {
            float t = stroke_density_resolution == 1 ? 0.0f : (float)i / (stroke_density_resolution - 1);
            float value = stroke_density_curve->sample(t);
            cached_stroke_density_lut.set(i, value * stroke_density_strength);
        }
    } else {
        for (int i = 0; i < stroke_density_resolution; i++) {
            cached_stroke_density_lut.set(i, stroke_density_strength);
        }
    }

    stroke_density_dirty = false;
    return cached_stroke_density_lut;
}

void PainterlyMaterial::set_palette_quantization_enabled(bool p_enabled) {
    if (palette_quantization_enabled == p_enabled) {
        return;
    }
    palette_quantization_enabled = p_enabled;
    _emit_changed(DIRTY_FEATURE_TOGGLE);
}

bool PainterlyMaterial::is_palette_quantization_enabled() const {
    return palette_quantization_enabled;
}

void PainterlyMaterial::set_brush_modulation_enabled(bool p_enabled) {
    if (brush_modulation_enabled == p_enabled) {
        return;
    }
    brush_modulation_enabled = p_enabled;
    _emit_changed(DIRTY_FEATURE_TOGGLE);
}

bool PainterlyMaterial::is_brush_modulation_enabled() const {
    return brush_modulation_enabled;
}

void PainterlyMaterial::set_lighting_stylization_enabled(bool p_enabled) {
    if (lighting_stylization_enabled == p_enabled) {
        return;
    }
    lighting_stylization_enabled = p_enabled;
    _emit_changed(DIRTY_FEATURE_TOGGLE);
}

bool PainterlyMaterial::is_lighting_stylization_enabled() const {
    return lighting_stylization_enabled;
}

void PainterlyMaterial::set_palette_colors(const PackedColorArray &p_colors) {
    palette_colors = p_colors;
    _clamp_palette();
    _emit_changed(DIRTY_PALETTE);
}

PackedColorArray PainterlyMaterial::get_palette_colors() const {
    return palette_colors;
}

void PainterlyMaterial::set_palette_blend_strength(float p_strength) {
    float clamped = CLAMP(p_strength, 0.0f, 1.0f);
    if (Math::is_equal_approx(clamped, palette_blend_strength)) {
        return;
    }
    palette_blend_strength = clamped;
    _emit_changed(DIRTY_PALETTE);
}

float PainterlyMaterial::get_palette_blend_strength() const {
    return palette_blend_strength;
}

void PainterlyMaterial::set_palette_noise_strength(float p_strength) {
    float clamped = CLAMP(p_strength, 0.0f, 1.0f);
    if (Math::is_equal_approx(clamped, palette_noise_strength)) {
        return;
    }
    palette_noise_strength = clamped;
    _emit_changed(DIRTY_PALETTE);
}

float PainterlyMaterial::get_palette_noise_strength() const {
    return palette_noise_strength;
}

void PainterlyMaterial::set_palette_preserve_luminance(bool p_preserve) {
    if (palette_preserve_luminance == p_preserve) {
        return;
    }
    palette_preserve_luminance = p_preserve;
    _emit_changed(DIRTY_PALETTE);
}

bool PainterlyMaterial::get_palette_preserve_luminance() const {
    return palette_preserve_luminance;
}

void PainterlyMaterial::set_shading_style(int p_style) {
    ShadingStyle style = SHADING_STYLE_REALISTIC;
    if (p_style == SHADING_STYLE_CEL) {
        style = SHADING_STYLE_CEL;
    } else if (p_style == SHADING_STYLE_PAINTERLY) {
        style = SHADING_STYLE_PAINTERLY;
    } else if (p_style == SHADING_STYLE_GOOCH) {
        style = SHADING_STYLE_GOOCH;
    }

    if (shading_style == style) {
        return;
    }

    shading_style = style;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_SHADING_STYLE);
}

int PainterlyMaterial::get_shading_style() const {
    return shading_style;
}

void PainterlyMaterial::set_style_preset(int p_preset) {
    StylePreset preset = STYLE_PRESET_CUSTOM;
    switch (p_preset) {
        case STYLE_PRESET_REALISTIC:
            preset = STYLE_PRESET_REALISTIC;
            break;
        case STYLE_PRESET_TOON:
            preset = STYLE_PRESET_TOON;
            break;
        case STYLE_PRESET_PAINTERLY:
            preset = STYLE_PRESET_PAINTERLY;
            break;
        case STYLE_PRESET_TECHNICAL:
            preset = STYLE_PRESET_TECHNICAL;
            break;
        default:
            break;
    }

    if (style_preset == preset) {
        return;
    }

    style_preset = preset;
    _apply_style_preset(preset);
    _emit_changed(DIRTY_SHADING_STYLE | DIRTY_LIGHTING | DIRTY_BRUSH);
}

int PainterlyMaterial::get_style_preset() const {
    return style_preset;
}

void PainterlyMaterial::set_cel_band_count(int p_bands) {
    int clamped = CLAMP(p_bands, 1, 16);
    if (cel_band_count == clamped) {
        return;
    }
    cel_band_count = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_SHADING_STYLE);
}

int PainterlyMaterial::get_cel_band_count() const {
    return cel_band_count;
}

void PainterlyMaterial::set_cel_smoothness(float p_smoothness) {
    float clamped = CLAMP(p_smoothness, 0.0f, 1.0f);
    if (Math::is_equal_approx(cel_smoothness, clamped)) {
        return;
    }
    cel_smoothness = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_SHADING_STYLE);
}

float PainterlyMaterial::get_cel_smoothness() const {
    return cel_smoothness;
}

void PainterlyMaterial::set_painterly_mix_strength(float p_strength) {
    float clamped = CLAMP(p_strength, 0.0f, 1.0f);
    if (Math::is_equal_approx(painterly_mix_strength, clamped)) {
        return;
    }
    painterly_mix_strength = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_SHADING_STYLE);
}

float PainterlyMaterial::get_painterly_mix_strength() const {
    return painterly_mix_strength;
}

void PainterlyMaterial::set_brush_texture_influence(float p_influence) {
    float clamped = CLAMP(p_influence, 0.0f, 1.0f);
    if (Math::is_equal_approx(brush_texture_influence, clamped)) {
        return;
    }
    brush_texture_influence = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_SHADING_STYLE);
}

float PainterlyMaterial::get_brush_texture_influence() const {
    return brush_texture_influence;
}

void PainterlyMaterial::set_brush_scale(float p_scale) {
    float clamped = MAX(p_scale, 0.1f);
    if (Math::is_equal_approx(clamped, brush_scale)) {
        return;
    }
    brush_scale = clamped;
    _emit_changed(DIRTY_BRUSH);
}

float PainterlyMaterial::get_brush_scale() const {
    return brush_scale;
}

void PainterlyMaterial::set_brush_softness(float p_softness) {
    float clamped = MAX(p_softness, 0.1f);
    if (Math::is_equal_approx(clamped, brush_softness)) {
        return;
    }
    brush_softness = clamped;
    _emit_changed(DIRTY_BRUSH);
}

float PainterlyMaterial::get_brush_softness() const {
    return brush_softness;
}

void PainterlyMaterial::set_brush_anisotropy(float p_anisotropy) {
    float clamped = MAX(p_anisotropy, 0.1f);
    if (Math::is_equal_approx(clamped, brush_anisotropy)) {
        return;
    }
    brush_anisotropy = clamped;
    _emit_changed(DIRTY_BRUSH);
}

float PainterlyMaterial::get_brush_anisotropy() const {
    return brush_anisotropy;
}

void PainterlyMaterial::set_brush_rotation_jitter(float p_jitter) {
    float clamped = CLAMP(p_jitter, 0.0f, 1.0f);
    if (Math::is_equal_approx(clamped, brush_rotation_jitter)) {
        return;
    }
    brush_rotation_jitter = clamped;
    _emit_changed(DIRTY_BRUSH);
}

float PainterlyMaterial::get_brush_rotation_jitter() const {
    return brush_rotation_jitter;
}

void PainterlyMaterial::set_brush_shape_noise(float p_noise) {
    float clamped = CLAMP(p_noise, 0.0f, 1.0f);
    if (Math::is_equal_approx(clamped, brush_shape_noise)) {
        return;
    }
    brush_shape_noise = clamped;
    _emit_changed(DIRTY_BRUSH);
}

float PainterlyMaterial::get_brush_shape_noise() const {
    return brush_shape_noise;
}

void PainterlyMaterial::set_light_color(const Color &p_color) {
    if (light_color == p_color) {
        return;
    }
    light_color = p_color;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

Color PainterlyMaterial::get_light_color() const {
    return light_color;
}

void PainterlyMaterial::set_ambient_color(const Color &p_color) {
    if (ambient_color == p_color) {
        return;
    }
    ambient_color = p_color;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

Color PainterlyMaterial::get_ambient_color() const {
    return ambient_color;
}

void PainterlyMaterial::set_light_direction(const Vector3 &p_direction) {
    Vector3 normalized = p_direction;
    if (normalized.length_squared() == 0.0f) {
        normalized = Vector3(0.0f, 0.0f, -1.0f);
    } else {
        normalized.normalize();
    }
    if (light_direction == normalized) {
        return;
    }
    light_direction = normalized;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

Vector3 PainterlyMaterial::get_light_direction() const {
    return light_direction;
}

void PainterlyMaterial::set_diffuse_strength(float p_strength) {
    float clamped = MAX(p_strength, 0.0f);
    if (Math::is_equal_approx(clamped, diffuse_strength)) {
        return;
    }
    diffuse_strength = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_diffuse_strength() const {
    return diffuse_strength;
}

void PainterlyMaterial::set_specular_strength(float p_strength) {
    float clamped = MAX(p_strength, 0.0f);
    if (Math::is_equal_approx(clamped, specular_strength)) {
        return;
    }
    specular_strength = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_specular_strength() const {
    return specular_strength;
}

void PainterlyMaterial::set_rim_strength(float p_strength) {
    float clamped = MAX(p_strength, 0.0f);
    if (Math::is_equal_approx(clamped, rim_strength)) {
        return;
    }
    rim_strength = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_rim_strength() const {
    return rim_strength;
}

void PainterlyMaterial::set_specular_power(float p_power) {
    float clamped = MAX(p_power, 1.0f);
    if (Math::is_equal_approx(clamped, specular_power)) {
        return;
    }
    specular_power = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_specular_power() const {
    return specular_power;
}

void PainterlyMaterial::set_rim_power(float p_power) {
    float clamped = MAX(p_power, 0.1f);
    if (Math::is_equal_approx(clamped, rim_power)) {
        return;
    }
    rim_power = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_rim_power() const {
    return rim_power;
}

void PainterlyMaterial::set_rim_color(const Color &p_color) {
    if (rim_color == p_color) {
        return;
    }
    rim_color = p_color;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

Color PainterlyMaterial::get_rim_color() const {
    return rim_color;
}

void PainterlyMaterial::set_shadow_color(const Color &p_color) {
    if (shadow_color == p_color) {
        return;
    }
    shadow_color = p_color;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

Color PainterlyMaterial::get_shadow_color() const {
    return shadow_color;
}

void PainterlyMaterial::set_highlight_color(const Color &p_color) {
    if (highlight_color == p_color) {
        return;
    }
    highlight_color = p_color;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

Color PainterlyMaterial::get_highlight_color() const {
    return highlight_color;
}

void PainterlyMaterial::set_color_temperature(float p_temperature) {
    float clamped = CLAMP(p_temperature, 1000.0f, 20000.0f);
    if (Math::is_equal_approx(color_temperature, clamped)) {
        return;
    }
    color_temperature = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_color_temperature() const {
    return color_temperature;
}

void PainterlyMaterial::set_color_temperature_strength(float p_strength) {
    float clamped = CLAMP(p_strength, 0.0f, 1.0f);
    if (Math::is_equal_approx(color_temperature_strength, clamped)) {
        return;
    }
    color_temperature_strength = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_color_temperature_strength() const {
    return color_temperature_strength;
}

void PainterlyMaterial::set_temporal_stability(float p_value) {
    float clamped = CLAMP(p_value, 0.0f, 1.0f);
    if (Math::is_equal_approx(temporal_stability, clamped)) {
        return;
    }
    temporal_stability = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_temporal_stability() const {
    return temporal_stability;
}

void PainterlyMaterial::set_gooch_cool_mix(float p_value) {
    float clamped = CLAMP(p_value, 0.0f, 1.0f);
    if (Math::is_equal_approx(gooch_cool_mix, clamped)) {
        return;
    }
    gooch_cool_mix = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_gooch_cool_mix() const {
    return gooch_cool_mix;
}

void PainterlyMaterial::set_gooch_warm_mix(float p_value) {
    float clamped = CLAMP(p_value, 0.0f, 1.0f);
    if (Math::is_equal_approx(gooch_warm_mix, clamped)) {
        return;
    }
    gooch_warm_mix = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_gooch_warm_mix() const {
    return gooch_warm_mix;
}

void PainterlyMaterial::set_lighting_intensity(float p_intensity) {
    float clamped = MAX(p_intensity, 0.0f);
    if (Math::is_equal_approx(lighting_intensity, clamped)) {
        return;
    }
    lighting_intensity = clamped;
    style_preset = STYLE_PRESET_CUSTOM;
    _emit_changed(DIRTY_LIGHTING);
}

float PainterlyMaterial::get_lighting_intensity() const {
    return lighting_intensity;
}

PackedStringArray PainterlyMaterial::get_shader_define_strings() const {
    PackedStringArray defines;

    if (palette_quantization_enabled) {
        defines.push_back("PAINTERLY_ENABLE_PALETTE");
    }
    if (brush_modulation_enabled) {
        defines.push_back("PAINTERLY_ENABLE_BRUSH");
    }
    if (lighting_stylization_enabled) {
        defines.push_back("PAINTERLY_ENABLE_LIGHTING");
    }

    return defines;
}

bool PainterlyMaterial::has_required_resources() const {
    return palette_textures.size() > 0 && noise_luts.size() > 0 && stroke_density_curve.is_valid();
}

Vector<String> PainterlyMaterial::get_missing_resources() const {
    Vector<String> missing;
    if (palette_textures.is_empty()) {
        missing.push_back("palette_textures");
    }
    if (noise_luts.is_empty()) {
        missing.push_back("noise_luts");
    }
    if (!stroke_density_curve.is_valid()) {
        missing.push_back("stroke_density_curve");
    }
    return missing;
}

Dictionary PainterlyMaterial::serialize() const {
    Dictionary data;
    data["palette_textures"] = get_palette_textures();
    data["noise_luts"] = get_noise_luts();
    data["stroke_density_curve"] = stroke_density_curve;
    data["stroke_density_resolution"] = stroke_density_resolution;
    data["stroke_density_strength"] = stroke_density_strength;

    data["palette_quantization_enabled"] = palette_quantization_enabled;
    data["brush_modulation_enabled"] = brush_modulation_enabled;
    data["lighting_stylization_enabled"] = lighting_stylization_enabled;
    data["palette_colors"] = palette_colors;
    data["palette_blend_strength"] = palette_blend_strength;
    data["palette_noise_strength"] = palette_noise_strength;
    data["palette_preserve_luminance"] = palette_preserve_luminance;

    data["shading_style"] = shading_style;
    data["style_preset"] = style_preset;
    data["cel_band_count"] = cel_band_count;
    data["cel_smoothness"] = cel_smoothness;
    data["painterly_mix_strength"] = painterly_mix_strength;
    data["brush_texture_influence"] = brush_texture_influence;

    data["brush_scale"] = brush_scale;
    data["brush_softness"] = brush_softness;
    data["brush_anisotropy"] = brush_anisotropy;
    data["brush_rotation_jitter"] = brush_rotation_jitter;
    data["brush_shape_noise"] = brush_shape_noise;

    data["light_color"] = light_color;
    data["ambient_color"] = ambient_color;
    data["light_direction"] = light_direction;
    data["diffuse_strength"] = diffuse_strength;
    data["specular_strength"] = specular_strength;
    data["rim_strength"] = rim_strength;
    data["specular_power"] = specular_power;
    data["rim_power"] = rim_power;
    data["rim_color"] = rim_color;
    data["shadow_color"] = shadow_color;
    data["highlight_color"] = highlight_color;
    data["color_temperature"] = color_temperature;
    data["color_temperature_strength"] = color_temperature_strength;
    data["temporal_stability"] = temporal_stability;
    data["gooch_cool_mix"] = gooch_cool_mix;
    data["gooch_warm_mix"] = gooch_warm_mix;
    data["lighting_intensity"] = lighting_intensity;

    return data;
}

void PainterlyMaterial::deserialize(const Dictionary &p_data) {
    _begin_bulk_update();
    if (p_data.has("palette_textures")) {
        TypedArray<Texture2D> textures = p_data["palette_textures"];
        set_palette_textures(textures);
    }
    if (p_data.has("noise_luts")) {
        TypedArray<Texture2D> luts = p_data["noise_luts"];
        set_noise_luts(luts);
    }
    if (p_data.has("stroke_density_curve")) {
        set_stroke_density_curve(p_data["stroke_density_curve"]);
    }
    if (p_data.has("stroke_density_resolution")) {
        set_stroke_density_resolution((int)p_data["stroke_density_resolution"]);
    }
    if (p_data.has("stroke_density_strength")) {
        set_stroke_density_strength((float)p_data["stroke_density_strength"]);
    }

    if (p_data.has("palette_quantization_enabled")) {
        set_palette_quantization_enabled(p_data["palette_quantization_enabled"]);
    }
    if (p_data.has("brush_modulation_enabled")) {
        set_brush_modulation_enabled(p_data["brush_modulation_enabled"]);
    }
    if (p_data.has("lighting_stylization_enabled")) {
        set_lighting_stylization_enabled(p_data["lighting_stylization_enabled"]);
    }
    if (p_data.has("palette_colors")) {
        set_palette_colors(p_data["palette_colors"]);
    }
    if (p_data.has("palette_blend_strength")) {
        set_palette_blend_strength((float)p_data["palette_blend_strength"]);
    }
    if (p_data.has("palette_noise_strength")) {
        set_palette_noise_strength((float)p_data["palette_noise_strength"]);
    }
    if (p_data.has("palette_preserve_luminance")) {
        set_palette_preserve_luminance(p_data["palette_preserve_luminance"]);
    }

    StylePreset saved_preset = (StylePreset)style_preset;
    if (p_data.has("style_preset")) {
        int preset_value = (int)p_data["style_preset"];
        preset_value = std::clamp(preset_value, (int)STYLE_PRESET_CUSTOM, (int)STYLE_PRESET_TECHNICAL);
        saved_preset = (StylePreset)preset_value;
    }
    if (p_data.has("shading_style")) {
        set_shading_style((int)p_data["shading_style"]);
    }
    if (p_data.has("cel_band_count")) {
        set_cel_band_count((int)p_data["cel_band_count"]);
    }
    if (p_data.has("cel_smoothness")) {
        set_cel_smoothness((float)p_data["cel_smoothness"]);
    }
    if (p_data.has("painterly_mix_strength")) {
        set_painterly_mix_strength((float)p_data["painterly_mix_strength"]);
    }
    if (p_data.has("brush_texture_influence")) {
        set_brush_texture_influence((float)p_data["brush_texture_influence"]);
    }

    if (p_data.has("brush_scale")) {
        set_brush_scale((float)p_data["brush_scale"]);
    }
    if (p_data.has("brush_softness")) {
        set_brush_softness((float)p_data["brush_softness"]);
    }
    if (p_data.has("brush_anisotropy")) {
        set_brush_anisotropy((float)p_data["brush_anisotropy"]);
    }
    if (p_data.has("brush_rotation_jitter")) {
        set_brush_rotation_jitter((float)p_data["brush_rotation_jitter"]);
    }
    if (p_data.has("brush_shape_noise")) {
        set_brush_shape_noise((float)p_data["brush_shape_noise"]);
    }

    if (p_data.has("light_color")) {
        set_light_color(p_data["light_color"]);
    }
    if (p_data.has("ambient_color")) {
        set_ambient_color(p_data["ambient_color"]);
    }
    if (p_data.has("light_direction")) {
        set_light_direction(p_data["light_direction"]);
    }
    if (p_data.has("diffuse_strength")) {
        set_diffuse_strength((float)p_data["diffuse_strength"]);
    }
    if (p_data.has("specular_strength")) {
        set_specular_strength((float)p_data["specular_strength"]);
    }
    if (p_data.has("rim_strength")) {
        set_rim_strength((float)p_data["rim_strength"]);
    }
    if (p_data.has("specular_power")) {
        set_specular_power((float)p_data["specular_power"]);
    }
    if (p_data.has("rim_power")) {
        set_rim_power((float)p_data["rim_power"]);
    }
    if (p_data.has("rim_color")) {
        set_rim_color(p_data["rim_color"]);
    }
    if (p_data.has("shadow_color")) {
        set_shadow_color(p_data["shadow_color"]);
    }
    if (p_data.has("highlight_color")) {
        set_highlight_color(p_data["highlight_color"]);
    }
    if (p_data.has("color_temperature")) {
        set_color_temperature((float)p_data["color_temperature"]);
    }
    if (p_data.has("color_temperature_strength")) {
        set_color_temperature_strength((float)p_data["color_temperature_strength"]);
    }
    if (p_data.has("temporal_stability")) {
        set_temporal_stability((float)p_data["temporal_stability"]);
    }
    if (p_data.has("gooch_cool_mix")) {
        set_gooch_cool_mix((float)p_data["gooch_cool_mix"]);
    }
    if (p_data.has("gooch_warm_mix")) {
        set_gooch_warm_mix((float)p_data["gooch_warm_mix"]);
    }
    if (p_data.has("lighting_intensity")) {
        set_lighting_intensity((float)p_data["lighting_intensity"]);
    }

    style_preset = saved_preset;
    _end_bulk_update();
}
