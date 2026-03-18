/**
 * @file gaussian_data_color_grading.cpp
 * @brief Companion .cpp for gaussian_data.h containing color grading
 *        bake/restore methods.
 *
 * These methods apply, revert, and evaluate color grading transformations
 * on the SH DC coefficients stored in each Gaussian.  They are split out
 * from the main gaussian_data.cpp to keep that file focused on core
 * storage, spatial queries, and GPU upload paths.
 */

#include "gaussian_data.h"

#include "../resources/color_grading_resource.h"
#include "core/error/error_macros.h"

Error GaussianData::bake_color_grading(const Ref<ColorGradingResource> &p_grading) {
    ERR_FAIL_COND_V(!p_grading.is_valid(), ERR_INVALID_PARAMETER);
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND_V(gaussians.size() == 0, ERR_INVALID_DATA);

    // Backup original SH DC coefficients (only once)
    if (!bake_info.is_baked) {
        bake_info.original_sh_dc.resize(gaussians.size());
        for (uint32_t i = 0; i < gaussians.size(); i++) {
            bake_info.original_sh_dc[i] = gaussians[i].sh_dc;
        }
    }

    // Apply color grading to each Gaussian's DC coefficients
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        Gaussian &g = gaussians[i];

        // Extract base color from SH DC coefficients
        Color base_color = g.sh_dc;

        // Apply color grading
        Color graded_color = apply_color_grading_cpu(base_color, p_grading);

        // Write back to SH DC coefficients (preserve alpha)
        g.sh_dc.r = graded_color.r;
        g.sh_dc.g = graded_color.g;
        g.sh_dc.b = graded_color.b;
        // sh_dc.a remains unchanged
    }

    bake_info.is_baked = true;
    bake_info.applied_grading = p_grading;

    // Mark dirty to trigger GPU re-upload
    _on_gaussian_storage_changed_locked();

    return OK;
}

void GaussianData::restore_original_colors() {
    RWLockWrite lock(data_rwlock);
    if (!bake_info.is_baked) {
        return;  // Nothing to restore
    }

    ERR_FAIL_COND(bake_info.original_sh_dc.size() != gaussians.size());

    // Restore original SH DC coefficients
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        gaussians[i].sh_dc = bake_info.original_sh_dc[i];
    }

    bake_info.is_baked = false;
    bake_info.applied_grading.unref();

    // Mark dirty to trigger GPU re-upload
    _on_gaussian_storage_changed_locked();
}

Color GaussianData::apply_color_grading_cpu(const Color &p_color, const Ref<ColorGradingResource> &p_grading) {
    Color result = p_color;

    if (!p_grading->get_enabled()) {
        return result;
    }

    // 1. Exposure
    float exposure_mult = Math::pow(2.0f, p_grading->get_exposure());
    result.r *= exposure_mult;
    result.g *= exposure_mult;
    result.b *= exposure_mult;

    // 2. Temperature & Tint
    float temp_factor = p_grading->get_temperature() * 0.01f;
    float tint_factor = p_grading->get_tint() * 0.01f;

    result.r += temp_factor * 0.5f;
    result.b -= temp_factor * 0.5f;
    result.g += tint_factor * 0.5f;
    result.r -= tint_factor * 0.25f;
    result.b -= tint_factor * 0.25f;

    result.r = MAX(result.r, 0.0f);
    result.g = MAX(result.g, 0.0f);
    result.b = MAX(result.b, 0.0f);

    // 3. Contrast
    result.r = (result.r - 0.5f) * p_grading->get_contrast() + 0.5f;
    result.g = (result.g - 0.5f) * p_grading->get_contrast() + 0.5f;
    result.b = (result.b - 0.5f) * p_grading->get_contrast() + 0.5f;

    // 4. Saturation & Hue shift (RGB -> HSV -> adjust -> RGB)
    float h = result.get_h();
    float s = result.get_s();
    float v = result.get_v();

    // Adjust saturation
    s *= p_grading->get_saturation();
    s = CLAMP(s, 0.0f, 1.0f);

    // Adjust hue
    h += (p_grading->get_hue_shift() / 360.0f);
    h = Math::fposmod(h, 1.0f);  // Wrap around

    result = Color::from_hsv(h, s, v);

    // Final clamp
    result.r = CLAMP(result.r, 0.0f, 65504.0f);
    result.g = CLAMP(result.g, 0.0f, 65504.0f);
    result.b = CLAMP(result.b, 0.0f, 65504.0f);

    return result;
}
