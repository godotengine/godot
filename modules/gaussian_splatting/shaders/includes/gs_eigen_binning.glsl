// gs_eigen_binning.glsl — Eigenvalue decomposition and opacity-aware bounds for tile binning.
//
// Requires: GAUSSIAN_EPSILON, MIN_VARIANCE constants to be defined before inclusion.

#ifndef GS_EIGEN_BINNING_GLSL_INCLUDED
#define GS_EIGEN_BINNING_GLSL_INCLUDED

struct EigenInfo {
    vec2 axis0;
    vec2 axis1;
    float radius0;
    float radius1;
    float raw_min_radius;  // Pre-aspect-clamp minimum radius for subpixel culling (#797)
    bool clamped;
};

// ============================================================================
// Opacity-Aware Bounding (FlashGS Optimization)
// ============================================================================
// Computes the effective sigma multiplier based on opacity and visibility
// threshold.  Reduces tile-Gaussian pairs by ~94% for low-opacity splats.
//
// Reference: FlashGS — Efficient Gaussian Splatting with Adaptive Bounds
// ============================================================================

// Compute the sigma multiplier for opacity-aware bounds.
// Returns the effective number of sigmas to use based on opacity.
// For high opacity (close to 1.0), returns close to max_sigma (conservative).
// For low opacity, returns a smaller value (aggressive culling).
float compute_opacity_aware_sigma(float opacity, float visibility_threshold, float max_sigma) {
    if (opacity <= visibility_threshold) {
        return 0.0;
    }

    float ln_ratio = log(opacity / visibility_threshold);
    if (ln_ratio <= 0.0) {
        return 0.0;
    }

    // The effective sigma count is sqrt(2 * ln(alpha/tau))
    // For alpha=1.0, tau=0.01: sqrt(2 * ln(100)) = sqrt(9.21) = 3.03 sigmas
    // For alpha=0.5, tau=0.01: sqrt(2 * ln(50)) = sqrt(7.82) = 2.80 sigmas
    // For alpha=0.1, tau=0.01: sqrt(2 * ln(10)) = sqrt(4.61) = 2.15 sigmas
    // For alpha=0.05, tau=0.01: sqrt(2 * ln(5)) = sqrt(3.22) = 1.79 sigmas
    float effective_sigma = sqrt(2.0 * ln_ratio);

    // Clamp to max_sigma to ensure we don't exceed the original conservative bound
    return min(effective_sigma, max_sigma);
}

// Compute eigenvalues and eigenvectors for tile binning heuristics.
EigenInfo compute_eigen(mat2 cov) {
    EigenInfo info;

    float trace = cov[0][0] + cov[1][1];
    float det = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
    det = max(det, MIN_VARIANCE * MIN_VARIANCE);

    float disc = max(trace * trace * 0.25 - det, 0.0);
    float root = sqrt(disc);

    float lambda0_raw = trace * 0.5 + root;
    float lambda1_raw = trace * 0.5 - root;
    float lambda0_orig = max(lambda0_raw, MIN_VARIANCE);
    float lambda1_orig = max(lambda1_raw, MIN_VARIANCE);
    bool clamped = (lambda0_raw != lambda0_orig) || (lambda1_raw != lambda1_orig);

    // Skip splats whose projected covariance is astronomically large.
    // These produce blocky artifacts no matter what we try.
    // Return a "skip" signal by setting radius to 0 (caller checks for this).
    const float MAX_RENDERABLE_EIGENVALUE = 1e8;  // Very permissive - only skip extreme cases
    float max_orig_eigen = max(lambda0_orig, lambda1_orig);
    if (max_orig_eigen > MAX_RENDERABLE_EIGENVALUE) {
        info.radius0 = 0.0;
        info.radius1 = 0.0;
        info.clamped = true;
        return info;
    }

    // Compute eigenvector axes using original eigenvalues.
    vec2 axis0;
    if (abs(cov[0][1]) > GAUSSIAN_EPSILON) {
        axis0 = normalize(vec2(lambda0_orig - cov[1][1], cov[0][1]));
    } else {
        axis0 = vec2(1.0, 0.0);
    }
    vec2 axis1 = vec2(-axis0.y, axis0.x);

    // Capture raw minimum radius BEFORE aspect ratio clamping for subpixel culling (#797)
    // This is the true screen-space minor radius, not the artificially inflated one.
    float lambda_small_raw = min(lambda0_orig, lambda1_orig);
    float raw_min_radius = sqrt(lambda_small_raw);

    // Keep eigen computation focused on numeric stability.
    // Visual/quality aspect clamping is handled later in main() using params.max_conic_aspect.
    float lambda0 = lambda0_orig;
    float lambda1 = lambda1_orig;

    // Cap eigenvalues to prevent tiny conic values from exploding numerically.
    const float MAX_EIGENVALUE = 10000.0;
    float lambda0_capped = min(lambda0, MAX_EIGENVALUE);
    float lambda1_capped = min(lambda1, MAX_EIGENVALUE);
    if (lambda0_capped != lambda0 || lambda1_capped != lambda1) {
        clamped = true;
    }
    lambda0 = lambda0_capped;
    lambda1 = lambda1_capped;

    info.axis0 = axis0;
    info.axis1 = axis1;
    info.radius0 = sqrt(lambda0);
    info.radius1 = sqrt(lambda1);
    info.raw_min_radius = raw_min_radius;
    info.clamped = clamped;
    return info;
}

#endif // GS_EIGEN_BINNING_GLSL_INCLUDED
