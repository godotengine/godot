// https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps
float mx_variance_shadow_occlusion(vec2 moments, float fragmentDepth)
{
    const float MIN_VARIANCE = 0.00001;

    // One-tailed inequality valid if fragmentDepth > moments.x.
    float p = (fragmentDepth <= moments.x) ? 1.0 : 0.0;

    // Compute variance.
    float variance = moments.y - mx_square(moments.x);
    variance = max(variance, MIN_VARIANCE);

    // Compute probabilistic upper bound.
    float d = fragmentDepth - moments.x;
    float pMax = variance / (variance + mx_square(d));
    return max(p, pMax);
}

vec2 mx_compute_depth_moments()
{
    float depth = gl_FragCoord.z;
    return vec2(depth, mx_square(depth));
}
