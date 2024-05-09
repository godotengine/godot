#include "mx_microfacet_specular.glsl"

// Construct an orthonormal basis from a unit vector.
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
mat3 mx_orthonormal_basis(vec3 N)
{
    float sign = (N.z < 0.0) ? -1.0 : 1.0;
    float a = -1.0 / (sign + N.z);
    float b = N.x * N.y * a;
    vec3 X = vec3(1.0 + sign * N.x * N.x * a, sign * b, -sign * N.x);
    vec3 Y = vec3(b, sign + N.y * N.y * a, -N.y);
    return mat3(X, Y, N);
}

// Return the alpha associated with the given mip level in a prefiltered environment.
float mx_latlong_lod_to_alpha(float lod)
{
    float lodBias = lod / float($envRadianceMips - 1);
    return (lodBias < 0.5) ? mx_square(lodBias) : 2.0 * (lodBias - 0.375);
}

// The inverse of mx_latlong_projection.
vec3 mx_latlong_map_projection_inverse(vec2 uv)
{
    float latitude = (uv.y - 0.5) * M_PI;
    float longitude = (uv.x - 0.5) * M_PI * 2.0;

    float x = -cos(latitude) * sin(longitude);
    float y = -sin(latitude);
    float z = cos(latitude) * cos(longitude);

    return vec3(x, y, z);
}

vec3 mx_generate_prefilter_env()
{
    // The tangent view vector is aligned with the normal.
    vec3 V = vec3(0.0, 0.0, 1.0);
    float NdotV = 1.0;

    // Compute derived properties.
    vec2 uv = gl_FragCoord.xy * pow(2.0, $envPrefilterMip) / vec2(textureSize($envRadiance, 0));
    vec3 worldN = mx_latlong_map_projection_inverse(uv);
    mat3 tangentToWorld = mx_orthonormal_basis(worldN);
    float alpha = mx_latlong_lod_to_alpha(float($envPrefilterMip));
    float G1V = mx_ggx_smith_G1(NdotV, alpha);

    // Integrate the LD term for the given environment and alpha.
    vec3 radiance = vec3(0.0, 0.0, 0.0);
    float weight = 0.0;
    int envRadianceSamples = 1024;
    for (int i = 0; i < envRadianceSamples; i++)
    {
        vec2 Xi = mx_spherical_fibonacci(i, envRadianceSamples);

        // Compute the half vector and incoming light direction.
        vec3 H = mx_ggx_importance_sample_VNDF(Xi, V, vec2(alpha));
        vec3 L = -V + 2.0 * H.z * H;

        // Compute dot products for this sample.
        float NdotL = clamp(L.z, M_FLOAT_EPS, 1.0);

        // Compute the geometric term.
        float G = mx_ggx_smith_G2(NdotL, NdotV, alpha);

        // Sample the environment light from the given direction.
        vec3 Lw = tangentToWorld * L;
        float pdf = mx_ggx_NDF(H, vec2(alpha)) * G1V / (4.0 * NdotV);
        float lod = mx_latlong_compute_lod(Lw, pdf, float($envRadianceMips - 1), envRadianceSamples);
        vec3 sampleColor = mx_latlong_map_lookup(Lw, $envMatrix, lod, $envRadiance);

        // Add the radiance contribution of this sample.
        radiance += G * sampleColor;
        weight += G;
    }

    return radiance / weight;
}
