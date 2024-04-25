#include "mx_microfacet_sheen.glsl"
#include "mx_microfacet_specular.glsl"

vec3 mx_generate_dir_albedo_table()
{
    vec2 uv = gl_FragCoord.xy / $albedoTableSize;
    vec2 ggxDirAlbedo = mx_ggx_dir_albedo(uv.x, uv.y, vec3(1, 0, 0), vec3(0, 1, 0)).xy;
    float sheenDirAlbedo = mx_imageworks_sheen_dir_albedo(uv.x, uv.y);
    return vec3(ggxDirAlbedo, sheenDirAlbedo);
}
