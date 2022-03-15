/* clang-format off */
[vertex]

/*
from VisualServer:

ARRAY_VERTEX=0,
ARRAY_NORMAL=1,
ARRAY_TANGENT=2,
ARRAY_COLOR=3,
ARRAY_TEX_UV=4,
ARRAY_TEX_UV2=5,
ARRAY_BONES=6,
ARRAY_WEIGHTS=7,
ARRAY_INDEX=8,
*/

#ifdef USE_2D_VERTEX
#define VFORMAT vec2
#else
#define VFORMAT vec3
#endif

/* INPUT ATTRIBS */

layout(location = 0) in highp VFORMAT vertex_attrib;
/* clang-format on */
#ifdef ENABLE_OCTAHEDRAL_COMPRESSION
layout(location = 1) in vec4 normal_tangent_attrib;
#else
layout(location = 1) in vec3 normal_attrib;
#endif

#ifdef ENABLE_TANGENT
#ifdef ENABLE_OCTAHEDRAL_COMPRESSION
// packed into normal_attrib zw component
#else
layout(location = 2) in vec4 tangent_attrib;
#endif
#endif

#ifdef ENABLE_COLOR
layout(location = 3) in vec4 color_attrib;
#endif

#ifdef ENABLE_UV
layout(location = 4) in vec2 uv_attrib;
#endif

#ifdef ENABLE_UV2
layout(location = 5) in vec2 uv2_attrib;
#endif

#ifdef ENABLE_SKELETON
layout(location = 6) in ivec4 bone_attrib;
layout(location = 7) in vec4 weight_attrib;
#endif

/* BLEND ATTRIBS */

#ifdef ENABLE_BLEND

layout(location = 8) in highp VFORMAT vertex_attrib_blend;
layout(location = 9) in vec3 normal_attrib_blend;

#ifdef ENABLE_TANGENT
layout(location = 10) in vec4 tangent_attrib_blend;
#endif

#ifdef ENABLE_COLOR
layout(location = 11) in vec4 color_attrib_blend;
#endif

#ifdef ENABLE_UV
layout(location = 12) in vec2 uv_attrib_blend;
#endif

#ifdef ENABLE_UV2
layout(location = 13) in vec2 uv2_attrib_blend;
#endif

#ifdef ENABLE_SKELETON
layout(location = 14) in ivec4 bone_attrib_blend;
layout(location = 15) in vec4 weight_attrib_blend;
#endif

#endif

/* OUTPUTS */

out VFORMAT vertex_out; //tfb:

#ifdef ENABLE_NORMAL
out vec3 normal_out; //tfb:ENABLE_NORMAL
#endif

#ifdef ENABLE_TANGENT
out vec4 tangent_out; //tfb:ENABLE_TANGENT
#endif

#ifdef ENABLE_COLOR
out vec4 color_out; //tfb:ENABLE_COLOR
#endif

#ifdef ENABLE_UV
out vec2 uv_out; //tfb:ENABLE_UV
#endif

#ifdef ENABLE_UV2
out vec2 uv2_out; //tfb:ENABLE_UV2
#endif

#ifdef ENABLE_SKELETON
out ivec4 bone_out; //tfb:ENABLE_SKELETON
out vec4 weight_out; //tfb:ENABLE_SKELETON
#endif

uniform float blend_amount;

#ifdef ENABLE_OCTAHEDRAL_COMPRESSION
vec3 oct_to_vec3(vec2 e) {
	vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
	float t = max(-v.z, 0.0);
	v.xy += t * -sign(v.xy);
	return normalize(v);
}
#endif

void main() {
#ifdef ENABLE_BLEND

	vertex_out = vertex_attrib_blend + vertex_attrib * blend_amount;

#ifdef ENABLE_NORMAL
#ifdef ENABLE_OCTAHEDRAL_COMPRESSION
	normal_out = normal_attrib_blend + oct_to_vec3(normal_tangent_attrib.xy) * blend_amount;
#else
	normal_out = normal_attrib_blend + normal_attrib * blend_amount;
#endif
#endif

#ifdef ENABLE_TANGENT
#ifdef ENABLE_OCTAHEDRAL_COMPRESSION
	tangent_out.xyz = tangent_attrib_blend.xyz + oct_to_vec3(vec2(normal_tangent_attrib.z, abs(normal_tangent_attrib.w) * 2.0 - 1.0)) * blend_amount;
	tangent_out.w = sign(tangent_attrib_blend.w);
#else
	tangent_out.xyz = tangent_attrib_blend.xyz + tangent_attrib.xyz * blend_amount;
	tangent_out.w = tangent_attrib_blend.w; //just copy, no point in blending his
#endif
#endif

#ifdef ENABLE_COLOR

	color_out = color_attrib_blend + color_attrib * blend_amount;
#endif

#ifdef ENABLE_UV

	uv_out = uv_attrib_blend + uv_attrib * blend_amount;
#endif

#ifdef ENABLE_UV2

	uv2_out = uv2_attrib_blend + uv2_attrib * blend_amount;
#endif

#ifdef ENABLE_SKELETON

	bone_out = bone_attrib_blend;
	weight_out = weight_attrib_blend + weight_attrib * blend_amount;
#endif

#else //ENABLE_BLEND

	vertex_out = vertex_attrib * blend_amount;

#ifdef ENABLE_NORMAL
#ifdef ENABLE_OCTAHEDRAL_COMPRESSION
	normal_out = oct_to_vec3(normal_tangent_attrib.xy) * blend_amount;
#else
	normal_out = normal_attrib * blend_amount;
#endif
#endif

#ifdef ENABLE_TANGENT
#ifdef ENABLE_OCTAHEDRAL_COMPRESSION
	tangent_out.xyz = oct_to_vec3(vec2(normal_tangent_attrib.z, abs(normal_tangent_attrib.w) * 2.0 - 1.0)) * blend_amount;
	tangent_out.w = sign(normal_tangent_attrib.w);
#else
	tangent_out.xyz = tangent_attrib.xyz * blend_amount;
	tangent_out.w = tangent_attrib.w; //just copy, no point in blending his
#endif
#endif

#ifdef ENABLE_COLOR

	color_out = color_attrib * blend_amount;
#endif

#ifdef ENABLE_UV

	uv_out = uv_attrib * blend_amount;
#endif

#ifdef ENABLE_UV2

	uv2_out = uv2_attrib * blend_amount;
#endif

#ifdef ENABLE_SKELETON

	bone_out = bone_attrib;
	weight_out = weight_attrib * blend_amount;
#endif

#endif
	gl_Position = vec4(0.0);
}

/* clang-format off */
[fragment]

void main() {

}
/* clang-format on */
