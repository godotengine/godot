#if defined(USE_MULTIVIEW) && defined(has_VK_KHR_multiview)
#extension GL_EXT_multiview : enable
#endif
#ifdef USE_MULTIVIEW
#ifdef has_VK_KHR_multiview
#define ViewIndex gl_ViewIndex
#else // has_VK_KHR_multiview
// !BAS! This needs to become an input once we implement our fallback!
#define ViewIndex 0
#endif // has_VK_KHR_multiview
#else // USE_MULTIVIEW
// Set to zero, not supported in non stereo
#define ViewIndex 0
#endif //USE_MULTIVIEW

#ifdef USE_MULTIVIEW
#define MV_SCREEN_UV vec3(uv, ViewIndex)
#define MV_SCREEN_INT_UV ivec3(uv, ViewIndex)
#define MV_SCREEN_UV_VEC3 uv
#define MV_TEXSIZE_TO_IVEC3(sz) sz
#define MV_SCREEN_SAMPLER sampler2DArray
#define texture2DScreen texture2DArray
#else
#define MV_SCREEN_UV uv
#define MV_SCREEN_INT_UV uv
#define MV_SCREEN_UV_VEC3 uv.xy
#define MV_TEXSIZE_TO_IVEC3(sz) ivec3(sz, 1)
#define MV_SCREEN_SAMPLER sampler2D
#define texture2DScreen texture2D
#endif

#define sampler2DScreen(tex, samp) tex, samp

ivec3 textureSize(texture2DScreen screenTex, sampler samp, int lod) {
	return MV_TEXSIZE_TO_IVEC3(textureSize(MV_SCREEN_SAMPLER(screenTex, samp), lod));
}

vec4 texture(texture2DScreen screenTex, sampler samp, vec2 uv) {
	return texture(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV);
}
vec4 texture(texture2DScreen screenTex, sampler samp, vec3 uv) {
	return texture(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3);
}
// bias versions failed to compile:
// vec4 texture(texture2DScreen screenTex, sampler samp, vec2 uv, float bias) {
//     return texture(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV, bias);
// }
// vec4 texture(texture2DScreen screenTex, sampler samp, vec3 uv, float bias) {
//     return texture(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3, bias);
// }

vec4 textureLod(texture2DScreen screenTex, sampler samp, vec2 uv, float lod) {
	return textureLod(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV, lod);
}
vec4 textureLod(texture2DScreen screenTex, sampler samp, vec3 uv, float lod) {
	return textureLod(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3, lod);
}

vec4 textureGrad(texture2DScreen screenTex, sampler samp, vec2 uv, vec2 dPdX, vec2 dPdY) {
	return textureGrad(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV, dPdX, dPdY);
}
vec4 textureGrad(texture2DScreen screenTex, sampler samp, vec3 uv, vec2 dPdX, vec2 dPdY) {
	return textureGrad(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3, dPdX, dPdY);
}

vec4 texelFetch(texture2DScreen screenTex, sampler samp, ivec2 uv, int lod) {
	return texelFetch(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_INT_UV, lod);
}
vec4 texelFetch(texture2DScreen screenTex, sampler samp, ivec3 uv, int lod) {
	return texelFetch(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3, lod);
}

vec4 textureGather(texture2DScreen screenTex, sampler samp, vec2 uv) {
	return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV);
}
vec4 textureGather(texture2DScreen screenTex, sampler samp, vec2 uv, int comp) {
	// cannot forward comp argument: must be a constant expression.
	switch (comp) {
		case 0:
			return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV, 0);
		case 1:
			return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV, 1);
		case 2:
			return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV, 2);
		case 3:
		default:
			return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV, 3);
	}
}
vec4 textureGather(texture2DScreen screenTex, sampler samp, vec3 uv) {
	return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3);
}
vec4 textureGather(texture2DScreen screenTex, sampler samp, vec3 uv, int comp) {
	// cannot forward comp argument: must be a constant expression.
	switch (comp) {
		case 0:
			return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3, 0);
		case 1:
			return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3, 1);
		case 2:
			return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3, 2);
		case 3:
		default:
			return textureGather(MV_SCREEN_SAMPLER(screenTex, samp), MV_SCREEN_UV_VEC3, 3);
	}
}

#undef MV_SCREEN_UV
#undef MV_SCREEN_INT_UV
#undef MV_SCREEN_UV_VEC3
#undef MV_TEXSIZE_TO_IVEC3
#undef MV_SCREEN_SAMPLER
