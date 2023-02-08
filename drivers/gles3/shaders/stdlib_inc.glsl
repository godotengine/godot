
#ifdef USE_GLES_OVER_GL
// Floating point pack/unpack functions are part of the GLSL ES 300 specification used by web and mobile.
uint float2half(uint f) {
	uint b = f + uint(0x00001000); // round-to-nearest-even: add last bit after truncated mantissa
    uint e = (b & uint(0x7f800000))>>uint(23); // exponent
    uint m = b & uint(0x007fffff); // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    return (b & uint(0x80000000))>>uint(16) | uint(e > uint(112)) * ((((e-uint(112))<<uint(10))& uint(0x7C00))|m>>uint(13)) | (uint(e<uint(113))&uint(e>uint(101)))*((((uint(0x007ff000)+m)>>(uint(125)-e))+uint(1))>>uint(1)) | uint(e>uint(143))*uint(0x7fff); // sign : normalized : denormalized : saturate
}

uint half2float(uint h) {
	uint e = (h & uint(0x7c00)) >> uint(10); // exponent
    uint m = (h & uint(0x03ff)) << uint(13); // mantissa
    uint v = uint(float(m)) >> uint(23); // evil log2 bit hack to count leading zeros in denormalized format
    return (h & uint(0x8000)) << uint(16) | uint(e != uint(0)) * ((e + uint(112)) << uint(23) | m) | (uint(e==uint(0)) & uint(m!=uint(0))) * ((v-uint(37)) << uint(23) | ((m<<(uint(150)-v)) & uint(0x007fe000))); // sign : normalized : denormalized
}

uint packHalf2x16(vec2 v) {
	return float2half(floatBitsToUint(v.x)) | float2half(floatBitsToUint(v.y)) << uint(16);
}

vec2 unpackHalf2x16(uint v) {
	return vec2(uintBitsToFloat(half2float(v & uint(0xffff))),
			uintBitsToFloat(half2float(v >> uint(16))));
}

uint packUnorm2x16(vec2 v) {
	uvec2 uv = uvec2(round(clamp(v, vec2(0.0), vec2(1.0)) * 65535.0));
	return uv.x | uv.y << uint(16);
}

vec2 unpackUnorm2x16(uint p) {
	return vec2(float(p & uint(0xffff)), float(p >> uint(16))) * 0.000015259021; // 1.0 / 65535.0 optimization
}

uint packSnorm2x16(vec2 v) {
	uvec2 uv = uvec2(round(clamp(v, vec2(-1.0), vec2(1.0)) * 32767.0) + 32767.0);
	return uv.x | uv.y << uint(16);
}

vec2 unpackSnorm2x16(uint p) {
	vec2 v = vec2(float(p & uint(0xffff)), float(p >> uint(16)));
	return clamp((v - 32767.0) * vec2(0.00003051851), vec2(-1.0), vec2(1.0));
}

#endif

// Compatibility renames. These are exposed with the "godot_" prefix
// to work around an Adreno bug which was exposing these ES310 functions
// in ES300 shaders. Internally, we must use the "godot_" prefix, but user shaders
// will be mapped automatically.
uint godot_packUnorm4x8(vec4 v) {
	uvec4 uv = uvec4(round(clamp(v, vec4(0.0), vec4(1.0)) * 255.0));
	return uv.x | (uv.y << uint(8)) | (uv.z << uint(16)) | (uv.w << uint(24));
}

vec4 godot_unpackUnorm4x8(uint p) {
	return vec4(float(p & uint(0xff)), float((p >> uint(8)) & uint(0xff)), float((p >> uint(16)) & uint(0xff)), float(p >> uint(24))) * 0.00392156862; // 1.0 / 255.0
}

uint godot_packSnorm4x8(vec4 v) {
	uvec4 uv = uvec4(round(clamp(v, vec4(-1.0), vec4(1.0)) * 127.0) + 127.0);
	return uv.x | uv.y << uint(8) | uv.z << uint(16) | uv.w << uint(24);
}

vec4 godot_unpackSnorm4x8(uint p) {
	vec4 v = vec4(float(p & uint(0xff)), float((p >> uint(8)) & uint(0xff)), float((p >> uint(16)) & uint(0xff)), float(p >> uint(24)));
	return clamp((v - vec4(127.0)) * vec4(0.00787401574), vec4(-1.0), vec4(1.0));
}

#define packUnorm4x8 godot_packUnorm4x8
#define unpackUnorm4x8 godot_unpackUnorm4x8
#define packSnorm4x8 godot_packSnorm4x8
#define unpackSnorm4x8 godot_unpackSnorm4x8
