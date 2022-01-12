//TODO: only needed by GLES_OVER_GL

uint float2half(uint f) {
	return ((f >> uint(16)) & uint(0x8000)) |
			((((f & uint(0x7f800000)) - uint(0x38000000)) >> uint(13)) & uint(0x7c00)) |
			((f >> uint(13)) & uint(0x03ff));
}

uint half2float(uint h) {
	return ((h & uint(0x8000)) << uint(16)) | (((h & uint(0x7c00)) + uint(0x1c000)) << uint(13)) | ((h & uint(0x03ff)) << uint(13));
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

uint packUnorm4x8(vec4 v) {
	uvec4 uv = uvec4(round(clamp(v, vec4(0.0), vec4(1.0)) * 255.0));
	return uv.x | uv.y << uint(8) | uv.z << uint(16) | uv.w << uint(24);
}

vec4 unpackUnorm4x8(uint p) {
	return vec4(float(p & uint(0xffff)), float((p >> uint(8)) & uint(0xffff)), float((p >> uint(16)) & uint(0xffff)), float(p >> uint(24))) * 0.00392156862; // 1.0 / 255.0
}

uint packSnorm4x8(vec4 v) {
	uvec4 uv = uvec4(round(clamp(v, vec4(-1.0), vec4(1.0)) * 127.0) + 127.0);
	return uv.x | uv.y << uint(8) | uv.z << uint(16) | uv.w << uint(24);
}

vec4 unpackSnorm4x8(uint p) {
	vec4 v = vec4(float(p & uint(0xffff)), float((p >> uint(8)) & uint(0xffff)), float((p >> uint(16)) & uint(0xffff)), float(p >> uint(24)));
	return clamp((v - vec4(127.0)) * vec4(0.00787401574), vec4(-1.0), vec4(1.0));
}
