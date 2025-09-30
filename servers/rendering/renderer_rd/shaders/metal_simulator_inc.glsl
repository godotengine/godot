#ifdef SIMULATE_CUBEMAP_ARRAYS
#define textureCubeArrayFix texture2DArray
#define samplerCubeArrayFix sampler2DArray
#else
#define textureCubeArrayFix textureCubeArray
#define samplerCubeArrayFix samplerCubeArray
#endif

#ifdef SIMULATE_CUBEMAP_ARRAYS
vec3 simulateCoords(vec4 p_coords) {
	vec3 dir = p_coords.xyz;
	float index = p_coords.w;
	vec3 absDir = abs(dir);
	float axis;
	vec2 uv;
	int face;
	if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
		axis = absDir.x;
		uv = vec2(dir.z, dir.y) / axis;
		face = dir.x > 0.0 ? 0 : 1;
		if (dir.x < 0.0) {
			uv.x = -uv.x;
		}
	} else if (absDir.y >= absDir.z) {
		axis = absDir.y;
		uv = vec2(dir.x, dir.z) / axis;
		face = dir.y > 0.0 ? 2 : 3;
		if (dir.y < 0.0) {
			uv.y = -uv.y;
		}
	} else {
		axis = absDir.z;
		uv = vec2(dir.x, dir.y) / axis;
		face = dir.z > 0.0 ? 4 : 5;
		if (dir.z < 0.0) {
			uv.x = -uv.x;
		}
	}
	uv = 0.5 * (uv + 1.0);
	float layer = index * 6.0 + float(face);
	return vec3(uv, layer);
}
#else
vec4 simulateCoords(vec4 p_coords) {
	return p_coords;
}
#endif

ivec3 simulateSize(ivec3 p_size) {
#ifdef SIMULATE_CUBEMAP_ARRAYS
	p_size.z /= 6;
#endif
	return p_size;
}

#ifdef SIMULATE_CUBEMAP_ARRAYS
struct Grad {
	vec2 dx;
	vec2 dy;
};

Grad simulateGrad(vec4 p_coords, vec3 p_dPdx, vec3 p_dPdy) {
	vec3 dir = p_coords.xyz;
	vec3 absDir = abs(dir);
	vec2 a, adx, ady;
	float b, bdx, bdy;
	if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
		a = vec2(dir.z, dir.y);
		adx = vec2(p_dPdx.z, p_dPdx.y);
		ady = vec2(p_dPdy.z, p_dPdy.y);
		b = dir.x;
		bdx = p_dPdx.x;
		if (dir.x < 0.0) {
			a.x *= -1.0;
			adx.x *= -1.0;
			ady.x *= -1.0;
		}
	} else if (absDir.y >= absDir.z) {
		a = vec2(dir.x, dir.z);
		adx = vec2(p_dPdx.x, p_dPdx.z);
		ady = vec2(p_dPdy.x, p_dPdy.z);
		b = dir.y;
		bdx = p_dPdx.y;
		bdy = p_dPdy.y;
		if (dir.y < 0.0) {
			a.y *= -1.0;
			adx.y *= -1.0;
			ady.y *= -1.0;
		}
	} else {
		a = vec2(dir.x, dir.y);
		adx = vec2(p_dPdx.x, p_dPdx.y);
		ady = vec2(p_dPdy.x, p_dPdy.y);
		b = dir.z;
		bdx = p_dPdx.z;
		bdy = p_dPdy.z;
		if (dir.z < 0.0) {
			a.x *= -1.0;
			adx.x *= -1.0;
			ady.x *= -1.0;
		}
	}
	vec2 dUVdx = (adx * b - a * bdx) / (b * b);
	vec2 dUVdy = (ady * b - a * bdy) / (b * b);
	return Grad(0.5 * dUVdx, 0.5 * dUVdy);
}
#else
struct Grad {
	vec3 dx;
	vec3 dy;
};

Grad simulateGrad(vec4 p_coords, vec3 p_dPdx, vec3 p_dPdy) {
	return Grad(p_dPdx, p_dPdy);
}
#endif

ivec3 textureSizeFix(samplerCubeArrayFix p_sampler, int p_lod) {
	return simulateSize(textureSize(p_sampler, p_lod));
}

ivec3 textureSizeFix(textureCubeArrayFix p_texture, sampler p_sampler, int p_lod) {
	return simulateSize(textureSize(samplerCubeArrayFix(p_texture, p_sampler), p_lod));
}

vec4 textureFix(samplerCubeArrayFix p_sampler, vec4 p_coords) {
	return texture(p_sampler, simulateCoords(p_coords));
}

vec4 textureFix(textureCubeArrayFix p_texture, sampler p_sampler, vec4 p_coords) {
	return texture(samplerCubeArrayFix(p_texture, p_sampler), simulateCoords(p_coords));
}

vec4 textureLodFix(samplerCubeArrayFix p_sampler, vec4 p_coords, float p_lod) {
	return textureLod(p_sampler, simulateCoords(p_coords), p_lod);
}

vec4 textureLodFix(textureCubeArrayFix p_texture, sampler p_sampler, vec4 p_coords, float p_lod) {
	return textureLod(samplerCubeArrayFix(p_texture, p_sampler), simulateCoords(p_coords), p_lod);
}

vec4 textureGradFix(samplerCubeArrayFix p_sampler, vec4 p_coords, vec3 p_dPdx, vec3 p_dPdy) {
	Grad grad = simulateGrad(p_coords, p_dPdx, p_dPdy);
	return textureGrad(p_sampler, simulateCoords(p_coords), grad.dx, grad.dy);
}

vec4 textureGradFix(textureCubeArrayFix p_texture, sampler p_sampler, vec4 p_coords, vec3 p_dPdx, vec3 p_dPdy) {
	Grad grad = simulateGrad(p_coords, p_dPdx, p_dPdy);
	return textureGrad(samplerCubeArrayFix(p_texture, p_sampler), simulateCoords(p_coords), grad.dx, grad.dy);
}
