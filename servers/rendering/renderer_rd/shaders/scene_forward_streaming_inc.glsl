// Helper functions for texture streaming feedback.
// Used by the Forward+ and Mobile renderers.

// Reference texture size used for LOD normalization: 4096x4096.
#define STREAMING_LOD_REFERENCE_LOG2 12.0

// Returns the pixel footprint in UV space.
float streaming_footprint_sq(vec2 ddx, vec2 ddy) {
	float px_sq = dot(ddx, ddx);
	float py_sq = dot(ddy, ddy);

	// Limit anisotropic filtering to 16x.
	const float MAX_ANISOTROPY = 16.0;

	return max(
			min(px_sq, py_sq),
			max(px_sq, py_sq) / (MAX_ANISOTROPY * MAX_ANISOTROPY));
}

// Computes the streaming LOD from texture coordinate gradients.
float streaming_lod_grad(vec2 ddx, vec2 ddy) {
	// Convert footprint size into a mip level.
	return 0.5 * log2(max(streaming_footprint_sq(ddx, ddy), 1e-30)) + STREAMING_LOD_REFERENCE_LOG2;
}

// Returns the streaming LOD from UV gradients.
float streaming_lod_uv(vec2 uv) {
	return streaming_lod_grad(dFdx(uv), dFdy(uv));
}

// Computes streaming LOD for planar/triplanar mapping using the dominant projection axis.
float streaming_lod_planar(vec3 pos_ddx, vec3 pos_ddy, vec3 weights) {
	vec2 ddx, ddy;
	if (weights.x >= weights.y && weights.x >= weights.z) {
		ddx = pos_ddx.zy;
		ddy = pos_ddy.zy;
	} else if (weights.y >= weights.z) {
		ddx = pos_ddx.xz;
		ddy = pos_ddy.xz;
	} else {
		ddx = pos_ddx.xy;
		ddy = pos_ddy.xy;
	}
	return streaming_lod_grad(ddx, ddy);
}

// Converts a mip level to the 4096x4096 reference scale.
float streaming_lod_level(float level, float texture_size) {
	return level + STREAMING_LOD_REFERENCE_LOG2 - log2(max(texture_size, 1.0));
}
