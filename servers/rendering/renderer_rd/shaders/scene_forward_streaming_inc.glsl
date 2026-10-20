// Helper functions for texture streaming feedback.
// Used by the Forward+ and Mobile renderers.

// Reference texture size used for LOD normalization: 4096x4096.
#define STREAMING_LOD_REFERENCE_LOG2 12.0

// Returns the squared pixel footprint in UV space.
float streaming_footprint_sq(vec2 uv_ddx, vec2 uv_ddy) {
	float uv_ddx_sq = dot(uv_ddx, uv_ddx);
	float uv_ddy_sq = dot(uv_ddy, uv_ddy);

	// Limit anisotropic filtering to 16x.
	const float MAX_ANISOTROPY = 16.0;

	return max(
			min(uv_ddx_sq, uv_ddy_sq),
			max(uv_ddx_sq, uv_ddy_sq) / (MAX_ANISOTROPY * MAX_ANISOTROPY));
}

// Undoes the reference offset on a shader-supplied STREAMING_LOD, so both feedback paths
// hand the same squared UV footprint to the encoder.
float streaming_footprint_sq_from_lod(float lod) {
	return exp2(2.0 * (clamp(lod, -32.0, 32.0) - STREAMING_LOD_REFERENCE_LOG2));
}
