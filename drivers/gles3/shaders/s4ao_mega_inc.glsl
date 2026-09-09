// S4AO (Stupid Simple Screen Space Ambient Occlusion) - Jonathan Dummer (O1S)
// The mega version uses N concentric rings of samples.

#if defined(USE_SSAO_MEGA)
const int rings = 4; // Start with the outer ring.
const int samps[] = int[](24, 18, 12, 6); // ( 9, 6, 3 ) is a minimum, but I want better.
#else
const int rings = 3; // Start with the outer ring.
const int samps[] = int[](15, 10, 5, 1); // ( 9, 6, 3 ) is a minimum, but I want better.
#endif
const float average_samples = 1.0 / float(samps[0] + samps[1] * int(rings > 1) + samps[2] * int(rings > 2) + samps[3] * int(rings > 3));
const float ssao_falloff_frac = 0.25;
// Perform the SSAO.
float s4ao(vec2 UV) {
#ifdef USE_MULTIVIEW
	float depth = texture(depth_buffer_array, vec3(UV, view)).r;
#else
	float depth = texture(depth_buffer, UV).r;
#endif
	float inv_falloff = 1.0f / max(1e-4f, depth * ssao_falloff_frac);
	// Random 2D rotation per pixel (0..1 -> parabola approximating a 180 deg arc)
	float r01 = fract(dot(UV, ssao_prn_UV));
	vec2 rcos = vec2(r01 - 0.5f, 2.0f * (r01 - r01 * r01)) * (2.0f * depth * ssao_radius_frac); // 180 degrees.
	vec2 rsin = rcos.yx * vec2(-1, 1); // Perpendicular to the random cosine vector.
	// Grab the samples and determine the occlusion.
	float occlusion = 0.0f;
	float ring_shrink = 0.75f; // Shrink every ring.
	for (int r = 0; r < rings; ++r) {
		float dt = (6.283185307f) / float(samps[r]);
		float t = float(r & 1) * 0.5f * dt;
		for (int s = 0; s < samps[r]; ++s) {
			vec2 duv = cos(t) * rcos + sin(t) * rsin;
#ifdef USE_MULTIVIEW
			float dz = texture(depth_buffer_array, vec3(UV + duv, view)).r - depth;
#else
			float dz = texture(depth_buffer, UV + duv).r - depth;
#endif
			// How 'directly overhead' is it?  Factor in the falloff depth.
			occlusion += normalize(vec3(duv, dz)).z * smoothstep(1.0f, 0.0f, dz * inv_falloff);
			t += dt;
		}
		// The next ring will be smaller.
		rcos *= ring_shrink;
		rsin *= ring_shrink;
	}
	// Adjust the occlusion for intensity, and # samples.
	occlusion *= ssao_intensity * average_samples;
	occlusion = 1.0f - clamp(occlusion, 0.0f, 1.0f);
	return occlusion * occlusion;
}
