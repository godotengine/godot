// S4AO (Stupid Simple Screen Space Ambient Occlusion) - Jonathan Dummer (O1S)

// The sample_width should be even, else the midpoint is at UV.
// Takes sample_width^2 samples in a grid, with the corners notched.
#if defined(USE_SSAO_LOW)
const int sample_width = 2;
#elif defined(USE_SSAO_HIGH)
const int sample_width = 6;
#else
const int sample_width = 4;
#endif
const int notch_01 = int(sample_width > 3); // Set to 1 to skip the corner samples, 0 to include them.
const float sample_mid = (float(sample_width) - 1.0) * 0.50001; // Can't be exactly 0.5 in case sample_width is odd.
#if defined(USE_SSAO_LOW)
const float inv_half_width = 1.0 / sample_mid; // The 2x2 sampling looks wider as all samples are at radius.
#else
const float inv_half_width = 1.7 / sample_mid; // Bake in the 1.7 scale for the random rotation.
#endif
const float average_samples = 1.0 / float(sample_width * sample_width - 4 * notch_01); //  1 / number_of_samples
const float ssao_falloff_frac = 0.25;
// Perform the SSAO.
float s4ao(vec2 UV) {
#ifdef USE_MULTIVIEW
	float depth = texture(depth_buffer_array, vec3(UV, view)).r;
#else
	float depth = texture(depth_buffer, UV).r;
#endif
	float radius = max(1e-4f, depth * ssao_radius_frac);
	float inv_falloff = 1.0f / max(1e-4f, depth * ssao_falloff_frac);
	// Random 2D rotation per pixel (+/-45 deg, with 0 having a lower probability).
	// The random cosine vector is vec2( 0.5, -0.5 to +0.5 ) and *1.7 makes the average length ~ 1.
	vec2 rcos = (inv_half_width * radius) * vec2(0.5f, fract(dot(UV, ssao_prn_UV)) - 0.5f);
	vec2 rsin = rcos.yx * vec2(-1, 1); // Perpendicular to the random cosine vector.
	// Grab the samples and determine the occlusion.
	float occlusion = 0.0f;
	vec2 base_duv = -sample_mid * rsin;
	for (int j = sample_width; --j >= 0;) {
#if defined(USE_SSAO_LOW)
		// Low quality uses 2x2 samples, no notching.
		vec2 duv = -sample_mid * rcos + base_duv;
		for (int i = sample_width; --i >= 0;) {
#else
		//	Will uses 4x4 or 6x6 samples, with the corners notched out.
		int o = /*notch_01 &*/ int((j <= 0) || (j >= (sample_width - 1))); // Notch corners of the grid.
		vec2 duv = (float(o) - sample_mid) * rcos + base_duv;
		for (int i = sample_width - o - o; --i >= 0;) {
#endif
#ifdef USE_MULTIVIEW
			float dz = texture(depth_buffer_array, vec3(UV + duv, view)).r - depth;
#else
			float dz = texture(depth_buffer, UV + duv).r - depth;
#endif
			float validity = smoothstep(1.0f, 0.0f, dz * inv_falloff);
			occlusion += normalize(vec3(duv, dz)).z * validity; // How 'directly overhead' is it?
			duv += rcos; // March along the rcos direction with i.
		}
		base_duv += rsin; // March along the rsin direction with j.
	}
	// Adjust the occlusion for intensity, and # samples.
	occlusion *= ssao_intensity * average_samples;
	occlusion = clamp(1.0f - occlusion, 0.0f, 1.0f);
	return occlusion * occlusion;
}
