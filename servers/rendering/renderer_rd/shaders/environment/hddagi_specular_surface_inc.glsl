ivec2 specular_footprint_begin(ivec2 pixel, ivec2 size, ivec2 screen_size) {
	return (pixel * screen_size + size - 1) / size;
}

// Half GI footprints can span three texels on an odd-sized viewport.
float specular_pack_view_z(float view_z, bool dynamic, ivec2 pixel, ivec2 size, ivec2 screen_pixel, ivec2 screen_size) {
	if (any(notEqual(size, screen_size))) {
		uvec2 offset = uvec2(screen_pixel - specular_footprint_begin(pixel, size, screen_size));
		view_z = uintBitsToFloat((floatBitsToUint(view_z) & ~15u) | (offset.y << 2u) | offset.x);
	}
	return dynamic ? -view_z : view_z;
}

ivec2 specular_surface_pixel(ivec2 pixel, ivec2 size, ivec2 screen_size, float packed_view_z) {
	if (all(equal(size, screen_size))) {
		return pixel;
	}
	uint bits = floatBitsToUint(packed_view_z);
	return specular_footprint_begin(pixel, size, screen_size) + ivec2(bits & 3u, (bits >> 2u) & 3u);
}
