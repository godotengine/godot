// Copyright © 2024 Cory Petkovsek, Roope Palmroos, and Contributors.

R"(

//INSERT: WORLD_NOISE1
// World Noise

uniform sampler2D _region_blend_map : hint_default_black, filter_linear, repeat_disable;
uniform int world_noise_max_octaves : hint_range(0, 15) = 4;
uniform int world_noise_min_octaves : hint_range(0, 15) = 2;
uniform float world_noise_lod_distance : hint_range(0, 40000, 1) = 7500.;
uniform float world_noise_scale : hint_range(0.25, 20, 0.01) = 5.0;
uniform float world_noise_height : hint_range(0, 1000, 0.1) = 64.0;
uniform vec3 world_noise_offset = vec3(0.0);
uniform float world_noise_blend_near : hint_range(0, .95, 0.01) = 0.5;
uniform float world_noise_blend_far : hint_range(.05, 1, 0.01) = 1.0;

float hashf(float f) {
	return fract(sin(f) * 1e4);
}

float hashv2(vec2 v) {
	return fract(1e4 * sin(17.0 * v.x + v.y * 0.1) * (0.1 + abs(sin(v.y * 13.0 + v.x))));
}

// https://iquilezles.org/articles/morenoise/
vec3 noise2D(vec2 x) {
    vec2 f = fract(x);
    // Quintic Hermine Curve.  Similar to SmoothStep()
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);

    vec2 p = floor(x);

	// Four corners in 2D of a tile
	float a = hashv2( p+vec2(0,0) );
    float b = hashv2( p+vec2(1,0) );
    float c = hashv2( p+vec2(0,1) );
    float d = hashv2( p+vec2(1,1) );

    // Mix 4 corner percentages
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   a - b - c + d;
    return vec3( k0 + k1 * u.x + k2 * u.y + k3 * u.x * u.y,
                du * ( vec2(k1, k2) + k3 * u.yx) );
}

float world_noise(vec2 p) {
    float a = 0.0;
    float b = 1.0;
    vec2  d = vec2(0.0);

    int octaves = int( clamp(
	float(world_noise_max_octaves) - floor(v_vertex_xz_dist/(world_noise_lod_distance)),
    float(world_noise_min_octaves), float(world_noise_max_octaves)) );
	
    for( int i=0; i < octaves; i++ ) {
        vec3 n = noise2D(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d,d));
        b *= 0.5;
        p = mat2( vec2(0.8, -0.6), vec2(0.6, 0.8) ) * p * 2.0;
    }
    return a;
}

// World Noise end

//INSERT: WORLD_NOISE2
	// World Noise
   	if (_background_mode == 2u) {
	    float weight = texture(_region_blend_map, (uv / float(_region_map_size)) + 0.5).r;
	    float rmap_half_size = float(_region_map_size) * .5;
	    if (abs(uv.x) > rmap_half_size + .5 || abs(uv.y) > rmap_half_size + .5) {
		    weight = 0.;
	    } else {
		    if (abs(uv.x) > rmap_half_size - .5) {
			    weight = mix(weight, 0., abs(uv.x) - (rmap_half_size-.5));
		    }
		    if (abs(uv.y) > rmap_half_size - .5) {
			    weight = mix(weight, 0., abs(uv.y) - (rmap_half_size-.5));
		    }
	    }
	    height = mix(height, world_noise((uv + world_noise_offset.xz) * world_noise_scale * .1) *
            world_noise_height * 10. + world_noise_offset.y * 100.,
		    clamp(smoothstep(world_noise_blend_near, world_noise_blend_far, 1.0 - weight), 0.0, 1.0));
    }
)"