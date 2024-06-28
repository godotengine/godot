// Copyright © 2023 Cory Petkovsek, Roope Palmroos, and Contributors.

// These DEBUG_* are special inserts that injected into the shader code before the last `}`
// which is assumed to belong to fragment()

R"(
//INSERT: DEBUG_CHECKERED
	// Show a checkered grid
	vec2 __p = uv * 1.0; // scale
	vec2 __ddx = dFdx(__p);
	vec2 __ddy = dFdy(__p);
	vec2 __w = max(abs(__ddx), abs(__ddy)) + 0.01;
	vec2 __i = 2.0 * (abs(fract((__p - 0.5 * __w) / 2.0) - 0.5) - abs(fract((__p + 0.5 * __w) / 2.0) - 0.5)) / __w;
	ALBEDO = vec3((0.5 - 0.5 * __i.x * __i.y) * 0.2 + 0.2);
	ROUGHNESS = 0.7;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_GREY
	// Show all grey
	ALBEDO = vec3(0.2);
	ROUGHNESS = 0.7;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_HEIGHTMAP
	// Show heightmap
	ALBEDO = vec3(smoothstep(-0.1, 2.0, 0.5 + get_height(uv2)/300.0));
	ROUGHNESS = 0.7;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_COLORMAP
	// Show colormap
	ALBEDO = color_map.rgb;
	ROUGHNESS = 0.7;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_ROUGHMAP
	// Show roughness map
	ALBEDO = vec3(color_map.a);
	ROUGHNESS = 0.7;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_CONTROL_TEXTURE
	// Show control map texture selection
	float __ctrl_base = weight_inv * (
	float(mat[0].base) * weights.x +
	float(mat[1].base) * weights.y +
	float(mat[2].base) * weights.z +
	float(mat[3].base) * weights.w )/96.;
	float __ctrl_over = weight_inv * (
	float(mat[0].over) * weights.x +
	float(mat[1].over) * weights.y +
	float(mat[2].over) * weights.z +
	float(mat[3].over) * weights.w )/96.;
	ALBEDO = vec3(__ctrl_base, __ctrl_over, 0.);
	ROUGHNESS = 1.;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_CONTROL_ANGLE
	ivec3 __auv = get_region_uv(floor(uv));
	uint __a_control = texelFetch(_control_maps, __auv, 0).r;
	uint __angle = (__a_control >>10u & 0xFu);
	vec3 __a_colors[16] = {
		vec3(1., .2, .0), vec3(.8, 0., .2), vec3(.6, .0, .4), vec3(.4, .0, .6),
		vec3(.2, 0., .8), vec3(.1, .1, .8), vec3(0., .2, .8), vec3(0., .4, .6),
		vec3(0., .6, .4), vec3(0., .8, .2), vec3(0., 1., 0.), vec3(.2, 1., 0.),
		vec3(.4, 1., 0.), vec3(.6, 1., 0.), vec3(.8, .6, 0.), vec3(1., .4, 0.)
	};
	ALBEDO = __a_colors[__angle];
	ROUGHNESS = 1.;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_CONTROL_SCALE
	ivec3 __suv = get_region_uv(floor(uv));
	uint __s_control = texelFetch(_control_maps, __suv, 0).r;
	uint __scale = (__s_control >>7u & 0x7u);
	vec3 __s_colors[8] = {
		vec3(.5, .5, .5), vec3(.675, .25, .375), vec3(.75, .125, .25), vec3(.875, .0, .125), vec3(1., 0., 0.),
		vec3(0., 0., 1.), vec3(.0, .166, .833), vec3(.166, .333, .666)
	};
	ALBEDO = __s_colors[__scale];
	ROUGHNESS = 1.;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5 ,0.5 ,1.0);

//INSERT: DEBUG_CONTROL_BLEND
	// Show control map blend values
	float __ctrl_blend = weight_inv * (
	float(mat[0].blend) * weights.x +
	float(mat[1].blend) * weights.y +
	float(mat[2].blend) * weights.z +
	float(mat[3].blend) * weights.w );
	ALBEDO = vec3(__ctrl_blend);
	ROUGHNESS = 1.;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_AUTOSHADER
	ivec3 __ruv = get_region_uv(floor(uv));
	uint __control = texelFetch(_control_maps, __ruv, 0).r;
	float __autoshader = float( bool(__control & 0x1u) || __ruv.z<0 );
	ALBEDO = vec3(__autoshader);
	ROUGHNESS = 1.;
	SPECULAR = 0.;

//INSERT: DEBUG_TEXTURE_HEIGHT
	// Show height textures
	ALBEDO = vec3(albedo_height.a);
	ROUGHNESS = 0.7;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_TEXTURE_NORMAL
	// Show normal map textures
	ALBEDO = normal_rough.rgb;
	ROUGHNESS = 0.7;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_TEXTURE_ROUGHNESS
	// Show roughness textures
	ALBEDO = vec3(normal_rough.a);
	ROUGHNESS = 0.7;
	SPECULAR = 0.;
	NORMAL_MAP = vec3(0.5, 0.5, 1.0);

//INSERT: DEBUG_VERTEX_GRID
	// Show region and vertex grids
	vec3 __pixel_pos = (INV_VIEW_MATRIX * vec4(VERTEX,1.0)).xyz;
	float __camera_dist = length(v_camera_pos - __pixel_pos);
	float __region_line = 0.5;		// Region line thickness
	float __grid_line = 0.05;		// Vertex grid line thickness
	float __grid_step = 1.0;			// Vertex grid size, 1.0 == integer units
	float __vertex_size = 4.;		// Size of vertices
	float __view_distance = 300.0;	// Visible distance of grid
	// Draw region grid
	__region_line = .1*sqrt(__camera_dist);
	if (mod(__pixel_pos.x * _mesh_vertex_density + __region_line*.5, _region_size) <= __region_line || 
		mod(__pixel_pos.z * _mesh_vertex_density + __region_line*.5, _region_size) <= __region_line ) {
		ALBEDO = vec3(1.);
	}
	vec3 __vertex_mul = vec3(0.);
	vec3 __vertex_add = vec3(0.);
	float __distance_factor = clamp(1.-__camera_dist/__view_distance, 0., 1.);
	// Draw vertex grid
	if ( mod(__pixel_pos.x * _mesh_vertex_density + __grid_line*.5, __grid_step) < __grid_line || 
	  	 mod(__pixel_pos.z * _mesh_vertex_density + __grid_line*.5, __grid_step) < __grid_line ) { 
		__vertex_mul = vec3(0.5) * __distance_factor;
	}
	// Draw Vertices
	if ( mod(UV.x + __grid_line*__vertex_size*.5, __grid_step) < __grid_line*__vertex_size &&
	  	 mod(UV.y + __grid_line*__vertex_size*.5, __grid_step) < __grid_line*__vertex_size ) { 
		__vertex_add = vec3(0.15) * __distance_factor;
	}
	ALBEDO = fma(ALBEDO, 1.-__vertex_mul, __vertex_add);

)"