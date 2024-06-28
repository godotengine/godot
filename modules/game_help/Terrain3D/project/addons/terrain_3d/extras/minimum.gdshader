// This shader is the minimum needed to allow the terrain to function, without any texturing.

shader_type spatial;
render_mode blend_mix,depth_draw_opaque,cull_back,diffuse_burley,specular_schlick_ggx;

// Private uniforms
uniform float _region_size = 1024.0;
uniform float _region_texel_size = 0.0009765625; // = 1/1024
uniform float _mesh_vertex_spacing = 1.0;
uniform float _mesh_vertex_density = 1.0; // = 1/_mesh_vertex_spacing
uniform int _region_map_size = 16;
uniform int _region_map[256];
uniform vec2 _region_offsets[256];
uniform sampler2DArray _height_maps : repeat_disable;
uniform usampler2DArray _control_maps : repeat_disable;
uniform sampler2DArray _color_maps : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform sampler2DArray _texture_array_albedo : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform sampler2DArray _texture_array_normal : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;

uniform float _texture_uv_scale_array[32];
uniform float _texture_uv_rotation_array[32];
uniform vec4 _texture_color_array[32];
uniform uint _background_mode = 1u;  // NONE = 0, FLAT = 1, NOISE = 2
uniform uint _mouse_layer = 0x80000000u; // Layer 32

varying flat vec2 v_uv_offset;
varying flat vec2 v_uv2_offset;

////////////////////////
// Vertex
////////////////////////

// Takes in UV world space coordinates, returns ivec3 with:
// XY: (0 to _region_size) coordinates within a region
// Z: layer index used for texturearrays, -1 if not in a region
ivec3 get_region_uv(vec2 uv) {
	uv *= _region_texel_size;
	ivec2 pos = ivec2(floor(uv)) + (_region_map_size / 2);
	int bounds = int(pos.x>=0 && pos.x<_region_map_size && pos.y>=0 && pos.y<_region_map_size);
	int layer_index = _region_map[ pos.y * _region_map_size + pos.x ] * bounds - 1;
	return ivec3(ivec2((uv - _region_offsets[layer_index]) * _region_size), layer_index);
}

// Takes in UV2 region space coordinates, returns vec3 with:
// XY: (0 to 1) coordinates within a region
// Z: layer index used for texturearrays, -1 if not in a region
vec3 get_region_uv2(vec2 uv) {
	// Vertex function added half a texel to UV2, to center the UV's.  vertex(), fragment() and get_height()
	// call this with reclaimed versions of UV2, so to keep the last row/column within the correct
	// window, take back the half pixel before the floor(). 
	ivec2 pos = ivec2(floor(uv - vec2(_region_texel_size * 0.5))) + (_region_map_size / 2);
	int bounds = int(pos.x>=0 && pos.x<_region_map_size && pos.y>=0 && pos.y<_region_map_size);
	int layer_index = _region_map[ pos.y * _region_map_size + pos.x ] * bounds - 1;
	// The return value is still texel-centered.
	return vec3(uv - _region_offsets[layer_index], float(layer_index));
}

// 1 lookup
float get_height(vec2 uv) {
	highp float height = 0.0;
	vec3 region = get_region_uv2(uv);
	if (region.z >= 0.) {
		height = texture(_height_maps, region).r;
	}
 	return height;
}

void vertex() {
	// Get vertex of flat plane in world coordinates and set world UV
	vec3 vertex = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	
	// UV coordinates in world space. Values are 0 to _region_size within regions
	UV = round(vertex.xz * _mesh_vertex_density);

	// Discard vertices for Holes. 1 lookup
	ivec3 region = get_region_uv(UV);
	uint control = texelFetch(_control_maps, region, 0).r;
	bool hole = bool(control >>2u & 0x1u);
	// Show holes to all cameras except mouse camera (on exactly 1 layer)
	if ( !(CAMERA_VISIBLE_LAYERS == _mouse_layer) && 
			(hole || (_background_mode == 0u && region.z < 0)) ) {
		VERTEX.x = 0./0.;
	} else {
		// UV coordinates in region space + texel offset. Values are 0 to 1 within regions
		UV2 = (UV + vec2(0.5)) * _region_texel_size;

		// Get final vertex location and save it
		VERTEX.y = get_height(UV2);
	}

	// Transform UVs to local to avoid poor precision during varying interpolation.
	v_uv_offset = MODEL_MATRIX[3].xz * _mesh_vertex_density;
	UV -= v_uv_offset;
	v_uv2_offset = v_uv_offset * _region_texel_size;
	UV2 -= v_uv2_offset;
}

////////////////////////
// Fragment
////////////////////////

// 3 lookups
vec3 get_normal(vec2 uv, out vec3 tangent, out vec3 binormal) {
	// Get the height of the current vertex
	float height = get_height(uv);

	// Get the heights to the right and in front, but because of hardware 
	// interpolation on the edges of the heightmaps, the values are off
	// causing the normal map to look weird. So, near the edges of the map
	// get the heights to the left or behind instead. Hacky solution that 
	// reduces the artifact, but doesn't fix it entirely. See #185.
	float u, v;
	if(mod(uv.y*_region_size, _region_size) > _region_size-2.) {
		v = get_height(uv + vec2(0, -_region_texel_size)) - height;
	} else {
		v = height - get_height(uv + vec2(0, _region_texel_size));
	}
	if(mod(uv.x*_region_size, _region_size) > _region_size-2.) {
		u = get_height(uv + vec2(-_region_texel_size, 0)) - height;		
	} else {
		u = height - get_height(uv + vec2(_region_texel_size, 0));
	}

	vec3 normal = vec3(u, _mesh_vertex_spacing, v);
	normal = normalize(normal);
	tangent = cross(normal, vec3(0, 0, 1));
	binormal = cross(normal, tangent);
	return normal;
}

void fragment() {
	// Recover UVs
	vec2 uv = UV + v_uv_offset;
	vec2 uv2 = UV2 + v_uv2_offset;

	// Calculate Terrain Normals. 4 lookups
	vec3 w_tangent, w_binormal;
	vec3 w_normal = get_normal(uv2, w_tangent, w_binormal);
	NORMAL = mat3(VIEW_MATRIX) * w_normal;
	TANGENT = mat3(VIEW_MATRIX) * w_tangent;
	BINORMAL = mat3(VIEW_MATRIX) * w_binormal;

	// Apply PBR
	ALBEDO=vec3(.2);
}

