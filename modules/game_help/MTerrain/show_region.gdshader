shader_type spatial;
render_mode unshaded;



uniform sampler2D mterrain_heightmap:filter_nearest;
uniform sampler2D mterrain_normals:filter_nearest;


uniform vec3 region_world_position;
uniform float region_size;
uniform float region_a;
uniform float region_b;
uniform int min_lod;
varying vec3 world_pos;
varying vec2 region_uv;

void vertex(){
	world_pos = (NODE_POSITION_WORLD + VERTEX);
	region_uv = (world_pos.xz - region_world_position.xz)/region_size;
	region_uv = region_uv*region_a + region_b;
	VERTEX.y = texture(mterrain_heightmap, region_uv).r;
	world_pos.y = VERTEX.y;
	vec3 n = texture(mterrain_normals, region_uv).rgb;
	n = fma(n,vec3(2.0),vec3(-1.0));
	NORMAL = n;
	//Scale world uv for terrain texture
	world_pos /= 100.0;

}

void fragment(){
	vec2 r = clamp(vec2(0.),vec2(1.),region_uv);
	ALBEDO = vec3(r,0.0)*0.2;
}
