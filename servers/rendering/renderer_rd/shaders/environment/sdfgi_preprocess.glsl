#[compute]

#version 450

#VERSION_DEFINES

#define REGION_SIZE 8

#define GROUP_SIZE 8

#ifdef MODE_REGION_STORE

layout(local_size_x = REGION_SIZE, local_size_y = REGION_SIZE, local_size_z = REGION_SIZE) in;

layout(r32ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_normal_bits;
layout(rg32ui, set = 0, binding = 2) uniform restrict writeonly uimage3D dst_voxels;
layout(r8ui, set = 0, binding = 3) uniform restrict writeonly uimage3D dst_region_bits;
layout(r16ui, set = 0, binding = 4) uniform restrict writeonly uimage3D region_version;

shared uint solid_bit_count;
shared uint region_bits[ ((REGION_SIZE/4) * (REGION_SIZE/4) * (REGION_SIZE/4))*2];

#endif

#ifdef MODE_LIGHT_STORE

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(r16ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_albedo;
layout(r32ui, set = 0, binding = 2) uniform restrict readonly uimage3D src_emission;
layout(r32ui, set = 0, binding = 3) uniform restrict readonly uimage3D src_emission_aniso;
layout(r32ui, set = 0, binding = 4) uniform restrict readonly uimage3D src_normal_bits;

layout(set = 0, binding = 7) uniform texture2DArray occlusion;
layout(set = 0, binding = 8) uniform sampler linear_sampler;

#endif

#if defined(MODE_LIGHT_STORE) || defined(MODE_LIGHT_SCROLL)


struct ProcessVoxel {
	uint position; // xyz 10 bit packed - then 2 extra bits for dynamic and static pending
	uint albedo_normal; // 0 - 16, 17 - 31 normal in octahedral format
	uint emission; // RGBE emission
	uint occlusion; // cached 4 bits occlusion for each 8 neighbouring probes
};

#endif

#if defined(MODE_LIGHT_STORE) || defined(MODE_LIGHT_SCROLL)


#define PROCESS_STATIC_PENDING_BIT 0x80000000
#define PROCESS_DYNAMIC_PENDING_BIT 0x40000000

layout(set = 0, binding = 5, std430) restrict buffer DispatchData {
	uint x;
	uint y;
	uint z;
	uint total_count;
}
dispatch_data;


layout(set = 0, binding = 6, std430) restrict buffer writeonly ProcessVoxels {
	ProcessVoxel data[];
}
dst_process_voxels;

shared uint store_position_count;
shared uint store_from_index;

#endif

#if defined(MODE_LIGHT_SCROLL)

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(r32ui, set = 0, binding = 1) uniform restrict uimage3D light_tex;


layout(set = 0, binding = 7, std430) restrict buffer SrcDispatchData {
	uint x;
	uint y;
	uint z;
	uint total_count;
}
src_dispatch_data;

layout(set = 0, binding = 8, std430) restrict buffer readonly SrcProcessVoxels {
	ProcessVoxel data[];
}
src_process_voxels;


#endif

// Some occlusion defines

#ifdef HIGH_DENSITY_PROBES
#define PROBE_CELLS 4
#else
#define PROBE_CELLS 8
#endif

#define OCC8_DISTANCE_MAX 15.0
#define OCC16_DISTANCE_MAX 256.0


#if defined(MODE_OCCLUSION) || defined(MODE_OCCLUSION_STORE)

layout(local_size_x = OCCLUSION_OCT_SIZE, local_size_y = OCCLUSION_OCT_SIZE, local_size_z = 1) in;


#endif

#ifdef MODE_OCCLUSION


layout(r32ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_solid_bits0;
layout(r32ui, set = 0, binding = 2) uniform restrict readonly uimage3D src_solid_bits1;
layout(r32ui, set = 0, binding = 3) uniform restrict readonly uimage3D src_normal_bits;
layout(r8, set = 0, binding = 4) uniform restrict writeonly image2DArray dst_occlusion;

// Bit voxels for DDA tracing.
shared uint bit_voxels[PROBE_CELLS*PROBE_CELLS*PROBE_CELLS*2];
shared uint bit_normals[PROBE_CELLS*PROBE_CELLS*PROBE_CELLS];
shared uint loaded_regions;

// Max distance in a 8^3 cube is 13.856406460551018, so..

#endif

#ifdef MODE_OCCLUSION_STORE

layout(r8, set = 0, binding = 1) uniform restrict readonly image2DArray src_occlusion;
layout(rg16, set = 0, binding = 2) uniform restrict writeonly image2DArray dst_occlusion;

shared vec3 neighbours[OCCLUSION_OCT_SIZE * OCCLUSION_OCT_SIZE];
shared int invalid_rays;

#endif

#ifdef MODE_LIGHTPROBE_SCROLL

layout(local_size_x = LIGHTPROBE_OCT_SIZE, local_size_y = LIGHTPROBE_OCT_SIZE, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 1) uniform restrict image2DArray lightprobe_specular_data;
layout(r32ui, set = 0, binding = 2) uniform restrict uimage2DArray lightprobe_diffuse_data;
layout(r32ui, set = 0, binding = 3) uniform restrict uimage2DArray lightprobe_ambient_data;
layout(r32ui, set = 0, binding = 4) uniform restrict uimage2DArray ray_hit_cache;
layout(set = 0, binding = 5) uniform texture2DArray occlusion_probes;
layout(set = 0, binding = 6) uniform sampler linear_sampler;


#define MAX_CASCADES 8

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 region_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 7, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

shared float occlusion[8];

#endif


layout(push_constant, std430) uniform Params {	
	ivec3 grid_size;
	uint region_version;

	ivec3 scroll;
	int cascade_count;

	ivec3 offset;
	int step;

	ivec3 limit;
	int cascade;

	ivec3 region_world_pos;
	int probe_cells;

	ivec3 probe_axis_size;
	int ray_hit_cache_frames;
}
params;

#define DISTANCE_MAX 15.0

float encode_distance(float p_distance) {

	return clamp((p_distance + 1.0) / DISTANCE_MAX, 0.0, 1.0);
}


uint encode_position(ivec3 p_position) {
	p_position += ivec3(512);
	p_position = clamp(p_position,ivec3(0),ivec3(1023));
	return uint(p_position.x | (p_position.y<<10) | (p_position.z<<20));
}

ivec3 decode_position(uint p_encoded_position) {
	return ((ivec3(p_encoded_position) >> ivec3(0,10,20)) & ivec3(0x3FF)) - ivec3(512);
}

int sq_distance(ivec3 p_from, ivec3 p_to) {

	ivec3 m = p_from - p_to;
	m*=m;
	return m.x + m.y + m.z;
}

vec2 octahedron_wrap(vec2 v) {
	vec2 signVal;
	signVal.x = v.x >= 0.0 ? 1.0 : -1.0;
	signVal.y = v.y >= 0.0 ? 1.0 : -1.0;
	return (1.0 - abs(v.yx)) * signVal;
}

vec2 octahedron_encode(vec3 n) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = n.z >= 0.0 ? n.xy : octahedron_wrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

vec3 octahedron_decode(vec2 f) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	f = f * 2.0 - 1.0;
	vec3 n = vec3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));
	float t = clamp(-n.z, 0.0, 1.0);
	n.x += n.x >= 0 ? -t : t;
	n.y += n.y >= 0 ? -t : t;
	return normalize(n);
}


uint rgbe_encode(vec3 color) {
	const float pow2to9 = 512.0f;
	const float B = 15.0f;
	const float N = 9.0f;
	const float LN2 = 0.6931471805599453094172321215;

	float cRed = clamp(color.r, 0.0, 65408.0);
	float cGreen = clamp(color.g, 0.0, 65408.0);
	float cBlue = clamp(color.b, 0.0, 65408.0);

	float cMax = max(cRed, max(cGreen, cBlue));

	float expp = max(-B - 1.0f, floor(log(cMax) / LN2)) + 1.0f + B;

	float sMax = floor((cMax / pow(2.0f, expp - B - N)) + 0.5f);

	float exps = expp + 1.0f;

	if (0.0 <= sMax && sMax < pow2to9) {
		exps = expp;
	}

	float sRed = floor((cRed / pow(2.0f, exps - B - N)) + 0.5f);
	float sGreen = floor((cGreen / pow(2.0f, exps - B - N)) + 0.5f);
	float sBlue = floor((cBlue / pow(2.0f, exps - B - N)) + 0.5f);
	return (uint(sRed) & 0x1FF) | ((uint(sGreen) & 0x1FF) << 9) | ((uint(sBlue) & 0x1FF) << 18) | ((uint(exps) & 0x1F) << 27);
}


vec3 rgbe_decode(uint p_rgbe) {
	vec4 rgbef = vec4((uvec4(p_rgbe) >> uvec4(0,9,18,27)) & uvec4(0x1FF,0x1FF,0x1FF,0x1F));
	return rgbef.rgb * pow( 2.0, rgbef.a - 15.0 - 9.0 );
}


ivec3 modi(ivec3 value, ivec3 p_y) {
	return ((value % p_y) + p_y) % p_y;
}

ivec2 probe_to_tex(ivec3 local_probe) {

	ivec3 cell = modi( params.region_world_pos + local_probe,ivec3(params.probe_axis_size));
	return cell.xy + ivec2(0,cell.z * int(params.probe_axis_size.y));

}

#ifdef MODE_LIGHTPROBE_SCROLL

ivec2 probe_to_texc(ivec3 local_probe,int cascade) {

	ivec3 cell = modi( cascades.data[cascade].region_world_offset + local_probe,ivec3(params.probe_axis_size));
	return cell.xy + ivec2(0,cell.z * int(params.probe_axis_size.y));

}

#endif

void main() {

#ifdef MODE_REGION_STORE

	ivec3 region_mask = (params.grid_size / REGION_SIZE) - 1;

	ivec3 src_pos = ivec3(gl_GlobalInvocationID.xyz) + params.offset;
	ivec3 region_pos = ivec3(gl_LocalInvocationID.xyz);
	ivec3 subregion = region_pos / 4;
	uint subregion_ofs = subregion.z * 4 + subregion.y * 2 + subregion.x;
	ivec3 subregion_pos = region_pos % 4;
	uint subregion_bit = subregion_pos.z * 16 + subregion_pos.y * 4 + subregion_pos.x;

	if (region_pos == ivec3(0)) {
		solid_bit_count = 0;
	}

	if (subregion_pos == ivec3(0)) {
		region_bits[subregion_ofs*2+0]=0;
		region_bits[subregion_ofs*2+1]=0;
	}

	groupMemoryBarrier();
	barrier();

	uint p = imageLoad(src_normal_bits, src_pos).r;

	if (p != 0) {
		atomicAdd(solid_bit_count,1);
		if (subregion_bit < 32) {
			atomicOr(region_bits[subregion_ofs*2+0],1<<subregion_bit);
		} else {
			atomicOr(region_bits[subregion_ofs*2+1],1<<(subregion_bit-32));
		}
	}

	groupMemoryBarrier();
	barrier();

	ivec3 dst_region = ((src_pos / REGION_SIZE) + params.region_world_pos) & region_mask;

	if (subregion_pos == ivec3(0)) {

		ivec3 dst_pos = (dst_region * REGION_SIZE + region_pos) / 4;
		uvec2 bits = uvec2(region_bits[subregion_ofs*2+0],region_bits[subregion_ofs*2+1]);
		dst_pos.y += params.cascade * (params.grid_size.y / 4);
		imageStore(dst_voxels,dst_pos,uvec4(bits,uvec2(0)));
	}

	if (region_pos == ivec3(0)) {
		dst_region.y += params.cascade * (params.grid_size.y / REGION_SIZE);
		imageStore(dst_region_bits,dst_region,uvec4(solid_bit_count > 0 ? 1 : 0));

		imageStore(region_version,dst_region,uvec4(params.region_version));

		// Store region version
	}


#endif

#ifdef MODE_LIGHT_SCROLL

	int src_index = int(gl_GlobalInvocationID).x;
	int local = int(gl_LocalInvocationID).x;

	if (src_index >= src_dispatch_data.total_count) {
		// Do not process.
		return;
	}

	if (local == 0) {
		store_position_count = 0; // Base one stores as zero, the others wait
		if (src_index == 0) {
			// This lone thread clears y and z.
			dispatch_data.y = 1;
			dispatch_data.z = 1;
		}
	}

	groupMemoryBarrier();
	barrier();

	ivec3 src_pos = (ivec3(src_process_voxels.data[src_index].position) >> ivec3(0,10,20)) & ivec3(0x3FF);
	bool inside_area = all(greaterThanEqual(src_pos, params.offset)) && all(lessThan(src_pos, params.limit));


	if (!inside_area) {

		ivec3 light_pos = src_pos;
		light_pos = (light_pos + (params.region_world_pos * REGION_SIZE)) & (params.grid_size-1);
		light_pos.y += params.grid_size.y * params.cascade;

		// As this will be a new area, clear the new region from the old values.
		imageStore(light_tex,light_pos,uvec4(0));
	}
	uint index;

	if (inside_area) {
		index = atomicAdd(store_position_count, 1);
	}

	groupMemoryBarrier();
	barrier();

	// global increment only once per group, to reduce pressure

	if (!inside_area || store_position_count == 0) {
		return;
	}

	if (index == 0) {
		store_from_index = atomicAdd(dispatch_data.total_count, store_position_count);
		uint group_count = (store_from_index + store_position_count - 1) / 64 + 1;
		atomicMax(dispatch_data.x, group_count);
	}

	groupMemoryBarrier();
	barrier();

	index+=store_from_index;

	ivec3 dst_pos = src_pos + params.scroll;

	uint src_pending_bits = src_process_voxels.data[src_index].position & (PROCESS_STATIC_PENDING_BIT|PROCESS_DYNAMIC_PENDING_BIT);

	dst_process_voxels.data[index].position = uint(dst_pos.x | (dst_pos.y << 10) | (dst_pos.z << 20)) | src_pending_bits;
	dst_process_voxels.data[index].albedo_normal = src_process_voxels.data[src_index].albedo_normal;
	dst_process_voxels.data[index].emission = src_process_voxels.data[src_index].emission;
	dst_process_voxels.data[index].occlusion = src_process_voxels.data[src_index].occlusion;

#endif

#ifdef MODE_LIGHT_STORE

	ivec3 local = ivec3(gl_LocalInvocationID.xyz);
	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz) + params.offset;
	uint solid = imageLoad(src_normal_bits, pos).r;

	if (local == ivec3(0)) {
		store_position_count = 0; // Base one stores as zero, the others wait
		if (pos == params.offset) {
			// This lone thread clears y and z.
			dispatch_data.y = 1;
			dispatch_data.z = 1;
		}
	}

	if (solid!=0) {
		return; // solid pixel, nothing to do.
	}

	vec4 albedo_accum = vec4(0.0);
	vec4 emission_accum = vec4(0.0);
	vec3 normal_accum = vec3(0.0);
	uint occlusionu = 0;

	//opposite to aniso dir
	const ivec3 offsets[6] = ivec3[](
			ivec3(1, 0, 0),
			ivec3(-1, 0, 0),
			ivec3(0, 1, 0),
			ivec3(0, -1, 0),
			ivec3(0, 0, 1),
			ivec3(0, 0, -1));

	const vec3 aniso_dir[6] = vec3[](
			vec3(-1, 0, 0),
			vec3(1, 0, 0),
			vec3(0, -1, 0),
			vec3(0, 1, 0),
			vec3(0, 0, -1),
			vec3(0, 0, 1));

	// aniso dir in bitform
	const uint aniso_mask[6] = uint[](
		(1<<0),
		(1<<1),
		(1<<2),
		(1<<3),
		(1<<4),
		(1<<5) );


	bool voxels_found=false;

	for(int i=0;i<6;i++) {

		ivec3 ofs = pos + offsets[i];
		if (any(lessThan(ofs, params.offset)) || any(greaterThanEqual(ofs, params.limit))) {
			// Outside range, continue.
			continue;
		}


		uint n = imageLoad(src_normal_bits, ofs).r;
		if (!bool(n & aniso_mask[i])) {
			// Not solid, continue.
			continue;
		}

		voxels_found = true;

		/*const int facing_direction_count =  26 ;
		const vec3 facing_directions[ 26 ]=vec3[]( vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0), vec3(-0.5773502691896258, -0.5773502691896258, -0.5773502691896258), vec3(-0.7071067811865475, -0.7071067811865475, 0.0), vec3(-0.5773502691896258, -0.5773502691896258, 0.5773502691896258), vec3(-0.7071067811865475, 0.0, -0.7071067811865475), vec3(-0.7071067811865475, 0.0, 0.7071067811865475), vec3(-0.5773502691896258, 0.5773502691896258, -0.5773502691896258), vec3(-0.7071067811865475, 0.7071067811865475, 0.0), vec3(-0.5773502691896258, 0.5773502691896258, 0.5773502691896258), vec3(0.0, -0.7071067811865475, -0.7071067811865475), vec3(0.0, -0.7071067811865475, 0.7071067811865475), vec3(0.0, 0.7071067811865475, -0.7071067811865475), vec3(0.0, 0.7071067811865475, 0.7071067811865475), vec3(0.5773502691896258, -0.5773502691896258, -0.5773502691896258), vec3(0.7071067811865475, -0.7071067811865475, 0.0), vec3(0.5773502691896258, -0.5773502691896258, 0.5773502691896258), vec3(0.7071067811865475, 0.0, -0.7071067811865475), vec3(0.7071067811865475, 0.0, 0.7071067811865475), vec3(0.5773502691896258, 0.5773502691896258, -0.5773502691896258), vec3(0.7071067811865475, 0.7071067811865475, 0.0), vec3(0.5773502691896258, 0.5773502691896258, 0.5773502691896258) );
		for(int j=0;j<facing_direction_count;j++) {
			if (bool(n & uint((1<<(j+6))))) {
				normal_accum += facing_directions[j];
			}
		}*/

		normal_accum += aniso_dir[i];

		ivec3 albedo_ofs = ofs>>1;
		albedo_ofs.z *= 6;
		albedo_ofs.z += i;

		uint a = imageLoad(src_albedo, albedo_ofs).r;
		albedo_accum += vec4(vec3((ivec3(a) >> ivec3(0,5,11)) & ivec3(0x1f,0x3f,0x1f)) / vec3(31.0,63.0,31.0), 1.0);

		uint rgbe = imageLoad(src_emission, ofs>>1).r;

		vec3 emission = rgbe_decode(rgbe);

		uint rgbe_aniso = imageLoad(src_emission_aniso, ofs>>1).r;
		float strength = ((rgbe_aniso >> (i * 5)) & 0x1F) / float(0x1F);
		emission_accum += vec4(emission * strength,1.0);
	}

	{
		// compute occlusion

		ivec3 base_probe = pos / PROBE_CELLS;
		float weights[8];
		float total_weight = 0.0;
		vec3 posf = vec3(pos) + 0.5; // Actual point in the center of the box.
		vec2 probe_tex_to_uv = 1.0 / vec2( (OCCLUSION_OCT_SIZE+2) * params.probe_axis_size.x, (OCCLUSION_OCT_SIZE+2) * params.probe_axis_size.y * params.probe_axis_size.z );
		for(int i=0;i<8;i++) {
			ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));

			vec3 probe_pos = vec3(probe * PROBE_CELLS);

			vec3 probe_to_pos = posf - probe_pos;
			vec3 n = normalize(probe_to_pos);
			float d = length(probe_to_pos);
			ivec2 tex_pos = probe_to_tex(probe);
			vec2 tex_uv = vec2(ivec2(tex_pos * (OCCLUSION_OCT_SIZE+2) + ivec2(1))) + octahedron_encode(n) * float(OCCLUSION_OCT_SIZE);
			tex_uv *= probe_tex_to_uv;
			vec2 o_o2 = texture(sampler2DArray(occlusion,linear_sampler),vec3(tex_uv,float(params.cascade))).rg * OCC16_DISTANCE_MAX;

			float mean = o_o2.x;
			float variance = abs((mean*mean) - o_o2.y);

			 // http://www.punkuser.net/vsm/vsm_paper.pdf; equation 5
			 // Need the max in the denominator because biasing can cause a negative displacement
			float dmean = max(d - mean, 0.0);
			float chebyshev_weight = variance / (variance + dmean*dmean);

			float weight = (d <= mean) ? 1.0 : chebyshev_weight;

			weight = max(0.000001, weight); // make sure not zero (only trilinear can be zero)

			vec3 trilinear = vec3(1.0) - abs(probe_to_pos / float(PROBE_CELLS));

			weight *= trilinear.x * trilinear.y * trilinear.z;

			weights[i]=weight;
			total_weight += weight;
		}

		for(int i=0;i<8;i++) {
			float w = weights[i] / total_weight;
			w *= 15.0;
			occlusionu|=uint(clamp(w,0.0,15.0)) << (i * 4);
		}

	}
	groupMemoryBarrier();
	barrier();

	uint index;

	if (voxels_found) {
		index = atomicAdd(store_position_count, 1);
	}

	groupMemoryBarrier();
	barrier();

	if (!voxels_found || store_position_count==0) {
		return;
	}

	// global increment only once per group, to reduce pressure

	if (index == 0) {
		store_from_index = atomicAdd(dispatch_data.total_count, store_position_count);
		uint group_count = (store_from_index + store_position_count - 1) / 64 + 1;
		atomicMax(dispatch_data.x, group_count);

	}

	groupMemoryBarrier();
	barrier();


	index += store_from_index;

	normal_accum = normalize(normal_accum);
	albedo_accum.rgb /= albedo_accum.a;
	emission_accum.rgb /= emission_accum.a;

	dst_process_voxels.data[index].position = uint(pos.x | (pos.y << 10) | (pos.z << 20)) | (PROCESS_STATIC_PENDING_BIT|PROCESS_DYNAMIC_PENDING_BIT);

	uint albedo_norm = 0;
	albedo_norm |= clamp(uint(albedo_accum.r * 31.0), 0, 31) << 0;
	albedo_norm |= clamp(uint(albedo_accum.g * 63.0), 0, 63) << 5;
	albedo_norm |= clamp(uint(albedo_accum.b * 31.0), 0, 31) << 11;

	vec2 octa_normal = octahedron_encode(normal_accum);
	uvec2 octa_unormal = clamp( uvec2(octa_normal * 255), uvec2(0), uvec2(255));
	albedo_norm |= (octa_unormal.x<<16) | (octa_unormal.y << 24);

	dst_process_voxels.data[index].albedo_normal = albedo_norm;
	dst_process_voxels.data[index].emission = rgbe_encode(emission_accum.rgb);

	dst_process_voxels.data[index].occlusion = occlusionu;

#endif


#ifdef MODE_OCCLUSION



	int linear_index = int(gl_LocalInvocationID.y * OCCLUSION_OCT_SIZE + gl_LocalInvocationID.x);

	if (linear_index==0) {
		loaded_regions = 0;
	}

	groupMemoryBarrier();
	barrier();


	{

		int index_mult = 3;

		int nonzero_found = 0;
		for(int i=0;i<index_mult;i++) {
			int load_index = linear_index * index_mult  + i;
			ivec3 load_pos;

			load_pos.z = load_index / (PROBE_CELLS*PROBE_CELLS);
			load_pos.y = (load_index / PROBE_CELLS) % PROBE_CELLS;
			load_pos.x = load_index % PROBE_CELLS;

			if (all(lessThan(load_pos,ivec3(PROBE_CELLS)))) {

				load_pos += ivec3(gl_WorkGroupID.xyz * PROBE_CELLS) + params.offset;
				uint n = imageLoad(src_normal_bits,load_pos).r;
				bit_normals[load_index] = n;

				if (n!=0) {
					// Save some bandwidth.
					uint p0 = imageLoad(src_solid_bits0,load_pos).r;
					uint p1 = imageLoad(src_solid_bits1,load_pos).r;
					bit_voxels[load_index * 2 + 0] = p0;
					bit_voxels[load_index * 2 + 1] = p1;

					nonzero_found++;
				} else {
					bit_voxels[load_index * 2 + 0] = 0;
					bit_voxels[load_index * 2 + 1] = 0;
				}
			}
		}

		if (nonzero_found > 0) {
			atomicAdd(loaded_regions,1);
		}
	}

	groupMemoryBarrier();
	barrier();


	// Special pattern to distribute rays evenely around octahedron sides
	//const uint probe_ofs[OCCLUSION_OCT_SIZE*OCCLUSION_OCT_SIZE]=uint[](7, 7, 7, 7, 7, 7, 7, 2, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 3, 2, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 3, 3, 3, 2, 2, 2, 6, 6, 6, 6, 7, 7, 7, 7, 3, 3, 3, 2, 2, 2, 6, 6, 6, 6, 7, 7, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 6, 6, 7, 7, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 6, 6, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 6, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 4, 4, 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 4, 4, 5, 5, 5, 5, 1, 1, 1, 0, 0, 0, 4, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1, 0, 0, 0, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 1, 0, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 1, 4, 4, 4, 4, 4, 4, 4);
	const uint probe_ofs[OCCLUSION_OCT_SIZE*OCCLUSION_OCT_SIZE]=uint[](7, 7, 7, 7, 7, 7, 3, 2, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 3, 3, 2, 2, 6, 6, 6, 6, 6, 7, 7, 7, 7, 3, 3, 3, 2, 2, 2, 6, 6, 6, 6, 7, 7, 7, 3, 3, 3, 3, 2, 2, 2, 2, 6, 6, 6, 7, 7, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 6, 6, 7, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 6, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 4, 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 4, 4, 5, 5, 5, 1, 1, 1, 1, 0, 0, 0, 0, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1, 0, 0, 0, 4, 4, 4, 4, 5, 5, 5, 5, 5, 1, 1, 0, 0, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 1, 0, 4, 4, 4, 4, 4, 4 );

	uint offset = probe_ofs[linear_index];


	vec3 ray_dir = octahedron_decode((vec2(gl_LocalInvocationID.xy) + (0.5)) / vec2(OCCLUSION_OCT_SIZE));

	ray_dir = mix( ray_dir, vec3(0.0), lessThan(abs(ray_dir),vec3(0.001))); // Avoid numerical error causing to go out of bounds


	ivec3 ray_from = mix(ivec3(0.0),ivec3(PROBE_CELLS*4-1),bvec3(ivec3(offset) & ivec3(1,2,4)));
	ivec3 ray_pos = ray_from;
	bool hit = false;
	uint hit_normals;

	if (loaded_regions!=0) {

#if 1

		const int LEVEL_BLOCK = 0;
		const int LEVEL_VOXEL = 1;
		const int MAX_LEVEL = 2;

	//#define HQ_RAY

		const int fp_bits = 12;

		const int fp_block_bits = fp_bits + 2;

		bvec3 limit_dir = greaterThan(ray_dir,vec3(0.0));
		ivec3 step = mix(ivec3(0),ivec3(1),limit_dir);
		ivec3 ray_sign = ivec3(sign(ray_dir));

		ivec3 ray_dir_fp = ivec3(ray_dir * float(1<<fp_bits));

		const float limit = 1.0/127.0;

		bvec3 ray_zero = lessThan(abs(ray_dir),vec3(limit));
		ivec3 inv_ray_dir_fp = ivec3( float(1<<fp_bits) / ray_dir );

		const ivec3 level_masks[MAX_LEVEL]=ivec3[](
			ivec3(1<<fp_block_bits) - ivec3(1),
			ivec3(1<<fp_bits) - ivec3(1)
		);

		ivec3 limits[MAX_LEVEL];

		limits[LEVEL_BLOCK] = ((ivec3(PROBE_CELLS*4) << fp_bits) - ivec3(1)) * step; // Region limit does not change, so initialize now.

		// Initialize to cascade
		int level = LEVEL_BLOCK;

		ivec3 pos = ray_pos << fp_bits;
		int block_offset;

		uint block_lo;
		uint block_hi;

		while(level >= 0) {
			// This loop is written so there is only one single main interation.
			// This ensures that different compute threads working on different
			// levels can still run together without blocking each other.

			if (level == LEVEL_VOXEL) {
				// The first level should be (in a worst case scenario) the most used
				// so it needs to appear first. The rest of the levels go from more to least used order.

				ivec3 block_local = (pos & level_masks[LEVEL_BLOCK]) >> fp_bits;
				uint block_index = uint(block_local.z * 16 + block_local.y * 4 + block_local.x);
				if (block_index < 32) {
					// Low 32 bits.
					if (bool(block_lo & uint(1<<block_index))) {
						hit=true;
						hit_normals = bit_normals[block_offset>>1];
						break;
					}
				} else {
					// High 32 bits.
					block_index-=32;
					if (bool(block_hi & uint(1<<block_index))) {
						hit=true;
						hit_normals = bit_normals[block_offset>>1];
						break;
					}
				}
			} else if (level == LEVEL_BLOCK) {
				ivec3 block_local = pos >> fp_block_bits;
				block_offset = (block_local.z * PROBE_CELLS*PROBE_CELLS + block_local.y * PROBE_CELLS + block_local.x) * 2;
				block_lo = bit_voxels[block_offset];
				block_hi = bit_voxels[block_offset+1];
				if (block_lo!=0 || block_hi!=0) {
					// Have voxels inside
					level = LEVEL_VOXEL;
					limits[LEVEL_VOXEL]= pos - (pos & level_masks[LEVEL_BLOCK]) + step * (level_masks[LEVEL_BLOCK] + ivec3(1));
					continue;
				}
			}

			// Fixed point, multi-level DDA.

			ivec3 mask = level_masks[level];
			ivec3 box = mask * step;
			ivec3 pos_diff = box - (pos & mask);
#ifdef HQ_RAY
			ivec3 mul_res = mul64(pos_diff,inv_ray_dir_fp,fp_bits);
#else
			ivec3 mul_res = (pos_diff * inv_ray_dir_fp) >> fp_bits;
#endif
			ivec3 tv = mix(mul_res,ivec3(0x7FFFFFFF),ray_zero);
			int t = min(tv.x,min(tv.y,tv.z));

			// The general idea here is that we _always_ need to increment to the closest next cell
			// (this is a DDA after all), so adv_box forces this increment for the minimum axis.

			ivec3 adv_box = pos_diff + ray_sign;
#ifdef HQ_RAY
			ivec3 adv_t = mul64(ray_dir_fp, ivec3(t), fp_bits);
#else
			ivec3 adv_t = (ray_dir_fp * t) >> fp_bits;
#endif
			pos += mix(adv_t,adv_box,equal(ivec3(t),tv));

			while(true) {
				bvec3 limit = lessThan(pos,limits[level]);
				bool inside = all(equal(limit,limit_dir));
				if (inside) {
					break;
				}
				level-=1;
				if (level == -1) {
					break;
				}
			}
		}

		ray_pos = pos >> fp_bits;

#else
		vec3 delta = min(abs(1.0 / ray_dir), vec3(PROBE_CELLS*4));
		ivec3 step = ivec3(sign(ray_dir));
		vec3 side = (sign(ray_dir) * vec3(-0.5) + (sign(ray_dir) * 0.5) + 0.5) * delta;



		while(all(greaterThanEqual(ray_pos,ivec3(0))) && all(lessThan(ray_pos,ivec3(PROBE_CELLS*4)))) {

			ivec3 ray_cell_pos = ray_pos >> ivec3(2);
			int cell_ofs = ray_cell_pos.z * PROBE_CELLS*PROBE_CELLS + ray_cell_pos.y * PROBE_CELLS + ray_cell_pos.x;
			cell_ofs*=2;
			ivec3 ray_bit_pos = ray_pos & ivec3(0x3);
			int bit_ofs = ray_bit_pos.z * 16 + ray_bit_pos.y * 4 + ray_bit_pos.x;

			if (bit_ofs >= 32) {
				bit_ofs-=32;
				cell_ofs+=1;
			}

			if (bool(bit_voxels[cell_ofs] & (1<<bit_ofs) )) {
				hit_normals = bit_normals[cell_ofs>>1];
				hit=true;
				break;
			}

			// because we deal with small bits and very precise directions, a perfect diagonal ray
			// can go through two pixels, so always ensure to advance one pixel at a time.
			// bvec3 mask = lessThanEqual(side.xyz, min(side.yzx, side.zxy));
			vec3 mask = vec3(1,0,0);
			float m = side.x;
			if (side.y < m) {
				mask = vec3(0,1,0);
				m = side.y;
			}
			if (side.z < m) {
				mask = vec3(0,0,1);
			}

			side += vec3(mask) * delta;
			ray_pos += ivec3(mask) * step;
		}

#endif

	}

	float d;

	if (hit) {

		const int facing_direction_count =  26 ;
		const vec3 facing_directions[ facing_direction_count ]=vec3[]( vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0), vec3(-0.5773502691896258, -0.5773502691896258, -0.5773502691896258), vec3(-0.7071067811865475, -0.7071067811865475, 0.0), vec3(-0.5773502691896258, -0.5773502691896258, 0.5773502691896258), vec3(-0.7071067811865475, 0.0, -0.7071067811865475), vec3(-0.7071067811865475, 0.0, 0.7071067811865475), vec3(-0.5773502691896258, 0.5773502691896258, -0.5773502691896258), vec3(-0.7071067811865475, 0.7071067811865475, 0.0), vec3(-0.5773502691896258, 0.5773502691896258, 0.5773502691896258), vec3(0.0, -0.7071067811865475, -0.7071067811865475), vec3(0.0, -0.7071067811865475, 0.7071067811865475), vec3(0.0, 0.7071067811865475, -0.7071067811865475), vec3(0.0, 0.7071067811865475, 0.7071067811865475), vec3(0.5773502691896258, -0.5773502691896258, -0.5773502691896258), vec3(0.7071067811865475, -0.7071067811865475, 0.0), vec3(0.5773502691896258, -0.5773502691896258, 0.5773502691896258), vec3(0.7071067811865475, 0.0, -0.7071067811865475), vec3(0.7071067811865475, 0.0, 0.7071067811865475), vec3(0.5773502691896258, 0.5773502691896258, -0.5773502691896258), vec3(0.7071067811865475, 0.7071067811865475, 0.0), vec3(0.5773502691896258, 0.5773502691896258, 0.5773502691896258) );
		bool any_frontface = false;
		for(int j=0;j<facing_direction_count;j++) {
			if (bool(hit_normals & uint((1<<(j+6))))) {
				if ( dot(ray_dir,facing_directions[j]) < -0.001 ) {
					any_frontface = true;
					break;
				}
			}
		}

		if (any_frontface) {
			// Hit a bafkace, go back one cell as penalty
			d = distance(vec3(ray_from),vec3(ray_pos)) / 4.0; // back to cells
		} else {
			d = OCC8_DISTANCE_MAX; //max(0.0,d - 1.0);
		}

	} else {

		const float edge = 0.25;
		vec3 posf = vec3(ray_from) + vec3(0.5);
		vec3 min_bounds = vec3(-edge);
		vec3 max_bounds = vec3(PROBE_CELLS * 4.0 + edge);

		vec3 plane = mix(min_bounds,max_bounds,greaterThan(ray_dir,vec3(0.0)));
		vec3 tv = mix( (plane - posf) / ray_dir, vec3(1e20), equal(ray_dir,vec3(0.0)));
		d = min(tv.x,min(tv.y,tv.z)) / 4.0;

		// Nope because it causes bending inwards.
		// d = distance(vec3(ray_from),vec3(ray_pos)) / 4.0; // save anyway

		// Nope because it causes bending outwards due to interpolation
		// d = OCC8_DISTANCE_MAX;
	}

	ivec3 base_probe = (params.offset / PROBE_CELLS) + ivec3(gl_WorkGroupID.xyz);


	if (bool(offset & 1)) {
		base_probe.x+=1;
	}
	if (bool(offset & 2)) {
		base_probe.y+=1;
	}
	if (bool(offset & 4)) {
		base_probe.z+=1;
	}

	ivec2 probe_tex_pos = probe_to_tex(base_probe);

	int probe_axis_size = (params.grid_size.y / PROBE_CELLS) + 1;
	ivec3 tex_uv = ivec3(probe_tex_pos * OCCLUSION_OCT_SIZE + ivec2(gl_LocalInvocationID).xy, params.cascade);

	imageStore(dst_occlusion,tex_uv,vec4(d / OCC8_DISTANCE_MAX));

#endif

#ifdef MODE_OCCLUSION_STORE

	const uint neighbour_max_weights = 20;
	const uint neighbour_weights[3920]= uint[](3951, 69319, 134200, 723903, 790094, 855812, 921287, 986474, 1706978, 1772898, 1838136, 10095551, 11013710, 11078626, 11865326, 11931396, 11996514, 12717294, 12783258, 0, 4063, 69780, 135023, 199624, 724684, 790523, 855901, 921445, 987022, 1051929, 1707416, 1772979, 1838327, 1903562, 11013555, 11865115, 11931231, 12783076, 0, 0, 4068, 70319, 136210, 201292, 659824, 725808, 791018, 855878, 921523, 987722, 1053287, 1117892, 1642396, 1707957, 1904315, 0, 0, 0, 0, 0, 69046, 135574, 201547, 266682, 331372, 527817, 594158, 660072, 725186, 986670, 1053068, 1118594, 1183365, 1576666, 1642193, 1969644, 0, 0, 0, 0, 200771, 266675, 331960, 397003, 462411, 528411, 594172, 659336, 1052000, 1118313, 1183811, 1248884, 1445069, 1510966, 1576473, 2035039, 2100378, 0, 0, 0, 199466, 265755, 331503, 396917, 462351, 527990, 593303, 658092, 1117287, 1183275, 1248773, 1314073, 1379496, 1445042, 1510556, 1575621, 2034493, 2100248, 2165788, 2362028, 265750, 331856, 397530, 462965, 528349, 593308, 1183536, 1249345, 1314896, 1380317, 1445618, 1510837, 2100715, 2166576, 2231830, 2297244, 2362805, 0, 0, 0, 265628, 331741, 397429, 463066, 528464, 593430, 1183157, 1249010, 1314781, 1380432, 1445953, 1511216, 2166197, 2231708, 2297366, 2363184, 2428395, 0, 0, 0, 199340, 265623, 331382, 396815, 462453, 528111, 593435, 658218, 1116869, 1182876, 1248434, 1313960, 1379609, 1445381, 1510955, 1576039, 2165420, 2362396, 2427928, 2493245, 200584, 266492, 331803, 396875, 462539, 528568, 594355, 659523, 1117721, 1183286, 1248461, 1445492, 1511491, 1577065, 1641824, 2428058, 2493791, 0, 0, 0, 135362, 201320, 266478, 331209, 527980, 594362, 660299, 725398, 789942, 1052369, 1117914, 1511045, 1577346, 1642892, 1707566, 2559468, 0, 0, 0, 0, 3910, 70122, 135984, 201072, 660044, 726034, 791215, 856036, 987061, 1052572, 1576644, 1643111, 1708618, 1773491, 2625211, 0, 0, 0, 0, 0, 3933, 69627, 134860, 658376, 724847, 790676, 856031, 921011, 986520, 1641753, 1707918, 1773413, 2624458, 2690295, 11013147, 11865523, 11931108, 12783199, 0, 0, 3844, 69198, 134079, 724024, 790215, 855919, 920930, 986082, 1707370, 1773255, 2690104, 10947519, 11013358, 11799522, 11865678, 11931290, 11996398, 12717410, 12783364, 0, 4063, 69477, 134391, 789939, 855647, 921748, 987022, 1051594, 1838959, 1903897, 2755528, 10096332, 11014139, 11079064, 11931485, 11996595, 12717083, 12783076, 0, 0, 3982, 69836, 135247, 199977, 724482, 790189, 855495, 921804, 987623, 1052715, 1839183, 1904683, 1969335, 2755881, 10096130, 11013805, 11931079, 0, 0, 0, 70271, 136566, 202059, 266673, 659963, 725557, 921766, 988452, 1054569, 1119462, 1839406, 1905871, 1971393, 0, 0, 0, 0, 0, 0, 0, 134915, 201349, 266987, 332018, 528055, 593984, 659382, 1053239, 1119335, 1184454, 1249200, 1970739, 2036307, 2101203, 2887290, 0, 0, 0, 0, 0, 199827, 266074, 331738, 397071, 462231, 527900, 593258, 1118048, 1183926, 1249267, 1314188, 1379313, 1969268, 2035539, 2101108, 2166268, 2952396, 3017851, 0, 0, 265280, 331285, 396920, 462158, 527552, 592602, 1117064, 1183313, 1249024, 1314325, 1379520, 1444642, 2034821, 2100773, 2166353, 2231360, 2296538, 2952015, 3017861, 3083144, 331033, 396917, 462351, 527528, 1182748, 1248773, 1314543, 1379958, 1445042, 1510060, 2100248, 2166315, 2231835, 2297239, 2362524, 3017533, 3083367, 3148586, 3213996, 3279557, 330920, 396815, 462453, 527641, 1182380, 1248434, 1314422, 1380079, 1445381, 1510428, 2165916, 2231703, 2297371, 2362923, 2427928, 3082949, 3148460, 3214122, 3279975, 3345213, 264922, 330944, 396622, 462456, 527893, 592960, 1248034, 1313984, 1379861, 1445632, 1510993, 1575816, 2231002, 2296896, 2362961, 2428453, 2493573, 3279752, 3345541, 3410767, 265578, 331292, 396695, 462607, 528346, 593754, 658579, 1313777, 1379724, 1445875, 1511606, 1576800, 2362876, 2428788, 2494291, 2559092, 3345531, 3411148, 0, 0, 200630, 266304, 331447, 528626, 594667, 660101, 724739, 1445808, 1512134, 1578087, 1643063, 2428883, 2495059, 2560563, 3477114, 0, 0, 0, 0, 0, 135733, 201211, 594353, 660811, 726390, 791167, 1578214, 1644393, 1709348, 1773734, 2561217, 2626767, 2691374, 0, 0, 0, 0, 0, 0, 0, 3527, 69293, 134658, 658729, 725071, 790732, 855950, 1642539, 1708519, 1773772, 2559159, 2625579, 2691151, 3607849, 10948098, 11865773, 12783047, 0, 0, 0, 3679, 69043, 724215, 790373, 856031, 1641418, 1707918, 1773716, 2624793, 2690927, 3607496, 10948300, 11799960, 11866107, 11931108, 11996187, 12717491, 12783453, 0, 0, 4068, 69555, 922287, 987722, 1052347, 1840146, 1905255, 2757196, 2821828, 9179504, 10097456, 10162076, 11014634, 11079605, 11931462, 0, 0, 0, 0, 0, 69798, 135470, 922239, 988452, 1053903, 1840502, 1906537, 1971393, 2757963, 2823398, 3674545, 9179643, 10097205, 0, 0, 0, 0, 0, 0, 0, 201005, 987367, 1053978, 1119546, 1184189, 1905946, 1972173, 2037089, 2756909, 2823482, 2889057, 2953736, 3740093, 0, 0, 0, 0, 0, 0, 0, 266084, 331604, 1118795, 1184430, 1249536, 1970763, 2036830, 2102035, 2166582, 2821748, 2888277, 2953933, 3018855, 3805124, 3870357, 0, 0, 0, 0, 0, 265411, 331396, 397006, 1117671, 1183892, 1249512, 1314436, 2035914, 2101741, 2166932, 2231491, 2887136, 2953362, 3018954, 3083751, 3804396, 3870176, 0, 0, 0, 331148, 397071, 462231, 527345, 1183228, 1249267, 1314778, 1379868, 2034811, 2101108, 2166966, 2232154, 2297194, 2952396, 3018579, 3084128, 3148947, 3935348, 0, 0, 397003, 462411, 1248884, 1315000, 1380379, 1445069, 2100378, 2166851, 2232755, 2298108, 2362934, 3018079, 3084393, 3149891, 3215240, 3280409, 4001120, 0, 0, 0, 396875, 462539, 1248461, 1314843, 1380536, 1445492, 2166326, 2232572, 2298291, 2363459, 2428058, 3083801, 3149704, 3215427, 3281001, 3345759, 4197728, 0, 0, 0, 330737, 396695, 462607, 527756, 1314332, 1380314, 1445875, 1510908, 2231658, 2297690, 2363574, 2428788, 2493563, 3214483, 3280736, 3346259, 3411148, 4263028, 0, 0, 462542, 528004, 593091, 1379972, 1446120, 1511572, 1576423, 2297027, 2363540, 2429421, 2494666, 3280359, 3346634, 3412114, 3476960, 4328928, 4394220, 0, 0, 0, 528212, 593764, 1446144, 1512110, 1577547, 2363190, 2429715, 2495582, 2560587, 3346535, 3412685, 3478101, 3542644, 4329109, 4394948, 0, 0, 0, 0, 0, 659757, 1511869, 1578298, 1643802, 1708263, 2495841, 2561997, 2626842, 3412488, 3478881, 3544378, 3608877, 4460989, 0, 0, 0, 0, 0, 0, 0, 725294, 790694, 1643727, 1709348, 1774207, 2561217, 2627433, 2692470, 3544294, 3609931, 4526513, 10031611, 10949173, 0, 0, 0, 0, 0, 0, 0, 790451, 856036, 1642171, 1708618, 1774255, 2626151, 2692114, 3542724, 3609164, 10031472, 10882972, 10949424, 11800501, 11866602, 12783430, 0, 0, 0, 0, 0, 921014, 986670, 1839510, 1905036, 1969644, 2757451, 2822530, 3674554, 3739269, 4591212, 7343561, 8261870, 9179752, 9244378, 10096834, 10161873, 0, 0, 0, 0, 1838851, 1905207, 1970739, 2035322, 2757253, 2823271, 2888275, 3674859, 3740358, 3805139, 4591858, 4657072, 7343799, 8261696, 9179062, 0, 0, 0, 0, 0, 1117812, 1970763, 2036309, 2101188, 2822731, 2888798, 2953933, 3018389, 3673956, 3740334, 3805971, 3870823, 4591444, 4657408, 4722486, 0, 0, 0, 0, 0, 1183623, 1249096, 1970011, 2036406, 2101949, 2166663, 2888374, 2954309, 3019446, 3739527, 3805885, 3871414, 3936091, 4656968, 4722567, 0, 0, 0, 0, 0, 1183542, 1249536, 1314644, 2035815, 2102035, 2167470, 2232164, 2887317, 2953933, 3019870, 3084875, 3805124, 3871317, 3936843, 4787828, 0, 0, 0, 0, 0, 1249200, 1315058, 1380023, 2101203, 2167494, 2233067, 2297920, 3019347, 3085415, 3150469, 3215286, 3870330, 3936819, 4002359, 4067075, 0, 0, 0, 0, 0, 1314412, 1379785, 2166405, 2232762, 2298094, 3084674, 3150667, 3215976, 3280602, 3935724, 4002188, 4067734, 4133058, 4198097, 4918830, 4984246, 0, 0, 0, 0, 1314249, 1379948, 2232558, 2298298, 2363013, 3083994, 3150440, 3216203, 3281282, 4001489, 4067522, 4133270, 4198796, 4263404, 5049782, 5115438, 0, 0, 0, 0, 1314487, 1380594, 1445808, 2232384, 2298603, 2364102, 2428883, 3149750, 3216005, 3282023, 3347027, 4132611, 4198967, 4264499, 4329082, 0, 0, 0, 0, 0, 1380180, 1446144, 1511222, 2297700, 2364078, 2429715, 2494567, 3281483, 3347550, 3412685, 3477141, 4264523, 4330069, 4394948, 5246580, 0, 0, 0, 0, 0, 1445704, 1511303, 2363271, 2429629, 2495158, 2559835, 3347126, 3413061, 3478198, 4263771, 4330166, 4395709, 4460423, 5312391, 5377864, 0, 0, 0, 0, 0, 1576564, 2428868, 2495061, 2560587, 3346069, 3412685, 3478622, 3543627, 4329575, 4395795, 4461230, 4525924, 5312310, 5378304, 5443412, 0, 0, 0, 0, 0, 2494074, 2560563, 2626103, 2690819, 3478099, 3544167, 3609221, 4394963, 4461254, 4526827, 5377968, 5443826, 8195767, 9113664, 10031030, 0, 0, 0, 0, 0, 1707566, 1772982, 2559468, 2625932, 2691478, 3543426, 3609419, 4460165, 4526522, 5443180, 8195529, 9113838, 9965274, 10031720, 10882769, 10948802, 0, 0, 0, 0, 1903968, 2756675, 2822249, 2887007, 3674547, 3739715, 3804314, 4591800, 4656756, 5508811, 6426187, 7344155, 7408845, 8261884, 8326710, 9179016, 9244185, 0, 0, 0, 1969268, 2755731, 2821984, 2887507, 2952396, 3673946, 3739830, 3805044, 3869819, 4591578, 4657139, 4722172, 5508879, 5574028, 6426007, 6491121, 7343644, 8260970, 0, 0, 2035168, 2100460, 2821607, 2887882, 2953362, 3018208, 3673283, 3739796, 3805677, 3870922, 4591236, 4657384, 4722836, 4787687, 5508814, 5574276, 5639363, 0, 0, 0, 2035349, 2101188, 2887783, 2953933, 3019349, 3083892, 3739446, 3805971, 3871838, 3936843, 4657408, 4723374, 4788811, 5574484, 5640036, 0, 0, 0, 0, 0, 2167229, 2953736, 3020129, 3085626, 3150125, 3872097, 3938253, 4003098, 4723133, 4789562, 4855066, 4919527, 5706029, 0, 0, 0, 0, 0, 0, 0, 2232753, 3085542, 3151179, 3215867, 3937473, 4003689, 4068726, 4133429, 4854991, 4920612, 4985471, 5771566, 5836966, 0, 0, 0, 0, 0, 0, 0, 3083972, 3150412, 3215728, 4002407, 4068370, 4133680, 4198300, 4853435, 4919882, 4985519, 5050858, 5115829, 5836723, 5902308, 5967686, 0, 0, 0, 0, 0, 3150192, 3215948, 3280580, 4001692, 4068144, 4133906, 4199015, 4919221, 4985322, 5051055, 5116490, 5181115, 5902150, 5967844, 6033331, 0, 0, 0, 0, 0, 2298289, 3150331, 3216715, 3282150, 4067893, 4134262, 4200297, 4265153, 5051007, 5117220, 5182671, 6033574, 6099246, 0, 0, 0, 0, 0, 0, 0, 2363837, 3215661, 3282234, 3347809, 3412488, 4199706, 4265933, 4330849, 5116135, 5182746, 5248314, 5312957, 6164781, 0, 0, 0, 0, 0, 0, 0, 2428868, 2494101, 3280500, 3347029, 3412685, 3477607, 4264523, 4330590, 4395795, 4460342, 5247563, 5313198, 5378304, 6229860, 6295380, 0, 0, 0, 0, 0, 2428140, 2493920, 3345888, 3412114, 3477706, 3542503, 4329674, 4395501, 4460692, 4525251, 5246439, 5312660, 5378280, 5443204, 6229187, 6295172, 6360782, 0, 0, 0, 2559092, 3411148, 3477331, 3542880, 3607699, 4328571, 4394868, 4460726, 4525914, 5311996, 5378035, 5443546, 6294924, 6360847, 7212017, 7277975, 8195612, 9112938, 0, 0, 2624864, 3476831, 3543145, 3608643, 4394138, 4460611, 4526515, 5377652, 5443768, 6360779, 7278155, 8129741, 8196123, 9047606, 9113852, 9965081, 10030984, 0, 0, 0, 2755370, 2821223, 2886461, 3673627, 3739179, 3804184, 4591343, 4656645, 4721692, 5508725, 5573913, 6426127, 6491304, 7343734, 7408818, 7473836, 8261015, 8326300, 9177772, 9243333, 2821000, 2886789, 2952015, 3673152, 3739217, 3804709, 3869829, 4591125, 4656896, 4722257, 4787080, 5508728, 5574165, 5639232, 6425934, 6491328, 6556378, 7343296, 7408418, 8260314, 2886779, 2952396, 3739132, 3805044, 3870547, 3935348, 4590988, 4657139, 4722870, 4788064, 5508879, 5574618, 5640026, 5704851, 6426007, 6491676, 6557034, 7343089, 0, 0, 3018362, 3805139, 3871315, 3936819, 4657072, 4723398, 4789351, 4854327, 5574898, 5640939, 5706373, 5771011, 6491831, 6557760, 6623158, 0, 0, 0, 0, 0, 3937473, 4003023, 4067630, 4789478, 4855657, 4920612, 4984998, 5640625, 5707083, 5772662, 5837439, 6623739, 6689333, 0, 0, 0, 0, 0, 0, 0, 3149097, 3935415, 4001835, 4067407, 4132354, 4853803, 4919783, 4985036, 5050029, 5705001, 5771343, 5837004, 5902222, 5967303, 6688258, 6753965, 6819271, 0, 0, 0, 3148744, 4001049, 4067183, 4132556, 4852682, 4919182, 4984980, 5050363, 5115288, 5770487, 5836645, 5902303, 5967709, 6032819, 6753715, 6819423, 6884836, 6949915, 0, 0, 3214280, 4067020, 4132719, 4197657, 4918680, 4984827, 5050516, 5115790, 5180362, 5836211, 5902173, 5967839, 6033253, 6098167, 6753307, 6819300, 6884959, 6950323, 0, 0, 3214633, 4066818, 4132943, 4198443, 4263095, 4984493, 5050572, 5116391, 5181483, 5901767, 5967758, 6033612, 6099023, 6163753, 6884807, 6950573, 7015938, 0, 0, 0, 4133166, 4199631, 4265153, 5050534, 5117220, 5183337, 5248230, 6034047, 6100342, 6165835, 6230449, 7017013, 7082491, 0, 0, 0, 0, 0, 0, 0, 3346042, 4264499, 4330067, 4394963, 5182007, 5248103, 5313222, 5377968, 6098691, 6165125, 6230763, 6295794, 7081910, 7147584, 7212727, 0, 0, 0, 0, 0, 3411148, 3476603, 4263028, 4329299, 4394868, 4460028, 5246816, 5312694, 5378035, 5442956, 6163603, 6229850, 6295514, 6360847, 7146858, 7212572, 7277975, 8195057, 0, 0, 3410767, 3476613, 3541896, 4328581, 4394533, 4460113, 4525120, 5245832, 5312081, 5377792, 5443093, 6229056, 6295061, 6360696, 7146202, 7212224, 7277902, 8129314, 8195264, 9112282, 3476285, 3542119, 3607338, 4394008, 4460075, 4525595, 5311516, 5377541, 5443311, 6294809, 6360693, 7212200, 7278095, 8063660, 8129714, 8195702, 9047196, 9112983, 9964229, 10029740, 3673622, 3739440, 3804651, 4591696, 4657217, 4722480, 5509338, 5574736, 5639702, 6426741, 6492125, 6557084, 7344093, 7409394, 7474613, 8261020, 8326581, 0, 0, 0, 3738652, 3804184, 3869501, 4590873, 4656645, 4722219, 4787303, 5508725, 5574383, 5639707, 5704490, 6426127, 6491766, 6557079, 6621868, 7343272, 7408818, 7474332, 7539397, 8325804, 3804314, 3870047, 4656756, 4722755, 4788329, 4853088, 5508811, 5574840, 5640627, 5705795, 6426187, 6492187, 6557948, 6623112, 7408845, 7474742, 7540249, 0, 0, 0, 3935724, 4722309, 4788610, 4854156, 4918830, 5574252, 5640634, 5706571, 5771670, 5836214, 6491593, 6557934, 6623848, 6688962, 7540442, 7605969, 0, 0, 0, 0, 4001467, 4787908, 4854375, 4919882, 4984755, 5706316, 5772306, 5837487, 5902308, 6623600, 6689584, 6754794, 6819654, 7606172, 7671733, 0, 0, 0, 0, 0, 4000714, 4066551, 4853017, 4919182, 4984677, 5049779, 5704648, 5771119, 5836948, 5902303, 5967455, 6688460, 6754299, 6819677, 6884836, 7671192, 7736755, 7801883, 0, 0, 4066360, 4131775, 4918634, 4984519, 5049934, 5114850, 5770296, 5836487, 5902191, 5967620, 6032738, 6687679, 6753870, 6819588, 6885018, 6950126, 7670754, 7736674, 7802094, 0, 4066239, 4131896, 4918242, 4984398, 5050055, 5115242, 5836130, 5902084, 5967727, 6033095, 6097976, 6753518, 6819482, 6885124, 6950478, 7015359, 7736558, 7802210, 7867362, 0, 4132087, 4197322, 4984243, 5050213, 5115790, 5180697, 5901919, 5967839, 6033556, 6098799, 6163400, 6819300, 6885213, 6950907, 7016140, 7736347, 7802291, 7867800, 0, 0, 4198075, 5050291, 5116490, 5182055, 5246660, 5967844, 6034095, 6099986, 6165068, 6885190, 6951402, 7017264, 7082352, 7868341, 7933852, 0, 0, 0, 0, 0, 4263404, 5115438, 5181836, 5247362, 5312133, 6032822, 6099350, 6165323, 6230458, 6295148, 7016642, 7082600, 7147758, 7212489, 7933649, 7999194, 0, 0, 0, 0, 4328799, 4394138, 5180768, 5247081, 5312579, 5377652, 6164547, 6230451, 6295736, 6360779, 7081864, 7147772, 7213083, 7278155, 7999001, 8064566, 8129741, 0, 0, 0, 4328253, 4394008, 4459548, 5246055, 5312043, 5377541, 5442841, 6163242, 6229531, 6295279, 6360693, 7080620, 7146903, 7212662, 7278095, 7998149, 8064156, 8129714, 8195240, 9046700, 4394475, 4460336, 4525590, 5312304, 5378113, 5443664, 6229526, 6295632, 6361306, 7146908, 7213021, 7278709, 8064437, 8130290, 8196061, 9047477, 9112988, 0, 0, 0, 3673500, 3739061, 4591581, 4656882, 4722101, 5509237, 5574621, 5639580, 6426842, 6492240, 6557206, 7344208, 7409729, 7474992, 8261142, 8326960, 8392171, 0, 0, 0, 3738284, 4590760, 4656306, 4721820, 4786885, 5508623, 5574262, 5639575, 5704364, 6426229, 6491887, 6557211, 6621994, 7343385, 7409157, 7474731, 7539815, 8326172, 8391704, 8457021, 4656333, 4722230, 4787737, 5508683, 5574683, 5640444, 5705608, 6426315, 6492344, 6558131, 6623299, 7409268, 7475267, 7540841, 7605600, 8391834, 8457567, 0, 0, 0, 4787930, 4853457, 5574089, 5640430, 5706344, 5771458, 6491756, 6558138, 6624075, 6689174, 6753718, 7474821, 7541122, 7606668, 7671342, 8523244, 0, 0, 0, 0, 4853660, 4919221, 5706096, 5772080, 5837290, 5902150, 6623820, 6689810, 6754991, 6819812, 7540420, 7606887, 7672394, 7737267, 8588987, 0, 0, 0, 0, 0, 4918680, 4984243, 5049371, 5770956, 5836795, 5902173, 5967332, 6622152, 6688623, 6754452, 6819807, 6884959, 7605529, 7671694, 7737189, 7802291, 8588234, 8654071, 0, 0, 4918242, 4984162, 5049582, 5770175, 5836366, 5902084, 5967514, 6032622, 6687800, 6753991, 6819695, 6885124, 6950242, 7671146, 7737031, 7802446, 7867362, 8653880, 8719295, 0, 4984046, 5049698, 5114850, 5836014, 5901978, 5967620, 6032974, 6097855, 6753634, 6819588, 6885231, 6950599, 7015480, 7670754, 7736910, 7802567, 7867754, 8653759, 8719416, 0, 4983835, 5049779, 5115288, 5901796, 5967709, 6033403, 6098636, 6819423, 6885343, 6951060, 7016303, 7080904, 7736755, 7802725, 7868302, 7933209, 8719607, 8784842, 0, 0, 5115829, 5181340, 5967686, 6033898, 6099760, 6164848, 6885348, 6951599, 7017490, 7082572, 7802803, 7869002, 7934567, 7999172, 8785595, 0, 0, 0, 0, 0, 5181137, 5246682, 6099138, 6165096, 6230254, 6294985, 6950326, 7016854, 7082827, 7147962, 7212652, 7867950, 7934348, 7999874, 8064645, 8850924, 0, 0, 0, 0, 5246489, 5312054, 5377229, 6164360, 6230268, 6295579, 6360651, 7082051, 7147955, 7213240, 7278283, 7933280, 7999593, 8065091, 8130164, 8916319, 8981658, 0, 0, 0, 4459180, 5245637, 5311644, 5377202, 5442728, 6163116, 6229399, 6295158, 6360591, 7080746, 7147035, 7212783, 7278197, 7998567, 8064555, 8130053, 8195353, 8915773, 8981528, 9047068, 4459957, 4525468, 5311925, 5377778, 5443549, 6229404, 6295517, 6361205, 7147030, 7213136, 7278810, 8064816, 8130625, 8196176, 8981995, 9047856, 9113110, 0, 0, 0, 2755244, 2820805, 3673495, 3738780, 4591222, 4656306, 4721324, 5508623, 5573800, 6426229, 6491417, 7343855, 7409157, 7474204, 8261147, 8326699, 8391704, 9177898, 9243751, 9308989, 3672794, 4590784, 4655906, 5508430, 5573824, 5638874, 6426232, 6491669, 6556736, 7343637, 7409408, 7474769, 7539592, 8260672, 8326737, 8392229, 8457349, 9243528, 9309317, 9374543, 4590577, 5508503, 5574172, 5639530, 6426383, 6492122, 6557530, 6622355, 7343500, 7409651, 7475382, 7540576, 8326652, 8392564, 8458067, 8522868, 9309307, 9374924, 0, 0, 5574327, 5640256, 5705654, 6492402, 6558443, 6623877, 6688515, 7409584, 7475910, 7541863, 7606839, 8392659, 8458835, 8524339, 9440890, 0, 0, 0, 0, 0, 5706235, 5771829, 6558129, 6624587, 6690166, 6754943, 7541990, 7608169, 7673124, 7737510, 8524993, 8590543, 8655150, 0, 0, 0, 0, 0, 0, 0, 5770754, 5836461, 5901767, 6622505, 6688847, 6754508, 6819726, 6884807, 7606315, 7672295, 7737548, 7802541, 8522935, 8589355, 8654927, 8719874, 9571625, 0, 0, 0, 5836211, 5901919, 5967332, 6032411, 6687991, 6754149, 6819807, 6885213, 6950323, 7605194, 7671694, 7737492, 7802875, 7867800, 8588569, 8654703, 8720076, 9571272, 0, 0, 5835803, 5901796, 5967455, 6032819, 6753715, 6819677, 6885343, 6950757, 7015671, 7671192, 7737339, 7803028, 7868302, 7932874, 8654540, 8720239, 8785177, 9636808, 0, 0, 5967303, 6033069, 6098434, 6819271, 6885262, 6951116, 7016527, 7081257, 7737005, 7803084, 7868903, 7933995, 8654338, 8720463, 8785963, 8850615, 9637161, 0, 0, 0, 6099509, 6164987, 6951551, 7017846, 7083339, 7147953, 7803046, 7869732, 7935849, 8000742, 8720686, 8787151, 8852673, 0, 0, 0, 0, 0, 0, 0, 6164406, 6230080, 6295223, 7016195, 7082629, 7148267, 7213298, 7934519, 8000615, 8065734, 8130480, 8852019, 8917587, 8982483, 9768570, 0, 0, 0, 0, 0, 5442545, 6229354, 6295068, 6360471, 7081107, 7147354, 7213018, 7278351, 7999328, 8065206, 8130547, 8195468, 8850548, 8916819, 8982388, 9047548, 9833676, 9899131, 0, 0, 4524762, 5376802, 5442752, 6228698, 6294720, 6360398, 7146560, 7212565, 7278200, 7998344, 8064593, 8130304, 8195605, 8916101, 8982053, 9047633, 9112640, 9833295, 9899141, 9964424, 3541701, 3607212, 4459676, 4525463, 5311148, 5377202, 5443190, 6294696, 6360591, 7212313, 7278197, 8064028, 8130053, 8195823, 8981528, 9047595, 9113115, 9898813, 9964647, 10029866, 2756488, 2821657, 3674364, 3739190, 4591643, 4656333, 5508683, 6426315, 7344312, 7409268, 8262067, 8327235, 8391834, 9179203, 9244777, 9309535, 10161504, 0, 0, 0, 3673450, 4591132, 5508503, 5573617, 6426383, 6491532, 7344090, 7409651, 7474684, 8261466, 8327350, 8392564, 8457339, 9178259, 9244512, 9310035, 9374924, 10226804, 0, 0, 6426318, 6491780, 6556867, 7343748, 7409896, 7475348, 7540199, 8260803, 8327316, 8393197, 8458442, 9244135, 9310410, 9375890, 9440736, 10292704, 10357996, 0, 0, 0, 6491988, 6557540, 7409920, 7475886, 7541323, 8326966, 8393491, 8459358, 8524363, 9310311, 9376461, 9441877, 9506420, 10292885, 10358724, 0, 0, 0, 0, 0, 6623533, 7475645, 7542074, 7607578, 7672039, 8459617, 8525773, 8590618, 9376264, 9442657, 9508154, 9572653, 10424765, 0, 0, 0, 0, 0, 0, 0, 6689070, 6754470, 7607503, 7673124, 7737983, 8524993, 8591209, 8656246, 8720949, 9508070, 9573707, 9638395, 10490289, 0, 0, 0, 0, 0, 0, 0, 6754227, 6819812, 6885190, 7605947, 7672394, 7738031, 7803370, 7868341, 8589927, 8655890, 8721200, 8785820, 9506500, 9572940, 9638256, 0, 0, 0, 0, 0, 6819654, 6885348, 6950835, 7671733, 7737834, 7803567, 7869002, 7933627, 8589212, 8655664, 8721426, 8786535, 9572720, 9638476, 9703108, 0, 0, 0, 0, 0, 6951078, 7016750, 7803519, 7869732, 7935183, 8655413, 8721782, 8787817, 8852673, 9572859, 9639243, 9704678, 10555825, 0, 0, 0, 0, 0, 0, 0, 7082285, 7868647, 7935258, 8000826, 8065469, 8787226, 8853453, 8918369, 9638189, 9704762, 9770337, 9835016, 10621373, 0, 0, 0, 0, 0, 0, 0, 7147364, 7212884, 8000075, 8065710, 8130816, 8852043, 8918110, 8983315, 9047862, 9703028, 9769557, 9835213, 9900135, 10686404, 10751637, 0, 0, 0, 0, 0, 7146691, 7212676, 7278286, 7998951, 8065172, 8130792, 8195716, 8917194, 8983021, 9048212, 9112771, 9768416, 9834642, 9900234, 9965031, 10685676, 10751456, 0, 0, 0, 4525418, 5443100, 6294513, 6360471, 7212428, 7278351, 8064508, 8130547, 8196058, 8916091, 8982388, 9048246, 9113434, 9833676, 9899859, 9965408, 10030227, 10816628, 0, 0, 3542553, 3608456, 4460086, 4526332, 5377229, 5443611, 6360651, 7278283, 8130164, 8196280, 8981658, 9048131, 9114035, 9899359, 9965673, 10031171, 10882400, 0, 0, 0, 1839298, 1904337, 2757224, 2821850, 3674350, 4591049, 7343724, 8262074, 8326789, 9179979, 9245058, 10097046, 10162572, 10227180, 11013558, 11079214, 0, 0, 0, 0, 2756534, 3674176, 4591287, 7344370, 7409584, 8262379, 8327878, 8392659, 9179781, 9245799, 9310803, 10096387, 10162743, 10228275, 10292858, 0, 0, 0, 0, 0, 7343956, 7409920, 7474998, 8261476, 8327854, 8393491, 8458343, 9245259, 9311326, 9376461, 9440917, 10228299, 10293845, 10358724, 11210356, 0, 0, 0, 0, 0, 7409480, 7475079, 8327047, 8393405, 8458934, 8523611, 9310902, 9376837, 9441974, 10227547, 10293942, 10359485, 10424199, 11276167, 11341640, 0, 0, 0, 0, 0, 7540340, 8392644, 8458837, 8524363, 9309845, 9376461, 9442398, 9507403, 10293351, 10359571, 10425006, 10489700, 11276086, 11342080, 11407188, 0, 0, 0, 0, 0, 8457850, 8524339, 8589879, 8654595, 9441875, 9507943, 9572997, 9637814, 10358739, 10425030, 10490603, 10555456, 11341744, 11407602, 11472567, 0, 0, 0, 0, 0, 7671342, 7736758, 8523244, 8589708, 8655254, 8720578, 8785617, 9507202, 9573195, 9638504, 9703130, 10423941, 10490298, 10555630, 11406956, 11472329, 0, 0, 0, 0, 7802294, 7867950, 8589009, 8655042, 8720790, 8786316, 8850924, 9506522, 9572968, 9638731, 9703810, 10490094, 10555834, 10620549, 11406793, 11472492, 0, 0, 0, 0, 8720131, 8786487, 8852019, 8916602, 9572278, 9638533, 9704551, 9769555, 10489920, 10556139, 10621638, 10686419, 11407031, 11473138, 11538352, 0, 0, 0, 0, 0, 7999092, 8852043, 8917589, 8982468, 9704011, 9770078, 9835213, 9899669, 10555236, 10621614, 10687251, 10752103, 11472724, 11538688, 11603766, 0, 0, 0, 0, 0, 8064903, 8130376, 8851291, 8917686, 8983229, 9047943, 9769654, 9835589, 9900726, 10620807, 10687165, 10752694, 10817371, 11538248, 11603847, 0, 0, 0, 0, 0, 8064822, 8130816, 8195924, 8917095, 8983315, 9048750, 9113444, 9768597, 9835213, 9901150, 9966155, 10686404, 10752597, 10818123, 11669108, 0, 0, 0, 0, 0, 3608502, 4526144, 5443255, 8130480, 8196338, 8982483, 9048774, 9114347, 9900627, 9966695, 10031749, 10751610, 10818099, 10883639, 10948355, 0, 0, 0, 0, 0, 2625233, 2691266, 3542746, 3609192, 4526318, 5443017, 8195692, 9047685, 9114042, 9965954, 10031947, 10817004, 10883468, 10949014, 11800110, 11865526, 0, 0, 0, 0, 3910, 922090, 987061, 1839920, 1904540, 2756976, 9179724, 9244356, 10097682, 10162791, 11014831, 11080266, 11144891, 11931620, 11997107, 0, 0, 0, 0, 0, 1839669, 2757115, 8262065, 9180491, 9245926, 10098038, 10164073, 10228929, 11014783, 11080996, 11146447, 11997350, 12063022, 0, 0, 0, 0, 0, 0, 0, 8327613, 9179437, 9246010, 9311585, 9376264, 10163482, 10229709, 10294625, 11079911, 11146522, 11212090, 11276733, 12128557, 0, 0, 0, 0, 0, 0, 0, 8392644, 8457877, 9244276, 9310805, 9376461, 9441383, 10228299, 10294366, 10359571, 10424118, 11211339, 11276974, 11342080, 12193636, 12259156, 0, 0, 0, 0, 0, 8391916, 8457696, 9309664, 9375890, 9441482, 9506279, 10293450, 10359277, 10424468, 10489027, 11210215, 11276436, 11342056, 11406980, 12192963, 12258948, 12324558, 0, 0, 0, 8522868, 9374924, 9441107, 9506656, 9571475, 10292347, 10358644, 10424502, 10489690, 10554730, 11275772, 11341811, 11407322, 11472412, 12258700, 12324623, 12389783, 12454897, 0, 0, 8588640, 9440607, 9506921, 9572419, 9637768, 9702937, 10357914, 10424387, 10490291, 10555644, 10620470, 11341428, 11407544, 11472923, 11537613, 12324555, 12389963, 0, 0, 0, 8785248, 9506329, 9572232, 9637955, 9703529, 9768287, 10423862, 10490108, 10555827, 10620995, 10685594, 11341005, 11407387, 11473080, 11538036, 12324427, 12390091, 0, 0, 0, 8850548, 9637011, 9703264, 9768787, 9833676, 10489194, 10555226, 10621110, 10686324, 10751099, 11406876, 11472858, 11538419, 11603452, 12258289, 12324247, 12390159, 12455308, 0, 0, 8916448, 8981740, 9702887, 9769162, 9834642, 9899488, 10554563, 10621076, 10686957, 10752202, 11472516, 11538664, 11604116, 11668967, 12390094, 12455556, 12520643, 0, 0, 0, 8916629, 8982468, 9769063, 9835213, 9900629, 9965172, 10620726, 10687251, 10753118, 10818123, 11538688, 11604654, 11670091, 12455764, 12521316, 0, 0, 0, 0, 0, 9048509, 9835016, 9901409, 9966906, 10031405, 10753377, 10819533, 10884378, 11604413, 11670842, 11736346, 11800807, 12587309, 0, 0, 0, 0, 0, 0, 0, 2691637, 3609083, 9114033, 9966822, 10032459, 10818753, 10884969, 10950006, 11736271, 11801892, 11866751, 12652846, 12718246, 0, 0, 0, 0, 0, 0, 0, 855878, 1707957, 1774058, 2625436, 2691888, 3608944, 9965252, 10031692, 10883687, 10949650, 11734715, 11801162, 11866799, 12718003, 12783588, 0, 0, 0, 0, 0, 3933, 69043, 789531, 855524, 921595, 986520, 1838796, 9178056, 10096495, 10161433, 11014292, 11079566, 11144138, 11931615, 11997029, 12061943, 12717491, 12783199, 0, 0, 3527, 921261, 1838594, 9178409, 10096719, 10162219, 10226871, 11014348, 11080167, 11145259, 11931534, 11997388, 12062799, 12127529, 12652034, 12717741, 12783047, 0, 0, 0, 10096942, 10163407, 10228929, 11014310, 11080996, 11147113, 11212006, 11997823, 12064118, 12129611, 12194225, 12587515, 12653109, 0, 0, 0, 0, 0, 0, 0, 9309818, 10228275, 10293843, 10358739, 11145783, 11211879, 11276998, 11341744, 12062467, 12128901, 12194539, 12259570, 12455607, 12521536, 12586934, 0, 0, 0, 0, 0, 9374924, 9440379, 10226804, 10293075, 10358644, 10423804, 11210592, 11276470, 11341811, 11406732, 11471857, 12127379, 12193626, 12259290, 12324623, 12389783, 12455452, 12520810, 0, 0, 9374543, 9440389, 9505672, 10292357, 10358309, 10423889, 10488896, 10554074, 11209608, 11275857, 11341568, 11406869, 11472064, 11537186, 12192832, 12258837, 12324472, 12389710, 12455104, 12520154, 9440061, 9505895, 9571114, 9636524, 9702085, 10357784, 10423851, 10489371, 10554775, 10620060, 11275292, 11341317, 11407087, 11472502, 11537586, 11602604, 12258585, 12324469, 12389903, 12455080, 9505477, 9570988, 9636650, 9702503, 9767741, 10423452, 10489239, 10554907, 10620459, 10685464, 11274924, 11340978, 11406966, 11472623, 11537925, 11602972, 12258472, 12324367, 12390005, 12455193, 9702280, 9768069, 9833295, 10488538, 10554432, 10620497, 10685989, 10751109, 11340578, 11406528, 11472405, 11538176, 11603537, 11668360, 12192474, 12258496, 12324174, 12390008, 12455445, 12520512, 9768059, 9833676, 10620412, 10686324, 10751827, 10816628, 11406321, 11472268, 11538419, 11604150, 11669344, 12193130, 12258844, 12324247, 12390159, 12455898, 12521306, 12586131, 0, 0, 9899642, 10686419, 10752595, 10818099, 11538352, 11604678, 11670631, 11735607, 12128182, 12193856, 12258999, 12456178, 12522219, 12587653, 12652291, 0, 0, 0, 0, 0, 10818753, 10884303, 10948910, 11670758, 11736937, 11801892, 11866278, 12063285, 12128763, 12521905, 12588363, 12653942, 12718719, 0, 0, 0, 0, 0, 0, 0, 855495, 1773229, 2690562, 10030377, 10816695, 10883115, 10948687, 11735083, 11801063, 11866316, 11931079, 11996845, 12062210, 12586281, 12652623, 12718284, 12783502, 0, 0, 0, 3556, 68635, 789939, 855901, 1707416, 1773563, 2690764, 10030024, 10882329, 10948463, 11733962, 11800462, 11866260, 11931231, 11996595, 12651767, 12717925, 12783583, 0, 0, 3844, 68962, 789742, 855706, 921166, 986082, 1772782, 1838015, 10095672, 11013831, 11079018, 11799522, 11865442, 11931503, 11996871, 12061752, 12651455, 12717646, 12783364, 0, 3679, 855524, 921011, 1772571, 10095863, 10161098, 11013989, 11079566, 11144473, 11799960, 11865523, 11931615, 11997332, 12062575, 12127176, 12652236, 12718075, 12783453, 0, 0, 10161851, 11014067, 11080266, 11145831, 11210436, 11734940, 11800501, 11931620, 11997871, 12063762, 12128844, 12587376, 12653360, 12718570, 12783430, 0, 0, 0, 0, 0, 10227180, 11079214, 11145612, 11211138, 11275909, 11669210, 11734737, 11996598, 12063126, 12129099, 12194234, 12258924, 12455369, 12521710, 12587624, 12652738, 0, 0, 0, 0, 10292575, 10357914, 11144544, 11210857, 11276355, 11341428, 11537613, 11603510, 11669017, 12128323, 12194227, 12259512, 12324555, 12389963, 12455963, 12521724, 12586888, 0, 0, 0, 10292029, 10357784, 10423324, 10619564, 11209831, 11275819, 11341317, 11406617, 11472040, 11537586, 11603100, 11668165, 12127018, 12193307, 12259055, 12324469, 12389903, 12455542, 12520855, 12585644, 10358251, 10424112, 10489366, 10554780, 10620341, 11276080, 11341889, 11407440, 11472861, 11538162, 11603381, 12193302, 12259408, 12325082, 12390517, 12455901, 12520860, 0, 0, 0, 10423733, 10489244, 10554902, 10620720, 10685931, 11275701, 11341554, 11407325, 11472976, 11538497, 11603760, 12193180, 12259293, 12324981, 12390618, 12456016, 12520982, 0, 0, 0, 10422956, 10619932, 10685464, 10750781, 11209413, 11275420, 11340978, 11406504, 11472153, 11537925, 11603499, 11668583, 12126892, 12193175, 12258934, 12324367, 12390005, 12455663, 12520987, 12585770, 10685594, 10751327, 11210265, 11275830, 11341005, 11538036, 11604035, 11669609, 11734368, 12128136, 12194044, 12259355, 12324427, 12390091, 12456120, 12521907, 12587075, 0, 0, 0, 10817004, 11144913, 11210458, 11603589, 11669890, 11735436, 11800110, 12062914, 12128872, 12194030, 12258761, 12455532, 12521914, 12587851, 12652950, 12717494, 0, 0, 0, 0, 10882747, 11079605, 11145116, 11669188, 11735655, 11801162, 11866035, 11931462, 11997674, 12063536, 12128624, 12587596, 12653586, 12718767, 12783588, 0, 0, 0, 0, 0, 3556, 855647, 920603, 1772979, 10881994, 10947831, 11013555, 11079064, 11734297, 11800462, 11865957, 11931485, 11997179, 12062412, 12585928, 12652399, 12718228, 12783583, 0, 0, 3738, 68846, 789858, 855812, 920814, 1706978, 1773134, 2689983, 10947640, 11013474, 11078626, 11799914, 11865799, 11931396, 11996750, 12061631, 12651576, 12717767, 12783471, 0);

	ivec2 local_pos = ivec2(gl_LocalInvocationID).xy;

	if (local_pos == ivec2(0)) {
		invalid_rays = 0;
	}

	groupMemoryBarrier();
	barrier();

	int linear_pos = local_pos.y * OCCLUSION_OCT_SIZE + local_pos.x;

	ivec3 base_probe = (params.offset / PROBE_CELLS) + ivec3(gl_WorkGroupID.xyz);
	ivec2 probe_tex_pos = probe_to_tex(base_probe);

	ivec3 src_tex_uv = ivec3(probe_tex_pos * OCCLUSION_OCT_SIZE , params.cascade);
	ivec3 dst_tex_uv = ivec3(probe_tex_pos * (OCCLUSION_OCT_SIZE+2) + ivec2(1), params.cascade);

	float o = imageLoad(src_occlusion,src_tex_uv + ivec3(local_pos,0)).r * OCC8_DISTANCE_MAX;

	float multiplier = 1.0; // Poison the filter if backface found.

	if ((o + 0.001) > OCC8_DISTANCE_MAX) {
		o = 0.0;
		multiplier = 0.0;
		atomicAdd(invalid_rays,1);
	}

	neighbours[linear_pos] = vec3(o,o*o,multiplier);

	groupMemoryBarrier();
	barrier();

	vec2 accum = vec2(0.0);
	for(uint i=0;i<neighbour_max_weights;i++) {
		uint n = neighbour_weights[ linear_pos * neighbour_max_weights + i];
		uint index = n>>16;
		float weight = float(n&0xFFFF) / float(0xFFFF);
		accum += neighbours[index].rg * weight;
		multiplier *= neighbours[index].b;
	}

	vec2 occlusion = accum * multiplier / OCC16_DISTANCE_MAX;

	if (invalid_rays>4) {
		occlusion = vec2(0.0);
	}

	ivec3 copy_to[4] = ivec3[](ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2));
	copy_to[0] = dst_tex_uv + ivec3(local_pos,0);

	if (local_pos == ivec2(0, 0)) {
		copy_to[1] = dst_tex_uv + ivec3(OCCLUSION_OCT_SIZE - 1, -1, 0);
		copy_to[2] = dst_tex_uv + ivec3(-1, OCCLUSION_OCT_SIZE - 1, 0);
		copy_to[3] = dst_tex_uv + ivec3(OCCLUSION_OCT_SIZE, OCCLUSION_OCT_SIZE, 0);
	} else if (local_pos == ivec2(OCCLUSION_OCT_SIZE - 1, 0)) {
		copy_to[1] = dst_tex_uv + ivec3(0, -1, 0);
		copy_to[2] = dst_tex_uv + ivec3(OCCLUSION_OCT_SIZE, OCCLUSION_OCT_SIZE - 1, 0);
		copy_to[3] = dst_tex_uv + ivec3(-1, OCCLUSION_OCT_SIZE, 0);
	} else if (local_pos == ivec2(0, OCCLUSION_OCT_SIZE - 1)) {
		copy_to[1] = dst_tex_uv + ivec3(-1, 0, 0);
		copy_to[2] = dst_tex_uv + ivec3(OCCLUSION_OCT_SIZE - 1, OCCLUSION_OCT_SIZE, 0);
		copy_to[3] = dst_tex_uv + ivec3(OCCLUSION_OCT_SIZE, -1, 0);
	} else if (local_pos == ivec2(OCCLUSION_OCT_SIZE - 1, OCCLUSION_OCT_SIZE - 1)) {
		copy_to[1] = dst_tex_uv + ivec3(0, OCCLUSION_OCT_SIZE, 0);
		copy_to[2] = dst_tex_uv + ivec3(OCCLUSION_OCT_SIZE, 0, 0);
		copy_to[3] = dst_tex_uv + ivec3(-1, -1, 0);
	} else if (local_pos.y == 0) {
		copy_to[1] = dst_tex_uv + ivec3(OCCLUSION_OCT_SIZE - local_pos.x - 1, local_pos.y - 1, 0);
	} else if (local_pos.x == 0) {
		copy_to[1] = dst_tex_uv + ivec3(local_pos.x - 1, OCCLUSION_OCT_SIZE - local_pos.y - 1, 0);
	} else if (local_pos.y == OCCLUSION_OCT_SIZE - 1) {
		copy_to[1] = dst_tex_uv + ivec3(OCCLUSION_OCT_SIZE - local_pos.x - 1, local_pos.y + 1, 0);
	} else if (local_pos.x == OCCLUSION_OCT_SIZE - 1) {
		copy_to[1] = dst_tex_uv + ivec3(local_pos.x + 1, OCCLUSION_OCT_SIZE - local_pos.y - 1, 0);
	}

	for (int i = 0; i < 4; i++) {
		if (copy_to[i] == ivec3(-2, -2, -2)) {
			continue;
		}
		imageStore(dst_occlusion, copy_to[i], vec4(occlusion,vec2(0.0)));
	}

#endif


#ifdef MODE_LIGHTPROBE_SCROLL

	ivec2 local_pos = ivec2(gl_LocalInvocationID).xy;

	ivec3 base_probe = (params.offset / PROBE_CELLS) + ivec3(gl_WorkGroupID.xyz);
	ivec2 probe_tex_pos = probe_to_tex(base_probe);

	vec3 posf = vec3(base_probe) * PROBE_CELLS;

	vec3 specular_light = vec3(0.0);
	vec3 diffuse_light = vec3(0.0);
	vec3 ambient_light = vec3(0.0);

	if (params.cascade < (params.cascade_count -1)) {

		// map to parent cascade space
		posf /= cascades.data[params.cascade].to_cell;
		posf += cascades.data[params.cascade].offset;

		int upper_cascade = params.cascade + 1;
		posf -= cascades.data[upper_cascade].offset;
		posf *= cascades.data[upper_cascade].to_cell;

		ivec3 base_probe = ivec3(posf / PROBE_CELLS);

		if (local_pos == ivec2(0)) {
			// Only the first thread needs to do this.
			float occlusion_total = 0;

			vec3 probe_axis_size = vec3(params.probe_axis_size);
			vec2 occ_probe_tex_to_uv = 1.0 / vec2( (OCCLUSION_OCT_SIZE+2) * probe_axis_size.x, (OCCLUSION_OCT_SIZE+2) * probe_axis_size.y * probe_axis_size.z );

			for(int i=0;i<8;i++) {
				ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
				vec3 probe_pos = vec3(probe * PROBE_CELLS);

				vec3 probe_to_pos = posf - probe_pos;
				vec3 n = normalize(probe_to_pos);
				float d = length(probe_to_pos);

				float weight = 1.0;

				ivec2 tex_pos = probe_to_texc(probe,upper_cascade);
				vec2 tex_uv = vec2(ivec2(tex_pos * (OCCLUSION_OCT_SIZE+2) + ivec2(1))) + octahedron_encode(n) * float(OCCLUSION_OCT_SIZE);
				tex_uv *= occ_probe_tex_to_uv;
				vec2 o_o2 = texture(sampler2DArray(occlusion_probes,linear_sampler),vec3(tex_uv,float(upper_cascade))).rg * OCC16_DISTANCE_MAX;

				float mean = o_o2.x;
				float variance = abs((mean*mean) - o_o2.y);

				 // http://www.punkuser.net/vsm/vsm_paper.pdf; equation 5
				 // Need the max in the denominator because biasing can cause a negative displacement
				float dmean = max(d - mean, 0.0);
				float chebyshev_weight = variance / (variance + dmean*dmean);

				chebyshev_weight = max(pow(chebyshev_weight,3.0), 0.0);

				weight *= (d <= mean) ? 1.0 : chebyshev_weight;

				weight = max(0.000001, weight); // make sure not zero (only trilinear can be zero)

				const float crushThreshold = 0.2;
				if (weight < crushThreshold) {
				      weight *= weight * weight * (1.0 / pow(crushThreshold,2.0));
				}

				vec3 trilinear = vec3(1.0) - abs(probe_to_pos / float(PROBE_CELLS));

				weight *= trilinear.x * trilinear.y * trilinear.z;

				occlusion[i]=weight;
				occlusion_total+=weight;
			}

			for(int i=0;i<8;i++) {
				occlusion[i]/=occlusion_total;
			}

		}

		groupMemoryBarrier();
		barrier();

		for(int i=0;i<8;i++) {
			ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
			ivec2 base_tex_pos = probe_to_texc(probe,upper_cascade);
			ivec2 tex_pos = base_tex_pos * (LIGHTPROBE_OCT_SIZE+2) + ivec2(1) + local_pos;
			ivec3 tex_array_pos = ivec3(tex_pos,upper_cascade);
			specular_light += imageLoad(lightprobe_specular_data,tex_array_pos).rgb * occlusion[i];
			diffuse_light += rgbe_decode(imageLoad(lightprobe_diffuse_data,tex_array_pos).r) * occlusion[i];

			if (local_pos == ivec2(0)) {
				tex_array_pos = ivec3(base_tex_pos,upper_cascade);
				ambient_light += rgbe_decode(imageLoad(lightprobe_ambient_data,tex_array_pos).r) * occlusion[i];
			}
		}
	}

	ivec3 dst_tex_uv = ivec3(probe_tex_pos * (LIGHTPROBE_OCT_SIZE+2) + ivec2(1), params.cascade);

	ivec3 copy_to[4] = ivec3[](ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2), ivec3(-2, -2, -2));
	copy_to[0] = dst_tex_uv + ivec3(local_pos,0);

	if (local_pos == ivec2(0, 0)) {
		copy_to[1] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE - 1, -1, 0);
		copy_to[2] = dst_tex_uv + ivec3(-1, LIGHTPROBE_OCT_SIZE - 1, 0);
		copy_to[3] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE, LIGHTPROBE_OCT_SIZE, 0);
	} else if (local_pos == ivec2(LIGHTPROBE_OCT_SIZE - 1, 0)) {
		copy_to[1] = dst_tex_uv + ivec3(0, -1, 0);
		copy_to[2] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE, LIGHTPROBE_OCT_SIZE - 1, 0);
		copy_to[3] = dst_tex_uv + ivec3(-1, LIGHTPROBE_OCT_SIZE, 0);
	} else if (local_pos == ivec2(0, LIGHTPROBE_OCT_SIZE - 1)) {
		copy_to[1] = dst_tex_uv + ivec3(-1, 0, 0);
		copy_to[2] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE - 1, LIGHTPROBE_OCT_SIZE, 0);
		copy_to[3] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE, -1, 0);
	} else if (local_pos == ivec2(LIGHTPROBE_OCT_SIZE - 1, LIGHTPROBE_OCT_SIZE - 1)) {
		copy_to[1] = dst_tex_uv + ivec3(0, LIGHTPROBE_OCT_SIZE, 0);
		copy_to[2] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE, 0, 0);
		copy_to[3] = dst_tex_uv + ivec3(-1, -1, 0);
	} else if (local_pos.y == 0) {
		copy_to[1] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE - local_pos.x - 1, local_pos.y - 1, 0);
	} else if (local_pos.x == 0) {
		copy_to[1] = dst_tex_uv + ivec3(local_pos.x - 1, LIGHTPROBE_OCT_SIZE - local_pos.y - 1, 0);
	} else if (local_pos.y == LIGHTPROBE_OCT_SIZE - 1) {
		copy_to[1] = dst_tex_uv + ivec3(LIGHTPROBE_OCT_SIZE - local_pos.x - 1, local_pos.y + 1, 0);
	} else if (local_pos.x == LIGHTPROBE_OCT_SIZE - 1) {
		copy_to[1] = dst_tex_uv + ivec3(local_pos.x + 1, LIGHTPROBE_OCT_SIZE - local_pos.y - 1, 0);
	}

	for (int i = 0; i < 4; i++) {
		if (copy_to[i] == ivec3(-2, -2, -2)) {
			continue;
		}
		imageStore(lightprobe_specular_data,copy_to[i],vec4(specular_light,1.0));
		imageStore(lightprobe_diffuse_data,copy_to[i],uvec4(rgbe_encode(diffuse_light)));
	}

	if (local_pos == ivec2(0)) {
		imageStore(lightprobe_ambient_data,ivec3(probe_tex_pos,params.cascade),uvec4(rgbe_encode(ambient_light)));
	}

	// Cache invalidation
	probe_tex_pos = probe_tex_pos * LIGHTPROBE_OCT_SIZE + local_pos;
	for(int i=0;i<params.ray_hit_cache_frames;i++) {
		ivec3 history_pos = ivec3(probe_tex_pos, params.cascade * params.ray_hit_cache_frames + i);
		// Completely invalidate cache frame.
		imageStore(ray_hit_cache,history_pos,uvec4(0));
	}



#endif


}
