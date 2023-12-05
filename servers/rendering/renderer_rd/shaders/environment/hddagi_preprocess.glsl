^#[compute]

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

layout(set = 0, binding = 7) uniform texture2DArray occlusion_direct;
layout(set = 0, binding = 8) uniform texture2DArray occlusion;
layout(set = 0, binding = 9) uniform sampler linear_sampler;

layout(r8ui, set = 0, binding = 10) uniform restrict writeonly uimage3D dst_disocclusion;

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



#ifdef MODE_OCCLUSION

layout(local_size_x = OCCLUSION_OCT_SIZE_HALF, local_size_y = OCCLUSION_OCT_SIZE_HALF, local_size_z = 1) in;


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

layout(local_size_x = OCCLUSION_OCT_SIZE, local_size_y = OCCLUSION_OCT_SIZE, local_size_z = 1) in;

layout(r8, set = 0, binding = 1) uniform restrict readonly image2DArray src_occlusion;
layout(rg16, set = 0, binding = 2) uniform restrict writeonly image2DArray dst_occlusion;
layout(r8ui, set = 0, binding = 3) uniform restrict writeonly uimage2DArray dst_neighbours;

shared vec3 neighbours[OCCLUSION_OCT_SIZE * OCCLUSION_OCT_SIZE];
shared int invalid_rays;

#endif

#ifdef MODE_LIGHTPROBE_SCROLL

layout(local_size_x = LIGHTPROBE_OCT_SIZE, local_size_y = LIGHTPROBE_OCT_SIZE, local_size_z = 1) in;

layout(r32ui, set = 0, binding = 1) uniform restrict uimage2DArray lightprobe_specular_data;
layout(r32ui, set = 0, binding = 2) uniform restrict uimage2DArray lightprobe_diffuse_data;
layout(r32ui, set = 0, binding = 3) uniform restrict uimage2DArray lightprobe_ambient_data;
layout(r32ui, set = 0, binding = 4) uniform restrict uimage2DArray ray_hit_cache;
layout(r32ui, set = 0, binding = 5) uniform restrict uimage2DArray lightprobe_moving_average_history;
layout(r32ui, set = 0, binding = 6) uniform restrict uimage2DArray lightprobe_moving_average;

layout(set = 0, binding = 7) uniform texture2DArray occlusion_probes;
layout(set = 0, binding = 8) uniform sampler linear_sampler;

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

	ivec3 upper_region_world_pos;
	uint pad;

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


uint rgbe_encode(vec3 rgb) {

    const float rgbe_max = uintBitsToFloat(0x477F8000);
    const float rgbe_min = uintBitsToFloat(0x37800000);

    rgb = clamp(rgb, 0, rgbe_max);

    float max_channel = max(max(rgbe_min, rgb.r), max(rgb.g, rgb.b));

    float bias = uintBitsToFloat((floatBitsToUint(max_channel) + 0x07804000) & 0x7F800000);

    uvec3 urgb = floatBitsToUint(rgb + bias);
    uint e = (floatBitsToUint(bias) << 4) + 0x10000000;
    return e | (urgb.b << 18) | (urgb.g << 9) | (urgb.r & 0x1FF);
}

vec3 rgbe_decode(uint p_rgbe) {
	vec4 rgbef = vec4((uvec4(p_rgbe) >> uvec4(0,9,18,27)) & uvec4(0x1FF,0x1FF,0x1FF,0x1F));
	return rgbef.rgb * pow( 2.0, rgbef.a - 15.0 - 9.0 );
}

#define FP_BITS 14
#define FP_MAX ((1<<22)-1)

uvec3 rgbe_decode_fp(uint p_rgbe,int p_bits) {
	uvec4 rgbe = (uvec4(p_rgbe) >> uvec4(0,9,18,27)) & uvec4(0x1FF,0x1FF,0x1FF,0x1F);
	int shift = int(rgbe.a) - 15 - 9 + p_bits;
	if (shift >= 0) {
		rgbe.rgb <<= uint(shift);
	} else {
		rgbe.rgb >>= uint(-shift);
	}
	return rgbe.rgb;
}

ivec3 modi(ivec3 value, ivec3 p_y) {
	// GLSL Specification says:
	// "Results are undefined if one or both operands are negative."
	// So..
	return mix( value % p_y, p_y - ((abs(value)-ivec3(1)) % p_y) -1, lessThan(sign(value), ivec3(0)) );
}

ivec2 probe_to_tex(ivec3 local_probe) {

	ivec3 cell = modi( params.region_world_pos + local_probe,ivec3(params.probe_axis_size));
	return cell.xy + ivec2(0,cell.z * int(params.probe_axis_size.y));

}


ivec2 probe_to_texp(ivec3 local_probe) {

	ivec3 cell = modi( params.upper_region_world_pos + local_probe,ivec3(params.probe_axis_size));
	return cell.xy + ivec2(0,cell.z * int(params.probe_axis_size.y));

}

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

	if (any(greaterThanEqual(pos,params.limit))) {
		// Storing is not a multiple of the workgroup, so invalid threads can happen.
		return;
	}

	uint solid = imageLoad(src_normal_bits, pos).r;

	if (local == ivec3(0)) {
		store_position_count = 0; // Base one stores as zero, the others wait
		if (pos == params.offset) {
			// This lone thread clears y and z.
			dispatch_data.y = 1;
			dispatch_data.z = 1;
		}
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

	const uint aniso_offset_mask[6] = uint[](
		(1<<1),
		(1<<0),
		(1<<3),
		(1<<2),
		(1<<5),
		(1<<4) );

	bool voxels_found=false;
	uint disocclusion = 0;

	for(int i=0;i<6;i++) {

		ivec3 ofs = pos + offsets[i];
		if (any(lessThan(ofs, params.offset)) || any(greaterThanEqual(ofs, params.limit))) {
			// Outside range, continue.
			continue;
		}


		uint n = imageLoad(src_normal_bits, ofs).r;
		if (n == 0) {
			disocclusion|=aniso_offset_mask[i];
		}

		if (solid != 0 || !bool(n & aniso_mask[i])) {
			// Not solid, continue.
			continue;
		}

		voxels_found = true;

		const int facing_direction_count =  26 ;
		const vec3 facing_directions[ 26 ]=vec3[]( vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0), vec3(-0.5773502691896258, -0.5773502691896258, -0.5773502691896258), vec3(-0.7071067811865475, -0.7071067811865475, 0.0), vec3(-0.5773502691896258, -0.5773502691896258, 0.5773502691896258), vec3(-0.7071067811865475, 0.0, -0.7071067811865475), vec3(-0.7071067811865475, 0.0, 0.7071067811865475), vec3(-0.5773502691896258, 0.5773502691896258, -0.5773502691896258), vec3(-0.7071067811865475, 0.7071067811865475, 0.0), vec3(-0.5773502691896258, 0.5773502691896258, 0.5773502691896258), vec3(0.0, -0.7071067811865475, -0.7071067811865475), vec3(0.0, -0.7071067811865475, 0.7071067811865475), vec3(0.0, 0.7071067811865475, -0.7071067811865475), vec3(0.0, 0.7071067811865475, 0.7071067811865475), vec3(0.5773502691896258, -0.5773502691896258, -0.5773502691896258), vec3(0.7071067811865475, -0.7071067811865475, 0.0), vec3(0.5773502691896258, -0.5773502691896258, 0.5773502691896258), vec3(0.7071067811865475, 0.0, -0.7071067811865475), vec3(0.7071067811865475, 0.0, 0.7071067811865475), vec3(0.5773502691896258, 0.5773502691896258, -0.5773502691896258), vec3(0.7071067811865475, 0.7071067811865475, 0.0), vec3(0.5773502691896258, 0.5773502691896258, 0.5773502691896258) );
		for(int j=0;j<facing_direction_count;j++) {
			if (bool(n & uint((1<<(j+6))))) {
				normal_accum += facing_directions[j];
			}
		}

		//normal_accum += aniso_dir[i];

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


	ivec3 dst_pos = (pos + params.region_world_pos * REGION_SIZE) & (params.grid_size -1 );
	dst_pos.y += params.grid_size.y * params.cascade;
	imageStore(dst_disocclusion,dst_pos,uvec4(disocclusion));

	if (solid!=0) {
		return; // No further use for this.
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
#if 1
			// Its just a voxel, why bothering with chebyshev.
			float o = texture(sampler2DArray(occlusion_direct,linear_sampler),vec3(tex_uv,float(params.cascade))).r;
			o *= OCC8_DISTANCE_MAX;

			float weight = d < o ? 1 : 0;

#else
			vec2 o_o2 = texture(sampler2DArray(occlusion,linear_sampler),vec3(tex_uv,float(params.cascade))).rg * OCC16_DISTANCE_MAX;

			float mean = o_o2.x;
			float variance = abs((mean*mean) - o_o2.y);

			 // http://www.punkuser.net/vsm/vsm_paper.pdf; equation 5
			 // Need the max in the denominator because biasing can cause a negative displacement
			float dmean = max(d - mean, 0.0);
			float chebyshev_weight = variance / (variance + dmean*dmean);

			float weight = (d <= mean) ? 1.0 : chebyshev_weight;

			weight = max(0.000001, weight); // make sure not zero (only trilinear can be zero)
#endif
			vec3 trilinear = vec3(1.0) - abs(probe_to_pos / float(PROBE_CELLS));

			weight *= trilinear.x * trilinear.y * trilinear.z;

			weights[i]=weight;
			total_weight += weight;
		}

		for(int i=0;i<8;i++) {
			float w;
			if (total_weight > 0.0) {
				w = weights[i] / total_weight;
				w *= 15.0;
			} else {
				w = 0;
			}
			occlusionu|=uint(clamp(w,0.0,15.0)) << (i * 4);
		}

	}


	index += store_from_index;

	normal_accum = normalize(normal_accum);
	albedo_accum.rgb /= albedo_accum.a;
	emission_accum.rgb /= emission_accum.a;

	dst_process_voxels.data[index].position = uint(pos.x | (pos.y << 10) | (pos.z << 20)) | PROCESS_STATIC_PENDING_BIT | PROCESS_DYNAMIC_PENDING_BIT;

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


	ivec3 workgroup = ivec3(gl_WorkGroupID.xyz);

	ivec2 local_pos = ivec2(gl_LocalInvocationID.xy) + (workgroup.xy & 1) * OCCLUSION_OCT_SIZE_HALF;

	workgroup.xy/=2;


	int linear_index = int(local_pos.y * OCCLUSION_OCT_SIZE + local_pos.x);

	if (gl_LocalInvocationID.xy == uvec2(0)) {
		loaded_regions = 0;
	}

	groupMemoryBarrier();
	barrier();


	{

		int cells_load_total = PROBE_CELLS * PROBE_CELLS * PROBE_CELLS;
		int group_total = OCCLUSION_OCT_SIZE_HALF * OCCLUSION_OCT_SIZE_HALF;
		int group_index = int(gl_LocalInvocationID.y * OCCLUSION_OCT_SIZE_HALF + gl_LocalInvocationID.x);

		int load_from = group_index * cells_load_total / group_total;
		int load_to = (group_index+1) * cells_load_total / group_total;

		int nonzero_found = 0;
		for(int i=load_from;i<load_to;i++) {
			ivec3 load_pos;

			load_pos.z = i / (PROBE_CELLS*PROBE_CELLS);
			load_pos.y = (i / PROBE_CELLS) % PROBE_CELLS;
			load_pos.x = i % PROBE_CELLS;


			load_pos += ivec3(workgroup.xyz * PROBE_CELLS) + params.offset;
			uint n = imageLoad(src_normal_bits,load_pos).r;
			bit_normals[i] = n;

			if (n!=0) {
				// Save some bandwidth.
				uint p0 = imageLoad(src_solid_bits0,load_pos).r;
				uint p1 = imageLoad(src_solid_bits1,load_pos).r;
				bit_voxels[i * 2 + 0] = p0;
				bit_voxels[i * 2 + 1] = p1;

				nonzero_found++;
			} else {
				bit_voxels[i * 2 + 0] = 0;
				bit_voxels[i * 2 + 1] = 0;
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


	vec3 ray_dir = octahedron_decode((vec2(local_pos.xy) + (0.5)) / vec2(OCCLUSION_OCT_SIZE));

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
	bool hit_backface = false;

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

		hit_backface = !any_frontface;
		d = distance(vec3(ray_from),vec3(ray_pos)) / 4.0; // back to cells
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

	if (hit_backface) {
		d = 0.0;
	} else {
		d += 0.1;
	}

	ivec3 base_probe = (params.offset / PROBE_CELLS) + ivec3(workgroup.xyz);


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

	ivec3 dst_tex_uv = ivec3(probe_tex_pos * (OCCLUSION_OCT_SIZE+2) + ivec2(1), params.cascade);

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

		imageStore(dst_occlusion, copy_to[i], vec4(d / OCC8_DISTANCE_MAX));
	}

#endif

#ifdef MODE_OCCLUSION_STORE

	// This shader needs to be optimized, mainly by detecting empty or invalid regions and not processing them.

	//const uint neighbour_max_weights = 8;
	//const uint neighbour_weights[1568]= uint[](8555, 73727, 794361, 860290, 925695, 11017977, 11935874, 12787613, 9440, 75396, 140251, 795937, 861106, 926660, 992292, 0, 76378, 142719, 207226, 732031, 993652, 1059254, 0, 0, 141737, 208307, 272893, 666521, 1059217, 1124730, 0, 0, 205760, 272105, 337089, 533352, 599382, 1123348, 1188800, 0, 270268, 336280, 401541, 466848, 532617, 597650, 1187807, 1253257, 337064, 402910, 468219, 533414, 1254534, 1320104, 1385382, 0, 336806, 402683, 468446, 533672, 1319846, 1385640, 1451142, 0, 269970, 336009, 401312, 467077, 532888, 597948, 1449865, 1515487, 271702, 336744, 533697, 599785, 664512, 1516480, 1582100, 0, 207769, 600573, 667059, 731561, 1583482, 1649041, 0, 0, 142207, 665978, 732543, 797274, 1649078, 1714548, 0, 0, 9138, 75041, 730075, 796292, 861408, 1713188, 1778628, 0, 8322, 73465, 794623, 860523, 1777663, 11869945, 11935645, 12787842, 9440, 74692, 927364, 992292, 1844187, 11019553, 11936690, 0, 74989, 140250, 926957, 993114, 1057674, 1844186, 1909642, 0, 147182, 212591, 998909, 1066147, 0, 0, 0, 0, 212448, 278326, 1131565, 1195706, 0, 0, 0, 0, 271289, 337122, 1123270, 1189601, 1254684, 2040744, 2106358, 0, 336869, 402658, 1188992, 1254974, 1319909, 2106385, 2172032, 0, 401541, 466848, 1253257, 1319320, 1384585, 2170847, 2236348, 2301586, 401312, 467077, 1319049, 1384856, 1449865, 2236050, 2301884, 2367455, 468194, 533477, 1385445, 1451582, 1516672, 2368640, 2434065, 0, 533730, 598969, 1451292, 1517281, 1582022, 2434038, 2499496, 0, 606006, 671200, 1523386, 1590317, 0, 0, 0, 0, 671343, 737006, 1655971, 1719805, 0, 0, 0, 0, 730074, 795885, 1647498, 1714010, 1778925, 2630538, 2696154, 0, 795588, 861408, 1713188, 1779332, 2696155, 11871521, 12788658, 0, 928346, 993652, 1846655, 1911222, 2763130, 10103679, 0, 0, 998909, 1851118, 1918115, 2768495, 0, 0, 0, 0, 2031615, 0, 0, 0, 0, 0, 0, 0, 1195561, 2048968, 2113404, 2965137, 0, 0, 0, 0, 1188845, 1254561, 2040929, 2107095, 2171885, 2958313, 3023969, 0, 1254684, 1320162, 2106358, 2172641, 2237369, 3023784, 3089350, 0, 1320129, 1385320, 2171840, 2238185, 2303318, 3089428, 3154880, 0, 1319784, 1385665, 2237782, 2303721, 2368448, 3220416, 3286036, 0, 1385698, 1451292, 2302905, 2369249, 2434038, 3285958, 3351464, 0, 1451169, 1516525, 2368493, 2434775, 2499681, 3351649, 3417065, 0, 1523241, 2441084, 2507720, 3423889, 0, 0, 0, 0, 2621439, 0, 0, 0, 0, 0, 0, 0, 1719805, 2639011, 2703086, 3620463, 0, 0, 0, 0, 1714548, 1780314, 2632118, 2698623, 3615098, 10955647, 0, 0, 1845673, 1911185, 2764211, 2828666, 3680765, 9186201, 0, 0, 2768352, 2835501, 3686198, 3751610, 0, 0, 0, 0, 2900936, 2965137, 3751465, 3817340, 0, 0, 0, 0, 2040863, 2106414, 2892831, 2959138, 3023903, 3810350, 3875871, 0, 2113404, 2178601, 2965137, 3032008, 0, 0, 0, 0, 2178746, 2244406, 3097645, 3161568, 0, 0, 0, 0, 2238973, 3090810, 3157427, 3222425, 4008337, 4073897, 0, 0, 2304509, 3156889, 3222963, 3287418, 4139433, 4204945, 0, 0, 2309942, 2375354, 3227104, 3294253, 0, 0, 0, 0, 2375209, 2441084, 3359688, 3423889, 0, 0, 0, 0, 2434094, 2499615, 3351583, 3417890, 3482655, 4334623, 4400174, 0, 3423889, 3490760, 4407164, 4472361, 0, 0, 0, 0, 3556397, 3620320, 4472506, 4538166, 0, 0, 0, 0, 2632081, 2697641, 3549562, 3616179, 4532733, 10038169, 0, 0, 2761664, 2827284, 3679977, 3744704, 4596929, 7349096, 8267094, 0, 2827206, 2892712, 3679161, 3745505, 3810294, 4596962, 4662556, 0, 2892897, 2958313, 3744749, 3811031, 3875937, 4662433, 4727789, 0, 2965137, 3817340, 3883976, 4734505, 0, 0, 0, 0, 3997695, 0, 0, 0, 0, 0, 0, 0, 3161711, 4015267, 4079342, 4931069, 0, 0, 0, 0, 3156346, 4008374, 4074879, 4139903, 4925812, 4991578, 0, 0, 3221882, 4074367, 4140415, 4204982, 5057114, 5122420, 0, 0, 3227247, 4144878, 4211875, 5127677, 0, 0, 0, 0, 4325375, 0, 0, 0, 0, 0, 0, 0, 3423889, 4342728, 4407164, 5324329, 0, 0, 0, 0, 3417065, 3482721, 4334689, 4400855, 4465645, 5317613, 5383329, 0, 3482536, 3548102, 4400118, 4466401, 4531129, 5383452, 5448930, 0, 3548180, 3613632, 4465600, 4531945, 5448897, 8201064, 9119062, 0, 3678140, 3743711, 4596120, 4661129, 5513349, 6430624, 7348361, 8265362, 3744896, 3810321, 4596709, 4662846, 4727936, 5514466, 5579749, 0, 3810294, 3875752, 4662556, 4728545, 4793286, 5580002, 5645241, 0, 4734650, 4801581, 5652278, 5717472, 0, 0, 0, 0, 4867235, 4931069, 5717615, 5783278, 0, 0, 0, 0, 4006794, 4072410, 4858762, 4925274, 4990189, 5776346, 5842157, 0, 4072411, 4924452, 4990596, 5055777, 5841860, 5907680, 5972914, 0, 4137947, 4990241, 5056132, 5121060, 5907378, 5973216, 6038468, 0, 4137946, 4203402, 5055725, 5121882, 5186442, 6038765, 6104026, 0, 5127677, 5194915, 6110958, 6176367, 0, 0, 0, 0, 5260333, 5324474, 6176224, 6242102, 0, 0, 0, 0, 4334504, 4400118, 5252038, 5318369, 5383452, 6235065, 6300898, 0, 4400145, 4465792, 5317760, 5383742, 5448677, 6300645, 6366434, 0, 4464607, 4530108, 5382025, 5448088, 6365317, 7282592, 8200329, 9117330, 4596904, 4662406, 5514718, 5579944, 6431995, 6497190, 7349158, 0, 4661129, 4726751, 5513349, 5579160, 5644220, 6430624, 6496393, 6561426, 4727744, 4793364, 5579969, 5646057, 5710784, 6497128, 6563158, 0, 4794746, 4860305, 5646845, 5713331, 5777833, 6630297, 0, 0, 4860342, 4925812, 5712250, 5778815, 5843546, 6695807, 0, 0, 4924452, 4989892, 5776347, 5842564, 5907680, 6759713, 6824882, 0, 4988927, 5054201, 5840895, 5906795, 5972098, 6758137, 6824066, 6889373, 4988665, 5054463, 5906562, 5972331, 6037503, 6823837, 6889602, 6954745, 5055428, 5121060, 5973216, 6039172, 6104027, 6890418, 6956321, 0, 5122420, 5188022, 6040154, 6106495, 6171002, 7023487, 0, 0, 5187985, 5253498, 6105513, 6172083, 6236669, 7089049, 0, 0, 5252116, 5317568, 6169536, 6235881, 6300865, 7152982, 7218024, 0, 5316575, 5382025, 6234044, 6300056, 6365317, 7151250, 7217289, 7282592, 5383302, 5448872, 6300840, 6366686, 7218086, 7283963, 8201126, 0, 4596646, 5514491, 5579686, 6432222, 6497448, 7349416, 7414918, 0, 5513120, 5578889, 5643922, 6430853, 6496664, 6561724, 7413641, 7479263, 5579624, 5645654, 6497473, 6563561, 6628288, 7480256, 7545876, 0, 5712793, 6564349, 6630835, 6695337, 7547258, 7612817, 0, 0, 5778303, 6629754, 6696319, 6761050, 7612854, 7678324, 0, 0, 5842209, 5907378, 6693851, 6760068, 6825184, 7676964, 7742404, 0, 5840633, 5906562, 5971869, 6758399, 6824299, 6889602, 7741439, 7806713, 5906333, 5972098, 6037241, 6824066, 6889835, 6955007, 7741177, 7806975, 5972914, 6038817, 6890720, 6956676, 7021531, 7807940, 7873572, 0, 6105983, 6957658, 7023999, 7088506, 7874932, 7940534, 0, 0, 6171545, 7023017, 7089587, 7154173, 7940497, 8006010, 0, 0, 6235478, 6300520, 7087040, 7153385, 7218369, 8004628, 8070080, 0, 6233746, 6299785, 6365088, 7151548, 7217560, 7282821, 8069087, 8134537, 5448614, 6300582, 6366459, 7218344, 7284190, 8135814, 8201384, 0, 3677842, 4595849, 5513120, 6430853, 7348632, 7413641, 8265660, 8331231, 6431970, 6497253, 7349221, 7415358, 7480448, 8332416, 8397841, 0, 6497506, 6562745, 7415068, 7481057, 7545798, 8397814, 8463272, 0, 6569782, 6634976, 7487162, 7554093, 0, 0, 0, 0, 6635119, 6700782, 7619747, 7683581, 0, 0, 0, 0, 6693850, 6759661, 7611274, 7677786, 7742701, 8594314, 8659930, 0, 6759364, 6825184, 6890418, 7676964, 7743108, 7808289, 8659931, 0, 6824882, 6890720, 6955972, 7742753, 7808644, 7873572, 8725467, 0, 6956269, 7021530, 7808237, 7874394, 7938954, 8725466, 8790922, 0, 7028462, 7093871, 7880189, 7947427, 0, 0, 0, 0, 7093728, 7159606, 8012845, 8076986, 0, 0, 0, 0, 7152569, 7218402, 8004550, 8070881, 8135964, 8922024, 8987638, 0, 7218149, 7283938, 8070272, 8136254, 8201189, 8987665, 9053312, 0, 4529810, 5447817, 6365088, 7282821, 8134537, 8200600, 9052127, 9117628, 3679574, 4596584, 7349441, 8267497, 8332224, 9184192, 9249812, 0, 7349474, 7415068, 8266681, 8333025, 8397814, 9249734, 9315240, 0, 7414945, 7480301, 8332269, 8398551, 8463457, 9315425, 9380841, 0, 7487017, 8404860, 8471496, 9387665, 0, 0, 0, 0, 8585215, 0, 0, 0, 0, 0, 0, 0, 7683581, 8602787, 8666862, 9584239, 0, 0, 0, 0, 7678324, 7744090, 8595894, 8662399, 8727423, 9578874, 0, 0, 7809626, 7874932, 8661887, 8727935, 8792502, 9644410, 0, 0, 7880189, 8732398, 8799395, 9649775, 0, 0, 0, 0, 8912895, 0, 0, 0, 0, 0, 0, 0, 8076841, 8930248, 8994684, 9846417, 0, 0, 0, 0, 8070125, 8135841, 8922209, 8988375, 9053165, 9839593, 9905249, 0, 8135964, 8201442, 8987638, 9053921, 9118649, 9905064, 9970630, 0, 4531542, 5448552, 8201409, 9053120, 9119465, 9970708, 10036160, 0, 2763673, 8268285, 9186739, 9251194, 10103209, 10168721, 0, 0, 8273718, 8339130, 9190880, 9258029, 0, 0, 0, 0, 8338985, 8404860, 9323464, 9387665, 0, 0, 0, 0, 8397870, 8463391, 9315359, 9381666, 9446431, 10298399, 10363950, 0, 9387665, 9454536, 10370940, 10436137, 0, 0, 0, 0, 9520173, 9584096, 10436282, 10501942, 0, 0, 0, 0, 8595857, 8661417, 9513338, 9579955, 9644953, 10496509, 0, 0, 8726953, 8792465, 9579417, 9645491, 9709946, 10562045, 0, 0, 9649632, 9716781, 10567478, 10632890, 0, 0, 0, 0, 9782216, 9846417, 10632745, 10698620, 0, 0, 0, 0, 8922143, 8987694, 9774111, 9840418, 9905183, 10691630, 10757151, 0, 8994684, 9059881, 9846417, 9913288, 0, 0, 0, 0, 9060026, 9125686, 9978925, 10042848, 0, 0, 0, 0, 3615641, 9120253, 9972090, 10038707, 10889617, 10955177, 0, 0, 1846143, 9185658, 10104191, 10168758, 11020890, 11086196, 0, 0, 9191023, 10108654, 10175651, 11091453, 0, 0, 0, 0, 10289151, 0, 0, 0, 0, 0, 0, 0, 9387665, 10306504, 10370940, 11288105, 0, 0, 0, 0, 9380841, 9446497, 10298465, 10364631, 10429421, 11281389, 11347105, 0, 9446312, 9511878, 10363894, 10430177, 10494905, 11347228, 11412706, 0, 9511956, 9577408, 10429376, 10495721, 10560854, 11412673, 11477864, 0, 9642944, 9708564, 10495318, 10561257, 10625984, 11412328, 11478209, 0, 9708486, 9773992, 10560441, 10626785, 10691574, 11478242, 11543836, 0, 9774177, 9839593, 10626029, 10692311, 10757217, 11543713, 11609069, 0, 9846417, 10698620, 10765256, 11615785, 0, 0, 0, 0, 10878975, 0, 0, 0, 0, 0, 0, 0, 10042991, 10896547, 10960622, 11812349, 0, 0, 0, 0, 2698111, 10037626, 10889654, 10956159, 11807092, 11872858, 0, 0, 9138, 927009, 10101723, 11019908, 11084836, 11936992, 12002244, 0, 10101722, 10167178, 11019501, 11085658, 11150218, 12002541, 12067802, 0, 11091453, 11158691, 12074734, 12140143, 0, 0, 0, 0, 11224109, 11288250, 12140000, 12205878, 0, 0, 0, 0, 10298280, 10363894, 11215814, 11282145, 11347228, 12198841, 12264674, 0, 10363921, 10429568, 11281536, 11347518, 11412453, 12264421, 12330210, 0, 10428383, 10493884, 10559122, 11345801, 11411864, 11477129, 12329093, 12394400, 10493586, 10559420, 10624991, 11411593, 11477400, 11542409, 12328864, 12394629, 10626176, 10691601, 11477989, 11544126, 11609216, 12395746, 12461029, 0, 10691574, 10757032, 11543836, 11609825, 11674566, 12461282, 12526521, 0, 11615930, 11682861, 12533558, 12598752, 0, 0, 0, 0, 11748515, 11812349, 12598895, 12664558, 0, 0, 0, 0, 10888074, 10953690, 11740042, 11806554, 11871469, 12657626, 12723437, 0, 861106, 1778977, 10953691, 11805732, 11871876, 12723140, 12788960, 0, 8322, 860061, 925433, 11018239, 11936107, 12001279, 12721913, 12787842, 11019204, 11084836, 11936992, 12002948, 12067803, 12723489, 12788658, 0, 11086196, 11151798, 12003930, 12070271, 12134778, 12659583, 0, 0, 11151761, 11217274, 12069289, 12135859, 12200445, 12594073, 0, 0, 11215892, 11281344, 12133312, 12199657, 12264641, 12460904, 12526934, 0, 11280351, 11345801, 12197820, 12263832, 12329093, 12394400, 12460169, 12525202, 11347078, 11412648, 11477926, 12264616, 12330462, 12395771, 12460966, 0, 11412390, 11478184, 11543686, 12264358, 12330235, 12395998, 12461224, 0, 11542409, 11608031, 12197522, 12263561, 12328864, 12394629, 12460440, 12525500, 11609024, 11674644, 12199254, 12264296, 12461249, 12527337, 12592064, 0, 11676026, 11741585, 12135321, 12528125, 12594611, 12659113, 0, 0, 11741622, 11807092, 12069759, 12593530, 12660095, 12724826, 0, 0, 11805732, 11871172, 11936690, 12002593, 12657627, 12723844, 12788960, 0, 8093, 860290, 1777401, 11870207, 11935874, 12001017, 12722175, 12788075);


	const uint neighbour_max_weights = 20;
	const uint neighbour_weights[3920]= uint[](3695, 69152, 134360, 724120, 789990, 855613, 921120, 986485, 1707178, 1772913, 1838296, 10095768, 11013606, 11078826, 11865399, 11931197, 11996529, 12717367, 12783114, 0, 4285, 69915, 135297, 725033, 790732, 856184, 921724, 987282, 1052469, 1707901, 1773453, 1838881, 11014029, 11865758, 11931630, 12783529, 0, 0, 0, 0, 4221, 70113, 135816, 201137, 659781, 725534, 790913, 856106, 921699, 987568, 1053118, 1118181, 1642583, 1708133, 1904608, 0, 0, 0, 0, 0, 136282, 202065, 267375, 332398, 594936, 660690, 725982, 987718, 1053780, 1119311, 1184382, 1577652, 1643182, 0, 0, 0, 0, 0, 0, 0, 200847, 266567, 331978, 397259, 462727, 528507, 594156, 659503, 1052421, 1118371, 1183887, 1249181, 1445553, 1511291, 1576811, 2035460, 0, 0, 0, 0, 200160, 266137, 331789, 397258, 462738, 528331, 593743, 1117856, 1183650, 1249165, 1314568, 1380038, 1445580, 1511103, 2035180, 2100849, 2166388, 0, 0, 0, 265875, 331695, 397297, 462785, 528247, 593491, 1183522, 1249192, 1314735, 1380215, 1445635, 1511008, 2100861, 2166562, 2231955, 2297427, 2362976, 0, 0, 0, 265811, 331639, 397249, 462833, 528303, 593555, 1183328, 1249027, 1314679, 1380271, 1445800, 1511202, 2166368, 2231891, 2297491, 2363170, 2428541, 0, 0, 0, 266063, 331723, 397202, 462794, 528397, 593817, 658912, 1183423, 1248972, 1314502, 1380104, 1445773, 1511330, 1576608, 2362996, 2428529, 2493932, 0, 0, 0, 200751, 266476, 331899, 397191, 462795, 528586, 594247, 659599, 1118059, 1183611, 1248945, 1445789, 1511567, 1577123, 1642245, 2494212, 0, 0, 0, 0, 136158, 201938, 267256, 529006, 595055, 660817, 726106, 1053358, 1118900, 1512062, 1578063, 1643604, 1708614, 0, 0, 0, 0, 0, 0, 0, 4138, 70017, 135710, 201029, 659889, 725640, 791009, 856189, 987237, 1052759, 1576933, 1642942, 1708464, 1773667, 2625504, 0, 0, 0, 0, 0, 4216, 69836, 135209, 725121, 790811, 856253, 921485, 987005, 1642293, 1708178, 1773692, 2690849, 11013790, 11865997, 11931561, 12783598, 0, 0, 0, 0, 3645, 69094, 134296, 724184, 790048, 855663, 920945, 986282, 1707381, 1773088, 2690264, 10947736, 11013431, 11799722, 11865574, 11931146, 11996471, 12717425, 12783165, 0, 4285, 69756, 134945, 790413, 856046, 921883, 987282, 1839233, 1904437, 10096681, 11014348, 11079549, 11931768, 11997069, 12717726, 12783529, 0, 0, 0, 0, 4147, 69846, 135318, 200422, 724832, 790460, 855871, 921814, 987489, 1052804, 1839254, 1904772, 2756326, 10096480, 11014076, 11931455, 0, 0, 0, 0, 70801, 136743, 202257, 267293, 660551, 726119, 988669, 1054499, 1119708, 1840082, 1906128, 1971657, 0, 0, 0, 0, 0, 0, 0, 0, 135753, 201807, 267399, 332653, 594694, 660148, 1053732, 1119569, 1184882, 1249969, 1971234, 2036788, 2101958, 0, 0, 0, 0, 0, 0, 0, 200220, 266135, 331735, 397168, 462506, 528113, 593554, 1118105, 1183811, 1249251, 1314469, 1969674, 2035603, 2101156, 2166497, 2952764, 3018254, 0, 0, 0, 265353, 331121, 396704, 462097, 527562, 592845, 1117226, 1183118, 1248736, 1314161, 1379530, 1444852, 2034861, 2100601, 2166158, 2231433, 2296781, 2952204, 3017901, 3083306, 331528, 397258, 462738, 528070, 1183348, 1249165, 1314829, 1380299, 1445580, 2100849, 2166690, 2232217, 2297679, 2363071, 3018220, 3083936, 3149280, 0, 0, 0, 331462, 397202, 462794, 528136, 1248972, 1314763, 1380365, 1445773, 1511028, 2166463, 2232143, 2297753, 2363298, 2428529, 3214816, 3280544, 3345900, 0, 0, 0, 265165, 330954, 396561, 462240, 527729, 593033, 1248244, 1313994, 1379697, 1445344, 1510798, 1575978, 2231245, 2296969, 2362766, 2428281, 2493613, 3279914, 3345581, 3410956, 265874, 331505, 396970, 462704, 528343, 593815, 658972, 1380005, 1445859, 1511491, 1576857, 2363105, 2428836, 2494355, 2559498, 3345934, 3411516, 0, 0, 0, 201396, 267014, 529261, 595079, 660559, 725577, 1446577, 1512562, 1578321, 1643556, 2429638, 2495540, 2561058, 0, 0, 0, 0, 0, 0, 0, 136295, 201799, 594973, 661009, 726567, 791697, 1578460, 1644323, 1709565, 2561481, 2627024, 2692050, 0, 0, 0, 0, 0, 0, 0, 0, 3903, 69564, 135008, 659174, 725142, 790742, 856115, 1642628, 1708385, 1773782, 2625668, 2691222, 3608294, 10948448, 11866044, 12783423, 0, 0, 0, 0, 4078, 69517, 724769, 790652, 856253, 1708178, 1773851, 2625333, 2691201, 10948649, 11800445, 11866316, 11931561, 11996830, 12717965, 12783736, 0, 0, 0, 0, 4221, 69731, 922081, 987568, 1052640, 1839752, 1905086, 2757041, 2822117, 9179461, 10097182, 10162263, 11014529, 11079781, 11931690, 0, 0, 0, 0, 0, 136146, 922769, 988669, 1054160, 1840679, 1906467, 1971657, 2758161, 2823644, 3675165, 9180231, 10097767, 0, 0, 0, 0, 0, 0, 0, 0, 1055236, 1120792, 1185754, 1907204, 1973153, 2038320, 2824728, 2890288, 2955276, 3741658, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 266882, 332408, 1119278, 1184868, 1250164, 1971246, 2037067, 2102427, 2167398, 2888755, 2954357, 3019548, 3805883, 0, 0, 0, 0, 0, 0, 0, 332209, 397790, 1118544, 1184488, 1250071, 1315249, 2036486, 2102184, 2167528, 2888011, 2953959, 3019526, 3084624, 3871051, 0, 0, 0, 0, 0, 0, 331429, 397168, 462506, 1183457, 1249251, 1314775, 1380081, 2035214, 2101156, 2166851, 2232215, 2297490, 2952764, 3018643, 3084185, 3149340, 3935754, 0, 0, 0, 397259, 462727, 1249181, 1315018, 1380475, 1445553, 2166927, 2232647, 2298092, 2363259, 3018500, 3084451, 3149967, 3215407, 3280747, 4001541, 0, 0, 0, 0, 397191, 462795, 1248945, 1314939, 1380554, 1445789, 2166651, 2232556, 2298183, 2363535, 3084139, 3149871, 3215503, 3281059, 3346180, 4198149, 0, 0, 0, 0, 396970, 462704, 528037, 1314545, 1380311, 1445859, 1511137, 2231954, 2297751, 2363459, 2428836, 2493966, 3214876, 3280793, 3346323, 3411516, 4263434, 0, 0, 0, 463326, 528817, 1380785, 1446679, 1512168, 1577296, 2364136, 2429864, 2495238, 3281232, 3347206, 3412711, 3477835, 4329803, 0, 0, 0, 0, 0, 0, 529016, 594562, 1446772, 1512548, 1578030, 2364006, 2430107, 2495819, 2561070, 3347228, 3413109, 3478579, 4395707, 0, 0, 0, 0, 0, 0, 0, 1513434, 1579544, 1645060, 2497072, 2562977, 2628100, 3414028, 3480112, 3545624, 4462554, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 725970, 1643984, 1709565, 1774737, 2561481, 2627363, 2692647, 3544540, 3610129, 4527133, 10032199, 10949735, 0, 0, 0, 0, 0, 0, 0, 0, 790627, 856189, 1642464, 1708464, 1774049, 2625982, 2691720, 3543013, 3609009, 10031429, 10883159, 10949150, 11800677, 11866497, 12783658, 0, 0, 0, 0, 0, 987718, 1840218, 1905748, 2757969, 2823247, 3675247, 3740286, 4592238, 8262648, 9180370, 9245364, 10097630, 10162862, 0, 0, 0, 0, 0, 0, 0, 1839689, 1905700, 1971234, 2757711, 2823505, 2888756, 3675271, 3740786, 3805894, 4592493, 4657841, 8262406, 9179828, 0, 0, 0, 0, 0, 0, 0, 1971246, 2036787, 2101947, 2823214, 2889035, 2954357, 3674754, 3740772, 3806363, 3871516, 4592248, 4658036, 4723302, 0, 0, 0, 0, 0, 0, 0, 1183821, 1249323, 1970230, 2036197, 2101736, 2166861, 2888165, 2953888, 3019237, 3739725, 3805672, 3871205, 3936310, 4657195, 4722765, 0, 0, 0, 0, 0, 1184358, 1250164, 1315448, 2036508, 2102427, 2167908, 2232962, 2954357, 3020107, 3085358, 3805883, 3871795, 3937326, 0, 0, 0, 0, 0, 0, 0, 1249969, 1315693, 2101958, 2167922, 2233479, 2298630, 3019828, 3085649, 3150927, 3216052, 3937314, 4002852, 4067913, 0, 0, 0, 0, 0, 0, 0, 1315438, 2167422, 2233455, 2298872, 3085391, 3151185, 3216594, 3281588, 4002900, 4068442, 4133854, 4199086, 4919878, 0, 0, 0, 0, 0, 0, 0, 1380974, 2233336, 2298991, 2364030, 3084980, 3151058, 3216721, 3281999, 4002478, 4068318, 4133978, 4199508, 5116486, 0, 0, 0, 0, 0, 0, 0, 1381229, 1446577, 2233094, 2299015, 2364530, 2429638, 3150516, 3216463, 3282257, 3347508, 4133449, 4199460, 4264994, 0, 0, 0, 0, 0, 0, 0, 1380984, 1446772, 1512038, 2298498, 2364516, 2430107, 2495260, 3281966, 3347787, 3413109, 4265006, 4330547, 4395707, 0, 0, 0, 0, 0, 0, 0, 1445931, 1511501, 2363469, 2429416, 2494949, 2560054, 3346917, 3412640, 3477989, 4263990, 4329957, 4395496, 4460621, 5312589, 5378091, 0, 0, 0, 0, 0, 2429627, 2495539, 2561070, 3413109, 3478859, 3544110, 4330268, 4396187, 4461668, 4526722, 5313126, 5378932, 5444216, 0, 0, 0, 0, 0, 0, 0, 2561058, 2626596, 2691657, 3478580, 3544401, 3609679, 4395718, 4461682, 4527239, 5378737, 5444461, 9114374, 10031796, 0, 0, 0, 0, 0, 0, 0, 1708614, 2626644, 2692186, 3544143, 3609937, 4461182, 4527215, 5444206, 9114616, 9966260, 10032338, 10883758, 10949598, 0, 0, 0, 0, 0, 0, 0, 1904389, 2756751, 2822307, 2887428, 3674439, 3739791, 4591818, 4657053, 5509067, 6426503, 7344251, 7409329, 8261868, 8327035, 9179183, 9244523, 0, 0, 0, 0, 1969674, 2756124, 2822041, 2887571, 2952764, 3674007, 3739715, 3805092, 3870222, 4591575, 4657123, 4722401, 5508976, 5574309, 6426282, 7343857, 8261266, 0, 0, 0, 2036043, 2822480, 2888454, 2953959, 3019083, 3740392, 3806120, 3871494, 4592049, 4657943, 4723432, 4788560, 5509598, 5575089, 0, 0, 0, 0, 0, 0, 2101947, 2888476, 2954357, 3019827, 3740262, 3806363, 3872075, 3937326, 4658036, 4723812, 4789294, 5575288, 5640834, 0, 0, 0, 0, 0, 0, 0, 2168794, 2955276, 3021360, 3086872, 3873328, 3939233, 4004356, 4724698, 4790808, 4856324, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2233373, 3085788, 3151377, 3216455, 3937737, 4003619, 4068903, 4133991, 4855248, 4920829, 4986001, 5772242, 0, 0, 0, 0, 0, 0, 0, 0, 3084261, 3150257, 3215685, 4002238, 4067976, 4133406, 4198487, 4853728, 4919728, 4985313, 5050753, 5116005, 5836899, 5902461, 5967914, 0, 0, 0, 0, 0, 3150149, 3215793, 3280869, 4001879, 4067870, 4133512, 4198846, 4919397, 4985217, 5050849, 5116336, 5181408, 5902378, 5967997, 6033507, 0, 0, 0, 0, 0, 2298909, 3150919, 3216913, 3282396, 4068455, 4134439, 4200227, 4265417, 5051537, 5117437, 5182928, 6099922, 0, 0, 0, 0, 0, 0, 0, 0, 2365402, 3283480, 3349040, 3414028, 4200964, 4266913, 4332080, 5184004, 5249560, 5314522, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2429627, 3347507, 3413109, 3478300, 4265006, 4330827, 4396187, 4461158, 5248046, 5313636, 5378932, 6230658, 6296184, 0, 0, 0, 0, 0, 0, 0, 2494795, 3346763, 3412711, 3478278, 3543376, 4330246, 4395944, 4461288, 5247312, 5313256, 5378839, 5444017, 6295985, 6361566, 0, 0, 0, 0, 0, 0, 2559498, 3411516, 3477395, 3542937, 3608092, 4328974, 4394916, 4460611, 4525975, 5312225, 5378019, 5443543, 6295205, 6360944, 7278250, 8195825, 9113234, 0, 0, 0, 2625285, 3477252, 3543203, 3608719, 4460687, 4526407, 5377949, 5443786, 6361035, 7278471, 8130225, 8196219, 9047931, 9113836, 9965419, 10031151, 0, 0, 0, 0, 2756064, 2821792, 2887148, 3674009, 3739554, 3804785, 4591629, 4657037, 4722292, 5509066, 5574408, 6426514, 6491846, 7344075, 7409356, 8261455, 8326847, 0, 0, 0, 2821162, 2886829, 2952204, 3673225, 3739022, 3804537, 3869869, 4590961, 4656608, 4722062, 4787242, 5508512, 5574001, 5639305, 6425873, 6491338, 6556621, 7343306, 7408628, 8260557, 2887182, 2952764, 3739361, 3805092, 3870611, 3935754, 4591269, 4657123, 4722755, 4788121, 5508976, 5574615, 5640087, 5705244, 6426282, 6491889, 6557330, 0, 0, 0, 3805894, 3871796, 3937314, 4657841, 4723826, 4789585, 4854820, 5575533, 5641351, 5706831, 5771849, 6558470, 6623924, 0, 0, 0, 0, 0, 0, 0, 3937737, 4003280, 4068306, 4789724, 4855587, 4920829, 5641245, 5707281, 5772839, 5837969, 6624327, 6689895, 0, 0, 0, 0, 0, 0, 0, 0, 3149542, 4001924, 4067478, 4132704, 4853892, 4919649, 4985046, 5050300, 5705446, 5771414, 5837014, 5902387, 5967679, 6688608, 6754236, 6819647, 0, 0, 0, 0, 4001589, 4067457, 4132905, 4919442, 4985115, 5050572, 5115773, 5771041, 5836924, 5902525, 5967992, 6033293, 6754189, 6819822, 6885289, 6950558, 0, 0, 0, 0, 4067369, 4132993, 4198197, 4919165, 4985036, 5050651, 5116050, 5836685, 5902456, 5968061, 6033532, 6098721, 6753950, 6819753, 6885358, 6950797, 0, 0, 0, 0, 3215078, 4067168, 4133014, 4198532, 4984764, 5050582, 5116257, 5181572, 5902143, 5967923, 6033622, 6099094, 6164198, 6885183, 6950844, 7016288, 0, 0, 0, 0, 4133842, 4199888, 4265417, 5117437, 5183267, 5248476, 6034577, 6100519, 6166033, 6231069, 7017575, 7083079, 0, 0, 0, 0, 0, 0, 0, 0, 4264994, 4330548, 4395718, 5182500, 5248337, 5313650, 5378737, 6099529, 6165583, 6231175, 6296429, 7082676, 7148294, 0, 0, 0, 0, 0, 0, 0, 3411516, 3477006, 4263434, 4329363, 4394916, 4460257, 5246873, 5312579, 5378019, 5443237, 6163996, 6229911, 6295511, 6360944, 7147154, 7212785, 7278250, 0, 0, 0, 3410956, 3476653, 3542058, 4328621, 4394361, 4459918, 4525193, 5245994, 5311886, 5377504, 5442929, 6229129, 6294897, 6360480, 7146445, 7212234, 7277841, 8129524, 8195274, 9112525, 3476972, 3542688, 3608032, 4394609, 4460450, 4525977, 5312116, 5377933, 5443597, 6295304, 6361034, 7212742, 7278482, 8130252, 8196043, 9047743, 9113423, 0, 0, 0, 3673747, 3739426, 3804797, 4591535, 4657064, 4722466, 5509105, 5574575, 5639827, 6426561, 6492023, 6557267, 7343991, 7409411, 7474784, 8261203, 8326752, 0, 0, 0, 3739252, 3804785, 3870188, 4591368, 4657037, 4722594, 4787872, 5509066, 5574669, 5640089, 5705184, 6426514, 6492107, 6557519, 7343814, 7409356, 7474879, 0, 0, 0, 3870468, 4657053, 4722831, 4788387, 4853509, 5509067, 5574858, 5640519, 5705871, 6426503, 6492283, 6557932, 6623279, 7409329, 7475067, 7540587, 0, 0, 0, 0, 4723326, 4789327, 4854868, 4919878, 5575278, 5641327, 5707089, 5772378, 6558712, 6624466, 6689758, 7541428, 7606958, 0, 0, 0, 0, 0, 0, 0, 4001760, 4788197, 4854206, 4919728, 4984931, 5706161, 5771912, 5837281, 5902461, 6623557, 6689310, 6754689, 6819882, 7606359, 7671909, 0, 0, 0, 0, 0, 4067105, 4853557, 4919442, 4984956, 5050253, 5771393, 5837083, 5902525, 5967854, 6688809, 6754508, 6819960, 6885289, 7671677, 7737229, 7802526, 0, 0, 0, 0, 4066520, 4131992, 4918645, 4984352, 5049830, 5115050, 5770456, 5836320, 5901935, 5967421, 6032753, 6687896, 6753766, 6819389, 6884874, 6950199, 7670954, 7736689, 7802167, 0, 4066456, 4132056, 4918442, 4984294, 5049888, 5115253, 5836145, 5901885, 5967471, 6032928, 6098136, 6753591, 6819338, 6884925, 6950374, 7015576, 7736631, 7802225, 7867562, 0, 4132641, 4984717, 5050492, 5116050, 5181237, 5902318, 5968061, 6033691, 6099073, 6819753, 6885496, 6951116, 7016489, 7736990, 7802765, 7868285, 0, 0, 0, 0, 4198368, 5050467, 5116336, 5181886, 5246949, 5967997, 6033889, 6099592, 6164913, 6885418, 6951297, 7016990, 7082309, 7868517, 7934039, 0, 0, 0, 0, 0, 5116486, 5182548, 5248079, 5313150, 6100058, 6165841, 6231151, 6296174, 7017438, 7083218, 7148536, 7934638, 8000180, 0, 0, 0, 0, 0, 0, 0, 4329220, 5181189, 5247139, 5312655, 5377949, 6164623, 6230343, 6295754, 6361035, 7082031, 7147756, 7213179, 7278471, 7999339, 8064891, 8130225, 0, 0, 0, 0, 4328940, 4394609, 4460148, 5246624, 5312418, 5377933, 5443336, 6163936, 6229913, 6295565, 6361034, 7147343, 7213003, 7278482, 8064703, 8130252, 8195782, 0, 0, 0, 4394621, 4460322, 4525715, 5312290, 5377960, 5443503, 6229651, 6295471, 6361073, 7147091, 7212919, 7278529, 8064608, 8130307, 8195959, 9047648, 9113171, 0, 0, 0, 3673683, 3739232, 4591479, 4656899, 4722272, 5509057, 5574519, 5639763, 6426609, 6492079, 6557331, 7344047, 7409576, 7474978, 8261267, 8326946, 8392317, 0, 0, 0, 4591302, 4656844, 4722367, 5509010, 5574603, 5640015, 6426570, 6492173, 6557593, 6622688, 7343880, 7409549, 7475106, 7540384, 8326772, 8392305, 8457708, 0, 0, 0, 4656817, 4722555, 4788075, 5508999, 5574779, 5640428, 5705775, 6426571, 6492362, 6558023, 6623375, 7409565, 7475343, 7540899, 7606021, 8457988, 0, 0, 0, 0, 4788916, 4854446, 5641208, 5706962, 5772254, 6492782, 6558831, 6624593, 6689882, 7475838, 7541839, 7607380, 7672390, 0, 0, 0, 0, 0, 0, 0, 4853847, 4919397, 5706053, 5771806, 5837185, 5902378, 6623665, 6689416, 6754785, 6819965, 7540709, 7606718, 7672240, 7737443, 8589280, 0, 0, 0, 0, 0, 4919165, 4984717, 5050014, 5771305, 5837004, 5902456, 5967785, 6688897, 6754587, 6820029, 6885358, 7606069, 7671954, 7737468, 7802765, 8654625, 0, 0, 0, 0, 4918442, 4984177, 5049655, 5770392, 5836262, 5901885, 5967370, 6032695, 6687960, 6753824, 6819439, 6884925, 6950257, 7671157, 7736864, 7802342, 7867562, 8654040, 8719512, 0, 4984119, 5049713, 5115050, 5836087, 5901834, 5967421, 6032870, 6098072, 6753649, 6819389, 6884975, 6950432, 7015640, 7670954, 7736806, 7802400, 7867765, 8653976, 8719576, 0, 4984478, 5050253, 5115773, 5902249, 5967992, 6033612, 6098985, 6819822, 6885565, 6951195, 7016577, 7737229, 7803004, 7868562, 7933749, 8720161, 0, 0, 0, 0, 5116005, 5181527, 5967914, 6033793, 6099486, 6164805, 6885501, 6951393, 7017096, 7082417, 7802979, 7868848, 7934398, 7999461, 8785888, 0, 0, 0, 0, 0, 5182126, 5247668, 6099934, 6165714, 6231032, 7017562, 7083345, 7148655, 7213678, 7868998, 7935060, 8000591, 8065662, 0, 0, 0, 0, 0, 0, 0, 5246827, 5312379, 5377713, 6164527, 6230252, 6295675, 6360967, 7082127, 7147847, 7213258, 7278539, 7933701, 7999651, 8065167, 8130461, 8916740, 0, 0, 0, 0, 5312191, 5377740, 5443270, 6229839, 6295499, 6360978, 7081440, 7147417, 7213069, 7278538, 7999136, 8064930, 8130445, 8195848, 8916460, 8982129, 9047668, 0, 0, 0, 4460128, 4525651, 5312096, 5377795, 5443447, 6229587, 6295415, 6361025, 7147155, 7212975, 7278577, 8064802, 8130472, 8196015, 8982141, 9047842, 9113235, 0, 0, 0, 3673935, 3739327, 4591563, 4656844, 5509010, 5574342, 6426570, 6491912, 7344141, 7409549, 7474804, 8261529, 8327074, 8392305, 9178592, 9244320, 9309676, 0, 0, 0, 3673037, 4590794, 4656116, 5508369, 5573834, 5639117, 6426016, 6491505, 6556809, 7343473, 7409120, 7474574, 7539754, 8260745, 8326542, 8392057, 8457389, 9243690, 9309357, 9374732, 5508778, 5574385, 5639826, 6426480, 6492119, 6557591, 6622748, 7343781, 7409635, 7475267, 7540633, 8326881, 8392612, 8458131, 8523274, 9309710, 9375292, 0, 0, 0, 5640966, 5706420, 6493037, 6558855, 6624335, 6689353, 7410353, 7476338, 7542097, 7607332, 8393414, 8459316, 8524834, 0, 0, 0, 0, 0, 0, 0, 5706823, 5772391, 6558749, 6624785, 6690343, 6755473, 7542236, 7608099, 7673341, 8525257, 8590800, 8655826, 0, 0, 0, 0, 0, 0, 0, 0, 5771104, 5836732, 5902143, 6622950, 6688918, 6754518, 6819891, 6885183, 7606404, 7672161, 7737558, 7802812, 8589444, 8654998, 8720224, 9572070, 0, 0, 0, 0, 5836685, 5902318, 5967785, 6033054, 6688545, 6754428, 6820029, 6885496, 6950797, 7671954, 7737627, 7803084, 7868285, 8589109, 8654977, 8720425, 0, 0, 0, 0, 5836446, 5902249, 5967854, 6033293, 6754189, 6819960, 6885565, 6951036, 7016225, 7671677, 7737548, 7803163, 7868562, 8654889, 8720513, 8785717, 0, 0, 0, 0, 5967679, 6033340, 6098784, 6819647, 6885427, 6951126, 7016598, 7081702, 7737276, 7803094, 7868769, 7934084, 8654688, 8720534, 8786052, 9637606, 0, 0, 0, 0, 6100071, 6165575, 6952081, 7018023, 7083537, 7148573, 7869949, 7935779, 8000988, 8721362, 8787408, 8852937, 0, 0, 0, 0, 0, 0, 0, 0, 6165172, 6230790, 7017033, 7083087, 7148679, 7213933, 7935012, 8000849, 8066162, 8131249, 8852514, 8918068, 8983238, 0, 0, 0, 0, 0, 0, 0, 6229650, 6295281, 6360746, 7081500, 7147415, 7213015, 7278448, 7999385, 8065091, 8130531, 8195749, 8850954, 8916883, 8982436, 9047777, 9834044, 9899534, 0, 0, 0, 4525005, 5377012, 5442762, 6228941, 6294730, 6360337, 7146633, 7212401, 7277984, 7998506, 8064398, 8130016, 8195441, 8916141, 8981881, 9047438, 9112713, 9833484, 9899181, 9964586, 4460223, 4525903, 5377740, 5443531, 6295238, 6360978, 7212808, 7278538, 8064628, 8130445, 8196109, 8982129, 9047970, 9113497, 9899500, 9965216, 10030560, 0, 0, 0, 2756655, 2821995, 3674348, 3739515, 4591739, 4656817, 5508999, 6426571, 7344330, 7409565, 8261959, 8327311, 9179279, 9244835, 9309956, 10161925, 0, 0, 0, 0, 3673746, 4591345, 5508778, 6426480, 6491813, 7344087, 7409635, 7474913, 8261527, 8327235, 8392612, 8457742, 9178652, 9244569, 9310099, 9375292, 10227210, 0, 0, 0, 6427102, 6492593, 7344561, 7410455, 7475944, 7541072, 8327912, 8393640, 8459014, 9245008, 9310982, 9376487, 9441611, 10293579, 0, 0, 0, 0, 0, 0, 6492792, 6558338, 7410548, 7476324, 7541806, 8327782, 8393883, 8459595, 8524846, 9311004, 9376885, 9442355, 10359483, 0, 0, 0, 0, 0, 0, 0, 7477210, 7543320, 7608836, 8460848, 8526753, 8591876, 9377804, 9443888, 9509400, 10426330, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6689746, 7607760, 7673341, 7738513, 8525257, 8591139, 8656423, 8721511, 9508316, 9573905, 9638983, 10490909, 0, 0, 0, 0, 0, 0, 0, 0, 6754403, 6819965, 6885418, 7606240, 7672240, 7737825, 7803265, 7868517, 8589758, 8655496, 8720926, 8786007, 9506789, 9572785, 9638213, 0, 0, 0, 0, 0, 6819882, 6885501, 6951011, 7671909, 7737729, 7803361, 7868848, 7933920, 8589399, 8655390, 8721032, 8786366, 9572677, 9638321, 9703397, 0, 0, 0, 0, 0, 7017426, 7804049, 7869949, 7935440, 8655975, 8721959, 8787747, 8852937, 9573447, 9639441, 9704924, 10556445, 0, 0, 0, 0, 0, 0, 0, 0, 7936516, 8002072, 8067034, 8788484, 8854433, 8919600, 9706008, 9771568, 9836556, 10622938, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7148162, 7213688, 8000558, 8066148, 8131444, 8852526, 8918347, 8983707, 9048678, 9770035, 9835637, 9900828, 10687163, 0, 0, 0, 0, 0, 0, 0, 7213489, 7279070, 7999824, 8065768, 8131351, 8196529, 8917766, 8983464, 9048808, 9769291, 9835239, 9900806, 9965904, 10752331, 0, 0, 0, 0, 0, 0, 4525714, 5443313, 6360746, 7212709, 7278448, 8064737, 8130531, 8196055, 8916494, 8982436, 9048131, 9113495, 9834044, 9899923, 9965465, 10030620, 10817034, 0, 0, 0, 3542891, 3608623, 4460411, 4526316, 5377713, 5443707, 6360967, 7278539, 8130461, 8196298, 9048207, 9113927, 9899780, 9965731, 10031247, 10882821, 0, 0, 0, 0, 1840094, 1905326, 2757842, 2822836, 3675128, 7344750, 8262767, 8327806, 9180497, 9245775, 10097754, 10163284, 11080262, 0, 0, 0, 0, 0, 0, 0, 2757300, 3674886, 7345005, 7410353, 8262791, 8328306, 8393414, 9180239, 9246033, 9311284, 10097225, 10163236, 10228770, 0, 0, 0, 0, 0, 0, 0, 7344760, 7410548, 7475814, 8262274, 8328292, 8393883, 8459036, 9245742, 9311563, 9376885, 10228782, 10294323, 10359483, 0, 0, 0, 0, 0, 0, 0, 7409707, 7475277, 8327245, 8393192, 8458725, 8523830, 9310693, 9376416, 9441765, 10227766, 10293733, 10359272, 10424397, 11276365, 11341867, 0, 0, 0, 0, 0, 8393403, 8459315, 8524846, 9376885, 9442635, 9507886, 10294044, 10359963, 10425444, 10490498, 11276902, 11342708, 11407992, 0, 0, 0, 0, 0, 0, 0, 8524834, 8590372, 8655433, 9442356, 9508177, 9573455, 9638580, 10359494, 10425458, 10491015, 10556166, 11342513, 11408237, 0, 0, 0, 0, 0, 0, 0, 7672390, 8590420, 8655962, 8721374, 8786606, 9507919, 9573713, 9639122, 9704116, 10424958, 10490991, 10556408, 11407982, 0, 0, 0, 0, 0, 0, 0, 7868998, 8589998, 8655838, 8721498, 8787028, 9507508, 9573586, 9639249, 9704527, 10490872, 10556527, 10621566, 11473518, 0, 0, 0, 0, 0, 0, 0, 8720969, 8786980, 8852514, 9573044, 9638991, 9704785, 9770036, 10490630, 10556551, 10622066, 10687174, 11473773, 11539121, 0, 0, 0, 0, 0, 0, 0, 8852526, 8918067, 8983227, 9704494, 9770315, 9835637, 10556034, 10622052, 10687643, 10752796, 11473528, 11539316, 11604582, 0, 0, 0, 0, 0, 0, 0, 8065101, 8130603, 8851510, 8917477, 8983016, 9048141, 9769445, 9835168, 9900517, 10621005, 10686952, 10752485, 10817590, 11538475, 11604045, 0, 0, 0, 0, 0, 8065638, 8131444, 8196728, 8917788, 8983707, 9049188, 9114242, 9835637, 9901387, 9966638, 10687163, 10753075, 10818606, 0, 0, 0, 0, 0, 0, 0, 3609268, 4526854, 8131249, 8196973, 8983238, 9049202, 9114759, 9901108, 9966929, 10032207, 10818594, 10884132, 10949193, 0, 0, 0, 0, 0, 0, 0, 2626222, 2692062, 3543732, 3609810, 4527096, 8196718, 9048702, 9114735, 9966671, 10032465, 10884180, 10949722, 11801158, 0, 0, 0, 0, 0, 0, 0, 4138, 921985, 987237, 1839646, 1904727, 2756933, 9179569, 9244645, 10097288, 10162622, 11014625, 11080112, 11145184, 11931773, 11997283, 0, 0, 0, 0, 0, 1840231, 2757703, 8262685, 9180689, 9246172, 10098215, 10164003, 10229193, 11015313, 11081213, 11146704, 12063698, 0, 0, 0, 0, 0, 0, 0, 0, 8329178, 9247256, 9312816, 9377804, 10164740, 10230689, 10295856, 11147780, 11213336, 11278298, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8393403, 9311283, 9376885, 9442076, 10228782, 10294603, 10359963, 10424934, 11211822, 11277412, 11342708, 12194434, 12259960, 0, 0, 0, 0, 0, 0, 0, 8458571, 9310539, 9376487, 9442054, 9507152, 10294022, 10359720, 10425064, 11211088, 11277032, 11342615, 11407793, 12259761, 12325342, 0, 0, 0, 0, 0, 0, 8523274, 9375292, 9441171, 9506713, 9571868, 10292750, 10358692, 10424387, 10489751, 10555026, 11276001, 11341795, 11407319, 11472625, 12258981, 12324720, 12390058, 0, 0, 0, 8589061, 9441028, 9506979, 9572495, 9637935, 9703275, 10424463, 10490183, 10555628, 10620795, 11341725, 11407562, 11473019, 11538097, 12324811, 12390279, 0, 0, 0, 0, 8785669, 9506667, 9572399, 9638031, 9703587, 9768708, 10424187, 10490092, 10555719, 10621071, 11341489, 11407483, 11473098, 11538333, 12324743, 12390347, 0, 0, 0, 0, 8850954, 9637404, 9703321, 9768851, 9834044, 10489490, 10555287, 10620995, 10686372, 10751502, 11407089, 11472855, 11538403, 11603681, 12324522, 12390256, 12455589, 0, 0, 0, 8917323, 9703760, 9769734, 9835239, 9900363, 10621672, 10687400, 10752774, 11473329, 11539223, 11604712, 11669840, 12390878, 12456369, 0, 0, 0, 0, 0, 0, 8983227, 9769756, 9835637, 9901107, 10621542, 10687643, 10753355, 10818606, 11539316, 11605092, 11670574, 12456568, 12522114, 0, 0, 0, 0, 0, 0, 0, 9050074, 9836556, 9902640, 9968152, 10754608, 10820513, 10885636, 11605978, 11672088, 11737604, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2692199, 3609671, 9114653, 9967068, 10032657, 10819017, 10884899, 10950183, 11736528, 11802109, 11867281, 12653522, 0, 0, 0, 0, 0, 0, 0, 0, 856106, 1708133, 1773953, 2625623, 2691614, 3608901, 9965541, 10031537, 10883518, 10949256, 11735008, 11801008, 11866593, 12718179, 12783741, 0, 0, 0, 0, 0, 4216, 69517, 790174, 855977, 921804, 987005, 1839145, 10096769, 10161973, 11014427, 11079826, 11931837, 11997308, 12062497, 12717965, 12783598, 0, 0, 0, 0, 3903, 921532, 1838944, 9178854, 10096790, 10162308, 11014358, 11080033, 11145348, 11931699, 11997398, 12062870, 12127974, 12652384, 12718012, 12783423, 0, 0, 0, 0, 10097618, 10163664, 10229193, 11081213, 11147043, 11212252, 11998353, 12064295, 12129809, 12194845, 12588103, 12653671, 0, 0, 0, 0, 0, 0, 0, 0, 10228770, 10294324, 10359494, 11146276, 11212113, 11277426, 11342513, 12063305, 12129359, 12194951, 12260205, 12522246, 12587700, 0, 0, 0, 0, 0, 0, 0, 9375292, 9440782, 10227210, 10293139, 10358692, 10424033, 11210649, 11276355, 11341795, 11407013, 12127772, 12193687, 12259287, 12324720, 12390058, 12455665, 12521106, 0, 0, 0, 9374732, 9440429, 9505834, 10292397, 10358137, 10423694, 10488969, 10554317, 11209770, 11275662, 11341280, 11406705, 11472074, 11537396, 12192905, 12258673, 12324256, 12389649, 12455114, 12520397, 9440748, 9506464, 9571808, 10358385, 10424226, 10489753, 10555215, 10620607, 11275892, 11341709, 11407373, 11472843, 11538124, 12259080, 12324810, 12390290, 12455622, 0, 0, 0, 9637344, 9703072, 9768428, 10423999, 10489679, 10555289, 10620834, 10686065, 11341516, 11407307, 11472909, 11538317, 11603572, 12259014, 12324754, 12390346, 12455688, 0, 0, 0, 9702442, 9768109, 9833484, 10488781, 10554505, 10620302, 10685817, 10751149, 11340788, 11406538, 11472241, 11537888, 11603342, 11668522, 12192717, 12258506, 12324113, 12389792, 12455281, 12520585, 9768462, 9834044, 10620641, 10686372, 10751891, 10817034, 11472549, 11538403, 11604035, 11669401, 12193426, 12259057, 12324522, 12390256, 12455895, 12521367, 12586524, 0, 0, 0, 10687174, 10753076, 10818594, 11539121, 11605106, 11670865, 11736100, 12128948, 12194566, 12456813, 12522631, 12588111, 12653129, 0, 0, 0, 0, 0, 0, 0, 10819017, 10884560, 10949586, 11671004, 11736867, 11802109, 12063847, 12129351, 12522525, 12588561, 12654119, 12719249, 0, 0, 0, 0, 0, 0, 0, 0, 855871, 1773500, 2690912, 10030822, 10883204, 10948758, 11735172, 11800929, 11866326, 11931455, 11997116, 12062560, 12586726, 12652694, 12718294, 12783667, 0, 0, 0, 0, 4009, 69278, 790413, 856184, 1707901, 1773772, 2691113, 10882869, 10948737, 11800722, 11866395, 11931630, 11997069, 12652321, 12718204, 12783805, 0, 0, 0, 0, 3645, 68977, 789815, 855562, 921062, 986282, 1772855, 1838232, 10095832, 11013664, 11079029, 11799722, 11865457, 11931247, 11996704, 12061912, 12651672, 12717542, 12783165, 0, 4078, 855977, 921485, 1773214, 10096417, 11014268, 11079826, 11145013, 11800445, 11865997, 11931837, 11997467, 12062849, 12652585, 12718284, 12783736, 0, 0, 0, 0, 10162144, 11014243, 11080112, 11145662, 11210725, 11735127, 11800677, 11931773, 11997665, 12063368, 12128689, 12587333, 12653086, 12718465, 12783658, 0, 0, 0, 0, 0, 11080262, 11146324, 11211855, 11276926, 11670196, 11735726, 12063834, 12129617, 12194927, 12259950, 12522488, 12588242, 12653534, 0, 0, 0, 0, 0, 0, 0, 10292996, 11144965, 11210915, 11276431, 11341725, 11538097, 11603835, 11669355, 12128399, 12194119, 12259530, 12324811, 12390279, 12456059, 12521708, 12587055, 0, 0, 0, 0, 10292716, 10358385, 10423924, 11210400, 11276194, 11341709, 11407112, 11472582, 11538124, 11603647, 12127712, 12193689, 12259341, 12324810, 12390290, 12455883, 12521295, 0, 0, 0, 10358397, 10424098, 10489491, 10554963, 10620512, 11276066, 11341736, 11407279, 11472759, 11538179, 11603552, 12193427, 12259247, 12324849, 12390337, 12455799, 12521043, 0, 0, 0, 10423904, 10489427, 10555027, 10620706, 10686077, 11275872, 11341571, 11407223, 11472815, 11538344, 11603746, 12193363, 12259191, 12324801, 12390385, 12455855, 12521107, 0, 0, 0, 10620532, 10686065, 10751468, 11275967, 11341516, 11407046, 11472648, 11538317, 11603874, 11669152, 12193615, 12259275, 12324754, 12390346, 12455949, 12521369, 12586464, 0, 0, 0, 10751748, 11210603, 11276155, 11341489, 11538333, 11604111, 11669667, 11734789, 12128303, 12194028, 12259451, 12324743, 12390347, 12456138, 12521799, 12587151, 0, 0, 0, 0, 11145902, 11211444, 11604606, 11670607, 11736148, 11801158, 12063710, 12129490, 12194808, 12456558, 12522607, 12588369, 12653658, 0, 0, 0, 0, 0, 0, 0, 10883040, 11079781, 11145303, 11669477, 11735486, 11801008, 11866211, 11931690, 11997569, 12063262, 12128581, 12587441, 12653192, 12718561, 12783741, 0, 0, 0, 0, 0, 4009, 856046, 921246, 1773453, 10948385, 11014029, 11079549, 11734837, 11800722, 11866236, 11931768, 11997388, 12062761, 12652673, 12718363, 12783805, 0, 0, 0, 0, 3594, 68919, 789873, 855613, 920887, 1707178, 1773030, 2690200, 10947800, 11013489, 11078826, 11799925, 11865632, 11931197, 11996646, 12061848, 12651736, 12717600, 12783215, 0 );

	ivec2 local_pos = ivec2(gl_LocalInvocationID).xy;

	if (local_pos == ivec2(0)) {
		invalid_rays = 0;
	}

	groupMemoryBarrier();
	barrier();

	int linear_pos = local_pos.y * OCCLUSION_OCT_SIZE + local_pos.x;

	ivec3 base_probe = (params.offset / PROBE_CELLS) + ivec3(gl_WorkGroupID.xyz);
	ivec2 probe_tex_pos = probe_to_tex(base_probe);

	ivec3 src_tex_uv = ivec3(probe_tex_pos * (OCCLUSION_OCT_SIZE+2) + ivec2(1), params.cascade);
	ivec3 dst_tex_uv = src_tex_uv;

	float o = imageLoad(src_occlusion,src_tex_uv + ivec3(local_pos,0)).r * OCC8_DISTANCE_MAX;

	float multiplier = 1.0; // Poison the filter if backface found.

	if (o == 0) {
		o = 0.0;
		multiplier = 0.0;
		atomicAdd(invalid_rays,1);
	} else {
		o = max(0.0, o - 0.1);
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

	if (local_pos == ivec2(0, 0)) {
		// Store neighbours for filtering.
		const vec3 aniso_dir[6] = vec3[](
				vec3(-1, 0, 0),
				vec3(1, 0, 0),
				vec3(0, -1, 0),
				vec3(0, 1, 0),
				vec3(0, 0, -1),
				vec3(0, 0, 1));

		uint neighbours = 0;
		for(int i=0;i<6;i++) {
			ivec3 neighbour_probe = base_probe + ivec3(aniso_dir[i]);
			if (any(lessThan(neighbour_probe,ivec3(0))) || any(greaterThanEqual(neighbour_probe,params.probe_axis_size))) {
				continue; // Outside range.
			}
			ivec3 test_uv = src_tex_uv + ivec3(octahedron_encode(aniso_dir[i]) * OCCLUSION_OCT_SIZE,0);
			o = imageLoad(src_occlusion, test_uv ).r * OCC8_DISTANCE_MAX;
			o = max(0.0, o - 0.1);

			if (o >= PROBE_CELLS - 0.5) {
				// Reaches neighbour.
				neighbours |= (1<<i);
			}

		}

		imageStore(dst_neighbours, ivec3(probe_tex_pos, params.cascade), uvec4(neighbours));
	}


#endif


#ifdef MODE_LIGHTPROBE_SCROLL

	ivec2 local_pos = ivec2(gl_LocalInvocationID).xy;

	vec3 specular_light = vec3(0.0);
	vec3 diffuse_light = vec3(0.0);
	vec3 ambient_light = vec3(0.0);

	int upper_cascade = params.cascade + 1;

	if (upper_cascade < params.cascade_count) {

		vec3 posi = params.offset + ivec3(gl_WorkGroupID.xyz) * PROBE_CELLS;
		// Convert cell to world
		posi += params.region_world_pos * REGION_SIZE - params.grid_size/2;
		posi -= (params.upper_region_world_pos * REGION_SIZE - params.grid_size/2)*2;

		vec3 posf = vec3(posi) / 2.0;

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

				if (d > 0.001) {

					ivec2 tex_pos = probe_to_texp(probe);
					vec2 tex_uv = vec2(ivec2(tex_pos * (OCCLUSION_OCT_SIZE+2) + ivec2(1))) + octahedron_encode(n) * float(OCCLUSION_OCT_SIZE);
					tex_uv *= occ_probe_tex_to_uv;
					vec2 o_o2 = texture(sampler2DArray(occlusion_probes,linear_sampler),vec3(tex_uv,float(upper_cascade))).rg * OCC16_DISTANCE_MAX;

					float mean = o_o2.x;
					float variance = abs((mean*mean) - o_o2.y);

					 // http://www.punkuser.net/vsm/vsm_paper.pdf; equation 5
					 // Need the max in the denominator because biasing can cause a negative displacement
					float dmean = max(d - mean, 0);
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

				}
				occlusion[i]=weight;
				occlusion_total+=weight;
			}

			for(int i=0;i<8;i++) {
				if (occlusion_total == 0.0) {
					occlusion[i] = 0;
				} else {
					occlusion[i]/=occlusion_total;
				}
			}

		}

		groupMemoryBarrier();
		barrier();

		for(int i=0;i<8;i++) {
			ivec3 probe = base_probe + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
			ivec2 base_tex_pos = probe_to_texp(probe);
			ivec2 tex_pos = base_tex_pos * (LIGHTPROBE_OCT_SIZE+2) + ivec2(1) + local_pos;
			ivec3 tex_array_pos = ivec3(tex_pos,upper_cascade);
			specular_light += rgbe_decode(imageLoad(lightprobe_specular_data,tex_array_pos).r) * occlusion[i];
			diffuse_light += rgbe_decode(imageLoad(lightprobe_diffuse_data,tex_array_pos).r) * occlusion[i];

			if (local_pos == ivec2(0)) {
				tex_array_pos = ivec3(base_tex_pos,upper_cascade);
				ambient_light += rgbe_decode(imageLoad(lightprobe_ambient_data,tex_array_pos).r) * occlusion[i];
			}
		}
	}

	ivec3 probe_from = (params.offset / PROBE_CELLS) + ivec3(gl_WorkGroupID.xyz);
	ivec2 probe_tex_pos = probe_to_tex(probe_from);

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

	uint specular_rgbe = rgbe_encode(specular_light);
	uint diffuse_rgbe = rgbe_encode(diffuse_light);

	for (int i = 0; i < 4; i++) {
		if (copy_to[i] == ivec3(-2, -2, -2)) {
			continue;
		}
		imageStore(lightprobe_specular_data,copy_to[i],uvec4(specular_rgbe));
		imageStore(lightprobe_diffuse_data,copy_to[i],uvec4(diffuse_rgbe));
	}


	if (local_pos == ivec2(0)) {
		imageStore(lightprobe_ambient_data,ivec3(probe_tex_pos,params.cascade),uvec4(rgbe_encode(ambient_light)));
	}

	// Cache and history invalidation
	probe_tex_pos = probe_tex_pos * LIGHTPROBE_OCT_SIZE + local_pos;

	for(int i=0;i<params.ray_hit_cache_frames;i++) {
		ivec3 history_pos = ivec3(probe_tex_pos, params.cascade * params.ray_hit_cache_frames + i);
		// Completely invalidate cache frame.
		imageStore(ray_hit_cache,history_pos,uvec4(0));
		imageStore(lightprobe_moving_average_history,history_pos,uvec4(specular_rgbe));
	}

	uvec3 moving_average = rgbe_decode_fp(specular_rgbe,FP_BITS);
	moving_average *= params.ray_hit_cache_frames;

	ivec3 ma_pos = ivec3(probe_tex_pos, params.cascade);
	ma_pos.x *= 3;

	imageStore(lightprobe_moving_average,ma_pos + ivec3(0,0,0),uvec4(moving_average.r));
	imageStore(lightprobe_moving_average,ma_pos + ivec3(1,0,0),uvec4(moving_average.g));
	imageStore(lightprobe_moving_average,ma_pos + ivec3(2,0,0),uvec4(moving_average.b));

#endif


}
