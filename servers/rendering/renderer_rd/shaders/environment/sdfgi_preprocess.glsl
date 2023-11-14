#[compute]

#version 450

#VERSION_DEFINES

#define JUMPFLOOD_UNASSIGNED 0xFFFFFFFF
#define GROUP_SIZE 8

#ifdef MODE_JUMPFLOOD_OPTIMIZED

layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE, local_size_z = GROUP_SIZE) in;

#elif defined(MODE_LIGHT_SCROLL) || defined(MODE_LIGHT_SCROLL_STORE)
//buffer layout
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#elif defined(MODE_SDF_SCROLL)
//buffer layout
layout(local_size_x = 120, local_size_y = 1, local_size_z = 1) in;



#else
//grid layout
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#endif


#ifdef MODE_SDF_SCROLL

layout(r8, set = 0, binding = 1) uniform restrict image3D sdf;
shared float group_distances[120];

#endif

#ifdef MODE_JUMPFLOOD_OPTIMIZED

layout(r32ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_normal_bits;
layout(r8, set = 0, binding = 2) uniform restrict writeonly image3D dst_sdf;

shared uint group_positions[GROUP_SIZE * GROUP_SIZE * GROUP_SIZE]; //4x4x4 with margins
shared uint solid_count;

void group_store(ivec3 p_pos,uint p_value) {
	uint offset = uint(p_pos.z * (GROUP_SIZE * GROUP_SIZE) + p_pos.y * GROUP_SIZE + p_pos.x);
	group_positions[offset].x = p_value;
}

uint group_load(ivec3 p_pos) {
	uint offset = uint(p_pos.z * (GROUP_SIZE * GROUP_SIZE) + p_pos.y * GROUP_SIZE + p_pos.x);
	return group_positions[offset];
}

#endif


#ifdef MODE_LIGHT_STORE

layout(r16ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_albedo;
layout(r32ui, set = 0, binding = 2) uniform restrict readonly uimage3D src_emission;
layout(r32ui, set = 0, binding = 3) uniform restrict readonly uimage3D src_emission_aniso;
layout(r32ui, set = 0, binding = 4) uniform restrict readonly uimage3D src_normal_bits;

#endif

#if defined(MODE_LIGHT_STORE) || defined(MODE_LIGHT_SCROLL) || defined(MODE_LIGHT_SCROLL_STORE)

struct ProcessVoxel {
	uint position; // xyz 10 bit packed
	uint albedo_emission_r; // 0 - 15, albedo 16-31 emission R
	uint emission_gb; // 0-15 emission G, 16-31 emission B
	uint normal; // 0-20 normal RG octahedron 21-32 cached occlusion?
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

#if defined(MODE_LIGHT_SCROLL) || defined(MODE_LIGHT_SCROLL_STORE)

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

layout(set = 0, binding = 9, std430) restrict buffer LightData {
	uint data[];
}
light_scroll;


#endif



layout(push_constant, std430) uniform Params {
	ivec3 scroll;
	int grid_size;

	ivec3 offset;
	int step_size;

	ivec3 limit;
	int cascade;
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


void main() {


#ifdef MODE_SDF_SCROLL

	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
	int index = int(gl_LocalInvocationID.x);

	int scroll;

	if (params.scroll.y != 0) {
		pos.xy = pos.yx;
		scroll = params.scroll.y;
	} else if (params.scroll.z != 0) {
		pos.xz = pos.zx;
		scroll = params.scroll.z;
	} else {
		scroll = params.scroll.x;
	}

	if (index >= (params.grid_size - abs(scroll))) {
		return; // out of range
	}

	pos.z += params.grid_size * params.cascade;

	ivec3 load_pos;
	ivec3 store_pos;

	if (scroll < 0) {
		load_pos = pos - params.scroll;
		store_pos = pos;
	} else {
		load_pos = pos;
		store_pos = pos + params.scroll;

	}

	group_distances[index] = imageLoad(sdf,load_pos).r;

	// sync
	groupMemoryBarrier();
	barrier();

	imageStore(sdf,store_pos,vec4(group_distances[index]));

#endif

#ifdef MODE_JUMPFLOOD_OPTIMIZED
	// optimized version using shared compute to do the whole jump flood using in-place processing

	ivec3 local_pos = ivec3(gl_LocalInvocationID.xyz);

	if (local_pos == ivec3(0)) {
		solid_count = 0;
	}

	groupMemoryBarrier();
	barrier();

	ivec3 global_pos = params.offset + ivec3(gl_GlobalInvocationID.xyz);

	uint p = imageLoad(src_normal_bits, global_pos).r;

	if (p==0) {
		p = JUMPFLOOD_UNASSIGNED;
	} else {
		p = encode_position(global_pos);
		atomicAdd(solid_count,1);
	}

	group_store(local_pos,p);

	// sync
	groupMemoryBarrier();
	barrier();

	if (solid_count == 0) {
		// No solids stored, store max distance and exit.
		imageStore(dst_sdf,global_pos + ivec3(0,0,params.grid_size * params.cascade) ,vec4(encode_distance(100.0)));
		return;
	}


	const uint offset_count = 27;
	const ivec3 offsets[offset_count] = ivec3[](
			ivec3(-1, -1, -1),
			ivec3(-1, -1, 0),
			ivec3(-1, -1, 1),
			ivec3(-1, 0, -1),
			ivec3(-1, 0, 0),
			ivec3(-1, 0, 1),
			ivec3(-1, 1, -1),
			ivec3(-1, 1, 0),
			ivec3(-1, 1, 1),
			ivec3(0, -1, -1),
			ivec3(0, -1, 0),
			ivec3(0, -1, 1),
			ivec3(0, 0, -1),
			ivec3(0, 0, 0),
			ivec3(0, 0, 1),
			ivec3(0, 1, -1),
			ivec3(0, 1, 0),
			ivec3(0, 1, 1),
			ivec3(1, -1, -1),
			ivec3(1, -1, 0),
			ivec3(1, -1, 1),
			ivec3(1, 0, -1),
			ivec3(1, 0, 0),
			ivec3(1, 0, 1),
			ivec3(1, 1, -1),
			ivec3(1, 1, 0),
			ivec3(1, 1, 1));


	int step = 4;

	uint closest;

	while(true) {

		int closest_dist = 0x7FFFFFFF;
		closest = JUMPFLOOD_UNASSIGNED;

		for (uint i = 0; i < offset_count; i++) {

			ivec3 group_ofs = local_pos + offsets[i] * step;

			if (any(lessThan(group_ofs,ivec3(0))) || any(greaterThanEqual(group_ofs,ivec3(GROUP_SIZE)))) {
				continue;
			}
			uint q = group_load(group_ofs);

			if (q == JUMPFLOOD_UNASSIGNED) {
				continue; //was not initialized yet, ignore
			}

			ivec3 q_pos = decode_position(q);

			// Squared distance.
			int q_dist = sq_distance(global_pos,q_pos);
			if (closest == JUMPFLOOD_UNASSIGNED || q_dist < closest_dist) {
				closest = q;
				closest_dist = q_dist;
			}
		}

		groupMemoryBarrier();
		barrier();
		step>>=1;
		if (step == 0) {
			break;
		} else {
			group_store(local_pos,closest);
		}
	}

	ivec3 pos = global_pos;
	ivec3 to_pos = decode_position(closest);

	float d;

	if (pos == to_pos) {
		d = -0.5;
	} else {
		vec3 posf = vec3(pos) + 0.5;
		vec3 to_posf = vec3(to_pos);

		if (pos.x > to_pos.x) {
			to_posf.x+=1.0;
		} else if (pos.x == to_pos.x) {
			to_posf.x+=0.5;
		}

		if (pos.y > to_pos.y) {
			to_posf.y+=1.0;
		} else if (pos.y == to_pos.y) {
			to_posf.y+=0.5;
		}

		if (pos.z > to_pos.z) {
			to_posf.z+=1.0;
		} else if (pos.z == to_pos.z) {
			to_posf.z+=0.5;
		}

		d = distance(posf,to_posf);
	}

	imageStore(dst_sdf,global_pos + ivec3(0,0,params.grid_size * params.cascade) ,vec4(encode_distance(d)));

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

		vec3 normal = aniso_dir[i];
		normal_accum += normal;
		ivec3 albedo_ofs = ofs>>1;
		albedo_ofs.z *= 6;
		albedo_ofs.z += i;

		uint a = imageLoad(src_albedo, albedo_ofs).r;
		albedo_accum += vec4(vec3((ivec3(a) >> ivec3(0,5,11)) & ivec3(0x1f,0x3f,0x1f)) / vec3(31.0,63.0,31.0), 1.0);

		uint rgbe = imageLoad(src_emission, ofs>>1).r;

		vec4 rgbef = vec4((uvec4(rgbe) >> uvec4(0,9,18,27)) & uvec4(0x1FF,0x1FF,0x1FF,0x1F));
		vec3 emission = rgbef.rgb * pow( 2.0, rgbef.a - 15.0 - 9.0 );

		uint rgbe_aniso = imageLoad(src_emission_aniso, ofs>>1).r;
		float strength = ((rgbe_aniso >> (i * 5)) & 0x1F) / float(0x1F);
		emission_accum += vec4(emission * strength,1.0);
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

	uint albedo_e = 0;
	albedo_e |= clamp(uint(albedo_accum.r * 31.0), 0, 31) << 0;
	albedo_e |= clamp(uint(albedo_accum.g * 63.0), 0, 63) << 5;
	albedo_e |= clamp(uint(albedo_accum.b * 31.0), 0, 31) << 11;

	albedo_e |= packHalf2x16(vec2(emission_accum.r,0))<<16;
	dst_process_voxels.data[index].albedo_emission_r = albedo_e;
	dst_process_voxels.data[index].emission_gb = packHalf2x16(emission_accum.gb);

	vec2 octa_normal = octahedron_encode(normal_accum);
	uvec2 octa_unormal = clamp( uvec2(octa_normal * 1023), uvec2(0), uvec2(1023));

	dst_process_voxels.data[index].normal = octa_unormal.x | (octa_unormal.y << 10);



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

	ivec3 light_pos = src_pos;
	light_pos.z += params.grid_size * params.cascade;

	if (!inside_area) {
		imageStore(light_tex,light_pos,uvec4(0)); // Clear directly here (without reading), since we will abort later.
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

	uint light = imageLoad(light_tex,light_pos).r;

	imageStore(light_tex,light_pos,uvec4(0)); // Clear after read
	light_scroll.data[index]=light;

	ivec3 dst_pos = src_pos + params.scroll;

	uint src_pending_bits = src_process_voxels.data[src_index].position & (PROCESS_STATIC_PENDING_BIT|PROCESS_DYNAMIC_PENDING_BIT);

	dst_process_voxels.data[index].position = uint(dst_pos.x | (dst_pos.y << 10) | (dst_pos.z << 20)) | src_pending_bits;
	dst_process_voxels.data[index].albedo_emission_r = src_process_voxels.data[src_index].albedo_emission_r;
	dst_process_voxels.data[index].emission_gb = src_process_voxels.data[src_index].emission_gb;
	dst_process_voxels.data[index].normal = src_process_voxels.data[src_index].normal;

	dst_pos.z+=params.grid_size * params.cascade;

#endif

#ifdef MODE_LIGHT_SCROLL_STORE

	int index = int(gl_GlobalInvocationID).x;

	if (index >= src_dispatch_data.total_count) {
		// Do not process.
		return;
	}

	ivec3 pos = (ivec3(src_process_voxels.data[index].position) >> ivec3(0,10,20)) & ivec3(0x3FF);
	uint light = light_scroll.data[index];

	ivec3 light_pos = pos;
	light_pos.z += params.grid_size * params.cascade;
	imageStore(light_tex,light_pos,uvec4(light));

#endif


}
