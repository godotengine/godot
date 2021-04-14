#[compute]

#version 450

#VERSION_DEFINES

#ifdef MODE_JUMPFLOOD_OPTIMIZED
#define GROUP_SIZE 8

layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE, local_size_z = GROUP_SIZE) in;

#elif defined(MODE_OCCLUSION) || defined(MODE_SCROLL)
//buffer layout
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#else
//grid layout
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#endif

#if defined(MODE_INITIALIZE_JUMP_FLOOD) || defined(MODE_INITIALIZE_JUMP_FLOOD_HALF)
layout(r16ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_color;
layout(rgba8ui, set = 0, binding = 2) uniform restrict writeonly uimage3D dst_positions;
#endif

#ifdef MODE_UPSCALE_JUMP_FLOOD
layout(r16ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_color;
layout(rgba8ui, set = 0, binding = 2) uniform restrict readonly uimage3D src_positions_half;
layout(rgba8ui, set = 0, binding = 3) uniform restrict writeonly uimage3D dst_positions;
#endif

#if defined(MODE_JUMPFLOOD) || defined(MODE_JUMPFLOOD_OPTIMIZED)
layout(rgba8ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_positions;
layout(rgba8ui, set = 0, binding = 2) uniform restrict writeonly uimage3D dst_positions;
#endif

#ifdef MODE_JUMPFLOOD_OPTIMIZED

shared uvec4 group_positions[(GROUP_SIZE + 2) * (GROUP_SIZE + 2) * (GROUP_SIZE + 2)]; //4x4x4 with margins

void group_store(ivec3 p_pos, uvec4 p_value) {
	uint offset = uint(p_pos.z * (GROUP_SIZE + 2) * (GROUP_SIZE + 2) + p_pos.y * (GROUP_SIZE + 2) + p_pos.x);
	group_positions[offset] = p_value;
}

uvec4 group_load(ivec3 p_pos) {
	uint offset = uint(p_pos.z * (GROUP_SIZE + 2) * (GROUP_SIZE + 2) + p_pos.y * (GROUP_SIZE + 2) + p_pos.x);
	return group_positions[offset];
}

#endif

#ifdef MODE_OCCLUSION

layout(r16ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_color;
layout(r8, set = 0, binding = 2) uniform restrict image3D dst_occlusion[8];
layout(r32ui, set = 0, binding = 3) uniform restrict readonly uimage3D src_facing;

const uvec2 group_size_offset[11] = uvec2[](uvec2(1, 0), uvec2(3, 1), uvec2(6, 4), uvec2(10, 10), uvec2(15, 20), uvec2(21, 35), uvec2(28, 56), uvec2(36, 84), uvec2(42, 120), uvec2(46, 162), uvec2(48, 208));
const uint group_pos[256] = uint[](0,
		65536, 256, 1,
		131072, 65792, 512, 65537, 257, 2,
		196608, 131328, 66048, 768, 131073, 65793, 513, 65538, 258, 3,
		262144, 196864, 131584, 66304, 1024, 196609, 131329, 66049, 769, 131074, 65794, 514, 65539, 259, 4,
		327680, 262400, 197120, 131840, 66560, 1280, 262145, 196865, 131585, 66305, 1025, 196610, 131330, 66050, 770, 131075, 65795, 515, 65540, 260, 5,
		393216, 327936, 262656, 197376, 132096, 66816, 1536, 327681, 262401, 197121, 131841, 66561, 1281, 262146, 196866, 131586, 66306, 1026, 196611, 131331, 66051, 771, 131076, 65796, 516, 65541, 261, 6,
		458752, 393472, 328192, 262912, 197632, 132352, 67072, 1792, 393217, 327937, 262657, 197377, 132097, 66817, 1537, 327682, 262402, 197122, 131842, 66562, 1282, 262147, 196867, 131587, 66307, 1027, 196612, 131332, 66052, 772, 131077, 65797, 517, 65542, 262, 7,
		459008, 393728, 328448, 263168, 197888, 132608, 67328, 458753, 393473, 328193, 262913, 197633, 132353, 67073, 1793, 393218, 327938, 262658, 197378, 132098, 66818, 1538, 327683, 262403, 197123, 131843, 66563, 1283, 262148, 196868, 131588, 66308, 1028, 196613, 131333, 66053, 773, 131078, 65798, 518, 65543, 263,
		459264, 393984, 328704, 263424, 198144, 132864, 459009, 393729, 328449, 263169, 197889, 132609, 67329, 458754, 393474, 328194, 262914, 197634, 132354, 67074, 1794, 393219, 327939, 262659, 197379, 132099, 66819, 1539, 327684, 262404, 197124, 131844, 66564, 1284, 262149, 196869, 131589, 66309, 1029, 196614, 131334, 66054, 774, 131079, 65799, 519,
		459520, 394240, 328960, 263680, 198400, 459265, 393985, 328705, 263425, 198145, 132865, 459010, 393730, 328450, 263170, 197890, 132610, 67330, 458755, 393475, 328195, 262915, 197635, 132355, 67075, 1795, 393220, 327940, 262660, 197380, 132100, 66820, 1540, 327685, 262405, 197125, 131845, 66565, 1285, 262150, 196870, 131590, 66310, 1030, 196615, 131335, 66055, 775);

shared uint occlusion_facing[((OCCLUSION_SIZE * 2) * (OCCLUSION_SIZE * 2) * (OCCLUSION_SIZE * 2)) / 4];

uint get_facing(ivec3 p_pos) {
	uint ofs = uint(p_pos.z * OCCLUSION_SIZE * 2 * OCCLUSION_SIZE * 2 + p_pos.y * OCCLUSION_SIZE * 2 + p_pos.x);
	uint v = occlusion_facing[ofs / 4];
	return (v >> ((ofs % 4) * 8)) & 0xFF;
}

#endif

#ifdef MODE_STORE

layout(rgba8ui, set = 0, binding = 1) uniform restrict readonly uimage3D src_positions;
layout(r16ui, set = 0, binding = 2) uniform restrict readonly uimage3D src_albedo;
layout(r8, set = 0, binding = 3) uniform restrict readonly image3D src_occlusion[8];
layout(r32ui, set = 0, binding = 4) uniform restrict readonly uimage3D src_light;
layout(r32ui, set = 0, binding = 5) uniform restrict readonly uimage3D src_light_aniso;
layout(r32ui, set = 0, binding = 6) uniform restrict readonly uimage3D src_facing;

layout(r8, set = 0, binding = 7) uniform restrict writeonly image3D dst_sdf;
layout(r16ui, set = 0, binding = 8) uniform restrict writeonly uimage3D dst_occlusion;

layout(set = 0, binding = 10, std430) restrict buffer DispatchData {
	uint x;
	uint y;
	uint z;
	uint total_count;
}
dispatch_data;

struct ProcessVoxel {
	uint position; //xyz 7 bit packed, extra 11 bits for neigbours
	uint albedo; //rgb bits 0-15 albedo, bits 16-21 are normal bits (set if geometry exists toward that side), extra 11 bits for neibhbours
	uint light; //rgbe8985 encoded total saved light, extra 2 bits for neighbours
	uint light_aniso; //55555 light anisotropy, extra 2 bits for neighbours
	//total neighbours: 26
};

layout(set = 0, binding = 11, std430) restrict buffer writeonly ProcessVoxels {
	ProcessVoxel data[];
}
dst_process_voxels;

shared ProcessVoxel store_positions[4 * 4 * 4];
shared uint store_position_count;
shared uint store_from_index;
#endif

#ifdef MODE_SCROLL

layout(r16ui, set = 0, binding = 1) uniform restrict writeonly uimage3D dst_albedo;
layout(r32ui, set = 0, binding = 2) uniform restrict writeonly uimage3D dst_facing;
layout(r32ui, set = 0, binding = 3) uniform restrict writeonly uimage3D dst_light;
layout(r32ui, set = 0, binding = 4) uniform restrict writeonly uimage3D dst_light_aniso;

layout(set = 0, binding = 5, std430) restrict buffer readonly DispatchData {
	uint x;
	uint y;
	uint z;
	uint total_count;
}
dispatch_data;

struct ProcessVoxel {
	uint position; //xyz 7 bit packed, extra 11 bits for neigbours
	uint albedo; //rgb bits 0-15 albedo, bits 16-21 are normal bits (set if geometry exists toward that side), extra 11 bits for neibhbours
	uint light; //rgbe8985 encoded total saved light, extra 2 bits for neighbours
	uint light_aniso; //55555 light anisotropy, extra 2 bits for neighbours
	//total neighbours: 26
};

layout(set = 0, binding = 6, std430) restrict buffer readonly ProcessVoxels {
	ProcessVoxel data[];
}
src_process_voxels;

#endif

#ifdef MODE_SCROLL_OCCLUSION

layout(r8, set = 0, binding = 1) uniform restrict image3D dst_occlusion[8];
layout(r16ui, set = 0, binding = 2) uniform restrict readonly uimage3D src_occlusion;

#endif

layout(push_constant, binding = 0, std430) uniform Params {
	ivec3 scroll;

	int grid_size;

	ivec3 probe_offset;
	int step_size;

	bool half_size;
	uint occlusion_index;
	int cascade;
	uint pad;
}
params;

void main() {
#ifdef MODE_SCROLL

	// Pixel being shaded
	int index = int(gl_GlobalInvocationID.x);
	if (index >= dispatch_data.total_count) { //too big
		return;
	}

	ivec3 read_pos = (ivec3(src_process_voxels.data[index].position) >> ivec3(0, 7, 14)) & ivec3(0x7F);
	ivec3 write_pos = read_pos + params.scroll;

	if (any(lessThan(write_pos, ivec3(0))) || any(greaterThanEqual(write_pos, ivec3(params.grid_size)))) {
		return; //fits outside the 3D texture, dont do anything
	}

	uint albedo = ((src_process_voxels.data[index].albedo & 0x7FFF) << 1) | 1; //add solid bit
	imageStore(dst_albedo, write_pos, uvec4(albedo));

	uint facing = (src_process_voxels.data[index].albedo >> 15) & 0x3F; //6 anisotropic facing bits
	imageStore(dst_facing, write_pos, uvec4(facing));

	uint light = src_process_voxels.data[index].light & 0x3fffffff; //30 bits of RGBE8985
	imageStore(dst_light, write_pos, uvec4(light));

	uint light_aniso = src_process_voxels.data[index].light_aniso & 0x3fffffff; //30 bits of 6 anisotropic 5 bits values
	imageStore(dst_light_aniso, write_pos, uvec4(light_aniso));

#endif

#ifdef MODE_SCROLL_OCCLUSION

	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
	if (any(greaterThanEqual(pos, ivec3(params.grid_size) - abs(params.scroll)))) { //too large, do nothing
		return;
	}

	ivec3 read_pos = pos + max(ivec3(0), -params.scroll);
	ivec3 write_pos = pos + max(ivec3(0), params.scroll);

	read_pos.z += params.cascade * params.grid_size;
	uint occlusion = imageLoad(src_occlusion, read_pos).r;
	read_pos.x += params.grid_size;
	occlusion |= imageLoad(src_occlusion, read_pos).r << 16;

	const uint occlusion_shift[8] = uint[](12, 8, 4, 0, 28, 24, 20, 16);

	for (uint i = 0; i < 8; i++) {
		float o = float((occlusion >> occlusion_shift[i]) & 0xF) / 15.0;
		imageStore(dst_occlusion[i], write_pos, vec4(o));
	}

#endif

#ifdef MODE_INITIALIZE_JUMP_FLOOD

	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);

	uint c = imageLoad(src_color, pos).r;
	uvec4 v;
	if (bool(c & 0x1)) {
		//bit set means this is solid
		v.xyz = uvec3(pos);
		v.w = 255; //not zero means used
	} else {
		v.xyz = uvec3(0);
		v.w = 0; // zero means unused
	}

	imageStore(dst_positions, pos, v);
#endif

#ifdef MODE_INITIALIZE_JUMP_FLOOD_HALF

	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
	ivec3 base_pos = pos * 2;

	//since we store in half size, lets kind of randomize what we store, so
	//the half size jump flood has a bit better chance to find something
	uvec4 closest[8];
	int closest_count = 0;

	for (uint i = 0; i < 8; i++) {
		ivec3 src_pos = base_pos + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
		uint c = imageLoad(src_color, src_pos).r;
		if (bool(c & 1)) {
			uvec4 v = uvec4(uvec3(src_pos), 255);
			closest[closest_count] = v;
			closest_count++;
		}
	}

	if (closest_count == 0) {
		imageStore(dst_positions, pos, uvec4(0));
	} else {
		ivec3 indexv = (pos & ivec3(1, 1, 1)) * ivec3(1, 2, 4);
		int index = (indexv.x | indexv.y | indexv.z) % closest_count;
		imageStore(dst_positions, pos, closest[index]);
	}

#endif

#ifdef MODE_JUMPFLOOD

	//regular jumpflood, efficient for large steps, inefficient for small steps
	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);

	vec3 posf = vec3(pos);

	if (params.half_size) {
		posf = posf * 2.0 + 0.5;
	}

	uvec4 p = imageLoad(src_positions, pos);

	if (!params.half_size && p == uvec4(uvec3(pos), 255)) {
		imageStore(dst_positions, pos, p);
		return; //points to itself and valid, nothing better can be done, just pass
	}

	float p_dist;

	if (p.w != 0) {
		p_dist = distance(posf, vec3(p.xyz));
	} else {
		p_dist = 0.0; //should not matter
	}

	const uint offset_count = 26;
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

	for (uint i = 0; i < offset_count; i++) {
		ivec3 ofs = pos + offsets[i] * params.step_size;
		if (any(lessThan(ofs, ivec3(0))) || any(greaterThanEqual(ofs, ivec3(params.grid_size)))) {
			continue;
		}
		uvec4 q = imageLoad(src_positions, ofs);

		if (q.w == 0) {
			continue; //was not initialized yet, ignore
		}

		float q_dist = distance(posf, vec3(q.xyz));
		if (p.w == 0 || q_dist < p_dist) {
			p = q; //just replace because current is unused
			p_dist = q_dist;
		}
	}

	imageStore(dst_positions, pos, p);
#endif

#ifdef MODE_JUMPFLOOD_OPTIMIZED
	//optimized version using shared compute memory

	ivec3 group_offset = ivec3(gl_WorkGroupID.xyz) % params.step_size;
	ivec3 group_pos = group_offset + (ivec3(gl_WorkGroupID.xyz) / params.step_size) * ivec3(GROUP_SIZE * params.step_size);

	//load data into local group memory

	if (all(lessThan(ivec3(gl_LocalInvocationID.xyz), ivec3((GROUP_SIZE + 2) / 2)))) {
		//use this thread for loading, this method uses less threads for this but its simpler and less divergent
		ivec3 base_pos = ivec3(gl_LocalInvocationID.xyz) * 2;
		for (uint i = 0; i < 8; i++) {
			ivec3 load_pos = base_pos + ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
			ivec3 load_global_pos = group_pos + (load_pos - ivec3(1)) * params.step_size;
			uvec4 q;
			if (all(greaterThanEqual(load_global_pos, ivec3(0))) && all(lessThan(load_global_pos, ivec3(params.grid_size)))) {
				q = imageLoad(src_positions, load_global_pos);
			} else {
				q = uvec4(0); //unused
			}

			group_store(load_pos, q);
		}
	}

	ivec3 global_pos = group_pos + ivec3(gl_LocalInvocationID.xyz) * params.step_size;

	if (any(lessThan(global_pos, ivec3(0))) || any(greaterThanEqual(global_pos, ivec3(params.grid_size)))) {
		return; //do nothing else, end here because outside range
	}

	//sync
	groupMemoryBarrier();
	barrier();

	ivec3 local_pos = ivec3(gl_LocalInvocationID.xyz) + ivec3(1);

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

	//only makes sense if point is inside screen
	uvec4 closest = uvec4(0);
	float closest_dist = 0.0;

	vec3 posf = vec3(global_pos);

	if (params.half_size) {
		posf = posf * 2.0 + 0.5;
	}

	for (uint i = 0; i < offset_count; i++) {
		uvec4 point = group_load(local_pos + offsets[i]);

		if (point.w == 0) {
			continue; //was not initialized yet, ignore
		}

		float dist = distance(posf, vec3(point.xyz));
		if (closest.w == 0 || dist < closest_dist) {
			closest = point;
			closest_dist = dist;
		}
	}

	imageStore(dst_positions, global_pos, closest);

#endif

#ifdef MODE_UPSCALE_JUMP_FLOOD

	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);

	uint c = imageLoad(src_color, pos).r;
	uvec4 v;
	if (bool(c & 1)) {
		//bit set means this is solid
		v.xyz = uvec3(pos);
		v.w = 255; //not zero means used
	} else {
		v = imageLoad(src_positions_half, pos >> 1);
		float d = length(vec3(ivec3(v.xyz) - pos));

		ivec3 vbase = ivec3(v.xyz - (v.xyz & uvec3(1)));

		//search around if there is a better candidate from the same block
		for (int i = 0; i < 8; i++) {
			ivec3 bits = ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
			ivec3 p = vbase + bits;

			float d2 = length(vec3(p - pos));
			if (d2 < d) { //check valid distance before test so we avoid a read
				uint c2 = imageLoad(src_color, p).r;
				if (bool(c2 & 1)) {
					v.xyz = uvec3(p);
					d = d2;
				}
			}
		}

		//could validate better position..
	}

	imageStore(dst_positions, pos, v);

#endif

#ifdef MODE_OCCLUSION

	uint invocation_idx = uint(gl_LocalInvocationID.x);
	ivec3 region = ivec3(gl_WorkGroupID);

	ivec3 region_offset = -ivec3(OCCLUSION_SIZE);
	region_offset += region * OCCLUSION_SIZE * 2;
	region_offset += params.probe_offset * OCCLUSION_SIZE;

	if (params.scroll != ivec3(0)) {
		//validate scroll region
		ivec3 region_offset_to = region_offset + ivec3(OCCLUSION_SIZE * 2);
		uvec3 scroll_mask = uvec3(notEqual(params.scroll, ivec3(0))); //save which axes acre scrolling
		ivec3 scroll_from = mix(ivec3(0), ivec3(params.grid_size) + params.scroll, lessThan(params.scroll, ivec3(0)));
		ivec3 scroll_to = mix(ivec3(params.grid_size), params.scroll, greaterThan(params.scroll, ivec3(0)));

		if ((uvec3(lessThanEqual(region_offset_to, scroll_from)) | uvec3(greaterThanEqual(region_offset, scroll_to))) * scroll_mask == scroll_mask) { //all axes that scroll are out, exit
			return; //region outside scroll bounds, quit
		}
	}

#define OCC_HALF_SIZE (OCCLUSION_SIZE / 2)

	ivec3 local_ofs = ivec3(uvec3(invocation_idx % OCC_HALF_SIZE, (invocation_idx % (OCC_HALF_SIZE * OCC_HALF_SIZE)) / OCC_HALF_SIZE, invocation_idx / (OCC_HALF_SIZE * OCC_HALF_SIZE))) * 4;

	/*	for(int i=0;i<64;i++) {
		ivec3 offset = region_offset + local_ofs + ((ivec3(i) >> ivec3(0,2,4)) & ivec3(3,3,3));
		uint facig =
		if (all(greaterThanEqual(offset,ivec3(0))) && all(lessThan(offset,ivec3(params.grid_size)))) {*/

	for (int i = 0; i < 16; i++) { //skip x, so it can be packed

		ivec3 offset = local_ofs + ((ivec3(i * 4) >> ivec3(0, 2, 4)) & ivec3(3, 3, 3));

		uint facing_pack = 0;
		for (int j = 0; j < 4; j++) {
			ivec3 foffset = region_offset + offset + ivec3(j, 0, 0);
			if (all(greaterThanEqual(foffset, ivec3(0))) && all(lessThan(foffset, ivec3(params.grid_size)))) {
				uint f = imageLoad(src_facing, foffset).r;
				facing_pack |= f << (j * 8);
			}
		}

		occlusion_facing[(offset.z * (OCCLUSION_SIZE * 2 * OCCLUSION_SIZE * 2) + offset.y * (OCCLUSION_SIZE * 2) + offset.x) / 4] = facing_pack;
	}

	//sync occlusion saved
	groupMemoryBarrier();
	barrier();

	//process occlusion

#define OCC_STEPS (OCCLUSION_SIZE * 3 - 2)
#define OCC_HALF_STEPS (OCC_STEPS / 2)

	for (int step = 0; step < OCC_STEPS; step++) {
		bool shrink = step >= OCC_HALF_STEPS;
		int occ_step = shrink ? OCC_HALF_STEPS - (step - OCC_HALF_STEPS) - 1 : step;

		if (invocation_idx < group_size_offset[occ_step].x) {
			uint pv = group_pos[group_size_offset[occ_step].y + invocation_idx];
			ivec3 proc_abs = (ivec3(int(pv)) >> ivec3(0, 8, 16)) & ivec3(0xFF);

			if (shrink) {
				proc_abs = ivec3(OCCLUSION_SIZE) - proc_abs - ivec3(1);
			}

			for (int i = 0; i < 8; i++) {
				ivec3 bits = ((ivec3(i) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1));
				ivec3 proc_sign = bits * 2 - 1;
				ivec3 local_offset = ivec3(OCCLUSION_SIZE) + proc_abs * proc_sign - (ivec3(1) - bits);
				ivec3 offset = local_offset + region_offset;
				if (all(greaterThanEqual(offset, ivec3(0))) && all(lessThan(offset, ivec3(params.grid_size)))) {
					float occ;

					uint facing = get_facing(local_offset);

					if (facing != 0) { //solid
						occ = 0.0;
					} else if (step == 0) {
#if 0
						occ = 0.0;
						if (get_facing(local_offset - ivec3(proc_sign.x,0,0))==0) {
							occ+=1.0;
						}
						if (get_facing(local_offset - ivec3(0,proc_sign.y,0))==0) {
							occ+=1.0;
						}
						if (get_facing(local_offset - ivec3(0,0,proc_sign.z))==0) {
							occ+=1.0;
						}
						/*
						if (get_facing(local_offset - proc_sign)==0) {
							occ+=1.0;
						}*/

						occ/=3.0;
#endif
						occ = 1.0;

					} else {
						ivec3 read_dir = -proc_sign;

						ivec3 major_axis;
						if (proc_abs.x < proc_abs.y) {
							if (proc_abs.z < proc_abs.y) {
								major_axis = ivec3(0, 1, 0);
							} else {
								major_axis = ivec3(0, 0, 1);
							}
						} else {
							if (proc_abs.z < proc_abs.x) {
								major_axis = ivec3(1, 0, 0);
							} else {
								major_axis = ivec3(0, 0, 1);
							}
						}

						float avg = 0.0;
						occ = 0.0;

						ivec3 read_x = offset + ivec3(read_dir.x, 0, 0) + (proc_abs.x == 0 ? major_axis * read_dir : ivec3(0));
						ivec3 read_y = offset + ivec3(0, read_dir.y, 0) + (proc_abs.y == 0 ? major_axis * read_dir : ivec3(0));
						ivec3 read_z = offset + ivec3(0, 0, read_dir.z) + (proc_abs.z == 0 ? major_axis * read_dir : ivec3(0));

						uint facing_x = get_facing(read_x - region_offset);
						if (facing_x == 0) {
							if (all(greaterThanEqual(read_x, ivec3(0))) && all(lessThan(read_x, ivec3(params.grid_size)))) {
								occ += imageLoad(dst_occlusion[params.occlusion_index], read_x).r;
								avg += 1.0;
							}
						} else {
							if (proc_abs.x != 0) { //do not occlude from voxels in the opposite octant
								avg += 1.0;
							}
						}

						uint facing_y = get_facing(read_y - region_offset);
						if (facing_y == 0) {
							if (all(greaterThanEqual(read_y, ivec3(0))) && all(lessThan(read_y, ivec3(params.grid_size)))) {
								occ += imageLoad(dst_occlusion[params.occlusion_index], read_y).r;
								avg += 1.0;
							}
						} else {
							if (proc_abs.y != 0) {
								avg += 1.0;
							}
						}

						uint facing_z = get_facing(read_z - region_offset);
						if (facing_z == 0) {
							if (all(greaterThanEqual(read_z, ivec3(0))) && all(lessThan(read_z, ivec3(params.grid_size)))) {
								occ += imageLoad(dst_occlusion[params.occlusion_index], read_z).r;
								avg += 1.0;
							}
						} else {
							if (proc_abs.z != 0) {
								avg += 1.0;
							}
						}

						if (avg > 0.0) {
							occ /= avg;
						}
					}

					imageStore(dst_occlusion[params.occlusion_index], offset, vec4(occ));
				}
			}
		}

		groupMemoryBarrier();
		barrier();
	}
#if 1
	//bias solid voxels away

	for (int i = 0; i < 64; i++) {
		ivec3 local_offset = local_ofs + ((ivec3(i) >> ivec3(0, 2, 4)) & ivec3(3, 3, 3));
		ivec3 offset = region_offset + local_offset;

		if (all(greaterThanEqual(offset, ivec3(0))) && all(lessThan(offset, ivec3(params.grid_size)))) {
			uint facing = get_facing(local_offset);

			if (facing != 0) {
				//only work on solids

				ivec3 proc_pos = local_offset - ivec3(OCCLUSION_SIZE);
				proc_pos += mix(ivec3(0), ivec3(1), greaterThanEqual(proc_pos, ivec3(0)));

				float avg = 0.0;
				float occ = 0.0;

				ivec3 read_dir = -sign(proc_pos);
				ivec3 read_dir_x = ivec3(read_dir.x, 0, 0);
				ivec3 read_dir_y = ivec3(0, read_dir.y, 0);
				ivec3 read_dir_z = ivec3(0, 0, read_dir.z);
				//solid
#if 0

				uvec3 facing_pos_base = (uvec3(facing) >> uvec3(0,1,2)) & uvec3(1,1,1);
				uvec3 facing_neg_base = (uvec3(facing) >> uvec3(3,4,5)) & uvec3(1,1,1);
				uvec3 facing_pos=  facing_pos_base &((~facing_neg_base)&uvec3(1,1,1));
				uvec3 facing_neg=  facing_neg_base &((~facing_pos_base)&uvec3(1,1,1));
#else
				uvec3 facing_pos = (uvec3(facing) >> uvec3(0, 1, 2)) & uvec3(1, 1, 1);
				uvec3 facing_neg = (uvec3(facing) >> uvec3(3, 4, 5)) & uvec3(1, 1, 1);
#endif
				bvec3 read_valid = bvec3(mix(facing_neg, facing_pos, greaterThan(read_dir, ivec3(0))));

				//sides
				if (read_valid.x) {
					ivec3 read_offset = local_offset + read_dir_x;
					uint f = get_facing(read_offset);
					if (f == 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occ += imageLoad(dst_occlusion[params.occlusion_index], read_offset).r;
							avg += 1.0;
						}
					}
				}

				if (read_valid.y) {
					ivec3 read_offset = local_offset + read_dir_y;
					uint f = get_facing(read_offset);
					if (f == 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occ += imageLoad(dst_occlusion[params.occlusion_index], read_offset).r;
							avg += 1.0;
						}
					}
				}

				if (read_valid.z) {
					ivec3 read_offset = local_offset + read_dir_z;
					uint f = get_facing(read_offset);
					if (f == 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occ += imageLoad(dst_occlusion[params.occlusion_index], read_offset).r;
							avg += 1.0;
						}
					}
				}

				//adjacents

				if (all(read_valid.yz)) {
					ivec3 read_offset = local_offset + read_dir_y + read_dir_z;
					uint f = get_facing(read_offset);
					if (f == 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occ += imageLoad(dst_occlusion[params.occlusion_index], read_offset).r;
							avg += 1.0;
						}
					}
				}

				if (all(read_valid.xz)) {
					ivec3 read_offset = local_offset + read_dir_x + read_dir_z;
					uint f = get_facing(read_offset);
					if (f == 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occ += imageLoad(dst_occlusion[params.occlusion_index], read_offset).r;
							avg += 1.0;
						}
					}
				}

				if (all(read_valid.xy)) {
					ivec3 read_offset = local_offset + read_dir_x + read_dir_y;
					uint f = get_facing(read_offset);
					if (f == 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occ += imageLoad(dst_occlusion[params.occlusion_index], read_offset).r;
							avg += 1.0;
						}
					}
				}

				//diagonal

				if (all(read_valid)) {
					ivec3 read_offset = local_offset + read_dir;
					uint f = get_facing(read_offset);
					if (f == 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occ += imageLoad(dst_occlusion[params.occlusion_index], read_offset).r;
							avg += 1.0;
						}
					}
				}

				if (avg > 0.0) {
					occ /= avg;
				}

				imageStore(dst_occlusion[params.occlusion_index], offset, vec4(occ));
			}
		}
	}

#endif

#if 1
	groupMemoryBarrier();
	barrier();

	for (int i = 0; i < 64; i++) {
		ivec3 local_offset = local_ofs + ((ivec3(i) >> ivec3(0, 2, 4)) & ivec3(3, 3, 3));
		ivec3 offset = region_offset + local_offset;

		if (all(greaterThanEqual(offset, ivec3(0))) && all(lessThan(offset, ivec3(params.grid_size)))) {
			uint facing = get_facing(local_offset);

			if (facing == 0) {
				ivec3 proc_pos = local_offset - ivec3(OCCLUSION_SIZE);
				proc_pos += mix(ivec3(0), ivec3(1), greaterThanEqual(proc_pos, ivec3(0)));

				ivec3 proc_abs = abs(proc_pos);

				ivec3 read_dir = sign(proc_pos); //opposite direction
				ivec3 read_dir_x = ivec3(read_dir.x, 0, 0);
				ivec3 read_dir_y = ivec3(0, read_dir.y, 0);
				ivec3 read_dir_z = ivec3(0, 0, read_dir.z);
				//solid
				uvec3 read_mask = mix(uvec3(1, 2, 4), uvec3(8, 16, 32), greaterThan(read_dir, ivec3(0))); //match positive with negative normals
				uvec3 block_mask = mix(uvec3(1, 2, 4), uvec3(8, 16, 32), lessThan(read_dir, ivec3(0))); //match positive with negative normals

				block_mask = uvec3(0);

				float visible = 0.0;
				float occlude_total = 0.0;

				if (proc_abs.x < OCCLUSION_SIZE) {
					ivec3 read_offset = local_offset + read_dir_x;
					uint x_mask = get_facing(read_offset);
					if (x_mask != 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occlude_total += 1.0;
							if (bool(x_mask & read_mask.x) && !bool(x_mask & block_mask.x)) {
								visible += 1.0;
							}
						}
					}
				}

				if (proc_abs.y < OCCLUSION_SIZE) {
					ivec3 read_offset = local_offset + read_dir_y;
					uint y_mask = get_facing(read_offset);
					if (y_mask != 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occlude_total += 1.0;
							if (bool(y_mask & read_mask.y) && !bool(y_mask & block_mask.y)) {
								visible += 1.0;
							}
						}
					}
				}

				if (proc_abs.z < OCCLUSION_SIZE) {
					ivec3 read_offset = local_offset + read_dir_z;
					uint z_mask = get_facing(read_offset);
					if (z_mask != 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occlude_total += 1.0;
							if (bool(z_mask & read_mask.z) && !bool(z_mask & block_mask.z)) {
								visible += 1.0;
							}
						}
					}
				}

				//if near the cartesian plane, test in opposite direction too

				read_mask = mix(uvec3(1, 2, 4), uvec3(8, 16, 32), lessThan(read_dir, ivec3(0))); //match negative with positive normals
				block_mask = mix(uvec3(1, 2, 4), uvec3(8, 16, 32), greaterThan(read_dir, ivec3(0))); //match negative with positive normals
				block_mask = uvec3(0);

				if (proc_abs.x == 1) {
					ivec3 read_offset = local_offset - read_dir_x;
					uint x_mask = get_facing(read_offset);
					if (x_mask != 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occlude_total += 1.0;
							if (bool(x_mask & read_mask.x) && !bool(x_mask & block_mask.x)) {
								visible += 1.0;
							}
						}
					}
				}

				if (proc_abs.y == 1) {
					ivec3 read_offset = local_offset - read_dir_y;
					uint y_mask = get_facing(read_offset);
					if (y_mask != 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occlude_total += 1.0;
							if (bool(y_mask & read_mask.y) && !bool(y_mask & block_mask.y)) {
								visible += 1.0;
							}
						}
					}
				}

				if (proc_abs.z == 1) {
					ivec3 read_offset = local_offset - read_dir_z;
					uint z_mask = get_facing(read_offset);
					if (z_mask != 0) {
						read_offset += region_offset;
						if (all(greaterThanEqual(read_offset, ivec3(0))) && all(lessThan(read_offset, ivec3(params.grid_size)))) {
							occlude_total += 1.0;
							if (bool(z_mask & read_mask.z) && !bool(z_mask & block_mask.z)) {
								visible += 1.0;
							}
						}
					}
				}

				if (occlude_total > 0.0) {
					float occ = imageLoad(dst_occlusion[params.occlusion_index], offset).r;
					occ *= visible / occlude_total;
					imageStore(dst_occlusion[params.occlusion_index], offset, vec4(occ));
				}
			}
		}
	}

#endif

	/*
	for(int i=0;i<8;i++) {
		ivec3 local_offset = local_pos + ((ivec3(i) >> ivec3(2,1,0)) & ivec3(1,1,1)) * OCCLUSION_SIZE;
		ivec3 offset = local_offset - ivec3(OCCLUSION_SIZE); //looking around probe, so starts negative
		offset += region * OCCLUSION_SIZE * 2; //offset by region
		offset += params.probe_offset * OCCLUSION_SIZE; // offset by probe offset
		if (all(greaterThanEqual(offset,ivec3(0))) && all(lessThan(offset,ivec3(params.grid_size)))) {
			imageStore(dst_occlusion[params.occlusion_index],offset,vec4( occlusion_data[ to_linear(local_offset) ]  ));
			//imageStore(dst_occlusion[params.occlusion_index],offset,vec4( occlusion_solid[ to_linear(local_offset) ] ));
		}
	}
*/

#endif

#ifdef MODE_STORE

	ivec3 local = ivec3(gl_LocalInvocationID.xyz);
	ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
	// store SDF
	uvec4 p = imageLoad(src_positions, pos);

	bool solid = false;
	float d;
	if (ivec3(p.xyz) == pos) {
		//solid block
		d = 0;
		solid = true;
	} else {
		//distance block
		d = 1.0 + length(vec3(p.xyz) - vec3(pos));
	}

	d /= 255.0;

	imageStore(dst_sdf, pos, vec4(d));

	// STORE OCCLUSION

	uint occlusion = 0;
	const uint occlusion_shift[8] = uint[](12, 8, 4, 0, 28, 24, 20, 16);
	for (int i = 0; i < 8; i++) {
		float occ = imageLoad(src_occlusion[i], pos).r;
		occlusion |= uint(clamp(occ * 15.0, 0.0, 15.0)) << occlusion_shift[i];
	}
	{
		ivec3 occ_pos = pos;
		occ_pos.z += params.cascade * params.grid_size;
		imageStore(dst_occlusion, occ_pos, uvec4(occlusion & 0xFFFF));
		occ_pos.x += params.grid_size;
		imageStore(dst_occlusion, occ_pos, uvec4(occlusion >> 16));
	}

	// STORE POSITIONS

	if (local == ivec3(0)) {
		store_position_count = 0; //base one stores as zero, the others wait
	}

	groupMemoryBarrier();
	barrier();

	if (solid) {
		uint index = atomicAdd(store_position_count, 1);
		// At least do the conversion work in parallel
		store_positions[index].position = uint(pos.x | (pos.y << 7) | (pos.z << 14));

		//see around which voxels point to this one, add them to the list
		uint bit_index = 0;
		uint neighbour_bits = 0;
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				for (int k = -1; k <= 1; k++) {
					if (i == 0 && j == 0 && k == 0) {
						continue;
					}
					ivec3 npos = pos + ivec3(i, j, k);
					if (all(greaterThanEqual(npos, ivec3(0))) && all(lessThan(npos, ivec3(params.grid_size)))) {
						p = imageLoad(src_positions, npos);
						if (ivec3(p.xyz) == pos) {
							neighbour_bits |= (1 << bit_index);
						}
					}
					bit_index++;
				}
			}
		}

		uint rgb = imageLoad(src_albedo, pos).r;
		uint facing = imageLoad(src_facing, pos).r;

		store_positions[index].albedo = rgb >> 1; //store as it comes (555) to avoid precision loss (and move away the alpha bit)
		store_positions[index].albedo |= (facing & 0x3F) << 15; // store facing in bits 15-21

		store_positions[index].albedo |= neighbour_bits << 21; //store lower 11 bits of neighbours with remaining albedo
		store_positions[index].position |= (neighbour_bits >> 11) << 21; //store 11 bits more of neighbours with position

		store_positions[index].light = imageLoad(src_light, pos).r;
		store_positions[index].light_aniso = imageLoad(src_light_aniso, pos).r;
		//add neighbours
		store_positions[index].light |= (neighbour_bits >> 22) << 30; //store 2 bits more of neighbours with light
		store_positions[index].light_aniso |= (neighbour_bits >> 24) << 30; //store 2 bits more of neighbours with aniso
	}

	groupMemoryBarrier();
	barrier();

	// global increment only once per group, to reduce pressure

	if (local == ivec3(0) && store_position_count > 0) {
		store_from_index = atomicAdd(dispatch_data.total_count, store_position_count);
		uint group_count = (store_from_index + store_position_count - 1) / 64 + 1;
		atomicMax(dispatch_data.x, group_count);
	}

	groupMemoryBarrier();
	barrier();

	uint read_index = uint(local.z * 4 * 4 + local.y * 4 + local.x);
	uint write_index = store_from_index + read_index;

	if (read_index < store_position_count) {
		dst_process_voxels.data[write_index] = store_positions[read_index];
	}

	if (pos == ivec3(0)) {
		//this thread clears y and z
		dispatch_data.y = 1;
		dispatch_data.z = 1;
	}
#endif
}
