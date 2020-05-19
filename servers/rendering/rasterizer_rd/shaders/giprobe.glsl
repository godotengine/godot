#[compute]

#version 450

VERSION_DEFINES

#ifdef MODE_DYNAMIC
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
#else
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
#endif

#ifndef MODE_DYNAMIC

#define NO_CHILDREN 0xFFFFFFFF
#define GREY_VEC vec3(0.33333, 0.33333, 0.33333)

struct CellChildren {
	uint children[8];
};

layout(set = 0, binding = 1, std430) buffer CellChildrenBuffer {
	CellChildren data[];
}
cell_children;

struct CellData {
	uint position; // xyz 10 bits
	uint albedo; //rgb albedo
	uint emission; //rgb normalized with e as multiplier
	uint normal; //RGB normal encoded
};

layout(set = 0, binding = 2, std430) buffer CellDataBuffer {
	CellData data[];
}
cell_data;

#endif // MODE DYNAMIC

#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_OMNI 1
#define LIGHT_TYPE_SPOT 2

#if defined(MODE_COMPUTE_LIGHT) || defined(MODE_DYNAMIC_LIGHTING)

struct Light {
	uint type;
	float energy;
	float radius;
	float attenuation;

	vec3 color;
	float spot_angle_radians;

	vec3 position;
	float spot_attenuation;

	vec3 direction;
	bool has_shadow;
};

layout(set = 0, binding = 3, std140) uniform Lights {
	Light data[MAX_LIGHTS];
}
lights;

#endif // MODE COMPUTE LIGHT

#ifdef MODE_SECOND_BOUNCE

layout(set = 0, binding = 5) uniform texture3D color_texture;

#ifdef MODE_ANISOTROPIC
layout(set = 0, binding = 7) uniform texture3D aniso_pos_texture;
layout(set = 0, binding = 8) uniform texture3D aniso_neg_texture;
#endif // MODE ANISOTROPIC

#endif // MODE_SECOND_BOUNCE

#ifndef MODE_DYNAMIC

layout(push_constant, binding = 0, std430) uniform Params {
	ivec3 limits;
	uint stack_size;

	float emission_scale;
	float propagation;
	float dynamic_range;

	uint light_count;
	uint cell_offset;
	uint cell_count;
	float aniso_strength;
	uint pad;
}
params;

layout(set = 0, binding = 4, std430) buffer Outputs {
	vec4 data[];
}
outputs;

#endif // MODE DYNAMIC

layout(set = 0, binding = 9) uniform texture3D texture_sdf;
layout(set = 0, binding = 10) uniform sampler texture_sampler;

#ifdef MODE_WRITE_TEXTURE

layout(rgba8, set = 0, binding = 5) uniform restrict writeonly image3D color_tex;

#ifdef MODE_ANISOTROPIC

layout(r16ui, set = 0, binding = 6) uniform restrict writeonly uimage3D aniso_pos_tex;
layout(r16ui, set = 0, binding = 7) uniform restrict writeonly uimage3D aniso_neg_tex;

#endif

#endif

#ifdef MODE_DYNAMIC

layout(push_constant, binding = 0, std430) uniform Params {
	ivec3 limits;
	uint light_count; //when not lighting
	ivec3 x_dir;
	float z_base;
	ivec3 y_dir;
	float z_sign;
	ivec3 z_dir;
	float pos_multiplier;
	ivec2 rect_pos;
	ivec2 rect_size;
	ivec2 prev_rect_ofs;
	ivec2 prev_rect_size;
	bool flip_x;
	bool flip_y;
	float dynamic_range;
	bool on_mipmap;
	float propagation;
	float pad[3];
}
params;

#ifdef MODE_DYNAMIC_LIGHTING

layout(rgba8, set = 0, binding = 5) uniform restrict readonly image2D source_albedo;
layout(rgba8, set = 0, binding = 6) uniform restrict readonly image2D source_normal;
layout(rgba8, set = 0, binding = 7) uniform restrict readonly image2D source_orm;
//layout (set=0,binding=8) uniform texture2D source_depth;
layout(rgba16f, set = 0, binding = 11) uniform restrict image2D emission;
layout(r32f, set = 0, binding = 12) uniform restrict image2D depth;

#endif

#ifdef MODE_DYNAMIC_SHRINK

layout(rgba16f, set = 0, binding = 5) uniform restrict readonly image2D source_light;
layout(r32f, set = 0, binding = 6) uniform restrict readonly image2D source_depth;

#ifdef MODE_DYNAMIC_SHRINK_WRITE

layout(rgba16f, set = 0, binding = 7) uniform restrict writeonly image2D light;
layout(r32f, set = 0, binding = 8) uniform restrict writeonly image2D depth;

#endif // MODE_DYNAMIC_SHRINK_WRITE

#ifdef MODE_DYNAMIC_SHRINK_PLOT

layout(rgba8, set = 0, binding = 11) uniform restrict image3D color_texture;

#ifdef MODE_ANISOTROPIC

layout(r16ui, set = 0, binding = 12) uniform restrict writeonly uimage3D aniso_pos_texture;
layout(r16ui, set = 0, binding = 13) uniform restrict writeonly uimage3D aniso_neg_texture;

#endif // MODE ANISOTROPIC

#endif //MODE_DYNAMIC_SHRINK_PLOT

#endif // MODE_DYNAMIC_SHRINK

//layout (rgba8,set=0,binding=5) uniform restrict writeonly image3D color_tex;

#endif // MODE DYNAMIC

#if defined(MODE_COMPUTE_LIGHT) || defined(MODE_DYNAMIC_LIGHTING)

float raymarch(float distance, float distance_adv, vec3 from, vec3 direction) {
	vec3 cell_size = 1.0 / vec3(params.limits);
	float occlusion = 1.0;
	while (distance > 0.5) { //use this to avoid precision errors
		float advance = texture(sampler3D(texture_sdf, texture_sampler), from * cell_size).r * 255.0 - 1.0;
		if (advance < 0.0) {
			occlusion = 0.0;
			break;
		}

		occlusion = min(advance, occlusion);

		advance = max(distance_adv, advance - mod(advance, distance_adv)); //should always advance in multiples of distance_adv

		from += direction * advance;
		distance -= advance;
	}

	return occlusion; //max(0.0,distance);
}

bool compute_light_vector(uint light, vec3 pos, out float attenuation, out vec3 light_pos) {
	if (lights.data[light].type == LIGHT_TYPE_DIRECTIONAL) {
		light_pos = pos - lights.data[light].direction * length(vec3(params.limits));
		attenuation = 1.0;

	} else {
		light_pos = lights.data[light].position;
		float distance = length(pos - light_pos);
		if (distance >= lights.data[light].radius) {
			return false;
		}

		attenuation = pow(clamp(1.0 - distance / lights.data[light].radius, 0.0001, 1.0), lights.data[light].attenuation);

		if (lights.data[light].type == LIGHT_TYPE_SPOT) {
			vec3 rel = normalize(pos - light_pos);
			float angle = acos(dot(rel, lights.data[light].direction));
			if (angle > lights.data[light].spot_angle_radians) {
				return false;
			}

			float d = clamp(angle / lights.data[light].spot_angle_radians, 0, 1);
			attenuation *= pow(1.0 - d, lights.data[light].spot_attenuation);
		}
	}

	return true;
}

float get_normal_advance(vec3 p_normal) {
	vec3 normal = p_normal;
	vec3 unorm = abs(normal);

	if ((unorm.x >= unorm.y) && (unorm.x >= unorm.z)) {
		// x code
		unorm = normal.x > 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(-1.0, 0.0, 0.0);
	} else if ((unorm.y > unorm.x) && (unorm.y >= unorm.z)) {
		// y code
		unorm = normal.y > 0.0 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, -1.0, 0.0);
	} else if ((unorm.z > unorm.x) && (unorm.z > unorm.y)) {
		// z code
		unorm = normal.z > 0.0 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 0.0, -1.0);
	} else {
		// oh-no we messed up code
		// has to be
		unorm = vec3(1.0, 0.0, 0.0);
	}

	return 1.0 / dot(normal, unorm);
}

void clip_segment(vec4 plane, vec3 begin, inout vec3 end) {
	vec3 segment = begin - end;
	float den = dot(plane.xyz, segment);

	//printf("den is %i\n",den);
	if (den < 0.0001) {
		return;
	}

	float dist = (dot(plane.xyz, begin) - plane.w) / den;

	if (dist < 0.0001 || dist > 1.0001) {
		return;
	}

	end = begin + segment * -dist;
}

bool compute_light_at_pos(uint index, vec3 pos, vec3 normal, inout vec3 light, inout vec3 light_dir) {
	float attenuation;
	vec3 light_pos;

	if (!compute_light_vector(index, pos, attenuation, light_pos)) {
		return false;
	}

	light_dir = normalize(pos - light_pos);

	if (attenuation < 0.01 || (length(normal) > 0.2 && dot(normal, light_dir) >= 0)) {
		return false; //not facing the light, or attenuation is near zero
	}

	if (lights.data[index].has_shadow) {
		float distance_adv = get_normal_advance(light_dir);

		vec3 to = pos;
		if (length(normal) > 0.2) {
			to += normal * distance_adv * 0.51;
		} else {
			to -= sign(light_dir) * 0.45; //go near the edge towards the light direction to avoid self occlusion
		}

		//clip
		clip_segment(mix(vec4(-1.0, 0.0, 0.0, 0.0), vec4(1.0, 0.0, 0.0, float(params.limits.x - 1)), bvec4(light_dir.x < 0.0)), to, light_pos);
		clip_segment(mix(vec4(0.0, -1.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, float(params.limits.y - 1)), bvec4(light_dir.y < 0.0)), to, light_pos);
		clip_segment(mix(vec4(0.0, 0.0, -1.0, 0.0), vec4(0.0, 0.0, 1.0, float(params.limits.z - 1)), bvec4(light_dir.z < 0.0)), to, light_pos);

		float distance = length(to - light_pos);
		if (distance < 0.1) {
			return false; // hit
		}

		distance += distance_adv - mod(distance, distance_adv); //make it reach the center of the box always
		light_pos = to - light_dir * distance;

		//from -= sign(light_dir)*0.45; //go near the edge towards the light direction to avoid self occlusion

		/*float dist = raymarch(distance,distance_adv,light_pos,light_dir);

		if (dist > distance_adv) {
			return false;
		}

		attenuation *= 1.0 - smoothstep(0.1*distance_adv,distance_adv,dist);
		*/

		float occlusion = raymarch(distance, distance_adv, light_pos, light_dir);

		if (occlusion == 0.0) {
			return false;
		}

		attenuation *= occlusion; //1.0 - smoothstep(0.1*distance_adv,distance_adv,dist);
	}

	light = lights.data[index].color * attenuation * lights.data[index].energy;
	return true;
}

#endif // MODE COMPUTE LIGHT

void main() {
#ifndef MODE_DYNAMIC

	uint cell_index = gl_GlobalInvocationID.x;
	if (cell_index >= params.cell_count) {
		return;
	}
	cell_index += params.cell_offset;

	uvec3 posu = uvec3(cell_data.data[cell_index].position & 0x7FF, (cell_data.data[cell_index].position >> 11) & 0x3FF, cell_data.data[cell_index].position >> 21);
	vec4 albedo = unpackUnorm4x8(cell_data.data[cell_index].albedo);

#endif

	/////////////////COMPUTE LIGHT///////////////////////////////

#ifdef MODE_COMPUTE_LIGHT

	vec3 pos = vec3(posu) + vec3(0.5);

	vec3 emission = vec3(uvec3(cell_data.data[cell_index].emission & 0x1ff, (cell_data.data[cell_index].emission >> 9) & 0x1ff, (cell_data.data[cell_index].emission >> 18) & 0x1ff)) * pow(2.0, float(cell_data.data[cell_index].emission >> 27) - 15.0 - 9.0);
	vec3 normal = unpackSnorm4x8(cell_data.data[cell_index].normal).xyz;

#ifdef MODE_ANISOTROPIC
	vec3 accum[6] = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
	const vec3 accum_dirs[6] = vec3[](vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0));
#else
	vec3 accum = vec3(0.0);
#endif

	for (uint i = 0; i < params.light_count; i++) {
		vec3 light;
		vec3 light_dir;
		if (!compute_light_at_pos(i, pos, normal.xyz, light, light_dir)) {
			continue;
		}

		light *= albedo.rgb;

#ifdef MODE_ANISOTROPIC
		for (uint j = 0; j < 6; j++) {
			accum[j] += max(0.0, dot(accum_dirs[j], -light_dir)) * light;
		}
#else
		if (length(normal) > 0.2) {
			accum += max(0.0, dot(normal, -light_dir)) * light;
		} else {
			//all directions
			accum += light;
		}
#endif
	}

#ifdef MODE_ANISOTROPIC

	for (uint i = 0; i < 6; i++) {
		vec3 light = accum[i];
		if (length(normal) > 0.2) {
			light += max(0.0, dot(accum_dirs[i], -normal)) * emission;
		} else {
			light += emission;
		}

		outputs.data[cell_index * 6 + i] = vec4(light, 0.0);
	}

#else
	outputs.data[cell_index] = vec4(accum + emission, 0.0);

#endif

#endif //MODE_COMPUTE_LIGHT

	/////////////////SECOND BOUNCE///////////////////////////////

#ifdef MODE_SECOND_BOUNCE
	vec3 pos = vec3(posu) + vec3(0.5);
	ivec3 ipos = ivec3(posu);
	vec4 normal = unpackSnorm4x8(cell_data.data[cell_index].normal);

#ifdef MODE_ANISOTROPIC
	vec3 accum[6];
	const vec3 accum_dirs[6] = vec3[](vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0));

	/*vec3 src_color = texelFetch(sampler3D(color_texture,texture_sampler),ipos,0).rgb * params.dynamic_range;
	vec3 src_aniso_pos = texelFetch(sampler3D(aniso_pos_texture,texture_sampler),ipos,0).rgb;
	vec3 src_anisp_neg = texelFetch(sampler3D(anisp_neg_texture,texture_sampler),ipos,0).rgb;
	accum[0]=src_col * src_aniso_pos.x;
	accum[1]=src_col * src_aniso_neg.x;
	accum[2]=src_col * src_aniso_pos.y;
	accum[3]=src_col * src_aniso_neg.y;
	accum[4]=src_col * src_aniso_pos.z;
	accum[5]=src_col * src_aniso_neg.z;*/

	accum[0] = outputs.data[cell_index * 6 + 0].rgb;
	accum[1] = outputs.data[cell_index * 6 + 1].rgb;
	accum[2] = outputs.data[cell_index * 6 + 2].rgb;
	accum[3] = outputs.data[cell_index * 6 + 3].rgb;
	accum[4] = outputs.data[cell_index * 6 + 4].rgb;
	accum[5] = outputs.data[cell_index * 6 + 5].rgb;

#else
	vec3 accum = outputs.data[cell_index].rgb;

#endif

	if (length(normal.xyz) > 0.2) {
		vec3 v0 = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
		vec3 tangent = normalize(cross(v0, normal.xyz));
		vec3 bitangent = normalize(cross(tangent, normal.xyz));
		mat3 normal_mat = mat3(tangent, bitangent, normal.xyz);

#define MAX_CONE_DIRS 6

		vec3 cone_dirs[MAX_CONE_DIRS] = vec3[](
				vec3(0.0, 0.0, 1.0),
				vec3(0.866025, 0.0, 0.5),
				vec3(0.267617, 0.823639, 0.5),
				vec3(-0.700629, 0.509037, 0.5),
				vec3(-0.700629, -0.509037, 0.5),
				vec3(0.267617, -0.823639, 0.5));

		float cone_weights[MAX_CONE_DIRS] = float[](0.25, 0.15, 0.15, 0.15, 0.15, 0.15);
		float tan_half_angle = 0.577;

		for (int i = 0; i < MAX_CONE_DIRS; i++) {
			vec3 direction = normal_mat * cone_dirs[i];
			vec4 color = vec4(0.0);
			{
				float dist = 1.5;
				float max_distance = length(vec3(params.limits));
				vec3 cell_size = 1.0 / vec3(params.limits);

#ifdef MODE_ANISOTROPIC
				vec3 aniso_normal = mix(direction, normal.xyz, params.aniso_strength);
#endif
				while (dist < max_distance && color.a < 0.95) {
					float diameter = max(1.0, 2.0 * tan_half_angle * dist);
					vec3 uvw_pos = (pos + dist * direction) * cell_size;
					float half_diameter = diameter * 0.5;
					//check if outside, then break
					//if ( any(greaterThan(abs(uvw_pos - 0.5),vec3(0.5f + half_diameter * cell_size)) ) ) {
					//	break;
					//}

					float log2_diameter = log2(diameter);
					vec4 scolor = textureLod(sampler3D(color_texture, texture_sampler), uvw_pos, log2_diameter);
#ifdef MODE_ANISOTROPIC

					vec3 aniso_neg = textureLod(sampler3D(aniso_neg_texture, texture_sampler), uvw_pos, log2_diameter).rgb;
					vec3 aniso_pos = textureLod(sampler3D(aniso_pos_texture, texture_sampler), uvw_pos, log2_diameter).rgb;

					scolor.rgb *= dot(max(vec3(0.0), (aniso_normal * aniso_pos)), vec3(1.0)) + dot(max(vec3(0.0), (-aniso_normal * aniso_neg)), vec3(1.0));
#endif
					float a = (1.0 - color.a);
					color += a * scolor;
					dist += half_diameter;
				}
			}
			color *= cone_weights[i] * vec4(albedo.rgb, 1.0) * params.dynamic_range; //restore range
#ifdef MODE_ANISOTROPIC
			for (uint j = 0; j < 6; j++) {
				accum[j] += max(0.0, dot(accum_dirs[j], direction)) * color.rgb;
			}
#else
			accum += color.rgb;
#endif
		}
	}

#ifdef MODE_ANISOTROPIC

	outputs.data[cell_index * 6 + 0] = vec4(accum[0], 0.0);
	outputs.data[cell_index * 6 + 1] = vec4(accum[1], 0.0);
	outputs.data[cell_index * 6 + 2] = vec4(accum[2], 0.0);
	outputs.data[cell_index * 6 + 3] = vec4(accum[3], 0.0);
	outputs.data[cell_index * 6 + 4] = vec4(accum[4], 0.0);
	outputs.data[cell_index * 6 + 5] = vec4(accum[5], 0.0);
#else
	outputs.data[cell_index] = vec4(accum, 0.0);

#endif

#endif // MODE_SECOND_BOUNCE

	/////////////////UPDATE MIPMAPS///////////////////////////////

#ifdef MODE_UPDATE_MIPMAPS

	{
#ifdef MODE_ANISOTROPIC
		vec3 light_accum[6] = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
#else
		vec3 light_accum = vec3(0.0);
#endif
		float count = 0.0;
		for (uint i = 0; i < 8; i++) {
			uint child_index = cell_children.data[cell_index].children[i];
			if (child_index == NO_CHILDREN) {
				continue;
			}
#ifdef MODE_ANISOTROPIC
			light_accum[0] += outputs.data[child_index * 6 + 0].rgb;
			light_accum[1] += outputs.data[child_index * 6 + 1].rgb;
			light_accum[2] += outputs.data[child_index * 6 + 2].rgb;
			light_accum[3] += outputs.data[child_index * 6 + 3].rgb;
			light_accum[4] += outputs.data[child_index * 6 + 4].rgb;
			light_accum[5] += outputs.data[child_index * 6 + 5].rgb;

#else
			light_accum += outputs.data[child_index].rgb;

#endif

			count += 1.0;
		}

		float divisor = mix(8.0, count, params.propagation);
#ifdef MODE_ANISOTROPIC
		outputs.data[cell_index * 6 + 0] = vec4(light_accum[0] / divisor, 0.0);
		outputs.data[cell_index * 6 + 1] = vec4(light_accum[1] / divisor, 0.0);
		outputs.data[cell_index * 6 + 2] = vec4(light_accum[2] / divisor, 0.0);
		outputs.data[cell_index * 6 + 3] = vec4(light_accum[3] / divisor, 0.0);
		outputs.data[cell_index * 6 + 4] = vec4(light_accum[4] / divisor, 0.0);
		outputs.data[cell_index * 6 + 5] = vec4(light_accum[5] / divisor, 0.0);

#else
		outputs.data[cell_index] = vec4(light_accum / divisor, 0.0);
#endif
	}
#endif

	///////////////////WRITE TEXTURE/////////////////////////////

#ifdef MODE_WRITE_TEXTURE
	{
#ifdef MODE_ANISOTROPIC
		vec3 accum_total = vec3(0.0);
		accum_total += outputs.data[cell_index * 6 + 0].rgb;
		accum_total += outputs.data[cell_index * 6 + 1].rgb;
		accum_total += outputs.data[cell_index * 6 + 2].rgb;
		accum_total += outputs.data[cell_index * 6 + 3].rgb;
		accum_total += outputs.data[cell_index * 6 + 4].rgb;
		accum_total += outputs.data[cell_index * 6 + 5].rgb;

		float accum_total_energy = max(dot(accum_total, GREY_VEC), 0.00001);
		vec3 iso_positive = vec3(dot(outputs.data[cell_index * 6 + 0].rgb, GREY_VEC), dot(outputs.data[cell_index * 6 + 2].rgb, GREY_VEC), dot(outputs.data[cell_index * 6 + 4].rgb, GREY_VEC)) / vec3(accum_total_energy);
		vec3 iso_negative = vec3(dot(outputs.data[cell_index * 6 + 1].rgb, GREY_VEC), dot(outputs.data[cell_index * 6 + 3].rgb, GREY_VEC), dot(outputs.data[cell_index * 6 + 5].rgb, GREY_VEC)) / vec3(accum_total_energy);

		{
			uint aniso_pos = uint(clamp(iso_positive.b * 31.0, 0.0, 31.0));
			aniso_pos |= uint(clamp(iso_positive.g * 63.0, 0.0, 63.0)) << 5;
			aniso_pos |= uint(clamp(iso_positive.r * 31.0, 0.0, 31.0)) << 11;
			imageStore(aniso_pos_tex, ivec3(posu), uvec4(aniso_pos));
		}

		{
			uint aniso_neg = uint(clamp(iso_negative.b * 31.0, 0.0, 31.0));
			aniso_neg |= uint(clamp(iso_negative.g * 63.0, 0.0, 63.0)) << 5;
			aniso_neg |= uint(clamp(iso_negative.r * 31.0, 0.0, 31.0)) << 11;
			imageStore(aniso_neg_tex, ivec3(posu), uvec4(aniso_neg));
		}

		imageStore(color_tex, ivec3(posu), vec4(accum_total / params.dynamic_range, albedo.a));

#else

		imageStore(color_tex, ivec3(posu), vec4(outputs.data[cell_index].rgb / params.dynamic_range, albedo.a));

#endif
	}
#endif

	///////////////////DYNAMIC LIGHTING/////////////////////////////

#ifdef MODE_DYNAMIC

	ivec2 pos_xy = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pos_xy, params.rect_size))) {
		return; //out of bounds
	}

	ivec2 uv_xy = pos_xy;
	if (params.flip_x) {
		uv_xy.x = params.rect_size.x - pos_xy.x - 1;
	}
	if (params.flip_y) {
		uv_xy.y = params.rect_size.y - pos_xy.y - 1;
	}

#ifdef MODE_DYNAMIC_LIGHTING

	{
		float z = params.z_base + imageLoad(depth, uv_xy).x * params.z_sign;

		ivec3 pos = params.x_dir * (params.rect_pos.x + pos_xy.x) + params.y_dir * (params.rect_pos.y + pos_xy.y) + abs(params.z_dir) * int(z);

		vec3 normal = imageLoad(source_normal, uv_xy).xyz * 2.0 - 1.0;
		normal = vec3(params.x_dir) * normal.x * mix(1.0, -1.0, params.flip_x) + vec3(params.y_dir) * normal.y * mix(1.0, -1.0, params.flip_y) - vec3(params.z_dir) * normal.z;

		vec4 albedo = imageLoad(source_albedo, uv_xy);

		//determine the position in space

		vec3 accum = vec3(0.0);
		for (uint i = 0; i < params.light_count; i++) {
			vec3 light;
			vec3 light_dir;
			if (!compute_light_at_pos(i, vec3(pos) * params.pos_multiplier, normal, light, light_dir)) {
				continue;
			}

			light *= albedo.rgb;

			accum += max(0.0, dot(normal, -light_dir)) * light;
		}

		accum += imageLoad(emission, uv_xy).xyz;

		imageStore(emission, uv_xy, vec4(accum, albedo.a));
		imageStore(depth, uv_xy, vec4(z));
	}

#endif // MODE DYNAMIC LIGHTING

#ifdef MODE_DYNAMIC_SHRINK

	{
		vec4 accum = vec4(0.0);
		float accum_z = 0.0;
		float count = 0.0;

		for (int i = 0; i < 4; i++) {
			ivec2 ofs = pos_xy * 2 + ivec2(i & 1, i >> 1) - params.prev_rect_ofs;
			if (any(lessThan(ofs, ivec2(0))) || any(greaterThanEqual(ofs, params.prev_rect_size))) {
				continue;
			}
			if (params.flip_x) {
				ofs.x = params.prev_rect_size.x - ofs.x - 1;
			}
			if (params.flip_y) {
				ofs.y = params.prev_rect_size.y - ofs.y - 1;
			}

			vec4 light = imageLoad(source_light, ofs);
			if (light.a == 0.0) { //ignore empty
				continue;
			}
			accum += light;
			float z = imageLoad(source_depth, ofs).x;
			accum_z += z * 0.5; //shrink half too
			count += 1.0;
		}

		if (params.on_mipmap) {
			accum.rgb /= mix(8.0, count, params.propagation);
			accum.a /= 8.0;
		} else {
			accum /= 4.0;
		}

		if (count == 0.0) {
			accum_z = 0.0; //avoid nan
		} else {
			accum_z /= count;
		}

#ifdef MODE_DYNAMIC_SHRINK_WRITE

		imageStore(light, uv_xy, accum);
		imageStore(depth, uv_xy, vec4(accum_z));
#endif

#ifdef MODE_DYNAMIC_SHRINK_PLOT

		if (accum.a < 0.001) {
			return; //do not blit if alpha is too low
		}

		ivec3 pos = params.x_dir * (params.rect_pos.x + pos_xy.x) + params.y_dir * (params.rect_pos.y + pos_xy.y) + abs(params.z_dir) * int(accum_z);

		float z_frac = fract(accum_z);

		for (int i = 0; i < 2; i++) {
			ivec3 pos3d = pos + abs(params.z_dir) * i;
			if (any(lessThan(pos3d, ivec3(0))) || any(greaterThanEqual(pos3d, params.limits))) {
				//skip if offlimits
				continue;
			}
			vec4 color_blit = accum * (i == 0 ? 1.0 - z_frac : z_frac);
			vec4 color = imageLoad(color_texture, pos3d);
			color.rgb *= params.dynamic_range;

#if 0
			color.rgb = mix(color.rgb,color_blit.rgb,color_blit.a);
			color.a+=color_blit.a;
#else

			float sa = 1.0 - color_blit.a;
			vec4 result;
			result.a = color.a * sa + color_blit.a;
			if (result.a == 0.0) {
				result = vec4(0.0);
			} else {
				result.rgb = (color.rgb * color.a * sa + color_blit.rgb * color_blit.a) / result.a;
				color = result;
			}

#endif
			color.rgb /= params.dynamic_range;
			imageStore(color_texture, pos3d, color);
			//imageStore(color_texture,pos3d,vec4(1,1,1,1));

#ifdef MODE_ANISOTROPIC
			//do not care about anisotropy for dynamic objects, just store full lit in all directions
			imageStore(aniso_pos_texture, pos3d, uvec4(0xFFFF));
			imageStore(aniso_neg_texture, pos3d, uvec4(0xFFFF));

#endif // ANISOTROPIC
		}
#endif // MODE_DYNAMIC_SHRINK_PLOT
	}
#endif

#endif // MODE DYNAMIC
}
