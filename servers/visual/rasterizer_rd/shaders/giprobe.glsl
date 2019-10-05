[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define NO_CHILDREN 0xFFFFFFFF
#define GREY_VEC vec3(0.33333,0.33333,0.33333)

struct CellChildren {
	uint children[8];
};

layout(set=0,binding=1,std430) buffer CellChildrenBuffer {
    CellChildren data[];
} cell_children;

struct CellData {
	uint position; // xyz 10 bits
	uint albedo; //rgb albedo
	uint emission; //rgb normalized with e as multiplier
	uint normal; //RGB normal encoded
};

layout(set=0,binding=2,std430) buffer CellDataBuffer {
    CellData data[];
} cell_data;

#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_OMNI 1
#define LIGHT_TYPE_SPOT 2

#ifdef MODE_COMPUTE_LIGHT

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


layout(set=0,binding=3,std140) uniform Lights {
    Light data[MAX_LIGHTS];
} lights;



#endif // MODE COMPUTE LIGHT


#ifdef MODE_SECOND_BOUNCE

layout (set=0,binding=5) uniform texture3D color_texture;
layout (set=0,binding=6) uniform sampler texture_sampler;

#ifdef MODE_ANISOTROPIC
layout (set=0,binding=7) uniform texture3D aniso_pos_texture;
layout (set=0,binding=8) uniform texture3D aniso_neg_texture;
#endif // MODE ANISOTROPIC

#endif // MODE_SECOND_BOUNCE


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

} params;


layout(set=0,binding=4,std430) buffer Outputs {
    vec4 data[];
} outputs;

#ifdef MODE_WRITE_TEXTURE

layout (rgba8,set=0,binding=5) uniform restrict writeonly image3D color_tex;

#ifdef MODE_ANISOTROPIC

layout (r16ui,set=0,binding=6) uniform restrict writeonly uimage3D aniso_pos_tex;
layout (r16ui,set=0,binding=7) uniform restrict writeonly uimage3D aniso_neg_tex;

#endif


#endif


#ifdef MODE_COMPUTE_LIGHT

uint raymarch(float distance,float distance_adv,vec3 from,vec3 direction) {

	uint result = NO_CHILDREN;

	ivec3 size = ivec3(max(max(params.limits.x,params.limits.y),params.limits.z));

	while (distance > -distance_adv) { //use this to avoid precision errors

		uint cell = 0;

		ivec3 pos = ivec3(from);

		if (all(greaterThanEqual(pos,ivec3(0))) && all(lessThan(pos,size))) {

			ivec3 ofs = ivec3(0);
			ivec3 half_size = size / 2;

			for (int i = 0; i < params.stack_size - 1; i++) {

				bvec3 greater = greaterThanEqual(pos,ofs+half_size);

				ofs += mix(ivec3(0),half_size,greater);

				uint child = 0; //wonder if this can be done faster
				if (greater.x) {
					child|=1;
				}
				if (greater.y) {
					child|=2;
				}
				if (greater.z) {
					child|=4;
				}

				cell = cell_children.data[cell].children[child];
				if (cell == NO_CHILDREN)
					break;

				half_size >>= ivec3(1);
			}

			if ( cell != NO_CHILDREN) {
				return cell; //found cell!
			}

		}

		from += direction * distance_adv;
		distance -= distance_adv;
	}

	return NO_CHILDREN;
}

bool compute_light_vector(uint light,uint cell, vec3 pos,out float attenuation, out vec3 light_pos) {


	if (lights.data[light].type==LIGHT_TYPE_DIRECTIONAL) {

		light_pos = pos - lights.data[light].direction * length(vec3(params.limits));
		attenuation = 1.0;

	} else {

		light_pos = lights.data[light].position;
		float distance = length(pos - light_pos);
		if (distance >= lights.data[light].radius) {
			return false;
		}


		attenuation = pow( clamp( 1.0 - distance / lights.data[light].radius, 0.0001, 1.0), lights.data[light].attenuation );


		if (lights.data[light].type==LIGHT_TYPE_SPOT) {

			vec3 rel = normalize(pos - light_pos);
			float angle = acos(dot(rel,lights.data[light].direction));
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

	return 1.0 / dot(normal,unorm);
}

#endif




void main() {

	uint cell_index = gl_GlobalInvocationID.x;;
	if (cell_index >= params.cell_count) {
		return;
	}
	cell_index += params.cell_offset;

	uvec3 posu = uvec3(cell_data.data[cell_index].position&0x7FF,(cell_data.data[cell_index].position>>11)&0x3FF,cell_data.data[cell_index].position>>21);
	vec4 albedo = unpackUnorm4x8(cell_data.data[cell_index].albedo);

/////////////////COMPUTE LIGHT///////////////////////////////

#ifdef MODE_COMPUTE_LIGHT

	vec3 pos = vec3(posu) + vec3(0.5);

	vec3 emission = vec3(uvec3(cell_data.data[cell_index].emission & 0x1ff,(cell_data.data[cell_index].emission >> 9) & 0x1ff,(cell_data.data[cell_index].emission >> 18) & 0x1ff)) * pow(2.0, float(cell_data.data[cell_index].emission >> 27) - 15.0 - 9.0);
	vec4 normal = unpackSnorm4x8(cell_data.data[cell_index].normal);

#ifdef MODE_ANISOTROPIC
	vec3 accum[6]=vec3[](vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0));
	const vec3 accum_dirs[6]=vec3[](vec3(1.0,0.0,0.0),vec3(-1.0,0.0,0.0),vec3(0.0,1.0,0.0),vec3(0.0,-1.0,0.0),vec3(0.0,0.0,1.0),vec3(0.0,0.0,-1.0));
#else
	vec3 accum = vec3(0.0);
#endif

	for(uint i=0;i<params.light_count;i++) {

		float attenuation;
		vec3 light_pos;

		if (!compute_light_vector(i,cell_index,pos,attenuation,light_pos)) {
			continue;
		}

		vec3 light_dir = pos - light_pos;
		float distance = length(light_dir);
		light_dir=normalize(light_dir);

		if (attenuation < 0.01 || (length(normal.xyz) > 0.2 && dot(normal.xyz,light_dir)>=0)) {
			continue; //not facing the light, or attenuation is near zero
		}

		if (lights.data[i].has_shadow) {

			float distance_adv = get_normal_advance(light_dir);


			distance += distance_adv - mod(distance, distance_adv); //make it reach the center of the box always

			vec3 from = pos - light_dir * distance; //approximate
			from -= sign(light_dir)*0.45; //go near the edge towards the light direction to avoid self occlusion



			uint result = raymarch(distance,distance_adv,from,light_dir);

			if (result != cell_index) {
				continue; //was occluded
			}
		}

		vec3 light = lights.data[i].color * albedo.rgb * attenuation * lights.data[i].energy;

#ifdef MODE_ANISOTROPIC
		for(uint j=0;j<6;j++) {

			accum[j]+=max(0.0,dot(accum_dirs[j],-light_dir))*light;
		}
#else
		if (length(normal.xyz) > 0.2) {
			accum+=max(0.0,dot(normal.xyz,-light_dir))*light;
		} else {
			//all directions
			accum+=light+emission;
		}
#endif
	}


#ifdef MODE_ANISOTROPIC

	outputs.data[cell_index*6+0]=vec4(accum[0] + emission,0.0);
	outputs.data[cell_index*6+1]=vec4(accum[1] + emission,0.0);
	outputs.data[cell_index*6+2]=vec4(accum[2] + emission,0.0);
	outputs.data[cell_index*6+3]=vec4(accum[3] + emission,0.0);
	outputs.data[cell_index*6+4]=vec4(accum[4] + emission,0.0);
	outputs.data[cell_index*6+5]=vec4(accum[5] + emission,0.0);
#else
	outputs.data[cell_index]=vec4(accum + emission,0.0);

#endif



#endif //MODE_COMPUTE_LIGHT

/////////////////SECOND BOUNCE///////////////////////////////
#ifdef MODE_SECOND_BOUNCE
	vec3 pos = vec3(posu) + vec3(0.5);
	ivec3 ipos = ivec3(posu);
	vec4 normal = unpackSnorm4x8(cell_data.data[cell_index].normal);


#ifdef MODE_ANISOTROPIC
	vec3 accum[6];
	const vec3 accum_dirs[6]=vec3[](vec3(1.0,0.0,0.0),vec3(-1.0,0.0,0.0),vec3(0.0,1.0,0.0),vec3(0.0,-1.0,0.0),vec3(0.0,0.0,1.0),vec3(0.0,0.0,-1.0));

	/*vec3 src_color = texelFetch(sampler3D(color_texture,texture_sampler),ipos,0).rgb * params.dynamic_range;
	vec3 src_aniso_pos = texelFetch(sampler3D(aniso_pos_texture,texture_sampler),ipos,0).rgb;
	vec3 src_anisp_neg = texelFetch(sampler3D(anisp_neg_texture,texture_sampler),ipos,0).rgb;
	accum[0]=src_col * src_aniso_pos.x;
	accum[1]=src_col * src_aniso_neg.x;
	accum[2]=src_col * src_aniso_pos.y;
	accum[3]=src_col * src_aniso_neg.y;
	accum[4]=src_col * src_aniso_pos.z;
	accum[5]=src_col * src_aniso_neg.z;*/

	accum[0] = outputs.data[cell_index*6+0].rgb;
	accum[1] = outputs.data[cell_index*6+1].rgb;
	accum[2] = outputs.data[cell_index*6+2].rgb;
	accum[3] = outputs.data[cell_index*6+3].rgb;
	accum[4] = outputs.data[cell_index*6+4].rgb;
	accum[5] = outputs.data[cell_index*6+5].rgb;

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
				vec3 aniso_normal = mix(direction,normal.xyz,params.aniso_strength);
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
					vec4 scolor = textureLod(sampler3D(color_texture,texture_sampler), uvw_pos, log2_diameter);
#ifdef MODE_ANISOTROPIC

					vec3 aniso_neg = textureLod(sampler3D(aniso_neg_texture,texture_sampler), uvw_pos, log2_diameter).rgb;
					vec3 aniso_pos = textureLod(sampler3D(aniso_pos_texture,texture_sampler), uvw_pos, log2_diameter).rgb;

					scolor.rgb*=dot(max(vec3(0.0),(aniso_normal * aniso_pos)),vec3(1.0)) + dot(max(vec3(0.0),(-aniso_normal * aniso_neg)),vec3(1.0));
#endif
					float a = (1.0 - color.a);
					color += a * scolor;
					dist += half_diameter;

				}

			}
			color *= cone_weights[i] * vec4(albedo.rgb,1.0) * params.dynamic_range; //restore range
#ifdef MODE_ANISOTROPIC
			for(uint j=0;j<6;j++) {

				accum[j]+=max(0.0,dot(accum_dirs[j],direction))*color.rgb;
			}
#else
			accum+=color.rgb;
#endif
		}
	}

#ifdef MODE_ANISOTROPIC

	outputs.data[cell_index*6+0]=vec4(accum[0],0.0);
	outputs.data[cell_index*6+1]=vec4(accum[1],0.0);
	outputs.data[cell_index*6+2]=vec4(accum[2],0.0);
	outputs.data[cell_index*6+3]=vec4(accum[3],0.0);
	outputs.data[cell_index*6+4]=vec4(accum[4],0.0);
	outputs.data[cell_index*6+5]=vec4(accum[5],0.0);
#else
	outputs.data[cell_index]=vec4(accum,0.0);

#endif

#endif // MODE_SECOND_BOUNCE
/////////////////UPDATE MIPMAPS///////////////////////////////

#ifdef MODE_UPDATE_MIPMAPS

	{
#ifdef MODE_ANISOTROPIC
		vec3 light_accum[6] = vec3[](vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0));
#else
		vec3 light_accum = vec3(0.0);
#endif
		float count = 0.0;
		for(uint i=0;i<8;i++) {
			uint child_index = cell_children.data[cell_index].children[i];
			if (child_index==NO_CHILDREN) {
				continue;
			}
#ifdef MODE_ANISOTROPIC
			light_accum[0] += outputs.data[child_index*6+0].rgb;
			light_accum[1] += outputs.data[child_index*6+1].rgb;
			light_accum[2] += outputs.data[child_index*6+2].rgb;
			light_accum[3] += outputs.data[child_index*6+3].rgb;
			light_accum[4] += outputs.data[child_index*6+4].rgb;
			light_accum[5] += outputs.data[child_index*6+5].rgb;

#else
			light_accum += outputs.data[child_index].rgb;

#endif

			count+=1.0;
		}

		float divisor = mix(8.0,count,params.propagation);
#ifdef MODE_ANISOTROPIC
		outputs.data[cell_index*6+0]=vec4(light_accum[0] / divisor,0.0);
		outputs.data[cell_index*6+1]=vec4(light_accum[1] / divisor,0.0);
		outputs.data[cell_index*6+2]=vec4(light_accum[2] / divisor,0.0);
		outputs.data[cell_index*6+3]=vec4(light_accum[3] / divisor,0.0);
		outputs.data[cell_index*6+4]=vec4(light_accum[4] / divisor,0.0);
		outputs.data[cell_index*6+5]=vec4(light_accum[5] / divisor,0.0);

#else
		outputs.data[cell_index]=vec4(light_accum / divisor,0.0);
#endif



	}
#endif

///////////////////WRITE TEXTURE/////////////////////////////

#ifdef MODE_WRITE_TEXTURE
	{

#ifdef MODE_ANISOTROPIC
		vec3 accum_total = vec3(0.0);
		accum_total += outputs.data[cell_index*6+0].rgb;
		accum_total += outputs.data[cell_index*6+1].rgb;
		accum_total += outputs.data[cell_index*6+2].rgb;
		accum_total += outputs.data[cell_index*6+3].rgb;
		accum_total += outputs.data[cell_index*6+4].rgb;
		accum_total += outputs.data[cell_index*6+5].rgb;

		float accum_total_energy = max(dot(accum_total,GREY_VEC),0.00001);
		vec3 iso_positive = vec3(dot(outputs.data[cell_index*6+0].rgb,GREY_VEC),dot(outputs.data[cell_index*6+2].rgb,GREY_VEC),dot(outputs.data[cell_index*6+4].rgb,GREY_VEC))/vec3(accum_total_energy);
		vec3 iso_negative = vec3(dot(outputs.data[cell_index*6+1].rgb,GREY_VEC),dot(outputs.data[cell_index*6+3].rgb,GREY_VEC),dot(outputs.data[cell_index*6+5].rgb,GREY_VEC))/vec3(accum_total_energy);


		{
			uint aniso_pos = uint(clamp(iso_positive.b * 31.0,0.0,31.0));
			aniso_pos |= uint(clamp(iso_positive.g * 63.0,0.0,63.0))<<5;
			aniso_pos |= uint(clamp(iso_positive.r * 31.0,0.0,31.0))<<11;
			imageStore(aniso_pos_tex,ivec3(posu),uvec4(aniso_pos));
		}

		{
			uint aniso_neg = uint(clamp(iso_negative.b * 31.0,0.0,31.0));
			aniso_neg |= uint(clamp(iso_negative.g * 63.0,0.0,63.0))<<5;
			aniso_neg |= uint(clamp(iso_negative.r * 31.0,0.0,31.0))<<11;
			imageStore(aniso_neg_tex,ivec3(posu),uvec4(aniso_neg));
		}

		imageStore(color_tex,ivec3(posu),vec4(accum_total / params.dynamic_range ,albedo.a));

#else

		imageStore(color_tex,ivec3(posu),vec4(outputs.data[cell_index].rgb / params.dynamic_range,albedo.a));

#endif


	}
#endif
}
