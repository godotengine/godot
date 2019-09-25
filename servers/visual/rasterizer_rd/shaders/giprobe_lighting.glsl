[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define NO_CHILDREN 0xFFFFFFFF
#define GREY_VEC vec3(0.33333,0.33333,0.33333)

struct CellPosition {
	uint children[8];
};


layout(set=0,binding=1,std140) buffer CellPositions {
    CellPosition data[];
} cell_positions;

struct CellMaterial {
	uint position; // xyz 10 bits
	uint albedo; //rgb albedo
	uint emission; //rgb normalized with e as multiplier
	uint normal; //RGB normal encoded
};

layout(set=0,binding=2,std140) buffer CellMaterials {
    CellMaterial data[];
} cell_materials;

#define LIGHT_TYPE_DIRECTIONAL 0
#define LIGHT_TYPE_OMNI 1
#define LIGHT_TYPE_SPOT 2

struct Light {

	uint type;
	float energy;
	float radius;
	float attenuation;

	vec3 color;
	float spot_angle_radians;

	float advance;
	float max_length;
	uint pad0;
	uint pad2;

	vec3 position;
	float spot_attenuation;


	vec3 direction;
	bool visible;

	vec4 clip_planes[3];
};

layout(set=0,binding=3,std140) buffer Lights {
    Light data[];
} lights;


layout(set=0,binding=4,std140) uniform Params {
	vec3 limits;
	float max_length;
	uint size;
	uint stack_size;
	uint light_count;
	float emission_scale;
} params;


layout (rgba8,set=0,binding=5) uniform restrict writeonly image3D color_tex;


uint raymarch(float distance,float distance_adv,vec3 from,vec3 direction) {

	uint result = NO_CHILDREN;

	while (distance > -distance_adv) { //use this to avoid precision errors

		uint cell = 0;

		ivec3 pos = ivec3(from);
		ivec3 ofs = ivec3(0);
		ivec3 half_size = ivec3(params.size) / 2;
		if (any(lessThan(pos,ivec3(0))) || any(greaterThanEqual(pos,ivec3(params.size)))) {
			return NO_CHILDREN; //outside range
		}

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

			cell = cell_positions.data[cell].children[child];
			if (cell == NO_CHILDREN)
				break;

			half_size >>= ivec3(1);
		}

		if ( cell != NO_CHILDREN) {
			return cell; //found cell!
		}

		from += direction * distance_adv;
		distance -= distance_adv;
	}

	return NO_CHILDREN;
}

bool compute_light_vector(uint light,uint cell, vec3 pos,out float attenuation, out vec3 light_pos) {

	if (lights.data[light].type==LIGHT_TYPE_DIRECTIONAL) {

		light_pos = pos - lights.data[light].direction * params.max_length;
		attenuation = 1.0;

	} else {

		light_pos = lights.data[light].position;
		float distance = length(pos - light_pos);
		if (distance >= lights.data[light].radius) {
			return false;
		}

		attenuation = pow( distance / lights.data[light].radius + 0.0001, lights.data[light].attenuation );


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

void main() {

	uint cell_index = gl_GlobalInvocationID.x;

	uvec3 posu = uvec3(cell_materials.data[cell_index].position&0x3FF,(cell_materials.data[cell_index].position>>10)&0x3FF,cell_materials.data[cell_index].position>>20);
	vec3 pos = vec3(posu);

	vec3 emission = vec3(ivec3(cell_materials.data[cell_index].emission&0x3FF,(cell_materials.data[cell_index].emission>>10)&0x7FF,cell_materials.data[cell_index].emission>>21)) * params.emission_scale;
	vec4 albedo = unpackUnorm4x8(cell_materials.data[cell_index].albedo);
	vec4 normal = unpackSnorm4x8(cell_materials.data[cell_index].normal); //w >0.5 means, all directions

#ifdef MODE_ANISOTROPIC
	vec3 accum[6]=vec3[](vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0));
	const vec3 accum_dirs[6]=vec3[](vec3(1.0,0.0,0.0),vec3(-1.0,0.0,0.0),vec3(0.0,1.0,0.0),vec3(0.0,-1.0,0.0),vec3(0.0,0.0,1.0),vec3(0.0,0.0,-1.0));
#else
	vec3 accum = vec3(0);
#endif

	for(uint i=0;i<params.light_count;i++) {

		float attenuation;
		vec3 light_pos;

		if (!compute_light_vector(i,cell_index,pos,attenuation,light_pos)) {
			continue;
		}

		float distance_adv = lights.data[i].advance;

		vec3 light_dir = pos - light_pos;
		float distance = length(light_dir);

		light_dir=normalize(light_dir);

		distance += distance_adv - mod(distance, distance_adv); //make it reach the center of the box always

		vec3 from = pos - light_dir * distance; //approximate

		if (normal.w < 0.5 && dot(normal.xyz,light_dir)>=0) {
			continue; //not facing the light
		}

		uint result = raymarch(distance,distance_adv,from,lights.data[i].direction);

		if (result != cell_index) {
			continue; //was occluded
		}

		vec3 light = lights.data[i].color * albedo.rgb * attenuation;

#ifdef MODE_ANISOTROPIC
		for(uint j=0;j<6;j++) {
			accum[j]+=max(0.0,dot(accum_dir,-light_dir))*light+emission;
		}
#else
		if (normal.w < 0.5) {
			accum+=max(0.0,dot(normal.xyz,-light_dir))*light+emission;
		} else {
			//all directions
			accum+=light+emission;
		}
#endif

	}

#ifdef MODE_ANISOTROPIC

	vec3 accum_total = accum[0]+accum[1]+accum[2]+accum[3]+accum[4]+accum[5];
	float accum_total_energy = max(dot(accum_total,GREY_VEC),0.00001);
	vec3 iso_positive = vec3(dot(aniso[0],GREY_VEC),dot(aniso[2],GREY_VEC),dot(aniso[4],GREY_VEC))/vec3(accum_total_energy);
	vec3 iso_negative = vec3(dot(aniso[1],GREY_VEC),dot(aniso[3],GREY_VEC),dot(aniso[5],GREY_VEC))/vec3(accum_total_energy);

	//store in 3D textures, total color, and isotropic magnitudes
#else
	//store in 3D texture pos, accum
	imageStore(color_tex,ivec3(posu),vec4(accum,albedo.a));
#endif

}
