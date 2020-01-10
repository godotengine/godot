/* clang-format off */
[compute]
/* clang-format on */
#version 450

VERSION_DEFINES

#define BLOCK_SIZE 8

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;

shared float tmp_data[BLOCK_SIZE*BLOCK_SIZE];


#ifdef READ_TEXTURE

//use for main texture
layout(set = 0, binding = 1) uniform texture2D source_texture;
layout(set = 0, binding = 2) uniform sampler source_sampler;

#else

//use for intermediate textures
layout(r32f, set = 0, binding = 1) uniform restrict readonly image2D source_luminance;

#endif

layout(r32f, set = 0, binding = 3) uniform restrict writeonly image2D dest_luminance;

layout(push_constant, binding = 0, std430) uniform Params {
	ivec2 source_size;
} params;

void main() {


	uint t = gl_LocalInvocationID.y * BLOCK_SIZE + gl_LocalInvocationID.x;
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(lessThan(pos,params.source_size))) {

#ifdef READ_TEXTURE
		vec3 v = texelFetch(sampler2D(source_texture,source_sampler),pos).rgb;
		avg += max(v.r,max(v.g,v.b));
		tmp_data[t] = 0.0;
#else
		tmp_data[t] = imageLoad(source_luminance, pos);
#endif
	} else {
		tmp_data[t] = 0.0;
	}

	groupMemoryBarrier();
	barrier();

	uint size = (BLOCK_SIZE * BLOCK_SIZE)>>1;

	do {
		if (t<size) {
			tmp_data[t]+=tmp_data[t+size];
		}
		groupMemoryBarrier();
		barrier();

		size>>=1;

	} while(size>1);

	if (t==0) {
		//compute rect size
		ivec2 rect_size = max(params.source_size - pos,ivec2(BLOCK_SIZE));
		float avg = tmp_data[0] / float(rect_size.x*rect_size.y);
		pos/=ivec2(BLOCK_SIZE);
		imageStore(dest_luminance, pos, vec4(avg));
	}
}
