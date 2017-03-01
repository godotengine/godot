[vertex]


layout(location=0) in highp vec4 vertex_attrib;

void main() {

	gl_Position = vertex_attrib;
}

[fragment]


#ifdef MINIFY_START

#define SDEPTH_TYPE highp sampler2D
uniform float camera_z_far;
uniform float camera_z_near;

#else

#define SDEPTH_TYPE mediump usampler2D

#endif

uniform SDEPTH_TYPE source_depth; //texunit:0

uniform ivec2 from_size;
uniform int source_mipmap;

layout(location = 0) out mediump uint depth;

void main() {


	ivec2 ssP = ivec2(gl_FragCoord.xy);

	  // Rotated grid subsampling to avoid XY directional bias or Z precision bias while downsampling.
	  // On DX9, the bit-and can be implemented with floating-point modulo

#ifdef MINIFY_START
	float fdepth = texelFetch(source_depth, clamp(ssP * 2 + ivec2(ssP.y & 1, ssP.x & 1), ivec2(0), from_size - ivec2(1)), source_mipmap).r;
	fdepth = fdepth * 2.0 - 1.0;
	fdepth = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - fdepth * (camera_z_far - camera_z_near));
	fdepth /= camera_z_far;
	depth = uint(clamp(fdepth*65535.0,0.0,65535.0));

#else
	depth = texelFetch(source_depth, clamp(ssP * 2 + ivec2(ssP.y & 1, ssP.x & 1), ivec2(0), from_size - ivec2(1)), source_mipmap).r;
#endif


}


