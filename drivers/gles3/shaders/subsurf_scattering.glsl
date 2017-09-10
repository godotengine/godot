[vertex]


layout(location=0) in highp vec4 vertex_attrib;
layout(location=4) in vec2 uv_in;

out vec2 uv_interp;


void main() {

	uv_interp = uv_in;
	gl_Position = vertex_attrib;
}

[fragment]

//#define QUALIFIER uniform // some guy on the interweb says it may be faster with this
#define QUALIFIER const

#ifdef USE_25_SAMPLES

const int kernel_size=25;
QUALIFIER vec2 kernel[25] = vec2[] (
    vec2(0.530605, 0.0),
    vec2(0.000973794, -3.0),
    vec2(0.00333804, -2.52083),
    vec2(0.00500364, -2.08333),
    vec2(0.00700976,  -1.6875),
    vec2(0.0094389, -1.33333),
    vec2(0.0128496,  -1.02083),
    vec2(0.017924,  -0.75),
    vec2(0.0263642,  -0.520833),
    vec2(0.0410172, -0.333333),
    vec2(0.0493588, -0.1875),
    vec2(0.0402784, -0.0833333),
    vec2(0.0211412,  -0.0208333),
    vec2(0.0211412,  0.0208333),
    vec2(0.0402784,  0.0833333),
    vec2(0.0493588,  0.1875),
    vec2(0.0410172,  0.333333),
    vec2(0.0263642,  0.520833),
    vec2(0.017924,  0.75),
    vec2(0.0128496, 1.02083),
    vec2(0.0094389,  1.33333),
    vec2(0.00700976,  1.6875),
    vec2(0.00500364,  2.08333),
    vec2(0.00333804,  2.52083),
    vec2(0.000973794,  3.0)
);

#endif //USE_25_SAMPLES

#ifdef USE_17_SAMPLES

const int kernel_size=17;

QUALIFIER vec2 kernel[17] = vec2[](
    vec2(0.536343,  0.0),
    vec2(0.00317394,  -2.0),
    vec2(0.0100386, -1.53125),
    vec2(0.0144609, -1.125),
    vec2(0.0216301,  -0.78125),
    vec2(0.0347317, -0.5),
    vec2(0.0571056,  -0.28125),
    vec2(0.0582416,  -0.125),
    vec2(0.0324462, -0.03125),
    vec2(0.0324462, 0.03125),
    vec2(0.0582416, 0.125),
    vec2(0.0571056,  0.28125),
    vec2(0.0347317, 0.5),
    vec2(0.0216301, 0.78125),
    vec2(0.0144609,  1.125),
    vec2(0.0100386,  1.53125),
    vec2(0.00317394,2.0)
);

#endif //USE_17_SAMPLES


#ifdef USE_11_SAMPLES

const int kernel_size=11;

QUALIFIER vec4 kernel[11] = vec4[](
    vec4(0.560479,  0.0),
    vec4(0.00471691,  -2.0),
    vec4(0.0192831, -1.28),
    vec4(0.03639, -0.72),
    vec4(0.0821904,  -0.32),
    vec4(0.0771802, -0.08),
    vec4(0.0771802,  0.08),
    vec4(0.0821904, 0.32),
    vec4(0.03639, 0.72),
    vec4(0.0192831, 1.28),
    vec4(0.00471691,2.0)
);

#endif //USE_11_SAMPLES



uniform float max_radius;
uniform float camera_z_far;
uniform float camera_z_near;
uniform float unit_size;
uniform vec2 dir;
in vec2 uv_interp;

uniform sampler2D source_diffuse; //texunit:0
uniform sampler2D source_sss; //texunit:1
uniform sampler2D source_depth; //texunit:2

layout(location = 0) out vec4 frag_color;

void main() {

	float strength = texture(source_sss,uv_interp).r;
	strength*=strength; //stored as sqrt

	// Fetch color of current pixel:
	vec4 base_color = texture(source_diffuse, uv_interp);


	if (strength>0.0) {


		// Fetch linear depth of current pixel:
		float depth = texture(source_depth, uv_interp).r * 2.0 - 1.0;
#ifdef USE_ORTHOGONAL_PROJECTION
		depth = ((depth + (camera_z_far + camera_z_near)/(camera_z_far - camera_z_near)) * (camera_z_far - camera_z_near))/2.0;
		float scale = unit_size; //remember depth is negative by default in OpenGL
#else
		depth = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - depth * (camera_z_far - camera_z_near));
		float scale = unit_size / depth; //remember depth is negative by default in OpenGL
#endif



		// Calculate the final step to fetch the surrounding pixels:
		vec2 step = max_radius * scale * dir;
		step *= strength; // Modulate it using the alpha channel.
		step *= 1.0 / 3.0; // Divide by 3 as the kernels range from -3 to 3.

		// Accumulate the center sample:
		vec3 color_accum = base_color.rgb;
		color_accum *= kernel[0].x;
#ifdef ENABLE_STRENGTH_WEIGHTING
		float color_weight = kernel[0].x;
#endif

		// Accumulate the other samples:
		for (int i = 1; i < kernel_size; i++) {
			// Fetch color and depth for current sample:
			vec2 offset = uv_interp + kernel[i].y * step;
			vec3 color = texture(source_diffuse, offset).rgb;

#ifdef ENABLE_FOLLOW_SURFACE
			// If the difference in depth is huge, we lerp color back to "colorM":
			float depth_cmp = texture(source_depth, offset).r *2.0 - 1.0;

#ifdef USE_ORTHOGONAL_PROJECTION
			depth_cmp = ((depth_cmp + (camera_z_far + camera_z_near)/(camera_z_far - camera_z_near)) * (camera_z_far - camera_z_near))/2.0;
#else
			depth_cmp = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - depth_cmp * (camera_z_far - camera_z_near));
#endif

			float s = clamp(300.0f * scale *
					       max_radius * abs(depth - depth_cmp),0.0,1.0);
			color = mix(color, base_color.rgb, s);
#endif

			// Accumulate:
			color*=kernel[i].x;

#ifdef ENABLE_STRENGTH_WEIGHTING
			float color_s = texture(source_sss, offset).r;
			color_weight+=color_s * kernel[i].x;
			color*=color_s;
#endif
			color_accum += color;

		}

#ifdef ENABLE_STRENGTH_WEIGHTING
		color_accum/=color_weight;
#endif
		frag_color = vec4(color_accum,base_color.a); //keep alpha (used for SSAO)
	} else {
		frag_color = base_color;
	}
}

