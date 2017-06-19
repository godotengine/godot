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
QUALIFIER vec4 kernel[25] = vec4[] (
    vec4(0.530605, 0.613514, 0.739601, 0.0),
    vec4(0.000973794, 1.11862e-005, 9.43437e-007, -3.0),
    vec4(0.00333804, 7.85443e-005, 1.2945e-005, -2.52083),
    vec4(0.00500364, 0.00020094, 5.28848e-005, -2.08333),
    vec4(0.00700976, 0.00049366, 0.000151938, -1.6875),
    vec4(0.0094389, 0.00139119, 0.000416598, -1.33333),
    vec4(0.0128496, 0.00356329, 0.00132016, -1.02083),
    vec4(0.017924, 0.00711691, 0.00347194, -0.75),
    vec4(0.0263642, 0.0119715, 0.00684598, -0.520833),
    vec4(0.0410172, 0.0199899, 0.0118481, -0.333333),
    vec4(0.0493588, 0.0367726, 0.0219485, -0.1875),
    vec4(0.0402784, 0.0657244, 0.04631, -0.0833333),
    vec4(0.0211412, 0.0459286, 0.0378196, -0.0208333),
    vec4(0.0211412, 0.0459286, 0.0378196, 0.0208333),
    vec4(0.0402784, 0.0657244, 0.04631, 0.0833333),
    vec4(0.0493588, 0.0367726, 0.0219485, 0.1875),
    vec4(0.0410172, 0.0199899, 0.0118481, 0.333333),
    vec4(0.0263642, 0.0119715, 0.00684598, 0.520833),
    vec4(0.017924, 0.00711691, 0.00347194, 0.75),
    vec4(0.0128496, 0.00356329, 0.00132016, 1.02083),
    vec4(0.0094389, 0.00139119, 0.000416598, 1.33333),
    vec4(0.00700976, 0.00049366, 0.000151938, 1.6875),
    vec4(0.00500364, 0.00020094, 5.28848e-005, 2.08333),
    vec4(0.00333804, 7.85443e-005, 1.2945e-005, 2.52083),
    vec4(0.000973794, 1.11862e-005, 9.43437e-007, 3.0)
);

#endif //USE_25_SAMPLES

#ifdef USE_17_SAMPLES

const int kernel_size=17;

QUALIFIER vec4 kernel[17] = vec4[](
    vec4(0.536343, 0.624624, 0.748867, 0.0),
    vec4(0.00317394, 0.000134823, 3.77269e-005, -2.0),
    vec4(0.0100386, 0.000914679, 0.000275702, -1.53125),
    vec4(0.0144609, 0.00317269, 0.00106399, -1.125),
    vec4(0.0216301, 0.00794618, 0.00376991, -0.78125),
    vec4(0.0347317, 0.0151085, 0.00871983, -0.5),
    vec4(0.0571056, 0.0287432, 0.0172844, -0.28125),
    vec4(0.0582416, 0.0659959, 0.0411329, -0.125),
    vec4(0.0324462, 0.0656718, 0.0532821, -0.03125),
    vec4(0.0324462, 0.0656718, 0.0532821, 0.03125),
    vec4(0.0582416, 0.0659959, 0.0411329, 0.125),
    vec4(0.0571056, 0.0287432, 0.0172844, 0.28125),
    vec4(0.0347317, 0.0151085, 0.00871983, 0.5),
    vec4(0.0216301, 0.00794618, 0.00376991, 0.78125),
    vec4(0.0144609, 0.00317269, 0.00106399, 1.125),
    vec4(0.0100386, 0.000914679, 0.000275702, 1.53125),
    vec4(0.00317394, 0.000134823, 3.77269e-005, 2.0)
);

#endif //USE_17_SAMPLES


#ifdef USE_11_SAMPLES

const int kernel_size=11;

QUALIFIER vec4 kernel[11] = vec4[](
    vec4(0.560479, 0.669086, 0.784728, 0.0),
    vec4(0.00471691, 0.000184771, 5.07566e-005, -2.0),
    vec4(0.0192831, 0.00282018, 0.00084214, -1.28),
    vec4(0.03639, 0.0130999, 0.00643685, -0.72),
    vec4(0.0821904, 0.0358608, 0.0209261, -0.32),
    vec4(0.0771802, 0.113491, 0.0793803, -0.08),
    vec4(0.0771802, 0.113491, 0.0793803, 0.08),
    vec4(0.0821904, 0.0358608, 0.0209261, 0.32),
    vec4(0.03639, 0.0130999, 0.00643685, 0.72),
    vec4(0.0192831, 0.00282018, 0.00084214, 1.28),
    vec4(0.00471691, 0.000184771, 5.07565e-005, 2.0)
);

#endif //USE_11_SAMPLES


uniform float max_radius;
uniform float fovy;
uniform float camera_z_far;
uniform float camera_z_near;
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
		depth = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - depth * (camera_z_far - camera_z_near));
		depth=-depth;


		// Calculate the radius scale (1.0 for a unit plane sitting on the
		// projection window):
		float distance = 1.0 / tan(0.5 * fovy);
		float scale = distance / -depth; //remember depth is negative by default in OpenGL

		// Calculate the final step to fetch the surrounding pixels:
		vec2 step = max_radius * scale * dir;
		step *= strength; // Modulate it using the alpha channel.
		step *= 1.0 / 3.0; // Divide by 3 as the kernels range from -3 to 3.

		// Accumulate the center sample:
		vec3 color_accum = base_color.rgb;
		color_accum *= kernel[0].rgb;

		// Accumulate the other samples:
		for (int i = 1; i < kernel_size; i++) {
			// Fetch color and depth for current sample:
			vec2 offset = uv_interp + kernel[i].a * step;
			vec3 color = texture(source_diffuse, offset).rgb;

#ifdef ENABLE_FOLLOW_SURFACE
			// If the difference in depth is huge, we lerp color back to "colorM":
			float depth_cmp = texture(source_depth, offset).r *2.0 - 1.0;
			depth_cmp = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - depth_cmp * (camera_z_far - camera_z_near));
			depth_cmp=-depth_cmp;

			float s = clamp(300.0f * distance *
					       max_radius * abs(depth - depth_cmp),0.0,1.0);
			color = mix(color, base_color.rgb, s);
#endif

			// Accumulate:
			color_accum += kernel[i].rgb * color;
		}

		frag_color = vec4(color_accum,base_color.a); //keep alpha (used for SSAO)
	} else {
		frag_color = base_color;
	}
}

