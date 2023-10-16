/* clang-format off */
#[modes]

mode_default =

#[specializations]

MODE_3D = false

#[vertex]

#include "stdlib_inc.glsl"

// ParticleData
layout(location = 0) in highp vec4 color;
layout(location = 1) in highp vec4 velocity_flags;
layout(location = 2) in highp vec4 custom;
layout(location = 3) in highp vec4 xform_1;
layout(location = 4) in highp vec4 xform_2;
#ifdef MODE_3D
layout(location = 5) in highp vec4 xform_3;
#endif

/* clang-format on */
out highp vec4 out_xform_1; //tfb:
out highp vec4 out_xform_2; //tfb:
#ifdef MODE_3D
out highp vec4 out_xform_3; //tfb:MODE_3D
#endif
flat out highp uvec4 instance_color_custom_data; //tfb:

uniform lowp vec3 sort_direction;
uniform highp float frame_remainder;

uniform highp vec3 align_up;
uniform highp uint align_mode;

uniform highp mat4 inv_emission_transform;

#define TRANSFORM_ALIGN_DISABLED uint(0)
#define TRANSFORM_ALIGN_Z_BILLBOARD uint(1)
#define TRANSFORM_ALIGN_Y_TO_VELOCITY uint(2)
#define TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY uint(3)

#define PARTICLE_FLAG_ACTIVE uint(1)

#define FLT_MAX float(3.402823466e+38)

void main() {
	// Set scale to zero and translate to -INF so particle will be invisible
	// even for materials that ignore rotation/scale (i.e. billboards).
	mat4 txform = mat4(vec4(0.0), vec4(0.0), vec4(0.0), vec4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0));
	if (bool(floatBitsToUint(velocity_flags.w) & PARTICLE_FLAG_ACTIVE)) {
#ifdef MODE_3D
		txform = transpose(mat4(xform_1, xform_2, xform_3, vec4(0.0, 0.0, 0.0, 1.0)));
#else
		txform = transpose(mat4(xform_1, xform_2, vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)));
#endif

		switch (align_mode) {
			case TRANSFORM_ALIGN_DISABLED: {
			} break; //nothing
			case TRANSFORM_ALIGN_Z_BILLBOARD: {
				mat3 local = mat3(normalize(cross(align_up, sort_direction)), align_up, sort_direction);
				local = local * mat3(txform);
				txform[0].xyz = local[0];
				txform[1].xyz = local[1];
				txform[2].xyz = local[2];

			} break;
			case TRANSFORM_ALIGN_Y_TO_VELOCITY: {
				vec3 v = velocity_flags.xyz;
				float s = (length(txform[0]) + length(txform[1]) + length(txform[2])) / 3.0;
				if (length(v) > 0.0) {
					txform[1].xyz = normalize(v);
				} else {
					txform[1].xyz = normalize(txform[1].xyz);
				}

				txform[0].xyz = normalize(cross(txform[1].xyz, txform[2].xyz));
				txform[2].xyz = vec3(0.0, 0.0, 1.0) * s;
				txform[0].xyz *= s;
				txform[1].xyz *= s;
			} break;
			case TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY: {
				vec3 sv = velocity_flags.xyz - sort_direction * dot(sort_direction, velocity_flags.xyz); //screen velocity

				if (length(sv) == 0.0) {
					sv = align_up;
				}

				sv = normalize(sv);

				txform[0].xyz = normalize(cross(sv, sort_direction)) * length(txform[0]);
				txform[1].xyz = sv * length(txform[1]);
				txform[2].xyz = sort_direction * length(txform[2]);

			} break;
		}

		txform[3].xyz += velocity_flags.xyz * frame_remainder;

#ifndef MODE_3D
		// In global mode, bring 2D particles to local coordinates
		// as they will be drawn with the node position as origin.
		txform = inv_emission_transform * txform;
#endif
	}
	txform = transpose(txform);

	instance_color_custom_data = uvec4(packHalf2x16(color.xy), packHalf2x16(color.zw), packHalf2x16(custom.xy), packHalf2x16(custom.zw));
	out_xform_1 = txform[0];
	out_xform_2 = txform[1];
#ifdef MODE_3D
	out_xform_3 = txform[2];
#endif
}

/* clang-format off */
#[fragment]

void main() {
}
/* clang-format on */
