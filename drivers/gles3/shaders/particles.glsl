[vertex]



layout(location=0) in highp vec4 color;
layout(location=1) in highp vec4 velocity_active;
layout(location=2) in highp vec4 custom;
layout(location=3) in highp vec4 xform_1;
layout(location=4) in highp vec4 xform_2;
layout(location=5) in highp vec4 xform_3;


struct Attractor {

	vec3 pos;
	vec3 dir;
	float radius;
	float eat_radius;
	float strength;
	float attenuation;
};

#define MAX_ATTRACTORS 64

uniform mat4 origin;
uniform float system_phase;
uniform float prev_system_phase;
uniform float total_particles;
uniform float explosiveness;
uniform vec4 time;
uniform float delta;
uniform vec3 gravity;
uniform int attractor_count;
uniform Attractor attractors[MAX_ATTRACTORS];


out highp vec4 out_color; //tfb:
out highp vec4 out_velocity_active; //tfb:
out highp vec4 out_custom; //tfb:
out highp vec4 out_xform_1; //tfb:
out highp vec4 out_xform_2; //tfb:
out highp vec4 out_xform_3; //tfb:

VERTEX_SHADER_GLOBALS

#if defined(USE_MATERIAL)

layout(std140) uniform UniformData { //ubo:0

MATERIAL_UNIFORMS

};

#endif

void main() {

	bool apply_forces=true;
	bool apply_velocity=true;

	float mass = 1.0;

	float restart_phase = float(gl_InstanceID)/total_particles;
	restart_phase*= explosiveness;
	bool restart=false;
	bool active = out_velocity_active.a > 0.5;

	if (system_phase > prev_system_phase) {
		restart = prev_system_phase < restart_phase && system_phase >= restart_phase;
	} else {
		restart = prev_system_phase < restart_phase || system_phase >= restart_phase;
	}

	if (restart) {
		active=true;
	}

	out_color=color;
	out_velocity_active=velocity_active;
	out_custom=custom;

	mat4 xform = transpose(mat4(xform_1,xform_2,xform_3,vec4(vec3(0.0),1.0)));


	out_rot_active=rot_active;

	if (active) {
		//execute shader

		{
			VERTEX_SHADER_CODE
		}

#if !defined(DISABLE_FORCE)

		{

			vec3 force = gravity;
			for(int i=0;i<attractor_count;i++) {

				vec3 rel_vec = out_pos_lifetime.xyz - attractors[i].pos;
				float dist = rel_vec.length();
				if (attractors[i].radius < dist)
					continue;
				if (attractors[i].eat_radius>0 &&  attractors[i].eat_radius > dist) {
					out_velocity_active.a=0.0;
				}

				rel_vec = normalize(rel_vec);

				float attenuation = pow(dist / attractors[i].radius,attractors[i].attenuation);

				if (attractors[i].dir==vec3(0.0)) {
					//towards center
					force+=attractors[i].strength * rel_vec * attenuation * mass;
				} else {
					force+=attractors[i].strength * attractors[i].dir * attenuation *mass;

				}
			}

			out_velocity_seed.xyz += force * delta;
		}
#endif

#if !defined(DISABLE_VELOCITY)

		{

			out_pos_lifetime.xyz += out_velocity_seed.xyz * delta;
		}
#endif
	}

	xform = transpose(xform);

	out_velocity_active.a = mix(0.0,1.0,active);

	out_xform_1 = xform[0];
	out_xform_2 = xform[1];
	out_xform_3 = xform[2];


}

[fragment]

//any code here is never executed, stuff is filled just so it works

FRAGMENT_SHADER_GLOBALS

#if defined(USE_MATERIAL)

layout(std140) uniform UniformData {

MATERIAL_UNIFORMS

};

#endif

void main() {

	{
		FRAGMENT_SHADER_CODE
	}
}
