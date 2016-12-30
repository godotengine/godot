[vertex]



layout(location=0) in highp vec4 pos_lifetime;
layout(location=1) in highp vec4 color;
layout(location=2) in highp vec4 velocity_seed;
layout(location=3) in highp vec4 rot_active;


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


out highp vec4 out_pos_lifetime; //tfb:
out highp vec4 out_color; //tfb:
out highp vec4 out_velocity_seed; //tfb:
out highp vec4 out_rot_active; //tfb:

void main() {

	bool apply_forces=true;
	bool apply_velocity=true;

	float mass = 1.0;

	float restart_phase = float(gl_InstanceID)/total_particles;
	restart_phase*= explosiveness;
	bool restart=false;

	if (system_phase > prev_system_phase) {
		restart = prev_system_phase < restart_phase && system_phase >= restart_phase;
	} else {
		restart = prev_system_phase < restart_phase || system_phase >= restart_phase;
	}

	if (restart) {
		out_rot_active.a=1.0;
	}

	out_pos_lifetime=pos_lifetime;
	out_color=color;
	out_velocity_seed=velocity_seed;
	out_rot_active=rot_active;

	if (out_rot_active.a) {
		//execute shader

	}


	if (apply_forces) {

		vec3 force = gravity;
		for(int i=0;i<attractor_count;i++) {

			vec3 rel_vec = out_pos_lifetime.xyz - attractors[i].pos;
			float dist = rel_vec.length();
			if (attractors[i].radius < dist)
				continue;
			if (attractors[i].eat_radius>0 &&  attractors[i].eat_radius > dist) {
				rot_active.a=0.0;
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

	if (apply_velocity) {

		out_pos_lifetime.xyz += out_velocity_seed.xyz * delta;
	}

}

[fragment]


void main() {


}
