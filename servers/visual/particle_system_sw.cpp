/*************************************************************************/
/*  particle_system_sw.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "particle_system_sw.h"
#include "sort.h"

ParticleSystemSW::ParticleSystemSW() {

	amount = 8;
	emitting = true;

	for (int i = 0; i < VS::PARTICLE_VAR_MAX; i++) {
		particle_randomness[i] = 0.0;
	}

	particle_vars[VS::PARTICLE_LIFETIME] = 2.0; //
	particle_vars[VS::PARTICLE_SPREAD] = 0.2; //
	particle_vars[VS::PARTICLE_GRAVITY] = 9.8; //
	particle_vars[VS::PARTICLE_LINEAR_VELOCITY] = 0.2; //
	particle_vars[VS::PARTICLE_ANGULAR_VELOCITY] = 0.0; //
	particle_vars[VS::PARTICLE_LINEAR_ACCELERATION] = 0.0; //
	particle_vars[VS::PARTICLE_RADIAL_ACCELERATION] = 0.0; //
	particle_vars[VS::PARTICLE_TANGENTIAL_ACCELERATION] = 1.0; //
	particle_vars[VS::PARTICLE_DAMPING] = 0.0; //
	particle_vars[VS::PARTICLE_INITIAL_SIZE] = 1.0;
	particle_vars[VS::PARTICLE_FINAL_SIZE] = 0.8;
	particle_vars[VS::PARTICLE_HEIGHT] = 1;
	particle_vars[VS::PARTICLE_HEIGHT_SPEED_SCALE] = 1;

	height_from_velocity = false;
	local_coordinates = false;

	particle_vars[VS::PARTICLE_INITIAL_ANGLE] = 0.0; //

	gravity_normal = Vector3(0, -1.0, 0);
	//emission_half_extents=Vector3(0.1,0.1,0.1);
	emission_half_extents = Vector3(1, 1, 1);
	color_phase_count = 0;
	color_phases[0].pos = 0.0;
	color_phases[0].color = Color(1.0, 0.0, 0.0);
	visibility_aabb = AABB(Vector3(-64, -64, -64), Vector3(128, 128, 128));

	attractor_count = 0;
}

ParticleSystemSW::~ParticleSystemSW() {
}

#define DEFAULT_SEED 1234567

_FORCE_INLINE_ static float _rand_from_seed(uint32_t *seed) {

	uint32_t k;
	uint32_t s = (*seed);
	if (s == 0)
		s = 0x12345987;
	k = s / 127773;
	s = 16807 * (s - k * 127773) - 2836 * k;
	if (s < 0)
		s += 2147483647;
	(*seed) = s;

	float v = ((float)((*seed) & 0xFFFFF)) / (float)0xFFFFF;
	v = v * 2.0 - 1.0;
	return v;
}

_FORCE_INLINE_ static uint32_t _irand_from_seed(uint32_t *seed) {

	uint32_t k;
	uint32_t s = (*seed);
	if (s == 0)
		s = 0x12345987;
	k = s / 127773;
	s = 16807 * (s - k * 127773) - 2836 * k;
	if (s < 0)
		s += 2147483647;
	(*seed) = s;

	return s;
}

void ParticleSystemProcessSW::process(const ParticleSystemSW *p_system, const Transform &p_transform, float p_time) {

	valid = false;
	if (p_system->amount <= 0) {
		ERR_EXPLAIN("Invalid amount of particles: " + itos(p_system->amount));
		ERR_FAIL_COND(p_system->amount <= 0);
	}
	if (p_system->attractor_count < 0 || p_system->attractor_count > VS::MAX_PARTICLE_ATTRACTORS) {
		ERR_EXPLAIN("Invalid amount of particle attractors.");
		ERR_FAIL_COND(p_system->attractor_count < 0 || p_system->attractor_count > VS::MAX_PARTICLE_ATTRACTORS);
	}
	float lifetime = p_system->particle_vars[VS::PARTICLE_LIFETIME];
	if (lifetime < CMP_EPSILON) {
		ERR_EXPLAIN("Particle system lifetime too small.");
		ERR_FAIL_COND(lifetime < CMP_EPSILON);
	}
	valid = true;
	int particle_count = MIN(p_system->amount, ParticleSystemSW::MAX_PARTICLES);
	;

	int emission_point_count = p_system->emission_points.size();
	DVector<Vector3>::Read r;
	if (emission_point_count)
		r = p_system->emission_points.read();

	if (particle_count != particle_data.size()) {

		//clear the whole system if particle amount changed
		particle_data.clear();
		particle_data.resize(p_system->amount);
		particle_system_time = 0;
	}

	float next_time = particle_system_time + p_time;

	if (next_time > lifetime)
		next_time = Math::fmod(next_time, lifetime);

	ParticleData *pdata = &particle_data[0];
	Vector3 attractor_positions[VS::MAX_PARTICLE_ATTRACTORS];

	for (int i = 0; i < p_system->attractor_count; i++) {

		attractor_positions[i] = p_transform.xform(p_system->attractors[i].pos);
	}

	for (int i = 0; i < particle_count; i++) {

		ParticleData &p = pdata[i];

		float restart_time = (i * lifetime / p_system->amount);

		bool restart = false;

		if (next_time < particle_system_time) {

			if (restart_time > particle_system_time || restart_time < next_time)
				restart = true;

		} else if (restart_time > particle_system_time && restart_time < next_time) {
			restart = true;
		}

		if (restart) {

			if (p_system->emitting) {
				if (emission_point_count == 0) { //use AABB
					if (p_system->local_coordinates)
						p.pos = p_system->emission_half_extents * Vector3(_rand_from_seed(&rand_seed), _rand_from_seed(&rand_seed), _rand_from_seed(&rand_seed));
					else
						p.pos = p_transform.xform(p_system->emission_half_extents * Vector3(_rand_from_seed(&rand_seed), _rand_from_seed(&rand_seed), _rand_from_seed(&rand_seed)));
				} else {
					//use preset positions
					if (p_system->local_coordinates)
						p.pos = r[_irand_from_seed(&rand_seed) % emission_point_count];
					else
						p.pos = p_transform.xform(r[_irand_from_seed(&rand_seed) % emission_point_count]);
				}

				float angle1 = _rand_from_seed(&rand_seed) * p_system->particle_vars[VS::PARTICLE_SPREAD] * Math_PI;
				float angle2 = _rand_from_seed(&rand_seed) * 20.0 * Math_PI; // make it more random like

				Vector3 rot_xz = Vector3(Math::sin(angle1), 0.0, Math::cos(angle1));
				Vector3 rot = Vector3(Math::cos(angle2) * rot_xz.x, Math::sin(angle2) * rot_xz.x, rot_xz.z);

				p.vel = (rot * p_system->particle_vars[VS::PARTICLE_LINEAR_VELOCITY] + rot * p_system->particle_randomness[VS::PARTICLE_LINEAR_VELOCITY] * _rand_from_seed(&rand_seed));
				if (!p_system->local_coordinates)
					p.vel = p_transform.basis.xform(p.vel);

				p.vel += p_system->emission_base_velocity;

				p.rot = p_system->particle_vars[VS::PARTICLE_INITIAL_ANGLE] + p_system->particle_randomness[VS::PARTICLE_INITIAL_ANGLE] * _rand_from_seed(&rand_seed);
				p.active = true;
				for (int r = 0; r < PARTICLE_RANDOM_NUMBERS; r++)
					p.random[r] = _rand_from_seed(&rand_seed);

			} else {

				p.pos = Vector3();
				p.rot = 0;
				p.vel = Vector3();
				p.active = false;
			}

		} else {

			if (!p.active)
				continue;

			Vector3 force;
			//apply gravity
			force = p_system->gravity_normal * (p_system->particle_vars[VS::PARTICLE_GRAVITY] + (p_system->particle_randomness[VS::PARTICLE_GRAVITY] * p.random[0]));
			//apply linear acceleration
			force += p.vel.normalized() * (p_system->particle_vars[VS::PARTICLE_LINEAR_ACCELERATION] + p_system->particle_randomness[VS::PARTICLE_LINEAR_ACCELERATION] * p.random[1]);
			//apply radial acceleration
			Vector3 org;
			if (!p_system->local_coordinates)
				org = p_transform.origin;
			force += (p.pos - org).normalized() * (p_system->particle_vars[VS::PARTICLE_RADIAL_ACCELERATION] + p_system->particle_randomness[VS::PARTICLE_RADIAL_ACCELERATION] * p.random[2]);
			//apply tangential acceleration
			force += (p.pos - org).cross(p_system->gravity_normal).normalized() * (p_system->particle_vars[VS::PARTICLE_TANGENTIAL_ACCELERATION] + p_system->particle_randomness[VS::PARTICLE_TANGENTIAL_ACCELERATION] * p.random[3]);
			//apply attractor forces
			for (int a = 0; a < p_system->attractor_count; a++) {

				force += (p.pos - attractor_positions[a]).normalized() * p_system->attractors[a].force;
			}

			p.vel += force * p_time;
			if (p_system->particle_vars[VS::PARTICLE_DAMPING]) {

				float v = p.vel.length();
				float damp = p_system->particle_vars[VS::PARTICLE_DAMPING] + p_system->particle_vars[VS::PARTICLE_DAMPING] * p_system->particle_randomness[VS::PARTICLE_DAMPING];
				v -= damp * p_time;
				if (v < 0) {
					p.vel = Vector3();
				} else {
					p.vel = p.vel.normalized() * v;
				}
			}
			p.rot += (p_system->particle_vars[VS::PARTICLE_ANGULAR_VELOCITY] + p_system->particle_randomness[VS::PARTICLE_ANGULAR_VELOCITY] * p.random[4]) * p_time;
			p.pos += p.vel * p_time;
		}
	}

	particle_system_time = Math::fmod(particle_system_time + p_time, lifetime);
}

ParticleSystemProcessSW::ParticleSystemProcessSW() {

	particle_system_time = 0;
	rand_seed = 1234567;
	valid = false;
}

struct _ParticleSorterSW {

	_FORCE_INLINE_ bool operator()(const ParticleSystemDrawInfoSW::ParticleDrawInfo *p_a, const ParticleSystemDrawInfoSW::ParticleDrawInfo *p_b) const {

		return p_a->d > p_b->d; // draw from further away to closest
	}
};

void ParticleSystemDrawInfoSW::prepare(const ParticleSystemSW *p_system, const ParticleSystemProcessSW *p_process, const Transform &p_system_transform, const Transform &p_camera_transform) {

	ERR_FAIL_COND(p_process->particle_data.size() != p_system->amount);
	ERR_FAIL_COND(p_system->amount <= 0 || p_system->amount >= ParticleSystemSW::MAX_PARTICLES);

	const ParticleSystemProcessSW::ParticleData *pdata = &p_process->particle_data[0];
	float time_pos = p_process->particle_system_time / p_system->particle_vars[VS::PARTICLE_LIFETIME];

	ParticleSystemSW::ColorPhase cphase[VS::MAX_PARTICLE_COLOR_PHASES];

	float last = -1;
	int col_count = 0;

	for (int i = 0; i < p_system->color_phase_count; i++) {

		if (p_system->color_phases[i].pos <= last)
			break;
		cphase[i] = p_system->color_phases[i];
		col_count++;
	}

	Vector3 camera_z_axis = p_camera_transform.basis.get_axis(2);

	for (int i = 0; i < p_system->amount; i++) {

		ParticleDrawInfo &pdi = draw_info[i];
		pdi.data = &pdata[i];
		pdi.transform.origin = pdi.data->pos;
		if (p_system->local_coordinates)
			pdi.transform.origin = p_system_transform.xform(pdi.transform.origin);

		pdi.d = -camera_z_axis.dot(pdi.transform.origin);

		// adjust particle size, color and rotation

		float time = ((float)i / p_system->amount);
		if (time < time_pos)
			time = time_pos - time;
		else
			time = (1.0 - time) + time_pos;

		Vector3 up = p_camera_transform.basis.get_axis(1); // up determines the rotation
		float up_scale = 1.0;

		if (p_system->height_from_velocity) {

			Vector3 veld = pdi.data->vel;
			Vector3 cam_z = camera_z_axis.normalized();
			float vc = Math::abs(veld.normalized().dot(cam_z));

			if (vc < (1.0 - CMP_EPSILON)) {
				up = Plane(cam_z, 0).project(veld).normalized();
				float h = p_system->particle_vars[VS::PARTICLE_HEIGHT] + p_system->particle_randomness[VS::PARTICLE_HEIGHT] * pdi.data->random[7];
				float velh = veld.length();
				h += velh * (p_system->particle_vars[VS::PARTICLE_HEIGHT_SPEED_SCALE] + p_system->particle_randomness[VS::PARTICLE_HEIGHT_SPEED_SCALE] * pdi.data->random[7]);

				up_scale = Math::lerp(1.0, h, (1.0 - vc));
			}

		} else if (pdi.data->rot) {

			up.rotate(camera_z_axis, pdi.data->rot);
		}

		{
			// matrix
			Vector3 v_z = (p_camera_transform.origin - pdi.transform.origin).normalized();
			//			Vector3 v_z = (p_camera_transform.origin-pdi.data->pos).normalized();
			Vector3 v_y = up;
			Vector3 v_x = v_y.cross(v_z);
			v_y = v_z.cross(v_x);
			v_x.normalize();
			v_y.normalize();

			float initial_scale, final_scale;
			initial_scale = p_system->particle_vars[VS::PARTICLE_INITIAL_SIZE] + p_system->particle_randomness[VS::PARTICLE_INITIAL_SIZE] * pdi.data->random[5];
			final_scale = p_system->particle_vars[VS::PARTICLE_FINAL_SIZE] + p_system->particle_randomness[VS::PARTICLE_FINAL_SIZE] * pdi.data->random[6];
			float scale = initial_scale + time * (final_scale - initial_scale);

			pdi.transform.basis.set_axis(0, v_x * scale);
			pdi.transform.basis.set_axis(1, v_y * scale * up_scale);
			pdi.transform.basis.set_axis(2, v_z * scale);
		}

		int cpos = 0;

		while (cpos < col_count) {

			if (cphase[cpos].pos > time)
				break;
			cpos++;
		}

		cpos--;

		if (cpos == -1)
			pdi.color = Color(1, 1, 1, 1);
		else {
			if (cpos == col_count - 1)
				pdi.color = cphase[cpos].color;
			else {
				float diff = (cphase[cpos + 1].pos - cphase[cpos].pos);
				if (diff > 0)
					pdi.color = cphase[cpos].color.linear_interpolate(cphase[cpos + 1].color, (time - cphase[cpos].pos) / diff);
				else
					pdi.color = cphase[cpos + 1].color;
			}
		}

		draw_info_order[i] = &pdi;
	}

	SortArray<ParticleDrawInfo *, _ParticleSorterSW> particle_sort;
	particle_sort.sort(&draw_info_order[0], p_system->amount);
}
