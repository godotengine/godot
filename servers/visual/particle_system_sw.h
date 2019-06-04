/*************************************************************************/
/*  particle_system_sw.h                                                 */
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
#ifndef PARTICLE_SYSTEM_SW_H
#define PARTICLE_SYSTEM_SW_H

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#include "servers/visual_server.h"

struct ParticleSystemSW {
	enum {

		MAX_PARTICLES = 1024
	};

	float particle_vars[VS::PARTICLE_VAR_MAX];
	float particle_randomness[VS::PARTICLE_VAR_MAX];

	Vector3 emission_half_extents;
	DVector<Vector3> emission_points;
	Vector3 gravity_normal;
	Vector3 emission_base_velocity;
	int amount;
	bool emitting;
	bool height_from_velocity;
	AABB visibility_aabb;
	bool sort;
	bool local_coordinates;

	struct ColorPhase {

		float pos;
		Color color;
		ColorPhase() {
			pos = 1.0;
			color = Color(0.0, 0.0, 1.0, 1.0);
		}
	};

	int color_phase_count;
	ColorPhase color_phases[VS::MAX_PARTICLE_COLOR_PHASES];

	struct Attractor {

		Vector3 pos;
		float force;
	};

	int attractor_count;
	Attractor attractors[VS::MAX_PARTICLE_ATTRACTORS];

	ParticleSystemSW();
	~ParticleSystemSW();
};

struct ParticleSystemProcessSW {

	enum {
		PARTICLE_RANDOM_NUMBERS = 8,
	};

	struct ParticleData {

		Vector3 pos;
		Vector3 vel;
		float rot;
		bool active;
		float random[PARTICLE_RANDOM_NUMBERS];

		ParticleData() {
			active = 0;
			rot = 0;
		}
	};

	bool valid;
	float particle_system_time;
	uint32_t rand_seed;
	Vector<ParticleData> particle_data;

	void process(const ParticleSystemSW *p_system, const Transform &p_transform, float p_time);

	ParticleSystemProcessSW();
};

struct ParticleSystemDrawInfoSW {

	struct ParticleDrawInfo {

		const ParticleSystemProcessSW::ParticleData *data;
		float d;
		Transform transform;
		Color color;
	};

	ParticleDrawInfo draw_info[ParticleSystemSW::MAX_PARTICLES];
	ParticleDrawInfo *draw_info_order[ParticleSystemSW::MAX_PARTICLES];

	void prepare(const ParticleSystemSW *p_system, const ParticleSystemProcessSW *p_process, const Transform &p_system_transform, const Transform &p_camera_transform);
};

#endif
