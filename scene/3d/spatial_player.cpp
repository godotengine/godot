/*************************************************************************/
/*  spatial_player.cpp                                                   */
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
#include "spatial_player.h"

#include "camera.h"
#include "scene/resources/surface_tool.h"
#include "servers/audio_server.h"
#include "servers/spatial_sound_server.h"

void SpatialPlayer::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_WORLD: {
			//find the sound space

			source_rid = SpatialSoundServer::get_singleton()->source_create(get_world()->get_sound_space());
			for (int i = 0; i < PARAM_MAX; i++)
				set_param(Param(i), params[i]);

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			SpatialSoundServer::get_singleton()->source_set_transform(source_rid, get_global_transform());

		} break;
		case NOTIFICATION_EXIT_WORLD: {

			if (source_rid.is_valid())
				SpatialSoundServer::get_singleton()->free(source_rid);

		} break;
	}
}

void SpatialPlayer::set_param(Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params[p_param] = p_value;
	if (p_param == PARAM_EMISSION_CONE_DEGREES) {
		update_gizmo();
	}
	if (source_rid.is_valid())
		SpatialSoundServer::get_singleton()->source_set_param(source_rid, (SpatialSoundServer::SourceParam)p_param, p_value);
}

float SpatialPlayer::get_param(Param p_param) const {

	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params[p_param];
}

bool SpatialPlayer::_can_gizmo_scale() const {

	return false;
}

RES SpatialPlayer::_get_gizmo_geometry() const {

	Ref<SurfaceTool> surface_tool(memnew(SurfaceTool));

	Ref<FixedMaterial> mat(memnew(FixedMaterial));

	mat->set_parameter(FixedMaterial::PARAM_DIFFUSE, Color(0.0, 0.6, 0.7, 0.05));
	mat->set_parameter(FixedMaterial::PARAM_EMISSION, Color(0.5, 0.7, 0.8));
	mat->set_blend_mode(Material::BLEND_MODE_ADD);
	mat->set_flag(Material::FLAG_DOUBLE_SIDED, true);
	//	mat->set_hint(Material::HINT_NO_DEPTH_DRAW,true);

	surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
	surface_tool->set_material(mat);

	int sides = 16;
	int sections = 24;

	//	float len=1;
	float deg = Math::deg2rad(params[PARAM_EMISSION_CONE_DEGREES]);
	if (deg == 180)
		deg = 179.5;

	Vector3 to = Vector3(0, 0, -1);

	for (int j = 0; j < sections; j++) {

		Vector3 p1 = Matrix3(Vector3(1, 0, 0), deg * j / sections).xform(to);
		Vector3 p2 = Matrix3(Vector3(1, 0, 0), deg * (j + 1) / sections).xform(to);

		for (int i = 0; i < sides; i++) {

			Vector3 p1r = Matrix3(Vector3(0, 0, 1), Math_PI * 2 * float(i) / sides).xform(p1);
			Vector3 p1s = Matrix3(Vector3(0, 0, 1), Math_PI * 2 * float(i + 1) / sides).xform(p1);
			Vector3 p2s = Matrix3(Vector3(0, 0, 1), Math_PI * 2 * float(i + 1) / sides).xform(p2);
			Vector3 p2r = Matrix3(Vector3(0, 0, 1), Math_PI * 2 * float(i) / sides).xform(p2);

			surface_tool->add_normal(p1r.normalized());
			surface_tool->add_vertex(p1r);
			surface_tool->add_normal(p1s.normalized());
			surface_tool->add_vertex(p1s);
			surface_tool->add_normal(p2s.normalized());
			surface_tool->add_vertex(p2s);

			surface_tool->add_normal(p1r.normalized());
			surface_tool->add_vertex(p1r);
			surface_tool->add_normal(p2s.normalized());
			surface_tool->add_vertex(p2s);
			surface_tool->add_normal(p2r.normalized());
			surface_tool->add_vertex(p2r);

			if (j == sections - 1) {

				surface_tool->add_normal(p2r.normalized());
				surface_tool->add_vertex(p2r);
				surface_tool->add_normal(p2s.normalized());
				surface_tool->add_vertex(p2s);
				surface_tool->add_normal(Vector3(0, 0, 1));
				surface_tool->add_vertex(Vector3());
			}
		}
	}

	Ref<Mesh> mesh = surface_tool->commit();

	Ref<FixedMaterial> mat_speaker(memnew(FixedMaterial));

	mat_speaker->set_parameter(FixedMaterial::PARAM_DIFFUSE, Color(0.3, 0.3, 0.6));
	mat_speaker->set_parameter(FixedMaterial::PARAM_SPECULAR, Color(0.5, 0.5, 0.6));
	//mat_speaker->set_blend_mode( Material::BLEND_MODE_MIX);
	//mat_speaker->set_flag(Material::FLAG_DOUBLE_SIDED,false);
	//mat_speaker->set_flag(Material::FLAG_UNSHADED,true);

	surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
	surface_tool->set_material(mat_speaker);

	//	float radius=1;

	const int speaker_points = 8;
	Vector3 speaker[speaker_points] = {
		Vector3(0, 0, 1) * 0.15,
		Vector3(1, 1, 1) * 0.15,
		Vector3(1, 1, 0) * 0.15,
		Vector3(2, 2, -1) * 0.15,
		Vector3(1, 1, -1) * 0.15,
		Vector3(0.8, 0.8, -1.2) * 0.15,
		Vector3(0.5, 0.5, -1.4) * 0.15,
		Vector3(0.0, 0.0, -1.6) * 0.15
	};

	int speaker_sides = 10;

	for (int i = 0; i < speaker_sides; i++) {

		Matrix3 ma(Vector3(0, 0, 1), Math_PI * 2 * float(i) / speaker_sides);
		Matrix3 mb(Vector3(0, 0, 1), Math_PI * 2 * float(i + 1) / speaker_sides);

		for (int j = 0; j < speaker_points - 1; j++) {

			Vector3 points[4] = {
				ma.xform(speaker[j]),
				mb.xform(speaker[j]),
				mb.xform(speaker[j + 1]),
				ma.xform(speaker[j + 1]),
			};

			Vector3 n = -Plane(points[0], points[1], points[2]).normal;

			surface_tool->add_normal(n);
			surface_tool->add_vertex(points[0]);
			surface_tool->add_normal(n);
			surface_tool->add_vertex(points[2]);
			surface_tool->add_normal(n);
			surface_tool->add_vertex(points[1]);

			surface_tool->add_normal(n);
			surface_tool->add_vertex(points[0]);
			surface_tool->add_normal(n);
			surface_tool->add_vertex(points[3]);
			surface_tool->add_normal(n);
			surface_tool->add_vertex(points[2]);
		}
	}

	return surface_tool->commit(mesh);
}

void SpatialPlayer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_param", "param", "value"), &SpatialPlayer::set_param);
	ObjectTypeDB::bind_method(_MD("get_param", "param"), &SpatialPlayer::get_param);

	BIND_CONSTANT(PARAM_VOLUME_DB);
	BIND_CONSTANT(PARAM_PITCH_SCALE);
	BIND_CONSTANT(PARAM_ATTENUATION_MIN_DISTANCE);
	BIND_CONSTANT(PARAM_ATTENUATION_MAX_DISTANCE);
	BIND_CONSTANT(PARAM_ATTENUATION_DISTANCE_EXP);
	BIND_CONSTANT(PARAM_EMISSION_CONE_DEGREES);
	BIND_CONSTANT(PARAM_EMISSION_CONE_ATTENUATION_DB);
	BIND_CONSTANT(PARAM_MAX);

	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/volume_db", PROPERTY_HINT_RANGE, "-80,24,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_VOLUME_DB);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/pitch_scale", PROPERTY_HINT_RANGE, "0.001,32,0.001"), _SCS("set_param"), _SCS("get_param"), PARAM_PITCH_SCALE);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/attenuation/min_distance", PROPERTY_HINT_RANGE, "0.01,4096,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION_MIN_DISTANCE);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/attenuation/max_distance", PROPERTY_HINT_RANGE, "0.01,4096,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION_MAX_DISTANCE);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/attenuation/distance_exp", PROPERTY_HINT_EXP_EASING, "attenuation"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION_DISTANCE_EXP);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/emission_cone/degrees", PROPERTY_HINT_RANGE, "0,180,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_EMISSION_CONE_DEGREES);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "params/emission_cone/attenuation_db", PROPERTY_HINT_RANGE, "-80,24,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_EMISSION_CONE_ATTENUATION_DB);
}

SpatialPlayer::SpatialPlayer() {

	params[PARAM_VOLUME_DB] = 0.0;
	params[PARAM_PITCH_SCALE] = 1.0;
	params[PARAM_ATTENUATION_MIN_DISTANCE] = 1;
	params[PARAM_ATTENUATION_MAX_DISTANCE] = 100;
	params[PARAM_ATTENUATION_DISTANCE_EXP] = 1.0; //linear (and not really good)
	params[PARAM_EMISSION_CONE_DEGREES] = 180.0; //cone disabled
	params[PARAM_EMISSION_CONE_ATTENUATION_DB] = -6.0; //minus 6 db attenuation
}

SpatialPlayer::~SpatialPlayer() {
}
