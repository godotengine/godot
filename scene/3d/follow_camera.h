/*************************************************************************/
/*  follow_camera.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef FOLLOW_CAMERA_H
#define FOLLOW_CAMERA_H

#include "scene/3d/camera.h"

class FollowCamera : public Camera {

	OBJ_TYPE( FollowCamera, Camera );

private:


	//used for follow
	Vector3 follow_pos;
	//used for fixed
	Vector2 initial_orbit;
	Vector2 orbit;
	float distance;

	float height;
	float target_width;

	float min_distance;
	float max_distance;

	float max_orbit_x;
	float min_orbit_x;

	float inclination;
	float extraclip;
	bool fullclip;

	bool clip;
	bool autoturn;
	float autoturn_tolerance;
	float autoturn_speed;

	bool smooth;
	float smooth_rot_ratio;
	float smooth_pos_ratio;



	struct ClipRay {
		RID query;
		bool clipped;
		Vector3 cast_pos;
		Vector3 clip_pos;
	};

	ClipRay clip_ray[3];
	Vector3 target_pos;
	float clip_len;

	Vector3 up_vector;


	virtual RES _get_gizmo_geometry() const;

	Transform ted;
	Vector3 proposed_pos;
	Transform accepted;
	Transform final;
	RID target_body;

	bool use_lookat_target;
	Vector3 lookat_target;

	void _compute_camera();

	bool queries_active;
	void _clear_queries();

	void _set_initial_orbit(const Vector2& p_orbit);

protected:

	virtual void _request_camera_update() {} //ignore

	void _notification(int p_what);

	static void _bind_methods();

	void _ray_collision(Vector3 p_point, Vector3 p_normal, int p_subindex, ObjectID p_against,int p_idx);

public:


	void set_orbit(const Vector2& p_orbit);
	void set_orbit_x(float p_x);
	void set_orbit_y(float p_y);
	Vector2 get_orbit() const;

	void set_height(float p_height);
	float get_height() const;

	void set_inclination(float p_degrees);
	float get_inclination() const;

	void set_max_orbit_x(float p_max);
	float get_max_orbit_x() const;

	void set_min_orbit_x(float p_min);
	float get_min_orbit_x() const;

	void rotate_orbit(const Vector2& p_relative);

	void set_distance(float p_distance);
	float get_distance() const;

	float get_min_distance() const;
	float get_max_distance() const;
	void set_min_distance(float p_min);
	void set_max_distance(float p_max);

	/** FINISH THIS AND CLEAN IT  UP */

	void set_clip(bool p_enabled);
	bool has_clip() const;

	void set_autoturn(bool p_enabled);
	bool has_autoturn() const;

	void set_autoturn_tolerance(float p_degrees);
	float get_autoturn_tolerance() const;

	void set_autoturn_speed(float p_speed);
	float get_autoturn_speed() const;

	void set_smoothing(bool p_enable);
	bool has_smoothing() const;

	void set_translation_smoothing(float p_amount);
	float get_translation_smoothing() const;

	void set_rotation_smoothing(float p_amount);
	float get_rotation_smoothing() const;

	void set_use_lookat_target(bool p_use, const Vector3 &p_lookat = Vector3());

	void set_up_vector(const Vector3& p_up);
	Vector3 get_up_vector() const;

	virtual Transform get_camera_transform() const;

	FollowCamera();
	~FollowCamera();
};



#endif // FOLLOW_CAMERA_H
