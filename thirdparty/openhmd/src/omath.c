/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Math Code Implementation */

#include <string.h>
#include "openhmdi.h"

// vector

float ovec3f_get_length(const vec3f* me)
{
	return sqrtf(POW2(me->x) + POW2(me->y) + POW2(me->z));
}

void ovec3f_normalize_me(vec3f* me)
{
	if(me->x == 0 && me->y == 0 && me->z == 0)
		return;

	float len = ovec3f_get_length(me);
	me->x /= len;
	me->y /= len;
	me->z /= len;
}

void ovec3f_subtract(const vec3f* a, const vec3f* b, vec3f* out)
{
	for(int i = 0; i < 3; i++)
		out->arr[i] = a->arr[i] - b->arr[i];
}

float ovec3f_get_dot(const vec3f* me, const vec3f* vec)
{
	return me->x * vec->x + me->y * vec->y + me->z * vec->z;
}

float ovec3f_get_angle(const vec3f* me, const vec3f* vec)
{
	float dot = ovec3f_get_dot(me, vec);
	float lengths = ovec3f_get_length(me) * ovec3f_get_length(vec);

	if(lengths == 0)
		return 0;

	return acosf(dot / lengths);
}


// quaternion

void oquatf_init_axis(quatf* me, const vec3f* vec, float angle)
{
	vec3f norm = *vec;
	ovec3f_normalize_me(&norm);

	me->x = norm.x * sinf(angle / 2.0f);
	me->y = norm.y * sinf(angle / 2.0f);
	me->z = norm.z * sinf(angle / 2.0f);
	me->w = cosf(angle / 2.0f);
}

void oquatf_get_rotated(const quatf* me, const vec3f* vec, vec3f* out_vec)
{
	quatf q = {{vec->x * me->w + vec->z * me->y - vec->y * me->z,
	            vec->y * me->w + vec->x * me->z - vec->z * me->x,
	            vec->z * me->w + vec->y * me->x - vec->x * me->y,
	            vec->x * me->x + vec->y * me->y + vec->z * me->z}};

	out_vec->x = me->w * q.x + me->x * q.w + me->y * q.z - me->z * q.y;
	out_vec->y = me->w * q.y + me->y * q.w + me->z * q.x - me->x * q.z;
	out_vec->z = me->w * q.z + me->z * q.w + me->x * q.y - me->y * q.x;
}

void oquatf_mult(const quatf* me, const quatf* q, quatf* out_q)
{
	out_q->x = me->w * q->x + me->x * q->w + me->y * q->z - me->z * q->y;
	out_q->y = me->w * q->y - me->x * q->z + me->y * q->w + me->z * q->x;
	out_q->z = me->w * q->z + me->x * q->y - me->y * q->x + me->z * q->w;
	out_q->w = me->w * q->w - me->x * q->x - me->y * q->y - me->z * q->z;
}

void oquatf_mult_me(quatf* me, const quatf* q)
{
	quatf tmp = *me;
	oquatf_mult(&tmp, q, me);
}

void oquatf_normalize_me(quatf* me)
{
	float len = oquatf_get_length(me);
	me->x /= len;
	me->y /= len;
	me->z /= len;
	me->w /= len;
}

float oquatf_get_length(const quatf* me)
{
	return sqrtf(me->x * me->x + me->y * me->y + me->z * me->z + me->w * me->w);
}

float oquatf_get_dot(const quatf* me, const quatf* q)
{
	return me->x * q->x + me->y * q->y + me->z * q->z + me->w * q->w;
}

void oquatf_inverse(quatf* me)
{
	float dot = oquatf_get_dot(me, me);

	// conjugate
	for(int i = 0; i < 3; i++)
		me->arr[i] = -me->arr[i];
	
	for(int i = 0; i < 4; i++)
		me->arr[i] /= dot;
}

void oquatf_diff(const quatf* me, const quatf* q, quatf* out_q)
{
	quatf inv = *me;
	oquatf_inverse(&inv);
	oquatf_mult(&inv, q, out_q);
}

void oquatf_slerp (float fT, const quatf* rkP, const quatf* rkQ, bool shortestPath, quatf* out_q)
{
	float fCos =  oquatf_get_dot(rkP, rkQ);
	quatf rkT;

	// Do we need to invert rotation?
	if (fCos < 0.0f && shortestPath)
	{
		fCos = -fCos;
		rkT = *rkQ;
		oquatf_inverse(&rkT);
	}
	else
	{
		rkT = *rkQ;
	}

	if (fabsf(fCos) < 1 - 0.001f)
	{
		// Standard case (slerp)
		float fSin = sqrtf(1 - (fCos*fCos));
		float fAngle = atan2f(fSin, fCos); 
		float fInvSin = 1.0f / fSin;
		float fCoeff0 = sin((1.0f - fT) * fAngle) * fInvSin;
		float fCoeff1 = sin(fT * fAngle) * fInvSin;
		
		out_q->x = fCoeff0 * rkP->x + fCoeff1 * rkT.x;
		out_q->y = fCoeff0 * rkP->y + fCoeff1 * rkT.y;
		out_q->z = fCoeff0 * rkP->z + fCoeff1 * rkT.z;
		out_q->w = fCoeff0 * rkP->w + fCoeff1 * rkT.w;
			
		//return fCoeff0 * rkP + fCoeff1 * rkT;
	}
	else
	{
		// There are two situations:
		// 1. "rkP" and "rkQ" are very close (fCos ~= +1), so we can do a linear
		//    interpolation safely.
		// 2. "rkP" and "rkQ" are almost inverse of each other (fCos ~= -1), there
		//    are an infinite number of possibilities interpolation. but we haven't
		//    have method to fix this case, so just use linear interpolation here.
		//Quaternion t = (1.0f - fT) * rkP + fT * rkT;
		
		out_q->x = (1.0f - fT) * rkP->x + fT * rkT.x;
		out_q->y = (1.0f - fT) * rkP->y + fT * rkT.y;
		out_q->z = (1.0f - fT) * rkP->z + fT * rkT.z;
		out_q->w = (1.0f - fT) * rkP->w + fT * rkT.w;
			
		oquatf_normalize_me(out_q);
		
		// taking the complement requires renormalisation
		//t.normalise();
		//return t;
	}
}

void oquatf_get_mat4x4(const quatf* me, const vec3f* point, float mat[4][4])
{
	mat[0][0] = 1 - 2 * me->y * me->y - 2 * me->z * me->z;
	mat[0][1] =     2 * me->x * me->y - 2 * me->w * me->z;
	mat[0][2] =     2 * me->x * me->z + 2 * me->w * me->y;
	mat[0][3] = point->x;

	mat[1][0] =     2 * me->x * me->y + 2 * me->w * me->z;
	mat[1][1] = 1 - 2 * me->x * me->x - 2 * me->z * me->z;
	mat[1][2] =     2 * me->y * me->z - 2 * me->w * me->x;
	mat[1][3] = point->y;

	mat[2][0] =     2 * me->x * me->z - 2 * me->w * me->y;
	mat[2][1] =     2 * me->y * me->z + 2 * me->w * me->x;
	mat[2][2] = 1 - 2 * me->x * me->x - 2 * me->y * me->y;
	mat[2][3] = point->z;

	mat[3][0] = 0;
	mat[3][1] = 0;
	mat[3][2] = 0;
	mat[3][3] = 1;
}


// matrix

void omat4x4f_init_ident(mat4x4f* me)
{
	memset(me, 0, sizeof(*me));
	me->m[0][0] = 1.0f;
	me->m[1][1] = 1.0f;
	me->m[2][2] = 1.0f;
	me->m[3][3] = 1.0f;
}

void omat4x4f_init_perspective(mat4x4f* me, float fovy_rad, float aspect, float znear, float zfar)
{
	float sine, cotangent, delta, half_fov;

	half_fov = fovy_rad / 2.0f;
	delta = zfar - znear;
	sine = sinf(half_fov);

	if ((delta == 0.0f) || (sine == 0.0f) || (aspect == 0.0f)) {
		omat4x4f_init_ident(me);
		return;
	}

	cotangent = cosf(half_fov) / sine;

	me->m[0][0] = cotangent / aspect;
	me->m[0][1] = 0;
	me->m[0][2] = 0;
	me->m[0][3] = 0;

	me->m[1][0] = 0;
	me->m[1][1] = cotangent;
	me->m[1][2] = 0;
	me->m[1][3] = 0;

	me->m[2][0] = 0;
	me->m[2][1] = 0;
	me->m[2][2] = -(zfar + znear) / delta;
	me->m[2][3] = -2.0f * znear * zfar / delta;

	me->m[3][0] = 0;
	me->m[3][1] = 0;
	me->m[3][2] = -1.0f;
	me->m[3][3] = 0;
}

void omat4x4f_init_frustum(mat4x4f* me, float left, float right, float bottom, float top, float znear, float zfar)
{
    omat4x4f_init_ident(me);

    float delta_x = right - left;
    float delta_y = top - bottom;
	float delta_z = zfar - znear;
	if ((delta_x == 0.0f) || (delta_y == 0.0f) || (delta_z == 0.0f)) {
        /* can't divide by zero, so just give back identity */
		return;
	}

	me->m[0][0] = 2.0f * znear / delta_x;
	me->m[0][1] = 0;
	me->m[0][2] = (right + left) / delta_x;
	me->m[0][3] = 0;

	me->m[1][0] = 0;
	me->m[1][1] = 2.0f * znear / delta_y;
	me->m[1][2] = (top + bottom) / delta_y;
	me->m[1][3] = 0;

	me->m[2][0] = 0;
	me->m[2][1] = 0;
	me->m[2][2] = -(zfar + znear) / delta_z;
	me->m[2][3] = -2.0f * zfar * znear / delta_z;

	me->m[3][0] = 0;
	me->m[3][1] = 0;
	me->m[3][2] = -1.0f;
	me->m[3][3] = 0;
}

void omat4x4f_init_look_at(mat4x4f* me, const quatf* rot, const vec3f* eye)
{
	quatf q;
	vec3f p;

	q.x = -rot->x;
	q.y = -rot->y;
	q.z = -rot->z;
	q.w =  rot->w;

	p.x = -eye->x;
	p.y = -eye->y;
	p.z = -eye->z;

	me->m[0][0] = 1 - 2 * q.y * q.y - 2 * q.z * q.z;
	me->m[0][1] =     2 * q.x * q.y - 2 * q.w * q.z;
	me->m[0][2] =     2 * q.x * q.z + 2 * q.w * q.y;
	me->m[0][3] = p.x * me->m[0][0] + p.y * me->m[0][1] + p.z * me->m[0][2];

	me->m[1][0] =     2 * q.x * q.y + 2 * q.w * q.z;
	me->m[1][1] = 1 - 2 * q.x * q.x - 2 * q.z * q.z;
	me->m[1][2] =     2 * q.y * q.z - 2 * q.w * q.x;
	me->m[1][3] = p.x * me->m[1][0] + p.y * me->m[1][1] + p.z * me->m[1][2];

	me->m[2][0] =     2 * q.x * q.z - 2 * q.w * q.y;
	me->m[2][1] =     2 * q.y * q.z + 2 * q.w * q.x;
	me->m[2][2] = 1 - 2 * q.x * q.x - 2 * q.y * q.y;
	me->m[2][3] = p.x * me->m[2][0] + p.y * me->m[2][1] + p.z * me->m[2][2];

	me->m[3][0] = 0;
	me->m[3][1] = 0;
	me->m[3][2] = 0;
	me->m[3][3] = 1;
}

void omat4x4f_init_translate(mat4x4f* me, float x, float y, float z)
{
	omat4x4f_init_ident(me);
	me->m[0][3] = x;
	me->m[1][3] = y;
	me->m[2][3] = z;
}

void omat4x4f_transpose(const mat4x4f* m, mat4x4f* o)
{
	o->m[0][0] = m->m[0][0];
	o->m[1][0] = m->m[0][1];
	o->m[2][0] = m->m[0][2];
	o->m[3][0] = m->m[0][3];

	o->m[0][1] = m->m[1][0];
	o->m[1][1] = m->m[1][1];
	o->m[2][1] = m->m[1][2];
	o->m[3][1] = m->m[1][3];

	o->m[0][2] = m->m[2][0];
	o->m[1][2] = m->m[2][1];
	o->m[2][2] = m->m[2][2];
	o->m[3][2] = m->m[2][3];

	o->m[0][3] = m->m[3][0];
	o->m[1][3] = m->m[3][1];
	o->m[2][3] = m->m[3][2];
	o->m[3][3] = m->m[3][3];
}

void omat4x4f_mult(const mat4x4f* l, const mat4x4f* r, mat4x4f *o)
{
	for(int i = 0; i < 4; i++){
		float a0 = l->m[i][0], a1 = l->m[i][1], a2 = l->m[i][2], a3 = l->m[i][3];
		o->m[i][0] = a0 * r->m[0][0] + a1 * r->m[1][0] + a2 * r->m[2][0] + a3 * r->m[3][0];
		o->m[i][1] = a0 * r->m[0][1] + a1 * r->m[1][1] + a2 * r->m[2][1] + a3 * r->m[3][1];
		o->m[i][2] = a0 * r->m[0][2] + a1 * r->m[1][2] + a2 * r->m[2][2] + a3 * r->m[3][2];
		o->m[i][3] = a0 * r->m[0][3] + a1 * r->m[1][3] + a2 * r->m[2][3] + a3 * r->m[3][3];
	}
}


// filter queue

void ofq_init(filter_queue* me, int size)
{
	memset(me, 0, sizeof(filter_queue));
	me->size = size;
}

void ofq_add(filter_queue* me, const vec3f* vec)
{
	me->elems[me->at] = *vec;
	me->at = ((me->at + 1) % me->size);
}

void ofq_get_mean(const filter_queue* me, vec3f* vec)
{
	vec->x = vec->y = vec->z = 0;
	for(int i = 0; i < me->size; i++){
		vec->x += me->elems[i].x;
		vec->y += me->elems[i].y;
		vec->z += me->elems[i].z;
	}

	vec->x /= (float)me->size;
	vec->y /= (float)me->size;
	vec->z /= (float)me->size;
}
