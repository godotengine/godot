/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Math */

#ifndef OMATH_H
#define OMATH_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define POW2(_x) ((_x) * (_x))
#define RAD_TO_DEG(_r) ((_r) * 360.0f / (2.0f * (float)M_PI))
#define DEG_TO_RAD(_d) ((_d) * (2.0f * (float)M_PI) / 360.0f)


// vector

typedef union { 
	struct { 
		float x, y, z; 
	}; 
	float arr[3]; 
} vec3f;

void ovec3f_normalize_me(vec3f* me);
float ovec3f_get_length(const vec3f* me);
float ovec3f_get_angle(const vec3f* me, const vec3f* vec); 
float ovec3f_get_dot(const vec3f* me, const vec3f* vec);
void ovec3f_subtract(const vec3f* a, const vec3f* b, vec3f* out);


// quaternion

typedef union { 
	struct { 
		float x, y, z, w; 
	}; 
	float arr[4]; 
} quatf;

void oquatf_init_axis(quatf* me, const vec3f* vec, float angle);

void oquatf_get_rotated(const quatf* me, const vec3f* vec, vec3f* out_vec);
void oquatf_mult_me(quatf* me, const quatf* q);
void oquatf_mult(const quatf* me, const quatf* q, quatf* out_q);
void oquatf_diff(const quatf* me, const quatf* q, quatf* out_q);
void oquatf_normalize_me(quatf* me);
float oquatf_get_length(const quatf* me);
float oquatf_get_dot(const quatf* me, const quatf* q);
void oquatf_inverse(quatf* me);

void oquatf_get_mat4x4(const quatf* me, const vec3f* point, float mat[4][4]);

// matrix

typedef union {
	float m[4][4];
	float arr[16];
} mat4x4f;

void omat4x4f_init_ident(mat4x4f* me);
void omat4x4f_init_perspective(mat4x4f* me, float fov_rad, float aspect, float znear, float zfar);
void omat4x4f_init_frustum(mat4x4f* me, float left, float right, float bottom, float top, float znear, float zfar);
void omat4x4f_init_look_at(mat4x4f* me, const quatf* ret, const vec3f* eye);
void omat4x4f_init_translate(mat4x4f* me, float x, float y, float z);
void omat4x4f_mult(const mat4x4f* left, const mat4x4f* right, mat4x4f* out_mat);
void omat4x4f_transpose(const mat4x4f* me, mat4x4f* out_mat);


// filter queue
#define FILTER_QUEUE_MAX_SIZE 256

typedef struct {
	int at, size;
	vec3f elems[FILTER_QUEUE_MAX_SIZE];
} filter_queue;

void ofq_init(filter_queue* me, int size);
void ofq_add(filter_queue* me, const vec3f* vec);
void ofq_get_mean(const filter_queue* me, vec3f* vec);

#endif
