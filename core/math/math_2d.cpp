/*************************************************************************/
/*  math_2d.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "math_2d.h"


real_t Vector2::angle() const {

	return Math::atan2(x,y);
}

float Vector2::length() const {

	return Math::sqrt( x*x + y*y );
}

float Vector2::length_squared() const {

	return x*x + y*y;
}

void Vector2::normalize() {

	float l = x*x + y*y;
	if (l!=0) {

		l=Math::sqrt(l);
		x/=l;
		y/=l;
	}
}

Vector2 Vector2::normalized() const {

	Vector2 v=*this;
	v.normalize();
	return v;
}

float Vector2::distance_to(const Vector2& p_vector2) const {

	return Math::sqrt( (x-p_vector2.x)*(x-p_vector2.x) + (y-p_vector2.y)*(y-p_vector2.y));
}

float Vector2::distance_squared_to(const Vector2& p_vector2) const {

	return (x-p_vector2.x)*(x-p_vector2.x) + (y-p_vector2.y)*(y-p_vector2.y);
}

float Vector2::angle_to(const Vector2& p_vector2) const  {

	return Math::atan2( tangent().dot(p_vector2), dot(p_vector2) );
}

float Vector2::angle_to_point(const Vector2& p_vector2) const  {

	return Math::atan2( x-p_vector2.x, y - p_vector2.y );
}

float Vector2::dot(const Vector2& p_other) const {

	return x*p_other.x + y*p_other.y;
}

float Vector2::cross(const Vector2& p_other) const {

	return x*p_other.y - y*p_other.x;
}

Vector2 Vector2::cross(real_t p_other) const {

	return Vector2(p_other*y,-p_other*x);
}


Vector2 Vector2::operator+(const Vector2& p_v) const {

	return Vector2(x+p_v.x,y+p_v.y);
}
void Vector2::operator+=(const Vector2& p_v) {

	x+=p_v.x; y+=p_v.y;
}
Vector2 Vector2::operator-(const Vector2& p_v) const {

	return Vector2(x-p_v.x,y-p_v.y);
}
void Vector2::operator-=(const Vector2& p_v) {

	x-=p_v.x; y-=p_v.y;
}

Vector2 Vector2::operator*(const Vector2 &p_v1) const {

	return Vector2(x * p_v1.x, y * p_v1.y);
};

Vector2 Vector2::operator*(const float &rvalue) const {

	return Vector2(x * rvalue, y * rvalue);
};
void Vector2::operator*=(const float &rvalue) {

	x *= rvalue; y *= rvalue;
};

Vector2 Vector2::operator/(const Vector2 &p_v1) const {

	return Vector2(x / p_v1.x, y / p_v1.y);
};

Vector2 Vector2::operator/(const float &rvalue) const {

	return Vector2(x / rvalue, y / rvalue);
};

void Vector2::operator/=(const float &rvalue) {

	x /= rvalue; y /= rvalue;
};

Vector2 Vector2::operator-() const {

	return Vector2(-x,-y);
}

bool Vector2::operator==(const Vector2& p_vec2) const {

	return x==p_vec2.x && y==p_vec2.y;
}
bool Vector2::operator!=(const Vector2& p_vec2) const {

	return x!=p_vec2.x || y!=p_vec2.y;
}
Vector2 Vector2::floor() const {

	return Vector2( Math::floor(x), Math::floor(y) );
}

Vector2 Vector2::rotated(float p_by) const {

	Vector2 v;
	v.set_rotation(angle()+p_by);
	v*=length();
	return v;
}

Vector2 Vector2::project(const Vector2& p_vec) const {

	Vector2 v1=p_vec;
	Vector2 v2=*this;
	return v2 * ( v1.dot(v2) / v2.dot(v2));
}

Vector2 Vector2::snapped(const Vector2& p_by) const {

	return Vector2(
		Math::stepify(x,p_by.x),
		Math::stepify(y,p_by.y)
	);
}

Vector2 Vector2::clamped(real_t p_len) const {

	real_t l = length();
	Vector2 v = *this;
	if (l>0 && p_len<l) {

		v/=l;
		v*=p_len;
	}

	return v;
}

Vector2 Vector2::cubic_interpolate_soft(const Vector2& p_b,const Vector2& p_pre_a, const Vector2& p_post_b,float p_t) const {
#if 0
	k[0] = ((*this) (vi[0] + 1, vi[1], vi[2])) - ((*this) (vi[0],
	vi[1],vi[2])); //fk = a0
	k[1] = (((*this) (vi[0] + 1, vi[1], vi[2])) - ((*this) ((int) (v(0) -
	1), vi[1],vi[2])))*0.5; //dk = a1
	k[2] = (((*this) ((int) (v(0) + 2), vi[1], vi[2])) - ((*this) (vi[0],
	vi[1],vi[2])))*0.5; //dk+1
	k[3] = k[0]*3 - k[1]*2 - k[2];//a2
	k[4] = k[1] + k[2] - k[0]*2;//a3

	//ip = a3(t-tk)³ + a2(t-tk)² + a1(t-tk) + a0
	//
	//a3 = dk + dk+1 - Dk
	//a2 = 3Dk - 2dk - dk+1
	//a1 = dk
	//a0 = fk
	//
	//dk = (fk+1 - fk-1)*0.5
	//Dk = (fk+1 - fk)

	float dk =
#endif

	return Vector2();
}

Vector2 Vector2::cubic_interpolate(const Vector2& p_b,const Vector2& p_pre_a, const Vector2& p_post_b,float p_t) const {



	Vector2 p0=p_pre_a;
	Vector2 p1=*this;
	Vector2 p2=p_b;
	Vector2 p3=p_post_b;

	float t = p_t;
	float t2 = t * t;
	float t3 = t2 * t;

	Vector2 out;
	out = 0.5f * ( ( p1 * 2.0f) +
	( -p0 + p2 ) * t +
	( 2.0f * p0 - 5.0f * p1 + 4 * p2 - p3 ) * t2 +
	( -p0 + 3.0f * p1 - 3.0f * p2 + p3 ) * t3 );
	return out;

/*
	float mu = p_t;
	float mu2 = mu*mu;

	Vector2 a0 = p_post_b - p_b - p_pre_a + *this;
	Vector2 a1 = p_pre_a - *this - a0;
	Vector2 a2 = p_b - p_pre_a;
	Vector2 a3 = *this;

	return ( a0*mu*mu2 + a1*mu2 + a2*mu + a3 );
*/
	/*
	float t = p_t;
	real_t t2 = t*t;
	real_t t3 = t2*t;

	real_t a =  2.0*t3- 3.0*t2 + 1;
	real_t b = -2.0*t3+ 3.0*t2;
	real_t c =    t3- 2.0*t2 + t;
	real_t d =    t3- t2;

	Vector2 p_a=*this;

	return Vector2(
		(a * p_a.x) + (b *p_b.x) + (c * p_pre_a.x) + (d * p_post_b.x),
		(a * p_a.y) + (b *p_b.y) + (c * p_pre_a.y) + (d * p_post_b.y)
	);
*/

}

Vector2 Vector2::slide(const Vector2& p_vec) const {

	return p_vec - *this * this->dot(p_vec);
}
Vector2 Vector2::reflect(const Vector2& p_vec) const {

	return p_vec - *this * this->dot(p_vec) * 2.0;

}


bool Rect2::intersects_segment(const Point2& p_from, const Point2& p_to, Point2* r_pos,Point2* r_normal) const {

	real_t min=0,max=1;
	int axis=0;
	float sign=0;

	for(int i=0;i<2;i++) {
		real_t seg_from=p_from[i];
		real_t seg_to=p_to[i];
		real_t box_begin=pos[i];
		real_t box_end=box_begin+size[i];
		real_t cmin,cmax;
		float csign;

		if (seg_from < seg_to) {

			if (seg_from > box_end || seg_to < box_begin)
				return false;
			real_t length=seg_to-seg_from;
			cmin = (seg_from < box_begin)?((box_begin - seg_from)/length):0;
			cmax = (seg_to > box_end)?((box_end - seg_from)/length):1;
			csign=-1.0;

		} else {

			if (seg_to > box_end || seg_from < box_begin)
				return false;
			real_t length=seg_to-seg_from;
			cmin = (seg_from > box_end)?(box_end - seg_from)/length:0;
			cmax = (seg_to < box_begin)?(box_begin - seg_from)/length:1;
			csign=1.0;
		}

		if (cmin > min) {
			min = cmin;
			axis=i;
			sign=csign;
		}
		if (cmax < max)
			max = cmax;
		if (max < min)
			return false;
	}


	Vector2 rel=p_to-p_from;

	if (r_normal) {
		Vector2 normal;
		normal[axis]=sign;
		*r_normal=normal;
	}

	if (r_pos)
		*r_pos=p_from+rel*min;

	return true;
}

/* Point2i */

Point2i Point2i::operator+(const Point2i& p_v) const {

	return Point2i(x+p_v.x,y+p_v.y);
}
void Point2i::operator+=(const Point2i& p_v) {

	x+=p_v.x; y+=p_v.y;
}
Point2i Point2i::operator-(const Point2i& p_v) const {

	return Point2i(x-p_v.x,y-p_v.y);
}
void Point2i::operator-=(const Point2i& p_v) {

	x-=p_v.x; y-=p_v.y;
}

Point2i Point2i::operator*(const Point2i &p_v1) const {

	return Point2i(x * p_v1.x, y * p_v1.y);
};

Point2i Point2i::operator*(const int &rvalue) const {

	return Point2i(x * rvalue, y * rvalue);
};
void Point2i::operator*=(const int &rvalue) {

	x *= rvalue; y *= rvalue;
};

Point2i Point2i::operator/(const Point2i &p_v1) const {

	return Point2i(x / p_v1.x, y / p_v1.y);
};

Point2i Point2i::operator/(const int &rvalue) const {

	return Point2i(x / rvalue, y / rvalue);
};

void Point2i::operator/=(const int &rvalue) {

	x /= rvalue; y /= rvalue;
};

Point2i Point2i::operator-() const {

	return Point2i(-x,-y);
}

bool Point2i::operator==(const Point2i& p_vec2) const {

	return x==p_vec2.x && y==p_vec2.y;
}
bool Point2i::operator!=(const Point2i& p_vec2) const {

	return x!=p_vec2.x || y!=p_vec2.y;
}

void Matrix32::invert() {

	SWAP(elements[0][1],elements[1][0]);
	elements[2] = basis_xform(-elements[2]);
}

Matrix32 Matrix32::inverse() const {

	Matrix32 inv=*this;
	inv.invert();
	return inv;

}

void Matrix32::affine_invert() {

	float det = basis_determinant();
	ERR_FAIL_COND(det==0);
	float idet = 1.0 / det;

	SWAP( elements[0][0],elements[1][1] );
	elements[0]*=Vector2(idet,-idet);
	elements[1]*=Vector2(-idet,idet);

	elements[2] = basis_xform(-elements[2]);

}

Matrix32 Matrix32::affine_inverse() const {

	Matrix32 inv=*this;
	inv.affine_invert();
	return inv;
}

void Matrix32::rotate(real_t p_phi) {

	Matrix32 rot(p_phi,Vector2());
	*this *= rot;
}

real_t Matrix32::get_rotation() const {

	return Math::atan2(elements[1].x,elements[1].y);
}

void Matrix32::set_rotation(real_t p_rot) {

	real_t cr = Math::cos(p_rot);
	real_t sr = Math::sin(p_rot);
	elements[0][0]=cr;
	elements[1][1]=cr;
	elements[0][1]=-sr;
	elements[1][0]=sr;
}

Matrix32::Matrix32(real_t p_rot, const Vector2& p_pos) {

	real_t cr = Math::cos(p_rot);
	real_t sr = Math::sin(p_rot);
	elements[0][0]=cr;
	elements[1][1]=cr;
	elements[0][1]=-sr;
	elements[1][0]=sr;
	elements[2]=p_pos;
}

Size2 Matrix32::get_scale() const {

	return Size2( elements[0].length(), elements[1].length() );
}

void Matrix32::scale(const Size2& p_scale) {

	elements[0]*=p_scale;
	elements[1]*=p_scale;
	elements[2]*=p_scale;
}
void Matrix32::scale_basis(const Size2& p_scale) {

	elements[0]*=p_scale;
	elements[1]*=p_scale;

}
void Matrix32::translate( real_t p_tx, real_t p_ty) {

	translate(Vector2(p_tx,p_ty));
}
void Matrix32::translate( const Vector2& p_translation ) {

	elements[2]+=basis_xform(p_translation);
}

void Matrix32::orthonormalize() {

	// Gram-Schmidt Process

	Vector2 x=elements[0];
	Vector2 y=elements[1];

	x.normalize();
	y = (y-x*(x.dot(y)));
	y.normalize();

	elements[0]=x;
	elements[1]=y;
}
Matrix32 Matrix32::orthonormalized() const {

	Matrix32 on=*this;
	on.orthonormalize();
	return on;

}

bool Matrix32::operator==(const Matrix32& p_transform) const {

	for(int i=0;i<3;i++) {
		if (elements[i]!=p_transform.elements[i])
			return false;
	}

	return true;
}

bool Matrix32::operator!=(const Matrix32& p_transform) const {

	for(int i=0;i<3;i++) {
		if (elements[i]!=p_transform.elements[i])
			return true;
	}

	return false;

}

void Matrix32::operator*=(const Matrix32& p_transform) {

	elements[2] = xform(p_transform.elements[2]);

	float x0,x1,y0,y1;

	x0 = tdotx(p_transform.elements[0]);
	x1 = tdoty(p_transform.elements[0]);
	y0 = tdotx(p_transform.elements[1]);
	y1 = tdoty(p_transform.elements[1]);

	elements[0][0]=x0;
	elements[0][1]=x1;
	elements[1][0]=y0;
	elements[1][1]=y1;
}


Matrix32 Matrix32::operator*(const Matrix32& p_transform) const {

	Matrix32 t = *this;
	t*=p_transform;
	return t;

}

Matrix32 Matrix32::scaled(const Size2& p_scale) const {

	Matrix32 copy=*this;
	copy.scale(p_scale);
	return copy;

}

Matrix32 Matrix32::basis_scaled(const Size2& p_scale) const {

	Matrix32 copy=*this;
	copy.scale_basis(p_scale);
	return copy;

}

Matrix32 Matrix32::untranslated() const {

	Matrix32 copy=*this;
	copy.elements[2]=Vector2();
	return copy;
}

Matrix32 Matrix32::translated(const Vector2& p_offset) const {

	Matrix32 copy=*this;
	copy.translate(p_offset);
	return copy;

}

Matrix32 Matrix32::rotated(float p_phi) const {

	Matrix32 copy=*this;
	copy.rotate(p_phi);
	return copy;

}

float Matrix32::basis_determinant() const {

	return elements[0].x * elements[1].y - elements[0].y * elements[1].x;
}

Matrix32 Matrix32::interpolate_with(const Matrix32& p_transform, float p_c) const {

	//extract parameters
	Vector2 p1 = get_origin();
	Vector2 p2 = p_transform.get_origin();

	real_t r1 = get_rotation();
	real_t r2 = p_transform.get_rotation();

	Size2 s1 = get_scale();
	Size2 s2 = p_transform.get_scale();

	//slerp rotation
	Vector2 v1(Math::cos(r1), Math::sin(r1));
	Vector2 v2(Math::cos(r2), Math::sin(r2));

	real_t dot = v1.dot(v2);

	dot = (dot < -1.0) ? -1.0 : ((dot > 1.0) ? 1.0 : dot); //clamp dot to [-1,1]

	Vector2 v;

	if (dot > 0.9995) {
		v = Vector2::linear_interpolate(v1, v2, p_c).normalized(); //linearly interpolate to avoid numerical precision issues
	} else {
		real_t angle = p_c*Math::acos(dot);
		Vector2 v3 = (v2 - v1*dot).normalized();
		v = v1*Math::cos(angle) + v3*Math::sin(angle);
	}

	//construct matrix
	Matrix32 res(Math::atan2(v.y, v.x), Vector2::linear_interpolate(p1, p2, p_c));
	res.scale_basis(Vector2::linear_interpolate(s1, s2, p_c));
	return res;
}

Matrix32::operator String() const {

	return String(String()+elements[0]+", "+elements[1]+", "+elements[2]);
}
