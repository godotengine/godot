/*
 * Copyright (c) 2007 Eric Jordan
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "b2Triangle.h"

namespace b2ConvexDecomp {

//Constructor automatically fixes orientation to ccw
b2Triangle::b2Triangle(float32 x1, float32 y1, float32 x2, float32 y2, float32 x3, float32 y3){
	x = new float32[3];
	y = new float32[3];
	float32 dx1 = x2-x1;
	float32 dx2 = x3-x1;
	float32 dy1 = y2-y1;
	float32 dy2 = y3-y1;
	float32 cross = dx1*dy2-dx2*dy1;
	bool ccw = (cross>0);
	if (ccw){
		x[0] = x1; x[1] = x2; x[2] = x3;
		y[0] = y1; y[1] = y2; y[2] = y3;
	} else{
		x[0] = x1; x[1] = x3; x[2] = x2;
		y[0] = y1; y[1] = y3; y[2] = y2;
	}
}
    
b2Triangle::b2Triangle(){
	x = new float32[3];
	y = new float32[3];
}

b2Triangle::~b2Triangle(){
	delete[] x;
	delete[] y;
}

void b2Triangle::Set(const b2Triangle& toMe) {
	for (int32 i=0; i<3; ++i) {
		x[i] = toMe.x[i];
		y[i] = toMe.y[i];
	}
}

bool b2Triangle::IsInside(float32 _x, float32 _y){
	if (_x < x[0] && _x < x[1] && _x < x[2]) return false;
	if (_x > x[0] && _x > x[1] && _x > x[2]) return false;
	if (_y < y[0] && _y < y[1] && _y < y[2]) return false;
	if (_y > y[0] && _y > y[1] && _y > y[2]) return false;
		
		float32 vx2 = _x-x[0]; float32 vy2 = _y-y[0];
		float32 vx1 = x[1]-x[0]; float32 vy1 = y[1]-y[0];
		float32 vx0 = x[2]-x[0]; float32 vy0 = y[2]-y[0];
		
		float32 dot00 = vx0*vx0+vy0*vy0;
		float32 dot01 = vx0*vx1+vy0*vy1;
		float32 dot02 = vx0*vx2+vy0*vy2;
		float32 dot11 = vx1*vx1+vy1*vy1;
		float32 dot12 = vx1*vx2+vy1*vy2;
		float32 invDenom = 1.0f / (dot00*dot11 - dot01*dot01);
		float32 u = (dot11*dot02 - dot01*dot12)*invDenom;
		float32 v = (dot00*dot12 - dot01*dot02)*invDenom;
		
		return ((u>=0)&&(v>=0)&&(u+v<=1));    
}


}
