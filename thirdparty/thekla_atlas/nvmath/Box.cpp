// This code is in the public domain -- castanyo@yahoo.es

#include "Box.h"
#include "Box.inl"
#include "Sphere.h"

using namespace nv;




// Clip the given segment against this box.
bool Box::clipSegment(const Vector3 & origin, const Vector3 & dir, float * t_near, float * t_far) const {

	// Avoid aliasing.
	float tnear = *t_near;
	float tfar = *t_far;

	// clip ray segment to box
	for (int i = 0; i < 3; i++)
	{
		const float pos = origin.component[i] + tfar * dir.component[i];
		const float dt = tfar - tnear;

		if (dir.component[i] < 0) {
			
			// clip end point
			if (pos < minCorner.component[i]) {
                tfar = tnear + dt * (origin.component[i] - minCorner.component[i]) / (origin.component[i] - pos);
			}
			
			// clip start point
			if (origin.component[i] > maxCorner.component[i]) {
				tnear = tnear + dt * (origin.component[i] - maxCorner.component[i]) / (tfar * dir.component[i]);
			}
		}
		else {

			// clip end point
			if (pos > maxCorner.component[i]) {
				tfar = tnear + dt * (maxCorner.component[i] - origin.component[i]) / (pos - origin.component[i]);
			}

			// clip start point
			if (origin.component[i] < minCorner.component[i]) {
				tnear = tnear + dt * (minCorner.component[i] - origin.component[i]) / (tfar * dir.component[i]);
			}
		}

		if (tnear > tfar) {
			// Clipped away.
			return false;
		}
	}

	// Return result.
	*t_near = tnear;
	*t_far = tfar;
	return true;
}


float nv::distanceSquared(const Box &box, const Vector3 &point) {
    Vector3 closest;

    if (point.x < box.minCorner.x) closest.x = box.minCorner.x;
    else if (point.x > box.maxCorner.x) closest.x = box.maxCorner.x;
    else closest.x = point.x;

    if (point.y < box.minCorner.y) closest.y = box.minCorner.y;
    else if (point.y > box.maxCorner.y) closest.y = box.maxCorner.y;
    else closest.y = point.y;

    if (point.z < box.minCorner.z) closest.z = box.minCorner.z;
    else if (point.z > box.maxCorner.z) closest.z = box.maxCorner.z;
    else closest.z = point.z;

    return lengthSquared(point - closest);
}

bool nv::overlap(const Box &box, const Sphere &sphere) {
    return distanceSquared(box, sphere.center) < sphere.radius * sphere.radius;
}


bool nv::intersect(const Box & box, const Vector3 & p, const Vector3 & id, float * t /*= NULL*/) {
    // Precompute these in ray structure?
    int sdx = (id.x < 0);
    int sdy = (id.y < 0);
    int sdz = (id.z < 0);

    float tmin = (box.corner(  sdx).x - p.x) * id.x;
    float tmax = (box.corner(1-sdx).x - p.x) * id.x;
    float tymin = (box.corner(  sdy).y - p.y) * id.y;
    float tymax = (box.corner(1-sdy).y - p.y) * id.y;

    if ((tmin > tymax) || (tymin > tmax)) 
        return false;

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    float tzmin = (box.corner(  sdz).z - p.z) * id.z;
    float tzmax = (box.corner(1-sdz).z - p.z) * id.z;

    if ((tmin > tzmax) || (tzmin > tmax)) 
        return false;

    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    if (tmax < 0) 
        return false;

    if (t != NULL) *t = tmin;

    return true;
}

