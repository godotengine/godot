// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "Sphere.h"
#include "Vector.inl"
#include "Box.inl"

#include <float.h> // FLT_MAX

using namespace nv;

const float radiusEpsilon = 1e-4f;

Sphere::Sphere(Vector3::Arg p0, Vector3::Arg p1)
{
    if (p0 == p1) *this = Sphere(p0);
    else {
        center = (p0 + p1) * 0.5f;
        radius = length(p0 - center) + radiusEpsilon;

        float d0 = length(p0 - center);
        float d1 = length(p1 - center);
        nvDebugCheck(equal(d0, radius - radiusEpsilon));
        nvDebugCheck(equal(d1, radius - radiusEpsilon));
    }
}

Sphere::Sphere(Vector3::Arg p0, Vector3::Arg p1, Vector3::Arg p2)
{
    if (p0 == p1 || p0 == p2) *this = Sphere(p1, p2);
    else if (p1 == p2) *this = Sphere(p0, p2);
    else {
        Vector3 a = p1 - p0;
        Vector3 b = p2 - p0;
        Vector3 c = cross(a, b);

        float denominator = 2.0f * lengthSquared(c);
        
        if (!isZero(denominator)) {
	        Vector3 d = (lengthSquared(b) * cross(c, a) + lengthSquared(a) * cross(b, c)) / denominator;

	        center = p0 + d;
	        radius = length(d) + radiusEpsilon;

            float d0 = length(p0 - center);
            float d1 = length(p1 - center);
            float d2 = length(p2 - center);
            nvDebugCheck(equal(d0, radius - radiusEpsilon));
            nvDebugCheck(equal(d1, radius - radiusEpsilon));
            nvDebugCheck(equal(d2, radius - radiusEpsilon));
        }
        else {
            // @@ This is a specialization of the code below, but really, the only thing we need to do here is to find the two most distant points.
            // Compute all possible spheres, invalidate those that do not contain the four points, keep the smallest.
            Sphere s0(p1, p2);
            float d0 = distanceSquared(s0, p0);
            if (d0 > 0) s0.radius = NV_FLOAT_MAX;

            Sphere s1(p0, p2);
            float d1 = distanceSquared(s1, p1);
            if (d1 > 0) s1.radius = NV_FLOAT_MAX;

            Sphere s2(p0, p1);
            float d2 = distanceSquared(s2, p2);
            if (d2 > 0) s1.radius = NV_FLOAT_MAX;

            if (s0.radius < s1.radius && s0.radius < s2.radius) {
                center = s0.center;
                radius = s0.radius;
            }
            else if (s1.radius < s2.radius) {
                center = s1.center;
                radius = s1.radius;
            }
            else {
                center = s2.center;
                radius = s2.radius;
            }
        }
    }
}

Sphere::Sphere(Vector3::Arg p0, Vector3::Arg p1, Vector3::Arg p2, Vector3::Arg p3)
{
    if (p0 == p1 || p0 == p2 || p0 == p3) *this = Sphere(p1, p2, p3);
    else if (p1 == p2 || p1 == p3) *this = Sphere(p0, p2, p3);
    else if (p2 == p3) *this = Sphere(p0, p1, p2);
    else {
        // @@ This only works if the points are not coplanar!
	    Vector3 a = p1 - p0;
	    Vector3 b = p2 - p0;
	    Vector3 c = p3 - p0;

        float denominator = 2.0f * dot(c, cross(a, b)); // triple product.

        if (!isZero(denominator)) {
	        Vector3 d = (lengthSquared(c) * cross(a, b) + lengthSquared(b) * cross(c, a) + lengthSquared(a) * cross(b, c)) / denominator;

	        center = p0 + d;
            radius = length(d) + radiusEpsilon;

            float d0 = length(p0 - center);
            float d1 = length(p1 - center);
            float d2 = length(p2 - center);
            float d3 = length(p3 - center);
            nvDebugCheck(equal(d0, radius - radiusEpsilon));
            nvDebugCheck(equal(d1, radius - radiusEpsilon));
            nvDebugCheck(equal(d2, radius - radiusEpsilon));
            nvDebugCheck(equal(d3, radius - radiusEpsilon));
        }
        else {
            // Compute all possible spheres, invalidate those that do not contain the four points, keep the smallest.
            Sphere s0(p1, p2, p3);
            float d0 = distanceSquared(s0, p0);
            if (d0 > 0) s0.radius = NV_FLOAT_MAX;

            Sphere s1(p0, p2, p3);
            float d1 = distanceSquared(s1, p1);
            if (d1 > 0) s1.radius = NV_FLOAT_MAX;

            Sphere s2(p0, p1, p3);
            float d2 = distanceSquared(s2, p2);
            if (d2 > 0) s2.radius = NV_FLOAT_MAX;

            Sphere s3(p0, p1, p2);
            float d3 = distanceSquared(s3, p3);
            if (d3 > 0) s2.radius = NV_FLOAT_MAX;

            if (s0.radius < s1.radius && s0.radius < s2.radius && s0.radius < s3.radius) {
                center = s0.center;
                radius = s0.radius;
            }
            else if (s1.radius < s2.radius && s1.radius < s3.radius) {
                center = s1.center;
                radius = s1.radius;
            }
            else if (s1.radius < s3.radius) {
                center = s2.center;
                radius = s2.radius;
            }
            else {
                center = s3.center;
                radius = s3.radius;
            }
        }
    }
}


float nv::distanceSquared(const Sphere & sphere, const Vector3 & point)
{
    return lengthSquared(sphere.center - point) - square(sphere.radius);
}



// Implementation of "MiniBall" based on:
// http://www.flipcode.com/archives/Smallest_Enclosing_Spheres.shtml

static Sphere recurseMini(const Vector3 *P[], uint p, uint b = 0)
{
	Sphere MB;

	switch(b)
	{
	case 0:
		MB = Sphere(*P[0]);
		break;
	case 1:
		MB = Sphere(*P[-1]);
		break;
	case 2:
		MB = Sphere(*P[-1], *P[-2]);
		break;
	case 3:
		MB = Sphere(*P[-1], *P[-2], *P[-3]);
		break;
	case 4:
		MB = Sphere(*P[-1], *P[-2], *P[-3], *P[-4]);
		return MB;
	}

	for (uint i = 0; i < p; i++)
    {
        if (distanceSquared(MB, *P[i]) > 0)   // Signed square distance to sphere
		{
			for (uint j = i; j > 0; j--)
			{
                swap(P[j], P[j-1]);
			}

			MB = recurseMini(P + 1, i, b + 1);
		}
    }

	return MB;
}

static bool allInside(const Sphere & sphere, const Vector3 * pointArray, const uint pointCount) {
    for (uint i = 0; i < pointCount; i++) {
        if (distanceSquared(sphere, pointArray[i]) >= NV_EPSILON) {
            return false;
        }
    }
    return true;
}


Sphere nv::miniBall(const Vector3 * pointArray, const uint pointCount)
{
    nvDebugCheck(pointArray != NULL);
    nvDebugCheck(pointCount > 0);

	const Vector3 **L = new const Vector3*[pointCount];

    for (uint i = 0; i < pointCount; i++) {
		L[i] = &pointArray[i];
    }

	Sphere sphere = recurseMini(L, pointCount);

	delete [] L;

    nvDebugCheck(allInside(sphere, pointArray, pointCount));

	return sphere;
}


// Approximate bounding sphere, based on "An Efficient Bounding Sphere" by Jack Ritter, from "Graphics Gems"
Sphere nv::approximateSphere_Ritter(const Vector3 * pointArray, const uint pointCount)
{
    nvDebugCheck(pointArray != NULL);
    nvDebugCheck(pointCount > 0);

    Vector3 xmin, xmax, ymin, ymax, zmin, zmax;

    xmin = xmax = ymin = ymax = zmin = zmax = pointArray[0];

    // FIRST PASS: find 6 minima/maxima points
    xmin.x = ymin.y = zmin.z = FLT_MAX;
    xmax.x = ymax.y = zmax.z = -FLT_MAX;

    for (uint i = 0; i < pointCount; i++)
	{
        const Vector3 & p = pointArray[i];
        if (p.x < xmin.x) xmin = p;
	    if (p.x > xmax.x) xmax = p;
	    if (p.y < ymin.y) ymin = p;
	    if (p.y > ymax.y) ymax = p;
	    if (p.z < zmin.z) zmin = p;
	    if (p.z > zmax.z) zmax = p;
	}

    float xspan = lengthSquared(xmax - xmin);
    float yspan = lengthSquared(ymax - ymin);
    float zspan = lengthSquared(zmax - zmin);

    // Set points dia1 & dia2 to the maximally separated pair.
    Vector3 dia1 = xmin; 
    Vector3 dia2 = xmax;
    float maxspan = xspan;
    if (yspan > maxspan) {
	    maxspan = yspan;
	    dia1 = ymin;
        dia2 = ymax;
	}
    if (zspan > maxspan) {
	    dia1 = zmin;
        dia2 = zmax;
	}

    // |dia1-dia2| is a diameter of initial sphere
    
    // calc initial center
    Sphere sphere;
    sphere.center = (dia1 + dia2) / 2.0f;

    // calculate initial radius**2 and radius
    float rad_sq = lengthSquared(dia2 - sphere.center);
    sphere.radius = sqrtf(rad_sq);


    // SECOND PASS: increment current sphere
    for (uint i = 0; i < pointCount; i++)
	{
        const Vector3 & p = pointArray[i];

        float old_to_p_sq = lengthSquared(p - sphere.center);

	    if (old_to_p_sq > rad_sq) 	// do r**2 test first
		{ 	
            // this point is outside of current sphere
		    float old_to_p = sqrtf(old_to_p_sq);

		    // calc radius of new sphere
            sphere.radius = (sphere.radius + old_to_p) / 2.0f;
		    rad_sq = sphere.radius * sphere.radius; 	// for next r**2 compare
    		
            float old_to_new = old_to_p - sphere.radius;

		    // calc center of new sphere
            sphere.center = (sphere.radius * sphere.center + old_to_new * p) / old_to_p;
		}	
	}

    nvDebugCheck(allInside(sphere, pointArray, pointCount));

    return sphere;
}


static float computeSphereRadius(const Vector3 & center, const Vector3 * pointArray, const uint pointCount) {

    float maxRadius2 = 0;

    for (uint i = 0; i < pointCount; i++)
	{
        const Vector3 & p = pointArray[i];

        float r2 = lengthSquared(center - p);

        if (r2 > maxRadius2) {
            maxRadius2 = r2;
        }
    }

    return sqrtf(maxRadius2) + radiusEpsilon;
}


Sphere nv::approximateSphere_AABB(const Vector3 * pointArray, const uint pointCount)
{
    nvDebugCheck(pointArray != NULL);
    nvDebugCheck(pointCount > 0);

    Box box;
    box.clearBounds();

    for (uint i = 0; i < pointCount; i++) {
        box.addPointToBounds(pointArray[i]);
    }

    Sphere sphere;
    sphere.center = box.center();
    sphere.radius = computeSphereRadius(sphere.center, pointArray, pointCount);

    nvDebugCheck(allInside(sphere, pointArray, pointCount));

    return sphere;
}


static void computeExtremalPoints(const Vector3 & dir, const Vector3 * pointArray, uint pointCount, Vector3 * minPoint, Vector3 * maxPoint) {
    nvDebugCheck(pointCount > 0);

    uint mini = 0;
    uint maxi = 0;
    float minDist = FLT_MAX;
    float maxDist = -FLT_MAX;

    for (uint i = 0; i < pointCount; i++) {
        float d = dot(dir, pointArray[i]);

        if (d < minDist) {
            minDist = d;
            mini = i;
        }
        if (d > maxDist) {
            maxDist = d;
            maxi = i;
        }
    }
    nvDebugCheck(minDist != FLT_MAX);
    nvDebugCheck(maxDist != -FLT_MAX);

    *minPoint = pointArray[mini];
    *maxPoint = pointArray[maxi];
}

// EPOS algorithm based on:
// http://www.ep.liu.se/ecp/034/009/ecp083409.pdf
Sphere nv::approximateSphere_EPOS6(const Vector3 * pointArray, uint pointCount)
{
    nvDebugCheck(pointArray != NULL);
    nvDebugCheck(pointCount > 0);

    Vector3 extremalPoints[6];

    // Compute 6 extremal points.
    computeExtremalPoints(Vector3(1, 0, 0), pointArray, pointCount, extremalPoints+0, extremalPoints+1);
    computeExtremalPoints(Vector3(0, 1, 0), pointArray, pointCount, extremalPoints+2, extremalPoints+3);
    computeExtremalPoints(Vector3(0, 0, 1), pointArray, pointCount, extremalPoints+4, extremalPoints+5);

    Sphere sphere = miniBall(extremalPoints, 6);
    sphere.radius = computeSphereRadius(sphere.center, pointArray, pointCount);

    nvDebugCheck(allInside(sphere, pointArray, pointCount));

    return sphere;
}

Sphere nv::approximateSphere_EPOS14(const Vector3 * pointArray, uint pointCount)
{
    nvDebugCheck(pointArray != NULL);
    nvDebugCheck(pointCount > 0);

    Vector3 extremalPoints[14];

    // Compute 14 extremal points.
    computeExtremalPoints(Vector3(1, 0, 0), pointArray, pointCount, extremalPoints+0, extremalPoints+1);
    computeExtremalPoints(Vector3(0, 1, 0), pointArray, pointCount, extremalPoints+2, extremalPoints+3);
    computeExtremalPoints(Vector3(0, 0, 1), pointArray, pointCount, extremalPoints+4, extremalPoints+5);

    float d = sqrtf(1.0f/3.0f);

    computeExtremalPoints(Vector3(d, d, d), pointArray, pointCount, extremalPoints+6, extremalPoints+7);
    computeExtremalPoints(Vector3(-d, d, d), pointArray, pointCount, extremalPoints+8, extremalPoints+9);
    computeExtremalPoints(Vector3(-d, -d, d), pointArray, pointCount, extremalPoints+10, extremalPoints+11);
    computeExtremalPoints(Vector3(d, -d, d), pointArray, pointCount, extremalPoints+12, extremalPoints+13);


    Sphere sphere = miniBall(extremalPoints, 14);
    sphere.radius = computeSphereRadius(sphere.center, pointArray, pointCount);

    nvDebugCheck(allInside(sphere, pointArray, pointCount));

    return sphere;
}



