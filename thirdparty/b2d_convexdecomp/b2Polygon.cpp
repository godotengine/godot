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

// This utility works with Box2d version 2.0 (or higher), and not with 1.4.3

#include "b2Triangle.h"
#include "b2Polygon.h"

#include <math.h>
#include <limits.h>
#include <assert.h>
#define b2Assert assert

namespace b2ConvexDecomp {


//If you're using 1.4.3, b2_toiSlop won't exist, so set this equal to 0
static const float32 toiSlop = 0.0f;

/*
 * Check if the lines a0->a1 and b0->b1 cross.
 * If they do, intersectionPoint will be filled
 * with the point of crossing.
 *
 * Grazing lines should not return true.
 */
bool intersect(const b2Vec2& a0, const b2Vec2& a1,
			   const b2Vec2& b0, const b2Vec2& b1, 
			   b2Vec2& intersectionPoint) {

	if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) return false;
	float x1 = a0.x; float y1 = a0.y;
	float x2 = a1.x; float y2 = a1.y;
	float x3 = b0.x; float y3 = b0.y;
	float x4 = b1.x; float y4 = b1.y;
	
	//AABB early exit
	if (b2Max(x1,x2) < b2Min(x3,x4) || b2Max(x3,x4) < b2Min(x1,x2) ) return false;
	if (b2Max(y1,y2) < b2Min(y3,y4) || b2Max(y3,y4) < b2Min(y1,y2) ) return false;
	
	float ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3));
	float ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3));
	float denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
	if (b2Abs(denom) < CMP_EPSILON) {
		//Lines are too close to parallel to call
		return false;
	}
	ua /= denom;
	ub /= denom;
	
	if ((0 < ua) && (ua < 1) && (0 < ub) && (ub < 1)) {
		//if (intersectionPoint){
			intersectionPoint.x = (x1 + ua * (x2 - x1));
			intersectionPoint.y = (y1 + ua * (y2 - y1));
		//}
		//printf("%f, %f -> %f, %f crosses %f, %f -> %f, %f\n",x1,y1,x2,y2,x3,y3,x4,y4);
		return true;
	}
	
	return false;
}

/*
 * True if line from a0->a1 intersects b0->b1
 */
bool intersect(const b2Vec2& a0, const b2Vec2& a1,
			   const b2Vec2& b0, const b2Vec2& b1) {
	b2Vec2 myVec(0.0f,0.0f);
	return intersect(a0, a1, b0, b1, myVec);
}

b2Polygon::b2Polygon(float32* _x, float32* _y, int32 nVert) {
        nVertices = nVert;
        x = new float32[nVertices];
        y = new float32[nVertices];
        for (int32 i = 0; i < nVertices; ++i) {
            x[i] = _x[i];
            y[i] = _y[i];
        }
		areaIsSet = false;
}
	
b2Polygon::b2Polygon(b2Vec2* v, int32 nVert) {
        nVertices = nVert;
        x = new float32[nVertices];
        y = new float32[nVertices];
        for (int32 i = 0; i < nVertices; ++i) {
            x[i] = v[i].x;
            y[i] = v[i].y;

        }
		areaIsSet = false;
}

b2Polygon::b2Polygon() {
	x = NULL;
	y = NULL;
	nVertices = 0;
	areaIsSet = false;
}
	
b2Polygon::~b2Polygon() {
	//printf("About to delete poly with %d vertices\n",nVertices);
	delete[] x;
	delete[] y;
}

float32 b2Polygon::GetArea() {
	// TODO: fix up the areaIsSet caching so that it can be used
	//if (areaIsSet) return area;
	area = 0.0f;
	
	//First do wraparound
	area += x[nVertices-1]*y[0]-x[0]*y[nVertices-1];
	for (int i=0; i<nVertices-1; ++i){
		area += x[i]*y[i+1]-x[i+1]*y[i];
	}
	area *= .5f;
	areaIsSet = true;
	return area;
}

bool b2Polygon::IsCCW() {
	return (GetArea() > 0.0f);
}
	
void b2Polygon::MergeParallelEdges(float32 tolerance) {
	if (nVertices <= 3) return; //Can't do anything useful here to a triangle
	bool* mergeMe = new bool[nVertices];
	int32 newNVertices = nVertices;
	for (int32 i = 0; i < nVertices; ++i) {
		int32 lower = (i == 0) ? (nVertices - 1) : (i - 1);
		int32 middle = i;
		int32 upper = (i == nVertices - 1) ? (0) : (i + 1);
		float32 dx0 = x[middle] - x[lower];
		float32 dy0 = y[middle] - y[lower];
		float32 dx1 = x[upper] - x[middle];
		float32 dy1 = y[upper] - y[middle];
		float32 norm0 = sqrtf(dx0*dx0+dy0*dy0);
		float32 norm1 = sqrtf(dx1*dx1+dy1*dy1);
		if ( !(norm0 > 0.0f && norm1 > 0.0f) && newNVertices > 3 ) {
			//Merge identical points
			mergeMe[i] = true;
			--newNVertices;
		}
		dx0 /= norm0; dy0 /= norm0;
		dx1 /= norm1; dy1 /= norm1;
		float32 cross = dx0 * dy1 - dx1 * dy0;
		float32 dot = dx0 * dx1 + dy0 * dy1;
		if (fabs(cross) < tolerance && dot > 0 && newNVertices > 3) {
			mergeMe[i] = true;
			--newNVertices;
		} else {
			mergeMe[i] = false;
		}
	}
	if(newNVertices == nVertices || newNVertices == 0) {
		delete[] mergeMe;
		return;
	}
	float32* newx = new float32[newNVertices];
	float32* newy = new float32[newNVertices];
	int32 currIndex = 0;
	for (int32 i=0; i < nVertices; ++i) {
		if (mergeMe[i] || newNVertices == 0 || currIndex == newNVertices) continue;
		b2Assert(currIndex < newNVertices);
		newx[currIndex] = x[i];
		newy[currIndex] = y[i];
		++currIndex;
	}
	delete[] x;
	delete[] y;
	delete[] mergeMe;
	x = newx;
	y = newy;
	nVertices = newNVertices;
	//printf("%d \n", newNVertices);
}
	
    /* 
	 *	Allocates and returns pointer to vector vertex array.
     *  Length of array is nVertices.
	 */
b2Vec2* b2Polygon::GetVertexVecs() {
        b2Vec2* out = new b2Vec2[nVertices];
        for (int32 i = 0; i < nVertices; ++i) {
            out[i].Set(x[i], y[i]);
        }
        return out;
}
	
b2Polygon::b2Polygon(b2Triangle& t) {
	nVertices = 3;
	x = new float[nVertices];
	y = new float[nVertices];
	for (int32 i = 0; i < nVertices; ++i) {
		x[i] = t.x[i];
		y[i] = t.y[i];
	}
}
	
void b2Polygon::Set(const b2Polygon& p) {
        if (nVertices != p.nVertices){
			nVertices = p.nVertices;
			delete[] x;
			delete[] y;
			x = new float32[nVertices];
			y = new float32[nVertices];
        }
		
        for (int32 i = 0; i < nVertices; ++i) {
            x[i] = p.x[i];
            y[i] = p.y[i];
        }
	areaIsSet = false;
}
	
    /*
     * Assuming the polygon is simple, checks if it is convex.
     */
bool b2Polygon::IsConvex() {
        bool isPositive = false;
        for (int32 i = 0; i < nVertices; ++i) {
            int32 lower = (i == 0) ? (nVertices - 1) : (i - 1);
            int32 middle = i;
            int32 upper = (i == nVertices - 1) ? (0) : (i + 1);
            float32 dx0 = x[middle] - x[lower];
            float32 dy0 = y[middle] - y[lower];
            float32 dx1 = x[upper] - x[middle];
            float32 dy1 = y[upper] - y[middle];
            float32 cross = dx0 * dy1 - dx1 * dy0;
            // Cross product should have same sign
            // for each vertex if poly is convex.
            bool newIsP = (cross >= 0) ? true : false;
            if (i == 0) {
                isPositive = newIsP;
            }
            else if (isPositive != newIsP) {
                return false;
            }
        }
        return true;
}

/*
 * Pulled from b2Shape.cpp, assertions removed
 */
static b2Vec2 PolyCentroid(const b2Vec2* vs, int32 count)
{
	b2Vec2 c; c.Set(0.0f, 0.0f);
	float32 area = 0.0f;

	const float32 inv3 = 1.0f / 3.0f;
	b2Vec2 pRef(0.0f, 0.0f);
	for (int32 i = 0; i < count; ++i)
	{
		// Triangle vertices.
		b2Vec2 p1 = pRef;
		b2Vec2 p2 = vs[i];
		b2Vec2 p3 = i + 1 < count ? vs[i+1] : vs[0];

		b2Vec2 e1 = p2 - p1;
		b2Vec2 e2 = p3 - p1;

		float32 D = b2Cross(e1, e2);

		float32 triangleArea = 0.5f * D;
		area += triangleArea;

		// Area weighted centroid
		c += (p1 + p2 + p3) * triangleArea * inv3;
	}

	// Centroid
	c *= 1.0f / area;
	return c;
}


/*
 * Checks if polygon is valid for use in Box2d engine.
 * Last ditch effort to ensure no invalid polygons are
 * added to world geometry.
 *
 * Performs a full check, for simplicity, convexity,
 * orientation, minimum angle, and volume.  This won't
 * be very efficient, and a lot of it is redundant when
 * other tools in this section are used.
 */
bool b2Polygon::IsUsable(bool printErrors){
	int32 error = -1;
	bool noError = true;
	if (nVertices < 3 || nVertices > b2_maxPolygonVertices) {noError = false; error = 0;}
	if (!IsConvex()) {noError = false; error = 1;}
	if (!IsSimple()) {noError = false; error = 2;}
	if (GetArea() < CMP_EPSILON) {noError = false; error = 3;}

	//Compute normals
	b2Vec2* normals = new b2Vec2[nVertices];
	b2Vec2* vertices = new b2Vec2[nVertices];
	for (int32 i = 0; i < nVertices; ++i){
		vertices[i].Set(x[i],y[i]);
		int32 i1 = i;
		int32 i2 = i + 1 < nVertices ? i + 1 : 0;
		b2Vec2 edge(x[i2]-x[i1],y[i2]-y[i1]);
		normals[i] = b2Cross(edge, 1.0f);
		normals[i].Normalize();
	}

	//Required side checks
	for (int32 i=0; i<nVertices; ++i){
		int32 iminus = (i==0)?nVertices-1:i-1;
		//int32 iplus = (i==nVertices-1)?0:i+1;

		//Parallel sides check
		float32 cross = b2Cross(normals[iminus], normals[i]);
		cross = b2Clamp(cross, -1.0f, 1.0f);
		float32 angle = asinf(cross);
		if(angle <= b2_angularSlop){
			noError = false;
			error = 4;
			break;
		}

		//Too skinny check
		for (int32 j=0; j<nVertices; ++j){
			if (j == i || j == (i + 1) % nVertices){
				continue;
			}
			float32 s = b2Dot(normals[i], vertices[j] - vertices[i]);
			if (s >= -b2_linearSlop){
				noError = false;
				error = 5;
			}
		}


		b2Vec2 centroid = PolyCentroid(vertices,nVertices);
		b2Vec2 n1 = normals[iminus];
		b2Vec2 n2 = normals[i];
		b2Vec2 v = vertices[i] - centroid;

		b2Vec2 d;
		d.x = b2Dot(n1, v) - toiSlop;
		d.y = b2Dot(n2, v) - toiSlop;

		// Shifting the edge inward by b2_toiSlop should
		// not cause the plane to pass the centroid.
		if ((d.x < 0.0f)||(d.y < 0.0f)){
			noError = false;
			error = 6;
		}

	}
	delete[] vertices;
	delete[] normals;

	if (!noError && printErrors){
		printf("Found invalid polygon, ");
		switch(error){
			case 0:
				printf("must have between 3 and %d vertices.\n",b2_maxPolygonVertices);
				break;
			case 1:
				printf("must be convex.\n");
				break;
			case 2:
				printf("must be simple (cannot intersect itself).\n");
				break;
			case 3:
				printf("area is too small.\n");
				break;
			case 4:
				printf("sides are too close to parallel.\n");
				break;
			case 5:
				printf("polygon is too thin.\n");
				break;
			case 6:
				printf("core shape generation would move edge past centroid (too thin).\n");
				break;
			default:
				printf("don't know why.\n");
		}
	}
	return noError;
}


bool b2Polygon::IsUsable(){
	return IsUsable(B2_POLYGON_REPORT_ERRORS);
}

//Check for edge crossings
bool b2Polygon::IsSimple() {
	for (int32 i=0; i<nVertices; ++i){
		int32 iplus = (i+1 > nVertices-1)?0:i+1;
		b2Vec2 a1(x[i],y[i]);
		b2Vec2 a2(x[iplus],y[iplus]);
		for (int32 j=i+1; j<nVertices; ++j){
			int32 jplus = (j+1 > nVertices-1)?0:j+1;
			b2Vec2 b1(x[j],y[j]);
			b2Vec2 b2(x[jplus],y[jplus]);
			if (intersect(a1,a2,b1,b2)){
				return false;
			}
		}
	}
	return true;
}
    
    /*
     * Tries to add a triangle to the polygon. Returns null if it can't connect
     * properly, otherwise returns a pointer to the new Polygon. Assumes bitwise
     * equality of joined vertex positions.
	 *
	 * Remember to delete the pointer afterwards.
	 * Todo: Make this return a b2Polygon instead
	 * of a pointer to a heap-allocated one.
	 *
	 * For internal use.
     */
b2Polygon* b2Polygon::Add(b2Triangle& t) {
        // First, find vertices that connect
        int32 firstP = -1;
        int32 firstT = -1;
        int32 secondP = -1;
        int32 secondT = -1;
        for (int32 i = 0; i < nVertices; i++) {
            if (t.x[0] == x[i] && t.y[0] == y[i]) {
                if (firstP == -1) {
                    firstP = i;
                    firstT = 0;
                }
                else {
                    secondP = i;
                    secondT = 0;
                }
            }
            else if (t.x[1] == x[i] && t.y[1] == y[i]) {
                if (firstP == -1) {
                    firstP = i;
                    firstT = 1;
                }
                else {
                    secondP = i;
                    secondT = 1;
                }
            }
            else if (t.x[2] == x[i] && t.y[2] == y[i]) {
                if (firstP == -1) {
                    firstP = i;
                    firstT = 2;
                }
                else {
                    secondP = i;
                    secondT = 2;
                }
            }
            else {
            }
        }
        // Fix ordering if first should be last vertex of poly
        if (firstP == 0 && secondP == nVertices - 1) {
            firstP = nVertices - 1;
            secondP = 0;
        }
		
        // Didn't find it
        if (secondP == -1) {
		    return NULL;
		}
		
        // Find tip index on triangle
        int32 tipT = 0;
        if (tipT == firstT || tipT == secondT)
            tipT = 1;
        if (tipT == firstT || tipT == secondT)
            tipT = 2;
		
        float32* newx = new float[nVertices + 1];
        float32* newy = new float[nVertices + 1];
        int32 currOut = 0;
        for (int32 i = 0; i < nVertices; i++) {
            newx[currOut] = x[i];
            newy[currOut] = y[i];
            if (i == firstP) {
                ++currOut;
                newx[currOut] = t.x[tipT];
                newy[currOut] = t.y[tipT];
            }
            ++currOut;
        }
        b2Polygon* result = new b2Polygon(newx, newy, nVertices+1);
        delete[] newx;
        delete[] newy;
        return result;
}
	
    /**
     * Adds this polygon to a PolyDef.
     */
#if 0
void b2Polygon::AddTo(b2FixtureDef& pd) {
	if (nVertices < 3) return;
	
	b2Assert(nVertices <= b2_maxPolygonVertices);
	
	b2Vec2* vecs = GetVertexVecs();
	b2Vec2* vecsToAdd = new b2Vec2[nVertices];

	int32 offset = 0;
	
	b2PolygonShape *polyShape = new b2PolygonShape;
	int32 ind;
	
    for (int32 i = 0; i < nVertices; ++i) {
		
		//Omit identical neighbors (including wraparound)
		ind = i - offset;
		if (vecs[i].x==vecs[remainder(i+1,nVertices)].x &&
			vecs[i].y==vecs[remainder(i+1,nVertices)].y){
				offset++;
				continue;
		}
		
		vecsToAdd[ind] = vecs[i];
		
    }
	
	polyShape->Set((const b2Vec2*)vecsToAdd, ind+1);
	pd.shape = polyShape;
	
    delete[] vecs;
	delete[] vecsToAdd;
}
#endif
	/**
	 * Finds and fixes "pinch points," points where two polygon
	 * vertices are at the same point.
	 *
	 * If a pinch point is found, pin is broken up into poutA and poutB
	 * and true is returned; otherwise, returns false.
	 *
	 * Mostly for internal use.
	 */
bool ResolvePinchPoint(const b2Polygon& pin, b2Polygon& poutA, b2Polygon& poutB){
	if (pin.nVertices < 3) return false;
	float32 tol = .001f;
	bool hasPinchPoint = false;
	int32 pinchIndexA = -1;
	int32 pinchIndexB = -1;
	for (int i=0; i<pin.nVertices; ++i){
		for (int j=i+1; j<pin.nVertices; ++j){
			//Don't worry about pinch points where the points
			//are actually just dupe neighbors
			if (b2Abs(pin.x[i]-pin.x[j])<tol&&b2Abs(pin.y[i]-pin.y[j])<tol&&j!=i+1){
				pinchIndexA = i;
				pinchIndexB = j;
				//printf("pinch: %f, %f == %f, %f\n",pin.x[i],pin.y[i],pin.x[j],pin.y[j]);
				//printf("at indexes %d, %d\n",i,j);
				hasPinchPoint = true;
				break;
			}
		}
		if (hasPinchPoint) break;
	}
	if (hasPinchPoint){
		//printf("Found pinch point\n");
		int32 sizeA = pinchIndexB - pinchIndexA;
		if (sizeA == pin.nVertices) return false;//has dupe points at wraparound, not a problem here
		float32* xA = new float32[sizeA];
		float32* yA = new float32[sizeA];
		for (int32 i=0; i < sizeA; ++i){
			int32 ind = remainder(pinchIndexA+i,pin.nVertices);
			xA[i] = pin.x[ind];
			yA[i] = pin.y[ind];
		}
		b2Polygon tempA(xA,yA,sizeA);
		poutA.Set(tempA);
		delete[] xA;
		delete[] yA;
		
		int32 sizeB = pin.nVertices - sizeA;
		float32* xB = new float32[sizeB];
		float32* yB = new float32[sizeB];
		for (int32 i=0; i<sizeB; ++i){
			int32 ind = remainder(pinchIndexB+i,pin.nVertices);
			xB[i] = pin.x[ind];
			yB[i] = pin.y[ind];
		}
		b2Polygon tempB(xB,yB,sizeB);
		poutB.Set(tempB);
		//printf("Size of a: %d, size of b: %d\n",sizeA,sizeB);
		delete[] xB;
		delete[] yB;
	}
	return hasPinchPoint;
}

    /**
     * Triangulates a polygon using simple ear-clipping algorithm. Returns
     * size of Triangle array unless the polygon can't be triangulated.
     * This should only happen if the polygon self-intersects,
     * though it will not _always_ return null for a bad polygon - it is the
     * caller's responsibility to check for self-intersection, and if it
     * doesn't, it should at least check that the return value is non-null
     * before using. You're warned!
	 *
	 * Triangles may be degenerate, especially if you have identical points
	 * in the input to the algorithm.  Check this before you use them.
     *
     * This is totally unoptimized, so for large polygons it should not be part
     * of the simulation loop.
     *
     * Returns:
     * -1 if algorithm fails (self-intersection most likely)
     * 0 if there are not enough vertices to triangulate anything.
     * Number of triangles if triangulation was successful.
     *
     * results will be filled with results - ear clipping always creates vNum - 2
     * or fewer (due to pinch point polygon snipping), so allocate an array of
	 * this size.
     */
	
int32 TriangulatePolygon(float32* xv, float32* yv, int32 vNum, b2Triangle* results) {
        if (vNum < 3)
            return 0;

		//Recurse and split on pinch points
		b2Polygon pA,pB;
		b2Polygon pin(xv,yv,vNum);
		if (ResolvePinchPoint(pin,pA,pB)){
			b2Triangle* mergeA = new b2Triangle[pA.nVertices];
			b2Triangle* mergeB = new b2Triangle[pB.nVertices];
			int32 nA = TriangulatePolygon(pA.x,pA.y,pA.nVertices,mergeA);
			int32 nB = TriangulatePolygon(pB.x,pB.y,pB.nVertices,mergeB);
			if (nA==-1 || nB==-1){
				delete[] mergeA;
				delete[] mergeB;
				return -1;
			}
			for (int32 i=0; i<nA; ++i){
				results[i].Set(mergeA[i]);
			}
			for (int32 i=0; i<nB; ++i){
				results[nA+i].Set(mergeB[i]);
			}
			delete[] mergeA;
			delete[] mergeB;
			return (nA+nB);
		}

        b2Triangle* buffer = new b2Triangle[vNum-2];
        int32 bufferSize = 0;
        float32* xrem = new float32[vNum];
        float32* yrem = new float32[vNum];
        for (int32 i = 0; i < vNum; ++i) {
            xrem[i] = xv[i];
            yrem[i] = yv[i];
        }
		
		int xremLength = vNum;
		
        while (vNum > 3) {
            // Find an ear
            int32 earIndex = -1;
			//float32 earVolume = -1.0f;
			float32 earMaxMinCross = -10.0f;
            for (int32 i = 0; i < vNum; ++i) {
                if (IsEar(i, xrem, yrem, vNum)) {
					int32 lower = remainder(i-1,vNum);
					int32 upper = remainder(i+1,vNum);
					b2Vec2 d1(xrem[upper]-xrem[i],yrem[upper]-yrem[i]);
					b2Vec2 d2(xrem[i]-xrem[lower],yrem[i]-yrem[lower]);
					b2Vec2 d3(xrem[lower]-xrem[upper],yrem[lower]-yrem[upper]);

					d1.Normalize();
					d2.Normalize();
					d3.Normalize();
					float32 cross12 = b2Abs( b2Cross(d1,d2) );
					float32 cross23 = b2Abs( b2Cross(d2,d3) );
					float32 cross31 = b2Abs( b2Cross(d3,d1) );
					//Find the maximum minimum angle
					float32 minCross = b2Min(cross12, b2Min(cross23,cross31));
					if (minCross > earMaxMinCross){
						earIndex = i;
						earMaxMinCross = minCross;
					}

					/*//This bit chooses the ear with greatest volume first
					float32 testVol = b2Abs( d1.x*d2.y-d2.x*d1.y );
					if (testVol > earVolume){
						earIndex = i;
						earVolume = testVol;
					}*/
                }
            }
			
            // If we still haven't found an ear, we're screwed.
            // Note: sometimes this is happening because the
			// remaining points are collinear.  Really these
			// should just be thrown out without halting triangulation.
			if (earIndex == -1){
				if (B2_POLYGON_REPORT_ERRORS){
					b2Polygon dump(xrem,yrem,vNum);
					printf("Couldn't find an ear, dumping remaining poly:\n");
					dump.printFormatted();
					printf("Please submit this dump to ewjordan at Box2d forums\n");
				}
				for (int32 i = 0; i < bufferSize; i++) {
					results[i].Set(buffer[i]);
				}
		
				delete[] buffer;
		
				if (bufferSize > 0) return bufferSize;
                else return -1;
			}
			
            // Clip off the ear:
            // - remove the ear tip from the list

            --vNum;
            float32* newx = new float32[vNum];
            float32* newy = new float32[vNum];
            int32 currDest = 0;
            for (int32 i = 0; i < vNum; ++i) {
                if (currDest == earIndex) ++currDest;
                newx[i] = xrem[currDest];
                newy[i] = yrem[currDest];
                ++currDest;
            }
			
            // - add the clipped triangle to the triangle list
            int32 under = (earIndex == 0) ? (vNum) : (earIndex - 1);
            int32 over = (earIndex == vNum) ? 0 : (earIndex + 1);
            b2Triangle toAdd = b2Triangle(xrem[earIndex], yrem[earIndex], xrem[over], yrem[over], xrem[under], yrem[under]);
            buffer[bufferSize].Set(toAdd);
            ++bufferSize;
			
            // - replace the old list with the new one
            delete[] xrem;
            delete[] yrem;
            xrem = newx;
            yrem = newy;
        }
		
        b2Triangle toAdd = b2Triangle(xrem[1], yrem[1], xrem[2], yrem[2],
								  xrem[0], yrem[0]);
        buffer[bufferSize].Set(toAdd);
        ++bufferSize;
		
        delete[] xrem;
        delete[] yrem;
		
        b2Assert(bufferSize == xremLength-2);
		
        for (int32 i = 0; i < bufferSize; i++) {
            results[i].Set(buffer[i]);
        }
		
        delete[] buffer;
		
        return bufferSize;
}

    /**
	 * Turns a list of triangles into a list of convex polygons. Very simple
     * method - start with a seed triangle, keep adding triangles to it until
     * you can't add any more without making the polygon non-convex.
     *
     * Returns an integer telling how many polygons were created.  Will fill
     * polys array up to polysLength entries, which may be smaller or larger
     * than the return value.
     * 
     * Takes O(N*P) where P is the number of resultant polygons, N is triangle
     * count.
     * 
     * The final polygon list will not necessarily be minimal, though in
     * practice it works fairly well.
     */
int32 PolygonizeTriangles(b2Triangle* triangulated, int32 triangulatedLength, b2Polygon* polys, int32 polysLength) {
        int32 polyIndex = 0;
		
        if (triangulatedLength <= 0) {
            return 0;
        }
        else {
            int* covered = new int[triangulatedLength];
            for (int32 i = 0; i < triangulatedLength; ++i) {
				covered[i] = 0;
				//Check here for degenerate triangles
				if ( ( (triangulated[i].x[0] == triangulated[i].x[1]) && (triangulated[i].y[0] == triangulated[i].y[1]) )
					 || ( (triangulated[i].x[1] == triangulated[i].x[2]) && (triangulated[i].y[1] == triangulated[i].y[2]) )
					 || ( (triangulated[i].x[0] == triangulated[i].x[2]) && (triangulated[i].y[0] == triangulated[i].y[2]) ) ) {
					covered[i] = 1;
				}
            }
			
            bool notDone = true;
            while (notDone) {
                int32 currTri = -1;
                for (int32 i = 0; i < triangulatedLength; ++i) {
                    if (covered[i])
                        continue;
                    currTri = i;
                    break;
                }
                if (currTri == -1) {
                    notDone = false;
                }
                else {
                    b2Polygon poly(triangulated[currTri]);
					covered[currTri] = 1;
					int32 index = 0;
                    for (int32 i = 0; i < 2*triangulatedLength; ++i,++index) {
						while (index >= triangulatedLength) index -= triangulatedLength;
                        if (covered[index]) {
                            continue;
						}
                        b2Polygon* newP = poly.Add(triangulated[index]);
                        if (!newP) {
                            continue;
						}
						if (newP->nVertices > b2Polygon::maxVerticesPerPolygon) {
							delete newP;
							newP = NULL;
                            continue;
						}
                        if (newP->IsConvex()) { //Or should it be IsUsable?  Maybe re-write IsConvex to apply the angle threshold from Box2d
                            poly.Set(*newP);
							delete newP;
							newP = NULL;
                            covered[index] = 1;
                        } else {
							delete newP;
							newP = NULL;
						}
                    }
                    if (polyIndex < polysLength){
						poly.MergeParallelEdges(b2_angularSlop);
						//If identical points are present, a triangle gets
						//borked by the MergeParallelEdges function, hence
						//the vertex number check
						if (poly.nVertices >= 3) polys[polyIndex].Set(poly);
						//else printf("Skipping corrupt poly\n");
					}
                    if (poly.nVertices >= 3) polyIndex++; //Must be outside (polyIndex < polysLength) test
                }
					//printf("MEMCHECK: %d\n",_CrtCheckMemory());
            }
            delete[] covered;
        }
        return polyIndex;
}
	
    /**
	 * Checks if vertex i is the tip of an ear in polygon defined by xv[] and
     * yv[].
	 *
	 * Assumes clockwise orientation of polygon...ick
     */
bool IsEar(int32 i, float32* xv, float32* yv, int32 xvLength) {
        float32 dx0, dy0, dx1, dy1;
        dx0 = dy0 = dx1 = dy1 = 0;
        if (i >= xvLength || i < 0 || xvLength < 3) {
            return false;
        }
        int32 upper = i + 1;
        int32 lower = i - 1;
        if (i == 0) {
            dx0 = xv[0] - xv[xvLength - 1];
            dy0 = yv[0] - yv[xvLength - 1];
            dx1 = xv[1] - xv[0];
            dy1 = yv[1] - yv[0];
            lower = xvLength - 1;
        }
        else if (i == xvLength - 1) {
            dx0 = xv[i] - xv[i - 1];
            dy0 = yv[i] - yv[i - 1];
            dx1 = xv[0] - xv[i];
            dy1 = yv[0] - yv[i];
            upper = 0;
        }
        else {
            dx0 = xv[i] - xv[i - 1];
            dy0 = yv[i] - yv[i - 1];
            dx1 = xv[i + 1] - xv[i];
            dy1 = yv[i + 1] - yv[i];
        }
        float32 cross = dx0 * dy1 - dx1 * dy0;
        if (cross > 0)
            return false;
        b2Triangle myTri(xv[i], yv[i], xv[upper], yv[upper],
								  xv[lower], yv[lower]);
        for (int32 j = 0; j < xvLength; ++j) {
            if (j == i || j == lower || j == upper)
                continue;
            if (myTri.IsInside(xv[j], yv[j]))
                return false;
        }
        return true;
}

void ReversePolygon(b2Polygon& p){
	ReversePolygon(p.x,p.y,p.nVertices);
}
	
void ReversePolygon(float* x, float* y, int n) {
        if (n == 1)
            return;
        int32 low = 0;
        int32 high = n - 1;
        while (low < high) {
            float32 buffer = x[low];
            x[low] = x[high];
            x[high] = buffer;
            buffer = y[low];
            y[low] = y[high];
            y[high] = buffer;
            ++low;
            --high;
        }
}

    /**
	 * Decomposes a non-convex polygon into a number of convex polygons, up
     * to maxPolys (remaining pieces are thrown out, but the total number
	 * is returned, so the return value can be greater than maxPolys).
     *
     * Each resulting polygon will have no more than maxVerticesPerPolygon
     * vertices (set to b2MaxPolyVertices by default, though you can change
	 * this).
     * 
     * Returns -1 if operation fails (usually due to self-intersection of
	 * polygon).
     */
int32 DecomposeConvex(b2Polygon* p, b2Polygon* results, int32 maxPolys) {
        if (p->nVertices < 3) return 0;

        b2Triangle* triangulated = new b2Triangle[p->nVertices - 2];
		int32 nTri;
        if (p->IsCCW()) {
			//printf("It is ccw \n");
			b2Polygon tempP;
			tempP.Set(*p);
			ReversePolygon(tempP.x, tempP.y, tempP.nVertices);
			nTri = TriangulatePolygon(tempP.x, tempP.y, tempP.nVertices, triangulated);
			//ReversePolygon(p->x, p->y, p->nVertices); //reset orientation
		} else {
			//printf("It is not ccw \n");
			nTri = TriangulatePolygon(p->x, p->y, p->nVertices, triangulated);
		}
		if (nTri < 1) {
            //Still no luck?  Oh well...
            delete[] triangulated;
            return -1;
        }
        int32 nPolys = PolygonizeTriangles(triangulated, nTri, results, maxPolys);
        delete[] triangulated;
        return nPolys;
}

    /**
	 * Decomposes a polygon into convex polygons and adds all pieces to a b2BodyDef
     * using a prototype b2PolyDef. All fields of the prototype are used for every
     * shape except the vertices (friction, restitution, density, etc).
     * 
     * If you want finer control, you'll have to add everything by hand.
     * 
     * This is the simplest method to add a complicated polygon to a body.
	 *
	 * Until Box2D's b2BodyDef behavior changes, this method returns a pointer to
	 * a heap-allocated array of b2PolyDefs, which must be deleted by the user
	 * after the b2BodyDef is added to the world.
     */
#if 0
void DecomposeConvexAndAddTo(b2Polygon* p, b2Body* bd, b2FixtureDef* prototype) {

        if (p->nVertices < 3) return;
        b2Polygon* decomposed = new b2Polygon[p->nVertices - 2]; //maximum number of polys
        int32 nPolys = DecomposeConvex(p, decomposed, p->nVertices - 2);
		//printf("npolys: %d",nPolys);
		b2FixtureDef* pdarray = new b2FixtureDef[2*p->nVertices];//extra space in case of splits
		int32 extra = 0;
        for (int32 i = 0; i < nPolys; ++i) {
            b2FixtureDef* toAdd = &pdarray[i+extra];
			 *toAdd = *prototype;
			 //Hmm, shouldn't have to do all this...
			 /*
			 toAdd->type = prototype->type;
			 toAdd->friction = prototype->friction;
			 toAdd->restitution = prototype->restitution;
			 toAdd->density = prototype->density;
			 toAdd->userData = prototype->userData;
			 toAdd->categoryBits = prototype->categoryBits;
			 toAdd->maskBits = prototype->maskBits;
			 toAdd->groupIndex = prototype->groupIndex;//*/
			 //decomposed[i].print();
			b2Polygon curr = decomposed[i];
			//TODO ewjordan: move this triangle handling to a better place so that
			//it happens even if this convenience function is not called.
			if (curr.nVertices == 3){
					//Check here for near-parallel edges, since we can't
					//handle this in merge routine
					for (int j=0; j<3; ++j){
						int32 lower = (j == 0) ? (curr.nVertices - 1) : (j - 1);
						int32 middle = j;
						int32 upper = (j == curr.nVertices - 1) ? (0) : (j + 1);
						float32 dx0 = curr.x[middle] - curr.x[lower]; float32 dy0 = curr.y[middle] - curr.y[lower];
						float32 dx1 = curr.x[upper] - curr.x[middle]; float32 dy1 = curr.y[upper] - curr.y[middle];
						float32 norm0 = sqrtf(dx0*dx0+dy0*dy0);	float32 norm1 = sqrtf(dx1*dx1+dy1*dy1);
						if ( !(norm0 > 0.0f && norm1 > 0.0f) ) {
							//Identical points, don't do anything!
							goto Skip;
						}
						dx0 /= norm0; dy0 /= norm0;
						dx1 /= norm1; dy1 /= norm1;
						float32 cross = dx0 * dy1 - dx1 * dy0;
						float32 dot = dx0*dx1 + dy0*dy1;
						if (fabs(cross) < b2_angularSlop && dot > 0) {
							//Angle too close, split the triangle across from this point.
							//This is guaranteed to result in two triangles that satify
							//the tolerance (one of the angles is 90 degrees)
							float32 dx2 = curr.x[lower] - curr.x[upper]; float32 dy2 = curr.y[lower] - curr.y[upper];
							float32 norm2 = sqrtf(dx2*dx2+dy2*dy2);
							if (norm2 == 0.0f) {
								goto Skip;
							}
							dx2 /= norm2; dy2 /= norm2;
							float32 thisArea = curr.GetArea();
							float32 thisHeight = 2.0f * thisArea / norm2;
							float32 buffer2 = dx2;
							dx2 = dy2; dy2 = -buffer2;
							//Make two new polygons
							//printf("dx2: %f, dy2: %f, thisHeight: %f, middle: %d\n",dx2,dy2,thisHeight,middle);
							float32 newX1[3] = { curr.x[middle]+dx2*thisHeight, curr.x[lower], curr.x[middle] };
							float32 newY1[3] = { curr.y[middle]+dy2*thisHeight, curr.y[lower], curr.y[middle] };
							float32 newX2[3] = { newX1[0], curr.x[middle], curr.x[upper] };
							float32 newY2[3] = { newY1[0], curr.y[middle], curr.y[upper] };
							b2Polygon p1(newX1,newY1,3);
							b2Polygon p2(newX2,newY2,3);
							if (p1.IsUsable()){
								p1.AddTo(*toAdd);
								
								
								bd->CreateFixture(toAdd);
								++extra;
							} else if (B2_POLYGON_REPORT_ERRORS){
								printf("Didn't add unusable polygon.  Dumping vertices:\n");
								p1.print();
							}
							if (p2.IsUsable()){
								p2.AddTo(pdarray[i+extra]);
								
								bd->CreateFixture(toAdd);
							} else if (B2_POLYGON_REPORT_ERRORS){
								printf("Didn't add unusable polygon.  Dumping vertices:\n");
								p2.print();
							}
							goto Skip;
						}
					}

			}
			if (decomposed[i].IsUsable()){
				decomposed[i].AddTo(*toAdd);
				
				bd->CreateFixture((const b2FixtureDef*)toAdd);
			} else if (B2_POLYGON_REPORT_ERRORS){
				printf("Didn't add unusable polygon.  Dumping vertices:\n");
				decomposed[i].print();
			}
Skip:
			;
        }
		delete[] pdarray;
        delete[] decomposed;
		return;// pdarray; //needs to be deleted after body is created
}

#endif
    /**
	 * Find the convex hull of a point cloud using "Gift-wrap" algorithm - start
     * with an extremal point, and walk around the outside edge by testing
     * angles.
     * 
     * Runs in O(N*S) time where S is number of sides of resulting polygon.
     * Worst case: point cloud is all vertices of convex polygon -> O(N^2).
     * 
     * There may be faster algorithms to do this, should you need one -
     * this is just the simplest. You can get O(N log N) expected time if you
     * try, I think, and O(N) if you restrict inputs to simple polygons.
     * 
     * Returns null if number of vertices passed is less than 3.
     * 
	 * Results should be passed through convex decomposition afterwards
	 * to ensure that each shape has few enough points to be used in Box2d.
	 *
     * FIXME?: May be buggy with colinear points on hull. Couldn't find a test
     * case that resulted in wrong behavior. If one turns up, the solution is to
     * supplement angle check with an equality resolver that always picks the
     * longer edge. I think the current solution is working, though it sometimes
     * creates an extra edge along a line.
     */
	
b2Polygon ConvexHull(b2Vec2* v, int nVert) {
        float32* cloudX = new float32[nVert];
        float32* cloudY = new float32[nVert];
        for (int32 i = 0; i < nVert; ++i) {
            cloudX[i] = v[i].x;
            cloudY[i] = v[i].y;
        }
        b2Polygon result = ConvexHull(cloudX, cloudY, nVert);
		delete[] cloudX;
		delete[] cloudY;
		return result;
}
	
b2Polygon ConvexHull(float32* cloudX, float32* cloudY, int32 nVert) {
		b2Assert(nVert > 2);
        int32* edgeList = new int32[nVert];
        int32 numEdges = 0;
		
	float32 minY = 1e10;
        int32 minYIndex = nVert;
        for (int32 i = 0; i < nVert; ++i) {
            if (cloudY[i] < minY) {
                minY = cloudY[i];
                minYIndex = i;
            }
        }
		
        int32 startIndex = minYIndex;
        int32 winIndex = -1;
        float32 dx = -1.0f;
        float32 dy = 0.0f;
        while (winIndex != minYIndex) {
            float32 newdx = 0.0f;
            float32 newdy = 0.0f;
            float32 maxDot = -2.0f;
            for (int32 i = 0; i < nVert; ++i) {
                if (i == startIndex)
                    continue;
                newdx = cloudX[i] - cloudX[startIndex];
                newdy = cloudY[i] - cloudY[startIndex];
                float32 nrm = sqrtf(newdx * newdx + newdy * newdy);
                nrm = (nrm == 0.0f) ? 1.0f : nrm;
                newdx /= nrm;
                newdy /= nrm;
                
                //Cross and dot products act as proxy for angle
                //without requiring inverse trig.
                //FIXED: don't need cross test
                //float32 newCross = newdx * dy - newdy * dx;
                float32 newDot = newdx * dx + newdy * dy;
                if (newDot > maxDot) {//newCross >= 0.0f && newDot > maxDot) {
                    maxDot = newDot;
                    winIndex = i;
                }
            }
            edgeList[numEdges++] = winIndex;
            dx = cloudX[winIndex] - cloudX[startIndex];
            dy = cloudY[winIndex] - cloudY[startIndex];
            float32 nrm = sqrtf(dx * dx + dy * dy);
            nrm = (nrm == 0.0f) ? 1.0f : nrm;
            dx /= nrm;
            dy /= nrm;
            startIndex = winIndex;
        }
		
        float32* xres = new float32[numEdges];
        float32* yres = new float32[numEdges];
        for (int32 i = 0; i < numEdges; i++) {
            xres[i] = cloudX[edgeList[i]];
            yres[i] = cloudY[edgeList[i]];
			//("%f, %f\n",xres[i],yres[i]);
        }
		
        b2Polygon returnVal(xres, yres, numEdges);

        delete[] xres;
		delete[] yres;
        delete[] edgeList;
		returnVal.MergeParallelEdges(b2_angularSlop);
		return returnVal;
}


/*
 * Given sines and cosines, tells if A's angle is less than B's on -Pi, Pi
 * (in other words, is A "righter" than B)
 */
bool IsRighter(float32 sinA, float32 cosA, float32 sinB, float32 cosB){
	if (sinA < 0){
		if (sinB > 0 || cosA <= cosB) return true;
		else return false;
	} else {
		if (sinB < 0 || cosA <= cosB) return false;
		else return true;
	}
}

//Fix for obnoxious behavior for the % operator for negative numbers...
int32 remainder(int32 x, int32 modulus){
	int32 rem = x % modulus;
	while (rem < 0){
		rem += modulus;
	}
	return rem;
}

/*
Method:
Start at vertex with minimum y (pick maximum x one if there are two).  
We aim our "lastDir" vector at (1.0, 0)
We look at the two rays going off from our start vertex, and follow whichever
has the smallest angle (in -Pi -> Pi) wrt lastDir ("rightest" turn)

Loop until we hit starting vertex:

We add our current vertex to the list.
We check the seg from current vertex to next vertex for intersections
  - if no intersections, follow to next vertex and continue
  - if intersections, pick one with minimum distance
    - if more than one, pick one with "rightest" next point (two possibilities for each)

*/

b2Polygon TraceEdge(b2Polygon* p){
	b2PolyNode* nodes = new b2PolyNode[p->nVertices*p->nVertices];//overkill, but sufficient (order of mag. is right)
	int32 nNodes = 0;

	//Add base nodes (raw outline)
	for (int32 i=0; i < p->nVertices; ++i){
		b2Vec2 pos(p->x[i],p->y[i]);
		nodes[i].position = pos;
		++nNodes;
		int32 iplus = (i==p->nVertices-1)?0:i+1;
		int32 iminus = (i==0)?p->nVertices-1:i-1;
		nodes[i].AddConnection(nodes[iplus]);
		nodes[i].AddConnection(nodes[iminus]);
	}

	//Process intersection nodes - horribly inefficient
	bool dirty = true;
	int counter = 0;
	while (dirty){
		dirty = false;
		for (int32 i=0; i < nNodes; ++i){
			for (int32 j=0; j < nodes[i].nConnected; ++j){
				for (int32 k=0; k < nNodes; ++k){
					if (k==i || &nodes[k] == nodes[i].connected[j]) continue;
					for (int32 l=0; l < nodes[k].nConnected; ++l){
				
						if ( nodes[k].connected[l] == nodes[i].connected[j] ||
							 nodes[k].connected[l] == &nodes[i]) continue;
						//Check intersection
						b2Vec2 intersectPt;
						//if (counter > 100) printf("checking intersection: %d, %d, %d, %d\n",i,j,k,l);
						bool crosses = intersect(nodes[i].position,nodes[i].connected[j]->position,
												 nodes[k].position,nodes[k].connected[l]->position,
												 intersectPt);
						if (crosses){
							/*if (counter > 100) {
								printf("Found crossing at %f, %f\n",intersectPt.x, intersectPt.y);
								printf("Locations: %f,%f - %f,%f | %f,%f - %f,%f\n",
												nodes[i].position.x, nodes[i].position.y,
												nodes[i].connected[j]->position.x, nodes[i].connected[j]->position.y,
												nodes[k].position.x,nodes[k].position.y,
												nodes[k].connected[l]->position.x,nodes[k].connected[l]->position.y);
								printf("Memory addresses: %d, %d, %d, %d\n",(int)&nodes[i],(int)nodes[i].connected[j],(int)&nodes[k],(int)nodes[k].connected[l]);
							}*/
							dirty = true;
							//Destroy and re-hook connections at crossing point
							b2PolyNode* connj = nodes[i].connected[j];
							b2PolyNode* connl = nodes[k].connected[l];
							nodes[i].connected[j]->RemoveConnection(nodes[i]);
							nodes[i].RemoveConnection(*connj);
							nodes[k].connected[l]->RemoveConnection(nodes[k]);
							nodes[k].RemoveConnection(*connl);
							nodes[nNodes] = b2PolyNode(intersectPt);
							nodes[nNodes].AddConnection(nodes[i]);
							nodes[i].AddConnection(nodes[nNodes]);
							nodes[nNodes].AddConnection(nodes[k]);
							nodes[k].AddConnection(nodes[nNodes]);
							nodes[nNodes].AddConnection(*connj);
							connj->AddConnection(nodes[nNodes]);
							nodes[nNodes].AddConnection(*connl);
							connl->AddConnection(nodes[nNodes]);
							++nNodes;
							goto SkipOut;
						}
					}
				}
			}
		}
		SkipOut:
		++counter;
		//if (counter > 100) printf("Counter: %d\n",counter);
	}
	
	/*
	// Debugging: check for connection consistency
	for (int32 i=0; i<nNodes; ++i) {
		int32 nConn = nodes[i].nConnected;
		for (int32 j=0; j<nConn; ++j) {
			if (nodes[i].connected[j]->nConnected == 0) b2Assert(false);
			b2PolyNode* connect = nodes[i].connected[j];
			bool found = false;
			for (int32 k=0; k<connect->nConnected; ++k) {
				if (connect->connected[k] == &nodes[i]) found = true;
			}
			b2Assert(found);
		}
	}*/

	//Collapse duplicate points
	bool foundDupe = true;
	int nActive = nNodes;
	while (foundDupe){
		foundDupe = false;
		for (int32 i=0; i < nNodes; ++i){
			if (nodes[i].nConnected == 0) continue;
			for (int32 j=i+1; j < nNodes; ++j){
				if (nodes[j].nConnected == 0) continue;
				b2Vec2 diff = nodes[i].position - nodes[j].position;
				if (diff.LengthSquared() <= COLLAPSE_DIST_SQR){
					if (nActive <= 3) return b2Polygon();
					//printf("Found dupe, %d left\n",nActive);
					--nActive;
					foundDupe = true;
					b2PolyNode* inode = &nodes[i];
					b2PolyNode* jnode = &nodes[j];
					//Move all of j's connections to i, and orphan j
					int32 njConn = jnode->nConnected;
					for (int32 k=0; k < njConn; ++k){
						b2PolyNode* knode = jnode->connected[k];
						b2Assert(knode != jnode);
						if (knode != inode) {
							inode->AddConnection(*knode);
							knode->AddConnection(*inode);
						}
						knode->RemoveConnection(*jnode);
						//printf("knode %d on node %d now has %d connections\n",k,j,knode->nConnected);
						//printf("Found duplicate point.\n");
					}
					/*
					printf("Orphaning node at address %d\n",(int)jnode);
					for (int32 k=0; k<njConn; ++k) {
						if (jnode->connected[k]->IsConnectedTo(*jnode)) printf("Problem!!!\n");
					}
					for (int32 k=0; k < njConn; ++k){
						jnode->RemoveConnectionByIndex(k);
					}
					*/
					jnode->nConnected = 0;
				}
			}
		}
	}
	
	/*
	// Debugging: check for connection consistency
	for (int32 i=0; i<nNodes; ++i) {
		int32 nConn = nodes[i].nConnected;
		printf("Node %d has %d connections\n",i,nConn);
		for (int32 j=0; j<nConn; ++j) {
			if (nodes[i].connected[j]->nConnected == 0) {
				printf("Problem with node %d connection at address %d\n",i,(int)(nodes[i].connected[j]));
				b2Assert(false);
			}
			b2PolyNode* connect = nodes[i].connected[j];
			bool found = false;
			for (int32 k=0; k<connect->nConnected; ++k) {
				if (connect->connected[k] == &nodes[i]) found = true;
			}
			if (!found) printf("Connection %d (of %d) on node %d (of %d) did not have reciprocal connection.\n",j,nConn,i,nNodes);
			b2Assert(found);
		}
	}//*/

	//Now walk the edge of the list

	//Find node with minimum y value (max x if equal)
	float32 minY = 1e10;
	float32 maxX = -1e10;
	int32 minYIndex = -1;
	for (int32 i = 0; i < nNodes; ++i) {
		if (nodes[i].position.y < minY && nodes[i].nConnected > 1) {
			minY = nodes[i].position.y;
			minYIndex = i;
			maxX = nodes[i].position.x;
		} else if (nodes[i].position.y == minY && nodes[i].position.x > maxX && nodes[i].nConnected > 1) {
			minYIndex = i;
			maxX = nodes[i].position.x;
		}
	}

	b2Vec2 origDir(1.0f,0.0f);
	b2Vec2* resultVecs = new b2Vec2[4*nNodes];// nodes may be visited more than once, unfortunately - change to growable array!
	int32 nResultVecs = 0;
	b2PolyNode* currentNode = &nodes[minYIndex];
	b2PolyNode* startNode = currentNode;
	b2Assert(currentNode->nConnected > 0);
	b2PolyNode* nextNode = currentNode->GetRightestConnection(origDir);
	if (!nextNode) goto CleanUp; // Borked, clean up our mess and return
	resultVecs[0] = startNode->position;
	++nResultVecs;
	while (nextNode != startNode){
		if (nResultVecs > 4*nNodes){
			/*
			printf("%d, %d, %d\n",(int)startNode,(int)currentNode,(int)nextNode);
			printf("%f, %f -> %f, %f\n",currentNode->position.x,currentNode->position.y, nextNode->position.x, nextNode->position.y);
				p->printFormatted();
				printf("Dumping connection graph: \n");
				for (int32 i=0; i<nNodes; ++i) {
					printf("nodex[%d] = %f; nodey[%d] = %f;\n",i,nodes[i].position.x,i,nodes[i].position.y);
					printf("//connected to\n");
					for (int32 j=0; j<nodes[i].nConnected; ++j) {
						printf("connx[%d][%d] = %f; conny[%d][%d] = %f;\n",i,j,nodes[i].connected[j]->position.x, i,j,nodes[i].connected[j]->position.y);
					}
				}
				printf("Dumping results thus far: \n");
				for (int32 i=0; i<nResultVecs; ++i) {
					printf("x[%d]=map(%f,-3,3,0,width); y[%d] = map(%f,-3,3,height,0);\n",i,resultVecs[i].x,i,resultVecs[i].y);
				}
			//*/
			b2Assert(false); //nodes should never be visited four times apiece (proof?), so we've probably hit a loop...crap
		}
		resultVecs[nResultVecs++] = nextNode->position;
		b2PolyNode* oldNode = currentNode;
		currentNode = nextNode;
		//printf("Old node connections = %d; address %d\n",oldNode->nConnected, (int)oldNode);
		//printf("Current node connections = %d; address %d\n",currentNode->nConnected, (int)currentNode);
		nextNode = currentNode->GetRightestConnection(oldNode);
		if (!nextNode) goto CleanUp; // There was a problem, so jump out of the loop and use whatever garbage we've generated so far
		//printf("nextNode address: %d\n",(int)nextNode);
	}

	CleanUp:
	
	float32* xres = new float32[nResultVecs];
	float32* yres = new float32[nResultVecs];
	for (int32 i=0; i<nResultVecs; ++i){
		xres[i] = resultVecs[i].x;
		yres[i] = resultVecs[i].y;
	}
	b2Polygon retval(xres,yres,nResultVecs);
	delete[] resultVecs;
	delete[] yres;
	delete[] xres;
	delete[] nodes;
	return retval;
}

b2PolyNode::b2PolyNode(){
	nConnected = 0;
	visited = false;
}
b2PolyNode::b2PolyNode(b2Vec2& pos){
	position = pos;
	nConnected = 0;
	visited = false;
}

void b2PolyNode::AddConnection(b2PolyNode& toMe){
	b2Assert(nConnected < MAX_CONNECTED);
	// Ignore duplicate additions
	for (int32 i=0; i<nConnected; ++i) {
		if (connected[i] == &toMe) return;
	}
	connected[nConnected] = &toMe;
	++nConnected;
}

void b2PolyNode::RemoveConnection(b2PolyNode& fromMe){
	bool isFound = false;
	int32 foundIndex = -1;
	for (int32 i=0; i<nConnected; ++i){
		if (&fromMe == connected[i]) {//.position == connected[i]->position){
			isFound = true;
			foundIndex = i;
			break;
		}
	}
	b2Assert(isFound);
	--nConnected;
	//printf("nConnected: %d\n",nConnected);
	for (int32 i=foundIndex; i < nConnected; ++i){
		connected[i] = connected[i+1];
	}
}
void b2PolyNode::RemoveConnectionByIndex(int32 index){
	--nConnected;
	//printf("New nConnected = %d\n",nConnected);
	for (int32 i=index; i < nConnected; ++i){
		connected[i] = connected[i+1];
	}
}
bool b2PolyNode::IsConnectedTo(b2PolyNode& me){
	bool isFound = false;
	for (int32 i=0; i<nConnected; ++i){
		if (&me == connected[i]) {//.position == connected[i]->position){
			isFound = true;
			break;
		}
	}
	return isFound;
}
b2PolyNode* b2PolyNode::GetRightestConnection(b2PolyNode* incoming){
	if (nConnected == 0) b2Assert(false); // This means the connection graph is inconsistent
	if (nConnected == 1) {
		//b2Assert(false);
		// Because of the possibility of collapsing nearby points,
		// we may end up with "spider legs" dangling off of a region.
		// The correct behavior here is to turn around.
		return incoming;
	}
	b2Vec2 inDir = position - incoming->position;
	float32 inLength = inDir.Normalize();
	b2Assert(inLength > CMP_EPSILON);
	
	b2PolyNode* result = NULL;
	for (int32 i=0; i<nConnected; ++i){
		if (connected[i] == incoming) continue;
		b2Vec2 testDir = connected[i]->position - position;
		float32 testLengthSqr = testDir.LengthSquared();
		testDir.Normalize();
		/*
		if (testLengthSqr < COLLAPSE_DIST_SQR) {
			printf("Problem with connection %d\n",i);
			printf("This node has %d connections\n",nConnected);
			printf("That one has %d\n",connected[i]->nConnected);
			if (this == connected[i]) printf("This points at itself.\n");
		}*/
		b2Assert (testLengthSqr >= COLLAPSE_DIST_SQR);
		float32 myCos = b2Dot(inDir,testDir);
		float32 mySin = b2Cross(inDir,testDir);
		if (result){
			b2Vec2 resultDir = result->position - position;
			resultDir.Normalize();
			float32 resCos = b2Dot(inDir,resultDir);
			float32 resSin = b2Cross(inDir,resultDir);
			if (IsRighter(mySin,myCos,resSin,resCos)){
				result = connected[i];
			}
		} else{
			result = connected[i];
		}
	}
	if (B2_POLYGON_REPORT_ERRORS && !result) {
		printf("nConnected = %d\n",nConnected);
		for (int32 i=0; i<nConnected; ++i) {
			printf("connected[%d] @ %d\n",i,0);//(int)connected[i]);
		}
	}
	b2Assert(result);

	return result;
}

b2PolyNode* b2PolyNode::GetRightestConnection(b2Vec2& incomingDir){
	b2Vec2 diff = position-incomingDir;
	b2PolyNode temp(diff);
	b2PolyNode* res = GetRightestConnection(&temp);
	b2Assert(res);
	return res;
}
}
