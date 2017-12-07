/*************************************************************************/
/*  b2d_decompose.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "b2d_decompose.h"

#include "thirdparty/b2d_convexdecomp/b2Polygon.h"

namespace b2ConvexDecomp {

void add_to_res(Vector<Vector<Vector2> > &res, const b2Polygon &p_poly) {

	Vector<Vector2> arr;
	for (int i = 0; i < p_poly.nVertices; i++) {

		arr.push_back(Vector2(p_poly.x[i], p_poly.y[i]));
	}

	res.push_back(arr);
}

static Vector<Vector<Vector2> > _b2d_decompose(const Vector<Vector2> &p_polygon) {

	Vector<Vector<Vector2> > res;
	if (p_polygon.size() < 3)
		return res;

	b2Vec2 *polys = memnew_arr(b2Vec2, p_polygon.size());
	for (int i = 0; i < p_polygon.size(); i++)
		polys[i] = b2Vec2(p_polygon[i].x, p_polygon[i].y);

	b2Polygon *p = new b2Polygon(polys, p_polygon.size());
	b2Polygon *decomposed = new b2Polygon[p->nVertices - 2]; //maximum number of polys

	memdelete_arr(polys);

	int32 nPolys = DecomposeConvex(p, decomposed, p->nVertices - 2);
	//int32 extra = 0;
	for (int32 i = 0; i < nPolys; ++i) {
		//   b2FixtureDef* toAdd = &pdarray[i+extra];
		// *toAdd = *prototype;
		//Hmm, shouldn't have to do all this...
		b2Polygon curr = decomposed[i];
		//TODO ewjordan: move this triangle handling to a better place so that
		//it happens even if this convenience function is not called.
		if (curr.nVertices == 3) {
			//Check here for near-parallel edges, since we can't
			//handle this in merge routine
			for (int j = 0; j < 3; ++j) {
				int32 lower = (j == 0) ? (curr.nVertices - 1) : (j - 1);
				int32 middle = j;
				int32 upper = (j == curr.nVertices - 1) ? (0) : (j + 1);
				float32 dx0 = curr.x[middle] - curr.x[lower];
				float32 dy0 = curr.y[middle] - curr.y[lower];
				float32 dx1 = curr.x[upper] - curr.x[middle];
				float32 dy1 = curr.y[upper] - curr.y[middle];
				float32 norm0 = sqrtf(dx0 * dx0 + dy0 * dy0);
				float32 norm1 = sqrtf(dx1 * dx1 + dy1 * dy1);
				if (!(norm0 > 0.0f && norm1 > 0.0f)) {
					//Identical points, don't do anything!
					goto Skip;
				}
				dx0 /= norm0;
				dy0 /= norm0;
				dx1 /= norm1;
				dy1 /= norm1;
				float32 cross = dx0 * dy1 - dx1 * dy0;
				float32 dot = dx0 * dx1 + dy0 * dy1;
				if (fabs(cross) < b2_angularSlop && dot > 0) {
					//Angle too close, split the triangle across from this point.
					//This is guaranteed to result in two triangles that satify
					//the tolerance (one of the angles is 90 degrees)
					float32 dx2 = curr.x[lower] - curr.x[upper];
					float32 dy2 = curr.y[lower] - curr.y[upper];
					float32 norm2 = sqrtf(dx2 * dx2 + dy2 * dy2);
					if (norm2 == 0.0f) {
						goto Skip;
					}
					dx2 /= norm2;
					dy2 /= norm2;
					float32 thisArea = curr.GetArea();
					float32 thisHeight = 2.0f * thisArea / norm2;
					float32 buffer2 = dx2;
					dx2 = dy2;
					dy2 = -buffer2;
					//Make two new polygons
					//printf("dx2: %f, dy2: %f, thisHeight: %f, middle: %d\n",dx2,dy2,thisHeight,middle);
					float32 newX1[3] = { curr.x[middle] + dx2 * thisHeight, curr.x[lower], curr.x[middle] };
					float32 newY1[3] = { curr.y[middle] + dy2 * thisHeight, curr.y[lower], curr.y[middle] };
					float32 newX2[3] = { newX1[0], curr.x[middle], curr.x[upper] };
					float32 newY2[3] = { newY1[0], curr.y[middle], curr.y[upper] };
					b2Polygon p1(newX1, newY1, 3);
					b2Polygon p2(newX2, newY2, 3);
					if (p1.IsUsable()) {
						add_to_res(res, p1);
						//++extra;
					} else if (B2_POLYGON_REPORT_ERRORS) {
						printf("Didn't add unusable polygon.  Dumping vertices:\n");
						p1.print();
					}
					if (p2.IsUsable()) {
						add_to_res(res, p2);

						//p2.AddTo(pdarray[i+extra]);

						//bd->CreateFixture(toAdd);
					} else if (B2_POLYGON_REPORT_ERRORS) {
						printf("Didn't add unusable polygon.  Dumping vertices:\n");
						p2.print();
					}
					goto Skip;
				}
			}
		}
		if (decomposed[i].IsUsable()) {
			add_to_res(res, decomposed[i]);

			//decomposed[i].AddTo(*toAdd);
			//bd->CreateFixture((const b2FixtureDef*)toAdd);
		} else if (B2_POLYGON_REPORT_ERRORS) {
			printf("Didn't add unusable polygon.  Dumping vertices:\n");
			decomposed[i].print();
		}
	Skip:;
	}
	//delete[] pdarray;
	delete[] decomposed;
	delete p;
	return res; // pdarray; //needs to be deleted after body is created
}
} // namespace b2ConvexDecomp

Vector<Vector<Vector2> > b2d_decompose(const Vector<Vector2> &p_polygon) {

	return b2ConvexDecomp::_b2d_decompose(p_polygon);
}
