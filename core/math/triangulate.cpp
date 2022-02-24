/*************************************************************************/
/*  triangulate.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "triangulate.h"

real_t Triangulate::get_area(const Vector<Vector2> &contour) {
	int n = contour.size();
	const Vector2 *c = &contour[0];

	real_t A = 0.0;

	for (int p = n - 1, q = 0; q < n; p = q++) {
		A += c[p].cross(c[q]);
	}
	return A * 0.5f;
}

/*
 * `is_inside_triangle` decides if a point P is inside the triangle
 * defined by A, B, C.
 */
bool Triangulate::is_inside_triangle(real_t Ax, real_t Ay,
		real_t Bx, real_t By,
		real_t Cx, real_t Cy,
		real_t Px, real_t Py,
		bool include_edges) {
	real_t ax, ay, bx, by, cx, cy, apx, apy, bpx, bpy, cpx, cpy;
	real_t cCROSSap, bCROSScp, aCROSSbp;

	ax = Cx - Bx;
	ay = Cy - By;
	bx = Ax - Cx;
	by = Ay - Cy;
	cx = Bx - Ax;
	cy = By - Ay;
	apx = Px - Ax;
	apy = Py - Ay;
	bpx = Px - Bx;
	bpy = Py - By;
	cpx = Px - Cx;
	cpy = Py - Cy;

	aCROSSbp = ax * bpy - ay * bpx;
	cCROSSap = cx * apy - cy * apx;
	bCROSScp = bx * cpy - by * cpx;

	if (include_edges) {
		return ((aCROSSbp > 0) && (bCROSScp > 0) && (cCROSSap > 0));
	} else {
		return ((aCROSSbp >= 0) && (bCROSScp >= 0) && (cCROSSap >= 0));
	}
}

bool Triangulate::snip(const Vector<Vector2> &p_contour, int u, int v, int w, int n, const Vector<int> &V, bool relaxed) {
	int p;
	real_t Ax, Ay, Bx, By, Cx, Cy, Px, Py;
	const Vector2 *contour = &p_contour[0];

	Ax = contour[V[u]].x;
	Ay = contour[V[u]].y;

	Bx = contour[V[v]].x;
	By = contour[V[v]].y;

	Cx = contour[V[w]].x;
	Cy = contour[V[w]].y;

	// It can happen that the triangulation ends up with three aligned vertices to deal with.
	// In this scenario, making the check below strict may reject the possibility of
	// forming a last triangle with these aligned vertices, preventing the triangulatiom
	// from completing.
	// To avoid that we allow zero-area triangles if all else failed.
	float threshold = relaxed ? -CMP_EPSILON : CMP_EPSILON;

	if (threshold > (((Bx - Ax) * (Cy - Ay)) - ((By - Ay) * (Cx - Ax)))) {
		return false;
	}

	for (p = 0; p < n; p++) {
		if ((p == u) || (p == v) || (p == w)) {
			continue;
		}
		Px = contour[V[p]].x;
		Py = contour[V[p]].y;
		if (is_inside_triangle(Ax, Ay, Bx, By, Cx, Cy, Px, Py, relaxed)) {
			return false;
		}
	}

	return true;
}

bool Triangulate::triangulate(const Vector<Vector2> &contour, Vector<int> &result) {
	/* allocate and initialize list of Vertices in polygon */

	int n = contour.size();
	if (n < 3) {
		return false;
	}

	Vector<int> V;
	V.resize(n);

	/* we want a counter-clockwise polygon in V */

	if (0 < get_area(contour)) {
		for (int v = 0; v < n; v++) {
			V.write[v] = v;
		}
	} else {
		for (int v = 0; v < n; v++) {
			V.write[v] = (n - 1) - v;
		}
	}

	bool relaxed = false;

	int nv = n;

	/*  remove nv-2 Vertices, creating 1 triangle every time */
	int count = 2 * nv; /* error detection */

	for (int v = nv - 1; nv > 2;) {
		/* if we loop, it is probably a non-simple polygon */
		if (0 >= (count--)) {
			if (relaxed) {
				//** Triangulate: ERROR - probable bad polygon!
				return false;
			} else {
				// There may be aligned vertices that the strict
				// checks prevent from triangulating. In this situation
				// we are better off adding flat triangles than
				// failing, so we relax the checks and try one last
				// round.
				// Only relaxing the constraints as a last resort avoids
				// degenerate triangles when they aren't necessary.
				count = 2 * nv;
				relaxed = true;
			}
		}

		/* three consecutive vertices in current polygon, <u,v,w> */
		int u = v;
		if (nv <= u) {
			u = 0; /* previous */
		}
		v = u + 1;
		if (nv <= v) {
			v = 0; /* new v    */
		}
		int w = v + 1;
		if (nv <= w) {
			w = 0; /* next     */
		}

		if (snip(contour, u, v, w, nv, V, relaxed)) {
			int a, b, c, s, t;

			/* true names of the vertices */
			a = V[u];
			b = V[v];
			c = V[w];

			/* output Triangle */
			result.push_back(a);
			result.push_back(b);
			result.push_back(c);

			/* remove v from remaining polygon */
			for (s = v, t = v + 1; t < nv; s++, t++) {
				V.write[s] = V[t];
			}

			nv--;

			/* reset error detection counter */
			count = 2 * nv;
		}
	}

	return true;
}
