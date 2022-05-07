/*************************************************************************/
/*  delaunay.h                                                           */
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

#ifndef DELAUNAY_H
#define DELAUNAY_H

#include "core/math/rect2.h"

class Delaunay2D {
public:
	struct Triangle {
		int points[3];
		bool bad;
		Triangle() { bad = false; }
		Triangle(int p_a, int p_b, int p_c) {
			points[0] = p_a;
			points[1] = p_b;
			points[2] = p_c;
			bad = false;
		}
	};

	struct Edge {
		int edge[2];
		bool bad;
		Edge() { bad = false; }
		Edge(int p_a, int p_b) {
			bad = false;
			edge[0] = p_a;
			edge[1] = p_b;
		}
	};

	static bool circum_circle_contains(const Vector<Vector2> &p_vertices, const Triangle &p_triangle, int p_vertex) {
		Vector2 p1 = p_vertices[p_triangle.points[0]];
		Vector2 p2 = p_vertices[p_triangle.points[1]];
		Vector2 p3 = p_vertices[p_triangle.points[2]];

		real_t ab = p1.x * p1.x + p1.y * p1.y;
		real_t cd = p2.x * p2.x + p2.y * p2.y;
		real_t ef = p3.x * p3.x + p3.y * p3.y;

		Vector2 circum(
				(ab * (p3.y - p2.y) + cd * (p1.y - p3.y) + ef * (p2.y - p1.y)) / (p1.x * (p3.y - p2.y) + p2.x * (p1.y - p3.y) + p3.x * (p2.y - p1.y)),
				(ab * (p3.x - p2.x) + cd * (p1.x - p3.x) + ef * (p2.x - p1.x)) / (p1.y * (p3.x - p2.x) + p2.y * (p1.x - p3.x) + p3.y * (p2.x - p1.x)));

		circum *= 0.5;
		float r = p1.distance_squared_to(circum);
		float d = p_vertices[p_vertex].distance_squared_to(circum);
		return d <= r;
	}

	static bool edge_compare(const Vector<Vector2> &p_vertices, const Edge &p_a, const Edge &p_b) {
		if (p_vertices[p_a.edge[0]].is_equal_approx(p_vertices[p_b.edge[0]]) && p_vertices[p_a.edge[1]].is_equal_approx(p_vertices[p_b.edge[1]])) {
			return true;
		}

		if (p_vertices[p_a.edge[0]].is_equal_approx(p_vertices[p_b.edge[1]]) && p_vertices[p_a.edge[1]].is_equal_approx(p_vertices[p_b.edge[0]])) {
			return true;
		}

		return false;
	}

	static Vector<Triangle> triangulate(const Vector<Vector2> &p_points) {
		Vector<Vector2> points = p_points;
		Vector<Triangle> triangles;

		Rect2 rect;
		for (int i = 0; i < p_points.size(); i++) {
			if (i == 0) {
				rect.position = p_points[i];
			} else {
				rect.expand_to(p_points[i]);
			}
		}

		float delta_max = MAX(rect.size.width, rect.size.height);
		Vector2 center = rect.position + rect.size * 0.5;

		points.push_back(Vector2(center.x - 20 * delta_max, center.y - delta_max));
		points.push_back(Vector2(center.x, center.y + 20 * delta_max));
		points.push_back(Vector2(center.x + 20 * delta_max, center.y - delta_max));

		triangles.push_back(Triangle(p_points.size() + 0, p_points.size() + 1, p_points.size() + 2));

		for (int i = 0; i < p_points.size(); i++) {
			//std::cout << "Traitement du point " << *p << std::endl;
			//std::cout << "_triangles contains " << _triangles.size() << " elements" << std::endl;

			Vector<Edge> polygon;

			for (int j = 0; j < triangles.size(); j++) {
				if (circum_circle_contains(points, triangles[j], i)) {
					triangles.write[j].bad = true;
					polygon.push_back(Edge(triangles[j].points[0], triangles[j].points[1]));
					polygon.push_back(Edge(triangles[j].points[1], triangles[j].points[2]));
					polygon.push_back(Edge(triangles[j].points[2], triangles[j].points[0]));
				}
			}

			for (int j = 0; j < triangles.size(); j++) {
				if (triangles[j].bad) {
					triangles.remove(j);
					j--;
				}
			}

			for (int j = 0; j < polygon.size(); j++) {
				for (int k = j + 1; k < polygon.size(); k++) {
					if (edge_compare(points, polygon[j], polygon[k])) {
						polygon.write[j].bad = true;
						polygon.write[k].bad = true;
					}
				}
			}

			for (int j = 0; j < polygon.size(); j++) {
				if (polygon[j].bad) {
					continue;
				}
				triangles.push_back(Triangle(polygon[j].edge[0], polygon[j].edge[1], i));
			}
		}

		for (int i = 0; i < triangles.size(); i++) {
			bool invalid = false;
			for (int j = 0; j < 3; j++) {
				if (triangles[i].points[j] >= p_points.size()) {
					invalid = true;
					break;
				}
			}
			if (invalid) {
				triangles.remove(i);
				i--;
			}
		}

		return triangles;
	}
};

#endif // DELAUNAY_H
