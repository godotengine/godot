/*************************************************************************/
/*  delaunay_2d.h                                                        */
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

#ifndef DELAUNAY_2D_H
#define DELAUNAY_2D_H

#include "core/math/rect2.h"
#include "core/templates/vector.h"

class Delaunay2D {
public:
	struct Triangle {
		int points[3];
		Vector2 circum_centre;
		real_t circum_radius_squared;
		Triangle() {}
		Triangle(int p_a, int p_b, int p_c) {
			points[0] = p_a;
			points[1] = p_b;
			points[2] = p_c;
		}
	};

	struct Edge {
		int points[2];
		bool bad = false;
		Edge() {}
		Edge(int p_a, int p_b) {
			// Store indices in a sorted manner to avoid having to check both orientations later.
			if (p_a > p_b) {
				points[0] = p_b;
				points[1] = p_a;
			} else {
				points[0] = p_a;
				points[1] = p_b;
			}
		}
	};

	static void get_circum_circle(const Vector<Vector2> &p_vertices, const Triangle &p_triangle, Vector2 &r_circum_center, real_t &r_radius_squared) {
		Vector2 a = p_vertices[p_triangle.points[1]] - p_vertices[p_triangle.points[0]];
		Vector2 b = p_vertices[p_triangle.points[2]] - p_vertices[p_triangle.points[0]];

		Vector2 O = (b * a.length_squared() - a * b.length_squared()) / (a.cross(b) * 2.0f);

		// Rotating by 90°.
		real_t temp = O.y;
		O.y = -O.x;
		O.x = temp;

		r_radius_squared = O.length_squared();
		r_circum_center = O + p_vertices[p_triangle.points[0]];
	}

	static Triangle create_triangle(const Vector<Vector2> &p_vertices, const int &p_a, const int &p_b, const int &p_c) {
		Triangle result(p_a, p_b, p_c);

		// Get the values of the circumcircle and store them inside the triangle object.
		get_circum_circle(p_vertices, result, result.circum_centre, result.circum_radius_squared);

		return result;
	}

	static Vector<Triangle> triangulate(const Vector<Vector2> &p_points) {
		Vector<Vector2> points = p_points;
		Vector<Triangle> triangles;

		// Get the bounding rect.
		Rect2 rect;
		for (int i = 0; i < p_points.size(); i++) {
			if (i == 0) {
				rect.position = p_points[i];
			} else {
				rect.expand_to(p_points[i]);
			}
		}

		real_t delta_max = MAX(rect.size.width, rect.size.height);
		Vector2 center = rect.get_center();

		// Construct the bounding triangle out of the rectangle.
		points.push_back(Vector2(center.x - 20 * delta_max, center.y - delta_max));
		points.push_back(Vector2(center.x, center.y + 20 * delta_max));
		points.push_back(Vector2(center.x + 20 * delta_max, center.y - delta_max));

		Triangle bounding_triangle = create_triangle(points, p_points.size() + 0, p_points.size() + 1, p_points.size() + 2);
		triangles.push_back(bounding_triangle);

		for (int i = 0; i < p_points.size(); i++) {
			Vector<Edge> polygon;

			// Save the edges of the triangles, circumcircles of which contain the i-th vertex.
			// Delete the triangles themselves.
			for (int j = triangles.size() - 1; j >= 0; j--) {
				if (points[i].distance_squared_to(triangles[j].circum_centre) >= triangles[j].circum_radius_squared) {
					continue;
				}

				polygon.push_back(Edge(triangles[j].points[0], triangles[j].points[1]));
				polygon.push_back(Edge(triangles[j].points[1], triangles[j].points[2]));
				polygon.push_back(Edge(triangles[j].points[2], triangles[j].points[0]));

				triangles.remove_at(j);
			}

			// Create a triangle for every unique edge.
			for (int j = 0; j < polygon.size(); j++) {
				if (polygon[j].bad) {
					continue;
				}

				for (int k = j + 1; k < polygon.size(); k++) {
					if (polygon[k].points[0] != polygon[j].points[0] || polygon[k].points[1] != polygon[j].points[1]) {
						continue;
					}

					polygon.write[j].bad = true;
					polygon.write[k].bad = true;

					break; // Since no more than two triangles can share an edge, no more than two edges can share vertices.
				}

				// Create triangles out of good edges.
				if (!polygon[j].bad) {
					triangles.push_back(create_triangle(points, polygon[j].points[0], polygon[j].points[1], i));
				}
			}
		}

		// Filter out the triangles containing the vertices of the bounding triangle.
		for (int i = triangles.size() - 1; i >= 0; i--) {
			for (int j = 0; j < 3; j++) {
				if (triangles[i].points[j] >= p_points.size()) {
					triangles.remove_at(i);
					break;
				}
			}
		}

		return triangles;
	}
};

#endif // DELAUNAY_2D_H
