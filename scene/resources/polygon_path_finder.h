#ifndef POLYGON_PATH_FINDER_H
#define POLYGON_PATH_FINDER_H

#include "resource.h"

class PolygonPathFinder : public Resource {

	OBJ_TYPE(PolygonPathFinder,Resource);

	struct Point {
		Vector2 pos;
		Set<int> connections;
		float distance;
		int prev;
	};

	struct Edge {

		int points[2];

		_FORCE_INLINE_ bool operator<(const Edge& p_edge) const {

			if (points[0]==p_edge.points[0])
				return points[1]<p_edge.points[1];
			else
				return points[0]<p_edge.points[0];
		}

		Edge(int a=0, int b=0) {

			if (a>b) {
				SWAP(a,b);
			}
		}
	};

	Vector2 outside_point;

	Vector<Point> points;
	Set<Edge> edges;

	bool _is_point_inside(const Vector2& p_point);

	void _set_data(const Dictionary& p_data);
	Dictionary _get_data() const;
protected:

	static void _bind_methods();
public:


	void setup(const Vector<Vector2>& p_points, const Vector<int>& p_connections);
	Vector<Vector2> find_path(const Vector2& p_from, const Vector2& p_to);

	PolygonPathFinder();
};

#endif // POLYGON_PATH_FINDER_H
