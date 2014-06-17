#include "polygon_path_finder.h"
#include "geometry.h"


bool PolygonPathFinder::_is_point_inside(const Vector2& p_point) const {

	int crosses=0;


	for (Set<Edge>::Element *E=edges.front();E;E=E->next()) {


		const Edge& e=E->get();

		Vector2 a = points[e.points[0]].pos;
		Vector2 b = points[e.points[1]].pos;


		if (Geometry::segment_intersects_segment_2d(a,b,p_point,outside_point,NULL)) {
			crosses++;
		}
	}

	return crosses&1;
}

void PolygonPathFinder::setup(const Vector<Vector2>& p_points, const Vector<int>& p_connections) {


	ERR_FAIL_COND(p_connections.size()&1);

	points.clear();
	edges.clear();

	//insert points

	int point_count=p_points.size();
	points.resize(point_count+2);
	bounds=Rect2();

	for(int i=0;i<p_points.size();i++) {

		points[i].pos=p_points[i];

		outside_point.x = i==0?p_points[0].x:(MAX( p_points[i].x, outside_point.x ));
		outside_point.y = i==0?p_points[0].y:(MAX( p_points[i].y, outside_point.y ));

		if (i==0) {
			bounds.pos=points[i].pos;
		} else {
			bounds.expand_to(points[i].pos);
		}
	}

	outside_point.x+=20.451+Math::randf()*10.2039;
	outside_point.y+=21.193+Math::randf()*12.5412;

	//insert edges (which are also connetions)

	for(int i=0;i<p_connections.size();i+=2) {

		Edge e(p_connections[i],p_connections[i+1]);
		ERR_FAIL_INDEX(e.points[0],point_count);
		ERR_FAIL_INDEX(e.points[1],point_count);
		points[p_connections[i]].connections.insert(p_connections[i+1]);
		points[p_connections[i+1]].connections.insert(p_connections[i]);
		edges.insert(e);
	}


	//fill the remaining connections based on visibility

	for(int i=0;i<point_count;i++) {

		for(int j=i+1;j<point_count;j++) {

			if (edges.has(Edge(i,j)))
				continue; //if in edge ignore

			Vector2 from=points[i].pos;
			Vector2 to=points[j].pos;

			if (!_is_point_inside(from*0.5+to*0.5)) //connection between points in inside space
				continue;

			bool valid=true;

			for (Set<Edge>::Element *E=edges.front();E;E=E->next()) {

				const Edge& e=E->get();
				if (e.points[0]==i || e.points[1]==i || e.points[0]==j || e.points[1]==j )
					continue;


				Vector2 a = points[e.points[0]].pos;
				Vector2 b = points[e.points[1]].pos;


				if (Geometry::segment_intersects_segment_2d(a,b,from,to,NULL)) {
					valid=false;
					break;
				}

			}

			if (valid) {
				points[i].connections.insert(j);
				points[j].connections.insert(i);
			}
		}
	}
}


Vector<Vector2> PolygonPathFinder::find_path(const Vector2& p_from, const Vector2& p_to) {

	Vector<Vector2> path;
	if (!_is_point_inside(p_from)) {
		printf("p_from outside\n");
		return path;
	};
	if (!_is_point_inside(p_to)) {
		printf("p_to outside\n");
		return path;
	};

	//test direct connection
	{

		bool can_see_eachother=true;

		for (Set<Edge>::Element *E=edges.front();E;E=E->next()) {

			const Edge& e=E->get();
			Vector2 a = points[e.points[0]].pos;
			Vector2 b = points[e.points[1]].pos;


			if (Geometry::segment_intersects_segment_2d(a,b,p_from,p_to,NULL)) {
				can_see_eachother=false;
				break;
			}

		}

		if (can_see_eachother) {

			path.push_back(p_from);
			path.push_back(p_to);
			return path;
		}
	}

	//add to graph

	int aidx = points.size()-2;
	int bidx = points.size()-1;
	points[aidx].pos=p_from;
	points[bidx].pos=p_to;
	points[aidx].distance=0;
	points[bidx].distance=0;
	points[aidx].prev=-1;
	points[bidx].prev=-1;


	for(int i=0;i<points.size()-2;i++) {


		bool valid_a=true;
		bool valid_b=true;
		points[i].prev=-1;
		points[i].distance=0;

		for (Set<Edge>::Element *E=edges.front();E;E=E->next()) {

			const Edge& e=E->get();

			if (e.points[0]==i || e.points[1]==i)
				continue;

			Vector2 a = points[e.points[0]].pos;
			Vector2 b = points[e.points[1]].pos;

			if (valid_a) {

				if (Geometry::segment_intersects_segment_2d(a,b,p_from,points[i].pos,NULL)) {
					valid_a=false;
				}
			}

			if (valid_b) {

				if (Geometry::segment_intersects_segment_2d(a,b,p_to,points[i].pos,NULL)) {
					valid_b=false;
				}
			}

			if (!valid_a && !valid_b)
				break;

		}

		if (valid_a) {
			points[i].connections.insert(aidx);
			points[aidx].connections.insert(i);
		}

		if (valid_b) {
			points[i].connections.insert(bidx);
			points[bidx].connections.insert(i);
		}

	}
	//solve graph

	Set<int> open_list;

	points[aidx].distance=0;
	points[aidx].prev=aidx;
	for(Set<int>::Element *E=points[aidx].connections.front();E;E=E->next()) {

		open_list.insert(E->get());
		points[E->get()].distance=p_from.distance_to(points[E->get()].pos);
		points[E->get()].prev=aidx;

	}


	bool found_route=false;

	while(true) {

		if (open_list.size()==0) {
			printf("open list empty\n");
			break;
		}
		//check open list

		int least_cost_point=-1;
		float least_cost=1e30;

		//this could be faster (cache previous results)
		for (Set<int>::Element *E=open_list.front();E;E=E->next()) {

			const Point& p =points[E->get()];
			float cost = p.distance;
			cost+=p.pos.distance_to(p_to);
			if (cost<least_cost) {

				least_cost_point=E->get();
				least_cost=cost;
			}
		}


		Point &np = points[least_cost_point];
		//open the neighbours for search

		for(Set<int>::Element *E=np.connections.front();E;E=E->next()) {

			Point& p =points[E->get()];
			float distance = np.pos.distance_to(p.pos) + np.distance;

			if (p.prev!=-1) {
				//oh this was visited already, can we win the cost?

				if (p.distance>distance) {

					p.prev=least_cost_point; //reasign previous
					p.distance=distance;
				}
			} else {
				//add to open neighbours

				p.prev=least_cost_point;
				p.distance=distance;
				open_list.insert(E->get());

				if (E->get()==bidx) {
					//oh my reached end! stop algorithm
					found_route=true;
					break;

				}

			}
		}

		if (found_route)
			break;

		open_list.erase(least_cost_point);
	}

	if (found_route) {
		int at = bidx;
		path.push_back(points[at].pos);
		do {
			at=points[at].prev;
			path.push_back(points[at].pos);
		} while (at!=aidx);

		path.invert();;
	}

	for(int i=0;i<points.size()-2;i++) {

		points[i].connections.erase(aidx);
		points[i].connections.erase(bidx);
		points[i].prev=-1;
		points[i].distance=0;
	}

	points[aidx].connections.clear();
	points[aidx].prev=-1;
	points[aidx].distance=0;
	points[bidx].connections.clear();
	points[bidx].prev=-1;
	points[bidx].distance=0;

	return path;
}

void PolygonPathFinder::_set_data(const Dictionary& p_data) {


	ERR_FAIL_COND(!p_data.has("points"));
	ERR_FAIL_COND(!p_data.has("connections"));
	ERR_FAIL_COND(!p_data.has("segments"));
	ERR_FAIL_COND(!p_data.has("bounds"));

	DVector<Vector2> p=p_data["points"];
	Array c=p_data["connections"];

	ERR_FAIL_COND(c.size()!=p.size());
	if (c.size())
		return;

	int pc = p.size();
	points.resize(pc+2);

	DVector<Vector2>::Read pr=p.read();
	for(int i=0;i<pc;i++) {
		points[i].pos=pr[i];
		DVector<int> con=c[i];
		DVector<int>::Read cr=con.read();
		int cc=con.size();
		for(int j=0;j<cc;j++) {

			points[i].connections.insert(cr[j]);
		}

	}

	DVector<int> segs=p_data["segments"];
	int sc=segs.size();
	ERR_FAIL_COND(sc&1);
	DVector<int>::Read sr = segs.read();
	for(int i=0;i<sc;i+=2) {

		Edge e(sr[i],sr[i+1]);
		edges.insert(e);
	}
	bounds=p_data["bounds"];

}

Dictionary PolygonPathFinder::_get_data() const{

	Dictionary d;
	DVector<Vector2> p;
	DVector<int> ind;
	Array connections;
	p.resize(points.size()-2);
	connections.resize(points.size()-2);
	ind.resize(edges.size()*2);
	{
		DVector<Vector2>::Write wp=p.write();
		for(int i=0;i<points.size()-2;i++) {
			wp[i]=points[i].pos;
			DVector<int> c;
			c.resize(points[i].connections.size());
			{
				DVector<int>::Write cw=c.write();
				int idx=0;
				for (Set<int>::Element *E=points[i].connections.front();E;E=E->next()) {
					cw[idx++]=E->get();
				}
			}
			connections[i]=c;
		}
	}
	{

		DVector<int>::Write iw=ind.write();
		int idx=0;
		for (Set<Edge>::Element *E=edges.front();E;E=E->next()) {
			iw[idx++]=E->get().points[0];
			iw[idx++]=E->get().points[1];
		}

	}

	d["bounds"]=bounds;
	d["points"]=p;
	d["connections"]=connections;
	d["segments"]=ind;

	return d;

}

bool PolygonPathFinder::is_point_inside(const Vector2& p_point) const {

	return _is_point_inside(p_point);
}

Vector2 PolygonPathFinder::get_closest_point(const Vector2& p_point) const {

	int closest_idx=-1;
	float closest_dist=1e20;
	for(int i=0;i<points.size()-2;i++) {

		float d = p_point.distance_squared_to(points[i].pos);
		if (d<closest_dist) {
			d=closest_dist;
			closest_idx=i;
		}

	}

	ERR_FAIL_COND_V(closest_idx==-1,Vector2());

	return points[closest_idx].pos;
}


Vector<Vector2> PolygonPathFinder::get_intersections(const Vector2& p_from, const Vector2& p_to) const {

	Vector<Vector2> inters;

	for (Set<Edge>::Element *E=edges.front();E;E=E->next()) {
		Vector2 a = points[E->get().points[0]].pos;
		Vector2 b = points[E->get().points[1]].pos;

		Vector2 res;
		if (Geometry::segment_intersects_segment_2d(a,b,p_from,p_to,&res)) {
			inters.push_back(res);
		}
	}

	return inters;

}

Rect2 PolygonPathFinder::get_bounds() const {

	return bounds;
}


void PolygonPathFinder::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("setup","points","connections"),&PolygonPathFinder::setup);
	ObjectTypeDB::bind_method(_MD("find_path","from","to"),&PolygonPathFinder::find_path);
	ObjectTypeDB::bind_method(_MD("get_intersections","from","to"),&PolygonPathFinder::get_intersections);
	ObjectTypeDB::bind_method(_MD("get_closest_point","point"),&PolygonPathFinder::get_closest_point);
	ObjectTypeDB::bind_method(_MD("is_point_inside","point"),&PolygonPathFinder::is_point_inside);
	ObjectTypeDB::bind_method(_MD("get_bounds"),&PolygonPathFinder::get_bounds);
	ObjectTypeDB::bind_method(_MD("_set_data"),&PolygonPathFinder::_set_data);
	ObjectTypeDB::bind_method(_MD("_get_data"),&PolygonPathFinder::_get_data);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY,"data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR),_SCS("_set_data"),_SCS("_get_data"));

}

PolygonPathFinder::PolygonPathFinder()
{
}


