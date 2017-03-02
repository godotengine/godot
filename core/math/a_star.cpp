/*************************************************************************/
/*  a_star.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "a_star.h"
#include "geometry.h"



void AStar::set_delegate(Object *p_to_object) {

	delegate = p_to_object;
}

Object *AStar::get_delegate() const {

	return delegate;
}

int AStar::get_available_point_id() const {

	if (points.empty()) {
		return 1;
	}

	return points.back()->key()+1;
}

void AStar::add_point(int p_id, const Vector3 &p_pos, real_t p_weight_scale,const Vector<Variant>& p_binds,uint32_t p_flags) {

	ERR_FAIL_COND(p_id<0);
	if (!points.has(p_id)) {
		Point *pt = memnew( Point );
		pt->id=p_id;
		pt->pos=p_pos;
		pt->weight_scale=p_weight_scale;
		pt->prev_point=NULL;
		pt->last_pass=0;
		pt->binds = p_binds;
		pt->flags = p_flags;
		points[p_id]=pt;
	} else {
		points[p_id]->pos=p_pos;
		points[p_id]->weight_scale=p_weight_scale;
		points[p_id]->binds = p_binds;
		points[p_id]->flags = p_flags;
	}
}

Vector3 AStar::get_point_pos(int p_id) const{

	ERR_FAIL_COND_V(!points.has(p_id),Vector3());

	return points[p_id]->pos;

}

void AStar::set_point_pos(int p_id, const Vector3 &p_pos){

	points[p_id]->pos = p_pos;

}

real_t AStar::get_point_weight_scale(int p_id) const{

	ERR_FAIL_COND_V(!points.has(p_id),0);

	return points[p_id]->weight_scale;

}

void AStar::set_point_weight_scale(int p_id, real_t p_weight_scale) {

	points[p_id]->weight_scale=p_weight_scale;

}

Vector<Variant> AStar::get_point_binds(int p_id) const{

	ERR_FAIL_COND_V(!points.has(p_id),Vector<Variant>());

	return points[p_id]->binds;

}

void AStar::set_point_binds(int p_id,const Vector<Variant>& p_binds) {

	points[p_id]->binds=p_binds;

}

uint32_t AStar::get_point_flags(int p_id) const {

	ERR_FAIL_COND_V(!points.has(p_id),0);

	return points[p_id]->flags;
}

void AStar::set_point_flags(int p_id, uint32_t p_flags) {

	points[p_id]->flags = p_flags;
}


void AStar::remove_point(int p_id){

	ERR_FAIL_COND(!points.has(p_id));

	Point* p = points[p_id];

	for(int i=0;i<p->neighbours.size();i++) {

		Segment s(p_id,p->neighbours[i]->id);
		segments.erase(s);
		p->neighbours[i]->neighbours.erase(p);
	}

	memdelete(p);
	points.erase(p_id);
}

void AStar::connect_points(int p_id,int p_with_id,bool bidirectional){

	ERR_FAIL_COND(!points.has(p_id));
	ERR_FAIL_COND(!points.has(p_with_id));
	ERR_FAIL_COND(p_id==p_with_id);


	Point* a = points[p_id];
	Point* b = points[p_with_id];
	a->neighbours.push_back(b);
	if (bidirectional)
		b->neighbours.push_back(a);

	Segment s(p_id,p_with_id);
	if (s.from==p_id) {
		s.from_point=a;
		s.to_point=b;
	} else {
		s.from_point=b;
		s.to_point=a;
	}

	segments.insert(s);


}
void AStar::disconnect_points(int p_id,int p_with_id){

	Segment s(p_id,p_with_id);
	ERR_FAIL_COND(!segments.has(s));


	segments.erase(s);

	Point *a = points[p_id];
	Point *b = points[p_with_id];
	a->neighbours.erase(b);
	b->neighbours.erase(a);

}
bool AStar::are_points_connected(int p_id,int p_with_id) const{

	Segment s(p_id,p_with_id);
	return segments.has(s);
}

void AStar::clear(){

	for (const Map<int,Point*>::Element *E=points.front();E;E=E->next()) {

		memdelete(E->get());
	}
	segments.clear();
	points.clear();
}


int AStar::get_closest_point(const Vector3& p_point) const{

	int closest_id=-1;
	real_t closest_dist=1e20;

	for (const Map<int,Point*>::Element *E=points.front();E;E=E->next()) {

		real_t d = p_point.distance_squared_to(E->get()->pos);
		if (closest_id<0 || d<closest_dist) {
			closest_dist=d;
			closest_id=E->key();
		}
	}

	return closest_id;


}
Vector3 AStar::get_closest_pos_in_segment(const Vector3& p_point) const {

	real_t closest_dist = 1e20;
	bool found=false;
	Vector3 closest_point;


	for (const Set<Segment>::Element *E=segments.front();E;E=E->next()) {

		Vector3 segment[2]={
			E->get().from_point->pos,
			E->get().to_point->pos,
		};

		Vector3 p = Geometry::get_closest_point_to_segment(p_point,segment);
		real_t d = p_point.distance_squared_to(p);
		if (!found || d<closest_dist) {

			closest_point=p;
			closest_dist=d;
			found=true;
		}
	}

	return closest_point;
}

bool AStar::_solve(Point* begin_point, Point* end_point, bool exhaustive) {

	pass++;

	SelfList<Point>::List open_list;

	bool found_route=false;

	for(int i=0;i<begin_point->neighbours.size();i++) {

		Point *n = begin_point->neighbours[i];
		n->prev_point=begin_point;
		n->distance = _compute_cost(n, begin_point, "compute_cost");
		n->distance *= n->weight_scale;
		n->last_pass=pass;
		open_list.add(&n->list);

		if (end_point==n) {
			found_route=true;
			if (!exhaustive)
				break;
		}
	}

	while(!found_route || exhaustive) {

		if (open_list.first()==NULL) {
			//found_route indicates if a route has been found
			break;
		}
		//check open list

		SelfList<Point> *least_cost_point=NULL;

		if (exhaustive) {
			// exhaustive search doesnt care about looking down the least cost path first
			// as it will search all paths for the best route
			least_cost_point = open_list.first();
		}
		else {
			//this could be faster (cache previous results)
			real_t least_cost = 1e30;
			for (SelfList<Point> *E=open_list.first();E;E=E->next()) {

				Point *p=E->self();

				real_t cost = p->distance;
				cost += _compute_cost(p, end_point, "estimate_cost"); // estimate as code might need to know the difference
				cost *= p->weight_scale;

				if (cost<least_cost) {

					least_cost_point=E;
					least_cost=cost;
				}
			}
		}


		Point *p=least_cost_point->self();
		//open the neighbours for search
		int es = p->neighbours.size();

		for(int i=0;i<es;i++) {


			Point* e=p->neighbours[i];


			real_t distance = _compute_cost(p, e, "compute_cost") + p->distance;
			distance *= e->weight_scale;



			if (e->last_pass==pass) {
				//oh this was visited already, can we win the cost?

				if (e->distance>distance) {

					e->prev_point=p;
					e->distance=distance;
				}
			} else {
				//add to open neighbours

				e->prev_point=p;
				e->distance=distance;
				e->last_pass=pass; //mark as used

				if (e==end_point) {
					//oh my reached end! stop algorithm if not exhaustive
					found_route=true;
					if (!exhaustive)
						break;
				}
				else {
					// dont put end point into the open list
					open_list.add(&e->list);
				}
			}
		}

		open_list.remove(least_cost_point);
	}

	//clear the openf list
	while(open_list.first()) {
		open_list.remove( open_list.first() );
	}

	return found_route;

}

real_t AStar::_compute_cost(Point *from, Point *to, const StringName& p_method) {
	if (delegate && delegate->has_method(p_method)) {

		Dictionary f;
		f["id"]=from->id;
		f["pos"]=from->pos;
		f["weight_scale"]=from->weight_scale;
		f["binds"]=from->binds;
		f["flags"]=from->flags;
		Variant fv(f);

		Dictionary t;
		t["id"]=to->id;
		t["pos"]=to->pos;
		t["weight_scale"]=to->weight_scale;
		t["binds"]=to->binds;
		t["flags"]=to->flags;
		Variant tv(t);

		Variant self(this);

		const Variant* args[] = { &self, &fv, &tv };
		Variant::CallError ce;
		real_t cost=delegate->call(p_method, args, 3, ce);

		return cost;
	}

	return from->pos.distance_to(to->pos);
}

PoolVector<Vector3> AStar::get_point_path(int p_from_id, int p_to_id, bool exhaustive) {

	ERR_FAIL_COND_V(!points.has(p_from_id),PoolVector<Vector3>());
	ERR_FAIL_COND_V(!points.has(p_to_id),PoolVector<Vector3>());

	pass++;

	Point* a = points[p_from_id];
	Point* b = points[p_to_id];

	if (a==b) {
		PoolVector<Vector3> ret;
		ret.push_back(a->pos);
		return ret;
	}


	Point *begin_point=a;
	Point *end_point=b;

	bool found_route=_solve(begin_point,end_point, exhaustive);

	if (!found_route)
		return PoolVector<Vector3>();

	//midpoints
	Point *p=end_point;
	int pc=1; //begin point
	while(p!=begin_point) {
		if (pc > points.size()) {
			ERR_PRINT("Infinite loop detected in AStar");
			return PoolVector<Vector3>();
		}

		pc++;
		p=p->prev_point;
	}

	PoolVector<Vector3> path;
	path.resize(pc);

	{
		PoolVector<Vector3>::Write w = path.write();

		Point *p=end_point;
		int idx=pc-1;
		while(p!=begin_point) {
			w[idx--]=p->pos;
			p=p->prev_point;
		}

		w[0]=p->pos; //assign first

	}

	return path;

}


PoolVector<int> AStar::get_id_path(int p_from_id, int p_to_id, bool exhaustive) {

	ERR_FAIL_COND_V(!points.has(p_from_id),PoolVector<int>());
	ERR_FAIL_COND_V(!points.has(p_to_id),PoolVector<int>());

	pass++;

	Point* a = points[p_from_id];
	Point* b = points[p_to_id];

	if (a==b) {
		PoolVector<int> ret;
		ret.push_back(a->id);
		return ret;
	}


	Point *begin_point=a;
	Point *end_point=b;

	bool found_route=_solve(begin_point,end_point, exhaustive);

	if (!found_route)
		return PoolVector<int>();

	//midpoints
	Point *p=end_point;
	int pc=1; //begin point
	while(p!=begin_point) {
		if (pc > points.size()) {
			ERR_PRINT("Infinite loop detected in AStar");
			return PoolVector<int>();
		}

		pc++;
		p=p->prev_point;
	}

	PoolVector<int> path;
	path.resize(pc);

	{
		PoolVector<int>::Write w = path.write();

		p=end_point;
		int idx=pc-1;
		while(p!=begin_point) {
			w[idx--]=p->id;
			p=p->prev_point;
		}

		w[0]=p->id; //assign first

	}

	return path;
}

void AStar::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_delegate"),&AStar::set_delegate);
	ClassDB::bind_method(D_METHOD("get_delegate"),&AStar::get_delegate);

	ClassDB::bind_method(D_METHOD("get_available_point_id"),&AStar::get_available_point_id);
	ClassDB::bind_method(D_METHOD("add_point","id","pos","weight_scale", "binds", "flags"),&AStar::add_point,DEFVAL(1.0), DEFVAL(Vector<Variant>()), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_point_pos","id"),&AStar::get_point_pos);
	ClassDB::bind_method(D_METHOD("set_point_pos","id","pos"),&AStar::set_point_pos);
	ClassDB::bind_method(D_METHOD("get_point_weight_scale","id"),&AStar::get_point_weight_scale);
	ClassDB::bind_method(D_METHOD("set_point_weight_scale","id","weight_scale"),&AStar::set_point_weight_scale);
	ClassDB::bind_method(D_METHOD("set_point_binds","id","binds"),&AStar::set_point_binds);
	ClassDB::bind_method(D_METHOD("get_point_binds","id"),&AStar::get_point_binds);
	ClassDB::bind_method(D_METHOD("set_point_flags","id","flags"),&AStar::set_point_flags);
	ClassDB::bind_method(D_METHOD("get_point_flags","id"),&AStar::get_point_flags);
	ClassDB::bind_method(D_METHOD("remove_point","id"),&AStar::remove_point);

	ClassDB::bind_method(D_METHOD("connect_points","id","to_id"),&AStar::connect_points,DEFVAL(true));
	ClassDB::bind_method(D_METHOD("disconnect_points","id","to_id"),&AStar::disconnect_points);
	ClassDB::bind_method(D_METHOD("are_points_connected","id","to_id"),&AStar::are_points_connected);

	ClassDB::bind_method(D_METHOD("clear"),&AStar::clear);

	ClassDB::bind_method(D_METHOD("get_closest_point","to_pos"),&AStar::get_closest_point);
	ClassDB::bind_method(D_METHOD("get_closest_pos_in_segment","to_pos"),&AStar::get_closest_pos_in_segment);

	ClassDB::bind_method(D_METHOD("get_point_path","from_id","to_id", "exhaustive"),&AStar::get_point_path,DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_id_path","from_id","to_id", "exhaustive"),&AStar::get_id_path,DEFVAL(false));

}


AStar::AStar() {

	pass=1;
	delegate = NULL;
}


AStar::~AStar() {

	pass=1;
}
