/*************************************************************************/
/*  curve.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "curve.h"
#include "core_string_names.h"

template <class T>
static _FORCE_INLINE_ T _bezier_interp(real_t t, T start, T control_1, T control_2, T end) {
	/* Formula from Wikipedia article on Bezier curves. */
	real_t omt = (1.0 - t);
	real_t omt2 = omt * omt;
	real_t omt3 = omt2 * omt;
	real_t t2 = t * t;
	real_t t3 = t2 * t;

	return start * omt3 + control_1 * omt2 * t * 3.0 + control_2 * omt * t2 * 3.0 + end * t3;
}

#if 0

int Curve2D::get_point_count() const {

	return points.size();
}
void Curve2D::add_point(const Vector2& p_pos, const Vector2& p_in, const Vector2& p_out) {

	Point n;
	n.pos=p_pos;
	n.in=p_in;
	n.out=p_out;
	points.push_back(n);
	emit_signal(CoreStringNames::get_singleton()->changed);
}
void Curve2D::set_point_pos(int p_index, const Vector2& p_pos) {

	ERR_FAIL_INDEX(p_index,points.size());

	points[p_index].pos=p_pos;
	emit_signal(CoreStringNames::get_singleton()->changed);

}
Vector2 Curve2D::get_point_pos(int p_index) const {

	ERR_FAIL_INDEX_V(p_index,points.size(),Vector2());
	return points[p_index].pos;

}


void Curve2D::set_point_in(int p_index, const Vector2& p_in) {

	ERR_FAIL_INDEX(p_index,points.size());

	points[p_index].in=p_in;
	emit_signal(CoreStringNames::get_singleton()->changed);

}
Vector2 Curve2D::get_point_in(int p_index) const {

	ERR_FAIL_INDEX_V(p_index,points.size(),Vector2());
	return points[p_index].in;

}

void Curve2D::set_point_out(int p_index, const Vector2& p_out) {

	ERR_FAIL_INDEX(p_index,points.size());

	points[p_index].out=p_out;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector2 Curve2D::get_point_out(int p_index) const {

	ERR_FAIL_INDEX_V(p_index,points.size(),Vector2());
	return points[p_index].out;

}


void Curve2D::remove_point(int p_index) {

	ERR_FAIL_INDEX(p_index,points.size());
	points.remove(p_index);
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector2 Curve2D::interpolate(int p_index, float p_offset) const {

	int pc = points.size();
	ERR_FAIL_COND_V(pc==0,Vector2());

	if (p_index >= pc-1)
		return points[pc-1].pos;
	else if (p_index<0)
		return points[0].pos;

	Vector2 p0 = points[p_index].pos;
	Vector2 p1 = p0+points[p_index].out;
	Vector2 p3 = points[p_index+1].pos;
	Vector2 p2 = p3+points[p_index+1].in;

	return _bezier_interp(p_offset,p0,p1,p2,p3);
}

Vector2 Curve2D::interpolatef(real_t p_findex) const {


	if (p_findex<0)
		p_findex=0;
	else if (p_findex>=points.size())
		p_findex=points.size();

	return interpolate((int)p_findex,Math::fmod(p_findex,1.0));

}

PoolVector<Point2> Curve2D::bake(int p_subdivs) const {

	int pc = points.size();

	PoolVector<Point2> ret;
	if (pc<2)
		return ret;

	ret.resize((pc-1)*p_subdivs+1);

	PoolVector<Point2>::Write w = ret.write();
	const Point *r = points.ptr();

	for(int i=0;i<pc;i++) {

		int ofs = pc*p_subdivs;

		int limit=(i==pc-1)?p_subdivs+1:p_subdivs;

		for(int j=0;j<limit;j++) {

			Vector2 p0 = r[i].pos;
			Vector2 p1 = p0+r[i].out;
			Vector2 p3 = r[i].pos;
			Vector2 p2 = p3+r[i].in;
			real_t t = j/(real_t)p_subdivs;

			w[ofs+j]=_bezier_interp(t,p0,p1,p2,p3);

		}
	}

	w = PoolVector<Point2>::Write();

	return ret;
}

void Curve2D::advance(real_t p_distance,int &r_index, real_t &r_pos) const {

	int pc = points.size();
	ERR_FAIL_COND(pc<2);
	if (r_index<0 || r_index>=(pc-1))
		return;

	Vector2 pos = interpolate(r_index,r_pos);

	float sign=p_distance<0 ? -1 : 1;
	p_distance=Math::abs(p_distance);

	real_t base = r_index+r_pos;
	real_t top = 0.1; //a tenth is in theory representative
	int iterations=32;



	for(int i=0;i<iterations;i++) {


		real_t o=base+top*sign;
		if (sign>0 && o >=pc) {
			top=pc-base;
			break;
		} else if (sign<0 && o <0) {
			top=-base;
			break;
		}

		Vector2 new_d = interpolatef(o);

		if (new_d.distance_to(pos) > p_distance)
			break;
		top*=2.0;
	}


	real_t bottom = 0.0;
	iterations=8;
	real_t final_offset;


	for(int i=0;i<iterations;i++) {

		real_t middle = (bottom+top)*0.5;
		real_t o=base+middle*sign;
		Vector2 new_d = interpolatef(o);

		if (new_d.distance_to(pos) > p_distance) {
			bottom=middle;
		} else {
			top=middle;
		}
		final_offset=o;
	}

	r_index=(int)final_offset;
	r_pos=Math::fmod(final_offset,1.0);

}

void Curve2D::get_approx_position_from_offset(real_t p_offset,int &r_index, real_t &r_pos,int p_subdivs) const {

	ERR_FAIL_COND(points.size()<2);

	real_t accum=0;



	for(int i=0;i<points.size();i++) {

		Vector2 prev_p=interpolate(i,0);


		for(int j=1;j<=p_subdivs;j++) {

			real_t frac = j/(real_t)p_subdivs;
			Vector2 p = interpolate(i,frac);
			real_t d = p.distance_to(prev_p);

			accum+=d;
			if (accum>p_offset) {


				r_index=j-1;
				if (d>0) {
					real_t mf = (p_offset-(accum-d)) / d;
					r_pos=frac-(1.0-mf);
				} else {
					r_pos=frac;
				}

				return;
			}

			prev_p=p;
		}
	}

	r_index=points.size()-1;
	r_pos=1.0;


}

void Curve2D::set_points_in(const Vector2Array& p_points) {

	points.resize(p_points.size());
	for (int i=0; i<p_points.size(); i++) {

		Point p = points[i];
		p.in = p_points[i];
		points[i] = p;
	};
};

void Curve2D::set_points_out(const Vector2Array& p_points) {

	points.resize(p_points.size());
	for (int i=0; i<p_points.size(); i++) {

		Point p = points[i];
		p.out = p_points[i];
		points[i] = p;
	};
};

void Curve2D::set_points_pos(const Vector2Array& p_points) {

	points.resize(p_points.size());
	for (int i=0; i<p_points.size(); i++) {

		Point p = points[i];
		p.pos = p_points[i];
		points[i] = p;
	};
};

Vector2Array Curve2D::get_points_in() const {
	Vector2Array ret;
	ret.resize(points.size());
	for (int i=0; i<points.size(); i++) {
		ret.set(i, points[i].in);
	};
	return ret;
};

Vector2Array Curve2D::get_points_out() const {
	Vector2Array ret;
	ret.resize(points.size());
	for (int i=0; i<points.size(); i++) {
		ret.set(i, points[i].out);
	};
	return ret;
};

Vector2Array Curve2D::get_points_pos() const {
	Vector2Array ret;
	ret.resize(points.size());
	for (int i=0; i<points.size(); i++) {
		ret.set(i, points[i].pos);
	};
	return ret;
};


void Curve2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_point_count"),&Curve2D::get_point_count);
	ClassDB::bind_method(D_METHOD("add_point","pos","in","out"),&Curve2D::add_point,DEFVAL(Vector2()),DEFVAL(Vector2()));
	ClassDB::bind_method(D_METHOD("set_point_pos","idx","pos"),&Curve2D::set_point_pos);
	ClassDB::bind_method(D_METHOD("get_point_pos","idx"),&Curve2D::get_point_pos);
	ClassDB::bind_method(D_METHOD("set_point_in","idx","pos"),&Curve2D::set_point_in);
	ClassDB::bind_method(D_METHOD("get_point_in","idx"),&Curve2D::get_point_in);
	ClassDB::bind_method(D_METHOD("set_point_out","idx","pos"),&Curve2D::set_point_out);
	ClassDB::bind_method(D_METHOD("get_point_out","idx"),&Curve2D::get_point_out);
	ClassDB::bind_method(D_METHOD("remove_point","idx"),&Curve2D::remove_point);
	ClassDB::bind_method(D_METHOD("interpolate","idx","t"),&Curve2D::interpolate);
	ClassDB::bind_method(D_METHOD("bake","subdivs"),&Curve2D::bake,DEFVAL(10));


	ClassDB::bind_method(D_METHOD("set_points_in"),&Curve2D::set_points_in);
	ClassDB::bind_method(D_METHOD("set_points_out"),&Curve2D::set_points_out);
	ClassDB::bind_method(D_METHOD("set_points_pos"),&Curve2D::set_points_pos);

	ClassDB::bind_method(D_METHOD("get_points_in"),&Curve2D::get_points_in);
	ClassDB::bind_method(D_METHOD("get_points_out"),&Curve2D::get_points_out);
	ClassDB::bind_method(D_METHOD("get_points_pos"),&Curve2D::get_points_pos);

	ADD_PROPERTY( PropertyInfo( Variant::VECTOR2_ARRAY, "points_in"), "set_points_in","get_points_in");
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR2_ARRAY, "points_out"), "set_points_out","get_points_out");
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR2_ARRAY, "points_pos"), "set_points_pos","get_points_pos");
}


Curve2D::Curve2D()
{
}

#endif

int Curve2D::get_point_count() const {

	return points.size();
}
void Curve2D::add_point(const Vector2 &p_pos, const Vector2 &p_in, const Vector2 &p_out, int p_atpos) {

	Point n;
	n.pos = p_pos;
	n.in = p_in;
	n.out = p_out;
	if (p_atpos >= 0 && p_atpos < points.size())
		points.insert(p_atpos, n);
	else
		points.push_back(n);

	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Curve2D::set_point_pos(int p_index, const Vector2 &p_pos) {

	ERR_FAIL_INDEX(p_index, points.size());

	points[p_index].pos = p_pos;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Vector2 Curve2D::get_point_pos(int p_index) const {

	ERR_FAIL_INDEX_V(p_index, points.size(), Vector2());
	return points[p_index].pos;
}

void Curve2D::set_point_in(int p_index, const Vector2 &p_in) {

	ERR_FAIL_INDEX(p_index, points.size());

	points[p_index].in = p_in;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Vector2 Curve2D::get_point_in(int p_index) const {

	ERR_FAIL_INDEX_V(p_index, points.size(), Vector2());
	return points[p_index].in;
}

void Curve2D::set_point_out(int p_index, const Vector2 &p_out) {

	ERR_FAIL_INDEX(p_index, points.size());

	points[p_index].out = p_out;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector2 Curve2D::get_point_out(int p_index) const {

	ERR_FAIL_INDEX_V(p_index, points.size(), Vector2());
	return points[p_index].out;
}

void Curve2D::remove_point(int p_index) {

	ERR_FAIL_INDEX(p_index, points.size());
	points.remove(p_index);
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Curve2D::clear_points() {
	if (!points.empty()) {
		points.clear();
		baked_cache_dirty = true;
		emit_signal(CoreStringNames::get_singleton()->changed);
	}
}

Vector2 Curve2D::interpolate(int p_index, float p_offset) const {

	int pc = points.size();
	ERR_FAIL_COND_V(pc == 0, Vector2());

	if (p_index >= pc - 1)
		return points[pc - 1].pos;
	else if (p_index < 0)
		return points[0].pos;

	Vector2 p0 = points[p_index].pos;
	Vector2 p1 = p0 + points[p_index].out;
	Vector2 p3 = points[p_index + 1].pos;
	Vector2 p2 = p3 + points[p_index + 1].in;

	return _bezier_interp(p_offset, p0, p1, p2, p3);
}

Vector2 Curve2D::interpolatef(real_t p_findex) const {

	if (p_findex < 0)
		p_findex = 0;
	else if (p_findex >= points.size())
		p_findex = points.size();

	return interpolate((int)p_findex, Math::fmod(p_findex, (real_t)1.0));
}

void Curve2D::_bake_segment2d(Map<float, Vector2> &r_bake, float p_begin, float p_end, const Vector2 &p_a, const Vector2 &p_out, const Vector2 &p_b, const Vector2 &p_in, int p_depth, int p_max_depth, float p_tol) const {

	float mp = p_begin + (p_end - p_begin) * 0.5;
	Vector2 beg = _bezier_interp(p_begin, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector2 mid = _bezier_interp(mp, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector2 end = _bezier_interp(p_end, p_a, p_a + p_out, p_b + p_in, p_b);

	Vector2 na = (mid - beg).normalized();
	Vector2 nb = (end - mid).normalized();
	float dp = na.dot(nb);

	if (dp < Math::cos(Math::deg2rad(p_tol))) {

		r_bake[mp] = mid;
	}

	if (p_depth < p_max_depth) {
		_bake_segment2d(r_bake, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
		_bake_segment2d(r_bake, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
	}
}

void Curve2D::_bake() const {

	if (!baked_cache_dirty)
		return;

	baked_max_ofs = 0;
	baked_cache_dirty = false;

	if (points.size() == 0) {
		baked_point_cache.resize(0);
		return;
	}

	if (points.size() == 1) {

		baked_point_cache.resize(1);
		baked_point_cache.set(0, points[0].pos);
		return;
	}

	Vector2 pos = points[0].pos;
	List<Vector2> pointlist;

	pointlist.push_back(pos); //start always from origin

	for (int i = 0; i < points.size() - 1; i++) {

		float step = 0.1; // at least 10 substeps ought to be enough?
		float p = 0;

		while (p < 1.0) {

			float np = p + step;
			if (np > 1.0)
				np = 1.0;

			Vector2 npp = _bezier_interp(np, points[i].pos, points[i].pos + points[i].out, points[i + 1].pos + points[i + 1].in, points[i + 1].pos);
			float d = pos.distance_to(npp);

			if (d > bake_interval) {
				// OK! between P and NP there _has_ to be Something, let's go searching!

				int iterations = 10; //lots of detail!

				float low = p;
				float hi = np;
				float mid = low + (hi - low) * 0.5;

				for (int j = 0; j < iterations; j++) {

					npp = _bezier_interp(mid, points[i].pos, points[i].pos + points[i].out, points[i + 1].pos + points[i + 1].in, points[i + 1].pos);
					d = pos.distance_to(npp);

					if (bake_interval < d)
						hi = mid;
					else
						low = mid;
					mid = low + (hi - low) * 0.5;
				}

				pos = npp;
				p = mid;
				pointlist.push_back(pos);
			} else {

				p = np;
			}
		}
	}

	Vector2 lastpos = points[points.size() - 1].pos;

	float rem = pos.distance_to(lastpos);
	baked_max_ofs = (pointlist.size() - 1) * bake_interval + rem;
	pointlist.push_back(lastpos);

	baked_point_cache.resize(pointlist.size());
	PoolVector2Array::Write w = baked_point_cache.write();
	int idx = 0;

	for (List<Vector2>::Element *E = pointlist.front(); E; E = E->next()) {

		w[idx] = E->get();
		idx++;
	}
}

float Curve2D::get_baked_length() const {

	if (baked_cache_dirty)
		_bake();

	return baked_max_ofs;
}
Vector2 Curve2D::interpolate_baked(float p_offset, bool p_cubic) const {

	if (baked_cache_dirty)
		_bake();

	//validate//
	int pc = baked_point_cache.size();
	if (pc == 0) {
		ERR_EXPLAIN("No points in Curve2D");
		ERR_FAIL_COND_V(pc == 0, Vector2());
	}

	if (pc == 1)
		return baked_point_cache.get(0);

	int bpc = baked_point_cache.size();
	PoolVector2Array::Read r = baked_point_cache.read();

	if (p_offset < 0)
		return r[0];
	if (p_offset >= baked_max_ofs)
		return r[bpc - 1];

	int idx = Math::floor((double)p_offset / (double)bake_interval);
	float frac = Math::fmod(p_offset, (float)bake_interval);

	if (idx >= bpc - 1) {
		return r[bpc - 1];
	} else if (idx == bpc - 2) {
		frac /= Math::fmod(baked_max_ofs, bake_interval);
	} else {
		frac /= bake_interval;
	}

	if (p_cubic) {

		Vector2 pre = idx > 0 ? r[idx - 1] : r[idx];
		Vector2 post = (idx < (bpc - 2)) ? r[idx + 2] : r[idx + 1];
		return r[idx].cubic_interpolate(r[idx + 1], pre, post, frac);
	} else {
		return r[idx].linear_interpolate(r[idx + 1], frac);
	}
}

PoolVector2Array Curve2D::get_baked_points() const {

	if (baked_cache_dirty)
		_bake();

	return baked_point_cache;
}

void Curve2D::set_bake_interval(float p_tolerance) {

	bake_interval = p_tolerance;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

float Curve2D::get_bake_interval() const {

	return bake_interval;
}

Dictionary Curve2D::_get_data() const {

	Dictionary dc;

	PoolVector2Array d;
	d.resize(points.size() * 3);
	PoolVector2Array::Write w = d.write();

	for (int i = 0; i < points.size(); i++) {

		w[i * 3 + 0] = points[i].in;
		w[i * 3 + 1] = points[i].out;
		w[i * 3 + 2] = points[i].pos;
	}

	w = PoolVector2Array::Write();

	dc["points"] = d;

	return dc;
}
void Curve2D::_set_data(const Dictionary &p_data) {

	ERR_FAIL_COND(!p_data.has("points"));

	PoolVector2Array rp = p_data["points"];
	int pc = rp.size();
	ERR_FAIL_COND(pc % 3 != 0);
	points.resize(pc / 3);
	PoolVector2Array::Read r = rp.read();

	for (int i = 0; i < points.size(); i++) {

		points[i].in = r[i * 3 + 0];
		points[i].out = r[i * 3 + 1];
		points[i].pos = r[i * 3 + 2];
	}

	baked_cache_dirty = true;
}

PoolVector2Array Curve2D::tesselate(int p_max_stages, float p_tolerance) const {

	PoolVector2Array tess;

	if (points.size() == 0) {
		return tess;
	}
	Vector<Map<float, Vector2> > midpoints;

	midpoints.resize(points.size() - 1);

	int pc = 1;
	for (int i = 0; i < points.size() - 1; i++) {

		_bake_segment2d(midpoints[i], 0, 1, points[i].pos, points[i].out, points[i + 1].pos, points[i + 1].in, 0, p_max_stages, p_tolerance);
		pc++;
		pc += midpoints[i].size();
	}

	tess.resize(pc);
	PoolVector2Array::Write bpw = tess.write();
	bpw[0] = points[0].pos;
	int pidx = 0;

	for (int i = 0; i < points.size() - 1; i++) {

		for (Map<float, Vector2>::Element *E = midpoints[i].front(); E; E = E->next()) {

			pidx++;
			bpw[pidx] = E->get();
		}

		pidx++;
		bpw[pidx] = points[i + 1].pos;
	}

	bpw = PoolVector2Array::Write();

	return tess;
}

void Curve2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_point_count"), &Curve2D::get_point_count);
	ClassDB::bind_method(D_METHOD("add_point", "pos", "in", "out", "atpos"), &Curve2D::add_point, DEFVAL(Vector2()), DEFVAL(Vector2()), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("set_point_pos", "idx", "pos"), &Curve2D::set_point_pos);
	ClassDB::bind_method(D_METHOD("get_point_pos", "idx"), &Curve2D::get_point_pos);
	ClassDB::bind_method(D_METHOD("set_point_in", "idx", "pos"), &Curve2D::set_point_in);
	ClassDB::bind_method(D_METHOD("get_point_in", "idx"), &Curve2D::get_point_in);
	ClassDB::bind_method(D_METHOD("set_point_out", "idx", "pos"), &Curve2D::set_point_out);
	ClassDB::bind_method(D_METHOD("get_point_out", "idx"), &Curve2D::get_point_out);
	ClassDB::bind_method(D_METHOD("remove_point", "idx"), &Curve2D::remove_point);
	ClassDB::bind_method(D_METHOD("clear_points"), &Curve2D::clear_points);
	ClassDB::bind_method(D_METHOD("interpolate", "idx", "t"), &Curve2D::interpolate);
	ClassDB::bind_method(D_METHOD("interpolatef", "fofs"), &Curve2D::interpolatef);
	//ClassDB::bind_method(D_METHOD("bake","subdivs"),&Curve2D::bake,DEFVAL(10));
	ClassDB::bind_method(D_METHOD("set_bake_interval", "distance"), &Curve2D::set_bake_interval);
	ClassDB::bind_method(D_METHOD("get_bake_interval"), &Curve2D::get_bake_interval);

	ClassDB::bind_method(D_METHOD("get_baked_length"), &Curve2D::get_baked_length);
	ClassDB::bind_method(D_METHOD("interpolate_baked", "offset", "cubic"), &Curve2D::interpolate_baked, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_baked_points"), &Curve2D::get_baked_points);
	ClassDB::bind_method(D_METHOD("tesselate", "max_stages", "tolerance_degrees"), &Curve2D::tesselate, DEFVAL(5), DEFVAL(4));

	ClassDB::bind_method(D_METHOD("_get_data"), &Curve2D::_get_data);
	ClassDB::bind_method(D_METHOD("_set_data"), &Curve2D::_set_data);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_interval", PROPERTY_HINT_RANGE, "0.01,512,0.01"), "set_bake_interval", "get_bake_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_data", "_get_data");
	/*ADD_PROPERTY( PropertyInfo( Variant::VECTOR3_ARRAY, "points_out"), "set_points_out","get_points_out");
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR3_ARRAY, "points_pos"), "set_points_pos","get_points_pos");
*/
}

Curve2D::Curve2D() {
	baked_cache_dirty = false;
	baked_max_ofs = 0;
	/*	add_point(Vector2(-1,0,0));
	add_point(Vector2(0,2,0));
	add_point(Vector2(0,3,5));*/
	bake_interval = 5;
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

int Curve3D::get_point_count() const {

	return points.size();
}
void Curve3D::add_point(const Vector3 &p_pos, const Vector3 &p_in, const Vector3 &p_out, int p_atpos) {

	Point n;
	n.pos = p_pos;
	n.in = p_in;
	n.out = p_out;
	if (p_atpos >= 0 && p_atpos < points.size())
		points.insert(p_atpos, n);
	else
		points.push_back(n);

	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
void Curve3D::set_point_pos(int p_index, const Vector3 &p_pos) {

	ERR_FAIL_INDEX(p_index, points.size());

	points[p_index].pos = p_pos;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Vector3 Curve3D::get_point_pos(int p_index) const {

	ERR_FAIL_INDEX_V(p_index, points.size(), Vector3());
	return points[p_index].pos;
}

void Curve3D::set_point_tilt(int p_index, float p_tilt) {

	ERR_FAIL_INDEX(p_index, points.size());

	points[p_index].tilt = p_tilt;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
float Curve3D::get_point_tilt(int p_index) const {

	ERR_FAIL_INDEX_V(p_index, points.size(), 0);
	return points[p_index].tilt;
}

void Curve3D::set_point_in(int p_index, const Vector3 &p_in) {

	ERR_FAIL_INDEX(p_index, points.size());

	points[p_index].in = p_in;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}
Vector3 Curve3D::get_point_in(int p_index) const {

	ERR_FAIL_INDEX_V(p_index, points.size(), Vector3());
	return points[p_index].in;
}

void Curve3D::set_point_out(int p_index, const Vector3 &p_out) {

	ERR_FAIL_INDEX(p_index, points.size());

	points[p_index].out = p_out;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

Vector3 Curve3D::get_point_out(int p_index) const {

	ERR_FAIL_INDEX_V(p_index, points.size(), Vector3());
	return points[p_index].out;
}

void Curve3D::remove_point(int p_index) {

	ERR_FAIL_INDEX(p_index, points.size());
	points.remove(p_index);
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Curve3D::clear_points() {

	if (!points.empty()) {
		points.clear();
		baked_cache_dirty = true;
		emit_signal(CoreStringNames::get_singleton()->changed);
	}
}

Vector3 Curve3D::interpolate(int p_index, float p_offset) const {

	int pc = points.size();
	ERR_FAIL_COND_V(pc == 0, Vector3());

	if (p_index >= pc - 1)
		return points[pc - 1].pos;
	else if (p_index < 0)
		return points[0].pos;

	Vector3 p0 = points[p_index].pos;
	Vector3 p1 = p0 + points[p_index].out;
	Vector3 p3 = points[p_index + 1].pos;
	Vector3 p2 = p3 + points[p_index + 1].in;

	return _bezier_interp(p_offset, p0, p1, p2, p3);
}

Vector3 Curve3D::interpolatef(real_t p_findex) const {

	if (p_findex < 0)
		p_findex = 0;
	else if (p_findex >= points.size())
		p_findex = points.size();

	return interpolate((int)p_findex, Math::fmod(p_findex, (real_t)1.0));
}

void Curve3D::_bake_segment3d(Map<float, Vector3> &r_bake, float p_begin, float p_end, const Vector3 &p_a, const Vector3 &p_out, const Vector3 &p_b, const Vector3 &p_in, int p_depth, int p_max_depth, float p_tol) const {

	float mp = p_begin + (p_end - p_begin) * 0.5;
	Vector3 beg = _bezier_interp(p_begin, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector3 mid = _bezier_interp(mp, p_a, p_a + p_out, p_b + p_in, p_b);
	Vector3 end = _bezier_interp(p_end, p_a, p_a + p_out, p_b + p_in, p_b);

	Vector3 na = (mid - beg).normalized();
	Vector3 nb = (end - mid).normalized();
	float dp = na.dot(nb);

	if (dp < Math::cos(Math::deg2rad(p_tol))) {

		r_bake[mp] = mid;
	}
	if (p_depth < p_max_depth) {
		_bake_segment3d(r_bake, p_begin, mp, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
		_bake_segment3d(r_bake, mp, p_end, p_a, p_out, p_b, p_in, p_depth + 1, p_max_depth, p_tol);
	}
}

void Curve3D::_bake() const {

	if (!baked_cache_dirty)
		return;

	baked_max_ofs = 0;
	baked_cache_dirty = false;

	if (points.size() == 0) {
		baked_point_cache.resize(0);
		baked_tilt_cache.resize(0);
		return;
	}

	if (points.size() == 1) {

		baked_point_cache.resize(1);
		baked_point_cache.set(0, points[0].pos);
		baked_tilt_cache.resize(1);
		baked_tilt_cache.set(0, points[0].tilt);
		return;
	}

	Vector3 pos = points[0].pos;
	List<Plane> pointlist;
	pointlist.push_back(Plane(pos, points[0].tilt));

	for (int i = 0; i < points.size() - 1; i++) {

		float step = 0.1; // at least 10 substeps ought to be enough?
		float p = 0;

		while (p < 1.0) {

			float np = p + step;
			if (np > 1.0)
				np = 1.0;

			Vector3 npp = _bezier_interp(np, points[i].pos, points[i].pos + points[i].out, points[i + 1].pos + points[i + 1].in, points[i + 1].pos);
			float d = pos.distance_to(npp);

			if (d > bake_interval) {
				// OK! between P and NP there _has_ to be Something, let's go searching!

				int iterations = 10; //lots of detail!

				float low = p;
				float hi = np;
				float mid = low + (hi - low) * 0.5;

				for (int j = 0; j < iterations; j++) {

					npp = _bezier_interp(mid, points[i].pos, points[i].pos + points[i].out, points[i + 1].pos + points[i + 1].in, points[i + 1].pos);
					d = pos.distance_to(npp);

					if (bake_interval < d)
						hi = mid;
					else
						low = mid;
					mid = low + (hi - low) * 0.5;
				}

				pos = npp;
				p = mid;
				Plane post;
				post.normal = pos;
				post.d = Math::lerp(points[i].tilt, points[i + 1].tilt, mid);
				pointlist.push_back(post);
			} else {

				p = np;
			}
		}
	}

	Vector3 lastpos = points[points.size() - 1].pos;
	float lastilt = points[points.size() - 1].tilt;

	float rem = pos.distance_to(lastpos);
	baked_max_ofs = (pointlist.size() - 1) * bake_interval + rem;
	pointlist.push_back(Plane(lastpos, lastilt));

	baked_point_cache.resize(pointlist.size());
	PoolVector3Array::Write w = baked_point_cache.write();
	int idx = 0;

	baked_tilt_cache.resize(pointlist.size());
	PoolRealArray::Write wt = baked_tilt_cache.write();

	for (List<Plane>::Element *E = pointlist.front(); E; E = E->next()) {

		w[idx] = E->get().normal;
		wt[idx] = E->get().d;
		idx++;
	}
}

float Curve3D::get_baked_length() const {

	if (baked_cache_dirty)
		_bake();

	return baked_max_ofs;
}
Vector3 Curve3D::interpolate_baked(float p_offset, bool p_cubic) const {

	if (baked_cache_dirty)
		_bake();

	//validate//
	int pc = baked_point_cache.size();
	if (pc == 0) {
		ERR_EXPLAIN("No points in Curve3D");
		ERR_FAIL_COND_V(pc == 0, Vector3());
	}

	if (pc == 1)
		return baked_point_cache.get(0);

	int bpc = baked_point_cache.size();
	PoolVector3Array::Read r = baked_point_cache.read();

	if (p_offset < 0)
		return r[0];
	if (p_offset >= baked_max_ofs)
		return r[bpc - 1];

	int idx = Math::floor((double)p_offset / (double)bake_interval);
	float frac = Math::fmod(p_offset, bake_interval);

	if (idx >= bpc - 1) {
		return r[bpc - 1];
	} else if (idx == bpc - 2) {
		frac /= Math::fmod(baked_max_ofs, bake_interval);
	} else {
		frac /= bake_interval;
	}

	if (p_cubic) {

		Vector3 pre = idx > 0 ? r[idx - 1] : r[idx];
		Vector3 post = (idx < (bpc - 2)) ? r[idx + 2] : r[idx + 1];
		return r[idx].cubic_interpolate(r[idx + 1], pre, post, frac);
	} else {
		return r[idx].linear_interpolate(r[idx + 1], frac);
	}
}

float Curve3D::interpolate_baked_tilt(float p_offset) const {

	if (baked_cache_dirty)
		_bake();

	//validate//
	int pc = baked_tilt_cache.size();
	if (pc == 0) {
		ERR_EXPLAIN("No tilts in Curve3D");
		ERR_FAIL_COND_V(pc == 0, 0);
	}

	if (pc == 1)
		return baked_tilt_cache.get(0);

	int bpc = baked_tilt_cache.size();
	PoolRealArray::Read r = baked_tilt_cache.read();

	if (p_offset < 0)
		return r[0];
	if (p_offset >= baked_max_ofs)
		return r[bpc - 1];

	int idx = Math::floor((double)p_offset / (double)bake_interval);
	float frac = Math::fmod(p_offset, bake_interval);

	if (idx >= bpc - 1) {
		return r[bpc - 1];
	} else if (idx == bpc - 2) {
		frac /= Math::fmod(baked_max_ofs, bake_interval);
	} else {
		frac /= bake_interval;
	}

	return Math::lerp(r[idx], r[idx + 1], frac);
}

PoolVector3Array Curve3D::get_baked_points() const {

	if (baked_cache_dirty)
		_bake();

	return baked_point_cache;
}

PoolRealArray Curve3D::get_baked_tilts() const {

	if (baked_cache_dirty)
		_bake();

	return baked_tilt_cache;
}

void Curve3D::set_bake_interval(float p_tolerance) {

	bake_interval = p_tolerance;
	baked_cache_dirty = true;
	emit_signal(CoreStringNames::get_singleton()->changed);
}

float Curve3D::get_bake_interval() const {

	return bake_interval;
}

Dictionary Curve3D::_get_data() const {

	Dictionary dc;

	PoolVector3Array d;
	d.resize(points.size() * 3);
	PoolVector3Array::Write w = d.write();
	PoolRealArray t;
	t.resize(points.size());
	PoolRealArray::Write wt = t.write();

	for (int i = 0; i < points.size(); i++) {

		w[i * 3 + 0] = points[i].in;
		w[i * 3 + 1] = points[i].out;
		w[i * 3 + 2] = points[i].pos;
		wt[i] = points[i].tilt;
	}

	w = PoolVector3Array::Write();
	wt = PoolRealArray::Write();

	dc["points"] = d;
	dc["tilts"] = t;

	return dc;
}
void Curve3D::_set_data(const Dictionary &p_data) {

	ERR_FAIL_COND(!p_data.has("points"));
	ERR_FAIL_COND(!p_data.has("tilts"));

	PoolVector3Array rp = p_data["points"];
	int pc = rp.size();
	ERR_FAIL_COND(pc % 3 != 0);
	points.resize(pc / 3);
	PoolVector3Array::Read r = rp.read();
	PoolRealArray rtl = p_data["tilts"];
	PoolRealArray::Read rt = rtl.read();

	for (int i = 0; i < points.size(); i++) {

		points[i].in = r[i * 3 + 0];
		points[i].out = r[i * 3 + 1];
		points[i].pos = r[i * 3 + 2];
		points[i].tilt = rt[i];
	}

	baked_cache_dirty = true;
}

PoolVector3Array Curve3D::tesselate(int p_max_stages, float p_tolerance) const {

	PoolVector3Array tess;

	if (points.size() == 0) {
		return tess;
	}
	Vector<Map<float, Vector3> > midpoints;

	midpoints.resize(points.size() - 1);

	int pc = 1;
	for (int i = 0; i < points.size() - 1; i++) {

		_bake_segment3d(midpoints[i], 0, 1, points[i].pos, points[i].out, points[i + 1].pos, points[i + 1].in, 0, p_max_stages, p_tolerance);
		pc++;
		pc += midpoints[i].size();
	}

	tess.resize(pc);
	PoolVector3Array::Write bpw = tess.write();
	bpw[0] = points[0].pos;
	int pidx = 0;

	for (int i = 0; i < points.size() - 1; i++) {

		for (Map<float, Vector3>::Element *E = midpoints[i].front(); E; E = E->next()) {

			pidx++;
			bpw[pidx] = E->get();
		}

		pidx++;
		bpw[pidx] = points[i + 1].pos;
	}

	bpw = PoolVector3Array::Write();

	return tess;
}

void Curve3D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_point_count"), &Curve3D::get_point_count);
	ClassDB::bind_method(D_METHOD("add_point", "pos", "in", "out", "atpos"), &Curve3D::add_point, DEFVAL(Vector3()), DEFVAL(Vector3()), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("set_point_pos", "idx", "pos"), &Curve3D::set_point_pos);
	ClassDB::bind_method(D_METHOD("get_point_pos", "idx"), &Curve3D::get_point_pos);
	ClassDB::bind_method(D_METHOD("set_point_tilt", "idx", "tilt"), &Curve3D::set_point_tilt);
	ClassDB::bind_method(D_METHOD("get_point_tilt", "idx"), &Curve3D::get_point_tilt);
	ClassDB::bind_method(D_METHOD("set_point_in", "idx", "pos"), &Curve3D::set_point_in);
	ClassDB::bind_method(D_METHOD("get_point_in", "idx"), &Curve3D::get_point_in);
	ClassDB::bind_method(D_METHOD("set_point_out", "idx", "pos"), &Curve3D::set_point_out);
	ClassDB::bind_method(D_METHOD("get_point_out", "idx"), &Curve3D::get_point_out);
	ClassDB::bind_method(D_METHOD("remove_point", "idx"), &Curve3D::remove_point);
	ClassDB::bind_method(D_METHOD("clear_points"), &Curve3D::clear_points);
	ClassDB::bind_method(D_METHOD("interpolate", "idx", "t"), &Curve3D::interpolate);
	ClassDB::bind_method(D_METHOD("interpolatef", "fofs"), &Curve3D::interpolatef);
	//ClassDB::bind_method(D_METHOD("bake","subdivs"),&Curve3D::bake,DEFVAL(10));
	ClassDB::bind_method(D_METHOD("set_bake_interval", "distance"), &Curve3D::set_bake_interval);
	ClassDB::bind_method(D_METHOD("get_bake_interval"), &Curve3D::get_bake_interval);

	ClassDB::bind_method(D_METHOD("get_baked_length"), &Curve3D::get_baked_length);
	ClassDB::bind_method(D_METHOD("interpolate_baked", "offset", "cubic"), &Curve3D::interpolate_baked, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_baked_points"), &Curve3D::get_baked_points);
	ClassDB::bind_method(D_METHOD("get_baked_tilts"), &Curve3D::get_baked_tilts);
	ClassDB::bind_method(D_METHOD("tesselate", "max_stages", "tolerance_degrees"), &Curve3D::tesselate, DEFVAL(5), DEFVAL(4));

	ClassDB::bind_method(D_METHOD("_get_data"), &Curve3D::_get_data);
	ClassDB::bind_method(D_METHOD("_set_data"), &Curve3D::_set_data);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_interval", PROPERTY_HINT_RANGE, "0.01,512,0.01"), "set_bake_interval", "get_bake_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_data", "_get_data");
	/*ADD_PROPERTY( PropertyInfo( Variant::VECTOR3_ARRAY, "points_out"), "set_points_out","get_points_out");
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR3_ARRAY, "points_pos"), "set_points_pos","get_points_pos");
*/
}

Curve3D::Curve3D() {
	baked_cache_dirty = false;
	baked_max_ofs = 0;
	/*	add_point(Vector3(-1,0,0));
	add_point(Vector3(0,2,0));
	add_point(Vector3(0,3,5));*/
	bake_interval = 0.2;
}
