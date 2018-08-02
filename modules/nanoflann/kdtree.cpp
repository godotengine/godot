/*************************************************************************/
/*  kdtree.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "kdtree.h"
#include <nanoflann.hpp>

using namespace nanoflann;

template <class U>
class KDTreeData {
	struct Adaptor {
		const Vector<U> *list;
		bool report_once = false;
		Adaptor(const Vector<U> *list) {
			this->list = list;
		}
		inline size_t kdtree_get_point_count() const {
			return list->size();
		}
		inline real_t kdtree_get_pt(const size_t idx, int dim) const;
		template <class BBOX>
		bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
	};
	typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<real_t, Adaptor>, Adaptor, 2> kdtree;
	Adaptor tdata;
	kdtree index;

public:
	KDTreeData(Vector<U> *list, int dims) :
			tdata(list),
			index(dims, tdata, KDTreeSingleIndexAdaptorParams(10)) {
		index.buildIndex();
	}
	void rebuild() {
		index.buildIndex();
	}
	std::vector<std::pair<size_t, real_t> > radius_search(const real_t *data, real_t radius) {
		std::vector<std::pair<size_t, real_t> > ret_matches;
		nanoflann::SearchParams params;
		size_t nmatches = index.radiusSearch(data, radius, ret_matches, params);
		return ret_matches;
	}
};
class KDTree2D::KDTreeData2D {
	KDTreeData<Vector2> *kddata;
	Vector<Vector2> list;

public:
	KDTreeData2D() {
		kddata = new KDTreeData<Vector2>(&list, 2);
	}
	~KDTreeData2D() {
		delete kddata;
	}
	Vector<Vector2> search(const Vector2 &coords, real_t radius);
	void add_point(const Vector2 &point) {
		list.push_back(point);
	}
	void add_points(const Vector<Vector2> &points) {
		list.append_array(points);
	}
	void rebuild() {
		delete kddata;
		kddata = new KDTreeData<Vector2>(&list, 2);
	}
};
class KDTree3D::KDTreeData3D {
	KDTreeData<Vector3> *kddata;
	Vector<Vector3> list;

public:
	KDTreeData3D() {
		kddata = new KDTreeData<Vector3>(&list, 3);
	}
	~KDTreeData3D() {
		delete kddata;
	}
	Vector<Vector3> search(const Vector3 &coords, real_t radius);
	void add_point(const Vector3 &point) {
		list.push_back(point);
	}
	void add_points(const Vector<Vector3> &points) {
		list.append_array(points);
	}
	void rebuild() {
		delete kddata;
		kddata = new KDTreeData<Vector3>(&list, 3);
	}
};
Vector<Vector2> KDTree2D::KDTreeData2D::search(const Vector2 &coords, real_t radius) {
	Vector<Vector2> ret;
	float search_data[] = { coords.x, coords.y };
	std::vector<std::pair<size_t, real_t> > ret_matches = kddata->radius_search(search_data, radius);
	ret.resize(ret_matches.size());
	for (int i = 0; i < ret_matches.size(); i++)
		ret.write[i] = list[ret_matches[i].first];
	return ret;
}
Vector<Vector3> KDTree3D::KDTreeData3D::search(const Vector3 &coords, real_t radius) {
	Vector<Vector3> ret;
	std::vector<std::pair<size_t, real_t> > ret_matches = kddata->radius_search(&coords.coord[0], radius);
	ret.resize(ret_matches.size());
	for (int i = 0; i < ret_matches.size(); i++)
		ret.write[i] = list[ret_matches[i].first];
	return ret;
}
KDTree2D::KDTree2D() {
	data = new KDTreeData2D();
}
void KDTree2D::rebuild() {
	data->rebuild();
}
Vector<Vector2> KDTree2D::search(const Vector2 &coords, real_t radius) {
	return data->search(coords, radius);
}
KDTree3D::KDTree3D() {
	data = new KDTreeData3D();
}
void KDTree3D::rebuild() {
	data->rebuild();
}
Vector<Vector3> KDTree3D::search(const Vector3 &coords, real_t radius) {
	return data->search(coords, radius);
}
void KDTree2D::add_point(const Vector2 &point) {
	data->add_point(point);
}
void KDTree3D::add_point(const Vector3 &point) {
	data->add_point(point);
}
void KDTree2D::add_points(const Vector<Vector2> &points) {
	data->add_points(points);
}
void KDTree3D::add_points(const Vector<Vector3> &points) {
	data->add_points(points);
}
KDTree2D::~KDTree2D() {
	delete data;
}
KDTree3D::~KDTree3D() {
	delete data;
}
template <>
inline real_t KDTreeData<Vector2>::Adaptor::kdtree_get_pt(const size_t idx, int dim) const {
	if (dim == 0)
		return list->get(idx).x;
	else if (dim == 1)
		return list->get(idx).y;
	return 0.0f;
}
template <>
inline real_t KDTreeData<Vector3>::Adaptor::kdtree_get_pt(const size_t idx, int dim) const {
	if (dim == 0)
		return list->get(idx).x;
	else if (dim == 1)
		return list->get(idx).y;
	else if (dim == 2)
		return list->get(idx).z;
	return 0.0f;
}
void KDTree2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("rebuild"), &KDTree2D::rebuild);
	ClassDB::bind_method(D_METHOD("search", "coords", "radius"), &KDTree2D::search);
	ClassDB::bind_method(D_METHOD("add_point", "point"), &KDTree2D::add_point);
	ClassDB::bind_method(D_METHOD("add_points", "points"), &KDTree2D::add_points);
}
void KDTree3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("rebuild"), &KDTree3D::rebuild);
	ClassDB::bind_method(D_METHOD("search", "coords", "radius"), &KDTree3D::search);
	ClassDB::bind_method(D_METHOD("add_point", "point"), &KDTree3D::add_point);
	ClassDB::bind_method(D_METHOD("add_points", "points"), &KDTree3D::add_points);
}
