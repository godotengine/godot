/*************************************************************************/
/*  spatial_indexer.h                                                    */
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
#ifndef SPATIAL_INDEXER_H
#define SPATIAL_INDEXER_H

#include "scene/3d/spatial.h"
#if 0

class Camera;
class ProximityArea;

class SpatialIndexer : public Object {

	GDCLASS( SpatialIndexer, Object );

	template<class T>
	struct TK {

		T *against;
		ProximityArea *area;
		bool operator<(const TK<T>& p_k) const { return against==p_k.against ? area < p_k.area : against < p_k.against; }
	};


	Set<Camera*> cameras; //cameras
	Set<ProximityArea*> proximity_areas;

	Set<TK<Camera> > camera_pairs;

	bool pending_update;
	void _update_pairs();
	void _request_update();

protected:

	static void _bind_methods();

friend class ProximityArea;
friend class Camera;

	void add_proximity_area(ProximityArea* p_area);
	void remove_proximity_area(ProximityArea* p_area);
	void update_proximity_area_transform(ProximityArea* p_area);
	void update_proximity_area_flags(ProximityArea* p_area);

	void add_camera(Camera* p_camera);
	void remove_camera(Camera* p_camera);
	void update_camera(Camera* p_camera);

public:


	SpatialIndexer();

};
#endif
#endif // SPATIAL_INDEXER_H
