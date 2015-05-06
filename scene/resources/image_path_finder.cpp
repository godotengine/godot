/*************************************************************************/
/*  image_path_finder.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "image_path_finder.h"


void ImagePathFinder::_unlock() {

	lock=DVector<Cell>::Write();
	cells=NULL;

}

void ImagePathFinder::_lock() {

	lock = cell_data.write();
	cells=lock.ptr();

}


bool ImagePathFinder::_can_go_straigth(const Point2& p_from, const Point2& p_to) const {

	int x1=p_from.x;
	int y1=p_from.y;
	int x2=p_to.x;
	int y2=p_to.y;

#define _TEST_VALID \
	{\
		uint32_t ofs=drawy*width+drawx;\
		if (cells[ofs].solid) {\
			if (!((drawx>0 && cells[ofs-1].visited) ||\
			    (drawx<width-1 && cells[ofs+1].visited) ||\
			    (drawy>0 && cells[ofs-width].visited) ||\
			    (drawy<height-1 && cells[ofs+width].visited))) {\
				return false;\
			}\
		}\
	}\


	int n, deltax, deltay, sgndeltax, sgndeltay, deltaxabs, deltayabs, x, y, drawx, drawy;
	deltax = x2 - x1;
	deltay = y2 - y1;
	deltaxabs = ABS(deltax);
	deltayabs = ABS(deltay);
	sgndeltax = SGN(deltax);
	sgndeltay = SGN(deltay);
	x = deltayabs >> 1;
	y = deltaxabs >> 1;
	drawx = x1;
	drawy = y1;
	int pc=0;

	_TEST_VALID

	if(deltaxabs >= deltayabs) {
		for(n = 0; n < deltaxabs; n++) {
			y += deltayabs;
			if(y >= deltaxabs){
				y -= deltaxabs;
				drawy += sgndeltay;
			}
			drawx += sgndeltax;
			_TEST_VALID
		}
	} else {
		for(n = 0; n < deltayabs; n++) {
			x += deltaxabs;
			if(x >= deltayabs) {
				x -= deltayabs;
				drawx += sgndeltax;
			}
			drawy += sgndeltay;
			_TEST_VALID
		}
	}
	return true;


}

bool ImagePathFinder::_is_linear_path(const Point2& p_from, const Point2& p_to) {

	int x1=p_from.x;
	int y1=p_from.y;
	int x2=p_to.x;
	int y2=p_to.y;

#define _TEST_CELL \
	if (cells[drawy*width+drawx].solid)\
		return false;


	int n, deltax, deltay, sgndeltax, sgndeltay, deltaxabs, deltayabs, x, y, drawx, drawy;
	deltax = x2 - x1;
	deltay = y2 - y1;
	deltaxabs = ABS(deltax);
	deltayabs = ABS(deltay);
	sgndeltax = SGN(deltax);
	sgndeltay = SGN(deltay);
	x = deltayabs >> 1;
	y = deltaxabs >> 1;
	drawx = x1;
	drawy = y1;
	int pc=0;

	_TEST_CELL

	if(deltaxabs >= deltayabs) {
		for(n = 0; n < deltaxabs; n++) {
			y += deltayabs;
			if(y >= deltaxabs){
				y -= deltaxabs;
				drawy += sgndeltay;
			}
			drawx += sgndeltax;
			_TEST_CELL
		}
	} else {
		for(n = 0; n < deltayabs; n++) {
			x += deltaxabs;
			if(x >= deltayabs) {
				x -= deltayabs;
				drawx += sgndeltax;
			}
			drawy += sgndeltay;
			_TEST_CELL
		}
	}
	return true;
}


DVector<Point2> ImagePathFinder::find_path(const Point2& p_from, const Point2& p_to,bool p_optimize) {


	Point2i from=p_from;
	Point2i to=p_to;

	ERR_FAIL_COND_V(from.x < 0,DVector<Point2>());
	ERR_FAIL_COND_V(from.y < 0,DVector<Point2>());
	ERR_FAIL_COND_V(from.x >=width,DVector<Point2>());
	ERR_FAIL_COND_V(from.y >=height,DVector<Point2>());
	ERR_FAIL_COND_V(to.x < 0,DVector<Point2>());
	ERR_FAIL_COND_V(to.y < 0,DVector<Point2>());
	ERR_FAIL_COND_V(to.x >=width,DVector<Point2>());
	ERR_FAIL_COND_V(to.y >=height,DVector<Point2>());

	if (from==to) {
		DVector<Point2> p;
		p.push_back(from);
		return p;
	}

	_lock();


	if (p_optimize) { //try a line first

		if (_is_linear_path(p_from,p_to)) {
			_unlock();
			DVector<Point2> p;
			p.push_back(from);
			p.push_back(to);
			return p;
		}
	}


	//clear all
	for(int i=0;i<width*height;i++) {

		bool s = cells[i].solid;
		cells[i].data=0;
		cells[i].solid=s;
	}

#define CELL_INDEX(m_p) (m_p.y*width+m_p.x)
#define CELL_COST(m_p) (cells[CELL_INDEX(m_p)].cost+( ABS(m_p.x-to.x)+ABS(m_p.y-to.y))*10)


	Set<Point2i> pending;
	pending.insert(from);

	//helper constants
	static const Point2i neighbour_rel[8]={
		Point2i(-1,-1), //0
		Point2i(-1, 0), //1
		Point2i(-1,+1), //2
		Point2i( 0,-1), //3
		Point2i( 0,+1), //4
		Point2i(+1,-1), //5
		Point2i(+1, 0), //6
		Point2i(+1,+1) }; //7

	static const int neighbour_cost[8]={
		14,
		10,
		14,
		10,
		10,
		14,
		10,
		14
	};

	static const int neighbour_parent[8]={
		7,
		6,
		5,
		4,
		3,
		2,
		1,
		0,
	};

	while(true) {

		if (pending.size() == 0) {
			_unlock();
			return DVector<Point2>(); // points don't connect
		}
		Point2i current;
		int lc=0x7FFFFFFF;
		{ //find the one with the least cost

			Set<Point2i>::Element *Efound=NULL;
			for (Set<Point2i>::Element *E=pending.front();E;E=E->next()) {

				int cc =CELL_COST(E->get());
				if (cc<lc) {
					lc=cc;
					current=E->get();
					Efound=E;

				}

			}
			pending.erase(Efound);
		}

		Cell &c = cells[CELL_INDEX(current)];

		//search around other cells


		int accum_cost = (from==current) ? 0 : cells[CELL_INDEX((current + neighbour_rel[c.parent]))].cost;

		bool done=false;

		for(int i=0;i<8;i++) {

			Point2i neighbour=current+neighbour_rel[i];
			if (neighbour.x<0 || neighbour.y<0 || neighbour.x>=width || neighbour.y>=height)
				continue;

			Cell &n = cells[CELL_INDEX(neighbour)];
			if (n.solid)
				continue; //no good

			int cost = neighbour_cost[i]+accum_cost;

			if (n.visited && n.cost < cost)
				continue;

			n.cost=cost;
			n.parent=neighbour_parent[i];
			n.visited=true;
			pending.insert(neighbour);
			if (neighbour==to)
				done=true;

		}

		if (done)
			break;
	}


	// go througuh poins twice, first compute amount, then add them

	Point2i current=to;
	int pcount=0;

	while(true) {

		Cell &c = cells[CELL_INDEX(current)];
		c.visited=true;
		pcount++;
		if (current==from)
			break;
		current+=neighbour_rel[ c.parent ];
	}

	//now place them in an array
	DVector<Vector2> result;
	result.resize(pcount);

	DVector<Vector2>::Write res=result.write();

	current=to;
	int pidx=pcount-1;

	while(true) {

		Cell &c = cells[CELL_INDEX(current)];
		res[pidx]=current;
		pidx--;
		if (current==from)
			break;
		current+=neighbour_rel[ c.parent ];
	}


	//simplify..


	if (p_optimize) {

		int p=pcount-1;
		while(p>0) {


			int limit=p;
			while(limit>0) {

				limit--;
				if (!_can_go_straigth(res[p],res[limit]))
					break;
			}


			if (limit<p-1) {
				int diff = p-limit-1;
				pcount-=diff;
				for(int i=limit+1;i<pcount;i++) {

					res[i]=res[i+diff];
				}
			}
			p=limit;
		}
	}

	res=DVector<Vector2>::Write();
	result.resize(pcount);
	return result;
}

Size2 ImagePathFinder::get_size() const {

	return Size2(width,height);
}
bool ImagePathFinder::is_solid(const Point2& p_pos) {


	Point2i pos = p_pos;

	ERR_FAIL_COND_V(pos.x<0,true);
	ERR_FAIL_COND_V(pos.y<0,true);
	ERR_FAIL_COND_V(pos.x>=width,true);
	ERR_FAIL_COND_V(pos.y>=height,true);

	return cell_data[pos.y*width+pos.x].solid;
}

void ImagePathFinder::create_from_image_alpha(const Image& p_image) {

	ERR_FAIL_COND(p_image.get_format() != Image::FORMAT_RGBA);
	width = p_image.get_width();
	height = p_image.get_height();
	DVector<uint8_t> data = p_image.get_data();
	cell_data.resize(width * height);
	DVector<uint8_t>::Read read = data.read();
	DVector<Cell>::Write write = cell_data.write();
	for (int i=0; i<width * height; i++) {
		Cell cell;
		cell.data = 0;
		cell.solid = read[i*4+3] < 128;
		write[i] = cell;
	};
};


void ImagePathFinder::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("find_path","from","to","optimize"),&ImagePathFinder::find_path,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("get_size"),&ImagePathFinder::get_size);
	ObjectTypeDB::bind_method(_MD("is_solid","pos"),&ImagePathFinder::is_solid);
	ObjectTypeDB::bind_method(_MD("create_from_image_alpha"),&ImagePathFinder::create_from_image_alpha);
}

ImagePathFinder::ImagePathFinder()
{

	cells=NULL;
	width=0;
	height=0;
}
