/*************************************************************************/
/*  image_path_finder.h                                                  */
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
#ifndef IMAGE_PATH_FINDER_H
#define IMAGE_PATH_FINDER_H

#include "resource.h"

class ImagePathFinder : public Resource{


	OBJ_TYPE(ImagePathFinder,Resource);
	union Cell {

		struct {
			bool solid:1;
			bool visited:1;
			bool final:1;
			uint8_t parent:3;
			uint32_t cost:26;
		};

		uint32_t data;
	};





	DVector<Cell>::Write lock;
	DVector<Cell> cell_data;

	uint32_t width;
	uint32_t height;
	Cell* cells; //when unlocked

	void _unlock();
	void _lock();


	_FORCE_INLINE_ bool _can_go_straigth(const Point2& p_from, const Point2& p_to) const;
	_FORCE_INLINE_ bool _is_linear_path(const Point2& p_from, const Point2& p_to);

protected:

	static void _bind_methods();
public:

	DVector<Point2> find_path(const Point2& p_from, const Point2& p_to,bool p_optimize=false);
	Size2 get_size() const;
	bool is_solid(const Point2& p_pos);
	void create_from_image_alpha(const Image& p_image);



	ImagePathFinder();
};

#endif // IMAGE_PATH_FINDER_H
