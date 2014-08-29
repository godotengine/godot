/*************************************************************************/
/*  editor_atlas.cpp                                                     */
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
#include "editor_atlas.h"
#include "vector3.h"

/*
BinPack2D: from https://github.com/chris-stones/BinPack2D

Copyright (c) 2013, christopher stones < chris.stones@zoho.com >
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include<vector>
#include<list>
#include<algorithm>

namespace BinPack2D {

class Coord : public Vector3
{
public:
	typedef std::vector<Coord>	Vector;
	typedef std::list<Coord>	List;

	Coord() { x = y = z = 0; }
	Coord(real_t p_x, real_t p_y, real_t p_z = 0) { x = p_x; y = p_y; z = p_z; }
};

template<typename T> class Content
{
public:
	typedef std::vector<Content<T> > Vector;
	
	bool rotated;
	Coord coord;
	Size2i  size;
	T content;

	Content(const Content<T> &src)
		: rotated(src.rotated)
		, coord(src.coord)
		, size(src.size)
		, content(src.content)
	{}

	Content(const T &content, const Coord &coord, const Size2i &size, bool rotated)
		: content(content)
		, coord(coord)
		, size(size)
		, rotated(rotated)
	{}
	
	void rotate() {

		rotated = !rotated;
		size = Size2i(size.height, size.width);
	}
	
	bool intersects(const Content<T> &that) const {

		if(this->coord.x >= (that.coord.x + that.size.width))
			return false;

		if(this->coord.y >= (that.coord.y + that.size.height))
			return false;

		if(that.coord.x >= (this->coord.x + this->size.width))
			return false;

		if(that.coord.y >= (this->coord.y + this->size.height))
			return false;

		return true;
	}
};

template<typename T> class Canvas
{
	Coord::List topLefts;
	typename Content<T>::Vector contentVector;
	bool needToSort;

public:
	typedef Canvas<T> CanvasT;
	typedef typename std::vector<CanvasT> Vector;
	
	static bool place(Vector &canvasVector, const typename Content<T>::Vector &contentVector, typename Content<T>::Vector &remainder) {
		typename Content<T>::Vector todo = contentVector;
		for(typename Vector::iterator itor = canvasVector.begin(); itor != canvasVector.end(); itor++) {

			Canvas <T> &canvas = *itor;
			remainder.clear();
			canvas.place(todo, remainder);
			todo = remainder;
		}
	
		if(remainder.size() == 0)
			return true;

		return false;
	}
	
	static bool place(Vector &canvasVector, const typename Content<T>::Vector &contentVector) {

		typename Content<T>::Vector remainder;
		return place(canvasVector, contentVector, remainder);
	}
	
	static bool place(Vector &canvasVector, const Content<T> &content) {

		typename Content<T>::Vector contentVector(1, content);
		return place(canvasVector, contentVector);
	}
	
	const int w;
	const int h;
	 
	Canvas(int w, int h)
		: needToSort(false)
		, w(w)
		, h(h)
	{  
		topLefts.push_back(Coord(0,0,0));
	}
	
	bool hasContent() const {

		return (contentVector.size() > 0) ;
	}
	
	const typename Content<T>::Vector &getContents() const {

		return contentVector;
	}
	
	bool operator < (const Canvas &that) const {

		if(this->w != that.w) return this->w < that.w;
		if(this->h != that.h) return this->h < that.h;
	}

	bool place(const typename Content<T>::Vector &contentVector, typename Content<T>::Vector &remainder) {
	
		bool placedAll = true;
		for(typename Content<T>::Vector::const_iterator itor = contentVector.begin(); itor != contentVector.end(); itor++) {

			const Content<T> & content = *itor;
			if(place(content) == false) {

				placedAll = false;
				remainder.push_back(content);
			}
		}
		return placedAll;
	}

	bool place(Content<T> content) {
 
		sort();

		for(Coord::List::iterator itor = topLefts.begin(); itor != topLefts.end(); itor++) {

			content.coord = *itor;
			if(fits(content)) {

				use(content);
				topLefts.erase(itor);
				return true;
			}
		}

		// Godot atlas does not support rotate
		//// EXPERIMENTAL - TRY ROTATED?
		//content.rotate();
		//for(Coord::List::iterator itor = topLefts.begin(); itor != topLefts.end(); itor++) {
		//
		//	content.coord = *itor;
		//
		//	if(fits(content)) {
		//
		//		use(content);
		//		topLefts.erase(itor);
		//		return true;
		//	}
		//}
		//////////////////////////////////
		return false;
	}
	
private:
	bool fits(const Content<T> &content) const {

		if((content.coord.x + content.size.width) > w)
			return false;
	
		if((content.coord.y + content.size.height) > h)
			return false;
	
		for(typename Content<T>::Vector::const_iterator itor = contentVector.begin(); itor != contentVector.end(); itor++)  
			if(content.intersects(*itor))
				return false;

		return true;
	}

	bool use(const Content<T> &content) {

		const Size2i  &size = content.size;
		const Coord &coord = content.coord;

		topLefts.push_front(Coord(
			coord.x + size.width,
			coord.y
		));

		topLefts.push_back(Coord(
			coord.x,
			coord.y + size.height
		));

		contentVector.push_back(content);
		needToSort = true;

		return true;
	}

private:
	struct TopToBottomLeftToRightSort {
		bool operator()(const Coord &a, const Coord &b) const {

			return (a.x * a.x + a.y * a.y) < (b.x * b.x + b.y * b.y);
		}
	};

public:

	void sort() {

		if(!needToSort)
			return;

			topLefts.sort(TopToBottomLeftToRightSort());
		needToSort = false;
	}
};

template <typename T> class ContentAccumulator {
	typename Content<T>::Vector contentVector;

public:

	ContentAccumulator()
	{}
	
	const typename Content<T>::Vector &get() const {

		return contentVector;
	}
	
	typename Content<T>::Vector &get() {

		return contentVector;
	}
	
	ContentAccumulator<T>& operator += (const Content<T> & content) {

		contentVector.push_back(content);
		return *this;
	}
	
	ContentAccumulator<T>& operator += (const typename Content<T>::Vector & content) {

		contentVector.insert(contentVector.end(), content.begin(), content.end());
		return *this;
	}
	
	ContentAccumulator<T> operator + (const Content<T> & content) {

		ContentAccumulator<T> temp = *this;
		temp += content;
		return temp;
	}
	
	ContentAccumulator<T> operator + (const typename Content<T>::Vector & content) {

		ContentAccumulator<T> temp = *this;
		temp += content;
		return temp;
	}

private:
	struct GreatestWidthThenGreatestHeightSort {

		bool operator()(const Content<T> &a, const Content<T> &b) const {
		  
			const Size2i &sa = a.size; 
			const Size2i &sb = b.size;

			return sa > sb;
		}
	};
	
	struct MakeHorizontal {

		Content<T> operator()(const Content<T> &elem) {

			if(elem.size.height > elem.size.width) {

				Content<T> r = elem;
				r.size.width = elem.size.height;
				r.size.height = elem.size.width;
				r.rotated = !elem.rotated;
				return r;
			}
			return elem;
		}
	};
	
public:
	
	void sort() {
	 
	//  if(allow_rotation)
	//	std::transform(contentVector.begin(), contentVector.end(), contentVector.begin(), MakeHorizontal());
		std::sort(contentVector.begin(), contentVector.end(), GreatestWidthThenGreatestHeightSort());
	}
};

template <typename T> class UniformCanvasArrayBuilder {

	int w;
	int h;
	int d;

public:
	UniformCanvasArrayBuilder(int w, int h, int d)
		: w(w)
		, h(h)
		, d(d)
	{}

	typename Canvas<T>::Vector Build() {

		return typename Canvas<T>::Vector(d, Canvas<T>(w, h));
	}  
};

template<typename T> class CanvasArray {

	typename Canvas<T>::Vector canvasArray;

public:  

	CanvasArray(const typename Canvas<T>::Vector &canvasArray)
		: canvasArray(canvasArray)
	{}

	bool place(const typename Content<T>::Vector &contentVector, typename Content<T>::Vector &remainder) {
 
		return Canvas<T>::place(canvasArray, contentVector, remainder);
	}

	bool place(const ContentAccumulator<T> &content, ContentAccumulator<T> &remainder) {

		return place(content.get(), remainder.get());
	}

	bool place(const typename Content<T>::Vector &contentVector) {

		return Canvas<T>::place(canvasArray, contentVector);
	}

	bool place(const ContentAccumulator<T> &content) {

		return place(content.get());
	}

	bool collectContent(typename Content<T>::Vector &contentVector) const {

		int z = 0;
		for(typename Canvas<T>::Vector::const_iterator itor = canvasArray.begin(); itor != canvasArray.end(); itor++) {

			const typename Content<T>::Vector &contents = itor->getContents();

			for(typename Content<T>::Vector::const_iterator itor = contents.begin(); itor != contents.end(); itor++) {

				Content<T> content = *itor;
				content.coord.z = z;
				contentVector.push_back(content);
			}
			z++;
		}
		return true;
	}
	
	bool collectContent(ContentAccumulator<T> &content) const {

		return collectContent(content.get());
	}
};

} /*** BinPack2D ***/

// Your data - whatever you want to associate with 'rectangle'
typedef size_t AtlasContent;

static bool binpack2d_fit(size_t w, size_t h, const BinPack2D::ContentAccumulator<AtlasContent>& inputContent, BinPack2D::ContentAccumulator<AtlasContent>& outputContent) {

	// Create some bins! (1 bins, w*h in this example)
	BinPack2D::CanvasArray<AtlasContent> canvasArray = 
		BinPack2D::UniformCanvasArrayBuilder<AtlasContent>(w, h, 1).Build();

	// A place to store content that didnt fit into the canvas array.
	BinPack2D::ContentAccumulator<AtlasContent> remainder;

	// try to pack content into the bins.
	if(!canvasArray.place(inputContent, remainder)) {

		outputContent.get().clear();
		return false;
	}

	// Read all placed content.
	canvasArray.collectContent(outputContent);

	return true;
}

void EditorAtlas::fit(const Vector<Size2i>& p_rects,Vector<Point2i>& r_result, Size2i& r_size) {


	// Create some 'content' to work on.
	BinPack2D::ContentAccumulator<AtlasContent> inputContent;

	for(size_t idx = 0; idx < p_rects.size(); idx++) {

		// size for this content
		const Size2i& size = p_rects[idx];
		// Add it
		inputContent += BinPack2D::Content<AtlasContent>(idx, BinPack2D::Coord(), Size2i(size.width, size.height), false);
	}

	// sort the input content by size... usually packs better.
	inputContent.sort();

	size_t pow_of_2[] = {
		64, 128, 256, 512, 1024, 2048, 4096
	};
	#define POW_SIZE (sizeof(pow_of_2) / sizeof(pow_of_2[0]))

	// A place to store packed content.
	BinPack2D::ContentAccumulator<AtlasContent> outputContent;

	for(size_t idx = 0; idx < POW_SIZE; idx++) {

		// Try to fit square(w * h) atlas
		{
			size_t w = pow_of_2[idx];
			size_t h = pow_of_2[idx];
			if(binpack2d_fit(w, h, inputContent, outputContent)) {

				r_size = Size2(w, h);
				break;
			}
		}
		if(idx == POW_SIZE)
			break;

		// Try to fit rectangle(w, h * 2) atlas
		{
			size_t w = pow_of_2[idx];
			size_t h = pow_of_2[idx + 1];
			if(binpack2d_fit(w, h, inputContent, outputContent)) {

				r_size = Size2(w, h);
				break;
			}
		}
		// Try to fit rectangle(w * 2, h) atlas
		{
			size_t w = pow_of_2[idx + 1];
			size_t h = pow_of_2[idx];
			if(binpack2d_fit(w, h, inputContent, outputContent)) {

				r_size = Size2(w, h);
				break;
			}
		}
	}

	r_result.resize(p_rects.size());

	// parse output.
	typedef BinPack2D::Content<AtlasContent>::Vector::iterator binpack2d_iterator;
	for(binpack2d_iterator itor = outputContent.get().begin(); itor != outputContent.get().end(); itor++) {
	
		const BinPack2D::Content<AtlasContent> &content = *itor;

		// retreive your data.
		const AtlasContent &index = content.content;
		r_result[index] = Size2i(content.coord.x, content.coord.y);
	}
}


