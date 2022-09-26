/*************************************************************************/
/*  test_path_2d.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_PATH_2D_H
#define TEST_PATH_2D_H

#include "scene/2d/path_2d.h"

#include "tests/test_macros.h"

namespace TestPath_2d {

TEST_CASE("[Path2D] Initialization") {
    SUBCASE("Path2D should not contain a curve when intialized") {
	    Path2D *path = memnew(Path2D);
	    CHECK(path->get_curve() == nullptr);
        memdelete(path);
    }
}

TEST_CASE("[Path2D] Assiging a curve") {
    SUBCASE("Path2D When a curve is assigned to a Path2D, the same curve should be returned by get_curve") {
	    Path2D *path = memnew(Path2D);
        
        Ref<Curve2D> *curve = memnew(Curve2D);
        path->set_curve(curve);

	    CHECK(path->get_curve() == curve);
        memdelete(path);
        memdelete(curve);
    }
    SUBCASE("Assigning a curve to a Path2D multiple times") {
	    Path2D *path = memnew(Path2D);
        
        Ref<Curve2D> *curve = memnew(Curve2D);
        path->set_curve(curve);
        path->set_curve(curve);
        path->set_curve(curve);

	    CHECK(path->get_curve() == curve);
        memdelete(path);
        memdelete(curve);
    }
    SUBCASE("Changing the curve assigned to a Path2D") {
	    Path2D *path = memnew(Path2D);
        
        Ref<Curve2D> *curve1 = memnew(Curve2D);
        Ref<Curve2D> *curve2 = memnew(Curve2D);

        path->set_curve(curve1);
	    CHECK(path->get_curve() == curve1);

        path->set_curve(curve2);
        CHECK(path->get_curve() == curve2);

        memdelete(path);
        memdelete(curve1);
        memdelete(curve2);
    }
    SUBCASE("Assigning the same curve to multiple Path2D objects") {
	    Path2D *path1 = memnew(Path2D);
        Path2D *path2 = memnew(Path2D);

        Ref<Curve2D> *curve = memnew(Curve2D);
        
        path1->set_curve(curve);
        path2->set_curve(curve);

	    CHECK(path1->get_curve() == path2->get_curve());
        memdelete(path1);
        memdelete(path2);
        memdelete(curve);
    }
    SUBCASE("Switching curves between two Path2D objects") {
	    Path2D *path1 = memnew(Path2D);
        Path2D *path2 = memnew(Path2D);

        Ref<Curve2D> *curve1 = memnew(Curve2D);
        Ref<Curve2D> *curve2 = memnew(Curve2D);


        path1->set_curve(curve1);
        path2->set_curve(curve2);
        CHECK(path1->get_curve() == curve1);
        CHECK(path2->get_curve() == curve2);

        Curve2D temp = path1->get_curve();
        path1->set_curve(path2->get_curve());
        path2->set_curve(temp);
        CHECK(path1->get_curve() == curve2);
        CHECK(path2->get_curve() == curve1);
        
        memdelete(path1);
        memdelete(path2);
        memdelete(curve1);
        memdelete(curve2);
    }
    
}



} // namespace TestPath_2d

#endif // TEST_PATH_2D_H
