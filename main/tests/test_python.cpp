/*************************************************************************/
/*  test_python.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "test_python.h"

#ifdef PYTHON_ENABLED

#include "Python.h"
#include "print_string.h"

namespace TestPython {

void test() {

	print_line("testing python");
	PyRun_SimpleString("import engine\n");
	PyRun_SimpleString("def test(self):\n\tprint(\"noway\")\n");
	PyRun_SimpleString("a=engine.ObjectPtr()\n");
	PyRun_SimpleString("a.noway(22,'hello')\n");
	PyRun_SimpleString("a.normalize()\n");
	PyRun_SimpleString("class Moch(engine.ObjectPtr):\n\tdef mooch(self):\n\t\tprint('muchi')\n");
	PyRun_SimpleString("b=Moch();\n");
	PyRun_SimpleString("b.mooch();\n");
	PyRun_SimpleString("b.meis();\n");
}
} // namespace TestPython

#endif
