/*************************************************************************/
/*  godot_nacl.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "os_nacl.h"
#include "main/main.h"

#include <stdio.h>
#include <string.h>

OSNacl* os_nacl = NULL;

int nacl_main(int argc, const char** argn, const char** argv) {

	// os is created in GodotModule::Init for the nacl module
	printf("called with %i args, %p, %p\n", argc, argn, argv);
	char* nargv[64];
	int nargc = 1;
	nargv[0] = (char*)argv[0];
	for (int i=1; i<argc; i++) {

		printf("arg %i is %s, %s\n", i, argn[i], argv[i]);
		if (strncmp(argn[i], "arg", 3) == 0) {

			printf("using arg %i, %s\n", nargc, argv[i]);
			nargv[nargc++] = (char*)argv[i];
		};
	};
	printf("total %i\n", nargc);
	nargv[nargc] = 0;

	printf("godot_nacl\n");
	for (int i=0; i<argc; i++) {
		printf("arg %i: %s\n", i, argv[i]);
	};

	printf("os created\n");
    Error err  = Main::setup("", nargc-1, &nargv[1]);
	printf("setup %i\n", err);
	if (err!=OK)
		return 255;

    printf("calling start\n");
    Main::start();

    return 0;
};


void nacl_cleanup() {

	Main::cleanup();
	delete os_nacl;
};


