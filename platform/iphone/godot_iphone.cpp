/*************************************************************************/
/*  godot_iphone.cpp                                                     */
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
#include "main/main.h"
#include "os_iphone.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static OSIPhone *os = NULL;

extern "C" {
int add_path(int p_argc, char **p_args);
int add_cmdline(int p_argc, char **p_args);
};

int iphone_main(int, int, int, char **);

int iphone_main(int width, int height, int argc, char **argv) {

	int len = strlen(argv[0]);

	while (len--) {
		if (argv[0][len] == '/') break;
	}

	if (len >= 0) {
		char path[512];
		memcpy(path, argv[0], len > sizeof(path) ? sizeof(path) : len);
		path[len] = 0;
		printf("Path: %s\n", path);
		chdir(path);
	}

	printf("godot_iphone %s\n", argv[0]);
	char cwd[512];
	getcwd(cwd, sizeof(cwd));
	printf("cwd %s\n", cwd);
	os = new OSIPhone(width, height);

	char *fargv[64];
	for (int i = 0; i < argc; i++) {
		fargv[i] = argv[i];
	};
	fargv[argc] = NULL;
	argc = add_path(argc, fargv);
	argc = add_cmdline(argc, fargv);

	printf("os created\n");
	Error err = Main::setup(fargv[0], argc - 1, &fargv[1], false);
	printf("setup %i\n", err);
	if (err != OK)
		return 255;

	return 0;
};

void iphone_finish() {

	printf("iphone_finish\n");
	Main::cleanup();
	delete os;
};
