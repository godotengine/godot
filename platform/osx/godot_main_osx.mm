/*************************************************************************/
/*  godot_main_osx.mm                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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

#include "os_osx.h"

#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
	int first_arg = 1;
	const char *dbg_arg = "-NSDocumentRevisionsDebugMode";
	printf("arguments\n");
	for (int i = 0; i < argc; i++) {
		if (strcmp(dbg_arg, argv[i]) == 0)
			first_arg = i + 2;
		printf("%i: %s\n", i, argv[i]);
	};

	if (argc >= 1 && argv[0][0] == '/') {
		//potentially launched from finder
		int len = strlen(argv[0]);
		while (len--) {
			if (argv[0][len] == '/') break;
		}
		if (len >= 0) {
			char *path = (char *)malloc(len + 1);
			memcpy(path, argv[0], len);
			path[len] = 0;

			char *pathinfo = (char *)malloc(strlen(path) + strlen("/../Info.plist") + 1);
			//in real code you would check for errors in malloc here
			strcpy(pathinfo, path);
			strcat(pathinfo, "/../Info.plist");

			FILE *f = fopen(pathinfo, "rb");
			if (f) {
				//running from app bundle, as Info.plist was found
				fclose(f);
				chdir(path);
				chdir("../Resources"); //data.pck, or just the files are here
			}

			free(path);
			free(pathinfo);
		}
	}

#ifdef DEBUG_ENABLED
	// lets report the path we made current after all that
	char cwd[4096];
	getcwd(cwd, 4096);
	printf("Current path: %s\n", cwd);
#endif

	OS_OSX os;

	Error err = Main::setup(argv[0], argc - first_arg, &argv[first_arg]);
	if (err != OK)
		return 255;

	if (Main::start())
		os.run(); // it is actually the OS that decides how to run

	Main::cleanup();

	return 0;
};
