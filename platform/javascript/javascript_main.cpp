/*************************************************************************/
/*  javascript_main.cpp                                                  */
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
#include "emscripten.h"
#include "io/resource_loader.h"
#include "main/main.h"
#include "os_javascript.h"
#include <GL/glut.h>
#include <string.h>

OS_JavaScript *os = NULL;

static void _gfx_init(void *ud, bool gl2, int w, int h, bool fs) {

	glutInitWindowSize(w, h);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutCreateWindow("godot");
}

static void _gfx_idle() {

	glutPostRedisplay();
}

int start_step = 0;

static void _godot_draw(void) {

	if (start_step == 1) {
		start_step = 2;
		Main::start();
		os->main_loop_begin();
	}

	if (start_step == 2) {
		os->main_loop_iterate();
	}

	glutSwapBuffers();
}

extern "C" {

void main_after_fs_sync() {

	start_step = 1;
}
}

int main(int argc, char *argv[]) {

	/* Initialize the window */
	printf("let it go dude!\n");
	glutInit(&argc, argv);
	os = new OS_JavaScript(argv[0], _gfx_init, NULL, NULL);

	Error err = Main::setup(argv[0], argc - 1, &argv[1]);

	ResourceLoader::set_abort_on_missing_resources(false); //ease up compatibility

	/* Set up glut callback functions */
	glutIdleFunc(_gfx_idle);
	//   glutReshapeFunc(gears_reshape);
	glutDisplayFunc(_godot_draw);
	//glutSpecialFunc(gears_special);

	//mount persistent file system
	/* clang-format off */
	EM_ASM(
		FS.mkdir('/userfs');
		FS.mount(IDBFS, {}, '/userfs');

		// sync from persistent state into memory and then
		// run the 'main_after_fs_sync' function
		FS.syncfs(true, function(err) {

			if (err) {
				Module.setStatus('Failed to load persistent data\nPlease allow (third-party) cookies');
				Module.printErr('Failed to populate IDB file system: ' + err.message);
				Module.exit();
			} else {
				Module.print('Successfully populated IDB file system');
				ccall('main_after_fs_sync', 'void', []);
			}
		});
	);
	/* clang-format on */

	glutMainLoop();

	return 0;
}

/*
 *
 *09] <azakai|2__> reduz: yes, define  TOTAL_MEMORY on Module. for example             var Module = { TOTAL_MEMORY: 12345.. };         before the main
 *
 */
