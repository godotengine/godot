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

static uint32_t _mouse_button_mask = 0;

static void _glut_mouse_button(int button, int state, int x, int y) {

	InputEvent ev;
	ev.type = InputEvent::MOUSE_BUTTON;
	switch (button) {
		case GLUT_LEFT_BUTTON: ev.mouse_button.button_index = BUTTON_LEFT; break;
		case GLUT_MIDDLE_BUTTON: ev.mouse_button.button_index = BUTTON_MIDDLE; break;
		case GLUT_RIGHT_BUTTON: ev.mouse_button.button_index = BUTTON_RIGHT; break;
		case 3: ev.mouse_button.button_index = BUTTON_WHEEL_UP; break;
		case 4: ev.mouse_button.button_index = BUTTON_WHEEL_DOWN; break;
	}

	ev.mouse_button.pressed = state == GLUT_DOWN;
	ev.mouse_button.x = x;
	ev.mouse_button.y = y;
	ev.mouse_button.global_x = x;
	ev.mouse_button.global_y = y;

	if (ev.mouse_button.button_index < 4) {
		if (ev.mouse_button.pressed) {
			_mouse_button_mask |= 1 << (ev.mouse_button.button_index - 1);
		} else {
			_mouse_button_mask &= ~(1 << (ev.mouse_button.button_index - 1));
		}
	}
	ev.mouse_button.button_mask = _mouse_button_mask;

	uint32_t m = glutGetModifiers();
	ev.mouse_button.mod.alt = (m & GLUT_ACTIVE_ALT) != 0;
	ev.mouse_button.mod.shift = (m & GLUT_ACTIVE_SHIFT) != 0;
	ev.mouse_button.mod.control = (m & GLUT_ACTIVE_CTRL) != 0;

	os->push_input(ev);

	if (ev.mouse_button.button_index == BUTTON_WHEEL_UP || ev.mouse_button.button_index == BUTTON_WHEEL_DOWN) {
		// GLUT doesn't send release events for mouse wheel, so send manually
		ev.mouse_button.pressed = false;
		os->push_input(ev);
	}
}

static int _glut_prev_x = 0;
static int _glut_prev_y = 0;

static void _glut_mouse_motion(int x, int y) {

	InputEvent ev;
	ev.type = InputEvent::MOUSE_MOTION;
	ev.mouse_motion.button_mask = _mouse_button_mask;
	ev.mouse_motion.x = x;
	ev.mouse_motion.y = y;
	ev.mouse_motion.global_x = x;
	ev.mouse_motion.global_y = y;
	ev.mouse_motion.relative_x = x - _glut_prev_x;
	ev.mouse_motion.relative_y = y - _glut_prev_y;
	_glut_prev_x = x;
	_glut_prev_y = y;

	uint32_t m = glutGetModifiers();
	ev.mouse_motion.mod.alt = (m & GLUT_ACTIVE_ALT) != 0;
	ev.mouse_motion.mod.shift = (m & GLUT_ACTIVE_SHIFT) != 0;
	ev.mouse_motion.mod.control = (m & GLUT_ACTIVE_CTRL) != 0;

	os->push_input(ev);
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

	glutMouseFunc(_glut_mouse_button);
	glutMotionFunc(_glut_mouse_motion);
	glutPassiveMotionFunc(_glut_mouse_motion);

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
