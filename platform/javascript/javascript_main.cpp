/*************************************************************************/
/*  javascript_main.cpp                                                  */
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
#include <GL/glut.h>
#include "os_javascript.h"
#include "main/main.h"
#include "io/resource_loader.h"
#include "os/keyboard.h"
#include "emscripten.h"
#include <string.h>

OS_JavaScript *os=NULL;

static void _gfx_init(void *ud,bool gl2,int w, int h,bool fs) {

	glutInitWindowSize(w, h);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutCreateWindow("godot");

}


static void _glut_skey(bool pressed,int key) {

	InputEvent ev;
	ev.type=InputEvent::KEY;
	ev.key.pressed=pressed;
	switch(key) {
		case GLUT_KEY_F1: ev.key.scancode=KEY_F1; break;
		case GLUT_KEY_F2: ev.key.scancode=KEY_F2; break;
		case GLUT_KEY_F3: ev.key.scancode=KEY_F3; break;
		case GLUT_KEY_F4: ev.key.scancode=KEY_F4; break;
		case GLUT_KEY_F5: ev.key.scancode=KEY_F5; break;
		case GLUT_KEY_F6: ev.key.scancode=KEY_F6; break;
		case GLUT_KEY_F7: ev.key.scancode=KEY_F7; break;
		case GLUT_KEY_F8: ev.key.scancode=KEY_F8; break;
		case GLUT_KEY_F9: ev.key.scancode=KEY_F9; break;
		case GLUT_KEY_F10: ev.key.scancode=KEY_F10; break;
		case GLUT_KEY_F11: ev.key.scancode=KEY_F11; break;
		case GLUT_KEY_F12: ev.key.scancode=KEY_F12; break;
		case GLUT_KEY_LEFT: ev.key.scancode=KEY_LEFT; break;
		case GLUT_KEY_UP: ev.key.scancode=KEY_UP; break;
		case GLUT_KEY_RIGHT: ev.key.scancode=KEY_RIGHT; break;
		case GLUT_KEY_DOWN: ev.key.scancode=KEY_DOWN; break;
		case GLUT_KEY_PAGE_UP: ev.key.scancode=KEY_PAGEUP; break;
		case GLUT_KEY_PAGE_DOWN: ev.key.scancode=KEY_PAGEDOWN; break;
		case GLUT_KEY_HOME: ev.key.scancode=KEY_HOME; break;
		case GLUT_KEY_END: ev.key.scancode=KEY_END; break;
		case GLUT_KEY_INSERT: ev.key.scancode=KEY_INSERT; break;
	}


	uint32_t m = glutGetModifiers();
	ev.key.mod.alt=(m&GLUT_ACTIVE_ALT)!=0;
	ev.key.mod.shift=(m&GLUT_ACTIVE_SHIFT)!=0;
	ev.key.mod.control=(m&GLUT_ACTIVE_CTRL)!=0;

	os->push_input(ev);
}

static void _glut_skey_up(int key, int x, int y) {

	_glut_skey(false,key);
}

static void _glut_skey_down(int key, int x, int y) {

	_glut_skey(true,key);
}

static void _glut_key(bool pressed,unsigned char key) {

	InputEvent ev;
	ev.type=InputEvent::KEY;
	ev.key.pressed=pressed;
	switch(key) {
		case '\n': ev.key.scancode=KEY_RETURN; break;
		case 0x1b: ev.key.scancode=KEY_ESCAPE; break;
		case 8: ev.key.scancode=KEY_BACKSPACE; break;
		case 0x7f: ev.key.scancode=KEY_DELETE; break;
		case 0x20: ev.key.scancode=KEY_SPACE; ev.key.unicode=key; break;
		default: {
			ev.key.unicode=key;
		}
	}


	uint32_t m = glutGetModifiers();
	ev.key.mod.alt=(m&GLUT_ACTIVE_ALT)!=0;
	ev.key.mod.shift=(m&GLUT_ACTIVE_SHIFT)!=0;
	ev.key.mod.control=(m&GLUT_ACTIVE_CTRL)!=0;

	os->push_input(ev);

}

static void _glut_key_up(unsigned char key, int x, int y) {

	_glut_key(false,key);
}

static void _glut_key_down(unsigned char key, int x, int y) {

	_glut_key(true,key);
}

static uint32_t _mouse_button_mask=0;

static void _glut_mouse_button(int button, int state, int x, int y) {

	InputEvent ev;
	ev.type=InputEvent::MOUSE_BUTTON;
	switch(button) {
		case GLUT_LEFT_BUTTON: ev.mouse_button.button_index=BUTTON_LEFT; break;
		case GLUT_MIDDLE_BUTTON: ev.mouse_button.button_index=BUTTON_MIDDLE; break;
		case GLUT_RIGHT_BUTTON: ev.mouse_button.button_index=BUTTON_RIGHT; break;
		case 3: ev.mouse_button.button_index=BUTTON_WHEEL_UP; break;
		case 4: ev.mouse_button.button_index=BUTTON_WHEEL_DOWN; break;
	}


	ev.mouse_button.pressed=state==GLUT_DOWN;
	ev.mouse_button.x=x;
	ev.mouse_button.y=y;
	ev.mouse_button.global_x=x;
	ev.mouse_button.global_y=y;

	if (ev.mouse_button.button_index<4) {
		if (ev.mouse_button.pressed) {
			_mouse_button_mask|=1<<ev.mouse_button.button_index;
		} else {
			_mouse_button_mask&=~(1<<ev.mouse_button.button_index);
		}
	}

	uint32_t m = glutGetModifiers();
	ev.mouse_button.mod.alt=(m&GLUT_ACTIVE_ALT)!=0;
	ev.mouse_button.mod.shift=(m&GLUT_ACTIVE_SHIFT)!=0;
	ev.mouse_button.mod.control=(m&GLUT_ACTIVE_CTRL)!=0;

	os->push_input(ev);

}


static int _glut_prev_x=0;
static int _glut_prev_y=0;

static void _glut_mouse_motion(int x, int y) {

	InputEvent ev;
	ev.type=InputEvent::MOUSE_MOTION;
	ev.mouse_motion.button_mask=_mouse_button_mask;
	ev.mouse_motion.x=x;
	ev.mouse_motion.y=y;
	ev.mouse_motion.global_x=x;
	ev.mouse_motion.global_y=y;
	ev.mouse_motion.relative_x=x-_glut_prev_x;
	ev.mouse_motion.relative_y=y-_glut_prev_y;
	_glut_prev_x=x;
	_glut_prev_y=y;

	uint32_t m = glutGetModifiers();
	ev.mouse_motion.mod.alt=(m&GLUT_ACTIVE_ALT)!=0;
	ev.mouse_motion.mod.shift=(m&GLUT_ACTIVE_SHIFT)!=0;
	ev.mouse_motion.mod.control=(m&GLUT_ACTIVE_CTRL)!=0;

	os->push_input(ev);

}

static void _gfx_idle() {

	glutPostRedisplay();
}

int start_step=0;

static void _godot_draw(void) {

	if (start_step==1) {
		start_step=2;
		Main::start();
		 os->main_loop_begin();
	}

	if (start_step==2) {
		os->main_loop_iterate();
	}

	glutSwapBuffers();
}



extern "C" {

void main_after_fs_sync(int value) {

	start_step=1;
	printf("FS SYNCHED!\n");
}

}

int main(int argc, char *argv[]) {


	/* Initialize the window */
	printf("let it go!\n");
	glutInit(&argc, argv);
	os = new OS_JavaScript(_gfx_init,NULL,NULL,NULL,NULL);
#if 0
	char *args[]={"-test","gui","-v",NULL};
	Error err  = Main::setup("apk",3,args);
#else
//	char *args[]={"-v",NULL};//
//	Error err  = Main::setup("",1,args);
	Error err  = Main::setup("",0,NULL);

#endif
	ResourceLoader::set_abort_on_missing_resources(false); //ease up compatibility


	glutSpecialUpFunc(_glut_skey_up);
	glutSpecialFunc(_glut_skey_down);
	glutKeyboardUpFunc(_glut_key_up);
	glutKeyboardFunc(_glut_key_down);
	glutMouseFunc(_glut_mouse_button);
	glutMotionFunc(_glut_mouse_motion);
	glutMotionFunc(_glut_mouse_motion);
	glutPassiveMotionFunc(_glut_mouse_motion);



   /* Set up glut callback functions */
	glutIdleFunc (_gfx_idle);
//   glutReshapeFunc(gears_reshape);
	glutDisplayFunc(_godot_draw);
   //glutSpecialFunc(gears_special);



	 //mount persistent filesystem
	 EM_ASM(
		 FS.mkdir('/userfs');
		 FS.mount(IDBFS, {}, '/userfs');



		 // sync from persisted state into memory and then
		 // run the 'test' function
		 FS.syncfs(true, function (err) {
			 assert(!err);
			 console.log("done syncinc!");
			 _after_sync_cb = Module.cwrap('main_after_fs_sync', 'void',['number']);
			 _after_sync_cb(0);

		 });

	  );

	glutMainLoop();



	return 0;
}


/*
 *
 *09] <azakai|2__> reduz: yes, define  TOTAL_MEMORY on Module. for example             var Module = { TOTAL_MEMORY: 12345.. };         before the main
 *
 */
