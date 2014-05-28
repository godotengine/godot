/*************************************************************************/
/*  context_gl_x11.cpp                                                   */
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
#include "context_gl_x11.h"

#ifdef X11_ENABLED
#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <GL/glx.h>

#define GLX_CONTEXT_MAJOR_VERSION_ARB		0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB		0x2092

typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

struct ContextGL_X11_Private { 

	::GLXContext glx_context;
};


void ContextGL_X11::release_current() {

	glXMakeCurrent(x11_display, None, NULL);
}

void ContextGL_X11::make_current() {

	glXMakeCurrent(x11_display, x11_window, p->glx_context);
}
void ContextGL_X11::swap_buffers() {

	glXSwapBuffers(x11_display,x11_window);
}
/*
static GLWrapperFuncPtr wrapper_get_proc_address(const char* p_function) {

	//print_line(String()+"getting proc of: "+p_function);
	GLWrapperFuncPtr func=(GLWrapperFuncPtr)glXGetProcAddress( (const GLubyte*) p_function);
	if (!func) {
		print_line("Couldn't find function: "+String(p_function));
	}

	return func;

}*/

Error ContextGL_X11::initialize() {

	
	GLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = NULL;
	
//	const char *extensions = glXQueryExtensionsString(x11_display, DefaultScreen(x11_display));
	
	glXCreateContextAttribsARB = (GLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");
	
	ERR_FAIL_COND_V( !glXCreateContextAttribsARB, ERR_UNCONFIGURED );


	static int visual_attribs[] = {
	    GLX_RENDER_TYPE, GLX_RGBA_BIT,
	    GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
	    GLX_DOUBLEBUFFER, true,
	    GLX_RED_SIZE, 1,
	    GLX_GREEN_SIZE, 1,
	    GLX_BLUE_SIZE, 1,
	    GLX_DEPTH_SIZE,0,
	    None 
	};

	int fbcount;
	GLXFBConfig *fbc = glXChooseFBConfig(x11_display, DefaultScreen(x11_display), visual_attribs, &fbcount);
	ERR_FAIL_COND_V(!fbc,ERR_UNCONFIGURED);
	
	XVisualInfo *vi = glXGetVisualFromFBConfig(x11_display, fbc[0]);

	XSetWindowAttributes swa;

	swa.colormap = XCreateColormap(x11_display, RootWindow(x11_display, vi->screen), vi->visual, AllocNone);
	swa.border_pixel = 0;
	swa.event_mask = StructureNotifyMask;

	/*
	char* windowid = getenv("GODOT_WINDOWID");
	if (windowid) {

		//freopen("/home/punto/stdout", "w", stdout);
		//reopen("/home/punto/stderr", "w", stderr);
		x11_window = atol(windowid);
	} else {
	*/
		x11_window = XCreateWindow(x11_display, RootWindow(x11_display, vi->screen), 0, 0, OS::get_singleton()->get_video_mode().width, OS::get_singleton()->get_video_mode().height, 0, vi->depth, InputOutput, vi->visual, CWBorderPixel|CWColormap|CWEventMask, &swa);
		ERR_FAIL_COND_V(!x11_window,ERR_UNCONFIGURED);
		XMapWindow(x11_display, x11_window);
		while(true) {
			// wait for mapnotify (window created)
			XEvent e;
			XNextEvent(x11_display, &e);
			if (e.type == MapNotify)
				break;
		}
		//};

	if (!OS::get_singleton()->get_video_mode().resizable) {
		XSizeHints *xsh;
		xsh = XAllocSizeHints();
		xsh->flags = PMinSize | PMaxSize;
		xsh->min_width = OS::get_singleton()->get_video_mode().width;
		xsh->max_width = OS::get_singleton()->get_video_mode().width;
		xsh->min_height = OS::get_singleton()->get_video_mode().height;
		xsh->max_height = OS::get_singleton()->get_video_mode().height;
		XSetWMNormalHints(x11_display, x11_window, xsh);
	}


	if (!opengl_3_context) {
		//oldstyle context:
		p->glx_context = glXCreateContext(x11_display, vi, 0, GL_TRUE);
	} else {
		static int context_attribs[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			None
		};
	
		p->glx_context = glXCreateContextAttribsARB(x11_display, fbc[0], NULL, true, context_attribs);
		ERR_FAIL_COND_V(!p->glx_context,ERR_UNCONFIGURED);
	}

	glXMakeCurrent(x11_display, x11_window, p->glx_context);

	/*
	glWrapperInit(wrapper_get_proc_address);
	glFlush();
	
	glXSwapBuffers(x11_display,x11_window);
*/
	//glXMakeCurrent(x11_display, None, NULL);

	return OK;
}

int ContextGL_X11::get_window_width() {

	XWindowAttributes xwa;
	XGetWindowAttributes(x11_display,x11_window,&xwa);
	
	return xwa.width;
}
int ContextGL_X11::get_window_height() {
	XWindowAttributes xwa;
	XGetWindowAttributes(x11_display,x11_window,&xwa);
	
	return xwa.height;

}


ContextGL_X11::ContextGL_X11(::Display *p_x11_display,::Window &p_x11_window,const OS::VideoMode& p_default_video_mode,bool p_opengl_3_context) : x11_window(p_x11_window) {

	default_video_mode=p_default_video_mode;
	x11_display=p_x11_display;
	
	opengl_3_context=p_opengl_3_context;
	
	double_buffer=false;
	direct_render=false;
	glx_minor=glx_major=0;
	p = memnew( ContextGL_X11_Private );
	p->glx_context=0;
}


ContextGL_X11::~ContextGL_X11() {

	memdelete( p );
}


#elif defined(GLES2_ENABLED)

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <GLES2/gl2.h>


#define GLX_CONTEXT_MAJOR_VERSION_ARB		0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB		0x2092

/*typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

struct ContextGL_X11_Private { 

	::GLXContext glx_context;
};*/


void ContextGL_X11::release_current() {

	eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

void ContextGL_X11::make_current() {

	eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
}
void ContextGL_X11::swap_buffers() {

	eglSwapBuffers(egl_display, egl_surface);
}

const char* EGLErrorString()
{
	EGLint nErr = eglGetError();
	switch(nErr){
		case EGL_SUCCESS:
			return "EGL_SUCCESS";
		case EGL_BAD_DISPLAY:
			return "EGL_BAD_DISPLAY";
		case EGL_NOT_INITIALIZED:
			return "EGL_NOT_INITIALIZED";
		case EGL_BAD_ACCESS:
			return "EGL_BAD_ACCESS";
		case EGL_BAD_ALLOC:
			return "EGL_BAD_ALLOC";
		case EGL_BAD_ATTRIBUTE:
			return "EGL_BAD_ATTRIBUTE";
		case EGL_BAD_CONFIG:
			return "EGL_BAD_CONFIG";
		case EGL_BAD_CONTEXT:
			return "EGL_BAD_CONTEXT";
		case EGL_BAD_CURRENT_SURFACE:
			return "EGL_BAD_CURRENT_SURFACE";
		case EGL_BAD_MATCH:
			return "EGL_BAD_MATCH";
		case EGL_BAD_NATIVE_PIXMAP:
			return "EGL_BAD_NATIVE_PIXMAP";
		case EGL_BAD_NATIVE_WINDOW:
			return "EGL_BAD_NATIVE_WINDOW";
		case EGL_BAD_PARAMETER:
			return "EGL_BAD_PARAMETER";
		case EGL_BAD_SURFACE:
			return "EGL_BAD_SURFACE";
		default:
			return "unknown";
	}
}

Error ContextGL_X11::initialize()
{

	if (!x11_display) {
		printf("no X11 display ... creating my own ...\n");
		x11_display = XOpenDisplay( NULL );
	}

	//_______________________________________________________________________________________
	// code from http://pandorawiki.org/GLESGAE#GLES1Context.h
    printf("I'm creating a XWindow!\n");
	// Create the actual window and store the pointer.
	x11_window = XCreateWindow(x11_display			// Pointer to the Display
				, RootWindow(x11_display,DefaultScreen(x11_display))	// Parent Window
				, 0				// X of top-left corner
				, 0				// Y of top-left corner
				, OS::get_singleton()->get_video_mode().width				// requested width
				, OS::get_singleton()->get_video_mode().height			// requested height
				, 0				// border width
				, CopyFromParent		// window depth
				, CopyFromParent		// window class - InputOutput / InputOnly / CopyFromParent
				, CopyFromParent		// visual type
				, 0				// value mask 
				, 0);				// attributes
						
	// Map the window to the display.
	XMapWindow(x11_display, x11_window);
	
	printf("I'm setting up a EGL Display!\n");
	// Initialise the EGL Display
	EGLBoolean eglb = eglInitialize(egl_display, NULL, NULL);
    if (0 == eglb)
    {
		printf("failed to init egl..\n");
        printf(EGLErrorString());
		printf("\n");
	}

	// Now we want to find an EGL Surface that will work for us...
	EGLint eglAttribs[] = {
		EGL_BUFFER_SIZE, 16			// 16bit Colour Buffer
    ,   EGL_DEPTH_SIZE, 24
    ,   EGL_SURFACE_TYPE, EGL_WINDOW_BIT
	,	EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT	// We want an ES2 config
	,	EGL_NONE
	};

	EGLConfig  eglConfig;
	EGLint     numConfig;
	if (0 == eglChooseConfig(egl_display, eglAttribs, &eglConfig, 1, &numConfig)) {
		printf("failed to get context..\n");
	}

	// Create the actual surface based upon the list of configs we've just gotten...
	egl_surface = eglCreateWindowSurface(egl_display, eglConfig, NULL, NULL);
	if (EGL_NO_SURFACE == egl_surface) {
		printf("failed to get surface..\n");
	}

	// Setup the EGL context
	EGLint contextAttribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2
	,	EGL_NONE
	};

	// Create our Context
	egl_context = eglCreateContext (egl_display, eglConfig, EGL_NO_CONTEXT, contextAttribs);
	if (EGL_NO_CONTEXT == egl_context) {
		printf("failed to get context...\n");
	}

	// Bind the Display, Surface and Contexts together
	eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
	

	// Setup the viewport
	glViewport(0, 0, default_video_mode.width, default_video_mode.height);
	//______________________________________________________________________________________
	printf("done!\n");
	return OK;
}

int ContextGL_X11::get_window_width() {

	XWindowAttributes xwa;
	XGetWindowAttributes(x11_display,x11_window,&xwa);
	
	return xwa.width;
}
int ContextGL_X11::get_window_height() {
	XWindowAttributes xwa;
	XGetWindowAttributes(x11_display,x11_window,&xwa);
	
	return xwa.height;

}


ContextGL_X11::ContextGL_X11(::Display *p_x11_display,::Window &p_x11_window,const OS::VideoMode& p_default_video_mode,bool p_opengl_3_context) : x11_window(p_x11_window) {

	default_video_mode=p_default_video_mode;
	x11_display=p_x11_display;
	egl_display=eglGetDisplay((EGLNativeDisplayType)EGL_DEFAULT_DISPLAY);
	
	opengl_3_context=p_opengl_3_context;
	
	double_buffer=false;
	direct_render=false;
	glx_minor=glx_major=0;
	/*p = memnew( ContextGL_X11_Private );
	p->glx_context=0;*/
}


ContextGL_X11::~ContextGL_X11() {

	//memdelete( p );
}

#elif defined(GLES1_ENABLED)

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <GLES/gl.h>


#define GLX_CONTEXT_MAJOR_VERSION_ARB		0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB		0x2092

/*typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

struct ContextGL_X11_Private { 

	::GLXContext glx_context;
};*/


void ContextGL_X11::release_current() {

	eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

void ContextGL_X11::make_current() {

	eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
}
void ContextGL_X11::swap_buffers() {

	eglSwapBuffers(egl_display, egl_surface);
}

Error ContextGL_X11::initialize() {

	//_______________________________________________________________________________________
	// code from http://pandorawiki.org/GLESGAE#GLES1Context.h
	printf("I'm creating a XWindow!");
	// Create the actual window and store the pointer.
	x11_window = XCreateWindow(x11_display			// Pointer to the Display
				, RootWindow(x11_display,DefaultScreen(x11_display))	// Parent Window
				, 0				// X of top-left corner
				, 0				// Y of top-left corner
				, OS::get_singleton()->get_video_mode().width				// requested width
				, OS::get_singleton()->get_video_mode().height			// requested height
				, 0				// border width
				, CopyFromParent		// window depth
				, CopyFromParent		// window class - InputOutput / InputOnly / CopyFromParent
				, CopyFromParent		// visual type
				, 0				// value mask 
				, 0);				// attributes
						
	// Map the window to the display.
	XMapWindow(x11_display, x11_window);
	
	printf("I'm setting up a EGL Display!\n");
	// Initialise the EGL Display
	if (0 == eglInitialize(egl_display, NULL, NULL)) {
		printf("failed to init egl..\n");
	}

	// Now we want to find an EGL Surface that will work for us...
	EGLint eglAttribs[] = {
		EGL_BUFFER_SIZE, 16			// 16bit Colour Buffer
    ,   EGL_DEPTH_SIZE, 24
	,	EGL_NONE
	};

	EGLConfig  eglConfig;
	EGLint     numConfig;
	if (0 == eglChooseConfig(egl_display, eglAttribs, &eglConfig, 1, &numConfig)) {
		printf("failed to get context..\n");
	}

	// Create the actual surface based upon the list of configs we've just gotten...
	egl_surface = eglCreateWindowSurface(egl_display, eglConfig, reinterpret_cast<EGLNativeWindowType>(/*x11_window.getWindow()*/x11_window), NULL);
	if (EGL_NO_SURFACE == egl_surface) {
		printf("failed to get surface..\n");
	}

	// Setup the EGL context
	EGLint contextAttribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 1
	,	EGL_NONE
	};

	// Create our Context
	egl_context = eglCreateContext (egl_display, eglConfig, EGL_NO_CONTEXT, contextAttribs);
	if (EGL_NO_CONTEXT == egl_context) {
		printf("failed to get context...\n");
	}

	// Bind the Display, Surface and Contexts together
	eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
	

	// Setup the viewport
	glViewport(0, 0, default_video_mode.width, default_video_mode.height);
	//______________________________________________________________________________________
	printf("done!\n");
	return OK;
}

int ContextGL_X11::get_window_width() {

	XWindowAttributes xwa;
	XGetWindowAttributes(x11_display,x11_window,&xwa);
	
	return xwa.width;
}
int ContextGL_X11::get_window_height() {
	XWindowAttributes xwa;
	XGetWindowAttributes(x11_display,x11_window,&xwa);
	
	return xwa.height;

}


ContextGL_X11::ContextGL_X11(::Display *p_x11_display,::Window &p_x11_window,const OS::VideoMode& p_default_video_mode,bool p_opengl_3_context) : x11_window(p_x11_window) {

	default_video_mode=p_default_video_mode;
	x11_display=p_x11_display;
	egl_display=eglGetDisplay((EGLNativeDisplayType)x11_display);
	
	opengl_3_context=p_opengl_3_context;
	
	double_buffer=false;
	direct_render=false;
	glx_minor=glx_major=0;
	/*p = memnew( ContextGL_X11_Private );
	p->glx_context=0;*/
}


ContextGL_X11::~ContextGL_X11() {

	//memdelete( p );
}

#endif
#endif
