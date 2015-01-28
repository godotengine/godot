/*************************************************/
/*  context_gl.cpp                               */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#include "context_gl.h"


#if defined(OPENGL_ENABLED)  || defined(GLES2_ENABLED)



ContextGL *ContextGL::singleton=NULL;

ContextGL *ContextGL::get_singleton() {

	return singleton;
}


ContextGL::ContextGL() {
	
	ERR_FAIL_COND(singleton);
	
	singleton=this;
}


ContextGL::~ContextGL() {

	if (singleton==this)
		singleton=NULL;
}

#endif
