#include "context_gl_android.h"

#include <GLES2/gl2.h>
#include "os/os.h"
void ContextGLAndroid::make_current() {

};

int ContextGLAndroid::get_window_width() {

	return OS::get_singleton()->get_default_video_mode().width;
};

int ContextGLAndroid::get_window_height() {

	return OS::get_singleton()->get_default_video_mode().height;

};

void ContextGLAndroid::swap_buffers() {

};

Error ContextGLAndroid::initialize() {

	return OK;
};


ContextGLAndroid::ContextGLAndroid() {


};

ContextGLAndroid::~ContextGLAndroid() {

};

