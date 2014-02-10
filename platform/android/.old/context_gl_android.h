#ifndef CONTEXT_GL_ANDROID_H
#define CONTEXT_GL_ANDROID_H

class ContextGLAndroid : public ContextGL {

	enum {
		COMMAND_BUFFER_SIZE = 1024 * 1024,
	};

public:

	virtual void make_current();

	virtual int get_window_width();
	virtual int get_window_height();
	virtual void swap_buffers();

	virtual Error initialize();

	ContextGLAndroid();
	~ContextGLAndroid();
};

#endif // CONTEXT_GL_ANDROID_H
