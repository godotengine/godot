#include "gl_context_egl.h"

#include "EGL/eglext.h"

using namespace Platform;

void ContextEGL::release_current() {

	eglMakeCurrent(mEglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, mEglContext);
};

void ContextEGL::make_current() {

	eglMakeCurrent(mEglDisplay, mEglSurface, mEglSurface, mEglContext);
};

int ContextEGL::get_window_width() {

	return width;
};

int ContextEGL::get_window_height() {

	return height;
};

void ContextEGL::reset() {

	cleanup();

	window = CoreWindow::GetForCurrentThread();
	initialize();
};

void ContextEGL::swap_buffers() {

	if (eglSwapBuffers(mEglDisplay, mEglSurface) != EGL_TRUE)
	{
		cleanup();

		window = CoreWindow::GetForCurrentThread();
		initialize();

		// tell rasterizer to reload textures and stuff?
	}
};

Error ContextEGL::initialize() {

	EGLint configAttribList[] = {
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_ALPHA_SIZE, 8,
		EGL_DEPTH_SIZE, 8,
		EGL_STENCIL_SIZE, 8,
		EGL_SAMPLE_BUFFERS, 0,
		EGL_NONE
	};

	EGLint surfaceAttribList[] = {
		EGL_NONE, EGL_NONE
	};

	EGLint numConfigs = 0;
	EGLint majorVersion = 1;
	EGLint minorVersion = 0;
	EGLDisplay display = EGL_NO_DISPLAY;
	EGLContext context = EGL_NO_CONTEXT;
	EGLSurface surface = EGL_NO_SURFACE;
	EGLConfig config = nullptr;
	EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE, EGL_NONE };

	try {

		const EGLint displayAttributes[] =
		{
			EGL_PLATFORM_ANGLE_TYPE_ANGLE, EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE,
			EGL_PLATFORM_ANGLE_MAX_VERSION_MAJOR_ANGLE, 9,
			EGL_PLATFORM_ANGLE_MAX_VERSION_MINOR_ANGLE, 3,
			EGL_NONE,
		};

		PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));

		if (!eglGetPlatformDisplayEXT)
		{
			throw Exception::CreateException(E_FAIL, L"Failed to get function eglGetPlatformDisplayEXT");
		}

		display = eglGetPlatformDisplayEXT(EGL_PLATFORM_ANGLE_ANGLE, EGL_DEFAULT_DISPLAY, displayAttributes);

		if (display == EGL_NO_DISPLAY)
		{
			throw Exception::CreateException(E_FAIL, L"Failed to get default EGL display");
		}

		if (eglInitialize(display, &majorVersion, &minorVersion) == EGL_FALSE)
		{
			throw Exception::CreateException(E_FAIL, L"Failed to initialize EGL");
		}

		if (eglGetConfigs(display, NULL, 0, &numConfigs) == EGL_FALSE)
		{
			throw Exception::CreateException(E_FAIL, L"Failed to get EGLConfig count");
		}

		if (eglChooseConfig(display, configAttribList, &config, 1, &numConfigs) == EGL_FALSE)
		{
			throw Exception::CreateException(E_FAIL, L"Failed to choose first EGLConfig count");
		}

		surface = eglCreateWindowSurface(display, config, reinterpret_cast<IInspectable*>(window), surfaceAttribList);
		if (surface == EGL_NO_SURFACE)
		{
			throw Exception::CreateException(E_FAIL, L"Failed to create EGL fullscreen surface");
		}

		context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
		if (context == EGL_NO_CONTEXT)
		{
			throw Exception::CreateException(E_FAIL, L"Failed to create EGL context");
		}

		if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE)
		{
			throw Exception::CreateException(E_FAIL, L"Failed to make fullscreen EGLSurface current");
		}
	} catch (...) {
		return FAILED;
	};

	mEglDisplay = display;
	mEglSurface = surface;
	mEglContext = context;

	eglQuerySurface(display,surface,EGL_WIDTH,&width);
	eglQuerySurface(display,surface,EGL_HEIGHT,&height);

	return OK;
};

void ContextEGL::cleanup() {

	if (mEglDisplay != EGL_NO_DISPLAY && mEglSurface != EGL_NO_SURFACE)
	{
		eglDestroySurface(mEglDisplay, mEglSurface);
		mEglSurface = EGL_NO_SURFACE;
	}

	if (mEglDisplay != EGL_NO_DISPLAY && mEglContext != EGL_NO_CONTEXT)
	{
		eglDestroyContext(mEglDisplay, mEglContext);
		mEglContext = EGL_NO_CONTEXT;
	}

	if (mEglDisplay != EGL_NO_DISPLAY)
	{
		eglTerminate(mEglDisplay);
		mEglDisplay = EGL_NO_DISPLAY;
	}
};

ContextEGL::ContextEGL(CoreWindow^ p_window) :
	mEglDisplay(EGL_NO_DISPLAY),
	mEglContext(EGL_NO_CONTEXT),
	mEglSurface(EGL_NO_SURFACE)
 {

	window = p_window;
};

ContextEGL::~ContextEGL() {

	cleanup();
};

