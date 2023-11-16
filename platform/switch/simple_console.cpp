#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <switch.h>

#include <EGL/egl.h> // EGL library
#include <EGL/eglext.h> // EGL extensions
#include <GL/gl.h>
#include <GLES3/gl3.h>
//#include "glad.h" // glad library (OpenGL loader)


//-----------------------------------------------------------------------------
// EGL initialization
//-----------------------------------------------------------------------------

static EGLDisplay s_display;
static EGLContext s_context;
static EGLSurface s_surface;

static bool initEgl(NWindow *win) {
	// Connect to the EGL default display
	s_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (!s_display) {
		printf("Could not connect to display! error: %d", eglGetError());
		goto _fail0;
	}

	// Initialize the EGL display connection
	eglInitialize(s_display, nullptr, nullptr);

	// Select OpenGL (Core) as the desired graphics API
	if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
		printf("Could not set API! error: %d", eglGetError());
		goto _fail1;
	}

	// Get an appropriate EGL framebuffer configuration
	EGLConfig config;
	EGLint numConfigs;
	static const EGLint framebufferAttributeList[] = {
		EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_ALPHA_SIZE, 8,
		EGL_DEPTH_SIZE, 24,
		EGL_STENCIL_SIZE, 8,
		EGL_NONE
	};
	eglChooseConfig(s_display, framebufferAttributeList, &config, 1, &numConfigs);
	if (numConfigs == 0) {
		printf("No config found! error: %d", eglGetError());
		goto _fail1;
	}

	// Create an EGL window surface
	s_surface = eglCreateWindowSurface(s_display, config, win, nullptr);
	if (!s_surface) {
		printf("Surface creation failed! error: %d", eglGetError());
		goto _fail1;
	}

	// Create an EGL rendering context
	static const EGLint contextAttributeList[] = {
		EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
		EGL_CONTEXT_MAJOR_VERSION_KHR, 4,
		EGL_CONTEXT_MINOR_VERSION_KHR, 3,
		EGL_NONE
	};
	s_context = eglCreateContext(s_display, config, EGL_NO_CONTEXT, contextAttributeList);
	if (!s_context) {
		printf("Context creation failed! error: %d", eglGetError());
		goto _fail2;
	}

	// Connect the context to the surface
	eglMakeCurrent(s_display, s_surface, s_surface, s_context);
	return true;

_fail2:
	eglDestroySurface(s_display, s_surface);
	s_surface = nullptr;
_fail1:
	eglTerminate(s_display);
	s_display = nullptr;
_fail0:
	return false;
}

static void deinitEgl() {
	if (s_display) {
		eglMakeCurrent(s_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		if (s_context) {
			eglDestroyContext(s_display, s_context);
			s_context = nullptr;
		}
		if (s_surface) {
			eglDestroySurface(s_display, s_surface);
			s_surface = nullptr;
		}
		eglTerminate(s_display);
		s_display = nullptr;
	}
}

//-----------------------------------------------------------------------------
// Main program
//-----------------------------------------------------------------------------

static void setMesaConfig() {
	// Uncomment below to disable error checking and save CPU time (useful for production):
	//setenv("MESA_NO_ERROR", "1", 1);

	// Uncomment below to enable Mesa logging:
	setenv("EGL_LOG_LEVEL", "debug", 1);
	setenv("MESA_VERBOSE", "all", 1);
	setenv("NOUVEAU_MESA_DEBUG", "1", 1);

	// Uncomment below to enable shader debugging in Nouveau:
	//setenv("NV50_PROG_OPTIMIZE", "0", 1);
	//setenv("NV50_PROG_DEBUG", "1", 1);
	//setenv("NV50_PROG_CHIPSET", "0x120", 1);
}

static const char *const vertexShaderSource = R"text(
    #version 330 core

    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;

    out vec3 ourColor;

    void main()
    {
        gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        ourColor = aColor;
    }
)text";

static const char *const fragmentShaderSource = R"text(
    #version 330 core

    in vec3 ourColor;

    out vec4 fragColor;

    void main()
    {
        fragColor = vec4(ourColor, 1.0f);
    }
)text";

static GLuint createAndCompileShader(GLenum type, const char *source) {
	GLint success;
	GLchar msg[512];

	GLuint handle = glCreateShader(type);
	if (!handle) {
		printf("%u: cannot create shader", type);
		return 0;
	}
	glShaderSource(handle, 1, &source, nullptr);
	glCompileShader(handle);
	glGetShaderiv(handle, GL_COMPILE_STATUS, &success);

	if (!success) {
		glGetShaderInfoLog(handle, sizeof(msg), nullptr, msg);
		printf("%u: %s\n", type, msg);
		glDeleteShader(handle);
		return 0;
	}

	return handle;
}

static GLuint s_program;
static GLuint s_vao, s_vbo;

static void sceneInit() {
	GLint vsh = createAndCompileShader(GL_VERTEX_SHADER, vertexShaderSource);
	GLint fsh = createAndCompileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

	s_program = glCreateProgram();
	glAttachShader(s_program, vsh);
	glAttachShader(s_program, fsh);
	glLinkProgram(s_program);

	GLint success;
	glGetProgramiv(s_program, GL_LINK_STATUS, &success);
	if (!success) {
		char buf[512];
		glGetProgramInfoLog(s_program, sizeof(buf), nullptr, buf);
		printf("Link error: %s", buf);
	}
	glDeleteShader(vsh);
	glDeleteShader(fsh);

	struct Vertex {
		float position[3];
		float color[3];
	};

	static const Vertex vertices[] = {
		{ { -0.5f, -0.5f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
		{ { 0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
		{ { 0.0f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f } },
	};

	glGenVertexArrays(1, &s_vao);
	glGenBuffers(1, &s_vbo);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(s_vao);

	glBindBuffer(GL_ARRAY_BUFFER, s_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, position));
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, color));
	glEnableVertexAttribArray(1);

	// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	glBindVertexArray(0);
}

static void sceneRender() {
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw our first triangle
	glUseProgram(s_program);
	glBindVertexArray(s_vao); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

static void sceneExit() {
	glDeleteBuffers(1, &s_vbo);
	glDeleteVertexArrays(1, &s_vao);
	glDeleteProgram(s_program);
}

int main(int argc, char *argv[]) {
	// Set mesa configuration (useful for debugging)
	setMesaConfig();

	socketInitializeDefault();
	nxlinkStdio();
	romfsInit();

	NWindow* window = nwindowGetDefault();

	printf("windows %p\n", window);
	unsigned int w,h;
	nwindowGetDimensions(window,&w,&h);
	printf("windows size (%d,%d)\n",w,h);
	printf("windows valid %d\n", nwindowIsValid(window));

	// Initialize EGL on the default window
	if (!initEgl(window))
		return EXIT_FAILURE;

	// Load OpenGL routines using glad
	//gladLoadGLLoader((GLADloadproc) eglGetProcAddress);

	// Initialize our scene
	sceneInit();

	// Configure our supported input layout: a single player with standard controller styles
	padConfigureInput(1, HidNpadStyleSet_NpadStandard);

	// Initialize the default gamepad (which reads handheld mode inputs as well as the first connected controller)
	PadState pad;
	padInitializeDefault(&pad);

	// Main graphics loop
	while (appletMainLoop()) {
		// Get and process input
		padUpdate(&pad);
		u32 kDown = padGetButtonsDown(&pad);
		if (kDown & HidNpadButton_Plus)
			break;

		// Render stuff!
		sceneRender();
		eglSwapBuffers(s_display, s_surface);
	}

	// Deinitialize our scene
	sceneExit();

	// Deinitialize EGL
	deinitEgl();

	romfsExit();
	socketExit();
	return EXIT_SUCCESS;
}