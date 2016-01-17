/*************************************************************************/
/*  godot_module.cpp                                                     */
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
#include "opengl_context.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ppapi/cpp/instance.h"
#include "ppapi/cpp/module.h"
#include "ppapi/gles2/gl2ext_ppapi.h"

#include "ppapi/cpp/rect.h"
#include "ppapi/cpp/size.h"
#include "ppapi/cpp/var.h"
#include "geturl_handler.h"

#include "core/variant.h"
#include "os_nacl.h"

extern int nacl_main(int argc, const char** argn, const char** argv);
extern void nacl_cleanup();

static String pkg_url;

pp::Instance* godot_instance = NULL;

struct StateData {
	int arg_count;
	Array args;
	String method;
};

extern OSNacl* os_nacl;

class GodotInstance : public pp::Instance {

	enum State {
		STATE_METHOD,
		STATE_PARAM_COUNT,
		STATE_PARAMS,
		STATE_CALL,
	};

	State state;
	StateData* sd;
	SharedOpenGLContext opengl_context_;
	int width;
	int height;

	#define MAX_ARGS 64
	uint32_t init_argc;
	char* init_argn[MAX_ARGS];
	char* init_argv[MAX_ARGS];

	bool package_loaded;
	GetURLHandler* package_pending;

public:
	explicit GodotInstance(PP_Instance instance) : pp::Instance(instance) {
		printf("GodotInstance!\n");
		state = STATE_METHOD;
		RequestInputEvents(PP_INPUTEVENT_CLASS_MOUSE | PP_INPUTEVENT_CLASS_KEYBOARD | PP_INPUTEVENT_CLASS_WHEEL | PP_INPUTEVENT_CLASS_TOUCH);
		sd = NULL;
		package_pending = NULL;
		package_loaded = false;
		godot_instance = this;
	}
	virtual ~GodotInstance() {

		nacl_cleanup();
	}

	/// Called by the browser to handle the postMessage() call in Javascript.
	/// Detects which method is being called from the message contents, and
	/// calls the appropriate function.  Posts the result back to the browser
	/// asynchronously.
	/// @param[in] var_message The message posted by the browser.  The possible
	///     messages are 'fortyTwo' and 'reverseText:Hello World'.  Note that
	///     the 'reverseText' form contains the string to reverse following a ':'
	///     separator.
	virtual void HandleMessage(const pp::Var& var_message);

	bool HandleInputEvent(const pp::InputEvent& event);

	bool Init(uint32_t argc, const char* argn[], const char* argv[]) {

		printf("******* init! %i, %p, %p\n", argc, argn, argv);
		fflush(stdout);
		if (opengl_context_ == NULL) {
			opengl_context_.reset(new OpenGLContext(this));
		};
		opengl_context_->InvalidateContext(this);
		opengl_context_->ResizeContext(pp::Size(0, 0));
		int current = opengl_context_->MakeContextCurrent(this);
		printf("current is %i\n", current);

		os_nacl = new OSNacl;

		pkg_url = "";
		for (uint32_t i=0; i<argc; i++) {
			if (strcmp(argn[i], "package") == 0) {
				pkg_url = argv[i];
			};
		};

		sd = memnew(StateData);

		if (pkg_url == "") {
			nacl_main(argc, argn, argv);
		} else {
			printf("starting package %ls\n", pkg_url.c_str());
			init_argc = MIN(argc, MAX_ARGS-1);
			for (uint32_t i=0; i<argc; i++) {

				int nlen = strlen(argn[i]);
				init_argn[i] = (char*)memalloc(nlen+1);
				strcpy(init_argn[i], argn[i]);
				init_argn[i+1] = NULL;

				int len = strlen(argv[i]);
				init_argv[i] = (char*)memalloc(len+1);
				strcpy(init_argv[i], argv[i]);
				init_argv[i+1] = NULL;
			};
			package_pending = memnew(GetURLHandler(this, pkg_url));
			package_pending->Start();
		};
		return true;
	};

	// Called whenever the in-browser window changes size.
	virtual void DidChangeView(const pp::Rect& position, const pp::Rect& clip) {

		if (position.size().width() == width &&
			position.size().height() == height)
			return;  // Size didn't change, no need to update anything.

		if (opengl_context_ == NULL) {
			opengl_context_.reset(new OpenGLContext(this));
		};
		opengl_context_->InvalidateContext(this);
		opengl_context_->ResizeContext(position.size());
		if (!opengl_context_->MakeContextCurrent(this))
			return;

		width = position.size().width();
		height = position.size().height();
		// init gl here?
		OS::VideoMode vm;
		vm.width = width;
		vm.height = height;
		vm.resizable = false;
		vm.fullscreen = true;
		OS::get_singleton()->set_video_mode(vm, 0);

		DrawSelf();
	};

	// Called to draw the contents of the module's browser area.
	void DrawSelf() {

		if (opengl_context_ == NULL)
			return;

		opengl_context_->FlushContext();
	};
};

static Variant to_variant(const pp::Var& p_var) {

	if (p_var.is_undefined() || p_var.is_null())
		return Variant();
	if (p_var.is_bool())
		return Variant(p_var.AsBool());
	if (p_var.is_double())
		return Variant(p_var.AsDouble());
	if (p_var.is_int())
		return Variant((int64_t)p_var.AsInt());
	if (p_var.is_string())
		return Variant(String::utf8(p_var.AsString().c_str()));

	return Variant();
};

void GodotInstance::HandleMessage(const pp::Var& var_message) {

	switch (state) {

		case STATE_METHOD: {

			ERR_FAIL_COND(!var_message.is_string());
			sd->method = var_message.AsString().c_str();
			state = STATE_PARAM_COUNT;
		} break;
		case STATE_PARAM_COUNT: {

			ERR_FAIL_COND(!var_message.is_number());
			sd->arg_count = var_message.AsInt();
			state = sd->arg_count>0?STATE_PARAMS:STATE_CALL;

		} break;
		case STATE_PARAMS: {

			Variant p = to_variant(var_message);
			sd->args.push_back(p);
			if (sd->args.size() >= sd->arg_count)
				state = STATE_CALL;
		} break;
		default:
			break;
	};

	if (state == STATE_CALL) {

		// call
		state = STATE_METHOD;


		if (sd->method == "package_finished") {

			GetURLHandler::Status status = package_pending->get_status();
			printf("status is %i, %i, %i\n", status, GetURLHandler::STATUS_ERROR, GetURLHandler::STATUS_COMPLETED);
			if (status == GetURLHandler::STATUS_ERROR) {
				printf("Error fetching package!\n");
			};
            if (status == GetURLHandler::STATUS_COMPLETED) {

				OSNacl* os = (OSNacl*)OS::get_singleton();
				os->add_package(pkg_url, package_pending->get_data());
			};
            memdelete(package_pending);
			package_pending = NULL;

            package_loaded = true;

            opengl_context_->MakeContextCurrent(this);
            nacl_main(init_argc, (const char**)init_argn, (const char**)init_argv);
            for (uint32_t i=0; i<init_argc; i++) {
				memfree(init_argn[i]);
				memfree(init_argv[i]);
			};
		};

		if (sd->method == "get_package_status") {

			if (package_loaded) {
				// post "loaded"
				PostMessage("loaded");
			} else if (package_pending == NULL) {
				// post "none"
				PostMessage("none");
			} else {
				// post package_pending->get_bytes_read();
				PostMessage(package_pending->get_bytes_read());
			};
		};
	};
}

bool GodotInstance::HandleInputEvent(const pp::InputEvent& event) {

	OSNacl* os = (OSNacl*)OS::get_singleton();
	os->handle_event(event);
	return true;
};

class GodotModule : public pp::Module {
 public:
  GodotModule() : pp::Module() {}
  virtual ~GodotModule() {
	  glTerminatePPAPI();
  }

  /// Create and return a GodotInstance object.
  /// @param[in] instance a handle to a plug-in instance.
  /// @return a newly created GodotInstance.
  /// @note The browser is responsible for calling @a delete when done.
  virtual pp::Instance* CreateInstance(PP_Instance instance) {
	printf("CreateInstance! %x\n", instance);
	return new GodotInstance(instance);
  }

  /// Called by the browser when the module is first loaded and ready to run.
  /// This is called once per module, not once per instance of the module on
  /// the page.
  virtual bool Init() {
	printf("GodotModule::init!\n");
	return glInitializePPAPI(get_browser_interface());
  }
};

namespace pp {
/// Factory function called by the browser when the module is first loaded.
/// The browser keeps a singleton of this module.  It calls the
/// CreateInstance() method on the object you return to make instances.  There
/// is one instance per <embed> tag on the page.  This is the main binding
/// point for your NaCl module with the browser.
/// @return new GodotModule.
/// @note The browser is responsible for deleting returned @a Module.
Module* CreateModule() {
  printf("CreateModule!\n");
  return new GodotModule();
}
}  // namespace pp
