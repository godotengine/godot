/*************************************************************************/
/*  pepper_main.cpp                                                      */
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
 IMPORTANT:  This Apple software is supplied to you by Apple Computer, Inc.
 ("Apple") in consideration of your agreement to the following terms, and
 your use, installation, modification or redistribution of this Apple software
 constitutes acceptance of these terms.  If you do not agree with these terms,
 please do not use, install, modify or redistribute this Apple software.

 In consideration of your agreement to abide by the following terms, and
 subject to these terms, Apple grants you a personal, non-exclusive license,
 under Apple's copyrights in this original Apple software
 (the "Apple Software"), to use, reproduce, modify and redistribute the Apple
 Software, with or without modifications, in source and/or binary forms;
 provided that if you redistribute the Apple Software in its entirety and
 without modifications, you must retain this notice and the following text and
 disclaimers in all such redistributions of the Apple Software.  Neither the
 name, trademarks, service marks or logos of Apple Computer, Inc. may be used
 to endorse or promote products derived from the Apple Software without
 specific prior written permission from Apple. Except as expressly stated in
 this notice, no other rights or licenses, express or implied, are granted by
 Apple herein, including but not limited to any patent rights that may be
 infringed by your derivative works or by other works in which the Apple
 Software may be incorporated.

 The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
 WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
 WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION ALONE OR IN
 COMBINATION WITH YOUR PRODUCTS.

 IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
 AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER UNDER
 THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY OR
 OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>

#include "os_nacl.h"
#include <GLES2/gl2.h>

#include <nacl/npupp.h>
#include <pgl/pgl.h>
#include <string.h>
#include <stdint.h>

static NPNetscapeFuncs kBrowserFuncs = { 0 };
static NPNetscapeFuncs* browser = &kBrowserFuncs;

static NPDevice* device3d_ = NULL;
static PGLContext pgl_context_;
static NPDeviceContext3D context3d_;
static int width_;
static int height_;

extern int nacl_main(int argc, char** argn, char** argv);
extern void nacl_cleanup();

NPExtensions* extensions = NULL;
static NPP npp_;

const int32_t kCommandBufferSize = 1024 * 1024;

// Plugin entry points
extern "C" {

// Plugin entry points

// Entrypoints -----------------------------------------------------------------

NPError NP_GetEntryPoints(NPPluginFuncs* plugin_funcs) {
  plugin_funcs->version = 11;
  plugin_funcs->size = sizeof(plugin_funcs);
  plugin_funcs->newp = NPP_New;
  plugin_funcs->destroy = NPP_Destroy;
  plugin_funcs->setwindow = NPP_SetWindow;
  plugin_funcs->newstream = NPP_NewStream;
  plugin_funcs->destroystream = NPP_DestroyStream;
  plugin_funcs->asfile = NPP_StreamAsFile;
  plugin_funcs->writeready = NPP_WriteReady;
  plugin_funcs->write = (NPP_WriteUPP)NPP_Write;
  plugin_funcs->print = NPP_Print;
  plugin_funcs->event = NPP_HandleEvent;
  plugin_funcs->urlnotify = NPP_URLNotify;
  plugin_funcs->getvalue = NPP_GetValue;
  plugin_funcs->setvalue = NPP_SetValue;

  return NPERR_NO_ERROR;
}

NPError NP_Shutdown() {
  pglTerminate();
  return NPERR_NO_ERROR;
}

NPError NP_GetValue(void* instance, NPPVariable variable, void* value);
char* NP_GetMIMEDescription();

NPError NP_Initialize(NPNetscapeFuncs* browser_funcs,
							NPPluginFuncs* plugin_funcs) {
	printf("NPP_Initialize\n");
  memcpy(&kBrowserFuncs, browser_funcs, sizeof(kBrowserFuncs));
  pglInitialize();
  return NP_GetEntryPoints(plugin_funcs);
}


}  // extern "C"

void Initialize3D() {
	// Initialize a 3D context.
	NPDeviceContext3DConfig config;
	config.commandBufferSize = kCommandBufferSize;
	NPError err = device3d_->initializeContext(npp_, &config, &context3d_);
	if (err != NPERR_NO_ERROR) {
		printf("Failed to initialize 3D context\n");
		exit(1);
	}

	// Create a PGL context.
	pgl_context_ = pglCreateContext(npp_, device3d_, &context3d_);

	// Initialize the demo GL state.
	//pglMakeCurrent(pgl_context_);
	//GLFromCPPInit();
	//pglMakeCurrent(NULL);
}

NPError NPP_New(NPMIMEType pluginType,
                NPP instance,
				uint16_t mode,
				int16_t argc, char* argn[], char* argv[],
                NPSavedData* saved) {
	printf("NPP_New\n");
	if (browser->version >= 14) {

		npp_ = instance;
		if (!extensions) {
			browser->getvalue(npp_, NPNVPepperExtensions,
			reinterpret_cast<void*>(&extensions));
			// CHECK(extensions);
		}

		printf("%s: %i\n", __FUNCTION__, __LINE__);

		device3d_ = extensions->acquireDevice(npp_, NPPepper3DDevice);
		if (device3d_ == NULL) {
			printf("Failed to acquire 3DDevice\n");
			exit(1);
		}
		printf("%s: %i\n", __FUNCTION__, __LINE__);

		/*
		deviceaudio_ = extensions->acquireDevice(npp_, NPPepperAudioDevice);
		if (deviceaudio_ == NULL) {
			printf("Failed to acquire AudioDevice\n");
			exit(1);
		}
		*/
		Initialize3D();
		pglMakeCurrent(pgl_context_);
		nacl_main(argc, argn, argv);
		pglMakeCurrent(NULL);
	};

  return NPERR_NO_ERROR;
}

NPError NPP_Destroy(NPP instance, NPSavedData** save) {

	nacl_cleanup();

	return NPERR_NO_ERROR;
}

void Destroy3D() {
	printf("destroy 3d\n");
	// Destroy the PGL context.
	pglDestroyContext(pgl_context_);
	pgl_context_ = NULL;

	// Destroy the Device3D context.
	device3d_->destroyContext(npp_, &context3d_);
}

static void iteration(void* data) {

	(void)data;
	OSNacl* os = (OSNacl*)OS::get_singleton();

	if (!pglMakeCurrent(pgl_context_) && pglGetError() == PGL_CONTEXT_LOST) {
		printf("******* Lost context! :O\n");
		Destroy3D();
		Initialize3D();
		pglMakeCurrent(pgl_context_);
	}

	glViewport(0, 0, width_, height_);

	os->iterate();

	pglSwapBuffers();
	pglMakeCurrent(NULL);

	browser->pluginthreadasynccall(npp_, iteration, NULL);
};

NPError NPP_SetWindow(NPP instance, NPWindow* window) {

	width_ = window->width;
	height_ = window->height;

	if (!pgl_context_)
		Initialize3D();

	// Schedule the first call to Draw.
	OSNacl* os = (OSNacl*)OS::get_singleton();
	OS::VideoMode vm;
	vm.width = width_;
	vm.height = height_;
	vm.resizable = false;
	vm.fullscreen = false;
	os->set_video_mode(vm);

	browser->pluginthreadasynccall(npp_, iteration, NULL);

	return NPERR_NO_ERROR;
}

NPError NPP_NewStream(NPP instance,
                      NPMIMEType type,
                      NPStream* stream,
                      NPBool seekable,
					  uint16_t* stype) {
  *stype = NP_ASFILEONLY;
  return NPERR_NO_ERROR;
}

NPError NPP_DestroyStream(NPP instance, NPStream* stream, NPReason reason) {
  return NPERR_NO_ERROR;
}

void NPP_StreamAsFile(NPP instance, NPStream* stream, const char* fname) {
}

int32_t NPP_Write(NPP instance,
                NPStream* stream,
				int32_t offset,
				int32_t len,
                void* buffer) {
  return 0;
}

int32_t NPP_WriteReady(NPP instance, NPStream* stream) {
  return 0;
}

void NPP_Print(NPP instance, NPPrint* platformPrint) {
}

int16_t NPP_HandleEvent(NPP instance, void* event) {

	OSNacl* os = (OSNacl*)OS::get_singleton();
	os->handle_event(event);
	return 1;
}

void NPP_URLNotify(NPP instance,
                   const char* url,
                   NPReason reason,
                   void* notify_data) {
  // PluginObject* obj = static_cast<PluginObject*>(instance->pdata);
}

static NPObject* Allocate(NPP npp, NPClass* npclass) {
  return new NPObject;
}

static void Deallocate(NPObject* object) {
  delete object;
}

// Return |true| if |method_name| is a recognized method.
static bool HasMethod(NPObject* obj, NPIdentifier method_name) {

  char *name = NPN_UTF8FromIdentifier(method_name);
  bool is_method = false;
  if (strcmp((const char *)name, "start_package") == 0) {
	is_method = true;
  } else if (strcmp((const char*)name, "add_package_chunk") == 0) {
	is_method = true;
  } else if (strcmp((const char*)name, "end_package") == 0) {
	is_method = true;
  } else if (strcmp((const char*)name, "start_scene") == 0) {
	is_method = true;
  }
  NPN_MemFree(name);
  return is_method;
}

// I don't know what this is
static bool InvokeDefault(NPObject *obj, const NPVariant *args,
						  uint32_t argCount, NPVariant *result) {
  if (result) {
	NULL_TO_NPVARIANT(*result);
  }
  return true;
}

static uint8_t* mem = NULL;
static int pkg_size = 0;
static String pkgname;

static bool variant_is_number(const NPVariant& v) {

	switch (v.type) {

	case NPVariantType_Int32:
	case NPVariantType_Double:
		return true;

	default:
		return false;
	}
	return false;
};

static double variant_as_number(const NPVariant& v) {

	switch (v.type) {

	case NPVariantType_Int32:
		return (double)v.value.intValue;
	case NPVariantType_Double:
		return (double)v.value.doubleValue;
	default:
		return 0;
	}

	return 0;
};

// Invoke() is called by the browser to invoke a function object whose name
// is |method_name|.
static bool Invoke(NPObject* obj,
				   NPIdentifier method_name,
				   const NPVariant *args,
				   uint32_t arg_count,
				   NPVariant *result) {
  NULL_TO_NPVARIANT(*result);
  char *name = NPN_UTF8FromIdentifier(method_name);
  if (name == NULL)
	return false;
  bool rval = false;

  OSNacl* os = (OSNacl*)OS::get_singleton();

  if (strcmp(name, "start_package") == 0) {

	  printf("arg count is %i\n", arg_count);
	  for (int i=0; i<arg_count; i++) {
		  printf("type for %i is %i\n", i, args[i].type);
	  }
	  if (arg_count != 2) {
		  return false;	// assuming "false" means error
	  };

	  if (args[0].type != NPVariantType_String) {
		  printf("arg 0 not string type %i\n", args[1].type);
		  return false;
	  };

	  if (!variant_is_number(args[1])) {

		  printf("arg 1 not number, type %i\n", args[1].type);
		  return false;
	  };

	  pkgname = String::utf8(args[0].value.stringValue.UTF8Characters, args[0].value.stringValue.UTF8Length);
	  pkg_size = (int)variant_as_number(args[1]);
	  mem = (uint8_t*)malloc(pkg_size);

	  printf("args %ls, %lf\n", pkgname.c_str(), variant_as_number(args[1]));

	  return true;
  };

  if (strcmp(name, "add_package_chunk") == 0) {

	  if (arg_count != 3) { // assuming arg_count starts from 1
		  return false;	// assuming "false" means error
	  };

	  if (!variant_is_number(args[0])) return false;
	  if (args[1].type != NPVariantType_String) return false;
	  if (!variant_is_number(args[2])) return false;

	  if (!mem)
		  return false;

	  int ofs = variant_as_number(args[0]);
	  int len = variant_as_number(args[2]);

	  String s;
	  if (s.parse_utf8(args[1].value.stringValue.UTF8Characters, args[1].value.stringValue.UTF8Length)) {
		  printf("error parsing?\n");
	  };
	  uint8_t* dst = mem + ofs;
	  for (int i=0; i<len; i++) {

		  dst[i] = s[i];
	  };

	  //memcpy(mem + ofs, args[1].value.stringValue.UTF8Characters, len);
	  return true;
  };

  if (strcmp(name, "end_package") == 0) {

	  os->add_package(pkgname, mem, pkg_size);
	  return true;
  };


  if (strcmp(name, "start_scene") == 0) {
printf("start_scene!\n");
	  if (arg_count != 1) {
		  return false;
	  };

	  if (args[0].type != NPVariantType_String) return false;
printf("calling with param %s\n", args[0].value.stringValue.UTF8Characters);

	printf("pepper iteration\n");
	if (!pglMakeCurrent(pgl_context_) && pglGetError() == PGL_CONTEXT_LOST) {
		printf("******* Lost context! :O\n");
		Destroy3D();
		Initialize3D();
		pglMakeCurrent(pgl_context_);
	}
	os->start_scene(String::utf8(args[0].value.stringValue.UTF8Characters));
	pglSwapBuffers();
	pglMakeCurrent(NULL);

printf("returning true\n");
	  return true;
  };

  NPN_MemFree(name);

  return rval;
}


static NPClass GodotClass = {
  NP_CLASS_STRUCT_VERSION,
  Allocate,
  Deallocate,
  NULL,  // Invalidate is not implemented
  HasMethod,
  Invoke,
  InvokeDefault,
  NULL,  // HasProperty is not implemented
  NULL,  // GetProperty is not implemented
  NULL,  // SetProperty is not implemented
};

static NPObject* npobject = NULL;

NPError NPP_GetValue(NPP instance, NPPVariable variable, void* value) {
  NPError err = NPERR_NO_ERROR;

  switch (variable) {
    case NPPVpluginNameString:
      *(reinterpret_cast<const char**>(value)) = "Pepper Test PlugIn";
      break;
    case NPPVpluginDescriptionString:
      *(reinterpret_cast<const char**>(value)) =
          "Simple Pepper plug-in for manual testing.";
      break;
    case NPPVpluginNeedsXEmbed:
	  *(reinterpret_cast<NPBool*>(value)) = 1;
      break;
	case NPPVpluginScriptableNPObject: {
	  if (npobject == NULL) {
		  npobject = NPN_CreateObject(instance, &GodotClass);
	  } else {
		  NPN_RetainObject(npobject);
	  };
	  void** v = reinterpret_cast<void**>(value);
	  *v = npobject;
	} break;
    default:
      fprintf(stderr, "Unhandled variable to NPP_GetValue\n");
      err = NPERR_GENERIC_ERROR;
      break;
  }

  return err;
}

NPError NPP_SetValue(NPP instance, NPNVariable variable, void* value) {
  return NPERR_GENERIC_ERROR;
}

NPError NP_GetValue(void* instance, NPPVariable variable, void* value) {
  return NPP_GetValue(reinterpret_cast<NPP>(instance), variable, value);
}

char* NP_GetMIMEDescription() {
  return const_cast<char*>("pepper-application/x-pepper-test-plugin;");
}
