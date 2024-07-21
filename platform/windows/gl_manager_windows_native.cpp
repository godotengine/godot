/**************************************************************************/
/*  gl_manager_windows_native.cpp                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "gl_manager_windows_native.h"

#if defined(WINDOWS_ENABLED) && defined(GLES3_ENABLED)

#include "core/config/project_settings.h"
#include "core/version.h"

#include "thirdparty/nvapi/nvapi_minimal.h"

#include <dwmapi.h>
#include <stdio.h>
#include <stdlib.h>

// Enable the DXGI present feature.
#define OPENGL_ON_DXGI_ENABLED

// Create and set the D3D11 render target.
// We don't actually need to set the render target 'cause we're rendering
// everything in OpenGL, right? As long as the buffer is written to it should
// just work?
//#define OPENGL_DXGI_SET_RENDER_TARGET

// Create a depth buffer in D3D11 and attach it to the render target.
// Requires OPENGL_DXGI_SET_RENDER_TARGET.
// For some reason Intel driver doesn't like it and gives GL_INVALID_OPERATION
// when trying to attach it with glFramebufferTexture2D.
// RasterizerGLES3 doesn't seem to actually need a depth buffer though. All
// it does is blit other FBOs onto the screen with GL_COLOR_BUFFER_BIT?
// Even if we do need a depth buffer, we could just create one in OpenGL.
//#define OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER

// Bind the color buffer as renderbuffer instead of texture 2D.
#define OPENGL_DXGI_USE_RENDERBUFFER

// Create and bind a depth buffer renderbuffer in OpenGL.
// Again, we probably don't need this at all.
//#define OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER

// Instead of rendering directly to the DXGI back buffer, render onto an
// intermediate buffer which is copied to the back buffer on present.
//#define OPENGL_DXGI_USE_INTERMEDIATE_BUFFER

// Use DXGI flip-discard. (WIP)
#define OPENGL_DXGI_USE_FLIP_MODEL

#ifdef OPENGL_ON_DXGI_ENABLED

#include "drivers/gles3/storage/texture_storage.h"
#include "platform_gl.h"

#include <d3d11.h>
#ifdef OPENGL_DXGI_USE_FLIP_MODEL
#include <d3d11_3.h>
#include <dxgi1_2.h>
#include <dxgi1_5.h>
#endif

#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

#endif // OPENGL_ON_DXGI_ENABLED

#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#define WGL_CONTEXT_FLAGS_ARB 0x2094
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x00000002
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001

#define _WGL_CONTEXT_DEBUG_BIT_ARB 0x0001

#if defined(__GNUC__)
// Workaround GCC warning from -Wcast-function-type.
#define GetProcAddress (void *)GetProcAddress
#endif

typedef HGLRC(APIENTRY *PFNWGLCREATECONTEXT)(HDC);
typedef BOOL(APIENTRY *PFNWGLDELETECONTEXT)(HGLRC);
typedef BOOL(APIENTRY *PFNWGLMAKECURRENT)(HDC, HGLRC);
typedef HGLRC(APIENTRY *PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int *);
typedef void *(APIENTRY *PFNWGLGETPROCADDRESS)(LPCSTR);

#ifdef OPENGL_ON_DXGI_ENABLED

typedef const char *(APIENTRY *PFNWGLGETEXTENSIONSSTRINGARB)(HDC);

#define WGL_ACCESS_READ_ONLY_NV 0x0000
#define WGL_ACCESS_READ_WRITE_NV 0x0001
#define WGL_ACCESS_WRITE_DISCARD_NV 0x0002

typedef BOOL(APIENTRY *PFNWGLDXSETRESOURCESHAREHANDLENVPROC)(void *dxObject, HANDLE shareHandle);
typedef HANDLE(APIENTRY *PFNWGLDXOPENDEVICENVPROC)(void *dxDevice);
typedef BOOL(APIENTRY *PFNWGLDXCLOSEDEVICENVPROC)(HANDLE hDevice);
typedef HANDLE(APIENTRY *PFNWGLDXREGISTEROBJECTNVPROC)(HANDLE hDevice, void *dxObject, GLuint name, GLenum type, GLenum access);
typedef BOOL(APIENTRY *PFNWGLDXUNREGISTEROBJECTNVPROC)(HANDLE hDevice, HANDLE hObject);
typedef BOOL(APIENTRY *PFNWGLDXOBJECTACCESSNVPROC)(HANDLE hObject, GLenum access);
typedef BOOL(APIENTRY *PFNWGLDXLOCKOBJECTSNVPROC)(HANDLE hDevice, GLint count, HANDLE *hObjects);
typedef BOOL(APIENTRY *PFNWGLDXUNLOCKOBJECTSNVPROC)(HANDLE hDevice, GLint count, HANDLE *hObjects);

PFNWGLDXSETRESOURCESHAREHANDLENVPROC gd_wglDXSetResourceShareHandleNV;
PFNWGLDXOPENDEVICENVPROC gd_wglDXOpenDeviceNV;
PFNWGLDXCLOSEDEVICENVPROC gd_wglDXCloseDeviceNV;
PFNWGLDXREGISTEROBJECTNVPROC gd_wglDXRegisterObjectNV;
PFNWGLDXUNREGISTEROBJECTNVPROC gd_wglDXUnregisterObjectNV;
PFNWGLDXOBJECTACCESSNVPROC gd_wglDXObjectAccessNV;
PFNWGLDXLOCKOBJECTSNVPROC gd_wglDXLockObjectsNV;
PFNWGLDXUNLOCKOBJECTSNVPROC gd_wglDXUnlockObjectsNV;

static bool load_nv_dx_interop(PFNWGLGETPROCADDRESS gd_wglGetProcAddress) {
	gd_wglDXSetResourceShareHandleNV = (PFNWGLDXSETRESOURCESHAREHANDLENVPROC)gd_wglGetProcAddress("wglDXSetResourceShareHandleNV");
	gd_wglDXOpenDeviceNV = (PFNWGLDXOPENDEVICENVPROC)gd_wglGetProcAddress("wglDXOpenDeviceNV");
	gd_wglDXCloseDeviceNV = (PFNWGLDXCLOSEDEVICENVPROC)gd_wglGetProcAddress("wglDXCloseDeviceNV");
	gd_wglDXRegisterObjectNV = (PFNWGLDXREGISTEROBJECTNVPROC)gd_wglGetProcAddress("wglDXRegisterObjectNV");
	gd_wglDXUnregisterObjectNV = (PFNWGLDXUNREGISTEROBJECTNVPROC)gd_wglGetProcAddress("wglDXUnregisterObjectNV");
	gd_wglDXObjectAccessNV = (PFNWGLDXOBJECTACCESSNVPROC)gd_wglGetProcAddress("wglDXObjectAccessNV");
	gd_wglDXLockObjectsNV = (PFNWGLDXLOCKOBJECTSNVPROC)gd_wglGetProcAddress("wglDXLockObjectsNV");
	gd_wglDXUnlockObjectsNV = (PFNWGLDXUNLOCKOBJECTSNVPROC)gd_wglGetProcAddress("wglDXUnlockObjectsNV");

	return gd_wglDXSetResourceShareHandleNV &&
			gd_wglDXOpenDeviceNV &&
			gd_wglDXCloseDeviceNV &&
			gd_wglDXRegisterObjectNV &&
			gd_wglDXUnregisterObjectNV &&
			gd_wglDXObjectAccessNV &&
			gd_wglDXLockObjectsNV &&
			gd_wglDXUnlockObjectsNV;
}

typedef decltype(&CreateDXGIFactory1) PFNCREATEDXGIFACTORY1;

static HMODULE module_dxgi = nullptr;
PFNCREATEDXGIFACTORY1 fptr_CreateDXGIFactory1 = nullptr;

typedef decltype(&D3D11CreateDeviceAndSwapChain) PFND3D11CREATEDEVICEANDSWAPCHAINPROC;
typedef decltype(&D3D11CreateDevice) PFND3D11CREATEDEVICEPROC;

static HMODULE module_d3d11 = nullptr;
PFND3D11CREATEDEVICEANDSWAPCHAINPROC fptr_D3D11CreateDeviceAndSwapChain = nullptr;
PFND3D11CREATEDEVICEPROC fptr_D3D11CreateDevice = nullptr;

#endif // OPENGL_ON_DXGI_ENABLED

static String format_error_message(DWORD id) {
	LPWSTR messageBuffer = nullptr;
	size_t size = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			nullptr, id, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR)&messageBuffer, 0, nullptr);

	String msg = "Error " + itos(id) + ": " + String::utf16((const char16_t *)messageBuffer, size);

	LocalFree(messageBuffer);

	return msg;
}

const int OGL_THREAD_CONTROL_ID = 0x20C1221E;
const int OGL_THREAD_CONTROL_DISABLE = 0x00000002;
const int OGL_THREAD_CONTROL_ENABLE = 0x00000001;
const int VRR_MODE_ID = 0x1194F158;
const int VRR_MODE_FULLSCREEN_ONLY = 0x1;

typedef int(__cdecl *NvAPI_Initialize_t)();
typedef int(__cdecl *NvAPI_Unload_t)();
typedef int(__cdecl *NvAPI_GetErrorMessage_t)(unsigned int, NvAPI_ShortString);
typedef int(__cdecl *NvAPI_DRS_CreateSession_t)(NvDRSSessionHandle *);
typedef int(__cdecl *NvAPI_DRS_DestroySession_t)(NvDRSSessionHandle);
typedef int(__cdecl *NvAPI_DRS_LoadSettings_t)(NvDRSSessionHandle);
typedef int(__cdecl *NvAPI_DRS_CreateProfile_t)(NvDRSSessionHandle, NVDRS_PROFILE *, NvDRSProfileHandle *);
typedef int(__cdecl *NvAPI_DRS_CreateApplication_t)(NvDRSSessionHandle, NvDRSProfileHandle, NVDRS_APPLICATION *);
typedef int(__cdecl *NvAPI_DRS_SaveSettings_t)(NvDRSSessionHandle);
typedef int(__cdecl *NvAPI_DRS_SetSetting_t)(NvDRSSessionHandle, NvDRSProfileHandle, NVDRS_SETTING *);
typedef int(__cdecl *NvAPI_DRS_FindProfileByName_t)(NvDRSSessionHandle, NvAPI_UnicodeString, NvDRSProfileHandle *);
typedef int(__cdecl *NvAPI_DRS_GetApplicationInfo_t)(NvDRSSessionHandle, NvDRSProfileHandle, NvAPI_UnicodeString, NVDRS_APPLICATION *);
typedef int(__cdecl *NvAPI_DRS_DeleteProfile_t)(NvDRSSessionHandle, NvDRSProfileHandle);
NvAPI_GetErrorMessage_t NvAPI_GetErrorMessage__;

static bool nvapi_err_check(const char *msg, int status) {
	if (status != 0) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			NvAPI_ShortString err_desc = { 0 };
			NvAPI_GetErrorMessage__(status, err_desc);
			print_verbose(vformat("%s: %s(code %d)", msg, err_desc, status));
		}
		return false;
	}
	return true;
}

// On windows we have to customize the NVIDIA application profile:
// * disable threaded optimization when using NVIDIA cards to avoid stuttering, see
//   https://stackoverflow.com/questions/36959508/nvidia-graphics-driver-causing-noticeable-frame-stuttering/37632948
//   https://github.com/Ryujinx/Ryujinx/blob/master/src/Ryujinx.Common/GraphicsDriver/NVThreadedOptimization.cs
// * disable G-SYNC in windowed mode, as it results in unstable editor refresh rates
void GLManagerNative_Windows::_nvapi_setup_profile() {
	HMODULE nvapi = nullptr;
#ifdef _WIN64
	nvapi = LoadLibraryA("nvapi64.dll");
#else
	nvapi = LoadLibraryA("nvapi.dll");
#endif

	if (nvapi == nullptr) {
		return;
	}

	void *(__cdecl * NvAPI_QueryInterface)(unsigned int interface_id) = nullptr;

	NvAPI_QueryInterface = (void *(__cdecl *)(unsigned int))(void *)GetProcAddress(nvapi, "nvapi_QueryInterface");

	if (NvAPI_QueryInterface == nullptr) {
		print_verbose("Error getting NVAPI NvAPI_QueryInterface");
		return;
	}

	// Setup NVAPI function pointers
	NvAPI_Initialize_t NvAPI_Initialize = (NvAPI_Initialize_t)NvAPI_QueryInterface(0x0150E828);
	NvAPI_GetErrorMessage__ = (NvAPI_GetErrorMessage_t)NvAPI_QueryInterface(0x6C2D048C);
	NvAPI_DRS_CreateSession_t NvAPI_DRS_CreateSession = (NvAPI_DRS_CreateSession_t)NvAPI_QueryInterface(0x0694D52E);
	NvAPI_DRS_DestroySession_t NvAPI_DRS_DestroySession = (NvAPI_DRS_DestroySession_t)NvAPI_QueryInterface(0xDAD9CFF8);
	NvAPI_Unload_t NvAPI_Unload = (NvAPI_Unload_t)NvAPI_QueryInterface(0xD22BDD7E);
	NvAPI_DRS_LoadSettings_t NvAPI_DRS_LoadSettings = (NvAPI_DRS_LoadSettings_t)NvAPI_QueryInterface(0x375DBD6B);
	NvAPI_DRS_CreateProfile_t NvAPI_DRS_CreateProfile = (NvAPI_DRS_CreateProfile_t)NvAPI_QueryInterface(0xCC176068);
	NvAPI_DRS_CreateApplication_t NvAPI_DRS_CreateApplication = (NvAPI_DRS_CreateApplication_t)NvAPI_QueryInterface(0x4347A9DE);
	NvAPI_DRS_SaveSettings_t NvAPI_DRS_SaveSettings = (NvAPI_DRS_SaveSettings_t)NvAPI_QueryInterface(0xFCBC7E14);
	NvAPI_DRS_SetSetting_t NvAPI_DRS_SetSetting = (NvAPI_DRS_SetSetting_t)NvAPI_QueryInterface(0x577DD202);
	NvAPI_DRS_FindProfileByName_t NvAPI_DRS_FindProfileByName = (NvAPI_DRS_FindProfileByName_t)NvAPI_QueryInterface(0x7E4A9A0B);
	NvAPI_DRS_GetApplicationInfo_t NvAPI_DRS_GetApplicationInfo = (NvAPI_DRS_GetApplicationInfo_t)NvAPI_QueryInterface(0xED1F8C69);
	NvAPI_DRS_DeleteProfile_t NvAPI_DRS_DeleteProfile = (NvAPI_DRS_DeleteProfile_t)NvAPI_QueryInterface(0x17093206);

	if (!nvapi_err_check("NVAPI: Init failed", NvAPI_Initialize())) {
		return;
	}

	print_verbose("NVAPI: Init OK!");

	NvDRSSessionHandle session_handle;

	if (NvAPI_DRS_CreateSession == nullptr) {
		return;
	}

	if (!nvapi_err_check("NVAPI: Error creating DRS session", NvAPI_DRS_CreateSession(&session_handle))) {
		NvAPI_Unload();
		return;
	}

	if (!nvapi_err_check("NVAPI: Error loading DRS settings", NvAPI_DRS_LoadSettings(session_handle))) {
		NvAPI_DRS_DestroySession(session_handle);
		NvAPI_Unload();
		return;
	}

	String app_executable_name = OS::get_singleton()->get_executable_path().get_file();
	String app_profile_name = GLOBAL_GET("application/config/name");
	// We need a name anyways, so let's use the engine name if an application name is not available
	// (this is used mostly by the Project Manager)
	if (app_profile_name.is_empty()) {
		app_profile_name = VERSION_NAME;
	}
	String old_profile_name = app_profile_name + " Nvidia Profile";
	Char16String app_profile_name_u16 = app_profile_name.utf16();
	Char16String old_profile_name_u16 = old_profile_name.utf16();
	Char16String app_executable_name_u16 = app_executable_name.utf16();

	// A previous error in app creation logic could result in invalid profiles,
	// clean these if they exist before proceeding.
	NvDRSProfileHandle old_profile_handle;

	int old_status = NvAPI_DRS_FindProfileByName(session_handle, (NvU16 *)(old_profile_name_u16.ptrw()), &old_profile_handle);

	if (old_status == 0) {
		print_verbose("NVAPI: Deleting old profile...");

		if (!nvapi_err_check("NVAPI: Error deleting old profile", NvAPI_DRS_DeleteProfile(session_handle, old_profile_handle))) {
			NvAPI_DRS_DestroySession(session_handle);
			NvAPI_Unload();
			return;
		}

		if (!nvapi_err_check("NVAPI: Error deleting old profile", NvAPI_DRS_SaveSettings(session_handle))) {
			NvAPI_DRS_DestroySession(session_handle);
			NvAPI_Unload();
			return;
		}
	}

	NvDRSProfileHandle profile_handle = nullptr;

	int profile_status = NvAPI_DRS_FindProfileByName(session_handle, (NvU16 *)(app_profile_name_u16.ptrw()), &profile_handle);

	if (profile_status != 0) {
		print_verbose("NVAPI: Profile not found, creating...");

		NVDRS_PROFILE profile_info;
		profile_info.version = NVDRS_PROFILE_VER;
		profile_info.isPredefined = 0;
		memcpy(profile_info.profileName, app_profile_name_u16.get_data(), sizeof(char16_t) * app_profile_name_u16.size());

		if (!nvapi_err_check("NVAPI: Error creating profile", NvAPI_DRS_CreateProfile(session_handle, &profile_info, &profile_handle))) {
			NvAPI_DRS_DestroySession(session_handle);
			NvAPI_Unload();
			return;
		}
	}

	NVDRS_APPLICATION_V4 app;
	app.version = NVDRS_APPLICATION_VER_V4;

	int app_status = NvAPI_DRS_GetApplicationInfo(session_handle, profile_handle, (NvU16 *)(app_executable_name_u16.ptrw()), &app);

	if (app_status != 0) {
		print_verbose("NVAPI: Application not found in profile, creating...");

		app.isPredefined = 0;
		memcpy(app.appName, app_executable_name_u16.get_data(), sizeof(char16_t) * app_executable_name_u16.size());
		memcpy(app.launcher, L"", sizeof(wchar_t));
		memcpy(app.fileInFolder, L"", sizeof(wchar_t));

		if (!nvapi_err_check("NVAPI: Error creating application", NvAPI_DRS_CreateApplication(session_handle, profile_handle, &app))) {
			NvAPI_DRS_DestroySession(session_handle);
			NvAPI_Unload();
			return;
		}
	}

	NVDRS_SETTING ogl_thread_control_setting = {};
	ogl_thread_control_setting.version = NVDRS_SETTING_VER;
	ogl_thread_control_setting.settingId = OGL_THREAD_CONTROL_ID;
	ogl_thread_control_setting.settingType = NVDRS_DWORD_TYPE;
	int thread_control_val = OGL_THREAD_CONTROL_DISABLE;
	if (!GLOBAL_GET("rendering/gl_compatibility/nvidia_disable_threaded_optimization")) {
		thread_control_val = OGL_THREAD_CONTROL_ENABLE;
	}
	ogl_thread_control_setting.u32CurrentValue = thread_control_val;

	if (!nvapi_err_check("NVAPI: Error calling NvAPI_DRS_SetSetting", NvAPI_DRS_SetSetting(session_handle, profile_handle, &ogl_thread_control_setting))) {
		NvAPI_DRS_DestroySession(session_handle);
		NvAPI_Unload();
		return;
	}

	NVDRS_SETTING vrr_mode_setting = {};
	vrr_mode_setting.version = NVDRS_SETTING_VER;
	vrr_mode_setting.settingId = VRR_MODE_ID;
	vrr_mode_setting.settingType = NVDRS_DWORD_TYPE;
	vrr_mode_setting.u32CurrentValue = VRR_MODE_FULLSCREEN_ONLY;

	if (!nvapi_err_check("NVAPI: Error calling NvAPI_DRS_SetSetting", NvAPI_DRS_SetSetting(session_handle, profile_handle, &vrr_mode_setting))) {
		NvAPI_DRS_DestroySession(session_handle);
		NvAPI_Unload();
		return;
	}

	if (!nvapi_err_check("NVAPI: Error saving settings", NvAPI_DRS_SaveSettings(session_handle))) {
		NvAPI_DRS_DestroySession(session_handle);
		NvAPI_Unload();
		return;
	}

	if (thread_control_val == OGL_THREAD_CONTROL_DISABLE) {
		print_verbose("NVAPI: Disabled OpenGL threaded optimization successfully");
	} else {
		print_verbose("NVAPI: Enabled OpenGL threaded optimization successfully");
	}
	print_verbose("NVAPI: Disabled G-SYNC for windowed mode successfully");

	NvAPI_DRS_DestroySession(session_handle);
}

#ifdef OPENGL_ON_DXGI_ENABLED
class GLManagerNative_Windows::DxgiSwapChain {
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> device_context;
	ComPtr<IDXGISwapChain> swap_chain;

#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	ComPtr<ID3D11Texture2D> depth_texture;
	ComPtr<ID3D11DepthStencilView> depth_stencil_view;
#endif

#ifdef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
	ComPtr<ID3D11Texture2D> intermediate_buffer;
#else
	ComPtr<ID3D11Texture2D> color_buffer;
#endif
#ifdef OPENGL_DXGI_SET_RENDER_TARGET
	ComPtr<ID3D11RenderTargetView> render_target_view;
#endif

#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	HANDLE frame_latency_waitable_obj{};
#endif

	HANDLE gldx_device{};
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	HANDLE gldx_color_buffer_rb{};
#else
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	HANDLE gldx_depth_texture{};
#endif
	HANDLE gldx_color_buffer_tex{};
#endif

	GLuint gl_fbo{};
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
#ifdef OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER
	GLuint gl_depth_buffer_rb{};
#endif
	GLuint gl_color_buffer_rb{};
#else
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	GLuint gl_depth_stencil_tex{};
#endif
	GLuint gl_color_buffer_tex{};
#endif

#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	bool supports_tearing = false;
#endif

	DxgiSwapChain() = default;
	~DxgiSwapChain() = default;

	template <typename T>
	friend void memdelete(T *);

public:
	static DxgiSwapChain *create(HWND p_hwnd, int p_width, int p_height, ID3D11Device *p_d3d11_device, ID3D11DeviceContext *p_d3d11_device_context);
	void destroy();

	void make_current();
	void release_current();

	void present(bool p_use_vsync);
	void resize_swap_chain(int p_width, int p_height);
	void set_use_vsync(bool p_use);

private:
#if defined(OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER) || defined(OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER)
	bool setup_depth_buffer(int p_width, int p_height);
	void release_depth_buffer();
#endif

#ifdef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
	bool setup_intermediate_buffer(int p_width, int p_height);
	void release_intermediate_buffer();
#else
	bool setup_render_target();
	void release_render_target();
#endif

	void lock_for_opengl();
	void unlock_from_opengl();
};

enum class DxgiStatus {
	UNINITIALIZED,
	DISABLED,
	LOADED_UNTESTED,
	LOADED_USABLE,
};

static DxgiStatus dxgi_status = DxgiStatus::UNINITIALIZED;

static bool load_dxgi_swap_chain_functions(PFNWGLGETPROCADDRESS gd_wglGetProcAddress, HDC hDC) {
	PFNWGLGETEXTENSIONSSTRINGARB gd_wglGetExtensionsStringARB = (PFNWGLGETEXTENSIONSSTRINGARB)gd_wglGetProcAddress("wglGetExtensionsStringARB");
	if (!gd_wglGetExtensionsStringARB) {
		return false;
	}

	const char *extensions = gd_wglGetExtensionsStringARB(hDC);
	if (!extensions || !strstr(extensions, "WGL_NV_DX_interop2")) {
		print_verbose("GLManagerNative_Windows: Extension WGL_NV_DX_interop2 not available.");
		return false;
	}

	if (!load_nv_dx_interop(gd_wglGetProcAddress)) {
		print_verbose("GLManagerNative_Windows: Failed to load WGL_NV_DX_interop functions.");
		return false;
	}

	print_verbose("GLManagerNative_Windows: Loaded WGL_NV_DX_interop functions.");

	module_dxgi = LoadLibraryW(L"dxgi.dll");
	if (!module_dxgi) {
		print_verbose("GLManagerNative_Windows: Failed to load DXGI.");
		return false;
	}

	fptr_CreateDXGIFactory1 = (PFNCREATEDXGIFACTORY1)GetProcAddress(module_dxgi, "CreateDXGIFactory1");
	if (!fptr_CreateDXGIFactory1) {
		print_verbose("GLManagerNative_Windows: Failed to load DXGI functions.")
				FreeLibrary(module_d3d11);
		return false;
	}

	module_d3d11 = LoadLibraryW(L"d3d11.dll");
	if (!module_d3d11) {
		print_verbose("GLManagerNative_Windows: Failed to load D3D11.");
		return false;
	}

	fptr_D3D11CreateDeviceAndSwapChain = (PFND3D11CREATEDEVICEANDSWAPCHAINPROC)GetProcAddress(module_d3d11, "D3D11CreateDeviceAndSwapChain");
	fptr_D3D11CreateDevice = (PFND3D11CREATEDEVICEPROC)GetProcAddress(module_d3d11, "D3D11CreateDevice");
	if (!fptr_D3D11CreateDeviceAndSwapChain || !fptr_D3D11CreateDevice) {
		print_verbose("GLManagerNative_Windows: Failed to load D3D11 functions.")
				FreeLibrary(module_d3d11);
		return false;
	}

	print_verbose("GLManagerNative_Windows: Loaded D3D11 functions.");
	return true;
}

static bool try_create_d3d11_device(ID3D11Device *&p_out_device, ID3D11DeviceContext *&p_out_device_context);
#endif // OPENGL_ON_DXGI_ENABLED

void GLManagerNative_Windows::set_prefer_dxgi_swap_chain(bool p_prefer) {
	prefer_dxgi = p_prefer;
}

bool GLManagerNative_Windows::is_using_dxgi_swap_chain() {
#ifdef OPENGL_ON_DXGI_ENABLED
#ifdef DEBUG_ENABLED
	if (dxgi_status == DxgiStatus::UNINITIALIZED) {
		WARN_PRINT("Do not attempt to check is_using_dxgi_swap_chain before first window!");
	}
#endif

	return dxgi_status == DxgiStatus::LOADED_USABLE;
#else
	return false;
#endif
}

int GLManagerNative_Windows::_find_or_create_display(GLWindow &win) {
	// find display NYI, only 1 supported so far
	if (_displays.size()) {
		return 0;
	}

	// create
	GLDisplay d_temp = {};
	_displays.push_back(d_temp);
	int new_display_id = _displays.size() - 1;

	// create context
	GLDisplay &d = _displays[new_display_id];
	Error err = _create_context(win, d);

	if (err != OK) {
		// not good
		// delete the _display?
		_displays.remove_at(new_display_id);
		return -1;
	}

	return new_display_id;
}

static Error _configure_pixel_format(HDC hDC) {
	static PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR), // Size Of This Pixel Format Descriptor
		1,
		PFD_DRAW_TO_WINDOW | // Format Must Support Window
				PFD_SUPPORT_OPENGL | // Format Must Support OpenGL
				PFD_DOUBLEBUFFER,
		(BYTE)PFD_TYPE_RGBA,
		(BYTE)(OS::get_singleton()->is_layered_allowed() ? 32 : 24),
		(BYTE)0, (BYTE)0, (BYTE)0, (BYTE)0, (BYTE)0, (BYTE)0, // Color Bits Ignored
		(BYTE)(OS::get_singleton()->is_layered_allowed() ? 8 : 0), // Alpha Buffer
		(BYTE)0, // Shift Bit Ignored
		(BYTE)0, // No Accumulation Buffer
		(BYTE)0, (BYTE)0, (BYTE)0, (BYTE)0, // Accumulation Bits Ignored
		(BYTE)24, // 24Bit Z-Buffer (Depth Buffer)
		(BYTE)0, // No Stencil Buffer
		(BYTE)0, // No Auxiliary Buffer
		(BYTE)PFD_MAIN_PLANE, // Main Drawing Layer
		(BYTE)0, // Reserved
		0, 0, 0 // Layer Masks Ignored
	};

	int pixel_format = ChoosePixelFormat(hDC, &pfd);
	if (!pixel_format) // Did Windows Find A Matching Pixel Format?
	{
		return ERR_CANT_CREATE; // Return FALSE
	}

	BOOL ret = SetPixelFormat(hDC, pixel_format, &pfd);
	if (!ret) // Are We Able To Set The Pixel Format?
	{
		return ERR_CANT_CREATE; // Return FALSE
	}

	return OK;
}

PFNWGLCREATECONTEXT gd_wglCreateContext;
PFNWGLMAKECURRENT gd_wglMakeCurrent;
PFNWGLDELETECONTEXT gd_wglDeleteContext;
PFNWGLGETPROCADDRESS gd_wglGetProcAddress;

Error GLManagerNative_Windows::_create_context(GLWindow &win, GLDisplay &gl_display) {
	Error err = _configure_pixel_format(win.hDC);
	if (err != OK) {
		return err;
	}

	HMODULE module = LoadLibraryW(L"opengl32.dll");
	if (!module) {
		return ERR_CANT_CREATE;
	}
	gd_wglCreateContext = (PFNWGLCREATECONTEXT)GetProcAddress(module, "wglCreateContext");
	gd_wglMakeCurrent = (PFNWGLMAKECURRENT)GetProcAddress(module, "wglMakeCurrent");
	gd_wglDeleteContext = (PFNWGLDELETECONTEXT)GetProcAddress(module, "wglDeleteContext");
	gd_wglGetProcAddress = (PFNWGLGETPROCADDRESS)GetProcAddress(module, "wglGetProcAddress");
	if (!gd_wglCreateContext || !gd_wglMakeCurrent || !gd_wglDeleteContext || !gd_wglGetProcAddress) {
		return ERR_CANT_CREATE;
	}

	gl_display.hRC = gd_wglCreateContext(win.hDC);
	if (!gl_display.hRC) // Are We Able To Get A Rendering Context?
	{
		return ERR_CANT_CREATE; // Return FALSE
	}

	if (!gd_wglMakeCurrent(win.hDC, gl_display.hRC)) {
		ERR_PRINT("Could not attach OpenGL context to newly created window: " + format_error_message(GetLastError()));
	}

	int attribs[] = {
		WGL_CONTEXT_MAJOR_VERSION_ARB, 3, //we want a 3.3 context
		WGL_CONTEXT_MINOR_VERSION_ARB, 3,
		//and it shall be forward compatible so that we can only use up to date functionality
		WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
		WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB /*| _WGL_CONTEXT_DEBUG_BIT_ARB*/,
		0
	}; //zero indicates the end of the array

	PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB = nullptr; //pointer to the method
	wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)gd_wglGetProcAddress("wglCreateContextAttribsARB");

	if (wglCreateContextAttribsARB == nullptr) //OpenGL 3.0 is not supported
	{
		gd_wglDeleteContext(gl_display.hRC);
		gl_display.hRC = nullptr;
		return ERR_CANT_CREATE;
	}

	HGLRC new_hRC = wglCreateContextAttribsARB(win.hDC, nullptr, attribs);
	if (!new_hRC) {
		gd_wglDeleteContext(gl_display.hRC);
		gl_display.hRC = nullptr;
		return ERR_CANT_CREATE;
	}

	if (!gd_wglMakeCurrent(win.hDC, nullptr)) {
		ERR_PRINT("Could not detach OpenGL context from newly created window: " + format_error_message(GetLastError()));
	}

	gd_wglDeleteContext(gl_display.hRC);
	gl_display.hRC = new_hRC;

	if (!gd_wglMakeCurrent(win.hDC, gl_display.hRC)) // Try to activate the rendering context.
	{
		ERR_PRINT("Could not attach OpenGL context to newly created window with replaced OpenGL context: " + format_error_message(GetLastError()));
		gd_wglDeleteContext(gl_display.hRC);
		gl_display.hRC = nullptr;
		return ERR_CANT_CREATE;
	}

	if (!wglSwapIntervalEXT) {
		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)gd_wglGetProcAddress("wglSwapIntervalEXT");
	}

#ifdef OPENGL_ON_DXGI_ENABLED
	if (dxgi_status == DxgiStatus::UNINITIALIZED) {
		if (prefer_dxgi) {
			if (load_dxgi_swap_chain_functions(gd_wglGetProcAddress, win.hDC)) {
				if (try_create_d3d11_device(d3d11_device, d3d11_device_context)) {
					dxgi_status = DxgiStatus::LOADED_UNTESTED;
				} else {
					dxgi_status = DxgiStatus::DISABLED;
				}
			} else {
				dxgi_status = DxgiStatus::DISABLED;
			}
		} else {
			dxgi_status = DxgiStatus::DISABLED;
		}
	}
#endif // OPENGL_ON_DXGI_ENABLED

	return OK;
}

Error GLManagerNative_Windows::window_create(DisplayServer::WindowID p_window_id, HWND p_hwnd, HINSTANCE p_hinstance, int p_width, int p_height) {
	HDC hDC = GetDC(p_hwnd);
	if (!hDC) {
		return ERR_CANT_CREATE;
	}

	// configure the HDC to use a compatible pixel format
	Error result = _configure_pixel_format(hDC);
	if (result != OK) {
		return result;
	}

	GLWindow win;
	win.hwnd = p_hwnd;
	win.hDC = hDC;

	win.gldisplay_id = _find_or_create_display(win);

	if (win.gldisplay_id == -1) {
		return FAILED;
	}

#ifdef OPENGL_ON_DXGI_ENABLED
	if (dxgi_status == DxgiStatus::LOADED_UNTESTED || dxgi_status == DxgiStatus::LOADED_USABLE) {
		win.dxgi = DxgiSwapChain::create(win.hwnd, p_width, p_height, d3d11_device, d3d11_device_context);
		if (win.dxgi) {
			if (dxgi_status == DxgiStatus::LOADED_UNTESTED) {
				dxgi_status = DxgiStatus::LOADED_USABLE;
				print_verbose("GLManagerNative_Windows: Presenting with D3D11 DXGI swap chain.")
			}
		} else {
			if (dxgi_status == DxgiStatus::LOADED_UNTESTED) {
				// If it failed during the first time creating a window,
				// just fall back to regular OpenGL SwapBuffers.
				dxgi_status = DxgiStatus::DISABLED;
				WARN_PRINT("GLManagerNative_Windows: Failed to initialize D3D11 DXGI swap chain, reverting to regular OpenGL.");
			} else {
				return ERR_CANT_CREATE;
			}
		}
	}
#endif // OPENGL_ON_DXGI_ENABLED

	// WARNING: p_window_id is an eternally growing integer since popup windows keep coming and going
	// and each of them has a higher id than the previous, so it must be used in a map not a vector
	_windows[p_window_id] = win;

	// make current
	window_make_current(p_window_id);

	return OK;
}

void GLManagerNative_Windows::window_destroy(DisplayServer::WindowID p_window_id) {
	GLWindow &win = get_window(p_window_id);

#ifdef OPENGL_ON_DXGI_ENABLED
	if (win.dxgi) {
		// We need to destroy some OpenGL resources.
		window_make_current(p_window_id);
		win.dxgi->destroy();
		win.dxgi = nullptr;
	}
#endif // OPENGL_ON_DXGI_ENABLED

	if (_current_window == &win) {
		_current_window = nullptr;
	}
	_windows.erase(p_window_id);
}

void GLManagerNative_Windows::window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) {
#ifdef OPENGL_ON_DXGI_ENABLED
	GLWindow &win = get_window(p_window_id);
	if (win.dxgi) {
		win.dxgi->resize_swap_chain(p_width, p_height);
	}
#endif // OPENGL_ON_DXGI_ENABLED
}

void GLManagerNative_Windows::release_current() {
	if (!_current_window) {
		return;
	}

#ifdef OPENGL_ON_DXGI_ENABLED
	if (_current_window->dxgi) {
		_current_window->dxgi->release_current();
	}
#endif // OPENGL_ON_DXGI_ENABLED

	if (!gd_wglMakeCurrent(_current_window->hDC, nullptr)) {
		ERR_PRINT("Could not detach OpenGL context from window marked current: " + format_error_message(GetLastError()));
	}

	_current_window = nullptr;
}

void GLManagerNative_Windows::window_make_current(DisplayServer::WindowID p_window_id) {
	if (p_window_id == -1) {
		return;
	}

	// crash if our data structures are out of sync, i.e. not found
	GLWindow &win = _windows[p_window_id];

	// noop
	if (&win == _current_window) {
		return;
	}

	const GLDisplay &disp = get_display(win.gldisplay_id);
	if (!gd_wglMakeCurrent(win.hDC, disp.hRC)) {
		ERR_PRINT("Could not switch OpenGL context to other window: " + format_error_message(GetLastError()));
	}

#ifdef OPENGL_ON_DXGI_ENABLED
	if (win.dxgi) {
		win.dxgi->make_current();
	}
#endif // OPENGL_ON_DXGI_ENABLED

	_current_window = &win;
}

void GLManagerNative_Windows::swap_buffers() {
#ifdef OPENGL_ON_DXGI_ENABLED
	GLWindow &win = *_current_window;
	if (win.dxgi) {
		win.dxgi->present(win.use_vsync);
	} else {
#else
	{
#endif // OPENGL_ON_DXGI_ENABLED
		SwapBuffers(_current_window->hDC);
	}
}

Error GLManagerNative_Windows::initialize() {
	_nvapi_setup_profile();
	return OK;
}

void GLManagerNative_Windows::set_use_vsync(DisplayServer::WindowID p_window_id, bool p_use) {
	GLWindow &win = get_window(p_window_id);

	if (&win != _current_window) {
		window_make_current(p_window_id);
	}

#ifdef OPENGL_ON_DXGI_ENABLED
	if (win.dxgi) {
		win.use_vsync = p_use;
		win.dxgi->set_use_vsync(p_use);
	} else {
#else
	{
#endif // OPENGL_ON_DXGI_ENABLED
		if (wglSwapIntervalEXT) {
			win.use_vsync = p_use;

			if (!wglSwapIntervalEXT(p_use ? 1 : 0)) {
				WARN_PRINT_ONCE("Could not set V-Sync mode, as changing V-Sync mode is not supported by the graphics driver.");
			}
		} else {
			WARN_PRINT_ONCE("Could not set V-Sync mode, as changing V-Sync mode is not supported by the graphics driver.");
		}
	}
}

bool GLManagerNative_Windows::is_using_vsync(DisplayServer::WindowID p_window_id) const {
	return get_window(p_window_id).use_vsync;
}

HDC GLManagerNative_Windows::get_hdc(DisplayServer::WindowID p_window_id) {
	return get_window(p_window_id).hDC;
}

HGLRC GLManagerNative_Windows::get_hglrc(DisplayServer::WindowID p_window_id) {
	const GLWindow &win = get_window(p_window_id);
	const GLDisplay &disp = get_display(win.gldisplay_id);
	return disp.hRC;
}

GLManagerNative_Windows::GLManagerNative_Windows() {
	direct_render = false;
	glx_minor = glx_major = 0;
	_current_window = nullptr;
}

GLManagerNative_Windows::~GLManagerNative_Windows() {
	release_current();

#ifdef OPENGL_ON_DXGI_ENABLED
	if (d3d11_device_context) {
		d3d11_device_context->Release();
	}
	if (d3d11_device) {
		d3d11_device->Release();
	}
#endif
}

#ifdef OPENGL_ON_DXGI_ENABLED

template <typename T = IDXGIFactory>
static ComPtr<T> get_dxgi_factory(ID3D11Device *device) {
	static_assert(std::is_convertible<T *, IDXGIFactory *>::value, "Template argument must be IDXGIFactory or a derived type.");

	ComPtr<IDXGIDevice> dxgi_device;
	HRESULT hr = device->QueryInterface(__uuidof(IDXGIDevice), &dxgi_device);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("Failed to get IDXGIDevice, HRESULT: 0x%08X", (unsigned)hr));
		return {};
	}
	ComPtr<IDXGIAdapter> dxgi_adapter;
	hr = dxgi_device->GetAdapter(&dxgi_adapter);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("Failed to get IDXGIAdapter, HRESULT: 0x%08X", (unsigned)hr));
		return {};
	}
	ComPtr<T> dxgi_factory;
	hr = dxgi_adapter->GetParent(__uuidof(T), &dxgi_factory);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("Failed to get IDXGIFactory, HRESULT: 0x%08X", (unsigned)hr));
		return {};
	}
	return dxgi_factory;
}

static bool try_create_d3d11_device(ID3D11Device *&p_out_device, ID3D11DeviceContext *&p_out_device_context) {
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> device_context;

	UINT flags = 0;
	if (OS::get_singleton()->is_stdout_verbose()) {
		flags |= D3D11_CREATE_DEVICE_DEBUG;
	}

	HRESULT hr;

	hr = fptr_D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &device_context);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("D3D11CreateDevice failed, HRESULT: 0x%08X", (unsigned)hr));
		return false;
	}

#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	ComPtr<IDXGIFactory2> dxgi_factory_2 = get_dxgi_factory<IDXGIFactory2>(device.Get());
	if (!dxgi_factory_2) {
		ERR_PRINT("Failed to get IDXGIFactory2.");
		return false;
	}
#endif

	p_out_device = device.Detach();
	p_out_device_context = device_context.Detach();
	return true;
}

GLManagerNative_Windows::DxgiSwapChain *GLManagerNative_Windows::DxgiSwapChain::create(HWND p_hwnd, int p_width, int p_height, ID3D11Device *p_d3d11_device, ID3D11DeviceContext *p_d3d11_device_context) {
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> device_context;
	ComPtr<IDXGISwapChain> swap_chain;

	UINT flags = 0;
	if (OS::get_singleton()->is_stdout_verbose()) {
		flags |= D3D11_CREATE_DEVICE_DEBUG;
	}

#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	HRESULT hr;

#if 0
	ComPtr<IDXGIFactory2> dxgi_factory_2;
	hr = fptr_CreateDXGIFactory1(__uuidof(IDXGIFactory2), &dxgi_factory_2);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("CreateDXGIFactory1 failed, HRESULT: 0x%08X", (unsigned)hr));
		return nullptr;
	}

	ComPtr<IDXGIAdapter1> dxgi_adapter_1;
	hr = dxgi_factory_2->EnumAdapters1(0, &dxgi_adapter_1);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("EnumAdapters1 failed, HRESULT: 0x%08X", (unsigned)hr));
		return nullptr;
	}
#endif

	device = p_d3d11_device;
	device_context = p_d3d11_device_context;

	ComPtr<IDXGIFactory2> dxgi_factory_2 = get_dxgi_factory<IDXGIFactory2>(device.Get());
	if (!dxgi_factory_2) {
		ERR_PRINT("Failed to get IDXGIFactory2.");
		return nullptr;
	}

	bool supports_tearing = false;
	{
		ComPtr<IDXGIFactory5> dxgi_factory_5;
		hr = dxgi_factory_2.As(&dxgi_factory_5);
		if (!SUCCEEDED(hr)) {
			ERR_PRINT(vformat("Failed to get IDXGIFactory5, HRESULT: 0x%08X", (unsigned)hr));
		} else {
			BOOL feature_allow_tearing = FALSE;
			hr = dxgi_factory_5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &feature_allow_tearing, sizeof(feature_allow_tearing));
			if (!SUCCEEDED(hr)) {
				ERR_PRINT(vformat("Failed to check DXGI_FEATURE_PRESENT_ALLOW_TEARING, HRESULT: 0x%08X", (unsigned)hr));
			} else {
				supports_tearing = feature_allow_tearing;
			}
		}
	}

	ComPtr<IDXGIFactory2> dxgi_factory_2;
	hr = dxgi_factory.As(&dxgi_factory_2);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("Failed to get IDXGIFactory2, HRESULT: 0x%08X", (unsigned)hr));
		return nullptr;
	}

	DXGI_SWAP_CHAIN_DESC1 swap_chain_desc_1 = {};
	swap_chain_desc_1.Width = p_width;
	swap_chain_desc_1.Height = p_height;
	swap_chain_desc_1.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swap_chain_desc_1.SampleDesc.Count = 1;
	swap_chain_desc_1.BufferCount = 3;
	swap_chain_desc_1.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swap_chain_desc_1.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swap_chain_desc_1.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
	if (supports_tearing) {
		swap_chain_desc_1.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
	}
	swap_chain_desc_1.Scaling = DXGI_SCALING_NONE;
	// TODO: ???
	swap_chain_desc_1.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

	ComPtr<IDXGISwapChain1> swap_chain_1;
	hr = dxgi_factory_2->CreateSwapChainForHwnd(device.Get(), p_hwnd, &swap_chain_desc_1, nullptr, nullptr, &swap_chain_1);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("CreateSwapChainForHwnd failed, HRESULT: 0x%08X", (unsigned)hr));
		return nullptr;
	}

	swap_chain = swap_chain_1;

	hr = dxgi_factory_2->MakeWindowAssociation(p_hwnd, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("MakeWindowAssociation failed, HRESULT: 0x%08X", (unsigned)hr));
	}

#else
	DXGI_SWAP_CHAIN_DESC swap_chain_desc = {};
	swap_chain_desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swap_chain_desc.SampleDesc.Count = 1;
#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	swap_chain_desc.BufferCount = 3;
#else
	swap_chain_desc.BufferCount = 2;
#endif
	swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swap_chain_desc.OutputWindow = p_hwnd;
	swap_chain_desc.Windowed = TRUE;
#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swap_chain_desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
#else
	swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
#endif

	// TODO: Change to use IDXGIFactory2::CreateSwapChainForHwnd so that we
	//       can specify DXGI_SCALING_NONE
	HRESULT hr = fptr_D3D11CreateDeviceAndSwapChain(nullptr, // Adapter
			D3D_DRIVER_TYPE_HARDWARE, // DriverType
			nullptr, // Software
			flags, // Flags
			nullptr, // pFeatureLevels
			0, // FeatureLevels
			D3D11_SDK_VERSION, // SDKVersion
			&swap_chain_desc, // pSwapChainDesc
			&swap_chain, // ppSwapChain
			&device, // ppDevice
			nullptr, // pFeatureLevel
			&device_context); // ppImmediateContext
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("D3D11CreateDeviceAndSwapChain failed, HRESULT: 0x%08X", (unsigned)hr));
		return nullptr;
	}

	{
		ComPtr<IDXGIFactory> dxgi_factory = get_dxgi_factory(device.Get());
		if (dxgi_factory) {
			hr = dxgi_factory->MakeWindowAssociation(p_hwnd, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
			if (!SUCCEEDED(hr)) {
				ERR_PRINT(vformat("MakeWindowAssociation failed, HRESULT: 0x%08X", (unsigned)hr));
			}
		}
	}
#endif

	HANDLE gldx_device = gd_wglDXOpenDeviceNV(device.Get());
	if (!gldx_device) {
		ERR_PRINT(vformat("Failed to connect D3D11 swap chain to WGL for interop. Error: %s", format_error_message(GetLastError())));
		gd_wglDXCloseDeviceNV(gldx_device);
		return nullptr;
	}

#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	HANDLE frame_latency_waitable_obj{};
	{
		ComPtr<IDXGISwapChain2> swap_chain_2;
		hr = swap_chain.As(&swap_chain_2);
		if (SUCCEEDED(hr)) {
			frame_latency_waitable_obj = swap_chain_2->GetFrameLatencyWaitableObject();
			DWORD wait = WaitForSingleObject(frame_latency_waitable_obj, 1000);
			if (wait != WAIT_OBJECT_0) {
				if (wait == WAIT_FAILED) {
					DWORD error = GetLastError();
					ERR_PRINT(vformat("Wait for frame latency waitable failed with error: 0x%08X", (unsigned)error));
				} else {
					ERR_PRINT(vformat("Wait for frame latency waitable failed, WaitForSingleObject returned 0x%08X", (unsigned)wait));
				}
			}
		} else {
			ERR_PRINT(vformat("Failed to get IDXGISwapChain2, HRESULT: 0x%08X", (unsigned)hr));
		}
	}
#endif

	// HACK: We need OpenGL functions _now_ but RasterizerGLES3 might not
	//       have been initialized yet.
	gladLoaderLoadGL();

	// Generate the FBO we use for the window.
	GLuint gl_fbo;
	glGenFramebuffers(1, &gl_fbo);

	// Generate texture names for render target and depth buffers.
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
#ifdef OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER
	GLuint renderbuffers[2] = {};
	glGenRenderbuffers(2, renderbuffers);
	GLuint gl_depth_buffer_rb = renderbuffers[0];
	GLuint gl_color_buffer_rb = renderbuffers[1];
#else
	GLuint gl_color_buffer_rb;
	glGenRenderbuffers(1, &gl_color_buffer_rb);
#endif
#else
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	GLuint textures[2] = {};
	glGenTextures(2, textures);
	GLuint gl_depth_stencil_tex = textures[0];
	GLuint gl_color_buffer_tex = textures[1];
#else
	GLuint gl_color_buffer_tex;
	glGenTextures(1, &gl_color_buffer_tex);
#endif
#endif

	GLManagerNative_Windows::DxgiSwapChain *dxgi = memnew(GLManagerNative_Windows::DxgiSwapChain);
	dxgi->device = std::move(device);
	dxgi->device_context = std::move(device_context);
	dxgi->swap_chain = std::move(swap_chain);
#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	dxgi->frame_latency_waitable_obj = frame_latency_waitable_obj;
#endif
	dxgi->gldx_device = gldx_device;
	dxgi->gl_fbo = gl_fbo;
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
#ifdef OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER
	dxgi->gl_depth_buffer_rb = gl_depth_buffer_rb;
#endif
	dxgi->gl_color_buffer_rb = gl_color_buffer_rb;
#else
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	dxgi->gl_depth_stencil_tex = gl_depth_stencil_tex;
#endif
	dxgi->gl_color_buffer_tex = gl_color_buffer_tex;
#endif
#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	dxgi->supports_tearing = supports_tearing;
#endif

	GLES3::TextureStorage::system_fbo = gl_fbo;

#if defined(OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER) || defined(OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER)
	if (!dxgi->setup_depth_buffer(p_width, p_height)) {
		GLES3::TextureStorage::system_fbo = 0;
		memdelete(dxgi);
		return nullptr;
	}
#endif

#ifdef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
	if (!dxgi->setup_intermediate_buffer(p_width, p_height)) {
#else
	if (!dxgi->setup_render_target()) {
#endif
		GLES3::TextureStorage::system_fbo = 0;
#if defined(OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER) || defined(OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER)
		dxgi->release_depth_buffer();
#endif
		memdelete(dxgi);
		return nullptr;
	}

#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	ComPtr<IDXGISwapChain2> swap_chain_2;
	hr = dxgi->swap_chain.As(&swap_chain_2);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("Failed to get IDXGISwapChain2, HRESULT: 0x%08X", (unsigned)hr));
	} else {
		// TODO: ???
		swap_chain_2->SetMaximumFrameLatency(1);
	}
#else
	ComPtr<IDXGIDevice1> device1;
	hr = dxgi->device.As(&device1);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("Failed to get IDXGIDevice1, HRESULT: 0x%08X", (unsigned)hr));
	} else {
		// TODO: ???
		device1->SetMaximumFrameLatency(1);
	}
#endif

	dxgi->lock_for_opengl();

	return dxgi;
}

void GLManagerNative_Windows::DxgiSwapChain::destroy() {
	if (GLES3::TextureStorage::system_fbo == gl_fbo) {
		GLES3::TextureStorage::system_fbo = 0;
	}

	unlock_from_opengl();
#ifdef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
	release_intermediate_buffer();
#else
	release_render_target();
#endif
#if defined(OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER) || defined(OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER)
	release_depth_buffer();
#endif

	// FIXME: Is this safe to do? Could our OpenGL context not be current?
	glDeleteFramebuffers(1, &gl_fbo);
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
#ifdef OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER
	GLuint renderbuffers[2] = { gl_depth_buffer_rb, gl_color_buffer_rb };
	glDeleteRenderbuffers(2, renderbuffers);
#else
	glDeleteRenderbuffers(1, &gl_color_buffer_rb);
#endif
#else
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	GLuint textures[2] = { gl_depth_stencil_tex, gl_color_buffer_tex };
	glDeleteTextures(2, textures);
#else
	glDeleteTextures(1, &gl_color_buffer_tex);
#endif
#endif

	gd_wglDXCloseDeviceNV(gldx_device);

#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	if (!CloseHandle(frame_latency_waitable_obj)) {
		ERR_PRINT(vformat("Failed to CloseHandle on frame latency waitable object. Error: %s", format_error_message(GetLastError())));
	}
#endif

	swap_chain.Reset();
	device_context.Reset();
	device.Reset();

	memdelete(this);
}

void GLManagerNative_Windows::DxgiSwapChain::make_current() {
	GLES3::TextureStorage::system_fbo = gl_fbo;
}

void GLManagerNative_Windows::DxgiSwapChain::release_current() {
	if (GLES3::TextureStorage::system_fbo != gl_fbo) {
#ifdef DEBUG_ENABLED
		WARN_PRINT("Trying to release D3D11 target but system_fbo has changed!");
#endif
		return;
	}
	GLES3::TextureStorage::system_fbo = 0;
}

#if defined(OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER) || defined(OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER)
bool GLManagerNative_Windows::DxgiSwapChain::setup_depth_buffer(int p_width, int p_height) {
#ifdef OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER
	glBindRenderbuffer(GL_RENDERBUFFER, gl_depth_buffer_rb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, p_width, p_height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	return true;
#else
	// Godot uses 24-bit depth buffer?
	ComPtr<ID3D11Texture2D> depth_texture_new;
	CD3D11_TEXTURE2D_DESC depth_buffer_desc(DXGI_FORMAT_R24G8_TYPELESS, p_width, p_height, 1, 1, D3D11_BIND_DEPTH_STENCIL);
	HRESULT hr = device->CreateTexture2D(
			&depth_buffer_desc,
			nullptr,
			&depth_texture_new);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("CreateTexture2D failed, HRESULT: 0x%08X", (unsigned)hr));
		return false;
	}

	D3D11_DEPTH_STENCIL_VIEW_DESC depth_stencil_view_desc{};
	depth_stencil_view_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depth_stencil_view_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	ComPtr<ID3D11DepthStencilView> depth_stencil_view_new;
	hr = device->CreateDepthStencilView(
			depth_texture_new.Get(),
			&depth_stencil_view_desc,
			&depth_stencil_view_new);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("CreateDepthStencilView failed, HRESULT: 0x%08X", (unsigned)hr));
		return false;
	}

	HANDLE gldx_depth_texture_new = gd_wglDXRegisterObjectNV(
			gldx_device,
			depth_texture_new.Get(),
			gl_depth_stencil_tex,
			GL_TEXTURE_2D,
			WGL_ACCESS_READ_WRITE_NV);
	if (!gldx_depth_texture_new) {
		ERR_PRINT(vformat("Failed to connect D3D11 depth texture to WGL for interop. Error: %s", format_error_message(GetLastError())));
		return false;
	}

	depth_texture = std::move(depth_texture_new);
	depth_stencil_view = std::move(depth_stencil_view_new);
	gldx_depth_texture = gldx_depth_texture_new;

	return true;
#endif
}

void GLManagerNative_Windows::DxgiSwapChain::release_depth_buffer() {
#ifdef OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER
	// no-op?
#else
	BOOL res = gd_wglDXUnregisterObjectNV(gldx_device, gldx_depth_texture);
	if (!res) {
		ERR_PRINT(vformat("Failed to unregister depth buffer for interop. Error: %s", format_error_message(GetLastError())));
	}

	gldx_depth_texture = nullptr;
	depth_stencil_view.Reset();
	depth_texture.Reset();
#endif
}
#endif // OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER

#ifdef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
bool GLManagerNative_Windows::DxgiSwapChain::setup_intermediate_buffer(int p_width, int p_height) {
	ComPtr<ID3D11Texture2D> intermediate_buffer_new;
	CD3D11_TEXTURE2D_DESC intermediate_buffer_desc(DXGI_FORMAT_R8G8B8A8_UNORM, p_width, p_height, 1, 1, D3D11_BIND_RENDER_TARGET);
	HRESULT hr = device->CreateTexture2D(
			&intermediate_buffer_desc,
			nullptr,
			&intermediate_buffer_new);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("CreateTexture2D failed, HRESULT: 0x%08X", (unsigned)hr));
		return false;
	}

	HANDLE gldx_color_buffer_rb_new = gd_wglDXRegisterObjectNV(
			gldx_device,
			intermediate_buffer_new.Get(),
			gl_color_buffer_rb,
			GL_RENDERBUFFER,
			WGL_ACCESS_READ_WRITE_NV);
	if (!gldx_color_buffer_rb_new) {
		ERR_PRINT(vformat("Failed to connect D3D11 intermediate texture to WGL for interop. Error: %s", format_error_message(GetLastError())));
		return false;
	}

	intermediate_buffer = std::move(intermediate_buffer_new);
	gldx_color_buffer_rb = gldx_color_buffer_rb_new;

	return true;
}

void GLManagerNative_Windows::DxgiSwapChain::release_intermediate_buffer() {
	BOOL res = gd_wglDXUnregisterObjectNV(gldx_device, gldx_color_buffer_rb);
	if (!res) {
		ERR_PRINT(vformat("Failed to unregister color buffer for interop. Error: %s", format_error_message(GetLastError())));
	}

	gldx_color_buffer_rb = nullptr;
	intermediate_buffer.Reset();
}

#else // OPENGL_DXGI_USE_INTERMEDIATE_BUFFER

bool GLManagerNative_Windows::DxgiSwapChain::setup_render_target() {
	// Get the current back buffer from the swap chain.
	ComPtr<ID3D11Texture2D> color_buffer_new;
	HRESULT hr = swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), &color_buffer_new);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("GetBuffer failed, HRESULT: 0x%08X", (unsigned)hr));
		return false;
	}

#ifdef OPENGL_DXGI_SET_RENDER_TARGET
	// TODO: Do we need to chedk OS::get_singleton()->is_layered_allowed() here?
	CD3D11_RENDER_TARGET_VIEW_DESC render_target_view_desc(D3D11_RTV_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM);
	ComPtr<ID3D11RenderTargetView> render_target_view_new;
	hr = device->CreateRenderTargetView(
			color_buffer_new.Get(),
			&render_target_view_desc,
			&render_target_view_new);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("CreateRenderTargetView failed, HRESULT: 0x%08X", (unsigned)hr));
		return false;
	}
#endif

	// Register for interop.
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	HANDLE gldx_color_buffer_rb_new = gd_wglDXRegisterObjectNV(gldx_device,
			color_buffer_new.Get(),
			gl_color_buffer_rb,
			GL_RENDERBUFFER,
			WGL_ACCESS_READ_WRITE_NV);
#else
	HANDLE gldx_color_buffer_tex_new = gd_wglDXRegisterObjectNV(gldx_device,
			color_buffer_new.Get(),
			gl_color_buffer_tex,
			GL_TEXTURE_2D,
			WGL_ACCESS_READ_WRITE_NV);
#endif
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	if (!gldx_color_buffer_rb_new) {
#else
	if (!gldx_color_buffer_tex_new) {
#endif
		ERR_PRINT(vformat("Failed to connect D3D11 color buffer to WGL for interop. Error: %s", format_error_message(GetLastError())));
		return false;
	}

#ifdef OPENGL_DXGI_SET_RENDER_TARGET
	// Attach back buffer and depth buffer to the render target.
	device_context->OMSetRenderTargets(1,
			render_target_view_new.GetAddressOf(),
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
			depth_stencil_view.Get()
#else
			nullptr
#endif
	);

	// TODO: Is it okay if we don't clear at all?
	// float clear_color[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	// device_context->ClearRenderTargetView(render_target_view_new.Get(), clear_color);
#endif

	color_buffer = std::move(color_buffer_new);
#ifdef OPENGL_DXGI_SET_RENDER_TARGET
	render_target_view = std::move(render_target_view_new);
#endif
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	gldx_color_buffer_rb = gldx_color_buffer_rb_new;
#else
	gldx_color_buffer_tex = gldx_color_buffer_tex_new;
#endif

	return true;
}

void GLManagerNative_Windows::DxgiSwapChain::release_render_target() {
	// Release the back buffer.
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	BOOL res = gd_wglDXUnregisterObjectNV(gldx_device, gldx_color_buffer_rb);
#else
	BOOL res = gd_wglDXUnregisterObjectNV(gldx_device, gldx_color_buffer_tex);
#endif
	if (!res) {
		ERR_PRINT(vformat("Failed to unregister color buffer for interop. Error: %s", format_error_message(GetLastError())));
	}

#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	gldx_color_buffer_rb = nullptr;
#else
	gldx_color_buffer_tex = nullptr;
#endif
#ifdef OPENGL_DXGI_SET_RENDER_TARGET
	render_target_view.Reset();
#endif
	color_buffer.Reset();
}
#endif // OPENGL_DXGI_USE_INTERMEDIATE_BUFFER

void GLManagerNative_Windows::DxgiSwapChain::lock_for_opengl() {
	// Lock the buffers for OpenGL access.
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	BOOL res = gd_wglDXLockObjectsNV(gldx_device, 1, &gldx_color_buffer_rb);
#else
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	void *handles[] = { gldx_depth_texture, gldx_color_buffer_tex };
	BOOL res = gd_wglDXLockObjectsNV(gldx_device, 2, handles);
#else
	BOOL res = gd_wglDXLockObjectsNV(gldx_device, 1, &gldx_color_buffer_tex);
#endif
#endif
	if (!res) {
		ERR_PRINT(vformat("Failed to lock DX objects for interop. Error: %s", format_error_message(GetLastError())));
	}

	// Attach color and depth buffers to FBO
	glBindFramebuffer(GL_FRAMEBUFFER, gl_fbo);
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, gl_color_buffer_rb);
#ifdef OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gl_depth_buffer_rb);
#endif
#else
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_color_buffer_tex, 0);
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gl_depth_stencil_tex, 0);
#endif
#endif
	if (GLES3::TextureStorage::system_fbo != gl_fbo) {
		glBindFramebuffer(GL_FRAMEBUFFER, GLES3::TextureStorage::system_fbo);
	}
}

void GLManagerNative_Windows::DxgiSwapChain::unlock_from_opengl() {
	// Detach color and depth buffers from FBO
	glBindFramebuffer(GL_FRAMEBUFFER, gl_fbo);
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, 0);
#ifdef OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
#endif
#else
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
#endif
#endif
	if (GLES3::TextureStorage::system_fbo != gl_fbo) {
		glBindFramebuffer(GL_FRAMEBUFFER, GLES3::TextureStorage::system_fbo);
	}

	// Unlock from OpenGL access.
#ifdef OPENGL_DXGI_USE_RENDERBUFFER
	BOOL res = gd_wglDXUnlockObjectsNV(gldx_device, 1, &gldx_color_buffer_rb);
#else
#ifdef OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER
	void *handles[] = { gldx_depth_texture, gldx_color_buffer_tex };
	BOOL res = gd_wglDXUnlockObjectsNV(gldx_device, 2, handles);
#else
	BOOL res = gd_wglDXUnlockObjectsNV(gldx_device, 1, &gldx_color_buffer_tex);
#endif
#endif
	if (!res) {
		ERR_PRINT(vformat("Failed to unlock DX objects for interop. Error: %s", format_error_message(GetLastError())));
	}
}

void GLManagerNative_Windows::DxgiSwapChain::present(bool p_use_vsync) {
	unlock_from_opengl();
#ifndef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
	release_render_target();
#endif

	HRESULT hr;

#ifdef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
	// Now we copy our intermediate buffer to the back buffer.
	ComPtr<ID3D11Texture2D> color_buffer_new;
	hr = swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), &color_buffer_new);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("GetBuffer failed, HRESULT: 0x%08X", (unsigned)hr));
		return false;
	}

	device_context->CopyResource(color_buffer_new.Get(), intermediate_buffer.Get());
#endif

#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	if (p_use_vsync) {
		hr = swap_chain->Present(1, 0);
	} else {
		hr = swap_chain->Present(0, supports_tearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
	}
	DWORD wait = WaitForSingleObject(frame_latency_waitable_obj, 1000);
	if (wait != WAIT_OBJECT_0) {
		if (wait == WAIT_FAILED) {
			DWORD error = GetLastError();
			ERR_PRINT(vformat("Wait for frame latency waitable failed with error: 0x%08X", (unsigned)error));
		} else {
			ERR_PRINT(vformat("Wait for frame latency waitable failed, WaitForSingleObject returned 0x%08X", (unsigned)wait));
		}
	}
#else
	// TODO: vsync???
	HRESULT hr = swap_chain->Present(p_use_vsync ? 1 : 0, 0);
#endif
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("Present failed, HRESULT: 0x%08X", (unsigned)hr));
	}

#ifndef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
	setup_render_target();
#endif
	lock_for_opengl();
}

void GLManagerNative_Windows::DxgiSwapChain::resize_swap_chain(int p_width, int p_height) {
	unlock_from_opengl();
#ifdef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
	release_intermediate_buffer();
#else
	release_render_target();
#endif
#if defined(OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER) || defined(OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER)
	release_depth_buffer();
#endif

	UINT flags = 0;
#ifdef OPENGL_DXGI_USE_FLIP_MODEL
	flags |= DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
	if (supports_tearing) {
		flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
	}
#endif
	HRESULT hr = swap_chain->ResizeBuffers(0, p_width, p_height, DXGI_FORMAT_UNKNOWN, flags);
	if (!SUCCEEDED(hr)) {
		ERR_PRINT(vformat("ResizeBuffers failed, HRESULT: 0x%08X", (unsigned)hr));
	}

#if defined(OPENGL_DXGI_USE_D3D11_DEPTH_BUFFER) || defined(OPENGL_DXGI_ADD_DEPTH_RENDERBUFFER)
	if (setup_depth_buffer(p_width, p_height)) {
#else
	{
#endif
#ifdef OPENGL_DXGI_USE_INTERMEDIATE_BUFFER
		setup_intermediate_buffer(p_width, p_height);
#else
		setup_render_target();
#endif
	}
	lock_for_opengl();
}

void GLManagerNative_Windows::DxgiSwapChain::set_use_vsync(bool p_use) {
	// TODO: ???
}

#endif // OPENGL_ON_DXGI_ENABLED

#endif // WINDOWS_ENABLED && GLES3_ENABLED
