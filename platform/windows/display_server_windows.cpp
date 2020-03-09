#include "display_server_windows.h"

DisplayServerWindows::DisplayServerWindows(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {

	drop_events = false;
	key_event_pos = 0;
	layered_window = false;
	hBitmap = NULL;

	alt_mem = false;
	gr_mem = false;
	shift_mem = false;
	control_mem = false;
	meta_mem = false;
	minimized = false;
	was_maximized = false;
	window_focused = true;
	console_visible = IsWindowVisible(GetConsoleWindow());

	pressrc = 0;
	old_invalid = true;
	mouse_mode = MOUSE_MODE_VISIBLE;

	main_loop = NULL;
	outside = true;
	window_has_focus = true;
	WNDCLASSEXW wc;

	if (is_hidpi_allowed()) {
		HMODULE Shcore = LoadLibraryW(L"Shcore.dll");

		if (Shcore != NULL) {
			typedef HRESULT(WINAPI * SetProcessDpiAwareness_t)(SHC_PROCESS_DPI_AWARENESS);

			SetProcessDpiAwareness_t SetProcessDpiAwareness = (SetProcessDpiAwareness_t)GetProcAddress(Shcore, "SetProcessDpiAwareness");

			if (SetProcessDpiAwareness) {
				SetProcessDpiAwareness(SHC_PROCESS_SYSTEM_DPI_AWARE);
			}
		}
	}

	video_mode = p_desired;

	//printf("**************** desired %s, mode %s\n", p_desired.fullscreen?"true":"false", video_mode.fullscreen?"true":"false");
	RECT WindowRect;

	WindowRect.left = 0;
	WindowRect.right = video_mode.width;
	WindowRect.top = 0;
	WindowRect.bottom = video_mode.height;

	memset(&wc, 0, sizeof(WNDCLASSEXW));
	wc.cbSize = sizeof(WNDCLASSEXW);
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS;
	wc.lpfnWndProc = (WNDPROC)::WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	//wc.hInstance = hInstance;
	wc.hInstance = godot_hinstance ? godot_hinstance : GetModuleHandle(NULL);
	wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
	wc.hCursor = NULL; //LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = NULL;
	wc.lpszMenuName = NULL;
	wc.lpszClassName = L"Engine";

	if (!RegisterClassExW(&wc)) {
		MessageBox(NULL, "Failed To Register The Window Class.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		return ERR_UNAVAILABLE;
	}

	use_raw_input = true;

	RAWINPUTDEVICE Rid[1];

	Rid[0].usUsagePage = 0x01;
	Rid[0].usUsage = 0x02;
	Rid[0].dwFlags = 0;
	Rid[0].hwndTarget = 0;

	if (RegisterRawInputDevices(Rid, 1, sizeof(Rid[0])) == FALSE) {
		//registration failed.
		use_raw_input = false;
	}

	pre_fs_valid = true;
	if (video_mode.fullscreen) {

		/* this returns DPI unaware size, commenting
		DEVMODE current;
		memset(&current, 0, sizeof(current));
		EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &current);

		WindowRect.right = current.dmPelsWidth;
		WindowRect.bottom = current.dmPelsHeight;

		*/

		EnumSizeData data = { 0, 0, Size2() };
		EnumDisplayMonitors(NULL, NULL, _MonitorEnumProcSize, (LPARAM)&data);

		WindowRect.right = data.size.width;
		WindowRect.bottom = data.size.height;

		/*  DEVMODE dmScreenSettings;
		memset(&dmScreenSettings,0,sizeof(dmScreenSettings));
		dmScreenSettings.dmSize=sizeof(dmScreenSettings);
		dmScreenSettings.dmPelsWidth	= video_mode.width;
		dmScreenSettings.dmPelsHeight	= video_mode.height;
		dmScreenSettings.dmBitsPerPel	= current.dmBitsPerPel;
		dmScreenSettings.dmFields=DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT;

		LONG err = ChangeDisplaySettings(&dmScreenSettings,CDS_FULLSCREEN);
		if (err!=DISP_CHANGE_SUCCESSFUL) {

			video_mode.fullscreen=false;
		}*/
		pre_fs_valid = false;
	}

	DWORD dwExStyle;
	DWORD dwStyle;

	if (video_mode.fullscreen || video_mode.borderless_window) {

		dwExStyle = WS_EX_APPWINDOW;
		dwStyle = WS_POPUP;

	} else {
		dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
		dwStyle = WS_OVERLAPPEDWINDOW;
		if (!video_mode.resizable) {
			dwStyle &= ~WS_THICKFRAME;
			dwStyle &= ~WS_MAXIMIZEBOX;
		}
	}

	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);

	char *windowid;
#ifdef MINGW_ENABLED
	windowid = getenv("GODOT_WINDOWID");
#else
	size_t len;
	_dupenv_s(&windowid, &len, "GODOT_WINDOWID");
#endif

	if (windowid) {

// strtoull on mingw
#ifdef MINGW_ENABLED
		hWnd = (HWND)strtoull(windowid, NULL, 0);
#else
		hWnd = (HWND)_strtoui64(windowid, NULL, 0);
#endif
		free(windowid);
		SetLastError(0);
		user_proc = (WNDPROC)GetWindowLongPtr(hWnd, GWLP_WNDPROC);
		SetWindowLongPtr(hWnd, GWLP_WNDPROC, (LONG_PTR)(WNDPROC)::WndProc);
		DWORD le = GetLastError();
		if (user_proc == 0 && le != 0) {

			printf("Error setting WNDPROC: %li\n", le);
		};
		GetWindowLongPtr(hWnd, GWLP_WNDPROC);

		RECT rect;
		if (!GetClientRect(hWnd, &rect)) {
			MessageBoxW(NULL, L"Window Creation Error.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
			return ERR_UNAVAILABLE;
		};
		video_mode.width = rect.right;
		video_mode.height = rect.bottom;
		video_mode.fullscreen = false;
	} else {

		hWnd = CreateWindowExW(
				dwExStyle,
				L"Engine", L"",
				dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
				(GetSystemMetrics(SM_CXSCREEN) - WindowRect.right) / 2,
				(GetSystemMetrics(SM_CYSCREEN) - WindowRect.bottom) / 2,
				WindowRect.right - WindowRect.left,
				WindowRect.bottom - WindowRect.top,
				NULL, NULL, hInstance, NULL);
		if (!hWnd) {
			MessageBoxW(NULL, L"Window Creation Error.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
			return ERR_UNAVAILABLE;
		}
	};

	if (video_mode.always_on_top) {
		SetWindowPos(hWnd, video_mode.always_on_top ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	}

	//!!!!!!!!!!!!!!!!!!!!!!!!!!
	//TODO - do Vulkan and GLES2 support checks, driver selection and fallback
	video_driver_index = p_video_driver;
	print_verbose("Driver: " + String(get_video_driver_name(video_driver_index)) + " [" + itos(video_driver_index) + "]");
	//!!!!!!!!!!!!!!!!!!!!!!!!!!

	// Init context and rendering device
#if defined(OPENGL_ENABLED)
	if (video_driver_index == VIDEO_DRIVER_GLES2) {

		context_gles2 = memnew(ContextGL_Windows(hWnd, false));

		if (context_gles2->initialize() != OK) {
			memdelete(context_gles2);
			context_gles2 = NULL;
			ERR_FAIL_V(ERR_UNAVAILABLE);
		}

		context_gles2->set_use_vsync(video_mode.use_vsync);
		set_vsync_via_compositor(video_mode.vsync_via_compositor);

		if (RasterizerGLES2::is_viable() == OK) {
			RasterizerGLES2::register_config();
			RasterizerGLES2::make_current();
		} else {
			memdelete(context_gles2);
			context_gles2 = NULL;
			ERR_FAIL_V(ERR_UNAVAILABLE);
		}
	}
#endif
#if defined(VULKAN_ENABLED)
	if (video_driver_index == VIDEO_DRIVER_VULKAN) {

		context_vulkan = memnew(VulkanContextWindows);
		if (context_vulkan->initialize() != OK) {
			memdelete(context_vulkan);
			context_vulkan = NULL;
			ERR_FAIL_V(ERR_UNAVAILABLE);
		}
		if (context_vulkan->window_create(hWnd, hInstance, get_video_mode().width, get_video_mode().height) == -1) {
			memdelete(context_vulkan);
			context_vulkan = NULL;
			ERR_FAIL_V(ERR_UNAVAILABLE);
		}

		//temporary
		rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		rendering_device_vulkan->initialize(context_vulkan);

		RasterizerRD::make_current();
	}
#endif

	visual_server = memnew(VisualServerRaster);
	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server, get_render_thread_mode() == RENDER_SEPARATE_THREAD));
	}

	visual_server->init();

	input = memnew(InputDefault);
	joypad = memnew(JoypadWindows(input, &hWnd));

	AudioDriverManager::initialize(p_audio_driver);

	TRACKMOUSEEVENT tme;
	tme.cbSize = sizeof(TRACKMOUSEEVENT);
	tme.dwFlags = TME_LEAVE;
	tme.hwndTrack = hWnd;
	tme.dwHoverTime = HOVER_DEFAULT;
	TrackMouseEvent(&tme);

	RegisterTouchWindow(hWnd, 0);

	_ensure_user_data_dir();

	DragAcceptFiles(hWnd, true);

	move_timer_id = 1;

	if (!is_no_window_mode_enabled()) {
		ShowWindow(hWnd, SW_SHOW); // Show The Window
		SetForegroundWindow(hWnd); // Slightly Higher Priority
		SetFocus(hWnd); // Sets Keyboard Focus To
	}

	if (p_desired.layered) {
		set_window_per_pixel_transparency_enabled(true);
	}

	// IME
	im_himc = ImmGetContext(hWnd);
	ImmReleaseContext(hWnd, im_himc);

	im_position = Vector2();

	set_ime_active(false);

	if (!OS::get_singleton()->is_in_low_processor_usage_mode()) {
		//SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
		SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
		DWORD index = 0;
		HANDLE handle = AvSetMmThreadCharacteristics("Games", &index);
		if (handle)
			AvSetMmThreadPriority(handle, AVRT_PRIORITY_CRITICAL);

		// This is needed to make sure that background work does not starve the main thread.
		// This is only setting priority of this thread, not the whole process.
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	}

	update_real_mouse_position();
}
