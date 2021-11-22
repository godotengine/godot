/*************************************************************************/
/*  app_uwp.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

//
// This file demonstrates how to initialize EGL in a Windows Store app, using ICoreWindow.
//

#include "app_uwp.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/keyboard.h"
#include "main/main.h"

#include "platform/windows/key_mapping_windows.h"

#include <collection.h>

using namespace Windows::ApplicationModel::Core;
using namespace Windows::ApplicationModel::Activation;
using namespace Windows::UI::Core;
using namespace Windows::UI::Input;
using namespace Windows::Devices::Input;
using namespace Windows::UI::Xaml::Input;
using namespace Windows::Foundation;
using namespace Windows::Graphics::Display;
using namespace Windows::System;
using namespace Windows::System::Threading::Core;
using namespace Microsoft::WRL;

using namespace GodotUWP;

// Helper to convert a length in device-independent pixels (DIPs) to a length in physical pixels.
inline float ConvertDipsToPixels(float dips, float dpi) {
	static const float dipsPerInch = 96.0f;
	return floor(dips * dpi / dipsPerInch + 0.5f); // Round to nearest integer.
}

// Implementation of the IFrameworkViewSource interface, necessary to run our app.
ref class GodotUWPViewSource sealed : Windows::ApplicationModel::Core::IFrameworkViewSource {
public:
	virtual Windows::ApplicationModel::Core::IFrameworkView ^ CreateView() {
		return ref new App();
	}
};

// The main function creates an IFrameworkViewSource for our app, and runs the app.
[Platform::MTAThread] int main(Platform::Array<Platform::String ^> ^) {
	auto godotApplicationSource = ref new GodotUWPViewSource();
	CoreApplication::Run(godotApplicationSource);
	return 0;
}

// The first method called when the IFrameworkView is being created.
void App::Initialize(CoreApplicationView ^ applicationView) {
	// Register event handlers for app lifecycle. This example includes Activated, so that we
	// can make the CoreWindow active and start rendering on the window.
	applicationView->Activated +=
			ref new TypedEventHandler<CoreApplicationView ^, IActivatedEventArgs ^>(this, &App::OnActivated);

	// Logic for other event handlers could go here.
	// Information about the Suspending and Resuming event handlers can be found here:
	// http://msdn.microsoft.com/en-us/library/windows/apps/xaml/hh994930.aspx

	os = new OS_UWP;
}

// Called when the CoreWindow object is created (or re-created).
void App::SetWindow(CoreWindow ^ p_window) {
	window = p_window;
	window->VisibilityChanged +=
			ref new TypedEventHandler<CoreWindow ^, VisibilityChangedEventArgs ^>(this, &App::OnVisibilityChanged);

	window->Closed +=
			ref new TypedEventHandler<CoreWindow ^, CoreWindowEventArgs ^>(this, &App::OnWindowClosed);

	window->SizeChanged +=
			ref new TypedEventHandler<CoreWindow ^, WindowSizeChangedEventArgs ^>(this, &App::OnWindowSizeChanged);

#if !(WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP)
	// Disable all pointer visual feedback for better performance when touching.
	// This is not supported on Windows Phone applications.
	auto pointerVisualizationSettings = PointerVisualizationSettings::GetForCurrentView();
	pointerVisualizationSettings->IsContactFeedbackEnabled = false;
	pointerVisualizationSettings->IsBarrelButtonFeedbackEnabled = false;
#endif

	window->PointerPressed +=
			ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &App::OnPointerPressed);
	window->PointerMoved +=
			ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &App::OnPointerMoved);
	window->PointerReleased +=
			ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &App::OnPointerReleased);
	window->PointerWheelChanged +=
			ref new TypedEventHandler<CoreWindow ^, PointerEventArgs ^>(this, &App::OnPointerWheelChanged);

	mouseChangedNotifier = SignalNotifier::AttachToEvent(L"os_mouse_mode_changed", ref new SignalHandler(this, &App::OnMouseModeChanged));

	mouseChangedNotifier->Enable();

	window->CharacterReceived +=
			ref new TypedEventHandler<CoreWindow ^, CharacterReceivedEventArgs ^>(this, &App::OnCharacterReceived);
	window->KeyDown +=
			ref new TypedEventHandler<CoreWindow ^, KeyEventArgs ^>(this, &App::OnKeyDown);
	window->KeyUp +=
			ref new TypedEventHandler<CoreWindow ^, KeyEventArgs ^>(this, &App::OnKeyUp);

	os->set_window(window);

	unsigned int argc;
	char **argv = get_command_line(&argc);

	Main::setup("uwp", argc, argv, false);

	UpdateWindowSize(Size(window->Bounds.Width, window->Bounds.Height));

	Main::setup2();
}

static MouseButton _get_button(Windows::UI::Input::PointerPoint ^ pt) {
	using namespace Windows::UI::Input;

#if WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP
	return MOUSE_BUTTON_LEFT;
#else
	switch (pt->Properties->PointerUpdateKind) {
		case PointerUpdateKind::LeftButtonPressed:
		case PointerUpdateKind::LeftButtonReleased:
			return MOUSE_BUTTON_LEFT;

		case PointerUpdateKind::RightButtonPressed:
		case PointerUpdateKind::RightButtonReleased:
			return MOUSE_BUTTON_RIGHT;

		case PointerUpdateKind::MiddleButtonPressed:
		case PointerUpdateKind::MiddleButtonReleased:
			return MOUSE_BUTTON_MIDDLE;

		case PointerUpdateKind::XButton1Pressed:
		case PointerUpdateKind::XButton1Released:
			return MOUSE_BUTTON_WHEEL_UP;

		case PointerUpdateKind::XButton2Pressed:
		case PointerUpdateKind::XButton2Released:
			return MOUSE_BUTTON_WHEEL_DOWN;

		default:
			break;
	}
#endif

	return MOUSE_BUTTON_NONE;
};

static bool _is_touch(Windows::UI::Input::PointerPoint ^ pointerPoint) {
#if WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP
	return true;
#else
	using namespace Windows::Devices::Input;
	switch (pointerPoint->PointerDevice->PointerDeviceType) {
		case PointerDeviceType::Touch:
		case PointerDeviceType::Pen:
			return true;
		default:
			return false;
	}
#endif
}

static Windows::Foundation::Point _get_pixel_position(CoreWindow ^ window, Windows::Foundation::Point rawPosition, OS *os) {
	Windows::Foundation::Point outputPosition;

// Compute coordinates normalized from 0..1.
// If the coordinates need to be sized to the SDL window,
// we'll do that after.
#if 1 || WINAPI_FAMILY != WINAPI_FAMILY_PHONE_APP
	outputPosition.X = rawPosition.X / window->Bounds.Width;
	outputPosition.Y = rawPosition.Y / window->Bounds.Height;
#else
	switch (DisplayProperties::CurrentOrientation) {
		case DisplayOrientations::Portrait:
			outputPosition.X = rawPosition.X / window->Bounds.Width;
			outputPosition.Y = rawPosition.Y / window->Bounds.Height;
			break;
		case DisplayOrientations::PortraitFlipped:
			outputPosition.X = 1.0f - (rawPosition.X / window->Bounds.Width);
			outputPosition.Y = 1.0f - (rawPosition.Y / window->Bounds.Height);
			break;
		case DisplayOrientations::Landscape:
			outputPosition.X = rawPosition.Y / window->Bounds.Height;
			outputPosition.Y = 1.0f - (rawPosition.X / window->Bounds.Width);
			break;
		case DisplayOrientations::LandscapeFlipped:
			outputPosition.X = 1.0f - (rawPosition.Y / window->Bounds.Height);
			outputPosition.Y = rawPosition.X / window->Bounds.Width;
			break;
		default:
			break;
	}
#endif

	OS::VideoMode vm = os->get_video_mode();
	outputPosition.X *= vm.width;
	outputPosition.Y *= vm.height;

	return outputPosition;
};

static int _get_finger(uint32_t p_touch_id) {
	return p_touch_id % 31; // for now
};

void App::pointer_event(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args, bool p_pressed, bool p_is_wheel) {
	Windows::UI::Input::PointerPoint ^ point = args->CurrentPoint;
	Windows::Foundation::Point pos = _get_pixel_position(window, point->Position, os);
	MouseButton but = _get_button(point);
	if (_is_touch(point)) {
		Ref<InputEventScreenTouch> screen_touch;
		screen_touch.instantiate();
		screen_touch->set_device(0);
		screen_touch->set_pressed(p_pressed);
		screen_touch->set_position(Vector2(pos.X, pos.Y));
		screen_touch->set_index(_get_finger(point->PointerId));

		last_touch_x[screen_touch->get_index()] = pos.X;
		last_touch_y[screen_touch->get_index()] = pos.Y;

		os->input_event(screen_touch);
	} else {
		Ref<InputEventMouseButton> mouse_button;
		mouse_button.instantiate();
		mouse_button->set_device(0);
		mouse_button->set_pressed(p_pressed);
		mouse_button->set_button_index(but);
		mouse_button->set_position(Vector2(pos.X, pos.Y));
		mouse_button->set_global_position(Vector2(pos.X, pos.Y));

		if (p_is_wheel) {
			if (point->Properties->MouseWheelDelta > 0) {
				mouse_button->set_button_index(point->Properties->IsHorizontalMouseWheel ? MOUSE_BUTTON_WHEEL_RIGHT : MOUSE_BUTTON_WHEEL_UP);
			} else if (point->Properties->MouseWheelDelta < 0) {
				mouse_button->set_button_index(point->Properties->IsHorizontalMouseWheel ? MOUSE_BUTTON_WHEEL_LEFT : MOUSE_BUTTON_WHEEL_DOWN);
			}
		}

		last_touch_x[31] = pos.X;
		last_touch_y[31] = pos.Y;

		os->input_event(mouse_button);

		if (p_is_wheel) {
			// Send release for mouse wheel
			mouse_button->set_pressed(false);
			os->input_event(mouse_button);
		}
	}
};

void App::OnPointerPressed(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args) {
	pointer_event(sender, args, true);
};

void App::OnPointerReleased(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args) {
	pointer_event(sender, args, false);
};

void App::OnPointerWheelChanged(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args) {
	pointer_event(sender, args, true, true);
}

void App::OnMouseModeChanged(Windows::System::Threading::Core::SignalNotifier ^ signalNotifier, bool timedOut) {
	OS::MouseMode mode = os->get_mouse_mode();
	SignalNotifier ^ notifier = mouseChangedNotifier;

	window->Dispatcher->RunAsync(
			CoreDispatcherPriority::High,
			ref new DispatchedHandler(
					[mode, notifier, this]() {
						if (mode == OS::MOUSE_MODE_CAPTURED) {
							this->MouseMovedToken = MouseDevice::GetForCurrentView()->MouseMoved +=
									ref new TypedEventHandler<MouseDevice ^, MouseEventArgs ^>(this, &App::OnMouseMoved);

						} else {
							MouseDevice::GetForCurrentView()->MouseMoved -= MouseMovedToken;
						}

						notifier->Enable();
					}));

	ResetEvent(os->mouse_mode_changed);
}

void App::OnPointerMoved(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args) {
	Windows::UI::Input::PointerPoint ^ point = args->CurrentPoint;
	Windows::Foundation::Point pos = _get_pixel_position(window, point->Position, os);

	if (_is_touch(point)) {
		Ref<InputEventScreenDrag> screen_drag;
		screen_drag.instantiate();
		screen_drag->set_device(0);
		screen_drag->set_position(Vector2(pos.X, pos.Y));
		screen_drag->set_index(_get_finger(point->PointerId));
		screen_drag->set_relative(Vector2(screen_drag->get_position().x - last_touch_x[screen_drag->get_index()], screen_drag->get_position().y - last_touch_y[screen_drag->get_index()]));

		os->input_event(screen_drag);
	} else {
		// In case the mouse grabbed, MouseMoved will handle this
		if (os->get_mouse_mode() == OS::MouseMode::MOUSE_MODE_CAPTURED) {
			return;
		}

		Ref<InputEventMouseMotion> mouse_motion;
		mouse_motion.instantiate();
		mouse_motion->set_device(0);
		mouse_motion->set_position(Vector2(pos.X, pos.Y));
		mouse_motion->set_global_position(Vector2(pos.X, pos.Y));
		mouse_motion->set_relative(Vector2(pos.X - last_touch_x[31], pos.Y - last_touch_y[31]));

		last_mouse_pos = pos;

		os->input_event(mouse_motion);
	}
}

void App::OnMouseMoved(MouseDevice ^ mouse_device, MouseEventArgs ^ args) {
	// In case the mouse isn't grabbed, PointerMoved will handle this
	if (os->get_mouse_mode() != OS::MouseMode::MOUSE_MODE_CAPTURED) {
		return;
	}

	Windows::Foundation::Point pos;
	pos.X = last_mouse_pos.X + args->MouseDelta.X;
	pos.Y = last_mouse_pos.Y + args->MouseDelta.Y;

	Ref<InputEventMouseMotion> mouse_motion;
	mouse_motion.instantiate();
	mouse_motion->set_device(0);
	mouse_motion->set_position(Vector2(pos.X, pos.Y));
	mouse_motion->set_global_position(Vector2(pos.X, pos.Y));
	mouse_motion->set_relative(Vector2(args->MouseDelta.X, args->MouseDelta.Y));

	last_mouse_pos = pos;

	os->input_event(mouse_motion);
}

void App::key_event(Windows::UI::Core::CoreWindow ^ sender, bool p_pressed, Windows::UI::Core::KeyEventArgs ^ key_args, Windows::UI::Core::CharacterReceivedEventArgs ^ char_args) {
	OS_UWP::KeyEvent ke;

	ke.control = sender->GetAsyncKeyState(VirtualKey::Control) == CoreVirtualKeyStates::Down;
	ke.alt = sender->GetAsyncKeyState(VirtualKey::Menu) == CoreVirtualKeyStates::Down;
	ke.shift = sender->GetAsyncKeyState(VirtualKey::Shift) == CoreVirtualKeyStates::Down;

	ke.pressed = p_pressed;

	if (key_args != nullptr) {
		ke.type = OS_UWP::KeyEvent::MessageType::KEY_EVENT_MESSAGE;
		ke.unicode = 0;
		ke.keycode = KeyMappingWindows::get_keysym((unsigned int)key_args->VirtualKey);
		ke.physical_keycode = KeyMappingWindows::get_scansym((unsigned int)key_args->KeyStatus.ScanCode, key_args->KeyStatus.IsExtendedKey);
		ke.echo = (!p_pressed && !key_args->KeyStatus.IsKeyReleased) || (p_pressed && key_args->KeyStatus.WasKeyDown);

	} else {
		ke.type = OS_UWP::KeyEvent::MessageType::CHAR_EVENT_MESSAGE;
		ke.unicode = char_args->KeyCode;
		ke.keycode = 0;
		ke.physical_keycode = 0;
		ke.echo = (!p_pressed && !char_args->KeyStatus.IsKeyReleased) || (p_pressed && char_args->KeyStatus.WasKeyDown);
	}

	os->queue_key_event(ke);
}

void App::OnKeyDown(CoreWindow ^ sender, KeyEventArgs ^ args) {
	key_event(sender, true, args);
}

void App::OnKeyUp(CoreWindow ^ sender, KeyEventArgs ^ args) {
	key_event(sender, false, args);
}

void App::OnCharacterReceived(CoreWindow ^ sender, CharacterReceivedEventArgs ^ args) {
	key_event(sender, true, nullptr, args);
}

// Initializes scene resources
void App::Load(Platform::String ^ entryPoint) {
}

// This method is called after the window becomes active.
void App::Run() {
	if (Main::start())
		os->run();
}

// Terminate events do not cause Uninitialize to be called. It will be called if your IFrameworkView
// class is torn down while the app is in the foreground.
void App::Uninitialize() {
	Main::cleanup();
	delete os;
}

// Application lifecycle event handler.
void App::OnActivated(CoreApplicationView ^ applicationView, IActivatedEventArgs ^ args) {
	// Run() won't start until the CoreWindow is activated.
	CoreWindow::GetForCurrentThread()->Activate();
}

// Window event handlers.
void App::OnVisibilityChanged(CoreWindow ^ sender, VisibilityChangedEventArgs ^ args) {
	mWindowVisible = args->Visible;
}

void App::OnWindowClosed(CoreWindow ^ sender, CoreWindowEventArgs ^ args) {
	mWindowClosed = true;
}

void App::OnWindowSizeChanged(CoreWindow ^ sender, WindowSizeChangedEventArgs ^ args) {
#if (WINAPI_FAMILY == WINAPI_FAMILY_PC_APP)
	// On Windows 8.1, apps are resized when they are snapped alongside other apps, or when the device is rotated.
	// The default framebuffer will be automatically resized when either of these occur.
	// In particular, on a 90 degree rotation, the default framebuffer's width and height will switch.
	UpdateWindowSize(args->Size);
#else if (WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP)
	// On Windows Phone 8.1, the window size changes when the device is rotated.
	// The default framebuffer will not be automatically resized when this occurs.
	// It is therefore up to the app to handle rotation-specific logic in its rendering code.
	//os->screen_size_changed();
	UpdateWindowSize(args->Size);
#endif
}

void App::UpdateWindowSize(Size size) {
	float dpi;
#if (WINAPI_FAMILY == WINAPI_FAMILY_PC_APP)
	DisplayInformation ^ currentDisplayInformation = DisplayInformation::GetForCurrentView();
	dpi = currentDisplayInformation->LogicalDpi;
#else if (WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP)
	dpi = DisplayProperties::LogicalDpi;
#endif
	Size pixelSize(ConvertDipsToPixels(size.Width, dpi), ConvertDipsToPixels(size.Height, dpi));

	mWindowWidth = static_cast<GLsizei>(pixelSize.Width);
	mWindowHeight = static_cast<GLsizei>(pixelSize.Height);

	OS::VideoMode vm;
	vm.width = mWindowWidth;
	vm.height = mWindowHeight;
	vm.fullscreen = true;
	vm.resizable = false;
	os->set_video_mode(vm);
}

char **App::get_command_line(unsigned int *out_argc) {
	static char *fail_cl[] = { "--path", "game", nullptr };
	*out_argc = 2;

	FILE *f = _wfopen(L"__cl__.cl", L"rb");

	if (f == nullptr) {
		wprintf(L"Couldn't open command line file.\n");
		return fail_cl;
	}

#define READ_LE_4(v) ((int)(##v[3] & 0xFF) << 24) | ((int)(##v[2] & 0xFF) << 16) | ((int)(##v[1] & 0xFF) << 8) | ((int)(##v[0] & 0xFF))
#define CMD_MAX_LEN 65535

	uint8_t len[4];
	int r = fread(len, sizeof(uint8_t), 4, f);

	Platform::Collections::Vector<Platform::String ^> cl;

	if (r < 4) {
		fclose(f);
		wprintf(L"Wrong cmdline length.\n");
		return (fail_cl);
	}

	int argc = READ_LE_4(len);

	for (int i = 0; i < argc; i++) {
		r = fread(len, sizeof(uint8_t), 4, f);

		if (r < 4) {
			fclose(f);
			wprintf(L"Wrong cmdline param length.\n");
			return (fail_cl);
		}

		int strlen = READ_LE_4(len);

		if (strlen > CMD_MAX_LEN) {
			fclose(f);
			wprintf(L"Wrong command length.\n");
			return (fail_cl);
		}

		char *arg = new char[strlen + 1];
		r = fread(arg, sizeof(char), strlen, f);
		arg[strlen] = '\0';

		if (r == strlen) {
			int warg_size = MultiByteToWideChar(CP_UTF8, 0, arg, -1, nullptr, 0);
			wchar_t *warg = new wchar_t[warg_size];

			MultiByteToWideChar(CP_UTF8, 0, arg, -1, warg, warg_size);

			cl.Append(ref new Platform::String(warg, warg_size));

		} else {
			delete[] arg;
			fclose(f);
			wprintf(L"Error reading command.\n");
			return (fail_cl);
		}
	}

#undef READ_LE_4
#undef CMD_MAX_LEN

	fclose(f);

	char **ret = new char *[cl.Size + 1];

	for (int i = 0; i < cl.Size; i++) {
		int arg_size = WideCharToMultiByte(CP_UTF8, 0, cl.GetAt(i)->Data(), -1, nullptr, 0, nullptr, nullptr);
		char *arg = new char[arg_size];

		WideCharToMultiByte(CP_UTF8, 0, cl.GetAt(i)->Data(), -1, arg, arg_size, nullptr, nullptr);

		ret[i] = arg;
	}
	ret[cl.Size] = nullptr;
	*out_argc = cl.Size;

	return ret;
}
