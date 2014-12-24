//
// This file demonstrates how to initialize EGL in a Windows Store app, using ICoreWindow.
//

#include "app.h"

#include "main/main.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"

using namespace Windows::ApplicationModel::Core;
using namespace Windows::ApplicationModel::Activation;
using namespace Windows::UI::Core;
using namespace Windows::UI::Input;
using namespace Windows::Foundation;
using namespace Windows::Graphics::Display;
using namespace Microsoft::WRL;
using namespace Platform;

using namespace $ext_safeprojectname$;

// Helper to convert a length in device-independent pixels (DIPs) to a length in physical pixels.
inline float ConvertDipsToPixels(float dips, float dpi)
{
    static const float dipsPerInch = 96.0f;
    return floor(dips * dpi / dipsPerInch + 0.5f); // Round to nearest integer.
}

// Implementation of the IFrameworkViewSource interface, necessary to run our app.
ref class HelloTriangleApplicationSource sealed : Windows::ApplicationModel::Core::IFrameworkViewSource
{
public:
    virtual Windows::ApplicationModel::Core::IFrameworkView^ CreateView()
    {
        return ref new App();
    }
};

// The main function creates an IFrameworkViewSource for our app, and runs the app.
[Platform::MTAThread]
int main(Platform::Array<Platform::String^>^)
{
    auto helloTriangleApplicationSource = ref new HelloTriangleApplicationSource();
    CoreApplication::Run(helloTriangleApplicationSource);
    return 0;
}

App::App() :
    mWindowClosed(false),
    mWindowVisible(true),
    mWindowWidth(0),
    mWindowHeight(0),
    mEglDisplay(EGL_NO_DISPLAY),
    mEglContext(EGL_NO_CONTEXT),
    mEglSurface(EGL_NO_SURFACE)
{
}

// The first method called when the IFrameworkView is being created.
void App::Initialize(CoreApplicationView^ applicationView)
{
    // Register event handlers for app lifecycle. This example includes Activated, so that we
    // can make the CoreWindow active and start rendering on the window.
    applicationView->Activated += 
        ref new TypedEventHandler<CoreApplicationView^, IActivatedEventArgs^>(this, &App::OnActivated);

    // Logic for other event handlers could go here.
    // Information about the Suspending and Resuming event handlers can be found here:
    // http://msdn.microsoft.com/en-us/library/windows/apps/xaml/hh994930.aspx

	os = new OSWinrt;
}

// Called when the CoreWindow object is created (or re-created).
void App::SetWindow(CoreWindow^ p_window)
{
	window = p_window;
    window->VisibilityChanged +=
        ref new TypedEventHandler<CoreWindow^, VisibilityChangedEventArgs^>(this, &App::OnVisibilityChanged);

    window->Closed += 
        ref new TypedEventHandler<CoreWindow^, CoreWindowEventArgs^>(this, &App::OnWindowClosed);

    window->SizeChanged += 
        ref new TypedEventHandler<CoreWindow^, WindowSizeChangedEventArgs^>(this, &App::OnWindowSizeChanged);

#if !(WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP)
    // Disable all pointer visual feedback for better performance when touching.
    // This is not supported on Windows Phone applications.
    auto pointerVisualizationSettings = PointerVisualizationSettings::GetForCurrentView();
    pointerVisualizationSettings->IsContactFeedbackEnabled = false;
    pointerVisualizationSettings->IsBarrelButtonFeedbackEnabled = false;
#endif


	window->PointerPressed +=
		ref new TypedEventHandler<CoreWindow^, PointerEventArgs^>(this, &App::OnPointerPressed);

	window->PointerMoved +=
		ref new TypedEventHandler<CoreWindow^, PointerEventArgs^>(this, &App::OnPointerMoved);

	window->PointerReleased +=
		ref new TypedEventHandler<CoreWindow^, PointerEventArgs^>(this, &App::OnPointerReleased);

	//window->PointerWheelChanged +=
	//	ref new TypedEventHandler<CoreWindow^, PointerEventArgs^>(this, &App::OnPointerWheelChanged);



	char* args[] = {"-path", "game", NULL};
	Main::setup("winrt", 2, args, false);

	// The CoreWindow has been created, so EGL can be initialized.
	ContextEGL* context = memnew(ContextEGL(window));
	os->set_gl_context(context);
	UpdateWindowSize(Size(window->Bounds.Width, window->Bounds.Height));

	Main::setup2();
}

static int _get_button(Windows::UI::Input::PointerPoint ^pt) {

	using namespace Windows::UI::Input;

#if WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP
	return BUTTON_LEFT;
#else
	switch (pt->Properties->PointerUpdateKind)
	{
		case PointerUpdateKind::LeftButtonPressed:
		case PointerUpdateKind::LeftButtonReleased:
			return BUTTON_LEFT;

		case PointerUpdateKind::RightButtonPressed:
		case PointerUpdateKind::RightButtonReleased:
			return BUTTON_RIGHT;

		case PointerUpdateKind::MiddleButtonPressed:
		case PointerUpdateKind::MiddleButtonReleased:
			return BUTTON_MIDDLE;

		case PointerUpdateKind::XButton1Pressed:
		case PointerUpdateKind::XButton1Released:
			return BUTTON_WHEEL_UP;

		case PointerUpdateKind::XButton2Pressed:
		case PointerUpdateKind::XButton2Released:
			return BUTTON_WHEEL_DOWN;

		default:
			break;
	}
#endif

	return 0;
};

static bool _is_touch(Windows::UI::Input::PointerPoint ^pointerPoint) {
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


static Windows::Foundation::Point _get_pixel_position(CoreWindow^ window, Windows::Foundation::Point rawPosition, OS* os) {

	Windows::Foundation::Point outputPosition;

	// Compute coordinates normalized from 0..1.
	// If the coordinates need to be sized to the SDL window,
	// we'll do that after.
	#if 1 || WINAPI_FAMILY != WINAPI_FAMILY_PHONE_APP
	outputPosition.X = rawPosition.X / window->Bounds.Width;
	outputPosition.Y = rawPosition.Y / window->Bounds.Height;
	#else
	switch (DisplayProperties::CurrentOrientation)
	{
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

void App::pointer_event(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args, bool p_pressed) {

	Windows::UI::Input::PointerPoint ^point = args->CurrentPoint;
	Windows::Foundation::Point pos = _get_pixel_position(window, point->Position, os);
	int but = _get_button(point);
	if (_is_touch(point)) {

		InputEvent event;
		event.type = InputEvent::SCREEN_TOUCH;
		event.device = 0;
		event.screen_touch.pressed = p_pressed;
		event.screen_touch.x = pos.X;
		event.screen_touch.y = pos.Y;
		event.screen_touch.index = _get_finger(point->PointerId);

		last_touch_x[event.screen_touch.index] = pos.X;
		last_touch_y[event.screen_touch.index] = pos.Y;

		os->input_event(event);
		if (event.screen_touch.index != 0)
			return;

	}; // fallthrought of sorts

	InputEvent event;
	event.type = InputEvent::MOUSE_BUTTON;
	event.device = 0;
	event.mouse_button.pressed = p_pressed;
	event.mouse_button.button_index = but;
	event.mouse_button.x = pos.X;
	event.mouse_button.y = pos.Y;
	event.mouse_button.global_x = pos.X;
	event.mouse_button.global_y = pos.Y;

	last_touch_x[31] = pos.X;
	last_touch_y[31] = pos.Y;

	os->input_event(event);
};


void App::OnPointerPressed(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args) {

	pointer_event(sender, args, true);
};


void App::OnPointerReleased(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args) {

	pointer_event(sender, args, false);
};

void App::OnPointerMoved(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args) {

	Windows::UI::Input::PointerPoint ^point = args->CurrentPoint;
	Windows::Foundation::Point pos = _get_pixel_position(window, point->Position, os);

	if (_is_touch(point)) {

		InputEvent event;
		event.type = InputEvent::SCREEN_DRAG;
		event.device = 0;
		event.screen_drag.x = pos.X;
		event.screen_drag.y = pos.Y;
		event.screen_drag.index = _get_finger(point->PointerId);
		event.screen_drag.relative_x = event.screen_drag.x - last_touch_x[event.screen_drag.index];
		event.screen_drag.relative_y = event.screen_drag.y - last_touch_y[event.screen_drag.index];

		os->input_event(event);
		if (event.screen_drag.index != 0)
			return;

	}; // fallthrought of sorts

	InputEvent event;
	event.type = InputEvent::MOUSE_MOTION;
	event.device = 0;
	event.mouse_motion.x = pos.X;
	event.mouse_motion.y = pos.Y;
	event.mouse_motion.global_x = pos.X;
	event.mouse_motion.global_y = pos.Y;
	event.mouse_motion.relative_x = pos.X - last_touch_x[31];
	event.mouse_motion.relative_y = pos.Y - last_touch_y[31];

	os->input_event(event);

};


// Initializes scene resources
void App::Load(Platform::String^ entryPoint)
{
	//char* args[] = {"-test", "render", NULL};
	//Main::setup("winrt", 2, args);
}

// This method is called after the window becomes active.
void App::Run()
{
	if (Main::start())
		os->run();
}

// Terminate events do not cause Uninitialize to be called. It will be called if your IFrameworkView
// class is torn down while the app is in the foreground.
void App::Uninitialize()
{
	Main::cleanup();
	delete os;
}

// Application lifecycle event handler.
void App::OnActivated(CoreApplicationView^ applicationView, IActivatedEventArgs^ args)
{
    // Run() won't start until the CoreWindow is activated.
    CoreWindow::GetForCurrentThread()->Activate();
}

// Window event handlers.
void App::OnVisibilityChanged(CoreWindow^ sender, VisibilityChangedEventArgs^ args)
{
    mWindowVisible = args->Visible;
}

void App::OnWindowClosed(CoreWindow^ sender, CoreWindowEventArgs^ args)
{
    mWindowClosed = true;
}

void App::OnWindowSizeChanged(CoreWindow^ sender, WindowSizeChangedEventArgs^ args)
{
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

void App::UpdateWindowSize(Size size)
{
	float dpi;
#if (WINAPI_FAMILY == WINAPI_FAMILY_PC_APP)
	DisplayInformation^ currentDisplayInformation = DisplayInformation::GetForCurrentView();
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
