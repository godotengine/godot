//
// This file demonstrates how to initialize EGL in a Windows Store app, using ICoreWindow.
//

#include "app.h"

#include "main/main.h"

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
void App::SetWindow(CoreWindow^ window)
{
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

    // The CoreWindow has been created, so EGL can be initialized.
	ContextEGL* context = memnew(ContextEGL(window));
	os->set_gl_context(context);
    UpdateWindowSize(Size(window->Bounds.Width, window->Bounds.Height));
}

// Initializes scene resources
void App::Load(Platform::String^ entryPoint)
{
	char** args = {NULL};
	Main::setup("winrt", 0, args);
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
#endif
}

void App::UpdateWindowSize(Size size)
{
	/*
    DisplayInformation^ currentDisplayInformation = DisplayInformation::GetForCurrentView();
    Size pixelSize(ConvertDipsToPixels(size.Width, currentDisplayInformation->LogicalDpi), ConvertDipsToPixels(size.Height, currentDisplayInformation->LogicalDpi));
    
    mWindowWidth = static_cast<GLsizei>(pixelSize.Width);
    mWindowHeight = static_cast<GLsizei>(pixelSize.Height);
	*/
}
