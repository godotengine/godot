#pragma once

#include <string>

#include <wrl.h>

#include "os_winrt.h"
#include "GLES2/gl2.h"

namespace $ext_safeprojectname$
{
    ref class App sealed : public Windows::ApplicationModel::Core::IFrameworkView
    {
    public:
        App();

        // IFrameworkView Methods.
        virtual void Initialize(Windows::ApplicationModel::Core::CoreApplicationView^ applicationView);
        virtual void SetWindow(Windows::UI::Core::CoreWindow^ window);
        virtual void Load(Platform::String^ entryPoint);
        virtual void Run();
        virtual void Uninitialize();

    private:
		void RecreateRenderer();

        // Application lifecycle event handlers.
        void OnActivated(Windows::ApplicationModel::Core::CoreApplicationView^ applicationView, Windows::ApplicationModel::Activation::IActivatedEventArgs^ args);

        // Window event handlers.
        void OnWindowSizeChanged(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::WindowSizeChangedEventArgs^ args);
        void OnVisibilityChanged(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::VisibilityChangedEventArgs^ args);
        void OnWindowClosed(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::CoreWindowEventArgs^ args);

		void pointer_event(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args, bool p_pressed);
		void OnPointerPressed(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args);
		void OnPointerReleased(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args);
		void OnPointerMoved(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args);


        void UpdateWindowSize(Windows::Foundation::Size size);
        void InitializeEGL(Windows::UI::Core::CoreWindow^ window);
        void CleanupEGL();

        bool mWindowClosed;
        bool mWindowVisible;
        GLsizei mWindowWidth;
        GLsizei mWindowHeight;
        
        EGLDisplay mEglDisplay;
        EGLContext mEglContext;
        EGLSurface mEglSurface;

		CoreWindow^ window;
		OSWinrt* os;

		int last_touch_x[32]; // 20 fingers, index 31 reserved for the mouse
		int last_touch_y[32];
	};

}
