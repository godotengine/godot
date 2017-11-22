/*************************************************************************/
/*  app.h                                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#pragma once

#include <string>

#include <wrl.h>

// ANGLE doesn't provide a specific lib for GLES3, so we keep using GLES2
#include "GLES2/gl2.h"
#include "os_uwp.h"

/** clang-format does not play nice with this C++/CX hybrid, needs investigation. */
/* clang-format off */

namespace GodotUWP
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

		property Windows::Foundation::EventRegistrationToken MouseMovedToken {
				Windows::Foundation::EventRegistrationToken get() { return this->mouseMovedToken; }
				void set(Windows::Foundation::EventRegistrationToken p_token) { this->mouseMovedToken = p_token; }
		}

	private:
		void RecreateRenderer();

		// Application lifecycle event handlers.
		void OnActivated(Windows::ApplicationModel::Core::CoreApplicationView^ applicationView, Windows::ApplicationModel::Activation::IActivatedEventArgs^ args);

		// Window event handlers.
		void OnWindowSizeChanged(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::WindowSizeChangedEventArgs^ args);
		void OnVisibilityChanged(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::VisibilityChangedEventArgs^ args);
		void OnWindowClosed(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::CoreWindowEventArgs^ args);

		void pointer_event(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args, bool p_pressed, bool p_is_wheel = false);
		void OnPointerPressed(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args);
		void OnPointerReleased(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args);
		void OnPointerMoved(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args);
		void OnMouseMoved(Windows::Devices::Input::MouseDevice^ mouse_device, Windows::Devices::Input::MouseEventArgs^ args);
		void OnPointerWheelChanged(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::PointerEventArgs^ args);

		Windows::System::Threading::Core::SignalNotifier^ mouseChangedNotifier;
		Windows::Foundation::EventRegistrationToken mouseMovedToken;
		void OnMouseModeChanged(Windows::System::Threading::Core::SignalNotifier^ signalNotifier, bool timedOut);

		void key_event(Windows::UI::Core::CoreWindow^ sender, bool p_pressed, Windows::UI::Core::KeyEventArgs^ key_args = nullptr, Windows::UI::Core::CharacterReceivedEventArgs^ char_args = nullptr);
		void OnKeyDown(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::KeyEventArgs^ args);
		void OnKeyUp(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::KeyEventArgs^ args);
		void OnCharacterReceived(Windows::UI::Core::CoreWindow^ sender, Windows::UI::Core::CharacterReceivedEventArgs^ args);

		void UpdateWindowSize(Windows::Foundation::Size size);
		void InitializeEGL(Windows::UI::Core::CoreWindow^ window);
		void CleanupEGL();

		char** get_command_line(unsigned int* out_argc);

		bool mWindowClosed;
		bool mWindowVisible;
		GLsizei mWindowWidth;
		GLsizei mWindowHeight;

		EGLDisplay mEglDisplay;
		EGLContext mEglContext;
		EGLSurface mEglSurface;

		CoreWindow^ window;
		OSUWP* os;

		int last_touch_x[32]; // 20 fingers, index 31 reserved for the mouse
		int last_touch_y[32];
		int number_of_contacts;
		Windows::Foundation::Point last_mouse_pos;
	};
}

/* clang-format on */
