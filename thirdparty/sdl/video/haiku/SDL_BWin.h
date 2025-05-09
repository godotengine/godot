/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#ifndef SDL_BWin_h_
#define SDL_BWin_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "SDL_internal.h"
#include "SDL_bframebuffer.h"

#ifdef __cplusplus
}
#endif

#include <stdio.h>
#include <AppKit.h>
#include <Cursor.h>
#include <InterfaceKit.h>
#ifdef SDL_VIDEO_OPENGL
#include <opengl/GLView.h>
#endif
#include "../../core/haiku/SDL_BApp.h"

enum WinCommands
{
    BWIN_MOVE_WINDOW,
    BWIN_RESIZE_WINDOW,
    BWIN_SHOW_WINDOW,
    BWIN_HIDE_WINDOW,
    BWIN_MAXIMIZE_WINDOW,
    BWIN_MINIMIZE_WINDOW,
    BWIN_RESTORE_WINDOW,
    BWIN_SET_TITLE,
    BWIN_SET_BORDERED,
    BWIN_SET_RESIZABLE,
    BWIN_FULLSCREEN,
    BWIN_UPDATE_FRAMEBUFFER,
    BWIN_MINIMUM_SIZE_WINDOW
};

// non-OpenGL framebuffer view
class SDL_BView : public BView
{
  public:
    SDL_BView(BRect frame, const char *name, uint32 resizingMode)
        : BView(frame, name, resizingMode, B_WILL_DRAW),
          fBitmap(NULL)
    {
    }

    void Draw(BRect dirty)
    {
        if (fBitmap != NULL)
            DrawBitmap(fBitmap, B_ORIGIN);
    }

    void SetBitmap(BBitmap *bitmap)
    {
        fBitmap = bitmap;
    }

  private:
    BBitmap *fBitmap;
};

class SDL_BWin : public BWindow
{
  public:
    // Constructor/Destructor
    SDL_BWin(BRect bounds, window_look look, uint32 flags)
        : BWindow(bounds, "Untitled", look, B_NORMAL_WINDOW_FEEL, flags)
    {
        _last_buttons = 0;

        _cur_view = NULL;
        _SDL_View = NULL;

#ifdef SDL_VIDEO_OPENGL
        _SDL_GLView = NULL;
        _gl_type = 0;
#endif
        _shown = false;
        _inhibit_resize = false;
        _mouse_focused = false;
        _prev_frame = NULL;
        _fullscreen = NULL;

        // Handle framebuffer stuff
        _buffer_locker = new BLocker();
        _bitmap = NULL;
    }

    virtual ~SDL_BWin()
    {
        Lock();

        if (_SDL_View != NULL && _SDL_View != _cur_view) {
            delete _SDL_View;
            _SDL_View = NULL;
        }

#ifdef SDL_VIDEO_OPENGL
        if (_SDL_GLView) {
            if (SDL_Looper->GetCurrentContext() == _SDL_GLView)
                SDL_Looper->SetCurrentContext(NULL);
            if (_SDL_GLView == _cur_view)
                RemoveChild(_SDL_GLView);
            _SDL_GLView = NULL;
            // _SDL_GLView deleted by HAIKU_GL_DestroyContext
        }

#endif
        Unlock();

        delete _prev_frame;

        // Clean up framebuffer stuff
        _buffer_locker->Lock();
        delete _buffer_locker;
    }

    void SetCurrentView(BView *view)
    {
        if (_cur_view != view) {
            if (_cur_view != NULL)
                RemoveChild(_cur_view);
            _cur_view = view;
            if (_cur_view != NULL)
                AddChild(_cur_view);
        }
    }

    void UpdateCurrentView()
    {
#ifdef SDL_VIDEO_OPENGL
        if (_SDL_GLView != NULL) {
            SetCurrentView(_SDL_GLView);
        } else
#endif
        if (_SDL_View != NULL) {
            SetCurrentView(_SDL_View);
        } else {
            SetCurrentView(NULL);
        }
    }

    SDL_BView *CreateView()
    {
        Lock();
        if (_SDL_View == NULL) {
            _SDL_View = new SDL_BView(Bounds(), "SDL View", B_FOLLOW_ALL_SIDES);
            UpdateCurrentView();
        }
        Unlock();
        return _SDL_View;
    }

    void RemoveView()
    {
        Lock();
        if (_SDL_View != NULL) {
            SDL_BView *oldView = _SDL_View;
            _SDL_View = NULL;
            UpdateCurrentView();
            delete oldView;
        }
        Unlock();
    }

    /* * * * * OpenGL functionality * * * * */
#ifdef SDL_VIDEO_OPENGL
    BGLView *CreateGLView(Uint32 gl_flags)
    {
        Lock();
        if (_SDL_GLView == NULL) {
            _SDL_GLView = new BGLView(Bounds(), "SDL GLView",
                                      B_FOLLOW_ALL_SIDES,
                                      (B_WILL_DRAW | B_FRAME_EVENTS),
                                      gl_flags);
            _gl_type = gl_flags;
            UpdateCurrentView();
        }
        Unlock();
        return _SDL_GLView;
    }

    void RemoveGLView()
    {
        Lock();
        if (_SDL_GLView != NULL) {
            if (SDL_Looper->GetCurrentContext() == _SDL_GLView)
                SDL_Looper->SetCurrentContext(NULL);
            _SDL_GLView = NULL;
            UpdateCurrentView();
            // _SDL_GLView deleted by HAIKU_GL_DestroyContext
        }
        Unlock();
    }

    void SwapBuffers(void)
    {
        _SDL_GLView->SwapBuffers();
    }
#endif

    /* * * * * Event sending * * * * */
    // Hook functions
    virtual void FrameMoved(BPoint origin)
    {
        // Post a message to the BApp so that it can handle the window event
        BMessage msg(BAPP_WINDOW_MOVED);
        msg.AddInt32("window-x", (int)origin.x);
        msg.AddInt32("window-y", (int)origin.y);
        _PostWindowEvent(msg);

        // Perform normal hook operations
        BWindow::FrameMoved(origin);
    }

    void FrameResized(float width, float height)
    {
        // Post a message to the BApp so that it can handle the window event
        BMessage msg(BAPP_WINDOW_RESIZED);

        msg.AddInt32("window-w", (int)width + 1);
        msg.AddInt32("window-h", (int)height + 1);
        _PostWindowEvent(msg);

        // Perform normal hook operations
        BWindow::FrameResized(width, height);
    }

    bool QuitRequested()
    {
        BMessage msg(BAPP_WINDOW_CLOSE_REQUESTED);
        _PostWindowEvent(msg);

        // We won't allow a quit unless asked by DestroyWindow()
        return false;
    }

    void WindowActivated(bool active)
    {
        BMessage msg(BAPP_KEYBOARD_FOCUS); // Mouse focus sold separately
        msg.AddBool("focusGained", active);
        _PostWindowEvent(msg);
    }

    void Zoom(BPoint origin,
              float width,
              float height)
    {
        BMessage msg(BAPP_MAXIMIZE); // Closest thing to maximization Haiku has
        _PostWindowEvent(msg);

        // Before the window zooms, record its size
        if (!_prev_frame)
            _prev_frame = new BRect(Frame());

        // Perform normal hook operations
        BWindow::Zoom(origin, width, height);
    }

    // Member functions
    void Show()
    {
        while (IsHidden()) {
            BWindow::Show();
        }
        _shown = true;

        BMessage msg(BAPP_SHOW);
        _PostWindowEvent(msg);
    }

    void Hide()
    {
        BWindow::Hide();
        _shown = false;

        BMessage msg(BAPP_HIDE);
        _PostWindowEvent(msg);
    }

    void Minimize(bool minimize)
    {
        BWindow::Minimize(minimize);
        int32 minState = (minimize ? BAPP_MINIMIZE : BAPP_RESTORE);

        BMessage msg(minState);
        _PostWindowEvent(msg);
    }

    void ScreenChanged(BRect screenFrame, color_space depth)
    {
        if (_fullscreen) {
            MoveTo(screenFrame.left, screenFrame.top);
            ResizeTo(screenFrame.Width(), screenFrame.Height());
        }
    }

    // BView message interruption
    void DispatchMessage(BMessage *msg, BHandler *target)
    {
        BPoint where;  // Used by mouse moved
        int32 buttons; // Used for mouse button events
        int32 key;     // Used for key events

        switch (msg->what) {
        case B_MOUSE_MOVED:
            int32 transit;
            if (msg->FindPoint("where", &where) == B_OK && msg->FindInt32("be:transit", &transit) == B_OK) {
                _MouseMotionEvent(where, transit);
            }
            break;

        case B_MOUSE_DOWN:
            if (msg->FindInt32("buttons", &buttons) == B_OK) {
                _MouseButtonEvent(buttons, true);
            }
            break;

        case B_MOUSE_UP:
            if (msg->FindInt32("buttons", &buttons) == B_OK) {
                _MouseButtonEvent(buttons, false);
            }
            break;

        case B_MOUSE_WHEEL_CHANGED:
            float x, y;
            if (msg->FindFloat("be:wheel_delta_x", &x) == B_OK && msg->FindFloat("be:wheel_delta_y", &y) == B_OK) {
                _MouseWheelEvent((int)x, (int)y);
            }
            break;

        case B_KEY_DOWN:
        {
            int32 i = 0;
            int8 byte;
            int8 bytes[4] = { 0, 0, 0, 0 };
            while (i < 4 && msg->FindInt8("byte", i, &byte) == B_OK) {
                bytes[i] = byte;
                i++;
            }
            if (msg->FindInt32("key", &key) == B_OK) {
                _KeyEvent((SDL_Scancode)key, &bytes[0], i, true);
            }
        } break;

        case B_UNMAPPED_KEY_DOWN: // modifier keys are unmapped
            if (msg->FindInt32("key", &key) == B_OK) {
                _KeyEvent((SDL_Scancode)key, NULL, 0, true);
            }
            break;

        case B_KEY_UP:
        case B_UNMAPPED_KEY_UP: // modifier keys are unmapped
            if (msg->FindInt32("key", &key) == B_OK) {
                _KeyEvent(key, NULL, 0, false);
            }
            break;

        default:
            /* move it after switch{} so it's always handled
               that way we keep Haiku features like:
               - CTRL+Q to close window (and other shortcuts)
               - PrintScreen to make screenshot into /boot/home
               - etc.. */
            // BWindow::DispatchMessage(msg, target);
            break;
        }

        BWindow::DispatchMessage(msg, target);
    }

    // Handle command messages
    void MessageReceived(BMessage *message)
    {
        switch (message->what) {
        // Handle commands from SDL
        case BWIN_SET_TITLE:
            _SetTitle(message);
            break;
        case BWIN_MOVE_WINDOW:
            _MoveTo(message);
            break;
        case BWIN_RESIZE_WINDOW:
            _ResizeTo(message);
            break;
        case BWIN_SET_BORDERED:
        {
            bool bEnabled;
            if (message->FindBool("window-border", &bEnabled) == B_OK)
                _SetBordered(bEnabled);
            break;
        }
        case BWIN_SET_RESIZABLE:
        {
            bool bEnabled;
            if (message->FindBool("window-resizable", &bEnabled) == B_OK)
                _SetResizable(bEnabled);
            break;
        }
        case BWIN_SHOW_WINDOW:
            Show();
            break;
        case BWIN_HIDE_WINDOW:
            Hide();
            break;
        case BWIN_MAXIMIZE_WINDOW:
            BWindow::Zoom();
            break;
        case BWIN_MINIMIZE_WINDOW:
            Minimize(true);
            break;
        case BWIN_RESTORE_WINDOW:
            _Restore();
            break;
        case BWIN_FULLSCREEN:
        {
            bool fullscreen;
            if (message->FindBool("fullscreen", &fullscreen) == B_OK)
                _SetFullScreen(fullscreen);
            break;
        }
        case BWIN_MINIMUM_SIZE_WINDOW:
            _SetMinimumSize(message);
            break;
        case BWIN_UPDATE_FRAMEBUFFER:
        {
            BMessage *pendingMessage;
            while ((pendingMessage = MessageQueue()->FindMessage(BWIN_UPDATE_FRAMEBUFFER, 0))) {
                MessageQueue()->RemoveMessage(pendingMessage);
                delete pendingMessage;
            }
            if (_bitmap != NULL) {
#ifdef SDL_VIDEO_OPENGL
                if (_SDL_GLView != NULL && _cur_view == _SDL_GLView) {
                    _SDL_GLView->CopyPixelsIn(_bitmap, B_ORIGIN);
                } else
#endif
                if (_SDL_View != NULL && _cur_view == _SDL_View) {
                    _SDL_View->Draw(Bounds());
                }
            }
            break;
        }
        default:
            // Perform normal message handling
            BWindow::MessageReceived(message);
            break;
        }
    }

    // Accessor methods
    bool IsShown() { return _shown; }
    int32 GetID() { return _id; }
    BBitmap *GetBitmap() { return _bitmap; }
    BView *GetCurView() { return _cur_view; }
    SDL_BView *GetView() { return _SDL_View; }
#ifdef SDL_VIDEO_OPENGL
    BGLView *GetGLView()
    {
        return _SDL_GLView;
    }
    Uint32 GetGLType() { return _gl_type; }
#endif

    // Setter methods
    void SetID(int32 id) { _id = id; }
    void LockBuffer() { _buffer_locker->Lock(); }
    void UnlockBuffer() { _buffer_locker->Unlock(); }
    void SetBitmap(BBitmap *bitmap)
    {
        _bitmap = bitmap;
        if (_SDL_View != NULL)
            _SDL_View->SetBitmap(bitmap);
    }

  private:
    // Event redirection
    void _MouseMotionEvent(BPoint &where, int32 transit)
    {
        if (transit == B_EXITED_VIEW) {
            // Change mouse focus
            if (_mouse_focused) {
                _MouseFocusEvent(false);
            }
        } else {
            // Change mouse focus
            if (!_mouse_focused) {
                _MouseFocusEvent(true);
            }
            BMessage msg(BAPP_MOUSE_MOVED);
            msg.AddInt32("x", (int)where.x);
            msg.AddInt32("y", (int)where.y);

            _PostWindowEvent(msg);
        }
    }

    void _MouseFocusEvent(bool focusGained)
    {
        _mouse_focused = focusGained;
        BMessage msg(BAPP_MOUSE_FOCUS);
        msg.AddBool("focusGained", focusGained);
        _PostWindowEvent(msg);

        /* FIXME: Why were these here?
         if false: be_app->SetCursor(B_HAND_CURSOR);
         if true:  SDL_SetCursor(NULL); */
    }

    void _MouseButtonEvent(int32 buttons, bool down)
    {
        int32 buttonStateChange = buttons ^ _last_buttons;

        if (buttonStateChange & B_PRIMARY_MOUSE_BUTTON) {
            _SendMouseButton(SDL_BUTTON_LEFT, down);
        }
        if (buttonStateChange & B_SECONDARY_MOUSE_BUTTON) {
            _SendMouseButton(SDL_BUTTON_RIGHT, down);
        }
        if (buttonStateChange & B_TERTIARY_MOUSE_BUTTON) {
            _SendMouseButton(SDL_BUTTON_MIDDLE, down);
        }

        _last_buttons = buttons;
    }

    void _SendMouseButton(int32 button, bool down)
    {
        BMessage msg(BAPP_MOUSE_BUTTON);
        msg.AddInt32("button-id", button);
        msg.AddBool("button-down", down);
        _PostWindowEvent(msg);
    }

    void _MouseWheelEvent(int32 x, int32 y)
    {
        // Create a message to pass along to the BeApp thread
        BMessage msg(BAPP_MOUSE_WHEEL);
        msg.AddInt32("xticks", x);
        msg.AddInt32("yticks", y);
        _PostWindowEvent(msg);
    }

    void _KeyEvent(int32 keyCode, const int8 *keyUtf8, const ssize_t &len, bool down)
    {
        // Create a message to pass along to the BeApp thread
        BMessage msg(BAPP_KEY);
        msg.AddInt32("key-scancode", keyCode);
        if (keyUtf8 != NULL) {
            msg.AddData("key-utf8", B_INT8_TYPE, (const void *)keyUtf8, len);
        }
        msg.AddBool("key-down", down);
        SDL_Looper->PostMessage(&msg);
    }

    void _RepaintEvent()
    {
        // Force a repaint: Call the SDL exposed event
        BMessage msg(BAPP_REPAINT);
        _PostWindowEvent(msg);
    }
    void _PostWindowEvent(BMessage &msg)
    {
        msg.AddInt32("window-id", _id);
        SDL_Looper->PostMessage(&msg);
    }

    // Command methods (functions called upon by SDL)
    void _SetTitle(BMessage *msg)
    {
        const char *title;
        if (
            msg->FindString("window-title", &title) != B_OK) {
            return;
        }
        SetTitle(title);
    }

    void _MoveTo(BMessage *msg)
    {
        int32 x, y;
        if (
            msg->FindInt32("window-x", &x) != B_OK ||
            msg->FindInt32("window-y", &y) != B_OK) {
            return;
        }
        if (_fullscreen)
            _non_fullscreen_frame.OffsetTo(x, y);
        else
            MoveTo(x, y);
    }

    void _ResizeTo(BMessage *msg)
    {
        int32 w, h;
        if (
            msg->FindInt32("window-w", &w) != B_OK ||
            msg->FindInt32("window-h", &h) != B_OK) {
            return;
        }
        if (_fullscreen) {
            _non_fullscreen_frame.right = _non_fullscreen_frame.left + w;
            _non_fullscreen_frame.bottom = _non_fullscreen_frame.top + h;
        } else
            ResizeTo(w, h);
    }

    void _SetBordered(bool bEnabled)
    {
        if (_fullscreen)
            _bordered = bEnabled;
        else
            SetLook(bEnabled ? B_TITLED_WINDOW_LOOK : B_NO_BORDER_WINDOW_LOOK);
    }

    void _SetResizable(bool bEnabled)
    {
        if (_fullscreen)
            _resizable = bEnabled;
        else {
            if (bEnabled) {
                SetFlags(Flags() & ~(B_NOT_RESIZABLE | B_NOT_ZOOMABLE));
            } else {
                SetFlags(Flags() | (B_NOT_RESIZABLE | B_NOT_ZOOMABLE));
            }
        }
    }

    void _SetMinimumSize(BMessage *msg)
    {
        float maxHeight;
        float maxWidth;
        float _;
        int32 minHeight;
        int32 minWidth;

        // This is a bit convoluted, we only want to set the minimum not the maximum
        // But there is no direct call to do that, so store the maximum size beforehand
        GetSizeLimits(&_, &maxWidth, &_, &maxHeight);
        if (msg->FindInt32("window-w", &minWidth) != B_OK)
            return;
        if (msg->FindInt32("window-h", &minHeight) != B_OK)
            return;
        SetSizeLimits((float)minWidth, maxWidth, (float)minHeight, maxHeight);
        UpdateSizeLimits();
    }

    void _Restore()
    {
        if (IsMinimized()) {
            Minimize(false);
        } else if (IsHidden()) {
            Show();
        } else if (_fullscreen) {

        } else if (_prev_frame != NULL) { // Zoomed
            MoveTo(_prev_frame->left, _prev_frame->top);
            ResizeTo(_prev_frame->Width(), _prev_frame->Height());
        }
    }

    void _SetFullScreen(bool fullscreen)
    {
        if (fullscreen != _fullscreen) {
            if (fullscreen) {
                BScreen screen(this);
                BRect screenFrame = screen.Frame();
                printf("screen frame: ");
                screenFrame.PrintToStream();
                printf("\n");
                _bordered = Look() != B_NO_BORDER_WINDOW_LOOK;
                _resizable = !(Flags() & B_NOT_RESIZABLE);
                _non_fullscreen_frame = Frame();
                _SetBordered(false);
                _SetResizable(false);
                MoveTo(screenFrame.left, screenFrame.top);
                ResizeTo(screenFrame.Width(), screenFrame.Height());
                _fullscreen = fullscreen;
            } else {
                _fullscreen = fullscreen;
                MoveTo(_non_fullscreen_frame.left, _non_fullscreen_frame.top);
                ResizeTo(_non_fullscreen_frame.Width(), _non_fullscreen_frame.Height());
                _SetBordered(_bordered);
                _SetResizable(_resizable);
            }
        }
    }

    // Members

    BView *_cur_view;
    SDL_BView *_SDL_View;
#ifdef SDL_VIDEO_OPENGL
    BGLView *_SDL_GLView;
    Uint32 _gl_type;
#endif

    int32 _last_buttons;
    int32 _id;           // Window id used by SDL_BApp
    bool _mouse_focused; // Does this window have mouse focus?
    bool _shown;
    bool _inhibit_resize;

    BRect *_prev_frame; // Previous position and size of the window
    bool _fullscreen;
    // valid only if fullscreen
    BRect _non_fullscreen_frame;
    bool _bordered;
    bool _resizable;

    // Framebuffer members
    BLocker *_buffer_locker;
    BBitmap *_bitmap;
};

/* FIXME:
 * An explanation of framebuffer flags.
 *
 * _connected -           Original variable used to let the drawing thread know
 *                         when changes are being made to the other framebuffer
 *                         members.
 * _connection_disabled - Used to signal to the drawing thread that the window
 *                         is closing, and the thread should exit.
 * _buffer_created -      True if the current buffer is valid
 * _buffer_dirty -        True if the window should be redrawn.
 * _trash_window_buffer - True if the window buffer needs to be trashed partway
 *                         through a draw cycle.  Occurs when the previous
 *                         buffer provided by DirectConnected() is invalidated.
 */
#endif // SDL_BWin_h_
