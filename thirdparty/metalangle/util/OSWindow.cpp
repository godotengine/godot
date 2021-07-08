//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "OSWindow.h"

#include <iostream>
#include <sstream>

#include "common/debug.h"

#ifndef DEBUG_EVENTS
#    define DEBUG_EVENTS 0
#endif

#if DEBUG_EVENTS
static const char *MouseButtonName(MouseButton button)
{
    switch (button)
    {
        case MOUSEBUTTON_UNKNOWN:
            return "Unknown";
        case MOUSEBUTTON_LEFT:
            return "Left";
        case MOUSEBUTTON_RIGHT:
            return "Right";
        case MOUSEBUTTON_MIDDLE:
            return "Middle";
        case MOUSEBUTTON_BUTTON4:
            return "Button4";
        case MOUSEBUTTON_BUTTON5:
            return "Button5";
        default:
            UNREACHABLE();
            return nullptr;
    }
}

static const char *KeyName(Key key)
{
    switch (key)
    {
        case KEY_UNKNOWN:
            return "Unknown";
        case KEY_A:
            return "A";
        case KEY_B:
            return "B";
        case KEY_C:
            return "C";
        case KEY_D:
            return "D";
        case KEY_E:
            return "E";
        case KEY_F:
            return "F";
        case KEY_G:
            return "G";
        case KEY_H:
            return "H";
        case KEY_I:
            return "I";
        case KEY_J:
            return "J";
        case KEY_K:
            return "K";
        case KEY_L:
            return "L";
        case KEY_M:
            return "M";
        case KEY_N:
            return "N";
        case KEY_O:
            return "O";
        case KEY_P:
            return "P";
        case KEY_Q:
            return "Q";
        case KEY_R:
            return "R";
        case KEY_S:
            return "S";
        case KEY_T:
            return "T";
        case KEY_U:
            return "U";
        case KEY_V:
            return "V";
        case KEY_W:
            return "W";
        case KEY_X:
            return "X";
        case KEY_Y:
            return "Y";
        case KEY_Z:
            return "Z";
        case KEY_NUM0:
            return "Num0";
        case KEY_NUM1:
            return "Num1";
        case KEY_NUM2:
            return "Num2";
        case KEY_NUM3:
            return "Num3";
        case KEY_NUM4:
            return "Num4";
        case KEY_NUM5:
            return "Num5";
        case KEY_NUM6:
            return "Num6";
        case KEY_NUM7:
            return "Num7";
        case KEY_NUM8:
            return "Num8";
        case KEY_NUM9:
            return "Num9";
        case KEY_ESCAPE:
            return "Escape";
        case KEY_LCONTROL:
            return "Left Control";
        case KEY_LSHIFT:
            return "Left Shift";
        case KEY_LALT:
            return "Left Alt";
        case KEY_LSYSTEM:
            return "Left System";
        case KEY_RCONTROL:
            return "Right Control";
        case KEY_RSHIFT:
            return "Right Shift";
        case KEY_RALT:
            return "Right Alt";
        case KEY_RSYSTEM:
            return "Right System";
        case KEY_MENU:
            return "Menu";
        case KEY_LBRACKET:
            return "Left Bracket";
        case KEY_RBRACKET:
            return "Right Bracket";
        case KEY_SEMICOLON:
            return "Semicolon";
        case KEY_COMMA:
            return "Comma";
        case KEY_PERIOD:
            return "Period";
        case KEY_QUOTE:
            return "Quote";
        case KEY_SLASH:
            return "Slash";
        case KEY_BACKSLASH:
            return "Backslash";
        case KEY_TILDE:
            return "Tilde";
        case KEY_EQUAL:
            return "Equal";
        case KEY_DASH:
            return "Dash";
        case KEY_SPACE:
            return "Space";
        case KEY_RETURN:
            return "Return";
        case KEY_BACK:
            return "Back";
        case KEY_TAB:
            return "Tab";
        case KEY_PAGEUP:
            return "Page Up";
        case KEY_PAGEDOWN:
            return "Page Down";
        case KEY_END:
            return "End";
        case KEY_HOME:
            return "Home";
        case KEY_INSERT:
            return "Insert";
        case KEY_DELETE:
            return "Delete";
        case KEY_ADD:
            return "Add";
        case KEY_SUBTRACT:
            return "Substract";
        case KEY_MULTIPLY:
            return "Multiply";
        case KEY_DIVIDE:
            return "Divide";
        case KEY_LEFT:
            return "Left";
        case KEY_RIGHT:
            return "Right";
        case KEY_UP:
            return "Up";
        case KEY_DOWN:
            return "Down";
        case KEY_NUMPAD0:
            return "Numpad 0";
        case KEY_NUMPAD1:
            return "Numpad 1";
        case KEY_NUMPAD2:
            return "Numpad 2";
        case KEY_NUMPAD3:
            return "Numpad 3";
        case KEY_NUMPAD4:
            return "Numpad 4";
        case KEY_NUMPAD5:
            return "Numpad 5";
        case KEY_NUMPAD6:
            return "Numpad 6";
        case KEY_NUMPAD7:
            return "Numpad 7";
        case KEY_NUMPAD8:
            return "Numpad 8";
        case KEY_NUMPAD9:
            return "Numpad 9";
        case KEY_F1:
            return "F1";
        case KEY_F2:
            return "F2";
        case KEY_F3:
            return "F3";
        case KEY_F4:
            return "F4";
        case KEY_F5:
            return "F5";
        case KEY_F6:
            return "F6";
        case KEY_F7:
            return "F7";
        case KEY_F8:
            return "F8";
        case KEY_F9:
            return "F9";
        case KEY_F10:
            return "F10";
        case KEY_F11:
            return "F11";
        case KEY_F12:
            return "F12";
        case KEY_F13:
            return "F13";
        case KEY_F14:
            return "F14";
        case KEY_F15:
            return "F15";
        case KEY_PAUSE:
            return "Pause";
        default:
            return "Unknown Key";
    }
}

static std::string KeyState(const Event::KeyEvent &event)
{
    if (event.Shift || event.Control || event.Alt || event.System)
    {
        std::ostringstream buffer;
        buffer << " [";

        if (event.Shift)
        {
            buffer << "Shift";
        }
        if (event.Control)
        {
            buffer << "Control";
        }
        if (event.Alt)
        {
            buffer << "Alt";
        }
        if (event.System)
        {
            buffer << "System";
        }

        buffer << "]";
        return buffer.str();
    }
    return "";
}

static void PrintEvent(const Event &event)
{
    switch (event.Type)
    {
        case Event::EVENT_CLOSED:
            std::cout << "Event: Window Closed" << std::endl;
            break;
        case Event::EVENT_MOVED:
            std::cout << "Event: Window Moved (" << event.Move.X << ", " << event.Move.Y << ")"
                      << std::endl;
            break;
        case Event::EVENT_RESIZED:
            std::cout << "Event: Window Resized (" << event.Size.Width << ", " << event.Size.Height
                      << ")" << std::endl;
            break;
        case Event::EVENT_LOST_FOCUS:
            std::cout << "Event: Window Lost Focus" << std::endl;
            break;
        case Event::EVENT_GAINED_FOCUS:
            std::cout << "Event: Window Gained Focus" << std::endl;
            break;
        case Event::EVENT_TEXT_ENTERED:
            // TODO(cwallez) show the character
            std::cout << "Event: Text Entered" << std::endl;
            break;
        case Event::EVENT_KEY_PRESSED:
            std::cout << "Event: Key Pressed (" << KeyName(event.Key.Code) << KeyState(event.Key)
                      << ")" << std::endl;
            break;
        case Event::EVENT_KEY_RELEASED:
            std::cout << "Event: Key Released (" << KeyName(event.Key.Code) << KeyState(event.Key)
                      << ")" << std::endl;
            break;
        case Event::EVENT_MOUSE_WHEEL_MOVED:
            std::cout << "Event: Mouse Wheel (" << event.MouseWheel.Delta << ")" << std::endl;
            break;
        case Event::EVENT_MOUSE_BUTTON_PRESSED:
            std::cout << "Event: Mouse Button Pressed " << MouseButtonName(event.MouseButton.Button)
                      << " at (" << event.MouseButton.X << ", " << event.MouseButton.Y << ")"
                      << std::endl;
            break;
        case Event::EVENT_MOUSE_BUTTON_RELEASED:
            std::cout << "Event: Mouse Button Released "
                      << MouseButtonName(event.MouseButton.Button) << " at (" << event.MouseButton.X
                      << ", " << event.MouseButton.Y << ")" << std::endl;
            break;
        case Event::EVENT_MOUSE_MOVED:
            std::cout << "Event: Mouse Moved (" << event.MouseMove.X << ", " << event.MouseMove.Y
                      << ")" << std::endl;
            break;
        case Event::EVENT_MOUSE_ENTERED:
            std::cout << "Event: Mouse Entered Window" << std::endl;
            break;
        case Event::EVENT_MOUSE_LEFT:
            std::cout << "Event: Mouse Left Window" << std::endl;
            break;
        case Event::EVENT_TEST:
            std::cout << "Event: Test" << std::endl;
            break;
        default:
            UNREACHABLE();
            break;
    }
}
#endif

OSWindow::OSWindow() : mX(0), mY(0), mWidth(0), mHeight(0) {}

OSWindow::~OSWindow() {}

int OSWindow::getX() const
{
    return mX;
}

int OSWindow::getY() const
{
    return mY;
}

int OSWindow::getWidth() const
{
    return mWidth;
}

int OSWindow::getHeight() const
{
    return mHeight;
}

bool OSWindow::takeScreenshot(uint8_t *pixelData)
{
    return false;
}

bool OSWindow::popEvent(Event *event)
{
    if (mEvents.size() > 0 && event)
    {
        *event = mEvents.front();
        mEvents.pop_front();
        return true;
    }
    else
    {
        return false;
    }
}

void OSWindow::pushEvent(Event event)
{
    switch (event.Type)
    {
        case Event::EVENT_MOVED:
            mX = event.Move.X;
            mY = event.Move.Y;
            break;
        case Event::EVENT_RESIZED:
            mWidth  = event.Size.Width;
            mHeight = event.Size.Height;
            break;
        default:
            break;
    }

    mEvents.push_back(event);

#if DEBUG_EVENTS
    PrintEvent(event);
#endif
}

bool OSWindow::didTestEventFire()
{
    Event topEvent;
    while (popEvent(&topEvent))
    {
        if (topEvent.Type == Event::EVENT_TEST)
        {
            return true;
        }
    }

    return false;
}

// static
void OSWindow::Delete(OSWindow **window)
{
    delete *window;
    *window = nullptr;
}
