//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// OSXWindow.h: Definition of the implementation of OSWindow for OSX

#ifndef UTIL_OSX_WINDOW_H_
#define UTIL_OSX_WINDOW_H_

#import <Cocoa/Cocoa.h>

#include "util/OSWindow.h"

class OSXWindow;

@interface WindowDelegate : NSObject {
    OSXWindow *mWindow;
}
- (id)initWithWindow:(OSXWindow *)window;
@end

@interface ContentView : NSView {
    OSXWindow *mWindow;
    NSTrackingArea *mTrackingArea;
    int mCurrentModifier;
}
- (id)initWithWindow:(OSXWindow *)window;
@end

class OSXWindow : public OSWindow
{
  public:
    OSXWindow();
    ~OSXWindow();

    bool initialize(const std::string &name, int width, int height) override;
    void destroy() override;

    void resetNativeWindow() override;
    EGLNativeWindowType getNativeWindow() const override;
    EGLNativeDisplayType getNativeDisplay() const override;

    void messageLoop() override;

    void setMousePosition(int x, int y) override;
    bool setPosition(int x, int y) override;
    bool resize(int width, int height) override;
    void setVisible(bool isVisible) override;

    void signalTestEvent() override;

    NSWindow *getNSWindow() const;

  private:
    NSWindow *mWindow;
    WindowDelegate *mDelegate;
    ContentView *mView;
};

#endif  // UTIL_OSX_WINDOW_H_
