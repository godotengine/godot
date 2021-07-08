//
// Copyright (c) 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef UTIL_IOS_WINDOW_H_
#define UTIL_IOS_WINDOW_H_

#include <memory>

#include "util/OSWindow.h"

class IOSWindow : public OSWindow
{
  public:
    IOSWindow();
    ~IOSWindow();

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

    bool hasOwnLoop() const override { return true; }
    int runOwnLoop(LoopStartDelegate initDelegate, LoopDelegate loopDelegate) override;

    // Register fuction to be executed when iOS app did finish launching.
    // Guaranteed to be executed before any callbacks passed to runOwnLoop().
    typedef std::function<void()> Delegate;
    static void RegisterAppStartDelegate(Delegate delegate);

    // For internal use only.
    void appDidFinishLaunching();
    void appWillTerminate();
    void loopIteration();
    void deviceOrientationDidChange();
    void viewDidAppear();
    void viewDidLayoutSubviews();

  private:
    void stopRunning();

    struct Impl;

    std::unique_ptr<Impl> mImpl;

    int mScreenWidth;
    int mScreenHeight;

    LoopStartDelegate mLoopInitializedDelegate;
    LoopDelegate mLoopIterationDelegate;
    bool mRunning;
};

#endif
