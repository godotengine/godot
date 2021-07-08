//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#import "MGLLayer.h"

NS_ASSUME_NONNULL_BEGIN

typedef enum MGLRenderingAPI : int
{
    kMGLRenderingAPIOpenGLES1 = 1,
    kMGLRenderingAPIOpenGLES2 = 2,
    kMGLRenderingAPIOpenGLES3 = 3,
} MGLRenderingAPI;

@interface MGLSharegroup : NSObject
@end

@interface MGLContext : NSObject

- (id)initWithAPI:(MGLRenderingAPI)api;

// NOTE: If you use sharegroup to share resources between contexts, make sure to call glFlush()
// when you finish the works of one context on a thread to make the changes visible to other
// contexts.
- (id)initWithAPI:(MGLRenderingAPI)api sharegroup:(MGLSharegroup *_Nullable)sharegroup;

- (MGLRenderingAPI)API;

@property(readonly) MGLSharegroup *sharegroup;

// Present the content of layer on screen as soon as possible.
- (BOOL)present:(MGLLayer *)layer;

+ (MGLContext *)currentContext;
+ (MGLLayer *)currentLayer;

// Set current context without layer. NOTE: this function is only useful if you want to
// create OpenGL resources such as textures/buffers before creating the presentation layer.
// Before drawing to a layer, use [MGLContext setCurrentContext: forLayer:] function instead.
+ (BOOL)setCurrentContext:(MGLContext *_Nullable)context;

// Set current context to render to the given layer.
+ (BOOL)setCurrentContext:(MGLContext *_Nullable)context forLayer:(MGLLayer *_Nullable)layer;

@end

NS_ASSUME_NONNULL_END
