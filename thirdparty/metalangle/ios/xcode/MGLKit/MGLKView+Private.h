//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef MGLKView_Private_h
#define MGLKView_Private_h

#import "MGLKView.h"

@class MGLKViewController;

@interface MGLKView ()

@property(atomic) BOOL drawing;
@property(nonatomic, weak) MGLKViewController *controller;

@end

#endif /* MGLKView_Private_h */
