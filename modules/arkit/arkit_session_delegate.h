/*************************************************************************/
/*  arkit_session_delegate.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef ARKIT_SESSION_DELEGATE_H
#define ARKIT_SESSION_DELEGATE_H

#import <ARKit/ARKit.h>
#import <UIKit/UIKit.h>

class ARKitInterface;

@interface ARKitSessionDelegate : NSObject <ARSessionDelegate> {
	ARKitInterface *arkit_interface;
}

@property(nonatomic) ARKitInterface *arkit_interface;

- (void)session:(ARSession *)session didAddAnchors:(NSArray<ARAnchor *> *)anchors API_AVAILABLE(ios(11.0));
- (void)session:(ARSession *)session didRemoveAnchors:(NSArray<ARAnchor *> *)anchors API_AVAILABLE(ios(11.0));
- (void)session:(ARSession *)session didUpdateAnchors:(NSArray<ARAnchor *> *)anchors API_AVAILABLE(ios(11.0));
@end

#endif /* !ARKIT_SESSION_DELEGATE_H */
