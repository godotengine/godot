/*************************************************************************/
/*  plugin_wrapper.mm                                                    */
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

#include "plugin_wrapper.h"

#import <Foundation/Foundation.h>

@implementation PluginWrapper
- (void)emitSignals:(NSString *)signalName withArgs:(NSArray<NSObject *> *)signalArgs {
	Object *obj = Engine::get_singleton()->get_singleton_object(String([@"GodotiOSPlugin" UTF8String]));
	iOSSingleton *singleton = (iOSSingleton *)obj;

	String signal_name = String([signalName UTF8String]);

	int count = [signalArgs count];
	ERR_FAIL_COND_MSG(count > VARIANT_ARG_MAX, "Maximum argument count exceeded!");

	// dispatch_async(dispatch_get_main_queue(), ^(void) {
	Variant variant_params[VARIANT_ARG_MAX];
	const Variant *args[VARIANT_ARG_MAX];

	for (int i = 0; i < count; i++) {
		variant_params[i] = get_objc_class_type(String(class_getName([signalArgs[i] class])), signalArgs[i]);
		args[i] = &variant_params[i];
	};

	singleton->emit_signal(signal_name, args, count);
}
@end
