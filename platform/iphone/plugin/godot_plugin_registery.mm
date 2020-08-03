/*************************************************************************/
/*  ios.mm                                                               */
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

#include "godot_plugin_registery.h"

#import <Foundation/Foundation.h>

void GodotPluginRegistery::register_plugin_classes() {
	Class godot_plugin_class = NSClassFromString(@"GodotPlugin");
	int classes_num = objc_getClassList(NULL, 0);
	Class *classes = NULL;

	classes = (__unsafe_unretained Class *)malloc(sizeof(Class) * classes_num);
	classes_num = objc_getClassList(classes, classes_num);

	for (NSInteger i = 0; i < classes_num; i++) {
		Class super_class = classes[i];
		do {
			super_class = class_getSuperclass(super_class);
		} while (super_class && super_class != godot_plugin_class);

		if (super_class == nil) {
			continue;
		}

		NSLog(@"Plugin Class Name: %@", NSStringFromClass(classes[i]));

		id plugin_class_instance = [[classes[i] alloc] init];

		SEL plugin_name_selector = NSSelectorFromString(@"getPluginName");
		NSString *plugin_name = (NSString *)[plugin_class_instance performSelector:plugin_name_selector];
		String plugin_name_str = String([plugin_name UTF8String]);

		//-------Custom method parse------------
		method_map.clear();
		SEL methods_name_selector = NSSelectorFromString(@"getMethodName");
		NSArray<NSString *> *methods_with_param = (NSArray<NSString *> *)[plugin_class_instance performSelector:methods_name_selector];
		for (NSString *str in methods_with_param) {
			NSArray *arr = [str componentsSeparatedByString:@"|"];

			// Must be 2 part, return type|method body
			ERR_FAIL_COND([arr count] != 2);

			NSString *return_type = [arr objectAtIndex:0];
			NSString *rest_of_method_body = [arr objectAtIndex:1];

			NSArray *arr2 = [rest_of_method_body componentsSeparatedByString:@"-"];
			NSString *method_name = [arr2 objectAtIndex:0];

			MethodMetaData mmd;
			mmd.method_name = String([method_name UTF8String]);
			mmd.return_type = String([return_type UTF8String]);

			NSLog(@"--------------");
			NSLog(@"Method Return Type Debug: %@", return_type);
			NSLog(@"Method Name Debug: %@", method_name);

			if ([arr2 count] == 1) {
				method_map[String([method_name UTF8String])] = mmd;
				continue;
			}

			NSString *method_args_str = [arr2 objectAtIndex:1];
			if ([method_args_str length] == 0) {
				method_map[String([method_name UTF8String])] = mmd;
				continue;
			}

			NSArray *arr3 = [method_args_str componentsSeparatedByString:@":"];
			List<String> method_args;
			for (int i = 0; i < [arr3 count]; i++) {
				NSLog(@"Method Arg Debug: %@", [arr3 objectAtIndex:i]);
				method_args.push_back(String([[arr3 objectAtIndex:i] UTF8String]));
			}
			mmd.method_args = method_args;
			method_map[String([method_name UTF8String])] = mmd;
			NSLog(@"--------------");
		}
		//-------Custom method parse------------

		iOSSingleton *ios_singleton = memnew(iOSSingleton);
		ios_singleton->set_instance(plugin_class_instance);
		unsigned int method_count = 0;
		Method *methods = class_copyMethodList(classes[i], &method_count);

		for (unsigned int i = 0; i < method_count; i++) {
			Method method = methods[i];
			const char *method_name_pointer = sel_getName(method_getName(method));
			NSString *instance_method_selector_name = [NSString stringWithCString:method_name_pointer encoding:NSASCIIStringEncoding];

			SEL instance_method_selector = NSSelectorFromString(instance_method_selector_name);
			NSArray *selector_parts = [instance_method_selector_name componentsSeparatedByString:@":"];
			NSString *instance_method_name = selector_parts[0];
			NSMethodSignature *method_signature = [plugin_class_instance methodSignatureForSelector:instance_method_selector];
			Vector<Variant::Type> types;

			Map<StringName, MethodMetaData>::Element *E = method_map.find(String([instance_method_name UTF8String]));
			if (E != NULL) {
				NSUInteger number_of_args = [method_signature numberOfArguments];
				ERR_FAIL_COND((number_of_args - 2) != E->get().method_args.size());
				for (const List<String>::Element *EL = E->get().method_args.front(); EL; EL = EL->next()) {
					NSLog(@"Method Read Debug: %@", [[[NSString alloc] initWithUTF8String:EL->get().utf8().get_data()] autorelease]);
					types.push_back(get_objc_type(EL->get()));
				}
				NSLog(@"Method Read Debug: %@", [[[NSString alloc] initWithUTF8String:E->get().return_type.utf8().get_data()] autorelease]);
				ios_singleton->add_method(String([instance_method_name UTF8String]), String(method_name_pointer), types, get_objc_type(E->get().return_type));

			} else {
				NSLog(@"Element is null %@", instance_method_name);
			}
		}

		Engine::get_singleton()->add_singleton(Engine::Singleton(plugin_name_str, ios_singleton));
		ProjectSettings::get_singleton()->set(plugin_name_str, ios_singleton);
		free(methods);
	}

	free(classes);
}

GodotPluginRegistery::GodotPluginRegistery(){};
