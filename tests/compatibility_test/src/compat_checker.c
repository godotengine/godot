/**************************************************************************/
/*  compat_checker.c                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "compat_checker.h"
#include <stdbool.h>

GDExtensionInterfaceClassdbGetMethodBind classdb_get_method_bind = NULL;
GDExtensionInterfaceVariantGetPtrBuiltinMethod variant_get_ptr_builtin_method = NULL;
GDExtensionInterfaceVariantGetPtrUtilityFunction variant_get_ptr_utility_function = NULL;

GDExtensionPtrDestructor string_name_destructor = NULL;
GDExtensionInterfaceStringNameNewWithLatin1Chars string_name_new_with_latin1_chars = NULL;

typedef struct
{
	uint8_t data[8];
} StringName;

/**
 * Platform APIs are being registered only after extensions, making them unavailable during initialization (on any level).
 *
 * Due to that we run the tests in Mainloop `startup_callback` (available since Godot 4.5), called after the initialization.
 */
void startup_func() {
	bool success = (builtin_methods_compatibility_test() && class_methods_compatibility_test() && utility_functions_compatibility_test());
	if (success) {
		fprintf(stdout, "Outcome = SUCCESS\n");
	} else {
		fprintf(stderr, "Outcome = FAILURE\n");
	}
	fprintf(stdout, "COMPATIBILITY TEST FINISHED.\n");
}

GDExtensionMainLoopCallbacks callbacks = {
	(GDExtensionMainLoopStartupCallback)startup_func,
	NULL,
	NULL
};

void initialize_compatibility_test(void *p_userdata, GDExtensionInitializationLevel p_level) {}

void deinitialize_compatibility_test(void *p_userdata, GDExtensionInitializationLevel p_level) {}

GDExtensionBool builtin_methods_compatibility_test() {
	FILE *file = fopen("./builtin_methods.txt", "r");
	if (file == NULL) {
		fprintf(stderr, "Failed to open file `builtin_methods.txt` \n");
		return false;
	}

	bool ret = true;
	char line[512];

	while (fgets(line, sizeof line, file) != NULL) {
		int variant_type;
		char method_name[128];
		GDExtensionInt hash;
		if (sscanf(line, "%d %s %ld", &variant_type, method_name, &hash) != 3) {
			continue;
		}

		StringName method_stringname;
		string_name_new_with_latin1_chars(&method_stringname, method_name, false);
		GDExtensionPtrBuiltInMethod method_bind = variant_get_ptr_builtin_method(variant_type, &method_stringname, hash);

		if (method_bind == NULL) {
			fprintf(stderr, "Method bind not found: %d::%s (hash: %ld)\n", variant_type, method_name, hash);
			ret = false;
		}

		string_name_destructor(&method_stringname);
	}

	fclose(file);
	return ret;
}

GDExtensionBool utility_functions_compatibility_test() {
	FILE *file = fopen("./utility_functions.txt", "r");
	if (file == NULL) {
		fprintf(stderr, "Failed to open file `utility_functions.txt` \n");
		return false;
	}

	bool ret = true;
	char line[256];

	while (fgets(line, sizeof line, file) != NULL) {
		char method_name[128];
		GDExtensionInt hash;
		if (sscanf(line, "%s %ld", method_name, &hash) != 2) {
			continue;
		}

		StringName method_stringname;
		string_name_new_with_latin1_chars(&method_stringname, method_name, false);
		GDExtensionPtrUtilityFunction function_bind = variant_get_ptr_utility_function(&method_stringname, hash);

		if (function_bind == NULL) {
			fprintf(stderr, "Utility function not found: %s (hash: %ld)\n", method_name, hash);
			ret = false;
		}

		string_name_destructor(&method_stringname);
	}

	fclose(file);
	return ret;
}

GDExtensionBool class_methods_compatibility_test() {
	FILE *file = fopen("./class_methods.txt", "r");
	if (file == NULL) {
		fprintf(stderr, "Failed to open file `class_methods.txt` \n");
		return false;
	}

	char current_class_name[128] = "";
	bool ret = true;
	char line[512];
	bool has_class_string = false;
	StringName p_classname;

	while (fgets(line, sizeof(line), file) != NULL) {
		GDExtensionInt hash;
		StringName p_methodname;
		char class_name[128];
		char method_name[128];

		if (sscanf(line, "%s %s %ld", class_name, method_name, &hash) != 3) {
			continue;
		}

		if (strcmp(current_class_name, class_name) != 0) {
			if (has_class_string) {
				string_name_destructor(&p_classname);
			}
			strcpy(current_class_name, class_name);

			string_name_new_with_latin1_chars(&p_classname, current_class_name, false);

			has_class_string = true;
		}

		string_name_new_with_latin1_chars(&p_methodname, method_name, false);
		GDExtensionMethodBindPtr method_bind = classdb_get_method_bind(&p_classname, &p_methodname, hash);

		if (method_bind == NULL) {
			fprintf(stderr, "Method bind not found: %s.%s (hash: %ld)\n", class_name, method_name, hash);
			ret = false;
		}

		string_name_destructor(&p_methodname);
	}

	if (has_class_string) {
		string_name_destructor(&p_classname);
	}

	fclose(file);
	return ret;
}

GDExtensionBool __attribute__((visibility("default"))) compatibility_test_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
	classdb_get_method_bind = (GDExtensionInterfaceClassdbGetMethodBind)p_get_proc_address("classdb_get_method_bind");
	if (classdb_get_method_bind == NULL) {
		fprintf(stderr, "Failed to load interface method `classdb_get_method_bind`\n");
		return false;
	}

	variant_get_ptr_builtin_method = (GDExtensionInterfaceVariantGetPtrBuiltinMethod)p_get_proc_address("variant_get_ptr_builtin_method");
	if (variant_get_ptr_builtin_method == NULL) {
		fprintf(stderr, "Failed to load interface method `variant_get_ptr_builtin_method`\n");
		return false;
	}

	variant_get_ptr_utility_function = (GDExtensionInterfaceVariantGetPtrUtilityFunction)p_get_proc_address("variant_get_ptr_utility_function");
	if (variant_get_ptr_utility_function == NULL) {
		fprintf(stderr, "Failed to load interface method `variant_get_ptr_utility_function`\n");
		return false;
	}

	GDExtensionInterfaceVariantGetPtrDestructor variant_get_ptr_destructor = (GDExtensionInterfaceVariantGetPtrDestructor)p_get_proc_address("variant_get_ptr_destructor");
	if (variant_get_ptr_destructor == NULL) {
		fprintf(stderr, "Failed to load interface method `variant_get_ptr_destructor`\n");
		return false;
	}
	string_name_destructor = variant_get_ptr_destructor(GDEXTENSION_VARIANT_TYPE_STRING_NAME);

	string_name_new_with_latin1_chars = (GDExtensionInterfaceStringNameNewWithLatin1Chars)p_get_proc_address("string_name_new_with_latin1_chars");
	if (classdb_get_method_bind == NULL) {
		fprintf(stderr, "Failed to load interface method `string_name_new_with_latin1_chars`\n");
		return false;
	}

	GDExtensionInterfaceRegisterMainLoopCallbacks register_main_loop_callbacks = (GDExtensionInterfaceRegisterMainLoopCallbacks)p_get_proc_address("register_main_loop_callbacks");
	if (register_main_loop_callbacks == NULL) {
		fprintf(stderr, "Failed to load interface method `register_main_loop_callbacks`\n");
		return false;
	}

	register_main_loop_callbacks(p_library, &callbacks);

	r_initialization->initialize = initialize_compatibility_test;
	r_initialization->deinitialize = deinitialize_compatibility_test;
	r_initialization->userdata = NULL;
	r_initialization->minimum_initialization_level = GDEXTENSION_INITIALIZATION_EDITOR;

	return true;
}
