/**************************************************************************/
/*  godot_switch.h                                                           */
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

#include <limits.h>
#include <locale.h>
#include <png.h>
#include <stdlib.h>
#include <unistd.h>
#include <zlib.h>

#include "main/main.h"

#include "os_switch.h"

// just to keep the main clean
#include <applet_splash.hpp>

int main(int argc, char *argv[]) {
	std::vector<std::string> args(argv, argv + argc);

	OS_Switch os(args);

	int apptype = appletGetAppletType();
	if (apptype != AppletType_Application && apptype != AppletType_SystemApplication) {
		// godot is not involved here we jsut display error message
		ERR_PRINT("application in applet mode!");
		display_applet_splash();
	} else {
		os.print("Main::setup\n");
		Error err = Main::setup(argv[0], argc - 1, &argv[1]);
		if (err != OK) {
			return 255;
		}
		os.print("Main::start\n");
		if (Main::start()) {
			os.run();
		}
		Main::cleanup();
	}
	os.print("godot switch exit\n");
	return 0;
}
