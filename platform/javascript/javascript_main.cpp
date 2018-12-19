/*************************************************************************/
/*  javascript_main.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/io/resource_loader.h"
#include "main/main.h"
#include "os_javascript.h"

#include <emscripten/emscripten.h>

extern "C" EMSCRIPTEN_KEEPALIVE void main_after_fs_sync(char *p_idbfs_err) {

	String idbfs_err = String::utf8(p_idbfs_err);
	if (!idbfs_err.empty()) {
		print_line("IndexedDB not available: " + idbfs_err);
	}
	OS_JavaScript *os = OS_JavaScript::get_singleton();
	os->set_idb_available(idbfs_err.empty());
	// Ease up compatibility.
	ResourceLoader::set_abort_on_missing_resources(false);
	Main::start();
	os->run_async();
}

int main(int argc, char *argv[]) {

	// Sync from persistent state into memory and then
	// run the 'main_after_fs_sync' function.
	/* clang-format off */
	EM_ASM(
		FS.mkdir('/userfs');
		FS.mount(IDBFS, {}, '/userfs');
		FS.syncfs(true, function(err) {
			ccall('main_after_fs_sync', null, ['string'], [err ? err.message : ""])
		});
	);
	/* clang-format on */

	new OS_JavaScript(argc, argv);
	// TODO: Check error return value.
	Main::setup(argv[0], argc - 1, &argv[1]);

	return 0;
	// Continued async in main_after_fs_sync() from the syncfs() callback.
}
