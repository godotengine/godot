/*************************************************************************/
/*  test_io.cpp                                                          */
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

#include "test_io.h"

#ifdef MINIZIP_ENABLED

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/dir_access.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/print_string.h"
#include "core/project_settings.h"
#include "scene/resources/texture.h"

#include "core/io/file_access_memory.h"

namespace TestIO {

class TestMainLoop : public MainLoop {

	bool quit;

public:
	virtual void input_event(const Ref<InputEvent> &p_event) {
	}
	virtual bool idle(float p_time) {
		return false;
	}

	virtual void request_quit() {

		quit = true;
	}
	virtual void init() {

		quit = true;
	}
	virtual bool iteration(float p_time) {

		return quit;
	}
	virtual void finish() {
	}
};

MainLoop *test() {

	print_line("this is test io");
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	da->change_dir(".");
	print_line("Opening current dir " + da->get_current_dir());
	String entry;
	da->list_dir_begin();
	while ((entry = da->get_next()) != "") {

		print_line("entry " + entry + " is dir: " + Variant(da->current_is_dir()));
	};
	da->list_dir_end();

	RES texture = ResourceLoader::load("test_data/rock.png");
	ERR_FAIL_COND_V(texture.is_null(), NULL);

	ResourceSaver::save("test_data/rock.xml", texture);

	print_line("localize paths");
	print_line(ProjectSettings::get_singleton()->localize_path("algo.xml"));
	print_line(ProjectSettings::get_singleton()->localize_path("c:\\windows\\algo.xml"));
	print_line(ProjectSettings::get_singleton()->localize_path(ProjectSettings::get_singleton()->get_resource_path() + "/something/something.xml"));
	print_line(ProjectSettings::get_singleton()->localize_path("somedir/algo.xml"));

	{

		FileAccess *z = FileAccess::open("test_data/archive.zip", FileAccess::READ);
		int len = z->get_len();
		Vector<uint8_t> zip;
		zip.resize(len);
		z->get_buffer(zip.ptrw(), len);
		z->close();
		memdelete(z);

		FileAccessMemory::register_file("a_package", zip);
		FileAccess::make_default<FileAccessMemory>(FileAccess::ACCESS_RESOURCES);
		FileAccess::make_default<FileAccessMemory>(FileAccess::ACCESS_FILESYSTEM);
		FileAccess::make_default<FileAccessMemory>(FileAccess::ACCESS_USERDATA);

		print_line("archive test");
	};

	print_line("test done");

	return memnew(TestMainLoop);
}
} // namespace TestIO

#else

namespace TestIO {

MainLoop *test() {

	return NULL;
}
} // namespace TestIO
#endif
