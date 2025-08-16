/**************************************************************************/
/*  low_level.cpp                                                         */
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

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/worker_thread_pool.hpp>
#include <godot_cpp/classes/xml_parser.hpp>

#include <godot_cpp/godot.hpp>

namespace godot {
Error XMLParser::_open_buffer(const uint8_t *p_buffer, size_t p_size) {
	return (Error)internal::gdextension_interface_xml_parser_open_buffer(_owner, p_buffer, p_size);
}

uint64_t FileAccess::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	return internal::gdextension_interface_file_access_get_buffer(_owner, p_dst, p_length);
}

void FileAccess::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	internal::gdextension_interface_file_access_store_buffer(_owner, p_src, p_length);
}

WorkerThreadPool::TaskID WorkerThreadPool::add_native_task(void (*p_func)(void *), void *p_userdata, bool p_high_priority, const String &p_description) {
	return (TaskID)internal::gdextension_interface_worker_thread_pool_add_native_task(_owner, p_func, p_userdata, p_high_priority, (GDExtensionConstStringPtr)&p_description);
}

WorkerThreadPool::GroupID WorkerThreadPool::add_native_group_task(void (*p_func)(void *, uint32_t), void *p_userdata, int p_elements, int p_tasks, bool p_high_priority, const String &p_description) {
	return (GroupID)internal::gdextension_interface_worker_thread_pool_add_native_group_task(_owner, p_func, p_userdata, p_elements, p_tasks, p_high_priority, (GDExtensionConstStringPtr)&p_description);
}

uint8_t *Image::ptrw() {
	return internal::gdextension_interface_image_ptrw(_owner);
}

const uint8_t *Image::ptr() {
	return internal::gdextension_interface_image_ptr(_owner);
}

} // namespace godot
