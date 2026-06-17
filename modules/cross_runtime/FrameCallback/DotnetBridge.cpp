/**************************************************************************/
/*  DotnetBridge.cpp                                                      */
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

#include "DotnetBridge.h"

#include <emscripten.h>

DotnetBridge *DotnetBridge::singleton = nullptr;

// Constructor: registers the singleton and defers SceneTree connection
// until the message queue is ready, since the tree may not exist yet at construction time.
DotnetBridge::DotnetBridge() {
	singleton = this;
	MessageQueue::get_singleton()->push_callable(
			callable_mp(this, &DotnetBridge::_try_connect));
}

// Deferred connection attempt: keeps re-queuing itself until the SceneTree
// is available, then connects to process_frame to drive the .NET game loop.
void DotnetBridge::_try_connect() {
	SceneTree *tree = SceneTree::get_singleton();
	if (!tree) {
		MessageQueue::get_singleton()->push_callable(
				callable_mp(this, &DotnetBridge::_try_connect));
		return;
	}
	if (!tree->is_connected("process_frame", callable_mp(this, &DotnetBridge::_on_process_frame))) {
		tree->connect("process_frame", callable_mp(this, &DotnetBridge::_on_process_frame));
	}
}

// Called every frame via SceneTree::process_frame.
// Reads the frame delta from the root viewport and forwards it to the .NET
// game loop via __stepFrame, which is registered by the JS bootstrap layer.
// NOTE: driving .NET from Godot's frame signal is a known architectural coupling —
// this will be revisited when a cleaner cross-runtime frame driver is available.
void DotnetBridge::_on_process_frame() {
	SceneTree *tree = SceneTree::get_singleton();
	if (!tree) {
		return;
	}
	double delta = tree->get_root()->get_process_delta_time();
	EM_ASM({
        if (globalThis.__stepFrame){
            globalThis.__stepFrame($0);
} }, delta);
}
