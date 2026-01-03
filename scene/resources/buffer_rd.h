/**************************************************************************/
/*  texture_rd.h                                                          */
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

#pragma once


#include "core/io/resource.h"
#include "servers/camera_server.h"
#include "servers/rendering_server.h"

class BufferRD : public Resource {
	enum BufferType {
		UniformBuffer,
		StorageBuffer
	};
	GDCLASS(BufferRD, Resource)

	mutable RID buffer_rid;
	RID buffer_rd_rid;
	Size2i size;
	BufferRD::BufferType buffer_type;


protected:
	static void _bind_methods();

public:
	virtual RID get_rid() const override;

	void set_buffer_rd_rid(RID p_buffer_rd_rid);
	RID get_buffer_rd_rid() const;

	// Internal function that should only be called from the rendering thread.
	void _set_buffer_rd_rid(RID p_buffer_rd_rid);

	BufferRD();
	~BufferRD();
};
