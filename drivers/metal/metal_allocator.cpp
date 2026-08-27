/**************************************************************************/
/*  metal_allocator.cpp                                                   */
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

#include "drivers/metal/metal_allocator.h"

#pragma mark - MetalAllocator

MetalAllocator *MetalAllocator::create(MTL::Device *p_device, bool p_use_heaps) {
	return memnew(MetalDeviceAllocator(p_device));
}

#pragma mark - MetalDeviceAllocator

MetalBuffer MetalDeviceAllocator::new_buffer(NS::UInteger p_length, MTL::ResourceOptions p_options) {
	MetalBuffer result;
	result.buffer = NS::TransferPtr(device->newBuffer(p_length, p_options));
	return result;
}

MetalTexture MetalDeviceAllocator::new_texture(const MTL::TextureDescriptor *p_desc) {
	MetalTexture result;
	result.texture = NS::TransferPtr(device->newTexture(p_desc));
	return result;
}

void MetalDeviceAllocator::free_buffer(MetalBuffer &p_buffer) {
	DEV_ASSERT(!p_buffer.allocation.is_valid());
	p_buffer.buffer.reset();
}

void MetalDeviceAllocator::free_texture(MetalTexture &p_texture) {
	DEV_ASSERT(!p_texture.allocation.is_valid());
	p_texture.texture.reset();
}

uint64_t MetalDeviceAllocator::get_heaps(LocalVector<MTL::Heap *> &r_heaps) {
	return 0;
}

uint64_t MetalDeviceAllocator::get_heap_generation() const {
	return 0;
}

#ifdef DEBUG_ENABLED
void MetalDeviceAllocator::get_stats(MetalAllocatorStats &r_stats) {
	r_stats = MetalAllocatorStats();
}
#endif
