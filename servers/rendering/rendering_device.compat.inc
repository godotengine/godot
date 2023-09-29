/**************************************************************************/
/*  rendering_device.compat.inc                                           */
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

#ifndef DISABLE_DEPRECATED

RID RenderingDevice::_shader_create_from_bytecode_bind_compat_79606(const Vector<uint8_t> &p_shader_binary) {
	return shader_create_from_bytecode(p_shader_binary, RID());
}

BitField<RenderingDevice::BarrierMask> RenderingDevice::_convert_barrier_mask_81356(BitField<BarrierMask> p_old_barrier) {
	if (p_old_barrier == 7) {
		return BARRIER_MASK_ALL_BARRIERS;
	} else if (p_old_barrier == 16) {
		return BARRIER_MASK_NO_BARRIER;
	}

	BitField<BarrierMask> new_barrier;
	if (p_old_barrier & 1) {
		new_barrier.set_flag(BARRIER_MASK_VERTEX);
	}
	if (p_old_barrier & 2) {
		new_barrier.set_flag(BARRIER_MASK_FRAGMENT);
	}
	if (p_old_barrier & 4) {
		new_barrier.set_flag(BARRIER_MASK_COMPUTE);
	}
	if (p_old_barrier & 8) {
		new_barrier.set_flag(BARRIER_MASK_TRANSFER);
	}
	return new_barrier;
}

void RenderingDevice::_draw_list_end_bind_compat_81356(BitField<BarrierMask> p_post_barrier) {
	draw_list_end(_convert_barrier_mask_81356(p_post_barrier));
}

void RenderingDevice::_compute_list_end_bind_compat_81356(BitField<BarrierMask> p_post_barrier) {
	compute_list_end(_convert_barrier_mask_81356(p_post_barrier));
}

void RenderingDevice::_barrier_bind_compat_81356(BitField<BarrierMask> p_from, BitField<BarrierMask> p_to) {
	barrier(_convert_barrier_mask_81356(p_from), _convert_barrier_mask_81356(p_to));
}

void RenderingDevice::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("shader_create_from_bytecode", "binary_data"), &RenderingDevice::_shader_create_from_bytecode_bind_compat_79606);
	ClassDB::bind_compatibility_method(D_METHOD("draw_list_end", "post_barrier"), &RenderingDevice::_draw_list_end_bind_compat_81356, DEFVAL(7));
	ClassDB::bind_compatibility_method(D_METHOD("compute_list_end", "post_barrier"), &RenderingDevice::_compute_list_end_bind_compat_81356, DEFVAL(7));
	ClassDB::bind_compatibility_method(D_METHOD("barrier", "from", "to"), &RenderingDevice::_barrier_bind_compat_81356, DEFVAL(7), DEFVAL(7));
}

#endif
