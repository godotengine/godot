
'''
/**************************************************************************
 *
 * Copyright 2009-2010 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/**
 * @file
 * Pixel format packing and unpacking functions.
 *
 * @author Jose Fonseca <jfonseca@vmware.com>
 */
'''

import sys

from u_format_parse import *


def inv_swizzles(swizzles):
    '''Return an array[4] of inverse swizzle terms'''
    '''Only pick the first matching value to avoid l8 getting blue and i8 getting alpha'''
    inv_swizzle = [None]*4
    for i in range(4):
        swizzle = swizzles[i]
        if swizzle < 4 and inv_swizzle[swizzle] == None:
            inv_swizzle[swizzle] = i
    return inv_swizzle

def print_channels(format, func):
    if format.nr_channels() <= 1:
        func(format.le_channels, format.le_swizzles)
    else:
        if (format.le_channels == format.be_channels and
            [c.shift for c in format.le_channels] ==
            [c.shift for c in format.be_channels] and
            format.le_swizzles == format.be_swizzles):
            func(format.le_channels, format.le_swizzles)
        else:
            print('#if UTIL_ARCH_BIG_ENDIAN')
            func(format.be_channels, format.be_swizzles)
            print('#else')
            func(format.le_channels, format.le_swizzles)
            print('#endif')

def generate_format_type(format):
    '''Generate a structure that describes the format.'''

    assert format.layout == PLAIN

    def generate_bitfields(channels, swizzles):
        for channel in channels:
            if channel.type == VOID:
                if channel.size:
                    print('   unsigned %s:%u;' % (channel.name, channel.size))
            elif channel.type == UNSIGNED:
                print('   unsigned %s:%u;' % (channel.name, channel.size))
            elif channel.type in (SIGNED, FIXED):
                print('   int %s:%u;' % (channel.name, channel.size))
            elif channel.type == FLOAT:
                if channel.size == 64:
                    print('   double %s;' % (channel.name))
                elif channel.size == 32:
                    print('   float %s;' % (channel.name))
                else:
                    print('   unsigned %s:%u;' % (channel.name, channel.size))
            else:
                assert 0

    def generate_full_fields(channels, swizzles):
        for channel in channels:
            assert channel.size % 8 == 0 and is_pot(channel.size)
            if channel.type == VOID:
                if channel.size:
                    print('   uint%u_t %s;' % (channel.size, channel.name))
            elif channel.type == UNSIGNED:
                print('   uint%u_t %s;' % (channel.size, channel.name))
            elif channel.type in (SIGNED, FIXED):
                print('   int%u_t %s;' % (channel.size, channel.name))
            elif channel.type == FLOAT:
                if channel.size == 64:
                    print('   double %s;' % (channel.name))
                elif channel.size == 32:
                    print('   float %s;' % (channel.name))
                elif channel.size == 16:
                    print('   uint16_t %s;' % (channel.name))
                else:
                    assert 0
            else:
                assert 0

    use_bitfields = False
    for channel in format.le_channels:
        if channel.size % 8 or not is_pot(channel.size):
            use_bitfields = True

    print('struct util_format_%s {' % format.short_name())
    if use_bitfields:
        print_channels(format, generate_bitfields)
    else:
        print_channels(format, generate_full_fields)
    print('};')
    print()


def is_format_supported(format):
    '''Determines whether we actually have the plumbing necessary to generate the
    to read/write to/from this format.'''

    # FIXME: Ideally we would support any format combination here.

    if format.layout != PLAIN:
        return False

    for i in range(4):
        channel = format.le_channels[i]
        if channel.type not in (VOID, UNSIGNED, SIGNED, FLOAT, FIXED):
            return False
        if channel.type == FLOAT and channel.size not in (16, 32, 64):
            return False

    return True

def native_type(format):
    '''Get the native appropriate for a format.'''

    if format.name == 'PIPE_FORMAT_R11G11B10_FLOAT':
        return 'uint32_t'
    if format.name == 'PIPE_FORMAT_R9G9B9E5_FLOAT':
        return 'uint32_t'

    if format.layout == PLAIN:
        if not format.is_array():
            # For arithmetic pixel formats return the integer type that matches the whole pixel
            return 'uint%u_t' % format.block_size()
        else:
            # For array pixel formats return the integer type that matches the color channel
            channel = format.array_element()
            if channel.type in (UNSIGNED, VOID):
                return 'uint%u_t' % channel.size
            elif channel.type in (SIGNED, FIXED):
                return 'int%u_t' % channel.size
            elif channel.type == FLOAT:
                if channel.size == 16:
                    return 'uint16_t'
                elif channel.size == 32:
                    return 'float'
                elif channel.size == 64:
                    return 'double'
                else:
                    assert False
            else:
                assert False
    else:
        assert False


def intermediate_native_type(bits, sign):
    '''Find a native type adequate to hold intermediate results of the request bit size.'''

    bytes = 4 # don't use anything smaller than 32bits
    while bytes * 8 < bits:
        bytes *= 2
    bits = bytes*8

    if sign:
        return 'int%u_t' % bits
    else:
        return 'uint%u_t' % bits


def get_one_shift(type):
    '''Get the number of the bit that matches unity for this type.'''
    if type.type == 'FLOAT':
        assert False
    if not type.norm:
        return 0
    if type.type == UNSIGNED:
        return type.size
    if type.type == SIGNED:
        return type.size - 1
    if type.type == FIXED:
        return type.size / 2
    assert False


def truncate_mantissa(x, bits):
    '''Truncate an integer so it can be represented exactly with a floating
    point mantissa'''

    assert isinstance(x, int)

    s = 1
    if x < 0:
        s = -1
        x = -x

    # We can represent integers up to mantissa + 1 bits exactly
    mask = (1 << (bits + 1)) - 1

    # Slide the mask until the MSB matches
    shift = 0
    while (x >> shift) & ~mask:
        shift += 1

    x &= mask << shift
    x *= s
    return x


def value_to_native(type, value):
    '''Get the value of unity for this type.'''
    if type.type == FLOAT:
        if type.size <= 32 \
            and isinstance(value, int):
            return truncate_mantissa(value, 23)
        return value
    if type.type == FIXED:
        return int(value * (1 << (type.size // 2)))
    if not type.norm:
        return int(value)
    if type.type == UNSIGNED:
        return int(value * ((1 << type.size) - 1))
    if type.type == SIGNED:
        return int(value * ((1 << (type.size - 1)) - 1))
    assert False


def native_to_constant(type, value):
    '''Get the value of unity for this type.'''
    if type.type == FLOAT:
        if type.size <= 32:
            return "%.1ff" % float(value)
        else:
            return "%.1f" % float(value)
    else:
        return str(int(value))


def get_one(type):
    '''Get the value of unity for this type.'''
    return value_to_native(type, 1)


def clamp_expr(src_channel, dst_channel, dst_native_type, value):
    '''Generate the expression to clamp the value in the source type to the
    destination type range.'''

    if src_channel == dst_channel:
        return value

    src_min = src_channel.min()
    src_max = src_channel.max()
    dst_min = dst_channel.min()
    dst_max = dst_channel.max()

    # Translate the destination range to the src native value
    dst_min_native = native_to_constant(src_channel, value_to_native(src_channel, dst_min))
    dst_max_native = native_to_constant(src_channel, value_to_native(src_channel, dst_max))

    if src_min < dst_min and src_max > dst_max:
        return 'CLAMP(%s, %s, %s)' % (value, dst_min_native, dst_max_native)

    if src_max > dst_max:
        return 'MIN2(%s, %s)' % (value, dst_max_native)

    if src_min < dst_min:
        return 'MAX2(%s, %s)' % (value, dst_min_native)

    return value


def conversion_expr(src_channel,
                    dst_channel, dst_native_type,
                    value,
                    clamp=True,
                    src_colorspace = RGB,
                    dst_colorspace = RGB):
    '''Generate the expression to convert a value between two types.'''

    if src_colorspace != dst_colorspace:
        if src_colorspace == SRGB:
            assert src_channel.type == UNSIGNED
            assert src_channel.norm
            assert src_channel.size <= 8
            assert src_channel.size >= 4
            assert dst_colorspace == RGB
            if src_channel.size < 8:
                value = '%s << %x | %s >> %x' % (value, 8 - src_channel.size, value, 2 * src_channel.size - 8)
            if dst_channel.type == FLOAT:
                return 'util_format_srgb_8unorm_to_linear_float(%s)' % value
            else:
                assert dst_channel.type == UNSIGNED
                assert dst_channel.norm
                assert dst_channel.size == 8
                return 'util_format_srgb_to_linear_8unorm(%s)' % value
        elif dst_colorspace == SRGB:
            assert dst_channel.type == UNSIGNED
            assert dst_channel.norm
            assert dst_channel.size <= 8
            assert src_colorspace == RGB
            if src_channel.type == FLOAT:
                value =  'util_format_linear_float_to_srgb_8unorm(%s)' % value
            else:
                assert src_channel.type == UNSIGNED
                assert src_channel.norm
                assert src_channel.size == 8
                value = 'util_format_linear_to_srgb_8unorm(%s)' % value
            # XXX rounding is all wrong.
            if dst_channel.size < 8:
                return '%s >> %x' % (value, 8 - dst_channel.size)
            else:
                return value
        elif src_colorspace == ZS:
            pass
        elif dst_colorspace == ZS:
            pass
        else:
            assert 0

    if src_channel == dst_channel:
        return value

    src_type = src_channel.type
    src_size = src_channel.size
    src_norm = src_channel.norm
    src_pure = src_channel.pure

    # Promote half to float
    if src_type == FLOAT and src_size == 16:
        value = '_mesa_half_to_float(%s)' % value
        src_size = 32

    # Special case for float <-> ubytes for more accurate results
    # Done before clamping since these functions already take care of that
    if src_type == UNSIGNED and src_norm and src_size == 8 and dst_channel.type == FLOAT and dst_channel.size == 32:
        return 'ubyte_to_float(%s)' % value
    if src_type == FLOAT and src_size == 32 and dst_channel.type == UNSIGNED and dst_channel.norm and dst_channel.size == 8:
        return 'float_to_ubyte(%s)' % value

    if clamp:
        if dst_channel.type != FLOAT or src_type != FLOAT:
            value = clamp_expr(src_channel, dst_channel, dst_native_type, value)

    if src_type in (SIGNED, UNSIGNED) and dst_channel.type in (SIGNED, UNSIGNED):
        if not src_norm and not dst_channel.norm:
            # neither is normalized -- just cast
            return '(%s)%s' % (dst_native_type, value)

        if src_norm and dst_channel.norm:
            return "_mesa_%snorm_to_%snorm(%s, %d, %d)" % ("s" if src_type == SIGNED else "u",
                                                           "s" if dst_channel.type == SIGNED else "u",
                                                           value, src_channel.size, dst_channel.size)
        else:
            # We need to rescale using an intermediate type big enough to hold the multiplication of both
            src_one = get_one(src_channel)
            dst_one = get_one(dst_channel)
            tmp_native_type = intermediate_native_type(src_size + dst_channel.size, src_channel.sign and dst_channel.sign)
            value = '((%s)%s)' % (tmp_native_type, value)
            value = '(%s)(%s * 0x%x / 0x%x)' % (dst_native_type, value, dst_one, src_one)
            return value


    # Promote to either float or double
    if src_type != FLOAT:
        if src_norm or src_type == FIXED:
            one = get_one(src_channel)
            if src_size <= 23:
                value = '(%s * (1.0f/0x%x))' % (value, one)
                if dst_channel.size <= 32:
                    value = '(float)%s' % value
                src_size = 32
            else:
                # bigger than single precision mantissa, use double
                value = '(%s * (1.0/0x%x))' % (value, one)
                src_size = 64
            src_norm = False
        else:
            if src_size <= 23 or dst_channel.size <= 32:
                value = '(float)%s' % value
                src_size = 32
            else:
                # bigger than single precision mantissa, use double
                value = '(double)%s' % value
                src_size = 64
        src_type = FLOAT

    # Convert double or float to non-float
    if dst_channel.type != FLOAT:
        if dst_channel.norm or dst_channel.type == FIXED:
            dst_one = get_one(dst_channel)
            if dst_channel.size <= 23:
                value = 'util_iround(%s * 0x%x)' % (value, dst_one)
            else:
                # bigger than single precision mantissa, use double
                value = '(%s * (double)0x%x)' % (value, dst_one)
        value = '(%s)%s' % (dst_native_type, value)
    else:
        # Cast double to float when converting to either half or float
        if dst_channel.size <= 32 and src_size > 32:
            value = '(float)%s' % value
            src_size = 32

        if dst_channel.size == 16:
            value = '_mesa_float_to_float16_rtz(%s)' % value
        elif dst_channel.size == 64 and src_size < 64:
            value = '(double)%s' % value

    return value


def generate_unpack_kernel(format, dst_channel, dst_native_type):

    if not is_format_supported(format):
        return

    assert format.layout == PLAIN

    def unpack_from_bitmask(channels, swizzles):
        depth = format.block_size()
        print('         uint%u_t value = *(const uint%u_t *)src;' % (depth, depth))

        # Compute the intermediate unshifted values
        for i in range(format.nr_channels()):
            src_channel = channels[i]
            value = 'value'
            shift = src_channel.shift
            if src_channel.type == UNSIGNED:
                if shift:
                    value = '%s >> %u' % (value, shift)
                if shift + src_channel.size < depth:
                    value = '(%s) & 0x%x' % (value, (1 << src_channel.size) - 1)
                print('         uint%u_t %s = %s;' % (depth, src_channel.name, value))
            elif src_channel.type == SIGNED:
                if shift + src_channel.size < depth:
                    # Align the sign bit
                    lshift = depth - (shift + src_channel.size)
                    value = '%s << %u' % (value, lshift)
                # Cast to signed
                value = '(int%u_t)(%s) ' % (depth, value)
                if src_channel.size < depth:
                    # Align the LSB bit
                    rshift = depth - src_channel.size
                    value = '(%s) >> %u' % (value, rshift)
                print('         int%u_t %s = %s;' % (depth, src_channel.name, value))
            else:
                value = None

        # Convert, swizzle, and store final values
        for i in range(4):
            swizzle = swizzles[i]
            if swizzle < 4:
                src_channel = channels[swizzle]
                src_colorspace = format.colorspace
                if src_colorspace == SRGB and i == 3:
                    # Alpha channel is linear
                    src_colorspace = RGB
                value = src_channel.name
                value = conversion_expr(src_channel,
                                        dst_channel, dst_native_type,
                                        value,
                                        src_colorspace = src_colorspace)
            elif swizzle == SWIZZLE_0:
                value = '0'
            elif swizzle == SWIZZLE_1:
                value = get_one(dst_channel)
            elif swizzle == SWIZZLE_NONE:
                value = '0'
            else:
                assert False
            print('         dst[%u] = %s; /* %s */' % (i, value, 'rgba'[i]))

    def unpack_from_struct(channels, swizzles):
        print('         struct util_format_%s pixel;' % format.short_name())
        print('         memcpy(&pixel, src, sizeof pixel);')

        for i in range(4):
            swizzle = swizzles[i]
            if swizzle < 4:
                src_channel = channels[swizzle]
                src_colorspace = format.colorspace
                if src_colorspace == SRGB and i == 3:
                    # Alpha channel is linear
                    src_colorspace = RGB
                value = 'pixel.%s' % src_channel.name
                value = conversion_expr(src_channel,
                                        dst_channel, dst_native_type,
                                        value,
                                        src_colorspace = src_colorspace)
            elif swizzle == SWIZZLE_0:
                value = '0'
            elif swizzle == SWIZZLE_1:
                value = get_one(dst_channel)
            elif swizzle == SWIZZLE_NONE:
                value = '0'
            else:
                assert False
            print('         dst[%u] = %s; /* %s */' % (i, value, 'rgba'[i]))

    if format.is_bitmask():
        print_channels(format, unpack_from_bitmask)
    else:
        print_channels(format, unpack_from_struct)


def generate_pack_kernel(format, src_channel, src_native_type):

    if not is_format_supported(format):
        return

    dst_native_type = native_type(format)

    assert format.layout == PLAIN

    def pack_into_bitmask(channels, swizzles):
        inv_swizzle = inv_swizzles(swizzles)

        depth = format.block_size()
        print('         uint%u_t value = 0;' % depth)

        for i in range(4):
            dst_channel = channels[i]
            shift = dst_channel.shift
            if inv_swizzle[i] is not None:
                value ='src[%u]' % inv_swizzle[i]
                dst_colorspace = format.colorspace
                if dst_colorspace == SRGB and inv_swizzle[i] == 3:
                    # Alpha channel is linear
                    dst_colorspace = RGB
                value = conversion_expr(src_channel,
                                        dst_channel, dst_native_type,
                                        value,
                                        dst_colorspace = dst_colorspace)
                if dst_channel.type in (UNSIGNED, SIGNED):
                    if shift + dst_channel.size < depth:
                        value = '(%s) & 0x%x' % (value, (1 << dst_channel.size) - 1)
                    if shift:
                        value = '(uint32_t)(%s) << %u' % (value, shift)
                    if dst_channel.type == SIGNED:
                        # Cast to unsigned
                        value = '(uint%u_t)(%s) ' % (depth, value)
                else:
                    value = None
                if value is not None:
                    print('         value |= %s;' % (value))

        print('         *(uint%u_t *)dst = value;' % depth)

    def pack_into_struct(channels, swizzles):
        inv_swizzle = inv_swizzles(swizzles)

        print('         struct util_format_%s pixel = {0};' % format.short_name())

        for i in range(4):
            dst_channel = channels[i]
            width = dst_channel.size
            if inv_swizzle[i] is None:
                continue
            dst_colorspace = format.colorspace
            if dst_colorspace == SRGB and inv_swizzle[i] == 3:
                # Alpha channel is linear
                dst_colorspace = RGB
            value ='src[%u]' % inv_swizzle[i]
            value = conversion_expr(src_channel,
                                    dst_channel, dst_native_type,
                                    value,
                                    dst_colorspace = dst_colorspace)
            print('         pixel.%s = %s;' % (dst_channel.name, value))

        print('         memcpy(dst, &pixel, sizeof pixel);')

    if format.is_bitmask():
        print_channels(format, pack_into_bitmask)
    else:
        print_channels(format, pack_into_struct)


def generate_format_unpack(format, dst_channel, dst_native_type, dst_suffix):
    '''Generate the function to unpack pixels from a particular format'''

    name = format.short_name()

    if "8unorm" in dst_suffix:
        dst_proto_type = dst_native_type
    else:
        dst_proto_type = 'void'

    proto = 'util_format_%s_unpack_%s(%s *restrict dst_row, const uint8_t *restrict src, unsigned width)' % (
        name, dst_suffix, dst_proto_type)
    print('void %s;' % proto, file=sys.stdout2)

    print('void')
    print(proto)
    print('{')

    if is_format_supported(format):
        print('   %s *dst = dst_row;' % (dst_native_type))
        print(
            '   for (unsigned x = 0; x < width; x += %u) {' % (format.block_width,))

        generate_unpack_kernel(format, dst_channel, dst_native_type)

        print('      src += %u;' % (format.block_size() / 8,))
        print('      dst += 4;')
        print('   }')

    print('}')
    print()


def generate_format_pack(format, src_channel, src_native_type, src_suffix):
    '''Generate the function to pack pixels to a particular format'''

    name = format.short_name()

    print('void')
    print('util_format_%s_pack_%s(uint8_t *restrict dst_row, unsigned dst_stride, const %s *restrict src_row, unsigned src_stride, unsigned width, unsigned height)' %
          (name, src_suffix, src_native_type))
    print('{')

    print('void util_format_%s_pack_%s(uint8_t *restrict dst_row, unsigned dst_stride, const %s *restrict src_row, unsigned src_stride, unsigned width, unsigned height);' %
          (name, src_suffix, src_native_type), file=sys.stdout2)

    if is_format_supported(format):
        print('   unsigned x, y;')
        print('   for(y = 0; y < height; y += %u) {' % (format.block_height,))
        print('      const %s *src = src_row;' % (src_native_type))
        print('      uint8_t *dst = dst_row;')
        print('      for(x = 0; x < width; x += %u) {' % (format.block_width,))

        generate_pack_kernel(format, src_channel, src_native_type)

        print('         src += 4;')
        print('         dst += %u;' % (format.block_size() / 8,))
        print('      }')
        print('      dst_row += dst_stride;')
        print('      src_row += src_stride/sizeof(*src_row);')
        print('   }')

    print('}')
    print()


def generate_format_fetch(format, dst_channel, dst_native_type):
    '''Generate the function to unpack pixels from a particular format'''

    name = format.short_name()

    proto = 'util_format_%s_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, UNUSED unsigned i, UNUSED unsigned j)' % (name)
    print('void %s;' % proto, file=sys.stdout2)

    print('void')
    print(proto)

    print('{')
    print('   %s *dst = in_dst;' % dst_native_type)

    if is_format_supported(format):
        generate_unpack_kernel(format, dst_channel, dst_native_type)

    print('}')
    print()


def is_format_hand_written(format):
    return format.layout != PLAIN or format.colorspace == ZS


def generate(formats):
    print()
    print('#include "pipe/p_compiler.h"')
    print('#include "util/u_math.h"')
    print('#include "util/half_float.h"')
    print('#include "u_format.h"')
    print('#include "u_format_other.h"')
    print('#include "util/format_srgb.h"')
    print('#include "format_utils.h"')
    print('#include "u_format_yuv.h"')
    print('#include "u_format_zs.h"')
    print('#include "u_format_pack.h"')
    print()

    for format in formats:
        if not is_format_hand_written(format):

            if is_format_supported(format) and not format.is_bitmask():
                generate_format_type(format)

            if format.is_pure_unsigned():
                native_type = 'unsigned'
                suffix = 'unsigned'
                channel = Channel(UNSIGNED, False, True, 32)

                generate_format_unpack(format, channel, native_type, suffix)
                generate_format_pack(format, channel, native_type, suffix)
                generate_format_fetch(format, channel, native_type)

                channel = Channel(SIGNED, False, True, 32)
                native_type = 'int'
                suffix = 'signed'
                generate_format_pack(format, channel, native_type, suffix)
            elif format.is_pure_signed():
                native_type = 'int'
                suffix = 'signed'
                channel = Channel(SIGNED, False, True, 32)

                generate_format_unpack(format, channel, native_type, suffix)
                generate_format_pack(format, channel, native_type, suffix)
                generate_format_fetch(format, channel, native_type)

                native_type = 'unsigned'
                suffix = 'unsigned'
                channel = Channel(UNSIGNED, False, True, 32)
                generate_format_pack(format, channel, native_type, suffix)
            else:
                channel = Channel(FLOAT, False, False, 32)
                native_type = 'float'
                suffix = 'rgba_float'

                generate_format_unpack(format, channel, native_type, suffix)
                generate_format_pack(format, channel, native_type, suffix)
                generate_format_fetch(format, channel, native_type)

                channel = Channel(UNSIGNED, True, False, 8)
                native_type = 'uint8_t'
                suffix = 'rgba_8unorm'

                generate_format_unpack(format, channel, native_type, suffix)
                generate_format_pack(format, channel, native_type, suffix)
