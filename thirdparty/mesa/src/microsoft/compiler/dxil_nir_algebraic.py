#
# Copyright (C) 2020 Microsoft Corporation
#
# Copyright (C) 2018 Alyssa Rosenzweig
#
# Copyright (C) 2016 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import argparse
import sys
import math

a = 'a'

# The nir_lower_bit_size() pass gets rid of all 8bit ALUs but insert new u2u8
# and i2i8 operations to convert the result back to the original type after the
# arithmetic operation is done. Those u2u8 and i2i8 operations, as any other
# 8bit operations, are not supported by DXIL and needs to be discarded. The
# dxil_nir_lower_8bit_conv() pass is here for that.
# Similarly, some hardware doesn't support 16bit values

no_8bit_conv = []
no_16bit_conv = []

def remove_unsupported_casts(arr, bit_size, mask, max_unsigned_float, min_signed_float, max_signed_float):
    for outer_op_type in ('u2u', 'i2i', 'u2f', 'i2f'):
        for outer_op_sz in (16, 32, 64):
            if outer_op_sz == bit_size:
                continue
            outer_op = outer_op_type + str(int(outer_op_sz))
            for inner_op_type in ('u2u', 'i2i'):
                inner_op = inner_op_type + str(int(bit_size))
                for src_sz in (16, 32, 64):
                    if (src_sz == bit_size):
                        continue
                    # Coming from integral, truncate appropriately
                    orig_seq = (outer_op, (inner_op, 'a@' + str(int(src_sz))))
                    if (outer_op[0] == 'u'):
                        new_seq = ('iand', a, mask)
                    else:
                        shift = src_sz - bit_size
                        new_seq = ('ishr', ('ishl', a, shift), shift)
                    # Make sure the destination is the right type/size
                    if outer_op_sz != src_sz or outer_op[2] != inner_op[0]:
                        new_seq = (outer_op, new_seq)
                    arr += [(orig_seq, new_seq)]
            for inner_op_type in ('f2u', 'f2i'):
                inner_op = inner_op_type + str(int(bit_size))
                if (outer_op[2] == 'f'):
                    # From float and to float, just truncate via min/max, and ensure the right float size
                    for src_sz in (16, 32, 64):
                        if (src_sz == bit_size):
                            continue
                        orig_seq = (outer_op, (inner_op, 'a@' + str(int(src_sz))))
                        if (outer_op[0] == 'u'):
                            new_seq = ('fmin', ('fmax', a, 0.0), max_unsigned_float)
                        else:
                            new_seq = ('fmin', ('fmax', a, min_signed_float), max_signed_float)
                        if outer_op_sz != src_sz:
                            new_seq = ('f2f' + str(int(outer_op_sz)), new_seq)
                        arr += [(orig_seq, new_seq)]
                else:
                    # From float to integral, convert to integral type first, then truncate
                    orig_seq = (outer_op, (inner_op, a))
                    float_conv = ('f2' + inner_op[2] + str(int(outer_op_sz)), a)
                    if (outer_op[0] == 'u'):
                        new_seq = ('iand', float_conv, mask)
                    else:
                        shift = outer_op_sz - bit_size
                        new_seq = ('ishr', ('ishl', float_conv, shift), shift)
                    arr += [(orig_seq, new_seq)]

remove_unsupported_casts(no_8bit_conv, 8, 0xff, 255.0, -128.0, 127.0)
remove_unsupported_casts(no_16bit_conv, 16, 0xffff, 65535.0, -32768.0, 32767.0)

lower_x2b = [
  (('b2b32', 'a'), ('b2i32', 'a')),
  (('b2b1', 'a'), ('ine', ('b2i32', a), 0)),
  (('f2b1', 'a'), ('fneu', a, 0)),
]

no_16bit_conv += [
  (('f2f32', ('u2u16', 'a@32')), ('unpack_half_2x16_split_x', 'a')),
  (('u2u32', ('f2f16_rtz', 'a@32')), ('pack_half_2x16_split', 'a', 0)),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--import-path', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.import_path)
    run()


def run():
    import nir_algebraic  # pylint: disable=import-error

    print('#include "dxil_nir.h"')

    print(nir_algebraic.AlgebraicPass("dxil_nir_lower_8bit_conv",
                                      no_8bit_conv).render())
    print(nir_algebraic.AlgebraicPass("dxil_nir_lower_16bit_conv",
                                      no_16bit_conv).render())
    print(nir_algebraic.AlgebraicPass("dxil_nir_lower_x2b",
                                      lower_x2b).render())

if __name__ == '__main__':
    main()
