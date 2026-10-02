#!/usr/bin/env perl
##
##  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

package thumb;

sub FixThumbInstructions($)
{
    # Write additions with shifts, such as "add r10, r11, lsl #8",
    # in three operand form, "add r10, r10, r11, lsl #8".
    s/(add\s+)(r\d+),\s*(r\d+),\s*(lsl #\d+)/$1$2, $2, $3, $4/g;

    # Convert additions with a non-constant shift into a sequence
    # with left shift, addition and a right shift (to restore the
    # register to the original value). Currently the right shift
    # isn't necessary in the code base since the values in these
    # registers aren't used, but doing the shift for consistency.
    # This converts instructions such as "add r12, r12, r5, lsl r4"
    # into the sequence "lsl r5, r4", "add r12, r12, r5", "lsr r5, r4".
    s/^(\s*)(add)(\s+)(r\d+),\s*(r\d+),\s*(r\d+),\s*lsl (r\d+)/$1lsl$3$6, $7\n$1$2$3$4, $5, $6\n$1lsr$3$6, $7/g;

    # Convert loads with right shifts in the indexing into a
    # sequence of an add, load and sub. This converts
    # "ldrb r4, [r9, lr, asr #1]" into "add r9, r9, lr, asr #1",
    # "ldrb r9, [r9]", "sub r9, r9, lr, asr #1".
    s/^(\s*)(ldrb)(\s+)(r\d+),\s*\[(\w+),\s*(\w+),\s*(asr #\d+)\]/$1add $3$5, $5, $6, $7\n$1$2$3$4, [$5]\n$1sub $3$5, $5, $6, $7/g;

    # Convert register indexing with writeback into a separate add
    # instruction. This converts "ldrb r12, [r1, r2]!" into
    # "ldrb r12, [r1, r2]", "add r1, r1, r2".
    s/^(\s*)(ldrb)(\s+)(r\d+),\s*\[(\w+),\s*(\w+)\]!/$1$2$3$4, [$5, $6]\n$1add $3$5, $6/g;

    # Convert negative register indexing into separate sub/add instructions.
    # This converts "ldrne r4, [src, -pstep, lsl #1]" into
    # "subne src, src, pstep, lsl #1", "ldrne r4, [src]",
    # "addne src, src, pstep, lsl #1". In a couple of cases where
    # this is used, it's used for two subsequent load instructions,
    # where a hand-written version of it could merge two subsequent
    # add and sub instructions.
    s/^(\s*)((ldr|str|pld)(ne)?)(\s+)(r\d+,\s*)?\[(\w+), -([^\]]+)\]/$1sub$4$5$7, $7, $8\n$1$2$5$6\[$7\]\n$1add$4$5$7, $7, $8/g;

    # Convert register post indexing to a separate add instruction.
    # This converts "ldrneb r9, [r0], r2" into "ldrneb r9, [r0]",
    # "addne r0, r0, r2".
    s/^(\s*)((ldr|str)(ne)?[bhd]?)(\s+)(\w+),(\s*\w+,)?\s*\[(\w+)\],\s*(\w+)/$1$2$5$6,$7 [$8]\n$1add$4$5$8, $8, $9/g;

    # Convert "mov pc, lr" into "bx lr", since the former only works
    # for switching from arm to thumb (and only in armv7), but not
    # from thumb to arm.
    s/mov(\s*)pc\s*,\s*lr/bx$1lr/g;
}

1;
