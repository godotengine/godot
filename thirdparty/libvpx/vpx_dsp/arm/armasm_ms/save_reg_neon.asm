; This file was created from a .asm file
;  using the ads2armasm_ms.pl script.
;
;  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;


    EXPORT  |vpx_push_neon|
    EXPORT  |vpx_pop_neon|

    
    

    AREA |.text|, CODE, READONLY, ALIGN=2

|vpx_push_neon| PROC
    vst1.i64            {d8, d9, d10, d11}, [r0]!
    vst1.i64            {d12, d13, d14, d15}, [r0]!
    bx              lr

    ENDP
    ALIGN 4

|vpx_pop_neon| PROC
    vld1.i64            {d8, d9, d10, d11}, [r0]!
    vld1.i64            {d12, d13, d14, d15}, [r0]!
    bx              lr

    ENDP
    ALIGN 4

    END

