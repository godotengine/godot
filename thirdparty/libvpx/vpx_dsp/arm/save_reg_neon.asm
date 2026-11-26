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

    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

|vpx_push_neon| PROC
    vstm            r0!, {d8-d15}
    bx              lr

    ENDP

|vpx_pop_neon| PROC
    vldm            r0!, {d8-d15}
    bx              lr

    ENDP

    END

