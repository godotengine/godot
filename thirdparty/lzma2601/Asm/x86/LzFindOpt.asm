; LzFindOpt.asm -- ASM version of GetMatchesSpecN_2() function
; 2024-06-18: Igor Pavlov : Public domain
;

ifndef x64
; x64=1
; .err <x64_IS_REQUIRED>
endif

include 7zAsm.asm

MY_ASM_START

ifndef Z7_LZ_FIND_OPT_ASM_USE_SEGMENT
if (IS_LINUX gt 0)
  Z7_LZ_FIND_OPT_ASM_USE_SEGMENT equ 1
else
  Z7_LZ_FIND_OPT_ASM_USE_SEGMENT equ 1
endif
endif

ifdef Z7_LZ_FIND_OPT_ASM_USE_SEGMENT
_TEXT$LZFINDOPT SEGMENT ALIGN(64) 'CODE'
MY_ALIGN macro num:req
        align  num
        ; align  16
endm
else
MY_ALIGN macro num:req
        ; We expect that ".text" is aligned for 16-bytes.
        ; So we don't need large alignment inside our function.
        align  16
endm
endif


MY_ALIGN_16 macro
        MY_ALIGN 16
endm

MY_ALIGN_32 macro
        MY_ALIGN 32
endm

MY_ALIGN_64 macro
        MY_ALIGN 64
endm


t0_L    equ x0_L
t0_x    equ x0
t0      equ r0
t1_x    equ x3
t1      equ r3

cp_x    equ t1_x
cp_r    equ t1
m       equ x5
m_r     equ r5
len_x   equ x6
len     equ r6
diff_x  equ x7
diff    equ r7
len0    equ r10
len1_x  equ x11
len1    equ r11
maxLen_x equ x12
maxLen  equ r12
d       equ r13
ptr0    equ r14
ptr1    equ r15

d_lim       equ m_r
cycSize     equ len_x
hash_lim    equ len0
delta1_x    equ len1_x
delta1_r    equ len1
delta_x     equ maxLen_x
delta_r     equ maxLen
hash        equ ptr0
src         equ ptr1



if (IS_LINUX gt 0)

; r1 r2  r8 r9        : win32
; r7 r6  r2 r1  r8 r9 : linux

lenLimit        equ r8
lenLimit_x      equ x8
; pos_r           equ r2
pos             equ x2
cur             equ r1
son             equ r9

else

lenLimit        equ REG_ABI_PARAM_2
lenLimit_x      equ REG_ABI_PARAM_2_x
pos             equ REG_ABI_PARAM_1_x
cur             equ REG_ABI_PARAM_0
son             equ REG_ABI_PARAM_3

endif


if (IS_LINUX gt 0)
    maxLen_OFFS         equ  (REG_SIZE * (6 + 1))
else
    cutValue_OFFS       equ  (REG_SIZE * (8 + 1 + 4))
    d_OFFS              equ  (REG_SIZE + cutValue_OFFS)
    maxLen_OFFS         equ  (REG_SIZE + d_OFFS)
endif
    hash_OFFS           equ  (REG_SIZE + maxLen_OFFS)
    limit_OFFS          equ  (REG_SIZE + hash_OFFS)
    size_OFFS           equ  (REG_SIZE + limit_OFFS)
    cycPos_OFFS         equ  (REG_SIZE + size_OFFS)
    cycSize_OFFS        equ  (REG_SIZE + cycPos_OFFS)
    posRes_OFFS         equ  (REG_SIZE + cycSize_OFFS)
    
if (IS_LINUX gt 0)
else
    cutValue_PAR        equ  [r0 + cutValue_OFFS]
    d_PAR               equ  [r0 + d_OFFS]
endif
    maxLen_PAR          equ  [r0 + maxLen_OFFS]
    hash_PAR            equ  [r0 + hash_OFFS]
    limit_PAR           equ  [r0 + limit_OFFS]
    size_PAR            equ  [r0 + size_OFFS]
    cycPos_PAR          equ  [r0 + cycPos_OFFS]
    cycSize_PAR         equ  [r0 + cycSize_OFFS]
    posRes_PAR          equ  [r0 + posRes_OFFS]


    cutValue_VAR        equ  DWORD PTR [r4 + 8 * 0]
    cutValueCur_VAR     equ  DWORD PTR [r4 + 8 * 0 + 4]
    cycPos_VAR          equ  DWORD PTR [r4 + 8 * 1 + 0]
    cycSize_VAR         equ  DWORD PTR [r4 + 8 * 1 + 4]
    hash_VAR            equ  QWORD PTR [r4 + 8 * 2]
    limit_VAR           equ  QWORD PTR [r4 + 8 * 3]
    size_VAR            equ  QWORD PTR [r4 + 8 * 4]
    distances           equ  QWORD PTR [r4 + 8 * 5]
    maxLen_VAR          equ  QWORD PTR [r4 + 8 * 6]

    Old_RSP             equ  QWORD PTR [r4 + 8 * 7]
    LOCAL_SIZE          equ  8 * 8

COPY_VAR_32 macro dest_var, src_var
        mov     x3, src_var
        mov     dest_var, x3
endm

COPY_VAR_64 macro dest_var, src_var
        mov     r3, src_var
        mov     dest_var, r3
endm


ifdef Z7_LZ_FIND_OPT_ASM_USE_SEGMENT
; MY_ALIGN_64
else
  MY_ALIGN_16
endif
MY_PROC GetMatchesSpecN_2, 13
MY_PUSH_PRESERVED_ABI_REGS
        mov     r0, RSP
        lea     r3, [r0 - LOCAL_SIZE]
        and     r3, -64
        mov     RSP, r3
        mov     Old_RSP, r0

if (IS_LINUX gt 0)
        mov     d,            REG_ABI_PARAM_5       ; r13 = r9
        mov     cutValue_VAR, REG_ABI_PARAM_4_x     ;     = r8
        mov     son,          REG_ABI_PARAM_3       ;  r9 = r1
        mov     r8,           REG_ABI_PARAM_2       ;  r8 = r2
        mov     pos,          REG_ABI_PARAM_1_x     ;  r2 = x6
        mov     r1,           REG_ABI_PARAM_0       ;  r1 = r7
else
        COPY_VAR_32 cutValue_VAR, cutValue_PAR
        mov     d, d_PAR
endif

        COPY_VAR_64 limit_VAR, limit_PAR
        
        mov     hash_lim, size_PAR
        mov     size_VAR, hash_lim
        
        mov     cp_x, cycPos_PAR
        mov     hash, hash_PAR

        mov     cycSize, cycSize_PAR
        mov     cycSize_VAR, cycSize
        
        ; we want cur in (rcx). So we change the cur and lenLimit variables
        sub     lenLimit, cur
        neg     lenLimit_x
        inc     lenLimit_x
        
        mov     t0_x, maxLen_PAR
        sub     t0, lenLimit
        mov     maxLen_VAR, t0

        jmp     main_loop

MY_ALIGN_64
fill_empty:
        ; ptr0 = *ptr1 = kEmptyHashValue;
        mov     QWORD PTR [ptr1], 0
        inc     pos
        inc     cp_x
        mov     DWORD PTR [d - 4], 0
        cmp     d, limit_VAR
        jae     fin
        cmp     hash, hash_lim
        je      fin

; MY_ALIGN_64
main_loop:
        ; UInt32 delta = *hash++;
        mov     diff_x, [hash]  ; delta
        add     hash, 4
        ; mov     cycPos_VAR, cp_x
       
        inc     cur
        add     d, 4
        mov     m, pos
        sub     m, diff_x;      ; matchPos
        
        ; CLzRef *ptr1 = son + ((size_t)(pos) << 1) - CYC_TO_POS_OFFSET * 2;
        lea     ptr1, [son + 8 * cp_r]
        ; mov     cycSize, cycSize_VAR
        cmp     pos, cycSize
        jb      directMode      ; if (pos < cycSize_VAR)
        
        ; CYC MODE

        cmp     diff_x, cycSize
        jae     fill_empty      ; if (delta >= cycSize_VAR)
        
        xor     t0_x, t0_x
        mov     cycPos_VAR, cp_x
        sub     cp_x, diff_x
        ; jae     prepare_for_tree_loop
        ; add     cp_x, cycSize
        cmovb   t0_x, cycSize
        add     cp_x, t0_x      ; cp_x +=  (cycPos < delta ? cycSize : 0)
        jmp     prepare_for_tree_loop
        
        
directMode:
        cmp     diff_x,  pos
        je      fill_empty      ; if (delta == pos)
        jae     fin_error       ; if (delta >= pos)
        
        mov     cycPos_VAR, cp_x
        mov     cp_x, m
        
prepare_for_tree_loop:
        mov     len0, lenLimit
        mov     hash_VAR, hash
        ; CLzRef *ptr0 = son + ((size_t)(pos) << 1) - CYC_TO_POS_OFFSET * 2 + 1;
        lea     ptr0, [ptr1 + 4]
        ; UInt32 *_distances = ++d;
        mov     distances, d

        neg     len0
        mov     len1, len0

        mov     t0_x, cutValue_VAR
        mov     maxLen, maxLen_VAR
        mov     cutValueCur_VAR, t0_x

MY_ALIGN_32
tree_loop:
        neg     diff
        mov     len, len0
        cmp     len1, len0
        cmovb   len, len1       ; len = (len1 < len0 ? len1 : len0);
        add     diff, cur

        mov     t0_x, [son + cp_r * 8]  ; prefetch
        movzx   t0_x, BYTE PTR [diff + 1 * len]
        lea     cp_r, [son + cp_r * 8]
        cmp     [cur + 1 * len], t0_L
        je      matched_1
        
        jb      left_0

        mov     [ptr1], m
        mov        m, [cp_r + 4]
        lea     ptr1, [cp_r + 4]
        sub     diff, cur ; FIX32
        jmp     next_node

MY_ALIGN_32
left_0:
        mov     [ptr0], m
        mov        m, [cp_r]
        mov     ptr0, cp_r
        sub     diff, cur ; FIX32
        ; jmp     next_node

; ------------ NEXT NODE ------------
; MY_ALIGN_32
next_node:
        mov     cycSize, cycSize_VAR
        dec     cutValueCur_VAR
        je      finish_tree
        
        add     diff_x, pos     ; prev_match = pos + diff
        cmp     m, diff_x
        jae     fin_error       ; if (new_match >= prev_match)
        
        mov     diff_x, pos
        sub     diff_x, m       ; delta = pos - new_match
        cmp     pos, cycSize
        jae     cyc_mode_2      ; if (pos >= cycSize)

        mov     cp_x, m
        test    m, m
        jne     tree_loop       ; if (m != 0)
        
finish_tree:
        ; ptr0 = *ptr1 = kEmptyHashValue;
        mov     DWORD PTR [ptr0], 0
        mov     DWORD PTR [ptr1], 0

        inc     pos
        
        ; _distances[-1] = (UInt32)(d - _distances);
        mov     t0, distances
        mov     t1, d
        sub     t1, t0
        shr     t1_x, 2
        mov     [t0 - 4], t1_x

        cmp     d, limit_VAR
        jae     fin             ; if (d >= limit)
   
        mov     cp_x, cycPos_VAR
        mov     hash, hash_VAR
        mov     hash_lim, size_VAR
        inc     cp_x
        cmp     hash, hash_lim
        jne     main_loop       ; if (hash != size)
        jmp     fin
        

MY_ALIGN_32
cyc_mode_2:
        cmp     diff_x, cycSize
        jae     finish_tree     ; if (delta >= cycSize)

        mov     cp_x, cycPos_VAR
        xor     t0_x, t0_x
        sub     cp_x, diff_x    ; cp_x = cycPos - delta
        cmovb   t0_x, cycSize
        add     cp_x, t0_x      ; cp_x += (cycPos < delta ? cycSize : 0)
        jmp     tree_loop

        
MY_ALIGN_32
matched_1:

        inc     len
        ; cmp     len_x, lenLimit_x
        je      short lenLimit_reach
        movzx   t0_x, BYTE PTR [diff + 1 * len]
        cmp     [cur + 1 * len], t0_L
        jne     mismatch

        
MY_ALIGN_32
match_loop:
        ;  while (++len != lenLimit)  (len[diff] != len[0]) ;

        inc     len
        ; cmp     len_x, lenLimit_x
        je      short lenLimit_reach
        movzx   t0_x, BYTE PTR [diff + 1 * len]
        cmp     BYTE PTR [cur + 1 * len], t0_L
        je      match_loop

mismatch:
        jb      left_2

        mov     [ptr1], m
        mov        m, [cp_r + 4]
        lea     ptr1, [cp_r + 4]
        mov     len1, len

        jmp     max_update
        
MY_ALIGN_32
left_2:
        mov     [ptr0], m
        mov        m, [cp_r]
        mov     ptr0, cp_r
        mov     len0, len

max_update:
        sub     diff, cur       ; restore diff

        cmp     maxLen, len
        jae     next_node
        
        mov     maxLen, len
        add     len, lenLimit
        mov     [d], len_x
        mov     t0_x, diff_x
        not     t0_x
        mov     [d + 4], t0_x
        add     d, 8
       
        jmp     next_node


        
MY_ALIGN_32
lenLimit_reach:

        mov     delta_r, cur
        sub     delta_r, diff
        lea     delta1_r, [delta_r - 1]

        mov     t0_x, [cp_r]
        mov     [ptr1], t0_x
        mov     t0_x, [cp_r + 4]
        mov     [ptr0], t0_x

        mov     [d], lenLimit_x
        mov     [d + 4], delta1_x
        add     d, 8

        ; _distances[-1] = (UInt32)(d - _distances);
        mov     t0, distances
        mov     t1, d
        sub     t1, t0
        shr     t1_x, 2
        mov     [t0 - 4], t1_x

        mov     hash, hash_VAR
        mov     hash_lim, size_VAR

        inc     pos
        mov     cp_x, cycPos_VAR
        inc     cp_x

        mov     d_lim, limit_VAR
        mov     cycSize, cycSize_VAR
        ; if (hash == size || *hash != delta || lenLimit[diff] != lenLimit[0] || d >= limit)
        ;    break;
        cmp     hash, hash_lim
        je      fin
        cmp     d, d_lim
        jae     fin
        cmp     delta_x, [hash]
        jne     main_loop
        movzx   t0_x, BYTE PTR [diff]
        cmp     [cur], t0_L
        jne     main_loop

        ; jmp     main_loop     ; bypass for debug
        
        mov     cycPos_VAR, cp_x
        shl     len, 3          ; cycSize * 8
        sub     diff, cur       ; restore diff
        xor     t0_x, t0_x
        cmp     cp_x, delta_x   ; cmp (cycPos_VAR, delta)
        lea     cp_r, [son + 8 * cp_r]  ; dest
        lea     src, [cp_r + 8 * diff]
        cmovb   t0, len         ; t0 =  (cycPos_VAR < delta ? cycSize * 8 : 0)
        add     src, t0
        add     len, son        ; len = son + cycSize * 8

       
MY_ALIGN_32
long_loop:
        add     hash, 4
        
        ; *(UInt64 *)(void *)ptr = ((const UInt64 *)(const void *)ptr)[diff];
        
        mov     t0, [src]
        add     src, 8
        mov     [cp_r], t0
        add     cp_r, 8
        cmp     src, len
        cmove   src, son       ; if end of (son) buffer is reached, we wrap to begin

        mov     DWORD PTR [d], 2
        mov     [d + 4], lenLimit_x
        mov     [d + 8], delta1_x
        add     d, 12

        inc     cur

        cmp     hash, hash_lim
        je      long_footer
        cmp     delta_x, [hash]
        jne     long_footer
        movzx   t0_x, BYTE PTR [diff + 1 * cur]
        cmp     [cur], t0_L
        jne     long_footer
        cmp     d, d_lim
        jb      long_loop

long_footer:
        sub     cp_r, son
        shr     cp_r, 3
        add     pos, cp_x
        sub     pos, cycPos_VAR
        mov     cycSize, cycSize_VAR
        
        cmp     d, d_lim
        jae     fin
        cmp     hash, hash_lim
        jne     main_loop
        jmp     fin



fin_error:
        xor     d, d
        
fin:
        mov     RSP, Old_RSP
        mov     t0, [r4 + posRes_OFFS]
        mov     [t0], pos
        mov     r0, d

MY_POP_PRESERVED_ABI_REGS
MY_ENDP

ifdef Z7_LZ_FIND_OPT_ASM_USE_SEGMENT
_TEXT$LZFINDOPT ENDS
endif

end
