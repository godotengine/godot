include ksamd64.inc

text        SEGMENT EXECUTE

public      __chkstk

__chkstk:
    sub         rsp,010h
    mov         QWORD PTR [rsp],r10
    mov         QWORD PTR [rsp+08h],r11
    xor         r11,r11
    lea         r10,[rsp+018h]
    sub         r10,rax
    cmovb       r10,r11
    mov         r11,QWORD PTR gs:[TeStackLimit]
    cmp         r10,r11
    jae         chkstk_finish
    and         r10w,0f000h
chkstk_loop:
    lea         r11,[r11-PAGE_SIZE]
    mov         BYTE PTR [r11],0h
    cmp         r10,r11
    jne         chkstk_loop
chkstk_finish:
    mov         r10,QWORD PTR [rsp]
    mov         r11,QWORD PTR [rsp+08h]
    add         rsp,010h
    ret
end
