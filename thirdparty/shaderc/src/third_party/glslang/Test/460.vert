#version 460 core

int i;
; // extraneous semicolon okay
float f;;;

void main()
{
    bool b1;
    b1 = anyInvocation(b1);
    b1 = allInvocations(b1);
    b1 = allInvocationsEqual(b1);
}
;
;
