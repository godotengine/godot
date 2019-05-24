
void MyFunc() { }

;;;
;
; ; ;     // HLSL allows stray global scope semicolons.

void MyFunc2() {;;;};

struct PS_OUTPUT { float4 color : SV_Target0; };;;;;

;PS_OUTPUT main()
{
    PS_OUTPUT ps_output;;;
    ;
    ps_output.color = 1.0;
    return ps_output;
};

