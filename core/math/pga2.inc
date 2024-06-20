// Generate from https://github.com/xiongyaohua/PGA_codegen

#define GP_0_0(r, a, b, op) \
    r.e op a.e*b.e;

#define GP_0_1(r, a, b, op) \
    r.e0 op a.e*b.e0;\
    r.e1 op a.e*b.e1;\
    r.e2 op a.e*b.e2;

#define GP_0_2(r, a, b, op) \
    r.e01 op a.e*b.e01;\
    r.e02 op a.e*b.e02;\
    r.e12 op a.e*b.e12;

#define GP_0_3(r, a, b, op) \
    r.e012 op a.e*b.e012;

#define GP_1_0(r, a, b, op) \
    r.e0 op a.e0*b.e;\
    r.e1 op a.e1*b.e;\
    r.e2 op a.e2*b.e;

#define GP_1_1(r, a, b, op) \
    r.e op a.e1*b.e1 + a.e2*b.e2;\
    r.e01 op a.e0*b.e1 - a.e1*b.e0;\
    r.e02 op a.e0*b.e2 - a.e2*b.e0;\
    r.e12 op a.e1*b.e2 - a.e2*b.e1;

#define GP_1_2(r, a, b, op) \
    r.e0 op -a.e1*b.e01 - a.e2*b.e02;\
    r.e1 op -a.e2*b.e12;\
    r.e2 op a.e1*b.e12;\
    r.e012 op a.e0*b.e12 - a.e1*b.e02 + a.e2*b.e01;

#define GP_1_3(r, a, b, op) \
    r.e01 op a.e2*b.e012;\
    r.e02 op -a.e1*b.e012;

#define GP_2_0(r, a, b, op) \
    r.e01 op a.e01*b.e;\
    r.e02 op a.e02*b.e;\
    r.e12 op a.e12*b.e;

#define GP_2_1(r, a, b, op) \
    r.e0 op a.e01*b.e1 + a.e02*b.e2;\
    r.e1 op a.e12*b.e2;\
    r.e2 op -a.e12*b.e1;\
    r.e012 op a.e01*b.e2 - a.e02*b.e1 + a.e12*b.e0;

#define GP_2_2(r, a, b, op) \
    r.e op -a.e12*b.e12;\
    r.e01 op -a.e02*b.e12 + a.e12*b.e02;\
    r.e02 op a.e01*b.e12 - a.e12*b.e01;

#define GP_2_3(r, a, b, op) \
    r.e0 op -a.e12*b.e012;

#define GP_3_0(r, a, b, op) \
    r.e012 op a.e012*b.e;

#define GP_3_1(r, a, b, op) \
    r.e01 op a.e012*b.e2;\
    r.e02 op -a.e012*b.e1;

#define GP_3_2(r, a, b, op) \
    r.e0 op -a.e012*b.e12;

#define GP_3_3(r, a, b, op) \


#define GP_FULL(r, a, b) \
    r.e = a.e*b.e + a.e1*b.e1 - a.e12*b.e12 + a.e2*b.e2;\
    r.e0 = a.e*b.e0 + a.e0*b.e + a.e01*b.e1 - a.e012*b.e12 + a.e02*b.e2 - a.e1*b.e01 - a.e12*b.e012 - a.e2*b.e02;\
    r.e1 = a.e*b.e1 + a.e1*b.e + a.e12*b.e2 - a.e2*b.e12;\
    r.e2 = a.e*b.e2 + a.e1*b.e12 - a.e12*b.e1 + a.e2*b.e;\
    r.e01 = a.e*b.e01 + a.e0*b.e1 + a.e01*b.e + a.e012*b.e2 - a.e02*b.e12 - a.e1*b.e0 + a.e12*b.e02 + a.e2*b.e012;\
    r.e02 = a.e*b.e02 + a.e0*b.e2 + a.e01*b.e12 - a.e012*b.e1 + a.e02*b.e - a.e1*b.e012 - a.e12*b.e01 - a.e2*b.e0;\
    r.e12 = a.e*b.e12 + a.e1*b.e2 + a.e12*b.e - a.e2*b.e1;\
    r.e012 = a.e*b.e012 + a.e0*b.e12 + a.e01*b.e2 + a.e012*b.e - a.e02*b.e1 - a.e1*b.e02 + a.e12*b.e0 + a.e2*b.e01;

#define OP_0_0(r, a, b, op) \
    r.e op a.e*b.e;

#define OP_0_1(r, a, b, op) \
    r.e0 op a.e*b.e0;\
    r.e1 op a.e*b.e1;\
    r.e2 op a.e*b.e2;

#define OP_0_2(r, a, b, op) \
    r.e01 op a.e*b.e01;\
    r.e02 op a.e*b.e02;\
    r.e12 op a.e*b.e12;

#define OP_0_3(r, a, b, op) \
    r.e012 op a.e*b.e012;

#define OP_1_0(r, a, b, op) \
    r.e0 op a.e0*b.e;\
    r.e1 op a.e1*b.e;\
    r.e2 op a.e2*b.e;

#define OP_1_1(r, a, b, op) \
    r.e01 op a.e0*b.e1 - a.e1*b.e0;\
    r.e02 op a.e0*b.e2 - a.e2*b.e0;\
    r.e12 op a.e1*b.e2 - a.e2*b.e1;

#define OP_1_2(r, a, b, op) \
    r.e012 op a.e0*b.e12 - a.e1*b.e02 + a.e2*b.e01;

#define OP_1_3(r, a, b, op) \


#define OP_2_0(r, a, b, op) \
    r.e01 op a.e01*b.e;\
    r.e02 op a.e02*b.e;\
    r.e12 op a.e12*b.e;

#define OP_2_1(r, a, b, op) \
    r.e012 op a.e01*b.e2 - a.e02*b.e1 + a.e12*b.e0;

#define OP_2_2(r, a, b, op) \


#define OP_2_3(r, a, b, op) \


#define OP_3_0(r, a, b, op) \
    r.e012 op a.e012*b.e;

#define OP_3_1(r, a, b, op) \


#define OP_3_2(r, a, b, op) \


#define OP_3_3(r, a, b, op) \


#define OP_FULL(r, a, b) \
    r.e = a.e*b.e;\
    r.e0 = a.e*b.e0 + a.e0*b.e;\
    r.e1 = a.e*b.e1 + a.e1*b.e;\
    r.e2 = a.e*b.e2 + a.e2*b.e;\
    r.e01 = a.e*b.e01 + a.e0*b.e1 + a.e01*b.e - a.e1*b.e0;\
    r.e02 = a.e*b.e02 + a.e0*b.e2 + a.e02*b.e - a.e2*b.e0;\
    r.e12 = a.e*b.e12 + a.e1*b.e2 + a.e12*b.e - a.e2*b.e1;\
    r.e012 = a.e*b.e012 + a.e0*b.e12 + a.e01*b.e2 + a.e012*b.e - a.e02*b.e1 - a.e1*b.e02 + a.e12*b.e0 + a.e2*b.e01;

#define IP_0_0(r, a, b, op) \
    r.e op a.e*b.e;

#define IP_0_1(r, a, b, op) \
    r.e0 op a.e*b.e0;\
    r.e1 op a.e*b.e1;\
    r.e2 op a.e*b.e2;

#define IP_0_2(r, a, b, op) \
    r.e01 op a.e*b.e01;\
    r.e02 op a.e*b.e02;\
    r.e12 op a.e*b.e12;

#define IP_0_3(r, a, b, op) \
    r.e012 op a.e*b.e012;

#define IP_1_0(r, a, b, op) \
    r.e0 op a.e0*b.e;\
    r.e1 op a.e1*b.e;\
    r.e2 op a.e2*b.e;

#define IP_1_1(r, a, b, op) \
    r.e op a.e1*b.e1 + a.e2*b.e2;

#define IP_1_2(r, a, b, op) \
    r.e0 op -a.e1*b.e01 - a.e2*b.e02;\
    r.e1 op -a.e2*b.e12;\
    r.e2 op a.e1*b.e12;

#define IP_1_3(r, a, b, op) \
    r.e01 op a.e2*b.e012;\
    r.e02 op -a.e1*b.e012;

#define IP_2_0(r, a, b, op) \
    r.e01 op a.e01*b.e;\
    r.e02 op a.e02*b.e;\
    r.e12 op a.e12*b.e;

#define IP_2_1(r, a, b, op) \
    r.e0 op a.e01*b.e1 + a.e02*b.e2;\
    r.e1 op a.e12*b.e2;\
    r.e2 op -a.e12*b.e1;

#define IP_2_2(r, a, b, op) \
    r.e op -a.e12*b.e12;

#define IP_2_3(r, a, b, op) \
    r.e0 op -a.e12*b.e012;

#define IP_3_0(r, a, b, op) \
    r.e012 op a.e012*b.e;

#define IP_3_1(r, a, b, op) \
    r.e01 op a.e012*b.e2;\
    r.e02 op -a.e012*b.e1;

#define IP_3_2(r, a, b, op) \
    r.e0 op -a.e012*b.e12;

#define IP_3_3(r, a, b, op) \


#define IP_FULL(r, a, b) \
    r.e = a.e*b.e + a.e1*b.e1 - a.e12*b.e12 + a.e2*b.e2;\
    r.e0 = a.e*b.e0 + a.e0*b.e + a.e01*b.e1 - a.e012*b.e12 + a.e02*b.e2 - a.e1*b.e01 - a.e12*b.e012 - a.e2*b.e02;\
    r.e1 = a.e*b.e1 + a.e1*b.e + a.e12*b.e2 - a.e2*b.e12;\
    r.e2 = a.e*b.e2 + a.e1*b.e12 - a.e12*b.e1 + a.e2*b.e;\
    r.e01 = a.e*b.e01 + a.e01*b.e + a.e012*b.e2 + a.e2*b.e012;\
    r.e02 = a.e*b.e02 - a.e012*b.e1 + a.e02*b.e - a.e1*b.e012;\
    r.e12 = a.e*b.e12 + a.e12*b.e;\
    r.e012 = a.e*b.e012 + a.e012*b.e;

#define RP_0_0(r, a, b, op) \


#define RP_0_1(r, a, b, op) \


#define RP_0_2(r, a, b, op) \


#define RP_0_3(r, a, b, op) \
    r.e op a.e*b.e012;

#define RP_1_0(r, a, b, op) \


#define RP_1_1(r, a, b, op) \


#define RP_1_2(r, a, b, op) \
    r.e op a.e0*b.e12 - a.e1*b.e02 + a.e2*b.e01;

#define RP_1_3(r, a, b, op) \
    r.e0 op a.e0*b.e012;\
    r.e1 op a.e1*b.e012;\
    r.e2 op a.e2*b.e012;

#define RP_2_0(r, a, b, op) \


#define RP_2_1(r, a, b, op) \
    r.e op a.e01*b.e2 - a.e02*b.e1 + a.e12*b.e0;

#define RP_2_2(r, a, b, op) \
    r.e0 op a.e01*b.e02 - a.e02*b.e01;\
    r.e1 op a.e01*b.e12 - a.e12*b.e01;\
    r.e2 op a.e02*b.e12 - a.e12*b.e02;

#define RP_2_3(r, a, b, op) \
    r.e01 op a.e01*b.e012;\
    r.e02 op a.e02*b.e012;\
    r.e12 op a.e12*b.e012;

#define RP_3_0(r, a, b, op) \
    r.e op a.e012*b.e;

#define RP_3_1(r, a, b, op) \
    r.e0 op a.e012*b.e0;\
    r.e1 op a.e012*b.e1;\
    r.e2 op a.e012*b.e2;

#define RP_3_2(r, a, b, op) \
    r.e01 op a.e012*b.e01;\
    r.e02 op a.e012*b.e02;\
    r.e12 op a.e012*b.e12;

#define RP_3_3(r, a, b, op) \
    r.e012 op a.e012*b.e012;

#define RP_FULL(r, a, b) \
    r.e = a.e*b.e012 + a.e0*b.e12 + a.e01*b.e2 + a.e012*b.e - a.e02*b.e1 - a.e1*b.e02 + a.e12*b.e0 + a.e2*b.e01;\
    r.e0 = a.e0*b.e012 + a.e01*b.e02 + a.e012*b.e0 - a.e02*b.e01;\
    r.e1 = a.e01*b.e12 + a.e012*b.e1 + a.e1*b.e012 - a.e12*b.e01;\
    r.e2 = a.e012*b.e2 + a.e02*b.e12 - a.e12*b.e02 + a.e2*b.e012;\
    r.e01 = a.e01*b.e012 + a.e012*b.e01;\
    r.e02 = a.e012*b.e02 + a.e02*b.e012;\
    r.e12 = a.e012*b.e12 + a.e12*b.e012;\
    r.e012 = a.e012*b.e012;

#define CP_0_0(r, a, b, op) \


#define CP_0_1(r, a, b, op) \


#define CP_0_2(r, a, b, op) \


#define CP_0_3(r, a, b, op) \


#define CP_1_0(r, a, b, op) \


#define CP_1_1(r, a, b, op) \
    r.e01 op a.e0*b.e1 - a.e1*b.e0;\
    r.e02 op a.e0*b.e2 - a.e2*b.e0;\
    r.e12 op a.e1*b.e2 - a.e2*b.e1;

#define CP_1_2(r, a, b, op) \
    r.e0 op -a.e1*b.e01 - a.e2*b.e02;\
    r.e1 op -a.e2*b.e12;\
    r.e2 op a.e1*b.e12;

#define CP_1_3(r, a, b, op) \


#define CP_2_0(r, a, b, op) \


#define CP_2_1(r, a, b, op) \
    r.e0 op a.e01*b.e1 + a.e02*b.e2;\
    r.e1 op a.e12*b.e2;\
    r.e2 op -a.e12*b.e1;

#define CP_2_2(r, a, b, op) \
    r.e01 op -a.e02*b.e12 + a.e12*b.e02;\
    r.e02 op a.e01*b.e12 - a.e12*b.e01;

#define CP_2_3(r, a, b, op) \


#define CP_3_0(r, a, b, op) \


#define CP_3_1(r, a, b, op) \


#define CP_3_2(r, a, b, op) \


#define CP_3_3(r, a, b, op) \


#define CP_FULL(r, a, b) \
    r.e0 = a.e01*b.e1 + a.e02*b.e2 - a.e1*b.e01 - a.e2*b.e02;\
    r.e1 = a.e12*b.e2 - a.e2*b.e12;\
    r.e2 = a.e1*b.e12 - a.e12*b.e1;\
    r.e01 = a.e0*b.e1 - a.e02*b.e12 - a.e1*b.e0 + a.e12*b.e02;\
    r.e02 = a.e0*b.e2 + a.e01*b.e12 - a.e12*b.e01 - a.e2*b.e0;\
    r.e12 = a.e1*b.e2 - a.e2*b.e1;

#define SP_FULL(r, a, b) \
    r = a.e*b.e + a.e1*b.e1 - a.e12*b.e12 + a.e2*b.e2;

#define REVERSE_0(r, a) \
    r.e = a.e;

#define REVERSE_1(r, a) \
    r.e0 = a.e0;\
    r.e1 = a.e1;\
    r.e2 = a.e2;

#define REVERSE_2(r, a) \
    r.e01 = -a.e01;\
    r.e02 = -a.e02;\
    r.e12 = -a.e12;

#define REVERSE_3(r, a) \
    r.e012 = -a.e012;

#define DUAL_0(r, a) \
    r.e012 = a.e;

#define DUAL_1(r, a) \
    r.e01 = a.e2;\
    r.e02 = -a.e1;\
    r.e12 = a.e0;

#define DUAL_2(r, a) \
    r.e0 = a.e12;\
    r.e1 = -a.e02;\
    r.e2 = a.e01;

#define DUAL_3(r, a) \
    r.e = a.e012;

#define GRADE_0(r, a) \
    r.e = a.e;

#define GRADE_1(r, a) \
    r.e0 = a.e0;\
    r.e1 = a.e1;\
    r.e2 = a.e2;

#define GRADE_2(r, a) \
    r.e01 = a.e01;\
    r.e02 = a.e02;\
    r.e12 = a.e12;

#define GRADE_3(r, a) \
    r.e012 = a.e012;

#define ADD_0(r, a, b, op) \
    r.e op a.e + b.e;

#define ADD_1(r, a, b, op) \
    r.e0 op a.e0 + b.e0;\
    r.e1 op a.e1 + b.e1;\
    r.e2 op a.e2 + b.e2;

#define ADD_2(r, a, b, op) \
    r.e01 op a.e01 + b.e01;\
    r.e02 op a.e02 + b.e02;\
    r.e12 op a.e12 + b.e12;

#define ADD_3(r, a, b, op) \
    r.e012 op a.e012 + b.e012;

#define SUB_0(r, a, b, op) \
    r.e op a.e - b.e;

#define SUB_1(r, a, b, op) \
    r.e0 op a.e0 - b.e0;\
    r.e1 op a.e1 - b.e1;\
    r.e2 op a.e2 - b.e2;

#define SUB_2(r, a, b, op) \
    r.e01 op a.e01 - b.e01;\
    r.e02 op a.e02 - b.e02;\
    r.e12 op a.e12 - b.e12;

#define SUB_3(r, a, b, op) \
    r.e012 op a.e012 - b.e012;

#define MUL_0(r, a, b, op) \
    r.e op a.e*b;

#define MUL_1(r, a, b, op) \
    r.e0 op a.e0*b;\
    r.e1 op a.e1*b;\
    r.e2 op a.e2*b;

#define MUL_2(r, a, b, op) \
    r.e01 op a.e01*b;\
    r.e02 op a.e02*b;\
    r.e12 op a.e12*b;

#define MUL_3(r, a, b, op) \
    r.e012 op a.e012*b;

#define DIV_0(r, a, b, op) \
    r.e op a.e/b;

#define DIV_1(r, a, b, op) \
    r.e0 op a.e0/b;\
    r.e1 op a.e1/b;\
    r.e2 op a.e2/b;

#define DIV_2(r, a, b, op) \
    r.e01 op a.e01/b;\
    r.e02 op a.e02/b;\
    r.e12 op a.e12/b;

#define DIV_3(r, a, b, op) \
    r.e012 op a.e012/b;