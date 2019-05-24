#version 110

#define ON

float sum = 0.0;

void main()
{

#ifdef ON
//yes
sum += 1.0;

    #ifdef OFF
    //no
    sum += 20.0;
    #endif

    #if defined(ON)
    //yes
    sum += 300.0;
    #endif

#endif


#if defined(OFF)
//no
sum += 4000.0;

#if !defined(ON)
//no
sum += 50000.0;
#endif

    //no
    sum += 0.1;
    #ifdef ON
        //no
        sum += 0.2;
    #endif

    //no
    sum += 0.01;
    #ifdef ON
        //no
        sum += 0.02;
    #else
        //no
        sum += 0.03;
    #endif

//no
    sum + 0.3;

#endif


#if !defined(OFF)
//yes
sum += 600000.0;

    #if defined(ON) && !defined(OFF)
    //yes
    sum += 80000000.0;

        #if defined(OFF) || defined(ON)
        //yes
        sum += 900000000.0;

            #if defined(ON) && defined(OFF)
                //no
                sum += 0.7;
            #elif !defined(OFF)
                //yes
                sum += 7000000.0;
            #endif

        #endif

    #endif

#endif

// sum should be 987600301.0
    gl_Position = vec4(sum);
}

#define  A 1
#define  C 0
#define  E 0
#define  F 1
#if A
    #if C
        #if E
            int selected4 = 1;
        #elif F
            int selected4 = 2;
        #else
            int selected4 = 3;
        #endif
    #endif
    int selected4 = 4;
#endif

#define  ZA 1
#define  ZC 1
#define  ZE 0
#define  ZF 1
#if ZA
    #if ZC
        #if ZE
            int selected2 = 1;
        #elif ZF
            int selected2 = 2;
        #else
            int selected2 = 3;
        #endif
    #endif
#endif

#define  AZA 1
#define  AZC 1
#define  AZE 0
#define  AZF 0
#if AZA
    #if AZC
        #if AZE
            int selected3 = 1;
        #elif AZF
            int selected3 = 2;
        #else
            int selected3 = 3;
        #endif
    #endif
#endif

// ERROR cases...

#if 0
int;
#else
int;
#elif 1
int;
#endif

#if 0
int;
#else
int;
#else
int;
#endif

#if 0
    #if 0
    int;
    #else
    int;
    #elif 1
    int;
    #endif

    #if 0
    int;
    #else
    int;
    #else
    int;
    #endif
#endif

#define FUNC(a,b)		a+b
void foo985(){	FUNC( (((2))), ((3),4)); }
// needs to be last test in file
void foo987(){	FUNC(((); }  // ERROR, EOF in argument
