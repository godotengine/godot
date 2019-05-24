#version 110

#define ON

float sum = 0.0;

void main()
{

#ifdef ON
//yes
sum += 1.0;
#endif

#ifdef OFF
    //no
    sum += 20.0;
#endif

    #if defined(ON)
    //yes
    sum += 300.0;
    #endif

    #if defined(OFF)
    //no
    sum += 4000.0;
    #endif

		  #if !defined(ON)
		//no
		sum += 50000.0;
		#endif

  	#if !defined(OFF)
		//yes
		sum += 600000.0;
		#endif

    #if defined (ON) && defined               (OFF)         
//no
sum += 7000000.0;
    #endif

#if        defined   (  ON         ) && !        defined(OFF)
//yes
sum += 80000000.0;
#endif

#if defined(OFF) || defined(ON)
//yes
sum += 900000000.0;
#endif

// sum should be 980600301.0
    gl_Position = vec4(sum);
}

#define FUNC(a,b)		a+b
// needs to be last test in file due to syntax error
void foo986(){	FUNC( (((2)))), 4); }  // ERROR, too few arguments )
