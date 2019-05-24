#version 300 es
#define ON1
#define ON2
float sum = 0.0;

void main()
{
#if defined(ON1) && (defined(OFF) || defined(ON2))
//yes
    sum += 1.0;
#endif

#if !defined(ON1) || (defined(OFF) || (!defined(OFF2) && defined(ON2)))
//yes
    sum += 20.0;
#endif

#if defined(ON1) && (defined(OFF) || !defined(ON2))
//no
    sum += 0.1;
#endif

#if !defined(ON1) || (defined(OFF) || !defined(OFF2) && !defined(ON2))
//no
    sum += 0.2;
#endif

#if !defined(ON1) || !defined(OFF) || defined(ON2) && defined(OFF2)
//yes
    sum += 300.0;
#endif

#if (!defined(ON1) || !defined(OFF) || defined(ON2)) && defined(OFF2)
//no
    sum += 0.4;
#endif

// sum should be 321.0
    gl_Position = vec4(sum);
}

#define ADD(a, b) a + b + ((a) + ((b)));

float foo()
{
    return ADD(gl_Position.xyxwx, 3.0)  // ERROR, should be this line number
    return ADD(gl_Position.y, 3.0)
}

#define BIG aonetuhanoethuanoenaoethu snaoetuhs onethausoentuas hnoethaueohnatuoeh santuoehsantouhe snathoensuta hsnoethuasntoe hsnuathoesnuathoenstuh nsoethantseuh toae ua \
    antoeh uantheount oentahoent uahnsoethasnutoehansteuo santhu sneoathu snoethasnut oesanthoesna thusenotha nsthasunoeth ausntehsunathoensuathoesnta uhnsoetha usntoeh uanhs unosethu \
    antoehunatoehu natoehua oentha neotuhan toehu natoehu ntahoe nutah eu natoheunathoen uasoenuasoent asntoehsan tuosnthnu aohenuath eontha untoh eunth unth anth unth nth nth nt \
    a ntoehanu tunth nsont uhansoethausn oehsanthnt heauo eanthuo sh nahnoethansu tohe sanuthoe snathuoesntha snuothe anthusonehtasuntoeh asnuthonsa teauhntoeha onetuha nth \
    anoethuan toentauh noethauntohe anuthoe nathu noethaun oethanuthoe nathuoe ntahu enotha unetha ntuhenaothu enotahun eotha ntoehu aoehuntha enotuh aonethau noethu anoethuna toheua \
    ontehanutoe hnuathoena aoteha aonetuha 

// identical
#define BIG aonetuhanoethuanoenaoethu snaoetuhs onethausoentuas hnoethaueohnatuoeh santuoehsantouhe snathoensuta hsnoethuasntoe hsnuathoesnuathoenstuh nsoethantseuh toae ua \
    antoeh uantheount oentahoent uahnsoethasnutoehansteuo santhu sneoathu snoethasnut oesanthoesna thusenotha nsthasunoeth ausntehsunathoensuathoesnta uhnsoetha usntoeh uanhs unosethu \
    antoehunatoehu natoehua oentha neotuhan toehu natoehu ntahoe nutah eu natoheunathoen uasoenuasoent asntoehsan tuosnthnu aohenuath eontha untoh eunth unth anth unth nth nth nt \
    a ntoehanu tunth nsont uhansoethausn oehsanthnt heauo eanthuo sh nahnoethansu tohe sanuthoe snathuoesntha snuothe anthusonehtasuntoeh asnuthonsa teauhntoeha onetuha nth \
    anoethuan toentauh noethauntohe anuthoe nathu noethaun oethanuthoe nathuoe ntahu enotha unetha ntuhenaothu enotahun eotha ntoehu aoehuntha enotuh aonethau noethu anoethuna toheua \
    ontehanutoe hnuathoena aoteha aonetuha 

// ERROR, one character different
#define BIG aonetuhanoethuanoenaoethu snaoetuhs onethausoentuas hnoethaueohnatuoeh santuoehsantouhe snathoensuta hsnoethuasntoe hsnuathoesnuathoenstuh nsoethantseuh toae ua \
    antoeh uantheount oentahoent uahnsoethasnutoehansteuo santhu sneoathu snoethasnut oesanthoesna thusenotha nsthasunoeth ausntehsunathoensuathoesnta uhnsoetha usntoeh uanhs unosethu \
    antoehunatoehu natoehua oentha neotuhan toehu natoehu ntahoe nutah eu natoheunathoen uasoenuasoent asntoehsan tuosnthnu aohenuath eontha untoh eunth unth anth unth nth nth nt \
    a ntoehanu tunth nsont uhansoethasn oehsanthnt heauo eanthuo sh nahnoethansu tohe sanuthoe snathuoesntha snuothe anthusonehtasuntoeh asnuthonsa teauhntoeha onetuha nth \
    anoethuan toentauh noethauntohe anuthoe nathu noethaun oethanuthoe nathuoe ntahu enotha unetha ntuhenaothu enotahun eotha ntoehu aoehuntha enotuh aonethau noethu anoethuna toheua \
    ontehanutoe hnuathoena aoteha aonetuha 

#define BIGARGS1(aonthanotehu, bonthanotehu, conthanotehu, donthanotehu, eonthanotehu, fonthanotehu, gonthanotehu, honthanotehu, ionthanotehu, jonthanotehu, konthanotehu) jonthanotehu
#define BIGARGS2(aonthanotehu, bonthanotehu, conthanotehu, donthanotehu, eonthanotehu, fonthanotehu, gonthanotehu, honthanotehu, ionthanotehu, jonthanotehu, konthanotehu) jonthanotehu
#define BIGARGS3(aonthanotehu, bonthanotehu, conthanotehu, donthanotehu, eonthanotehu, fonthanotehu, gonthanotehu, honthanotehu, ionthanotehu, jonthanotehu, konthanotehu) jonthanotehu
#define BIGARGS4(aonthanotehu, bonthanotehu, conthanotehu, donthanotehu, eonthanotehu, fonthanotehu, gonthanotehu, honthanotehu, ionthanotehu, jonthanotehu, konthanotehu) jonthanotehu


#define foobar(a, b) a + b

#if foobar(1.1, 2.2)
#error good macro
#else
#error bad macro
#endif

#if foobar(1
;
#
#
#endif
#if foobar(1,
;
#
#
#endif
float c = foobar(1.1, 2.2
       );
#if foobar(1.1, 2.2
)
#if foobar(1.1, 2.2
#endif
#endif

#define VAL0 0
#define VAL1 1

#if UNDEF
#error bad 0
#else
#error good 0
#endif

#if VAL1 || UNDEF
#error good 1
#else
#error bad 1
#endif

#if VAL1 && UNDEF // UNDEF ERROR
#endif

#if VAL0 || UNDEF // UNDEF ERROR
#endif

#if VAL0 && UNDEF
#error bad 2
#else
#error good 1
#endif

#if VAL1 || (VAL1 && UNDEF)
#error good 3
#else
#error bad 3
#endif

#if VAL1 && (VAL1 || UNDEF)
#error good 4
#else
#error bad 4
#endif

#if VAL1 < VAL1 || VAL1 > VAL1 || UNDEF // UNDEF ERROR
#endif

#if VAL1 < VAL1 || VAL1 > VAL1 && UNDEF
#endif

#if VAL1 || UNDEF && UNDEF2
#endif

#if VAL0 || UNDEF && UNDEF2  // UNDEF ERROR
#endif

#if (VAL1 || UNDEF) && UNDEF2 // UNDEF2 ERROR
#endif

#if (VAL0 && UNDEF) || UNDEF2 // UNDEF2 ERROR
#endif

#line 3000
#error line of this error should be 3000
    
#define __LINE__ 30
#define __FILE__
#define __VERSION__
#define GL_SOME_EXTENSION
#undef __LINE__
#undef __FILE__
#undef __VERSION__
#undef GL_SOME_EXTENSION

#line 4000
#line 200 % 0     // ERROR, div by 0
#if __LINE__ / 0  // ERROR, div by 0
#endif

#if 7%            // ERROR incomplete expression

#line 10000
#if 0
// ERROR, EOF