#include "trex.h"
#include <stdio.h>
#include <string.h>

#ifdef _UNICODE
#define trex_sprintf swprintf
#else
#define trex_sprintf sprintf
#endif

int main(int argc, char* argv[])
{
	const TRexChar *begin,*end;
	TRexChar sTemp[200];
	const TRexChar *error = NULL;
	TRex *x = trex_compile(_TREXC("(x{1,5})xx"),&error);
	if(x) {
		trex_sprintf(sTemp,_TREXC("xxxxxxx"));
		if(trex_search(x,sTemp,&begin,&end))
		{
			int i,n = trex_getsubexpcount(x);
			TRexMatch match;
			for(i = 0; i < n; i++)
			{
				TRexChar t[200];
				trex_getsubexp(x,i,&match);
				trex_sprintf(t,_TREXC("[%%d]%%.%ds\n"),match.len);
				trex_printf(t,i,match.begin);
			}
			trex_printf(_TREXC("match! %d sub matches\n"),trex_getsubexpcount(x));
		}
		else {
			trex_printf(_TREXC("no match!\n"));
		}
		trex_free(x);
	}
	else {
		trex_printf(_TREXC("compilation error [%s]!\n"),error?error:_TREXC("undefined"));
	}
	return 0;
}
