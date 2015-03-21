#include "rtlfixer.h"
#include "ustring.h"
#include "fribidi/fribidi.h"

String RTLFixer::getFixedText(const String& text)
{

    FriBidiChar *us, *out_us;
    char *outstr;

    int size;
    String line = "";
    String vtext = "";
    FriBidiParType base;
    for(int i = 0;i<text.length();i++)
    {
        CharType current= text[i];
        if(current!='\n')
        {
            line+=current;
        }else
        {

            size=line.utf8().length();
            FriBidiStrIndex *ltov =(  FriBidiStrIndex *) memalloc(size*sizeof(FriBidiStrIndex));
            FriBidiStrIndex  *vtol=(  FriBidiStrIndex *) memalloc(size*sizeof(FriBidiStrIndex));
            FriBidiLevel *levels=(  FriBidiLevel *) memalloc(size*sizeof(FriBidiLevel));
            us=(FriBidiChar *) memalloc(size*sizeof(FriBidiChar));


            //base = FRIBIDI_TYPE_N;
            base =64;
            size = fribidi_charset_to_unicode(FRIBIDI_CHAR_SET_UTF8, line.utf8().get_data(), size, us);
            out_us=(FriBidiChar *) memalloc((size+1)*sizeof(FriBidiChar));
            outstr=(char *) memalloc(size*sizeof(char)*4);
            fribidi_log2vis(us, size, &base, out_us, ltov, vtol, levels);
            fribidi_unicode_to_charset(FRIBIDI_CHAR_SET_UTF8, out_us, size, outstr);


            vtext += String::utf8(outstr);
            vtext +='\n';
            memfree(out_us);
            memfree(outstr);
            memfree(us);
            memfree(levels);
            memfree(vtol);
            memfree(ltov);

            line="";
        }


    }

    if(line.length()>0)
    {
        size=line.utf8().length();
        FriBidiStrIndex *ltov =(  FriBidiStrIndex *) memalloc(size*sizeof(FriBidiStrIndex));
        FriBidiStrIndex  *vtol=(  FriBidiStrIndex *) memalloc(size*sizeof(FriBidiStrIndex));
        FriBidiLevel *levels=(  FriBidiLevel *) memalloc(size*sizeof(FriBidiLevel));
        us=(FriBidiChar *) memalloc(size*sizeof(FriBidiChar));


        //base = FRIBIDI_TYPE_N;
        base =64;
        size = fribidi_charset_to_unicode(FRIBIDI_CHAR_SET_UTF8, line.utf8().get_data(), size, us);
        out_us=(FriBidiChar *) memalloc((size+1)*sizeof(FriBidiChar));
        outstr=(char *) memalloc(size*sizeof(char)*4);
        fribidi_log2vis(us, size, &base, out_us, ltov, vtol, levels);
        fribidi_unicode_to_charset(FRIBIDI_CHAR_SET_UTF8, out_us, size, outstr);


        vtext += String::utf8(outstr);

        memfree(out_us);
        memfree(outstr);
        memfree(us);
        memfree(levels);
        memfree(vtol);
        memfree(ltov);
      }
    return vtext;

}
