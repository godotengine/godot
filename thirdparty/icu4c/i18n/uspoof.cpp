// © 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
***************************************************************************
* Copyright (C) 2008-2015, International Business Machines Corporation
* and others. All Rights Reserved.
***************************************************************************
*   file name:  uspoof.cpp
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2008Feb13
*   created by: Andy Heninger
*
*   Unicode Spoof Detection
*/
#include "unicode/ubidi.h"
#include "unicode/utypes.h"
#include "unicode/normalizer2.h"
#include "unicode/uspoof.h"
#include "unicode/ustring.h"
#include "unicode/utf16.h"
#include "cmemory.h"
#include "cstring.h"
#include "mutex.h"
#include "scriptset.h"
#include "uassert.h"
#include "ucln_in.h"
#include "uspoof_impl.h"
#include "umutex.h"


#if !UCONFIG_NO_NORMALIZATION

U_NAMESPACE_USE


//
// Static Objects used by the spoof impl, their thread safe initialization and their cleanup.
//
static UnicodeSet *gInclusionSet = nullptr;
static UnicodeSet *gRecommendedSet = nullptr;
static const Normalizer2 *gNfdNormalizer = nullptr;
static UInitOnce gSpoofInitStaticsOnce {};

namespace {

UBool U_CALLCONV
uspoof_cleanup() {
    delete gInclusionSet;
    gInclusionSet = nullptr;
    delete gRecommendedSet;
    gRecommendedSet = nullptr;
    gNfdNormalizer = nullptr;
    gSpoofInitStaticsOnce.reset();
    return true;
}

void U_CALLCONV initializeStatics(UErrorCode &status) {
    static const char16_t *inclusionPat =
        u"['\\-.\\:\\u00B7\\u0375\\u058A\\u05F3\\u05F4\\u06FD\\u06FE\\u0F0B\\u2010"
        u"\\u2019\\u2027\\u30A0\\u30FB]";
    gInclusionSet = new UnicodeSet(UnicodeString(inclusionPat), status);
    if (gInclusionSet == nullptr) {
        status = U_MEMORY_ALLOCATION_ERROR;
        return;
    }
    gInclusionSet->freeze();

    // Note: data from IdentifierStatus.txt & IdentifierType.txt
    // There is tooling to generate this constant in the unicodetools project:
    //      org.unicode.text.tools.RecommendedSetGenerator
    // It will print the Java and C++ code to the console for easy copy-paste into this file.
    static const char16_t *recommendedPat =
        u"[0-9A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u0131\\u0134-\\u013E"
        u"\\u0141-\\u0148\\u014A-\\u017E\\u018F\\u01A0\\u01A1\\u01AF\\u01B0\\u01CD-"
        u"\\u01DC\\u01DE-\\u01E3\\u01E6-\\u01F0\\u01F4\\u01F5\\u01F8-\\u021B\\u021E"
        u"\\u021F\\u0226-\\u0233\\u0259\\u02BB\\u02BC\\u02EC\\u0300-\\u0304\\u0306-"
        u"\\u030C\\u030F-\\u0311\\u0313\\u0314\\u031B\\u0323-\\u0328\\u032D\\u032E"
        u"\\u0330\\u0331\\u0335\\u0338\\u0339\\u0342\\u0345\\u037B-\\u037D\\u0386"
        u"\\u0388-\\u038A\\u038C\\u038E-\\u03A1\\u03A3-\\u03CE\\u03FC-\\u045F\\u048A-"
        u"\\u04FF\\u0510-\\u0529\\u052E\\u052F\\u0531-\\u0556\\u0559\\u0561-\\u0586"
        u"\\u05B4\\u05D0-\\u05EA\\u05EF-\\u05F2\\u0620-\\u063F\\u0641-\\u0655\\u0660-"
        u"\\u0669\\u0670-\\u0672\\u0674\\u0679-\\u068D\\u068F-\\u06A0\\u06A2-\\u06D3"
        u"\\u06D5\\u06E5\\u06E6\\u06EE-\\u06FC\\u06FF\\u0750-\\u07B1\\u0870-\\u0887"
        u"\\u0889-\\u088E\\u08A0-\\u08AC\\u08B2\\u08B5-\\u08C9\\u0901-\\u094D\\u094F"
        u"\\u0950\\u0956\\u0957\\u0960-\\u0963\\u0966-\\u096F\\u0971-\\u0977\\u0979-"
        u"\\u097F\\u0981-\\u0983\\u0985-\\u098C\\u098F\\u0990\\u0993-\\u09A8\\u09AA-"
        u"\\u09B0\\u09B2\\u09B6-\\u09B9\\u09BC-\\u09C4\\u09C7\\u09C8\\u09CB-\\u09CE"
        u"\\u09D7\\u09E0-\\u09E3\\u09E6-\\u09F1\\u09FE\\u0A01-\\u0A03\\u0A05-\\u0A0A"
        u"\\u0A0F\\u0A10\\u0A13-\\u0A28\\u0A2A-\\u0A30\\u0A32\\u0A35\\u0A38\\u0A39"
        u"\\u0A3C\\u0A3E-\\u0A42\\u0A47\\u0A48\\u0A4B-\\u0A4D\\u0A5C\\u0A66-\\u0A74"
        u"\\u0A81-\\u0A83\\u0A85-\\u0A8D\\u0A8F-\\u0A91\\u0A93-\\u0AA8\\u0AAA-\\u0AB0"
        u"\\u0AB2\\u0AB3\\u0AB5-\\u0AB9\\u0ABC-\\u0AC5\\u0AC7-\\u0AC9\\u0ACB-\\u0ACD"
        u"\\u0AD0\\u0AE0-\\u0AE3\\u0AE6-\\u0AEF\\u0AFA-\\u0AFF\\u0B01-\\u0B03\\u0B05-"
        u"\\u0B0C\\u0B0F\\u0B10\\u0B13-\\u0B28\\u0B2A-\\u0B30\\u0B32\\u0B33\\u0B35-"
        u"\\u0B39\\u0B3C-\\u0B43\\u0B47\\u0B48\\u0B4B-\\u0B4D\\u0B55-\\u0B57\\u0B5F-"
        u"\\u0B61\\u0B66-\\u0B6F\\u0B71\\u0B82\\u0B83\\u0B85-\\u0B8A\\u0B8E-\\u0B90"
        u"\\u0B92-\\u0B95\\u0B99\\u0B9A\\u0B9C\\u0B9E\\u0B9F\\u0BA3\\u0BA4\\u0BA8-"
        u"\\u0BAA\\u0BAE-\\u0BB9\\u0BBE-\\u0BC2\\u0BC6-\\u0BC8\\u0BCA-\\u0BCD\\u0BD0"
        u"\\u0BD7\\u0BE6-\\u0BEF\\u0C01-\\u0C0C\\u0C0E-\\u0C10\\u0C12-\\u0C28\\u0C2A-"
        u"\\u0C33\\u0C35-\\u0C39\\u0C3C-\\u0C44\\u0C46-\\u0C48\\u0C4A-\\u0C4D\\u0C55"
        u"\\u0C56\\u0C5D\\u0C60\\u0C61\\u0C66-\\u0C6F\\u0C80\\u0C82\\u0C83\\u0C85-"
        u"\\u0C8C\\u0C8E-\\u0C90\\u0C92-\\u0CA8\\u0CAA-\\u0CB3\\u0CB5-\\u0CB9\\u0CBC-"
        u"\\u0CC4\\u0CC6-\\u0CC8\\u0CCA-\\u0CCD\\u0CD5\\u0CD6\\u0CDD\\u0CE0-\\u0CE3"
        u"\\u0CE6-\\u0CEF\\u0CF1-\\u0CF3\\u0D00\\u0D02\\u0D03\\u0D05-\\u0D0C\\u0D0E-"
        u"\\u0D10\\u0D12-\\u0D3A\\u0D3D-\\u0D43\\u0D46-\\u0D48\\u0D4A-\\u0D4E\\u0D54-"
        u"\\u0D57\\u0D60\\u0D61\\u0D66-\\u0D6F\\u0D7A-\\u0D7F\\u0D82\\u0D83\\u0D85-"
        u"\\u0D8E\\u0D91-\\u0D96\\u0D9A-\\u0DA5\\u0DA7-\\u0DB1\\u0DB3-\\u0DBB\\u0DBD"
        u"\\u0DC0-\\u0DC6\\u0DCA\\u0DCF-\\u0DD4\\u0DD6\\u0DD8-\\u0DDE\\u0DF2\\u0E01-"
        u"\\u0E32\\u0E34-\\u0E3A\\u0E40-\\u0E4E\\u0E50-\\u0E59\\u0E81\\u0E82\\u0E84"
        u"\\u0E86-\\u0E8A\\u0E8C-\\u0EA3\\u0EA5\\u0EA7-\\u0EB2\\u0EB4-\\u0EBD\\u0EC0-"
        u"\\u0EC4\\u0EC6\\u0EC8-\\u0ECE\\u0ED0-\\u0ED9\\u0EDE\\u0EDF\\u0F00\\u0F20-"
        u"\\u0F29\\u0F35\\u0F37\\u0F3E-\\u0F42\\u0F44-\\u0F47\\u0F49-\\u0F4C\\u0F4E-"
        u"\\u0F51\\u0F53-\\u0F56\\u0F58-\\u0F5B\\u0F5D-\\u0F68\\u0F6A-\\u0F6C\\u0F71"
        u"\\u0F72\\u0F74\\u0F7A-\\u0F80\\u0F82-\\u0F84\\u0F86-\\u0F92\\u0F94-\\u0F97"
        u"\\u0F99-\\u0F9C\\u0F9E-\\u0FA1\\u0FA3-\\u0FA6\\u0FA8-\\u0FAB\\u0FAD-\\u0FB8"
        u"\\u0FBA-\\u0FBC\\u0FC6\\u1000-\\u1049\\u1050-\\u109D\\u10C7\\u10CD\\u10D0-"
        u"\\u10F0\\u10F7-\\u10FA\\u10FD-\\u10FF\\u1200-\\u1248\\u124A-\\u124D\\u1250-"
        u"\\u1256\\u1258\\u125A-\\u125D\\u1260-\\u1288\\u128A-\\u128D\\u1290-\\u12B0"
        u"\\u12B2-\\u12B5\\u12B8-\\u12BE\\u12C0\\u12C2-\\u12C5\\u12C8-\\u12D6\\u12D8-"
        u"\\u1310\\u1312-\\u1315\\u1318-\\u135A\\u135D-\\u135F\\u1380-\\u138F\\u1780-"
        u"\\u17A2\\u17A5-\\u17A7\\u17A9-\\u17B3\\u17B6-\\u17CD\\u17D0\\u17D2\\u17D7"
        u"\\u17DC\\u17E0-\\u17E9\\u1C90-\\u1CBA\\u1CBD-\\u1CBF\\u1E00-\\u1E99\\u1E9E"
        u"\\u1EA0-\\u1EF9\\u1F00-\\u1F15\\u1F18-\\u1F1D\\u1F20-\\u1F45\\u1F48-\\u1F4D"
        u"\\u1F50-\\u1F57\\u1F59\\u1F5B\\u1F5D\\u1F5F-\\u1F70\\u1F72\\u1F74\\u1F76"
        u"\\u1F78\\u1F7A\\u1F7C\\u1F80-\\u1FB4\\u1FB6-\\u1FBA\\u1FBC\\u1FC2-\\u1FC4"
        u"\\u1FC6-\\u1FC8\\u1FCA\\u1FCC\\u1FD0-\\u1FD2\\u1FD6-\\u1FDA\\u1FE0-\\u1FE2"
        u"\\u1FE4-\\u1FEA\\u1FEC\\u1FF2-\\u1FF4\\u1FF6-\\u1FF8\\u1FFA\\u1FFC\\u2D27"
        u"\\u2D2D\\u2D80-\\u2D96\\u2DA0-\\u2DA6\\u2DA8-\\u2DAE\\u2DB0-\\u2DB6\\u2DB8-"
        u"\\u2DBE\\u2DC0-\\u2DC6\\u2DC8-\\u2DCE\\u2DD0-\\u2DD6\\u2DD8-\\u2DDE\\u3005-"
        u"\\u3007\\u3041-\\u3096\\u3099\\u309A\\u309D\\u309E\\u30A1-\\u30FA\\u30FC-"
        u"\\u30FE\\u3105-\\u312D\\u312F\\u31A0-\\u31BF\\u3400-\\u4DBF\\u4E00-\\u9FFF"
        u"\\uA67F\\uA717-\\uA71F\\uA788\\uA78D\\uA792\\uA793\\uA7AA\\uA7C0-\\uA7CA"
        u"\\uA7D0\\uA7D1\\uA7D3\\uA7D5-\\uA7D9\\uA9E7-\\uA9FE\\uAA60-\\uAA76\\uAA7A-"
        u"\\uAA7F\\uAB01-\\uAB06\\uAB09-\\uAB0E\\uAB11-\\uAB16\\uAB20-\\uAB26\\uAB28-"
        u"\\uAB2E\\uAB66\\uAB67\\uAC00-\\uD7A3\\uFA0E\\uFA0F\\uFA11\\uFA13\\uFA14"
        u"\\uFA1F\\uFA21\\uFA23\\uFA24\\uFA27-\\uFA29\\U00011301\\U00011303"
        u"\\U0001133B\\U0001133C\\U00016FF0\\U00016FF1\\U0001B11F-\\U0001B122"
        u"\\U0001B132\\U0001B150-\\U0001B152\\U0001B155\\U0001B164-\\U0001B167"
        u"\\U0001DF00-\\U0001DF1E\\U0001DF25-\\U0001DF2A\\U0001E08F\\U0001E7E0-"
        u"\\U0001E7E6\\U0001E7E8-\\U0001E7EB\\U0001E7ED\\U0001E7EE\\U0001E7F0-"
        u"\\U0001E7FE\\U00020000-\\U0002A6DF\\U0002A700-\\U0002B739\\U0002B740-"
        u"\\U0002B81D\\U0002B820-\\U0002CEA1\\U0002CEB0-\\U0002EBE0\\U0002EBF0-"
        u"\\U0002EE5D\\U00030000-\\U0003134A\\U00031350-\\U000323AF]";

    gRecommendedSet = new UnicodeSet(UnicodeString(recommendedPat), status);
    if (gRecommendedSet == nullptr) {
        status = U_MEMORY_ALLOCATION_ERROR;
        delete gInclusionSet;
        return;
    }
    gRecommendedSet->freeze();
    gNfdNormalizer = Normalizer2::getNFDInstance(status);
    ucln_i18n_registerCleanup(UCLN_I18N_SPOOF, uspoof_cleanup);
}

}  // namespace

U_CFUNC void uspoof_internalInitStatics(UErrorCode *status) {
    umtx_initOnce(gSpoofInitStaticsOnce, &initializeStatics, *status);
}

U_CAPI USpoofChecker * U_EXPORT2
uspoof_open(UErrorCode *status) {
    umtx_initOnce(gSpoofInitStaticsOnce, &initializeStatics, *status);
    if (U_FAILURE(*status)) {
        return nullptr;
    }
    SpoofImpl *si = new SpoofImpl(*status);
    if (si == nullptr) {
        *status = U_MEMORY_ALLOCATION_ERROR;
        return nullptr;
    }
    if (U_FAILURE(*status)) {
        delete si;
        return nullptr;
    }
    return si->asUSpoofChecker();
}


U_CAPI USpoofChecker * U_EXPORT2
uspoof_openFromSerialized(const void *data, int32_t length, int32_t *pActualLength,
                          UErrorCode *status) {
    if (U_FAILURE(*status)) {
        return nullptr;
    }

    if (data == nullptr) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return nullptr;
    }

    umtx_initOnce(gSpoofInitStaticsOnce, &initializeStatics, *status);
    if (U_FAILURE(*status))
    {
        return nullptr;
    }

    SpoofData *sd = new SpoofData(data, length, *status);
    if (sd == nullptr) {
        *status = U_MEMORY_ALLOCATION_ERROR;
        return nullptr;
    }

    if (U_FAILURE(*status)) {
        delete sd;
        return nullptr;
    }

    SpoofImpl *si = new SpoofImpl(sd, *status);
    if (si == nullptr) {
        *status = U_MEMORY_ALLOCATION_ERROR;
        delete sd; // explicit delete as the destructor for si won't be called.
        return nullptr;
    }

    if (U_FAILURE(*status)) {
        delete si; // no delete for sd, as the si destructor will delete it.
        return nullptr;
    }

    if (pActualLength != nullptr) {
        *pActualLength = sd->size();
    }
    return si->asUSpoofChecker();
}


U_CAPI USpoofChecker * U_EXPORT2
uspoof_clone(const USpoofChecker *sc, UErrorCode *status) {
    const SpoofImpl *src = SpoofImpl::validateThis(sc, *status);
    if (src == nullptr) {
        return nullptr;
    }
    SpoofImpl *result = new SpoofImpl(*src, *status);   // copy constructor
    if (result == nullptr) {
        *status = U_MEMORY_ALLOCATION_ERROR;
        return nullptr;
    }
    if (U_FAILURE(*status)) {
        delete result;
        result = nullptr;
    }
    return result->asUSpoofChecker();
}


U_CAPI void U_EXPORT2
uspoof_close(USpoofChecker *sc) {
    UErrorCode status = U_ZERO_ERROR;
    SpoofImpl *This = SpoofImpl::validateThis(sc, status);
    delete This;
}


U_CAPI void U_EXPORT2
uspoof_setChecks(USpoofChecker *sc, int32_t checks, UErrorCode *status) {
    SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (This == nullptr) {
        return;
    }

    // Verify that the requested checks are all ones (bits) that 
    //   are acceptable, known values.
    if (checks & ~(USPOOF_ALL_CHECKS | USPOOF_AUX_INFO)) {
        *status = U_ILLEGAL_ARGUMENT_ERROR; 
        return;
    }

    This->fChecks = checks;
}


U_CAPI int32_t U_EXPORT2
uspoof_getChecks(const USpoofChecker *sc, UErrorCode *status) {
    const SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (This == nullptr) {
        return 0;
    }
    return This->fChecks;
}

U_CAPI void U_EXPORT2
uspoof_setRestrictionLevel(USpoofChecker *sc, URestrictionLevel restrictionLevel) {
    UErrorCode status = U_ZERO_ERROR;
    SpoofImpl *This = SpoofImpl::validateThis(sc, status);
    if (This != nullptr) {
        This->fRestrictionLevel = restrictionLevel;
        This->fChecks |= USPOOF_RESTRICTION_LEVEL;
    }
}

U_CAPI URestrictionLevel U_EXPORT2
uspoof_getRestrictionLevel(const USpoofChecker *sc) {
    UErrorCode status = U_ZERO_ERROR;
    const SpoofImpl *This = SpoofImpl::validateThis(sc, status);
    if (This == nullptr) {
        return USPOOF_UNRESTRICTIVE;
    }
    return This->fRestrictionLevel;
}

U_CAPI void U_EXPORT2
uspoof_setAllowedLocales(USpoofChecker *sc, const char *localesList, UErrorCode *status) {
    SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (This == nullptr) {
        return;
    }
    This->setAllowedLocales(localesList, *status);
}

U_CAPI const char * U_EXPORT2
uspoof_getAllowedLocales(USpoofChecker *sc, UErrorCode *status) {
    SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (This == nullptr) {
        return nullptr;
    }
    return This->getAllowedLocales(*status);
}


U_CAPI const USet * U_EXPORT2
uspoof_getAllowedChars(const USpoofChecker *sc, UErrorCode *status) {
    const UnicodeSet *result = uspoof_getAllowedUnicodeSet(sc, status);
    return result->toUSet();
}

U_CAPI const UnicodeSet * U_EXPORT2
uspoof_getAllowedUnicodeSet(const USpoofChecker *sc, UErrorCode *status) {
    const SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (This == nullptr) {
        return nullptr;
    }
    return This->fAllowedCharsSet;
}


U_CAPI void U_EXPORT2
uspoof_setAllowedChars(USpoofChecker *sc, const USet *chars, UErrorCode *status) {
    const UnicodeSet *set = UnicodeSet::fromUSet(chars);
    uspoof_setAllowedUnicodeSet(sc, set, status);
}


U_CAPI void U_EXPORT2
uspoof_setAllowedUnicodeSet(USpoofChecker *sc, const UnicodeSet *chars, UErrorCode *status) {
    SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (This == nullptr) {
        return;
    }
    if (chars->isBogus()) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return;
    }
    UnicodeSet *clonedSet = chars->clone();
    if (clonedSet == nullptr || clonedSet->isBogus()) {
        *status = U_MEMORY_ALLOCATION_ERROR;
        return;
    }
    clonedSet->freeze();
    delete This->fAllowedCharsSet;
    This->fAllowedCharsSet = clonedSet;
    This->fChecks |= USPOOF_CHAR_LIMIT;
}


U_CAPI int32_t U_EXPORT2
uspoof_check(const USpoofChecker *sc,
             const char16_t *id, int32_t length,
             int32_t *position,
             UErrorCode *status) {

    // Backwards compatibility:
    if (position != nullptr) {
        *position = 0;
    }

    // Delegate to uspoof_check2
    return uspoof_check2(sc, id, length, nullptr, status);
}


U_CAPI int32_t U_EXPORT2
uspoof_check2(const USpoofChecker *sc,
    const char16_t* id, int32_t length,
    USpoofCheckResult* checkResult,
    UErrorCode *status) {

    const SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (This == nullptr) {
        return 0;
    }
    if (length < -1) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }
    UnicodeString idStr((length == -1), id, length);  // Aliasing constructor.
    int32_t result = uspoof_check2UnicodeString(sc, idStr, checkResult, status);
    return result;
}


U_CAPI int32_t U_EXPORT2
uspoof_checkUTF8(const USpoofChecker *sc,
                 const char *id, int32_t length,
                 int32_t *position,
                 UErrorCode *status) {

    // Backwards compatibility:
    if (position != nullptr) {
        *position = 0;
    }

    // Delegate to uspoof_check2
    return uspoof_check2UTF8(sc, id, length, nullptr, status);
}


U_CAPI int32_t U_EXPORT2
uspoof_check2UTF8(const USpoofChecker *sc,
    const char *id, int32_t length,
    USpoofCheckResult* checkResult,
    UErrorCode *status) {

    if (U_FAILURE(*status)) {
        return 0;
    }
    UnicodeString idStr = UnicodeString::fromUTF8(StringPiece(id, length>=0 ? length : static_cast<int32_t>(uprv_strlen(id))));
    int32_t result = uspoof_check2UnicodeString(sc, idStr, checkResult, status);
    return result;
}


U_CAPI int32_t U_EXPORT2
uspoof_areConfusable(const USpoofChecker *sc,
                     const char16_t *id1, int32_t length1,
                     const char16_t *id2, int32_t length2,
                     UErrorCode *status) {
    SpoofImpl::validateThis(sc, *status);
    if (U_FAILURE(*status)) {
        return 0;
    }
    if (length1 < -1 || length2 < -1) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }
        
    UnicodeString id1Str((length1==-1), id1, length1);  // Aliasing constructor
    UnicodeString id2Str((length2==-1), id2, length2);  // Aliasing constructor
    return uspoof_areConfusableUnicodeString(sc, id1Str, id2Str, status);
}


U_CAPI int32_t U_EXPORT2
uspoof_areConfusableUTF8(const USpoofChecker *sc,
                         const char *id1, int32_t length1,
                         const char *id2, int32_t length2,
                         UErrorCode *status) {
    SpoofImpl::validateThis(sc, *status);
    if (U_FAILURE(*status)) {
        return 0;
    }
    if (length1 < -1 || length2 < -1) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }
    UnicodeString id1Str = UnicodeString::fromUTF8(StringPiece(id1, length1>=0? length1 : static_cast<int32_t>(uprv_strlen(id1))));
    UnicodeString id2Str = UnicodeString::fromUTF8(StringPiece(id2, length2>=0? length2 : static_cast<int32_t>(uprv_strlen(id2))));
    int32_t results = uspoof_areConfusableUnicodeString(sc, id1Str, id2Str, status);
    return results;
}
 

U_CAPI int32_t U_EXPORT2
uspoof_areConfusableUnicodeString(const USpoofChecker *sc,
                                  const icu::UnicodeString &id1,
                                  const icu::UnicodeString &id2,
                                  UErrorCode *status) {
    const SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (U_FAILURE(*status)) {
        return 0;
    }
    //
    // See section 4 of UAX 39 for the algorithm for checking whether two strings are confusable,
    //   and for definitions of the types (single, whole, mixed-script) of confusables.
    
    // We only care about a few of the check flags.  Ignore the others.
    // If no tests relevant to this function have been specified, return an error.
    // TODO:  is this really the right thing to do?  It's probably an error on the caller's part,
    //        but logically we would just return 0 (no error).
    if ((This->fChecks & USPOOF_CONFUSABLE) == 0) {
        *status = U_INVALID_STATE_ERROR;
        return 0;
    }

    // Compute the skeletons and check for confusability.
    UnicodeString id1Skeleton;
    uspoof_getSkeletonUnicodeString(sc, 0 /* deprecated */, id1, id1Skeleton, status);
    UnicodeString id2Skeleton;
    uspoof_getSkeletonUnicodeString(sc, 0 /* deprecated */, id2, id2Skeleton, status);
    if (U_FAILURE(*status)) { return 0; }
    if (id1Skeleton != id2Skeleton) {
        return 0;
    }

    // If we get here, the strings are confusable.  Now we just need to set the flags for the appropriate classes
    // of confusables according to UTS 39 section 4.
    // Start by computing the resolved script sets of id1 and id2.
    ScriptSet id1RSS;
    This->getResolvedScriptSet(id1, id1RSS, *status);
    ScriptSet id2RSS;
    This->getResolvedScriptSet(id2, id2RSS, *status);

    // Turn on all applicable flags
    int32_t result = 0;
    if (id1RSS.intersects(id2RSS)) {
        result |= USPOOF_SINGLE_SCRIPT_CONFUSABLE;
    } else {
        result |= USPOOF_MIXED_SCRIPT_CONFUSABLE;
        if (!id1RSS.isEmpty() && !id2RSS.isEmpty()) {
            result |= USPOOF_WHOLE_SCRIPT_CONFUSABLE;
        }
    }

    // Turn off flags that the user doesn't want
    if ((This->fChecks & USPOOF_SINGLE_SCRIPT_CONFUSABLE) == 0) {
        result &= ~USPOOF_SINGLE_SCRIPT_CONFUSABLE;
    }
    if ((This->fChecks & USPOOF_MIXED_SCRIPT_CONFUSABLE) == 0) {
        result &= ~USPOOF_MIXED_SCRIPT_CONFUSABLE;
    }
    if ((This->fChecks & USPOOF_WHOLE_SCRIPT_CONFUSABLE) == 0) {
        result &= ~USPOOF_WHOLE_SCRIPT_CONFUSABLE;
    }

    return result;
}

U_CAPI uint32_t U_EXPORT2 uspoof_areBidiConfusable(const USpoofChecker *sc, UBiDiDirection direction,
                                                  const char16_t *id1, int32_t length1,
                                                  const char16_t *id2, int32_t length2,
                                                   UErrorCode *status) {
    UnicodeString id1Str((length1 == -1), id1, length1); // Aliasing constructor
    UnicodeString id2Str((length2 == -1), id2, length2); // Aliasing constructor
    if (id1Str.isBogus() || id2Str.isBogus()) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }
    return uspoof_areBidiConfusableUnicodeString(sc, direction, id1Str, id2Str, status);
}

U_CAPI uint32_t U_EXPORT2 uspoof_areBidiConfusableUTF8(const USpoofChecker *sc, UBiDiDirection direction,
                                                      const char *id1, int32_t length1, const char *id2,
                                                      int32_t length2, UErrorCode *status) {
    if (length1 < -1 || length2 < -1) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }
    UnicodeString id1Str = UnicodeString::fromUTF8(
        StringPiece(id1, length1 >= 0 ? length1 : static_cast<int32_t>(uprv_strlen(id1))));
    UnicodeString id2Str = UnicodeString::fromUTF8(
        StringPiece(id2, length2 >= 0 ? length2 : static_cast<int32_t>(uprv_strlen(id2))));
    return uspoof_areBidiConfusableUnicodeString(sc, direction, id1Str, id2Str, status);
}

U_CAPI uint32_t U_EXPORT2 uspoof_areBidiConfusableUnicodeString(const USpoofChecker *sc,
                                                               UBiDiDirection direction,
                                                               const icu::UnicodeString &id1,
                                                               const icu::UnicodeString &id2,
                                                               UErrorCode *status) {
    const SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (U_FAILURE(*status)) {
        return 0;
    }
    //
    // See section 4 of UTS 39 for the algorithm for checking whether two strings are confusable,
    //   and for definitions of the types (single, whole, mixed-script) of confusables.

    // We only care about a few of the check flags.  Ignore the others.
    // If no tests relevant to this function have been specified, return an error.
    // TODO:  is this really the right thing to do?  It's probably an error on the caller's part,
    //        but logically we would just return 0 (no error).
    if ((This->fChecks & USPOOF_CONFUSABLE) == 0) {
        *status = U_INVALID_STATE_ERROR;
        return 0;
    }

    // Compute the skeletons and check for confusability.
    UnicodeString id1Skeleton;
    uspoof_getBidiSkeletonUnicodeString(sc, direction, id1, id1Skeleton, status);
    UnicodeString id2Skeleton;
    uspoof_getBidiSkeletonUnicodeString(sc, direction, id2, id2Skeleton, status);
    if (U_FAILURE(*status)) {
        return 0;
    }
    if (id1Skeleton != id2Skeleton) {
        return 0;
    }

    // If we get here, the strings are confusable.  Now we just need to set the flags for the appropriate
    // classes of confusables according to UTS 39 section 4. Start by computing the resolved script sets
    // of id1 and id2.
    ScriptSet id1RSS;
    This->getResolvedScriptSet(id1, id1RSS, *status);
    ScriptSet id2RSS;
    This->getResolvedScriptSet(id2, id2RSS, *status);

    // Turn on all applicable flags
    uint32_t result = 0;
    if (id1RSS.intersects(id2RSS)) {
        result |= USPOOF_SINGLE_SCRIPT_CONFUSABLE;
    } else {
        result |= USPOOF_MIXED_SCRIPT_CONFUSABLE;
        if (!id1RSS.isEmpty() && !id2RSS.isEmpty()) {
            result |= USPOOF_WHOLE_SCRIPT_CONFUSABLE;
        }
    }

    // Turn off flags that the user doesn't want
    return result & This->fChecks;
}


U_CAPI int32_t U_EXPORT2
uspoof_checkUnicodeString(const USpoofChecker *sc,
                          const icu::UnicodeString &id,
                          int32_t *position,
                          UErrorCode *status) {

    // Backwards compatibility:
    if (position != nullptr) {
        *position = 0;
    }

    // Delegate to uspoof_check2
    return uspoof_check2UnicodeString(sc, id, nullptr, status);
}

namespace {

int32_t checkImpl(const SpoofImpl* This, const UnicodeString& id, CheckResult* checkResult, UErrorCode* status) {
    U_ASSERT(This != nullptr);
    U_ASSERT(checkResult != nullptr);
    checkResult->clear();
    int32_t result = 0;

    if (0 != (This->fChecks & USPOOF_RESTRICTION_LEVEL)) {
        URestrictionLevel idRestrictionLevel = This->getRestrictionLevel(id, *status);
        if (idRestrictionLevel > This->fRestrictionLevel) {
            result |= USPOOF_RESTRICTION_LEVEL;
        }
        checkResult->fRestrictionLevel = idRestrictionLevel;
    }

    if (0 != (This->fChecks & USPOOF_MIXED_NUMBERS)) {
        UnicodeSet numerics;
        This->getNumerics(id, numerics, *status);
        if (numerics.size() > 1) {
            result |= USPOOF_MIXED_NUMBERS;
        }
        checkResult->fNumerics = numerics;  // UnicodeSet::operator=
    }

    if (0 != (This->fChecks & USPOOF_HIDDEN_OVERLAY)) {
        int32_t index = This->findHiddenOverlay(id, *status);
        if (index != -1) {
            result |= USPOOF_HIDDEN_OVERLAY;
        }
    }


    if (0 != (This->fChecks & USPOOF_CHAR_LIMIT)) {
        int32_t i;
        UChar32 c;
        int32_t length = id.length();
        for (i=0; i<length ;) {
            c = id.char32At(i);
            i += U16_LENGTH(c);
            if (!This->fAllowedCharsSet->contains(c)) {
                result |= USPOOF_CHAR_LIMIT;
                break;
            }
        }
    }

    if (0 != (This->fChecks & USPOOF_INVISIBLE)) {
        // This check needs to be done on NFD input
        UnicodeString nfdText;
        gNfdNormalizer->normalize(id, nfdText, *status);
        int32_t nfdLength = nfdText.length();

        // scan for more than one occurrence of the same non-spacing mark
        // in a sequence of non-spacing marks.
        int32_t     i;
        UChar32     c;
        UChar32     firstNonspacingMark = 0;
        UBool       haveMultipleMarks = false;  
        UnicodeSet  marksSeenSoFar;   // Set of combining marks in a single combining sequence.
        
        for (i=0; i<nfdLength ;) {
            c = nfdText.char32At(i);
            i += U16_LENGTH(c);
            if (u_charType(c) != U_NON_SPACING_MARK) {
                firstNonspacingMark = 0;
                if (haveMultipleMarks) {
                    marksSeenSoFar.clear();
                    haveMultipleMarks = false;
                }
                continue;
            }
            if (firstNonspacingMark == 0) {
                firstNonspacingMark = c;
                continue;
            }
            if (!haveMultipleMarks) {
                marksSeenSoFar.add(firstNonspacingMark);
                haveMultipleMarks = true;
            }
            if (marksSeenSoFar.contains(c)) {
                // report the error, and stop scanning.
                // No need to find more than the first failure.
                result |= USPOOF_INVISIBLE;
                break;
            }
            marksSeenSoFar.add(c);
        }
    }

    checkResult->fChecks = result;
    return checkResult->toCombinedBitmask(This->fChecks);
}

}  // namespace

U_CAPI int32_t U_EXPORT2
uspoof_check2UnicodeString(const USpoofChecker *sc,
                          const icu::UnicodeString &id,
                          USpoofCheckResult* checkResult,
                          UErrorCode *status) {
    const SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (This == nullptr) {
        return false;
    }

    if (checkResult != nullptr) {
        CheckResult* ThisCheckResult = CheckResult::validateThis(checkResult, *status);
        if (ThisCheckResult == nullptr) {
            return false;
        }
        return checkImpl(This, id, ThisCheckResult, status);
    } else {
        // Stack-allocate the checkResult since this method doesn't return it
        CheckResult stackCheckResult;
        return checkImpl(This, id, &stackCheckResult, status);
    }
}


U_CAPI int32_t U_EXPORT2
uspoof_getSkeleton(const USpoofChecker *sc,
                   uint32_t type,
                   const char16_t *id,  int32_t length,
                   char16_t *dest, int32_t destCapacity,
                   UErrorCode *status) {

    SpoofImpl::validateThis(sc, *status);
    if (U_FAILURE(*status)) {
        return 0;
    }
    if (length<-1 || destCapacity<0 || (destCapacity==0 && dest!=nullptr)) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }

    UnicodeString idStr((length==-1), id, length);  // Aliasing constructor
    UnicodeString destStr;
    uspoof_getSkeletonUnicodeString(sc, type, idStr, destStr, status);
    destStr.extract(dest, destCapacity, *status);
    return destStr.length();
}

U_CAPI int32_t U_EXPORT2 uspoof_getBidiSkeleton(const USpoofChecker *sc, UBiDiDirection direction,
                                                const UChar *id, int32_t length, UChar *dest,
                                                int32_t destCapacity, UErrorCode *status) {
    UnicodeString idStr((length == -1), id, length); // Aliasing constructor
    if (idStr.isBogus()) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }
    UnicodeString destStr;
    uspoof_getBidiSkeletonUnicodeString(sc, direction, idStr, destStr, status);
    return destStr.extract(dest, destCapacity, *status);
}



U_I18N_API UnicodeString &U_EXPORT2 uspoof_getBidiSkeletonUnicodeString(const USpoofChecker *sc,
                                                                        UBiDiDirection direction,
                                                                        const UnicodeString &id,
                                                                        UnicodeString &dest,
                                                                        UErrorCode *status) {
    dest.remove();
    if (direction != UBIDI_LTR && direction != UBIDI_RTL) {
      *status = U_ILLEGAL_ARGUMENT_ERROR;
      return dest;
    }
    UBiDi *bidi = ubidi_open();
    ubidi_setPara(bidi, id.getBuffer(), id.length(), direction,
                  /*embeddingLevels*/ nullptr, status);
    if (U_FAILURE(*status)) {
        ubidi_close(bidi);
        return dest;
    }
    UnicodeString reordered;
    int32_t const size = ubidi_getProcessedLength(bidi);
    UChar* const reorderedBuffer = reordered.getBuffer(size);
    if (reorderedBuffer == nullptr) {
        *status = U_MEMORY_ALLOCATION_ERROR;
        ubidi_close(bidi);
        return dest;
    }
    ubidi_writeReordered(bidi, reorderedBuffer, size,
                         UBIDI_KEEP_BASE_COMBINING | UBIDI_DO_MIRRORING, status);
    reordered.releaseBuffer(size);
    ubidi_close(bidi);

    if (U_FAILURE(*status)) {
        return dest;
    }

    // The type parameter is deprecated since ICU 58; any number may be passed.
    constexpr uint32_t deprecatedType = 58;
    return uspoof_getSkeletonUnicodeString(sc, deprecatedType, reordered, dest, status);
}



U_I18N_API UnicodeString &  U_EXPORT2
uspoof_getSkeletonUnicodeString(const USpoofChecker *sc,
                                uint32_t /*type*/,
                                const UnicodeString &id,
                                UnicodeString &dest,
                                UErrorCode *status) {
    const SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (U_FAILURE(*status)) {
        return dest;
    }

    UnicodeString nfdId;
    gNfdNormalizer->normalize(id, nfdId, *status);

    // Apply the skeleton mapping to the NFD normalized input string
    // Accumulate the skeleton, possibly unnormalized, in a UnicodeString.
    int32_t inputIndex = 0;
    UnicodeString skelStr;
    int32_t normalizedLen = nfdId.length();
    for (inputIndex=0; inputIndex < normalizedLen; ) {
        UChar32 c = nfdId.char32At(inputIndex);
        inputIndex += U16_LENGTH(c);
        if (!u_hasBinaryProperty(c, UCHAR_DEFAULT_IGNORABLE_CODE_POINT)) {
            This->fSpoofData->confusableLookup(c, skelStr);
        }
    }

    gNfdNormalizer->normalize(skelStr, dest, *status);
    return dest;
}

U_CAPI int32_t U_EXPORT2 uspoof_getSkeletonUTF8(const USpoofChecker *sc, uint32_t type, const char *id,
                                                int32_t length, char *dest, int32_t destCapacity,
                       UErrorCode *status) {
    SpoofImpl::validateThis(sc, *status);
    if (U_FAILURE(*status)) {
        return 0;
    }
    if (length<-1 || destCapacity<0 || (destCapacity==0 && dest!=nullptr)) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }

    UnicodeString srcStr = UnicodeString::fromUTF8(
        StringPiece(id, length >= 0 ? length : static_cast<int32_t>(uprv_strlen(id))));
    UnicodeString destStr;
    uspoof_getSkeletonUnicodeString(sc, type, srcStr, destStr, status);
    if (U_FAILURE(*status)) {
        return 0;
    }

    int32_t lengthInUTF8 = 0;
    u_strToUTF8(dest, destCapacity, &lengthInUTF8, destStr.getBuffer(), destStr.length(), status);
    return lengthInUTF8;
}

U_CAPI int32_t U_EXPORT2 uspoof_getBidiSkeletonUTF8(const USpoofChecker *sc, UBiDiDirection direction,
                                                    const char *id, int32_t length, char *dest,
                                                    int32_t destCapacity, UErrorCode *status) {
    if (length < -1) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return 0;
    }

    UnicodeString srcStr = UnicodeString::fromUTF8(
        StringPiece(id, length >= 0 ? length : static_cast<int32_t>(uprv_strlen(id))));
    UnicodeString destStr;
    uspoof_getBidiSkeletonUnicodeString(sc, direction, srcStr, destStr, status);
    if (U_FAILURE(*status)) {
        return 0;
    }

    int32_t lengthInUTF8 = 0;
    u_strToUTF8(dest, destCapacity, &lengthInUTF8, destStr.getBuffer(), destStr.length(), status);
    return lengthInUTF8;
}


U_CAPI int32_t U_EXPORT2
uspoof_serialize(USpoofChecker *sc,void *buf, int32_t capacity, UErrorCode *status) {
    SpoofImpl *This = SpoofImpl::validateThis(sc, *status);
    if (This == nullptr) {
        U_ASSERT(U_FAILURE(*status));
        return 0;
    }

    return This->fSpoofData->serialize(buf, capacity, *status);
}

U_CAPI const USet * U_EXPORT2
uspoof_getInclusionSet(UErrorCode *status) {
    umtx_initOnce(gSpoofInitStaticsOnce, &initializeStatics, *status);
    return gInclusionSet->toUSet();
}

U_CAPI const USet * U_EXPORT2
uspoof_getRecommendedSet(UErrorCode *status) {
    umtx_initOnce(gSpoofInitStaticsOnce, &initializeStatics, *status);
    return gRecommendedSet->toUSet();
}

U_I18N_API const UnicodeSet * U_EXPORT2
uspoof_getInclusionUnicodeSet(UErrorCode *status) {
    umtx_initOnce(gSpoofInitStaticsOnce, &initializeStatics, *status);
    return gInclusionSet;
}

U_I18N_API const UnicodeSet * U_EXPORT2
uspoof_getRecommendedUnicodeSet(UErrorCode *status) {
    umtx_initOnce(gSpoofInitStaticsOnce, &initializeStatics, *status);
    return gRecommendedSet;
}

//------------------
// CheckResult APIs
//------------------

U_CAPI USpoofCheckResult* U_EXPORT2
uspoof_openCheckResult(UErrorCode *status) {
    CheckResult* checkResult = new CheckResult();
    if (checkResult == nullptr) {
        *status = U_MEMORY_ALLOCATION_ERROR;
        return nullptr;
    }
    return checkResult->asUSpoofCheckResult();
}

U_CAPI void U_EXPORT2
uspoof_closeCheckResult(USpoofCheckResult* checkResult) {
    UErrorCode status = U_ZERO_ERROR;
    CheckResult* This = CheckResult::validateThis(checkResult, status);
    delete This;
}

U_CAPI int32_t U_EXPORT2
uspoof_getCheckResultChecks(const USpoofCheckResult *checkResult, UErrorCode *status) {
    const CheckResult* This = CheckResult::validateThis(checkResult, *status);
    if (U_FAILURE(*status)) { return 0; }
    return This->fChecks;
}

U_CAPI URestrictionLevel U_EXPORT2
uspoof_getCheckResultRestrictionLevel(const USpoofCheckResult *checkResult, UErrorCode *status) {
    const CheckResult* This = CheckResult::validateThis(checkResult, *status);
    if (U_FAILURE(*status)) { return USPOOF_UNRESTRICTIVE; }
    return This->fRestrictionLevel;
}

U_CAPI const USet* U_EXPORT2
uspoof_getCheckResultNumerics(const USpoofCheckResult *checkResult, UErrorCode *status) {
    const CheckResult* This = CheckResult::validateThis(checkResult, *status);
    if (U_FAILURE(*status)) { return nullptr; }
    return This->fNumerics.toUSet();
}



#endif // !UCONFIG_NO_NORMALIZATION
