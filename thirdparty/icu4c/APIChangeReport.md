
  
<!--
 Copyright © 2019 and later: Unicode, Inc. and others.
 License & terms of use: http://www.unicode.org/copyright.html
-->

# ICU4C API Comparison: ICU 67 with ICU 68

> _Note_ Markdown format of this document is new for ICU 65.

- [Removed from ICU 67](#removed)
- [Deprecated or Obsoleted in ICU 68](#deprecated)
- [Changed in  ICU 68](#changed)
- [Promoted to stable in ICU 68](#promoted)
- [Added in ICU 68](#added)
- [Other existing drafts in ICU 68](#other)
- [Signature Simplifications](#simplifications)

## Removed

Removed from ICU 67
  
| File | API | ICU 67 | ICU 68 |
|---|---|---|---|
| fmtable.h | const UFormattable* icu::Formattable::toUFormattable() |  StableICU 52 | (missing)
| measunit.h | LocalArray&lt;MeasureUnit&gt; icu::MeasureUnit::splitToSingleUnits(int32_t&amp;, UErrorCode&amp;) const |  InternalICU 67 | (missing)
| measunit.h | int32_t icu::MeasureUnit::getIndex() const |  Internal | (missing)
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::resolveUnitPerUnit(const MeasureUnit&amp;, const MeasureUnit&amp;, bool*) |  Internal | (missing)
| measunit.h | <tt>static</tt> int32_t icu::MeasureUnit::getIndexCount() |  Internal | (missing)
| measunit.h | <tt>static</tt> int32_t icu::MeasureUnit::internalGetIndexForTypeAndSubtype(const char*, const char*) |  Internal | (missing)
| nounit.h | UClassID icu::NoUnit::getDynamicClassID() const |  DraftICU 60 | (missing)
| nounit.h | icu::NoUnit::NoUnit(const NoUnit&amp;) |  DraftICU 60 | (missing)
| nounit.h | icu::NoUnit::~NoUnit() |  DraftICU 60 | (missing)
| nounit.h | <tt>static</tt> NoUnit icu::NoUnit::base() |  DraftICU 60 | (missing)
| nounit.h | <tt>static</tt> NoUnit icu::NoUnit::percent() |  DraftICU 60 | (missing)
| nounit.h | <tt>static</tt> NoUnit icu::NoUnit::permille() |  DraftICU 60 | (missing)
| nounit.h | <tt>static</tt> UClassID icu::NoUnit::getStaticClassID() |  DraftICU 60 | (missing)
| nounit.h | void* icu::NoUnit::clone() const |  DraftICU 60 | (missing)
| uniset.h | const USet* icu::UnicodeSet::toUSet() |  StableICU 4.2 | (missing)

## Deprecated

Deprecated or Obsoleted in ICU 68
  
| File | API | ICU 67 | ICU 68 |
|---|---|---|---|
| numberrangeformatter.h | UnicodeString icu::number::FormattedNumberRange::getFirstDecimal(UErrorCode&amp;) const |  DraftICU 63 | DeprecatedICU 68
| numberrangeformatter.h | UnicodeString icu::number::FormattedNumberRange::getSecondDecimal(UErrorCode&amp;) const |  DraftICU 63 | DeprecatedICU 68
| umachine.h | <tt>#define</tt> FALSE |  StableICU 2.0 | DeprecatedICU 68
| umachine.h | <tt>#define</tt> TRUE |  StableICU 2.0 | DeprecatedICU 68

## Changed

Changed in  ICU 68 (old, new)


  
| File | API | ICU 67 | ICU 68 |
|---|---|---|---|
| bytestrie.h | BytesTrie&amp; icu::BytesTrie::resetToState64(uint64_t) |  Draft→StableICU 65
| bytestrie.h | uint64_t icu::BytesTrie::getState64() const |  Draft→StableICU 65
| listformatter.h | <tt>static</tt> ListFormatter* icu::ListFormatter::createInstance(const Locale&amp;, UListFormatterType, UListFormatterWidth, UErrorCode&amp;) |  Draft→StableICU 67
| localebuilder.h | UBool icu::LocaleBuilder::copyErrorTo(UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::addSupportedLocale(const Locale&amp;) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::operator=(Builder&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setDefaultLocale(const Locale*) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setDemotionPerDesiredLocale(ULocMatchDemotion) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setFavorSubtag(ULocMatchFavorSubtag) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setSupportedLocales(Iter, Iter) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setSupportedLocales(Locale::Iterator&amp;) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setSupportedLocalesFromListString(StringPiece) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setSupportedLocalesViaConverter(Iter, Iter, Conv) |  Draft→StableICU 65
| localematcher.h | Locale icu::LocaleMatcher::Result::makeResolvedLocale(UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | LocaleMatcher icu::LocaleMatcher::Builder::build(UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | LocaleMatcher&amp; icu::LocaleMatcher::operator=(LocaleMatcher&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | Result icu::LocaleMatcher::getBestMatchResult(Locale::Iterator&amp;, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | Result icu::LocaleMatcher::getBestMatchResult(const Locale&amp;, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | Result&amp; icu::LocaleMatcher::Result::operator=(Result&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | UBool icu::LocaleMatcher::Builder::copyErrorTo(UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::Result::getDesiredLocale() const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::Result::getSupportedLocale() const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::getBestMatch(Locale::Iterator&amp;, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::getBestMatch(const Locale&amp;, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::getBestMatchForListString(StringPiece, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | <tt>enum</tt> ULocMatchDemotion::ULOCMATCH_DEMOTION_NONE |  Draft→StableICU 65
| localematcher.h | <tt>enum</tt> ULocMatchDemotion::ULOCMATCH_DEMOTION_REGION |  Draft→StableICU 65
| localematcher.h | <tt>enum</tt> ULocMatchFavorSubtag::ULOCMATCH_FAVOR_LANGUAGE |  Draft→StableICU 65
| localematcher.h | <tt>enum</tt> ULocMatchFavorSubtag::ULOCMATCH_FAVOR_SCRIPT |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Builder::Builder() |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Builder::Builder(Builder&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Builder::~Builder() |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::LocaleMatcher(LocaleMatcher&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Result::Result(Result&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Result::~Result() |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::~LocaleMatcher() |  Draft→StableICU 65
| localematcher.h | int32_t icu::LocaleMatcher::Result::getDesiredIndex() const |  Draft→StableICU 65
| localematcher.h | int32_t icu::LocaleMatcher::Result::getSupportedIndex() const |  Draft→StableICU 65
| locid.h | UBool icu::Locale::ConvertingIterator&lt; Iter, Conv &gt;::hasNext() const override |  Draft→StableICU 65
| locid.h | UBool icu::Locale::Iterator::hasNext() const |  Draft→StableICU 65
| locid.h | UBool icu::Locale::RangeIterator&lt; Iter &gt;::hasNext() const override |  Draft→StableICU 65
| locid.h | const Locale&amp; icu::Locale::ConvertingIterator&lt; Iter, Conv &gt;::next() override |  Draft→StableICU 65
| locid.h | const Locale&amp; icu::Locale::Iterator::next() |  Draft→StableICU 65
| locid.h | const Locale&amp; icu::Locale::RangeIterator&lt; Iter &gt;::next() override |  Draft→StableICU 65
| locid.h | icu::Locale::ConvertingIterator&lt; Iter, Conv &gt;::ConvertingIterator(Iter, Iter, Conv) |  Draft→StableICU 65
| locid.h | icu::Locale::Iterator::~Iterator() |  Draft→StableICU 65
| locid.h | icu::Locale::RangeIterator&lt; Iter &gt;::RangeIterator(Iter, Iter) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getBar() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDecade() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDotPerCentimeter() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDotPerInch() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getEm() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getMegapixel() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getPascal() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getPixel() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getPixelPerCentimeter() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getPixelPerInch() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getThermUs() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createBar(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDecade(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDotPerCentimeter(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDotPerInch(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createEm(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createMegapixel(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createPascal(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createPixel(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createPixelPerCentimeter(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createPixelPerInch(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createThermUs(UErrorCode&amp;) |  Draft→StableICU 65
| numberformatter.h | StringClass icu::number::FormattedNumber::toDecimalNumber(UErrorCode&amp;) const |  Draft→StableICU 65
| numberrangeformatter.h | UnicodeString icu::number::FormattedNumberRange::getFirstDecimal(UErrorCode&amp;) const |  DraftICU 63 | DeprecatedICU 68
| numberrangeformatter.h | UnicodeString icu::number::FormattedNumberRange::getSecondDecimal(UErrorCode&amp;) const |  DraftICU 63 | DeprecatedICU 68
| reldatefmt.h | <tt>enum</tt> UDateAbsoluteUnit::UDAT_ABSOLUTE_HOUR |  Draft→StableICU 65
| reldatefmt.h | <tt>enum</tt> UDateAbsoluteUnit::UDAT_ABSOLUTE_MINUTE |  Draft→StableICU 65
| stringpiece.h | icu::StringPiece::StringPiece(T) |  Draft→StableICU 65
| ucal.h | int32_t ucal_getHostTimeZone(UChar*, int32_t, UErrorCode*) |  Draft→StableICU 65
| ucharstrie.h | UCharsTrie&amp; icu::UCharsTrie::resetToState64(uint64_t) |  Draft→StableICU 65
| ucharstrie.h | uint64_t icu::UCharsTrie::getState64() const |  Draft→StableICU 65
| ulistformatter.h | UListFormatter* ulistfmt_openForType(const char*, UListFormatterType, UListFormatterWidth, UErrorCode*) |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterType::ULISTFMT_TYPE_AND |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterType::ULISTFMT_TYPE_OR |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterType::ULISTFMT_TYPE_UNITS |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterWidth::ULISTFMT_WIDTH_NARROW |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterWidth::ULISTFMT_WIDTH_SHORT |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterWidth::ULISTFMT_WIDTH_WIDE |  Draft→StableICU 67
| uloc.h | UEnumeration* uloc_openAvailableByType(ULocAvailableType, UErrorCode*) |  Draft→StableICU 65
| uloc.h | <tt>enum</tt> ULocAvailableType::ULOC_AVAILABLE_DEFAULT |  Draft→StableICU 65
| uloc.h | <tt>enum</tt> ULocAvailableType::ULOC_AVAILABLE_ONLY_LEGACY_ALIASES |  Draft→StableICU 65
| uloc.h | <tt>enum</tt> ULocAvailableType::ULOC_AVAILABLE_WITH_LEGACY_ALIASES |  Draft→StableICU 65
| umachine.h | <tt>#define</tt> FALSE |  StableICU 2.0 | DeprecatedICU 68
| umachine.h | <tt>#define</tt> TRUE |  StableICU 2.0 | DeprecatedICU 68
| utrace.h | <tt>enum</tt> UTraceFunctionNumber::UTRACE_UDATA_BUNDLE |  Draft→StableICU 65
| utrace.h | <tt>enum</tt> UTraceFunctionNumber::UTRACE_UDATA_DATA_FILE |  Draft→StableICU 65
| utrace.h | <tt>enum</tt> UTraceFunctionNumber::UTRACE_UDATA_RES_FILE |  Draft→StableICU 65
| utrace.h | <tt>enum</tt> UTraceFunctionNumber::UTRACE_UDATA_START |  Draft→StableICU 65

## Promoted

Promoted to stable in ICU 68
  
| File | API | ICU 67 | ICU 68 |
|---|---|---|---|
| bytestrie.h | BytesTrie&amp; icu::BytesTrie::resetToState64(uint64_t) |  Draft→StableICU 65
| bytestrie.h | uint64_t icu::BytesTrie::getState64() const |  Draft→StableICU 65
| fmtable.h | UFormattable* icu::Formattable::toUFormattable() |  (missing) | StableICU 52
| listformatter.h | <tt>static</tt> ListFormatter* icu::ListFormatter::createInstance(const Locale&amp;, UListFormatterType, UListFormatterWidth, UErrorCode&amp;) |  Draft→StableICU 67
| localebuilder.h | UBool icu::LocaleBuilder::copyErrorTo(UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::addSupportedLocale(const Locale&amp;) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::operator=(Builder&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setDefaultLocale(const Locale*) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setDemotionPerDesiredLocale(ULocMatchDemotion) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setFavorSubtag(ULocMatchFavorSubtag) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setSupportedLocales(Iter, Iter) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setSupportedLocales(Locale::Iterator&amp;) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setSupportedLocalesFromListString(StringPiece) |  Draft→StableICU 65
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setSupportedLocalesViaConverter(Iter, Iter, Conv) |  Draft→StableICU 65
| localematcher.h | Locale icu::LocaleMatcher::Result::makeResolvedLocale(UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | LocaleMatcher icu::LocaleMatcher::Builder::build(UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | LocaleMatcher&amp; icu::LocaleMatcher::operator=(LocaleMatcher&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | Result icu::LocaleMatcher::getBestMatchResult(Locale::Iterator&amp;, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | Result icu::LocaleMatcher::getBestMatchResult(const Locale&amp;, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | Result&amp; icu::LocaleMatcher::Result::operator=(Result&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | UBool icu::LocaleMatcher::Builder::copyErrorTo(UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::Result::getDesiredLocale() const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::Result::getSupportedLocale() const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::getBestMatch(Locale::Iterator&amp;, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::getBestMatch(const Locale&amp;, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | const Locale* icu::LocaleMatcher::getBestMatchForListString(StringPiece, UErrorCode&amp;) const |  Draft→StableICU 65
| localematcher.h | <tt>enum</tt> ULocMatchDemotion::ULOCMATCH_DEMOTION_NONE |  Draft→StableICU 65
| localematcher.h | <tt>enum</tt> ULocMatchDemotion::ULOCMATCH_DEMOTION_REGION |  Draft→StableICU 65
| localematcher.h | <tt>enum</tt> ULocMatchFavorSubtag::ULOCMATCH_FAVOR_LANGUAGE |  Draft→StableICU 65
| localematcher.h | <tt>enum</tt> ULocMatchFavorSubtag::ULOCMATCH_FAVOR_SCRIPT |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Builder::Builder() |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Builder::Builder(Builder&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Builder::~Builder() |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::LocaleMatcher(LocaleMatcher&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Result::Result(Result&amp;&amp;) |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::Result::~Result() |  Draft→StableICU 65
| localematcher.h | icu::LocaleMatcher::~LocaleMatcher() |  Draft→StableICU 65
| localematcher.h | int32_t icu::LocaleMatcher::Result::getDesiredIndex() const |  Draft→StableICU 65
| localematcher.h | int32_t icu::LocaleMatcher::Result::getSupportedIndex() const |  Draft→StableICU 65
| locid.h | UBool icu::Locale::ConvertingIterator&lt; Iter, Conv &gt;::hasNext() const override |  Draft→StableICU 65
| locid.h | UBool icu::Locale::Iterator::hasNext() const |  Draft→StableICU 65
| locid.h | UBool icu::Locale::RangeIterator&lt; Iter &gt;::hasNext() const override |  Draft→StableICU 65
| locid.h | const Locale&amp; icu::Locale::ConvertingIterator&lt; Iter, Conv &gt;::next() override |  Draft→StableICU 65
| locid.h | const Locale&amp; icu::Locale::Iterator::next() |  Draft→StableICU 65
| locid.h | const Locale&amp; icu::Locale::RangeIterator&lt; Iter &gt;::next() override |  Draft→StableICU 65
| locid.h | icu::Locale::ConvertingIterator&lt; Iter, Conv &gt;::ConvertingIterator(Iter, Iter, Conv) |  Draft→StableICU 65
| locid.h | icu::Locale::Iterator::~Iterator() |  Draft→StableICU 65
| locid.h | icu::Locale::RangeIterator&lt; Iter &gt;::RangeIterator(Iter, Iter) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getBar() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDecade() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDotPerCentimeter() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDotPerInch() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getEm() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getMegapixel() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getPascal() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getPixel() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getPixelPerCentimeter() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getPixelPerInch() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getThermUs() |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createBar(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDecade(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDotPerCentimeter(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDotPerInch(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createEm(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createMegapixel(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createPascal(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createPixel(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createPixelPerCentimeter(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createPixelPerInch(UErrorCode&amp;) |  Draft→StableICU 65
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createThermUs(UErrorCode&amp;) |  Draft→StableICU 65
| numberformatter.h | StringClass icu::number::FormattedNumber::toDecimalNumber(UErrorCode&amp;) const |  Draft→StableICU 65
| reldatefmt.h | <tt>enum</tt> UDateAbsoluteUnit::UDAT_ABSOLUTE_HOUR |  Draft→StableICU 65
| reldatefmt.h | <tt>enum</tt> UDateAbsoluteUnit::UDAT_ABSOLUTE_MINUTE |  Draft→StableICU 65
| stringpiece.h | icu::StringPiece::StringPiece(T) |  Draft→StableICU 65
| ucal.h | int32_t ucal_getHostTimeZone(UChar*, int32_t, UErrorCode*) |  Draft→StableICU 65
| ucharstrie.h | UCharsTrie&amp; icu::UCharsTrie::resetToState64(uint64_t) |  Draft→StableICU 65
| ucharstrie.h | uint64_t icu::UCharsTrie::getState64() const |  Draft→StableICU 65
| ulistformatter.h | UListFormatter* ulistfmt_openForType(const char*, UListFormatterType, UListFormatterWidth, UErrorCode*) |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterType::ULISTFMT_TYPE_AND |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterType::ULISTFMT_TYPE_OR |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterType::ULISTFMT_TYPE_UNITS |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterWidth::ULISTFMT_WIDTH_NARROW |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterWidth::ULISTFMT_WIDTH_SHORT |  Draft→StableICU 67
| ulistformatter.h | <tt>enum</tt> UListFormatterWidth::ULISTFMT_WIDTH_WIDE |  Draft→StableICU 67
| uloc.h | UEnumeration* uloc_openAvailableByType(ULocAvailableType, UErrorCode*) |  Draft→StableICU 65
| uloc.h | <tt>enum</tt> ULocAvailableType::ULOC_AVAILABLE_DEFAULT |  Draft→StableICU 65
| uloc.h | <tt>enum</tt> ULocAvailableType::ULOC_AVAILABLE_ONLY_LEGACY_ALIASES |  Draft→StableICU 65
| uloc.h | <tt>enum</tt> ULocAvailableType::ULOC_AVAILABLE_WITH_LEGACY_ALIASES |  Draft→StableICU 65
| uniset.h | USet* icu::UnicodeSet::toUSet() |  (missing) | StableICU 4.2
| utrace.h | <tt>enum</tt> UTraceFunctionNumber::UTRACE_UDATA_BUNDLE |  Draft→StableICU 65
| utrace.h | <tt>enum</tt> UTraceFunctionNumber::UTRACE_UDATA_DATA_FILE |  Draft→StableICU 65
| utrace.h | <tt>enum</tt> UTraceFunctionNumber::UTRACE_UDATA_RES_FILE |  Draft→StableICU 65
| utrace.h | <tt>enum</tt> UTraceFunctionNumber::UTRACE_UDATA_START |  Draft→StableICU 65

## Added

Added in ICU 68
  
| File | API | ICU 67 | ICU 68 |
|---|---|---|---|
| dtitvfmt.h | UDisplayContext icu::DateIntervalFormat::getContext(UDisplayContextType, UErrorCode&amp;) const |  (missing) | DraftICU 68
| dtitvfmt.h | void icu::DateIntervalFormat::setContext(UDisplayContext, UErrorCode&amp;) |  (missing) | DraftICU 68
| dtptngen.h | <tt>static</tt> DateTimePatternGenerator* icu::DateTimePatternGenerator::createInstanceNoStdPat(const Locale&amp;, UErrorCode&amp;) |  (missing) | Internal
| fmtable.h | UFormattable* icu::Formattable::toUFormattable() |  (missing) | StableICU 52
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setMaxDistance(const Locale&amp;, const Locale&amp;) |  (missing) | DraftICU 68
| localematcher.h | Builder&amp; icu::LocaleMatcher::Builder::setNoDefaultLocale() |  (missing) | DraftICU 68
| localematcher.h | UBool icu::LocaleMatcher::isMatch(const Locale&amp;, const Locale&amp;, UErrorCode&amp;) const |  (missing) | DraftICU 68
| measunit.h | int32_t icu::MeasureUnit::getOffset() const |  (missing) | Internal
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getCandela() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDessertSpoon() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDessertSpoonImperial() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDot() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDram() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getDrop() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getEarthRadius() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getGrain() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getJigger() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getLumen() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getPinch() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit icu::MeasureUnit::getQuartImperial() |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createCandela(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDessertSpoon(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDessertSpoonImperial(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDot(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDram(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createDrop(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createEarthRadius(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createGrain(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createJigger(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createLumen(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createPinch(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | <tt>static</tt> MeasureUnit* icu::MeasureUnit::createQuartImperial(UErrorCode&amp;) |  (missing) | DraftICU 68
| measunit.h | std::pair&lt; LocalArray&lt; MeasureUnit &gt;, int32_t &gt; icu::MeasureUnit::splitToSingleUnits(UErrorCode&amp;) const |  (missing) | DraftICU 68
| numberformatter.h | Derived icu::number::NumberFormatterSettings&lt; Derived &gt;::usage(StringPiece) const&amp; |  (missing) | DraftICU 68
| numberformatter.h | Derived icu::number::NumberFormatterSettings&lt; Derived &gt;::usage(StringPiece)&amp;&amp; |  (missing) | DraftICU 68
| numberformatter.h | MeasureUnit icu::number::FormattedNumber::getOutputUnit(UErrorCode&amp;) const |  (missing) | DraftICU 68
| numberformatter.h | Usage&amp; icu::number::impl::Usage::operator=(Usage&amp;&amp;) |  (missing) | Internal
| numberformatter.h | Usage&amp; icu::number::impl::Usage::operator=(const Usage&amp;) |  (missing) | Internal
| numberformatter.h | bool icu::number::impl::Usage::isSet() const |  (missing) | Internal
| numberformatter.h | icu::number::impl::Usage::Usage(Usage&amp;&amp;) |  (missing) | Internal
| numberformatter.h | icu::number::impl::Usage::Usage(const Usage&amp;) |  (missing) | Internal
| numberformatter.h | icu::number::impl::Usage::~Usage() |  (missing) | Internal
| numberformatter.h | int16_t icu::number::impl::Usage::length() const |  (missing) | Internal
| numberformatter.h | void icu::number::impl::Usage::set(StringPiece) |  (missing) | Internal
| numberrangeformatter.h | std::pair&lt; StringClass, StringClass &gt; icu::number::FormattedNumberRange::getDecimalNumbers(UErrorCode&amp;) const |  (missing) | DraftICU 68
| plurrule.h | UnicodeString icu::PluralRules::select(const number::FormattedNumberRange&amp;, UErrorCode&amp;) const |  (missing) | DraftICU 68
| plurrule.h | UnicodeString icu::PluralRules::select(const number::impl::UFormattedNumberRangeData*, UErrorCode&amp;) const |  (missing) | Internal
| plurrule.h | int32_t icu::PluralRules::getSamples(const UnicodeString&amp;, FixedDecimal*, int32_t, UErrorCode&amp;) |  (missing) | Internal
| timezone.h | <tt>static</tt> TimeZone* icu::TimeZone::forLocaleOrDefault(const Locale&amp;) |  (missing) | Internal
| ucurr.h | <tt>enum</tt> UCurrNameStyle::UCURR_FORMAL_SYMBOL_NAME |  (missing) | DraftICU 68
| ucurr.h | <tt>enum</tt> UCurrNameStyle::UCURR_VARIANT_SYMBOL_NAME |  (missing) | DraftICU 68
| udateintervalformat.h | UDisplayContext udtitvfmt_getContext(const UDateIntervalFormat*, UDisplayContextType, UErrorCode*) |  (missing) | DraftICU 68
| udateintervalformat.h | void udtitvfmt_setContext(UDateIntervalFormat*, UDisplayContext, UErrorCode*) |  (missing) | DraftICU 68
| umachine.h | <tt>#define</tt> U_DEFINE_FALSE_AND_TRUE |  (missing) | InternalICU 68
| uniset.h | USet* icu::UnicodeSet::toUSet() |  (missing) | StableICU 4.2
| unum.h | <tt>enum</tt> UNumberFormatMinimumGroupingDigits::UNUM_MINIMUM_GROUPING_DIGITS_AUTO |  (missing) | DraftICU 68
| unum.h | <tt>enum</tt> UNumberFormatMinimumGroupingDigits::UNUM_MINIMUM_GROUPING_DIGITS_MIN2 |  (missing) | DraftICU 68
| unumberformatter.h | <tt>enum</tt> UNumberUnitWidth::UNUM_UNIT_WIDTH_FORMAL |  (missing) | DraftICU 68
| unumberformatter.h | <tt>enum</tt> UNumberUnitWidth::UNUM_UNIT_WIDTH_VARIANT |  (missing) | DraftICU 68
| unumberformatter.h | int32_t unumf_resultToDecimalNumber(const UFormattedNumber*, char*, int32_t, UErrorCode*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | UFormattedNumberRange* unumrf_openResult(UErrorCode*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | UNumberRangeFormatter* unumrf_openForSkeletonWithCollapseAndIdentityFallback(const UChar*, int32_t, UNumberRangeCollapse, UNumberRangeIdentityFallback, const char*, UParseError*, UErrorCode*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | UNumberRangeIdentityResult unumrf_resultGetIdentityResult(const UFormattedNumberRange*, UErrorCode*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | const UFormattedValue* unumrf_resultAsValue(const UFormattedNumberRange*, UErrorCode*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | int32_t unumrf_resultGetFirstDecimalNumber(const UFormattedNumberRange*, char*, int32_t, UErrorCode*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | int32_t unumrf_resultGetSecondDecimalNumber(const UFormattedNumberRange*, char*, int32_t, UErrorCode*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | void unumrf_close(UNumberRangeFormatter*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | void unumrf_closeResult(UFormattedNumberRange*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | void unumrf_formatDecimalRange(const UNumberRangeFormatter*, const char*, int32_t, const char*, int32_t, UFormattedNumberRange*, UErrorCode*) |  (missing) | DraftICU 68
| unumberrangeformatter.h | void unumrf_formatDoubleRange(const UNumberRangeFormatter*, double, double, UFormattedNumberRange*, UErrorCode*) |  (missing) | DraftICU 68
| upluralrules.h | int32_t uplrules_selectForRange(const UPluralRules*, const struct UFormattedNumberRange*, UChar*, int32_t, UErrorCode*) |  (missing) | DraftICU 68

## Other

Other existing drafts in ICU 68

| File | API | ICU 67 | ICU 68 |
|---|---|---|---|
| bytestream.h |  void icu::ByteSink::AppendU8(const char*, int32_t) | DraftICU 67 | 
| bytestream.h |  void icu::ByteSink::AppendU8(const char8_t*, int32_t) | DraftICU 67 | 
| dtptngen.h |  UDateFormatHourCycle icu::DateTimePatternGenerator::getDefaultHourCycle(UErrorCode&amp;) const | DraftICU 67 | 
| localematcher.h |  Builder&amp; icu::LocaleMatcher::Builder::setDirection(ULocMatchDirection) | DraftICU 67 | 
| localematcher.h |  <tt>enum</tt> ULocMatchDirection::ULOCMATCH_DIRECTION_ONLY_TWO_WAY | DraftICU 67 | 
| localematcher.h |  <tt>enum</tt> ULocMatchDirection::ULOCMATCH_DIRECTION_WITH_ONE_WAY | DraftICU 67 | 
| locid.h |  void icu::Locale::canonicalize(UErrorCode&amp;) | DraftICU 67 | 
| measfmt.h |  void icu::MeasureFormat::parseObject(const UnicodeString&amp;, Formattable&amp;, ParsePosition&amp;) const | DraftICU 53 | 
| measunit.h |  MeasureUnit icu::MeasureUnit::product(const MeasureUnit&amp;, UErrorCode&amp;) const | DraftICU 67 | 
| measunit.h |  MeasureUnit icu::MeasureUnit::reciprocal(UErrorCode&amp;) const | DraftICU 67 | 
| measunit.h |  MeasureUnit icu::MeasureUnit::withDimensionality(int32_t, UErrorCode&amp;) const | DraftICU 67 | 
| measunit.h |  MeasureUnit icu::MeasureUnit::withSIPrefix(UMeasureSIPrefix, UErrorCode&amp;) const | DraftICU 67 | 
| measunit.h |  MeasureUnit&amp; icu::MeasureUnit::operator=(MeasureUnit&amp;&amp;) noexcept | DraftICU 67 | 
| measunit.h |  UMeasureSIPrefix icu::MeasureUnit::getSIPrefix(UErrorCode&amp;) const | DraftICU 67 | 
| measunit.h |  UMeasureUnitComplexity icu::MeasureUnit::getComplexity(UErrorCode&amp;) const | DraftICU 67 | 
| measunit.h |  const char* icu::MeasureUnit::getIdentifier() const | DraftICU 67 | 
| measunit.h |  icu::MeasureUnit::MeasureUnit(MeasureUnit&amp;&amp;) noexcept | DraftICU 67 | 
| measunit.h |  int32_t icu::MeasureUnit::getDimensionality(UErrorCode&amp;) const | DraftICU 67 | 
| measunit.h |  <tt>static</tt> MeasureUnit icu::MeasureUnit::forIdentifier(StringPiece, UErrorCode&amp;) | DraftICU 67 | 
| stringpiece.h |  icu::StringPiece::StringPiece(const char8_t*) | DraftICU 67 | 
| stringpiece.h |  icu::StringPiece::StringPiece(const char8_t*, int32_t) | DraftICU 67 | 
| stringpiece.h |  icu::StringPiece::StringPiece(const std::u8string&amp;) | DraftICU 67 | 
| stringpiece.h |  icu::StringPiece::StringPiece(std::nullptr_t) | DraftICU 67 | 
| stringpiece.h |  int32_t icu::StringPiece::compare(StringPiece) | DraftICU 67 | 
| stringpiece.h |  int32_t icu::StringPiece::find(StringPiece, int32_t) | DraftICU 67 | 
| stringpiece.h |  void icu::StringPiece::set(const char8_t*) | DraftICU 67 | 
| stringpiece.h |  void icu::StringPiece::set(const char8_t*, int32_t) | DraftICU 67 | 
| udat.h |  <tt>enum</tt> UDateFormatHourCycle::UDAT_HOUR_CYCLE_11 | DraftICU 67 | 
| udat.h |  <tt>enum</tt> UDateFormatHourCycle::UDAT_HOUR_CYCLE_12 | DraftICU 67 | 
| udat.h |  <tt>enum</tt> UDateFormatHourCycle::UDAT_HOUR_CYCLE_23 | DraftICU 67 | 
| udat.h |  <tt>enum</tt> UDateFormatHourCycle::UDAT_HOUR_CYCLE_24 | DraftICU 67 | 
| udateintervalformat.h |  void udtitvfmt_formatCalendarToResult(const UDateIntervalFormat*, UCalendar*, UCalendar*, UFormattedDateInterval*, UErrorCode*) | DraftICU 67 | 
| udateintervalformat.h |  void udtitvfmt_formatToResult(const UDateIntervalFormat*, UDate, UDate, UFormattedDateInterval*, UErrorCode*) | DraftICU 67 | 
| udatpg.h |  UDateFormatHourCycle udatpg_getDefaultHourCycle(const UDateTimePatternGenerator*, UErrorCode*) | DraftICU 67 | 
| uregex.h |  <tt>enum</tt> URegexpFlag::UREGEX_CANON_EQ | DraftICU 2.4 | 
| utrace.h |  <tt>enum</tt> UTraceFunctionNumber::UTRACE_UBRK_CREATE_BREAK_ENGINE | DraftICU 67 | 
| utrace.h |  <tt>enum</tt> UTraceFunctionNumber::UTRACE_UBRK_CREATE_CHARACTER | DraftICU 67 | 
| utrace.h |  <tt>enum</tt> UTraceFunctionNumber::UTRACE_UBRK_CREATE_LINE | DraftICU 67 | 
| utrace.h |  <tt>enum</tt> UTraceFunctionNumber::UTRACE_UBRK_CREATE_SENTENCE | DraftICU 67 | 
| utrace.h |  <tt>enum</tt> UTraceFunctionNumber::UTRACE_UBRK_CREATE_TITLE | DraftICU 67 | 
| utrace.h |  <tt>enum</tt> UTraceFunctionNumber::UTRACE_UBRK_CREATE_WORD | DraftICU 67 | 
| utrace.h |  <tt>enum</tt> UTraceFunctionNumber::UTRACE_UBRK_START | DraftICU 67 | 

## Simplifications

This section shows cases where the signature was "simplified" for the sake of comparison. The simplified form is in bold, followed by
    all possible variations in "original" form.


## Colophon

Contents generated by StableAPI tool on Fri Oct 23 11:32:42 PDT 2020

Copyright © 2019 and later: Unicode, Inc. and others.
License & terms of use: http://www.unicode.org/copyright.html
  