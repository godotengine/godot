/*
******************************************************************************
* Copyright (C) 2014, International Business Machines
* Corporation and others.  All Rights Reserved.
******************************************************************************
* simplepatternformatter.h
*/

#ifndef __SIMPLEPATTERNFORMATTER_H__
#define __SIMPLEPATTERNFORMATTER_H__ 

#define EXPECTED_PLACEHOLDER_COUNT 3

#include "cmemory.h"
#include "unicode/utypes.h"
#include "unicode/unistr.h"

U_NAMESPACE_BEGIN

class SimplePatternFormatterPlaceholderValues;

struct PlaceholderInfo {
  int32_t id;
  int32_t offset;
};

/**
 * Compiled version of a pattern string such as "{1} was born in {0}".
 * <p>
 * Using SimplePatternFormatter is both faster and safer than adhoc replacement.
 * They are faster because they are precompiled; they are safer because they
 * account for curly braces escaped by apostrophe (').
 * 
 * Placeholders are of the form \{[0-9]+\}. If a curly brace is preceded
 * by a single quote, it becomes a curly brace instead of the start of a
 * placeholder. Two single quotes resolve to one single quote. 
 * <p>
 * Example:
 * <pre>
 * SimplePatternFormatter fmt("{1} '{born} in {0}");
 * UnicodeString result;
 * UErrorCode status = U_ZERO_ERROR;
 * // Evaluates to: "paul {born} in england"
 * fmt.format("england", "paul", result, status);
 * </pre>
 */
class U_COMMON_API SimplePatternFormatter : public UMemory {
public:
    /**
     * Default constructor
     */
    SimplePatternFormatter();

    /**
     * Construct from a pattern. Will never fail if pattern has three or
     * fewer placeholders in it.
     */
    explicit SimplePatternFormatter(const UnicodeString& pattern);

    /**
     * Copy constructor.
     */
    SimplePatternFormatter(const SimplePatternFormatter& other);

    /**
     * Assignment operator
     */
    SimplePatternFormatter &operator=(const SimplePatternFormatter& other);

    /**
     * Destructor.
     */
    ~SimplePatternFormatter();

    /**
     * Compiles pattern and makes this object represent pattern.
     *
     * Returns TRUE on success; FALSE on failure. Will not fail if
     * there are three or fewer placeholders in pattern. May fail with
     * U_MEMORY_ALLOCATION_ERROR if there are more than three placeholders.
     */
    UBool compile(const UnicodeString &pattern, UErrorCode &status);

    /**
     * Returns (maxPlaceholderId + 1). For example
     * <code>SimplePatternFormatter("{0} {2}").getPlaceholderCount()
     * evaluates to 3.
     * Callers use this function to find out how many values this object
     * expects when formatting.
     */
    int32_t getPlaceholderCount() const {
        return placeholderCount;
    }

    /**
     * Returns this pattern with none of the placeholders.
     */
    const UnicodeString &getPatternWithNoPlaceholders() const {
        return noPlaceholders;
    }

    /**
     * Formats given value. arg0 cannot be appendTo.
     */
    UnicodeString &format(
            const UnicodeString &args0,
            UnicodeString &appendTo,
            UErrorCode &status) const;
    
    /**
     * Formats given values. Neither arg0 nor arg1 can be appendTo.
     */
    UnicodeString &format(
            const UnicodeString &args0,
            const UnicodeString &args1,
            UnicodeString &appendTo,
            UErrorCode &status) const;
    
    /**
     * Formats given values. Neither arg0, arg1, nor arg2 can be appendTo.
     */
    UnicodeString &format(
            const UnicodeString &args0,
            const UnicodeString &args1,
            const UnicodeString &args2,
            UnicodeString &appendTo,
            UErrorCode &status) const;
    
    /**
     * Formats given values.
     *
     * The caller retains ownership of all pointers.
     * @param placeholderValues 1st one corresponds to {0}; 2nd to {1};
     *  3rd to {2} etc. If any of these point to appendTo, this method
     *  sets status to U_ILLEGAL_ARGUMENT_ERROR.
     * @param placeholderValueCount the number of placeholder values
     *  must be at least large enough to provide values for all placeholders
     *  in this object. Otherwise status set to U_ILLEGAL_ARGUMENT_ERROR.
     * @param appendTo resulting string appended here.
     * @param offsetArray The offset of each placeholder value in appendTo
     *  stored here. The first value gets the offset of the value for {0};
     *  the 2nd for {1}; the 3rd for {2} etc. -1 means that the corresponding
     *  placeholder does not exist in this object. If caller is not
     *  interested in offsets, it may pass NULL and 0 for the length.
     * @param offsetArrayLength the size of offsetArray. If less than
     *  placeholderValueCount only the first offsets get recorded. If
     * greater than placeholderValueCount, then extra values in offset
     * array are set to -1.
     * @param status any error stored here.
     */
    UnicodeString &formatAndAppend(
            const UnicodeString * const *placeholderValues,
            int32_t placeholderValueCount,
            UnicodeString &appendTo,
            int32_t *offsetArray,
            int32_t offsetArrayLength,
            UErrorCode &status) const;

    /**
     * Formats given values.
     *
     * The caller retains ownership of all pointers.
     * @param placeholderValues 1st one corresponds to {0}; 2nd to {1};
     *  3rd to {2} etc. May include pointer to result in which case
     *  the previous value of result is used for the corresponding
     *  placeholder.
     * @param placeholderValueCount the number of placeholder values
     *  must be at least large enough to provide values for all placeholders
     *  in this object. Otherwise status set to U_ILLEGAL_ARGUMENT_ERROR.
     * @param result resulting string stored here overwriting any previous
     *   value.
     * @param offsetArray The offset of each placeholder value in result
     *  stored here. The first value gets the offset of the value for {0};
     *  the 2nd for {1}; the 3rd for {2} etc. -1 means that the corresponding
     *  placeholder does not exist in this object. If caller is not
     *  interested in offsets, it may pass NULL and 0 for the length.
     * @param offsetArrayLength the size of offsetArray. If less than
     *  placeholderValueCount only the first offsets get recorded. If
     * greater than placeholderValueCount, then extra values in offset
     * array are set to -1.
     * @param status any error stored here.
     */
    UnicodeString &formatAndReplace(
            const UnicodeString * const *placeholderValues,
            int32_t placeholderValueCount,
            UnicodeString &result,
            int32_t *offsetArray,
            int32_t offsetArrayLength,
            UErrorCode &status) const;
private:
    UnicodeString noPlaceholders;
    MaybeStackArray<PlaceholderInfo, 3> placeholders;
    int32_t placeholderSize;
    int32_t placeholderCount;
    UBool firstPlaceholderReused;

    // A Placeholder value that is the same as appendTo is treated as the
    // empty string.
    UnicodeString &formatAndAppend(
            const SimplePatternFormatterPlaceholderValues &placeholderValues,
            UnicodeString &appendTo,
            int32_t *offsetArray,
            int32_t offsetArrayLength) const;

    // Returns the placeholder at the beginning of this pattern
    // (e.g 3 for placeholder {3}). Returns -1 if the beginning of pattern
    // is text or if the placeholder at the beginning of this pattern
    // is used again in the middle of the pattern.
    int32_t getUniquePlaceholderAtStart() const;
    
    // ensureCapacity ensures that the capacity of the placeholders array
    // is desiredCapacity. If ensureCapacity must resize the placeholders
    // array, the first placeholderSize elements stay in the array. Note
    // that ensureCapcity NEVER changes the value of placeholderSize only
    // the capacity of the placeholders array.
    // If there is no memory allocation error when resizing, this
    // function returns desiredCapacity. If there is a memory allocation
    // error, this function leaves the placeholders array unchanged and
    // returns the smaller, old capacity. ensureCapacity resizes only if
    // the current capacity of placeholders array is less than desiredCapacity.
    // Otherwise, it leaves the placeholders array unchanged. If caller
    // specifies an allocation size, then it must be at least as large as
    // desiredCapacity. In that case, if ensureCapacity resizes, it will
    // allocate allocationSize spots instead of desiredCapacity spots in
    // the array. If caller is calling ensureCapacity in a loop while adding
    // elements, it is recommended that it use an allocationSize of
    // approximately twice desiredCapacity to avoid memory allocation with
    // every call to ensureCapacity.
    int32_t ensureCapacity(int32_t desiredCapacity, int32_t allocationSize=0);

    // Records the offset of an individual placeholder in the noPlaceholders
    // string.
    UBool addPlaceholder(int32_t id, int32_t offset);
};

U_NAMESPACE_END

#endif
