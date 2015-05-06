T-REX 1.3 http://tiny-rex.sourceforge.net
----------------------------------------------------------------------
	T-Rex a tiny regular expression library

	Copyright (C) 2003-2006 Alberto Demichelis

	This software is provided 'as-is', without any express 
	or implied warranty. In no event will the authors be held 
	liable for any damages arising from the use of this software.

	Permission is granted to anyone to use this software for 
	any purpose, including commercial applications, and to alter
	it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented;
		you must not claim that you wrote the original software.
		If you use this software in a product, an acknowledgment
		in the product documentation would be appreciated but
		is not required.

		2. Altered source versions must be plainly marked as such,
		and must not be misrepresented as being the original software.

		3. This notice may not be removed or altered from any
		source distribution.
		
----------------------------------------------------------------------
TRex implements the following expressions

\	Quote the next metacharacter
^	Match the beginning of the string
.	Match any character
$	Match the end of the string
|	Alternation
()	Grouping (creates a capture)
[]	Character class  

==GREEDY CLOSURES==
*	   Match 0 or more times
+	   Match 1 or more times
?	   Match 1 or 0 times
{n}    Match exactly n times
{n,}   Match at least n times
{n,m}  Match at least n but not more than m times  

==ESCAPE CHARACTERS==
\t		tab                   (HT, TAB)
\n		newline               (LF, NL)
\r		return                (CR)
\f		form feed             (FF)

==PREDEFINED CLASSES==
\l		lowercase next char
\u		uppercase next char
\a		letters
\A		non letters
\w		alphanimeric [0-9a-zA-Z]
\W		non alphanimeric
\s		space
\S		non space
\d		digits
\D		non nondigits
\x		exadecimal digits
\X		non exadecimal digits
\c		control charactrs
\C		non control charactrs
\p		punctation
\P		non punctation
\b		word boundary
\B		non word boundary

----------------------------------------------------------------------
API DOC
----------------------------------------------------------------------
TRex *trex_compile(const TRexChar *pattern,const TRexChar **error);

compiles an expression and returns a pointer to the compiled version.
in case of failure returns NULL.The returned object has to be deleted
through the function trex_free().

pattern
	a pointer to a zero terminated string containing the pattern that 
	has to be compiled.
error
	apointer to a string pointer that will be set with an error string
	in case of failure.
	
----------------------------------------------------------------------
void trex_free(TRex *exp)

deletes a expression structure created with trex_compile()

exp
	the expression structure that has to be deleted

----------------------------------------------------------------------
TRexBool trex_match(TRex* exp,const TRexChar* text)

returns TRex_True if the string specified in the parameter text is an
exact match of the expression, otherwise returns TRex_False.

exp
	the compiled expression
text
	the string that has to be tested
	
----------------------------------------------------------------------
TRexBool trex_search(TRex* exp,const TRexChar* text, const TRexChar** out_begin, const TRexChar** out_end)

searches the first match of the expressin in the string specified in the parameter text.
if the match is found returns TRex_True and the sets out_begin to the beginning of the
match and out_end at the end of the match; otherwise returns TRex_False.

exp
	the compiled expression
text
	the string that has to be tested
out_begin
	a pointer to a string pointer that will be set with the beginning of the match
out_end
	a pointer to a string pointer that will be set with the end of the match

----------------------------------------------------------------------
TREX_API TRexBool trex_searchrange(TRex* exp,const TRexChar* text_begin,const TRexChar* text_end,const TRexChar** out_begin, const TRexChar** out_end)

searches the first match of the expressin in the string delimited 
by the parameter text_begin and text_end.
if the match is found returns TRex_True and the sets out_begin to the beginning of the
match and out_end at the end of the match; otherwise returns TRex_False.

exp
	the compiled expression
text_begin
	a pointer to the beginnning of the string that has to be tested
text_end
	a pointer to the end of the string that has to be tested
out_begin
	a pointer to a string pointer that will be set with the beginning of the match
out_end
	a pointer to a string pointer that will be set with the end of the match
	
----------------------------------------------------------------------
int trex_getsubexpcount(TRex* exp)

returns the number of sub expressions matched by the expression

exp
	the compiled expression

---------------------------------------------------------------------
TRexBool trex_getsubexp(TRex* exp, int n, TRexMatch *submatch)

retrieve the begin and and pointer to the length of the sub expression indexed
by n. The result is passed trhough the struct TRexMatch:

typedef struct {
	const TRexChar *begin;
	int len;
} TRexMatch;

the function returns TRex_True if n is valid index otherwise TRex_False.

exp
	the compiled expression
n
	the index of the submatch
submatch
	a pointer to structure that will store the result
	
this function works also after a match operation has been performend.
	
