## Creates parameter structures for parameterized tests.
##
## This is a static class accessible in a [GutTest] script through
## [member GutTest.ParameterFactory].  It contains methods for constructing parameters to be
## used in parameterized tests.  It currently only has one, if you have anyu
## ideas for more, make an issue.  More of them would be great since I prematurely
## decided to make this static class and it has such a long name.  I'd feel a lot
## better about it if there was more in here.
## [br]
## Additional Helper Ideas?[br]
## [li]File.  IDK what it would look like.  csv maybe.[/li]
## [li]Random values within a range?[/li]
## [li]All int values in a range or add an optioanal step.[/li]




## Creates an array of dictionaries.  It pairs up the names array with each set
## of values in values.  If more names than values are specified then the missing
## values will be filled with nulls.  If more values than names are specified
## those values will be ignored.
##
## Example:
##[codeblock]
## create_named_parameters(['a', 'b'], [[1, 2], ['one', 'two']]) returns
##    [{a:1, b:2}, {a:'one', b:'two'}]
##[/codeblock]
## [br]
## This allows you to increase readability of your parameterized tests:
## [br]
##[codeblock]
## var params = create_named_parameters(['a', 'b'], [[1, 2], ['one', 'two']])
## func test_foo(p = use_parameters(params)):
##    assert_eq(p.a, p.b)
##[/codeblock]
## [br]
## Parameters:[br]
##[li]names:  an array of names to be used as keys in the dictionaries[/li]
##[li]values:  an array of arrays of values.[/li]
static func named_parameters(names, values):
	var named = []
	for i in range(values.size()):
		var entry = {}

		var parray = values[i]
		if(typeof(parray) != TYPE_ARRAY):
			parray = [values[i]]

		for j in range(names.size()):
			if(j >= parray.size()):
				entry[names[j]] = null
			else:
				entry[names[j]] = parray[j]
		named.append(entry)

	return named



# ##############################################################################
#(G)odot (U)nit (T)est class
#
# ##############################################################################
# The MIT License (MIT)
# =====================
#
# Copyright (c) 2025 Tom "Butch" Wesley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ##############################################################################
# This is the home for all parameter creation helpers.  These functions should
# all return an array of values to be used as parameters for parameterized
# tests.
# ##############################################################################