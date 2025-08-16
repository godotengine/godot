# ------------------------------------------------------------------------------
# This is used to track the change in orphans over different intervals.
# You use this by adding a counter at the start of an interval and then
# using get_orphans_since to find out how many orphans have been created since
# that counter was added.
#
# For example, when a test starts, gut adds a counter for "test" which
# creates/sets the counter's value to the current orphan count.  At the end of
# the test GUT uses get_orphans_since("test") to find out how many orphans
# were created by the test.
# ------------------------------------------------------------------------------
var _counters = {}

func orphan_count():
	return Performance.get_monitor(Performance.OBJECT_ORPHAN_NODE_COUNT)

func add_counter(name):
	_counters[name] = orphan_count()

# Returns the number of orphans created since add_counter was last called for
# the name.  Returns -1 to avoid blowing up with an invalid name but still
# be somewhat visible that we've done something wrong.
func get_orphans_since(name):
	return orphan_count() - _counters[name] if _counters.has(name) else -1

func get_count(name):
	return _counters.get(name, -1)

func print_orphans(name, lgr):
	var count = get_orphans_since(name)

	if(count > 0):
		var o = 'orphan'
		if(count > 1):
			o = 'orphans'
		lgr.orphan(str(count, ' new ', o, ' in ', name, '.'))

func print_all():
	var msg = str("Total Orphans ", orphan_count(), "\n", JSON.stringify(_counters, "    "))
	print(msg)



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
# This is a utility for tracking changes in the orphan count.  Each time
# add_counter is called it adds/resets the value in the dictionary to the
# current number of orphans.  Each call to get_counter will return the change
# in orphans since add_counter was last called.
# ##############################################################################