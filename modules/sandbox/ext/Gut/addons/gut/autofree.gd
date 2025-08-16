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
# Class used to keep track of objects to be freed and utilities to free them.
# ##############################################################################
var _to_free = []
var _to_queue_free = []

func add_free(thing):
	if(typeof(thing) == TYPE_OBJECT):
		if(!thing is RefCounted):
			_to_free.append(thing)

func add_queue_free(thing):
	_to_queue_free.append(thing)

func get_queue_free_count():
	return _to_queue_free.size()

func get_free_count():
	return _to_free.size()

func free_all():
	for i in range(_to_free.size()):
		if(is_instance_valid(_to_free[i])):
			_to_free[i].free()
	_to_free.clear()

	for i in range(_to_queue_free.size()):
		if(is_instance_valid(_to_queue_free[i])):
			_to_queue_free[i].queue_free()
	_to_queue_free.clear()
