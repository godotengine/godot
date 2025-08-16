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

# Some arbitrary string that should never show up by accident.  If it does, then
# shame on  you.
const ARG_NOT_SET = '_*_argument_*_is_*_not_set_*_'

# This hash holds the objects that are being watched, the signals that are being
# watched, and an array of arrays that contains arguments that were passed
# each time the signal was emitted.
#
# For example:
#	_watched_signals => {
#		ref1 => {
#			'signal1' => [[], [], []],
#			'signal2' => [[p1, p2]],
#			'signal3' => [[p1]]
#		},
#		ref2 => {
#			'some_signal' => [],
#			'other_signal' => [[p1, p2, p3], [p1, p2, p3], [p1, p2, p3]]
#		}
#	}
#
# In this sample:
#	- signal1 on the ref1 object was emitted 3 times and each time, zero
#	parameters were passed.
#	- signal3 on ref1 was emitted once and passed a single parameter
#	- some_signal on ref2 was never emitted.
#	- other_signal on ref2 was emitted 3 times, each time with 3 parameters.
var _watched_signals = {}
var _lgr = GutUtils.get_logger()

func _add_watched_signal(obj, name):
	# SHORTCIRCUIT - ignore dupes
	if(_watched_signals.has(obj) and _watched_signals[obj].has(name)):
		return

	if(!_watched_signals.has(obj)):
		_watched_signals[obj] = {name:[]}
	else:
		_watched_signals[obj][name] = []
	obj.connect(name,Callable(self,'_on_watched_signal').bind(obj,name))

# This handles all the signals that are watched.  It supports up to 9 parameters
# which could be emitted by the signal and the two parameters used when it is
# connected via watch_signal.  I chose 9 since you can only specify up to 9
# parameters when dynamically calling a method via call (per the Godot
# documentation, i.e. some_object.call('some_method', 1, 2, 3...)).
#
# Based on the documentation of emit_signal, it appears you can only pass up
# to 4 parameters when firing a signal.  I haven't verified this, but this should
# future proof this some if the value ever grows.
func _on_watched_signal(arg1=ARG_NOT_SET, arg2=ARG_NOT_SET, arg3=ARG_NOT_SET, \
						arg4=ARG_NOT_SET, arg5=ARG_NOT_SET, arg6=ARG_NOT_SET, \
						arg7=ARG_NOT_SET, arg8=ARG_NOT_SET, arg9=ARG_NOT_SET, \
						arg10=ARG_NOT_SET, arg11=ARG_NOT_SET):
	var args = [arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11]

	# strip off any unused vars.
	var idx = args.size() -1
	while(str(args[idx]) == ARG_NOT_SET):
		args.remove_at(idx)
		idx -= 1

	# retrieve object and signal name from the array and remove_at them.  These
	# will always be at the end since they are added when the connect happens.
	var signal_name = args[args.size() -1]
	args.pop_back()
	var object = args[args.size() -1]
	args.pop_back()

	if(_watched_signals.has(object)):
		_watched_signals[object][signal_name].append(args)
	else:
		_lgr.error(str("signal_watcher._on_watched_signal:  Got signal for unwatched object:  ", object, '::', signal_name))

# This parameter stuff should go into test.gd not here.  This thing works
# just fine the way it is.
func _obj_name_pair(obj_or_signal, signal_name=null):
	var to_return = {
		'object' : obj_or_signal,
		'signal_name' : signal_name
	}
	if(obj_or_signal is Signal):
		to_return.object =  obj_or_signal.get_object()
		to_return.signal_name = obj_or_signal.get_name()

	return to_return


func does_object_have_signal(object, signal_name):
	var signals = object.get_signal_list()
	for i in range(signals.size()):
		if(signals[i]['name'] == signal_name):
			return true
	return false

func watch_signals(object):
	var signals = object.get_signal_list()
	for i in range(signals.size()):
		_add_watched_signal(object, signals[i]['name'])

func watch_signal(object, signal_name):
	var did = false
	if(does_object_have_signal(object, signal_name)):
		_add_watched_signal(object, signal_name)
		did = true
	else:
		GutUtils.get_logger().warn(str(object, ' does not have signal ', signal_name))
	return did

func get_emit_count(object, signal_name):
	var to_return = -1
	if(is_watching(object, signal_name)):
		to_return = _watched_signals[object][signal_name].size()
	return to_return

func did_emit(object, signal_name=null):
	var vals = _obj_name_pair(object, signal_name)
	var did = false
	if(is_watching(vals.object, vals.signal_name)):
		did = get_emit_count(vals.object, vals.signal_name) != 0
	return did

func print_object_signals(object):
	var list = object.get_signal_list()
	for i in range(list.size()):
		print(list[i].name, "\n  ", list[i])

func get_signal_parameters(object, signal_name, index=-1):
	var params = null
	if(is_watching(object, signal_name)):
		var all_params = _watched_signals[object][signal_name]
		if(all_params.size() > 0):
			if(index == -1):
				index = all_params.size() -1
			params = all_params[index]
	return params

func is_watching_object(object):
	return _watched_signals.has(object)

func is_watching(object, signal_name):
	return _watched_signals.has(object) and _watched_signals[object].has(signal_name)

func clear():
	for obj in _watched_signals:
		if(GutUtils.is_not_freed(obj)):
			for signal_name in _watched_signals[obj]:
				obj.disconnect(signal_name, Callable(self,'_on_watched_signal'))
	_watched_signals.clear()

# Returns a list of all the signal names that were emitted by the object.
# If the object is not being watched then an empty list is returned.
func get_signals_emitted(obj):
	var emitted = []
	if(is_watching_object(obj)):
		for signal_name in _watched_signals[obj]:
			if(_watched_signals[obj][signal_name].size() > 0):
				emitted.append(signal_name)

	return emitted


func get_signal_summary(obj):
	var emitted = {}
	if(is_watching_object(obj)):
		for signal_name in _watched_signals[obj]:
			if(_watched_signals[obj][signal_name].size() > 0):
				# maybe this could return parameters if any were sent.  should
				# have an empty list if no parameters were ever sent to the
				# signal.  Or this all just gets moved into print_signal_summary
				# since this wouldn't be that useful without more data in the
				# summary.
				var entry = {
					emit_count = get_emit_count(obj, signal_name)
				}
				emitted[signal_name] = entry

	return emitted


func print_signal_summary(obj):
	if(!is_watching_object(obj)):
		var msg = str('Not watching signals for ', obj)
		GutUtils.get_logger().warn(msg)
		return

	var summary = get_signal_summary(obj)
	print(obj, '::Signals')
	var sorted = summary.keys()
	sorted.sort()
	for key in sorted:
		print(' -  ', key, ' x ', summary[key].emit_count)
