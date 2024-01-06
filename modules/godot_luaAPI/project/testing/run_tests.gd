extends Node2D

var time_elapsed = 0

var done = false
var testCount = 0
var failures = 0

var tests: Array[UnitTest]
var doneTests: Array[UnitTest]
var currentTest: UnitTest

var logFile: FileAccess

func logPrint(msg: String):
	logFile.store_line(msg)
	print(msg)

func sort_tests(a, b):
	if a.id < b.id:
		return true
	return false

func _ready():
	logFile = FileAccess.open("res://log.txt", FileAccess.WRITE_READ)
	logPrint("LuaAPI Testing framework for v2-alpha")
	logPrint("Starting time: %s\n" % str(time_elapsed))
	load_tests()
	for test in tests:
		add_child(test)
		test.set_process(false)
		test._ready()
	tests.sort_custom(sort_tests)
	testCount = tests.size()
	logPrint("Loaded %d tests\n" % testCount)

func _process(delta):
	if done:
		get_tree().quit(failures)
		return

	time_elapsed += delta
	var doneCount = 0
	if currentTest == null:
		currentTest = tests.pop_back()

	if not currentTest.done:
		currentTest._process(delta)

	if currentTest.done:
		logPrint("Test #%d: " % currentTest.id + currentTest.testName + " has finished.")
		doneTests.append(currentTest)
		currentTest = null

	if doneTests.size() == testCount:
		finish()

func finish():
	logPrint("\nFinished!")
	logPrint("End time: %s\n" % str(time_elapsed))
	logPrint("Report:\n")
	for test in doneTests:
		logPrint("Test Name: %s" % test.testName)
		logPrint("-------------------------------")
		logPrint("Test id(start order): %d" % test.id)
		logPrint("Test Description:")
		logPrint(test.testDescription)
		logPrint("Frames: %d" % test.frames)
		logPrint("Time: %s" % str(test.time))

		if test.status:
			logPrint("Test finished with no errors.")
			logPrint("-------------------------------\n")
			test._finalize()
			test.free()
			continue

		failures += 1

		logPrint("Test finished with %d errors." % test.errors.size())
		for err in test.errors:
			logPrint("\nERROR %d: " % err.type + err.message)
		logPrint("-------------------------------\n")
		test._finalize()
		test.free()

	doneTests.clear()
	logPrint("%d/" % failures + "%d tests failed." % testCount)
	done=true


func load_tests():
	var dir = DirAccess.open("res://testing/tests")
	dir.list_dir_begin()

	while true:
		var file = dir.get_next()
		if file == "":
			break
		elif not file.begins_with(".") and  file.ends_with(".gd"):
			var test = load("res://testing/tests/%s" % file).new()
			tests.append(test)

	dir.list_dir_end()
