class_name GutOutputTest
extends GutInternalTester
## Base class for tests that require the output be manually inspected.  This
## has not been applied to everything yet.


func look_for(text):
	_lgr.log(str("Look for:  ", text), _lgr.fmts.bold)
	pass_test("check for output")



func should_not_error():
	_lgr.log("There should not be any errors in this test.", _lgr.fmts.bold)
	pass_test("no errors")



func should_error(text=""):
	_lgr.log(str("There should be ERRORs:  ", text), _lgr.fmts.bold)
	pass_test("check for errors")



func should_warn(text=""):
	_lgr.log(str("There should be Warnings:  ", text), _lgr.fmts.bold)
	pass_test("check for warnings")


func just_look_at_it(text=""):
	if(text == ""):
		text = "Look at the text, see if you like it or not, or whatever"
	_lgr.log(text, _lgr.fmts.bold)
	pass_test("")