var m1 # No init.
var m2 = 22 # Init.
var m3: String # No init, typed.
var m4: String = "44" # Init, typed.


func test():
	var loc5 # No init, local.
	var loc6 = 66 # Init, local.
	var loc7: String # No init, typed.
	var loc8: String = "88" # Init, typed.

	m1 = 11
	m3 = "33"

	loc5 = 55
	loc7 = "77"

	prints(m1, m2, m3, m4, loc5, loc6, loc7, loc8)
	print("OK")
