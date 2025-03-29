enum CustomEnum { ENUM_VALUE = 3 }

func test():
	print("non-formatted string {not a slot}")
	print(f"without slot")
	print(f"") # empty string
	print(f"with double quote ' delimiters")
	print(f'with single quote " delimiters')
	print(f"with bool literal {true} slot")
	print(f"with int literal {4} slot")
	print(f"with float literal {1.234} slot")
	print(f"with enum literal {CustomEnum.ENUM_VALUE} slot")
	print(f"with string literal {"foo"} slot")

	var bool_var: bool = true
	var int_var: int = 5
	var float_var: float = 5.234
	var enum_var: CustomEnum = CustomEnum.ENUM_VALUE
	var string_var: String = "foo"

	print(f"with bool variable {bool_var}")
	print(f"with int variable {int_var} slot")
	print(f"with float variable {float_var} slot")
	print(f"with enum variable {enum_var} slot")
	print(f"with string variable {string_var} slot")

	print(f"with bool expression {2 == 3+4}")
	print(f"with int expression {7*5-4} slot")
	print(f"with float expression {4.3*2 + 1.1} slot")
	print(f"with double quote string double quote expression {"foo"+"bar"} slot")
	print(f"with single quote string single quote expression {'foo'+'bar'} slot")
	print(f"with double quote string single quote expression {"foo"+'bar'} slot")
	print(f"with single quote string double quote expression {'foo'+"bar"} slot")

	print(f"with int function {double_me(5)} slot")
	print(f"with {1+2} multiple {1.1+2.2} slots")
	print(f"with slot { 1 +	2 } with whitespace inside")

	print(f"with {{escaped}} and not escaped {2+3} slots")
	print(fr"raw with {{escaped}} and not escaped {2+3} slots")
	print("non-formatted string doesn't honor {{escaped}} braces")
	print(f"with nested { f"formatted {2+3}" + " " + f"string {6*2}" } slots")

	print(fr"raw with raw \t chars and {2+3} slots")
	print(f"""multiline with newlines
and {2+3} slots""")

	# Any code tokens, including newlines, can go within a slot, even in
	# a non-multiline formatted string (probably not worth advertising this).
	print(f"with newlines and {
		"comments" # this is a comment
	} within a slot")

	print(fr'''raw and multiline with raw \t chars, carriage-returns
and {2+3} slots, using alternate quote char''')

	print(f"Handle slot with embedded braces { {"a":1, "b":2, "c":3}["b"]}")

func double_me(x:int)->int:
	return x*2
