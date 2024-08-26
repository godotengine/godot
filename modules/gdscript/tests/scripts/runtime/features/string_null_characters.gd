func test():
	print("\0" != null)
	print("\0" != "")
	print("\0" == "\u0000")
	print("\0" == String.chr(0))
	print("\0".length() == 1)

	var text := ""
	for n in 10:
		text += "\0"
	print(text.length() == 10)

	text = "string\0split\u0000by%snull" % String.chr(0)
	var split := text.split("\0")
	print(split)
	var join := "\u0000".join(split)
	print(join == text)

	var escape := text.c_escape()
	print(escape)
	join = escape.c_unescape()
	print(join == text)

	var buffer := text.to_ascii_buffer()
	print(buffer)
	join = buffer.get_string_from_ascii()
	print(join != text)

	buffer = text.to_utf8_buffer()
	print(buffer)
	join = buffer.get_string_from_utf8()
	print(join == text)

	buffer = text.to_utf16_buffer()
	print(buffer)
	join = buffer.get_string_from_utf16()
	print(join == text)

	buffer = text.to_utf32_buffer()
	print(buffer)
	join = buffer.get_string_from_utf32()
	print(join == text)

	buffer = [0x74, 0x72, 0x61, 0x69, 0x6c, 0, 0, 0, 0, 0] # "trail\0\0\0\0\0"
	print(buffer.get_string_from_ascii() == "trail")
	print(buffer.get_string_from_utf8() == "trail\0\0\0\0\0")
