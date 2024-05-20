signal ok()

@warning_ignore("return_value_discarded")
func test():
	ok.connect(func(): print('ok'))
	emit_signal(&'ok')
