def can_build(env, platform):
	return platform=="windows" or platform=="server"

def configure(env):
	pass

def get_doc_classes():
	return [
		"InstancePool",
		"WorkPool",
		"Future",
	]

def get_doc_path():
	return "doc_classes"
