def can_build(platform):
	# we'll be supporting more eventually...
    return platform == 'windows' or platform == 'x11' or platform == 'osx'


def configure(env):
    pass
