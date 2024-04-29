# 4.0+ replacement for `setget`:
var _backing: int = 0
var property:
	get:
		return _backing + 1000
	set(value):
		_backing = value - 1000


func test():
	print("Not using self:")
	print(property)
	print(_backing)
	property = 5000
	print(property)
	print(_backing)
	_backing = -50
	print(property)
	print(_backing)
	property = 5000
	print(property)
	print(_backing)

	# In Godot 4.0 and later, using `self` no longer makes a difference for
	# getter/setter execution in GDScript.
	print("Using self:")
	print(self.property)
	print(self._backing)
	self.property = 5000
	print(self.property)
	print(self._backing)
	self._backing = -50
	print(self.property)
	print(self._backing)
	self.property = 5000
	print(self.property)
	print(self._backing)
