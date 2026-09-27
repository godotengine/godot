class_name InnerBase

class A extends InnerA:
	class InnerA:
		pass

class B extends InnerB:
	const InnerB = A

class C extends InnerC.InnerInnerC:
	class InnerC:
		class InnerInnerC:
			pass

class D extends InnerBase.D:
	pass

class E extends InnerBase.E.InnerE:
	class InnerE:
		pass

class F extends F:
	pass

class G extends G.InnerG:
	class InnerG:
		pass

class H extends InnerBase:
	pass

const _I = I
class I extends _I:
	pass

const _J = J
class J extends _J.InnerJ:
	class InnerJ:
		pass
