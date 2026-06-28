class Outer:
	enum OuterEnum { OuterValue = 3 }
	const OuterConst := OuterEnum

	class Inner:
		enum InnerEnum { InnerValue = 7 }
		const InnerConst := InnerEnum

		static func test() -> void:
			print(OuterEnum.size());
			print(OuterEnum.OuterValue);
			print(OuterConst.size());
			print(OuterConst.OuterValue);
			print(Outer.OuterEnum.size());
			print(Outer.OuterEnum.OuterValue);
			print(Outer.OuterConst.size());
			print(Outer.OuterConst.OuterValue);

			print(InnerEnum.size());
			print(InnerEnum.InnerValue);
			print(InnerConst.size());
			print(InnerConst.InnerValue);
			print(Inner.InnerEnum.size());
			print(Inner.InnerEnum.InnerValue);
			print(Inner.InnerConst.size());
			print(Inner.InnerConst.InnerValue);

func test():
	Outer.Inner.test()
