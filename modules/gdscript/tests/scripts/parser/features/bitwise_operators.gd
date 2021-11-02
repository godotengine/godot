enum Flags {
	FIRE = 1 << 1,
	ICE = 1 << 2,
	SLIPPERY = 1 << 3,
	STICKY = 1 << 4,
	NONSOLID = 1 << 5,

	ALL = FIRE | ICE | SLIPPERY | STICKY | NONSOLID,
}


func test():
	var flags = Flags.FIRE | Flags.SLIPPERY
	print(flags)

	flags = Flags.FIRE & Flags.SLIPPERY
	print(flags)

	flags = Flags.FIRE ^ Flags.SLIPPERY
	print(flags)

	flags = Flags.ALL & (Flags.FIRE | Flags.ICE)
	print(flags)

	flags = (Flags.ALL & Flags.FIRE) | Flags.ICE
	print(flags)

	flags = Flags.ALL & Flags.FIRE | Flags.ICE
	print(flags)

	# Enum value must be casted to an integer. Otherwise, a parser error is emitted.
	flags &= int(Flags.ICE)
	print(flags)

	flags ^= int(Flags.ICE)
	print(flags)

	flags |= int(Flags.STICKY | Flags.SLIPPERY)
	print(flags)

	print()

	var num = 2 << 4
	print(num)

	num <<= 2
	print(num)

	num >>= 2
	print(num)
