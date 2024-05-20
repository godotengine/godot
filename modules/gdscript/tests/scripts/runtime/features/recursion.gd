func is_prime(number: int, divisor: int = 2) -> bool:
	print(divisor)
	if number <= 2:
		return (number == 2)
	elif number % divisor == 0:
		return false
	elif divisor * divisor > number:
		return true

	return is_prime(number, divisor + 1)

func test():
	# Not a prime number.
	print(is_prime(989))

	print()

	# Largest prime number below 10000.
	print(is_prime(9973))
