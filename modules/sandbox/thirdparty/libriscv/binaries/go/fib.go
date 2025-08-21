package main

import "os"

func fib(n uint, acc uint, prev uint) uint {
	if n == 0 {
		return acc
	} else {
		return fib(n-1, prev+acc, acc)
	}
}

func main() {
	os.Exit(int(fib(2560000, 0, 0)))
}
