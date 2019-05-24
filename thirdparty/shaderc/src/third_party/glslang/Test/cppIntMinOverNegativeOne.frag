#if (-2147483648 / -1) != 0
#error INT_MIN / -1 should yield 0, something went wrong.
#endif
#if (-2147483648 % -1) != 0
#error INT_MIN % -1 should yield 0, something went wrong.
#endif