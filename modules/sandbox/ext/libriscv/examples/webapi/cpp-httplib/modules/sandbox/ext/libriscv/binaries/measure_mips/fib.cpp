#include <cstdlib>

static long fib(long n)
{
    auto func = [](long n, long acc = 0, long prev = 1, auto &self)
    {
        if (n < 1)
            return acc;
        else
            return self(n - 1, prev + acc, acc, self);
    };

    return func(n, 0, 1, func);
}

int main(int argc, char** argv)
{
    const long n = atol(argv[1]);
    return fib(n);
}
