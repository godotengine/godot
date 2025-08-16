#include <stdexcept>
#include <iostream>

int main(int, char** argv)
{
    try {
        throw std::runtime_error(argv[1]);
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        return 0;
    }
    return 1;
}

extern "C"
int my_function(const char* str)
{
    std::cout << str << std::endl;
    return 1234;
}
