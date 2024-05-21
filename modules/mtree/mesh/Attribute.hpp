#pragma once
#include <vector>
#include <array>
#include <iostream>


namespace Mtree
{

    struct AbstractAttribute
    {
        virtual void add_data() = 0;
    };


    template <typename T>
    struct Attribute : AbstractAttribute
    {
        std::string name;
        std::vector<T> data;

        Attribute(std::string name) : name{ name } {};

        virtual void add_data()
        {
            data.emplace_back();
        };
    };
}