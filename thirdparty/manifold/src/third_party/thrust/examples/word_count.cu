#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

#include <iostream>

// This example computes the number of words in a text sample
// with a single call to thrust::inner_product.  The algorithm
// counts the number of characters which start a new word, i.e.
// the number of characters where input[i] is an alphabetical
// character and input[i-1] is not an alphabetical character.


// determines whether the character is alphabetical
__host__ __device__
bool is_alpha(const char c)
{
    return (c >= 'A' && c <= 'z');
}

// determines whether the right character begins a new word
struct is_word_start
    : public thrust::binary_function<const char&, const char&, bool>
{
    __host__ __device__
    bool operator()(const char& left, const char& right) const
    {
        return is_alpha(right) && !is_alpha(left);
    }
};


int word_count(const thrust::device_vector<char>& input)
{
    // check for empty string
    if (input.empty())
        return 0;

    // compute the number characters that start a new word
    int wc = thrust::inner_product(input.begin(), input.end() - 1,  // sequence of left characters
                                   input.begin() + 1,               // sequence of right characters
                                   0,                               // initialize sum to 0
                                   thrust::plus<int>(),             // sum values together
                                   is_word_start());       // how to compare the left and right characters
    
    // if the first character is alphabetical, then it also begins a word
    if (is_alpha(input.front()))
        wc++;
    
    return wc;
}


int main(void)
{
    // Paragraph from 'The Raven' by Edgar Allan Poe
    // http://en.wikipedia.org/wiki/The_Raven
    const char raw_input[] = "  But the raven, sitting lonely on the placid bust, spoke only,\n"
                             "  That one word, as if his soul in that one word he did outpour.\n"
                             "  Nothing further then he uttered - not a feather then he fluttered -\n"
                             "  Till I scarcely more than muttered `Other friends have flown before -\n"
                             "  On the morrow he will leave me, as my hopes have flown before.'\n"
                             "  Then the bird said, `Nevermore.'\n";

    std::cout << "Text sample:" << std::endl;
    std::cout << raw_input << std::endl;
    
    // transfer to device
    thrust::device_vector<char> input(raw_input, raw_input + sizeof(raw_input));

    // count words
    int wc = word_count(input);
    
    std::cout << "Text sample contains " << wc << " words" << std::endl;
        
    return 0;
}

