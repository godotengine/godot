//  NREX: Node RegEx
//
//  Copyright (c) 2015, Zher Huei Lee
//  All rights reserved.
//
//  This software is provided 'as-is', without any express or implied
//  warranty.  In no event will the authors be held liable for any damages
//  arising from the use of this software.
//
//  Permission is granted to anyone to use this software for any purpose,
//  including commercial applications, and to alter it and redistribute it
//  freely, subject to the following restrictions:
//
//   1. The origin of this software must not be misrepresented; you must not
//      claim that you wrote the original software. If you use this software
//      in a product, an acknowledgment in the product documentation would
//      be appreciated but is not required.
//
//   2. Altered source versions must be plainly marked as such, and must not
//      be misrepresented as being the original software.
//
//   3. This notice may not be removed or altered from any source
//      distribution.
//

#include "nrex.hpp"

#ifdef NREX_UNICODE
#include <wctype.h>
#include <wchar.h>
#define NREX_ISALPHANUM iswalnum
#define NREX_STRLEN wcslen
#else
#include <ctype.h>
#include <string.h>
#define NREX_ISALPHANUM isalnum
#define NREX_STRLEN strlen
#endif

#ifdef NREX_THROW_ERROR
#define NREX_COMPILE_ERROR(M) throw nrex_compile_error(M)
#else
#define NREX_COMPILE_ERROR(M) reset(); return false
#endif

#ifndef NREX_NEW
#define NREX_NEW(X) new X
#define NREX_NEW_ARRAY(X, N) new X[N]
#define NREX_DELETE(X) delete X
#define NREX_DELETE_ARRAY(X) delete[] X
#endif

template<typename T>
class nrex_array
{
    private:
        T* _data;
        unsigned int _reserved;
        unsigned int _size;
    public:
        nrex_array()
            : _data(NREX_NEW_ARRAY(T, 2))
            , _reserved(2)
            , _size(0)
        {
        }

        ~nrex_array()
        {
            NREX_DELETE_ARRAY(_data);
        }

        unsigned int size() const
        {
            return _size;
        }

        void reserve(unsigned int size)
        {
            T* old = _data;
            _data = NREX_NEW_ARRAY(T, size);
            _reserved = size;
            for (unsigned int i = 0; i < _size; ++i)
            {
                _data[i] = old[i];
            }
            NREX_DELETE_ARRAY(old);
        }

        void push(T item)
        {
            if (_size == _reserved)
            {
                reserve(_reserved * 2);
            }
            _data[_size] = item;
            _size++;
        }

        T& top()
        {
            return _data[_size - 1];
        }

        const T& operator[] (unsigned int i) const
        {
            return _data[i];
        }

        void pop()
        {
            if (_size > 0)
            {
                --_size;
            }
        }
};

static nrex_char nrex_unescape(nrex_char repr)
{
    switch (repr)
    {
        case '^': return '^';
        case '$': return '$';
        case '(': return '(';
        case ')': return ')';
        case '\\': return '\\';
        case '.': return '.';
        case '+': return '+';
        case '*': return '*';
        case '?': return '?';
        case '-': return '-';
        case 'a': return '\a';
        case 'e': return '\e';
        case 'f': return '\f';
        case 'n': return '\n';
        case 'r': return '\r';
        case 't': return '\t';
        case 'v': return '\v';
    }
    return 0;
}

struct nrex_search
{
    public:
        const nrex_char* str;
        nrex_result* captures;
        int end;
        bool complete;

        nrex_char at(int pos)
        {
            return str[pos];
        }

        nrex_search(const nrex_char* str, nrex_result* captures)
            : str(str)
            , captures(captures)
            , end(0)
        {
        }
};

struct nrex_node
{
        nrex_node* next;
        nrex_node* previous;
        nrex_node* parent;
        bool quantifiable;

        nrex_node(bool quantify = false)
            : next(NULL)
            , previous(NULL)
            , parent(NULL)
            , quantifiable(quantify)
        {
        }

        virtual ~nrex_node()
        {
            if (next)
            {
                NREX_DELETE(next);
            }
        }

        virtual int test(nrex_search* s, int pos) const
        {
            return next ? next->test(s, pos) : -1;
        }

        virtual int test_parent(nrex_search* s, int pos) const
        {
            if (next)
            {
                pos = next->test(s, pos);
            }
            if (parent && pos >= 0)
            {
                pos = parent->test_parent(s, pos);
            }
            if (pos >= 0)
            {
                s->complete = true;
            }
            return pos;
        }
};

struct nrex_node_group : public nrex_node
{
        int capturing;
        bool negate;
        nrex_array<nrex_node*> childset;
        nrex_node* back;

        nrex_node_group(int capturing)
            : nrex_node(true)
            , capturing(capturing)
            , negate(false)
            , back(NULL)
        {
        }

        virtual ~nrex_node_group()
        {
            for (unsigned int i = 0; i < childset.size(); ++i)
            {
                NREX_DELETE(childset[i]);
            }

        }

        int test(nrex_search* s, int pos) const
        {
            if (capturing >= 0)
            {
                s->captures[capturing].start = pos;
            }
            for (unsigned int i = 0; i < childset.size(); ++i)
            {
                s->complete = false;
                int res = childset[i]->test(s, pos);
                if (s->complete)
                {
                    return res;
                }
                if (negate)
                {
                    if (res < 0)
                    {
                        res = pos + 1;
                    }
                    else
                    {
                        return -1;
                    }
                }
                if (res >= 0)
                {
                    if (capturing >= 0)
                    {
                        s->captures[capturing].length = res - pos;
                    }
                    return next ? next->test(s, res) : res;
                }
            }
            return -1;
        }

        virtual int test_parent(nrex_search* s, int pos) const
        {
            if (capturing >= 0)
            {
                s->captures[capturing].length = pos - s->captures[capturing].start;
            }
            return nrex_node::test_parent(s, pos);
        }

        void add_childset()
        {
            back = NULL;
        }

        void add_child(nrex_node* node)
        {
            node->parent = this;
            node->previous = back;
            if (back)
            {
                back->next = node;
            }
            else
            {
                childset.push(node);
            }
            back = node;
        }

        nrex_node* swap_back(nrex_node* node)
        {
            if (!back)
            {
                add_child(node);
                return NULL;
            }
            nrex_node* old = back;
            if (!old->previous)
            {
                childset.pop();
            }
            back = old->previous;
            add_child(node);
            return old;
        }
};

struct nrex_node_char : public nrex_node
{
        nrex_char ch;

        nrex_node_char(nrex_char c)
            : nrex_node(true)
            , ch(c)
        {
        }

        int test(nrex_search* s, int pos) const
        {
            if (s->end == pos || s->at(pos) != ch)
            {
                return -1;
            }
            return next ? next->test(s, pos + 1) : pos + 1;
        }
};

struct nrex_node_range : public nrex_node
{
        nrex_char start;
        nrex_char end;

        nrex_node_range(nrex_char s, nrex_char e)
            : nrex_node(true)
            , start(s)
            , end(e)
        {
        }

        int test(nrex_search* s, int pos) const
        {
            if (s->end == pos)
            {
                return -1;
            }
            nrex_char c = s->at(pos);
            if (c < start || end < c)
            {
                return -1;
            }
            return next ? next->test(s, pos + 1) : pos + 1;
        }
};

static bool nrex_is_whitespace(nrex_char repr)
{
    switch (repr)
    {
        case ' ':
        case '\t':
        case '\r':
        case '\n':
        case '\f':
            return true;
    }
    return false;
}

static bool nrex_is_shorthand(nrex_char repr)
{
    switch (repr)
    {
        case 'W':
        case 'w':
        case 'D':
        case 'd':
        case 'S':
        case 's':
            return true;
    }
    return false;
}

struct nrex_node_shorthand : public nrex_node
{
        nrex_char repr;

        nrex_node_shorthand(nrex_char c)
            : nrex_node(true)
            , repr(c)
        {
        }

        int test(nrex_search* s, int pos) const
        {
            if (s->end == pos)
            {
                return -1;
            }
            bool found = false;
            bool invert = false;
            nrex_char c = s->at(pos);
            switch (repr)
            {
                case '.':
                    found = true;
                    break;
                case 'W':
                    invert = true;
                case 'w':
                    if (c == '_' || NREX_ISALPHANUM(c))
                    {
                        found = true;
                    }
                    break;
                case 'D':
                    invert = true;
                case 'd':
                    if ('0' <= c && c <= '9')
                    {
                        found = true;
                    }
                    break;
                case 'S':
                    invert = true;
                case 's':
                    if (nrex_is_whitespace(c))
                    {
                        found = true;
                    }
                    break;
            }
            if (found == invert)
            {
                return -1;
            }
            return next ? next->test(s, pos + 1) : pos + 1;
        }
};

static bool nrex_is_quantifier(nrex_char repr)
{
    switch (repr)
    {
        case '?':
        case '*':
        case '+':
        case '{':
            return true;
    }
    return false;
}

struct nrex_node_quantifier : public nrex_node
{
        int min;
        int max;
        bool greedy;
        nrex_node* child;

        nrex_node_quantifier()
            : nrex_node()
            , min(0)
            , max(0)
            , greedy(true)
            , child(NULL)
        {
        }

        virtual ~nrex_node_quantifier()
        {
            if (child)
            {
                NREX_DELETE(child);
            }
        }

        int test(nrex_search* s, int pos) const
        {
            nrex_array<int> backtrack;
            backtrack.push(pos);
            while (backtrack.top() <= s->end)
            {
                if (max >= 1 && backtrack.size() > (unsigned int)max)
                {
                    break;
                }
                if (!greedy && (unsigned int)min < backtrack.size())
                {
                    int res = backtrack.top();
                    if (next)
                    {
                        res = next->test(s, res);
                    }
                    if (s->complete)
                    {
                        return res;
                    }
                    if (res >= 0 && parent->test_parent(s, res) >= 0)
                    {
                        return res;
                    }
                }
                int res = child->test(s, backtrack.top());
                if (s->complete)
                {
                    return res;
                }
                if (res < 0 || res == backtrack.top())
                {
                    break;
                }
                backtrack.push(res);
            }
            while (greedy && (unsigned int) min < backtrack.size())
            {
                int res = backtrack.top();
                if (next)
                {
                    res = next->test(s, res);
                }
                if (res >= 0 && parent->test_parent(s, res) >= 0)
                {
                    return res;
                }
                if (s->complete)
                {
                    return res;
                }
                backtrack.pop();
            }
            return -1;
        }
};

struct nrex_node_anchor : public nrex_node
{
        bool end;

        nrex_node_anchor(bool end)
            : nrex_node()
            , end(end)
        {
        }

        int test(nrex_search* s, int pos) const
        {
            if (!end && pos != 0)
            {
                return -1;
            }
            else if (end && pos != s->end)
            {
                return -1;
            }
            return next ? next->test(s, pos) : pos;
        }
};

struct nrex_node_backreference : public nrex_node
{
        int ref;

        nrex_node_backreference(int ref)
            : nrex_node(true)
            , ref(ref)
        {
        }

        int test(nrex_search* s, int pos) const
        {
            nrex_result& r = s->captures[ref];
            for (int i = 0; i < r.length; ++i)
            {
                if (pos + i >= s->end)
                {
                    return -1;
                }
                if (s->at(r.start + i) != s->at(pos + i))
                {
                    return -1;
                }
            }
            return next ? next->test(s, pos + r.length) : pos + r.length;
        }
};

nrex::nrex()
    : _capturing(0)
    , _root(NULL)
{
}

nrex::~nrex()
{
    if (_root)
    {
        NREX_DELETE(_root);
    }
}

bool nrex::valid() const
{
    return (_root != NULL);
}

void nrex::reset()
{
    _capturing = 0;
    if (_root)
    {
        NREX_DELETE(_root);
    }
    _root = NULL;
}

int nrex::capture_size() const
{
    return _capturing + 1;
}

bool nrex::compile(const nrex_char* pattern)
{
    reset();
    nrex_node_group* root = NREX_NEW(nrex_node_group(_capturing));
    nrex_array<nrex_node_group*> stack;
    stack.push(root);
    _root = root;

    for (const nrex_char* c = pattern; c[0] != '\0'; ++c)
    {
        if (c[0] == '(')
        {
            if (c[1] == '?')
            {
                if (c[2] == ':')
                {
                    c = &c[2];
                    nrex_node_group* group = NREX_NEW(nrex_node_group(-1));
                    stack.top()->add_child(group);
                    stack.push(group);
                }
                else
                {
                    NREX_COMPILE_ERROR("unrecognised qualifier for parenthesis");
                }
            }
            else if (_capturing < 99)
            {
                nrex_node_group* group = NREX_NEW(nrex_node_group(++_capturing));
                stack.top()->add_child(group);
                stack.push(group);
            }
            else
            {
                nrex_node_group* group = NREX_NEW(nrex_node_group(-1));
                stack.top()->add_child(group);
                stack.push(group);
            }
        }
        else if (c[0] == ')')
        {
            if (stack.size() > 1)
            {
                stack.pop();
            }
            else
            {
                NREX_COMPILE_ERROR("unexpected ')'");
            }
        }
        else if (c[0] == '[')
        {
            nrex_node_group* group = NREX_NEW(nrex_node_group(-1));
            stack.top()->add_child(group);
            if (c[1] == '^')
            {
                group->negate = true;
                ++c;
            }
            while (true)
            {
                group->add_childset();
                ++c;
                if (c[0] == '\0')
                {
                    NREX_COMPILE_ERROR("unclosed character class '[]'");
                }
                if (c[0] == ']')
                {
                    break;
                }
                else if (c[0] == '\\')
                {
                    nrex_char unescaped = nrex_unescape(c[1]);
                    if (unescaped)
                    {
                        group->add_child(NREX_NEW(nrex_node_char(unescaped)));
                        ++c;
                    }
                    else if (nrex_is_shorthand(c[1]))
                    {
                        group->add_child(NREX_NEW(nrex_node_shorthand(c[1])));
                        ++c;
                    }
                    else
                    {
                        NREX_COMPILE_ERROR("escape token not recognised");
                    }
                }
                else
                {
                    if (c[1] == '-' && c[2] != '\0')
                    {
                        bool range = false;
                        if ('A' <= c[0] && c[0] <= 'Z' && 'A' <= c[2] && c[2] <= 'Z')
                        {
                            range = true;
                        }
                        if ('a' <= c[0] && c[0] <= 'z' && 'a' <= c[2] && c[2] <= 'z')
                        {
                            range = true;
                        }
                        if ('0' <= c[0] && c[0] <= '9' && '0' <= c[2] && c[2] <= '9')
                        {
                            range = true;
                        }
                        if (range)
                        {
                            group->add_child(NREX_NEW(nrex_node_range(c[0], c[2])));
                            c = &c[2];
                            continue;
                        }
                    }
                    group->add_child(NREX_NEW(nrex_node_char(c[0])));
                }

            }
        }
        else if (nrex_is_quantifier(c[0]))
        {
            nrex_node_quantifier* quant = NREX_NEW(nrex_node_quantifier);
            quant->child = stack.top()->swap_back(quant);
            if (quant->child == NULL || !quant->child->quantifiable)
            {
                NREX_COMPILE_ERROR("element not quantifiable");
            }
            quant->child->previous = NULL;
            quant->child->next = NULL;
            quant->child->parent = quant;
            if (c[0] == '?')
            {
                quant->min = 0;
                quant->max = 1;
            }
            else if (c[0] == '+')
            {
                quant->min = 1;
                quant->max = -1;
            }
            else if (c[0] == '*')
            {
                quant->min = 0;
                quant->max = -1;
            }
            else if (c[0] == '{')
            {
                bool max_set = false;
                quant->min = 0;
                quant->max = -1;
                while (true)
                {
                    ++c;
                    if (c[0] == '\0')
                    {
                        NREX_COMPILE_ERROR("unclosed range quantifier '{}'");
                    }
                    else if (c[0] == '}')
                    {
                        break;
                    }
                    else if (c[0] == ',')
                    {
                        max_set = true;
                        continue;
                    }
                    else if (c[0] < '0' || '9' < c[0])
                    {
                        NREX_COMPILE_ERROR("expected numeric digits, ',' or '}'");
                    }
                    if (max_set)
                    {
                        if (quant->max < 0)
                        {
                            quant->max = int(c[0] - '0');
                        }
                        else
                        {
                            quant->max = quant->max * 10 + int(c[0] - '0');
                        }
                    }
                    else
                    {
                        quant->min = quant->min * 10 + int(c[0] - '0');
                    }
                }
                if (!max_set)
                {
                    quant->max = quant->min;
                }
            }
            if (c[1] == '?')
            {
                quant->greedy = false;
                ++c;
            }
        }
        else if (c[0] == '|')
        {
            stack.top()->add_childset();
        }
        else if (c[0] == '^' || c[0] == '$')
        {
            stack.top()->add_child(NREX_NEW(nrex_node_anchor((c[0] == '$'))));
        }
        else if (c[0] == '.')
        {
            stack.top()->add_child(NREX_NEW(nrex_node_shorthand('.')));
        }
        else if (c[0] == '\\')
        {
            nrex_char unescaped = nrex_unescape(c[1]);
            if (unescaped)
            {
                stack.top()->add_child(NREX_NEW(nrex_node_char(unescaped)));
                ++c;
            }
            else if (nrex_is_shorthand(c[1]))
            {
                stack.top()->add_child(NREX_NEW(nrex_node_shorthand(c[1])));
                ++c;
            }
            else if ('1' <= c[1] && c[1] <= '9')
            {
                int ref = 0;
                if ('0' <= c[2] && c[2] <= '9')
                {
                    ref = int(c[1] - '0') * 10 + int(c[2] - '0');
                    c = &c[2];
                }
                else
                {
                    ref = int(c[1] - '0');
                    ++c;
                }
                if (ref > _capturing)
                {
                    NREX_COMPILE_ERROR("backreference to non-existent capture");
                }
                stack.top()->add_child(NREX_NEW(nrex_node_backreference(ref)));
            }
            else
            {
                NREX_COMPILE_ERROR("escape token not recognised");
            }
        }
        else
        {
            stack.top()->add_child(NREX_NEW(nrex_node_char(c[0])));
        }
    }
    return true;
}

bool nrex::match(const nrex_char* str, nrex_result* captures, int offset, int end) const
{
    nrex_search s(str, captures);
    if (end >= offset)
    {
        s.end = end;
    }
    else
    {
        s.end = NREX_STRLEN(str);
    }
    for (int i = offset; i < s.end; ++i)
    {
        for (int c = 0; c <= _capturing; ++c)
        {
            captures[c].start = 0;
            captures[c].length = 0;
        }
        if (_root->test(&s, i) >= 0)
        {
            return true;
        }
    }
    return false;
}
