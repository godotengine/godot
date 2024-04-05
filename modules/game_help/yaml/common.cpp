#include "common.hpp"

#ifndef RYML_NO_DEFAULT_CALLBACKS
#   include <stdlib.h>
#   include <stdio.h>
#endif // RYML_NO_DEFAULT_CALLBACKS

namespace c4 {
namespace yml {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

namespace {
Callbacks s_default_callbacks;
} // anon namespace

#ifndef RYML_NO_DEFAULT_CALLBACKS
void report_error_impl(const char* msg, size_t length, Location loc, FILE *f)
{
    if(!f)
        f = stderr;
    if(loc)
    {
        if(!loc.name.empty())
        {
            fwrite(loc.name.str, 1, loc.name.len, f);
            fputc(':', f);
        }
        fprintf(f, "%zu:", loc.line);
        if(loc.col)
            fprintf(f, "%zu:", loc.col);
        if(loc.offset)
            fprintf(f, " (%zuB):", loc.offset);
    }
    fprintf(f, "%.*s\n", (int)length, msg);
    fflush(f);
}

void error_impl(const char* msg, size_t length, Location loc, void * /*user_data*/)
{
    report_error_impl(msg, length, loc, nullptr);
    ::abort();
}

void* allocate_impl(size_t length, void * /*hint*/, void * /*user_data*/)
{
    void *mem = ::malloc(length);
    if(mem == nullptr)
    {
        const char msg[] = "could not allocate memory";
        error_impl(msg, sizeof(msg)-1, {}, nullptr);
    }
    return mem;
}

void free_impl(void *mem, size_t /*length*/, void * /*user_data*/)
{
    ::free(mem);
}
#endif // RYML_NO_DEFAULT_CALLBACKS



Callbacks::Callbacks()
    :
    m_user_data(nullptr),
    #ifndef RYML_NO_DEFAULT_CALLBACKS
    m_allocate(allocate_impl),
    m_free(free_impl),
    m_error(error_impl)
    #else
    m_allocate(nullptr),
    m_free(nullptr),
    m_error(nullptr)
    #endif
{
}

Callbacks::Callbacks(void *user_data, pfn_allocate alloc_, pfn_free free_, pfn_error error_)
    :
    m_user_data(user_data),
    #ifndef RYML_NO_DEFAULT_CALLBACKS
    m_allocate(alloc_ ? alloc_ : allocate_impl),
    m_free(free_ ? free_ : free_impl),
    m_error(error_ ? error_ : error_impl)
    #else
    m_allocate(alloc_),
    m_free(free_),
    m_error(error_)
    #endif
{
    C4_CHECK(m_allocate);
    C4_CHECK(m_free);
    C4_CHECK(m_error);
}


void set_callbacks(Callbacks const& c)
{
    s_default_callbacks = c;
}

Callbacks const& get_callbacks()
{
    return s_default_callbacks;
}

void reset_callbacks()
{
    set_callbacks(Callbacks());
}

void error(const char *msg, size_t msg_len, Location loc)
{
    s_default_callbacks.m_error(msg, msg_len, loc, s_default_callbacks.m_user_data);
}

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace yml
} // namespace c4
