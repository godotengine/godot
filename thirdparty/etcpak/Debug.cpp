#include <algorithm>
#include <vector>
#include "Debug.hpp"

static std::vector<DebugLog::Callback*> s_callbacks;

void DebugLog::Message( const char* msg )
{
    for( auto it = s_callbacks.begin(); it != s_callbacks.end(); ++it )
    {
        (*it)->OnDebugMessage( msg );
    }
}

void DebugLog::AddCallback( Callback* c )
{
    const auto it = std::find( s_callbacks.begin(), s_callbacks.end(), c );
    if( it == s_callbacks.end() )
    {
        s_callbacks.push_back( c );
    }
}

void DebugLog::RemoveCallback( Callback* c )
{
    const auto it = std::find( s_callbacks.begin(), s_callbacks.end(), c );
    if( it != s_callbacks.end() )
    {
        s_callbacks.erase( it );
    }
}
