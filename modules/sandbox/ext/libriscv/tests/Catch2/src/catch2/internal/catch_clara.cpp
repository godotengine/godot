
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_clara.hpp>
#include <catch2/internal/catch_console_width.hpp>
#include <catch2/internal/catch_platform.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/internal/catch_textflow.hpp>
#include <catch2/internal/catch_reusable_string_stream.hpp>

#include <algorithm>
#include <ostream>

namespace {
    bool isOptPrefix( char c ) {
        return c == '-'
#ifdef CATCH_PLATFORM_WINDOWS
               || c == '/'
#endif
            ;
    }

    Catch::StringRef normaliseOpt( Catch::StringRef optName ) {
        if ( optName[0] == '-'
#if defined(CATCH_PLATFORM_WINDOWS)
             || optName[0] == '/'
#endif
        ) {
            return optName.substr( 1, optName.size() );
        }

        return optName;
    }

    static size_t find_first_separator(Catch::StringRef sr) {
        auto is_separator = []( char c ) {
            return c == ' ' || c == ':' || c == '=';
        };
        size_t pos = 0;
        while (pos < sr.size()) {
            if (is_separator(sr[pos])) { return pos; }
            ++pos;
        }

        return Catch::StringRef::npos;
    }

} // namespace

namespace Catch {
    namespace Clara {
        namespace Detail {

            void TokenStream::loadBuffer() {
                m_tokenBuffer.clear();

                // Skip any empty strings
                while ( it != itEnd && it->empty() ) {
                    ++it;
                }

                if ( it != itEnd ) {
                    StringRef next = *it;
                    if ( isOptPrefix( next[0] ) ) {
                        auto delimiterPos = find_first_separator(next);
                        if ( delimiterPos != StringRef::npos ) {
                            m_tokenBuffer.push_back(
                                { TokenType::Option,
                                  next.substr( 0, delimiterPos ) } );
                            m_tokenBuffer.push_back(
                                { TokenType::Argument,
                                  next.substr( delimiterPos + 1, next.size() ) } );
                        } else {
                            if ( next.size() > 1 && next[1] != '-' && next.size() > 2 ) {
                                // Combined short args, e.g. "-ab" for "-a -b"
                                for ( size_t i = 1; i < next.size(); ++i ) {
                                    m_tokenBuffer.push_back(
                                        { TokenType::Option,
                                          next.substr( i, 1 ) } );
                                }
                            } else {
                                m_tokenBuffer.push_back(
                                    { TokenType::Option, next } );
                            }
                        }
                    } else {
                        m_tokenBuffer.push_back(
                            { TokenType::Argument, next } );
                    }
                }
            }

            TokenStream::TokenStream( Args const& args ):
                TokenStream( args.m_args.begin(), args.m_args.end() ) {}

            TokenStream::TokenStream( Iterator it_, Iterator itEnd_ ):
                it( it_ ), itEnd( itEnd_ ) {
                loadBuffer();
            }

            TokenStream& TokenStream::operator++() {
                if ( m_tokenBuffer.size() >= 2 ) {
                    m_tokenBuffer.erase( m_tokenBuffer.begin() );
                } else {
                    if ( it != itEnd )
                        ++it;
                    loadBuffer();
                }
                return *this;
            }

            ParserResult convertInto( std::string const& source,
                                      std::string& target ) {
                target = source;
                return ParserResult::ok( ParseResultType::Matched );
            }

            ParserResult convertInto( std::string const& source,
                                      bool& target ) {
                std::string srcLC = toLower( source );

                if ( srcLC == "y" || srcLC == "1" || srcLC == "true" ||
                     srcLC == "yes" || srcLC == "on" ) {
                    target = true;
                } else if ( srcLC == "n" || srcLC == "0" || srcLC == "false" ||
                            srcLC == "no" || srcLC == "off" ) {
                    target = false;
                } else {
                    return ParserResult::runtimeError(
                        "Expected a boolean value but did not recognise: '" +
                        source + '\'' );
                }
                return ParserResult::ok( ParseResultType::Matched );
            }

            size_t ParserBase::cardinality() const { return 1; }

            InternalParseResult ParserBase::parse( Args const& args ) const {
                return parse( static_cast<std::string>(args.exeName()), TokenStream( args ) );
            }

            ParseState::ParseState( ParseResultType type,
                                    TokenStream remainingTokens ):
                m_type( type ), m_remainingTokens( CATCH_MOVE(remainingTokens) ) {}

            ParserResult BoundFlagRef::setFlag( bool flag ) {
                m_ref = flag;
                return ParserResult::ok( ParseResultType::Matched );
            }

            ResultBase::~ResultBase() = default;

            bool BoundRef::isContainer() const { return false; }

            bool BoundRef::isFlag() const { return false; }

            bool BoundFlagRefBase::isFlag() const { return true; }

} // namespace Detail

        Detail::InternalParseResult Arg::parse(std::string const&,
                                               Detail::TokenStream tokens) const {
            auto validationResult = validate();
            if (!validationResult)
                return Detail::InternalParseResult(validationResult);

            auto token = *tokens;
            if (token.type != Detail::TokenType::Argument)
                return Detail::InternalParseResult::ok(Detail::ParseState(
                    ParseResultType::NoMatch, CATCH_MOVE(tokens)));

            assert(!m_ref->isFlag());
            auto valueRef =
                static_cast<Detail::BoundValueRefBase*>(m_ref.get());

            auto result = valueRef->setValue(static_cast<std::string>(token.token));
            if ( !result )
                return Detail::InternalParseResult( result );
            else
                return Detail::InternalParseResult::ok(
                    Detail::ParseState( ParseResultType::Matched,
                                        CATCH_MOVE( ++tokens ) ) );
        }

        Opt::Opt(bool& ref) :
            ParserRefImpl(std::make_shared<Detail::BoundFlagRef>(ref)) {}

        Detail::HelpColumns Opt::getHelpColumns() const {
            ReusableStringStream oss;
            bool first = true;
            for (auto const& opt : m_optNames) {
                if (first)
                    first = false;
                else
                    oss << ", ";
                oss << opt;
            }
            if (!m_hint.empty())
                oss << " <" << m_hint << '>';
            return { oss.str(), m_description };
        }

        bool Opt::isMatch(StringRef optToken) const {
            auto normalisedToken = normaliseOpt(optToken);
            for (auto const& name : m_optNames) {
                if (normaliseOpt(name) == normalisedToken)
                    return true;
            }
            return false;
        }

        Detail::InternalParseResult Opt::parse(std::string const&,
                                       Detail::TokenStream tokens) const {
            auto validationResult = validate();
            if (!validationResult)
                return Detail::InternalParseResult(validationResult);

            if (tokens &&
                tokens->type == Detail::TokenType::Option) {
                auto const& token = *tokens;
                if (isMatch(token.token)) {
                    if (m_ref->isFlag()) {
                        auto flagRef =
                            static_cast<Detail::BoundFlagRefBase*>(
                                m_ref.get());
                        auto result = flagRef->setFlag(true);
                        if (!result)
                            return Detail::InternalParseResult(result);
                        if (result.value() ==
                            ParseResultType::ShortCircuitAll)
                            return Detail::InternalParseResult::ok(Detail::ParseState(
                                result.value(), CATCH_MOVE(tokens)));
                    } else {
                        auto valueRef =
                            static_cast<Detail::BoundValueRefBase*>(
                                m_ref.get());
                        ++tokens;
                        if (!tokens)
                            return Detail::InternalParseResult::runtimeError(
                                "Expected argument following " +
                                token.token);
                        auto const& argToken = *tokens;
                        if (argToken.type != Detail::TokenType::Argument)
                            return Detail::InternalParseResult::runtimeError(
                                "Expected argument following " +
                                token.token);
                        const auto result = valueRef->setValue(static_cast<std::string>(argToken.token));
                        if (!result)
                            return Detail::InternalParseResult(result);
                        if (result.value() ==
                            ParseResultType::ShortCircuitAll)
                            return Detail::InternalParseResult::ok(Detail::ParseState(
                                result.value(), CATCH_MOVE(tokens)));
                    }
                    return Detail::InternalParseResult::ok(Detail::ParseState(
                        ParseResultType::Matched, CATCH_MOVE(++tokens)));
                }
            }
            return Detail::InternalParseResult::ok(
                Detail::ParseState(ParseResultType::NoMatch, CATCH_MOVE(tokens)));
        }

        Detail::Result Opt::validate() const {
            if (m_optNames.empty())
                return Detail::Result::logicError("No options supplied to Opt");
            for (auto const& name : m_optNames) {
                if (name.empty())
                    return Detail::Result::logicError(
                        "Option name cannot be empty");
#ifdef CATCH_PLATFORM_WINDOWS
                if (name[0] != '-' && name[0] != '/')
                    return Detail::Result::logicError(
                        "Option name must begin with '-' or '/'");
#else
                if (name[0] != '-')
                    return Detail::Result::logicError(
                        "Option name must begin with '-'");
#endif
            }
            return ParserRefImpl::validate();
        }

        ExeName::ExeName() :
            m_name(std::make_shared<std::string>("<executable>")) {}

        ExeName::ExeName(std::string& ref) : ExeName() {
            m_ref = std::make_shared<Detail::BoundValueRef<std::string>>(ref);
        }

        Detail::InternalParseResult
            ExeName::parse(std::string const&,
                           Detail::TokenStream tokens) const {
            return Detail::InternalParseResult::ok(
                Detail::ParseState(ParseResultType::NoMatch, CATCH_MOVE(tokens)));
        }

        ParserResult ExeName::set(std::string const& newName) {
            auto lastSlash = newName.find_last_of("\\/");
            auto filename = (lastSlash == std::string::npos)
                ? newName
                : newName.substr(lastSlash + 1);

            *m_name = filename;
            if (m_ref)
                return m_ref->setValue(filename);
            else
                return ParserResult::ok(ParseResultType::Matched);
        }




        Parser& Parser::operator|=( Parser const& other ) {
            m_options.insert( m_options.end(),
                              other.m_options.begin(),
                              other.m_options.end() );
            m_args.insert(
                m_args.end(), other.m_args.begin(), other.m_args.end() );
            return *this;
        }

        std::vector<Detail::HelpColumns> Parser::getHelpColumns() const {
            std::vector<Detail::HelpColumns> cols;
            cols.reserve( m_options.size() );
            for ( auto const& o : m_options ) {
                cols.push_back(o.getHelpColumns());
            }
            return cols;
        }

        void Parser::writeToStream( std::ostream& os ) const {
            if ( !m_exeName.name().empty() ) {
                os << "usage:\n"
                   << "  " << m_exeName.name() << ' ';
                bool required = true, first = true;
                for ( auto const& arg : m_args ) {
                    if ( first )
                        first = false;
                    else
                        os << ' ';
                    if ( arg.isOptional() && required ) {
                        os << '[';
                        required = false;
                    }
                    os << '<' << arg.hint() << '>';
                    if ( arg.cardinality() == 0 )
                        os << " ... ";
                }
                if ( !required )
                    os << ']';
                if ( !m_options.empty() )
                    os << " options";
                os << "\n\nwhere options are:\n";
            }

            auto rows = getHelpColumns();
            size_t consoleWidth = CATCH_CONFIG_CONSOLE_WIDTH;
            size_t optWidth = 0;
            for ( auto const& cols : rows )
                optWidth = ( std::max )( optWidth, cols.left.size() + 2 );

            optWidth = ( std::min )( optWidth, consoleWidth / 2 );

            for ( auto& cols : rows ) {
                auto row = TextFlow::Column( CATCH_MOVE(cols.left) )
                               .width( optWidth )
                               .indent( 2 ) +
                           TextFlow::Spacer( 4 ) +
                           TextFlow::Column( static_cast<std::string>(cols.descriptions) )
                               .width( consoleWidth - 7 - optWidth );
                os << row << '\n';
            }
        }

        Detail::Result Parser::validate() const {
            for ( auto const& opt : m_options ) {
                auto result = opt.validate();
                if ( !result )
                    return result;
            }
            for ( auto const& arg : m_args ) {
                auto result = arg.validate();
                if ( !result )
                    return result;
            }
            return Detail::Result::ok();
        }

        Detail::InternalParseResult
        Parser::parse( std::string const& exeName,
                       Detail::TokenStream tokens ) const {

            struct ParserInfo {
                ParserBase const* parser = nullptr;
                size_t count = 0;
            };
            std::vector<ParserInfo> parseInfos;
            parseInfos.reserve( m_options.size() + m_args.size() );
            for ( auto const& opt : m_options ) {
                parseInfos.push_back( { &opt, 0 } );
            }
            for ( auto const& arg : m_args ) {
                parseInfos.push_back( { &arg, 0 } );
            }

            m_exeName.set( exeName );

            auto result = Detail::InternalParseResult::ok(
                Detail::ParseState( ParseResultType::NoMatch, CATCH_MOVE(tokens) ) );
            while ( result.value().remainingTokens() ) {
                bool tokenParsed = false;

                for ( auto& parseInfo : parseInfos ) {
                    if ( parseInfo.parser->cardinality() == 0 ||
                         parseInfo.count < parseInfo.parser->cardinality() ) {
                        result = parseInfo.parser->parse(
                            exeName, CATCH_MOVE(result).value().remainingTokens() );
                        if ( !result )
                            return result;
                        if ( result.value().type() !=
                             ParseResultType::NoMatch ) {
                            tokenParsed = true;
                            ++parseInfo.count;
                            break;
                        }
                    }
                }

                if ( result.value().type() == ParseResultType::ShortCircuitAll )
                    return result;
                if ( !tokenParsed )
                    return Detail::InternalParseResult::runtimeError(
                        "Unrecognised token: " +
                        result.value().remainingTokens()->token );
            }
            // !TBD Check missing required options
            return result;
        }

        Args::Args(int argc, char const* const* argv) :
            m_exeName(argv[0]), m_args(argv + 1, argv + argc) {}

        Args::Args(std::initializer_list<StringRef> args) :
            m_exeName(*args.begin()),
            m_args(args.begin() + 1, args.end()) {}


        Help::Help( bool& showHelpFlag ):
            Opt( [&]( bool flag ) {
                showHelpFlag = flag;
                return ParserResult::ok( ParseResultType::ShortCircuitAll );
            } ) {
            static_cast<Opt&> ( *this )(
                "display usage information" )["-?"]["-h"]["--help"]
                .optional();
        }

    } // namespace Clara
} // namespace Catch
