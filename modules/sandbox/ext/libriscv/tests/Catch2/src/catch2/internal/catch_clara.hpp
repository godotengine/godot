
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_CLARA_HPP_INCLUDED
#define CATCH_CLARA_HPP_INCLUDED

#if defined( __clang__ )
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#    pragma clang diagnostic ignored "-Wshadow"
#    pragma clang diagnostic ignored "-Wdeprecated"
#endif

#if defined( __GNUC__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#ifndef CLARA_CONFIG_OPTIONAL_TYPE
#    ifdef __has_include
#        if __has_include( <optional>) && __cplusplus >= 201703L
#            include <optional>
#            define CLARA_CONFIG_OPTIONAL_TYPE std::optional
#        endif
#    endif
#endif

#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_noncopyable.hpp>
#include <catch2/internal/catch_void_type.hpp>

#include <cassert>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace Catch {
    namespace Clara {

        class Args;
        class Parser;

        // enum of result types from a parse
        enum class ParseResultType {
            Matched,
            NoMatch,
            ShortCircuitAll,
            ShortCircuitSame
        };

        struct accept_many_t {};
        constexpr accept_many_t accept_many {};

        namespace Detail {
            struct fake_arg {
                template <typename T>
                operator T();
            };

            template <typename F, typename = void>
            static constexpr bool is_unary_function_v = false;

            template <typename F>
            static constexpr bool is_unary_function_v<
                F,
                Catch::Detail::void_t<decltype( std::declval<F>()(
                    fake_arg() ) )>> = true;

            // Traits for extracting arg and return type of lambdas (for single
            // argument lambdas)
            template <typename L>
            struct UnaryLambdaTraits
                : UnaryLambdaTraits<decltype( &L::operator() )> {};

            template <typename ClassT, typename ReturnT, typename... Args>
            struct UnaryLambdaTraits<ReturnT ( ClassT::* )( Args... ) const> {
                static const bool isValid = false;
            };

            template <typename ClassT, typename ReturnT, typename ArgT>
            struct UnaryLambdaTraits<ReturnT ( ClassT::* )( ArgT ) const> {
                static const bool isValid = true;
                using ArgType = std::remove_const_t<std::remove_reference_t<ArgT>>;
                using ReturnType = ReturnT;
            };

            class TokenStream;

            // Wraps a token coming from a token stream. These may not directly
            // correspond to strings as a single string may encode an option +
            // its argument if the : or = form is used
            enum class TokenType { Option, Argument };
            struct Token {
                TokenType type;
                StringRef token;
            };

            // Abstracts iterators into args as a stream of tokens, with option
            // arguments uniformly handled
            class TokenStream {
                using Iterator = std::vector<StringRef>::const_iterator;
                Iterator it;
                Iterator itEnd;
                std::vector<Token> m_tokenBuffer;
                void loadBuffer();

            public:
                explicit TokenStream( Args const& args );
                TokenStream( Iterator it, Iterator itEnd );

                explicit operator bool() const {
                    return !m_tokenBuffer.empty() || it != itEnd;
                }

                size_t count() const {
                    return m_tokenBuffer.size() + ( itEnd - it );
                }

                Token operator*() const {
                    assert( !m_tokenBuffer.empty() );
                    return m_tokenBuffer.front();
                }

                Token const* operator->() const {
                    assert( !m_tokenBuffer.empty() );
                    return &m_tokenBuffer.front();
                }

                TokenStream& operator++();
            };

            //! Denotes type of a parsing result
            enum class ResultType {
                Ok,          ///< No errors
                LogicError,  ///< Error in user-specified arguments for
                             ///< construction
                RuntimeError ///< Error in parsing inputs
            };

            class ResultBase {
            protected:
                ResultBase( ResultType type ): m_type( type ) {}
                virtual ~ResultBase(); // = default;


                ResultBase(ResultBase const&) = default;
                ResultBase& operator=(ResultBase const&) = default;
                ResultBase(ResultBase&&) = default;
                ResultBase& operator=(ResultBase&&) = default;

                virtual void enforceOk() const = 0;

                ResultType m_type;
            };

            template <typename T>
            class ResultValueBase : public ResultBase {
            public:
                T const& value() const& {
                    enforceOk();
                    return m_value;
                }
                T&& value() && {
                    enforceOk();
                    return CATCH_MOVE( m_value );
                }

            protected:
                ResultValueBase( ResultType type ): ResultBase( type ) {}

                ResultValueBase( ResultValueBase const& other ):
                    ResultBase( other ) {
                    if ( m_type == ResultType::Ok )
                        new ( &m_value ) T( other.m_value );
                }
                ResultValueBase( ResultValueBase&& other ):
                    ResultBase( other ) {
                    if ( m_type == ResultType::Ok )
                        new ( &m_value ) T( CATCH_MOVE(other.m_value) );
                }


                ResultValueBase( ResultType, T const& value ):
                    ResultBase( ResultType::Ok ) {
                    new ( &m_value ) T( value );
                }
                ResultValueBase( ResultType, T&& value ):
                    ResultBase( ResultType::Ok ) {
                    new ( &m_value ) T( CATCH_MOVE(value) );
                }

                ResultValueBase& operator=( ResultValueBase const& other ) {
                    if ( m_type == ResultType::Ok )
                        m_value.~T();
                    ResultBase::operator=( other );
                    if ( m_type == ResultType::Ok )
                        new ( &m_value ) T( other.m_value );
                    return *this;
                }
                ResultValueBase& operator=( ResultValueBase&& other ) {
                    if ( m_type == ResultType::Ok ) m_value.~T();
                    ResultBase::operator=( other );
                    if ( m_type == ResultType::Ok )
                        new ( &m_value ) T( CATCH_MOVE(other.m_value) );
                    return *this;
                }


                ~ResultValueBase() override {
                    if ( m_type == ResultType::Ok )
                        m_value.~T();
                }

                union {
                    T m_value;
                };
            };

            template <> class ResultValueBase<void> : public ResultBase {
            protected:
                using ResultBase::ResultBase;
            };

            template <typename T = void>
            class BasicResult : public ResultValueBase<T> {
            public:
                template <typename U>
                explicit BasicResult( BasicResult<U> const& other ):
                    ResultValueBase<T>( other.type() ),
                    m_errorMessage( other.errorMessage() ) {
                    assert( type() != ResultType::Ok );
                }

                template <typename U>
                static auto ok( U&& value ) -> BasicResult {
                    return { ResultType::Ok, CATCH_FORWARD(value) };
                }
                static auto ok() -> BasicResult { return { ResultType::Ok }; }
                static auto logicError( std::string&& message )
                    -> BasicResult {
                    return { ResultType::LogicError, CATCH_MOVE(message) };
                }
                static auto runtimeError( std::string&& message )
                    -> BasicResult {
                    return { ResultType::RuntimeError, CATCH_MOVE(message) };
                }

                explicit operator bool() const {
                    return m_type == ResultType::Ok;
                }
                auto type() const -> ResultType { return m_type; }
                auto errorMessage() const -> std::string const& {
                    return m_errorMessage;
                }

            protected:
                void enforceOk() const override {

                    // Errors shouldn't reach this point, but if they do
                    // the actual error message will be in m_errorMessage
                    assert( m_type != ResultType::LogicError );
                    assert( m_type != ResultType::RuntimeError );
                    if ( m_type != ResultType::Ok )
                        std::abort();
                }

                std::string
                    m_errorMessage; // Only populated if resultType is an error

                BasicResult( ResultType type,
                             std::string&& message ):
                    ResultValueBase<T>( type ), m_errorMessage( CATCH_MOVE(message) ) {
                    assert( m_type != ResultType::Ok );
                }

                using ResultValueBase<T>::ResultValueBase;
                using ResultBase::m_type;
            };

            class ParseState {
            public:
                ParseState( ParseResultType type,
                            TokenStream remainingTokens );

                ParseResultType type() const { return m_type; }
                TokenStream const& remainingTokens() const& {
                    return m_remainingTokens;
                }
                TokenStream&& remainingTokens() && {
                    return CATCH_MOVE( m_remainingTokens );
                }

            private:
                ParseResultType m_type;
                TokenStream m_remainingTokens;
            };

            using Result = BasicResult<void>;
            using ParserResult = BasicResult<ParseResultType>;
            using InternalParseResult = BasicResult<ParseState>;

            struct HelpColumns {
                std::string left;
                StringRef descriptions;
            };

            template <typename T>
            ParserResult convertInto( std::string const& source, T& target ) {
                std::stringstream ss( source );
                ss >> target;
                if ( ss.fail() ) {
                    return ParserResult::runtimeError(
                        "Unable to convert '" + source +
                        "' to destination type" );
                } else {
                    return ParserResult::ok( ParseResultType::Matched );
                }
            }
            ParserResult convertInto( std::string const& source,
                                      std::string& target );
            ParserResult convertInto( std::string const& source, bool& target );

#ifdef CLARA_CONFIG_OPTIONAL_TYPE
            template <typename T>
            auto convertInto( std::string const& source,
                              CLARA_CONFIG_OPTIONAL_TYPE<T>& target )
                -> ParserResult {
                T temp;
                auto result = convertInto( source, temp );
                if ( result )
                    target = CATCH_MOVE( temp );
                return result;
            }
#endif // CLARA_CONFIG_OPTIONAL_TYPE

            struct BoundRef : Catch::Detail::NonCopyable {
                virtual ~BoundRef() = default;
                virtual bool isContainer() const;
                virtual bool isFlag() const;
            };
            struct BoundValueRefBase : BoundRef {
                virtual auto setValue( std::string const& arg )
                    -> ParserResult = 0;
            };
            struct BoundFlagRefBase : BoundRef {
                virtual auto setFlag( bool flag ) -> ParserResult = 0;
                bool isFlag() const override;
            };

            template <typename T> struct BoundValueRef : BoundValueRefBase {
                T& m_ref;

                explicit BoundValueRef( T& ref ): m_ref( ref ) {}

                ParserResult setValue( std::string const& arg ) override {
                    return convertInto( arg, m_ref );
                }
            };

            template <typename T>
            struct BoundValueRef<std::vector<T>> : BoundValueRefBase {
                std::vector<T>& m_ref;

                explicit BoundValueRef( std::vector<T>& ref ): m_ref( ref ) {}

                auto isContainer() const -> bool override { return true; }

                auto setValue( std::string const& arg )
                    -> ParserResult override {
                    T temp;
                    auto result = convertInto( arg, temp );
                    if ( result )
                        m_ref.push_back( temp );
                    return result;
                }
            };

            struct BoundFlagRef : BoundFlagRefBase {
                bool& m_ref;

                explicit BoundFlagRef( bool& ref ): m_ref( ref ) {}

                ParserResult setFlag( bool flag ) override;
            };

            template <typename ReturnType> struct LambdaInvoker {
                static_assert(
                    std::is_same<ReturnType, ParserResult>::value,
                    "Lambda must return void or clara::ParserResult" );

                template <typename L, typename ArgType>
                static auto invoke( L const& lambda, ArgType const& arg )
                    -> ParserResult {
                    return lambda( arg );
                }
            };

            template <> struct LambdaInvoker<void> {
                template <typename L, typename ArgType>
                static auto invoke( L const& lambda, ArgType const& arg )
                    -> ParserResult {
                    lambda( arg );
                    return ParserResult::ok( ParseResultType::Matched );
                }
            };

            template <typename ArgType, typename L>
            auto invokeLambda( L const& lambda, std::string const& arg )
                -> ParserResult {
                ArgType temp{};
                auto result = convertInto( arg, temp );
                return !result ? result
                               : LambdaInvoker<typename UnaryLambdaTraits<
                                     L>::ReturnType>::invoke( lambda, temp );
            }

            template <typename L> struct BoundLambda : BoundValueRefBase {
                L m_lambda;

                static_assert(
                    UnaryLambdaTraits<L>::isValid,
                    "Supplied lambda must take exactly one argument" );
                explicit BoundLambda( L const& lambda ): m_lambda( lambda ) {}

                auto setValue( std::string const& arg )
                    -> ParserResult override {
                    return invokeLambda<typename UnaryLambdaTraits<L>::ArgType>(
                        m_lambda, arg );
                }
            };

            template <typename L> struct BoundManyLambda : BoundLambda<L> {
                explicit BoundManyLambda( L const& lambda ): BoundLambda<L>( lambda ) {}
                bool isContainer() const override { return true; }
            };

            template <typename L> struct BoundFlagLambda : BoundFlagRefBase {
                L m_lambda;

                static_assert(
                    UnaryLambdaTraits<L>::isValid,
                    "Supplied lambda must take exactly one argument" );
                static_assert(
                    std::is_same<typename UnaryLambdaTraits<L>::ArgType,
                                 bool>::value,
                    "flags must be boolean" );

                explicit BoundFlagLambda( L const& lambda ):
                    m_lambda( lambda ) {}

                auto setFlag( bool flag ) -> ParserResult override {
                    return LambdaInvoker<typename UnaryLambdaTraits<
                        L>::ReturnType>::invoke( m_lambda, flag );
                }
            };

            enum class Optionality { Optional, Required };

            class ParserBase {
            public:
                virtual ~ParserBase() = default;
                virtual auto validate() const -> Result { return Result::ok(); }
                virtual auto parse( std::string const& exeName,
                                    TokenStream tokens ) const
                    -> InternalParseResult = 0;
                virtual size_t cardinality() const;

                InternalParseResult parse( Args const& args ) const;
            };

            template <typename DerivedT>
            class ComposableParserImpl : public ParserBase {
            public:
                template <typename T>
                auto operator|( T const& other ) const -> Parser;
            };

            // Common code and state for Args and Opts
            template <typename DerivedT>
            class ParserRefImpl : public ComposableParserImpl<DerivedT> {
            protected:
                Optionality m_optionality = Optionality::Optional;
                std::shared_ptr<BoundRef> m_ref;
                StringRef m_hint;
                StringRef m_description;

                explicit ParserRefImpl( std::shared_ptr<BoundRef> const& ref ):
                    m_ref( ref ) {}

            public:
                template <typename LambdaT>
                ParserRefImpl( accept_many_t,
                               LambdaT const& ref,
                               StringRef hint ):
                    m_ref( std::make_shared<BoundManyLambda<LambdaT>>( ref ) ),
                    m_hint( hint ) {}

                template <typename T,
                          typename = typename std::enable_if_t<
                              !Detail::is_unary_function_v<T>>>
                ParserRefImpl( T& ref, StringRef hint ):
                    m_ref( std::make_shared<BoundValueRef<T>>( ref ) ),
                    m_hint( hint ) {}

                template <typename LambdaT,
                          typename = typename std::enable_if_t<
                              Detail::is_unary_function_v<LambdaT>>>
                ParserRefImpl( LambdaT const& ref, StringRef hint ):
                    m_ref( std::make_shared<BoundLambda<LambdaT>>( ref ) ),
                    m_hint( hint ) {}

                DerivedT& operator()( StringRef description ) & {
                    m_description = description;
                    return static_cast<DerivedT&>( *this );
                }
                DerivedT&& operator()( StringRef description ) && {
                    m_description = description;
                    return static_cast<DerivedT&&>( *this );
                }

                auto optional() -> DerivedT& {
                    m_optionality = Optionality::Optional;
                    return static_cast<DerivedT&>( *this );
                }

                auto required() -> DerivedT& {
                    m_optionality = Optionality::Required;
                    return static_cast<DerivedT&>( *this );
                }

                auto isOptional() const -> bool {
                    return m_optionality == Optionality::Optional;
                }

                auto cardinality() const -> size_t override {
                    if ( m_ref->isContainer() )
                        return 0;
                    else
                        return 1;
                }

                StringRef hint() const { return m_hint; }
            };

        } // namespace detail


        // A parser for arguments
        class Arg : public Detail::ParserRefImpl<Arg> {
        public:
            using ParserRefImpl::ParserRefImpl;
            using ParserBase::parse;

            Detail::InternalParseResult
                parse(std::string const&,
                      Detail::TokenStream tokens) const override;
        };

        // A parser for options
        class Opt : public Detail::ParserRefImpl<Opt> {
        protected:
            std::vector<StringRef> m_optNames;

        public:
            template <typename LambdaT>
            explicit Opt(LambdaT const& ref) :
                ParserRefImpl(
                    std::make_shared<Detail::BoundFlagLambda<LambdaT>>(ref)) {}

            explicit Opt(bool& ref);

            template <typename LambdaT,
                      typename = typename std::enable_if_t<
                          Detail::is_unary_function_v<LambdaT>>>
            Opt( LambdaT const& ref, StringRef hint ):
                ParserRefImpl( ref, hint ) {}

            template <typename LambdaT>
            Opt( accept_many_t, LambdaT const& ref, StringRef hint ):
                ParserRefImpl( accept_many, ref, hint ) {}

            template <typename T,
                      typename = typename std::enable_if_t<
                          !Detail::is_unary_function_v<T>>>
            Opt( T& ref, StringRef hint ):
                ParserRefImpl( ref, hint ) {}

            Opt& operator[]( StringRef optName ) & {
                m_optNames.push_back(optName);
                return *this;
            }
            Opt&& operator[]( StringRef optName ) && {
                m_optNames.push_back( optName );
                return CATCH_MOVE(*this);
            }

            Detail::HelpColumns getHelpColumns() const;

            bool isMatch(StringRef optToken) const;

            using ParserBase::parse;

            Detail::InternalParseResult
                parse(std::string const&,
                      Detail::TokenStream tokens) const override;

            Detail::Result validate() const override;
        };

        // Specifies the name of the executable
        class ExeName : public Detail::ComposableParserImpl<ExeName> {
            std::shared_ptr<std::string> m_name;
            std::shared_ptr<Detail::BoundValueRefBase> m_ref;

        public:
            ExeName();
            explicit ExeName(std::string& ref);

            template <typename LambdaT>
            explicit ExeName(LambdaT const& lambda) : ExeName() {
                m_ref = std::make_shared<Detail::BoundLambda<LambdaT>>(lambda);
            }

            // The exe name is not parsed out of the normal tokens, but is
            // handled specially
            Detail::InternalParseResult
                parse(std::string const&,
                      Detail::TokenStream tokens) const override;

            std::string const& name() const { return *m_name; }
            Detail::ParserResult set(std::string const& newName);
        };


        // A Combined parser
        class Parser : Detail::ParserBase {
            mutable ExeName m_exeName;
            std::vector<Opt> m_options;
            std::vector<Arg> m_args;

        public:

            auto operator|=(ExeName const& exeName) -> Parser& {
                m_exeName = exeName;
                return *this;
            }

            auto operator|=(Arg const& arg) -> Parser& {
                m_args.push_back(arg);
                return *this;
            }

            friend Parser& operator|=( Parser& p, Opt const& opt ) {
                p.m_options.push_back( opt );
                return p;
            }
            friend Parser& operator|=( Parser& p, Opt&& opt ) {
                p.m_options.push_back( CATCH_MOVE(opt) );
                return p;
            }

            Parser& operator|=(Parser const& other);

            template <typename T>
            friend Parser operator|( Parser const& p, T&& rhs ) {
                Parser temp( p );
                temp |= rhs;
                return temp;
            }

            template <typename T>
            friend Parser operator|( Parser&& p, T&& rhs ) {
                p |= CATCH_FORWARD(rhs);
                return CATCH_MOVE(p);
            }

            std::vector<Detail::HelpColumns> getHelpColumns() const;

            void writeToStream(std::ostream& os) const;

            friend auto operator<<(std::ostream& os, Parser const& parser)
                -> std::ostream& {
                parser.writeToStream(os);
                return os;
            }

            Detail::Result validate() const override;

            using ParserBase::parse;
            Detail::InternalParseResult
                parse(std::string const& exeName,
                      Detail::TokenStream tokens) const override;
        };

        /**
         * Wrapper over argc + argv, assumes that the inputs outlive it
         */
        class Args {
            friend Detail::TokenStream;
            StringRef m_exeName;
            std::vector<StringRef> m_args;

        public:
            Args(int argc, char const* const* argv);
            // Helper constructor for testing
            Args(std::initializer_list<StringRef> args);

            StringRef exeName() const { return m_exeName; }
        };


        // Convenience wrapper for option parser that specifies the help option
        struct Help : Opt {
            Help(bool& showHelpFlag);
        };

        // Result type for parser operation
        using Detail::ParserResult;

        namespace Detail {
            template <typename DerivedT>
            template <typename T>
            Parser
                ComposableParserImpl<DerivedT>::operator|(T const& other) const {
                return Parser() | static_cast<DerivedT const&>(*this) | other;
            }
        }

    } // namespace Clara
} // namespace Catch

#if defined( __clang__ )
#    pragma clang diagnostic pop
#endif

#if defined( __GNUC__ )
#    pragma GCC diagnostic pop
#endif

#endif // CATCH_CLARA_HPP_INCLUDED
