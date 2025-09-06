
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_GENERATORS_HPP_INCLUDED
#define CATCH_GENERATORS_HPP_INCLUDED

#include <catch2/catch_tostring.hpp>
#include <catch2/interfaces/catch_interfaces_generatortracker.hpp>
#include <catch2/internal/catch_source_line_info.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_unique_name.hpp>

#include <vector>
#include <tuple>

namespace Catch {

namespace Generators {

namespace Detail {

    //! Throws GeneratorException with the provided message
    [[noreturn]]
    void throw_generator_exception(char const * msg);

} // end namespace detail

    template<typename T>
    class IGenerator : public GeneratorUntypedBase {
        std::string stringifyImpl() const override {
            return ::Catch::Detail::stringify( get() );
        }

    public:
        // Returns the current element of the generator
        //
        // \Precondition The generator is either freshly constructed,
        // or the last call to `next()` returned true
        virtual T const& get() const = 0;
        using type = T;
    };

    template <typename T>
    using GeneratorPtr = Catch::Detail::unique_ptr<IGenerator<T>>;

    template <typename T>
    class GeneratorWrapper final {
        GeneratorPtr<T> m_generator;
    public:
        //! Takes ownership of the passed pointer.
        GeneratorWrapper(IGenerator<T>* generator):
            m_generator(generator) {}
        GeneratorWrapper(GeneratorPtr<T> generator):
            m_generator(CATCH_MOVE(generator)) {}

        T const& get() const {
            return m_generator->get();
        }
        bool next() {
            return m_generator->countedNext();
        }
    };


    template<typename T>
    class SingleValueGenerator final : public IGenerator<T> {
        T m_value;
    public:
        SingleValueGenerator(T const& value) :
            m_value(value)
        {}
        SingleValueGenerator(T&& value):
            m_value(CATCH_MOVE(value))
        {}

        T const& get() const override {
            return m_value;
        }
        bool next() override {
            return false;
        }
    };

    template<typename T>
    class FixedValuesGenerator final : public IGenerator<T> {
        static_assert(!std::is_same<T, bool>::value,
            "FixedValuesGenerator does not support bools because of std::vector<bool>"
            "specialization, use SingleValue Generator instead.");
        std::vector<T> m_values;
        size_t m_idx = 0;
    public:
        FixedValuesGenerator( std::initializer_list<T> values ) : m_values( values ) {}

        T const& get() const override {
            return m_values[m_idx];
        }
        bool next() override {
            ++m_idx;
            return m_idx < m_values.size();
        }
    };

    template <typename T, typename DecayedT = std::decay_t<T>>
    GeneratorWrapper<DecayedT> value( T&& value ) {
        return GeneratorWrapper<DecayedT>(
            Catch::Detail::make_unique<SingleValueGenerator<DecayedT>>(
                CATCH_FORWARD( value ) ) );
    }
    template <typename T>
    GeneratorWrapper<T> values(std::initializer_list<T> values) {
        return GeneratorWrapper<T>(Catch::Detail::make_unique<FixedValuesGenerator<T>>(values));
    }

    template<typename T>
    class Generators : public IGenerator<T> {
        std::vector<GeneratorWrapper<T>> m_generators;
        size_t m_current = 0;

        void add_generator( GeneratorWrapper<T>&& generator ) {
            m_generators.emplace_back( CATCH_MOVE( generator ) );
        }
        void add_generator( T const& val ) {
            m_generators.emplace_back( value( val ) );
        }
        void add_generator( T&& val ) {
            m_generators.emplace_back( value( CATCH_MOVE( val ) ) );
        }
        template <typename U>
        std::enable_if_t<!std::is_same<std::decay_t<U>, T>::value>
        add_generator( U&& val ) {
            add_generator( T( CATCH_FORWARD( val ) ) );
        }

        template <typename U> void add_generators( U&& valueOrGenerator ) {
            add_generator( CATCH_FORWARD( valueOrGenerator ) );
        }

        template <typename U, typename... Gs>
        void add_generators( U&& valueOrGenerator, Gs&&... moreGenerators ) {
            add_generator( CATCH_FORWARD( valueOrGenerator ) );
            add_generators( CATCH_FORWARD( moreGenerators )... );
        }

    public:
        template <typename... Gs>
        Generators(Gs &&... moreGenerators) {
            m_generators.reserve(sizeof...(Gs));
            add_generators(CATCH_FORWARD(moreGenerators)...);
        }

        T const& get() const override {
            return m_generators[m_current].get();
        }

        bool next() override {
            if (m_current >= m_generators.size()) {
                return false;
            }
            const bool current_status = m_generators[m_current].next();
            if (!current_status) {
                ++m_current;
            }
            return m_current < m_generators.size();
        }
    };


    template <typename... Ts>
    GeneratorWrapper<std::tuple<std::decay_t<Ts>...>>
    table( std::initializer_list<std::tuple<std::decay_t<Ts>...>> tuples ) {
        return values<std::tuple<Ts...>>( tuples );
    }

    // Tag type to signal that a generator sequence should convert arguments to a specific type
    template <typename T>
    struct as {};

    template<typename T, typename... Gs>
    auto makeGenerators( GeneratorWrapper<T>&& generator, Gs &&... moreGenerators ) -> Generators<T> {
        return Generators<T>(CATCH_MOVE(generator), CATCH_FORWARD(moreGenerators)...);
    }
    template<typename T>
    auto makeGenerators( GeneratorWrapper<T>&& generator ) -> Generators<T> {
        return Generators<T>(CATCH_MOVE(generator));
    }
    template<typename T, typename... Gs>
    auto makeGenerators( T&& val, Gs &&... moreGenerators ) -> Generators<std::decay_t<T>> {
        return makeGenerators( value( CATCH_FORWARD( val ) ), CATCH_FORWARD( moreGenerators )... );
    }
    template<typename T, typename U, typename... Gs>
    auto makeGenerators( as<T>, U&& val, Gs &&... moreGenerators ) -> Generators<T> {
        return makeGenerators( value( T( CATCH_FORWARD( val ) ) ), CATCH_FORWARD( moreGenerators )... );
    }

    IGeneratorTracker* acquireGeneratorTracker( StringRef generatorName,
                                                SourceLineInfo const& lineInfo );
    IGeneratorTracker* createGeneratorTracker( StringRef generatorName,
                                               SourceLineInfo lineInfo,
                                               GeneratorBasePtr&& generator );

    template<typename L>
    auto generate( StringRef generatorName, SourceLineInfo const& lineInfo, L const& generatorExpression ) -> typename decltype(generatorExpression())::type {
        using UnderlyingType = typename decltype(generatorExpression())::type;

        IGeneratorTracker* tracker = acquireGeneratorTracker( generatorName, lineInfo );
        // Creation of tracker is delayed after generator creation, so
        // that constructing generator can fail without breaking everything.
        if (!tracker) {
            tracker = createGeneratorTracker(
                generatorName,
                lineInfo,
                Catch::Detail::make_unique<Generators<UnderlyingType>>(
                    generatorExpression() ) );
        }

        auto const& generator = static_cast<IGenerator<UnderlyingType> const&>( *tracker->getGenerator() );
        return generator.get();
    }

} // namespace Generators
} // namespace Catch

#define CATCH_INTERNAL_GENERATOR_STRINGIZE_IMPL( ... ) #__VA_ARGS__##_catch_sr
#define CATCH_INTERNAL_GENERATOR_STRINGIZE(...) CATCH_INTERNAL_GENERATOR_STRINGIZE_IMPL(__VA_ARGS__)

#define GENERATE( ... ) \
    Catch::Generators::generate( CATCH_INTERNAL_GENERATOR_STRINGIZE(INTERNAL_CATCH_UNIQUE_NAME(generator)), \
                                 CATCH_INTERNAL_LINEINFO, \
                                 [ ]{ using namespace Catch::Generators; return makeGenerators( __VA_ARGS__ ); } ) //NOLINT(google-build-using-namespace)
#define GENERATE_COPY( ... ) \
    Catch::Generators::generate( CATCH_INTERNAL_GENERATOR_STRINGIZE(INTERNAL_CATCH_UNIQUE_NAME(generator)), \
                                 CATCH_INTERNAL_LINEINFO, \
                                 [=]{ using namespace Catch::Generators; return makeGenerators( __VA_ARGS__ ); } ) //NOLINT(google-build-using-namespace)
#define GENERATE_REF( ... ) \
    Catch::Generators::generate( CATCH_INTERNAL_GENERATOR_STRINGIZE(INTERNAL_CATCH_UNIQUE_NAME(generator)), \
                                 CATCH_INTERNAL_LINEINFO, \
                                 [&]{ using namespace Catch::Generators; return makeGenerators( __VA_ARGS__ ); } ) //NOLINT(google-build-using-namespace)

#endif // CATCH_GENERATORS_HPP_INCLUDED
