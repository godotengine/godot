
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

//  Catch v3.9.1
//  Generated: 2025-08-09 00:29:21.552225
//  ----------------------------------------------------------
//  This file is an amalgamation of multiple different files.
//  You probably shouldn't edit it directly.
//  ----------------------------------------------------------

#include "catch_amalgamated.hpp"


#ifndef CATCH_WINDOWS_H_PROXY_HPP_INCLUDED
#define CATCH_WINDOWS_H_PROXY_HPP_INCLUDED


#if defined(CATCH_PLATFORM_WINDOWS)

// We might end up with the define made globally through the compiler,
// and we don't want to trigger warnings for this
#if !defined(NOMINMAX)
#  define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#  define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#endif // defined(CATCH_PLATFORM_WINDOWS)

#endif // CATCH_WINDOWS_H_PROXY_HPP_INCLUDED




namespace Catch {
    namespace Benchmark {
        namespace Detail {
            ChronometerConcept::~ChronometerConcept() = default;
        } // namespace Detail
    } // namespace Benchmark
} // namespace Catch


// Adapted from donated nonius code.


#include <vector>

namespace Catch {
    namespace Benchmark {
        namespace Detail {
            SampleAnalysis analyse(const IConfig &cfg, FDuration* first, FDuration* last) {
                if (!cfg.benchmarkNoAnalysis()) {
                    std::vector<double> samples;
                    samples.reserve(static_cast<size_t>(last - first));
                    for (auto current = first; current != last; ++current) {
                        samples.push_back( current->count() );
                    }

                    auto analysis = Catch::Benchmark::Detail::analyse_samples(
                        cfg.benchmarkConfidenceInterval(),
                        cfg.benchmarkResamples(),
                        samples.data(),
                        samples.data() + samples.size() );
                    auto outliers = Catch::Benchmark::Detail::classify_outliers(
                        samples.data(), samples.data() + samples.size() );

                    auto wrap_estimate = [](Estimate<double> e) {
                        return Estimate<FDuration> {
                            FDuration(e.point),
                                FDuration(e.lower_bound),
                                FDuration(e.upper_bound),
                                e.confidence_interval,
                        };
                    };
                    std::vector<FDuration> samples2;
                    samples2.reserve(samples.size());
                    for (auto s : samples) {
                        samples2.push_back( FDuration( s ) );
                    }

                    return {
                        CATCH_MOVE(samples2),
                        wrap_estimate(analysis.mean),
                        wrap_estimate(analysis.standard_deviation),
                        outliers,
                        analysis.outlier_variance,
                    };
                } else {
                    std::vector<FDuration> samples;
                    samples.reserve(static_cast<size_t>(last - first));

                    FDuration mean = FDuration(0);
                    int i = 0;
                    for (auto it = first; it < last; ++it, ++i) {
                        samples.push_back(*it);
                        mean += *it;
                    }
                    mean /= i;

                    return SampleAnalysis{
                        CATCH_MOVE(samples),
                        Estimate<FDuration>{ mean, mean, mean, 0.0 },
                        Estimate<FDuration>{ FDuration( 0 ),
                                             FDuration( 0 ),
                                             FDuration( 0 ),
                                             0.0 },
                        OutlierClassification{},
                        0.0
                    };
                }
            }
        } // namespace Detail
    } // namespace Benchmark
} // namespace Catch




namespace Catch {
    namespace Benchmark {
        namespace Detail {
            struct do_nothing {
                void operator()() const {}
            };

            BenchmarkFunction::callable::~callable() = default;
            BenchmarkFunction::BenchmarkFunction():
                f( new model<do_nothing>{ {} } ){}
        } // namespace Detail
    } // namespace Benchmark
} // namespace Catch




#include <exception>

namespace Catch {
    namespace Benchmark {
        namespace Detail {
            struct optimized_away_error : std::exception {
                const char* what() const noexcept override;
            };

            const char* optimized_away_error::what() const noexcept {
                return "could not measure benchmark, maybe it was optimized away";
            }

            void throw_optimized_away_error() {
                Catch::throw_exception(optimized_away_error{});
            }

        } // namespace Detail
    } // namespace Benchmark
} // namespace Catch


// Adapted from donated nonius code.



#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>


#if defined(CATCH_CONFIG_USE_ASYNC)
#include <future>
#endif

namespace Catch {
    namespace Benchmark {
        namespace Detail {
            namespace {

                template <typename URng, typename Estimator>
                static sample
                resample( URng& rng,
                          unsigned int resamples,
                          double const* first,
                          double const* last,
                          Estimator& estimator ) {
                    auto n = static_cast<size_t>( last - first );
                    Catch::uniform_integer_distribution<size_t> dist( 0, n - 1 );

                    sample out;
                    out.reserve( resamples );
                    std::vector<double> resampled;
                    resampled.reserve( n );
                    for ( size_t i = 0; i < resamples; ++i ) {
                        resampled.clear();
                        for ( size_t s = 0; s < n; ++s ) {
                            resampled.push_back( first[dist( rng )] );
                        }
                        const auto estimate =
                            estimator( resampled.data(), resampled.data() + resampled.size() );
                        out.push_back( estimate );
                    }
                    std::sort( out.begin(), out.end() );
                    return out;
                }

                static double outlier_variance( Estimate<double> mean,
                                                Estimate<double> stddev,
                                                int n ) {
                    double sb = stddev.point;
                    double mn = mean.point / n;
                    double mg_min = mn / 2.;
                    double sg = (std::min)( mg_min / 4., sb / std::sqrt( n ) );
                    double sg2 = sg * sg;
                    double sb2 = sb * sb;

                    auto c_max = [n, mn, sb2, sg2]( double x ) -> double {
                        double k = mn - x;
                        double d = k * k;
                        double nd = n * d;
                        double k0 = -n * nd;
                        double k1 = sb2 - n * sg2 + nd;
                        double det = k1 * k1 - 4 * sg2 * k0;
                        return static_cast<int>( -2. * k0 /
                                                 ( k1 + std::sqrt( det ) ) );
                    };

                    auto var_out = [n, sb2, sg2]( double c ) {
                        double nc = n - c;
                        return ( nc / n ) * ( sb2 - nc * sg2 );
                    };

                    return (std::min)( var_out( 1 ),
                                       var_out(
                                           (std::min)( c_max( 0. ),
                                                       c_max( mg_min ) ) ) ) /
                           sb2;
                }

                static double erf_inv( double x ) {
                    // Code accompanying the article "Approximating the erfinv
                    // function" in GPU Computing Gems, Volume 2
                    double w, p;

                    w = -log( ( 1.0 - x ) * ( 1.0 + x ) );

                    if ( w < 6.250000 ) {
                        w = w - 3.125000;
                        p = -3.6444120640178196996e-21;
                        p = -1.685059138182016589e-19 + p * w;
                        p = 1.2858480715256400167e-18 + p * w;
                        p = 1.115787767802518096e-17 + p * w;
                        p = -1.333171662854620906e-16 + p * w;
                        p = 2.0972767875968561637e-17 + p * w;
                        p = 6.6376381343583238325e-15 + p * w;
                        p = -4.0545662729752068639e-14 + p * w;
                        p = -8.1519341976054721522e-14 + p * w;
                        p = 2.6335093153082322977e-12 + p * w;
                        p = -1.2975133253453532498e-11 + p * w;
                        p = -5.4154120542946279317e-11 + p * w;
                        p = 1.051212273321532285e-09 + p * w;
                        p = -4.1126339803469836976e-09 + p * w;
                        p = -2.9070369957882005086e-08 + p * w;
                        p = 4.2347877827932403518e-07 + p * w;
                        p = -1.3654692000834678645e-06 + p * w;
                        p = -1.3882523362786468719e-05 + p * w;
                        p = 0.0001867342080340571352 + p * w;
                        p = -0.00074070253416626697512 + p * w;
                        p = -0.0060336708714301490533 + p * w;
                        p = 0.24015818242558961693 + p * w;
                        p = 1.6536545626831027356 + p * w;
                    } else if ( w < 16.000000 ) {
                        w = sqrt( w ) - 3.250000;
                        p = 2.2137376921775787049e-09;
                        p = 9.0756561938885390979e-08 + p * w;
                        p = -2.7517406297064545428e-07 + p * w;
                        p = 1.8239629214389227755e-08 + p * w;
                        p = 1.5027403968909827627e-06 + p * w;
                        p = -4.013867526981545969e-06 + p * w;
                        p = 2.9234449089955446044e-06 + p * w;
                        p = 1.2475304481671778723e-05 + p * w;
                        p = -4.7318229009055733981e-05 + p * w;
                        p = 6.8284851459573175448e-05 + p * w;
                        p = 2.4031110387097893999e-05 + p * w;
                        p = -0.0003550375203628474796 + p * w;
                        p = 0.00095328937973738049703 + p * w;
                        p = -0.0016882755560235047313 + p * w;
                        p = 0.0024914420961078508066 + p * w;
                        p = -0.0037512085075692412107 + p * w;
                        p = 0.005370914553590063617 + p * w;
                        p = 1.0052589676941592334 + p * w;
                        p = 3.0838856104922207635 + p * w;
                    } else {
                        w = sqrt( w ) - 5.000000;
                        p = -2.7109920616438573243e-11;
                        p = -2.5556418169965252055e-10 + p * w;
                        p = 1.5076572693500548083e-09 + p * w;
                        p = -3.7894654401267369937e-09 + p * w;
                        p = 7.6157012080783393804e-09 + p * w;
                        p = -1.4960026627149240478e-08 + p * w;
                        p = 2.9147953450901080826e-08 + p * w;
                        p = -6.7711997758452339498e-08 + p * w;
                        p = 2.2900482228026654717e-07 + p * w;
                        p = -9.9298272942317002539e-07 + p * w;
                        p = 4.5260625972231537039e-06 + p * w;
                        p = -1.9681778105531670567e-05 + p * w;
                        p = 7.5995277030017761139e-05 + p * w;
                        p = -0.00021503011930044477347 + p * w;
                        p = -0.00013871931833623122026 + p * w;
                        p = 1.0103004648645343977 + p * w;
                        p = 4.8499064014085844221 + p * w;
                    }
                    return p * x;
                }

                static double
                standard_deviation( double const* first, double const* last ) {
                    auto m = Catch::Benchmark::Detail::mean( first, last );
                    double variance =
                        std::accumulate( first,
                                         last,
                                         0.,
                                         [m]( double a, double b ) {
                                             double diff = b - m;
                                             return a + diff * diff;
                                         } ) /
                        static_cast<double>( last - first );
                    return std::sqrt( variance );
                }

                static sample jackknife( double ( *estimator )( double const*,
                                                                double const* ),
                                         double* first,
                                         double* last ) {
                    const auto second = first + 1;
                    sample results;
                    results.reserve( static_cast<size_t>( last - first ) );

                    for ( auto it = first; it != last; ++it ) {
                        std::iter_swap( it, first );
                        results.push_back( estimator( second, last ) );
                    }

                    return results;
                }


            } // namespace
        }     // namespace Detail
    }         // namespace Benchmark
} // namespace Catch

namespace Catch {
    namespace Benchmark {
        namespace Detail {

            double weighted_average_quantile( int k,
                                              int q,
                                              double* first,
                                              double* last ) {
                auto count = last - first;
                double idx = static_cast<double>((count - 1) * k) / static_cast<double>(q);
                int j = static_cast<int>(idx);
                double g = idx - j;
                std::nth_element(first, first + j, last);
                auto xj = first[j];
                if ( Catch::Detail::directCompare( g, 0 ) ) {
                    return xj;
                }

                auto xj1 = *std::min_element(first + (j + 1), last);
                return xj + g * (xj1 - xj);
            }

            OutlierClassification
            classify_outliers( double const* first, double const* last ) {
                std::vector<double> copy( first, last );

                auto q1 = weighted_average_quantile( 1, 4, copy.data(), copy.data() + copy.size() );
                auto q3 = weighted_average_quantile( 3, 4, copy.data(), copy.data() + copy.size() );
                auto iqr = q3 - q1;
                auto los = q1 - ( iqr * 3. );
                auto lom = q1 - ( iqr * 1.5 );
                auto him = q3 + ( iqr * 1.5 );
                auto his = q3 + ( iqr * 3. );

                OutlierClassification o;
                for ( ; first != last; ++first ) {
                    const double t = *first;
                    if ( t < los ) {
                        ++o.low_severe;
                    } else if ( t < lom ) {
                        ++o.low_mild;
                    } else if ( t > his ) {
                        ++o.high_severe;
                    } else if ( t > him ) {
                        ++o.high_mild;
                    }
                    ++o.samples_seen;
                }
                return o;
            }

            double mean( double const* first, double const* last ) {
                auto count = last - first;
                double sum = 0.;
                while (first != last) {
                    sum += *first;
                    ++first;
                }
                return sum / static_cast<double>(count);
            }

            double normal_cdf( double x ) {
                return std::erfc( -x / std::sqrt( 2.0 ) ) / 2.0;
            }

            double erfc_inv(double x) {
                return erf_inv(1.0 - x);
            }

            double normal_quantile(double p) {
                static const double ROOT_TWO = std::sqrt(2.0);

                double result = 0.0;
                assert(p >= 0 && p <= 1);
                if (p < 0 || p > 1) {
                    return result;
                }

                result = -erfc_inv(2.0 * p);
                // result *= normal distribution standard deviation (1.0) * sqrt(2)
                result *= /*sd * */ ROOT_TWO;
                // result += normal disttribution mean (0)
                return result;
            }

            Estimate<double>
            bootstrap( double confidence_level,
                       double* first,
                       double* last,
                       sample const& resample,
                       double ( *estimator )( double const*, double const* ) ) {
                auto n_samples = last - first;

                double point = estimator( first, last );
                // Degenerate case with a single sample
                if ( n_samples == 1 )
                    return { point, point, point, confidence_level };

                sample jack = jackknife( estimator, first, last );
                double jack_mean =
                    mean( jack.data(), jack.data() + jack.size() );
                double sum_squares = 0, sum_cubes = 0;
                for ( double x : jack ) {
                    auto difference = jack_mean - x;
                    auto square = difference * difference;
                    auto cube = square * difference;
                    sum_squares += square;
                    sum_cubes += cube;
                }

                double accel = sum_cubes / ( 6 * std::pow( sum_squares, 1.5 ) );
                long n = static_cast<long>( resample.size() );
                double prob_n = static_cast<double>(
                    std::count_if( resample.begin(),
                                   resample.end(),
                                   [point]( double x ) { return x < point; } )) /
                    static_cast<double>( n );
                // degenerate case with uniform samples
                if ( Catch::Detail::directCompare( prob_n, 0. ) ) {
                    return { point, point, point, confidence_level };
                }

                double bias = normal_quantile( prob_n );
                double z1 = normal_quantile( ( 1. - confidence_level ) / 2. );

                auto cumn = [n]( double x ) -> long {
                    return std::lround( normal_cdf( x ) *
                                        static_cast<double>( n ) );
                };
                auto a = [bias, accel]( double b ) {
                    return bias + b / ( 1. - accel * b );
                };
                double b1 = bias + z1;
                double b2 = bias - z1;
                double a1 = a( b1 );
                double a2 = a( b2 );
                auto lo = static_cast<size_t>( (std::max)( cumn( a1 ), 0l ) );
                auto hi =
                    static_cast<size_t>( (std::min)( cumn( a2 ), n - 1 ) );

                return { point, resample[lo], resample[hi], confidence_level };
            }

            bootstrap_analysis analyse_samples(double confidence_level,
                                               unsigned int n_resamples,
                                               double* first,
                                               double* last) {
                auto mean = &Detail::mean;
                auto stddev = &standard_deviation;

#if defined(CATCH_CONFIG_USE_ASYNC)
                auto Estimate = [=](double(*f)(double const*, double const*)) {
                    std::random_device rd;
                    auto seed = rd();
                    return std::async(std::launch::async, [=] {
                        SimplePcg32 rng( seed );
                        auto resampled = resample(rng, n_resamples, first, last, f);
                        return bootstrap(confidence_level, first, last, resampled, f);
                    });
                };

                auto mean_future = Estimate(mean);
                auto stddev_future = Estimate(stddev);

                auto mean_estimate = mean_future.get();
                auto stddev_estimate = stddev_future.get();
#else
                auto Estimate = [=](double(*f)(double const* , double const*)) {
                    std::random_device rd;
                    auto seed = rd();
                    SimplePcg32 rng( seed );
                    auto resampled = resample(rng, n_resamples, first, last, f);
                    return bootstrap(confidence_level, first, last, resampled, f);
                };

                auto mean_estimate = Estimate(mean);
                auto stddev_estimate = Estimate(stddev);
#endif // CATCH_USE_ASYNC

                auto n = static_cast<int>(last - first); // seriously, one can't use integral types without hell in C++
                double outlier_variance = Detail::outlier_variance(mean_estimate, stddev_estimate, n);

                return { mean_estimate, stddev_estimate, outlier_variance };
            }
        } // namespace Detail
    } // namespace Benchmark
} // namespace Catch



#include <cmath>
#include <limits>

namespace {

// Performs equivalent check of std::fabs(lhs - rhs) <= margin
// But without the subtraction to allow for INFINITY in comparison
bool marginComparison(double lhs, double rhs, double margin) {
    return (lhs + margin >= rhs) && (rhs + margin >= lhs);
}

}

namespace Catch {

    Approx::Approx ( double value )
    :   m_epsilon( static_cast<double>(std::numeric_limits<float>::epsilon())*100. ),
        m_margin( 0.0 ),
        m_scale( 0.0 ),
        m_value( value )
    {}

    Approx Approx::custom() {
        return Approx( 0 );
    }

    Approx Approx::operator-() const {
        auto temp(*this);
        temp.m_value = -temp.m_value;
        return temp;
    }


    std::string Approx::toString() const {
        ReusableStringStream rss;
        rss << "Approx( " << ::Catch::Detail::stringify( m_value ) << " )";
        return rss.str();
    }

    bool Approx::equalityComparisonImpl(const double other) const {
        // First try with fixed margin, then compute margin based on epsilon, scale and Approx's value
        // Thanks to Richard Harris for his help refining the scaled margin value
        return marginComparison(m_value, other, m_margin)
            || marginComparison(m_value, other, m_epsilon * (m_scale + std::fabs(std::isinf(m_value)? 0 : m_value)));
    }

    void Approx::setMargin(double newMargin) {
        CATCH_ENFORCE(newMargin >= 0,
            "Invalid Approx::margin: " << newMargin << '.'
            << " Approx::Margin has to be non-negative.");
        m_margin = newMargin;
    }

    void Approx::setEpsilon(double newEpsilon) {
        CATCH_ENFORCE(newEpsilon >= 0 && newEpsilon <= 1.0,
            "Invalid Approx::epsilon: " << newEpsilon << '.'
            << " Approx::epsilon has to be in [0, 1]");
        m_epsilon = newEpsilon;
    }

namespace literals {
    Approx operator ""_a(long double val) {
        return Approx(val);
    }
    Approx operator ""_a(unsigned long long val) {
        return Approx(val);
    }
} // end namespace literals

std::string StringMaker<Catch::Approx>::convert(Catch::Approx const& value) {
    return value.toString();
}

} // end namespace Catch



namespace Catch {

    AssertionResultData::AssertionResultData(ResultWas::OfType _resultType, LazyExpression const& _lazyExpression):
        lazyExpression(_lazyExpression),
        resultType(_resultType) {}

    std::string AssertionResultData::reconstructExpression() const {

        if( reconstructedExpression.empty() ) {
            if( lazyExpression ) {
                ReusableStringStream rss;
                rss << lazyExpression;
                reconstructedExpression = rss.str();
            }
        }
        return reconstructedExpression;
    }

    AssertionResult::AssertionResult( AssertionInfo const& info, AssertionResultData&& data )
    :   m_info( info ),
        m_resultData( CATCH_MOVE(data) )
    {}

    // Result was a success
    bool AssertionResult::succeeded() const {
        return Catch::isOk( m_resultData.resultType );
    }

    // Result was a success, or failure is suppressed
    bool AssertionResult::isOk() const {
        return Catch::isOk( m_resultData.resultType ) || shouldSuppressFailure( m_info.resultDisposition );
    }

    ResultWas::OfType AssertionResult::getResultType() const {
        return m_resultData.resultType;
    }

    bool AssertionResult::hasExpression() const {
        return !m_info.capturedExpression.empty();
    }

    bool AssertionResult::hasMessage() const {
        return !m_resultData.message.empty();
    }

    std::string AssertionResult::getExpression() const {
        // Possibly overallocating by 3 characters should be basically free
        std::string expr; expr.reserve(m_info.capturedExpression.size() + 3);
        if (isFalseTest(m_info.resultDisposition)) {
            expr += "!(";
        }
        expr += m_info.capturedExpression;
        if (isFalseTest(m_info.resultDisposition)) {
            expr += ')';
        }
        return expr;
    }

    std::string AssertionResult::getExpressionInMacro() const {
        if ( m_info.macroName.empty() ) {
            return static_cast<std::string>( m_info.capturedExpression );
        }
        std::string expr;
        expr.reserve( m_info.macroName.size() + m_info.capturedExpression.size() + 4 );
        expr += m_info.macroName;
        expr += "( ";
        expr += m_info.capturedExpression;
        expr += " )";
        return expr;
    }

    bool AssertionResult::hasExpandedExpression() const {
        return hasExpression() && getExpandedExpression() != getExpression();
    }

    std::string AssertionResult::getExpandedExpression() const {
        std::string expr = m_resultData.reconstructExpression();
        return expr.empty()
                ? getExpression()
                : expr;
    }

    StringRef AssertionResult::getMessage() const {
        return m_resultData.message;
    }
    SourceLineInfo AssertionResult::getSourceInfo() const {
        return m_info.lineInfo;
    }

    StringRef AssertionResult::getTestMacroName() const {
        return m_info.macroName;
    }

} // end namespace Catch



#include <fstream>

namespace Catch {

    namespace {
        static bool enableBazelEnvSupport() {
#if defined( CATCH_CONFIG_BAZEL_SUPPORT )
            return true;
#else
            return Detail::getEnv( "BAZEL_TEST" ) != nullptr;
#endif
        }

        struct bazelShardingOptions {
            unsigned int shardIndex, shardCount;
            std::string shardFilePath;
        };

        static Optional<bazelShardingOptions> readBazelShardingOptions() {
            const auto bazelShardIndex = Detail::getEnv( "TEST_SHARD_INDEX" );
            const auto bazelShardTotal = Detail::getEnv( "TEST_TOTAL_SHARDS" );
            const auto bazelShardInfoFile = Detail::getEnv( "TEST_SHARD_STATUS_FILE" );


            const bool has_all =
                bazelShardIndex && bazelShardTotal && bazelShardInfoFile;
            if ( !has_all ) {
                // We provide nice warning message if the input is
                // misconfigured.
                auto warn = []( const char* env_var ) {
                    Catch::cerr()
                        << "Warning: Bazel shard configuration is missing '"
                        << env_var << "'. Shard configuration is skipped.\n";
                };
                if ( !bazelShardIndex ) {
                    warn( "TEST_SHARD_INDEX" );
                }
                if ( !bazelShardTotal ) {
                    warn( "TEST_TOTAL_SHARDS" );
                }
                if ( !bazelShardInfoFile ) {
                    warn( "TEST_SHARD_STATUS_FILE" );
                }
                return {};
            }

            auto shardIndex = parseUInt( bazelShardIndex );
            if ( !shardIndex ) {
                Catch::cerr()
                    << "Warning: could not parse 'TEST_SHARD_INDEX' ('" << bazelShardIndex
                    << "') as unsigned int.\n";
                return {};
            }
            auto shardTotal = parseUInt( bazelShardTotal );
            if ( !shardTotal ) {
                Catch::cerr()
                    << "Warning: could not parse 'TEST_TOTAL_SHARD' ('"
                    << bazelShardTotal << "') as unsigned int.\n";
                return {};
            }

            return bazelShardingOptions{
                *shardIndex, *shardTotal, bazelShardInfoFile };

        }
    } // end namespace


    bool operator==( ProcessedReporterSpec const& lhs,
                     ProcessedReporterSpec const& rhs ) {
        return lhs.name == rhs.name &&
               lhs.outputFilename == rhs.outputFilename &&
               lhs.colourMode == rhs.colourMode &&
               lhs.customOptions == rhs.customOptions;
    }

    Config::Config( ConfigData const& data ):
        m_data( data ) {
        // We need to trim filter specs to avoid trouble with superfluous
        // whitespace (esp. important for bdd macros, as those are manually
        // aligned with whitespace).

        for (auto& elem : m_data.testsOrTags) {
            elem = trim(elem);
        }
        for (auto& elem : m_data.sectionsToRun) {
            elem = trim(elem);
        }

        // Insert the default reporter if user hasn't asked for a specific one
        if ( m_data.reporterSpecifications.empty() ) {
#if defined( CATCH_CONFIG_DEFAULT_REPORTER )
            const auto default_spec = CATCH_CONFIG_DEFAULT_REPORTER;
#else
            const auto default_spec = "console";
#endif
            auto parsed = parseReporterSpec(default_spec);
            CATCH_ENFORCE( parsed,
                           "Cannot parse the provided default reporter spec: '"
                               << default_spec << '\'' );
            m_data.reporterSpecifications.push_back( std::move( *parsed ) );
        }

        if ( enableBazelEnvSupport() ) {
            readBazelEnvVars();
        }

        // Bazel support can modify the test specs, so parsing has to happen
        // after reading Bazel env vars.
        TestSpecParser parser( ITagAliasRegistry::get() );
        if ( !m_data.testsOrTags.empty() ) {
            m_hasTestFilters = true;
            for ( auto const& testOrTags : m_data.testsOrTags ) {
                parser.parse( testOrTags );
            }
        }
        m_testSpec = parser.testSpec();


        // We now fixup the reporter specs to handle default output spec,
        // default colour spec, etc
        bool defaultOutputUsed = false;
        for ( auto const& reporterSpec : m_data.reporterSpecifications ) {
            // We do the default-output check separately, while always
            // using the default output below to make the code simpler
            // and avoid superfluous copies.
            if ( reporterSpec.outputFile().none() ) {
                CATCH_ENFORCE( !defaultOutputUsed,
                               "Internal error: cannot use default output for "
                               "multiple reporters" );
                defaultOutputUsed = true;
            }

            m_processedReporterSpecs.push_back( ProcessedReporterSpec{
                reporterSpec.name(),
                reporterSpec.outputFile() ? *reporterSpec.outputFile()
                                          : data.defaultOutputFilename,
                reporterSpec.colourMode().valueOr( data.defaultColourMode ),
                reporterSpec.customOptions() } );
        }
    }

    Config::~Config() = default;


    bool Config::listTests() const          { return m_data.listTests; }
    bool Config::listTags() const           { return m_data.listTags; }
    bool Config::listReporters() const      { return m_data.listReporters; }
    bool Config::listListeners() const      { return m_data.listListeners; }

    std::vector<std::string> const& Config::getTestsOrTags() const { return m_data.testsOrTags; }
    std::vector<std::string> const& Config::getSectionsToRun() const { return m_data.sectionsToRun; }

    std::vector<ReporterSpec> const& Config::getReporterSpecs() const {
        return m_data.reporterSpecifications;
    }

    std::vector<ProcessedReporterSpec> const&
    Config::getProcessedReporterSpecs() const {
        return m_processedReporterSpecs;
    }

    TestSpec const& Config::testSpec() const { return m_testSpec; }
    bool Config::hasTestFilters() const { return m_hasTestFilters; }

    bool Config::showHelp() const { return m_data.showHelp; }

    // IConfig interface
    bool Config::allowThrows() const                   { return !m_data.noThrow; }
    StringRef Config::name() const { return m_data.name.empty() ? m_data.processName : m_data.name; }
    bool Config::includeSuccessfulResults() const      { return m_data.showSuccessfulTests; }
    bool Config::warnAboutMissingAssertions() const {
        return !!( m_data.warnings & WarnAbout::NoAssertions );
    }
    bool Config::warnAboutUnmatchedTestSpecs() const {
        return !!( m_data.warnings & WarnAbout::UnmatchedTestSpec );
    }
    bool Config::zeroTestsCountAsSuccess() const       { return m_data.allowZeroTests; }
    ShowDurations Config::showDurations() const        { return m_data.showDurations; }
    double Config::minDuration() const                 { return m_data.minDuration; }
    TestRunOrder Config::runOrder() const              { return m_data.runOrder; }
    uint32_t Config::rngSeed() const                   { return m_data.rngSeed; }
    unsigned int Config::shardCount() const            { return m_data.shardCount; }
    unsigned int Config::shardIndex() const            { return m_data.shardIndex; }
    ColourMode Config::defaultColourMode() const       { return m_data.defaultColourMode; }
    bool Config::shouldDebugBreak() const              { return m_data.shouldDebugBreak; }
    int Config::abortAfter() const                     { return m_data.abortAfter; }
    bool Config::showInvisibles() const                { return m_data.showInvisibles; }
    Verbosity Config::verbosity() const                { return m_data.verbosity; }

    bool Config::skipBenchmarks() const                           { return m_data.skipBenchmarks; }
    bool Config::benchmarkNoAnalysis() const                      { return m_data.benchmarkNoAnalysis; }
    unsigned int Config::benchmarkSamples() const                 { return m_data.benchmarkSamples; }
    double Config::benchmarkConfidenceInterval() const            { return m_data.benchmarkConfidenceInterval; }
    unsigned int Config::benchmarkResamples() const               { return m_data.benchmarkResamples; }
    std::chrono::milliseconds Config::benchmarkWarmupTime() const { return std::chrono::milliseconds(m_data.benchmarkWarmupTime); }

    void Config::readBazelEnvVars() {
        // Register a JUnit reporter for Bazel. Bazel sets an environment
        // variable with the path to XML output. If this file is written to
        // during test, Bazel will not generate a default XML output.
        // This allows the XML output file to contain higher level of detail
        // than what is possible otherwise.
        const auto bazelOutputFile = Detail::getEnv( "XML_OUTPUT_FILE" );

        if ( bazelOutputFile ) {
            m_data.reporterSpecifications.push_back(
                { "junit", std::string( bazelOutputFile ), {}, {} } );
        }

        const auto bazelTestSpec = Detail::getEnv( "TESTBRIDGE_TEST_ONLY" );
        if ( bazelTestSpec ) {
            // Presumably the test spec from environment should overwrite
            // the one we got from CLI (if we got any)
            m_data.testsOrTags.clear();
            m_data.testsOrTags.push_back( bazelTestSpec );
        }

        const auto bazelShardOptions = readBazelShardingOptions();
        if ( bazelShardOptions ) {
            std::ofstream f( bazelShardOptions->shardFilePath,
                             std::ios_base::out | std::ios_base::trunc );
            if ( f.is_open() ) {
                f << "";
                m_data.shardIndex = bazelShardOptions->shardIndex;
                m_data.shardCount = bazelShardOptions->shardCount;
            }
        }
    }

} // end namespace Catch





namespace Catch {
    std::uint32_t getSeed() {
        return getCurrentContext().getConfig()->rngSeed();
    }
}



#include <cassert>
#include <stack>

namespace Catch {

    ////////////////////////////////////////////////////////////////////////////


    ScopedMessage::ScopedMessage( MessageBuilder&& builder ):
        m_info( CATCH_MOVE(builder.m_info) ) {
        m_info.message = builder.m_stream.str();
        getResultCapture().pushScopedMessage( m_info );
    }

    ScopedMessage::ScopedMessage( ScopedMessage&& old ) noexcept:
        m_info( CATCH_MOVE( old.m_info ) ) {
        old.m_moved = true;
    }

    ScopedMessage::~ScopedMessage() {
        if ( !m_moved ){
            getResultCapture().popScopedMessage(m_info);
        }
    }


    Capturer::Capturer( StringRef macroName,
                        SourceLineInfo const& lineInfo,
                        ResultWas::OfType resultType,
                        StringRef names ):
        m_resultCapture( getResultCapture() ) {
        auto trimmed = [&] (size_t start, size_t end) {
            while (names[start] == ',' || isspace(static_cast<unsigned char>(names[start]))) {
                ++start;
            }
            while (names[end] == ',' || isspace(static_cast<unsigned char>(names[end]))) {
                --end;
            }
            return names.substr(start, end - start + 1);
        };
        auto skipq = [&] (size_t start, char quote) {
            for (auto i = start + 1; i < names.size() ; ++i) {
                if (names[i] == quote)
                    return i;
                if (names[i] == '\\')
                    ++i;
            }
            CATCH_INTERNAL_ERROR("CAPTURE parsing encountered unmatched quote");
        };

        size_t start = 0;
        std::stack<char> openings;
        for (size_t pos = 0; pos < names.size(); ++pos) {
            char c = names[pos];
            switch (c) {
            case '[':
            case '{':
            case '(':
            // It is basically impossible to disambiguate between
            // comparison and start of template args in this context
//            case '<':
                openings.push(c);
                break;
            case ']':
            case '}':
            case ')':
//           case '>':
                openings.pop();
                break;
            case '"':
            case '\'':
                pos = skipq(pos, c);
                break;
            case ',':
                if (start != pos && openings.empty()) {
                    m_messages.emplace_back(macroName, lineInfo, resultType);
                    m_messages.back().message = static_cast<std::string>(trimmed(start, pos));
                    m_messages.back().message += " := ";
                    start = pos;
                }
                break;
            default:; // noop
            }
        }
        assert(openings.empty() && "Mismatched openings");
        m_messages.emplace_back(macroName, lineInfo, resultType);
        m_messages.back().message = static_cast<std::string>(trimmed(start, names.size() - 1));
        m_messages.back().message += " := ";
    }
    Capturer::~Capturer() {
        assert( m_captured == m_messages.size() );
        for ( size_t i = 0; i < m_captured; ++i )
            m_resultCapture.popScopedMessage( m_messages[i] );
    }

    void Capturer::captureValue( size_t index, std::string const& value ) {
        assert( index < m_messages.size() );
        m_messages[index].message += value;
        m_resultCapture.pushScopedMessage( m_messages[index] );
        m_captured++;
    }

} // end namespace Catch




#include <exception>

namespace Catch {

    namespace {

        class RegistryHub : public IRegistryHub,
                            public IMutableRegistryHub,
                            private Detail::NonCopyable {

        public: // IRegistryHub
            RegistryHub() = default;
            ReporterRegistry const& getReporterRegistry() const override {
                return m_reporterRegistry;
            }
            ITestCaseRegistry const& getTestCaseRegistry() const override {
                return m_testCaseRegistry;
            }
            IExceptionTranslatorRegistry const& getExceptionTranslatorRegistry() const override {
                return m_exceptionTranslatorRegistry;
            }
            ITagAliasRegistry const& getTagAliasRegistry() const override {
                return m_tagAliasRegistry;
            }
            StartupExceptionRegistry const& getStartupExceptionRegistry() const override {
                return m_exceptionRegistry;
            }

        public: // IMutableRegistryHub
            void registerReporter( std::string const& name, IReporterFactoryPtr factory ) override {
                m_reporterRegistry.registerReporter( name, CATCH_MOVE(factory) );
            }
            void registerListener( Detail::unique_ptr<EventListenerFactory> factory ) override {
                m_reporterRegistry.registerListener( CATCH_MOVE(factory) );
            }
            void registerTest( Detail::unique_ptr<TestCaseInfo>&& testInfo, Detail::unique_ptr<ITestInvoker>&& invoker ) override {
                m_testCaseRegistry.registerTest( CATCH_MOVE(testInfo), CATCH_MOVE(invoker) );
            }
            void registerTranslator( Detail::unique_ptr<IExceptionTranslator>&& translator ) override {
                m_exceptionTranslatorRegistry.registerTranslator( CATCH_MOVE(translator) );
            }
            void registerTagAlias( std::string const& alias, std::string const& tag, SourceLineInfo const& lineInfo ) override {
                m_tagAliasRegistry.add( alias, tag, lineInfo );
            }
            void registerStartupException() noexcept override {
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
                m_exceptionRegistry.add(std::current_exception());
#else
                CATCH_INTERNAL_ERROR("Attempted to register active exception under CATCH_CONFIG_DISABLE_EXCEPTIONS!");
#endif
            }
            IMutableEnumValuesRegistry& getMutableEnumValuesRegistry() override {
                return m_enumValuesRegistry;
            }

        private:
            TestRegistry m_testCaseRegistry;
            ReporterRegistry m_reporterRegistry;
            ExceptionTranslatorRegistry m_exceptionTranslatorRegistry;
            TagAliasRegistry m_tagAliasRegistry;
            StartupExceptionRegistry m_exceptionRegistry;
            Detail::EnumValuesRegistry m_enumValuesRegistry;
        };
    }

    using RegistryHubSingleton = Singleton<RegistryHub, IRegistryHub, IMutableRegistryHub>;

    IRegistryHub const& getRegistryHub() {
        return RegistryHubSingleton::get();
    }
    IMutableRegistryHub& getMutableRegistryHub() {
        return RegistryHubSingleton::getMutable();
    }
    void cleanUp() {
        cleanupSingletons();
        cleanUpContext();
    }
    std::string translateActiveException() {
        return getRegistryHub().getExceptionTranslatorRegistry().translateActiveException();
    }


} // end namespace Catch



#include <cassert>
#include <exception>
#include <iomanip>
#include <set>

namespace Catch {

    namespace {
        IEventListenerPtr createReporter(std::string const& reporterName, ReporterConfig&& config) {
            auto reporter = Catch::getRegistryHub().getReporterRegistry().create(reporterName, CATCH_MOVE(config));
            CATCH_ENFORCE(reporter, "No reporter registered with name: '" << reporterName << '\'');

            return reporter;
        }

        IEventListenerPtr prepareReporters(Config const* config) {
            if (Catch::getRegistryHub().getReporterRegistry().getListeners().empty()
                    && config->getProcessedReporterSpecs().size() == 1) {
                auto const& spec = config->getProcessedReporterSpecs()[0];
                return createReporter(
                    spec.name,
                    ReporterConfig( config,
                                    makeStream( spec.outputFilename ),
                                    spec.colourMode,
                                    spec.customOptions ) );
            }

            auto multi = Detail::make_unique<MultiReporter>(config);

            auto const& listeners = Catch::getRegistryHub().getReporterRegistry().getListeners();
            for (auto const& listener : listeners) {
                multi->addListener(listener->create(config));
            }

            for ( auto const& reporterSpec : config->getProcessedReporterSpecs() ) {
                multi->addReporter( createReporter(
                    reporterSpec.name,
                    ReporterConfig( config,
                                    makeStream( reporterSpec.outputFilename ),
                                    reporterSpec.colourMode,
                                    reporterSpec.customOptions ) ) );
            }

            return multi;
        }

        class TestGroup {
        public:
            explicit TestGroup(IEventListenerPtr&& reporter, Config const* config):
                m_reporter(reporter.get()),
                m_config{config},
                m_context{config, CATCH_MOVE(reporter)} {

                assert( m_config->testSpec().getInvalidSpecs().empty() &&
                        "Invalid test specs should be handled before running tests" );

                auto const& allTestCases = getAllTestCasesSorted(*m_config);
                auto const& testSpec = m_config->testSpec();
                if ( !testSpec.hasFilters() ) {
                    for ( auto const& test : allTestCases ) {
                        if ( !test.getTestCaseInfo().isHidden() ) {
                            m_tests.emplace( &test );
                        }
                    }
                } else {
                    m_matches =
                        testSpec.matchesByFilter( allTestCases, *m_config );
                    for ( auto const& match : m_matches ) {
                        m_tests.insert( match.tests.begin(),
                                        match.tests.end() );
                    }
                }

                m_tests = createShard(m_tests, m_config->shardCount(), m_config->shardIndex());
            }

            Totals execute() {
                Totals totals;
                for (auto const& testCase : m_tests) {
                    if (!m_context.aborting())
                        totals += m_context.runTest(*testCase);
                    else
                        m_reporter->skipTest(testCase->getTestCaseInfo());
                }

                for (auto const& match : m_matches) {
                    if (match.tests.empty()) {
                        m_unmatchedTestSpecs = true;
                        m_reporter->noMatchingTestCases( match.name );
                    }
                }

                return totals;
            }

            bool hadUnmatchedTestSpecs() const {
                return m_unmatchedTestSpecs;
            }


        private:
            IEventListener* m_reporter;
            Config const* m_config;
            RunContext m_context;
            std::set<TestCaseHandle const*> m_tests;
            TestSpec::Matches m_matches;
            bool m_unmatchedTestSpecs = false;
        };

        void applyFilenamesAsTags() {
            for (auto const& testInfo : getRegistryHub().getTestCaseRegistry().getAllInfos()) {
                testInfo->addFilenameTag();
            }
        }

    } // anon namespace

    Session::Session() {
        static bool alreadyInstantiated = false;
        if( alreadyInstantiated ) {
            CATCH_TRY { CATCH_INTERNAL_ERROR( "Only one instance of Catch::Session can ever be used" ); }
            CATCH_CATCH_ALL { getMutableRegistryHub().registerStartupException(); }
        }

        // There cannot be exceptions at startup in no-exception mode.
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
        const auto& exceptions = getRegistryHub().getStartupExceptionRegistry().getExceptions();
        if ( !exceptions.empty() ) {
            config();
            getCurrentMutableContext().setConfig(m_config.get());

            m_startupExceptions = true;
            auto errStream = makeStream( "%stderr" );
            auto colourImpl = makeColourImpl(
                ColourMode::PlatformDefault, errStream.get() );
            auto guard = colourImpl->guardColour( Colour::Red );
            errStream->stream() << "Errors occurred during startup!" << '\n';
            // iterate over all exceptions and notify user
            for ( const auto& ex_ptr : exceptions ) {
                try {
                    std::rethrow_exception(ex_ptr);
                } catch ( std::exception const& ex ) {
                    errStream->stream() << TextFlow::Column( ex.what() ).indent(2) << '\n';
                }
            }
        }
#endif

        alreadyInstantiated = true;
        m_cli = makeCommandLineParser( m_configData );
    }
    Session::~Session() {
        Catch::cleanUp();
    }

    void Session::showHelp() const {
        Catch::cout()
                << "\nCatch2 v" << libraryVersion() << '\n'
                << m_cli << '\n'
                << "For more detailed usage please see the project docs\n\n" << std::flush;
    }
    void Session::libIdentify() {
        Catch::cout()
                << std::left << std::setw(16) << "description: " << "A Catch2 test executable\n"
                << std::left << std::setw(16) << "category: " << "testframework\n"
                << std::left << std::setw(16) << "framework: " << "Catch2\n"
                << std::left << std::setw(16) << "version: " << libraryVersion() << '\n' << std::flush;
    }

    int Session::applyCommandLine( int argc, char const * const * argv ) {
        if ( m_startupExceptions ) { return UnspecifiedErrorExitCode; }

        auto result = m_cli.parse( Clara::Args( argc, argv ) );

        if( !result ) {
            config();
            getCurrentMutableContext().setConfig(m_config.get());
            auto errStream = makeStream( "%stderr" );
            auto colour = makeColourImpl( ColourMode::PlatformDefault, errStream.get() );

            errStream->stream()
                << colour->guardColour( Colour::Red )
                << "\nError(s) in input:\n"
                << TextFlow::Column( result.errorMessage() ).indent( 2 )
                << "\n\n";
            errStream->stream() << "Run with -? for usage\n\n" << std::flush;
            return UnspecifiedErrorExitCode;
        }

        if( m_configData.showHelp )
            showHelp();
        if( m_configData.libIdentify )
            libIdentify();

        m_config.reset();
        return 0;
    }

#if defined(CATCH_CONFIG_WCHAR) && defined(_WIN32) && defined(UNICODE)
    int Session::applyCommandLine( int argc, wchar_t const * const * argv ) {

        char **utf8Argv = new char *[ argc ];

        for ( int i = 0; i < argc; ++i ) {
            int bufSize = WideCharToMultiByte( CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr );

            utf8Argv[ i ] = new char[ bufSize ];

            WideCharToMultiByte( CP_UTF8, 0, argv[i], -1, utf8Argv[i], bufSize, nullptr, nullptr );
        }

        int returnCode = applyCommandLine( argc, utf8Argv );

        for ( int i = 0; i < argc; ++i )
            delete [] utf8Argv[ i ];

        delete [] utf8Argv;

        return returnCode;
    }
#endif

    void Session::useConfigData( ConfigData const& configData ) {
        m_configData = configData;
        m_config.reset();
    }

    int Session::run() {
        if( ( m_configData.waitForKeypress & WaitForKeypress::BeforeStart ) != 0 ) {
            Catch::cout() << "...waiting for enter/ return before starting\n" << std::flush;
            static_cast<void>(std::getchar());
        }
        int exitCode = runInternal();
        if( ( m_configData.waitForKeypress & WaitForKeypress::BeforeExit ) != 0 ) {
            Catch::cout() << "...waiting for enter/ return before exiting, with code: " << exitCode << '\n' << std::flush;
            static_cast<void>(std::getchar());
        }
        return exitCode;
    }

    Clara::Parser const& Session::cli() const {
        return m_cli;
    }
    void Session::cli( Clara::Parser const& newParser ) {
        m_cli = newParser;
    }
    ConfigData& Session::configData() {
        return m_configData;
    }
    Config& Session::config() {
        if( !m_config )
            m_config = Detail::make_unique<Config>( m_configData );
        return *m_config;
    }

    int Session::runInternal() {
        if ( m_startupExceptions ) { return UnspecifiedErrorExitCode; }

        if (m_configData.showHelp || m_configData.libIdentify) {
            return 0;
        }

        if ( m_configData.shardIndex >= m_configData.shardCount ) {
            Catch::cerr() << "The shard count (" << m_configData.shardCount
                          << ") must be greater than the shard index ("
                          << m_configData.shardIndex << ")\n"
                          << std::flush;
            return UnspecifiedErrorExitCode;
        }

        CATCH_TRY {
            config(); // Force config to be constructed

            seedRng( *m_config );

            if (m_configData.filenamesAsTags) {
                applyFilenamesAsTags();
            }

            // Set up global config instance before we start calling into other functions
            getCurrentMutableContext().setConfig(m_config.get());

            // Create reporter(s) so we can route listings through them
            auto reporter = prepareReporters(m_config.get());

            auto const& invalidSpecs = m_config->testSpec().getInvalidSpecs();
            if ( !invalidSpecs.empty() ) {
                for ( auto const& spec : invalidSpecs ) {
                    reporter->reportInvalidTestSpec( spec );
                }
                return InvalidTestSpecExitCode;
            }


            // Handle list request
            if (list(*reporter, *m_config)) {
                return 0;
            }

            TestGroup tests { CATCH_MOVE(reporter), m_config.get() };
            auto const totals = tests.execute();

            if ( tests.hadUnmatchedTestSpecs()
                && m_config->warnAboutUnmatchedTestSpecs() ) {
                // UnmatchedTestSpecExitCode
                return UnmatchedTestSpecExitCode;
            }

            if ( totals.testCases.total() == 0
                && !m_config->zeroTestsCountAsSuccess() ) {
                return NoTestsRunExitCode;
            }

            if ( totals.testCases.total() > 0 &&
                 totals.testCases.total() == totals.testCases.skipped
                && !m_config->zeroTestsCountAsSuccess() ) {
                return AllTestsSkippedExitCode;
            }

            if ( totals.assertions.failed ) { return TestFailureExitCode; }
            return 0;

        }
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
        catch( std::exception& ex ) {
            Catch::cerr() << ex.what() << '\n' << std::flush;
            return UnspecifiedErrorExitCode;
        }
#endif
    }

} // end namespace Catch




namespace Catch {

    RegistrarForTagAliases::RegistrarForTagAliases(char const* alias, char const* tag, SourceLineInfo const& lineInfo) {
        CATCH_TRY {
            getMutableRegistryHub().registerTagAlias(alias, tag, lineInfo);
        } CATCH_CATCH_ALL {
            // Do not throw when constructing global objects, instead register the exception to be processed later
            getMutableRegistryHub().registerStartupException();
        }
    }

}



#include <cassert>
#include <cctype>
#include <algorithm>

namespace Catch {

    namespace {
        using TCP_underlying_type = uint8_t;
        static_assert(sizeof(TestCaseProperties) == sizeof(TCP_underlying_type),
                      "The size of the TestCaseProperties is different from the assumed size");

        constexpr TestCaseProperties operator|(TestCaseProperties lhs, TestCaseProperties rhs) {
            return static_cast<TestCaseProperties>(
                static_cast<TCP_underlying_type>(lhs) | static_cast<TCP_underlying_type>(rhs)
            );
        }

        constexpr TestCaseProperties& operator|=(TestCaseProperties& lhs, TestCaseProperties rhs) {
            lhs = static_cast<TestCaseProperties>(
                static_cast<TCP_underlying_type>(lhs) | static_cast<TCP_underlying_type>(rhs)
            );
            return lhs;
        }

        constexpr TestCaseProperties operator&(TestCaseProperties lhs, TestCaseProperties rhs) {
            return static_cast<TestCaseProperties>(
                static_cast<TCP_underlying_type>(lhs) & static_cast<TCP_underlying_type>(rhs)
            );
        }

        constexpr bool applies(TestCaseProperties tcp) {
            static_assert(static_cast<TCP_underlying_type>(TestCaseProperties::None) == 0,
                          "TestCaseProperties::None must be equal to 0");
            return tcp != TestCaseProperties::None;
        }

        TestCaseProperties parseSpecialTag( StringRef tag ) {
            if( !tag.empty() && tag[0] == '.' )
                return TestCaseProperties::IsHidden;
            else if( tag == "!throws"_sr )
                return TestCaseProperties::Throws;
            else if( tag == "!shouldfail"_sr )
                return TestCaseProperties::ShouldFail;
            else if( tag == "!mayfail"_sr )
                return TestCaseProperties::MayFail;
            else if( tag == "!nonportable"_sr )
                return TestCaseProperties::NonPortable;
            else if( tag == "!benchmark"_sr )
                return TestCaseProperties::Benchmark | TestCaseProperties::IsHidden;
            else
                return TestCaseProperties::None;
        }
        bool isReservedTag( StringRef tag ) {
            return parseSpecialTag( tag ) == TestCaseProperties::None
                && tag.size() > 0
                && !std::isalnum( static_cast<unsigned char>(tag[0]) );
        }
        void enforceNotReservedTag( StringRef tag, SourceLineInfo const& _lineInfo ) {
            CATCH_ENFORCE( !isReservedTag(tag),
                          "Tag name: [" << tag << "] is not allowed.\n"
                          << "Tag names starting with non alphanumeric characters are reserved\n"
                          << _lineInfo );
        }

        std::string makeDefaultName() {
            static size_t counter = 0;
            return "Anonymous test case " + std::to_string(++counter);
        }

        constexpr StringRef extractFilenamePart(StringRef filename) {
            size_t lastDot = filename.size();
            while (lastDot > 0 && filename[lastDot - 1] != '.') {
                --lastDot;
            }
            // In theory we could have filename without any extension in it
            if ( lastDot == 0 ) { return StringRef(); }

            --lastDot;
            size_t nameStart = lastDot;
            while (nameStart > 0 && filename[nameStart - 1] != '/' && filename[nameStart - 1] != '\\') {
                --nameStart;
            }

            return filename.substr(nameStart, lastDot - nameStart);
        }

        // Returns the upper bound on size of extra tags ([#file]+[.])
        constexpr size_t sizeOfExtraTags(StringRef filepath) {
            // [.] is 3, [#] is another 3
            const size_t extras = 3 + 3;
            return extractFilenamePart(filepath).size() + extras;
        }
    } // end unnamed namespace

    bool operator<(  Tag const& lhs, Tag const& rhs ) {
        Detail::CaseInsensitiveLess cmp;
        return cmp( lhs.original, rhs.original );
    }
    bool operator==( Tag const& lhs, Tag const& rhs ) {
        Detail::CaseInsensitiveEqualTo cmp;
        return cmp( lhs.original, rhs.original );
    }

    Detail::unique_ptr<TestCaseInfo>
        makeTestCaseInfo(StringRef _className,
                         NameAndTags const& nameAndTags,
                         SourceLineInfo const& _lineInfo ) {
        return Detail::make_unique<TestCaseInfo>(_className, nameAndTags, _lineInfo);
    }

    TestCaseInfo::TestCaseInfo(StringRef _className,
                               NameAndTags const& _nameAndTags,
                               SourceLineInfo const& _lineInfo):
        name( _nameAndTags.name.empty() ? makeDefaultName() : _nameAndTags.name ),
        className( _className ),
        lineInfo( _lineInfo )
    {
        StringRef originalTags = _nameAndTags.tags;
        // We need to reserve enough space to store all of the tags
        // (including optional hidden tag and filename tag)
        auto requiredSize = originalTags.size() + sizeOfExtraTags(_lineInfo.file);
        backingTags.reserve(requiredSize);

        // We cannot copy the tags directly, as we need to normalize
        // some tags, so that [.foo] is copied as [.][foo].
        size_t tagStart = 0;
        size_t tagEnd = 0;
        bool inTag = false;
        for (size_t idx = 0; idx < originalTags.size(); ++idx) {
            auto c = originalTags[idx];
            if (c == '[') {
                CATCH_ENFORCE(
                    !inTag,
                    "Found '[' inside a tag while registering test case '"
                        << _nameAndTags.name << "' at " << _lineInfo );

                inTag = true;
                tagStart = idx;
            }
            if (c == ']') {
                CATCH_ENFORCE(
                    inTag,
                    "Found unmatched ']' while registering test case '"
                        << _nameAndTags.name << "' at " << _lineInfo );

                inTag = false;
                tagEnd = idx;
                assert(tagStart < tagEnd);

                // We need to check the tag for special meanings, copy
                // it over to backing storage and actually reference the
                // backing storage in the saved tags
                StringRef tagStr = originalTags.substr(tagStart+1, tagEnd - tagStart - 1);
                CATCH_ENFORCE( !tagStr.empty(),
                               "Found an empty tag while registering test case '"
                                   << _nameAndTags.name << "' at "
                                   << _lineInfo );

                enforceNotReservedTag(tagStr, lineInfo);
                properties |= parseSpecialTag(tagStr);
                // When copying a tag to the backing storage, we need to
                // check if it is a merged hide tag, such as [.foo], and
                // if it is, we need to handle it as if it was [foo].
                if (tagStr.size() > 1 && tagStr[0] == '.') {
                    tagStr = tagStr.substr(1, tagStr.size() - 1);
                }
                // We skip over dealing with the [.] tag, as we will add
                // it later unconditionally and then sort and unique all
                // the tags.
                internalAppendTag(tagStr);
            }
        }
        CATCH_ENFORCE( !inTag,
                       "Found an unclosed tag while registering test case '"
                           << _nameAndTags.name << "' at " << _lineInfo );


        // Add [.] if relevant
        if (isHidden()) {
            internalAppendTag("."_sr);
        }

        // Sort and prepare tags
        std::sort(begin(tags), end(tags));
        tags.erase(std::unique(begin(tags), end(tags)),
                   end(tags));
    }

    bool TestCaseInfo::isHidden() const {
        return applies( properties & TestCaseProperties::IsHidden );
    }
    bool TestCaseInfo::throws() const {
        return applies( properties & TestCaseProperties::Throws );
    }
    bool TestCaseInfo::okToFail() const {
        return applies( properties & (TestCaseProperties::ShouldFail | TestCaseProperties::MayFail ) );
    }
    bool TestCaseInfo::expectedToFail() const {
        return applies( properties & (TestCaseProperties::ShouldFail) );
    }

    void TestCaseInfo::addFilenameTag() {
        std::string combined("#");
        combined += extractFilenamePart(lineInfo.file);
        internalAppendTag(combined);
    }

    std::string TestCaseInfo::tagsAsString() const {
        std::string ret;
        // '[' and ']' per tag
        std::size_t full_size = 2 * tags.size();
        for (const auto& tag : tags) {
            full_size += tag.original.size();
        }
        ret.reserve(full_size);
        for (const auto& tag : tags) {
            ret.push_back('[');
            ret += tag.original;
            ret.push_back(']');
        }

        return ret;
    }

    void TestCaseInfo::internalAppendTag(StringRef tagStr) {
        backingTags += '[';
        const auto backingStart = backingTags.size();
        backingTags += tagStr;
        const auto backingEnd = backingTags.size();
        backingTags += ']';
        tags.emplace_back(StringRef(backingTags.c_str() + backingStart, backingEnd - backingStart));
    }

    bool operator<( TestCaseInfo const& lhs, TestCaseInfo const& rhs ) {
        // We want to avoid redoing the string comparisons multiple times,
        // so we store the result of a three-way comparison before using
        // it in the actual comparison logic.
        const auto cmpName = lhs.name.compare( rhs.name );
        if ( cmpName != 0 ) {
            return cmpName < 0;
        }
        const auto cmpClassName = lhs.className.compare( rhs.className );
        if ( cmpClassName != 0 ) {
            return cmpClassName < 0;
        }
        return lhs.tags < rhs.tags;
    }

} // end namespace Catch



#include <algorithm>
#include <string>
#include <vector>
#include <ostream>

namespace Catch {

    TestSpec::Pattern::Pattern( std::string const& name )
    : m_name( name )
    {}

    TestSpec::Pattern::~Pattern() = default;

    std::string const& TestSpec::Pattern::name() const {
        return m_name;
    }


    TestSpec::NamePattern::NamePattern( std::string const& name, std::string const& filterString )
    : Pattern( filterString )
    , m_wildcardPattern( toLower( name ), CaseSensitive::No )
    {}

    bool TestSpec::NamePattern::matches( TestCaseInfo const& testCase ) const {
        return m_wildcardPattern.matches( testCase.name );
    }

    void TestSpec::NamePattern::serializeTo( std::ostream& out ) const {
        out << '"' << name() << '"';
    }


    TestSpec::TagPattern::TagPattern( std::string const& tag, std::string const& filterString )
    : Pattern( filterString )
    , m_tag( tag )
    {}

    bool TestSpec::TagPattern::matches( TestCaseInfo const& testCase ) const {
        return std::find( begin( testCase.tags ),
                          end( testCase.tags ),
                          Tag( m_tag ) ) != end( testCase.tags );
    }

    void TestSpec::TagPattern::serializeTo( std::ostream& out ) const {
        out << name();
    }

    bool TestSpec::Filter::matches( TestCaseInfo const& testCase ) const {
        bool should_use = !testCase.isHidden();
        for (auto const& pattern : m_required) {
            should_use = true;
            if (!pattern->matches(testCase)) {
                return false;
            }
        }
        for (auto const& pattern : m_forbidden) {
            if (pattern->matches(testCase)) {
                return false;
            }
        }
        return should_use;
    }

    void TestSpec::Filter::serializeTo( std::ostream& out ) const {
        bool first = true;
        for ( auto const& pattern : m_required ) {
            if ( !first ) {
                out << ' ';
            }
            out << *pattern;
            first = false;
        }
        for ( auto const& pattern : m_forbidden ) {
            if ( !first ) {
                out << ' ';
            }
            out << *pattern;
            first = false;
        }
    }


    std::string TestSpec::extractFilterName( Filter const& filter ) {
        Catch::ReusableStringStream sstr;
        sstr << filter;
        return sstr.str();
    }

    bool TestSpec::hasFilters() const {
        return !m_filters.empty();
    }

    bool TestSpec::matches( TestCaseInfo const& testCase ) const {
        return std::any_of( m_filters.begin(), m_filters.end(), [&]( Filter const& f ){ return f.matches( testCase ); } );
    }

    TestSpec::Matches TestSpec::matchesByFilter( std::vector<TestCaseHandle> const& testCases, IConfig const& config ) const {
        Matches matches;
        matches.reserve( m_filters.size() );
        for ( auto const& filter : m_filters ) {
            std::vector<TestCaseHandle const*> currentMatches;
            for ( auto const& test : testCases )
                if ( isThrowSafe( test, config ) &&
                     filter.matches( test.getTestCaseInfo() ) )
                    currentMatches.emplace_back( &test );
            matches.push_back(
                FilterMatch{ extractFilterName( filter ), currentMatches } );
        }
        return matches;
    }

    const TestSpec::vectorStrings& TestSpec::getInvalidSpecs() const {
        return m_invalidSpecs;
    }

    void TestSpec::serializeTo( std::ostream& out ) const {
        bool first = true;
        for ( auto const& filter : m_filters ) {
            if ( !first ) {
                out << ',';
            }
            out << filter;
            first = false;
        }
    }

}



#include <chrono>

namespace Catch {

    namespace {
        static auto getCurrentNanosecondsSinceEpoch() -> uint64_t {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        }
    } // end unnamed namespace

    void Timer::start() {
       m_nanoseconds = getCurrentNanosecondsSinceEpoch();
    }
    auto Timer::getElapsedNanoseconds() const -> uint64_t {
        return getCurrentNanosecondsSinceEpoch() - m_nanoseconds;
    }
    auto Timer::getElapsedMicroseconds() const -> uint64_t {
        return getElapsedNanoseconds()/1000;
    }
    auto Timer::getElapsedMilliseconds() const -> unsigned int {
        return static_cast<unsigned int>(getElapsedMicroseconds()/1000);
    }
    auto Timer::getElapsedSeconds() const -> double {
        return static_cast<double>(getElapsedMicroseconds())/1000000.0;
    }


} // namespace Catch




#include <iomanip>

namespace Catch {

namespace Detail {

    namespace {
        const int hexThreshold = 255;

        struct Endianness {
            enum Arch : uint8_t {
                Big,
                Little
            };

            static Arch which() {
                int one = 1;
                // If the lowest byte we read is non-zero, we can assume
                // that little endian format is used.
                auto value = *reinterpret_cast<char*>(&one);
                return value ? Little : Big;
            }
        };

        template<typename T>
        std::string fpToString(T value, int precision) {
            if (Catch::isnan(value)) {
                return "nan";
            }

            ReusableStringStream rss;
            rss << std::setprecision(precision)
                << std::fixed
                << value;
            std::string d = rss.str();
            std::size_t i = d.find_last_not_of('0');
            if (i != std::string::npos && i != d.size() - 1) {
                if (d[i] == '.')
                    i++;
                d = d.substr(0, i + 1);
            }
            return d;
        }
    } // end unnamed namespace

    std::string convertIntoString(StringRef string, bool escapeInvisibles) {
        std::string ret;
        // This is enough for the "don't escape invisibles" case, and a good
        // lower bound on the "escape invisibles" case.
        ret.reserve(string.size() + 2);

        if (!escapeInvisibles) {
            ret += '"';
            ret += string;
            ret += '"';
            return ret;
        }

        ret += '"';
        for (char c : string) {
            switch (c) {
            case '\r':
                ret.append("\\r");
                break;
            case '\n':
                ret.append("\\n");
                break;
            case '\t':
                ret.append("\\t");
                break;
            case '\f':
                ret.append("\\f");
                break;
            default:
                ret.push_back(c);
                break;
            }
        }
        ret += '"';

        return ret;
    }

    std::string convertIntoString(StringRef string) {
        return convertIntoString(string, getCurrentContext().getConfig()->showInvisibles());
    }

    std::string rawMemoryToString( const void *object, std::size_t size ) {
        // Reverse order for little endian architectures
        int i = 0, end = static_cast<int>( size ), inc = 1;
        if( Endianness::which() == Endianness::Little ) {
            i = end-1;
            end = inc = -1;
        }

        unsigned char const *bytes = static_cast<unsigned char const *>(object);
        ReusableStringStream rss;
        rss << "0x" << std::setfill('0') << std::hex;
        for( ; i != end; i += inc )
             rss << std::setw(2) << static_cast<unsigned>(bytes[i]);
       return rss.str();
    }

    std::string makeExceptionHappenedString() {
        return "{ stringification failed with an exception: \"" +
               translateActiveException() + "\" }";

    }

} // end Detail namespace



//// ======================================================= ////
//
//   Out-of-line defs for full specialization of StringMaker
//
//// ======================================================= ////

std::string StringMaker<std::string>::convert(const std::string& str) {
    return Detail::convertIntoString( str );
}

#ifdef CATCH_CONFIG_CPP17_STRING_VIEW
std::string StringMaker<std::string_view>::convert(std::string_view str) {
    return Detail::convertIntoString( StringRef( str.data(), str.size() ) );
}
#endif

std::string StringMaker<char const*>::convert(char const* str) {
    if (str) {
        return Detail::convertIntoString( str );
    } else {
        return{ "{null string}" };
    }
}
std::string StringMaker<char*>::convert(char* str) { // NOLINT(readability-non-const-parameter)
    if (str) {
        return Detail::convertIntoString( str );
    } else {
        return{ "{null string}" };
    }
}

#ifdef CATCH_CONFIG_WCHAR
std::string StringMaker<std::wstring>::convert(const std::wstring& wstr) {
    std::string s;
    s.reserve(wstr.size());
    for (auto c : wstr) {
        s += (c <= 0xff) ? static_cast<char>(c) : '?';
    }
    return ::Catch::Detail::stringify(s);
}

# ifdef CATCH_CONFIG_CPP17_STRING_VIEW
std::string StringMaker<std::wstring_view>::convert(std::wstring_view str) {
    return StringMaker<std::wstring>::convert(std::wstring(str));
}
# endif

std::string StringMaker<wchar_t const*>::convert(wchar_t const * str) {
    if (str) {
        return ::Catch::Detail::stringify(std::wstring{ str });
    } else {
        return{ "{null string}" };
    }
}
std::string StringMaker<wchar_t *>::convert(wchar_t * str) {
    if (str) {
        return ::Catch::Detail::stringify(std::wstring{ str });
    } else {
        return{ "{null string}" };
    }
}
#endif

#if defined(CATCH_CONFIG_CPP17_BYTE)
#include <cstddef>
std::string StringMaker<std::byte>::convert(std::byte value) {
    return ::Catch::Detail::stringify(std::to_integer<unsigned long long>(value));
}
#endif // defined(CATCH_CONFIG_CPP17_BYTE)

std::string StringMaker<int>::convert(int value) {
    return ::Catch::Detail::stringify(static_cast<long long>(value));
}
std::string StringMaker<long>::convert(long value) {
    return ::Catch::Detail::stringify(static_cast<long long>(value));
}
std::string StringMaker<long long>::convert(long long value) {
    ReusableStringStream rss;
    rss << value;
    if (value > Detail::hexThreshold) {
        rss << " (0x" << std::hex << value << ')';
    }
    return rss.str();
}

std::string StringMaker<unsigned int>::convert(unsigned int value) {
    return ::Catch::Detail::stringify(static_cast<unsigned long long>(value));
}
std::string StringMaker<unsigned long>::convert(unsigned long value) {
    return ::Catch::Detail::stringify(static_cast<unsigned long long>(value));
}
std::string StringMaker<unsigned long long>::convert(unsigned long long value) {
    ReusableStringStream rss;
    rss << value;
    if (value > Detail::hexThreshold) {
        rss << " (0x" << std::hex << value << ')';
    }
    return rss.str();
}

std::string StringMaker<signed char>::convert(signed char value) {
    if (value == '\r') {
        return "'\\r'";
    } else if (value == '\f') {
        return "'\\f'";
    } else if (value == '\n') {
        return "'\\n'";
    } else if (value == '\t') {
        return "'\\t'";
    } else if ('\0' <= value && value < ' ') {
        return ::Catch::Detail::stringify(static_cast<unsigned int>(value));
    } else {
        char chstr[] = "' '";
        chstr[1] = value;
        return chstr;
    }
}
std::string StringMaker<char>::convert(char c) {
    return ::Catch::Detail::stringify(static_cast<signed char>(c));
}
std::string StringMaker<unsigned char>::convert(unsigned char value) {
    return ::Catch::Detail::stringify(static_cast<char>(value));
}

int StringMaker<float>::precision = std::numeric_limits<float>::max_digits10;

std::string StringMaker<float>::convert(float value) {
    return Detail::fpToString(value, precision) + 'f';
}

int StringMaker<double>::precision = std::numeric_limits<double>::max_digits10;

std::string StringMaker<double>::convert(double value) {
    return Detail::fpToString(value, precision);
}

} // end namespace Catch



namespace Catch {

    Counts Counts::operator - ( Counts const& other ) const {
        Counts diff;
        diff.passed = passed - other.passed;
        diff.failed = failed - other.failed;
        diff.failedButOk = failedButOk - other.failedButOk;
        diff.skipped = skipped - other.skipped;
        return diff;
    }

    Counts& Counts::operator += ( Counts const& other ) {
        passed += other.passed;
        failed += other.failed;
        failedButOk += other.failedButOk;
        skipped += other.skipped;
        return *this;
    }

    std::uint64_t Counts::total() const {
        return passed + failed + failedButOk + skipped;
    }
    bool Counts::allPassed() const {
        return failed == 0 && failedButOk == 0 && skipped == 0;
    }
    bool Counts::allOk() const {
        return failed == 0;
    }

    Totals Totals::operator - ( Totals const& other ) const {
        Totals diff;
        diff.assertions = assertions - other.assertions;
        diff.testCases = testCases - other.testCases;
        return diff;
    }

    Totals& Totals::operator += ( Totals const& other ) {
        assertions += other.assertions;
        testCases += other.testCases;
        return *this;
    }

    Totals Totals::delta( Totals const& prevTotals ) const {
        Totals diff = *this - prevTotals;
        if( diff.assertions.failed > 0 )
            ++diff.testCases.failed;
        else if( diff.assertions.failedButOk > 0 )
            ++diff.testCases.failedButOk;
        else if ( diff.assertions.skipped > 0 )
            ++ diff.testCases.skipped;
        else
            ++diff.testCases.passed;
        return diff;
    }

}




namespace Catch {
    namespace Detail {
        void registerTranslatorImpl(
            Detail::unique_ptr<IExceptionTranslator>&& translator ) {
            getMutableRegistryHub().registerTranslator(
                CATCH_MOVE( translator ) );
        }
    } // namespace Detail
} // namespace Catch


#include <ostream>

namespace Catch {

    Version::Version
        (   unsigned int _majorVersion,
            unsigned int _minorVersion,
            unsigned int _patchNumber,
            char const * const _branchName,
            unsigned int _buildNumber )
    :   majorVersion( _majorVersion ),
        minorVersion( _minorVersion ),
        patchNumber( _patchNumber ),
        branchName( _branchName ),
        buildNumber( _buildNumber )
    {}

    std::ostream& operator << ( std::ostream& os, Version const& version ) {
        os  << version.majorVersion << '.'
            << version.minorVersion << '.'
            << version.patchNumber;
        // branchName is never null -> 0th char is \0 if it is empty
        if (version.branchName[0]) {
            os << '-' << version.branchName
               << '.' << version.buildNumber;
        }
        return os;
    }

    Version const& libraryVersion() {
        static Version version( 3, 9, 1, "", 0 );
        return version;
    }

}




namespace Catch {

    const char* GeneratorException::what() const noexcept {
        return m_msg;
    }

} // end namespace Catch




namespace Catch {

    IGeneratorTracker::~IGeneratorTracker() = default;

namespace Generators {

namespace Detail {

    [[noreturn]]
    void throw_generator_exception(char const* msg) {
        Catch::throw_exception(GeneratorException{ msg });
    }
} // end namespace Detail

    GeneratorUntypedBase::~GeneratorUntypedBase() = default;

    IGeneratorTracker* acquireGeneratorTracker(StringRef generatorName, SourceLineInfo const& lineInfo ) {
        return getResultCapture().acquireGeneratorTracker( generatorName, lineInfo );
    }

    IGeneratorTracker* createGeneratorTracker( StringRef generatorName,
                                 SourceLineInfo lineInfo,
                                 GeneratorBasePtr&& generator ) {
        return getResultCapture().createGeneratorTracker(
            generatorName, lineInfo, CATCH_MOVE( generator ) );
    }

} // namespace Generators
} // namespace Catch




#include <random>

namespace Catch {
    namespace Generators {
        namespace Detail {
            std::uint32_t getSeed() { return sharedRng()(); }
        } // namespace Detail

        struct RandomFloatingGenerator<long double>::PImpl {
            PImpl( long double a, long double b, uint32_t seed ):
                rng( seed ), dist( a, b ) {}

            Catch::SimplePcg32 rng;
            std::uniform_real_distribution<long double> dist;
        };

        RandomFloatingGenerator<long double>::RandomFloatingGenerator(
            long double a, long double b, std::uint32_t seed) :
            m_pimpl(Catch::Detail::make_unique<PImpl>(a, b, seed)) {
            static_cast<void>( next() );
        }

        RandomFloatingGenerator<long double>::~RandomFloatingGenerator() =
            default;
        bool RandomFloatingGenerator<long double>::next() {
            m_current_number = m_pimpl->dist( m_pimpl->rng );
            return true;
        }
    } // namespace Generators
} // namespace Catch




namespace Catch {
    IResultCapture::~IResultCapture() = default;
}




namespace Catch {
    IConfig::~IConfig() = default;
}




namespace Catch {
    IExceptionTranslator::~IExceptionTranslator() = default;
    IExceptionTranslatorRegistry::~IExceptionTranslatorRegistry() = default;
}



#include <string>

namespace Catch {
    namespace Generators {

        bool GeneratorUntypedBase::countedNext() {
            auto ret = next();
            if ( ret ) {
                m_stringReprCache.clear();
                ++m_currentElementIndex;
            }
            return ret;
        }

        StringRef GeneratorUntypedBase::currentElementAsString() const {
            if ( m_stringReprCache.empty() ) {
                m_stringReprCache = stringifyImpl();
            }
            return m_stringReprCache;
        }

    } // namespace Generators
} // namespace Catch




namespace Catch {
    IRegistryHub::~IRegistryHub() = default;
    IMutableRegistryHub::~IMutableRegistryHub() = default;
}



#include <cassert>

namespace Catch {

    ReporterConfig::ReporterConfig(
        IConfig const* _fullConfig,
        Detail::unique_ptr<IStream> _stream,
        ColourMode colourMode,
        std::map<std::string, std::string> customOptions ):
        m_stream( CATCH_MOVE(_stream) ),
        m_fullConfig( _fullConfig ),
        m_colourMode( colourMode ),
        m_customOptions( CATCH_MOVE( customOptions ) ) {}

    Detail::unique_ptr<IStream> ReporterConfig::takeStream() && {
        assert( m_stream );
        return CATCH_MOVE( m_stream );
    }
    IConfig const * ReporterConfig::fullConfig() const { return m_fullConfig; }
    ColourMode ReporterConfig::colourMode() const { return m_colourMode; }

    std::map<std::string, std::string> const&
    ReporterConfig::customOptions() const {
        return m_customOptions;
    }

    ReporterConfig::~ReporterConfig() = default;

    AssertionStats::AssertionStats( AssertionResult const& _assertionResult,
                                    std::vector<MessageInfo> const& _infoMessages,
                                    Totals const& _totals )
    :   assertionResult( _assertionResult ),
        infoMessages( _infoMessages ),
        totals( _totals )
    {
        if( assertionResult.hasMessage() ) {
            // Copy message into messages list.
            // !TBD This should have been done earlier, somewhere
            MessageBuilder builder( assertionResult.getTestMacroName(), assertionResult.getSourceInfo(), assertionResult.getResultType() );
            builder.m_info.message = static_cast<std::string>(assertionResult.getMessage());

            infoMessages.push_back( CATCH_MOVE(builder.m_info) );
        }
    }

    SectionStats::SectionStats(  SectionInfo&& _sectionInfo,
                                 Counts const& _assertions,
                                 double _durationInSeconds,
                                 bool _missingAssertions )
    :   sectionInfo( CATCH_MOVE(_sectionInfo) ),
        assertions( _assertions ),
        durationInSeconds( _durationInSeconds ),
        missingAssertions( _missingAssertions )
    {}


    TestCaseStats::TestCaseStats(  TestCaseInfo const& _testInfo,
                                   Totals const& _totals,
                                   std::string&& _stdOut,
                                   std::string&& _stdErr,
                                   bool _aborting )
    : testInfo( &_testInfo ),
        totals( _totals ),
        stdOut( CATCH_MOVE(_stdOut) ),
        stdErr( CATCH_MOVE(_stdErr) ),
        aborting( _aborting )
    {}


    TestRunStats::TestRunStats(   TestRunInfo const& _runInfo,
                    Totals const& _totals,
                    bool _aborting )
    :   runInfo( _runInfo ),
        totals( _totals ),
        aborting( _aborting )
    {}

    IEventListener::~IEventListener() = default;

} // end namespace Catch




namespace Catch {
    IReporterFactory::~IReporterFactory() = default;
    EventListenerFactory::~EventListenerFactory() = default;
}




namespace Catch {
    ITestCaseRegistry::~ITestCaseRegistry() = default;
}



namespace Catch {

    AssertionHandler::AssertionHandler
        (   StringRef macroName,
            SourceLineInfo const& lineInfo,
            StringRef capturedExpression,
            ResultDisposition::Flags resultDisposition )
    :   m_assertionInfo{ macroName, lineInfo, capturedExpression, resultDisposition },
        m_resultCapture( getResultCapture() )
    {
        m_resultCapture.notifyAssertionStarted( m_assertionInfo );
    }

    void AssertionHandler::handleExpr( ITransientExpression const& expr ) {
        m_resultCapture.handleExpr( m_assertionInfo, expr, m_reaction );
    }
    void AssertionHandler::handleMessage(ResultWas::OfType resultType, std::string&& message) {
        m_resultCapture.handleMessage( m_assertionInfo, resultType, CATCH_MOVE(message), m_reaction );
    }

    auto AssertionHandler::allowThrows() const -> bool {
        return getCurrentContext().getConfig()->allowThrows();
    }

    void AssertionHandler::complete() {
        m_completed = true;
        if( m_reaction.shouldDebugBreak ) {

            // If you find your debugger stopping you here then go one level up on the
            // call-stack for the code that caused it (typically a failed assertion)

            // (To go back to the test and change execution, jump over the throw, next)
            CATCH_BREAK_INTO_DEBUGGER();
        }
        if (m_reaction.shouldThrow) {
            throw_test_failure_exception();
        }
        if ( m_reaction.shouldSkip ) {
            throw_test_skip_exception();
        }
    }

    void AssertionHandler::handleUnexpectedInflightException() {
        m_resultCapture.handleUnexpectedInflightException( m_assertionInfo, Catch::translateActiveException(), m_reaction );
    }

    void AssertionHandler::handleExceptionThrownAsExpected() {
        m_resultCapture.handleNonExpr(m_assertionInfo, ResultWas::Ok, m_reaction);
    }
    void AssertionHandler::handleExceptionNotThrownAsExpected() {
        m_resultCapture.handleNonExpr(m_assertionInfo, ResultWas::Ok, m_reaction);
    }

    void AssertionHandler::handleUnexpectedExceptionNotThrown() {
        m_resultCapture.handleUnexpectedExceptionNotThrown( m_assertionInfo, m_reaction );
    }

    void AssertionHandler::handleThrowingCallSkipped() {
        m_resultCapture.handleNonExpr(m_assertionInfo, ResultWas::Ok, m_reaction);
    }

    // This is the overload that takes a string and infers the Equals matcher from it
    // The more general overload, that takes any string matcher, is in catch_capture_matchers.cpp
    void handleExceptionMatchExpr( AssertionHandler& handler, std::string const& str ) {
        handleExceptionMatchExpr( handler, Matchers::Equals( str ) );
    }

} // namespace Catch




#include <algorithm>

namespace Catch {
    namespace Detail {

        bool CaseInsensitiveLess::operator()( StringRef lhs,
                                              StringRef rhs ) const {
            return std::lexicographical_compare(
                lhs.begin(), lhs.end(),
                rhs.begin(), rhs.end(),
                []( char l, char r ) { return toLower( l ) < toLower( r ); } );
        }

        bool
        CaseInsensitiveEqualTo::operator()( StringRef lhs,
                                            StringRef rhs ) const {
            return std::equal(
                lhs.begin(), lhs.end(),
                rhs.begin(), rhs.end(),
                []( char l, char r ) { return toLower( l ) == toLower( r ); } );
        }

    } // namespace Detail
} // namespace Catch




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




#include <fstream>
#include <string>

namespace Catch {

    Clara::Parser makeCommandLineParser( ConfigData& config ) {

        using namespace Clara;

        auto const setWarning = [&]( std::string const& warning ) {
            if ( warning == "NoAssertions" ) {
                config.warnings = static_cast<WarnAbout::What>(config.warnings | WarnAbout::NoAssertions);
                return ParserResult::ok( ParseResultType::Matched );
            } else if ( warning == "UnmatchedTestSpec" ) {
                config.warnings = static_cast<WarnAbout::What>(config.warnings | WarnAbout::UnmatchedTestSpec);
                return ParserResult::ok( ParseResultType::Matched );
            }

            return ParserResult ::runtimeError(
                "Unrecognised warning option: '" + warning + '\'' );
        };
        auto const loadTestNamesFromFile = [&]( std::string const& filename ) {
                std::ifstream f( filename.c_str() );
                if( !f.is_open() )
                    return ParserResult::runtimeError( "Unable to load input file: '" + filename + '\'' );

                std::string line;
                while( std::getline( f, line ) ) {
                    line = trim(line);
                    if( !line.empty() && !startsWith( line, '#' ) ) {
                        if( !startsWith( line, '"' ) )
                            line = '"' + CATCH_MOVE(line) + '"';
                        config.testsOrTags.push_back( line );
                        config.testsOrTags.emplace_back( "," );
                    }
                }
                //Remove comma in the end
                if(!config.testsOrTags.empty())
                    config.testsOrTags.erase( config.testsOrTags.end()-1 );

                return ParserResult::ok( ParseResultType::Matched );
            };
        auto const setTestOrder = [&]( std::string const& order ) {
                if( startsWith( "declared", order ) )
                    config.runOrder = TestRunOrder::Declared;
                else if( startsWith( "lexical", order ) )
                    config.runOrder = TestRunOrder::LexicographicallySorted;
                else if( startsWith( "random", order ) )
                    config.runOrder = TestRunOrder::Randomized;
                else
                    return ParserResult::runtimeError( "Unrecognised ordering: '" + order + '\'' );
                return ParserResult::ok( ParseResultType::Matched );
            };
        auto const setRngSeed = [&]( std::string const& seed ) {
                if( seed == "time" ) {
                    config.rngSeed = generateRandomSeed(GenerateFrom::Time);
                    return ParserResult::ok(ParseResultType::Matched);
                } else if (seed == "random-device") {
                    config.rngSeed = generateRandomSeed(GenerateFrom::RandomDevice);
                    return ParserResult::ok(ParseResultType::Matched);
                }

                // TODO: ideally we should be parsing uint32_t directly
                //       fix this later when we add new parse overload
                auto parsedSeed = parseUInt( seed, 0 );
                if ( !parsedSeed ) {
                    return ParserResult::runtimeError( "Could not parse '" + seed + "' as seed" );
                }
                config.rngSeed = *parsedSeed;
                return ParserResult::ok( ParseResultType::Matched );
            };
        auto const setDefaultColourMode = [&]( std::string const& colourMode ) {
            Optional<ColourMode> maybeMode = Catch::Detail::stringToColourMode(toLower( colourMode ));
            if ( !maybeMode ) {
                return ParserResult::runtimeError(
                    "colour mode must be one of: default, ansi, win32, "
                    "or none. '" +
                    colourMode + "' is not recognised" );
            }
            auto mode = *maybeMode;
            if ( !isColourImplAvailable( mode ) ) {
                return ParserResult::runtimeError(
                    "colour mode '" + colourMode +
                    "' is not supported in this binary" );
            }
            config.defaultColourMode = mode;
            return ParserResult::ok( ParseResultType::Matched );
        };
        auto const setWaitForKeypress = [&]( std::string const& keypress ) {
                auto keypressLc = toLower( keypress );
                if (keypressLc == "never")
                    config.waitForKeypress = WaitForKeypress::Never;
                else if( keypressLc == "start" )
                    config.waitForKeypress = WaitForKeypress::BeforeStart;
                else if( keypressLc == "exit" )
                    config.waitForKeypress = WaitForKeypress::BeforeExit;
                else if( keypressLc == "both" )
                    config.waitForKeypress = WaitForKeypress::BeforeStartAndExit;
                else
                    return ParserResult::runtimeError( "keypress argument must be one of: never, start, exit or both. '" + keypress + "' not recognised" );
            return ParserResult::ok( ParseResultType::Matched );
            };
        auto const setVerbosity = [&]( std::string const& verbosity ) {
            auto lcVerbosity = toLower( verbosity );
            if( lcVerbosity == "quiet" )
                config.verbosity = Verbosity::Quiet;
            else if( lcVerbosity == "normal" )
                config.verbosity = Verbosity::Normal;
            else if( lcVerbosity == "high" )
                config.verbosity = Verbosity::High;
            else
                return ParserResult::runtimeError( "Unrecognised verbosity, '" + verbosity + '\'' );
            return ParserResult::ok( ParseResultType::Matched );
        };
        auto const setReporter = [&]( std::string const& userReporterSpec ) {
            if ( userReporterSpec.empty() ) {
                return ParserResult::runtimeError( "Received empty reporter spec." );
            }

            Optional<ReporterSpec> parsed =
                parseReporterSpec( userReporterSpec );
            if ( !parsed ) {
                return ParserResult::runtimeError(
                    "Could not parse reporter spec '" + userReporterSpec +
                    "'" );
            }

            auto const& reporterSpec = *parsed;

            auto const& factories =
                getRegistryHub().getReporterRegistry().getFactories();
            auto result = factories.find( reporterSpec.name() );

            if ( result == factories.end() ) {
                return ParserResult::runtimeError(
                    "Unrecognized reporter, '" + reporterSpec.name() +
                    "'. Check available with --list-reporters" );
            }


            const bool hadOutputFile = reporterSpec.outputFile().some();
            config.reporterSpecifications.push_back( CATCH_MOVE( *parsed ) );
            // It would be enough to check this only once at the very end, but
            // there is  not a place where we could call this check, so do it
            // every time it could fail. For valid inputs, this is still called
            // at most once.
            if (!hadOutputFile) {
                int n_reporters_without_file = 0;
                for (auto const& spec : config.reporterSpecifications) {
                    if (spec.outputFile().none()) {
                        n_reporters_without_file++;
                    }
                }
                if (n_reporters_without_file > 1) {
                    return ParserResult::runtimeError( "Only one reporter may have unspecified output file." );
                }
            }

            return ParserResult::ok( ParseResultType::Matched );
        };
        auto const setShardCount = [&]( std::string const& shardCount ) {
            auto parsedCount = parseUInt( shardCount );
            if ( !parsedCount ) {
                return ParserResult::runtimeError(
                    "Could not parse '" + shardCount + "' as shard count" );
            }
            if ( *parsedCount == 0 ) {
                return ParserResult::runtimeError(
                    "Shard count must be positive" );
            }
            config.shardCount = *parsedCount;
            return ParserResult::ok( ParseResultType::Matched );
        };

        auto const setShardIndex = [&](std::string const& shardIndex) {
            auto parsedIndex = parseUInt( shardIndex );
            if ( !parsedIndex ) {
                return ParserResult::runtimeError(
                    "Could not parse '" + shardIndex + "' as shard index" );
            }
            config.shardIndex = *parsedIndex;
            return ParserResult::ok( ParseResultType::Matched );
        };

        auto cli
            = ExeName( config.processName )
            | Help( config.showHelp )
            | Opt( config.showSuccessfulTests )
                ["-s"]["--success"]
                ( "include successful tests in output" )
            | Opt( config.shouldDebugBreak )
                ["-b"]["--break"]
                ( "break into debugger on failure" )
            | Opt( config.noThrow )
                ["-e"]["--nothrow"]
                ( "skip exception tests" )
            | Opt( config.showInvisibles )
                ["-i"]["--invisibles"]
                ( "show invisibles (tabs, newlines)" )
            | Opt( config.defaultOutputFilename, "filename" )
                ["-o"]["--out"]
                ( "default output filename" )
            | Opt( accept_many, setReporter, "name[::key=value]*" )
                ["-r"]["--reporter"]
                ( "reporter to use (defaults to console)" )
            | Opt( config.name, "name" )
                ["-n"]["--name"]
                ( "suite name" )
            | Opt( [&]( bool ){ config.abortAfter = 1; } )
                ["-a"]["--abort"]
                ( "abort at first failure" )
            | Opt( [&]( int x ){ config.abortAfter = x; }, "no. failures" )
                ["-x"]["--abortx"]
                ( "abort after x failures" )
            | Opt( accept_many, setWarning, "warning name" )
                ["-w"]["--warn"]
                ( "enable warnings" )
            | Opt( [&]( bool flag ) { config.showDurations = flag ? ShowDurations::Always : ShowDurations::Never; }, "yes|no" )
                ["-d"]["--durations"]
                ( "show test durations" )
            | Opt( config.minDuration, "seconds" )
                ["-D"]["--min-duration"]
                ( "show test durations for tests taking at least the given number of seconds" )
            | Opt( loadTestNamesFromFile, "filename" )
                ["-f"]["--input-file"]
                ( "load test names to run from a file" )
            | Opt( config.filenamesAsTags )
                ["-#"]["--filenames-as-tags"]
                ( "adds a tag for the filename" )
            | Opt( config.sectionsToRun, "section name" )
                ["-c"]["--section"]
                ( "specify section to run" )
            | Opt( setVerbosity, "quiet|normal|high" )
                ["-v"]["--verbosity"]
                ( "set output verbosity" )
            | Opt( config.listTests )
                ["--list-tests"]
                ( "list all/matching test cases" )
            | Opt( config.listTags )
                ["--list-tags"]
                ( "list all/matching tags" )
            | Opt( config.listReporters )
                ["--list-reporters"]
                ( "list all available reporters" )
            | Opt( config.listListeners )
                ["--list-listeners"]
                ( "list all listeners" )
            | Opt( setTestOrder, "decl|lex|rand" )
                ["--order"]
                ( "test case order (defaults to decl)" )
            | Opt( setRngSeed, "'time'|'random-device'|number" )
                ["--rng-seed"]
                ( "set a specific seed for random numbers" )
            | Opt( setDefaultColourMode, "ansi|win32|none|default" )
                ["--colour-mode"]
                ( "what color mode should be used as default" )
            | Opt( config.libIdentify )
                ["--libidentify"]
                ( "report name and version according to libidentify standard" )
            | Opt( setWaitForKeypress, "never|start|exit|both" )
                ["--wait-for-keypress"]
                ( "waits for a keypress before exiting" )
            | Opt( config.skipBenchmarks)
                ["--skip-benchmarks"]
                ( "disable running benchmarks")
            | Opt( config.benchmarkSamples, "samples" )
                ["--benchmark-samples"]
                ( "number of samples to collect (default: 100)" )
            | Opt( config.benchmarkResamples, "resamples" )
                ["--benchmark-resamples"]
                ( "number of resamples for the bootstrap (default: 100000)" )
            | Opt( config.benchmarkConfidenceInterval, "confidence interval" )
                ["--benchmark-confidence-interval"]
                ( "confidence interval for the bootstrap (between 0 and 1, default: 0.95)" )
            | Opt( config.benchmarkNoAnalysis )
                ["--benchmark-no-analysis"]
                ( "perform only measurements; do not perform any analysis" )
            | Opt( config.benchmarkWarmupTime, "benchmarkWarmupTime" )
                ["--benchmark-warmup-time"]
                ( "amount of time in milliseconds spent on warming up each test (default: 100)" )
            | Opt( setShardCount, "shard count" )
                ["--shard-count"]
                ( "split the tests to execute into this many groups" )
            | Opt( setShardIndex, "shard index" )
                ["--shard-index"]
                ( "index of the group of tests to execute (see --shard-count)" )
            | Opt( config.allowZeroTests )
                ["--allow-running-no-tests"]
                ( "Treat 'No tests run' as a success" )
            | Arg( config.testsOrTags, "test name|pattern|tags" )
                ( "which test or tests to use" );

        return cli;
    }

} // end namespace Catch


#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif



#include <cassert>
#include <ostream>
#include <utility>

namespace Catch {

    ColourImpl::~ColourImpl() = default;

    ColourImpl::ColourGuard ColourImpl::guardColour( Colour::Code colourCode ) {
        return ColourGuard(colourCode, this );
    }

    void ColourImpl::ColourGuard::engageImpl( std::ostream& stream ) {
        assert( &stream == &m_colourImpl->m_stream->stream() &&
                "Engaging colour guard for different stream than used by the "
                "parent colour implementation" );
        static_cast<void>( stream );

        m_engaged = true;
        m_colourImpl->use( m_code );
    }

    ColourImpl::ColourGuard::ColourGuard( Colour::Code code,
                                          ColourImpl const* colour ):
        m_colourImpl( colour ), m_code( code ) {
    }
    ColourImpl::ColourGuard::ColourGuard( ColourGuard&& rhs ) noexcept:
        m_colourImpl( rhs.m_colourImpl ),
        m_code( rhs.m_code ),
        m_engaged( rhs.m_engaged ) {
        rhs.m_engaged = false;
    }
    ColourImpl::ColourGuard&
    ColourImpl::ColourGuard::operator=( ColourGuard&& rhs ) noexcept {
        using std::swap;
        swap( m_colourImpl, rhs.m_colourImpl );
        swap( m_code, rhs.m_code );
        swap( m_engaged, rhs.m_engaged );

        return *this;
    }
    ColourImpl::ColourGuard::~ColourGuard() {
        if ( m_engaged ) {
            m_colourImpl->use( Colour::None );
        }
    }

    ColourImpl::ColourGuard&
    ColourImpl::ColourGuard::engage( std::ostream& stream ) & {
        engageImpl( stream );
        return *this;
    }

    ColourImpl::ColourGuard&&
    ColourImpl::ColourGuard::engage( std::ostream& stream ) && {
        engageImpl( stream );
        return CATCH_MOVE(*this);
    }

    namespace {
        //! A do-nothing implementation of colour, used as fallback for unknown
        //! platforms, and when the user asks to deactivate all colours.
        class NoColourImpl final : public ColourImpl {
        public:
            NoColourImpl( IStream* stream ): ColourImpl( stream ) {}

        private:
            void use( Colour::Code ) const override {}
        };
    } // namespace


} // namespace Catch


#if defined ( CATCH_CONFIG_COLOUR_WIN32 ) /////////////////////////////////////////

namespace Catch {
namespace {

    class Win32ColourImpl final : public ColourImpl {
    public:
        Win32ColourImpl(IStream* stream):
            ColourImpl(stream) {
            CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
            GetConsoleScreenBufferInfo( GetStdHandle( STD_OUTPUT_HANDLE ),
                                        &csbiInfo );
            originalForegroundAttributes = csbiInfo.wAttributes & ~( BACKGROUND_GREEN | BACKGROUND_RED | BACKGROUND_BLUE | BACKGROUND_INTENSITY );
            originalBackgroundAttributes = csbiInfo.wAttributes & ~( FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY );
        }

        static bool useImplementationForStream(IStream const& stream) {
            // Win32 text colour APIs can only be used on console streams
            // We cannot check that the output hasn't been redirected,
            // so we just check that the original stream is console stream.
            return stream.isConsole();
        }

    private:
        void use( Colour::Code _colourCode ) const override {
            switch( _colourCode ) {
                case Colour::None:      return setTextAttribute( originalForegroundAttributes );
                case Colour::White:     return setTextAttribute( FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE );
                case Colour::Red:       return setTextAttribute( FOREGROUND_RED );
                case Colour::Green:     return setTextAttribute( FOREGROUND_GREEN );
                case Colour::Blue:      return setTextAttribute( FOREGROUND_BLUE );
                case Colour::Cyan:      return setTextAttribute( FOREGROUND_BLUE | FOREGROUND_GREEN );
                case Colour::Yellow:    return setTextAttribute( FOREGROUND_RED | FOREGROUND_GREEN );
                case Colour::Grey:      return setTextAttribute( 0 );

                case Colour::LightGrey:     return setTextAttribute( FOREGROUND_INTENSITY );
                case Colour::BrightRed:     return setTextAttribute( FOREGROUND_INTENSITY | FOREGROUND_RED );
                case Colour::BrightGreen:   return setTextAttribute( FOREGROUND_INTENSITY | FOREGROUND_GREEN );
                case Colour::BrightWhite:   return setTextAttribute( FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE );
                case Colour::BrightYellow:  return setTextAttribute( FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN );

                case Colour::Bright: CATCH_INTERNAL_ERROR( "not a colour" );

                default:
                    CATCH_ERROR( "Unknown colour requested" );
            }
        }

        void setTextAttribute( WORD _textAttribute ) const {
            SetConsoleTextAttribute( GetStdHandle( STD_OUTPUT_HANDLE ),
                                     _textAttribute |
                                         originalBackgroundAttributes );
        }
        WORD originalForegroundAttributes;
        WORD originalBackgroundAttributes;
    };

} // end anon namespace
} // end namespace Catch

#endif // Windows/ ANSI/ None


#if defined( CATCH_PLATFORM_LINUX ) || defined( CATCH_PLATFORM_MAC ) || defined( __GLIBC__ )
#    define CATCH_INTERNAL_HAS_ISATTY
#    include <unistd.h>
#endif

namespace Catch {
namespace {

    class ANSIColourImpl final : public ColourImpl {
    public:
        ANSIColourImpl( IStream* stream ): ColourImpl( stream ) {}

        static bool useImplementationForStream(IStream const& stream) {
            // This is kinda messy due to trying to support a bunch of
            // different platforms at once.
            // The basic idea is that if we are asked to do autodetection (as
            // opposed to being told to use posixy colours outright), then we
            // only want to use the colours if we are writing to console.
            // However, console might be redirected, so we make an attempt at
            // checking for that on platforms where we know how to do that.
            bool useColour = stream.isConsole();
#if defined( CATCH_INTERNAL_HAS_ISATTY ) && \
    !( defined( __DJGPP__ ) && defined( __STRICT_ANSI__ ) )
            ErrnoGuard _; // for isatty
            useColour = useColour && isatty( STDOUT_FILENO );
#    endif
#    if defined( CATCH_PLATFORM_MAC ) || defined( CATCH_PLATFORM_IPHONE )
            useColour = useColour && !isDebuggerActive();
#    endif

            return useColour;
        }

    private:
        void use( Colour::Code _colourCode ) const override {
            auto setColour = [&out =
                                  m_stream->stream()]( char const* escapeCode ) {
                // The escape sequence must be flushed to console, otherwise
                // if stdin and stderr are intermixed, we'd get accidentally
                // coloured output.
                out << '\033' << escapeCode << std::flush;
            };
            switch( _colourCode ) {
                case Colour::None:
                case Colour::White:     return setColour( "[0m" );
                case Colour::Red:       return setColour( "[0;31m" );
                case Colour::Green:     return setColour( "[0;32m" );
                case Colour::Blue:      return setColour( "[0;34m" );
                case Colour::Cyan:      return setColour( "[0;36m" );
                case Colour::Yellow:    return setColour( "[0;33m" );
                case Colour::Grey:      return setColour( "[1;30m" );

                case Colour::LightGrey:     return setColour( "[0;37m" );
                case Colour::BrightRed:     return setColour( "[1;31m" );
                case Colour::BrightGreen:   return setColour( "[1;32m" );
                case Colour::BrightWhite:   return setColour( "[1;37m" );
                case Colour::BrightYellow:  return setColour( "[1;33m" );

                case Colour::Bright: CATCH_INTERNAL_ERROR( "not a colour" );
                default: CATCH_INTERNAL_ERROR( "Unknown colour requested" );
            }
        }
    };

} // end anon namespace
} // end namespace Catch

namespace Catch {

    Detail::unique_ptr<ColourImpl> makeColourImpl( ColourMode colourSelection,
                                                   IStream* stream ) {
#if defined( CATCH_CONFIG_COLOUR_WIN32 )
        if ( colourSelection == ColourMode::Win32 ) {
            return Detail::make_unique<Win32ColourImpl>( stream );
        }
#endif
        if ( colourSelection == ColourMode::ANSI ) {
            return Detail::make_unique<ANSIColourImpl>( stream );
        }
        if ( colourSelection == ColourMode::None ) {
            return Detail::make_unique<NoColourImpl>( stream );
        }

        if ( colourSelection == ColourMode::PlatformDefault) {
#if defined( CATCH_CONFIG_COLOUR_WIN32 )
            if ( Win32ColourImpl::useImplementationForStream( *stream ) ) {
                return Detail::make_unique<Win32ColourImpl>( stream );
            }
#endif
            if ( ANSIColourImpl::useImplementationForStream( *stream ) ) {
                return Detail::make_unique<ANSIColourImpl>( stream );
            }
            return Detail::make_unique<NoColourImpl>( stream );
        }

        CATCH_ERROR( "Could not create colour impl for selection " << static_cast<int>(colourSelection) );
    }

    bool isColourImplAvailable( ColourMode colourSelection ) {
        switch ( colourSelection ) {
#if defined( CATCH_CONFIG_COLOUR_WIN32 )
        case ColourMode::Win32:
#endif
        case ColourMode::ANSI:
        case ColourMode::None:
        case ColourMode::PlatformDefault:
            return true;
        default:
            return false;
        }
    }


} // end namespace Catch

#if defined(__clang__)
#    pragma clang diagnostic pop
#endif




namespace Catch {

    Context* Context::currentContext = nullptr;

    void cleanUpContext() {
        delete Context::currentContext;
        Context::currentContext = nullptr;
    }
    void Context::createContext() {
        currentContext = new Context();
    }

    Context& getCurrentMutableContext() {
        if ( !Context::currentContext ) { Context::createContext(); }
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.UndefReturn)
        return *Context::currentContext;
    }

    SimplePcg32& sharedRng() {
        static SimplePcg32 s_rng;
        return s_rng;
    }

}





#include <ostream>

#if defined(CATCH_CONFIG_ANDROID_LOGWRITE)
#include <android/log.h>

    namespace Catch {
        void writeToDebugConsole( std::string const& text ) {
            __android_log_write( ANDROID_LOG_DEBUG, "Catch", text.c_str() );
        }
    }

#elif defined(CATCH_PLATFORM_WINDOWS)

    namespace Catch {
        void writeToDebugConsole( std::string const& text ) {
            ::OutputDebugStringA( text.c_str() );
        }
    }

#else

    namespace Catch {
        void writeToDebugConsole( std::string const& text ) {
            // !TBD: Need a version for Mac/ XCode and other IDEs
            Catch::cout() << text;
        }
    }

#endif // Platform



#if defined(CATCH_PLATFORM_MAC) || defined(CATCH_PLATFORM_IPHONE)

#  include <cassert>
#  include <sys/types.h>
#  include <unistd.h>
#  include <cstddef>
#  include <ostream>

#ifdef __apple_build_version__
    // These headers will only compile with AppleClang (XCode)
    // For other compilers (Clang, GCC, ... ) we need to exclude them
#  include <sys/sysctl.h>
#endif

    namespace Catch {
        #ifdef __apple_build_version__
        // The following function is taken directly from the following technical note:
        // https://developer.apple.com/library/archive/qa/qa1361/_index.html

        // Returns true if the current process is being debugged (either
        // running under the debugger or has a debugger attached post facto).
        bool isDebuggerActive(){
            int                 mib[4];
            struct kinfo_proc   info;
            std::size_t         size;

            // Initialize the flags so that, if sysctl fails for some bizarre
            // reason, we get a predictable result.

            info.kp_proc.p_flag = 0;

            // Initialize mib, which tells sysctl the info we want, in this case
            // we're looking for information about a specific process ID.

            mib[0] = CTL_KERN;
            mib[1] = KERN_PROC;
            mib[2] = KERN_PROC_PID;
            mib[3] = getpid();

            // Call sysctl.

            size = sizeof(info);
            if( sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, nullptr, 0) != 0 ) {
                Catch::cerr() << "\n** Call to sysctl failed - unable to determine if debugger is active **\n\n" << std::flush;
                return false;
            }

            // We're being debugged if the P_TRACED flag is set.

            return ( (info.kp_proc.p_flag & P_TRACED) != 0 );
        }
        #else
        bool isDebuggerActive() {
            // We need to find another way to determine this for non-appleclang compilers on macOS
            return false;
        }
        #endif
    } // namespace Catch

#elif defined(CATCH_PLATFORM_LINUX)
    #include <fstream>
    #include <string>

    namespace Catch{
        // The standard POSIX way of detecting a debugger is to attempt to
        // ptrace() the process, but this needs to be done from a child and not
        // this process itself to still allow attaching to this process later
        // if wanted, so is rather heavy. Under Linux we have the PID of the
        // "debugger" (which doesn't need to be gdb, of course, it could also
        // be strace, for example) in /proc/$PID/status, so just get it from
        // there instead.
        bool isDebuggerActive(){
            // Libstdc++ has a bug, where std::ifstream sets errno to 0
            // This way our users can properly assert over errno values
            ErrnoGuard guard;
            std::ifstream in("/proc/self/status");
            for( std::string line; std::getline(in, line); ) {
                static const int PREFIX_LEN = 11;
                if( line.compare(0, PREFIX_LEN, "TracerPid:\t") == 0 ) {
                    // We're traced if the PID is not 0 and no other PID starts
                    // with 0 digit, so it's enough to check for just a single
                    // character.
                    return line.length() > PREFIX_LEN && line[PREFIX_LEN] != '0';
                }
            }

            return false;
        }
    } // namespace Catch
#elif defined(_MSC_VER)
    extern "C" __declspec(dllimport) int __stdcall IsDebuggerPresent();
    namespace Catch {
        bool isDebuggerActive() {
            return IsDebuggerPresent() != 0;
        }
    }
#elif defined(__MINGW32__)
    extern "C" __declspec(dllimport) int __stdcall IsDebuggerPresent();
    namespace Catch {
        bool isDebuggerActive() {
            return IsDebuggerPresent() != 0;
        }
    }
#else
    namespace Catch {
       bool isDebuggerActive() { return false; }
    }
#endif // Platform




namespace Catch {

    void ITransientExpression::streamReconstructedExpression(
        std::ostream& os ) const {
        // We can't make this function pure virtual to keep ITransientExpression
        // constexpr, so we write error message instead
        os << "Some class derived from ITransientExpression without overriding streamReconstructedExpression";
    }

    void formatReconstructedExpression( std::ostream &os, std::string const& lhs, StringRef op, std::string const& rhs ) {
        if( lhs.size() + rhs.size() < 40 &&
                lhs.find('\n') == std::string::npos &&
                rhs.find('\n') == std::string::npos )
            os << lhs << ' ' << op << ' ' << rhs;
        else
            os << lhs << '\n' << op << '\n' << rhs;
    }
}



#include <stdexcept>


namespace Catch {
#if defined(CATCH_CONFIG_DISABLE_EXCEPTIONS) && !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS_CUSTOM_HANDLER)
    [[noreturn]]
    void throw_exception(std::exception const& e) {
        Catch::cerr() << "Catch will terminate because it needed to throw an exception.\n"
                      << "The message was: " << e.what() << '\n';
        std::terminate();
    }
#endif

    [[noreturn]]
    void throw_logic_error(std::string const& msg) {
        throw_exception(std::logic_error(msg));
    }

    [[noreturn]]
    void throw_domain_error(std::string const& msg) {
        throw_exception(std::domain_error(msg));
    }

    [[noreturn]]
    void throw_runtime_error(std::string const& msg) {
        throw_exception(std::runtime_error(msg));
    }



} // namespace Catch;



#include <cassert>

namespace Catch {

    IMutableEnumValuesRegistry::~IMutableEnumValuesRegistry() = default;

    namespace Detail {

        namespace {
            // Extracts the actual name part of an enum instance
            // In other words, it returns the Blue part of Bikeshed::Colour::Blue
            StringRef extractInstanceName(StringRef enumInstance) {
                // Find last occurrence of ":"
                size_t name_start = enumInstance.size();
                while (name_start > 0 && enumInstance[name_start - 1] != ':') {
                    --name_start;
                }
                return enumInstance.substr(name_start, enumInstance.size() - name_start);
            }
        }

        std::vector<StringRef> parseEnums( StringRef enums ) {
            auto enumValues = splitStringRef( enums, ',' );
            std::vector<StringRef> parsed;
            parsed.reserve( enumValues.size() );
            for( auto const& enumValue : enumValues ) {
                parsed.push_back(trim(extractInstanceName(enumValue)));
            }
            return parsed;
        }

        EnumInfo::~EnumInfo() = default;

        StringRef EnumInfo::lookup( int value ) const {
            for( auto const& valueToName : m_values ) {
                if( valueToName.first == value )
                    return valueToName.second;
            }
            return "{** unexpected enum value **}"_sr;
        }

        Catch::Detail::unique_ptr<EnumInfo> makeEnumInfo( StringRef enumName, StringRef allValueNames, std::vector<int> const& values ) {
            auto enumInfo = Catch::Detail::make_unique<EnumInfo>();
            enumInfo->m_name = enumName;
            enumInfo->m_values.reserve( values.size() );

            const auto valueNames = Catch::Detail::parseEnums( allValueNames );
            assert( valueNames.size() == values.size() );
            std::size_t i = 0;
            for( auto value : values )
                enumInfo->m_values.emplace_back(value, valueNames[i++]);

            return enumInfo;
        }

        EnumInfo const& EnumValuesRegistry::registerEnum( StringRef enumName, StringRef allValueNames, std::vector<int> const& values ) {
            m_enumInfos.push_back(makeEnumInfo(enumName, allValueNames, values));
            return *m_enumInfos.back();
        }

    } // Detail
} // Catch





#include <cerrno>

namespace Catch {
        ErrnoGuard::ErrnoGuard():m_oldErrno(errno){}
        ErrnoGuard::~ErrnoGuard() { errno = m_oldErrno; }
}



#include <exception>

namespace Catch {

#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
    namespace {
        static std::string tryTranslators(
            std::vector<
                Detail::unique_ptr<IExceptionTranslator const>> const& translators ) {
            if ( translators.empty() ) {
                std::rethrow_exception( std::current_exception() );
            } else {
                return translators[0]->translate( translators.begin() + 1,
                                                  translators.end() );
            }
        }

    }
#endif //!defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)

    ExceptionTranslatorRegistry::~ExceptionTranslatorRegistry() = default;

    void ExceptionTranslatorRegistry::registerTranslator( Detail::unique_ptr<IExceptionTranslator>&& translator ) {
        m_translators.push_back( CATCH_MOVE( translator ) );
    }

#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
    std::string ExceptionTranslatorRegistry::translateActiveException() const {
        // Compiling a mixed mode project with MSVC means that CLR
        // exceptions will be caught in (...) as well. However, these do
        // do not fill-in std::current_exception and thus lead to crash
        // when attempting rethrow.
        // /EHa switch also causes structured exceptions to be caught
        // here, but they fill-in current_exception properly, so
        // at worst the output should be a little weird, instead of
        // causing a crash.
        if ( std::current_exception() == nullptr ) {
            return "Non C++ exception. Possibly a CLR exception.";
        }

        // First we try user-registered translators. If none of them can
        // handle the exception, it will be rethrown handled by our defaults.
        try {
            return tryTranslators(m_translators);
        }
        // To avoid having to handle TFE explicitly everywhere, we just
        // rethrow it so that it goes back up the caller.
        catch( TestFailureException& ) {
            return "{ nested assertion failed }";
        }
        catch( TestSkipException& ) {
            return "{ nested SKIP() called }";
        }
        catch( std::exception const& ex ) {
            return ex.what();
        }
        catch( std::string const& msg ) {
            return msg;
        }
        catch( const char* msg ) {
            return msg;
        }
        catch(...) {
            return "Unknown exception";
        }
    }

#else // ^^ Exceptions are enabled // Exceptions are disabled vv
    std::string ExceptionTranslatorRegistry::translateActiveException() const {
        CATCH_INTERNAL_ERROR("Attempted to translate active exception under CATCH_CONFIG_DISABLE_EXCEPTIONS!");
    }
#endif

}



/** \file
 * This file provides platform specific implementations of FatalConditionHandler
 *
 * This means that there is a lot of conditional compilation, and platform
 * specific code. Currently, Catch2 supports a dummy handler (if no
 * handler is desired), and 2 platform specific handlers:
 *  * Windows' SEH
 *  * POSIX signals
 *
 * Consequently, various pieces of code below are compiled if either of
 * the platform specific handlers is enabled, or if none of them are
 * enabled. It is assumed that both cannot be enabled at the same time,
 * and doing so should cause a compilation error.
 *
 * If another platform specific handler is added, the compile guards
 * below will need to be updated taking these assumptions into account.
 */



#include <algorithm>

#if !defined( CATCH_CONFIG_WINDOWS_SEH ) && !defined( CATCH_CONFIG_POSIX_SIGNALS )

namespace Catch {

    // If neither SEH nor signal handling is required, the handler impls
    // do not have to do anything, and can be empty.
    void FatalConditionHandler::engage_platform() {}
    void FatalConditionHandler::disengage_platform() noexcept {}
    FatalConditionHandler::FatalConditionHandler() = default;
    FatalConditionHandler::~FatalConditionHandler() = default;

} // end namespace Catch

#endif // !CATCH_CONFIG_WINDOWS_SEH && !CATCH_CONFIG_POSIX_SIGNALS

#if defined( CATCH_CONFIG_WINDOWS_SEH ) && defined( CATCH_CONFIG_POSIX_SIGNALS )
#error "Inconsistent configuration: Windows' SEH handling and POSIX signals cannot be enabled at the same time"
#endif // CATCH_CONFIG_WINDOWS_SEH && CATCH_CONFIG_POSIX_SIGNALS

#if defined( CATCH_CONFIG_WINDOWS_SEH ) || defined( CATCH_CONFIG_POSIX_SIGNALS )

namespace {
    //! Signals fatal error message to the run context
    void reportFatal( char const * const message ) {
        Catch::getCurrentContext().getResultCapture()->handleFatalErrorCondition( message );
    }

    //! Minimal size Catch2 needs for its own fatal error handling.
    //! Picked empirically, so it might not be sufficient on all
    //! platforms, and for all configurations.
    constexpr std::size_t minStackSizeForErrors = 32 * 1024;
} // end unnamed namespace

#endif // CATCH_CONFIG_WINDOWS_SEH || CATCH_CONFIG_POSIX_SIGNALS

#if defined( CATCH_CONFIG_WINDOWS_SEH )

namespace Catch {

    struct SignalDefs { DWORD id; const char* name; };

    // There is no 1-1 mapping between signals and windows exceptions.
    // Windows can easily distinguish between SO and SigSegV,
    // but SigInt, SigTerm, etc are handled differently.
    static constexpr SignalDefs signalDefs[] = {
        { EXCEPTION_ILLEGAL_INSTRUCTION,  "SIGILL - Illegal instruction signal" },
        { EXCEPTION_STACK_OVERFLOW, "SIGSEGV - Stack overflow" },
        { EXCEPTION_ACCESS_VIOLATION, "SIGSEGV - Segmentation violation signal" },
        { EXCEPTION_INT_DIVIDE_BY_ZERO, "Divide by zero error" },
    };

    static LONG CALLBACK topLevelExceptionFilter(PEXCEPTION_POINTERS ExceptionInfo) {
        for (auto const& def : signalDefs) {
            if (ExceptionInfo->ExceptionRecord->ExceptionCode == def.id) {
                reportFatal(def.name);
            }
        }
        // If its not an exception we care about, pass it along.
        // This stops us from eating debugger breaks etc.
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // Since we do not support multiple instantiations, we put these
    // into global variables and rely on cleaning them up in outlined
    // constructors/destructors
    static LPTOP_LEVEL_EXCEPTION_FILTER previousTopLevelExceptionFilter = nullptr;


    // For MSVC, we reserve part of the stack memory for handling
    // memory overflow structured exception.
    FatalConditionHandler::FatalConditionHandler() {
        ULONG guaranteeSize = static_cast<ULONG>(minStackSizeForErrors);
        if (!SetThreadStackGuarantee(&guaranteeSize)) {
            // We do not want to fully error out, because needing
            // the stack reserve should be rare enough anyway.
            Catch::cerr()
                << "Failed to reserve piece of stack."
                << " Stack overflows will not be reported successfully.";
        }
    }

    // We do not attempt to unset the stack guarantee, because
    // Windows does not support lowering the stack size guarantee.
    FatalConditionHandler::~FatalConditionHandler() = default;


    void FatalConditionHandler::engage_platform() {
        // Register as a the top level exception filter.
        previousTopLevelExceptionFilter = SetUnhandledExceptionFilter(topLevelExceptionFilter);
    }

    void FatalConditionHandler::disengage_platform() noexcept {
        if (SetUnhandledExceptionFilter(previousTopLevelExceptionFilter) != topLevelExceptionFilter) {
            Catch::cerr()
                << "Unexpected SEH unhandled exception filter on disengage."
                << " The filter was restored, but might be rolled back unexpectedly.";
        }
        previousTopLevelExceptionFilter = nullptr;
    }

} // end namespace Catch

#endif // CATCH_CONFIG_WINDOWS_SEH

#if defined( CATCH_CONFIG_POSIX_SIGNALS )

#include <signal.h>

namespace Catch {

    struct SignalDefs {
        int id;
        const char* name;
    };

    static constexpr SignalDefs signalDefs[] = {
        { SIGINT,  "SIGINT - Terminal interrupt signal" },
        { SIGILL,  "SIGILL - Illegal instruction signal" },
        { SIGFPE,  "SIGFPE - Floating point error signal" },
        { SIGSEGV, "SIGSEGV - Segmentation violation signal" },
        { SIGTERM, "SIGTERM - Termination request signal" },
        { SIGABRT, "SIGABRT - Abort (abnormal termination) signal" }
    };

// Older GCCs trigger -Wmissing-field-initializers for T foo = {}
// which is zero initialization, but not explicit. We want to avoid
// that.
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

    static char* altStackMem = nullptr;
    static std::size_t altStackSize = 0;
    static stack_t oldSigStack{};
    static struct sigaction oldSigActions[sizeof(signalDefs) / sizeof(SignalDefs)]{};

    static void restorePreviousSignalHandlers() noexcept {
        // We set signal handlers back to the previous ones. Hopefully
        // nobody overwrote them in the meantime, and doesn't expect
        // their signal handlers to live past ours given that they
        // installed them after ours..
        for (std::size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
            sigaction(signalDefs[i].id, &oldSigActions[i], nullptr);
        }
        // Return the old stack
        sigaltstack(&oldSigStack, nullptr);
    }

    static void handleSignal( int sig ) {
        char const * name = "<unknown signal>";
        for (auto const& def : signalDefs) {
            if (sig == def.id) {
                name = def.name;
                break;
            }
        }
        // We need to restore previous signal handlers and let them do
        // their thing, so that the users can have the debugger break
        // when a signal is raised, and so on.
        restorePreviousSignalHandlers();
        reportFatal( name );
        raise( sig );
    }

    FatalConditionHandler::FatalConditionHandler() {
        assert(!altStackMem && "Cannot initialize POSIX signal handler when one already exists");
        if (altStackSize == 0) {
            altStackSize = std::max(static_cast<size_t>(SIGSTKSZ), minStackSizeForErrors);
        }
        altStackMem = new char[altStackSize]();
    }

    FatalConditionHandler::~FatalConditionHandler() {
        delete[] altStackMem;
        // We signal that another instance can be constructed by zeroing
        // out the pointer.
        altStackMem = nullptr;
    }

    void FatalConditionHandler::engage_platform() {
        stack_t sigStack;
        sigStack.ss_sp = altStackMem;
        sigStack.ss_size = altStackSize;
        sigStack.ss_flags = 0;
        sigaltstack(&sigStack, &oldSigStack);
        struct sigaction sa = { };

        sa.sa_handler = handleSignal;
        sa.sa_flags = SA_ONSTACK;
        for (std::size_t i = 0; i < sizeof(signalDefs)/sizeof(SignalDefs); ++i) {
            sigaction(signalDefs[i].id, &sa, &oldSigActions[i]);
        }
    }

#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif


    void FatalConditionHandler::disengage_platform() noexcept {
        restorePreviousSignalHandlers();
    }

} // end namespace Catch

#endif // CATCH_CONFIG_POSIX_SIGNALS




#include <cstring>

namespace Catch {
    namespace Detail {

        uint32_t convertToBits(float f) {
            static_assert(sizeof(float) == sizeof(uint32_t), "Important ULP matcher assumption violated");
            uint32_t i;
            std::memcpy(&i, &f, sizeof(f));
            return i;
        }

        uint64_t convertToBits(double d) {
            static_assert(sizeof(double) == sizeof(uint64_t), "Important ULP matcher assumption violated");
            uint64_t i;
            std::memcpy(&i, &d, sizeof(d));
            return i;
        }

#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        bool directCompare( float lhs, float rhs ) { return lhs == rhs; }
        bool directCompare( double lhs, double rhs ) { return lhs == rhs; }
#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic pop
#endif


    } // end namespace Detail
} // end namespace Catch






#include <cstdlib>

namespace Catch {
    namespace Detail {

#if !defined (CATCH_CONFIG_GETENV)
        char const* getEnv( char const* ) { return nullptr; }
#else

        char const* getEnv( char const* varName ) {
#    if defined( _MSC_VER )
#        pragma warning( push )
#        pragma warning( disable : 4996 ) // use getenv_s instead of getenv
#    endif

            return std::getenv( varName );

#    if defined( _MSC_VER )
#        pragma warning( pop )
#    endif
        }
#endif
} // namespace Detail
} // namespace Catch




#include <cstdio>
#include <fstream>

namespace Catch {

    Catch::IStream::~IStream() = default;

namespace Detail {
    namespace {
        template<typename WriterF, std::size_t bufferSize=256>
        class StreamBufImpl final : public std::streambuf {
            char data[bufferSize];
            WriterF m_writer;

        public:
            StreamBufImpl() {
                setp( data, data + sizeof(data) );
            }

            ~StreamBufImpl() noexcept override {
                StreamBufImpl::sync();
            }

        private:
            int overflow( int c ) override {
                sync();

                if( c != EOF ) {
                    if( pbase() == epptr() )
                        m_writer( std::string( 1, static_cast<char>( c ) ) );
                    else
                        sputc( static_cast<char>( c ) );
                }
                return 0;
            }

            int sync() override {
                if( pbase() != pptr() ) {
                    m_writer( std::string( pbase(), static_cast<std::string::size_type>( pptr() - pbase() ) ) );
                    setp( pbase(), epptr() );
                }
                return 0;
            }
        };

        ///////////////////////////////////////////////////////////////////////////

        struct OutputDebugWriter {

            void operator()( std::string const& str ) {
                if ( !str.empty() ) {
                    writeToDebugConsole( str );
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////////

        class FileStream final : public IStream {
            std::ofstream m_ofs;
        public:
            FileStream( std::string const& filename ) {
                m_ofs.open( filename.c_str() );
                CATCH_ENFORCE( !m_ofs.fail(), "Unable to open file: '" << filename << '\'' );
                m_ofs << std::unitbuf;
            }
        public: // IStream
            std::ostream& stream() override {
                return m_ofs;
            }
        };

        ///////////////////////////////////////////////////////////////////////////

        class CoutStream final : public IStream {
            std::ostream m_os;
        public:
            // Store the streambuf from cout up-front because
            // cout may get redirected when running tests
            CoutStream() : m_os( Catch::cout().rdbuf() ) {}

        public: // IStream
            std::ostream& stream() override { return m_os; }
            bool isConsole() const override { return true; }
        };

        class CerrStream : public IStream {
            std::ostream m_os;

        public:
            // Store the streambuf from cerr up-front because
            // cout may get redirected when running tests
            CerrStream(): m_os( Catch::cerr().rdbuf() ) {}

        public: // IStream
            std::ostream& stream() override { return m_os; }
            bool isConsole() const override { return true; }
        };

        ///////////////////////////////////////////////////////////////////////////

        class DebugOutStream final : public IStream {
            Detail::unique_ptr<StreamBufImpl<OutputDebugWriter>> m_streamBuf;
            std::ostream m_os;
        public:
            DebugOutStream()
            :   m_streamBuf( Detail::make_unique<StreamBufImpl<OutputDebugWriter>>() ),
                m_os( m_streamBuf.get() )
            {}

        public: // IStream
            std::ostream& stream() override { return m_os; }
        };

    } // unnamed namespace
} // namespace Detail

    ///////////////////////////////////////////////////////////////////////////

    auto makeStream( std::string const& filename ) -> Detail::unique_ptr<IStream> {
        if ( filename.empty() || filename == "-" ) {
            return Detail::make_unique<Detail::CoutStream>();
        }
        if( filename[0] == '%' ) {
            if ( filename == "%debug" ) {
                return Detail::make_unique<Detail::DebugOutStream>();
            } else if ( filename == "%stderr" ) {
                return Detail::make_unique<Detail::CerrStream>();
            } else if ( filename == "%stdout" ) {
                return Detail::make_unique<Detail::CoutStream>();
            } else {
                CATCH_ERROR( "Unrecognised stream: '" << filename << '\'' );
            }
        }
        return Detail::make_unique<Detail::FileStream>( filename );
    }

}



namespace Catch {
    void JsonUtils::indent( std::ostream& os, std::uint64_t level ) {
        for ( std::uint64_t i = 0; i < level; ++i ) {
            os << "  ";
        }
    }
    void JsonUtils::appendCommaNewline( std::ostream& os,
                                        bool& should_comma,
                                        std::uint64_t level ) {
        if ( should_comma ) { os << ','; }
        should_comma = true;
        os << '\n';
        indent( os, level );
    }

    JsonObjectWriter::JsonObjectWriter( std::ostream& os ):
        JsonObjectWriter{ os, 0 } {}

    JsonObjectWriter::JsonObjectWriter( std::ostream& os,
                                        std::uint64_t indent_level ):
        m_os{ os }, m_indent_level{ indent_level } {
        m_os << '{';
    }
    JsonObjectWriter::JsonObjectWriter( JsonObjectWriter&& source ) noexcept:
        m_os{ source.m_os },
        m_indent_level{ source.m_indent_level },
        m_should_comma{ source.m_should_comma },
        m_active{ source.m_active } {
        source.m_active = false;
    }

    JsonObjectWriter::~JsonObjectWriter() {
        if ( !m_active ) { return; }

        m_os << '\n';
        JsonUtils::indent( m_os, m_indent_level );
        m_os << '}';
    }

    JsonValueWriter JsonObjectWriter::write( StringRef key ) {
        JsonUtils::appendCommaNewline(
            m_os, m_should_comma, m_indent_level + 1 );

        m_os << '"' << key << "\": ";
        return JsonValueWriter{ m_os, m_indent_level + 1 };
    }

    JsonArrayWriter::JsonArrayWriter( std::ostream& os ):
        JsonArrayWriter{ os, 0 } {}
    JsonArrayWriter::JsonArrayWriter( std::ostream& os,
                                      std::uint64_t indent_level ):
        m_os{ os }, m_indent_level{ indent_level } {
        m_os << '[';
    }
    JsonArrayWriter::JsonArrayWriter( JsonArrayWriter&& source ) noexcept:
        m_os{ source.m_os },
        m_indent_level{ source.m_indent_level },
        m_should_comma{ source.m_should_comma },
        m_active{ source.m_active } {
        source.m_active = false;
    }
    JsonArrayWriter::~JsonArrayWriter() {
        if ( !m_active ) { return; }

        m_os << '\n';
        JsonUtils::indent( m_os, m_indent_level );
        m_os << ']';
    }

    JsonObjectWriter JsonArrayWriter::writeObject() {
        JsonUtils::appendCommaNewline(
            m_os, m_should_comma, m_indent_level + 1 );
        return JsonObjectWriter{ m_os, m_indent_level + 1 };
    }

    JsonArrayWriter JsonArrayWriter::writeArray() {
        JsonUtils::appendCommaNewline(
            m_os, m_should_comma, m_indent_level + 1 );
        return JsonArrayWriter{ m_os, m_indent_level + 1 };
    }

    JsonArrayWriter& JsonArrayWriter::write( bool value ) {
        return writeImpl( value );
    }

    JsonValueWriter::JsonValueWriter( std::ostream& os ):
        JsonValueWriter{ os, 0 } {}

    JsonValueWriter::JsonValueWriter( std::ostream& os,
                                      std::uint64_t indent_level ):
        m_os{ os }, m_indent_level{ indent_level } {}

    JsonObjectWriter JsonValueWriter::writeObject() && {
        return JsonObjectWriter{ m_os, m_indent_level };
    }

    JsonArrayWriter JsonValueWriter::writeArray() && {
        return JsonArrayWriter{ m_os, m_indent_level };
    }

    void JsonValueWriter::write( Catch::StringRef value ) && {
        writeImpl( value, true );
    }

    void JsonValueWriter::write( bool value ) && {
        writeImpl( value ? "true"_sr : "false"_sr, false );
    }

    void JsonValueWriter::writeImpl( Catch::StringRef value, bool quote ) {
        if ( quote ) { m_os << '"'; }
        for (char c : value) {
            // Escape list taken from https://www.json.org/json-en.html,
            // string definition.
            // Note that while forward slash _can_ be escaped, it does
            // not have to be, if JSON is not further embedded somewhere
            // where forward slash is meaningful.
            if ( c == '"' ) {
                m_os << "\\\"";
            } else if ( c == '\\' ) {
                m_os << "\\\\";
            } else if ( c == '\b' ) {
                m_os << "\\b";
            } else if ( c == '\f' ) {
                m_os << "\\f";
            } else if ( c == '\n' ) {
                m_os << "\\n";
            } else if ( c == '\r' ) {
                m_os << "\\r";
            } else if ( c == '\t' ) {
                m_os << "\\t";
            } else {
                m_os << c;
            }
        }
        if ( quote ) { m_os << '"'; }
    }

} // namespace Catch




namespace Catch {

    auto operator << (std::ostream& os, LazyExpression const& lazyExpr) -> std::ostream& {
        if (lazyExpr.m_isNegated)
            os << '!';

        if (lazyExpr) {
            if (lazyExpr.m_isNegated && lazyExpr.m_transientExpression->isBinaryExpression())
                os << '(' << *lazyExpr.m_transientExpression << ')';
            else
                os << *lazyExpr.m_transientExpression;
        } else {
            os << "{** error - unchecked empty expression requested **}";
        }
        return os;
    }

} // namespace Catch




#ifdef CATCH_CONFIG_WINDOWS_CRTDBG
#include <crtdbg.h>

namespace Catch {

    LeakDetector::LeakDetector() {
        int flag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
        flag |= _CRTDBG_LEAK_CHECK_DF;
        flag |= _CRTDBG_ALLOC_MEM_DF;
        _CrtSetDbgFlag(flag);
        _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
        _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
        // Change this to leaking allocation's number to break there
        _CrtSetBreakAlloc(-1);
    }
}

#else // ^^ Windows crt debug heap enabled // Windows crt debug heap disabled vv

    Catch::LeakDetector::LeakDetector() = default;

#endif // CATCH_CONFIG_WINDOWS_CRTDBG

Catch::LeakDetector::~LeakDetector() {
    Catch::cleanUp();
}




namespace Catch {
    namespace {

        void listTests(IEventListener& reporter, IConfig const& config) {
            auto const& testSpec = config.testSpec();
            auto matchedTestCases = filterTests(getAllTestCasesSorted(config), testSpec, config);
            reporter.listTests(matchedTestCases);
        }

        void listTags(IEventListener& reporter, IConfig const& config) {
            auto const& testSpec = config.testSpec();
            std::vector<TestCaseHandle> matchedTestCases = filterTests(getAllTestCasesSorted(config), testSpec, config);

            std::map<StringRef, TagInfo, Detail::CaseInsensitiveLess> tagCounts;
            for (auto const& testCase : matchedTestCases) {
                for (auto const& tagName : testCase.getTestCaseInfo().tags) {
                    auto it = tagCounts.find(tagName.original);
                    if (it == tagCounts.end())
                        it = tagCounts.insert(std::make_pair(tagName.original, TagInfo())).first;
                    it->second.add(tagName.original);
                }
            }

            std::vector<TagInfo> infos; infos.reserve(tagCounts.size());
            for (auto& tagc : tagCounts) {
                infos.push_back(CATCH_MOVE(tagc.second));
            }

            reporter.listTags(infos);
        }

        void listReporters(IEventListener& reporter) {
            std::vector<ReporterDescription> descriptions;

            auto const& factories = getRegistryHub().getReporterRegistry().getFactories();
            descriptions.reserve(factories.size());
            for (auto const& fac : factories) {
                descriptions.push_back({ fac.first, fac.second->getDescription() });
            }

            reporter.listReporters(descriptions);
        }

        void listListeners(IEventListener& reporter) {
            std::vector<ListenerDescription> descriptions;

            auto const& factories =
                getRegistryHub().getReporterRegistry().getListeners();
            descriptions.reserve( factories.size() );
            for ( auto const& fac : factories ) {
                descriptions.push_back( { fac->getName(), fac->getDescription() } );
            }

            reporter.listListeners( descriptions );
        }

    } // end anonymous namespace

    void TagInfo::add( StringRef spelling ) {
        ++count;
        spellings.insert( spelling );
    }

    std::string TagInfo::all() const {
        // 2 per tag for brackets '[' and ']'
        size_t size =  spellings.size() * 2;
        for (auto const& spelling : spellings) {
            size += spelling.size();
        }

        std::string out; out.reserve(size);
        for (auto const& spelling : spellings) {
            out += '[';
            out += spelling;
            out += ']';
        }
        return out;
    }

    bool list( IEventListener& reporter, Config const& config ) {
        bool listed = false;
        if (config.listTests()) {
            listed = true;
            listTests(reporter, config);
        }
        if (config.listTags()) {
            listed = true;
            listTags(reporter, config);
        }
        if (config.listReporters()) {
            listed = true;
            listReporters(reporter);
        }
        if ( config.listListeners() ) {
            listed = true;
            listListeners( reporter );
        }
        return listed;
    }

} // end namespace Catch



namespace Catch {
    CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
    CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS
    static const LeakDetector leakDetector;
    CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
}

// Allow users of amalgamated .cpp file to remove our main and provide their own.
#if !defined(CATCH_AMALGAMATED_CUSTOM_MAIN)

#if defined(CATCH_CONFIG_WCHAR) && defined(CATCH_PLATFORM_WINDOWS) && defined(_UNICODE) && !defined(DO_NOT_USE_WMAIN)
// Standard C/C++ Win32 Unicode wmain entry point
extern "C" int __cdecl wmain (int argc, wchar_t * argv[], wchar_t * []) {
#else
// Standard C/C++ main entry point
int main (int argc, char * argv[]) {
#endif

    // We want to force the linker not to discard the global variable
    // and its constructor, as it (optionally) registers leak detector
    (void)&Catch::leakDetector;

    return Catch::Session().run( argc, argv );
}

#endif // !defined(CATCH_AMALGAMATED_CUSTOM_MAIN




namespace Catch {

    MessageInfo::MessageInfo(   StringRef _macroName,
                                SourceLineInfo const& _lineInfo,
                                ResultWas::OfType _type )
    :   macroName( _macroName ),
        lineInfo( _lineInfo ),
        type( _type ),
        sequence( ++globalCount )
    {}

    // This may need protecting if threading support is added
    unsigned int MessageInfo::globalCount = 0;

} // end namespace Catch



#include <cstdio>
#include <cstring>
#include <iosfwd>
#include <sstream>

#if defined( CATCH_CONFIG_NEW_CAPTURE )
#    if defined( _MSC_VER )
#        include <io.h> //_dup and _dup2
#        define dup _dup
#        define dup2 _dup2
#        define fileno _fileno
#    else
#        include <unistd.h> // dup and dup2
#    endif
#endif

namespace Catch {

    namespace {
        //! A no-op implementation, used if no reporter wants output
        //! redirection.
        class NoopRedirect : public OutputRedirect {
            void activateImpl() override {}
            void deactivateImpl() override {}
            std::string getStdout() override { return {}; }
            std::string getStderr() override { return {}; }
            void clearBuffers() override {}
        };

        /**
         * Redirects specific stream's rdbuf with another's.
         *
         * Redirection can be stopped and started on-demand, assumes
         * that the underlying stream's rdbuf aren't changed by other
         * users.
         */
        class RedirectedStreamNew {
            std::ostream& m_originalStream;
            std::ostream& m_redirectionStream;
            std::streambuf* m_prevBuf;

        public:
            RedirectedStreamNew( std::ostream& originalStream,
                                 std::ostream& redirectionStream ):
                m_originalStream( originalStream ),
                m_redirectionStream( redirectionStream ),
                m_prevBuf( m_originalStream.rdbuf() ) {}

            void startRedirect() {
                m_originalStream.rdbuf( m_redirectionStream.rdbuf() );
            }
            void stopRedirect() { m_originalStream.rdbuf( m_prevBuf ); }
        };

        /**
         * Redirects the `std::cout`, `std::cerr`, `std::clog` streams,
         * but does not touch the actual `stdout`/`stderr` file descriptors.
         */
        class StreamRedirect : public OutputRedirect {
            ReusableStringStream m_redirectedOut, m_redirectedErr;
            RedirectedStreamNew m_cout, m_cerr, m_clog;

        public:
            StreamRedirect():
                m_cout( Catch::cout(), m_redirectedOut.get() ),
                m_cerr( Catch::cerr(), m_redirectedErr.get() ),
                m_clog( Catch::clog(), m_redirectedErr.get() ) {}

            void activateImpl() override {
                m_cout.startRedirect();
                m_cerr.startRedirect();
                m_clog.startRedirect();
            }
            void deactivateImpl() override {
                m_cout.stopRedirect();
                m_cerr.stopRedirect();
                m_clog.stopRedirect();
            }
            std::string getStdout() override { return m_redirectedOut.str(); }
            std::string getStderr() override { return m_redirectedErr.str(); }
            void clearBuffers() override {
                m_redirectedOut.str( "" );
                m_redirectedErr.str( "" );
            }
        };

#if defined( CATCH_CONFIG_NEW_CAPTURE )

        // Windows's implementation of std::tmpfile is terrible (it tries
        // to create a file inside system folder, thus requiring elevated
        // privileges for the binary), so we have to use tmpnam(_s) and
        // create the file ourselves there.
        class TempFile {
        public:
            TempFile( TempFile const& ) = delete;
            TempFile& operator=( TempFile const& ) = delete;
            TempFile( TempFile&& ) = delete;
            TempFile& operator=( TempFile&& ) = delete;

#    if defined( _MSC_VER )
            TempFile() {
                if ( tmpnam_s( m_buffer ) ) {
                    CATCH_RUNTIME_ERROR( "Could not get a temp filename" );
                }
                if ( fopen_s( &m_file, m_buffer, "wb+" ) ) {
                    char buffer[100];
                    if ( strerror_s( buffer, errno ) ) {
                        CATCH_RUNTIME_ERROR(
                            "Could not translate errno to a string" );
                    }
                    CATCH_RUNTIME_ERROR( "Could not open the temp file: '"
                                         << m_buffer
                                         << "' because: " << buffer );
                }
            }
#    else
            TempFile() {
                m_file = std::tmpfile();
                if ( !m_file ) {
                    CATCH_RUNTIME_ERROR( "Could not create a temp file." );
                }
            }
#    endif

            ~TempFile() {
                // TBD: What to do about errors here?
                std::fclose( m_file );
                // We manually create the file on Windows only, on Linux
                // it will be autodeleted
#    if defined( _MSC_VER )
                std::remove( m_buffer );
#    endif
            }

            std::FILE* getFile() { return m_file; }
            std::string getContents() {
                ReusableStringStream sstr;
                constexpr long buffer_size = 100;
                char buffer[buffer_size + 1] = {};
                long current_pos = ftell( m_file );
                CATCH_ENFORCE( current_pos >= 0,
                               "ftell failed, errno: " << errno );
                std::rewind( m_file );
                while ( current_pos > 0 ) {
                    auto read_characters =
                        std::fread( buffer,
                                    1,
                                    std::min( buffer_size, current_pos ),
                                    m_file );
                    buffer[read_characters] = '\0';
                    sstr << buffer;
                    current_pos -= static_cast<long>( read_characters );
                }
                return sstr.str();
            }

            void clear() { std::rewind( m_file ); }

        private:
            std::FILE* m_file = nullptr;
            char m_buffer[L_tmpnam] = { 0 };
        };

        /**
         * Redirects the actual `stdout`/`stderr` file descriptors.
         *
         * Works by replacing the file descriptors numbered 1 and 2
         * with an open temporary file.
         */
        class FileRedirect : public OutputRedirect {
            TempFile m_outFile, m_errFile;
            int m_originalOut = -1;
            int m_originalErr = -1;

            // Flushes cout/cerr/clog streams and stdout/stderr FDs
            void flushEverything() {
                Catch::cout() << std::flush;
                fflush( stdout );
                // Since we support overriding these streams, we flush cerr
                // even though std::cerr is unbuffered
                Catch::cerr() << std::flush;
                Catch::clog() << std::flush;
                fflush( stderr );
            }

        public:
            FileRedirect():
                m_originalOut( dup( fileno( stdout ) ) ),
                m_originalErr( dup( fileno( stderr ) ) ) {
                CATCH_ENFORCE( m_originalOut >= 0, "Could not dup stdout" );
                CATCH_ENFORCE( m_originalErr >= 0, "Could not dup stderr" );
            }

            std::string getStdout() override { return m_outFile.getContents(); }
            std::string getStderr() override { return m_errFile.getContents(); }
            void clearBuffers() override {
                m_outFile.clear();
                m_errFile.clear();
            }

            void activateImpl() override {
                // We flush before starting redirect, to ensure that we do
                // not capture the end of message sent before activation.
                flushEverything();

                int ret;
                ret = dup2( fileno( m_outFile.getFile() ), fileno( stdout ) );
                CATCH_ENFORCE( ret >= 0,
                               "dup2 to stdout has failed, errno: " << errno );
                ret = dup2( fileno( m_errFile.getFile() ), fileno( stderr ) );
                CATCH_ENFORCE( ret >= 0,
                               "dup2 to stderr has failed, errno: " << errno );
            }
            void deactivateImpl() override {
                // We flush before ending redirect, to ensure that we
                // capture all messages sent while the redirect was active.
                flushEverything();

                int ret;
                ret = dup2( m_originalOut, fileno( stdout ) );
                CATCH_ENFORCE(
                    ret >= 0,
                    "dup2 of original stdout has failed, errno: " << errno );
                ret = dup2( m_originalErr, fileno( stderr ) );
                CATCH_ENFORCE(
                    ret >= 0,
                    "dup2 of original stderr has failed, errno: " << errno );
            }
        };

#endif // CATCH_CONFIG_NEW_CAPTURE

    } // end namespace

    bool isRedirectAvailable( OutputRedirect::Kind kind ) {
        switch ( kind ) {
        // These two are always available
        case OutputRedirect::None:
        case OutputRedirect::Streams:
            return true;
#if defined( CATCH_CONFIG_NEW_CAPTURE )
        case OutputRedirect::FileDescriptors:
            return true;
#endif
        default:
            return false;
        }
    }

    Detail::unique_ptr<OutputRedirect> makeOutputRedirect( bool actual ) {
        if ( actual ) {
            // TODO: Clean this up later
#if defined( CATCH_CONFIG_NEW_CAPTURE )
            return Detail::make_unique<FileRedirect>();
#else
            return Detail::make_unique<StreamRedirect>();
#endif
        } else {
            return Detail::make_unique<NoopRedirect>();
        }
    }

    RedirectGuard scopedActivate( OutputRedirect& redirectImpl ) {
        return RedirectGuard( true, redirectImpl );
    }

    RedirectGuard scopedDeactivate( OutputRedirect& redirectImpl ) {
        return RedirectGuard( false, redirectImpl );
    }

    OutputRedirect::~OutputRedirect() = default;

    RedirectGuard::RedirectGuard( bool activate, OutputRedirect& redirectImpl ):
        m_redirect( &redirectImpl ),
        m_activate( activate ),
        m_previouslyActive( redirectImpl.isActive() ) {

        // Skip cases where there is no actual state change.
        if ( m_activate == m_previouslyActive ) { return; }

        if ( m_activate ) {
            m_redirect->activate();
        } else {
            m_redirect->deactivate();
        }
    }

    RedirectGuard::~RedirectGuard() noexcept( false ) {
        if ( m_moved ) { return; }
        // Skip cases where there is no actual state change.
        if ( m_activate == m_previouslyActive ) { return; }

        if ( m_activate ) {
            m_redirect->deactivate();
        } else {
            m_redirect->activate();
        }
    }

    RedirectGuard::RedirectGuard( RedirectGuard&& rhs ) noexcept:
        m_redirect( rhs.m_redirect ),
        m_activate( rhs.m_activate ),
        m_previouslyActive( rhs.m_previouslyActive ),
        m_moved( false ) {
        rhs.m_moved = true;
    }

    RedirectGuard& RedirectGuard::operator=( RedirectGuard&& rhs ) noexcept {
        m_redirect = rhs.m_redirect;
        m_activate = rhs.m_activate;
        m_previouslyActive = rhs.m_previouslyActive;
        m_moved = false;
        rhs.m_moved = true;
        return *this;
    }

} // namespace Catch

#if defined( CATCH_CONFIG_NEW_CAPTURE )
#    if defined( _MSC_VER )
#        undef dup
#        undef dup2
#        undef fileno
#    endif
#endif




#include <limits>
#include <stdexcept>

namespace Catch {

    Optional<unsigned int> parseUInt(std::string const& input, int base) {
        auto trimmed = trim( input );
        // std::stoull is annoying and accepts numbers starting with '-',
        // it just negates them into unsigned int
        if ( trimmed.empty() || trimmed[0] == '-' ) {
            return {};
        }

        CATCH_TRY {
            size_t pos = 0;
            const auto ret = std::stoull( trimmed, &pos, base );

            // We did not consume the whole input, so there is an issue
            // This can be bunch of different stuff, like multiple numbers
            // in the input, or invalid digits/characters and so on. Either
            // way, we do not want to return the partially parsed result.
            if ( pos != trimmed.size() ) {
                return {};
            }
            // Too large
            if ( ret > std::numeric_limits<unsigned int>::max() ) {
                return {};
            }
            return static_cast<unsigned int>(ret);
        }
        CATCH_CATCH_ANON( std::invalid_argument const& ) {
            // no conversion could be performed
        }
        CATCH_CATCH_ANON( std::out_of_range const& ) {
            // the input does not fit into an unsigned long long
        }
        return {};
    }

} // namespace Catch




#include <cmath>

namespace Catch {

#if !defined(CATCH_CONFIG_POLYFILL_ISNAN)
    bool isnan(float f) {
        return std::isnan(f);
    }
    bool isnan(double d) {
        return std::isnan(d);
    }
#else
    // For now we only use this for embarcadero
    bool isnan(float f) {
        return std::_isnan(f);
    }
    bool isnan(double d) {
        return std::_isnan(d);
    }
#endif

#if !defined( CATCH_CONFIG_GLOBAL_NEXTAFTER )
    float nextafter( float x, float y ) { return std::nextafter( x, y ); }
    double nextafter( double x, double y ) { return std::nextafter( x, y ); }
#else
    float nextafter( float x, float y ) { return ::nextafterf( x, y ); }
    double nextafter( double x, double y ) { return ::nextafter( x, y ); }
#endif

} // end namespace Catch



#if defined( __clang__ )
#    define CATCH2_CLANG_NO_SANITIZE_INTEGER \
        __attribute__( ( no_sanitize( "unsigned-integer-overflow" ) ) )
#else
#    define CATCH2_CLANG_NO_SANITIZE_INTEGER
#endif
namespace Catch {

namespace {

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4146) // we negate uint32 during the rotate
#endif
        // Safe rotr implementation thanks to John Regehr
        CATCH2_CLANG_NO_SANITIZE_INTEGER
        uint32_t rotate_right(uint32_t val, uint32_t count) {
            const uint32_t mask = 31;
            count &= mask;
            return (val >> count) | (val << (-count & mask));
        }

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

}


    SimplePcg32::SimplePcg32(result_type seed_) {
        seed(seed_);
    }


    void SimplePcg32::seed(result_type seed_) {
        m_state = 0;
        (*this)();
        m_state += seed_;
        (*this)();
    }

    void SimplePcg32::discard(uint64_t skip) {
        // We could implement this to run in O(log n) steps, but this
        // should suffice for our use case.
        for (uint64_t s = 0; s < skip; ++s) {
            static_cast<void>((*this)());
        }
    }

    CATCH2_CLANG_NO_SANITIZE_INTEGER
    SimplePcg32::result_type SimplePcg32::operator()() {
        // prepare the output value
        const uint32_t xorshifted = static_cast<uint32_t>(((m_state >> 18u) ^ m_state) >> 27u);
        const auto output = rotate_right(xorshifted, static_cast<uint32_t>(m_state >> 59u));

        // advance state
        m_state = m_state * 6364136223846793005ULL + s_inc;

        return output;
    }

    bool operator==(SimplePcg32 const& lhs, SimplePcg32 const& rhs) {
        return lhs.m_state == rhs.m_state;
    }

    bool operator!=(SimplePcg32 const& lhs, SimplePcg32 const& rhs) {
        return lhs.m_state != rhs.m_state;
    }
}





#include <ctime>
#include <random>

namespace Catch {

    std::uint32_t generateRandomSeed( GenerateFrom from ) {
        switch ( from ) {
        case GenerateFrom::Time:
            return static_cast<std::uint32_t>( std::time( nullptr ) );

        case GenerateFrom::Default:
        case GenerateFrom::RandomDevice: {
            std::random_device rd;
            return Detail::fillBitsFrom<std::uint32_t>( rd );
        }

        default:
            CATCH_ERROR("Unknown generation method");
        }
    }

} // end namespace Catch




namespace Catch {
    struct ReporterRegistry::ReporterRegistryImpl {
        std::vector<Detail::unique_ptr<EventListenerFactory>> listeners;
        std::map<std::string, IReporterFactoryPtr, Detail::CaseInsensitiveLess>
            factories;
    };

    ReporterRegistry::ReporterRegistry():
        m_impl( Detail::make_unique<ReporterRegistryImpl>() ) {
        // Because it is impossible to move out of initializer list,
        // we have to add the elements manually
        m_impl->factories["Automake"] =
            Detail::make_unique<ReporterFactory<AutomakeReporter>>();
        m_impl->factories["compact"] =
            Detail::make_unique<ReporterFactory<CompactReporter>>();
        m_impl->factories["console"] =
            Detail::make_unique<ReporterFactory<ConsoleReporter>>();
        m_impl->factories["JUnit"] =
            Detail::make_unique<ReporterFactory<JunitReporter>>();
        m_impl->factories["SonarQube"] =
            Detail::make_unique<ReporterFactory<SonarQubeReporter>>();
        m_impl->factories["TAP"] =
            Detail::make_unique<ReporterFactory<TAPReporter>>();
        m_impl->factories["TeamCity"] =
            Detail::make_unique<ReporterFactory<TeamCityReporter>>();
        m_impl->factories["XML"] =
            Detail::make_unique<ReporterFactory<XmlReporter>>();
        m_impl->factories["JSON"] =
            Detail::make_unique<ReporterFactory<JsonReporter>>();
    }

    ReporterRegistry::~ReporterRegistry() = default;

    IEventListenerPtr
    ReporterRegistry::create( std::string const& name,
                              ReporterConfig&& config ) const {
        auto it = m_impl->factories.find( name );
        if ( it == m_impl->factories.end() ) return nullptr;
        return it->second->create( CATCH_MOVE( config ) );
    }

    void ReporterRegistry::registerReporter( std::string const& name,
                                             IReporterFactoryPtr factory ) {
        CATCH_ENFORCE( name.find( "::" ) == name.npos,
                       "'::' is not allowed in reporter name: '" + name +
                           '\'' );
        auto ret = m_impl->factories.emplace( name, CATCH_MOVE( factory ) );
        CATCH_ENFORCE( ret.second,
                       "reporter using '" + name +
                           "' as name was already registered" );
    }
    void ReporterRegistry::registerListener(
        Detail::unique_ptr<EventListenerFactory> factory ) {
        m_impl->listeners.push_back( CATCH_MOVE( factory ) );
    }

    std::map<std::string,
             IReporterFactoryPtr,
             Detail::CaseInsensitiveLess> const&
    ReporterRegistry::getFactories() const {
        return m_impl->factories;
    }

    std::vector<Detail::unique_ptr<EventListenerFactory>> const&
    ReporterRegistry::getListeners() const {
        return m_impl->listeners;
    }
} // namespace Catch





#include <algorithm>

namespace Catch {

    namespace {
        struct kvPair {
            StringRef key, value;
        };

        kvPair splitKVPair(StringRef kvString) {
            auto splitPos = static_cast<size_t>(
                std::find( kvString.begin(), kvString.end(), '=' ) -
                kvString.begin() );

            return { kvString.substr( 0, splitPos ),
                     kvString.substr( splitPos + 1, kvString.size() ) };
        }
    }

    namespace Detail {
        std::vector<std::string> splitReporterSpec( StringRef reporterSpec ) {
            static constexpr auto separator = "::";
            static constexpr size_t separatorSize = 2;

            size_t separatorPos = 0;
            auto findNextSeparator = [&reporterSpec]( size_t startPos ) {
                static_assert(
                    separatorSize == 2,
                    "The code below currently assumes 2 char separator" );

                auto currentPos = startPos;
                do {
                    while ( currentPos < reporterSpec.size() &&
                            reporterSpec[currentPos] != separator[0] ) {
                        ++currentPos;
                    }
                    if ( currentPos + 1 < reporterSpec.size() &&
                         reporterSpec[currentPos + 1] == separator[1] ) {
                        return currentPos;
                    }
                    ++currentPos;
                } while ( currentPos < reporterSpec.size() );

                return static_cast<size_t>( -1 );
            };

            std::vector<std::string> parts;

            while ( separatorPos < reporterSpec.size() ) {
                const auto nextSeparator = findNextSeparator( separatorPos );
                parts.push_back( static_cast<std::string>( reporterSpec.substr(
                    separatorPos, nextSeparator - separatorPos ) ) );

                if ( nextSeparator == static_cast<size_t>( -1 ) ) {
                    break;
                }
                separatorPos = nextSeparator + separatorSize;
            }

            // Handle a separator at the end.
            // This is not a valid spec, but we want to do validation in a
            // centralized place
            if ( separatorPos == reporterSpec.size() ) {
                parts.emplace_back();
            }

            return parts;
        }

        Optional<ColourMode> stringToColourMode( StringRef colourMode ) {
            if ( colourMode == "default" ) {
                return ColourMode::PlatformDefault;
            } else if ( colourMode == "ansi" ) {
                return ColourMode::ANSI;
            } else if ( colourMode == "win32" ) {
                return ColourMode::Win32;
            } else if ( colourMode == "none" ) {
                return ColourMode::None;
            } else {
                return {};
            }
        }
    } // namespace Detail


    bool operator==( ReporterSpec const& lhs, ReporterSpec const& rhs ) {
        return lhs.m_name == rhs.m_name &&
               lhs.m_outputFileName == rhs.m_outputFileName &&
               lhs.m_colourMode == rhs.m_colourMode &&
               lhs.m_customOptions == rhs.m_customOptions;
    }

    Optional<ReporterSpec> parseReporterSpec( StringRef reporterSpec ) {
        auto parts = Detail::splitReporterSpec( reporterSpec );

        assert( parts.size() > 0 && "Split should never return empty vector" );

        std::map<std::string, std::string> kvPairs;
        Optional<std::string> outputFileName;
        Optional<ColourMode> colourMode;

        // First part is always reporter name, so we skip it
        for ( size_t i = 1; i < parts.size(); ++i ) {
            auto kv = splitKVPair( parts[i] );
            auto key = kv.key, value = kv.value;

            if ( key.empty() || value.empty() ) { // NOLINT(bugprone-branch-clone)
                return {};
            } else if ( key[0] == 'X' ) {
                // This is a reporter-specific option, we don't check these
                // apart from basic sanity checks
                if ( key.size() == 1 ) {
                    return {};
                }

                auto ret = kvPairs.emplace( std::string(kv.key), std::string(kv.value) );
                if ( !ret.second ) {
                    // Duplicated key. We might want to handle this differently,
                    // e.g. by overwriting the existing value?
                    return {};
                }
            } else if ( key == "out" ) {
                // Duplicated key
                if ( outputFileName ) {
                    return {};
                }
                outputFileName = static_cast<std::string>( value );
            } else if ( key == "colour-mode" ) {
                // Duplicated key
                if ( colourMode ) {
                    return {};
                }
                colourMode = Detail::stringToColourMode( value );
                // Parsing failed
                if ( !colourMode ) {
                    return {};
                }
            } else {
                // Unrecognized option
                return {};
            }
        }

        return ReporterSpec{ CATCH_MOVE( parts[0] ),
                             CATCH_MOVE( outputFileName ),
                             CATCH_MOVE( colourMode ),
                             CATCH_MOVE( kvPairs ) };
    }

ReporterSpec::ReporterSpec(
        std::string name,
        Optional<std::string> outputFileName,
        Optional<ColourMode> colourMode,
        std::map<std::string, std::string> customOptions ):
        m_name( CATCH_MOVE( name ) ),
        m_outputFileName( CATCH_MOVE( outputFileName ) ),
        m_colourMode( CATCH_MOVE( colourMode ) ),
        m_customOptions( CATCH_MOVE( customOptions ) ) {}

} // namespace Catch



#include <cstdio>
#include <sstream>
#include <vector>

namespace Catch {

    // This class encapsulates the idea of a pool of ostringstreams that can be reused.
    struct StringStreams {
        std::vector<Detail::unique_ptr<std::ostringstream>> m_streams;
        std::vector<std::size_t> m_unused;
        std::ostringstream m_referenceStream; // Used for copy state/ flags from

        auto add() -> std::size_t {
            if( m_unused.empty() ) {
                m_streams.push_back( Detail::make_unique<std::ostringstream>() );
                return m_streams.size()-1;
            }
            else {
                auto index = m_unused.back();
                m_unused.pop_back();
                return index;
            }
        }

        void release( std::size_t index ) {
            m_streams[index]->copyfmt( m_referenceStream ); // Restore initial flags and other state
            m_unused.push_back(index);
        }
    };

    ReusableStringStream::ReusableStringStream()
    :   m_index( Singleton<StringStreams>::getMutable().add() ),
        m_oss( Singleton<StringStreams>::getMutable().m_streams[m_index].get() )
    {}

    ReusableStringStream::~ReusableStringStream() {
        static_cast<std::ostringstream*>( m_oss )->str("");
        m_oss->clear();
        Singleton<StringStreams>::getMutable().release( m_index );
    }

    std::string ReusableStringStream::str() const {
        return static_cast<std::ostringstream*>( m_oss )->str();
    }

    void ReusableStringStream::str( std::string const& str ) {
        static_cast<std::ostringstream*>( m_oss )->str( str );
    }


}




#include <cassert>
#include <algorithm>

namespace Catch {

    namespace Generators {
        namespace {
            struct GeneratorTracker final : TestCaseTracking::TrackerBase,
                                      IGeneratorTracker {
                GeneratorBasePtr m_generator;

                GeneratorTracker(
                    TestCaseTracking::NameAndLocation&& nameAndLocation,
                    TrackerContext& ctx,
                    ITracker* parent ):
                    TrackerBase( CATCH_MOVE( nameAndLocation ), ctx, parent ) {}

                static GeneratorTracker*
                acquire( TrackerContext& ctx,
                         TestCaseTracking::NameAndLocationRef const&
                             nameAndLocation ) {
                    GeneratorTracker* tracker;

                    ITracker& currentTracker = ctx.currentTracker();
                    // Under specific circumstances, the generator we want
                    // to acquire is also the current tracker. If this is
                    // the case, we have to avoid looking through current
                    // tracker's children, and instead return the current
                    // tracker.
                    // A case where this check is important is e.g.
                    //     for (int i = 0; i < 5; ++i) {
                    //         int n = GENERATE(1, 2);
                    //     }
                    //
                    // without it, the code above creates 5 nested generators.
                    if ( currentTracker.nameAndLocation() == nameAndLocation ) {
                        auto thisTracker = currentTracker.parent()->findChild(
                            nameAndLocation );
                        assert( thisTracker );
                        assert( thisTracker->isGeneratorTracker() );
                        tracker = static_cast<GeneratorTracker*>( thisTracker );
                    } else if ( ITracker* childTracker =
                                    currentTracker.findChild(
                                        nameAndLocation ) ) {
                        assert( childTracker );
                        assert( childTracker->isGeneratorTracker() );
                        tracker =
                            static_cast<GeneratorTracker*>( childTracker );
                    } else {
                        return nullptr;
                    }

                    if ( !tracker->isComplete() ) { tracker->open(); }

                    return tracker;
                }

                // TrackerBase interface
                bool isGeneratorTracker() const override { return true; }
                auto hasGenerator() const -> bool override {
                    return !!m_generator;
                }
                void close() override {
                    TrackerBase::close();
                    // If a generator has a child (it is followed by a section)
                    // and none of its children have started, then we must wait
                    // until later to start consuming its values.
                    // This catches cases where `GENERATE` is placed between two
                    // `SECTION`s.
                    // **The check for m_children.empty cannot be removed**.
                    // doing so would break `GENERATE` _not_ followed by
                    // `SECTION`s.
                    const bool should_wait_for_child = [&]() {
                        // No children -> nobody to wait for
                        if ( m_children.empty() ) { return false; }
                        // If at least one child started executing, don't wait
                        if ( std::find_if(
                                 m_children.begin(),
                                 m_children.end(),
                                 []( TestCaseTracking::ITrackerPtr const&
                                         tracker ) {
                                     return tracker->hasStarted();
                                 } ) != m_children.end() ) {
                            return false;
                        }

                        // No children have started. We need to check if they
                        // _can_ start, and thus we should wait for them, or
                        // they cannot start (due to filters), and we shouldn't
                        // wait for them
                        ITracker* parent = m_parent;
                        // This is safe: there is always at least one section
                        // tracker in a test case tracking tree
                        while ( !parent->isSectionTracker() ) {
                            parent = parent->parent();
                        }
                        assert( parent &&
                                "Missing root (test case) level section" );

                        auto const& parentSection =
                            static_cast<SectionTracker const&>( *parent );
                        auto const& filters = parentSection.getFilters();
                        // No filters -> no restrictions on running sections
                        if ( filters.empty() ) { return true; }

                        for ( auto const& child : m_children ) {
                            if ( child->isSectionTracker() &&
                                 std::find( filters.begin(),
                                            filters.end(),
                                            static_cast<SectionTracker const&>(
                                                *child )
                                                .trimmedName() ) !=
                                     filters.end() ) {
                                return true;
                            }
                        }
                        return false;
                    }();

                    // This check is a bit tricky, because m_generator->next()
                    // has a side-effect, where it consumes generator's current
                    // value, but we do not want to invoke the side-effect if
                    // this generator is still waiting for any child to start.
                    assert( m_generator && "Tracker without generator" );
                    if ( should_wait_for_child ||
                         ( m_runState == CompletedSuccessfully &&
                           m_generator->countedNext() ) ) {
                        m_children.clear();
                        m_runState = Executing;
                    }
                }

                // IGeneratorTracker interface
                auto getGenerator() const -> GeneratorBasePtr const& override {
                    return m_generator;
                }
                void setGenerator( GeneratorBasePtr&& generator ) override {
                    m_generator = CATCH_MOVE( generator );
                }
            };
        } // namespace
    }

    namespace Detail {
        // Assertions are owned by the thread that is executing them.
        // This allows for lock-free progress in common cases where we
        // do not need to send the assertion events to the reporter.
        // This also implies that messages are owned by their respective
        // threads, and should not be shared across different threads.
        //
        // For simplicity, we disallow messages in multi-threaded contexts,
        // but in the future we can enable them under this logic.
        //
        // This implies that various pieces of metadata referring to last
        // assertion result/source location/message handling, etc
        // should also be thread local. For now we just use naked globals
        // below, in the future we will want to allocate piece of memory
        // from heap, to avoid consuming too much thread-local storage.

        // This is used for the "if" part of CHECKED_IF/CHECKED_ELSE
        static thread_local bool g_lastAssertionPassed = false;
        // Should we clear message scopes before sending off the messages to
        // reporter? Set in `assertionPassedFastPath` to avoid doing the full
        // clear there for performance reasons.
        static thread_local bool g_clearMessageScopes = false;
        // This is the source location for last encountered macro. It is
        // used to provide the users with more precise location of error
        // when an unexpected exception/fatal error happens.
        static thread_local SourceLineInfo g_lastKnownLineInfo("DummyLocation", static_cast<size_t>(-1));
    }

    RunContext::RunContext(IConfig const* _config, IEventListenerPtr&& reporter)
    :   m_runInfo(_config->name()),
        m_config(_config),
        m_reporter(CATCH_MOVE(reporter)),
        m_outputRedirect( makeOutputRedirect( m_reporter->getPreferences().shouldRedirectStdOut ) ),
        m_abortAfterXFailedAssertions( m_config->abortAfter() ),
        m_reportAssertionStarting( m_reporter->getPreferences().shouldReportAllAssertionStarts ),
        m_includeSuccessfulResults( m_config->includeSuccessfulResults() || m_reporter->getPreferences().shouldReportAllAssertions ),
        m_shouldDebugBreak( m_config->shouldDebugBreak() )
    {
        getCurrentMutableContext().setResultCapture( this );
        m_reporter->testRunStarting(m_runInfo);
    }

    RunContext::~RunContext() {
        updateTotalsFromAtomics();
        m_reporter->testRunEnded(TestRunStats(m_runInfo, m_totals, aborting()));
    }

    Totals RunContext::runTest(TestCaseHandle const& testCase) {
        updateTotalsFromAtomics();
        const Totals prevTotals = m_totals;

        auto const& testInfo = testCase.getTestCaseInfo();
        m_reporter->testCaseStarting(testInfo);
        testCase.prepareTestCase();
        m_activeTestCase = &testCase;


        ITracker& rootTracker = m_trackerContext.startRun();
        assert(rootTracker.isSectionTracker());
        static_cast<SectionTracker&>(rootTracker).addInitialFilters(m_config->getSectionsToRun());

        // We intentionally only seed the internal RNG once per test case,
        // before it is first invoked. The reason for that is a complex
        // interplay of generator/section implementation details and the
        // Random*Generator types.
        //
        // The issue boils down to us needing to seed the Random*Generators
        // with different seed each, so that they return different sequences
        // of random numbers. We do this by giving them a number from the
        // shared RNG instance as their seed.
        //
        // However, this runs into an issue if the reseeding happens each
        // time the test case is entered (as opposed to first time only),
        // because multiple generators could get the same seed, e.g. in
        // ```cpp
        // TEST_CASE() {
        //     auto i = GENERATE(take(10, random(0, 100));
        //     SECTION("A") {
        //         auto j = GENERATE(take(10, random(0, 100));
        //     }
        //     SECTION("B") {
        //         auto k = GENERATE(take(10, random(0, 100));
        //     }
        // }
        // ```
        // `i` and `j` would properly return values from different sequences,
        // but `i` and `k` would return the same sequence, because their seed
        // would be the same.
        // (The reason their seeds would be the same is that the generator
        //  for k would be initialized when the test case is entered the second
        //  time, after the shared RNG instance was reset to the same value
        //  it had when the generator for i was initialized.)
        seedRng( *m_config );

        uint64_t testRuns = 0;
        std::string redirectedCout;
        std::string redirectedCerr;
        do {
            m_trackerContext.startCycle();
            m_testCaseTracker = &SectionTracker::acquire(m_trackerContext, TestCaseTracking::NameAndLocationRef(testInfo.name, testInfo.lineInfo));

            m_reporter->testCasePartialStarting(testInfo, testRuns);

            updateTotalsFromAtomics();
            const auto beforeRunTotals = m_totals;
            runCurrentTest();
            std::string oneRunCout = m_outputRedirect->getStdout();
            std::string oneRunCerr = m_outputRedirect->getStderr();
            m_outputRedirect->clearBuffers();
            redirectedCout += oneRunCout;
            redirectedCerr += oneRunCerr;

            updateTotalsFromAtomics();
            const auto singleRunTotals = m_totals.delta(beforeRunTotals);
            auto statsForOneRun = TestCaseStats(testInfo, singleRunTotals, CATCH_MOVE(oneRunCout), CATCH_MOVE(oneRunCerr), aborting());
            m_reporter->testCasePartialEnded(statsForOneRun, testRuns);

            ++testRuns;
        } while (!m_testCaseTracker->isSuccessfullyCompleted() && !aborting());

        Totals deltaTotals = m_totals.delta(prevTotals);
        if (testInfo.expectedToFail() && deltaTotals.testCases.passed > 0) {
            deltaTotals.assertions.failed++;
            deltaTotals.testCases.passed--;
            deltaTotals.testCases.failed++;
        }
        m_totals.testCases += deltaTotals.testCases;
        testCase.tearDownTestCase();
        m_reporter->testCaseEnded(TestCaseStats(testInfo,
                                  deltaTotals,
                                  CATCH_MOVE(redirectedCout),
                                  CATCH_MOVE(redirectedCerr),
                                  aborting()));

        m_activeTestCase = nullptr;
        m_testCaseTracker = nullptr;

        return deltaTotals;
    }


    void RunContext::assertionEnded(AssertionResult&& result) {
        Detail::g_lastKnownLineInfo = result.m_info.lineInfo;
        if (result.getResultType() == ResultWas::Ok) {
            m_atomicAssertionCount.passed++;
            Detail::g_lastAssertionPassed = true;
        } else if (result.getResultType() == ResultWas::ExplicitSkip) {
            m_atomicAssertionCount.skipped++;
            Detail::g_lastAssertionPassed = true;
        } else if (!result.succeeded()) {
            Detail::g_lastAssertionPassed = false;
            if (result.isOk()) {
            }
            else if( m_activeTestCase->getTestCaseInfo().okToFail() ) // Read from a shared state established before the threads could start, this is fine
                m_atomicAssertionCount.failedButOk++;
            else
                m_atomicAssertionCount.failed++;
        }
        else {
            Detail::g_lastAssertionPassed = true;
        }

        // From here, we are touching shared state and need mutex.
        Detail::LockGuard lock( m_assertionMutex );
        {
            if ( Detail::g_clearMessageScopes ) {
                m_messageScopes.clear();
                Detail::g_clearMessageScopes = false;
            }
            auto _ = scopedDeactivate( *m_outputRedirect );
            updateTotalsFromAtomics();
            m_reporter->assertionEnded( AssertionStats( result, m_messages, m_totals ) );
        }

        if ( result.getResultType() != ResultWas::Warning ) {
            m_messageScopes.clear();
        }

        // Reset working state. assertion info will be reset after
        // populateReaction is run if it is needed
        m_lastResult = CATCH_MOVE( result );
    }

    void RunContext::notifyAssertionStarted( AssertionInfo const& info ) {
        if (m_reportAssertionStarting) {
            Detail::LockGuard lock( m_assertionMutex );
            auto _ = scopedDeactivate( *m_outputRedirect );
            m_reporter->assertionStarting( info );
        }
    }

    bool RunContext::sectionStarted( StringRef sectionName,
                                     SourceLineInfo const& sectionLineInfo,
                                     Counts& assertions ) {
        ITracker& sectionTracker =
            SectionTracker::acquire( m_trackerContext,
                                     TestCaseTracking::NameAndLocationRef(
                                         sectionName, sectionLineInfo ) );

        if (!sectionTracker.isOpen())
            return false;
        m_activeSections.push_back(&sectionTracker);

        SectionInfo sectionInfo( sectionLineInfo, static_cast<std::string>(sectionName) );
        Detail::g_lastKnownLineInfo = sectionLineInfo;

        {
            auto _ = scopedDeactivate( *m_outputRedirect );
            m_reporter->sectionStarting( sectionInfo );
        }

        updateTotalsFromAtomics();
        assertions = m_totals.assertions;

        return true;
    }
    IGeneratorTracker*
    RunContext::acquireGeneratorTracker( StringRef generatorName,
                                         SourceLineInfo const& lineInfo ) {
        auto* tracker = Generators::GeneratorTracker::acquire(
            m_trackerContext,
            TestCaseTracking::NameAndLocationRef(
                 generatorName, lineInfo ) );
        Detail::g_lastKnownLineInfo = lineInfo;
        return tracker;
    }

    IGeneratorTracker* RunContext::createGeneratorTracker(
        StringRef generatorName,
        SourceLineInfo lineInfo,
        Generators::GeneratorBasePtr&& generator ) {

        auto nameAndLoc = TestCaseTracking::NameAndLocation( static_cast<std::string>( generatorName ), lineInfo );
        auto& currentTracker = m_trackerContext.currentTracker();
        assert(
            currentTracker.nameAndLocation() != nameAndLoc &&
            "Trying to create tracker for a generator that already has one" );

        auto newTracker = Catch::Detail::make_unique<Generators::GeneratorTracker>(
            CATCH_MOVE(nameAndLoc), m_trackerContext, &currentTracker );
        auto ret = newTracker.get();
        currentTracker.addChild( CATCH_MOVE( newTracker ) );

        ret->setGenerator( CATCH_MOVE( generator ) );
        ret->open();
        return ret;
    }

    bool RunContext::testForMissingAssertions(Counts& assertions) {
        if (assertions.total() != 0)
            return false;
        if (!m_config->warnAboutMissingAssertions())
            return false;
        if (m_trackerContext.currentTracker().hasChildren())
            return false;
        m_atomicAssertionCount.failed++;
        assertions.failed++;
        return true;
    }

    void RunContext::sectionEnded(SectionEndInfo&& endInfo) {
        updateTotalsFromAtomics();
        Counts assertions = m_totals.assertions - endInfo.prevAssertions;
        bool missingAssertions = testForMissingAssertions(assertions);

        if (!m_activeSections.empty()) {
            m_activeSections.back()->close();
            m_activeSections.pop_back();
        }

        {
            auto _ = scopedDeactivate( *m_outputRedirect );
            m_reporter->sectionEnded(
                SectionStats( CATCH_MOVE( endInfo.sectionInfo ),
                              assertions,
                              endInfo.durationInSeconds,
                              missingAssertions ) );
        }
    }

    void RunContext::sectionEndedEarly(SectionEndInfo&& endInfo) {
        if ( m_unfinishedSections.empty() ) {
            m_activeSections.back()->fail();
        } else {
            m_activeSections.back()->close();
        }
        m_activeSections.pop_back();

        m_unfinishedSections.push_back(CATCH_MOVE(endInfo));
    }

    void RunContext::benchmarkPreparing( StringRef name ) {
        auto _ = scopedDeactivate( *m_outputRedirect );
        m_reporter->benchmarkPreparing( name );
    }
    void RunContext::benchmarkStarting( BenchmarkInfo const& info ) {
        auto _ = scopedDeactivate( *m_outputRedirect );
        m_reporter->benchmarkStarting( info );
    }
    void RunContext::benchmarkEnded( BenchmarkStats<> const& stats ) {
        auto _ = scopedDeactivate( *m_outputRedirect );
        m_reporter->benchmarkEnded( stats );
    }
    void RunContext::benchmarkFailed( StringRef error ) {
        auto _ = scopedDeactivate( *m_outputRedirect );
        m_reporter->benchmarkFailed( error );
    }

    void RunContext::pushScopedMessage(MessageInfo const & message) {
        m_messages.push_back(message);
    }

    void RunContext::popScopedMessage( MessageInfo const& message ) {
        // Note: On average, it would probably be better to look for the message
        //       backwards. However, we do not expect to have to deal with more
        //       messages than low single digits, so the optimization is tiny,
        //       and we would have to hand-write the loop to avoid terrible
        //       codegen of reverse iterators in debug mode.
        m_messages.erase(
            std::find_if( m_messages.begin(),
                          m_messages.end(),
                          [id = message.sequence]( MessageInfo const& msg ) {
                              return msg.sequence == id;
                          } ) );
    }

    void RunContext::emplaceUnscopedMessage( MessageBuilder&& builder ) {
        m_messageScopes.emplace_back( CATCH_MOVE(builder) );
    }

    std::string RunContext::getCurrentTestName() const {
        return m_activeTestCase
            ? m_activeTestCase->getTestCaseInfo().name
            : std::string();
    }

    const AssertionResult * RunContext::getLastResult() const {
        // m_lastResult is updated inside the assertion slow-path, under
        // a mutex, so the read needs to happen under mutex as well.

        // TBD: The last result only makes sense if it is a thread-local
        //      thing, because the answer is different per thread, like
        //      last line info, whether last assertion passed, and so on.
        //
        //      However, the last result was also never updated in the
        //      assertion fast path, so it was always somewhat broken,
        //      and since IResultCapture::getLastResult is deprecated,
        //      we will leave it as is, until it is finally removed.
        Detail::LockGuard _( m_assertionMutex );
        return &(*m_lastResult);
    }

    void RunContext::exceptionEarlyReported() {
        m_shouldReportUnexpected = false;
    }

    void RunContext::handleFatalErrorCondition( StringRef message ) {
        // We lock only when touching the reporters directly, to avoid
        // deadlocks when we call into other functions that also want
        // to lock the mutex before touching reporters.
        //
        // This does mean that we allow other threads to run while handling
        // a fatal error, but this is all a best effort attempt anyway.
        {
            Detail::LockGuard lock( m_assertionMutex );
            // TODO: scoped deactivate here? Just give up and do best effort?
            //       the deactivation can break things further, OTOH so can the
            //       capture
            auto _ = scopedDeactivate( *m_outputRedirect );

            // First notify reporter that bad things happened
            m_reporter->fatalErrorEncountered( message );
        }

        // Don't rebuild the result -- the stringification itself can cause more fatal errors
        // Instead, fake a result data.
        AssertionResultData tempResult( ResultWas::FatalErrorCondition, { false } );
        tempResult.message = static_cast<std::string>(message);
        AssertionResult result( makeDummyAssertionInfo(),
                                CATCH_MOVE( tempResult ) );

        assertionEnded(CATCH_MOVE(result) );


        // At this point we touch sections/test cases from this thread
        // to try and end them. Technically that is not supported when
        // using multiple threads, but the worst thing that can happen
        // is that the process aborts harder :-D
        Detail::LockGuard lock( m_assertionMutex );

        // Best effort cleanup for sections that have not been destructed yet
        // Since this is a fatal error, we have not had and won't have the opportunity to destruct them properly
        while (!m_activeSections.empty()) {
            auto const& nl = m_activeSections.back()->nameAndLocation();
            SectionEndInfo endInfo{ SectionInfo(nl.location, nl.name), {}, 0.0 };
            sectionEndedEarly(CATCH_MOVE(endInfo));
        }
        handleUnfinishedSections();

        // Recreate section for test case (as we will lose the one that was in scope)
        auto const& testCaseInfo = m_activeTestCase->getTestCaseInfo();
        SectionInfo testCaseSection(testCaseInfo.lineInfo, testCaseInfo.name);

        Counts assertions;
        assertions.failed = 1;
        SectionStats testCaseSectionStats(CATCH_MOVE(testCaseSection), assertions, 0, false);
        m_reporter->sectionEnded( testCaseSectionStats );

        auto const& testInfo = m_activeTestCase->getTestCaseInfo();

        Totals deltaTotals;
        deltaTotals.testCases.failed = 1;
        deltaTotals.assertions.failed = 1;
        m_reporter->testCaseEnded(TestCaseStats(testInfo,
                                  deltaTotals,
                                  std::string(),
                                  std::string(),
                                  false));
        m_totals.testCases.failed++;
        updateTotalsFromAtomics();
        m_reporter->testRunEnded(TestRunStats(m_runInfo, m_totals, false));
    }

    bool RunContext::lastAssertionPassed() {
        return Detail::g_lastAssertionPassed;
    }

    void RunContext::assertionPassedFastPath(SourceLineInfo lineInfo) {
        // We want to save the line info for better experience with unexpected assertions
        Detail::g_lastKnownLineInfo = lineInfo;
        ++m_atomicAssertionCount.passed;
        Detail::g_lastAssertionPassed = true;
        Detail::g_clearMessageScopes = true;
    }

    void RunContext::updateTotalsFromAtomics() {
        m_totals.assertions = Counts{
            m_atomicAssertionCount.passed,
            m_atomicAssertionCount.failed,
            m_atomicAssertionCount.failedButOk,
            m_atomicAssertionCount.skipped,
        };
    }

    bool RunContext::aborting() const {
        return m_atomicAssertionCount.failed >= m_abortAfterXFailedAssertions;
    }

    void RunContext::runCurrentTest() {
        auto const& testCaseInfo = m_activeTestCase->getTestCaseInfo();
        SectionInfo testCaseSection(testCaseInfo.lineInfo, testCaseInfo.name);
        m_reporter->sectionStarting(testCaseSection);
        updateTotalsFromAtomics();
        Counts prevAssertions = m_totals.assertions;
        double duration = 0;
        m_shouldReportUnexpected = true;
        Detail::g_lastKnownLineInfo = testCaseInfo.lineInfo;

        Timer timer;
        CATCH_TRY {
            {
                auto _ = scopedActivate( *m_outputRedirect );
                timer.start();
                invokeActiveTestCase();
            }
            duration = timer.getElapsedSeconds();
        } CATCH_CATCH_ANON (TestFailureException&) {
            // This just means the test was aborted due to failure
        } CATCH_CATCH_ANON (TestSkipException&) {
            // This just means the test was explicitly skipped
        } CATCH_CATCH_ALL {
            // Under CATCH_CONFIG_FAST_COMPILE, unexpected exceptions under REQUIRE assertions
            // are reported without translation at the point of origin.
            if ( m_shouldReportUnexpected ) {
                AssertionReaction dummyReaction;
                handleUnexpectedInflightException( makeDummyAssertionInfo(),
                                                   translateActiveException(),
                                                   dummyReaction );
            }
        }
        updateTotalsFromAtomics();
        Counts assertions = m_totals.assertions - prevAssertions;
        bool missingAssertions = testForMissingAssertions(assertions);

        m_testCaseTracker->close();
        handleUnfinishedSections();
        m_messageScopes.clear();
        // TBD: At this point, m_messages should be empty. Do we want to
        //      assert that this is true, or keep the defensive clear call?
        m_messages.clear();

        SectionStats testCaseSectionStats(CATCH_MOVE(testCaseSection), assertions, duration, missingAssertions);
        m_reporter->sectionEnded(testCaseSectionStats);
    }

    void RunContext::invokeActiveTestCase() {
        // We need to engage a handler for signals/structured exceptions
        // before running the tests themselves, or the binary can crash
        // without failed test being reported.
        FatalConditionHandlerGuard _(&m_fatalConditionhandler);
        // We keep having issue where some compilers warn about an unused
        // variable, even though the type has non-trivial constructor and
        // destructor. This is annoying and ugly, but it makes them stfu.
        (void)_;

        m_activeTestCase->invoke();
    }

    void RunContext::handleUnfinishedSections() {
        // If sections ended prematurely due to an exception we stored their
        // infos here so we can tear them down outside the unwind process.
        for ( auto it = m_unfinishedSections.rbegin(),
                   itEnd = m_unfinishedSections.rend();
              it != itEnd;
              ++it ) {
            sectionEnded( CATCH_MOVE( *it ) );
        }
        m_unfinishedSections.clear();
    }

    void RunContext::handleExpr(
        AssertionInfo const& info,
        ITransientExpression const& expr,
        AssertionReaction& reaction
    ) {
        bool negated = isFalseTest( info.resultDisposition );
        bool result = expr.getResult() != negated;

        if( result ) {
            if (!m_includeSuccessfulResults) {
                assertionPassedFastPath(info.lineInfo);
            }
            else {
                reportExpr(info, ResultWas::Ok, &expr, negated);
            }
        }
        else {
            reportExpr(info, ResultWas::ExpressionFailed, &expr, negated );
            populateReaction(
                reaction, info.resultDisposition & ResultDisposition::Normal );
        }
    }
    void RunContext::reportExpr(
            AssertionInfo const &info,
            ResultWas::OfType resultType,
            ITransientExpression const *expr,
            bool negated ) {

        Detail::g_lastKnownLineInfo = info.lineInfo;
        AssertionResultData data( resultType, LazyExpression( negated ) );

        AssertionResult assertionResult{ info, CATCH_MOVE( data ) };
        assertionResult.m_resultData.lazyExpression.m_transientExpression = expr;

        assertionEnded( CATCH_MOVE(assertionResult) );
    }

    void RunContext::handleMessage(
            AssertionInfo const& info,
            ResultWas::OfType resultType,
            std::string&& message,
            AssertionReaction& reaction
    ) {
        Detail::g_lastKnownLineInfo = info.lineInfo;

        AssertionResultData data( resultType, LazyExpression( false ) );
        data.message = CATCH_MOVE( message );
        AssertionResult assertionResult{ info,
                                         CATCH_MOVE( data ) };

        const auto isOk = assertionResult.isOk();
        assertionEnded( CATCH_MOVE(assertionResult) );
        if ( !isOk ) {
            populateReaction(
                reaction, info.resultDisposition & ResultDisposition::Normal );
        } else if ( resultType == ResultWas::ExplicitSkip ) {
            // TODO: Need to handle this explicitly, as ExplicitSkip is
            // considered "OK"
            reaction.shouldSkip = true;
        }
    }

    void RunContext::handleUnexpectedExceptionNotThrown(
            AssertionInfo const& info,
            AssertionReaction& reaction
    ) {
        handleNonExpr(info, Catch::ResultWas::DidntThrowException, reaction);
    }

    void RunContext::handleUnexpectedInflightException(
            AssertionInfo const& info,
            std::string&& message,
            AssertionReaction& reaction
    ) {
        Detail::g_lastKnownLineInfo = info.lineInfo;

        AssertionResultData data( ResultWas::ThrewException, LazyExpression( false ) );
        data.message = CATCH_MOVE(message);
        AssertionResult assertionResult{ info, CATCH_MOVE(data) };
        assertionEnded( CATCH_MOVE(assertionResult) );
        populateReaction( reaction,
                          info.resultDisposition & ResultDisposition::Normal );
    }

    void RunContext::populateReaction( AssertionReaction& reaction,
                                       bool has_normal_disposition ) {
        reaction.shouldDebugBreak = m_shouldDebugBreak;
        reaction.shouldThrow = aborting() || has_normal_disposition;
    }

    AssertionInfo RunContext::makeDummyAssertionInfo() {
        const bool testCaseJustStarted =
            Detail::g_lastKnownLineInfo ==
            m_activeTestCase->getTestCaseInfo().lineInfo;

        return AssertionInfo{
            testCaseJustStarted ? "TEST_CASE"_sr : StringRef(),
            Detail::g_lastKnownLineInfo,
            testCaseJustStarted ? StringRef() : "{Unknown expression after the reported line}"_sr,
            ResultDisposition::Normal
        };
    }

    void RunContext::handleIncomplete(
            AssertionInfo const& info
    ) {
        using namespace std::string_literals;
        Detail::g_lastKnownLineInfo = info.lineInfo;

        AssertionResultData data( ResultWas::ThrewException, LazyExpression( false ) );
        data.message = "Exception translation was disabled by CATCH_CONFIG_FAST_COMPILE"s;
        AssertionResult assertionResult{ info, CATCH_MOVE( data ) };
        assertionEnded( CATCH_MOVE(assertionResult) );
    }

    void RunContext::handleNonExpr(
            AssertionInfo const &info,
            ResultWas::OfType resultType,
            AssertionReaction &reaction
    ) {
        AssertionResultData data( resultType, LazyExpression( false ) );
        AssertionResult assertionResult{ info, CATCH_MOVE( data ) };

        const auto isOk = assertionResult.isOk();
        if ( isOk && !m_includeSuccessfulResults ) {
            assertionPassedFastPath( info.lineInfo );
            return;
        }

        assertionEnded( CATCH_MOVE(assertionResult) );
        if ( !isOk ) {
            populateReaction(
                reaction, info.resultDisposition & ResultDisposition::Normal );
        }
    }

    IResultCapture& getResultCapture() {
        if (auto* capture = getCurrentContext().getResultCapture())
            return *capture;
        else
            CATCH_INTERNAL_ERROR("No result capture instance");
    }

    void seedRng(IConfig const& config) {
        sharedRng().seed(config.rngSeed());
    }

    unsigned int rngSeed() {
        return getCurrentContext().getConfig()->rngSeed();
    }

}



namespace Catch {

    Section::Section( SectionInfo&& info ):
        m_info( CATCH_MOVE( info ) ),
        m_sectionIncluded(
            getResultCapture().sectionStarted( m_info.name, m_info.lineInfo, m_assertions ) ) {
        // Non-"included" sections will not use the timing information
        // anyway, so don't bother with the potential syscall.
        if (m_sectionIncluded) {
            m_timer.start();
        }
    }

    Section::Section( SourceLineInfo const& _lineInfo,
                      StringRef _name,
                      const char* const ):
        m_info( { "invalid", static_cast<std::size_t>( -1 ) }, std::string{} ),
        m_sectionIncluded(
            getResultCapture().sectionStarted( _name, _lineInfo, m_assertions ) ) {
        // We delay initialization the SectionInfo member until we know
        // this section needs it, so we avoid allocating std::string for name.
        // We also delay timer start to avoid the potential syscall unless we
        // will actually use the result.
        if ( m_sectionIncluded ) {
            m_info.name = static_cast<std::string>( _name );
            m_info.lineInfo = _lineInfo;
            m_timer.start();
        }
    }

    Section::~Section() {
        if( m_sectionIncluded ) {
            SectionEndInfo endInfo{ CATCH_MOVE(m_info), m_assertions, m_timer.getElapsedSeconds() };
            if ( uncaught_exceptions() ) {
                getResultCapture().sectionEndedEarly( CATCH_MOVE(endInfo) );
            } else {
                getResultCapture().sectionEnded( CATCH_MOVE( endInfo ) );
            }
        }
    }

    // This indicates whether the section should be executed or not
    Section::operator bool() const {
        return m_sectionIncluded;
    }


} // end namespace Catch



#include <vector>

namespace Catch {

    namespace {
        static auto getSingletons() -> std::vector<ISingleton*>*& {
            static std::vector<ISingleton*>* g_singletons = nullptr;
            if( !g_singletons )
                g_singletons = new std::vector<ISingleton*>();
            return g_singletons;
        }
    }

    ISingleton::~ISingleton() = default;

    void addSingleton(ISingleton* singleton ) {
        getSingletons()->push_back( singleton );
    }
    void cleanupSingletons() {
        auto& singletons = getSingletons();
        for( auto singleton : *singletons )
            delete singleton;
        delete singletons;
        singletons = nullptr;
    }

} // namespace Catch



#include <cstring>
#include <ostream>

namespace Catch {

    bool SourceLineInfo::operator == ( SourceLineInfo const& other ) const noexcept {
        return line == other.line && (file == other.file || std::strcmp(file, other.file) == 0);
    }
    bool SourceLineInfo::operator < ( SourceLineInfo const& other ) const noexcept {
        // We can assume that the same file will usually have the same pointer.
        // Thus, if the pointers are the same, there is no point in calling the strcmp
        return line < other.line || ( line == other.line && file != other.file && (std::strcmp(file, other.file) < 0));
    }

    std::ostream& operator << ( std::ostream& os, SourceLineInfo const& info ) {
#ifndef __GNUG__
        os << info.file << '(' << info.line << ')';
#else
        os << info.file << ':' << info.line;
#endif
        return os;
    }

} // end namespace Catch




namespace Catch {
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
    void StartupExceptionRegistry::add( std::exception_ptr const& exception ) noexcept {
        CATCH_TRY {
            m_exceptions.push_back(exception);
        } CATCH_CATCH_ALL {
            // If we run out of memory during start-up there's really not a lot more we can do about it
            std::terminate();
        }
    }

    std::vector<std::exception_ptr> const& StartupExceptionRegistry::getExceptions() const noexcept {
        return m_exceptions;
    }
#endif

} // end namespace Catch





#include <iostream>

namespace Catch {

// If you #define this you must implement these functions
#if !defined( CATCH_CONFIG_NOSTDOUT )
    std::ostream& cout() { return std::cout; }
    std::ostream& cerr() { return std::cerr; }
    std::ostream& clog() { return std::clog; }
#endif

} // namespace Catch



#include <ostream>
#include <cstring>
#include <cctype>
#include <vector>

namespace Catch {

    bool startsWith( std::string const& s, std::string const& prefix ) {
        return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
    }
    bool startsWith( StringRef s, char prefix ) {
        return !s.empty() && s[0] == prefix;
    }
    bool endsWith( std::string const& s, std::string const& suffix ) {
        return s.size() >= suffix.size() && std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
    }
    bool endsWith( std::string const& s, char suffix ) {
        return !s.empty() && s[s.size()-1] == suffix;
    }
    bool contains( std::string const& s, std::string const& infix ) {
        return s.find( infix ) != std::string::npos;
    }
    void toLowerInPlace( std::string& s ) {
        for ( char& c : s ) {
            c = toLower( c );
        }
    }
    std::string toLower( std::string const& s ) {
        std::string lc = s;
        toLowerInPlace( lc );
        return lc;
    }
    char toLower(char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    std::string trim( std::string const& str ) {
        static char const* whitespaceChars = "\n\r\t ";
        std::string::size_type start = str.find_first_not_of( whitespaceChars );
        std::string::size_type end = str.find_last_not_of( whitespaceChars );

        return start != std::string::npos ? str.substr( start, 1+end-start ) : std::string();
    }

    StringRef trim(StringRef ref) {
        const auto is_ws = [](char c) {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r';
        };
        size_t real_begin = 0;
        while (real_begin < ref.size() && is_ws(ref[real_begin])) { ++real_begin; }
        size_t real_end = ref.size();
        while (real_end > real_begin && is_ws(ref[real_end - 1])) { --real_end; }

        return ref.substr(real_begin, real_end - real_begin);
    }

    bool replaceInPlace( std::string& str, std::string const& replaceThis, std::string const& withThis ) {
        std::size_t i = str.find( replaceThis );
        if (i == std::string::npos) {
            return false;
        }
        std::size_t copyBegin = 0;
        std::string origStr = CATCH_MOVE(str);
        str.clear();
        // There is at least one replacement, so reserve with the best guess
        // we can make without actually counting the number of occurrences.
        str.reserve(origStr.size() - replaceThis.size() + withThis.size());
        do {
            str.append(origStr, copyBegin, i-copyBegin );
            str += withThis;
            copyBegin = i + replaceThis.size();
            if( copyBegin < origStr.size() )
                i = origStr.find( replaceThis, copyBegin );
            else
                i = std::string::npos;
        } while( i != std::string::npos );
        if ( copyBegin < origStr.size() ) {
            str.append(origStr, copyBegin, origStr.size() );
        }
        return true;
    }

    std::vector<StringRef> splitStringRef( StringRef str, char delimiter ) {
        std::vector<StringRef> subStrings;
        std::size_t start = 0;
        for(std::size_t pos = 0; pos < str.size(); ++pos ) {
            if( str[pos] == delimiter ) {
                if( pos - start > 1 )
                    subStrings.push_back( str.substr( start, pos-start ) );
                start = pos+1;
            }
        }
        if( start < str.size() )
            subStrings.push_back( str.substr( start, str.size()-start ) );
        return subStrings;
    }

    std::ostream& operator << ( std::ostream& os, pluralise const& pluraliser ) {
        os << pluraliser.m_count << ' ' << pluraliser.m_label;
        if( pluraliser.m_count != 1 )
            os << 's';
        return os;
    }

}



#include <algorithm>
#include <ostream>
#include <cstring>

namespace Catch {
    StringRef::StringRef( char const* rawChars ) noexcept
    : StringRef( rawChars, std::strlen(rawChars) )
    {}


    bool StringRef::operator<(StringRef rhs) const noexcept {
        if (m_size < rhs.m_size) {
            return strncmp(m_start, rhs.m_start, m_size) <= 0;
        }
        return strncmp(m_start, rhs.m_start, rhs.m_size) < 0;
    }

    int StringRef::compare( StringRef rhs ) const {
        auto cmpResult =
            strncmp( m_start, rhs.m_start, std::min( m_size, rhs.m_size ) );

        // This means that strncmp found a difference before the strings
        // ended, and we can return it directly
        if ( cmpResult != 0 ) {
            return cmpResult;
        }

        // If strings are equal up to length, then their comparison results on
        // their size
        if ( m_size < rhs.m_size ) {
            return -1;
        } else if ( m_size > rhs.m_size ) {
            return 1;
        } else {
            return 0;
        }
    }

    auto operator << ( std::ostream& os, StringRef str ) -> std::ostream& {
        return os.write(str.data(), static_cast<std::streamsize>(str.size()));
    }

    std::string operator+(StringRef lhs, StringRef rhs) {
        std::string ret;
        ret.reserve(lhs.size() + rhs.size());
        ret += lhs;
        ret += rhs;
        return ret;
    }

    auto operator+=( std::string& lhs, StringRef rhs ) -> std::string& {
        lhs.append(rhs.data(), rhs.size());
        return lhs;
    }

} // namespace Catch



namespace Catch {

    TagAliasRegistry::~TagAliasRegistry() = default;

    TagAlias const* TagAliasRegistry::find( std::string const& alias ) const {
        auto it = m_registry.find( alias );
        if( it != m_registry.end() )
            return &(it->second);
        else
            return nullptr;
    }

    std::string TagAliasRegistry::expandAliases( std::string const& unexpandedTestSpec ) const {
        std::string expandedTestSpec = unexpandedTestSpec;
        for( auto const& registryKvp : m_registry ) {
            std::size_t pos = expandedTestSpec.find( registryKvp.first );
            if( pos != std::string::npos ) {
                expandedTestSpec =  expandedTestSpec.substr( 0, pos ) +
                                    registryKvp.second.tag +
                                    expandedTestSpec.substr( pos + registryKvp.first.size() );
            }
        }
        return expandedTestSpec;
    }

    void TagAliasRegistry::add( std::string const& alias, std::string const& tag, SourceLineInfo const& lineInfo ) {
        CATCH_ENFORCE( startsWith(alias, "[@") && endsWith(alias, ']'),
                      "error: tag alias, '" << alias << "' is not of the form [@alias name].\n" << lineInfo );

        CATCH_ENFORCE( m_registry.insert(std::make_pair(alias, TagAlias(tag, lineInfo))).second,
                      "error: tag alias, '" << alias << "' already registered.\n"
                      << "\tFirst seen at: " << find(alias)->lineInfo << "\n"
                      << "\tRedefined at: " << lineInfo );
    }

    ITagAliasRegistry::~ITagAliasRegistry() = default;

    ITagAliasRegistry const& ITagAliasRegistry::get() {
        return getRegistryHub().getTagAliasRegistry();
    }

} // end namespace Catch




namespace Catch {
    TestCaseInfoHasher::TestCaseInfoHasher( hash_t seed ): m_seed( seed ) {}

    uint32_t TestCaseInfoHasher::operator()( TestCaseInfo const& t ) const {
        // FNV-1a hash algorithm that is designed for uniqueness:
        const hash_t prime = 1099511628211u;
        hash_t hash = 14695981039346656037u;
        for ( const char c : t.name ) {
            hash ^= c;
            hash *= prime;
        }
        for ( const char c : t.className ) {
            hash ^= c;
            hash *= prime;
        }
        for ( const Tag& tag : t.tags ) {
            for ( const char c : tag.original ) {
                hash ^= c;
                hash *= prime;
            }
        }
        hash ^= m_seed;
        hash *= prime;
        const uint32_t low{ static_cast<uint32_t>( hash ) };
        const uint32_t high{ static_cast<uint32_t>( hash >> 32 ) };
        return low * high;
    }
} // namespace Catch




#include <algorithm>
#include <set>

namespace Catch {

    namespace {
        static void enforceNoDuplicateTestCases(
            std::vector<TestCaseHandle> const& tests ) {
            auto testInfoCmp = []( TestCaseInfo const* lhs,
                                   TestCaseInfo const* rhs ) {
                return *lhs < *rhs;
            };
            std::set<TestCaseInfo const*, decltype( testInfoCmp )&> seenTests(
                testInfoCmp );
            for ( auto const& test : tests ) {
                const auto infoPtr = &test.getTestCaseInfo();
                const auto prev = seenTests.insert( infoPtr );
                CATCH_ENFORCE( prev.second,
                               "error: test case \""
                                   << infoPtr->name << "\", with tags \""
                                   << infoPtr->tagsAsString()
                                   << "\" already defined.\n"
                                   << "\tFirst seen at "
                                   << ( *prev.first )->lineInfo << "\n"
                                   << "\tRedefined at " << infoPtr->lineInfo );
            }
        }

        static bool matchTest( TestCaseHandle const& testCase,
                               TestSpec const& testSpec,
                               IConfig const& config ) {
            return testSpec.matches( testCase.getTestCaseInfo() ) &&
                   isThrowSafe( testCase, config );
        }

    } // end unnamed namespace

    std::vector<TestCaseHandle> sortTests( IConfig const& config, std::vector<TestCaseHandle> const& unsortedTestCases ) {
        switch (config.runOrder()) {
        case TestRunOrder::Declared:
            return unsortedTestCases;

        case TestRunOrder::LexicographicallySorted: {
            std::vector<TestCaseHandle> sorted = unsortedTestCases;
            std::sort(
                sorted.begin(),
                sorted.end(),
                []( TestCaseHandle const& lhs, TestCaseHandle const& rhs ) {
                    return lhs.getTestCaseInfo() < rhs.getTestCaseInfo();
                }
            );
            return sorted;
        }
        case TestRunOrder::Randomized: {
            using TestWithHash = std::pair<TestCaseInfoHasher::hash_t, TestCaseHandle>;

            TestCaseInfoHasher h{ config.rngSeed() };
            std::vector<TestWithHash> indexed_tests;
            indexed_tests.reserve(unsortedTestCases.size());

            for (auto const& handle : unsortedTestCases) {
                indexed_tests.emplace_back(h(handle.getTestCaseInfo()), handle);
            }

            std::sort( indexed_tests.begin(),
                       indexed_tests.end(),
                       []( TestWithHash const& lhs, TestWithHash const& rhs ) {
                           if ( lhs.first == rhs.first ) {
                               return lhs.second.getTestCaseInfo() <
                                      rhs.second.getTestCaseInfo();
                           }
                           return lhs.first < rhs.first;
                       } );

            std::vector<TestCaseHandle> randomized;
            randomized.reserve(indexed_tests.size());

            for (auto const& indexed : indexed_tests) {
                randomized.push_back(indexed.second);
            }

            return randomized;
        }
        }

        CATCH_INTERNAL_ERROR("Unknown test order value!");
    }

    bool isThrowSafe( TestCaseHandle const& testCase, IConfig const& config ) {
        return !testCase.getTestCaseInfo().throws() || config.allowThrows();
    }

    std::vector<TestCaseHandle> filterTests( std::vector<TestCaseHandle> const& testCases, TestSpec const& testSpec, IConfig const& config ) {
        std::vector<TestCaseHandle> filtered;
        filtered.reserve( testCases.size() );
        for (auto const& testCase : testCases) {
            if ((!testSpec.hasFilters() && !testCase.getTestCaseInfo().isHidden()) ||
                (testSpec.hasFilters() && matchTest(testCase, testSpec, config))) {
                filtered.push_back(testCase);
            }
        }
        return createShard(filtered, config.shardCount(), config.shardIndex());
    }
    std::vector<TestCaseHandle> const& getAllTestCasesSorted( IConfig const& config ) {
        return getRegistryHub().getTestCaseRegistry().getAllTestsSorted( config );
    }

    TestRegistry::~TestRegistry() = default;

    void TestRegistry::registerTest(Detail::unique_ptr<TestCaseInfo> testInfo, Detail::unique_ptr<ITestInvoker> testInvoker) {
        m_handles.emplace_back(testInfo.get(), testInvoker.get());
        m_viewed_test_infos.push_back(testInfo.get());
        m_owned_test_infos.push_back(CATCH_MOVE(testInfo));
        m_invokers.push_back(CATCH_MOVE(testInvoker));
    }

    std::vector<TestCaseInfo*> const& TestRegistry::getAllInfos() const {
        return m_viewed_test_infos;
    }

    std::vector<TestCaseHandle> const& TestRegistry::getAllTests() const {
        return m_handles;
    }
    std::vector<TestCaseHandle> const& TestRegistry::getAllTestsSorted( IConfig const& config ) const {
        if( m_sortedFunctions.empty() )
            enforceNoDuplicateTestCases( m_handles );

        if(  m_currentSortOrder != config.runOrder() || m_sortedFunctions.empty() ) {
            m_sortedFunctions = sortTests( config, m_handles );
            m_currentSortOrder = config.runOrder();
        }
        return m_sortedFunctions;
    }

} // end namespace Catch




#include <algorithm>
#include <cassert>

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif

namespace Catch {
namespace TestCaseTracking {

    NameAndLocation::NameAndLocation( std::string&& _name, SourceLineInfo const& _location )
    :   name( CATCH_MOVE(_name) ),
        location( _location )
    {}


    ITracker::~ITracker() = default;

    void ITracker::markAsNeedingAnotherRun() {
        m_runState = NeedsAnotherRun;
    }

    void ITracker::addChild( ITrackerPtr&& child ) {
        m_children.push_back( CATCH_MOVE(child) );
    }

    ITracker* ITracker::findChild( NameAndLocationRef const& nameAndLocation ) {
        auto it = std::find_if(
            m_children.begin(),
            m_children.end(),
            [&nameAndLocation]( ITrackerPtr const& tracker ) {
                auto const& tnameAndLoc = tracker->nameAndLocation();
                if ( tnameAndLoc.location.line !=
                     nameAndLocation.location.line ) {
                    return false;
                }
                return tnameAndLoc == nameAndLocation;
            } );
        return ( it != m_children.end() ) ? it->get() : nullptr;
    }

    bool ITracker::isSectionTracker() const { return false; }
    bool ITracker::isGeneratorTracker() const { return false; }

    bool ITracker::isOpen() const {
        return m_runState != NotStarted && !isComplete();
    }

    bool ITracker::hasStarted() const { return m_runState != NotStarted; }

    void ITracker::openChild() {
        if (m_runState != ExecutingChildren) {
            m_runState = ExecutingChildren;
            if (m_parent) {
                m_parent->openChild();
            }
        }
    }

    ITracker& TrackerContext::startRun() {
        using namespace std::string_literals;
        m_rootTracker = Catch::Detail::make_unique<SectionTracker>(
            NameAndLocation( "{root}"s, CATCH_INTERNAL_LINEINFO ),
            *this,
            nullptr );
        m_currentTracker = nullptr;
        m_runState = Executing;
        return *m_rootTracker;
    }

    void TrackerContext::completeCycle() {
        m_runState = CompletedCycle;
    }

    bool TrackerContext::completedCycle() const {
        return m_runState == CompletedCycle;
    }
    void TrackerContext::setCurrentTracker( ITracker* tracker ) {
        m_currentTracker = tracker;
    }


    TrackerBase::TrackerBase( NameAndLocation&& nameAndLocation, TrackerContext& ctx, ITracker* parent ):
        ITracker(CATCH_MOVE(nameAndLocation), parent),
        m_ctx( ctx )
    {}

    bool TrackerBase::isComplete() const {
        return m_runState == CompletedSuccessfully || m_runState == Failed;
    }

    void TrackerBase::open() {
        m_runState = Executing;
        moveToThis();
        if( m_parent )
            m_parent->openChild();
    }

    void TrackerBase::close() {

        // Close any still open children (e.g. generators)
        while( &m_ctx.currentTracker() != this )
            m_ctx.currentTracker().close();

        switch( m_runState ) {
            case NeedsAnotherRun:
                break;

            case Executing:
                m_runState = CompletedSuccessfully;
                break;
            case ExecutingChildren:
                if( std::all_of(m_children.begin(), m_children.end(), [](ITrackerPtr const& t){ return t->isComplete(); }) )
                    m_runState = CompletedSuccessfully;
                break;

            case NotStarted:
            case CompletedSuccessfully:
            case Failed:
                CATCH_INTERNAL_ERROR( "Illogical state: " << m_runState );

            default:
                CATCH_INTERNAL_ERROR( "Unknown state: " << m_runState );
        }
        moveToParent();
        m_ctx.completeCycle();
    }
    void TrackerBase::fail() {
        m_runState = Failed;
        if( m_parent )
            m_parent->markAsNeedingAnotherRun();
        moveToParent();
        m_ctx.completeCycle();
    }

    void TrackerBase::moveToParent() {
        assert( m_parent );
        m_ctx.setCurrentTracker( m_parent );
    }
    void TrackerBase::moveToThis() {
        m_ctx.setCurrentTracker( this );
    }

    SectionTracker::SectionTracker( NameAndLocation&& nameAndLocation, TrackerContext& ctx, ITracker* parent )
    :   TrackerBase( CATCH_MOVE(nameAndLocation), ctx, parent ),
        m_trimmed_name(trim(StringRef(ITracker::nameAndLocation().name)))
    {
        if( parent ) {
            while ( !parent->isSectionTracker() ) {
                parent = parent->parent();
            }

            SectionTracker& parentSection = static_cast<SectionTracker&>( *parent );
            addNextFilters( parentSection.m_filters );
        }
    }

    bool SectionTracker::isComplete() const {
        bool complete = true;

        if (m_filters.empty()
            || m_filters[0].empty()
            || std::find(m_filters.begin(), m_filters.end(), m_trimmed_name) != m_filters.end()) {
            complete = TrackerBase::isComplete();
        }
        return complete;
    }

    bool SectionTracker::isSectionTracker() const { return true; }

    SectionTracker& SectionTracker::acquire( TrackerContext& ctx, NameAndLocationRef const& nameAndLocation ) {
        SectionTracker* tracker;

        ITracker& currentTracker = ctx.currentTracker();
        if ( ITracker* childTracker =
                 currentTracker.findChild( nameAndLocation ) ) {
            assert( childTracker );
            assert( childTracker->isSectionTracker() );
            tracker = static_cast<SectionTracker*>( childTracker );
        } else {
            auto newTracker = Catch::Detail::make_unique<SectionTracker>(
                NameAndLocation{ static_cast<std::string>(nameAndLocation.name),
                                 nameAndLocation.location },
                ctx,
                &currentTracker );
            tracker = newTracker.get();
            currentTracker.addChild( CATCH_MOVE( newTracker ) );
        }

        if ( !ctx.completedCycle() ) {
            tracker->tryOpen();
        }

        return *tracker;
    }

    void SectionTracker::tryOpen() {
        if( !isComplete() )
            open();
    }

    void SectionTracker::addInitialFilters( std::vector<std::string> const& filters ) {
        if( !filters.empty() ) {
            m_filters.reserve( m_filters.size() + filters.size() + 2 );
            m_filters.emplace_back(StringRef{}); // Root - should never be consulted
            m_filters.emplace_back(StringRef{}); // Test Case - not a section filter
            m_filters.insert( m_filters.end(), filters.begin(), filters.end() );
        }
    }
    void SectionTracker::addNextFilters( std::vector<StringRef> const& filters ) {
        if( filters.size() > 1 )
            m_filters.insert( m_filters.end(), filters.begin()+1, filters.end() );
    }

    StringRef SectionTracker::trimmedName() const {
        return m_trimmed_name;
    }

} // namespace TestCaseTracking

} // namespace Catch

#if defined(__clang__)
#    pragma clang diagnostic pop
#endif




namespace Catch {

    void throw_test_failure_exception() {
#if !defined( CATCH_CONFIG_DISABLE_EXCEPTIONS )
        throw TestFailureException{};
#else
        CATCH_ERROR( "Test failure requires aborting test!" );
#endif
    }

    void throw_test_skip_exception() {
#if !defined( CATCH_CONFIG_DISABLE_EXCEPTIONS )
        throw Catch::TestSkipException();
#else
        CATCH_ERROR( "Explicitly skipping tests during runtime requires exceptions" );
#endif
    }

} // namespace Catch



#include <algorithm>
#include <iterator>

namespace Catch {
    void ITestInvoker::prepareTestCase() {}
    void ITestInvoker::tearDownTestCase() {}
    ITestInvoker::~ITestInvoker() = default;

    namespace {
        static StringRef extractClassName( StringRef classOrMethodName ) {
            if ( !startsWith( classOrMethodName, '&' ) ) {
                return classOrMethodName;
            }

            // Remove the leading '&' to avoid having to special case it later
            const auto methodName =
                classOrMethodName.substr( 1, classOrMethodName.size() );

            auto reverseStart = std::make_reverse_iterator( methodName.end() );
            auto reverseEnd = std::make_reverse_iterator( methodName.begin() );

            // We make a simplifying assumption that ":" is only present
            // in the input as part of "::" from C++ typenames (this is
            // relatively safe assumption because the input is generated
            // as stringification of type through preprocessor).
            auto lastColons = std::find( reverseStart, reverseEnd, ':' ) + 1;
            auto secondLastColons =
                std::find( lastColons + 1, reverseEnd, ':' );

            auto const startIdx = reverseEnd - secondLastColons;
            auto const classNameSize = secondLastColons - lastColons - 1;

            return methodName.substr(
                static_cast<std::size_t>( startIdx ),
                static_cast<std::size_t>( classNameSize ) );
        }

        class TestInvokerAsFunction final : public ITestInvoker {
            using TestType = void ( * )();
            TestType m_testAsFunction;

        public:
            constexpr TestInvokerAsFunction( TestType testAsFunction ) noexcept:
                m_testAsFunction( testAsFunction ) {}

            void invoke() const override { m_testAsFunction(); }
        };

    } // namespace

    Detail::unique_ptr<ITestInvoker> makeTestInvoker( void(*testAsFunction)() ) {
        return Detail::make_unique<TestInvokerAsFunction>( testAsFunction );
    }

    AutoReg::AutoReg( Detail::unique_ptr<ITestInvoker> invoker, SourceLineInfo const& lineInfo, StringRef classOrMethod, NameAndTags const& nameAndTags ) noexcept {
        CATCH_TRY {
            getMutableRegistryHub()
                    .registerTest(
                        makeTestCaseInfo(
                            extractClassName( classOrMethod ),
                            nameAndTags,
                            lineInfo),
                        CATCH_MOVE(invoker)
                    );
        } CATCH_CATCH_ALL {
            // Do not throw when constructing global objects, instead register the exception to be processed later
            getMutableRegistryHub().registerStartupException();
        }
    }
}





namespace Catch {

    TestSpecParser::TestSpecParser( ITagAliasRegistry const& tagAliases ) : m_tagAliases( &tagAliases ) {}

    TestSpecParser& TestSpecParser::parse( std::string const& arg ) {
        m_mode = None;
        m_exclusion = false;
        m_arg = m_tagAliases->expandAliases( arg );
        m_escapeChars.clear();
        m_substring.reserve(m_arg.size());
        m_patternName.reserve(m_arg.size());
        m_realPatternPos = 0;

        for( m_pos = 0; m_pos < m_arg.size(); ++m_pos )
          //if visitChar fails
           if( !visitChar( m_arg[m_pos] ) ){
               m_testSpec.m_invalidSpecs.push_back(arg);
               break;
           }
        endMode();
        return *this;
    }
    TestSpec TestSpecParser::testSpec() {
        addFilter();
        return CATCH_MOVE(m_testSpec);
    }
    bool TestSpecParser::visitChar( char c ) {
        if( (m_mode != EscapedName) && (c == '\\') ) {
            escape();
            addCharToPattern(c);
            return true;
        }else if((m_mode != EscapedName) && (c == ',') )  {
            return separate();
        }

        switch( m_mode ) {
        case None:
            if( processNoneChar( c ) )
                return true;
            break;
        case Name:
            processNameChar( c );
            break;
        case EscapedName:
            endMode();
            addCharToPattern(c);
            return true;
        default:
        case Tag:
        case QuotedName:
            if( processOtherChar( c ) )
                return true;
            break;
        }

        m_substring += c;
        if( !isControlChar( c ) ) {
            m_patternName += c;
            m_realPatternPos++;
        }
        return true;
    }
    // Two of the processing methods return true to signal the caller to return
    // without adding the given character to the current pattern strings
    bool TestSpecParser::processNoneChar( char c ) {
        switch( c ) {
        case ' ':
            return true;
        case '~':
            m_exclusion = true;
            return false;
        case '[':
            startNewMode( Tag );
            return false;
        case '"':
            startNewMode( QuotedName );
            return false;
        default:
            startNewMode( Name );
            return false;
        }
    }
    void TestSpecParser::processNameChar( char c ) {
        if( c == '[' ) {
            if( m_substring == "exclude:" )
                m_exclusion = true;
            else
                endMode();
            startNewMode( Tag );
        }
    }
    bool TestSpecParser::processOtherChar( char c ) {
        if( !isControlChar( c ) )
            return false;
        m_substring += c;
        endMode();
        return true;
    }
    void TestSpecParser::startNewMode( Mode mode ) {
        m_mode = mode;
    }
    void TestSpecParser::endMode() {
        switch( m_mode ) {
        case Name:
        case QuotedName:
            return addNamePattern();
        case Tag:
            return addTagPattern();
        case EscapedName:
            revertBackToLastMode();
            return;
        case None:
        default:
            return startNewMode( None );
        }
    }
    void TestSpecParser::escape() {
        saveLastMode();
        m_mode = EscapedName;
        m_escapeChars.push_back(m_realPatternPos);
    }
    bool TestSpecParser::isControlChar( char c ) const {
        switch( m_mode ) {
            default:
                return false;
            case None:
                return c == '~';
            case Name:
                return c == '[';
            case EscapedName:
                return true;
            case QuotedName:
                return c == '"';
            case Tag:
                return c == '[' || c == ']';
        }
    }

    void TestSpecParser::addFilter() {
        if( !m_currentFilter.m_required.empty() || !m_currentFilter.m_forbidden.empty() ) {
            m_testSpec.m_filters.push_back( CATCH_MOVE(m_currentFilter) );
            m_currentFilter = TestSpec::Filter();
        }
    }

    void TestSpecParser::saveLastMode() {
      lastMode = m_mode;
    }

    void TestSpecParser::revertBackToLastMode() {
      m_mode = lastMode;
    }

    bool TestSpecParser::separate() {
      if( (m_mode==QuotedName) || (m_mode==Tag) ){
         //invalid argument, signal failure to previous scope.
         m_mode = None;
         m_pos = m_arg.size();
         m_substring.clear();
         m_patternName.clear();
         m_realPatternPos = 0;
         return false;
      }
      endMode();
      addFilter();
      return true; //success
    }

    std::string TestSpecParser::preprocessPattern() {
        std::string token = m_patternName;
        for (std::size_t i = 0; i < m_escapeChars.size(); ++i)
            token = token.substr(0, m_escapeChars[i] - i) + token.substr(m_escapeChars[i] - i + 1);
        m_escapeChars.clear();
        if (startsWith(token, "exclude:")) {
            m_exclusion = true;
            token = token.substr(8);
        }

        m_patternName.clear();
        m_realPatternPos = 0;

        return token;
    }

    void TestSpecParser::addNamePattern() {
        auto token = preprocessPattern();

        if (!token.empty()) {
            if (m_exclusion) {
                m_currentFilter.m_forbidden.emplace_back(Detail::make_unique<TestSpec::NamePattern>(token, m_substring));
            } else {
                m_currentFilter.m_required.emplace_back(Detail::make_unique<TestSpec::NamePattern>(token, m_substring));
            }
        }
        m_substring.clear();
        m_exclusion = false;
        m_mode = None;
    }

    void TestSpecParser::addTagPattern() {
        auto token = preprocessPattern();

        if (!token.empty()) {
            // If the tag pattern is the "hide and tag" shorthand (e.g. [.foo])
            // we have to create a separate hide tag and shorten the real one
            if (token.size() > 1 && token[0] == '.') {
                token.erase(token.begin());
                if (m_exclusion) {
                    m_currentFilter.m_forbidden.emplace_back(Detail::make_unique<TestSpec::TagPattern>(".", m_substring));
                } else {
                    m_currentFilter.m_required.emplace_back(Detail::make_unique<TestSpec::TagPattern>(".", m_substring));
                }
            }
            if (m_exclusion) {
                m_currentFilter.m_forbidden.emplace_back(Detail::make_unique<TestSpec::TagPattern>(token, m_substring));
            } else {
                m_currentFilter.m_required.emplace_back(Detail::make_unique<TestSpec::TagPattern>(token, m_substring));
            }
        }
        m_substring.clear();
        m_exclusion = false;
        m_mode = None;
    }

} // namespace Catch



#include <algorithm>
#include <cstring>
#include <ostream>

namespace {
    bool isWhitespace( char c ) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }

    bool isBreakableBefore( char c ) {
        static const char chars[] = "[({<|";
        return std::memchr( chars, c, sizeof( chars ) - 1 ) != nullptr;
    }

    bool isBreakableAfter( char c ) {
        static const char chars[] = "])}>.,:;*+-=&/\\";
        return std::memchr( chars, c, sizeof( chars ) - 1 ) != nullptr;
    }

} // namespace

namespace Catch {
    namespace TextFlow {
        void AnsiSkippingString::preprocessString() {
            for ( auto it = m_string.begin(); it != m_string.end(); ) {
                // try to read through an ansi sequence
                while ( it != m_string.end() && *it == '\033' &&
                        it + 1 != m_string.end() && *( it + 1 ) == '[' ) {
                    auto cursor = it + 2;
                    while ( cursor != m_string.end() &&
                            ( isdigit( *cursor ) || *cursor == ';' ) ) {
                        ++cursor;
                    }
                    if ( cursor == m_string.end() || *cursor != 'm' ) {
                        break;
                    }
                    // 'm' -> 0xff
                    *cursor = AnsiSkippingString::sentinel;
                    // if we've read an ansi sequence, set the iterator and
                    // return to the top of the loop
                    it = cursor + 1;
                }
                if ( it != m_string.end() ) {
                    ++m_size;
                    ++it;
                }
            }
        }

        AnsiSkippingString::AnsiSkippingString( std::string const& text ):
            m_string( text ) {
            preprocessString();
        }

        AnsiSkippingString::AnsiSkippingString( std::string&& text ):
            m_string( CATCH_MOVE( text ) ) {
            preprocessString();
        }

        AnsiSkippingString::const_iterator AnsiSkippingString::begin() const {
            return const_iterator( m_string );
        }

        AnsiSkippingString::const_iterator AnsiSkippingString::end() const {
            return const_iterator( m_string, const_iterator::EndTag{} );
        }

        std::string AnsiSkippingString::substring( const_iterator begin,
                                                   const_iterator end ) const {
            // There's one caveat here to an otherwise simple substring: when
            // making a begin iterator we might have skipped ansi sequences at
            // the start. If `begin` here is a begin iterator, skipped over
            // initial ansi sequences, we'll use the true beginning of the
            // string. Lastly: We need to transform any chars we replaced with
            // 0xff back to 'm'
            auto str = std::string( begin == this->begin() ? m_string.begin()
                                                           : begin.m_it,
                                    end.m_it );
            std::transform( str.begin(), str.end(), str.begin(), []( char c ) {
                return c == AnsiSkippingString::sentinel ? 'm' : c;
            } );
            return str;
        }

        void AnsiSkippingString::const_iterator::tryParseAnsiEscapes() {
            // check if we've landed on an ansi sequence, and if so read through
            // it
            while ( m_it != m_string->end() && *m_it == '\033' &&
                    m_it + 1 != m_string->end() &&  *( m_it + 1 ) == '[' ) {
                auto cursor = m_it + 2;
                while ( cursor != m_string->end() &&
                        ( isdigit( *cursor ) || *cursor == ';' ) ) {
                    ++cursor;
                }
                if ( cursor == m_string->end() ||
                     *cursor != AnsiSkippingString::sentinel ) {
                    break;
                }
                // if we've read an ansi sequence, set the iterator and
                // return to the top of the loop
                m_it = cursor + 1;
            }
        }

        void AnsiSkippingString::const_iterator::advance() {
            assert( m_it != m_string->end() );
            m_it++;
            tryParseAnsiEscapes();
        }

        void AnsiSkippingString::const_iterator::unadvance() {
            assert( m_it != m_string->begin() );
            m_it--;
            // if *m_it is 0xff, scan back to the \033 and then m_it-- once more
            // (and repeat check)
            while ( *m_it == AnsiSkippingString::sentinel ) {
                while ( *m_it != '\033' ) {
                    assert( m_it != m_string->begin() );
                    m_it--;
                }
                // if this happens, we must have been a begin iterator that had
                // skipped over ansi sequences at the start of a string
                assert( m_it != m_string->begin() );
                assert( *m_it == '\033' );
                m_it--;
            }
        }

        static bool isBoundary( AnsiSkippingString const& line,
                                AnsiSkippingString::const_iterator it ) {
            return it == line.end() ||
                   ( isWhitespace( *it ) &&
                     !isWhitespace( *it.oneBefore() ) ) ||
                   isBreakableBefore( *it ) ||
                   isBreakableAfter( *it.oneBefore() );
        }

        void Column::const_iterator::calcLength() {
            m_addHyphen = false;
            m_parsedTo = m_lineStart;
            AnsiSkippingString const& current_line = m_column.m_string;

            if ( m_parsedTo == current_line.end() ) {
                m_lineEnd = m_parsedTo;
                return;
            }

            assert( m_lineStart != current_line.end() );
            if ( *m_lineStart == '\n' ) { ++m_parsedTo; }

            const auto maxLineLength = m_column.m_width - indentSize();
            std::size_t lineLength = 0;
            while ( m_parsedTo != current_line.end() &&
                    lineLength < maxLineLength && *m_parsedTo != '\n' ) {
                ++m_parsedTo;
                ++lineLength;
            }

            // If we encountered a newline before the column is filled,
            // then we linebreak at the newline and consider this line
            // finished.
            if ( lineLength < maxLineLength ) {
                m_lineEnd = m_parsedTo;
            } else {
                // Look for a natural linebreak boundary in the column
                // (We look from the end, so that the first found boundary is
                // the right one)
                m_lineEnd = m_parsedTo;
                while ( lineLength > 0 &&
                        !isBoundary( current_line, m_lineEnd ) ) {
                    --lineLength;
                    --m_lineEnd;
                }
                while ( lineLength > 0 &&
                        isWhitespace( *m_lineEnd.oneBefore() ) ) {
                    --lineLength;
                    --m_lineEnd;
                }

                // If we found one, then that is where we linebreak, otherwise
                // we have to split text with a hyphen
                if ( lineLength == 0 ) {
                    m_addHyphen = true;
                    m_lineEnd = m_parsedTo.oneBefore();
                }
            }
        }

        size_t Column::const_iterator::indentSize() const {
            auto initial = m_lineStart == m_column.m_string.begin()
                               ? m_column.m_initialIndent
                               : std::string::npos;
            return initial == std::string::npos ? m_column.m_indent : initial;
        }

        std::string Column::const_iterator::addIndentAndSuffix(
            AnsiSkippingString::const_iterator start,
            AnsiSkippingString::const_iterator end ) const {
            std::string ret;
            const auto desired_indent = indentSize();
            // ret.reserve( desired_indent + (end - start) + m_addHyphen );
            ret.append( desired_indent, ' ' );
            // ret.append( start, end );
            ret += m_column.m_string.substring( start, end );
            if ( m_addHyphen ) { ret.push_back( '-' ); }

            return ret;
        }

        Column::const_iterator::const_iterator( Column const& column ):
            m_column( column ),
            m_lineStart( column.m_string.begin() ),
            m_lineEnd( column.m_string.begin() ),
            m_parsedTo( column.m_string.begin() ) {
            assert( m_column.m_width > m_column.m_indent );
            assert( m_column.m_initialIndent == std::string::npos ||
                    m_column.m_width > m_column.m_initialIndent );
            calcLength();
            if ( m_lineStart == m_lineEnd ) {
                m_lineStart = m_column.m_string.end();
            }
        }

        std::string Column::const_iterator::operator*() const {
            assert( m_lineStart <= m_parsedTo );
            return addIndentAndSuffix( m_lineStart, m_lineEnd );
        }

        Column::const_iterator& Column::const_iterator::operator++() {
            m_lineStart = m_lineEnd;
            AnsiSkippingString const& current_line = m_column.m_string;
            if ( m_lineStart != current_line.end() && *m_lineStart == '\n' ) {
                m_lineStart++;
            } else {
                while ( m_lineStart != current_line.end() &&
                        isWhitespace( *m_lineStart ) ) {
                    ++m_lineStart;
                }
            }

            if ( m_lineStart != current_line.end() ) { calcLength(); }
            return *this;
        }

        Column::const_iterator Column::const_iterator::operator++( int ) {
            const_iterator prev( *this );
            operator++();
            return prev;
        }

        std::ostream& operator<<( std::ostream& os, Column const& col ) {
            bool first = true;
            for ( auto line : col ) {
                if ( first ) {
                    first = false;
                } else {
                    os << '\n';
                }
                os << line;
            }
            return os;
        }

        Column Spacer( size_t spaceWidth ) {
            Column ret{ "" };
            ret.width( spaceWidth );
            return ret;
        }

        Columns::iterator::iterator( Columns const& columns, EndTag ):
            m_columns( columns.m_columns ), m_activeIterators( 0 ) {

            m_iterators.reserve( m_columns.size() );
            for ( auto const& col : m_columns ) {
                m_iterators.push_back( col.end() );
            }
        }

        Columns::iterator::iterator( Columns const& columns ):
            m_columns( columns.m_columns ),
            m_activeIterators( m_columns.size() ) {

            m_iterators.reserve( m_columns.size() );
            for ( auto const& col : m_columns ) {
                m_iterators.push_back( col.begin() );
            }
        }

        std::string Columns::iterator::operator*() const {
            std::string row, padding;

            for ( size_t i = 0; i < m_columns.size(); ++i ) {
                const auto width = m_columns[i].width();
                if ( m_iterators[i] != m_columns[i].end() ) {
                    std::string col = *m_iterators[i];
                    row += padding;
                    row += col;

                    padding.clear();
                    if ( col.size() < width ) {
                        padding.append( width - col.size(), ' ' );
                    }
                } else {
                    padding.append( width, ' ' );
                }
            }
            return row;
        }

        Columns::iterator& Columns::iterator::operator++() {
            for ( size_t i = 0; i < m_columns.size(); ++i ) {
                if ( m_iterators[i] != m_columns[i].end() ) {
                    ++m_iterators[i];
                }
            }
            return *this;
        }

        Columns::iterator Columns::iterator::operator++( int ) {
            iterator prev( *this );
            operator++();
            return prev;
        }

        std::ostream& operator<<( std::ostream& os, Columns const& cols ) {
            bool first = true;
            for ( auto line : cols ) {
                if ( first ) {
                    first = false;
                } else {
                    os << '\n';
                }
                os << line;
            }
            return os;
        }

        Columns operator+( Column const& lhs, Column const& rhs ) {
            Columns cols;
            cols += lhs;
            cols += rhs;
            return cols;
        }
        Columns operator+( Column&& lhs, Column&& rhs ) {
            Columns cols;
            cols += CATCH_MOVE( lhs );
            cols += CATCH_MOVE( rhs );
            return cols;
        }

        Columns& operator+=( Columns& lhs, Column const& rhs ) {
            lhs.m_columns.push_back( rhs );
            return lhs;
        }
        Columns& operator+=( Columns& lhs, Column&& rhs ) {
            lhs.m_columns.push_back( CATCH_MOVE( rhs ) );
            return lhs;
        }
        Columns operator+( Columns const& lhs, Column const& rhs ) {
            auto combined( lhs );
            combined += rhs;
            return combined;
        }
        Columns operator+( Columns&& lhs, Column&& rhs ) {
            lhs += CATCH_MOVE( rhs );
            return CATCH_MOVE( lhs );
        }

    } // namespace TextFlow
} // namespace Catch




#include <exception>

namespace Catch {
    bool uncaught_exceptions() {
#if defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
        return false;
#elif defined(CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS)
        return std::uncaught_exceptions() > 0;
#else
        return std::uncaught_exception();
#endif
  }
} // end namespace Catch



namespace Catch {

    WildcardPattern::WildcardPattern( std::string const& pattern,
                                      CaseSensitive caseSensitivity )
    :   m_caseSensitivity( caseSensitivity ),
        m_pattern( normaliseString( pattern ) )
    {
        if( startsWith( m_pattern, '*' ) ) {
            m_pattern = m_pattern.substr( 1 );
            m_wildcard = WildcardAtStart;
        }
        if( endsWith( m_pattern, '*' ) ) {
            m_pattern = m_pattern.substr( 0, m_pattern.size()-1 );
            m_wildcard = static_cast<WildcardPosition>( m_wildcard | WildcardAtEnd );
        }
    }

    bool WildcardPattern::matches( std::string const& str ) const {
        switch( m_wildcard ) {
            case NoWildcard:
                return m_pattern == normaliseString( str );
            case WildcardAtStart:
                return endsWith( normaliseString( str ), m_pattern );
            case WildcardAtEnd:
                return startsWith( normaliseString( str ), m_pattern );
            case WildcardAtBothEnds:
                return contains( normaliseString( str ), m_pattern );
            default:
                CATCH_INTERNAL_ERROR( "Unknown enum" );
        }
    }

    std::string WildcardPattern::normaliseString( std::string const& str ) const {
        return trim( m_caseSensitivity == CaseSensitive::No ? toLower( str ) : str );
    }
}


// Note: swapping these two includes around causes MSVC to error out
//       while in /permissive- mode. No, I don't know why.
//       Tested on VS 2019, 18.{3, 4}.x

#include <cstdint>
#include <iomanip>
#include <type_traits>

namespace Catch {

namespace {

    size_t trailingBytes(unsigned char c) {
        if ((c & 0xE0) == 0xC0) {
            return 2;
        }
        if ((c & 0xF0) == 0xE0) {
            return 3;
        }
        if ((c & 0xF8) == 0xF0) {
            return 4;
        }
        CATCH_INTERNAL_ERROR("Invalid multibyte utf-8 start byte encountered");
    }

    uint32_t headerValue(unsigned char c) {
        if ((c & 0xE0) == 0xC0) {
            return c & 0x1F;
        }
        if ((c & 0xF0) == 0xE0) {
            return c & 0x0F;
        }
        if ((c & 0xF8) == 0xF0) {
            return c & 0x07;
        }
        CATCH_INTERNAL_ERROR("Invalid multibyte utf-8 start byte encountered");
    }

    void hexEscapeChar(std::ostream& os, unsigned char c) {
        std::ios_base::fmtflags f(os.flags());
        os << "\\x"
            << std::uppercase << std::hex << std::setfill('0') << std::setw(2)
            << static_cast<int>(c);
        os.flags(f);
    }

    constexpr bool shouldNewline(XmlFormatting fmt) {
        return !!(static_cast<std::underlying_type_t<XmlFormatting>>(fmt & XmlFormatting::Newline));
    }

    constexpr bool shouldIndent(XmlFormatting fmt) {
        return !!(static_cast<std::underlying_type_t<XmlFormatting>>(fmt & XmlFormatting::Indent));
    }

} // anonymous namespace

    void XmlEncode::encodeTo( std::ostream& os ) const {
        // Apostrophe escaping not necessary if we always use " to write attributes
        // (see: http://www.w3.org/TR/xml/#syntax)

        for( std::size_t idx = 0; idx < m_str.size(); ++ idx ) {
            unsigned char c = static_cast<unsigned char>(m_str[idx]);
            switch (c) {
            case '<':   os << "&lt;"; break;
            case '&':   os << "&amp;"; break;

            case '>':
                // See: http://www.w3.org/TR/xml/#syntax
                if (idx > 2 && m_str[idx - 1] == ']' && m_str[idx - 2] == ']')
                    os << "&gt;";
                else
                    os << c;
                break;

            case '\"':
                if (m_forWhat == ForAttributes)
                    os << "&quot;";
                else
                    os << c;
                break;

            default:
                // Check for control characters and invalid utf-8

                // Escape control characters in standard ascii
                // see http://stackoverflow.com/questions/404107/why-are-control-characters-illegal-in-xml-1-0
                if (c < 0x09 || (c > 0x0D && c < 0x20) || c == 0x7F) {
                    hexEscapeChar(os, c);
                    break;
                }

                // Plain ASCII: Write it to stream
                if (c < 0x7F) {
                    os << c;
                    break;
                }

                // UTF-8 territory
                // Check if the encoding is valid and if it is not, hex escape bytes.
                // Important: We do not check the exact decoded values for validity, only the encoding format
                // First check that this bytes is a valid lead byte:
                // This means that it is not encoded as 1111 1XXX
                // Or as 10XX XXXX
                if (c <  0xC0 ||
                    c >= 0xF8) {
                    hexEscapeChar(os, c);
                    break;
                }

                auto encBytes = trailingBytes(c);
                // Are there enough bytes left to avoid accessing out-of-bounds memory?
                if (idx + encBytes - 1 >= m_str.size()) {
                    hexEscapeChar(os, c);
                    break;
                }
                // The header is valid, check data
                // The next encBytes bytes must together be a valid utf-8
                // This means: bitpattern 10XX XXXX and the extracted value is sane (ish)
                bool valid = true;
                uint32_t value = headerValue(c);
                for (std::size_t n = 1; n < encBytes; ++n) {
                    unsigned char nc = static_cast<unsigned char>(m_str[idx + n]);
                    valid &= ((nc & 0xC0) == 0x80);
                    value = (value << 6) | (nc & 0x3F);
                }

                if (
                    // Wrong bit pattern of following bytes
                    (!valid) ||
                    // Overlong encodings
                    (value < 0x80) ||
                    (0x80 <= value && value < 0x800   && encBytes > 2) ||
                    (0x800 < value && value < 0x10000 && encBytes > 3) ||
                    // Encoded value out of range
                    (value >= 0x110000)
                    ) {
                    hexEscapeChar(os, c);
                    break;
                }

                // If we got here, this is in fact a valid(ish) utf-8 sequence
                for (std::size_t n = 0; n < encBytes; ++n) {
                    os << m_str[idx + n];
                }
                idx += encBytes - 1;
                break;
            }
        }
    }

    std::ostream& operator << ( std::ostream& os, XmlEncode const& xmlEncode ) {
        xmlEncode.encodeTo( os );
        return os;
    }

    XmlWriter::ScopedElement::ScopedElement( XmlWriter* writer, XmlFormatting fmt )
    :   m_writer( writer ),
        m_fmt(fmt)
    {}

    XmlWriter::ScopedElement::ScopedElement( ScopedElement&& other ) noexcept
    :   m_writer( other.m_writer ),
        m_fmt(other.m_fmt)
    {
        other.m_writer = nullptr;
        other.m_fmt = XmlFormatting::None;
    }
    XmlWriter::ScopedElement& XmlWriter::ScopedElement::operator=( ScopedElement&& other ) noexcept {
        if ( m_writer ) {
            m_writer->endElement();
        }
        m_writer = other.m_writer;
        other.m_writer = nullptr;
        m_fmt = other.m_fmt;
        other.m_fmt = XmlFormatting::None;
        return *this;
    }


    XmlWriter::ScopedElement::~ScopedElement() {
        if (m_writer) {
            m_writer->endElement(m_fmt);
        }
    }

    XmlWriter::ScopedElement&
    XmlWriter::ScopedElement::writeText( StringRef text, XmlFormatting fmt ) {
        m_writer->writeText( text, fmt );
        return *this;
    }

    XmlWriter::ScopedElement&
    XmlWriter::ScopedElement::writeAttribute( StringRef name,
                                              StringRef attribute ) {
        m_writer->writeAttribute( name, attribute );
        return *this;
    }


    XmlWriter::XmlWriter( std::ostream& os ) : m_os( os )
    {
        writeDeclaration();
    }

    XmlWriter::~XmlWriter() {
        while (!m_tags.empty()) {
            endElement();
        }
        newlineIfNecessary();
    }

    XmlWriter& XmlWriter::startElement( std::string const& name, XmlFormatting fmt ) {
        ensureTagClosed();
        newlineIfNecessary();
        if (shouldIndent(fmt)) {
            m_os << m_indent;
            m_indent += "  ";
        }
        m_os << '<' << name;
        m_tags.push_back( name );
        m_tagIsOpen = true;
        applyFormatting(fmt);
        return *this;
    }

    XmlWriter::ScopedElement XmlWriter::scopedElement( std::string const& name, XmlFormatting fmt ) {
        ScopedElement scoped( this, fmt );
        startElement( name, fmt );
        return scoped;
    }

    XmlWriter& XmlWriter::endElement(XmlFormatting fmt) {
        m_indent = m_indent.substr(0, m_indent.size() - 2);

        if( m_tagIsOpen ) {
            m_os << "/>";
            m_tagIsOpen = false;
        } else {
            newlineIfNecessary();
            if (shouldIndent(fmt)) {
                m_os << m_indent;
            }
            m_os << "</" << m_tags.back() << '>';
        }
        m_os << std::flush;
        applyFormatting(fmt);
        m_tags.pop_back();
        return *this;
    }

    XmlWriter& XmlWriter::writeAttribute( StringRef name,
                                          StringRef attribute ) {
        if( !name.empty() && !attribute.empty() )
            m_os << ' ' << name << "=\"" << XmlEncode( attribute, XmlEncode::ForAttributes ) << '"';
        return *this;
    }

    XmlWriter& XmlWriter::writeAttribute( StringRef name, bool attribute ) {
        writeAttribute(name, (attribute ? "true"_sr : "false"_sr));
        return *this;
    }

    XmlWriter& XmlWriter::writeAttribute( StringRef name,
                                          char const* attribute ) {
        writeAttribute( name, StringRef( attribute ) );
        return *this;
    }

    XmlWriter& XmlWriter::writeText( StringRef text, XmlFormatting fmt ) {
        CATCH_ENFORCE(!m_tags.empty(), "Cannot write text as top level element");
        if( !text.empty() ){
            bool tagWasOpen = m_tagIsOpen;
            ensureTagClosed();
            if (tagWasOpen && shouldIndent(fmt)) {
                m_os << m_indent;
            }
            m_os << XmlEncode( text, XmlEncode::ForTextNodes );
            applyFormatting(fmt);
        }
        return *this;
    }

    XmlWriter& XmlWriter::writeComment( StringRef text, XmlFormatting fmt ) {
        ensureTagClosed();
        if (shouldIndent(fmt)) {
            m_os << m_indent;
        }
        m_os << "<!-- " << text << " -->";
        applyFormatting(fmt);
        return *this;
    }

    void XmlWriter::writeStylesheetRef( StringRef url ) {
        m_os << R"(<?xml-stylesheet type="text/xsl" href=")" << url << R"("?>)" << '\n';
    }

    void XmlWriter::ensureTagClosed() {
        if( m_tagIsOpen ) {
            m_os << '>' << std::flush;
            newlineIfNecessary();
            m_tagIsOpen = false;
        }
    }

    void XmlWriter::applyFormatting(XmlFormatting fmt) {
        m_needsNewline = shouldNewline(fmt);
    }

    void XmlWriter::writeDeclaration() {
        m_os << R"(<?xml version="1.0" encoding="UTF-8"?>)" << '\n';
    }

    void XmlWriter::newlineIfNecessary() {
        if( m_needsNewline ) {
            m_os << '\n' << std::flush;
            m_needsNewline = false;
        }
    }
}





namespace Catch {
namespace Matchers {

    std::string MatcherUntypedBase::toString() const {
        if (m_cachedToString.empty()) {
            m_cachedToString = describe();
        }
        return m_cachedToString;
    }

    MatcherUntypedBase::~MatcherUntypedBase() = default;

} // namespace Matchers
} // namespace Catch




namespace Catch {
namespace Matchers {

    std::string IsEmptyMatcher::describe() const {
        return "is empty";
    }

    std::string HasSizeMatcher::describe() const {
        ReusableStringStream sstr;
        sstr << "has size == " << m_target_size;
        return sstr.str();
    }

    IsEmptyMatcher IsEmpty() {
        return {};
    }

    HasSizeMatcher SizeIs(std::size_t sz) {
        return HasSizeMatcher{ sz };
    }

} // end namespace Matchers
} // end namespace Catch



namespace Catch {
namespace Matchers {

bool ExceptionMessageMatcher::match(std::exception const& ex) const {
    return ex.what() == m_message;
}

std::string ExceptionMessageMatcher::describe() const {
    return "exception message matches \"" + m_message + '"';
}

ExceptionMessageMatcher Message(std::string const& message) {
    return ExceptionMessageMatcher(message);
}

} // namespace Matchers
} // namespace Catch



#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <limits>


namespace Catch {
namespace {

    template <typename FP>
    bool almostEqualUlps(FP lhs, FP rhs, uint64_t maxUlpDiff) {
        // Comparison with NaN should always be false.
        // This way we can rule it out before getting into the ugly details
        if (Catch::isnan(lhs) || Catch::isnan(rhs)) {
            return false;
        }

        // This should also handle positive and negative zeros, infinities
        const auto ulpDist = ulpDistance(lhs, rhs);

        return ulpDist <= maxUlpDiff;
    }


template <typename FP>
FP step(FP start, FP direction, uint64_t steps) {
    for (uint64_t i = 0; i < steps; ++i) {
        start = Catch::nextafter(start, direction);
    }
    return start;
}

// Performs equivalent check of std::fabs(lhs - rhs) <= margin
// But without the subtraction to allow for INFINITY in comparison
bool marginComparison(double lhs, double rhs, double margin) {
    return (lhs + margin >= rhs) && (rhs + margin >= lhs);
}

template <typename FloatingPoint>
void write(std::ostream& out, FloatingPoint num) {
    out << std::scientific
        << std::setprecision(std::numeric_limits<FloatingPoint>::max_digits10 - 1)
        << num;
}

} // end anonymous namespace

namespace Matchers {
namespace Detail {

    enum class FloatingPointKind : uint8_t {
        Float,
        Double
    };

} // end namespace Detail


    WithinAbsMatcher::WithinAbsMatcher(double target, double margin)
        :m_target{ target }, m_margin{ margin } {
        CATCH_ENFORCE(margin >= 0, "Invalid margin: " << margin << '.'
            << " Margin has to be non-negative.");
    }

    // Performs equivalent check of std::fabs(lhs - rhs) <= margin
    // But without the subtraction to allow for INFINITY in comparison
    bool WithinAbsMatcher::match(double const& matchee) const {
        return (matchee + m_margin >= m_target) && (m_target + m_margin >= matchee);
    }

    std::string WithinAbsMatcher::describe() const {
        return "is within " + ::Catch::Detail::stringify(m_margin) + " of " + ::Catch::Detail::stringify(m_target);
    }


    WithinUlpsMatcher::WithinUlpsMatcher(double target, uint64_t ulps, Detail::FloatingPointKind baseType)
        :m_target{ target }, m_ulps{ ulps }, m_type{ baseType } {
        CATCH_ENFORCE(m_type == Detail::FloatingPointKind::Double
                   || m_ulps < (std::numeric_limits<uint32_t>::max)(),
            "Provided ULP is impossibly large for a float comparison.");
        CATCH_ENFORCE( std::numeric_limits<double>::is_iec559,
                       "WithinUlp matcher only supports platforms with "
                       "IEEE-754 compatible floating point representation" );
    }

#if defined(__clang__)
#pragma clang diagnostic push
// Clang <3.5 reports on the default branch in the switch below
#pragma clang diagnostic ignored "-Wunreachable-code"
#endif

    bool WithinUlpsMatcher::match(double const& matchee) const {
        switch (m_type) {
        case Detail::FloatingPointKind::Float:
            return almostEqualUlps<float>(static_cast<float>(matchee), static_cast<float>(m_target), m_ulps);
        case Detail::FloatingPointKind::Double:
            return almostEqualUlps<double>(matchee, m_target, m_ulps);
        default:
            CATCH_INTERNAL_ERROR( "Unknown Detail::FloatingPointKind value" );
        }
    }

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    std::string WithinUlpsMatcher::describe() const {
        std::stringstream ret;

        ret << "is within " << m_ulps << " ULPs of ";

        if (m_type == Detail::FloatingPointKind::Float) {
            write(ret, static_cast<float>(m_target));
            ret << 'f';
        } else {
            write(ret, m_target);
        }

        ret << " ([";
        if (m_type == Detail::FloatingPointKind::Double) {
            write( ret,
                   step( m_target,
                         -std::numeric_limits<double>::infinity(),
                         m_ulps ) );
            ret << ", ";
            write( ret,
                   step( m_target,
                         std::numeric_limits<double>::infinity(),
                         m_ulps ) );
        } else {
            // We have to cast INFINITY to float because of MinGW, see #1782
            write( ret,
                   step( static_cast<float>( m_target ),
                         -std::numeric_limits<float>::infinity(),
                         m_ulps ) );
            ret << ", ";
            write( ret,
                   step( static_cast<float>( m_target ),
                         std::numeric_limits<float>::infinity(),
                         m_ulps ) );
        }
        ret << "])";

        return ret.str();
    }

    WithinRelMatcher::WithinRelMatcher(double target, double epsilon):
        m_target(target),
        m_epsilon(epsilon){
        CATCH_ENFORCE(m_epsilon >= 0., "Relative comparison with epsilon <  0 does not make sense.");
        CATCH_ENFORCE(m_epsilon  < 1., "Relative comparison with epsilon >= 1 does not make sense.");
    }

    bool WithinRelMatcher::match(double const& matchee) const {
        const auto relMargin = m_epsilon * (std::max)(std::fabs(matchee), std::fabs(m_target));
        return marginComparison(matchee, m_target,
                                std::isinf(relMargin)? 0 : relMargin);
    }

    std::string WithinRelMatcher::describe() const {
        Catch::ReusableStringStream sstr;
        sstr << "and " << ::Catch::Detail::stringify(m_target) << " are within " << m_epsilon * 100. << "% of each other";
        return sstr.str();
    }


WithinUlpsMatcher WithinULP(double target, uint64_t maxUlpDiff) {
    return WithinUlpsMatcher(target, maxUlpDiff, Detail::FloatingPointKind::Double);
}

WithinUlpsMatcher WithinULP(float target, uint64_t maxUlpDiff) {
    return WithinUlpsMatcher(target, maxUlpDiff, Detail::FloatingPointKind::Float);
}

WithinAbsMatcher WithinAbs(double target, double margin) {
    return WithinAbsMatcher(target, margin);
}

WithinRelMatcher WithinRel(double target, double eps) {
    return WithinRelMatcher(target, eps);
}

WithinRelMatcher WithinRel(double target) {
    return WithinRelMatcher(target, std::numeric_limits<double>::epsilon() * 100);
}

WithinRelMatcher WithinRel(float target, float eps) {
    return WithinRelMatcher(target, eps);
}

WithinRelMatcher WithinRel(float target) {
    return WithinRelMatcher(target, std::numeric_limits<float>::epsilon() * 100);
}



bool IsNaNMatcher::match( double const& matchee ) const {
    return std::isnan( matchee );
}

std::string IsNaNMatcher::describe() const {
    using namespace std::string_literals;
    return "is NaN"s;
}

IsNaNMatcher IsNaN() { return IsNaNMatcher(); }

    } // namespace Matchers
} // namespace Catch




std::string Catch::Matchers::Detail::finalizeDescription(const std::string& desc) {
    if (desc.empty()) {
        return "matches undescribed predicate";
    } else {
        return "matches predicate: \"" + desc + '"';
    }
}



namespace Catch {
    namespace Matchers {
        std::string AllTrueMatcher::describe() const { return "contains only true"; }

        AllTrueMatcher AllTrue() { return AllTrueMatcher{}; }

        std::string NoneTrueMatcher::describe() const { return "contains no true"; }

        NoneTrueMatcher NoneTrue() { return NoneTrueMatcher{}; }

        std::string AnyTrueMatcher::describe() const { return "contains at least one true"; }

        AnyTrueMatcher AnyTrue() { return AnyTrueMatcher{}; }
    } // namespace Matchers
} // namespace Catch



#include <regex>

namespace Catch {
namespace Matchers {

    CasedString::CasedString( std::string const& str, CaseSensitive caseSensitivity )
    :   m_caseSensitivity( caseSensitivity ),
        m_str( adjustString( str ) )
    {}
    std::string CasedString::adjustString( std::string const& str ) const {
        return m_caseSensitivity == CaseSensitive::No
               ? toLower( str )
               : str;
    }
    StringRef CasedString::caseSensitivitySuffix() const {
        return m_caseSensitivity == CaseSensitive::Yes
                   ? StringRef()
                   : " (case insensitive)"_sr;
    }


    StringMatcherBase::StringMatcherBase( StringRef operation, CasedString const& comparator )
    : m_comparator( comparator ),
      m_operation( operation ) {
    }

    std::string StringMatcherBase::describe() const {
        std::string description;
        description.reserve(5 + m_operation.size() + m_comparator.m_str.size() +
                                    m_comparator.caseSensitivitySuffix().size());
        description += m_operation;
        description += ": \"";
        description += m_comparator.m_str;
        description += '"';
        description += m_comparator.caseSensitivitySuffix();
        return description;
    }

    StringEqualsMatcher::StringEqualsMatcher( CasedString const& comparator ) : StringMatcherBase( "equals"_sr, comparator ) {}

    bool StringEqualsMatcher::match( std::string const& source ) const {
        return m_comparator.adjustString( source ) == m_comparator.m_str;
    }


    StringContainsMatcher::StringContainsMatcher( CasedString const& comparator ) : StringMatcherBase( "contains"_sr, comparator ) {}

    bool StringContainsMatcher::match( std::string const& source ) const {
        return contains( m_comparator.adjustString( source ), m_comparator.m_str );
    }


    StartsWithMatcher::StartsWithMatcher( CasedString const& comparator ) : StringMatcherBase( "starts with"_sr, comparator ) {}

    bool StartsWithMatcher::match( std::string const& source ) const {
        return startsWith( m_comparator.adjustString( source ), m_comparator.m_str );
    }


    EndsWithMatcher::EndsWithMatcher( CasedString const& comparator ) : StringMatcherBase( "ends with"_sr, comparator ) {}

    bool EndsWithMatcher::match( std::string const& source ) const {
        return endsWith( m_comparator.adjustString( source ), m_comparator.m_str );
    }



    RegexMatcher::RegexMatcher(std::string regex, CaseSensitive caseSensitivity): m_regex(CATCH_MOVE(regex)), m_caseSensitivity(caseSensitivity) {}

    bool RegexMatcher::match(std::string const& matchee) const {
        auto flags = std::regex::ECMAScript; // ECMAScript is the default syntax option anyway
        if (m_caseSensitivity == CaseSensitive::No) {
            flags |= std::regex::icase;
        }
        auto reg = std::regex(m_regex, flags);
        return std::regex_match(matchee, reg);
    }

    std::string RegexMatcher::describe() const {
        return "matches " + ::Catch::Detail::stringify(m_regex) + ((m_caseSensitivity == CaseSensitive::Yes)? " case sensitively" : " case insensitively");
    }


    StringEqualsMatcher Equals( std::string const& str, CaseSensitive caseSensitivity ) {
        return StringEqualsMatcher( CasedString( str, caseSensitivity) );
    }
    StringContainsMatcher ContainsSubstring( std::string const& str, CaseSensitive caseSensitivity ) {
        return StringContainsMatcher( CasedString( str, caseSensitivity) );
    }
    EndsWithMatcher EndsWith( std::string const& str, CaseSensitive caseSensitivity ) {
        return EndsWithMatcher( CasedString( str, caseSensitivity) );
    }
    StartsWithMatcher StartsWith( std::string const& str, CaseSensitive caseSensitivity ) {
        return StartsWithMatcher( CasedString( str, caseSensitivity) );
    }

    RegexMatcher Matches(std::string const& regex, CaseSensitive caseSensitivity) {
        return RegexMatcher(regex, caseSensitivity);
    }

} // namespace Matchers
} // namespace Catch



namespace Catch {
namespace Matchers {
    MatcherGenericBase::~MatcherGenericBase() = default;

    namespace Detail {

        std::string describe_multi_matcher(StringRef combine, std::string const* descriptions_begin, std::string const* descriptions_end) {
            std::string description;
            std::size_t combined_size = 4;
            for ( auto desc = descriptions_begin; desc != descriptions_end; ++desc ) {
                combined_size += desc->size();
            }
            combined_size += static_cast<size_t>(descriptions_end - descriptions_begin - 1) * combine.size();

            description.reserve(combined_size);

            description += "( ";
            bool first = true;
            for( auto desc = descriptions_begin; desc != descriptions_end; ++desc ) {
                if( first )
                    first = false;
                else
                    description += combine;
                description += *desc;
            }
            description += " )";
            return description;
        }

    } // namespace Detail
} // namespace Matchers
} // namespace Catch




namespace Catch {

    // This is the general overload that takes a any string matcher
    // There is another overload, in catch_assertionhandler.h/.cpp, that only takes a string and infers
    // the Equals matcher (so the header does not mention matchers)
    void handleExceptionMatchExpr( AssertionHandler& handler, StringMatcher const& matcher ) {
        std::string exceptionMessage = Catch::translateActiveException();
        MatchExpr<std::string, StringMatcher const&> expr( CATCH_MOVE(exceptionMessage), matcher );
        handler.handleExpr( expr );
    }

} // namespace Catch



#include <ostream>

namespace Catch {

    AutomakeReporter::~AutomakeReporter() = default;

    void AutomakeReporter::testCaseEnded(TestCaseStats const& _testCaseStats) {
        // Possible values to emit are PASS, XFAIL, SKIP, FAIL, XPASS and ERROR.
        m_stream << ":test-result: ";
        if ( _testCaseStats.totals.testCases.skipped > 0 ) {
            m_stream << "SKIP";
        } else if (_testCaseStats.totals.assertions.allPassed()) {
            m_stream << "PASS";
        } else if (_testCaseStats.totals.assertions.allOk()) {
            m_stream << "XFAIL";
        } else {
            m_stream << "FAIL";
        }
        m_stream << ' ' << _testCaseStats.testInfo->name << '\n';
        StreamingReporterBase::testCaseEnded(_testCaseStats);
    }

    void AutomakeReporter::skipTest(TestCaseInfo const& testInfo) {
        m_stream << ":test-result: SKIP " << testInfo.name << '\n';
    }

} // end namespace Catch






namespace Catch {
    ReporterBase::ReporterBase( ReporterConfig&& config ):
        IEventListener( config.fullConfig() ),
        m_wrapped_stream( CATCH_MOVE(config).takeStream() ),
        m_stream( m_wrapped_stream->stream() ),
        m_colour( makeColourImpl( config.colourMode(), m_wrapped_stream.get() ) ),
        m_customOptions( config.customOptions() )
    {}

    ReporterBase::~ReporterBase() = default;

    void ReporterBase::listReporters(
        std::vector<ReporterDescription> const& descriptions ) {
        defaultListReporters(m_stream, descriptions, m_config->verbosity());
    }

    void ReporterBase::listListeners(
        std::vector<ListenerDescription> const& descriptions ) {
        defaultListListeners( m_stream, descriptions );
    }

    void ReporterBase::listTests(std::vector<TestCaseHandle> const& tests) {
        defaultListTests(m_stream,
                         m_colour.get(),
                         tests,
                         m_config->hasTestFilters(),
                         m_config->verbosity());
    }

    void ReporterBase::listTags(std::vector<TagInfo> const& tags) {
        defaultListTags( m_stream, tags, m_config->hasTestFilters() );
    }

} // namespace Catch




#include <ostream>

namespace Catch {
namespace {

    // Colour::LightGrey
    static constexpr Colour::Code compactDimColour = Colour::FileName;

#ifdef CATCH_PLATFORM_MAC
    static constexpr Catch::StringRef compactFailedString = "FAILED"_sr;
    static constexpr Catch::StringRef compactPassedString = "PASSED"_sr;
#else
    static constexpr Catch::StringRef compactFailedString = "failed"_sr;
    static constexpr Catch::StringRef compactPassedString = "passed"_sr;
#endif

// Implementation of CompactReporter formatting
class AssertionPrinter {
public:
    AssertionPrinter& operator= (AssertionPrinter const&) = delete;
    AssertionPrinter(AssertionPrinter const&) = delete;
    AssertionPrinter(std::ostream& _stream, AssertionStats const& _stats, bool _printInfoMessages, ColourImpl* colourImpl_)
        : stream(_stream)
        , result(_stats.assertionResult)
        , messages(_stats.infoMessages)
        , itMessage(_stats.infoMessages.begin())
        , printInfoMessages(_printInfoMessages)
        , colourImpl(colourImpl_)
    {}

    void print() {
        printSourceInfo();

        itMessage = messages.begin();

        switch (result.getResultType()) {
        case ResultWas::Ok:
            printResultType(Colour::ResultSuccess, compactPassedString);
            printOriginalExpression();
            printReconstructedExpression();
            if (!result.hasExpression())
                printRemainingMessages(Colour::None);
            else
                printRemainingMessages();
            break;
        case ResultWas::ExpressionFailed:
            if (result.isOk())
                printResultType(Colour::ResultSuccess, compactFailedString + " - but was ok"_sr);
            else
                printResultType(Colour::Error, compactFailedString);
            printOriginalExpression();
            printReconstructedExpression();
            printRemainingMessages();
            break;
        case ResultWas::ThrewException:
            printResultType(Colour::Error, compactFailedString);
            printIssue("unexpected exception with message:");
            printMessage();
            printExpressionWas();
            printRemainingMessages();
            break;
        case ResultWas::FatalErrorCondition:
            printResultType(Colour::Error, compactFailedString);
            printIssue("fatal error condition with message:");
            printMessage();
            printExpressionWas();
            printRemainingMessages();
            break;
        case ResultWas::DidntThrowException:
            printResultType(Colour::Error, compactFailedString);
            printIssue("expected exception, got none");
            printExpressionWas();
            printRemainingMessages();
            break;
        case ResultWas::Info:
            printResultType(Colour::None, "info"_sr);
            printMessage();
            printRemainingMessages();
            break;
        case ResultWas::Warning:
            printResultType(Colour::None, "warning"_sr);
            printMessage();
            printRemainingMessages();
            break;
        case ResultWas::ExplicitFailure:
            printResultType(Colour::Error, compactFailedString);
            printIssue("explicitly");
            printRemainingMessages(Colour::None);
            break;
        case ResultWas::ExplicitSkip:
            printResultType(Colour::Skip, "skipped"_sr);
            printMessage();
            printRemainingMessages();
            break;
            // These cases are here to prevent compiler warnings
        case ResultWas::Unknown:
        case ResultWas::FailureBit:
        case ResultWas::Exception:
            printResultType(Colour::Error, "** internal error **");
            break;
        }
    }

private:
    void printSourceInfo() const {
        stream << colourImpl->guardColour( Colour::FileName )
               << result.getSourceInfo() << ':';
    }

    void printResultType(Colour::Code colour, StringRef passOrFail) const {
        if (!passOrFail.empty()) {
            stream << colourImpl->guardColour(colour) << ' ' << passOrFail;
            stream << ':';
        }
    }

    void printIssue(char const* issue) const {
        stream << ' ' << issue;
    }

    void printExpressionWas() {
        if (result.hasExpression()) {
            stream << ';';
            {
                stream << colourImpl->guardColour(compactDimColour) << " expression was:";
            }
            printOriginalExpression();
        }
    }

    void printOriginalExpression() const {
        if (result.hasExpression()) {
            stream << ' ' << result.getExpression();
        }
    }

    void printReconstructedExpression() const {
        if (result.hasExpandedExpression()) {
            stream << colourImpl->guardColour(compactDimColour) << " for: ";
            stream << result.getExpandedExpression();
        }
    }

    void printMessage() {
        if (itMessage != messages.end()) {
            stream << " '" << itMessage->message << '\'';
            ++itMessage;
        }
    }

    void printRemainingMessages(Colour::Code colour = compactDimColour) {
        if (itMessage == messages.end())
            return;

        const auto itEnd = messages.cend();
        const auto N = static_cast<std::size_t>(itEnd - itMessage);

        stream << colourImpl->guardColour( colour ) << " with "
               << pluralise( N, "message"_sr ) << ':';

        while (itMessage != itEnd) {
            // If this assertion is a warning ignore any INFO messages
            if (printInfoMessages || itMessage->type != ResultWas::Info) {
                printMessage();
                if (itMessage != itEnd) {
                    stream << colourImpl->guardColour(compactDimColour) << " and";
                }
                continue;
            }
            ++itMessage;
        }
    }

private:
    std::ostream& stream;
    AssertionResult const& result;
    std::vector<MessageInfo> const& messages;
    std::vector<MessageInfo>::const_iterator itMessage;
    bool printInfoMessages;
    ColourImpl* colourImpl;
};

} // anon namespace

        std::string CompactReporter::getDescription() {
            return "Reports test results on a single line, suitable for IDEs";
        }

        void CompactReporter::noMatchingTestCases( StringRef unmatchedSpec ) {
            m_stream << "No test cases matched '" << unmatchedSpec << "'\n";
        }

        void CompactReporter::testRunStarting( TestRunInfo const& ) {
            if ( m_config->testSpec().hasFilters() ) {
                m_stream << m_colour->guardColour( Colour::BrightYellow )
                         << "Filters: "
                         << m_config->testSpec()
                         << '\n';
            }
            m_stream << "RNG seed: " << getSeed() << '\n'
                     << std::flush;
        }

        void CompactReporter::assertionEnded( AssertionStats const& _assertionStats ) {
            AssertionResult const& result = _assertionStats.assertionResult;

            bool printInfoMessages = true;

            // Drop out if result was successful and we're not printing those
            if( !m_config->includeSuccessfulResults() && result.isOk() ) {
                if( result.getResultType() != ResultWas::Warning && result.getResultType() != ResultWas::ExplicitSkip )
                    return;
                printInfoMessages = false;
            }

            AssertionPrinter printer( m_stream, _assertionStats, printInfoMessages, m_colour.get() );
            printer.print();

            m_stream << '\n' << std::flush;
        }

        void CompactReporter::sectionEnded(SectionStats const& _sectionStats) {
            double dur = _sectionStats.durationInSeconds;
            if ( shouldShowDuration( *m_config, dur ) ) {
                m_stream << getFormattedDuration( dur ) << " s: " << _sectionStats.sectionInfo.name << '\n' << std::flush;
            }
        }

        void CompactReporter::testRunEnded( TestRunStats const& _testRunStats ) {
            printTestRunTotals( m_stream, *m_colour, _testRunStats.totals );
            m_stream << "\n\n" << std::flush;
            StreamingReporterBase::testRunEnded( _testRunStats );
        }

        CompactReporter::~CompactReporter() = default;

} // end namespace Catch




#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4061) // Not all labels are EXPLICITLY handled in switch
 // Note that 4062 (not all labels are handled and default is missing) is enabled
#endif

#if defined(__clang__)
#  pragma clang diagnostic push
// For simplicity, benchmarking-only helpers are always enabled
#  pragma clang diagnostic ignored "-Wunused-function"
#endif



namespace Catch {

namespace {

// Formatter impl for ConsoleReporter
class ConsoleAssertionPrinter {
public:
    ConsoleAssertionPrinter& operator= (ConsoleAssertionPrinter const&) = delete;
    ConsoleAssertionPrinter(ConsoleAssertionPrinter const&) = delete;
    ConsoleAssertionPrinter(std::ostream& _stream, AssertionStats const& _stats, ColourImpl* colourImpl_, bool _printInfoMessages)
        : stream(_stream),
        stats(_stats),
        result(_stats.assertionResult),
        colour(Colour::None),
        messages(_stats.infoMessages),
        colourImpl(colourImpl_),
        printInfoMessages(_printInfoMessages) {
        switch (result.getResultType()) {
        case ResultWas::Ok:
            colour = Colour::Success;
            passOrFail = "PASSED"_sr;
            //if( result.hasMessage() )
            if (messages.size() == 1)
                messageLabel = "with message"_sr;
            if (messages.size() > 1)
                messageLabel = "with messages"_sr;
            break;
        case ResultWas::ExpressionFailed:
            if (result.isOk()) {
                colour = Colour::Success;
                passOrFail = "FAILED - but was ok"_sr;
            } else {
                colour = Colour::Error;
                passOrFail = "FAILED"_sr;
            }
            if (messages.size() == 1)
                messageLabel = "with message"_sr;
            if (messages.size() > 1)
                messageLabel = "with messages"_sr;
            break;
        case ResultWas::ThrewException:
            colour = Colour::Error;
            passOrFail = "FAILED"_sr;
            // todo switch
            switch (messages.size()) { case 0:
                messageLabel = "due to unexpected exception with "_sr;
                break;
            case 1:
                messageLabel = "due to unexpected exception with message"_sr;
                break;
            default:
                messageLabel = "due to unexpected exception with messages"_sr;
                break;
            }
            break;
        case ResultWas::FatalErrorCondition:
            colour = Colour::Error;
            passOrFail = "FAILED"_sr;
            messageLabel = "due to a fatal error condition"_sr;
            break;
        case ResultWas::DidntThrowException:
            colour = Colour::Error;
            passOrFail = "FAILED"_sr;
            messageLabel = "because no exception was thrown where one was expected"_sr;
            break;
        case ResultWas::Info:
            messageLabel = "info"_sr;
            break;
        case ResultWas::Warning:
            messageLabel = "warning"_sr;
            break;
        case ResultWas::ExplicitFailure:
            passOrFail = "FAILED"_sr;
            colour = Colour::Error;
            if (messages.size() == 1)
                messageLabel = "explicitly with message"_sr;
            if (messages.size() > 1)
                messageLabel = "explicitly with messages"_sr;
            break;
        case ResultWas::ExplicitSkip:
            colour = Colour::Skip;
            passOrFail = "SKIPPED"_sr;
            if (messages.size() == 1)
                messageLabel = "explicitly with message"_sr;
            if (messages.size() > 1)
                messageLabel = "explicitly with messages"_sr;
            break;
            // These cases are here to prevent compiler warnings
        case ResultWas::Unknown:
        case ResultWas::FailureBit:
        case ResultWas::Exception:
            passOrFail = "** internal error **"_sr;
            colour = Colour::Error;
            break;
        }
    }

    void print() const {
        printSourceInfo();
        if (stats.totals.assertions.total() > 0) {
            printResultType();
            printOriginalExpression();
            printReconstructedExpression();
        } else {
            stream << '\n';
        }
        printMessage();
    }

private:
    void printResultType() const {
        if (!passOrFail.empty()) {
            stream << colourImpl->guardColour(colour) << passOrFail << ":\n";
        }
    }
    void printOriginalExpression() const {
        if (result.hasExpression()) {
            stream << colourImpl->guardColour( Colour::OriginalExpression )
                   << "  " << result.getExpressionInMacro() << '\n';
        }
    }
    void printReconstructedExpression() const {
        if (result.hasExpandedExpression()) {
            stream << "with expansion:\n";
            stream << colourImpl->guardColour( Colour::ReconstructedExpression )
                   << TextFlow::Column( result.getExpandedExpression() )
                          .indent( 2 )
                   << '\n';
        }
    }
    void printMessage() const {
        if (!messageLabel.empty())
            stream << messageLabel << ':' << '\n';
        for (auto const& msg : messages) {
            // If this assertion is a warning ignore any INFO messages
            if (printInfoMessages || msg.type != ResultWas::Info)
                stream << TextFlow::Column(msg.message).indent(2) << '\n';
        }
    }
    void printSourceInfo() const {
        stream << colourImpl->guardColour( Colour::FileName )
               << result.getSourceInfo() << ": ";
    }

    std::ostream& stream;
    AssertionStats const& stats;
    AssertionResult const& result;
    Colour::Code colour;
    StringRef passOrFail;
    StringRef messageLabel;
    std::vector<MessageInfo> const& messages;
    ColourImpl* colourImpl;
    bool printInfoMessages;
};

std::size_t makeRatio( std::uint64_t number, std::uint64_t total ) {
    const auto ratio = total > 0 ? CATCH_CONFIG_CONSOLE_WIDTH * number / total : 0;
    return (ratio == 0 && number > 0) ? 1 : static_cast<std::size_t>(ratio);
}

std::size_t&
findMax( std::size_t& i, std::size_t& j, std::size_t& k, std::size_t& l ) {
    if (i > j && i > k && i > l)
        return i;
    else if (j > k && j > l)
        return j;
    else if (k > l)
        return k;
    else
        return l;
}

struct ColumnBreak {};
struct RowBreak {};
struct OutputFlush {};

class Duration {
    enum class Unit : uint8_t {
        Auto,
        Nanoseconds,
        Microseconds,
        Milliseconds,
        Seconds,
        Minutes
    };
    static const uint64_t s_nanosecondsInAMicrosecond = 1000;
    static const uint64_t s_nanosecondsInAMillisecond = 1000 * s_nanosecondsInAMicrosecond;
    static const uint64_t s_nanosecondsInASecond = 1000 * s_nanosecondsInAMillisecond;
    static const uint64_t s_nanosecondsInAMinute = 60 * s_nanosecondsInASecond;

    double m_inNanoseconds;
    Unit m_units;

public:
    explicit Duration(double inNanoseconds, Unit units = Unit::Auto)
        : m_inNanoseconds(inNanoseconds),
        m_units(units) {
        if (m_units == Unit::Auto) {
            if (m_inNanoseconds < s_nanosecondsInAMicrosecond)
                m_units = Unit::Nanoseconds;
            else if (m_inNanoseconds < s_nanosecondsInAMillisecond)
                m_units = Unit::Microseconds;
            else if (m_inNanoseconds < s_nanosecondsInASecond)
                m_units = Unit::Milliseconds;
            else if (m_inNanoseconds < s_nanosecondsInAMinute)
                m_units = Unit::Seconds;
            else
                m_units = Unit::Minutes;
        }

    }

    auto value() const -> double {
        switch (m_units) {
        case Unit::Microseconds:
            return m_inNanoseconds / static_cast<double>(s_nanosecondsInAMicrosecond);
        case Unit::Milliseconds:
            return m_inNanoseconds / static_cast<double>(s_nanosecondsInAMillisecond);
        case Unit::Seconds:
            return m_inNanoseconds / static_cast<double>(s_nanosecondsInASecond);
        case Unit::Minutes:
            return m_inNanoseconds / static_cast<double>(s_nanosecondsInAMinute);
        default:
            return m_inNanoseconds;
        }
    }
    StringRef unitsAsString() const {
        switch (m_units) {
        case Unit::Nanoseconds:
            return "ns"_sr;
        case Unit::Microseconds:
            return "us"_sr;
        case Unit::Milliseconds:
            return "ms"_sr;
        case Unit::Seconds:
            return "s"_sr;
        case Unit::Minutes:
            return "m"_sr;
        default:
            return "** internal error **"_sr;
        }

    }
    friend auto operator << (std::ostream& os, Duration const& duration) -> std::ostream& {
        return os << duration.value() << ' ' << duration.unitsAsString();
    }
};
} // end anon namespace

enum class Justification : uint8_t {
    Left,
    Right
};

struct ColumnInfo {
    std::string name;
    std::size_t width;
    Justification justification;
};

class TablePrinter {
    std::ostream& m_os;
    std::vector<ColumnInfo> m_columnInfos;
    ReusableStringStream m_oss;
    int m_currentColumn = -1;
    bool m_isOpen = false;

public:
    TablePrinter( std::ostream& os, std::vector<ColumnInfo> columnInfos )
    :   m_os( os ),
        m_columnInfos( CATCH_MOVE( columnInfos ) ) {}

    auto columnInfos() const -> std::vector<ColumnInfo> const& {
        return m_columnInfos;
    }

    void open() {
        if (!m_isOpen) {
            m_isOpen = true;
            *this << RowBreak();

			TextFlow::Columns headerCols;
			for (auto const& info : m_columnInfos) {
                assert(info.width > 2);
				headerCols += TextFlow::Column(info.name).width(info.width - 2);
                headerCols += TextFlow::Spacer( 2 );
			}
			m_os << headerCols << '\n';

            m_os << lineOfChars('-') << '\n';
        }
    }
    void close() {
        if (m_isOpen) {
            *this << RowBreak();
            m_os << '\n' << std::flush;
            m_isOpen = false;
        }
    }

    template<typename T>
    friend TablePrinter& operator<< (TablePrinter& tp, T const& value) {
        tp.m_oss << value;
        return tp;
    }

    friend TablePrinter& operator<< (TablePrinter& tp, ColumnBreak) {
        auto colStr = tp.m_oss.str();
        const auto strSize = colStr.size();
        tp.m_oss.str("");
        tp.open();
        if (tp.m_currentColumn == static_cast<int>(tp.m_columnInfos.size() - 1)) {
            tp.m_currentColumn = -1;
            tp.m_os << '\n';
        }
        tp.m_currentColumn++;

        auto colInfo = tp.m_columnInfos[tp.m_currentColumn];
        auto padding = (strSize + 1 < colInfo.width)
            ? std::string(colInfo.width - (strSize + 1), ' ')
            : std::string();
        if (colInfo.justification == Justification::Left)
            tp.m_os << colStr << padding << ' ';
        else
            tp.m_os << padding << colStr << ' ';
        return tp;
    }

    friend TablePrinter& operator<< (TablePrinter& tp, RowBreak) {
        if (tp.m_currentColumn > 0) {
            tp.m_os << '\n';
            tp.m_currentColumn = -1;
        }
        return tp;
    }

    friend TablePrinter& operator<<(TablePrinter& tp, OutputFlush) {
        tp.m_os << std::flush;
        return tp;
    }
};

ConsoleReporter::ConsoleReporter(ReporterConfig&& config):
    StreamingReporterBase( CATCH_MOVE( config ) ),
    m_tablePrinter(Detail::make_unique<TablePrinter>(m_stream,
        [&config]() -> std::vector<ColumnInfo> {
        if (config.fullConfig()->benchmarkNoAnalysis())
        {
            return{
                { "benchmark name", CATCH_CONFIG_CONSOLE_WIDTH - 43, Justification::Left },
                { "     samples", 14, Justification::Right },
                { "  iterations", 14, Justification::Right },
                { "        mean", 14, Justification::Right }
            };
        }
        else
        {
            return{
                { "benchmark name", CATCH_CONFIG_CONSOLE_WIDTH - 43, Justification::Left },
                { "samples      mean       std dev", 14, Justification::Right },
                { "iterations   low mean   low std dev", 14, Justification::Right },
                { "est run time high mean  high std dev", 14, Justification::Right }
            };
        }
    }())) {
    m_preferences.shouldReportAllAssertionStarts = false;
}

ConsoleReporter::~ConsoleReporter() = default;

std::string ConsoleReporter::getDescription() {
    return "Reports test results as plain lines of text";
}

void ConsoleReporter::noMatchingTestCases( StringRef unmatchedSpec ) {
    m_stream << "No test cases matched '" << unmatchedSpec << "'\n";
}

void ConsoleReporter::reportInvalidTestSpec( StringRef arg ) {
    m_stream << "Invalid Filter: " << arg << '\n';
}

void ConsoleReporter::assertionEnded(AssertionStats const& _assertionStats) {
    AssertionResult const& result = _assertionStats.assertionResult;

    bool includeResults = m_config->includeSuccessfulResults() || !result.isOk();

    // Drop out if result was successful but we're not printing them.
    // TODO: Make configurable whether skips should be printed
    if (!includeResults && result.getResultType() != ResultWas::Warning && result.getResultType() != ResultWas::ExplicitSkip)
        return;

    lazyPrint();

    ConsoleAssertionPrinter printer(m_stream, _assertionStats, m_colour.get(), includeResults);
    printer.print();
    m_stream << '\n' << std::flush;
}

void ConsoleReporter::sectionStarting(SectionInfo const& _sectionInfo) {
    m_tablePrinter->close();
    m_headerPrinted = false;
    StreamingReporterBase::sectionStarting(_sectionInfo);
}
void ConsoleReporter::sectionEnded(SectionStats const& _sectionStats) {
    m_tablePrinter->close();
    if (_sectionStats.missingAssertions) {
        lazyPrint();
        auto guard =
            m_colour->guardColour( Colour::ResultError ).engage( m_stream );
        if (m_sectionStack.size() > 1)
            m_stream << "\nNo assertions in section";
        else
            m_stream << "\nNo assertions in test case";
        m_stream << " '" << _sectionStats.sectionInfo.name << "'\n\n" << std::flush;
    }
    double dur = _sectionStats.durationInSeconds;
    if (shouldShowDuration(*m_config, dur)) {
        m_stream << getFormattedDuration(dur) << " s: " << _sectionStats.sectionInfo.name << '\n' << std::flush;
    }
    if (m_headerPrinted) {
        m_headerPrinted = false;
    }
    StreamingReporterBase::sectionEnded(_sectionStats);
}

void ConsoleReporter::benchmarkPreparing( StringRef name ) {
	lazyPrintWithoutClosingBenchmarkTable();

	auto nameCol = TextFlow::Column( static_cast<std::string>( name ) )
                       .width( m_tablePrinter->columnInfos()[0].width - 2 );

	bool firstLine = true;
	for (auto line : nameCol) {
		if (!firstLine)
			(*m_tablePrinter) << ColumnBreak() << ColumnBreak() << ColumnBreak();
		else
			firstLine = false;

		(*m_tablePrinter) << line << ColumnBreak();
	}
}

void ConsoleReporter::benchmarkStarting(BenchmarkInfo const& info) {
    (*m_tablePrinter) << info.samples << ColumnBreak()
        << info.iterations << ColumnBreak();
    if ( !m_config->benchmarkNoAnalysis() ) {
        ( *m_tablePrinter )
            << Duration( info.estimatedDuration ) << ColumnBreak();
    }
    ( *m_tablePrinter ) << OutputFlush{};
}
void ConsoleReporter::benchmarkEnded(BenchmarkStats<> const& stats) {
    if (m_config->benchmarkNoAnalysis())
    {
        (*m_tablePrinter) << Duration(stats.mean.point.count()) << ColumnBreak();
    }
    else
    {
        (*m_tablePrinter) << ColumnBreak()
            << Duration(stats.mean.point.count()) << ColumnBreak()
            << Duration(stats.mean.lower_bound.count()) << ColumnBreak()
            << Duration(stats.mean.upper_bound.count()) << ColumnBreak() << ColumnBreak()
            << Duration(stats.standardDeviation.point.count()) << ColumnBreak()
            << Duration(stats.standardDeviation.lower_bound.count()) << ColumnBreak()
            << Duration(stats.standardDeviation.upper_bound.count()) << ColumnBreak() << ColumnBreak() << ColumnBreak() << ColumnBreak() << ColumnBreak();
    }
}

void ConsoleReporter::benchmarkFailed( StringRef error ) {
    auto guard = m_colour->guardColour( Colour::Red ).engage( m_stream );
    (*m_tablePrinter)
        << "Benchmark failed (" << error << ')'
        << ColumnBreak() << RowBreak();
}

void ConsoleReporter::testCaseEnded(TestCaseStats const& _testCaseStats) {
    m_tablePrinter->close();
    StreamingReporterBase::testCaseEnded(_testCaseStats);
    m_headerPrinted = false;
}
void ConsoleReporter::testRunEnded(TestRunStats const& _testRunStats) {
    printTotalsDivider(_testRunStats.totals);
    printTestRunTotals( m_stream, *m_colour, _testRunStats.totals );
    m_stream << '\n' << std::flush;
    StreamingReporterBase::testRunEnded(_testRunStats);
}
void ConsoleReporter::testRunStarting(TestRunInfo const& _testRunInfo) {
    StreamingReporterBase::testRunStarting(_testRunInfo);
    if ( m_config->testSpec().hasFilters() ) {
        m_stream << m_colour->guardColour( Colour::BrightYellow ) << "Filters: "
                 << m_config->testSpec() << '\n';
    }
    m_stream << "Randomness seeded to: " << getSeed() << '\n'
             << std::flush;
}

void ConsoleReporter::lazyPrint() {

    m_tablePrinter->close();
    lazyPrintWithoutClosingBenchmarkTable();
}

void ConsoleReporter::lazyPrintWithoutClosingBenchmarkTable() {

    if ( !m_testRunInfoPrinted ) {
        lazyPrintRunInfo();
    }
    if (!m_headerPrinted) {
        printTestCaseAndSectionHeader();
        m_headerPrinted = true;
    }
}
void ConsoleReporter::lazyPrintRunInfo() {
    m_stream << '\n'
             << lineOfChars( '~' ) << '\n'
             << m_colour->guardColour( Colour::SecondaryText )
             << currentTestRunInfo.name << " is a Catch2 v" << libraryVersion()
             << " host application.\n"
             << "Run with -? for options\n\n";

    m_testRunInfoPrinted = true;
}
void ConsoleReporter::printTestCaseAndSectionHeader() {
    assert(!m_sectionStack.empty());
    printOpenHeader(currentTestCaseInfo->name);

    if (m_sectionStack.size() > 1) {
        auto guard = m_colour->guardColour( Colour::Headers ).engage( m_stream );

        auto
            it = m_sectionStack.begin() + 1, // Skip first section (test case)
            itEnd = m_sectionStack.end();
        for (; it != itEnd; ++it)
            printHeaderString(it->name, 2);
    }

    SourceLineInfo lineInfo = m_sectionStack.back().lineInfo;


    m_stream << lineOfChars( '-' ) << '\n'
             << m_colour->guardColour( Colour::FileName ) << lineInfo << '\n'
             << lineOfChars( '.' ) << "\n\n"
             << std::flush;
}

void ConsoleReporter::printClosedHeader(std::string const& _name) {
    printOpenHeader(_name);
    m_stream << lineOfChars('.') << '\n';
}
void ConsoleReporter::printOpenHeader(std::string const& _name) {
    m_stream << lineOfChars('-') << '\n';
    {
        auto guard = m_colour->guardColour( Colour::Headers ).engage( m_stream );
        printHeaderString(_name);
    }
}

void ConsoleReporter::printHeaderString(std::string const& _string, std::size_t indent) {
    // We want to get a bit fancy with line breaking here, so that subsequent
    // lines start after ":" if one is present, e.g.
    // ```
    // blablabla: Fancy
    //            linebreaking
    // ```
    // but we also want to avoid problems with overly long indentation causing
    // the text to take up too many lines, e.g.
    // ```
    // blablabla: F
    //            a
    //            n
    //            c
    //            y
    //            .
    //            .
    //            .
    // ```
    // So we limit the prefix indentation check to first quarter of the possible
    // width
    std::size_t idx = _string.find( ": " );
    if ( idx != std::string::npos && idx < CATCH_CONFIG_CONSOLE_WIDTH / 4 ) {
        idx += 2;
    } else {
        idx = 0;
    }
    m_stream << TextFlow::Column( _string )
                  .indent( indent + idx )
                  .initialIndent( indent )
           << '\n';
}

void ConsoleReporter::printTotalsDivider(Totals const& totals) {
    if (totals.testCases.total() > 0) {
        std::size_t failedRatio = makeRatio(totals.testCases.failed, totals.testCases.total());
        std::size_t failedButOkRatio = makeRatio(totals.testCases.failedButOk, totals.testCases.total());
        std::size_t passedRatio = makeRatio(totals.testCases.passed, totals.testCases.total());
        std::size_t skippedRatio = makeRatio(totals.testCases.skipped, totals.testCases.total());
        while (failedRatio + failedButOkRatio + passedRatio + skippedRatio < CATCH_CONFIG_CONSOLE_WIDTH - 1)
            findMax(failedRatio, failedButOkRatio, passedRatio, skippedRatio)++;
        while (failedRatio + failedButOkRatio + passedRatio > CATCH_CONFIG_CONSOLE_WIDTH - 1)
            findMax(failedRatio, failedButOkRatio, passedRatio, skippedRatio)--;

        m_stream << m_colour->guardColour( Colour::Error )
                 << std::string( failedRatio, '=' )
                 << m_colour->guardColour( Colour::ResultExpectedFailure )
                 << std::string( failedButOkRatio, '=' );
        if ( totals.testCases.allPassed() ) {
            m_stream << m_colour->guardColour( Colour::ResultSuccess )
                     << std::string( passedRatio, '=' );
        } else {
            m_stream << m_colour->guardColour( Colour::Success )
                     << std::string( passedRatio, '=' );
        }
        m_stream << m_colour->guardColour( Colour::Skip )
                 << std::string( skippedRatio, '=' );
    } else {
        m_stream << m_colour->guardColour( Colour::Warning )
                 << std::string( CATCH_CONFIG_CONSOLE_WIDTH - 1, '=' );
    }
    m_stream << '\n';
}

} // end namespace Catch

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(__clang__)
#  pragma clang diagnostic pop
#endif




#include <algorithm>
#include <cassert>

namespace Catch {
    namespace {
        struct BySectionInfo {
            BySectionInfo( SectionInfo const& other ): m_other( other ) {}
            BySectionInfo( BySectionInfo const& other ) = default;
            bool operator()(
                Detail::unique_ptr<CumulativeReporterBase::SectionNode> const&
                    node ) const {
                return (
                    ( node->stats.sectionInfo.name == m_other.name ) &&
                    ( node->stats.sectionInfo.lineInfo == m_other.lineInfo ) );
            }
            void operator=( BySectionInfo const& ) = delete;

        private:
            SectionInfo const& m_other;
        };

    } // namespace

    namespace Detail {
        AssertionOrBenchmarkResult::AssertionOrBenchmarkResult(
            AssertionStats const& assertion ):
            m_assertion( assertion ) {}

        AssertionOrBenchmarkResult::AssertionOrBenchmarkResult(
            BenchmarkStats<> const& benchmark ):
            m_benchmark( benchmark ) {}

        bool AssertionOrBenchmarkResult::isAssertion() const {
            return m_assertion.some();
        }
        bool AssertionOrBenchmarkResult::isBenchmark() const {
            return m_benchmark.some();
        }

        AssertionStats const& AssertionOrBenchmarkResult::asAssertion() const {
            assert(m_assertion.some());

            return *m_assertion;
        }
        BenchmarkStats<> const& AssertionOrBenchmarkResult::asBenchmark() const {
            assert(m_benchmark.some());

            return *m_benchmark;
        }

    }

    CumulativeReporterBase::~CumulativeReporterBase() = default;

    void CumulativeReporterBase::benchmarkEnded(BenchmarkStats<> const& benchmarkStats) {
        m_sectionStack.back()->assertionsAndBenchmarks.emplace_back(benchmarkStats);
    }

    void
    CumulativeReporterBase::sectionStarting( SectionInfo const& sectionInfo ) {
        // We need a copy, because SectionStats expect to take ownership
        SectionStats incompleteStats( SectionInfo(sectionInfo), Counts(), 0, false );
        SectionNode* node;
        if ( m_sectionStack.empty() ) {
            if ( !m_rootSection ) {
                m_rootSection =
                    Detail::make_unique<SectionNode>( incompleteStats );
            }
            node = m_rootSection.get();
        } else {
            SectionNode& parentNode = *m_sectionStack.back();
            auto it = std::find_if( parentNode.childSections.begin(),
                                    parentNode.childSections.end(),
                                    BySectionInfo( sectionInfo ) );
            if ( it == parentNode.childSections.end() ) {
                auto newNode =
                    Detail::make_unique<SectionNode>( incompleteStats );
                node = newNode.get();
                parentNode.childSections.push_back( CATCH_MOVE( newNode ) );
            } else {
                node = it->get();
            }
        }

        m_deepestSection = node;
        m_sectionStack.push_back( node );
    }

    void CumulativeReporterBase::assertionEnded(
        AssertionStats const& assertionStats ) {
        assert( !m_sectionStack.empty() );
        // AssertionResult holds a pointer to a temporary DecomposedExpression,
        // which getExpandedExpression() calls to build the expression string.
        // Our section stack copy of the assertionResult will likely outlive the
        // temporary, so it must be expanded or discarded now to avoid calling
        // a destroyed object later.
        if ( m_shouldStoreFailedAssertions &&
             !assertionStats.assertionResult.isOk() ) {
            static_cast<void>(
                assertionStats.assertionResult.getExpandedExpression() );
        }
        if ( m_shouldStoreSuccesfulAssertions &&
             assertionStats.assertionResult.isOk() ) {
            static_cast<void>(
                assertionStats.assertionResult.getExpandedExpression() );
        }
        SectionNode& sectionNode = *m_sectionStack.back();
        sectionNode.assertionsAndBenchmarks.emplace_back( assertionStats );
    }

    void CumulativeReporterBase::sectionEnded( SectionStats const& sectionStats ) {
        assert( !m_sectionStack.empty() );
        SectionNode& node = *m_sectionStack.back();
        node.stats = sectionStats;
        m_sectionStack.pop_back();
    }

    void CumulativeReporterBase::testCaseEnded(
        TestCaseStats const& testCaseStats ) {
        auto node = Detail::make_unique<TestCaseNode>( testCaseStats );
        assert( m_sectionStack.size() == 0 );
        node->children.push_back( CATCH_MOVE(m_rootSection) );
        m_testCases.push_back( CATCH_MOVE(node) );

        assert( m_deepestSection );
        m_deepestSection->stdOut = testCaseStats.stdOut;
        m_deepestSection->stdErr = testCaseStats.stdErr;
    }


    void CumulativeReporterBase::testRunEnded( TestRunStats const& testRunStats ) {
        assert(!m_testRun && "CumulativeReporterBase assumes there can only be one test run");
        m_testRun = Detail::make_unique<TestRunNode>( testRunStats );
        m_testRun->children.swap( m_testCases );
        testRunEndedCumulative();
    }

    bool CumulativeReporterBase::SectionNode::hasAnyAssertions() const {
        return std::any_of(
            assertionsAndBenchmarks.begin(),
            assertionsAndBenchmarks.end(),
            []( Detail::AssertionOrBenchmarkResult const& res ) {
                return res.isAssertion();
            } );
    }

} // end namespace Catch




namespace Catch {

    void EventListenerBase::fatalErrorEncountered( StringRef ) {}

    void EventListenerBase::benchmarkPreparing( StringRef ) {}
    void EventListenerBase::benchmarkStarting( BenchmarkInfo const& ) {}
    void EventListenerBase::benchmarkEnded( BenchmarkStats<> const& ) {}
    void EventListenerBase::benchmarkFailed( StringRef ) {}

    void EventListenerBase::assertionStarting( AssertionInfo const& ) {}

    void EventListenerBase::assertionEnded( AssertionStats const& ) {}
    void EventListenerBase::listReporters(
        std::vector<ReporterDescription> const& ) {}
    void EventListenerBase::listListeners(
        std::vector<ListenerDescription> const& ) {}
    void EventListenerBase::listTests( std::vector<TestCaseHandle> const& ) {}
    void EventListenerBase::listTags( std::vector<TagInfo> const& ) {}
    void EventListenerBase::noMatchingTestCases( StringRef ) {}
    void EventListenerBase::reportInvalidTestSpec( StringRef ) {}
    void EventListenerBase::testRunStarting( TestRunInfo const& ) {}
    void EventListenerBase::testCaseStarting( TestCaseInfo const& ) {}
    void EventListenerBase::testCasePartialStarting(TestCaseInfo const&, uint64_t) {}
    void EventListenerBase::sectionStarting( SectionInfo const& ) {}
    void EventListenerBase::sectionEnded( SectionStats const& ) {}
    void EventListenerBase::testCasePartialEnded(TestCaseStats const&, uint64_t) {}
    void EventListenerBase::testCaseEnded( TestCaseStats const& ) {}
    void EventListenerBase::testRunEnded( TestRunStats const& ) {}
    void EventListenerBase::skipTest( TestCaseInfo const& ) {}
} // namespace Catch




#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <ostream>
#include <iomanip>

namespace Catch {

    namespace {
        void listTestNamesOnly(std::ostream& out,
                               std::vector<TestCaseHandle> const& tests) {
            for (auto const& test : tests) {
                auto const& testCaseInfo = test.getTestCaseInfo();

                if (startsWith(testCaseInfo.name, '#')) {
                    out << '"' << testCaseInfo.name << '"';
                } else {
                    out << testCaseInfo.name;
                }

                out << '\n';
            }
            out << std::flush;
        }
    } // end unnamed namespace


    // Because formatting using c++ streams is stateful, drop down to C is
    // required Alternatively we could use stringstream, but its performance
    // is... not good.
    std::string getFormattedDuration( double duration ) {
        // Max exponent + 1 is required to represent the whole part
        // + 1 for decimal point
        // + 3 for the 3 decimal places
        // + 1 for null terminator
        const std::size_t maxDoubleSize = DBL_MAX_10_EXP + 1 + 1 + 3 + 1;
        char buffer[maxDoubleSize];

        // Save previous errno, to prevent sprintf from overwriting it
        ErrnoGuard guard;
#ifdef _MSC_VER
        size_t printedLength = static_cast<size_t>(
            sprintf_s( buffer, "%.3f", duration ) );
#else
        size_t printedLength = static_cast<size_t>(
            std::snprintf( buffer, maxDoubleSize, "%.3f", duration ) );
#endif
        return std::string( buffer, printedLength );
    }

    bool shouldShowDuration( IConfig const& config, double duration ) {
        if ( config.showDurations() == ShowDurations::Always ) {
            return true;
        }
        if ( config.showDurations() == ShowDurations::Never ) {
            return false;
        }
        const double min = config.minDuration();
        return min >= 0 && duration >= min;
    }

    std::string serializeFilters( std::vector<std::string> const& filters ) {
        // We add a ' ' separator between each filter
        size_t serialized_size = filters.size() - 1;
        for (auto const& filter : filters) {
            serialized_size += filter.size();
        }

        std::string serialized;
        serialized.reserve(serialized_size);
        bool first = true;

        for (auto const& filter : filters) {
            if (!first) {
                serialized.push_back(' ');
            }
            first = false;
            serialized.append(filter);
        }

        return serialized;
    }

    std::ostream& operator<<( std::ostream& out, lineOfChars value ) {
        for ( size_t idx = 0; idx < CATCH_CONFIG_CONSOLE_WIDTH - 1; ++idx ) {
            out.put( value.c );
        }
        return out;
    }

    void
    defaultListReporters( std::ostream& out,
                          std::vector<ReporterDescription> const& descriptions,
                          Verbosity verbosity ) {
        out << "Available reporters:\n";
        const auto maxNameLen =
            std::max_element( descriptions.begin(),
                              descriptions.end(),
                              []( ReporterDescription const& lhs,
                                  ReporterDescription const& rhs ) {
                                  return lhs.name.size() < rhs.name.size();
                              } )
                ->name.size();

        for ( auto const& desc : descriptions ) {
            if ( verbosity == Verbosity::Quiet ) {
                out << TextFlow::Column( desc.name )
                           .indent( 2 )
                           .width( 5 + maxNameLen )
                    << '\n';
            } else {
                out << TextFlow::Column( desc.name + ':' )
                               .indent( 2 )
                               .width( 5 + maxNameLen ) +
                           TextFlow::Column( desc.description )
                               .initialIndent( 0 )
                               .indent( 2 )
                               .width( CATCH_CONFIG_CONSOLE_WIDTH - maxNameLen - 8 )
                    << '\n';
            }
        }
        out << '\n' << std::flush;
    }

    void defaultListListeners( std::ostream& out,
                               std::vector<ListenerDescription> const& descriptions ) {
        out << "Registered listeners:\n";

        if(descriptions.empty()) {
            return;
        }

        const auto maxNameLen =
            std::max_element( descriptions.begin(),
                              descriptions.end(),
                              []( ListenerDescription const& lhs,
                                  ListenerDescription const& rhs ) {
                                  return lhs.name.size() < rhs.name.size();
                              } )
                ->name.size();

        for ( auto const& desc : descriptions ) {
            out << TextFlow::Column( static_cast<std::string>( desc.name ) +
                                     ':' )
                           .indent( 2 )
                           .width( maxNameLen + 5 ) +
                       TextFlow::Column( desc.description )
                           .initialIndent( 0 )
                           .indent( 2 )
                           .width( CATCH_CONFIG_CONSOLE_WIDTH - maxNameLen - 8 )
                << '\n';
        }

        out << '\n' << std::flush;
    }

    void defaultListTags( std::ostream& out,
                          std::vector<TagInfo> const& tags,
                          bool isFiltered ) {
        if ( isFiltered ) {
            out << "Tags for matching test cases:\n";
        } else {
            out << "All available tags:\n";
        }

        // minimum whitespace to pad tag counts, possibly overwritten below
        size_t maxTagCountLen = 2;

        // determine necessary padding for tag count column
        if ( ! tags.empty() ) {
            const auto maxTagCount =
                std::max_element( tags.begin(),
                                  tags.end(),
                                  []( auto const& lhs, auto const& rhs ) {
                                      return lhs.count < rhs.count;
                                  } )
                    ->count;
            
            // more padding necessary for 3+ digits
            if (maxTagCount >= 100) {
                auto numDigits = 1 + std::floor( std::log10( maxTagCount ) );
                maxTagCountLen = static_cast<size_t>( numDigits );
            }
        }

        for ( auto const& tagCount : tags ) {
            ReusableStringStream rss;
            rss << "  " << std::setw( maxTagCountLen ) << tagCount.count << "  ";
            auto str = rss.str();
            auto wrapper = TextFlow::Column( tagCount.all() )
                               .initialIndent( 0 )
                               .indent( str.size() )
                               .width( CATCH_CONFIG_CONSOLE_WIDTH - 10 );
            out << str << wrapper << '\n';
        }
        out << pluralise(tags.size(), "tag"_sr) << "\n\n" << std::flush;
    }

    void defaultListTests(std::ostream& out, ColourImpl* streamColour, std::vector<TestCaseHandle> const& tests, bool isFiltered, Verbosity verbosity) {
        // We special case this to provide the equivalent of old
        // `--list-test-names-only`, which could then be used by the
        // `--input-file` option.
        if (verbosity == Verbosity::Quiet) {
            listTestNamesOnly(out, tests);
            return;
        }

        if (isFiltered) {
            out << "Matching test cases:\n";
        } else {
            out << "All available test cases:\n";
        }

        for (auto const& test : tests) {
            auto const& testCaseInfo = test.getTestCaseInfo();
            Colour::Code colour = testCaseInfo.isHidden()
                ? Colour::SecondaryText
                : Colour::None;
            auto colourGuard = streamColour->guardColour( colour ).engage( out );

            out << TextFlow::Column(testCaseInfo.name).indent(2) << '\n';
            if (verbosity >= Verbosity::High) {
                out << TextFlow::Column(Catch::Detail::stringify(testCaseInfo.lineInfo)).indent(4) << '\n';
            }
            if (!testCaseInfo.tags.empty() &&
                verbosity > Verbosity::Quiet) {
                out << TextFlow::Column(testCaseInfo.tagsAsString()).indent(6) << '\n';
            }
        }

        if (isFiltered) {
            out << pluralise(tests.size(), "matching test case"_sr);
        } else {
            out << pluralise(tests.size(), "test case"_sr);
        }
        out << "\n\n" << std::flush;
    }

    namespace {
        class SummaryColumn {
        public:
            SummaryColumn( std::string suffix, Colour::Code colour ):
                m_suffix( CATCH_MOVE( suffix ) ), m_colour( colour ) {}

            SummaryColumn&& addRow( std::uint64_t count ) && {
                std::string row = std::to_string(count);
                auto const new_width = std::max( m_width, row.size() );
                if ( new_width > m_width ) {
                    for ( auto& oldRow : m_rows ) {
                        oldRow.insert( 0, new_width - m_width, ' ' );
                    }
                } else {
                    row.insert( 0, m_width - row.size(), ' ' );
                }
                m_width = new_width;
                m_rows.push_back( row );
                return std::move( *this );
            }

            std::string const& getSuffix() const { return m_suffix; }
            Colour::Code getColour() const { return m_colour; }
            std::string const& getRow( std::size_t index ) const {
                return m_rows[index];
            }

        private:
            std::string m_suffix;
            Colour::Code m_colour;
            std::size_t m_width = 0;
            std::vector<std::string> m_rows;
        };

        void printSummaryRow( std::ostream& stream,
                              ColourImpl& colour,
                              StringRef label,
                              std::vector<SummaryColumn> const& cols,
                              std::size_t row ) {
            for ( auto const& col : cols ) {
                auto const& value = col.getRow( row );
                auto const& suffix = col.getSuffix();
                if ( suffix.empty() ) {
                    stream << label << ": ";
                    if ( value != "0" ) {
                        stream << value;
                    } else {
                        stream << colour.guardColour( Colour::Warning )
                               << "- none -";
                    }
                } else if ( value != "0" ) {
                    stream << colour.guardColour( Colour::LightGrey ) << " | "
                           << colour.guardColour( col.getColour() ) << value
                           << ' ' << suffix;
                }
            }
            stream << '\n';
        }
    } // namespace

    void printTestRunTotals( std::ostream& stream,
                             ColourImpl& streamColour,
                             Totals const& totals ) {
        if ( totals.testCases.total() == 0 ) {
            stream << streamColour.guardColour( Colour::Warning )
                   << "No tests ran\n";
            return;
        }

        if ( totals.assertions.total() > 0 && totals.testCases.allPassed() ) {
            stream << streamColour.guardColour( Colour::ResultSuccess )
                   << "All tests passed";
            stream << " ("
                   << pluralise( totals.assertions.passed, "assertion"_sr )
                   << " in "
                   << pluralise( totals.testCases.passed, "test case"_sr )
                   << ')' << '\n';
            return;
        }

        std::vector<SummaryColumn> columns;
        // Don't include "skipped assertions" in total count
        const auto totalAssertionCount =
            totals.assertions.total() - totals.assertions.skipped;
        columns.push_back( SummaryColumn( "", Colour::None )
                               .addRow( totals.testCases.total() )
                               .addRow( totalAssertionCount ) );
        columns.push_back( SummaryColumn( "passed", Colour::Success )
                               .addRow( totals.testCases.passed )
                               .addRow( totals.assertions.passed ) );
        columns.push_back( SummaryColumn( "failed", Colour::ResultError )
                               .addRow( totals.testCases.failed )
                               .addRow( totals.assertions.failed ) );
        columns.push_back( SummaryColumn( "skipped", Colour::Skip )
                               .addRow( totals.testCases.skipped )
                               // Don't print "skipped assertions"
                               .addRow( 0 ) );
        columns.push_back(
            SummaryColumn( "failed as expected", Colour::ResultExpectedFailure )
                .addRow( totals.testCases.failedButOk )
                .addRow( totals.assertions.failedButOk ) );
        printSummaryRow( stream, streamColour, "test cases"_sr, columns, 0 );
        printSummaryRow( stream, streamColour, "assertions"_sr, columns, 1 );
    }

} // namespace Catch


//

namespace Catch {
    namespace {
        void writeSourceInfo( JsonObjectWriter& writer,
                              SourceLineInfo const& sourceInfo ) {
            auto source_location_writer =
                writer.write( "source-location"_sr ).writeObject();
            source_location_writer.write( "filename"_sr )
                .write( sourceInfo.file );
            source_location_writer.write( "line"_sr ).write( sourceInfo.line );
        }

        void writeTags( JsonArrayWriter writer, std::vector<Tag> const& tags ) {
            for ( auto const& tag : tags ) {
                writer.write( tag.original );
            }
        }

        void writeProperties( JsonArrayWriter writer,
                              TestCaseInfo const& info ) {
            if ( info.isHidden() ) { writer.write( "is-hidden"_sr ); }
            if ( info.okToFail() ) { writer.write( "ok-to-fail"_sr ); }
            if ( info.expectedToFail() ) {
                writer.write( "expected-to-fail"_sr );
            }
            if ( info.throws() ) { writer.write( "throws"_sr ); }
        }

    } // namespace

    JsonReporter::JsonReporter( ReporterConfig&& config ):
        StreamingReporterBase{ CATCH_MOVE( config ) } {

        m_preferences.shouldRedirectStdOut = true;
        // TBD: Do we want to report all assertions? XML reporter does
        //      not, but for machine-parseable reporters I think the answer
        //      should be yes.
        m_preferences.shouldReportAllAssertions = true;
        // We only handle assertions when they end
        m_preferences.shouldReportAllAssertionStarts = false;

        m_objectWriters.emplace( m_stream );
        m_writers.emplace( Writer::Object );
        auto& writer = m_objectWriters.top();

        writer.write( "version"_sr ).write( 1 );

        {
            auto metadata_writer = writer.write( "metadata"_sr ).writeObject();
            metadata_writer.write( "name"_sr ).write( m_config->name() );
            metadata_writer.write( "rng-seed"_sr ).write( m_config->rngSeed() );
            metadata_writer.write( "catch2-version"_sr )
                .write( libraryVersion() );
            if ( m_config->testSpec().hasFilters() ) {
                metadata_writer.write( "filters"_sr )
                    .write( m_config->testSpec() );
            }
        }
    }

    JsonReporter::~JsonReporter() {
        endListing();
        // TODO: Ensure this closes the top level object, add asserts
        assert( m_writers.size() == 1 && "Only the top level object should be open" );
        assert( m_writers.top() == Writer::Object );
        endObject();
        m_stream << '\n' << std::flush;
        assert( m_writers.empty() );
    }

    JsonArrayWriter& JsonReporter::startArray() {
        m_arrayWriters.emplace( m_arrayWriters.top().writeArray() );
        m_writers.emplace( Writer::Array );
        return m_arrayWriters.top();
    }
    JsonArrayWriter& JsonReporter::startArray( StringRef key ) {
        m_arrayWriters.emplace(
            m_objectWriters.top().write( key ).writeArray() );
        m_writers.emplace( Writer::Array );
        return m_arrayWriters.top();
    }

    JsonObjectWriter& JsonReporter::startObject() {
        m_objectWriters.emplace( m_arrayWriters.top().writeObject() );
        m_writers.emplace( Writer::Object );
        return m_objectWriters.top();
    }
    JsonObjectWriter& JsonReporter::startObject( StringRef key ) {
        m_objectWriters.emplace(
            m_objectWriters.top().write( key ).writeObject() );
        m_writers.emplace( Writer::Object );
        return m_objectWriters.top();
    }

    void JsonReporter::endObject() {
        assert( isInside( Writer::Object ) );
        m_objectWriters.pop();
        m_writers.pop();
    }
    void JsonReporter::endArray() {
        assert( isInside( Writer::Array ) );
        m_arrayWriters.pop();
        m_writers.pop();
    }

    bool JsonReporter::isInside( Writer writer ) {
        return !m_writers.empty() && m_writers.top() == writer;
    }

    void JsonReporter::startListing() {
        if ( !m_startedListing ) { startObject( "listings"_sr ); }
        m_startedListing = true;
    }
    void JsonReporter::endListing() {
        if ( m_startedListing ) { endObject(); }
        m_startedListing = false;
    }

    std::string JsonReporter::getDescription() {
        return "Outputs listings as JSON. Test listing is Work-in-Progress!";
    }

    void JsonReporter::testRunStarting( TestRunInfo const& runInfo ) {
        StreamingReporterBase::testRunStarting( runInfo );
        endListing();

        assert( isInside( Writer::Object ) );
        startObject( "test-run"_sr );
        startArray( "test-cases"_sr );
    }

     static void writeCounts( JsonObjectWriter&& writer, Counts const& counts ) {
        writer.write( "passed"_sr ).write( counts.passed );
        writer.write( "failed"_sr ).write( counts.failed );
        writer.write( "fail-but-ok"_sr ).write( counts.failedButOk );
        writer.write( "skipped"_sr ).write( counts.skipped );
    }

    void JsonReporter::testRunEnded(TestRunStats const& runStats) {
        assert( isInside( Writer::Array ) );
        // End "test-cases"
        endArray();

        {
            auto totals =
                m_objectWriters.top().write( "totals"_sr ).writeObject();
            writeCounts( totals.write( "assertions"_sr ).writeObject(),
                         runStats.totals.assertions );
            writeCounts( totals.write( "test-cases"_sr ).writeObject(),
                         runStats.totals.testCases );
        }

        // End the "test-run" object
        endObject();
    }

    void JsonReporter::testCaseStarting( TestCaseInfo const& tcInfo ) {
        StreamingReporterBase::testCaseStarting( tcInfo );

        assert( isInside( Writer::Array ) &&
                "We should be in the 'test-cases' array" );
        startObject();
        // "test-info" prelude
        {
            auto testInfo =
                m_objectWriters.top().write( "test-info"_sr ).writeObject();
            // TODO: handle testName vs className!!
            testInfo.write( "name"_sr ).write( tcInfo.name );
            writeSourceInfo(testInfo, tcInfo.lineInfo);
            writeTags( testInfo.write( "tags"_sr ).writeArray(), tcInfo.tags );
            writeProperties( testInfo.write( "properties"_sr ).writeArray(),
                             tcInfo );
        }


        // Start the array for individual test runs (testCasePartial pairs)
        startArray( "runs"_sr );
    }

    void JsonReporter::testCaseEnded( TestCaseStats const& tcStats ) {
        StreamingReporterBase::testCaseEnded( tcStats );

        // We need to close the 'runs' array before finishing the test case
        assert( isInside( Writer::Array ) );
        endArray();

        {
            auto totals =
                m_objectWriters.top().write( "totals"_sr ).writeObject();
            writeCounts( totals.write( "assertions"_sr ).writeObject(),
                         tcStats.totals.assertions );
            // We do not write the test case totals, because there will always be just one test case here.
            // TODO: overall "result" -> success, skip, fail here? Or in partial result?
        }
        // We do not write out stderr/stdout, because we instead wrote those out in partial runs

        // TODO: aborting?

        // And we also close this test case's object
        assert( isInside( Writer::Object ) );
        endObject();
    }

    void JsonReporter::testCasePartialStarting( TestCaseInfo const& /*tcInfo*/,
                                                uint64_t index ) {
        startObject();
        m_objectWriters.top().write( "run-idx"_sr ).write( index );
        startArray( "path"_sr );
        // TODO: we want to delay most of the printing to the 'root' section
        // TODO: childSection key name?
    }

    void JsonReporter::testCasePartialEnded( TestCaseStats const& tcStats,
                                             uint64_t /*index*/ ) {
        // Fixme: the top level section handles this.
        //// path object
        endArray();
        if ( !tcStats.stdOut.empty() ) {
            m_objectWriters.top()
                .write( "captured-stdout"_sr )
                .write( tcStats.stdOut );
        }
        if ( !tcStats.stdErr.empty() ) {
            m_objectWriters.top()
                .write( "captured-stderr"_sr )
                .write( tcStats.stdErr );
        }
        {
            auto totals =
                m_objectWriters.top().write( "totals"_sr ).writeObject();
            writeCounts( totals.write( "assertions"_sr ).writeObject(),
                         tcStats.totals.assertions );
            // We do not write the test case totals, because there will
            // always be just one test case here.
            // TODO: overall "result" -> success, skip, fail here? Or in
            // partial result?
        }
        // TODO: aborting?
        // run object
        endObject();
    }

    void JsonReporter::sectionStarting( SectionInfo const& sectionInfo ) {
        assert( isInside( Writer::Array ) &&
                "Section should always start inside an object" );
        // We want to nest top level sections, even though it shares name
        // and source loc with the TEST_CASE
        auto& sectionObject = startObject();
        sectionObject.write( "kind"_sr ).write( "section"_sr );
        sectionObject.write( "name"_sr ).write( sectionInfo.name );
        writeSourceInfo( m_objectWriters.top(), sectionInfo.lineInfo );


        // TBD: Do we want to create this event lazily? It would become
        //      rather complex, but we could do it, and it would look
        //      better for empty sections. OTOH, empty sections should
        //      be rare.
        startArray( "path"_sr );
    }
    void JsonReporter::sectionEnded( SectionStats const& /*sectionStats */) {
        // End the subpath array
        endArray();
        // TODO: metadata
        // TODO: what info do we have here?

        // End the section object
        endObject();
    }

    void JsonReporter::assertionEnded( AssertionStats const& assertionStats ) {
        // TODO: There is lot of different things to handle here, but
        //       we can fill it in later, after we show that the basic
        //       outline and streaming reporter impl works well enough.
        //if ( !m_config->includeSuccessfulResults()
        //    && assertionStats.assertionResult.isOk() ) {
        //    return;
        //}
        assert( isInside( Writer::Array ) );
        auto assertionObject = m_arrayWriters.top().writeObject();

        assertionObject.write( "kind"_sr ).write( "assertion"_sr );
        writeSourceInfo( assertionObject,
                         assertionStats.assertionResult.getSourceInfo() );
        assertionObject.write( "status"_sr )
            .write( assertionStats.assertionResult.isOk() );
        // TODO: handling of result.
        // TODO: messages
        // TODO: totals?
    }


    void JsonReporter::benchmarkPreparing( StringRef name ) { (void)name; }
    void JsonReporter::benchmarkStarting( BenchmarkInfo const& ) {}
    void JsonReporter::benchmarkEnded( BenchmarkStats<> const& ) {}
    void JsonReporter::benchmarkFailed( StringRef error ) { (void)error; }

    void JsonReporter::listReporters(
        std::vector<ReporterDescription> const& descriptions ) {
        startListing();

        auto writer =
            m_objectWriters.top().write( "reporters"_sr ).writeArray();
        for ( auto const& desc : descriptions ) {
            auto desc_writer = writer.writeObject();
            desc_writer.write( "name"_sr ).write( desc.name );
            desc_writer.write( "description"_sr ).write( desc.description );
        }
    }
    void JsonReporter::listListeners(
        std::vector<ListenerDescription> const& descriptions ) {
        startListing();

        auto writer =
            m_objectWriters.top().write( "listeners"_sr ).writeArray();

        for ( auto const& desc : descriptions ) {
            auto desc_writer = writer.writeObject();
            desc_writer.write( "name"_sr ).write( desc.name );
            desc_writer.write( "description"_sr ).write( desc.description );
        }
    }
    void JsonReporter::listTests( std::vector<TestCaseHandle> const& tests ) {
        startListing();

        auto writer = m_objectWriters.top().write( "tests"_sr ).writeArray();

        for ( auto const& test : tests ) {
            auto desc_writer = writer.writeObject();
            auto const& info = test.getTestCaseInfo();

            desc_writer.write( "name"_sr ).write( info.name );
            desc_writer.write( "class-name"_sr ).write( info.className );
            {
                auto tag_writer = desc_writer.write( "tags"_sr ).writeArray();
                for ( auto const& tag : info.tags ) {
                    tag_writer.write( tag.original );
                }
            }
            writeSourceInfo( desc_writer, info.lineInfo );
        }
    }
    void JsonReporter::listTags( std::vector<TagInfo> const& tags ) {
        startListing();

        auto writer = m_objectWriters.top().write( "tags"_sr ).writeArray();
        for ( auto const& tag : tags ) {
            auto tag_writer = writer.writeObject();
            {
                auto aliases_writer =
                    tag_writer.write( "aliases"_sr ).writeArray();
                for ( auto alias : tag.spellings ) {
                    aliases_writer.write( alias );
                }
            }
            tag_writer.write( "count"_sr ).write( tag.count );
        }
    }
} // namespace Catch




#include <cassert>
#include <ctime>
#include <algorithm>
#include <iomanip>

namespace Catch {

    namespace {
        std::string getCurrentTimestamp() {
            time_t rawtime;
            std::time(&rawtime);

            std::tm timeInfo = {};
#if defined (_MSC_VER) || defined (__MINGW32__)
            gmtime_s(&timeInfo, &rawtime);
#elif defined (CATCH_PLATFORM_PLAYSTATION)
            gmtime_s(&rawtime, &timeInfo);
#elif defined (__IAR_SYSTEMS_ICC__)
            timeInfo = *std::gmtime(&rawtime);
#else
            gmtime_r(&rawtime, &timeInfo);
#endif

            auto const timeStampSize = sizeof("2017-01-16T17:06:45Z");
            char timeStamp[timeStampSize];
            const char * const fmt = "%Y-%m-%dT%H:%M:%SZ";

            std::strftime(timeStamp, timeStampSize, fmt, &timeInfo);

            return std::string(timeStamp, timeStampSize - 1);
        }

        std::string fileNameTag(std::vector<Tag> const& tags) {
            auto it = std::find_if(begin(tags),
                                   end(tags),
                                   [] (Tag const& tag) {
                                       return tag.original.size() > 0
                                           && tag.original[0] == '#'; });
            if (it != tags.end()) {
                return static_cast<std::string>(
                    it->original.substr(1, it->original.size() - 1)
                );
            }
            return std::string();
        }

        // Formats the duration in seconds to 3 decimal places.
        // This is done because some genius defined Maven Surefire schema
        // in a way that only accepts 3 decimal places, and tools like
        // Jenkins use that schema for validation JUnit reporter output.
        std::string formatDuration( double seconds ) {
            ReusableStringStream rss;
            rss << std::fixed << std::setprecision( 3 ) << seconds;
            return rss.str();
        }

        static void normalizeNamespaceMarkers(std::string& str) {
            std::size_t pos = str.find( "::" );
            while ( pos != std::string::npos ) {
                str.replace( pos, 2, "." );
                pos += 1;
                pos = str.find( "::", pos );
            }
        }

    } // anonymous namespace

    JunitReporter::JunitReporter( ReporterConfig&& _config )
        :   CumulativeReporterBase( CATCH_MOVE(_config) ),
            xml( m_stream )
        {
            m_preferences.shouldRedirectStdOut = true;
            m_preferences.shouldReportAllAssertions = false;
            m_preferences.shouldReportAllAssertionStarts = false;
            m_shouldStoreSuccesfulAssertions = false;
        }

    std::string JunitReporter::getDescription() {
        return "Reports test results in an XML format that looks like Ant's junitreport target";
    }

    void JunitReporter::testRunStarting( TestRunInfo const& runInfo )  {
        CumulativeReporterBase::testRunStarting( runInfo );
        xml.startElement( "testsuites" );
        suiteTimer.start();
        stdOutForSuite.clear();
        stdErrForSuite.clear();
        unexpectedExceptions = 0;
    }

    void JunitReporter::testCaseStarting( TestCaseInfo const& testCaseInfo ) {
        m_okToFail = testCaseInfo.okToFail();
    }

    void JunitReporter::assertionEnded( AssertionStats const& assertionStats ) {
        if( assertionStats.assertionResult.getResultType() == ResultWas::ThrewException && !m_okToFail )
            unexpectedExceptions++;
        CumulativeReporterBase::assertionEnded( assertionStats );
    }

    void JunitReporter::testCaseEnded( TestCaseStats const& testCaseStats ) {
        stdOutForSuite += testCaseStats.stdOut;
        stdErrForSuite += testCaseStats.stdErr;
        CumulativeReporterBase::testCaseEnded( testCaseStats );
    }

    void JunitReporter::testRunEndedCumulative() {
        const auto suiteTime = suiteTimer.getElapsedSeconds();
        writeRun( *m_testRun, suiteTime );
        xml.endElement();
    }

    void JunitReporter::writeRun( TestRunNode const& testRunNode, double suiteTime ) {
        XmlWriter::ScopedElement e = xml.scopedElement( "testsuite" );

        TestRunStats const& stats = testRunNode.value;
        xml.writeAttribute( "name"_sr, stats.runInfo.name );
        xml.writeAttribute( "errors"_sr, unexpectedExceptions );
        xml.writeAttribute( "failures"_sr, stats.totals.assertions.failed-unexpectedExceptions );
        xml.writeAttribute( "skipped"_sr, stats.totals.assertions.skipped );
        xml.writeAttribute( "tests"_sr, stats.totals.assertions.total() );
        xml.writeAttribute( "hostname"_sr, "tbd"_sr ); // !TBD
        if( m_config->showDurations() == ShowDurations::Never )
            xml.writeAttribute( "time"_sr, ""_sr );
        else
            xml.writeAttribute( "time"_sr, formatDuration( suiteTime ) );
        xml.writeAttribute( "timestamp"_sr, getCurrentTimestamp() );

        // Write properties
        {
            auto properties = xml.scopedElement("properties");
            xml.scopedElement("property")
                .writeAttribute("name"_sr, "random-seed"_sr)
                .writeAttribute("value"_sr, m_config->rngSeed());
            if (m_config->testSpec().hasFilters()) {
                xml.scopedElement("property")
                    .writeAttribute("name"_sr, "filters"_sr)
                    .writeAttribute("value"_sr, m_config->testSpec());
            }
        }

        // Write test cases
        for( auto const& child : testRunNode.children )
            writeTestCase( *child );

        xml.scopedElement( "system-out" ).writeText( trim( stdOutForSuite ), XmlFormatting::Newline );
        xml.scopedElement( "system-err" ).writeText( trim( stdErrForSuite ), XmlFormatting::Newline );
    }

    void JunitReporter::writeTestCase( TestCaseNode const& testCaseNode ) {
        TestCaseStats const& stats = testCaseNode.value;

        // All test cases have exactly one section - which represents the
        // test case itself. That section may have 0-n nested sections
        assert( testCaseNode.children.size() == 1 );
        SectionNode const& rootSection = *testCaseNode.children.front();

        std::string className =
            static_cast<std::string>( stats.testInfo->className );

        if( className.empty() ) {
            className = fileNameTag(stats.testInfo->tags);
            if ( className.empty() ) {
                className = "global";
            }
        }

        if ( !m_config->name().empty() )
            className = static_cast<std::string>(m_config->name()) + '.' + className;

        normalizeNamespaceMarkers(className);

        writeSection( className, "", rootSection, stats.testInfo->okToFail() );
    }

    void JunitReporter::writeSection( std::string const& className,
                                      std::string const& rootName,
                                      SectionNode const& sectionNode,
                                      bool testOkToFail) {
        std::string name = trim( sectionNode.stats.sectionInfo.name );
        if( !rootName.empty() )
            name = rootName + '/' + name;

        if ( sectionNode.stats.assertions.total() > 0
           || !sectionNode.stdOut.empty()
           || !sectionNode.stdErr.empty() ) {
            XmlWriter::ScopedElement e = xml.scopedElement( "testcase" );
            if( className.empty() ) {
                xml.writeAttribute( "classname"_sr, name );
                xml.writeAttribute( "name"_sr, "root"_sr );
            }
            else {
                xml.writeAttribute( "classname"_sr, className );
                xml.writeAttribute( "name"_sr, name );
            }
            xml.writeAttribute( "time"_sr, formatDuration( sectionNode.stats.durationInSeconds ) );
            // This is not ideal, but it should be enough to mimic gtest's
            // junit output.
            // Ideally the JUnit reporter would also handle `skipTest`
            // events and write those out appropriately.
            xml.writeAttribute( "status"_sr, "run"_sr );

            if (sectionNode.stats.assertions.failedButOk) {
                xml.scopedElement("skipped")
                    .writeAttribute("message", "TEST_CASE tagged with !mayfail");
            }

            writeAssertions( sectionNode );


            if( !sectionNode.stdOut.empty() )
                xml.scopedElement( "system-out" ).writeText( trim( sectionNode.stdOut ), XmlFormatting::Newline );
            if( !sectionNode.stdErr.empty() )
                xml.scopedElement( "system-err" ).writeText( trim( sectionNode.stdErr ), XmlFormatting::Newline );
        }
        for( auto const& childNode : sectionNode.childSections )
            if( className.empty() )
                writeSection( name, "", *childNode, testOkToFail );
            else
                writeSection( className, name, *childNode, testOkToFail );
    }

    void JunitReporter::writeAssertions( SectionNode const& sectionNode ) {
        for (auto const& assertionOrBenchmark : sectionNode.assertionsAndBenchmarks) {
            if (assertionOrBenchmark.isAssertion()) {
                writeAssertion(assertionOrBenchmark.asAssertion());
            }
        }
    }

    void JunitReporter::writeAssertion( AssertionStats const& stats ) {
        AssertionResult const& result = stats.assertionResult;
        if ( !result.isOk() ||
             result.getResultType() == ResultWas::ExplicitSkip ) {
            std::string elementName;
            switch( result.getResultType() ) {
                case ResultWas::ThrewException:
                case ResultWas::FatalErrorCondition:
                    elementName = "error";
                    break;
                case ResultWas::ExplicitFailure:
                case ResultWas::ExpressionFailed:
                case ResultWas::DidntThrowException:
                    elementName = "failure";
                    break;
                case ResultWas::ExplicitSkip:
                    elementName = "skipped";
                    break;
                // We should never see these here:
                case ResultWas::Info:
                case ResultWas::Warning:
                case ResultWas::Ok:
                case ResultWas::Unknown:
                case ResultWas::FailureBit:
                case ResultWas::Exception:
                    elementName = "internalError";
                    break;
            }

            XmlWriter::ScopedElement e = xml.scopedElement( elementName );

            xml.writeAttribute( "message"_sr, result.getExpression() );
            xml.writeAttribute( "type"_sr, result.getTestMacroName() );

            ReusableStringStream rss;
            if ( result.getResultType() == ResultWas::ExplicitSkip ) {
                rss << "SKIPPED\n";
            } else {
                rss << "FAILED" << ":\n";
                if (result.hasExpression()) {
                    rss << "  ";
                    rss << result.getExpressionInMacro();
                    rss << '\n';
                }
                if (result.hasExpandedExpression()) {
                    rss << "with expansion:\n";
                    rss << TextFlow::Column(result.getExpandedExpression()).indent(2) << '\n';
                }
            }

            if( result.hasMessage() )
                rss << result.getMessage() << '\n';
            for( auto const& msg : stats.infoMessages )
                if( msg.type == ResultWas::Info )
                    rss << msg.message << '\n';

            rss << "at " << result.getSourceInfo();
            xml.writeText( rss.str(), XmlFormatting::Newline );
        }
    }

} // end namespace Catch




#include <ostream>

namespace Catch {
    void MultiReporter::updatePreferences(IEventListener const& reporterish) {
        m_preferences.shouldRedirectStdOut |=
            reporterish.getPreferences().shouldRedirectStdOut;
        m_preferences.shouldReportAllAssertions |=
            reporterish.getPreferences().shouldReportAllAssertions;
        m_preferences.shouldReportAllAssertionStarts |=
            reporterish.getPreferences().shouldReportAllAssertionStarts;
    }

    void MultiReporter::addListener( IEventListenerPtr&& listener ) {
        updatePreferences(*listener);
        m_reporterLikes.insert(m_reporterLikes.begin() + m_insertedListeners, CATCH_MOVE(listener) );
        ++m_insertedListeners;
    }

    void MultiReporter::addReporter( IEventListenerPtr&& reporter ) {
        updatePreferences(*reporter);

        // We will need to output the captured stdout if there are reporters
        // that do not want it captured.
        // We do not consider listeners, because it is generally assumed that
        // listeners are output-transparent, even though they can ask for stdout
        // capture to do something with it.
        m_haveNoncapturingReporters |= !reporter->getPreferences().shouldRedirectStdOut;

        // Reporters can always be placed to the back without breaking the
        // reporting order
        m_reporterLikes.push_back( CATCH_MOVE( reporter ) );
    }

    void MultiReporter::noMatchingTestCases( StringRef unmatchedSpec ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->noMatchingTestCases( unmatchedSpec );
        }
    }

    void MultiReporter::fatalErrorEncountered( StringRef error ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->fatalErrorEncountered( error );
        }
    }

    void MultiReporter::reportInvalidTestSpec( StringRef arg ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->reportInvalidTestSpec( arg );
        }
    }

    void MultiReporter::benchmarkPreparing( StringRef name ) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->benchmarkPreparing(name);
        }
    }
    void MultiReporter::benchmarkStarting( BenchmarkInfo const& benchmarkInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->benchmarkStarting( benchmarkInfo );
        }
    }
    void MultiReporter::benchmarkEnded( BenchmarkStats<> const& benchmarkStats ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->benchmarkEnded( benchmarkStats );
        }
    }

    void MultiReporter::benchmarkFailed( StringRef error ) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->benchmarkFailed(error);
        }
    }

    void MultiReporter::testRunStarting( TestRunInfo const& testRunInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testRunStarting( testRunInfo );
        }
    }

    void MultiReporter::testCaseStarting( TestCaseInfo const& testInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testCaseStarting( testInfo );
        }
    }

    void
    MultiReporter::testCasePartialStarting( TestCaseInfo const& testInfo,
                                                uint64_t partNumber ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testCasePartialStarting( testInfo, partNumber );
        }
    }

    void MultiReporter::sectionStarting( SectionInfo const& sectionInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->sectionStarting( sectionInfo );
        }
    }

    void MultiReporter::assertionStarting( AssertionInfo const& assertionInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->assertionStarting( assertionInfo );
        }
    }

    void MultiReporter::assertionEnded( AssertionStats const& assertionStats ) {
        const bool reportByDefault =
            assertionStats.assertionResult.getResultType() != ResultWas::Ok ||
            m_config->includeSuccessfulResults();

        for ( auto & reporterish : m_reporterLikes ) {
            if ( reportByDefault ||
                 reporterish->getPreferences().shouldReportAllAssertions ) {
                    reporterish->assertionEnded( assertionStats );
            }
        }
    }

    void MultiReporter::sectionEnded( SectionStats const& sectionStats ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->sectionEnded( sectionStats );
        }
    }

    void MultiReporter::testCasePartialEnded( TestCaseStats const& testStats,
                                                  uint64_t partNumber ) {
        if ( m_preferences.shouldRedirectStdOut &&
             m_haveNoncapturingReporters ) {
            if ( !testStats.stdOut.empty() ) {
                Catch::cout() << testStats.stdOut << std::flush;
            }
            if ( !testStats.stdErr.empty() ) {
                Catch::cerr() << testStats.stdErr << std::flush;
            }
        }

        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testCasePartialEnded( testStats, partNumber );
        }
    }

    void MultiReporter::testCaseEnded( TestCaseStats const& testCaseStats ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testCaseEnded( testCaseStats );
        }
    }

    void MultiReporter::testRunEnded( TestRunStats const& testRunStats ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->testRunEnded( testRunStats );
        }
    }


    void MultiReporter::skipTest( TestCaseInfo const& testInfo ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->skipTest( testInfo );
        }
    }

    void MultiReporter::listReporters(std::vector<ReporterDescription> const& descriptions) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->listReporters(descriptions);
        }
    }

    void MultiReporter::listListeners(
        std::vector<ListenerDescription> const& descriptions ) {
        for ( auto& reporterish : m_reporterLikes ) {
            reporterish->listListeners( descriptions );
        }
    }

    void MultiReporter::listTests(std::vector<TestCaseHandle> const& tests) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->listTests(tests);
        }
    }

    void MultiReporter::listTags(std::vector<TagInfo> const& tags) {
        for (auto& reporterish : m_reporterLikes) {
            reporterish->listTags(tags);
        }
    }

} // end namespace Catch





namespace Catch {
    namespace Detail {

        void registerReporterImpl( std::string const& name,
                                   IReporterFactoryPtr reporterPtr ) {
            CATCH_TRY {
                getMutableRegistryHub().registerReporter(
                    name, CATCH_MOVE( reporterPtr ) );
            }
            CATCH_CATCH_ALL {
                // Do not throw when constructing global objects, instead
                // register the exception to be processed later
                getMutableRegistryHub().registerStartupException();
            }
        }

        void registerListenerImpl( Detail::unique_ptr<EventListenerFactory> listenerFactory ) {
            getMutableRegistryHub().registerListener( CATCH_MOVE(listenerFactory) );
        }


    } // namespace Detail
} // namespace Catch




#include <map>

namespace Catch {

    namespace {
        std::string createMetadataString(IConfig const& config) {
            ReusableStringStream sstr;
            if ( config.testSpec().hasFilters() ) {
                sstr << "filters='"
                         << config.testSpec()
                         << "' ";
            }
            sstr << "rng-seed=" << config.rngSeed();
            return sstr.str();
        }
    }

    void SonarQubeReporter::testRunStarting(TestRunInfo const& testRunInfo) {
        CumulativeReporterBase::testRunStarting(testRunInfo);

        xml.writeComment( createMetadataString( *m_config ) );
        xml.startElement("testExecutions");
        xml.writeAttribute("version"_sr, '1');
    }

    void SonarQubeReporter::writeRun( TestRunNode const& runNode ) {
        std::map<StringRef, std::vector<TestCaseNode const*>> testsPerFile;

        for ( auto const& child : runNode.children ) {
            testsPerFile[child->value.testInfo->lineInfo.file].push_back(
                child.get() );
        }

        for ( auto const& kv : testsPerFile ) {
            writeTestFile( kv.first, kv.second );
        }
    }

    void SonarQubeReporter::writeTestFile(StringRef filename, std::vector<TestCaseNode const*> const& testCaseNodes) {
        XmlWriter::ScopedElement e = xml.scopedElement("file");
        xml.writeAttribute("path"_sr, filename);

        for (auto const& child : testCaseNodes)
            writeTestCase(*child);
    }

    void SonarQubeReporter::writeTestCase(TestCaseNode const& testCaseNode) {
        // All test cases have exactly one section - which represents the
        // test case itself. That section may have 0-n nested sections
        assert(testCaseNode.children.size() == 1);
        SectionNode const& rootSection = *testCaseNode.children.front();
        writeSection("", rootSection, testCaseNode.value.testInfo->okToFail());
    }

    void SonarQubeReporter::writeSection(std::string const& rootName, SectionNode const& sectionNode, bool okToFail) {
        std::string name = trim(sectionNode.stats.sectionInfo.name);
        if (!rootName.empty())
            name = rootName + '/' + name;

        if ( sectionNode.stats.assertions.total() > 0
            || !sectionNode.stdOut.empty()
            || !sectionNode.stdErr.empty() ) {
            XmlWriter::ScopedElement e = xml.scopedElement("testCase");
            xml.writeAttribute("name"_sr, name);
            xml.writeAttribute("duration"_sr, static_cast<long>(sectionNode.stats.durationInSeconds * 1000));

            writeAssertions(sectionNode, okToFail);
        }

        for (auto const& childNode : sectionNode.childSections)
            writeSection(name, *childNode, okToFail);
    }

    void SonarQubeReporter::writeAssertions(SectionNode const& sectionNode, bool okToFail) {
        for (auto const& assertionOrBenchmark : sectionNode.assertionsAndBenchmarks) {
            if (assertionOrBenchmark.isAssertion()) {
                writeAssertion(assertionOrBenchmark.asAssertion(), okToFail);
            }
        }
    }

    void SonarQubeReporter::writeAssertion(AssertionStats const& stats, bool okToFail) {
        AssertionResult const& result = stats.assertionResult;
        if ( !result.isOk() ||
             result.getResultType() == ResultWas::ExplicitSkip ) {
            std::string elementName;
            if (okToFail) {
                elementName = "skipped";
            } else {
                switch (result.getResultType()) {
                case ResultWas::ThrewException:
                case ResultWas::FatalErrorCondition:
                    elementName = "error";
                    break;
                case ResultWas::ExplicitFailure:
                case ResultWas::ExpressionFailed:
                case ResultWas::DidntThrowException:
                    elementName = "failure";
                    break;
                case ResultWas::ExplicitSkip:
                    elementName = "skipped";
                    break;
                    // We should never see these here:
                case ResultWas::Info:
                case ResultWas::Warning:
                case ResultWas::Ok:
                case ResultWas::Unknown:
                case ResultWas::FailureBit:
                case ResultWas::Exception:
                    elementName = "internalError";
                    break;
                }
            }

            XmlWriter::ScopedElement e = xml.scopedElement(elementName);

            ReusableStringStream messageRss;
            messageRss << result.getTestMacroName() << '(' << result.getExpression() << ')';
            xml.writeAttribute("message"_sr, messageRss.str());

            ReusableStringStream textRss;
            if ( result.getResultType() == ResultWas::ExplicitSkip ) {
                textRss << "SKIPPED\n";
            } else {
                textRss << "FAILED:\n";
                if (result.hasExpression()) {
                    textRss << '\t' << result.getExpressionInMacro() << '\n';
                }
                if (result.hasExpandedExpression()) {
                    textRss << "with expansion:\n\t" << result.getExpandedExpression() << '\n';
                }
            }

            if (result.hasMessage())
                textRss << result.getMessage() << '\n';

            for (auto const& msg : stats.infoMessages)
                if (msg.type == ResultWas::Info)
                    textRss << msg.message << '\n';

            textRss << "at " << result.getSourceInfo();
            xml.writeText(textRss.str(), XmlFormatting::Newline);
        }
    }

} // end namespace Catch



namespace Catch {

    StreamingReporterBase::~StreamingReporterBase() = default;

    void
    StreamingReporterBase::testRunStarting( TestRunInfo const& _testRunInfo ) {
        currentTestRunInfo = _testRunInfo;
    }

    void StreamingReporterBase::testRunEnded( TestRunStats const& ) {
        currentTestCaseInfo = nullptr;
    }

} // end namespace Catch



#include <algorithm>
#include <ostream>

namespace Catch {

    namespace {
        // Yes, this has to be outside the class and namespaced by naming.
        // Making older compiler happy is hard.
        static constexpr StringRef tapFailedString = "not ok"_sr;
        static constexpr StringRef tapPassedString = "ok"_sr;
        static constexpr Colour::Code tapDimColour = Colour::FileName;

        class TapAssertionPrinter {
        public:
            TapAssertionPrinter& operator= (TapAssertionPrinter const&) = delete;
            TapAssertionPrinter(TapAssertionPrinter const&) = delete;
            TapAssertionPrinter(std::ostream& _stream, AssertionStats const& _stats, std::size_t _counter, ColourImpl* colour_)
                : stream(_stream)
                , result(_stats.assertionResult)
                , messages(_stats.infoMessages)
                , itMessage(_stats.infoMessages.begin())
                , printInfoMessages(true)
                , counter(_counter)
                , colourImpl( colour_ ) {}

            void print() {
                itMessage = messages.begin();

                switch (result.getResultType()) {
                case ResultWas::Ok:
                    printResultType(tapPassedString);
                    printOriginalExpression();
                    printReconstructedExpression();
                    if (!result.hasExpression())
                        printRemainingMessages(Colour::None);
                    else
                        printRemainingMessages();
                    break;
                case ResultWas::ExpressionFailed:
                    if (result.isOk()) {
                        printResultType(tapPassedString);
                    } else {
                        printResultType(tapFailedString);
                    }
                    printOriginalExpression();
                    printReconstructedExpression();
                    if (result.isOk()) {
                        printIssue(" # TODO");
                    }
                    printRemainingMessages();
                    break;
                case ResultWas::ThrewException:
                    printResultType(tapFailedString);
                    printIssue("unexpected exception with message:"_sr);
                    printMessage();
                    printExpressionWas();
                    printRemainingMessages();
                    break;
                case ResultWas::FatalErrorCondition:
                    printResultType(tapFailedString);
                    printIssue("fatal error condition with message:"_sr);
                    printMessage();
                    printExpressionWas();
                    printRemainingMessages();
                    break;
                case ResultWas::DidntThrowException:
                    printResultType(tapFailedString);
                    printIssue("expected exception, got none"_sr);
                    printExpressionWas();
                    printRemainingMessages();
                    break;
                case ResultWas::Info:
                    printResultType("info"_sr);
                    printMessage();
                    printRemainingMessages();
                    break;
                case ResultWas::Warning:
                    printResultType("warning"_sr);
                    printMessage();
                    printRemainingMessages();
                    break;
                case ResultWas::ExplicitFailure:
                    printResultType(tapFailedString);
                    printIssue("explicitly"_sr);
                    printRemainingMessages(Colour::None);
                    break;
                case ResultWas::ExplicitSkip:
                    printResultType(tapPassedString);
                    printIssue(" # SKIP"_sr);
                    printMessage();
                    printRemainingMessages();
                    break;
                    // These cases are here to prevent compiler warnings
                case ResultWas::Unknown:
                case ResultWas::FailureBit:
                case ResultWas::Exception:
                    printResultType("** internal error **"_sr);
                    break;
                }
            }

        private:
            void printResultType(StringRef passOrFail) const {
                if (!passOrFail.empty()) {
                    stream << passOrFail << ' ' << counter << " -";
                }
            }

            void printIssue(StringRef issue) const {
                stream << ' ' << issue;
            }

            void printExpressionWas() {
                if (result.hasExpression()) {
                    stream << ';';
                    stream << colourImpl->guardColour( tapDimColour )
                           << " expression was:";
                    printOriginalExpression();
                }
            }

            void printOriginalExpression() const {
                if (result.hasExpression()) {
                    stream << ' ' << result.getExpression();
                }
            }

            void printReconstructedExpression() const {
                if (result.hasExpandedExpression()) {
                    stream << colourImpl->guardColour( tapDimColour ) << " for: ";

                    std::string expr = result.getExpandedExpression();
                    std::replace(expr.begin(), expr.end(), '\n', ' ');
                    stream << expr;
                }
            }

            void printMessage() {
                if (itMessage != messages.end()) {
                    stream << " '" << itMessage->message << '\'';
                    ++itMessage;
                }
            }

            void printRemainingMessages(Colour::Code colour = tapDimColour) {
                if (itMessage == messages.end()) {
                    return;
                }

                // using messages.end() directly (or auto) yields compilation error:
                std::vector<MessageInfo>::const_iterator itEnd = messages.end();
                const std::size_t N = static_cast<std::size_t>(itEnd - itMessage);

                stream << colourImpl->guardColour( colour ) << " with "
                       << pluralise( N, "message"_sr ) << ':';

                for (; itMessage != itEnd; ) {
                    // If this assertion is a warning ignore any INFO messages
                    if (printInfoMessages || itMessage->type != ResultWas::Info) {
                        stream << " '" << itMessage->message << '\'';
                        if (++itMessage != itEnd) {
                            stream << colourImpl->guardColour(tapDimColour) << " and";
                        }
                    }
                }
            }

        private:
            std::ostream& stream;
            AssertionResult const& result;
            std::vector<MessageInfo> const& messages;
            std::vector<MessageInfo>::const_iterator itMessage;
            bool printInfoMessages;
            std::size_t counter;
            ColourImpl* colourImpl;
        };

    } // End anonymous namespace

    void TAPReporter::testRunStarting( TestRunInfo const& ) {
        if ( m_config->testSpec().hasFilters() ) {
            m_stream << "# filters: " << m_config->testSpec() << '\n';
        }
        m_stream << "# rng-seed: " << m_config->rngSeed() << '\n'
                 << std::flush;
    }

    void TAPReporter::noMatchingTestCases( StringRef unmatchedSpec ) {
        m_stream << "# No test cases matched '" << unmatchedSpec << "'\n";
    }

    void TAPReporter::assertionEnded(AssertionStats const& _assertionStats) {
        ++counter;

        m_stream << "# " << currentTestCaseInfo->name << '\n';
        TapAssertionPrinter printer(m_stream, _assertionStats, counter, m_colour.get());
        printer.print();

        m_stream << '\n' << std::flush;
    }

    void TAPReporter::testRunEnded(TestRunStats const& _testRunStats) {
        m_stream << "1.." << _testRunStats.totals.assertions.total();
        if (_testRunStats.totals.testCases.total() == 0) {
            m_stream << " # Skipped: No tests ran.";
        }
        m_stream << "\n\n" << std::flush;
        StreamingReporterBase::testRunEnded(_testRunStats);
    }




} // end namespace Catch




#include <cassert>
#include <ostream>

namespace Catch {

    namespace {
        // if string has a : in first line will set indent to follow it on
        // subsequent lines
        void printHeaderString(std::ostream& os, std::string const& _string, std::size_t indent = 0) {
            std::size_t i = _string.find(": ");
            if (i != std::string::npos)
                i += 2;
            else
                i = 0;
            os << TextFlow::Column(_string)
                  .indent(indent + i)
                  .initialIndent(indent) << '\n';
        }

        std::string escape(StringRef str) {
            std::string escaped = static_cast<std::string>(str);
            replaceInPlace(escaped, "|", "||");
            replaceInPlace(escaped, "'", "|'");
            replaceInPlace(escaped, "\n", "|n");
            replaceInPlace(escaped, "\r", "|r");
            replaceInPlace(escaped, "[", "|[");
            replaceInPlace(escaped, "]", "|]");
            return escaped;
        }
    } // end anonymous namespace


    TeamCityReporter::~TeamCityReporter() = default;

    void TeamCityReporter::testRunStarting( TestRunInfo const& runInfo ) {
        m_stream << "##teamcity[testSuiteStarted name='" << escape( runInfo.name )
               << "']\n";
    }

    void TeamCityReporter::testRunEnded( TestRunStats const& runStats ) {
        m_stream << "##teamcity[testSuiteFinished name='"
               << escape( runStats.runInfo.name ) << "']\n";
    }

    void TeamCityReporter::assertionEnded(AssertionStats const& assertionStats) {
        AssertionResult const& result = assertionStats.assertionResult;
        if ( !result.isOk() ||
             result.getResultType() == ResultWas::ExplicitSkip ) {

            ReusableStringStream msg;
            if (!m_headerPrintedForThisSection)
                printSectionHeader(msg.get());
            m_headerPrintedForThisSection = true;

            msg << result.getSourceInfo() << '\n';

            switch (result.getResultType()) {
            case ResultWas::ExpressionFailed:
                msg << "expression failed";
                break;
            case ResultWas::ThrewException:
                msg << "unexpected exception";
                break;
            case ResultWas::FatalErrorCondition:
                msg << "fatal error condition";
                break;
            case ResultWas::DidntThrowException:
                msg << "no exception was thrown where one was expected";
                break;
            case ResultWas::ExplicitFailure:
                msg << "explicit failure";
                break;
            case ResultWas::ExplicitSkip:
                msg << "explicit skip";
                break;

                // We shouldn't get here because of the isOk() test
            case ResultWas::Ok:
            case ResultWas::Info:
            case ResultWas::Warning:
                CATCH_ERROR("Internal error in TeamCity reporter");
                // These cases are here to prevent compiler warnings
            case ResultWas::Unknown:
            case ResultWas::FailureBit:
            case ResultWas::Exception:
                CATCH_ERROR("Not implemented");
            }
            if (assertionStats.infoMessages.size() == 1)
                msg << " with message:";
            if (assertionStats.infoMessages.size() > 1)
                msg << " with messages:";
            for (auto const& messageInfo : assertionStats.infoMessages)
                msg << "\n  \"" << messageInfo.message << '"';


            if (result.hasExpression()) {
                msg <<
                    "\n  " << result.getExpressionInMacro() << "\n"
                    "with expansion:\n"
                    "  " << result.getExpandedExpression() << '\n';
            }

            if ( result.getResultType() == ResultWas::ExplicitSkip ) {
                m_stream << "##teamcity[testIgnored";
            } else if ( currentTestCaseInfo->okToFail() ) {
                msg << "- failure ignore as test marked as 'ok to fail'\n";
                m_stream << "##teamcity[testIgnored";
            } else {
                m_stream << "##teamcity[testFailed";
            }
            m_stream << " name='" << escape( currentTestCaseInfo->name ) << '\''
                     << " message='" << escape( msg.str() ) << '\'' << "]\n";
        }
        m_stream.flush();
    }

    void TeamCityReporter::testCaseStarting(TestCaseInfo const& testInfo) {
        m_testTimer.start();
        StreamingReporterBase::testCaseStarting(testInfo);
        m_stream << "##teamcity[testStarted name='"
            << escape(testInfo.name) << "']\n";
        m_stream.flush();
    }

    void TeamCityReporter::testCaseEnded(TestCaseStats const& testCaseStats) {
        StreamingReporterBase::testCaseEnded(testCaseStats);
        auto const& testCaseInfo = *testCaseStats.testInfo;
        if (!testCaseStats.stdOut.empty())
            m_stream << "##teamcity[testStdOut name='"
            << escape(testCaseInfo.name)
            << "' out='" << escape(testCaseStats.stdOut) << "']\n";
        if (!testCaseStats.stdErr.empty())
            m_stream << "##teamcity[testStdErr name='"
            << escape(testCaseInfo.name)
            << "' out='" << escape(testCaseStats.stdErr) << "']\n";
        m_stream << "##teamcity[testFinished name='"
            << escape(testCaseInfo.name) << "' duration='"
            << m_testTimer.getElapsedMilliseconds() << "']\n";
        m_stream.flush();
    }

    void TeamCityReporter::printSectionHeader(std::ostream& os) {
        assert(!m_sectionStack.empty());

        if (m_sectionStack.size() > 1) {
            os << lineOfChars('-') << '\n';

            std::vector<SectionInfo>::const_iterator
                it = m_sectionStack.begin() + 1, // Skip first section (test case)
                itEnd = m_sectionStack.end();
            for (; it != itEnd; ++it)
                printHeaderString(os, it->name);
            os << lineOfChars('-') << '\n';
        }

        SourceLineInfo lineInfo = m_sectionStack.front().lineInfo;

        os << lineInfo << '\n';
        os << lineOfChars('.') << "\n\n";
    }

} // end namespace Catch




#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4061) // Not all labels are EXPLICITLY handled in switch
                              // Note that 4062 (not all labels are handled
                              // and default is missing) is enabled
#endif

namespace Catch {
    XmlReporter::XmlReporter( ReporterConfig&& _config )
    :   StreamingReporterBase( CATCH_MOVE(_config) ),
        m_xml(m_stream)
    {
        m_preferences.shouldRedirectStdOut = true;
        m_preferences.shouldReportAllAssertions = true;
        m_preferences.shouldReportAllAssertionStarts = false;
    }

    XmlReporter::~XmlReporter() = default;

    std::string XmlReporter::getDescription() {
        return "Reports test results as an XML document";
    }

    std::string XmlReporter::getStylesheetRef() const {
        return std::string();
    }

    void XmlReporter::writeSourceInfo( SourceLineInfo const& sourceInfo ) {
        m_xml
            .writeAttribute( "filename"_sr, sourceInfo.file )
            .writeAttribute( "line"_sr, sourceInfo.line );
    }

    void XmlReporter::testRunStarting( TestRunInfo const& testInfo ) {
        StreamingReporterBase::testRunStarting( testInfo );
        std::string stylesheetRef = getStylesheetRef();
        if( !stylesheetRef.empty() )
            m_xml.writeStylesheetRef( stylesheetRef );
        m_xml.startElement("Catch2TestRun")
             .writeAttribute("name"_sr, m_config->name())
             .writeAttribute("rng-seed"_sr, m_config->rngSeed())
             .writeAttribute("xml-format-version"_sr, 3)
             .writeAttribute("catch2-version"_sr, libraryVersion());
        if ( m_config->testSpec().hasFilters() ) {
            m_xml.writeAttribute( "filters"_sr, m_config->testSpec() );
        }
    }

    void XmlReporter::testCaseStarting( TestCaseInfo const& testInfo ) {
        StreamingReporterBase::testCaseStarting(testInfo);
        m_xml.startElement( "TestCase" )
            .writeAttribute( "name"_sr, trim( StringRef(testInfo.name) ) )
            .writeAttribute( "tags"_sr, testInfo.tagsAsString() );

        writeSourceInfo( testInfo.lineInfo );

        if ( m_config->showDurations() == ShowDurations::Always )
            m_testCaseTimer.start();
        m_xml.ensureTagClosed();
    }

    void XmlReporter::sectionStarting( SectionInfo const& sectionInfo ) {
        StreamingReporterBase::sectionStarting( sectionInfo );
        if( m_sectionDepth++ > 0 ) {
            m_xml.startElement( "Section" )
                .writeAttribute( "name"_sr, trim( StringRef(sectionInfo.name) ) );
            writeSourceInfo( sectionInfo.lineInfo );
            m_xml.ensureTagClosed();
        }
    }

    void XmlReporter::assertionEnded( AssertionStats const& assertionStats ) {

        AssertionResult const& result = assertionStats.assertionResult;

        bool includeResults = m_config->includeSuccessfulResults() || !result.isOk();

        if( includeResults || result.getResultType() == ResultWas::Warning ) {
            // Print any info messages in <Info> tags.
            for( auto const& msg : assertionStats.infoMessages ) {
                if( msg.type == ResultWas::Info && includeResults ) {
                    auto t = m_xml.scopedElement( "Info" );
                    writeSourceInfo( msg.lineInfo );
                    t.writeText( msg.message );
                } else if ( msg.type == ResultWas::Warning ) {
                    auto t = m_xml.scopedElement( "Warning" );
                    writeSourceInfo( msg.lineInfo );
                    t.writeText( msg.message );
                }
            }
        }

        // Drop out if result was successful but we're not printing them.
        if ( !includeResults && result.getResultType() != ResultWas::Warning &&
             result.getResultType() != ResultWas::ExplicitSkip ) {
            return;
        }

        // Print the expression if there is one.
        if( result.hasExpression() ) {
            m_xml.startElement( "Expression" )
                .writeAttribute( "success"_sr, result.succeeded() )
                .writeAttribute( "type"_sr, result.getTestMacroName() );

            writeSourceInfo( result.getSourceInfo() );

            m_xml.scopedElement( "Original" )
                .writeText( result.getExpression() );
            m_xml.scopedElement( "Expanded" )
                .writeText( result.getExpandedExpression() );
        }

        // And... Print a result applicable to each result type.
        switch( result.getResultType() ) {
            case ResultWas::ThrewException:
                m_xml.startElement( "Exception" );
                writeSourceInfo( result.getSourceInfo() );
                m_xml.writeText( result.getMessage() );
                m_xml.endElement();
                break;
            case ResultWas::FatalErrorCondition:
                m_xml.startElement( "FatalErrorCondition" );
                writeSourceInfo( result.getSourceInfo() );
                m_xml.writeText( result.getMessage() );
                m_xml.endElement();
                break;
            case ResultWas::Info:
                m_xml.scopedElement( "Info" )
                     .writeText( result.getMessage() );
                break;
            case ResultWas::Warning:
                // Warning will already have been written
                break;
            case ResultWas::ExplicitFailure:
                m_xml.startElement( "Failure" );
                writeSourceInfo( result.getSourceInfo() );
                m_xml.writeText( result.getMessage() );
                m_xml.endElement();
                break;
            case ResultWas::ExplicitSkip:
                m_xml.startElement( "Skip" );
                writeSourceInfo( result.getSourceInfo() );
                m_xml.writeText( result.getMessage() );
                m_xml.endElement();
                break;
            default:
                break;
        }

        if( result.hasExpression() )
            m_xml.endElement();
    }

    void XmlReporter::sectionEnded( SectionStats const& sectionStats ) {
        StreamingReporterBase::sectionEnded( sectionStats );
        if ( --m_sectionDepth > 0 ) {
            {
                XmlWriter::ScopedElement e = m_xml.scopedElement( "OverallResults" );
                e.writeAttribute( "successes"_sr, sectionStats.assertions.passed );
                e.writeAttribute( "failures"_sr, sectionStats.assertions.failed );
                e.writeAttribute( "expectedFailures"_sr, sectionStats.assertions.failedButOk );
                e.writeAttribute( "skipped"_sr, sectionStats.assertions.skipped > 0 );

                if ( m_config->showDurations() == ShowDurations::Always )
                    e.writeAttribute( "durationInSeconds"_sr, sectionStats.durationInSeconds );
            }
            // Ends assertion tag
            m_xml.endElement();
        }
    }

    void XmlReporter::testCaseEnded( TestCaseStats const& testCaseStats ) {
        StreamingReporterBase::testCaseEnded( testCaseStats );
        XmlWriter::ScopedElement e = m_xml.scopedElement( "OverallResult" );
        e.writeAttribute( "success"_sr, testCaseStats.totals.assertions.allOk() );
        e.writeAttribute( "skips"_sr, testCaseStats.totals.assertions.skipped );

        if ( m_config->showDurations() == ShowDurations::Always )
            e.writeAttribute( "durationInSeconds"_sr, m_testCaseTimer.getElapsedSeconds() );
        if( !testCaseStats.stdOut.empty() )
            m_xml.scopedElement( "StdOut" ).writeText( trim( StringRef(testCaseStats.stdOut) ), XmlFormatting::Newline );
        if( !testCaseStats.stdErr.empty() )
            m_xml.scopedElement( "StdErr" ).writeText( trim( StringRef(testCaseStats.stdErr) ), XmlFormatting::Newline );

        m_xml.endElement();
    }

    void XmlReporter::testRunEnded( TestRunStats const& testRunStats ) {
        StreamingReporterBase::testRunEnded( testRunStats );
        m_xml.scopedElement( "OverallResults" )
            .writeAttribute( "successes"_sr, testRunStats.totals.assertions.passed )
            .writeAttribute( "failures"_sr, testRunStats.totals.assertions.failed )
            .writeAttribute( "expectedFailures"_sr, testRunStats.totals.assertions.failedButOk )
            .writeAttribute( "skips"_sr, testRunStats.totals.assertions.skipped );
        m_xml.scopedElement( "OverallResultsCases")
            .writeAttribute( "successes"_sr, testRunStats.totals.testCases.passed )
            .writeAttribute( "failures"_sr, testRunStats.totals.testCases.failed )
            .writeAttribute( "expectedFailures"_sr, testRunStats.totals.testCases.failedButOk )
            .writeAttribute( "skips"_sr, testRunStats.totals.testCases.skipped );
        m_xml.endElement();
    }

    void XmlReporter::benchmarkPreparing( StringRef name ) {
        m_xml.startElement("BenchmarkResults")
             .writeAttribute("name"_sr, name);
    }

    void XmlReporter::benchmarkStarting(BenchmarkInfo const &info) {
        m_xml.writeAttribute("samples"_sr, info.samples)
            .writeAttribute("resamples"_sr, info.resamples)
            .writeAttribute("iterations"_sr, info.iterations)
            .writeAttribute("clockResolution"_sr, info.clockResolution)
            .writeAttribute("estimatedDuration"_sr, info.estimatedDuration)
            .writeComment("All values in nano seconds"_sr);
    }

    void XmlReporter::benchmarkEnded(BenchmarkStats<> const& benchmarkStats) {
        m_xml.scopedElement("mean")
            .writeAttribute("value"_sr, benchmarkStats.mean.point.count())
            .writeAttribute("lowerBound"_sr, benchmarkStats.mean.lower_bound.count())
            .writeAttribute("upperBound"_sr, benchmarkStats.mean.upper_bound.count())
            .writeAttribute("ci"_sr, benchmarkStats.mean.confidence_interval);
        m_xml.scopedElement("standardDeviation")
            .writeAttribute("value"_sr, benchmarkStats.standardDeviation.point.count())
            .writeAttribute("lowerBound"_sr, benchmarkStats.standardDeviation.lower_bound.count())
            .writeAttribute("upperBound"_sr, benchmarkStats.standardDeviation.upper_bound.count())
            .writeAttribute("ci"_sr, benchmarkStats.standardDeviation.confidence_interval);
        m_xml.scopedElement("outliers")
            .writeAttribute("variance"_sr, benchmarkStats.outlierVariance)
            .writeAttribute("lowMild"_sr, benchmarkStats.outliers.low_mild)
            .writeAttribute("lowSevere"_sr, benchmarkStats.outliers.low_severe)
            .writeAttribute("highMild"_sr, benchmarkStats.outliers.high_mild)
            .writeAttribute("highSevere"_sr, benchmarkStats.outliers.high_severe);
        m_xml.endElement();
    }

    void XmlReporter::benchmarkFailed(StringRef error) {
        m_xml.scopedElement("failed").
            writeAttribute("message"_sr, error);
        m_xml.endElement();
    }

    void XmlReporter::listReporters(std::vector<ReporterDescription> const& descriptions) {
        auto outerTag = m_xml.scopedElement("AvailableReporters");
        for (auto const& reporter : descriptions) {
            auto inner = m_xml.scopedElement("Reporter");
            m_xml.startElement("Name", XmlFormatting::Indent)
                 .writeText(reporter.name, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            m_xml.startElement("Description", XmlFormatting::Indent)
                 .writeText(reporter.description, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
        }
    }

    void XmlReporter::listListeners(std::vector<ListenerDescription> const& descriptions) {
        auto outerTag = m_xml.scopedElement( "RegisteredListeners" );
        for ( auto const& listener : descriptions ) {
            auto inner = m_xml.scopedElement( "Listener" );
            m_xml.startElement( "Name", XmlFormatting::Indent )
                .writeText( listener.name, XmlFormatting::None )
                .endElement( XmlFormatting::Newline );
            m_xml.startElement( "Description", XmlFormatting::Indent )
                .writeText( listener.description, XmlFormatting::None )
                .endElement( XmlFormatting::Newline );
        }
    }

    void XmlReporter::listTests(std::vector<TestCaseHandle> const& tests) {
        auto outerTag = m_xml.scopedElement("MatchingTests");
        for (auto const& test : tests) {
            auto innerTag = m_xml.scopedElement("TestCase");
            auto const& testInfo = test.getTestCaseInfo();
            m_xml.startElement("Name", XmlFormatting::Indent)
                 .writeText(testInfo.name, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            m_xml.startElement("ClassName", XmlFormatting::Indent)
                 .writeText(testInfo.className, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            m_xml.startElement("Tags", XmlFormatting::Indent)
                 .writeText(testInfo.tagsAsString(), XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);

            auto sourceTag = m_xml.scopedElement("SourceInfo");
            m_xml.startElement("File", XmlFormatting::Indent)
                 .writeText(testInfo.lineInfo.file, XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            m_xml.startElement("Line", XmlFormatting::Indent)
                 .writeText(std::to_string(testInfo.lineInfo.line), XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
        }
    }

    void XmlReporter::listTags(std::vector<TagInfo> const& tags) {
        auto outerTag = m_xml.scopedElement("TagsFromMatchingTests");
        for (auto const& tag : tags) {
            auto innerTag = m_xml.scopedElement("Tag");
            m_xml.startElement("Count", XmlFormatting::Indent)
                 .writeText(std::to_string(tag.count), XmlFormatting::None)
                 .endElement(XmlFormatting::Newline);
            auto aliasTag = m_xml.scopedElement("Aliases");
            for (auto const& alias : tag.spellings) {
                m_xml.startElement("Alias", XmlFormatting::Indent)
                     .writeText(alias, XmlFormatting::None)
                     .endElement(XmlFormatting::Newline);
            }
        }
    }

} // end namespace Catch

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
