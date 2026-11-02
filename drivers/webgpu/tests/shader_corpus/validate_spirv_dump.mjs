/**
 * SPIR-V Dump Validator
 *
 * Validates all .spv files in a given directory through the Tint CLI.
 * Designed to run after `GODOT_DUMP_SPIRV=<dir>` produces SPIR-V files during
 * engine shader compilation.
 *
 * Known failures are tracked in expected_failures.json: shader variants the
 * WebGPU Mobile renderer does not use. Only NEW failures beyond the baseline fail.
 *
 * Usage:
 *   node validate_spirv_dump.mjs <spirv-directory>
 *   node validate_spirv_dump.mjs ./spirv_dump/
 *   node validate_spirv_dump.mjs ./spirv_dump/ --update-baseline
 *
 * Exit codes:
 *   0 = no regressions (all failures are expected)
 *   1 = new failures detected beyond expected baseline
 */

import { readFileSync, readdirSync, writeFileSync, existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { convertSpirvFile, findTintCli } from '../tint_cli.mjs';

const EXPECTED_FAILURES_PATH = join(dirname(fileURLToPath(import.meta.url)), 'expected_failures.json');

// ─── Main ─────────────────────────────────────────────────────────────────────

function main() {
    const spvDir = process.argv[2];
    const updateBaseline = process.argv.includes('--update-baseline');

    if (!spvDir) {
        console.error('Usage: node validate_spirv_dump.mjs <spirv-directory> [--update-baseline]');
        process.exit(1);
    }

    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║   SPIR-V Dump Validator (Tint)                           ║');
    console.log('╚══════════════════════════════════════════════════════════╝\n');

    const tintCli = findTintCli();
    console.log(`Using Tint CLI: ${tintCli}\n`);

    // Load expected failures baseline
    const baseline = existsSync(EXPECTED_FAILURES_PATH) ? JSON.parse(readFileSync(EXPECTED_FAILURES_PATH, 'utf8')) : {};
    const expectedFailures = new Set(updateBaseline ? [] : baseline.expected_failures || []);
    console.log(`Expected failures baseline: ${expectedFailures.size} shaders\n`);

    // Find all .spv files
    let spvFiles;
    try {
        spvFiles = readdirSync(spvDir).filter(f => f.endsWith('.spv')).sort();
    } catch (e) {
        console.error(`ERROR: Cannot read directory: ${spvDir}`);
        console.error(e.message);
        process.exit(1);
    }

    if (spvFiles.length === 0) {
        console.error(`ERROR: No .spv files found in ${spvDir}`);
        console.error('Did you run with GODOT_DUMP_SPIRV set?');
        process.exit(1);
    }

    console.log(`Found ${spvFiles.length} SPIR-V files in ${spvDir}\n`);

    let passed = 0;
    let failed = 0;
    let expectedFailed = 0;
    const failures = [];
    const regressions = [];

    for (const spvFile of spvFiles) {
        const spvPath = join(spvDir, spvFile);

        process.stdout.write(`  ${spvFile.padEnd(55)}`);

        const startTime = performance.now();
        const { wgsl, error: errorMsg } = convertSpirvFile(tintCli, spvPath);
        if (wgsl !== null) {
            const elapsed = (performance.now() - startTime).toFixed(1);
            console.log(`PASS  (${elapsed}ms, ${wgsl.length} chars)`);
            passed++;
        } else {
            const elapsed = (performance.now() - startTime).toFixed(1);
            const isExpected = expectedFailures.has(spvFile);

            if (isExpected) {
                console.log(`XFAIL (${elapsed}ms)  [expected]`);
                expectedFailed++;
            } else {
                console.log(`FAIL  (${elapsed}ms)`);
                console.log(`         ${errorMsg.split('\n')[0]}`);
                regressions.push({ file: spvFile, error: errorMsg });
            }
            failed++;
            failures.push({ file: spvFile, error: errorMsg });
        }
    }

    // Summary
    console.log('\n' + '─'.repeat(60));
    console.log(`\nResults: ${passed} passed, ${failed} failed (${expectedFailed} expected), ${spvFiles.length} total`);
    console.log(`Pass rate: ${((passed / spvFiles.length) * 100).toFixed(1)}%`);

    if (regressions.length > 0) {
        console.log(`\nREGRESSIONS (${regressions.length} new failures):`);
        for (const f of regressions) {
            console.log(`  - ${f.file}: ${f.error.substring(0, 100)}`);
        }
    }

    if (expectedFailed > 0 && !updateBaseline) {
        console.log(`\nExpected failures (${expectedFailed}):`);
        for (const f of failures.filter(f => expectedFailures.has(f.file))) {
            console.log(`  - ${f.file}`);
        }
    }

    // Write report
    const report = {
        timestamp: new Date().toISOString(),
        converter: 'tint',
        spirv_directory: spvDir,
        total: spvFiles.length,
        passed,
        failed,
        expected_failed: expectedFailed,
        regressions: regressions.length,
        failures,
    };

    const reportPath = join(spvDir, 'validation_report.json');
    writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\nReport: ${reportPath}`);

    // Update baseline if requested, keeping the categories of shaders that still fail.
    if (updateBaseline) {
        const failing = new Set(failures.map(f => f.file));
        const categories = Object.entries(baseline.failure_categories || {})
            .map(([category, files]) => [category, files.filter(f => failing.has(f))])
            .filter(([, files]) => files.length > 0);
        const updated = {
            description: baseline.description,
            updated: new Date().toISOString(),
            expected_failures: [...failing].sort(),
            failure_categories: Object.fromEntries(categories),
        };
        writeFileSync(EXPECTED_FAILURES_PATH, JSON.stringify(updated, null, 2) + '\n');
        console.log(`\nBaseline updated: ${EXPECTED_FAILURES_PATH} (${failures.length} expected failures)`);
        process.exit(0);
    }

    // Exit: fail only on regressions
    if (regressions.length > 0) {
        console.error(`\nFAILED: ${regressions.length} new shader(s) cannot be converted.`);
        console.error('If these variants are unused by the WebGPU renderer, run with --update-baseline');
        process.exit(1);
    }

    console.log('\nPASSED: No regressions detected.');
    process.exit(0);
}

main();
