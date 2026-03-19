.PHONY: all build build-tests build-linux build-windows build-macos guard history-audit test-module test-runtime test-baseline test test-quick docs docs-stage docs-site benchmark format lint clean

# Determine parallelism for SCons builds; fall back to a single job if detection fails.
NPROC ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
PLATFORM ?= linuxbsd
GODOT_SOURCE_DIR ?= .

define WITH_GODOT_BINARY
	@set -e; \
	godot_bin="$${GODOT_BINARY:-}"; \
	if [ -z "$$godot_bin" ]; then \
		godot_bin="$$(ls $(GODOT_SOURCE_DIR)/bin/godot.$(PLATFORM).editor.dev.* 2>/dev/null | head -n 1)"; \
	fi; \
	if [ -z "$$godot_bin" ]; then \
		godot_bin="$$(ls $(GODOT_SOURCE_DIR)/bin/godot.$(PLATFORM).editor.* 2>/dev/null | head -n 1)"; \
	fi; \
	if [ -z "$$godot_bin" ]; then \
		echo "Could not locate a Godot editor binary under $(GODOT_SOURCE_DIR)/bin."; \
		echo "Build one first (for tests: 'make build-tests') or set GODOT_BINARY."; \
		exit 1; \
	fi; \
	echo "Using GODOT_BINARY=$$godot_bin"; \
	$(1)
endef

# Default target
all: build

# Build targets
build:
	scons -C $(GODOT_SOURCE_DIR) platform=$(PLATFORM) target=editor dev_build=yes -j$(NPROC)

build-tests:
	scons -C $(GODOT_SOURCE_DIR) platform=$(PLATFORM) target=editor dev_build=yes tests=yes -j$(NPROC)

build-linux:
	$(MAKE) build PLATFORM=linuxbsd

build-windows:
	$(MAKE) build PLATFORM=windows

build-macos:
	$(MAKE) build PLATFORM=macos

# Testing
guard:
	GS_CI_HISTORY_ARTIFACT_GUARD_MODE=$${GS_CI_HISTORY_ARTIFACT_GUARD_MODE:-warn} python3 tests/ci/run_module_tests.py --guard-only

history-audit:
	python3 scripts/repo/history_artifact_audit.py

test-module:
	$(call WITH_GODOT_BINARY,GODOT_BINARY="$$godot_bin" python3 tests/ci/run_module_tests.py --godot-binary "$$godot_bin")

test-runtime:
	$(call WITH_GODOT_BINARY,GODOT_BINARY="$$godot_bin" python3 tests/runtime/run_runtime_validation.py --godot-binary "$$godot_bin" --gd-mode headless)

test-baseline:
	$(call WITH_GODOT_BINARY,GODOT_BINARY="$$godot_bin" python3 tests/ci/run_baseline_qa.py --godot "$$godot_bin")

test-module test-runtime test-baseline: build-tests

test: guard test-baseline

test-quick: build-tests guard
	$(call WITH_GODOT_BINARY,GODOT_BINARY="$$godot_bin" python3 tests/ci/run_baseline_qa.py --godot "$$godot_bin" --quick)

# Development
format:
	clang-format -i modules/gaussian_splatting/**/*.{cpp,h}
	python3 -m black scripts test_data tests/ci

lint:
	python3 -m flake8 scripts test_data tests/ci

# Documentation
docs:
	python3 scripts/build_documentation.py --all

docs-stage:
	python3 scripts/stage_public_docs.py --source docs --output .site/public-docs

docs-site:
	python3 scripts/build_docs_site.py --strict

# Benchmarking
benchmark:
	$(call WITH_GODOT_BINARY,GODOT_BINARY="$$godot_bin" python3 tests/runtime/run_benchmark_suite.py --godot-binary "$$godot_bin" --profile quick)

# Cleanup
clean:
	scons -C $(GODOT_SOURCE_DIR) --clean
	rm -rf $(GODOT_SOURCE_DIR)/bin .import/ logs/
