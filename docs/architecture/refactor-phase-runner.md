# Refactor Phase Runner

This workflow automates the repeatable parts of the Gaussian renderer refactor.

What it does:
- runs `git diff --check`
- runs local guard-only validation
- regenerates the architecture pack for major phases
- writes a native Windows batch runner for build + module tests
- creates local checkpoint commits after a phase is verified

What it does not do:
- blindly implement the remaining refactor phases
- replace phase-by-phase review for risky mutator and seam work
- run the native Windows editor/test lane from WSL

## Commands

List the remaining large phases:

```bash
python3 scripts/refactor_phase_runner.py list
```

Run local checks for a phase:

```bash
python3 scripts/refactor_phase_runner.py local-checks --phase 1b.2a
```

Generate the Windows verification helper:

```bash
python3 scripts/refactor_phase_runner.py write-windows-runner --phase 1b.2a
```

That writes:

`ci/scripts/run_refactor_phase_windows.bat`

Run it from native Windows `cmd.exe`:

```bat
cd /d C:\projects\godotgs-clean-refactor
ci\scripts\run_refactor_phase_windows.bat
```

Create a checkpoint commit after the phase is verified:

```bash
python3 scripts/refactor_phase_runner.py checkpoint-commit \
  --phase 1b.2a \
  --stage-all-tracked \
  --path docs/architecture/gaussian-renderer-refactor-memory.md \
  --message "refactor(renderer): complete phase 1b.2a checkpoint"
```

## Recommended use

1. Implement one large coherent phase batch.
2. Run `local-checks`.
3. Run the generated Windows batch helper.
4. Update the memory log with the verification result.
5. Create a checkpoint commit.

## Sequence discipline

Keep the original guardrails:
- `GaussianSplatRenderer` remains the stable external facade.
- Do not drift into debug/painterly redesign outside the planned phases.
- Keep composition-root and sorting cleanup tightly scoped.
- Replace test internal mutations before final accessor lockdown.
- Regenerate the architecture pack after each major phase.
