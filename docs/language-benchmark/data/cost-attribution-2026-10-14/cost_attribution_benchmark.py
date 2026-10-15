"""Tests-only cost diagnostics, not a replacement language-parity scorecard."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from interleaved_ab_benchmark import (ROOT, ProcessMonitor, dependency_hashes, run_child,
                                      sha256, wait_for_idle)

PROBES = {'raw0','scoped0','bridge_plan0','bridge_unplanned0','raw4','bridge_plan4',
          'bridge_unplanned4','instance0','object0','phase_prepare','phase_execute_read','phase_unprepare'}
PROBES |= {f'{kind}_{op}' for kind in ('vec','color') for op in
           ('scalar','operator','inplace','generic_inplace','read','copy','assign','default')}


def parse_costs(text, run):
    if '[doctest] Status: SUCCESS!' not in text or 'COST_UNSUPPORTED' in text or 'ERROR:' in text:
        raise ValueError('test did not complete successfully')
    rows=[]
    for line in text.splitlines():
        if not line.startswith('COST|'): continue
        _,name,rep,n,us,checksum=line.split('|')
        row=dict(run=run,probe=name,rep=int(rep),iters=int(n),elapsed_us=int(us),checksum=int(checksum))
        factor=10 if name.endswith('4') or (name.startswith('color_') and not name.endswith('default')) else 3 if name.startswith('vec_') and not name.endswith('default') else 1
        expected_n=204800 if name.startswith('phase_') else 200000
        if row['iters']!=expected_n or row['elapsed_us']<=0 or row['checksum']!=expected_n*factor:
            raise ValueError(f'invalid timing/work: {row}')
        rows.append(row)
    keys={(r['probe'],r['rep']) for r in rows}
    if keys!={(name,rep) for name in PROBES for rep in range(1,8)} or len(rows)!=len(keys):
        raise ValueError('missing/extra/duplicate cost samples')
    return rows


def summarize(rows, runs):
    keys={(r['run'],r['probe'],r['rep']) for r in rows}
    if keys!={(run,name,rep) for run in range(1,runs+1) for name in PROBES for rep in range(1,8)} or len(keys)!=len(rows):
        raise ValueError('incomplete process matrix')
    result=[]
    for name in sorted(PROBES):
        values=[statistics.median(r['elapsed_us']*1000/r['iters'] for r in rows if r['probe']==name and r['run']==run) for run in range(1,runs+1)]
        result.append(dict(probe=name,median_ns=statistics.median(values),min_process_ns=min(values),
                           max_process_ns=max(values),process_medians_ns=values))
    return result


def source_hashes():
    paths=[Path(__file__), Path(__file__).with_name('interleaved_ab_benchmark.py'),
           Path(__file__).with_name('test_angelscript_cost_attribution.cpp'),
           Path(__file__).with_name('test_angelscript_cost_attribution.h')]
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',type=Path,default=ROOT/'bin/godot.windows.editor.x86_64.mono.exe')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runs',type=int,default=6)
    args=parser.parse_args()
    if os.name!='nt' or args.runs<1: parser.error('Windows and positive runs required')
    exe=args.exe.resolve()
    if '.console.' in exe.name: parser.error('actual exe required')
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    env=json.loads((ROOT/'.scons_env.json').read_text())
    if env.get('angelscript_force_generic') or env.get('angelscript_pod_geometry_probe'):
        raise ValueError('default native/non-POD build required')
    meta=dict(started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              exe=str(exe),exe_sha256=sha256(exe),dependencies=dependency_hashes(exe),sources=source_hashes(),runs=args.runs,
              repetitions=7,warmups=1,affinity='0xf000',priority='AboveNormal',
              idle_gate=dict(budget_s=60,sample_timeout_s=3,consecutive=3,cpu_below=25,failure='abort_no_retry'),
              compiler={k:env.get(k) for k in ('CCFLAGS','optimize','precision','module_mono_enabled','tests','angelscript_force_generic','angelscript_pod_geometry_probe')})
    save('metadata.json',meta)
    (args.output/'source.diff').write_bytes(subprocess.check_output(['git','diff','--','modules/angelscript']))
    for path in meta['sources']:
        with (args.output/Path(path).name).open('xb') as output: output.write((ROOT/path).read_bytes())
    rows=[]; gates=[]; processes=[]
    outcome=dict(status='running')
    save('result.json',outcome)
    monitor=ProcessMonitor()
    try:
        with (args.output/'telemetry.jsonl').open('x',encoding='utf-8') as telemetry, (args.output/'samples.csv').open('x',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,['run','probe','rep','iters','elapsed_us','checksum']); writer.writeheader()
            for run in range(1,args.runs+1):
                def record(sample,pid=None):
                    sample.update(run=run,phase='idle' if pid is None else 'run',pid=pid)
                    telemetry.write(json.dumps(sample)+'\n'); telemetry.flush()
                gate=wait_for_idle(monitor,record=record); gate['run']=run; gates.append(gate); save('gates.json',gates)
                print(f"run {run} gate={gate['passed']} reason={gate['reason']} elapsed={gate['elapsed_s']:.3f}s",flush=True)
                if not gate['passed']:
                    outcome.update(status='idle_unavailable',stopped_before_run=run); break
                raw=args.output/f'{run:02d}.log'
                entry=dict(run=run,reverse=run%2==0,status='started'); processes.append(entry); save('processes.json',processes)
                entry.update(run_child(exe,ROOT,raw,0xf000,monitor,record,timeout=180,
                    arguments=['--headless','--test','--test-case=*cost attribution*','--no-breaks'],
                    extra_env={'AS_COST_TIMING':'1','AS_COST_REVERSE':str(int(entry['reverse']))}))
                if entry['exit_code']!=0: raise RuntimeError(f'probe exit {entry["exit_code"]}')
                parsed=parse_costs(raw.read_text(encoding='utf-8',errors='replace'),run)
                rows.extend(parsed); writer.writerows(parsed); file.flush()
                entry['status']='validated'; save('processes.json',processes)
            else:
                outcome['status']='complete'
    except Exception as error:
        outcome.update(status='failed',error=f'{type(error).__name__}: {error}')
    finally:
        outcome['monitor_cleanup']=monitor.close()
        save('processes.json',processes); save('gates.json',gates)
        checks=dict(exe=sha256(exe)==meta['exe_sha256'],dependencies=dependency_hashes(exe)==meta['dependencies'],sources=source_hashes()==meta['sources'])
        outcome.update(final_identity=checks,rows=len(rows),completed_processes=sum(p.get('status')=='validated' for p in processes),finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        if not all(checks.values()) or outcome['monitor_cleanup']['alive']: outcome['status']='failed'
        save('result.json',outcome)
    if outcome['status']=='complete':
        summary=summarize(rows,args.runs); save('summary.json',summary)
        with (args.output/'summary.csv').open('x',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,list(summary[0])); writer.writeheader(); writer.writerows(summary)
        print(f'PASS: {len(rows)} raw repetition records, {args.runs} processes',flush=True)
        return 0
    print('STOP: '+json.dumps(outcome),flush=True)
    return 2 if outcome['status']=='idle_unavailable' else 1


if __name__=='__main__': raise SystemExit(main())
