import unittest

from cost_attribution_benchmark import PROBES, parse_costs, summarize


def fixture():
    lines = ['[doctest] Status: SUCCESS!']
    for name in PROBES:
        n = 204800 if name.startswith('phase_') else 200000
        factor = 10 if name.endswith('4') or (name.startswith('color_') and not name.endswith('default')) else 3 if name.startswith('vec_') and not name.endswith('default') else 1
        for rep in range(1,8):
            lines.append(f'COST|{name}|{rep}|{n}|1000|{n*factor}')
    return '\n'.join(lines)


class CostTests(unittest.TestCase):
    def test_complete_raw_repetitions(self):
        rows=parse_costs(fixture(),1)
        self.assertEqual(len(rows),28*7)
        self.assertEqual(len(summarize(rows,1)),28)

    def test_missing_duplicate_bad_time_or_checksum(self):
        text=fixture()
        lines=text.splitlines()
        cases=[ '\n'.join(lines[:-1]), text+'\n'+lines[-1], text.replace('|1000|','|0|',1),
                text.replace('|200000|1000|200000','|200000|1000|0',1), text.replace('SUCCESS!','FAILURE!') ]
        for changed in cases:
            with self.assertRaises(ValueError): parse_costs(changed,1)

    def test_reject_partial_process_matrix(self):
        rows=parse_costs(fixture(),1)
        with self.assertRaises(ValueError): summarize(rows,2)

    def test_summary_uses_process_medians(self):
        rows=parse_costs(fixture(),1)+parse_costs(fixture(),2)
        rows[-1]['elapsed_us']=100000
        result=summarize(rows,2)
        self.assertEqual(next(r for r in result if r['probe']=='vec_scalar')['median_ns'],5)


if __name__=='__main__': unittest.main()
