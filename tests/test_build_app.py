"""Exercise build-app.sh's real UI compiler queue without a PS5 SDK.

The extracted fragment is deliberately bounded by stable surrounding commands;
if the build script is reorganized, update this harness rather than testing a
copy of the scheduling algorithm.
"""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]

# One locked record makes overlapping fake compiler processes observable without
# timing assertions. Both the C and C++ paths use this executable.
COMPILER = r'''
import fcntl
import json
import os
from pathlib import Path
import sys
import time

source = Path(sys.argv[sys.argv.index('-c') + 1])
output = Path(sys.argv[sys.argv.index('-o') + 1])
state_path = Path(os.environ['QUEUE_STATE'])
def record(starting):
    with state_path.open('r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = json.load(stream)
        state['active'] += 1 if starting else -1
        state['peak'] = max(state['peak'], state['active'])
        if starting:
            state['started'].append(source.name)
        stream.seek(0)
        json.dump(state, stream)
        stream.truncate()

record(True)
failed = source.name == os.environ.get('FAIL_SOURCE')
try:
    # A failure can finish before the parent reaches wait; successful compilers
    # stay alive briefly so the concurrency bound is meaningfully exercised.
    if not failed:
        time.sleep(0.08)
        output.write_text('fake object\n')
finally:
    record(False)
sys.exit(17 if failed else 0)
'''


class BuildAppQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = (ROOT / 'scripts/build-app.sh').read_text()
        start = script.index('kit_objects=()\n')
        end = script.index('rm -f "$obj/libps5lm_kit.a"', start)
        cls.queue = script[start:end]

    def run_queue(self, jobs='2', fail=None, cached=(), count=5, host_cpus=2):
        with tempfile.TemporaryDirectory(prefix='ps5lm-build-queue-') as directory:
            root = Path(directory)
            kit = root / 'kit'
            obj = root / 'obj'
            (kit / 'src').mkdir(parents=True)
            (obj / 'kit').mkdir(parents=True)
            (root / 'ps5/app').mkdir(parents=True)
            compiler = root / 'ps5/app/ps5cc'
            compiler.write_text(f'#!{sys.executable}\n' + COMPILER)
            compiler.chmod(0o755)
            state_path = root / 'state.json'
            state_path.write_text(json.dumps({'active': 0, 'peak': 0, 'started': []}))
            sources = []
            for index in range(count):
                extension = 'c' if index % 2 else 'cpp'
                source = kit / 'src' / f'{index}.{extension}'
                source.write_text('// fake input\n')
                sources.append(source)
                if index in cached:
                    output = obj / 'kit' / (source.name + '.o')
                    output.write_text('cached object\n')
                    # Avoid depending on filesystem timestamp granularity.
                    os.utime(output, (time.time() + 10, time.time() + 10))

            setup = '\n'.join([
                'set -euo pipefail',
                f'root={shlex.quote(str(root))}',
                f'kit={shlex.quote(str(kit))}',
                f'obj={shlex.quote(str(obj))}',
                'kitflags=()',
                'kit_sources=(' + ' '.join(shlex.quote(str(p)) for p in sources) + ')',
                'cxx() { "$root/ps5/app/ps5cc" "$@"; }',
                # Keep the default deterministic, independent of runner size.
                f'nproc() {{ printf "{host_cpus}\\n"; }}',
            ])
            env = os.environ.copy()
            env.pop('CMAKE_BUILD_PARALLEL_LEVEL', None)
            env.pop('FAIL_SOURCE', None)
            if jobs is not None:
                env['CMAKE_BUILD_PARALLEL_LEVEL'] = jobs
            if fail is not None:
                env['FAIL_SOURCE'] = sources[fail].name
            env['QUEUE_STATE'] = str(state_path)
            completed = subprocess.run(
                ['bash', '-c', setup + '\n' + self.queue],
                env=env, capture_output=True, text=True, timeout=10,
            )
            state = json.loads(state_path.read_text())
            objects = {p.name: p.read_text() for p in (obj / 'kit').glob('*.o')}
            return completed, state, objects

    def test_success_respects_job_limit(self):
        for jobs in ('1', '2', '3'):
            with self.subTest(jobs=jobs):
                result, state, objects = self.run_queue(jobs=jobs)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(state['started']), 5)
                self.assertEqual(state['active'], 0)
                self.assertGreaterEqual(state['peak'], 1)
                self.assertLessEqual(state['peak'], int(jobs))
                self.assertEqual(len(objects), 5)

    def test_fast_compiler_failures_are_not_lost(self):
        # The old nproc/bare-wait loop loses failures on a many-core host;
        # CMAKE_BUILD_PARALLEL_LEVEL must take precedence over that host size.
        for position in (0, 2, 4):
            with self.subTest(position=position):
                result, state, _ = self.run_queue(fail=position, host_cpus=64)
                self.assertEqual(result.returncode, 17, result.stderr)
                self.assertEqual(state['active'], 0)
                self.assertLessEqual(state['peak'], 2)

    def test_cached_objects_skip_compilation(self):
        result, state, objects = self.run_queue(cached=(0, 2, 4))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(state['started']), ['1.c', '3.c'])
        self.assertEqual(state['active'], 0)
        for index in (0, 2, 4):
            self.assertEqual(objects[f'{index}.cpp.o'], 'cached object\n')

    def test_all_cached_or_empty_sources_need_no_wait(self):
        for count, cached in ((0, ()), (5, range(5))):
            with self.subTest(count=count):
                result, state, _ = self.run_queue(count=count, cached=cached)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(state['started'], [])
                self.assertEqual(state['active'], 0)

    def test_invalid_job_limits_fail_before_compiling(self):
        for jobs in ('0', '-1', '01', 'two'):
            with self.subTest(jobs=jobs):
                result, state, _ = self.run_queue(jobs=jobs)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('invalid job limit', result.stderr)
                self.assertEqual(state['started'], [])

    def test_default_uses_nproc(self):
        result, state, _ = self.run_queue(jobs=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLessEqual(state['peak'], 2)
        self.assertEqual(len(state['started']), 5)


class NativeBuildPreflightTests(unittest.TestCase):
    def test_early_failure_removes_previous_success_report(self):
        cases = (
            ({'BUILD_JOBS': '0'}, 'BUILD_JOBS must be a positive integer'),
            ({'BUILD_JOBS': '2', 'LLVM_CONFIG': 'ps5lm-nonexistent-llvm-config-for-test'},
             'LLVM_CONFIG'),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory(prefix='ps5lm-native-preflight-') as directory:
                    root = Path(directory)
                    (root / 'scripts').mkdir()
                    script = root / 'scripts/test-native-build.sh'
                    script.write_text((ROOT / 'scripts/test-native-build.sh').read_text())
                    report = root / 'build/native-ci/result.json'
                    report.parent.mkdir(parents=True)
                    report.write_text('{"status": "passed"}\n')
                    env = os.environ.copy()
                    env.update(overrides)
                    result = subprocess.run(
                        ['bash', str(script)], env=env,
                        capture_output=True, text=True, timeout=5,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(report.exists())
                    # These failures must occur before any dependency bootstrap.
                    self.assertFalse((root / '.deps').exists())


if __name__ == '__main__':
    unittest.main()
