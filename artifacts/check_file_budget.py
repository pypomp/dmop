"""Keep tracked files and distinct paths across reachable history below 1,800."""
import subprocess

LIMIT = 1800
tracked = set(subprocess.check_output(['git', 'ls-files', '-z']).split(b'\0'))-{b''}
history = set(subprocess.check_output(['git', 'log', '--all', '--no-renames', '--format=', '--name-only']).splitlines())-{b''}
# Staged additions must count even before their first commit.
history |= tracked
print(f'{len(tracked)} tracked files; {len(history)} distinct historical paths; budget {LIMIT}')
if max(len(tracked), len(history)) > LIMIT:
    raise SystemExit('Archive bulk results before committing; do not force-add result trees.')
