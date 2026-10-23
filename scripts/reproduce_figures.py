"""Replay publication figures from frozen tables in an isolated workspace."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

REPO = Path(__file__).resolve().parents[1]
JOBS = {
    '1': ('03_analysis/2026-10-15_figure1_visual_r3/draw_figure1_r3.py', 'drawing_config.yaml'),
    '2': ('03_analysis/2026-10-16_figure2_matched_chronos_recovery/draw_complete_figure2.py', 'drawing_config.yaml'),
    '3': ('03_analysis/2026-10-18_figure3_r4/draw_figure3_r4.py', 'run_config.yaml'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--figures', nargs='+', choices=sorted(JOBS), default=sorted(JOBS))
    args = parser.parse_args()
    destination = args.output_directory.resolve()
    archive = REPO / 'archive' / 'project'
    if destination.exists():
        raise SystemExit('Choose a new, nonexistent output directory; existing evidence is never overwritten.')
    if destination == archive or archive in destination.parents:
        raise SystemExit('Output must be outside the archived project.')
    destination.mkdir(parents=True)
    workspace = destination / 'workspace'
    shutil.copytree(archive, workspace)
    logs = destination / 'logs'
    logs.mkdir()
    env = dict(os.environ, PROJECT_ROOT=str(workspace), MPLCONFIGDIR=str(destination / 'matplotlib'))
    status = {'operation': 'figure_replay', 'new_model_fits': 0, 'jobs': []}
    for figure in args.figures:
        relative_script, config_name = JOBS[figure]
        script = workspace / relative_script
        config_path = script.with_name(config_name)
        cfg = yaml.safe_load(config_path.read_text())
        cfg['project_root'] = str(workspace)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        log = logs / ('figure' + figure + '.log')
        with log.open('w') as stream:
            completed = subprocess.run([sys.executable, str(script)], cwd=workspace, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status['jobs'].append({'figure': figure, 'returncode': completed.returncode, 'log': str(log.relative_to(destination))})
        (destination / 'REPLAY_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
        if completed.returncode:
            raise SystemExit('Figure replay stopped; see ' + str(log))
    print('Figures saved under ' + str(workspace / '05_figures'))

if __name__ == '__main__':
    main()
