
import argparse
import os
import subprocess

def run_command(command: list, cwd: str = None, timeout: int = None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        return e.stdout or "", e.stderr or f"Command timed out after {timeout} seconds", -1

ALL_SCOPES = [
    'Alzheimer_SingleCell',
    'Alzheimer_Spatial',
    'Parkinson_SingleCell',
    'Frontotemporal_Dementia_SingleCell',
    'Multiple_Sclerosis_SingleCell',

    'Spinal_Muscular_Atrophy_SingleCell',
    'Amyotrophic_Lateral_Sclerosis_SingleCell',
    'Spinocerebellar_ataxia_SingleCell',
    'Huntingtons_SingleCell',
    'Prion_diseases_SingleCell',

    'Parkinson_Spatial',
    'Frontotemporal_Dementia_Spatial',
    'Multiple_Sclerosis_Spatial',
    'Amyotrophic_Lateral_Sclerosis_Spatial',
    'Spinal_Muscular_Atrophy_Spatial',
    'Spinocerebellar_ataxia_Spatial',
    'Huntington_disease_Spatial',
    'Prion_diseases_Spatial',
]

def main(scopes, mindate=None, maxdate=None):
    for scope in scopes:
        cmd = ["python", "./app_script.py", "-s", scope]
        if mindate:
            cmd += ["--mindate", mindate]
        if maxdate:
            cmd += ["--maxdate", maxdate]
        out, error, code = run_command(cmd)
        if code != 0:
            with open(f"./{scope}_error.log", "w") as fobj:
                fobj.write(str(error))
        with open(f"./{scope}_success.log", "w") as fobj:
            fobj.write(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run identify workflow across scopes.")
    parser.add_argument(
        "-s", "--scope", nargs="+", default=None,
        metavar="SCOPE",
        help=f"one or more scopes to run (default: all). choices: {ALL_SCOPES}",
    )
    parser.add_argument("--mindate", default=None, help="override mindate (YYYY/MM/DD), e.g. 2025/05/30")
    parser.add_argument("--maxdate", default=None, help="override maxdate (YYYY/MM/DD), e.g. 2026/07/10")
    args = parser.parse_args()

    scopes = args.scope if args.scope else ALL_SCOPES
    invalid = [s for s in scopes if s not in ALL_SCOPES]
    if invalid:
        parser.error(f"Unknown scope(s): {invalid}. Valid choices: {ALL_SCOPES}")

    main(scopes, mindate=args.mindate, maxdate=args.maxdate)
