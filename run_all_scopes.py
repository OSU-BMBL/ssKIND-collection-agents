
import os
import subprocess

def run_command(command: list, cwd: str = None, timeout: int = None):
    """
    Run a shell command with optional timeout and return stdout, stderr, and return code.
    """
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

scopes_to_run = [
    # 'Alzheimer_SingleCell', 
    # 'Alzheimer_Spatial', 
    # 'Parkinson_SingleCell', 
    # 'Frontotemporal_Dementia_SingleCell', 
    # 'Multiple_Sclerosis_SingleCell', 
    # 
    # 'Spinal_Muscular_Atrophy_SingleCell', 
    # 'Amyotrophic_Lateral_Sclerosis_SingleCell', 
    # 'Spinocerebellar_ataxia_SingleCell', 
    # 'Huntingtons_SingleCell', 
    # 'Prion_diseases_SingleCell', 
    # 
    # 'Parkinson_Spatial', 
    "Frontotemporal_Dementia_Spatial",
    # 'Multiple_Sclerosis_Spatial', 
    # 'Amyotrophic_Lateral_Sclerosis_Spatial', 
    # 'Spinal_Muscular_Atrophy_Spatial', 
    # 'Spinocerebellar_ataxia_Spatial', 
    # 'Huntington_disease_Spatial', 
    # 'Prion_diseases_Spatial',
]

def main():
    for scope in scopes_to_run:
        out, error, code = run_command([
            "python", "./app_script.py", "-s", scope
        ])
        if code != 0:
            with open(f"./{scope}_error.log", "w") as fobj:
                fobj.write(str(error))
        with open(f"./{scope}_success.log", "w") as fobj:
            fobj.write(out)


if __name__ == "__main__":
    main()
