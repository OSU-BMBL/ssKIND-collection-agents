
import argparse
import json
import logging
import os

from app_script import main_execute

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

logger = logging.getLogger(__name__)


def main(scopes, mindate=None, maxdate=None):
    all_results = {}
    for scope in scopes:
        logger.info("=" * 64)
        logger.info("Running scope: %s", scope)
        try:
            valid_pmids = main_execute(scope, mindate=mindate, maxdate=maxdate)
            all_results[scope] = valid_pmids
            logger.info("Scope %s done — %d valid PMIDs: %s", scope, len(valid_pmids), valid_pmids)
        except Exception as exc:
            logger.error("Scope %s failed: %s", scope, exc)
            all_results[scope] = []

    # Write consolidated results
    out_path = "all_scopes_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Wrote consolidated results to %s", out_path)

    total = sum(len(v) for v in all_results.values())
    logger.info("All scopes complete. Total valid PMIDs across all scopes: %d", total)
    return all_results


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
