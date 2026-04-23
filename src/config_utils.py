import yaml

_CONFIG_PATH = "./config/scope_config.yaml"
_SPATIAL_DEFAULTS_KEY = "_spatial_defaults"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _is_spatial_scope(sc_config: dict) -> bool:
    return "disease_clause" in sc_config


def _assemble_spatial_query(sc_config: dict, defaults: dict) -> str:
    disease_clause = sc_config["disease_clause"].strip()
    tech_clause = defaults["tech_clause"].strip()
    return f"({disease_clause}) AND ({tech_clause})"


def _assemble_spatial_original_instructions(sc_config: dict, defaults: dict) -> str:
    base = defaults["original_instructions_base"]
    note = sc_config.get("original_instructions_note", "")
    return base + note + "\n"


def read_config_query(scope: str) -> tuple[str, str, str]:
    config = _load_config()
    sc_config = config[scope]
    if _is_spatial_scope(sc_config):
        defaults = config[_SPATIAL_DEFAULTS_KEY]
        query = _assemble_spatial_query(sc_config, defaults)
        mindate = sc_config.get("mindate", defaults["mindate"])
        maxdate = sc_config.get("maxdate", defaults["maxdate"])
    else:
        query = sc_config["query"]
        mindate = sc_config["mindate"]
        maxdate = sc_config["maxdate"]
    return query, mindate, maxdate


def read_config_identify_original_instructions(scope: str) -> str:
    config = _load_config()
    sc_config = config[scope]
    if _is_spatial_scope(sc_config):
        defaults = config[_SPATIAL_DEFAULTS_KEY]
        return _assemble_spatial_original_instructions(sc_config, defaults)
    return sc_config.get("identify_original_instructions", "N/A")


def read_config_identify_relevant_instructions(scope: str) -> str:
    config = _load_config()
    sc_config = config[scope]
    return sc_config.get("identify_relevant_instructions", "N/A")


def read_config_scopes() -> list:
    config = _load_config()
    return [k for k in config.keys() if not k.startswith("_")]
