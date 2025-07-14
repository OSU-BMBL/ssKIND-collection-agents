import yaml

def read_config_query(scope: str):
    with open("./config/scope_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
        sc_config = config[scope]
    return sc_config["query"], sc_config["mindate"], sc_config["maxdate"]

def read_config_identify_original_instructions(scope: str):
    with open("./config/scope_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
        sc_config = config[scope]
    return sc_config.get("identify_original_instructions", "N/A")

def read_config_identify_relevant_instructions(scope: str):
    with open("./config/scope_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
        sc_config = config[scope]
    return sc_config.get("identify_relevant_instructions", "N/A")

def read_config_scopes() -> list:
    with open("./config/scope_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    return list(config.keys())

