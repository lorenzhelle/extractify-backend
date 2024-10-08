import os


def get_api_key_from_env_file(name: str) -> str:
    api_key_env = os.environ.get(name)

    if api_key_env is not None:
        return api_key_env
    else:
        raise ValueError(f"{name} is not set in the environment variables.")
