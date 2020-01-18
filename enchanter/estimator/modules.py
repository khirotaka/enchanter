

def is_jupyter() -> bool:
    if "get_ipython" not in globals():
        return False
    env = get_ipython().__class__.__name__
    if env == "TerminalInteractiveShell":
        return False
    return True
