import importlib
import logging

from snowdrop_tangled_game_engine import GameAgentBase


def import_agent(agent_name: str) -> type[GameAgentBase]:
    """
    Dynamically import a class from a string specification in the format 'module_name.ClassName'.

    Args:
        agent_name (str): The module and class name as a single string.

    Returns:
        type: The class type that was dynamically imported.
    """
    try:
        # Split the input into module and class
        module_name, class_name = agent_name.rsplit('.', 1)
        logging.debug(f"Importing module {module_name} and class {class_name}")

        # Import the module
        module = importlib.import_module(module_name)

        # Get the class from the module
        clazz = getattr(module, class_name)
        if not issubclass(clazz, GameAgentBase):
            raise ValueError(f"Class '{class_name}' is not a subclass of GameAgentBase.")
        return clazz
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(
            f"Could not import class {agent_name}. Ensure it exists and is correctly specified (e.g. module_name.ClassName).") from e
