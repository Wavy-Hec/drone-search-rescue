"""Task registration hook for the ``isaaclab`` CLI without ``pip install``.

``isaaclab train/play`` discover installed projects through the ``isaaclab.tasks`` entry point. Without an
install, put ``src`` on ``PYTHONPATH`` and pass ``--rl_library rsl_rl --external_callback
drone_lab.register.register_tasks``; the callback imports the task packages (their ``gym.register`` calls) and
returns None, which tells the CLI that it consumed no arguments.
"""


def register_tasks() -> None:
    """Import every task package of the project so its Gymnasium ids are registered."""
    import drone_lab.tasks  # noqa: F401

    return None
