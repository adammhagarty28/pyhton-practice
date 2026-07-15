# Source Dependencies

This project distinguishes between two kinds of dependencies.

## 1. Installable Python packages

Normal Python packages are documented in:

    requirements/runtime.txt

The canonical SIMCenter environment is:

    jax-forge

Missing installable libraries should be added to that environment when
necessary rather than creating a new project-specific Conda environment.

## 2. Source-code dependencies

Any external source code that the project imports directly must be stored
inside the Git repository.

This prevents the project from working on one computer only because a
manually downloaded source file exists somewhere on that machine.

### JAX-PULSE

The current project imports:

    relevant_PULSE_files.jax_kernels
    relevant_PULSE_files.jax_pulse

The required source currently lives under:

    src/relevant_PULSE_files/

At minimum, the working project currently depends on functionality including:

    Pose
    deposit
    Pulse

These source files must remain version-controlled with the project.

Do not replace or modify the active JAX-PULSE implementation during the
current behavior-preserving refactor unless a change is explicitly required.

Future cleanup may move external source dependencies into a clearer location
such as:

    third_party/jax_pulse/

or:

    src/spray_cooling/vendor/jax_pulse/

That move should only occur after all current imports have been identified
and updated safely.

## Project dependency rule

If the dependency can be installed normally:

    document it in requirements/runtime.txt

If the project requires downloaded source code that is imported directly:

    store that source in the repository and track it with Git

If a large external project should not be copied into this repository:

    pin it explicitly using a Git submodule or another reproducible source
    reference rather than relying on an untracked local folder
