#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Configuration loading: Jinja2-templated YAML."""

import os

import yaml
from jinja2 import Environment
from jinja2.runtime import ChainableUndefined


def load_config(path, context=None):
    """Render a Jinja2-templated YAML config file and return the parsed dict.

    The template is rendered twice: a first permissive pass resolves the
    ``frequency`` value declared in the file, which is then made available to a
    second pass so that ``{{ frequency }}`` placeholders in output file names
    pick up the value actually set in the config (rather than a hardcoded one).
    An explicit ``context`` still wins over the value read from the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found at {path}")

    context = dict(context or {})

    with open(path) as f:
        template_str = f.read()

    env = Environment(undefined=ChainableUndefined)

    # First pass: tolerate undefined placeholders, just to read `frequency`.
    first = yaml.safe_load(env.from_string(template_str).render(context)) or {}
    render_context = {}
    if first.get("frequency") is not None:
        render_context["frequency"] = first["frequency"]
    render_context.update(context)

    rendered = env.from_string(template_str).render(render_context)
    return yaml.safe_load(rendered)
