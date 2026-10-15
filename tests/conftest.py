"""Shared pytest fixtures: force a headless matplotlib backend before anything imports pyplot."""

import matplotlib

matplotlib.use("Agg")
