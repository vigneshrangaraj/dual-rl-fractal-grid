# DualRL_Project

DualRL_Project is a research framework for dual-level reinforcement learning applied to microgrid control. The project integrates:

- **Tertiary Level:** Macro-level economic dispatch and tie-line control using Soft Actor-Critic (SAC).
- **Secondary Level:** Local voltage regulation via per-inverter (DER) control using Independent Asynchronous Advantage Actor-Critic (IA3C) and inter-agent communication.

## Overview

This project simulates a fractal grid where each microgrid is modeled as an IEEE 34 bus system. Each microgrid can include configurable distributed energy resources (DERs) such as solar, wind, and Battery Energy Storage Systems (BESS). The secondary layer uses per-DER agents to regulate local voltage with cooperative communication, while the tertiary layer handles economic dispatch decisions that affect the overall grid.

Key features:
- **Dual-RL Architecture:** Hierarchical control combining tertiary (SAC) and secondary (IA3C) agents.
- **Microgrid Modeling:** Each microgrid is modeled as an IEEE 34 bus system using PandaPower.
- **Inter-Agent Communication:** Secondary DER agents exchange information with a Gaussian distance-masked communication module.
- **AC Power Flow:** Use PandaPower to compute AC power flow and update grid states after control actions.
- **Configurable Components:** All parameters are modularly defined in the `utils/config.py` file.