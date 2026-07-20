"""Minimal PyTorch reimplementation of DAG-GFlowNet (Deleu et al. 2022).

Scoped to synthetic datasets with a mix of multinomial (categorical) and
Gaussian variables, scored with a decomposable BIC score under the
conditional-Gaussian assumption (discrete nodes have only discrete parents).
"""
