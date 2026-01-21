"""
Circuit Extraction Module for Certified Proof-Carrying Circuits

This module provides tools for extracting sparse, interpretable circuits
from neural networks with certified error bounds using BlockCert
Lipschitz composition.

Two extraction modes:
1. BlockCertExtractor: Full certified pipeline (recommended)
2. CircuitExtractor: Legacy simple extraction
"""

from .circuit_extractor import (
    # Legacy API (backward compatibility)
    CircuitExtractor,
    CircuitComponent,
    ErrorBound,
    extract_transformer_circuit,
    # BlockCert API (recommended)
    BlockCertExtractor,
    extract_certified_circuit,
)

# BlockCert submodules
from .blockcert import (
    BlockIR,
    AttentionIR,
    MLPIR,
    NormIR,
    BlockInterpreter,
    ActivationTracer,
    TraceDataset,
    BlockCertifier,
    CertificationMetrics,
    Certificate,
    generate_certificate,
)

__all__ = [
    # Legacy API
    'CircuitExtractor',
    'CircuitComponent',
    'ErrorBound',
    'extract_transformer_circuit',
    # BlockCert API
    'BlockCertExtractor',
    'extract_certified_circuit',
    # IR
    'BlockIR',
    'AttentionIR',
    'MLPIR',
    'NormIR',
    # Components
    'BlockInterpreter',
    'ActivationTracer',
    'TraceDataset',
    'BlockCertifier',
    'CertificationMetrics',
    'Certificate',
    'generate_certificate',
]

__version__ = '1.1.0'
