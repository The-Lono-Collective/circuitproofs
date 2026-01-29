"""
Circuit Extraction Module for Certified Proof-Carrying Circuits

This module provides tools for extracting sparse, interpretable circuits
from neural networks with certified error bounds using BlockCert
Lipschitz composition.
"""

from extraction.blockcert import (
    BlockIR,
    AttentionIR,
    MLPIR,
    NormIR,
    TraceRecord,
    TraceDataset,
    BlockInterpreter,
    BlockCertifier,
    CertificationMetrics,
    Certificate,
    generate_certificate,
)

__all__ = [
    # IR
    'BlockIR',
    'AttentionIR',
    'MLPIR',
    'NormIR',
    # Trace data
    'TraceRecord',
    'TraceDataset',
    # Components
    'BlockInterpreter',
    'BlockCertifier',
    'CertificationMetrics',
    'Certificate',
    'generate_certificate',
]

__version__ = '1.1.0'
