"""
BlockCert: Certified Circuit Extraction Framework

This module implements the BlockCert methodology for extracting circuits
from transformer models with certified error bounds.

Components:
- ir: Intermediate Representation (.npz storage)
- interpreter: Pure Python block execution
- tracer: Hook-based activation tracing
- certifier: Metrics computation with auto-LiRPA
- certificate: Certificate artifact generation
"""

from .ir import BlockIR, AttentionIR, MLPIR, NormIR
from .interpreter import BlockInterpreter
from .tracer import ActivationTracer, TraceDataset
from .certifier import BlockCertifier, CertificationMetrics
from .certificate import Certificate, generate_certificate

__all__ = [
    # IR
    "BlockIR",
    "AttentionIR",
    "MLPIR",
    "NormIR",
    # Interpreter
    "BlockInterpreter",
    # Tracer
    "ActivationTracer",
    "TraceDataset",
    # Certifier
    "BlockCertifier",
    "CertificationMetrics",
    # Certificate
    "Certificate",
    "generate_certificate",
]

__version__ = "0.1.0"
