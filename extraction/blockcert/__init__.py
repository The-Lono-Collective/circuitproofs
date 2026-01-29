"""
BlockCert: Certified Circuit Extraction Framework

This module implements the BlockCert methodology for extracting circuits
from transformer models with certified error bounds.

Components:
- ir: Intermediate Representation (.npz storage)
- interpreter: Pure Python block execution
- certifier: Metrics computation with auto-LiRPA
- certificate: Certificate artifact generation
"""

from extraction.blockcert.ir import BlockIR, AttentionIR, MLPIR, NormIR, TraceRecord, TraceDataset
from extraction.blockcert.interpreter import BlockInterpreter
from extraction.blockcert.certifier import BlockCertifier, CertificationMetrics
from extraction.blockcert.certificate import Certificate, generate_certificate

__all__ = [
    # IR
    "BlockIR",
    "AttentionIR",
    "MLPIR",
    "NormIR",
    # Trace data
    "TraceRecord",
    "TraceDataset",
    # Interpreter
    "BlockInterpreter",
    # Certifier
    "BlockCertifier",
    "CertificationMetrics",
    # Certificate
    "Certificate",
    "generate_certificate",
]

__version__ = "0.1.0"
