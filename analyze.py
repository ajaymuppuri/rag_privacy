#!/usr/bin/env python3
"""Run RAG analysis on the privacy document corpus."""

from rag.analyze import print_summary, run_analysis

if __name__ == "__main__":
    report = run_analysis()
    print_summary(report)
