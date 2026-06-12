#!/usr/bin/env python
"""06_summarize_claims - make claim-oriented JSON summaries."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.summarize_claims import main


if __name__ == "__main__":
    main()
