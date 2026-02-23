# EloSRM

An Elo-based rating system for TopCoder Single Round Matches

## Features

- Log-rank performance transform
- Tuned by minimization of L₁ loss
- Designed to predict performance and estimate proficiency
- Optimized implementations in both C++ and Python

## Documentation

https://arxiv.org/abs/1905.00961

## License

MIT license

## Repository Structure
```
EloSRM/
├── README.md
├── LICENSE
├── TCSRM.md                    dataset description
├── cpp/
│   ├── EloSRM.h                C++ implementation
│   ├── runner.cpp              simple runner
│   └── runner.out.txt          sample output
├── python/
│   ├── elosrm.py               Python implementation
│   ├── runner.py               simple runner
│   ├── requirements.txt        Python dependencies
│   └── runner.out.txt          sample output
└── data/
    └── community.topcoder.com  TC SRM data files
```

## Implementation

**EloSRM.h / elosrm.py:**
- Core rating algorithm
- Matches the arXiv paper specification
- All system parameters documented in code

**runner.cpp / runner.py:**
- Parses TopCoder XML data files
- Computes ratings for all rounds
- Reports statistics

## Quick Start

### C++ Version
```bash
cd cpp
g++ -O3 -fopenmp -std=c++17 runner.cpp -o runner
./runner ../data/community.topcoder.com
```

### Python Version
```bash
cd python
pip install -r requirements.txt
python runner.py ../data/community.topcoder.com
```
