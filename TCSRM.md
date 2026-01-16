# TopCoder SRM dataset

## Overview

Historical data from TopCoder Single Round Matches (2001-2024):
- **1,402** total rounds in round list
- **1,396** result files with participant data  
- **1,362** rated rounds
- **82,174** unique players
- **817,969** rating changes computed

## Data Files
```
data/community.topcoder.com/
├── tc_module=BasicData&c=dd_round_list (list of 1402 rounds)
└── tc_module=BasicData&c=dd_round_results&rd=<round_id> (1396 files)
```

## Data Format

Files are in XML format as originally provided by TopCoder's data feeds.

**Note:** Since TopCoder occasionally modified or removed results after initial publication, files downloaded at different times may differ slightly.

## Parsing Rules

The following rules are applied when processing the data:

1. **Timestamps:** Interpreted as UTC (no DST adjustment)
2. **Rated participants:** Only rows with `rated_flag = 1` are included
3. **Duplicate handling:** Rows with duplicate `coder_id` within a round are excluded

## Data Availability

This historical dataset is no longer available from TopCoder's website and is preserved in this repository for research purposes.

