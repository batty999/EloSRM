import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

from elosrm import EloSRM


# ---------------------------------------------------------
# Data structures
# ---------------------------------------------------------

@dataclass
class Row:
    coder_id: int
    final_score: float


@dataclass
class Round:
    round_id: int
    date: float
    divisions: List[List[Row]] = None


def parse_utc(date_str: str) -> float:
    """Parse YYYY-MM-DD HH:MM string to timestamp"""
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
        dt = dt.replace(tzinfo=timezone.utc) # No DST
        return dt.timestamp()
    
    except ValueError:
        return 0.0


def extract_tag(xml_block: str, tag_name: str) -> str:
    """Extract content using string operations """
    start_tag = f'<{tag_name}>'
    end_tag = f'</{tag_name}>'
    
    start_idx = xml_block.find(start_tag)
    if start_idx == -1:
        return ''
    
    start_idx += len(start_tag)
    end_idx = xml_block.find(end_tag, start_idx)
    
    if end_idx == -1:
        return ''
    
    return xml_block[start_idx:end_idx]


def parse_rounds(content: str) -> List[Round]:
    """Parse dd_round_list content into Round objects."""
    rounds = []
    pos = 0

    while True:
        row_start = content.find("<row>", pos)
        if row_start == -1: break
        row_start += 5

        row_end = content.find("</row>", row_start)
        if row_end == -1: break
        pos = row_end + 6

        block = content[row_start:row_end]

        # Extract round_id
        rid = extract_tag(block, "round_id")
        if not rid:
            continue

        # Extract date string
        date_str = extract_tag(block, "date")
        if not date_str:
            continue

        # Convert
        try:
            round_id = int(rid)
        except ValueError:
            continue

        date_ts = parse_utc(date_str)
        if date_ts <= 0:
            continue

        rounds.append(Round(round_id, date_ts))

    return rounds


def parse_results(content: str) -> List[List[Row]]:
    """Parse dd_round_results into division lists."""
    divisions = [[], []]
    seen_coders = set()
    
    # Find all row blocks using string operations
    pos = 0
    while True:
        # Find next <row> tag
        row_start = content.find('<row>', pos)
        if row_start == -1: break
        row_start += 5
        
        # Find closing </row>
        row_end = content.find('</row>', row_start)
        if row_end == -1: break
        pos = row_end + 6  # Move past </row>

        # Extract the block
        block = content[row_start:row_end]

        # Check rated_flag
        rated = extract_tag(block, 'rated_flag')
        if rated != "1":
            continue

        # coder
        cid = extract_tag(block, 'coder_id')
        if not cid:
            continue
        
        coder_id = int(cid)
        if coder_id in seen_coders: # skip dups
            continue 
        seen_coders.add(coder_id)
    
        # score
        score_str = extract_tag(block, 'final_points')
        score = float(score_str) if score_str else 0.0
    
        # division
        div_str = extract_tag(block, 'division')
        div = int(div_str) if div_str else 1
        assert div in [1, 2]
        rows = divisions[div - 1]

        rows.append(Row(coder_id, score))

    return divisions

# ---------------------------------------------------------
# Multiprocessing
# ---------------------------------------------------------
def process_single_round(args):
    """Worker: load and parse a single dd_round_results file."""
    round_id, data_dir = args
    
    path = data_dir / f"tc_module=BasicData&c=dd_round_results&rd={round_id}"

    try:
        with open(path, 'r', encoding='ISO-8859-1', errors='ignore') as f:
            content = f.read()
    except:
        return []
    
    return parse_results(content) if content else []

def parse_rounds_parallel(rounds: List[Round], data_dir: Path) -> None:
    """Parse all rounds in parallel and fill round.divisions."""
    
    # Prepare arguments for each round
    args = ((r.round_id, data_dir) for r in rounds)

    num_workers = min(cpu_count() // 2, len(rounds))
    # print(f"Using {num_workers} worker processes")
    
    with Pool(num_workers) as pool:
        results = pool.map(process_single_round, args)

    # Assign results back to rounds
    for r, divisions in zip(rounds, results):
        r.divisions = divisions


# ---------------------------------------------------------
# Main Rating Procedure
# ---------------------------------------------------------

def main():
    parse_start = time.time()

    # Determine data dir
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/community.topcoder.com")
    print(f"Using data directory: {data_dir}")

    rl_path = data_dir / "tc_module=BasicData&c=dd_round_list"
    if not rl_path.exists():
        print("Round list file not found.")
        return 1
    
    # Read and parse round list
    try:
        rl_content = rl_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        print("Couldn't read round list file")
        return 1

    rounds = parse_rounds(rl_content)
    if not rounds:
        print("No rounds found.")
        return 1

    rounds.sort(key=lambda r: r.date)
    print(f"Processing {len(rounds)} rounds...")

    # Phase 1: parse result files
    parse_rounds_parallel(rounds, data_dir)
    parse_end = time.time()

    # Phase 2: rating pass
    players: Dict[int, EloSRM.Player] = {}
    
    # Initialize rating system with first round date
    if rounds:
        EloSRM.init(rounds[0].date)
    
    num_deltas = 0
    max_delta = 0.0
    sum_deltas = 0.0
    sum_delta2 = 0.0
    sum_err = 0.0
    sum_errc = 0.0
    max_rating = 0.0
    
    rating_start = time.time()

    # Preallocated result objects
    RESULT_POOL_SIZE = 4000
    result_pool = [EloSRM.Result(None, 0.0) for _ in range(RESULT_POOL_SIZE)]

    for ridx, round in enumerate(rounds):
        for rows in round.divisions:
            if not rows: continue

            # Setup results for this div
            n = len(rows)
            
            # Expand pool if needed
            if n > RESULT_POOL_SIZE:
                result_pool.extend([EloSRM.Result(None, 0.0) 
                        for _ in range(n - RESULT_POOL_SIZE)])
                RESULT_POOL_SIZE = n
                  
            results = result_pool[:n]
            for i, row in enumerate(rows):
                if row.coder_id not in players:
                    players[row.coder_id] = EloSRM.Player()
                
                result = results[i]
                result.player = players[row.coder_id]
                result.score = row.final_score

            # Rate the round
            EloSRM.rate_round(results, round.date)

            # Collect statistics
            for res in results:
                delta = res.delta_r
                rating = res.player.rating
                if rating > max_rating: max_rating = rating
                if delta > max_delta: max_delta = delta
                sum_deltas += delta
                sum_delta2 += delta * delta
                sum_errc += abs(res.perf) 
                t = sum_err + sum_errc
                sum_errc -= t - sum_err
                sum_err = t
            
            num_deltas += n

    rating_end = time.time()
    
    if num_deltas == 0:
        print("No rating changes computed.")
        return 0
    
    # Calculate statistics
    mean_delta = sum_deltas / num_deltas
    stddev_delta = math.sqrt(sum_delta2 / num_deltas - mean_delta ** 2)
    avg_error = sum_err / num_deltas

    print(f"\nRating changes computed: {num_deltas}")
    print(f"Max rating: {max_rating:.4f}")
    print(f"Max delta: {max_delta:.4f}")
    print(f"Mean delta: {mean_delta:.4f}")
    print(f"Stddev: {stddev_delta:.4f}")
    print(f"Error: {avg_error}")
    
    parse_time = parse_end - parse_start
    rating_time = rating_end - rating_start
    
    print(f"\nParse time: {parse_time:.3f}s")
    print(f"Rating time: {rating_time:.3f}s")
    now = datetime.now()
    print("Date: ", now.strftime("%b %d %Y %H:%M:%S"))

    return 0

# ---------------------------------------------------------
# Profiling entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    import cProfile, pstats, io, sys
    profiler = cProfile.Profile()

    prof = 0
    if prof: profiler.enable()

    exit_code = main()

    if prof: 
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats(pstats.SortKey.TIME)
        ps.print_stats(30)
        sys.stdout.write(s.getvalue())

    sys.exit(exit_code)
