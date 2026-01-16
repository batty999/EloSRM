// EloSRM rating runner
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <chrono>
#include <charconv>
#include <assert.h>

#include "EloSRM.h"


using namespace std;
namespace fs = std::filesystem;


struct Row { 
    int coder_id = 0; 
    double final_score = 0; 
};

struct Round { 
    int round_id = 0; 
    time_t date = 0;
    size_t num_rated = 0;
    vector<Row> divisions[2];
};

static vector<Round> rounds;
static unordered_map<int, EloSRM::Player> players;

static time_t parseutc(const char* s) {
    std::tm t{};
    if (sscanf(s, "%d-%d-%d %d:%d", &t.tm_year, &t.tm_mon,
                    &t.tm_mday, &t.tm_hour, &t.tm_min) != 5)
        return 0;
    t.tm_year -= 1900;    // years since 1900
    t.tm_mon  -= 1;       // months since January
    t.tm_sec  = 0;
#ifdef _WIN32
    return _mkgmtime(&t);
#else
    return timegm(&t);
#endif
}

// Extract tag content from XML block
static string_view tag(string_view b, const char *t) {
    char open[64], close[64];
    auto ol = snprintf(open, sizeof(open), "<%s>", t);
    snprintf(close, sizeof(close), "</%s>", t);
    size_t pa = b.find(open);
    if (pa == string_view::npos) return {};
    pa += ol;
    size_t pb = b.find(close, pa);
    if (pb == string_view::npos) return {};
    return b.substr(pa, pb - pa);
}

static int tag_int(std::string_view block, const char* name)
{
    auto sv = tag(block, name);
    int value = 0;
    std::from_chars(sv.data(), sv.data() + sv.size(), value);
    return value;
}

static double tag_double(std::string_view block, const char* name)
{
    auto sv = tag(block, name);
    double value = 0;
    std::from_chars(sv.data(), sv.data() + sv.size(), value);
    return value;
}

static void parse_rounds(const string &c) {
    rounds.clear();
    for (size_t pos = 0; (pos = c.find("<row", pos)) != string::npos;) {
        if ((pos = c.find('>', pos)) == string::npos) break;
        size_t start = ++pos;
        if ((pos = c.find("</row>", pos)) == string::npos) break;
        string_view b(c.c_str() + start, pos - start);
        Round r;
        r.round_id = tag_int(b, "round_id");
        auto ds = tag(b, "date");
        r.date = parseutc(ds.data());
        rounds.push_back(r);
        pos += 6;
    }
}

static void parse_results(Round& round, const string &c) {
    vector<Row> rows;
    unordered_set<int> seen;
    for (size_t pos = 0; (pos = c.find("<row", pos)) != string::npos;) {
        if ((pos = c.find('>', pos)) == string::npos) break;
        size_t start = ++pos;
        if ((pos = c.find("</row>", pos)) == string::npos) break;
        string_view b(c.c_str() + start, pos - start);
        pos += 6;
        if (!tag_int(b, "rated_flag")) continue;
        Row r;
        r.coder_id = tag_int(b, "coder_id");
        if (!seen.insert(r.coder_id).second) continue;    // skip dups
        r.final_score = tag_double(b, "final_points");
        int div = tag_int(b, "division");
        assert(div == 1 || div == 2);
        round.divisions[div-1].push_back(r);
        round.num_rated++;
    }
}

static bool read_file(const fs::path& p, string& c) {
    ifstream in(p, ios::binary);
    if (!in) return false;
    in.seekg(0, ios::end);
    c.resize(in.tellg());
    in.seekg(0, ios::beg);
    in.read(&c[0], c.size());
    return true;
}


int main(int argc, char **argv) {
    using clk = chrono::high_resolution_clock;
    auto parse_t0 = clk::now();
 
    // Allow optional argv[1] to override data directory
    string dataDir = argc > 1 ? argv[1] : "../data/community.topcoder.com";
    cout << "Using data directory: " << dataDir << "\n";

    fs::path rlPath = fs::path(dataDir) / "tc_module=BasicData&c=dd_round_list";
    if (!fs::exists(rlPath)) {
        cerr << "Round list file not found: " << rlPath << "\n";
        return 1;
    }

    string rlContent;
    if (!read_file(rlPath, rlContent)) {
        cerr << "Couldn't open round list.\n"; 
        return 1;
    }

    parse_rounds(rlContent);
    if (rounds.empty()) { 
        cerr << "dd_round_list is empty.\n"; 
        return 1; 
    }
    stable_sort(rounds.begin(), rounds.end(), [](const auto &a, const auto &b) { 
        return a.date < b.date; 
    });

    // Phase 1: parse all result files
    cout << "Processing " << rounds.size() << " rounds...\n";
#pragma omp parallel for
    for (int i = 0; i < (int)rounds.size(); ++i) {
        const auto &ri = rounds[i];
        char fname[128];
        snprintf(fname, sizeof(fname), "tc_module=BasicData&c=dd_round_results&rd=%d", ri.round_id);
        fs::path p = fs::path(dataDir) / string(fname);
        if (!fs::exists(p)) continue;
    
        string content;
        if (!read_file(p, content)) continue;
        
        parse_results(rounds[i], content);
    }
    auto parse_t1 = clk::now();

    // Phase 2: rating pass
    players.clear();
    EloSRM::init(rounds.front().date);
    size_t numRounds = 0;
    size_t numDeltas = 0;
    double maxDelta = 0.0;
    double sumDeltas = 0.0;
    double sumDelta2 = 0.0;
    double sumErr = 0.0;
    double sumErrC = 0.0;
    double maxRating = 0.0;
    auto rate_t0 = clk::now();

    for (const auto &round : rounds) {
        if (round.num_rated) numRounds++;
        for (const auto& rows : round.divisions) {
            if (rows.empty()) continue;

            vector<EloSRM::Result> results;
            results.reserve(rows.size());
            for (const auto& r : rows) {
                auto& pl = players[r.coder_id];
                results.push_back({ &pl, r.final_score });
            }

            EloSRM::rateRound(results, round.date);

            for (const auto& res : results) {
                auto d = res.delta_r;
                maxRating = max(maxRating, res.player->rating);
                maxDelta = max(maxDelta, d);
                sumDeltas += d;
                sumDelta2 += d * d;
                sumErrC += fabs(res.perf);
                double t = sumErr + sumErrC;
                sumErrC -= t - sumErr;
                sumErr = t;
            }
            numDeltas += results.size();
        }
    }
    auto rate_t1 = clk::now();

    if (!numDeltas) { 
        cout << "No rating changes computed.\n"; 
        return 0; 
    }

    double mean = sumDeltas / numDeltas;
    double stddev = sqrt(sumDelta2 / numDeltas - mean * mean);

    cout << "Rated rounds: " << numRounds << "\n";
    cout << "Unique players: " << players.size() << "\n";
    cout << "\nRating changes computed: " << numDeltas << "\n";
    cout << fixed << setprecision(4);
    cout << "Max rating: " << maxRating << "\n";
    cout << "Max delta: " << maxDelta << "\n";
    cout << "Mean delta: " << mean << "\n";
    cout << "Stddev: " << stddev << "\n";
    double err = sumErr / numDeltas;
    cout << setprecision(16);
    cout << "Error: " << err << "\n";

    double parse_secs = chrono::duration<double>(parse_t1 - parse_t0).count();
    double rate_secs = chrono::duration<double>(rate_t1 - rate_t0).count();
    cout << fixed << setprecision(3);
    cout << "\nParse time: " << parse_secs << "s\n";
    cout << "Rating time: " << rate_secs << "s\n";
    cout << "Built: " << __DATE__ " " __TIME__ << "\n";
    return 0;
}
