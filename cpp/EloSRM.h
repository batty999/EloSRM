//
// EloSRM rating system for TopCoder SRM
// https://github.com/batty999/EloSRM
// (c) 2019-2026 Fred Batty
//

#include <cmath>
#include <vector>

#ifndef M_LN10
#define M_LN10 2.302585092994046
#define M_LN2 0.6931471805599453
#define M_LOG2E 1.4426950408889634
#endif


namespace EloSRM
{
    // Constants
    const int VERSION = 7;
    const double EloScale = M_LN10 / 400;
    const double K0 = 400 * M_LN2 / M_LN10;
    const double R0 = 1200;                     // Initial rating

    // System parameters
    const double K = 648.3147599935407;         // Base K-factor
    const double C = 3.8884120557511483;        // Competition factor
    const double M = 4.44015774770823;          // Max performance
    const double B = 41.84027892146161;         // Performance bonus
    const double W1 = 0.20058285315948782;      // Experience weight
    const double D = 112.17636272246507;        // Recent period
    const double gamma = 0.4599945496035186;    // Frequency factor
    const double G = 50.67198262989024;         // Inflation per year

    // Runtime computed values
    time_t t0;                                  // First round date
    double R_init = R0;                         // adjusted for inflation
    double B2;
    double lambda;
    double G_sec;

    struct Player {
        int num_ratings = 0;
        double rating = R_init;
        double recent_rounds = 0;
        time_t last_round = 0;
    };

    struct Result {
        Player* player;
        double score;
        double rate;            // 10 ** rating / 400
        double delta_r;
        double perf;
    };

    void rateDivision(Result* results, int n) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            Player* pi = results[i].player;
            double si = results[i].score;
            double ri = results[i].rate;
            double erank = 1, arank = 1;
            double mu = 1, var = 1;
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                double sj = results[j].score;
                double rj = results[j].rate;

                double wj = rj / (ri + rj);     // win probability
                mu += wj;
                var += wj * (1 - wj);
                if (si == sj) {
                    erank += .5;
                    arank += .5;
                }
                else {
                    erank += wj;
                    arank += si < sj;
                }
            }
            double perf = log(erank / arank) * M_LOG2E;
            double perf1 = var / mu;

            double pa = perf * M / (M + abs(perf));
            pa += B2 * perf1;
            double ef = sqrt(1. + pi->num_ratings * W1);
            double cf = 1 + C * perf1;
            double ff = pow(pi->recent_rounds, gamma);
            double w = ef * cf * ff;
            double delta_r = K * pa / w;

            results[i].delta_r = delta_r;
            results[i].perf = perf;
        }
    }

    void rateRound(std::vector<Result>& results, time_t round_time) {
        R_init = R0 + G_sec * (round_time - t0);
        int n = (int)results.size();
        for (int i = 0; i < n; i++) {
            Result* ri = &results[i];
            Player* pi = ri->player;
            if (!pi->num_ratings) {
                pi->rating = R_init;
                pi->recent_rounds = 1;
            }
            else {
                auto t_diff = round_time - pi->last_round;
                double decay = exp(lambda * t_diff);
                pi->recent_rounds = 1 + pi->recent_rounds * decay;
            }
            ri->rate = exp(pi->rating * EloScale);
        }
        rateDivision(&results[0], n);

        for (int i = 0; i < n; i++) {
            Player* pi = results[i].player;
            pi->num_ratings++;
            pi->rating += results[i].delta_r;
            pi->last_round = round_time;
        }
    }

    void init(time_t first_round_time) {
        t0 = first_round_time;
        R_init = R0;
        G_sec = G / (365.25 * 24 * 3600);
        B2 = B / K0;
        lambda = -M_LN2 / (D * 24 * 3600);
    }
}
