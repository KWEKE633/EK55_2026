/*
 * MIT License
 *
 * Copyright (c) 2026 [EK55]
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma GCC optimize("O3,unroll-loops,tree-vectorize")
#pragma GCC optimize("align-functions=64")
#pragma GCC optimize("schedule-insns")
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vl,popcnt,lzcnt,bmi2")

#include "io_struct.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>

using io_struct::input_t;
using io_struct::output_t;
using io_struct::PARAMETER_COUNT;

alignas(64) static uint64_t comp_Adj[64];
alignas(64) static long long W[64];
alignas(64) static __m512i v_vec[64][8];
alignas(64) static __m512i M_vec[8];

static long long best_W_S = 1000000000000000000LL;
static uint64_t best_S = 0;
static uint64_t init_P_global = 0;

#define ADD_VEC(n, v_idx)                           \
	n##0 = _mm512_add_epi32(n##0, v_vec[v_idx][0]); \
	n##1 = _mm512_add_epi32(n##1, v_vec[v_idx][1]); \
	n##2 = _mm512_add_epi32(n##2, v_vec[v_idx][2]); \
	n##3 = _mm512_add_epi32(n##3, v_vec[v_idx][3]); \
	n##4 = _mm512_add_epi32(n##4, v_vec[v_idx][4]); \
	n##5 = _mm512_add_epi32(n##5, v_vec[v_idx][5]); \
	n##6 = _mm512_add_epi32(n##6, v_vec[v_idx][6]); \
	n##7 = _mm512_add_epi32(n##7, v_vec[v_idx][7]);

#define BRANCH_SINGLE(v_idx, next_S, next_U, next_W)                                                          \
	do                                                                                                        \
	{                                                                                                         \
		if (next_W < best_W_S)                                                                                \
		{                                                                                                     \
			__m512i n0 = _mm512_add_epi32(s0, v_vec[v_idx][0]);                                               \
			__m512i n1 = _mm512_add_epi32(s1, v_vec[v_idx][1]);                                               \
			if (!(_mm512_cmpgt_epi32_mask(n0, M_vec[0]) | _mm512_cmpgt_epi32_mask(n1, M_vec[1])))             \
			{                                                                                                 \
				__m512i n2 = _mm512_add_epi32(s2, v_vec[v_idx][2]);                                           \
				__m512i n3 = _mm512_add_epi32(s3, v_vec[v_idx][3]);                                           \
				if (!(_mm512_cmpgt_epi32_mask(n2, M_vec[2]) | _mm512_cmpgt_epi32_mask(n3, M_vec[3])))         \
				{                                                                                             \
					__m512i n4 = _mm512_add_epi32(s4, v_vec[v_idx][4]);                                       \
					__m512i n5 = _mm512_add_epi32(s5, v_vec[v_idx][5]);                                       \
					if (!(_mm512_cmpgt_epi32_mask(n4, M_vec[4]) | _mm512_cmpgt_epi32_mask(n5, M_vec[5])))     \
					{                                                                                         \
						__m512i n6 = _mm512_add_epi32(s6, v_vec[v_idx][6]);                                   \
						__m512i n7 = _mm512_add_epi32(s7, v_vec[v_idx][7]);                                   \
						if (!(_mm512_cmpgt_epi32_mask(n6, M_vec[6]) | _mm512_cmpgt_epi32_mask(n7, M_vec[7]))) \
						{                                                                                     \
							dfs_vc(next_S, next_U, next_W, n0, n1, n2, n3, n4, n5, n6, n7);                   \
						}                                                                                     \
					}                                                                                         \
				}                                                                                             \
			}                                                                                                 \
		}                                                                                                     \
	} while (0)

__attribute__((hot, noinline)) static void dfs_vc(uint64_t S, uint64_t U, long long cur_W_S,
												  __m512i s0, __m512i s1, __m512i s2, __m512i s3,
												  __m512i s4, __m512i s5, __m512i s6, __m512i s7)
{
	if (cur_W_S >= best_W_S)
		return;
	int max_deg = 0;
	int best_u = -1;
	long long lower_bound = 0;
	long long allowed_W = best_W_S - cur_W_S;
	__m512i lb0 = s0, lb1 = s1, lb2 = s2, lb3 = s3, lb4 = s4, lb5 = s5, lb6 = s6, lb7 = s7;
	uint64_t temp_U = U;
	uint64_t unmatched = U;
	while (temp_U)
	{
		uint64_t bit_v = temp_U & -temp_U;
		int v = __builtin_ctzll(bit_v);
		temp_U &= temp_U - 1;
		uint64_t adj_U = comp_Adj[v] & U;
		if (!adj_U)
		{
			U ^= bit_v;
			unmatched ^= bit_v;
			continue;
		}
		int deg = __builtin_popcountll(adj_U);
		if (deg > max_deg)
		{
			max_deg = deg;
			best_u = v;
		}
		if (unmatched & bit_v)
		{
			uint64_t avail = adj_U & unmatched;
			if (avail)
			{
				uint64_t bit_w = avail & -avail;
				int w = __builtin_ctzll(bit_w);

				lower_bound += W[w];
				if (lower_bound >= allowed_W)
					return;
				lb0 = _mm512_add_epi32(lb0, _mm512_min_epi32(v_vec[v][0], v_vec[w][0]));
				lb1 = _mm512_add_epi32(lb1, _mm512_min_epi32(v_vec[v][1], v_vec[w][1]));
				lb2 = _mm512_add_epi32(lb2, _mm512_min_epi32(v_vec[v][2], v_vec[w][2]));
				lb3 = _mm512_add_epi32(lb3, _mm512_min_epi32(v_vec[v][3], v_vec[w][3]));
				lb4 = _mm512_add_epi32(lb4, _mm512_min_epi32(v_vec[v][4], v_vec[w][4]));
				lb5 = _mm512_add_epi32(lb5, _mm512_min_epi32(v_vec[v][5], v_vec[w][5]));
				lb6 = _mm512_add_epi32(lb6, _mm512_min_epi32(v_vec[v][6], v_vec[w][6]));
				lb7 = _mm512_add_epi32(lb7, _mm512_min_epi32(v_vec[v][7], v_vec[w][7]));

				unmatched ^= bit_v | bit_w;
			}
			else
			{
				unmatched ^= bit_v;
			}
		}
	}
	do
	{
		if (_mm512_cmpgt_epi32_mask(lb0, M_vec[0]) | _mm512_cmpgt_epi32_mask(lb1, M_vec[1]))
			return;
		if (_mm512_cmpgt_epi32_mask(lb2, M_vec[2]) | _mm512_cmpgt_epi32_mask(lb3, M_vec[3]))
			return;
		if (_mm512_cmpgt_epi32_mask(lb4, M_vec[4]) | _mm512_cmpgt_epi32_mask(lb5, M_vec[5]))
			return;
		if (_mm512_cmpgt_epi32_mask(lb6, M_vec[6]) | _mm512_cmpgt_epi32_mask(lb7, M_vec[7]))
			return;
	} while (0);

	if (max_deg == 0)
	{
		best_W_S = cur_W_S;
		best_S = S;
		return;
	}
	if (max_deg == 1)
	{
		long long greedy_W = cur_W_S;
		uint64_t greedy_S = S;
		uint64_t temp_U2 = U;
		__m512i g0 = s0, g1 = s1, g2 = s2, g3 = s3, g4 = s4, g5 = s5, g6 = s6, g7 = s7;
		bool possible = true;
		while (temp_U2)
		{
			uint64_t bit_v = temp_U2 & -temp_U2;
			int v = __builtin_ctzll(bit_v);
			uint64_t adj = comp_Adj[v] & temp_U2;
			if (adj)
			{
				uint64_t bit_w = adj & -adj;
				int w = __builtin_ctzll(bit_w);
				int pick = (W[v] < W[w]) ? v : w;

				greedy_W += W[pick];
				if (greedy_W >= best_W_S)
				{
					possible = false;
					break;
				}
				greedy_S |= (1ULL << pick);

				ADD_VEC(g, pick);
				temp_U2 ^= bit_v | bit_w;
			}
			else
			{
				temp_U2 &= temp_U2 - 1;
			}
		}
		if (possible)
		{
			do
			{
				if (_mm512_cmpgt_epi32_mask(g0, M_vec[0]) | _mm512_cmpgt_epi32_mask(g1, M_vec[1]))
					break;
				if (_mm512_cmpgt_epi32_mask(g2, M_vec[2]) | _mm512_cmpgt_epi32_mask(g3, M_vec[3]))
					break;
				if (_mm512_cmpgt_epi32_mask(g4, M_vec[4]) | _mm512_cmpgt_epi32_mask(g5, M_vec[5]))
					break;
				if (_mm512_cmpgt_epi32_mask(g6, M_vec[6]) | _mm512_cmpgt_epi32_mask(g7, M_vec[7]))
					break;
				best_W_S = greedy_W;
				best_S = greedy_S;
				return;
			} while (0);
		}
	}
	uint64_t N_u = comp_Adj[best_u] & U;
	long long w_Nu = 0;
	uint64_t temp_Nu = N_u;
	while (temp_Nu)
	{
		uint64_t bit = temp_Nu & -temp_Nu;
		w_Nu += W[__builtin_ctzll(bit)];
		temp_Nu &= temp_Nu - 1;
	}
	uint64_t bit_best_u = 1ULL << best_u;
	uint64_t next_U_branch1 = U ^ (N_u | bit_best_u);
	uint64_t next_U_branch2 = U ^ bit_best_u;
	if (w_Nu < W[best_u])
	{
		if (cur_W_S + w_Nu < best_W_S)
		{
			__m512i n0 = s0, n1 = s1, n2 = s2, n3 = s3, n4 = s4, n5 = s5, n6 = s6, n7 = s7;
			uint64_t temp1 = N_u;
			while (temp1)
			{
				uint64_t bit = temp1 & -temp1;
				int v_idx = __builtin_ctzll(bit);
				ADD_VEC(n, v_idx);
				temp1 &= temp1 - 1;
			}
			do
			{
				if (_mm512_cmpgt_epi32_mask(n0, M_vec[0]) | _mm512_cmpgt_epi32_mask(n1, M_vec[1]))
					break;
				if (_mm512_cmpgt_epi32_mask(n2, M_vec[2]) | _mm512_cmpgt_epi32_mask(n3, M_vec[3]))
					break;
				if (_mm512_cmpgt_epi32_mask(n4, M_vec[4]) | _mm512_cmpgt_epi32_mask(n5, M_vec[5]))
					break;
				if (_mm512_cmpgt_epi32_mask(n6, M_vec[6]) | _mm512_cmpgt_epi32_mask(n7, M_vec[7]))
					break;
				dfs_vc(S | N_u, next_U_branch1, cur_W_S + w_Nu, n0, n1, n2, n3, n4, n5, n6, n7);
			} while (0);
		}
		BRANCH_SINGLE(best_u, S | bit_best_u, next_U_branch2, cur_W_S + W[best_u]);
	}
	else
	{
		BRANCH_SINGLE(best_u, S | bit_best_u, next_U_branch2, cur_W_S + W[best_u]);

		if (cur_W_S + w_Nu < best_W_S)
		{
			__m512i n0 = s0, n1 = s1, n2 = s2, n3 = s3, n4 = s4, n5 = s5, n6 = s6, n7 = s7;
			uint64_t temp1 = N_u;
			while (temp1)
			{
				uint64_t bit = temp1 & -temp1;
				int v_idx = __builtin_ctzll(bit);
				ADD_VEC(n, v_idx);
				temp1 &= temp1 - 1;
			}
			do
			{
				if (_mm512_cmpgt_epi32_mask(n0, M_vec[0]) | _mm512_cmpgt_epi32_mask(n1, M_vec[1]))
					break;
				if (_mm512_cmpgt_epi32_mask(n2, M_vec[2]) | _mm512_cmpgt_epi32_mask(n3, M_vec[3]))
					break;
				if (_mm512_cmpgt_epi32_mask(n4, M_vec[4]) | _mm512_cmpgt_epi32_mask(n5, M_vec[5]))
					break;
				if (_mm512_cmpgt_epi32_mask(n6, M_vec[6]) | _mm512_cmpgt_epi32_mask(n7, M_vec[7]))
					break;
				dfs_vc(S | N_u, next_U_branch1, cur_W_S + w_Nu, n0, n1, n2, n3, n4, n5, n6, n7);
			} while (0);
		}
	}
}

void solve(input_t &input, output_t &output)
{
	io_struct::InitOutput(output);
	const int N = input.N;
	if (N == 0)
		return;
	std::pair<long long, int> w_idx[64];
	long long V_total_W = 0;
	for (int i = 0; i < N; ++i)
	{
		__m512i vsum = _mm512_setzero_epi32();
#pragma GCC unroll 8
		for (int k = 0; k < 8; ++k)
		{
			vsum = _mm512_add_epi32(vsum, _mm512_loadu_epi32(&input.v[i][k * 16]));
		}
		long long sum = _mm512_reduce_add_epi32(vsum);
		w_idx[i] = {sum, i};
		V_total_W += sum;
	}
	std::sort(w_idx, w_idx + N, [](const auto &a, const auto &b)
			  {
		if (a.first != b.first) return a.first > b.first;
		return a.second < b.second; });
	int map_n2o[64], map_o2n[64];
	for (int i = 0; i < N; ++i)
	{
		int orig = w_idx[i].second;
		map_n2o[i] = orig;
		map_o2n[orig] = i;
		W[i] = w_idx[i].first;
	}
	init_P_global = (N == 64) ? ~0ULL : ((1ULL << N) - 1);
	for (int i = 0; i < N; ++i)
		comp_Adj[i] = 0;
	for (int i = 0; i < N; ++i)
	{
		int o_i = map_o2n[i];
		uint64_t adj_i = 0;
		for (int j = 0; j < N; ++j)
		{
			if (!input.A[i][j] && i != j)
				adj_i |= (1ULL << map_o2n[j]);
		}
		comp_Adj[o_i] = adj_i;
	}
	for (int i = 0; i < N; ++i)
	{
		int orig = map_n2o[i];
#pragma GCC unroll 8
		for (int k = 0; k < 8; ++k)
			v_vec[i][k] = _mm512_loadu_epi32(&input.v[orig][k * 16]);
	}
	__m512i req_vec[8];
#pragma GCC unroll 8
	for (int k = 0; k < 8; ++k)
		req_vec[k] = _mm512_loadu_epi32(&input.r[k * 16]);
	alignas(64) __m512i V_sum_AVX[8];
#pragma GCC unroll 8
	for (int k = 0; k < 8; ++k)
		V_sum_AVX[k] = _mm512_setzero_epi32();
	for (int i = 0; i < N; ++i)
	{
#pragma GCC unroll 8
		for (int k = 0; k < 8; ++k)
			V_sum_AVX[k] = _mm512_add_epi32(V_sum_AVX[k], v_vec[i][k]);
	}
	bool impossible = false;
	__m512i zero_vec = _mm512_setzero_epi32();
#pragma GCC unroll 8
	for (int k = 0; k < 8; ++k)
	{
		M_vec[k] = _mm512_sub_epi32(V_sum_AVX[k], req_vec[k]);
		if (_mm512_cmpgt_epi32_mask(zero_vec, M_vec[k]))
			impossible = true;
	}
	if (impossible)
	{
		output.K_size = 0;
		asm volatile("" ::: "memory");
		std::_Exit(0);
	}
	uint64_t initial_C = 0;
	long long initial_C_W = 0;
	for (int i = 0; i < N; ++i)
	{
		if ((comp_Adj[i] & init_P_global) == 0)
		{
			initial_C |= (1ULL << i);
			initial_C_W += W[i];
		}
	}
	uint64_t initial_U = init_P_global & ~initial_C;
	best_W_S = 1000000000000000000LL;
	best_S = 0;
	int max_trials = (N <= 35) ? 2 : 5;
	for (int trial = 0; trial < max_trials; ++trial)
	{
		uint64_t R = initial_C;
		uint64_t P_greedy = initial_U;
		long long cur_W = initial_C_W;
		while (P_greedy)
		{
			int v = -1;
			if (trial == 0)
			{
				uint64_t bit_v = P_greedy & -P_greedy;
				v = __builtin_ctzll(bit_v);
			}
			else if (trial == 1)
			{
				int min_deg = 1000;
				uint64_t temp = P_greedy;
				while (temp)
				{
					uint64_t bit = temp & -temp;
					int cand = __builtin_ctzll(bit);
					int deg = __builtin_popcountll(comp_Adj[cand] & P_greedy);
					if (deg < min_deg)
					{
						min_deg = deg;
						v = cand;
					}
					temp &= temp - 1;
				}
			}
			else if (trial == 2)
			{
				int min_score = 10000;
				uint64_t temp = P_greedy;
				while (temp)
				{
					uint64_t bit = temp & -temp;
					int cand = __builtin_ctzll(bit);
					int deg = __builtin_popcountll(comp_Adj[cand] & P_greedy);
					int score = cand + deg * 4;
					if (score < min_score)
					{
						min_score = score;
						v = cand;
					}
					temp &= temp - 1;
				}
			}
			else if (trial == 3)
			{
				int min_score = 10000;
				uint64_t temp = P_greedy;
				while (temp)
				{
					uint64_t bit = temp & -temp;
					int cand = __builtin_ctzll(bit);
					int deg = __builtin_popcountll(comp_Adj[cand] & P_greedy);
					int score = (cand + deg) * 3;
					if (score < min_score)
					{
						min_score = score;
						v = cand;
					}
					temp &= temp - 1;
				}
			}
			else if (trial == 4)
			{
				int min_score = 10000;
				uint64_t temp = P_greedy;
				while (temp)
				{
					uint64_t bit = temp & -temp;
					int cand = __builtin_ctzll(bit);
					int deg = __builtin_popcountll(comp_Adj[cand] & P_greedy);
					int score = cand * 4 + deg;
					if (score < min_score)
					{
						min_score = score;
						v = cand;
					}
					temp &= temp - 1;
				}
			}
			uint64_t bit = 1ULL << v;
			R |= bit;
			cur_W += W[v];
			P_greedy &= ~comp_Adj[v];
			P_greedy &= ~bit;
		}

		long long cur_S_W = V_total_W - cur_W;
		__m512i c0 = _mm512_setzero_epi32(), c1 = _mm512_setzero_epi32();
		__m512i c2 = _mm512_setzero_epi32(), c3 = _mm512_setzero_epi32();
		__m512i c4 = _mm512_setzero_epi32(), c5 = _mm512_setzero_epi32();
		__m512i c6 = _mm512_setzero_epi32(), c7 = _mm512_setzero_epi32();
		uint64_t temp_S = init_P_global & ~R;
		while (temp_S)
		{
			uint64_t bit = temp_S & -temp_S;
			int v_idx = __builtin_ctzll(bit);
			ADD_VEC(c, v_idx);
			temp_S &= temp_S - 1;
		}

		bool valid = false;
		if (cur_S_W < best_W_S)
		{
			do
			{
				if (_mm512_cmpgt_epi32_mask(c0, M_vec[0]) | _mm512_cmpgt_epi32_mask(c1, M_vec[1]))
					break;
				if (_mm512_cmpgt_epi32_mask(c2, M_vec[2]) | _mm512_cmpgt_epi32_mask(c3, M_vec[3]))
					break;
				if (_mm512_cmpgt_epi32_mask(c4, M_vec[4]) | _mm512_cmpgt_epi32_mask(c5, M_vec[5]))
					break;
				if (_mm512_cmpgt_epi32_mask(c6, M_vec[6]) | _mm512_cmpgt_epi32_mask(c7, M_vec[7]))
					break;
				best_W_S = cur_S_W;
				best_S = init_P_global & ~R;
				valid = true;
			} while (0);
		}

		bool local_improved = true;
		while (local_improved)
		{
			local_improved = false;
			uint64_t not_R = init_P_global & ~R;

			uint64_t cand_add = not_R;
			while (cand_add)
			{
				uint64_t bit_v = cand_add & -cand_add;
				int v = __builtin_ctzll(bit_v);
				cand_add &= cand_add - 1;
				if ((comp_Adj[v] & R) == 0)
				{
					R |= bit_v;
					cur_W += W[v];
					c0 = _mm512_sub_epi32(c0, v_vec[v][0]);
					c1 = _mm512_sub_epi32(c1, v_vec[v][1]);
					c2 = _mm512_sub_epi32(c2, v_vec[v][2]);
					c3 = _mm512_sub_epi32(c3, v_vec[v][3]);
					c4 = _mm512_sub_epi32(c4, v_vec[v][4]);
					c5 = _mm512_sub_epi32(c5, v_vec[v][5]);
					c6 = _mm512_sub_epi32(c6, v_vec[v][6]);
					c7 = _mm512_sub_epi32(c7, v_vec[v][7]);
					long long new_cur_S_W = V_total_W - cur_W;
					if (valid && new_cur_S_W < best_W_S)
					{
						best_W_S = new_cur_S_W;
						best_S = init_P_global & ~R;
					}
					local_improved = true;
					break;
				}
			}
			if (local_improved)
				continue;

			while (not_R)
			{
				uint64_t bit_v = not_R & -not_R;
				int v = __builtin_ctzll(bit_v);
				not_R &= not_R - 1;
				uint64_t conflict = comp_Adj[v] & R;
				if (conflict && (conflict & (conflict - 1)) == 0)
				{
					int u = __builtin_ctzll(conflict);
					if (W[v] > W[u] && ((1ULL << u) & initial_C) == 0)
					{
						long long new_cur_S_W = V_total_W - (cur_W - W[u] + W[v]);
						if (new_cur_S_W < best_W_S)
						{
							__m512i n0 = _mm512_add_epi32(c0, _mm512_sub_epi32(v_vec[u][0], v_vec[v][0]));
							__m512i n1 = _mm512_add_epi32(c1, _mm512_sub_epi32(v_vec[u][1], v_vec[v][1]));
							if (_mm512_cmpgt_epi32_mask(n0, M_vec[0]) | _mm512_cmpgt_epi32_mask(n1, M_vec[1]))
								continue;

							__m512i n2 = _mm512_add_epi32(c2, _mm512_sub_epi32(v_vec[u][2], v_vec[v][2]));
							__m512i n3 = _mm512_add_epi32(c3, _mm512_sub_epi32(v_vec[u][3], v_vec[v][3]));
							if (_mm512_cmpgt_epi32_mask(n2, M_vec[2]) | _mm512_cmpgt_epi32_mask(n3, M_vec[3]))
								continue;

							__m512i n4 = _mm512_add_epi32(c4, _mm512_sub_epi32(v_vec[u][4], v_vec[v][4]));
							__m512i n5 = _mm512_add_epi32(c5, _mm512_sub_epi32(v_vec[u][5], v_vec[v][5]));
							if (_mm512_cmpgt_epi32_mask(n4, M_vec[4]) | _mm512_cmpgt_epi32_mask(n5, M_vec[5]))
								continue;

							__m512i n6 = _mm512_add_epi32(c6, _mm512_sub_epi32(v_vec[u][6], v_vec[v][6]));
							__m512i n7 = _mm512_add_epi32(c7, _mm512_sub_epi32(v_vec[u][7], v_vec[v][7]));
							if (_mm512_cmpgt_epi32_mask(n6, M_vec[6]) | _mm512_cmpgt_epi32_mask(n7, M_vec[7]))
								continue;

							R = (R ^ conflict) | bit_v;
							cur_W = cur_W - W[u] + W[v];
							best_W_S = new_cur_S_W;
							best_S = init_P_global & ~R;
							c0 = n0;
							c1 = n1;
							c2 = n2;
							c3 = n3;
							c4 = n4;
							c5 = n5;
							c6 = n6;
							c7 = n7;
							valid = true;
							local_improved = true;
							break;
						}
					}
				}
			}
		}
	}
	if (initial_U)
	{
		__m512i z = _mm512_setzero_epi32();
		dfs_vc(0, initial_U, 0, z, z, z, z, z, z, z, z);
	}
	if (best_W_S == 1000000000000000000LL)
	{
		output.K_size = 0;
		asm volatile("" ::: "memory");
		std::_Exit(0);
	}
	uint64_t R_final = init_P_global & ~best_S;
	int k_size = 0;
	long long best_T = 0;
	while (R_final)
	{
		uint64_t bit = R_final & -R_final;
		int new_i = __builtin_ctzll(bit);
		R_final &= R_final - 1;
		output.members[k_size++] = map_n2o[new_i] + 1;
		best_T += W[new_i];
	}
	std::sort(output.members.begin(), output.members.begin() + k_size);
	output.K_size = k_size;
	output.T = best_T;
	asm volatile("" ::: "memory");
	std::_Exit(0);
}
