# =============================================================================
# 스트리밍 알고리즘 구현 및 분석
# Bloom Filter + Count-Min Sketch
# 데이터: MovieLens 1M (ratings.dat — 1,000,209 레코드)
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import math
import time
import random
import hashlib
import tracemalloc
import collections
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트
plt.rcParams['axes.unicode_minus'] = False
try:
    fp = r'C:\Windows\Fonts\malgun.ttf'
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = 'Malgun Gothic'
except Exception:
    pass

BASE   = r"C:\Users\chldl\OneDrive\바탕 화면\대학\빅데이터\streaming"
FIGS   = os.path.join(BASE, "figures")
DATA   = os.path.join(BASE, "ml-1m", "ratings.dat")
os.makedirs(FIGS, exist_ok=True)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, f"{name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {name}.png")

def stream_ratings(path=DATA):
    """ratings.dat을 한 줄씩 스트리밍. 각 레코드: (user_id, movie_id, rating)"""
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 3:
                yield parts[0], parts[1], parts[2]   # user_id, movie_id, rating


# =============================================================================
# 1. Bloom Filter
# =============================================================================
class BloomFilter:
    """
    Bloom Filter: 원소 포함 여부 근사 판정 (False Positive 가능, False Negative 없음)

    핵심 로직:
      - m 비트 배열 초기화 (모두 0)
      - 삽입: k개 해시 함수 → 해당 비트 위치를 1로 설정
      - 조회: k개 위치 모두 1이면 "존재 가능", 하나라도 0이면 "확실히 없음"
    """
    def __init__(self, n_bits: int, n_hash: int, seed: int = 42):
        self.m = n_bits
        self.k = n_hash
        self.bits = bytearray(math.ceil(n_bits / 8))
        self.seeds = [seed + i * 2654435761 for i in range(n_hash)]
        self.n_inserted = 0

    def _hashes(self, item: str):
        b = item.encode('utf-8')
        for s in self.seeds:
            h = int(hashlib.md5(b + s.to_bytes(8, 'little')).hexdigest(), 16)
            yield h % self.m

    def add(self, item: str):
        for pos in self._hashes(item):
            self.bits[pos >> 3] |= (1 << (pos & 7))
        self.n_inserted += 1

    def __contains__(self, item: str) -> bool:
        return all((self.bits[pos >> 3] >> (pos & 7)) & 1 for pos in self._hashes(item))

    @property
    def memory_bytes(self):
        return len(self.bits)

    @staticmethod
    def optimal_params(n_items: int, fpr: float):
        """목표 FPR에 대한 최적 m, k 계산"""
        m = math.ceil(-n_items * math.log(fpr) / (math.log(2) ** 2))
        k = max(1, round((m / n_items) * math.log(2)))
        return m, k


# =============================================================================
# 2. Count-Min Sketch
# =============================================================================
class CountMinSketch:
    """
    Count-Min Sketch: 항목별 빈도 근사 추정 (과추정 가능, 과소추정 없음)

    핵심 로직:
      - depth × width 크기의 2D 카운터 배열 초기화 (모두 0)
      - 삽입: 각 행(depth)에 독립 해시 → 해당 열(width)의 카운터 +1
      - 조회: depth개 위치의 카운터 중 최솟값 반환 (Count-Min)
    """
    def __init__(self, width: int, depth: int, seed: int = 42):
        self.w = width
        self.d = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.seeds = [(seed + i * 0x9e3779b9) & 0xFFFFFFFF for i in range(depth)]

    def _hash(self, item: str, row: int) -> int:
        s = self.seeds[row]
        h = int(hashlib.md5(item.encode() + s.to_bytes(4, 'little')).hexdigest(), 16)
        return h % self.w

    def add(self, item: str, count: int = 1):
        for r in range(self.d):
            self.table[r, self._hash(item, r)] += count

    def query(self, item: str) -> int:
        return int(min(self.table[r, self._hash(item, r)] for r in range(self.d)))

    @property
    def memory_bytes(self):
        return self.table.nbytes


# =============================================================================
# STEP 01. 데이터 준비 및 Ground Truth 계산
# =============================================================================
print("=" * 70)
print("STEP 01. 데이터 준비 및 Ground Truth 계산")
print("=" * 70)

tracemalloc.start()
t0 = time.perf_counter()

exact_counts   = collections.Counter()   # movie_id 정확 빈도
exact_users    = set()                   # 정확 고유 사용자 집합
exact_pairs    = set()                   # (user_id, movie_id) 정확 집합
total_records  = 0

for uid, mid, rating in stream_ratings():
    exact_counts[mid] += 1
    exact_users.add(uid)
    exact_pairs.add(f"{uid}:{mid}")
    total_records += 1

gt_time = time.perf_counter() - t0
_, gt_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

N_ITEMS        = len(exact_pairs)       # Bloom Filter 기준 항목 수
N_UNIQUE_USERS = len(exact_users)
N_UNIQUE_MOVIES= len(exact_counts)

print(f"  총 레코드  : {total_records:,}")
print(f"  고유 사용자: {N_UNIQUE_USERS:,}")
print(f"  고유 영화  : {N_UNIQUE_MOVIES:,}")
print(f"  고유 (user,movie) 쌍: {N_ITEMS:,}")
print(f"  Ground Truth 처리 시간: {gt_time:.2f}s")
print(f"  Ground Truth 메모리 피크: {gt_peak/1024/1024:.1f} MB")


# =============================================================================
# STEP 02-A. Bloom Filter 실험
# =============================================================================
print("\n" + "=" * 70)
print("STEP 02-A. Bloom Filter 실험")
print("=" * 70)

# 테스트용 비존재 쌍 생성 (FPR 측정)
rng = random.Random(42)
fake_pairs = set()
while len(fake_pairs) < 10000:
    u = str(rng.randint(10000, 99999))
    m = str(rng.randint(10000, 99999))
    k = f"{u}:{m}"
    if k not in exact_pairs:
        fake_pairs.add(k)
fake_pairs = list(fake_pairs)

bf_results = []

# 파라미터 실험: FPR 목표 0.001 / 0.01 / 0.05 / 0.1
for target_fpr in [0.001, 0.01, 0.05, 0.1]:
    m_bits, k_hash = BloomFilter.optimal_params(N_ITEMS, target_fpr)

    tracemalloc.start()
    t_start = time.perf_counter()
    bf = BloomFilter(m_bits, k_hash)
    for uid, mid, _ in stream_ratings():
        bf.add(f"{uid}:{mid}")
    t_end = time.perf_counter()
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # FPR 측정
    fp = sum(1 for p in fake_pairs if p in bf)
    fpr_actual = fp / len(fake_pairs)

    # FNR 측정 (없어야 함)
    sample_true = rng.sample(list(exact_pairs), 1000)
    fn = sum(1 for p in sample_true if p not in bf)

    elapsed = t_end - t_start
    mem_mb  = bf.memory_bytes / 1024 / 1024

    bf_results.append({
        'target_fpr': target_fpr, 'm_bits': m_bits, 'k_hash': k_hash,
        'actual_fpr': fpr_actual, 'fnr': fn/1000,
        'time_s': elapsed, 'mem_mb': mem_mb,
    })

    print(f"  FPR목표={target_fpr:.3f} | m={m_bits:,} k={k_hash} | "
          f"실제FPR={fpr_actual:.4f} | FNR={fn/1000:.4f} | "
          f"시간={elapsed:.2f}s | 메모리={mem_mb:.2f}MB")


# =============================================================================
# STEP 02-B. Count-Min Sketch 실험
# =============================================================================
print("\n" + "=" * 70)
print("STEP 02-B. Count-Min Sketch 실험")
print("=" * 70)

cms_results = []

# 파라미터 실험: (width, depth) 조합
param_sets = [(500, 3), (1000, 4), (5000, 5), (10000, 7)]

for width, depth in param_sets:
    tracemalloc.start()
    t_start = time.perf_counter()
    cms = CountMinSketch(width, depth)
    for uid, mid, _ in stream_ratings():
        cms.add(mid)
    t_end = time.perf_counter()
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = t_end - t_start
    mem_mb  = cms.memory_bytes / 1024 / 1024

    # 상위 100 영화에 대해 오차 측정
    top_movies = [m for m, _ in exact_counts.most_common(100)]
    errors = [abs(cms.query(m) - exact_counts[m]) / exact_counts[m]
              for m in top_movies]
    mre = np.mean(errors)          # Mean Relative Error
    max_re = np.max(errors)

    # 전체 오차 샘플 (500개)
    sample_movies = rng.sample(list(exact_counts.keys()), min(500, N_UNIQUE_MOVIES))
    all_errors = [abs(cms.query(m) - exact_counts[m]) / max(exact_counts[m], 1)
                  for m in sample_movies]
    overall_mre = np.mean(all_errors)

    cms_results.append({
        'width': width, 'depth': depth,
        'mre_top100': mre, 'max_re_top100': max_re,
        'mre_overall': overall_mre,
        'time_s': elapsed, 'mem_mb': mem_mb,
    })

    print(f"  w={width:>6} d={depth} | MRE(top100)={mre:.4f} | MaxRE={max_re:.4f} | "
          f"MRE(전체)={overall_mre:.4f} | 시간={elapsed:.2f}s | 메모리={mem_mb:.4f}MB")


# =============================================================================
# STEP 03. Ground Truth vs Sketch 상세 비교 (최고 파라미터)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 03. 상세 비교 (최고 파라미터 CMS w=10000, d=7)")
print("=" * 70)

best_cms = CountMinSketch(10000, 7)
for uid, mid, _ in stream_ratings():
    best_cms.add(mid)

top20 = exact_counts.most_common(20)
print(f"\n{'영화ID':>10} {'실제빈도':>10} {'CMS추정':>10} {'오차율':>8}")
print("-" * 44)
for mid, true_cnt in top20:
    est = best_cms.query(mid)
    err = (est - true_cnt) / true_cnt * 100
    print(f"{mid:>10} {true_cnt:>10,} {est:>10,} {err:>7.2f}%")


# =============================================================================
# STEP 04. 시각화
# =============================================================================
print("\n" + "=" * 70)
print("STEP 04. 시각화")
print("=" * 70)

# ── 그림 1: Bloom Filter FPR vs 메모리 ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fpr_targets = [r['target_fpr'] for r in bf_results]
fpr_actuals = [r['actual_fpr'] for r in bf_results]
mems_bf     = [r['mem_mb']     for r in bf_results]
times_bf    = [r['time_s']     for r in bf_results]

ax = axes[0]
ax.plot(fpr_targets, fpr_actuals, 'o-', color='steelblue', label='실제 FPR')
ax.plot(fpr_targets, fpr_targets, '--', color='gray',      label='목표 FPR')
ax.set_xlabel('목표 FPR')
ax.set_ylabel('FPR')
ax.set_title('Bloom Filter: 목표 vs 실제 FPR')
ax.legend()

ax = axes[1]
ax.bar([str(t) for t in fpr_targets], mems_bf, color='coral')
ax.set_xlabel('목표 FPR')
ax.set_ylabel('메모리 (MB)')
ax.set_title('Bloom Filter: 목표 FPR별 메모리 사용량')

ax = axes[2]
ax.bar([str(t) for t in fpr_targets], times_bf, color='mediumseagreen')
ax.set_xlabel('목표 FPR')
ax.set_ylabel('처리 시간 (s)')
ax.set_title('Bloom Filter: 목표 FPR별 처리 시간')
save_fig('01_bloom_filter_analysis')

# ── 그림 2: Count-Min Sketch 파라미터 분석 ──────────────────────────────────
labels = [f"w={r['width']}\nd={r['depth']}" for r in cms_results]
mres   = [r['mre_top100']   for r in cms_results]
mems   = [r['mem_mb']       for r in cms_results]
times  = [r['time_s']       for r in cms_results]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].bar(labels, mres, color='steelblue')
axes[0].set_ylabel('평균 상대오차 (MRE, Top-100)')
axes[0].set_title('Count-Min Sketch: 파라미터별 정확도')

axes[1].bar(labels, mems, color='coral')
axes[1].set_ylabel('메모리 (MB)')
axes[1].set_title('Count-Min Sketch: 파라미터별 메모리')

axes[2].bar(labels, times, color='mediumseagreen')
axes[2].set_ylabel('처리 시간 (s)')
axes[2].set_title('Count-Min Sketch: 파라미터별 처리 시간')
save_fig('02_cms_analysis')

# ── 그림 3: Top-20 영화 빈도 실제 vs CMS 추정 ───────────────────────────────
movie_ids  = [m for m, _ in top20]
true_cnts  = [c for _, c in top20]
est_cnts   = [best_cms.query(m) for m in movie_ids]

x = np.arange(len(movie_ids))
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - 0.2, true_cnts, 0.4, label='실제 빈도', color='steelblue')
ax.bar(x + 0.2, est_cnts,  0.4, label='CMS 추정',  color='coral', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"M{m}" for m in movie_ids], rotation=45, fontsize=8)
ax.set_ylabel('빈도수')
ax.set_title('Top-20 영화: 실제 빈도 vs CMS 추정 (w=10000, d=7)')
ax.legend()
save_fig('03_cms_top20_comparison')

# ── 그림 4: 정확도-메모리 Trade-off 종합 ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bloom Filter
ax = axes[0]
ax.scatter(mems_bf, fpr_actuals, s=100, c='steelblue', zorder=5)
for r in bf_results:
    ax.annotate(f"FPR={r['target_fpr']}", (r['mem_mb'], r['actual_fpr']),
                textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.set_xlabel('메모리 (MB)')
ax.set_ylabel('실제 FPR')
ax.set_title('Bloom Filter: 정확도-메모리 Trade-off')

# Count-Min Sketch
ax = axes[1]
ax.scatter(mems, mres, s=100, c='coral', zorder=5)
for r in cms_results:
    ax.annotate(f"w={r['width']},d={r['depth']}", (r['mem_mb'], r['mre_top100']),
                textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.set_xlabel('메모리 (MB)')
ax.set_ylabel('MRE (Top-100)')
ax.set_title('Count-Min Sketch: 정확도-메모리 Trade-off')
save_fig('04_tradeoff')

# ── 그림 5: 알고리즘 비교 (처리시간 vs 정확도 산점도) ───────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
# BF: x=time, y=1-fpr (정밀도로 표현)
for r in bf_results:
    ax.scatter(r['time_s'], 1 - r['actual_fpr'],
               s=r['mem_mb'] * 30, color='steelblue', alpha=0.7,
               label=f"BF FPR={r['target_fpr']}")
# CMS: x=time, y=1-mre
for r in cms_results:
    ax.scatter(r['time_s'], 1 - r['mre_top100'],
               s=max(r['mem_mb'] * 5000, 50), color='coral', alpha=0.7,
               marker='^', label=f"CMS w={r['width']},d={r['depth']}")
ax.set_xlabel('처리 시간 (s)')
ax.set_ylabel('정확도 (1 - 오차율)')
ax.set_title('알고리즘 종합 비교 (원 크기 = 메모리)')
ax.legend(fontsize=7, loc='lower right')
save_fig('05_algorithm_comparison')

# ── 그림 6: CMS 오차 분포 히스토그램 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
all_true = list(exact_counts.values())
all_est  = [best_cms.query(m) for m in exact_counts]
rel_errs = [(e - t) / t * 100 for t, e in zip(all_true, all_est)]

axes[0].hist(rel_errs, bins=50, color='coral', edgecolor='white')
axes[0].set_xlabel('상대 오차 (%)')
axes[0].set_ylabel('빈도')
axes[0].set_title('CMS 전체 상대 오차 분포 (w=10000, d=7)')
axes[0].axvline(0, color='black', linestyle='--')

axes[1].hist([abs(e) for e in rel_errs], bins=50, color='steelblue', edgecolor='white')
axes[1].set_xlabel('절대 상대 오차 (%)')
axes[1].set_ylabel('빈도')
axes[1].set_title('CMS 절대 오차 분포')
save_fig('06_cms_error_distribution')


# =============================================================================
# 최종 요약 출력
# =============================================================================
print("\n" + "=" * 70)
print("최종 성능 요약")
print("=" * 70)

print("\n[Bloom Filter 요약]")
print(f"{'목표FPR':>10} {'m(비트)':>12} {'k(해시)':>8} {'실제FPR':>10} {'메모리MB':>10} {'시간(s)':>8}")
for r in bf_results:
    print(f"{r['target_fpr']:>10.3f} {r['m_bits']:>12,} {r['k_hash']:>8} "
          f"{r['actual_fpr']:>10.4f} {r['mem_mb']:>10.2f} {r['time_s']:>8.2f}")

print("\n[Count-Min Sketch 요약]")
print(f"{'width':>8} {'depth':>6} {'MRE(top100)':>12} {'MaxRE':>8} {'메모리MB':>10} {'시간(s)':>8}")
for r in cms_results:
    print(f"{r['width']:>8} {r['depth']:>6} {r['mre_top100']:>12.4f} "
          f"{r['max_re_top100']:>8.4f} {r['mem_mb']:>10.4f} {r['time_s']:>8.2f}")

print(f"\n  그래프: {FIGS}")
print("=" * 70)

# 결과 CSV 저장
import json
summary = {'bloom_filter': bf_results, 'count_min_sketch': cms_results,
           'ground_truth': {'total': total_records, 'unique_users': N_UNIQUE_USERS,
                            'unique_movies': N_UNIQUE_MOVIES, 'unique_pairs': N_ITEMS}}
with open(os.path.join(BASE, "results.json"), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("결과 저장: results.json")
