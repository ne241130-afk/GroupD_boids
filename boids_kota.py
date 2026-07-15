#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from alifebook_lib.visualizers import SwarmVisualizer

# visualizerの初期化 (Appendix参照)
visualizer = SwarmVisualizer()

# シミュレーションパラメタ
N = 256

# --- 動的なマルチグループ用セッティング ---
# 力の強さ
COHESION_FORCE = 0.25
SEPARATION_FORCE = 0.05
ALIGNMENT_FORCE = 0.25

# 力の働く距離
COHESION_DISTANCE = 0.12
SEPARATION_DISTANCE = 0.20
ALIGNMENT_DISTANCE = 0.5

# 力の働く角度
COHESION_ANGLE = np.pi
SEPARATION_ANGLE = np.pi     # 全方位（360度）からの他グループの接近を警戒
ALIGNMENT_ANGLE = np.pi / 2  # 前方の仲間にだけ合わせる（シャープな直進性）

# 境界で働く力（中央に押し戻す力を少し優しくして、のびのび泳がせる）
BOUNDARY_FORCE = 0.01
# --- 速度の上限/下限 ---
MIN_VEL = 0.008
MAX_VEL = 0.025 
LEADER_MAX_VEL = 0.045

LEADER_FORCE = 0.03
LEADER_DISTANCE = 0.3

# 位置と速度
x = np.random.rand(N, 3) * 2 - 1
v = (np.random.rand(N, 3) * 2 - 1 ) * MIN_VEL


# リーダーだけ速くする
v[N-1] = np.array([LEADER_MAX_VEL, 0.0, 0.0])

# 各力を代入する変数
dv_coh = np.empty((N,3))
dv_sep = np.empty((N,3))
dv_ali = np.empty((N,3))
dv_boundary = np.empty((N,3))
dv_leader = np.empty((N,3))


while visualizer:

    for i in range(N):
        x_this = x[i]
        v_this = v[i]

        # 自分以外の個体データ
        x_that = np.delete(x, i, axis=0)
        v_that = np.delete(v, i, axis=0)

        distance = np.linalg.norm(x_that - x_this, axis=1)

        # ゼロ除算対策
        denom = np.linalg.norm(v_this) * np.linalg.norm((x_that - x_this), axis=1)
        denom = np.where(denom == 0, 1e-8, denom)
        angle = np.arccos(
            np.clip(np.dot(v_this, (x_that - x_this).T) / denom, -1.0, 1.0)
        )

        if i == N - 1:
            # リーダーは他個体の影響を受けない
            dv_coh[i] = 0
            dv_sep[i] = 0
            dv_ali[i] = 0
            dv_leader[i] = 0

        else:
            # 一般個体（0〜N-2番）の行動ルール
            # 通常の3つの力の計算
            coh_agents_x = x_that[(distance < COHESION_DISTANCE) & (angle < COHESION_ANGLE)]
            sep_agents_x = x_that[(distance < SEPARATION_DISTANCE) & (angle < SEPARATION_ANGLE)]
            ali_agents_v = v_that[(distance < ALIGNMENT_DISTANCE) & (angle < ALIGNMENT_ANGLE)]

            dv_coh[i] = COHESION_FORCE * (np.average(coh_agents_x, axis=0) - x_this) if len(coh_agents_x) > 0 else 0
            dv_sep[i] = SEPARATION_FORCE * np.sum(x_this - sep_agents_x, axis=0) if len(sep_agents_x) > 0 else 0
            dv_ali[i] = ALIGNMENT_FORCE * (np.average(ali_agents_v, axis=0) - v_this) if len(ali_agents_v) > 0 else 0

            # リーダーについていく
            leader_pos = x[N - 1]
            dist = np.linalg.norm(leader_pos - x_this)

            if dist < LEADER_DISTANCE:
                dv_leader[i] = LEADER_FORCE * (leader_pos - x_this)
            else:
                dv_leader[i] = 0

        # 境界の力（全員共通）
        dist_center = np.linalg.norm(x_this)
        dv_boundary[i] = (
            -BOUNDARY_FORCE * x_this * (dist_center - 1) / dist_center
            if dist_center > 1
            else 0
        )

    # 速度のアップデート
    v += dv_coh + dv_sep + dv_ali + dv_boundary + dv_leader
    
    # 速度制限
    for i in range(N):
        v_abs = np.linalg.norm(v[i])
        if i == N-1:
            # 外敵の速度制限（一般個体より速い）
            if v_abs > LEADER_MAX_VEL:
                v[i] = LEADER_MAX_VEL * v[i] / v_abs
        else:
            # 一般個体の速度制限
            if (v_abs < MIN_VEL):
                v[i] = MIN_VEL * v[i] / v_abs
            elif (v_abs > MAX_VEL):
                v[i] = MAX_VEL * v[i] / v_abs
                
    # 位置のアップデート
    x += v
    visualizer.update(x, v)