#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from alifebook_lib.visualizers_ryota import SwarmVisualizer

# visualizerの初期化 (Appendix参照)
visualizer = SwarmVisualizer()

# シミュレーションパラメタ
N = 250
# 力の強さ
COHESION_FORCE = 0.00000001
SEPARATION_FORCE = 0.5
ALIGNMENT_FORCE = 0.01
# 力の働く距離
COHESION_DISTANCE = 0.8
SEPARATION_DISTANCE = 0.03
ALIGNMENT_DISTANCE = 0.5
# 力の働く角度
COHESION_ANGLE = np.pi / 2
SEPARATION_ANGLE = np.pi / 2
ALIGNMENT_ANGLE = np.pi / 3
# 速度の上限/下限
MIN_VEL = 0.005
MAX_VEL = 0.03
LEADER_SPEED = MAX_VEL
# 境界で働く力（0にすると自由境界）
BOUNDARY_FORCE = 0.005
# リーダー個体が他個体を引き寄せる力
LEADER_ATTRACTION_FORCE = 0.004
LEADER_ATTRACTION_DISTANCE = 1
LEADER_ANGLE = np.pi
# 嫌われ者個体から逃げる力
OUTCAST_AVOIDANCE_FORCE = 2.0
OUTCAST_AVOIDANCE_DISTANCE = 0.5
OUTCAST_ANGLE = np.pi
# 嫌われ者自身がグループから離れる力
OUTCAST_ESCAPE_FORCE = 0.6
OUTCAST_ESCAPE_DISTANCE = 0.3

# 位置と速度
# -----------------------------
# 初期位置を5つのグループに分ける
# -----------------------------
centers = np.array([
    [-0.8, -0.8, 0],
    [ 0.8, -0.8, 0],
    [-0.8,  0.8, 0],
    [ 0.8,  0.8, 0],
    [ 0.0,  0.0, 0]
])

x = np.empty((N, 3))

group_size = N // 5  # 50匹ずつ

for i in range(5):
    start = i * group_size
    end = (i + 1) * group_size

    # 各中心の周りにランダム配置
    x[start:end] = centers[i] + (np.random.rand(group_size, 3) - 0.5) * 0.15

# 初速度
v = (np.random.rand(N, 3) * 2 - 1) * MIN_VEL

# リーダー(赤)は全体で1体、嫌われ者(青)は各グループに1体ずつ配置
is_leader = np.zeros(N, dtype=bool)
is_outcast = np.zeros(N, dtype=bool)
is_leader[0] = True
for i in range(5):
    start = i * group_size
    is_outcast[start + 1] = True
leader_v_abs = np.linalg.norm(v[is_leader], axis=1)
v[is_leader] = LEADER_SPEED * v[is_leader] / leader_v_abs[:, np.newaxis]

# 表示色（通常: 灰色, リーダー: 赤, 嫌われ者: 青）
colors = np.full((N, 4), 0.7)
colors[is_leader] = (1, 0, 0, 1)
colors[is_outcast] = (0, 0, 1, 1)

# cohesion, separation, alignmentの３つの力を代入する変数
dv_coh = np.empty((N,3))
dv_sep = np.empty((N,3))
dv_ali = np.empty((N,3))
# リーダー/嫌われ者に関する力を代入する変数
dv_leader = np.empty((N,3))
dv_outcast = np.empty((N,3))
# 境界で働く力を代入する変数
dv_boundary = np.empty((N,3))

while visualizer:
    for i in range(N):
        # ここで計算する個体の位置と速度
        x_this = x[i]
        v_this = v[i]
        # それ以外の個体の位置と速度の配列
        x_that = np.delete(x, i, axis=0)
        v_that = np.delete(v, i, axis=0)
        is_leader_that = np.delete(is_leader, i)
        is_outcast_that = np.delete(is_outcast, i)
        # 個体間の距離と角度
        distance = np.linalg.norm(x_that - x_this, axis=1)
        angle = np.arccos(np.dot(v_this, (x_that-x_this).T) / (np.linalg.norm(v_this) * np.linalg.norm((x_that-x_this), axis=1)))
        # 各力が働く範囲内の個体のリスト
        other_agents_mask = ~is_leader_that if is_leader[i] else np.ones(N - 1, dtype=bool)
        coh_agents_x = x_that[ other_agents_mask & (distance < COHESION_DISTANCE) & (angle < COHESION_ANGLE) ]
        sep_agents_x = x_that[ other_agents_mask & (distance < SEPARATION_DISTANCE) & (angle < SEPARATION_ANGLE) ]
        ali_agents_v = v_that[ other_agents_mask & (distance < ALIGNMENT_DISTANCE) & (angle < ALIGNMENT_ANGLE) ]
        leader_agents_x = x_that[ other_agents_mask & is_leader_that & (distance < LEADER_ATTRACTION_DISTANCE) & (angle < LEADER_ANGLE) ]
        outcast_agents_x = x_that[ is_outcast_that & (distance < OUTCAST_AVOIDANCE_DISTANCE) & (angle < OUTCAST_ANGLE) ]
        group_agents_x = x_that[ (~is_outcast_that) & (distance < OUTCAST_ESCAPE_DISTANCE) ]
        # 各力の計算
        if is_outcast[i]:
            dv_coh[i] = 0
            dv_sep[i] = SEPARATION_FORCE * np.sum(x_this - sep_agents_x, axis=0) if (len(sep_agents_x) > 0) else 0
            dv_ali[i] = 0
            dv_leader[i] = 0
            dv_outcast[i] = OUTCAST_ESCAPE_FORCE * np.sum(x_this - group_agents_x, axis=0) if (len(group_agents_x) > 0) else 0
        else:
            dv_coh[i] = COHESION_FORCE * (np.average(coh_agents_x, axis=0) - x_this) if (len(coh_agents_x) > 0) else 0
            dv_sep[i] = SEPARATION_FORCE * np.sum(x_this - sep_agents_x, axis=0) if (len(sep_agents_x) > 0) else 0
            dv_ali[i] = ALIGNMENT_FORCE * (np.average(ali_agents_v, axis=0) - v_this) if (len(ali_agents_v) > 0) else 0
            dv_leader[i] = LEADER_ATTRACTION_FORCE * np.sum(leader_agents_x - x_this, axis=0) if (len(leader_agents_x) > 0) else 0
            dv_outcast[i] = OUTCAST_AVOIDANCE_FORCE * np.sum(x_this - outcast_agents_x, axis=0) if (len(outcast_agents_x) > 0) else 0
        dist_center = np.linalg.norm(x_this) # 原点からの距離
        dv_boundary[i] = - BOUNDARY_FORCE * x_this * (dist_center - 1) / dist_center if (dist_center > 1) else 0
    # 速度のアップデートと上限/下限のチェック
    v += dv_coh + dv_sep + dv_ali + dv_leader + dv_outcast + dv_boundary
    for i in range(N):
        v_abs = np.linalg.norm(v[i])
        if is_leader[i]:
            v[i] = LEADER_SPEED * v[i] / v_abs if (v_abs > 0) else np.array([LEADER_SPEED, 0, 0])
        elif (v_abs < MIN_VEL):
            v[i] = MIN_VEL * v[i] / v_abs
        elif (v_abs > MAX_VEL):
            v[i] = MAX_VEL * v[i] / v_abs
    # 位置のアップデート
    x += v
    visualizer.update(x, v, colors)
