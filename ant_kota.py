#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

sys.path.append(os.pardir)
from kota_alifebook_lib.visualizers import MatrixVisualizer

# 環境と個体数
N = 70                            # シミュレーションに存在するアリの総数
SCOUT = 0                         # リーダーとして表示する先発アリの番号
INITIAL_SCOUTS = 4                # 最初から未知のフィールドを探索するアリの数
MAX_RECRUITED_FORAGERS = 16       # 経路発見後に採餌へ参加できる最大個体数
NEST = np.array([-0.72, -0.60])   # 巣の中心座標
FOOD_SOURCES = np.array([[0.66, 0.48], [0.55, -0.38], [-0.18, 0.60]])  # 餌場の座標一覧
FOOD_STOCK_INITIAL = np.array([70, 45, 55])  # 各餌場で取得できる初期餌数
NEST_RADIUS = 0.075               # この距離以内を「巣に到着」と判定する半径
NEST_EXIT_RADIUS = 0.105         # 巣から再出発する時の出口の半径
# 帰巣判定と同じ大きさになる巣の描画半径（格子マス数）。
NEST_DISPLAY_RADIUS = 6
FOOD_RADIUS = 0.075               # 餌そのものの半径（描画・接触の基準）
FOOD_PICKUP_RADIUS = 0.105        # 少し手前でも取得できる判定半径（終点での停滞を防ぐ）
OBSTACLES = np.array([[-0.05, -0.05, 0.16], [0.28, 0.22, 0.12]])  # [中心x, 中心y, 半径]
OBSTACLE_MARGIN = 0.06            # 障害物に接近する前から回避を始める余白

# アリの状態
WAITING, SEARCHING, RETURNING = 0, 1, 2  # 待機・探索・帰巣を表す状態番号
STATE_NAMES = ("waiting", "searching", "returning")  # 状態番号に対応する表示名
MIN_SPEED = 0.0025                 # 停止判定に使う基準速度
MAX_SPEED = 0.0045                 # 標準的な最大移動速度
MAX_SEARCH_STEPS = 600             # 空振り時も自分の経路を辿って帰巣する
STUCK_FRAME_LIMIT = 30             # この時間ほぼ動かなければ探索を再開する
RECRUIT_INTERVAL = 20              # フェロモン経路完成後、待機アリを出す間隔
PATH_POINT_DISTANCE = 0.012        # 経路記憶へ新しい地点を追加する最小移動距離
SEPARATION_DISTANCE = 0.040        # 触角や体がぶつからない程度の距離
SEPARATION_WEIGHT = 0.22           # 仲間を避ける力の強さ

# フェロモン
GRID_SIZE = 150                    # フェロモン地図・描画画像の縦横マス数
EVAPORATION = 0.999                # 1フレーム後に残るフェロモンの割合
DIFFUSION_CENTER = 0.48            # 拡散後も同じマスに残るフェロモンの割合
DIFFUSION_SIDE = 0.065             # 中心 + 4近傍 = 1.0
PHEROMONE_DROP = 0.90              # 帰還アリが1フレームに置くフェロモン量
SENSOR_DISTANCE = 0.050            # 触角で前方を調べる距離
SENSOR_ANGLE = np.deg2rad(35)      # 左右の触角を向ける角度
FOLLOW_PHEROMONE_PROBABILITY = 0.80  # 匂いを検知した時に従う確率
SCOUT_NOISE = 0.10               # 先発アリの小さな探索揺らぎ
GUIDE_NOISE = 0.13               # 共有経路を辿るアリの揺らぎ
RETURN_NOISE = 0.05              # 帰還時の揺らぎ（経路を見失わない程度）

WAYPOINT_RADIUS = 0.085            # 経路上の目標地点へ十分近付いたとみなす距離
EXPLORATION_TARGET_RADIUS = 0.10  # ランダム探索地点を切り替える距離
FOOD_SMELL_RADIUS = 0.22          # 近付いた餌だけは触角で検知できる距離

# 色（RGB）
COLOR_BACKGROUND = np.array([0.025, 0.040, 0.070])  # 背景色
COLOR_PHEROMONE = np.array([1.00, 0.23, 0.02])       # フェロモンの色
COLOR_NEST = np.array([1.00, 0.12, 0.16])            # 巣の色
COLOR_FOOD = np.array([1.00, 0.92, 0.00])            # 残量がある餌の色
COLOR_WAITING = np.array([0.55, 0.55, 0.60])         # 待機アリ用の色（現在は非表示）
COLOR_SEARCHING = np.array([0.00, 0.82, 1.00])       # 探索・採餌へ向かうアリの色
COLOR_RETURNING = np.array([0.15, 1.00, 0.30])  # 餌を運ぶ帰還アリ（明るい緑）
COLOR_SCOUT = np.array([0.72, 0.20, 1.00])      # 最初に探索する先発アリ（紫）


def unit(vector):
    """ゼロベクトルでも安全な単位ベクトル。"""
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-12 else np.zeros_like(vector)


def grid_index(position):
    """[-1, 1] の座標をフェロモン格子の添字へ変換する。"""
    gx = int(np.clip((position[0] + 1.0) * 0.5 * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
    gy = int(np.clip((position[1] + 1.0) * 0.5 * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
    return gx, gy


def rotate(direction, angle):
    """2次元の進行方向を左右へ回転する。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * direction[0] - s * direction[1],
                     s * direction[0] + c * direction[1]])


def diffuse_and_evaporate(field):
    """フェロモンを4近傍へ広げ、少しずつ蒸発させる。"""
    sides = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
             + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1))
    result = EVAPORATION * (DIFFUSION_CENTER * field + DIFFUSION_SIDE * sides)
    result[[0, -1], :] = 0.0
    result[:, [0, -1]] = 0.0
    return result


def pheromone_direction(position, velocity, field):
    """触角に相当する前・左前・右前の3方向から最も濃い匂いを選ぶ。"""
    heading = unit(velocity)
    if not np.any(heading):
        heading = np.array([1.0, 0.0])
    directions = (heading, rotate(heading, SENSOR_ANGLE), rotate(heading, -SENSOR_ANGLE))
    values = []
    for direction in directions:
        gx, gy = grid_index(position + SENSOR_DISTANCE * direction)
        values.append(field[gx, gy])
    best = int(np.argmax(values))
    return directions[best] if values[best] > 0.015 else None


def draw_square(image, position, radius, color):
    """色付きの環境物／アリをフェロモン地図に重ねる。"""
    gx, gy = grid_index(position)
    image[max(0, gx - radius):gx + radius + 1,
          max(0, gy - radius):gy + radius + 1] = color


def avoidance_direction(index, position, state):
    """近すぎる仲間から少し離れる。行列が完全に重ならないための動き。"""
    offsets = position[index] - position
    distances = np.linalg.norm(offsets, axis=1)
    neighbours = ((state != WAITING) & (distances > 1e-10)
                  & (distances < SEPARATION_DISTANCE))
    if not np.any(neighbours):
        return np.zeros(2)
    return unit(np.sum(offsets[neighbours] / distances[neighbours, None], axis=0))


def obstacle_avoidance(point):
    """円形障害物の外側へ向ける回避方向。"""
    force = np.zeros(2)
    for ox, oy, radius in OBSTACLES:
        away = point - np.array([ox, oy])
        distance = np.linalg.norm(away)
        if distance < radius + OBSTACLE_MARGIN:
            force += unit(away if distance > 1e-10 else np.random.normal(size=2))
    return unit(force)


def segment_hits_obstacle(start, end, center, radius):
    """移動線分が円形障害物に入るかを調べる。"""
    segment = end - start
    length_sq = np.dot(segment, segment)
    if length_sq < 1e-12:
        nearest = start
    else:
        t = np.clip(np.dot(center - start, segment) / length_sq, 0.0, 1.0)
        nearest = start + t * segment
    return np.linalg.norm(nearest - center) < radius + OBSTACLE_MARGIN


def avoid_obstacle_collision(start, velocity):
    """障害物を横切る移動を、外周に沿う移動へ変更する。"""
    speed = np.linalg.norm(velocity)
    if speed < 1e-12:
        return velocity
    corrected = velocity.copy()
    for ox, oy, radius in OBSTACLES:
        center = np.array([ox, oy])
        if segment_hits_obstacle(start, start + corrected, center, radius):
            outward = unit(start - center)
            if not np.any(outward):
                outward = unit(np.random.normal(0.0, 1.0, 2))
            tangent = np.array([-outward[1], outward[0]])
            if np.dot(tangent, corrected) < 0:
                tangent *= -1
            # 少し外へ押し出しながら、障害物の縁を通る。
            corrected = unit(0.80 * tangent + 0.55 * outward) * speed
    return corrected


def food_near(point, radius, stock):
    """まだ取れる餌場が近くにあれば、その添字を返す。"""
    available = np.flatnonzero(stock > 0)
    if len(available) == 0:
        return None
    distances = np.linalg.norm(FOOD_SOURCES[available] - point, axis=1)
    index = int(available[np.argmin(distances)])
    return index if np.min(distances) < radius else None


def reinforce_route(field, route, amount=2.5):
    """最初の帰還時に、巣から餌までの道しるべを明確に残す。"""
    for point in route:
        if np.linalg.norm(point - NEST) > NEST_RADIUS:
            gx, gy = grid_index(point)
            field[gx, gy] += amount


def route_exit_index(route):
    """巣の内側の記録点を飛ばし、巣の外から共有経路を辿り始める。"""
    for index, point in enumerate(route):
        if np.linalg.norm(point - NEST) > NEST_EXIT_RADIUS:
            return index
    return 0


def nearest_route_index(route, point):
    """停止したアリを最も近い共有経路の少し先へ戻す。"""
    if not route:
        return -1
    distances = [np.linalg.norm(route_point - point) for route_point in route]
    return min(int(np.argmin(distances)) + 1, len(route) - 1)


def make_image(pheromone, position, state, carrying, stock):
    """フェロモン・巣・餌・アリを一枚のカラー地図として描く。"""
    image = np.full((GRID_SIZE, GRID_SIZE, 3), COLOR_BACKGROUND)
    strength = np.sqrt(np.clip(pheromone / 7.0, 0.0, 1.0))[:, :, None]
    image += strength * COLOR_PHEROMONE

    draw_square(image, NEST, NEST_DISPLAY_RADIUS, COLOR_NEST)
    for food, amount in zip(FOOD_SOURCES, stock):
        draw_square(image, food, 2, COLOR_FOOD if amount > 0 else np.array([0.25, 0.25, 0.25]))
    for ox, oy, radius in OBSTACLES:
        gx, gy = grid_index(np.array([ox, oy]))
        r = int(radius * (GRID_SIZE - 1) / 2)
        image[max(0, gx-r):gx+r+1, max(0, gy-r):gy+r+1] = np.array([0.08, 0.08, 0.10])
    for i in range(N):
        # 巣の中で待機する個体は外から見えないものとして描画しない。
        if state[i] == WAITING:
            continue
        if carrying[i]:
            color = COLOR_RETURNING
        elif i < INITIAL_SCOUTS:
            # 最初に未知のフィールドを探索した4匹を先発アリとして区別する。
            color = COLOR_SCOUT
        else:
            color = COLOR_SEARCHING
        draw_square(image, position[i], 0, color)
    return np.clip(image, 0.0, 1.0)


def random_exploration_target():
    """餌の位置とは無関係な、フィールド内のランダムな探索地点。"""
    return np.random.uniform(-0.85, 0.85, 2)


# 初期状態: 先発アリだけが出発し、他は巣の周辺で待機する。
position = NEST + np.random.normal(0.0, 0.018, (N, 2))  # 全アリの現在座標 [N, 2]
velocity = np.zeros((N, 2))          # 全アリの現在速度ベクトル [N, 2]
# 同じ種類のアリでも移動速度に少し個体差を持たせる。
speed_factor = np.random.uniform(0.80, 1.18, N)
state = np.full(N, WAITING, dtype=int)  # 全アリの行動状態 [N]
state[:INITIAL_SCOUTS] = SEARCHING
velocity[:INITIAL_SCOUTS] = np.array([
    unit(np.random.normal(0.0, 1.0, 2)) * MAX_SPEED * speed_factor[i]
    for i in range(INITIAL_SCOUTS)
])
has_joined_foraging = np.zeros(N, dtype=bool)  # 一度でも採餌役に選ばれたか [N]
has_joined_foraging[:INITIAL_SCOUTS] = True
exploration_target = np.array([random_exploration_target() for _ in range(N)])  # 各探索アリの一時目標
carrying = np.zeros(N, dtype=bool)       # 各アリが現在餌を運んでいるか [N]
search_steps = np.zeros(N, dtype=int)    # 各アリが探索を続けたフレーム数 [N]
path_history = [[] for _ in range(N)]    # 各アリが記憶した往路の座標列
return_index = np.full(N, -1, dtype=int) # 帰巣時に次に辿る経路記憶の添字 [N]
pheromone = np.zeros((GRID_SIZE, GRID_SIZE))  # 地面に残るフェロモン濃度マップ

trail_ready = False                 # 後続アリが使える共有経路が存在するか
trail_source = -1                # 現在共有している経路が通じる餌場
recruit_clock = 0                   # 次の待機アリを採餌へ出すまでの経過フレーム
food_stock = FOOD_STOCK_INITIAL.copy()  # 現在の各餌場の残量
delivered_food = 0                  # 巣まで運び込まれた餌の累計（記録用）
established_route = []              # 帰還アリが共有した巣から餌までの代表経路
guide_index = np.full(N, -1, dtype=int)  # 共有経路を教わったアリの現在の案内地点
carrying_source = np.full(N, -1, dtype=int)  # 各アリが運んでいる餌場の番号 [N]
stuck_steps = np.zeros(N, dtype=int)         # 各アリがほぼ移動できない連続フレーム数 [N]

visualizer = MatrixVisualizer(value_range_min=0, value_range_max=1,
                              title="Ant foraging: pheromone trail")


while visualizer:
    pheromone = diffuse_and_evaporate(pheromone)

    # 最初の帰還後、待機アリを一匹ずつ採餌に参加させる。
    if trail_ready:
        recruit_clock += 1
        if recruit_clock >= RECRUIT_INTERVAL:
            # 上限に達するまで、巣の待機アリを新たに採餌役へ加える。
            new_waiters = np.flatnonzero((state == WAITING) & ~has_joined_foraging)
            if len(new_waiters) > 0 and np.sum(has_joined_foraging) < MAX_RECRUITED_FORAGERS:
                recruit = np.random.choice(new_waiters)
                has_joined_foraging[recruit] = True
            else:
                recruit = None

            if recruit is not None:
                state[recruit] = SEARCHING
                # 帰還アリが残した経路の最初の向きで巣を出る。
                if len(established_route) >= 2:
                    guide_index[recruit] = route_exit_index(established_route)
                    departure = unit(established_route[guide_index[recruit]] - position[recruit])
                else:
                    departure = unit(np.random.normal(0.0, 1.0, 2))
                velocity[recruit] = departure * MAX_SPEED * speed_factor[recruit]
            recruit_clock = 0

    for i in range(N):
        if state[i] == WAITING:
            # 一度でも採餌役になったアリは巣で停止させない。経路があれば
            # それを辿り、餌場が枯渇して経路がなければ広域探索へ戻る。
            if has_joined_foraging[i]:
                if trail_ready and len(established_route) >= 2:
                    state[i] = SEARCHING
                    search_steps[i] = 0
                    guide_index[i] = route_exit_index(established_route)
                    direction = unit(established_route[guide_index[i]] - position[i])
                    exit_direction = unit(direction + np.random.normal(0.0, 0.28, 2))
                    position[i] = NEST + NEST_EXIT_RADIUS * exit_direction
                    velocity[i] = direction * MAX_SPEED * speed_factor[i]
                    continue
                if i < INITIAL_SCOUTS:
                    state[i] = SEARCHING
                    search_steps[i] = 0
                    guide_index[i] = -1
                    exploration_target[i] = random_exploration_target()
                    direction = unit(exploration_target[i] - position[i])
                    exit_direction = unit(direction + np.random.normal(0.0, 0.28, 2))
                    position[i] = NEST + NEST_EXIT_RADIUS * exit_direction
                    velocity[i] = direction * MAX_SPEED * speed_factor[i]
                    continue
            velocity[i] = 0.0
            continue

        if state[i] == SEARCHING:
            search_steps[i] += 1

            # 通った道を、後で自力で戻るために記憶する。
            if (not path_history[i]
                    or np.linalg.norm(position[i] - path_history[i][-1]) > PATH_POINT_DISTANCE):
                path_history[i].append(position[i].copy())

            if i < INITIAL_SCOUTS and not trail_ready:
                # 先発アリは餌の位置を知らない。前の向きを少し保った
                # 広いフィールドのランダムな探索地点を渡り歩く。
                if np.linalg.norm(position[i] - exploration_target[i]) < EXPLORATION_TARGET_RADIUS:
                    exploration_target[i] = random_exploration_target()
                nearby_food = food_near(position[i], FOOD_SMELL_RADIUS, food_stock)
                if nearby_food is not None:
                    # 餌の近くで初めて触角により方向が分かる。
                    target_direction = unit(FOOD_SOURCES[nearby_food] - position[i])
                else:
                    target_direction = unit(exploration_target[i] - position[i])
                direction = unit(
                    0.82 * target_direction
                    + 0.18 * unit(velocity[i])
                    + np.random.normal(0.0, SCOUT_NOISE, 2)
                )
            elif guide_index[i] >= 0 and established_route:
                # 帰還アリから教わった共有経路を、巣→餌の順に辿る。
                guide_index[i] = min(guide_index[i], len(established_route) - 1)
                while (guide_index[i] < len(established_route) - 1
                       and np.linalg.norm(position[i] - established_route[guide_index[i]]) < 0.030):
                    guide_index[i] += 1
                route_direction = unit(established_route[guide_index[i]] - position[i])
                # 経路の最後では、揺らぎよりも餌への接近を優先する。
                # これにより餌場の手前を往復して空振り帰巣するのを防ぐ。
                at_route_end = guide_index[i] == len(established_route) - 1
                if at_route_end and trail_source >= 0 and food_stock[trail_source] > 0:
                    route_direction = unit(FOOD_SOURCES[trail_source] - position[i])
                direction = unit(
                    (0.93 if at_route_end else 0.78) * route_direction
                    + (0.05 if at_route_end else 0.18) * unit(velocity[i])
                    + np.random.normal(0.0, GUIDE_NOISE * (0.25 if at_route_end else 1.0), 2)
                )
            else:
                trail = pheromone_direction(position[i], velocity[i], pheromone)
                if trail is not None and np.random.random() < FOLLOW_PHEROMONE_PROBABILITY:
                    direction = trail
                else:
                    # 完全なランダムではなく、前の向きをある程度維持する探索。
                    direction = unit(0.65 * unit(velocity[i])
                                     + np.random.normal(0.0, 0.55, 2))

            velocity[i] = direction * MAX_SPEED * speed_factor[i]

            # 餌の取得、または探索失敗時の帰還。
            # 餌の縁で小さく揺れて取得半径に入れない状況を避けるため、
            # 実際の餌半径より少し広い範囲で取得する。
            food_index = food_near(position[i], FOOD_PICKUP_RADIUS, food_stock)
            if food_index is not None:
                # 共有する経路の終点を餌の中心へ固定する。これがないと、
                # 記録間隔の都合で餌の少し手前が終点になり、後続アリが
                # 餌場の縁で止まってしまうことがある。
                path_history[i].append(FOOD_SOURCES[food_index].copy())
                carrying[i] = True
                carrying_source[i] = food_index
                food_stock[food_index] -= 1
                # 別の餌場を発見したら、帰還後にその餌場の経路へ切り替える。
                if food_index != trail_source:
                    trail_ready = False
                    established_route = []
                    guide_index[:] = -1
                if food_stock[food_index] == 0:
                    # 最後の一個を取った時点で、他のアリを古い餌場への
                    # 案内から解放し、次の餌場の探索へ切り替える。
                    trail_ready = False
                    established_route = []
                    guide_index[:] = -1
                    for scout in range(INITIAL_SCOUTS):
                        if state[scout] != RETURNING:
                            state[scout] = SEARCHING
                            exploration_target[scout] = random_exploration_target()
                state[i] = RETURNING
                return_index[i] = len(path_history[i]) - 1
            elif search_steps[i] >= MAX_SEARCH_STEPS:
                carrying[i] = False
                state[i] = RETURNING
                return_index[i] = len(path_history[i]) - 1

        elif state[i] == RETURNING:
            # 保存した往路を逆順に辿る。巣座標を移動方向には使わない。
            while (return_index[i] >= 0
                   and np.linalg.norm(position[i] - path_history[i][return_index[i]]) < 0.018):
                return_index[i] -= 1

            if return_index[i] >= 0:
                route_direction = unit(path_history[i][return_index[i]] - position[i])
                direction = unit(
                    0.88 * route_direction
                    + 0.08 * unit(velocity[i])
                    + np.random.normal(0.0, RETURN_NOISE, 2)
                )
                velocity[i] = direction * MAX_SPEED * speed_factor[i] * 1.05
            else:
                velocity[i] *= 0.8

            # 餌を運ぶ帰還アリだけが、後続への道しるべを残す。
            if carrying[i] and np.linalg.norm(position[i] - NEST) > NEST_RADIUS:
                gx, gy = grid_index(position[i])
                pheromone[gx, gy] += PHEROMONE_DROP

            # 巣へ届いたら荷物を置く。空振り帰巣もここで待機へ戻る。
            if np.linalg.norm(position[i] - NEST) < NEST_RADIUS:
                if carrying[i]:
                    delivered_food += 1
                    # 最初の発見者の経路を巣に持ち帰り、後続が辿れるよう
                    # フェロモンを補強する。
                    source = carrying_source[i]
                    if food_stock[source] > 0 and (not trail_ready or trail_source != source):
                        established_route = [point.copy() for point in path_history[i]]
                        reinforce_route(pheromone, established_route)
                        trail_ready = True
                        trail_source = source
                    # 帰還アリが増えるほど、同じ経路の匂いが長く強く残る。
                    reinforce_route(pheromone, path_history[i], amount=1.0)
                    carrying_source[i] = -1
                    # 餌が枯渇したら、古い共有経路を解除して残る餌場を探索する。
                    if food_stock[source] == 0:
                        trail_ready = False
                        if trail_source == source:
                            trail_source = -1
                        established_route = []
                        guide_index[:] = -1
                        for scout in range(INITIAL_SCOUTS):
                            state[scout] = SEARCHING
                            exploration_target[scout] = random_exploration_target()
                carrying[i] = False
                state[i] = WAITING
                search_steps[i] = 0
                path_history[i] = []
                return_index[i] = -1
                guide_index[i] = -1
                velocity[i] = 0.0

                # 餌がまだ見つからない間は、先発アリだけ再びランダム探索へ出る。
                if not trail_ready and i < INITIAL_SCOUTS:
                    state[i] = SEARCHING
                    exploration_target[i] = random_exploration_target()
                    velocity[i] = unit(np.random.normal(0.0, 1.0, 2)) * MAX_SPEED * speed_factor[i]

        # 正方形のフィールドから出ないように反射させる。
        # 仲間との接近を避ける小さな横ぶれを加える。
        avoid = avoidance_direction(i, position, state)
        obstacle_avoid = obstacle_avoidance(position[i])
        if np.any(avoid) or np.any(obstacle_avoid):
            speed = np.linalg.norm(velocity[i])
            velocity[i] = unit(velocity[i] + SEPARATION_WEIGHT * speed * avoid
                               + 0.55 * speed * obstacle_avoid) * speed
        velocity[i] = avoid_obstacle_collision(position[i], velocity[i])
        next_position = position[i] + velocity[i]
        for axis in (0, 1):
            if abs(next_position[axis]) > 0.97:
                velocity[i, axis] *= -1
        previous_position = position[i].copy()
        position[i] += velocity[i]

        # 経路終点や障害物の縁で止まった個体は、経路を一度忘れて
        # ランダム探索へ復帰させる。群れ全体が停止するのを防ぐ。
        if np.linalg.norm(position[i] - previous_position) < MIN_SPEED * 0.10:
            stuck_steps[i] += 1
        else:
            stuck_steps[i] = 0
        if state[i] != WAITING and stuck_steps[i] >= STUCK_FRAME_LIMIT:
            if state[i] == RETURNING and carrying[i]:
                # 餌を持つ帰還アリは情報源なので、ランダム探索へ戻して
                # 荷物を失わせない。次の経路記憶へ強制的に向きを戻す。
                if return_index[i] >= 0:
                    direction = unit(path_history[i][return_index[i]] - position[i])
                else:
                    direction = unit(NEST - position[i])
                velocity[i] = direction * MAX_SPEED * speed_factor[i]
            else:
                carrying[i] = False
                path_history[i] = []
                return_index[i] = -1
                search_steps[i] = 0
                if trail_ready and established_route:
                    state[i] = SEARCHING
                    guide_index[i] = nearest_route_index(established_route, position[i])
                    direction = unit(established_route[guide_index[i]] - position[i])
                    velocity[i] = direction * MAX_SPEED * speed_factor[i]
                elif i < INITIAL_SCOUTS:
                    state[i] = SEARCHING
                    guide_index[i] = -1
                    exploration_target[i] = random_exploration_target()
                    velocity[i] = unit(exploration_target[i] - position[i]) * MAX_SPEED * speed_factor[i]
                else:
                    state[i] = WAITING
                    velocity[i] = 0.0
            stuck_steps[i] = 0

    visualizer.update(make_image(pheromone, position, state, carrying, food_stock))
