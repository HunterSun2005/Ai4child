"""
EVGS Feature Engineering v2: 从COCO-WholeBody 133关键点提取临床步态角度特征

基于爱丁堡视觉步态评分 (Edinburgh Visual Gait Score, EVGS) 的17个评分项，
将2D关键点坐标转化为临床相关的角度/距离特征。

v2 修复内容:
- 修复1: forward/toward视频左右镜像 + left/right行走方向符号翻转
- 修复2: foot_clearance和heel_height用身体比例归一化
- 修复3: 踝关节角度公式验证与修正
- 修复4: 加入步态周期分割 (initial contact detection)
- 修复5: 重新设计pelvic rotation计算 (用髋宽/肩宽比)
- 修复6: 冠状面脚部特征加时序平滑
- 修复7: load_frame_json支持多实例选择 (用bbox面积最大的)
- 修复8: 按步态阶段分别聚合特征

作者: Zihan
日期: 2026-04-05
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 1. COCO-WholeBody 关键点定义
# =============================================================================

NOSE = 0
LEFT_EYE = 1; RIGHT_EYE = 2
LEFT_EAR = 3; RIGHT_EAR = 4
LEFT_SHOULDER = 5; RIGHT_SHOULDER = 6
LEFT_ELBOW = 7; RIGHT_ELBOW = 8
LEFT_WRIST = 9; RIGHT_WRIST = 10
LEFT_HIP = 11; RIGHT_HIP = 12
LEFT_KNEE = 13; RIGHT_KNEE = 14
LEFT_ANKLE = 15; RIGHT_ANKLE = 16

LEFT_BIG_TOE = 17; LEFT_SMALL_TOE = 18; LEFT_HEEL = 19
RIGHT_BIG_TOE = 20; RIGHT_SMALL_TOE = 21; RIGHT_HEEL = 22

# 左侧/右侧肢体关键点映射
SIDE_KEYPOINTS = {
    'left': {
        'shoulder': LEFT_SHOULDER, 'hip': LEFT_HIP,
        'knee': LEFT_KNEE, 'ankle': LEFT_ANKLE,
        'big_toe': LEFT_BIG_TOE, 'small_toe': LEFT_SMALL_TOE, 'heel': LEFT_HEEL,
    },
    'right': {
        'shoulder': RIGHT_SHOULDER, 'hip': RIGHT_HIP,
        'knee': RIGHT_KNEE, 'ankle': RIGHT_ANKLE,
        'big_toe': RIGHT_BIG_TOE, 'small_toe': RIGHT_SMALL_TOE, 'heel': RIGHT_HEEL,
    }
}


# =============================================================================
# 2. 基础几何工具
# =============================================================================

def compute_angle_3pts(p1: np.ndarray, p_vertex: np.ndarray, p2: np.ndarray) -> float:
    """
    计算以 p_vertex 为顶点，p1-p_vertex-p2 构成的角度（度数）。
    返回值范围: [0, 180] 度
    """
    v1 = p1 - p_vertex
    v2 = p2 - p_vertex
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product < 1e-8:
        return 0.0  # 退化情况: 点重合
    cos_angle = np.dot(v1, v2) / norm_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def signed_angle_2d(v_from: np.ndarray, v_to: np.ndarray) -> float:
    """
    计算从 v_from 旋转到 v_to 的有符号角度(度数)。
    在图像坐标系(y向下)中:
      正值 = 顺时针旋转
      负值 = 逆时针旋转
    范围: [-180, 180]
    """
    cross = v_from[0] * v_to[1] - v_from[1] * v_to[0]
    dot = np.dot(v_from, v_to)
    return np.degrees(np.arctan2(cross, dot))


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2.0


def compute_body_height(keypoints: np.ndarray) -> float:
    """
    计算身体参考高度(像素), 用于归一化距离类特征。
    使用 shoulder中点 到 ankle中点 的距离作为参考。
    """
    shoulder_mid = midpoint(keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER])
    ankle_mid = midpoint(keypoints[LEFT_ANKLE], keypoints[RIGHT_ANKLE])
    height = np.linalg.norm(shoulder_mid - ankle_mid)
    return max(height, 1.0)  # 避免除零


def smooth_time_series(values: np.ndarray, window: int = 7) -> np.ndarray:
    """
    移动平均时序平滑滤波 (纯numpy实现, 不依赖scipy)。
    修复6: 用于冠状面脚部特征等噪声较大的信号。
    """
    if len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    # 用 'same' 模式保持长度, 边缘用原始值
    padded = np.pad(values, window // 2, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(values)]


def find_peaks_simple(signal: np.ndarray, min_distance: int = 15,
                      min_prominence: float = 5.0) -> np.ndarray:
    """
    简单峰值检测 (纯numpy实现, 替代scipy.signal.find_peaks)。

    找到 signal 中的局部最大值, 满足:
    1. 比相邻点都大
    2. 相邻峰之间间距 >= min_distance
    3. 峰值突出度 >= min_prominence
    """
    if len(signal) < 3:
        return np.array([], dtype=int)

    # 找所有局部最大值
    candidates = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            # 计算prominence: 与左右最近低谷的高度差的较小值
            left_min = np.min(signal[max(0, i - min_distance):i])
            right_min = np.min(signal[i + 1:min(len(signal), i + min_distance + 1)])
            prominence = signal[i] - max(left_min, right_min)
            if prominence >= min_prominence:
                candidates.append((i, signal[i], prominence))

    if not candidates:
        return np.array([], dtype=int)

    # 按值从大到小排序, 贪心选择满足min_distance的峰
    candidates.sort(key=lambda x: -x[1])
    selected = []
    for idx, val, prom in candidates:
        if all(abs(idx - s) >= min_distance for s in selected):
            selected.append(idx)

    return np.array(sorted(selected), dtype=int)


# =============================================================================
# 3. 修复1: 视角方向预处理
# =============================================================================
# - left视频: 人从右→左走, x正方向 = 行走反方向(后方)
# - right视频: 人从左→右走, x正方向 = 行走方向(前方)
# - forward视频: 人背对摄像头走, 画面左=人的左, 无需翻转
# - toward视频: 人面对摄像头走, 画面左=人的右, 需要水平翻转
#
# 关于COCO标注: COCO的left/right始终是"人体自身的左/右"，不随视角变化。
# 因此 toward 视频中虽然画面是镜像的，但标注已经处理了——
# LEFT_KNEE始终是人的左膝。
# 所以关键点索引不需要swap, 但冠状面角度的正负号(左偏/右偏)
# 需要根据forward/toward取反, 因为x轴投影方向翻转了。
# =============================================================================

def get_sagittal_direction_sign(view: str) -> float:
    """
    返回矢状面中"前方"对应的x轴符号。
    - right视频(人从左→右走): 前方 = x正方向, return +1
    - left视频(人从右→左走): 前方 = x负方向, return -1
    """
    if view == 'right':
        return 1.0
    elif view == 'left':
        return -1.0
    else:
        raise ValueError(f"Sagittal direction only for left/right views, got: {view}")


def get_coronal_lateral_sign(view: str) -> float:
    """
    返回冠状面中"人体左侧"对应的x轴符号。
    - forward视频(背对摄像头): 画面左 = 人的左, return +1
    - toward视频(面对摄像头): 画面左 = 人的右, return -1
    """
    if view == 'forward':
        return 1.0
    elif view == 'toward':
        return -1.0
    else:
        raise ValueError(f"Coronal direction only for forward/toward views, got: {view}")


# =============================================================================
# 4. 矢状面特征 (Sagittal Features)
# =============================================================================

def compute_ankle_angle_sagittal(keypoints: np.ndarray, side: str,
                                  direction_sign: float) -> float:
    """
    EVGS #3/#7: 踝关节背屈/跖屈角度 (矢状面)

    修复3 v2: 使用 heel→toe 作为足底方向(而非 ankle→toe),
    这样无论行走方向如何, 角度计算都是对称的, 无需direction_sign。

    方法:
      shin_vec = knee - ankle (胫骨方向, 向上)
      sole_vec = toe_mid - heel (足底方向, 正常时水平向前)
      dorsiflexion = 90° - angle_between(shin_vec, sole_vec)

    原理:
      中立位: shin向上, sole水平 → 夹角≈90° → dorsi=0°
      背屈: sole向shin方向收拢 → 夹角<90° → dorsi为正
      跖屈(如马蹄足): sole向下, 与shin近乎反向 → 夹角>>90° → dorsi为负

    返回: 背屈角度(°)
      正值 = 背屈 (dorsiflexion)
      负值 = 跖屈 (plantarflexion)
    """
    kp = SIDE_KEYPOINTS[side]
    knee = keypoints[kp['knee']]
    ankle = keypoints[kp['ankle']]
    heel = keypoints[kp['heel']]
    toe_mid = midpoint(keypoints[kp['big_toe']], keypoints[kp['small_toe']])

    shin_vec = knee - ankle      # 胫骨方向 (向上)
    sole_vec = toe_mid - heel    # 足底方向 (正常时水平向前)

    raw_angle = compute_angle_3pts(
        knee, ankle,
        ankle + sole_vec  # 构造虚拟点: ankle + 足底方向
    )
    # 更简洁: 直接用两向量夹角
    norm_product = np.linalg.norm(shin_vec) * np.linalg.norm(sole_vec)
    if norm_product < 1e-8:
        return 0.0
    cos_angle = np.dot(shin_vec, sole_vec) / norm_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_between = np.degrees(np.arccos(cos_angle))

    dorsiflexion = 90.0 - angle_between
    return dorsiflexion


def compute_knee_angle_sagittal(keypoints: np.ndarray, side: str,
                                 direction_sign: float) -> float:
    """
    EVGS #9/#10/#11: 膝关节屈曲/伸展角度 (矢状面)

    hip-knee-ankle三点角, 180°=完全伸展。

    返回: 屈曲角度(°)
      正值 = 屈曲 (flexion)
      负值 = 过伸 (hyperextension)
    """
    kp = SIDE_KEYPOINTS[side]
    hip = keypoints[kp['hip']]
    knee = keypoints[kp['knee']]
    ankle = keypoints[kp['ankle']]

    raw_angle = compute_angle_3pts(hip, knee, ankle)
    flexion = 180.0 - raw_angle

    # 判断屈曲/过伸方向
    # 大腿向量和小腿向量
    thigh_vec = hip - knee    # knee → hip
    shank_vec = ankle - knee  # knee → ankle
    cross = thigh_vec[0] * shank_vec[1] - thigh_vec[1] * shank_vec[0]

    # 人面朝右(direction_sign=+1): 屈曲时膝关节向前突出, cross < 0
    # 人面朝左(direction_sign=-1): 屈曲时cross > 0
    if cross * direction_sign > 0:
        flexion = -flexion  # 过伸

    return flexion


def compute_hip_angle_sagittal(keypoints: np.ndarray, side: str,
                                direction_sign: float) -> float:
    """
    EVGS #12/#13: 髋关节屈曲/伸展角度 (矢状面)

    shoulder-hip-knee三点角。

    返回: 髋关节角度(°)
      正值 = 屈曲 (flexion, 大腿在躯干前方)
      负值 = 伸展 (extension, 大腿在躯干后方)
    正常: 站立末期 0-20° 伸展, 摆动相峰值 25-45° 屈曲
    """
    kp = SIDE_KEYPOINTS[side]
    shoulder = keypoints[kp['shoulder']]
    hip = keypoints[kp['hip']]
    knee = keypoints[kp['knee']]

    raw_angle = compute_angle_3pts(shoulder, hip, knee)
    deviation = 180.0 - raw_angle

    # 判断屈曲/伸展方向
    trunk_vec = shoulder - hip  # hip → shoulder (向上)
    thigh_vec = knee - hip      # hip → knee (向下)
    cross = trunk_vec[0] * thigh_vec[1] - trunk_vec[1] * thigh_vec[0]

    # 人面朝右(direction_sign=+1): 屈曲=大腿在躯干前方(向右), cross > 0
    # 人面朝左(direction_sign=-1): 屈曲=大腿在躯干前方(向左), cross < 0
    if cross * direction_sign > 0:
        return deviation   # 屈曲
    else:
        return -deviation  # 伸展


def compute_trunk_inclination_sagittal(keypoints: np.ndarray,
                                        direction_sign: float) -> float:
    """
    EVGS #16: 躯干矢状面姿势

    躯干(shoulder中点 → hip中点)相对于垂直方向的倾斜。

    返回: 倾斜角度(°)
      正值 = 前倾 (forward lean)
      负值 = 后仰 (backward lean)
    正常: -5° 到 5°
    """
    shoulder_mid = midpoint(keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER])
    hip_mid = midpoint(keypoints[LEFT_HIP], keypoints[RIGHT_HIP])

    # 躯干向量: shoulder → hip (向下)
    trunk_vec = hip_mid - shoulder_mid
    vertical = np.array([0.0, 1.0])  # 图像坐标系垂直向下

    # 有符号角度: 从垂直方向到躯干方向
    angle = signed_angle_2d(vertical, trunk_vec)

    # direction_sign 校正:
    # signed_angle_2d 中, 前倾(trunk偏向行走方向)在图像坐标中为负旋转(人朝右时)
    # 所以需要取反: -angle * direction_sign
    # 人面朝右(+1): 前倾→angle为负 → -(-)*1 = 正 ✓
    # 人面朝左(-1): 前倾→angle为正 → -(+)*(-1) = 正 ✓
    return -angle * direction_sign


def compute_initial_contact_angle(keypoints: np.ndarray, side: str,
                                   direction_sign: float) -> float:
    """
    EVGS #1: 初始着地角度

    heel→toe向量相对于水平面的角度, 用于判断着地方式:
    - 正角度(toe高于heel) = 足跟着地 (正常)
    - 接近0 = 平足着地
    - 负角度(toe低于heel) = 足尖着地

    修复3: 使用direction_sign确定"前方"方向。
    """
    kp = SIDE_KEYPOINTS[side]
    heel = keypoints[kp['heel']]
    toe_mid = midpoint(keypoints[kp['big_toe']], keypoints[kp['small_toe']])

    vec = toe_mid - heel  # heel → toe

    # 在行走方向上的投影
    forward_component = vec[0] * direction_sign  # 正值=toe在heel前方
    vertical_component = -vec[1]  # 正值=toe高于heel(图像坐标y向下)

    # 角度: toe相对于heel的仰角
    angle = np.degrees(np.arctan2(vertical_component, abs(forward_component) + 1e-8))
    return angle


def compute_heel_height_normalized(keypoints: np.ndarray, side: str) -> float:
    """
    EVGS #2: 足跟上提 (归一化)

    修复2: 用身体高度归一化, 而不是返回原始像素差。

    返回: 足跟相对于脚趾的归一化高度差
      正值 = 足跟比脚趾低(着地状态)
      负值 = 足跟比脚趾高(已上提)
      量级约 [-0.1, 0.1], 无量纲
    """
    kp = SIDE_KEYPOINTS[side]
    heel = keypoints[kp['heel']]
    toe_mid = midpoint(keypoints[kp['big_toe']], keypoints[kp['small_toe']])
    body_h = compute_body_height(keypoints)

    # y向下, heel_y > toe_y → heel更低(着地)
    raw_diff = heel[1] - toe_mid[1]
    return raw_diff / body_h


def compute_foot_clearance_normalized(keypoints: np.ndarray, side: str) -> float:
    """
    EVGS #6: 足廓清 (归一化)

    修复2: 用身体高度归一化。
    计算脚趾中点相对于身体最低点(两踝中点)的高度差, 归一化后返回。

    返回: 归一化足部离地高度
      0 ≈ 脚在地面
      正值 = 脚抬离地面
      量级约 [0, 0.3]
    """
    kp = SIDE_KEYPOINTS[side]
    toe_mid = midpoint(keypoints[kp['big_toe']], keypoints[kp['small_toe']])
    ankle_mid = midpoint(keypoints[LEFT_ANKLE], keypoints[RIGHT_ANKLE])
    body_h = compute_body_height(keypoints)

    # 用站立侧的脚趾y坐标作为地面参考
    # 在正常步态中, 站立侧的脚趾y最大(最低)
    # 这里简化: 用两踝中点的y + 一个小偏移作为地面近似
    ground_y = max(keypoints[LEFT_ANKLE][1], keypoints[RIGHT_ANKLE][1]) + 30  # 粗略估计

    # 足部离地高度 (正值=离地)
    clearance = (ground_y - toe_mid[1]) / body_h
    return max(clearance, 0.0)


def extract_sagittal_features(keypoints: np.ndarray, side: str,
                               view: str) -> Dict[str, float]:
    """
    提取单帧的所有矢状面(侧面)特征。

    修复1: 接受view参数, 内部处理行走方向。
    """
    direction_sign = get_sagittal_direction_sign(view)
    features = {}
    prefix = f"sag_{side}"

    features[f"{prefix}_initial_contact_angle"] = \
        compute_initial_contact_angle(keypoints, side, direction_sign)

    features[f"{prefix}_heel_height_norm"] = \
        compute_heel_height_normalized(keypoints, side)

    features[f"{prefix}_ankle_angle"] = \
        compute_ankle_angle_sagittal(keypoints, side, direction_sign)

    features[f"{prefix}_foot_clearance_norm"] = \
        compute_foot_clearance_normalized(keypoints, side)

    features[f"{prefix}_knee_angle"] = \
        compute_knee_angle_sagittal(keypoints, side, direction_sign)

    features[f"{prefix}_hip_angle"] = \
        compute_hip_angle_sagittal(keypoints, side, direction_sign)

    features[f"{prefix}_trunk_inclination"] = \
        compute_trunk_inclination_sagittal(keypoints, direction_sign)

    return features


# =============================================================================
# 5. 冠状面特征 (Coronal Features)
# =============================================================================

def compute_hindfoot_angle_coronal(keypoints: np.ndarray, side: str,
                                    lateral_sign: float) -> float:
    """
    EVGS #4: 后足内翻/外翻 (冠状面)

    ankle→heel线段相对于垂直方向的角度。

    返回: 内翻/外翻角度(°)
      正值 = 外翻 (valgus, heel偏向外侧)
      负值 = 内翻 (varus, heel偏向内侧)
    """
    kp = SIDE_KEYPOINTS[side]
    ankle = keypoints[kp['ankle']]
    heel = keypoints[kp['heel']]

    vec = heel - ankle
    vertical = np.array([0.0, 1.0])
    angle = signed_angle_2d(vertical, vec)

    # 校正方向: 外翻=heel偏向身体外侧
    # 左脚: 外侧=图像左方(forward时), heel向左偏=外翻
    # 右脚: 外侧=图像右方(forward时), heel向右偏=外翻
    if side == 'left':
        return -angle * lateral_sign
    else:
        return angle * lateral_sign


def compute_foot_rotation_coronal(keypoints: np.ndarray, side: str,
                                   lateral_sign: float) -> float:
    """
    EVGS #5: 足部旋转 (冠状面)

    从正面看, toe相对于heel的水平偏移, 归一化后用arcsin转角度。

    返回: 旋转角度(°)
      正值 = 外旋 (toe-out)
      负值 = 内旋 (toe-in)
    """
    kp = SIDE_KEYPOINTS[side]
    heel = keypoints[kp['heel']]
    toe_mid = midpoint(keypoints[kp['big_toe']], keypoints[kp['small_toe']])

    dx = toe_mid[0] - heel[0]
    foot_length = np.linalg.norm(toe_mid - heel) + 1e-8
    rotation_ratio = np.clip(dx / foot_length, -1.0, 1.0)
    angle = np.degrees(np.arcsin(rotation_ratio))

    # 校正: 外旋=toe偏向身体外侧
    if side == 'left':
        return -angle * lateral_sign
    else:
        return angle * lateral_sign


def compute_knee_progression_angle_coronal(keypoints: np.ndarray, side: str,
                                            lateral_sign: float) -> float:
    """
    EVGS #8: 膝前进角 (冠状面)

    膝关节相对于 hip-ankle 连线的水平偏移。

    返回: 偏移角度(°)
      正值 = 外旋 (膝盖指向外侧)
      负值 = 内旋 (膝盖指向内侧)
    """
    kp = SIDE_KEYPOINTS[side]
    hip = keypoints[kp['hip']]
    knee = keypoints[kp['knee']]
    ankle = keypoints[kp['ankle']]

    # hip到ankle连线上knee对应位置的x坐标(线性插值)
    t = (knee[1] - hip[1]) / (ankle[1] - hip[1] + 1e-8)
    t = np.clip(t, 0, 1)
    expected_x = hip[0] + t * (ankle[0] - hip[0])

    knee_offset = knee[0] - expected_x
    thigh_length = np.linalg.norm(knee - hip) + 1e-8
    offset_ratio = np.clip(knee_offset / thigh_length, -1.0, 1.0)
    angle = np.degrees(np.arcsin(offset_ratio))

    # 校正方向
    if side == 'left':
        return -angle * lateral_sign  # 左腿, 膝盖偏左=外旋
    else:
        return angle * lateral_sign   # 右腿, 膝盖偏右=外旋


def compute_pelvic_obliquity_coronal(keypoints: np.ndarray,
                                      lateral_sign: float) -> float:
    """
    EVGS #14: 骨盆倾斜 (冠状面)

    左右髋连线相对于水平方向的角度。

    返回: 倾斜角度(°)
      正值 = 支撑侧上提 (正常或代偿)
      负值 = 支撑侧下降 (Trendelenburg)

    注意: 需要知道哪侧是支撑侧才能完整解读。
    这里返回的是"右髋相对于左髋的高度差"。
    """
    left_hip = keypoints[LEFT_HIP]
    right_hip = keypoints[RIGHT_HIP]

    vec = right_hip - left_hip
    # 图像坐标y向下, 所以 -vec[1] = 右髋比左髋高多少
    angle = np.degrees(np.arctan2(-vec[1], vec[0]))

    return angle * lateral_sign


def compute_pelvic_rotation_coronal(keypoints: np.ndarray,
                                     lateral_sign: float) -> float:
    """
    EVGS #15: 骨盆旋转 (冠状面)

    修复5: 重新设计。
    骨盆旋转在2D正面视角的主要表现是: 左右髋投影宽度相对于肩宽的变化。
    - 无旋转: hip_width / shoulder_width ≈ 常数
    - 有旋转: 骨盆转动后, 投影宽度缩小, 比值减小

    同时, 旋转方向通过两侧不对称性判断:
    旋转时离摄像机近的一侧髋看起来偏向画面外侧。

    返回: 旋转估计值
      正值 = 顺时针旋转(从上方看, 左侧向前)
      负值 = 逆时针旋转(从上方看, 右侧向前)
    """
    left_hip = keypoints[LEFT_HIP]
    right_hip = keypoints[RIGHT_HIP]
    left_shoulder = keypoints[LEFT_SHOULDER]
    right_shoulder = keypoints[RIGHT_SHOULDER]

    hip_width = abs(right_hip[0] - left_hip[0]) + 1e-8
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0]) + 1e-8

    # 宽度比 (正常约0.6-0.8)
    width_ratio = hip_width / shoulder_width

    # 不对称性: 左侧髋到中线的距离 vs 右侧
    body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    left_dist = abs(left_hip[0] - body_center_x)
    right_dist = abs(right_hip[0] - body_center_x)

    # 不对称比: 正值=左侧更远(左侧向摄像机, 即右侧向前旋转)
    asymmetry = (left_dist - right_dist) / (hip_width + 1e-8)

    # 综合: 旋转角度近似 = asymmetry * 缩放因子
    # 这里用arcsin做非线性映射
    rotation = np.degrees(np.arcsin(np.clip(asymmetry, -1.0, 1.0)))

    return rotation * lateral_sign


def compute_lateral_trunk_shift_coronal(keypoints: np.ndarray,
                                         lateral_sign: float) -> float:
    """
    EVGS #17: 最大侧方偏移 (冠状面)

    肩中点相对于髋中点的水平偏移, 用髋宽归一化。

    返回: 归一化侧方偏移(无量纲)
      正值 = 向右偏移(forward视角中)
      负值 = 向左偏移
    """
    shoulder_mid = midpoint(keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER])
    hip_mid = midpoint(keypoints[LEFT_HIP], keypoints[RIGHT_HIP])
    hip_width = abs(keypoints[RIGHT_HIP][0] - keypoints[LEFT_HIP][0]) + 1e-8

    lateral_shift = (shoulder_mid[0] - hip_mid[0]) / hip_width
    return lateral_shift * lateral_sign


def extract_coronal_features(keypoints: np.ndarray, side: str,
                              view: str) -> Dict[str, float]:
    """
    提取单帧的所有冠状面(正面)特征。

    修复1: 接受view参数, 处理forward/toward的方向差异。
    """
    lateral_sign = get_coronal_lateral_sign(view)
    features = {}
    prefix = f"cor_{side}"

    features[f"{prefix}_hindfoot_angle"] = \
        compute_hindfoot_angle_coronal(keypoints, side, lateral_sign)
    features[f"{prefix}_foot_rotation"] = \
        compute_foot_rotation_coronal(keypoints, side, lateral_sign)
    features[f"{prefix}_knee_progression_angle"] = \
        compute_knee_progression_angle_coronal(keypoints, side, lateral_sign)

    # 以下三个特征与肢体侧无关, 只计算一次
    features["cor_pelvic_obliquity"] = \
        compute_pelvic_obliquity_coronal(keypoints, lateral_sign)
    features["cor_pelvic_rotation"] = \
        compute_pelvic_rotation_coronal(keypoints, lateral_sign)
    features["cor_lateral_trunk_shift"] = \
        compute_lateral_trunk_shift_coronal(keypoints, lateral_sign)

    return features


# =============================================================================
# 6. 步态周期分割 (修复4)
# =============================================================================

def detect_initial_contacts(ankle_y_series: np.ndarray,
                             fps: int = 30,
                             min_cycle_frames: int = 15) -> np.ndarray:
    """
    检测 initial contact (IC) 事件——足跟着地时刻。

    原理: 在矢状面视角中, 着地时踝关节y坐标达到局部最大值(最低点)。
    用 find_peaks 在 ankle_y 信号上找峰值(y向下所以峰值=最低点)。

    Args:
        ankle_y_series: 踝关节y坐标时间序列
        fps: 视频帧率
        min_cycle_frames: 两次IC之间的最小帧数(避免误检)

    Returns:
        IC帧索引数组
    """
    # 先平滑
    if len(ankle_y_series) < 10:
        return np.array([])

    smoothed = smooth_time_series(ankle_y_series, window=11)

    # 找峰值 (y坐标的局部最大值 = 脚的最低位置 = 着地)
    peaks = find_peaks_simple(
        smoothed,
        min_distance=min_cycle_frames,
        min_prominence=5.0
    )

    return peaks


def split_stance_swing(ic_indices: np.ndarray,
                        total_frames: int) -> List[Dict[str, Tuple[int, int]]]:
    """
    根据IC事件将步态划分为站立相和摆动相。

    简化假设: 站立相约占步态周期的60%, 摆动相约40%。
    每个周期: IC[i] → IC[i] + 0.6*(IC[i+1]-IC[i]) = 站立相
              → IC[i+1] = 摆动相

    Returns:
        列表, 每个元素是一个字典:
        {'stance': (start, end), 'swing': (start, end)}
    """
    cycles = []
    for i in range(len(ic_indices) - 1):
        start = ic_indices[i]
        end = ic_indices[i + 1]
        cycle_length = end - start

        stance_end = start + int(cycle_length * 0.6)

        cycles.append({
            'stance': (start, stance_end),
            'swing': (stance_end, end),
            'full': (start, end),
        })

    return cycles


# =============================================================================
# 7. 数据加载 (修复7)
# =============================================================================

def load_frame_json(json_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    从比赛JSON文件中加载单帧数据。

    修复7: 如果有多个instance, 选择bounding box面积最大的(最可能是目标患儿)。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    instances = data['instance_info']

    if len(instances) == 1:
        instance = instances[0]
    else:
        # 选择bbox面积最大的instance
        best_idx = 0
        best_area = 0
        for idx, inst in enumerate(instances):
            bbox = inst.get('gt_bbox_xywh_px', [0, 0, 0, 0])
            area = bbox[2] * bbox[3]  # width * height
            if area > best_area:
                best_area = area
                best_idx = idx
        instance = instances[best_idx]

    kp_flat = instance['keypoints']
    keypoints = np.array(kp_flat).reshape(-1, 2)

    scores = np.array(instance['keypoint_scores'])

    video_info = data.get('video_info', {})

    return keypoints, scores, video_info


def determine_view_from_filename(filename: str) -> str:
    """根据视频文件名推断拍摄视角。"""
    filename_lower = filename.lower()
    
    # 新增兼容：如果文件名有 backward，按背对摄像头(forward)的几何逻辑处理
    if 'backward' in filename_lower:
        return 'forward'
        
    for view in ['forward', 'toward', 'left', 'right']:
        if view in filename_lower:
            return view
            
    return 'unknown'

# =============================================================================
# 8. 完整Pipeline (修复8: 按步态阶段聚合)
# =============================================================================

def extract_video_features(frames_keypoints: List[np.ndarray],
                           view: str,
                           side: str = 'left',
                           fps: int = 30) -> Dict[str, np.ndarray]:
    """
    从整个视频(多帧)中提取逐帧特征时间序列。

    修复6: 对冠状面脚部特征做时序平滑。
    """
    all_features = []
    for kp in frames_keypoints:
        if view in ('left', 'right'):
            features = extract_sagittal_features(kp, side, view)
        elif view in ('forward', 'toward'):
            features = extract_coronal_features(kp, side, view)
        else:
            raise ValueError(f"Unknown view: {view}")
        all_features.append(features)

    if not all_features:
        return {}

    result = {}
    for key in all_features[0]:
        values = np.array([f[key] for f in all_features])
        result[key] = values

    # 修复6: 对冠状面脚部特征做平滑
    noisy_keys = ['hindfoot_angle', 'foot_rotation']
    for key in result:
        if any(nk in key for nk in noisy_keys):
            result[key] = smooth_time_series(result[key], window=11)

    return result


def aggregate_by_gait_phase(time_series: Dict[str, np.ndarray],
                             frames_keypoints: List[np.ndarray],
                             side: str,
                             fps: int = 30) -> Dict[str, float]:
    """
    修复8: 按步态阶段分别聚合特征, 而不是整段视频粗暴取统计量。

    对矢状面特征, 先检测步态周期, 然后:
    - 站立相特征(#3,#9,#12): 在stance phase内取峰值
    - 摆动相特征(#7,#10,#11,#13): 在swing phase内取峰值
    - 初始着地(#1): 在IC时刻取值
    - 全周期特征(#2,#6,#16): 对整个周期聚合

    对冠状面特征, 用站立中期(mid-stance)的值。
    """
    aggregated = {}

    # 判断是矢状面还是冠状面
    sample_key = list(time_series.keys())[0] if time_series else ""
    is_sagittal = sample_key.startswith("sag_")

    if is_sagittal and frames_keypoints:
        # 用目标侧踝关节y坐标检测IC
        kp_indices = SIDE_KEYPOINTS[side]
        ankle_y = np.array([kp[kp_indices['ankle']][1] for kp in frames_keypoints])
        ic_indices = detect_initial_contacts(ankle_y, fps)
        cycles = split_stance_swing(ic_indices, len(frames_keypoints))

        if len(cycles) >= 1:
            # ---- 按阶段提取峰值 ----
            # EVGS子项到特征key和阶段的映射
            phase_mapping = {
                'ankle_angle': [('stance', 'max'), ('swing', 'max')],    # #3(stance max背屈), #7(swing max背屈)
                'knee_angle':  [('stance', 'min'), ('swing', 'min'),     # #9(stance min=max伸展)
                                ('swing', 'max')],                        # #11(swing max=peak屈曲)
                'hip_angle':   [('stance', 'min'), ('swing', 'max')],    # #12(stance min=max伸展), #13(swing max=peak屈曲)
            }

            for feat_key, phases in phase_mapping.items():
                full_key = None
                for k in time_series:
                    if feat_key in k:
                        full_key = k
                        break
                if full_key is None:
                    continue

                values = time_series[full_key]
                for phase_name, agg_type in phases:
                    phase_values = []
                    for cycle in cycles:
                        s, e = cycle[phase_name]
                        if s < len(values) and e <= len(values) and s < e:
                            segment = values[s:e]
                            if agg_type == 'max':
                                phase_values.append(np.max(segment))
                            elif agg_type == 'min':
                                phase_values.append(np.min(segment))

                    if phase_values:
                        tag = f"{full_key}_{phase_name}_{agg_type}"
                        aggregated[tag] = float(np.mean(phase_values))
                        aggregated[f"{tag}_std"] = float(np.std(phase_values))

            # 初始着地角度: 在IC时刻取值
            for k in time_series:
                if 'initial_contact' in k:
                    ic_vals = [time_series[k][ic] for ic in ic_indices
                               if ic < len(time_series[k])]
                    if ic_vals:
                        aggregated[f"{k}_at_ic"] = float(np.mean(ic_vals))
                        aggregated[f"{k}_at_ic_std"] = float(np.std(ic_vals))

            # 足跟高度: 在stance早期评估
            for k in time_series:
                if 'heel_height' in k:
                    for cycle in cycles:
                        s, e = cycle['stance']
                        # 站立相后半段检查足跟是否上提
                        mid = (s + e) // 2
                        if mid < len(time_series[k]) and e <= len(time_series[k]):
                            late_stance = time_series[k][mid:e]
                            if len(late_stance) > 0:
                                # 最小值=足跟最高(最大上提量)
                                aggregated.setdefault(f"{k}_late_stance_min_list", []).append(
                                    float(np.min(late_stance)))
                    # 平均所有周期
                    list_key = f"{k}_late_stance_min_list"
                    if list_key in aggregated:
                        vals = aggregated.pop(list_key)
                        aggregated[f"{k}_late_stance_min"] = float(np.mean(vals))

        # 无论是否检测到步态周期, 都输出全局统计量作为fallback
        for feat_name, values in time_series.items():
            aggregated[f"{feat_name}_mean"] = float(np.mean(values))
            aggregated[f"{feat_name}_std"] = float(np.std(values))
            aggregated[f"{feat_name}_max"] = float(np.max(values))
            aggregated[f"{feat_name}_min"] = float(np.min(values))
            aggregated[f"{feat_name}_range"] = float(np.ptp(values))

    else:
        # 冠状面: 用站立中期(mid-stance)的值
        # 简化: 取中间50%帧的统计量, 避免步态起止的不稳定
        for feat_name, values in time_series.items():
            n = len(values)
            q1, q3 = n // 4, 3 * n // 4
            mid_values = values[q1:q3] if q3 > q1 else values

            aggregated[f"{feat_name}_mean"] = float(np.mean(values))
            aggregated[f"{feat_name}_std"] = float(np.std(values))
            aggregated[f"{feat_name}_max"] = float(np.max(values))
            aggregated[f"{feat_name}_min"] = float(np.min(values))
            aggregated[f"{feat_name}_range"] = float(np.ptp(values))
            aggregated[f"{feat_name}_mid_mean"] = float(np.mean(mid_values))
            aggregated[f"{feat_name}_mid_std"] = float(np.std(mid_values))

    return aggregated


# =============================================================================
# 9. 顶层处理函数
# =============================================================================

def process_video_directory(video_dir: str,
                            side: str = 'left',
                            fps: int = 30) -> Dict[str, float]:
    """
    处理一个视频目录下的所有帧JSON文件, 返回聚合特征。
    """
    json_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.json')])
    if not json_files:
        return {}

    first_kp, _, video_info = load_frame_json(os.path.join(video_dir, json_files[0]))
    
    # 核心修复在这里：👇
    video_name = video_info.get('video_name', '')
    if not video_name:  
        # 如果 JSON 里没有存 video_name，就直接用所在文件夹的名字！
        video_name = os.path.basename(video_dir)
        
    view = determine_view_from_filename(video_name)
    detected_fps = video_info.get('fps', fps)

    frames_keypoints = []
    for jf in json_files:
        kp, scores, _ = load_frame_json(os.path.join(video_dir, jf))
        frames_keypoints.append(kp)

    time_series = extract_video_features(frames_keypoints, view, side, detected_fps)
    aggregated = aggregate_by_gait_phase(time_series, frames_keypoints, side, detected_fps)

    # 附加元信息
    aggregated['_view'] = hash(view) % 100  # encode view as numeric
    aggregated['_num_frames'] = len(frames_keypoints)

    return aggregated


# =============================================================================
# 10. EVGS 映射表
# =============================================================================

EVGS_VIEW_MAPPING = {
    1:  {"name": "初始着地", "en": "Initial Contact",
         "view": "sagittal", "feature": "initial_contact_angle", "phase": "ic"},
    2:  {"name": "足跟上提", "en": "Heel Rise",
         "view": "sagittal", "feature": "heel_height_norm", "phase": "late_stance"},
    3:  {"name": "最大踝背屈(站立相)", "en": "Max Ankle Dorsiflexion Stance",
         "view": "sagittal", "feature": "ankle_angle", "phase": "stance"},
    4:  {"name": "后足内翻/外翻", "en": "Hindfoot Varus/Valgus",
         "view": "coronal", "feature": "hindfoot_angle", "phase": "stance"},
    5:  {"name": "足部旋转", "en": "Foot Rotation",
         "view": "coronal", "feature": "foot_rotation", "phase": "stance"},
    6:  {"name": "足廓清", "en": "Foot Clearance",
         "view": "sagittal", "feature": "foot_clearance_norm", "phase": "swing"},
    7:  {"name": "最大踝背屈(摆动相)", "en": "Max Ankle Dorsiflexion Swing",
         "view": "sagittal", "feature": "ankle_angle", "phase": "swing"},
    8:  {"name": "站立中期膝前进角", "en": "Mid-stance Knee Progression Angle",
         "view": "coronal", "feature": "knee_progression_angle", "phase": "mid_stance"},
    9:  {"name": "站立相膝伸展峰值", "en": "Knee Extension Peak Stance",
         "view": "sagittal", "feature": "knee_angle", "phase": "stance"},
    10: {"name": "摆动末期位置", "en": "Terminal Swing Position",
         "view": "sagittal", "feature": "knee_angle", "phase": "late_swing"},
    11: {"name": "摆动相膝屈曲峰值", "en": "Swing Knee Flexion Peak",
         "view": "sagittal", "feature": "knee_angle", "phase": "swing"},
    12: {"name": "站立相髋伸展峰值", "en": "Hip Extension Peak Stance",
         "view": "sagittal", "feature": "hip_angle", "phase": "stance"},
    13: {"name": "摆动相髋屈曲峰值", "en": "Swing Hip Flexion Peak",
         "view": "sagittal", "feature": "hip_angle", "phase": "swing"},
    14: {"name": "站立中期骨盆倾斜", "en": "Mid-stance Pelvic Obliquity",
         "view": "coronal", "feature": "pelvic_obliquity", "phase": "mid_stance"},
    15: {"name": "站立中期骨盆旋转", "en": "Mid-stance Pelvic Rotation",
         "view": "coronal", "feature": "pelvic_rotation", "phase": "mid_stance"},
    16: {"name": "矢状面峰值姿势", "en": "Trunk Sagittal Posture",
         "view": "sagittal", "feature": "trunk_inclination", "phase": "full"},
    17: {"name": "最大侧方偏移", "en": "Maximum Lateral Shift",
         "view": "coronal", "feature": "lateral_trunk_shift", "phase": "full"},
}


# =============================================================================
# 11. 验证测试
# =============================================================================

def _make_sagittal_kp(pose: str = 'normal') -> np.ndarray:
    """矢状面测试关键点 (人面朝右走)"""
    kp = np.zeros((133, 2))
    if pose == 'normal':
        kp[LEFT_SHOULDER] = kp[RIGHT_SHOULDER] = [960, 350]
        kp[LEFT_HIP] = kp[RIGHT_HIP] = [960, 580]
        kp[LEFT_KNEE] = kp[RIGHT_KNEE] = [980, 760]   # 膝微前
        kp[LEFT_ANKLE] = kp[RIGHT_ANKLE] = [960, 930]
        kp[LEFT_BIG_TOE] = kp[RIGHT_BIG_TOE] = [1040, 960]
        kp[LEFT_SMALL_TOE] = kp[RIGHT_SMALL_TOE] = [1030, 960]
        kp[LEFT_HEEL] = kp[RIGHT_HEEL] = [920, 960]
    elif pose == 'crouch':
        kp[LEFT_SHOULDER] = kp[RIGHT_SHOULDER] = [940, 380]
        kp[LEFT_HIP] = kp[RIGHT_HIP] = [940, 580]
        kp[LEFT_KNEE] = kp[RIGHT_KNEE] = [1030, 740]  # 膝大幅前移
        kp[LEFT_ANKLE] = kp[RIGHT_ANKLE] = [960, 920]
        kp[LEFT_BIG_TOE] = kp[RIGHT_BIG_TOE] = [1040, 950]
        kp[LEFT_SMALL_TOE] = kp[RIGHT_SMALL_TOE] = [1030, 950]
        kp[LEFT_HEEL] = kp[RIGHT_HEEL] = [920, 950]
    elif pose == 'equinus':
        kp[LEFT_SHOULDER] = kp[RIGHT_SHOULDER] = [960, 350]
        kp[LEFT_HIP] = kp[RIGHT_HIP] = [960, 580]
        kp[LEFT_KNEE] = kp[RIGHT_KNEE] = [975, 760]
        kp[LEFT_ANKLE] = kp[RIGHT_ANKLE] = [990, 900]
        # 足尖着地, 足跟悬空
        kp[LEFT_BIG_TOE] = kp[RIGHT_BIG_TOE] = [1000, 960]
        kp[LEFT_SMALL_TOE] = kp[RIGHT_SMALL_TOE] = [995, 960]
        kp[LEFT_HEEL] = kp[RIGHT_HEEL] = [970, 910]
    return kp


def _make_coronal_kp(pose: str = 'normal') -> np.ndarray:
    """冠状面测试关键点 (forward视角)"""
    kp = np.zeros((133, 2))
    if pose == 'normal':
        kp[LEFT_SHOULDER] = [880, 350]; kp[RIGHT_SHOULDER] = [1040, 350]
        kp[LEFT_HIP] = [910, 580]; kp[RIGHT_HIP] = [1010, 580]
        kp[LEFT_KNEE] = [910, 760]; kp[RIGHT_KNEE] = [1010, 760]
        kp[LEFT_ANKLE] = [910, 930]; kp[RIGHT_ANKLE] = [1010, 930]
        kp[LEFT_BIG_TOE] = [900, 960]; kp[LEFT_SMALL_TOE] = [920, 960]; kp[LEFT_HEEL] = [910, 960]
        kp[RIGHT_BIG_TOE] = [1020, 960]; kp[RIGHT_SMALL_TOE] = [1000, 960]; kp[RIGHT_HEEL] = [1010, 960]
    elif pose == 'pelvic_drop':
        kp[LEFT_SHOULDER] = [860, 350]; kp[RIGHT_SHOULDER] = [1020, 350]
        kp[LEFT_HIP] = [900, 570]; kp[RIGHT_HIP] = [1000, 600]  # 右髋下降
        kp[LEFT_KNEE] = [940, 760]; kp[RIGHT_KNEE] = [980, 760]  # 膝内偏
        kp[LEFT_ANKLE] = [900, 930]; kp[RIGHT_ANKLE] = [1000, 930]
        kp[LEFT_BIG_TOE] = [890, 960]; kp[LEFT_SMALL_TOE] = [910, 960]; kp[LEFT_HEEL] = [900, 960]
        kp[RIGHT_BIG_TOE] = [1010, 960]; kp[RIGHT_SMALL_TOE] = [990, 960]; kp[RIGHT_HEEL] = [1000, 960]
    return kp


def run_validation():
    print("=" * 60)
    print("EVGS Feature Engineering v2 — 验证测试")
    print("=" * 60)

    for pose in ['normal', 'crouch', 'equinus']:
        print(f"\n--- 矢状面 {pose.upper()} (right view, 人面朝右) ---")
        kp = _make_sagittal_kp(pose)
        feats = extract_sagittal_features(kp, 'left', 'right')
        for k, v in feats.items():
            print(f"  {k}: {v:+.2f}")

    for pose in ['normal', 'pelvic_drop']:
        print(f"\n--- 冠状面 {pose.upper()} (forward view) ---")
        kp = _make_coronal_kp(pose)
        feats = extract_coronal_features(kp, 'left', 'forward')
        for k, v in feats.items():
            print(f"  {k}: {v:+.2f}")

    # 测试步态周期检测
    print("\n--- 步态周期检测测试 ---")
    # 模拟2个步态周期的ankle y坐标
    t = np.linspace(0, 4 * np.pi, 120)
    ankle_y = 930 + 30 * np.sin(t)  # 峰值=着地
    ics = detect_initial_contacts(ankle_y, fps=30)
    print(f"  模拟120帧, 检测到 {len(ics)} 个IC事件, 帧索引: {ics}")
    cycles = split_stance_swing(ics, 120)
    for i, c in enumerate(cycles):
        print(f"  周期{i}: stance={c['stance']}, swing={c['swing']}")


if __name__ == "__main__":
    run_validation()
