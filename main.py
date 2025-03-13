import pygame
import random
import math
import numpy as np
from PIL import Image
from datetime import datetime

# Hard code
config_dict = {
    'null': ([0.0, 0.0, 0.0], 0),
    '0': [([90, 90, 180], 1), ([0, 0, 180], 1), ([0, 90, 90], 1), ([270, 180, 180], 1), ([0, 0, 180], 1), ([0, 0, 270], 1)],
    '1': [([0, 0, 0], 0)] * 3 + [([180, 180, 180], 1), ([0, 0, 180], 1), ([0, 0, 0], 1)],
    '2': [([90, 90, 90], 1), ([90, 90, 180], 1), ([0, 0, 90], 1), ([270, 180, 180], 1), ([0, 0, 270], 1), ([270, 270, 270], 1)],
    '3': [([90, 90, 90], 1)] * 3 + [([270, 180, 180], 1), ([0, 180, 270], 1), ([0, 0, 270], 1)],
    '4': [([180, 180, 180], 1), ([0, 0, 90], 1), ([0, 0, 0], 0), ([180, 180, 180], 1), ([0, 180, 270], 1), ([0, 0, 0], 1)],
    '5': [([90, 90, 180], 1), ([0, 0, 90], 1), ([90, 90, 90], 1), ([270, 270, 270], 1), ([180, 180, 270], 1), ([0, 0, 270], 1)],
    '6': [([90, 90, 180], 1), ([0, 90, 180], 1), ([0, 0, 90], 1), ([270, 270, 270], 1), ([180, 180, 270], 1), ([0, 0, 270], 1)],
    '7': [([90, 90, 90], 1), ([0, 0, 0], 0), ([0, 0, 0], 0), ([180, 180, 270], 1), ([0, 0, 180], 1), ([0, 0, 0], 1)],
    '8': [([90, 90, 180], 1), ([0, 90, 180], 1), ([0, 0, 90], 1), ([180, 180, 270], 1), ([0, 180, 270], 1), ([0, 0, 270], 1)],
    '9': [([90, 90, 180], 1), ([0, 0, 90], 1), ([90, 90, 90], 1), ([180, 180, 270], 1), ([0, 180, 270], 1), ([0, 0, 270], 1)],
    ':': [([0, 0, 0], 0), ([90, 90, 90], 0.5), ([90, 90, 90], 0.5)],
    '-': [([0, 0, 0], 0), ([90, 90, 270], 1), ([0, 0, 0], 0)],
    '_': [([0.0, 0.0, 0.0], 0)] * 3,
    'A': [([70, 70, 180], 1), ([0, 90, 180], 1), ([0, 0, 0], 1), ([180, 180, 270+20], 1), ([0, 180, 270], 1), ([0, 0, 0], 1)],
    'B': [([90, 90, 180], 1), ([0, 90, 180], 1), ([0, 0, 90], 1), ([180-30, 180-30, 270], 1), ([30, 150, 270], 1), ([30, 30, 270], 1)],
    'C': [([90, 90, 180], 1), ([0, 0, 180], 1), ([0, 0, 90], 1), ([270, 270, 270], 1), ([0.0, 0.0, 0.0], 0), ([270, 270, 270], 1)],
    'D': [([80, 80, 180], 1), ([0, 0, 180], 1), ([0, 0, 90], 1), ([180, 180, 280], 1), ([0, 0, 180], 1), ([5, 5, 270], 1)],
    'E': [([90, 90, 180], 1), ([0, 90, 180], 1), ([0, 0, 90], 1), ([270, 270, 270], 1), ([270, 270, 270], 1), ([270, 270, 270], 1)],
    'F': [([90, 90, 180], 1), ([0, 90, 180], 1), ([0, 0, 0], 1), ([270, 270, 270], 1), ([270, 270, 270], 1), ([0, 0, 0], 0)],
    'G': [([80, 80, 180], 1), ([0, 0, 180], 1), ([0, 0, 100], 1), ([280, 280, 280], 1), ([180, 180, 270], 1), ([5, 260, 260], 1)],
    'H': [([180, 180, 180], 1), ([0, 90, 180], 1), ([0, 0, 0], 1), ([180, 180, 180], 1), ([0, 180, 270], 1), ([0, 0, 0], 1)],
    'I': [([90, 180, 270], 1), ([0, 0, 180], 1), ([0, 90, 270], 1)],
    'J': [([90, 90, 90], 1), ([0, 0, 0], 0), ([100, 100, 100], 1), ([90, 180, 270], 1), ([0, 0, 180], 1), ([0, 0, 260], 1)],
    'K': [([180, 180, 180], 1), ([0, 45, 180], 1), ([0, 0, 0], 1), ([180+45, ]*3, 1), ([0, ]*3, 0), ([270+45, ]*3, 1)],
    'L': [([180,]*3, 1), ([0, 0, 180], 1), ([0, 0, 90], 1), ([0, ]*3, 0), ([0, ]*3, 0), ([270, ]*3, 1)],
    'M': [([135, 135, 180], 1), ([0, 0, 180], 1), [(0, 0, 0), 1], ([0, 0, 0], 0), ([45, 180, 270+45], 1), ([0, ]*3, 1), ([180, 180, 180+45], 1), ([0, 0, 180], 1), [(0, 0, 0), 1]],
    'N': [([180-30, 180-30, 180], 1), ([0, 0, 180], 1), [(0, 0, 0), 1], ([180, 180, 180], 1), ([0, 0, 180], 1), [(0, 0, 360-30), 1]],
    'O': [([80, 80, 180], 1), ([0, 0, 180], 1), ([0, 0, 100], 1), ([180, 180, 280], 1), ([0, 0, 180], 1), ([0, 0, 260], 1)],
    'P': [([90, 90, 180], 1), ([0, 90, 180], 1), ([0, 0, 0], 1), ([180, 180, 270], 1), ([0, 0, 270], 1), ([0, 0, 0], 0)],
    'Q': [([80, 80, 180], 1), ([0, 0, 180], 1), ([0, 0, 100], 1), ([180, 180, 280], 1), ([0, 0, 180], 1), ([0, 120, 260], 1)],
    'R': [([90, 90, 180], 1), ([0, 120, 180], 1), ([0, 0, 0], 1), ([180, 180, 270], 1), ([0, 0, 270], 1), ([270+45, ]*3, 1)],
    'S': [([90, 90, 180], 1), ([0, 90, 90], 1), ([90, 90, 90], 1), ([270, 270, 270], 1), ([180, 180, 270], 1), ([0, 0, 270], 1)],
    'T': [([90, 90, 90], 1), ([0, ]*3, 0), ([0, ]*3, 0), ([90, 180, 270], 1), ([0, 0, 180], 1), ([0, 0, 0], 1), ([270, ]*3, 1), ([0, ]*3, 0), ([0, ]*3, 0)],
    'U': [([180, 180, 180], 1), ([0, 0, 180], 1), [(0, 0, 90), 1], ([0, 0, 0], 0), ([0, 0, 0], 0), ([90, 90, 270], 1), ([180, 180, 180], 1), ([0, 0, 180], 1), [(0, 0, 270), 1]],
    'V': [([180, 180, 180], 1), ([0, 0, 180-45], 1), [(0, 0, 0), 0], ([0, 0, 0], 0), ([0, 0, 0], 0), ([90-45, 90-45, 270+45], 1), ([180, 180, 180], 1), ([0, 0, 180+45], 1), [(0, 0, 0), 0]],
    'W': [([180, ]*3, 1), ([0, 0, 180], 1), ([0, 0, 45], 1), ([180, ]*3, 1), ([0, 135, 180+45], 1), ([0, ]*3, 0), ([180, ]*3, 1), ([0, 0, 180], 1), ([0, 0, 270+45], 1)],
    'X': [([135, ]*3, 1), ([0, ]*3, 0), ([45, ]*3, 1), ([0, ]*3, 0), ([45, 180+45, 270+45], 1), ([0, ]*3, 0), ([180+45, ]*3, 1), ([0, ]*3, 0), ([270+45, ]*3, 1)],
    'Y': [([135, ]*3, 1), ([0, ]*3, 0), ([0, ]*3, 0), ([0, ]*3, 0), ([45, 180, 270+45], 1), ([0, ]*3, 1), ([180+45, ]*3, 1), ([0, ]*3, 0), ([0, ]*3, 0)],
    'Z': [([90, ]*3, 1), ([0, ]*3, 0), ([45, 45, 90], 1), ([90, 90, 270], 1), ([45, 45, 180+45], 1), ([90, 90, 270], 1), ([180+45, 180+45, 270], 1), ([0, ]*3, 0), (([270, ]*3, 1))],
    'dog_1': [([45, 45, 180+45], 1), ([270+45, 180+45, 180+45], 1), ([270+45, 180+45, 180+45], 1),
              ([270+45, 90, 90], 1), ([45, 45, 45], 1), ([90, 90, 90], 1),
              ([90, 90, 270], 1), ([180-20, 180-20, 180+20], 1), ([90, 90, 270], 1),
              ([45, 45, 270], 1), ([270+45, 270+45, 270+45], 1), ([270, 270, 270], 1),
              ([270+45, 90+45, 90+45], 1), ([45, 45, 90+45], 1), ([45, 45, 90+45], 1),
              ] + [([0, 0, 0], 0)]*3 +
             [([180, 180, 180], 1), ([0, 90, 180], 1), ([0, 0, 0], 1)] + [([180, 180, 180], 1), ([0, 270, 180], 1), ([0, 0, 0], 1)] +
             [([90, 90, 180], 1), ([0, 90, 180], 1), ([0, 0, 90], 1)] + [([270, 270, 270], 1), ] * 3 +
             [([180, 180, 180], 1), ([0, 180, 180], 1), ([0, 0, 90], 1)] * 2 +
             [([90, 180, 90], 1), ([0, 0, 180], 1), ([0, 0, 90], 1), ([180, 180, 270], 1), ([0, 0, 180], 1), ([0, 0, 270], 1)],
    'dog_2': [([45, 45, 180+45], 1), ([270+45, 180+45, 180+45], 1), ([270+45, 180+45, 180+45], 1),
              ([270+45, 90, 90], 1), ([70, 70, 70], 1), ([90-10, 90-10, 90-10], 1),
              ([90, 90, 270], 1), ([180-35, 180-35, 180+35], 1), ([90-10, 90-10, 270+10], 1),
              ([45, 45, 270], 1), ([270+15, 270+15, 270+15], 1), ([270+10, 270+10, 270+10], 1),
              ([270+45, 90+45, 90+45], 1), ([45, 45, 90+45], 1), ([45, 45, 90+45], 1),
              ] + [([0, 0, 0], 0)]*3 +
             [([180, 180, 180], 1), ([0, 90, 180], 1), ([0, 0, 0], 1)] + [([180, 180, 180], 1), ([0, 270, 180], 1), ([0, 0, 0], 1)] +
             [([90, 90, 180], 1), ([0, 90, 180], 1), ([0, 0, 90], 1)] + [([270, 270, 270], 1), ] * 3 +
             [([180, 180, 180], 1), ([0, 180, 180], 1), ([0, 0, 90], 1)] * 2 +
             [([90, 180, 90], 1), ([0, 0, 180], 1), ([0, 0, 90], 1), ([180, 180, 270], 1), ([0, 0, 180], 1), ([0, 0, 270], 1)]
}


class Clock:
    active_animations = []
    _image_cache = {}
    _cache_enabled = True

    def __init__(self, x, y, size):
        self.target_lengths = None
        self.target_angles = None
        self.start_lengths = None
        self.start_angles = None
        self.start_time = None
        self.center = (x + size // 2, y + size // 2)
        self.size = size
        self.original_clock_length = size // 2
        self.current_angles = [0.0, 0.0, 0.0]
        self.current_length = [1.0, 1.0, 1.0]
        self.nearest = False
        self.visible = True

        # 预加载原始图片
        self.original_clock_images = [
            self._load_image('needle.png', int(size*0.83)),
            self._load_image('needle.png', int(size*0.83)),
            self._load_image('needle.png', int(size*0.83))
        ]
        self.original_sizes = [img.size for img in self.original_clock_images]

        cell_image = Image.open("circle.png").convert("RGBA")
        self.scale = 0.9

        self.pos_offset = self.size * self.scale / 2
        cell_image = cell_image.resize((int(self.size * self.scale), int(self.size * self.scale)),
                                       Image.Resampling.LANCZOS)
        self.cell_surface = pygame.image.fromstring(
            cell_image.tobytes(), cell_image.size, cell_image.mode
        )

    def update_center(self, x, y):
        self.center = (x, y)

    def set_nearest(self, nearest=False):
        self.nearest = nearest

    def set_visible(self, visible=True):
        self.visible = visible

    def _load_image(self, path, size):
        """使用PIL加载并缓存原始图片"""
        key = (path, size)
        if key not in self._image_cache:
            img = Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
            self._image_cache[key] = img
        return self._image_cache[key]

    def control(self, target_angles, target_length):
        """启动新的动画"""
        target_lengths = [max(0.00001, min(1.0, target_length)),] * 3

        self.start_time = pygame.time.get_ticks()
        self.start_angles = self.current_angles.copy()
        self.start_lengths = self.current_length.copy()
        self.target_angles = [a % 360 for a in target_angles]
        self.target_lengths = target_lengths

        if self not in Clock.active_animations:
            Clock.active_animations.append(self)

    def update(self):
        """更新当前动画状态"""
        if self not in self.active_animations:
            return

        elapsed = pygame.time.get_ticks() - self.start_time
        progress = min(elapsed / 1000, 1.0)  # 固定1000ms动画时长
        t = self._ease(progress)

        for i in range(3):
            # 计算角度插值
            start_a = self.start_angles[i]
            target_a = self.target_angles[i]

            if self.nearest:
                if target_a > start_a:
                    angle_delta = (target_a - start_a) % 360
                    self.current_angles[i] = (start_a + t * angle_delta) % 360
                else:
                    angle_delta = (start_a - target_a) % 360
                    self.current_angles[i] = (start_a - t * angle_delta) % 360
            else:
                angle_delta = (target_a - start_a) % 360
                self.current_angles[i] = (start_a + t * angle_delta) % 360

            # 计算长度插值
            start_l = self.start_lengths[i]
            target_l = self.target_lengths[i]
            self.current_length[i] = start_l + t * (target_l - start_l)

        if progress >= 1.0:
            self.active_animations.remove(self)

    def draw(self, surface):
        if not self.visible:
            return
        surface.blit(self.cell_surface, (self.center[0] - self.pos_offset, self.center[1] - self.pos_offset))
        """绘制三个指针到目标surface"""

        # # 假设三个指针的长度都一致
        # length = self.current_length[0]
        # for angle in set(self.current_angles):
        #     img = self._get_transformed_image(angle, length)
        #     rect = img.get_rect(center=self.center)
        #     surface.blit(img, rect)

        # 如果指针长度不一样，可以使用以下代码
        for i in range(3):
            angle = self.current_angles[i]
            length = self.current_length[i]
            if length < 0.01:
                continue
            img = self._get_transformed_image(angle, length)
            if img:
                rect = img.get_rect(center=self.center)
                surface.blit(img, rect)

    def _get_transformed_image(self, angle, length):
        """获取处理后的图像（带缓存）"""
        cache_key = (round(angle * 2), round(length, 2))

        if self._cache_enabled and cache_key in self._image_cache:
            return self._image_cache[cache_key]

        # 使用PIL进行图像变换
        original_img = self.original_clock_images[0]
        orig_w, orig_h = self.original_sizes[0]
        new_w = math.ceil(orig_w * length)

        # 缩放处理
        scaled_img = original_img.resize((new_w, orig_h), Image.Resampling.LANCZOS)

        # 旋转处理（角度转换为PIL坐标系）
        rotated_img = scaled_img.rotate(
            -angle + 90,  # 角度校准
            center=(new_w // 2, orig_h // 2),
            expand=True,
            resample=Image.BILINEAR
        )

        # 转换为Pygame Surface
        mode = rotated_img.mode
        size = rotated_img.size
        data = rotated_img.tobytes()
        pygame_surface = pygame.image.fromstring(data, size, mode)

        if self._cache_enabled:
            self._image_cache[cache_key] = pygame_surface
        return pygame_surface

    @staticmethod
    def _ease(t):
        """缓动函数"""
        # return t * t * (3 - 2 * t)
        return t


class CollisionEffect:
    def __init__(self, center_list, radius, width, height):
        self.initial_centers = [(x, y) for x, y in center_list]
        self.radius = radius
        self.width = width
        self.height = height
        self.positions = [[x, y] for x, y in center_list]
        self.velocities = []
        self.recovering = False
        self.recover_duration = 2000
        self.recover_start_time = 0
        self.start_recover_positions = []

    def random_collision(self):
        self.velocities = []
        for _ in range(len(self.positions)):
            speed = random.uniform(8, 12)
            angle = random.uniform(0, 2 * math.pi)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            self.velocities.append([vx, vy])
        self.recovering = False

    def start_recover(self):
        self.recovering = True
        self.start_time = pygame.time.get_ticks()  # 记录恢复开始时间
        self.start_positions = [p.copy() for p in self.positions]  # 记录初始位置

    def finish_recover(self):
        self.recovering = False

    def update(self):
        if self.recovering:
            # 添加位置插值动画
            t = min((pygame.time.get_ticks()-self.start_time)/self.recover_duration, 1.0)
            for i in range(len(self.positions)):
                dx = self.initial_centers[i][0] - self.start_positions[i][0]
                dy = self.initial_centers[i][1] - self.start_positions[i][1]
                self.positions[i][0] = self.start_positions[i][0] + dx * t
                self.positions[i][1] = self.start_positions[i][1] + dy * t
            if t >= 1.0:
                self.recovering = False
        else:
            # 更新位置
            for i in range(len(self.positions)):
                self.positions[i][0] += self.velocities[i][0]
                self.positions[i][1] += self.velocities[i][1]

            # 边界碰撞处理
            for i in range(len(self.positions)):
                pos = self.positions[i]
                vel = self.velocities[i]
                if pos[0] < self.radius:
                    vel[0] *= -1
                    pos[0] = self.radius
                elif pos[0] > self.width - self.radius:
                    vel[0] *= -1
                    pos[0] = self.width - self.radius
                if pos[1] < self.radius:
                    vel[1] *= -1
                    pos[1] = self.radius
                elif pos[1] > self.height - self.radius:
                    vel[1] *= -1
                    pos[1] = self.height - self.radius

            # 球体间碰撞处理
            for i in range(len(self.positions)):
                for j in range(i + 1, len(self.positions)):
                    dx = self.positions[j][0] - self.positions[i][0]
                    dy = self.positions[j][1] - self.positions[i][1]
                    distance = math.hypot(dx, dy)
                    if distance < 2 * self.radius and distance != 0:
                        # 计算碰撞响应
                        nx = dx / distance
                        ny = dy / distance
                        v1 = self.velocities[i]
                        v2 = self.velocities[j]

                        # 计算法向速度分量
                        v1n = np.dot(v1, [nx, ny])
                        v2n = np.dot(v2, [nx, ny])

                        # 交换法向速度（质量相同）
                        v1_new = [v1[0] + (v2n - v1n) * nx, v1[1] + (v2n - v1n) * ny]
                        v2_new = [v2[0] + (v1n - v2n) * nx, v2[1] + (v1n - v2n) * ny]

                        self.velocities[i] = v1_new
                        self.velocities[j] = v2_new

                        # 修正位置避免重叠
                        overlap = (2 * self.radius - distance) / 2
                        self.positions[i][0] -= overlap * nx
                        self.positions[i][1] -= overlap * ny
                        self.positions[j][0] += overlap * nx
                        self.positions[j][1] += overlap * ny


class Layout:
    def __init__(self, width, height, num_row, num_col):
        if num_col < 14:
            raise ValueError('Please use a large num_col >= 14')
        self.width = width
        self.height = height
        self.num_row = num_row
        self.num_col = num_col
        self.cell_width = width // num_col
        self.offset_x = (width - self.cell_width * num_col) // 2
        self.offset_y = (height - self.cell_width * num_row) // 2
        self.clock_list = []
        self._create_clocks()
        
        self.collision_clock_idx_list = []
        self.display_clock_idx_list = []
        self._init_clock_function()
        self.star_effect_init = [int(360 * random.uniform(0, 1)) for _ in self.collision_clock_idx_list]
        
        # 保存参与碰撞的球的初始位置
        self.initial_collision_centers = [self.clock_list[i].center for i in self.collision_clock_idx_list]
        
        self.collision_effect = None
        self.radius = self.cell_width * 0.4

    def enable_collision_effect(self, enable=True):
        if enable:
            # 使用保存的初始位置创建碰撞效果
            self.collision_effect = CollisionEffect(
                self.initial_collision_centers, 
                self.radius, 
                self.width, 
                self.height
            )
            self.collision_effect.random_collision()
        else:
            # 重置参与碰撞的球到初始位置
            for idx, center in zip(self.collision_clock_idx_list, self.initial_collision_centers):
                self.clock_list[idx].update_center(center[0], center[1])
            self.collision_effect = None


    def _init_clock_function(self):
        for row_idx in [0, 4]:
            self.collision_clock_idx_list.extend([row_idx + i * self.num_row for i in range(self.num_col)])
        for col_idx in range(self.num_col):
            for row_idx in [1, 2, 3]:
                self.display_clock_idx_list.append(row_idx + col_idx * self.num_row)

    def _create_clocks(self):
        for x in range(self.offset_x, self.num_col*self.cell_width+self.offset_x, self.cell_width):
            for y in range(self.offset_y, self.num_row*self.cell_width+self.offset_y, self.cell_width):
                self.clock_list.append(Clock(x, y, self.cell_width))

    def set_nearest(self, nearest=False):
        for clock in self.clock_list:
            clock.set_nearest(nearest)

    def update(self):
        for clock in self.clock_list:
            clock.update()
            
        if self.collision_effect:
            self.collision_effect.update()
            for idx, pos in zip(self.collision_clock_idx_list, self.collision_effect.positions):
                self.clock_list[idx].update_center(pos[0], pos[1])

    def draw(self, surface):
        for idx, clock in enumerate(self.clock_list):
            if self.collision_effect and idx in self.display_clock_idx_list:
                continue  # 碰撞效果时隐藏显示用指针
            clock.draw(surface)

    def display_long_line(self):
        for clock_idx in self.collision_clock_idx_list:
            self.clock_list[clock_idx].control([90, 90, 270], 1)
    
    def clear(self):
        for clock_idx in self.display_clock_idx_list:
            self.clock_list[clock_idx].control(config_dict['null'][0], config_dict['null'][1])

    def _group_clocks(self, clock_row, clock_col, str_):
        begin_idx = clock_col * self.num_row + clock_row
        if str_ in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                    'G', 'H', 'J', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', }:
            return [begin_idx, begin_idx + 1, begin_idx + 2, begin_idx + self.num_row, begin_idx + self.num_row + 1,
                    begin_idx + self.num_row + 2], (clock_row, clock_col+2)
        if str_ in {':', '/', '-', 'I', '_'}:
            return [begin_idx, begin_idx + 1, begin_idx + 2], (clock_row, clock_col+1)
        if str_ in {'M', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}:
            return [begin_idx, begin_idx + 1, begin_idx + 2, begin_idx + self.num_row, begin_idx + self.num_row + 1,
             begin_idx + self.num_row + 2, begin_idx + 2 * self.num_row, begin_idx + 2 * self.num_row + 1,
             begin_idx + 2 * self.num_row + 2], (clock_row, clock_col + 3)
        raise ValueError(f'character {str_} is not supported')

    def padding_str(self, input_):
        total_col_needed = 0
        for str_ in input_:
            if str_ in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                    'G', 'H', 'J', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', }:
                total_col_needed += 2
            elif str_ in {':', '/', '-', 'I', '_'}:
                total_col_needed += 1
            elif str_ in {'M', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}:
                total_col_needed += 3
            else:
                raise ValueError(f'character {str_} is not supported')
        padding = (self.num_col - total_col_needed) // 2
        return '_' * padding + input_ + '_' * padding


    def display_image(self, config_list, nearest=True):
        self.clear()
        for clock_idx, config in zip(self.display_clock_idx_list, config_list):
            self.clock_list[clock_idx].control(config[0], config[1])
            self.clock_list[clock_idx].set_nearest(nearest)

    def display_single_char_general(self, clock_row, clock_col, str_):
        clock_idx_list, (clock_row_next, clock_col_next) = self._group_clocks(clock_row, clock_col, str_)

        for clock_idx, config in zip(clock_idx_list, config_dict[str_]):
            self.clock_list[clock_idx].control(config[0], config[1])
        return clock_row_next, clock_col_next

    def display_string_general(self, chars, clock_row=1, clock_col=0):
        chars = self.padding_str(chars)
        for char_ in chars:
            clock_row, clock_col = self.display_single_char_general(clock_row, clock_col, char_)

    def display_star(self, reverse=False):
        for clock_idx, angle in zip(self.collision_clock_idx_list, self.star_effect_init):
            if reverse:
                self.clock_list[clock_idx].control([angle, angle+120, angle+240], 1)
            else:
                self.clock_list[clock_idx].control([angle+180, angle+120+180, angle+240+180], 1)


class EffectManager:
    def __init__(self, layout):
        self.layout = layout
        self.states = ['sentence', 'time', 'date', 'dog', 'collision']
        self.sentence = 'Welcome'   # Sentence to display, you can modify this
        self.words_list = self.sentence.upper().split(' ')
        self.word_index = 0
        self.current_state = -1
        self.dog_type = True
        self.state_start_time = 0
        self.durations = [len(self.words_list)*3000+1000, 18000, 5000, 8000, 22000] # display durations (in ms) for each type
        self.layout.display_long_line()
        self._star_flag = True
        self.last_switch_time_list = [0] * len(self.states)
        self.time_switch_interval_list = [3000, 2000, 2000, 2000, 1000]


    def next_state(self):
        self.current_state = (self.current_state + 1) % len(self.states)
        self.state_start_time = pygame.time.get_ticks()
        
        if self.states[self.current_state] == 'collision':
            self.layout.enable_collision_effect(True)
        else:
            self.layout.enable_collision_effect(False)

    def update(self):
        elapsed = pygame.time.get_ticks() - self.state_start_time
        if elapsed > self.durations[self.current_state]:
            if self.states[self.current_state] == 'collision':
                self.layout.collision_effect.finish_recover()
                self.layout.display_long_line()
            elif self.states[self.current_state] == 'sentence':
                self.word_index = 0
            elif self.states[self.current_state] == 'dog':
                self.layout.set_nearest(False)
            self.next_state()
            
        # 根据状态调用对应的显示方法
        if self.states[self.current_state] == 'sentence':
            self.show_sentence()
        elif self.states[self.current_state] == 'time':
            self.show_time()
        elif self.states[self.current_state] == 'date':
            self.show_date()
        elif self.states[self.current_state] == 'dog':
            self.show_dog()
        elif self.states[self.current_state] == 'collision':
            self.show_collision()

    def show_collision(self):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.state_start_time
        if elapsed >= (self.durations[self.current_state] - self.layout.collision_effect.recover_duration) and not self.layout.collision_effect.recovering:
            self.layout.collision_effect.start_recover()
            self.layout.display_long_line()
        else:
            if current_time - self.last_switch_time_list[self.current_state] >= self.time_switch_interval_list[self.current_state]:
                self.layout.display_star(self._star_flag)
                self._star_flag = not self._star_flag
                self.last_switch_time_list[self.current_state] = current_time

    def show_time(self):
        """时间更新"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_switch_time_list[self.current_state] >= self.time_switch_interval_list[self.current_state]:
            time_string = datetime.now().strftime('%H:%M:%S')
            self.layout.display_string_general(time_string)
            self.last_switch_time_list[self.current_state] = current_time

    def show_date(self):
        """日期更新"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_switch_time_list[self.current_state] >= self.time_switch_interval_list[self.current_state]:
            date_string = datetime.now().strftime('%y-%m-%d')
            self.layout.display_string_general(date_string)
            self.last_switch_time_list[self.current_state] = current_time

    def show_sentence(self):
        current_time = pygame.time.get_ticks()
        word = self.words_list[self.word_index % len(self.words_list)]
        if current_time - self.last_switch_time_list[self.current_state] >= self.time_switch_interval_list[self.current_state]:
            self.layout.clear()
            self.layout.display_string_general(word)
            self.word_index += 1
            self.last_switch_time_list[self.current_state] = current_time

    def show_dog(self):
        """定时切换小狗动画"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_switch_time_list[self.current_state] >= self.time_switch_interval_list[self.current_state]:
            config = 'dog_1' if self.dog_type else 'dog_2'
            self.layout.display_image(config_dict['_'] * 2 + config_dict[config], nearest=True)
            self.dog_type = not self.dog_type
            self.last_switch_time_list[self.current_state] = current_time

# 主程序
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
width, height = screen.get_size()

# 加载背景
bg_image = Image.open("fill.png").convert("RGBA")
bg_image = bg_image.resize((width, height), Image.Resampling.LANCZOS)
bg_surface = pygame.image.fromstring(bg_image.tobytes(), bg_image.size, bg_image.mode)

layout = Layout(width, height, 5, 16)
effect_manager = EffectManager(layout)
effect_manager.next_state()  # 初始化第一个状态

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    effect_manager.update()
    layout.update()
    
    screen.blit(bg_surface, (0, 0))
    layout.draw(screen)
    pygame.display.flip()
    clock.tick(30)

pygame.quit()