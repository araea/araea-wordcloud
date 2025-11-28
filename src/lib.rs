/*!
 * Araea WordCloud Library
 *
 * 一个纯 Rust 实现的词云可视化库。
 */

use fontdue::{Font, FontSettings};
use image::GenericImageView;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use thiserror::Error;
use tiny_skia::{Pixmap, Transform};

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, Error)]
pub enum Error {
    #[error("Font error: {0}")]
    Font(String),
    #[error("Image error: {0}")]
    Image(String),
    #[error("SVG error: {0}")]
    Svg(String),
    #[error("Render error: {0}")]
    Render(String),
    #[error("Invalid input: {0}")]
    Input(String),
}

// =============================================================================
// Public Data Types
// =============================================================================

/// 单词输入项
#[derive(Debug, Clone)]
pub struct WordInput {
    pub text: String,
    pub weight: f32,
}

impl WordInput {
    pub fn new(text: impl Into<String>, weight: f32) -> Self {
        Self {
            text: text.into(),
            weight: weight.max(0.0),
        }
    }
}

/// 已布局的单词
#[derive(Debug, Clone)]
pub struct PlacedWord {
    pub text: String,
    pub font_size: f32,
    pub x: f32,
    pub y: f32,
    pub rotation: f32,
    pub color: String,
}

/// 预设配色方案
#[derive(Debug, Clone, Copy, Default)]
pub enum ColorScheme {
    #[default]
    Ocean,
    Sunset,
    Forest,
    Berry,
    Monochrome,
    Rainbow,
}

impl ColorScheme {
    pub fn colors(&self) -> Vec<&'static str> {
        match self {
            ColorScheme::Ocean => vec!["#264653", "#287271", "#2a9d8f", "#8ab17d", "#e9c46a"],
            ColorScheme::Sunset => vec!["#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f"],
            ColorScheme::Forest => vec!["#2d6a4f", "#40916c", "#52b788", "#74c69d", "#95d5b2"],
            ColorScheme::Berry => vec!["#7b2cbf", "#9d4edd", "#c77dff", "#e0aaff", "#ff6d00"],
            ColorScheme::Monochrome => vec!["#212529", "#495057", "#6c757d", "#adb5bd", "#ced4da"],
            ColorScheme::Rainbow => {
                vec![
                    "#e63946", "#f4a261", "#e9c46a", "#2a9d8f", "#457b9d", "#7b2cbf",
                ]
            }
        }
    }
}

// =============================================================================
// Preset Masks
// =============================================================================

/// 内置蒙版形状
#[derive(Debug, Clone, Copy, Default)]
pub enum MaskShape {
    #[default]
    Circle,
    Cloud,
    Heart,
    Skull,
    Star,
    Triangle,
}

impl MaskShape {
    /// 获取蒙版的 SVG 字节数据
    pub fn bytes(&self) -> &'static [u8] {
        match self {
            MaskShape::Circle => include_bytes!("../assets/circle.svg"),
            MaskShape::Cloud => include_bytes!("../assets/cloud.svg"),
            MaskShape::Heart => include_bytes!("../assets/heart.svg"),
            MaskShape::Skull => include_bytes!("../assets/skull.svg"),
            MaskShape::Star => include_bytes!("../assets/star.svg"),
            MaskShape::Triangle => include_bytes!("../assets/triangle.svg"),
        }
    }
}

// =============================================================================
// Font Info
// =============================================================================

struct FontInfo {
    data: Vec<u8>,
    family_name: String,
}

fn extract_font_family_name(font_data: &[u8]) -> Option<String> {
    let mut db = usvg::fontdb::Database::new();
    db.load_font_source(usvg::fontdb::Source::Binary(Arc::new(font_data.to_vec())));
    for face in db.faces() {
        if let Some((name, _)) = face.families.first() {
            return Some(name.clone());
        }
    }
    None
}

// =============================================================================
// Builder
// =============================================================================

pub struct WordCloudBuilder {
    width: u32,
    height: u32,
    background: String,
    colors: Vec<String>,
    font_data: Option<Vec<u8>>,
    mask_data: Option<Vec<u8>>,
    padding: u32,
    min_font_size: f32,
    max_font_size: f32,
    angles: Vec<f32>,
    seed: Option<u64>,
    word_spacing: f32,
}

impl Default for WordCloudBuilder {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            background: "#FFFFFF".into(),
            colors: ColorScheme::Ocean
                .colors()
                .into_iter()
                .map(String::from)
                .collect(),
            font_data: None,
            mask_data: None,
            padding: 2,
            min_font_size: 14.0,
            max_font_size: 120.0,
            angles: vec![0.0],
            seed: None,
            word_spacing: 4.0,
        }
    }
}

impl WordCloudBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width.max(100);
        self.height = height.max(100);
        self
    }

    pub fn background(mut self, color: impl Into<String>) -> Self {
        self.background = color.into();
        self
    }

    pub fn color_scheme(mut self, scheme: ColorScheme) -> Self {
        self.colors = scheme.colors().into_iter().map(String::from).collect();
        self
    }

    pub fn colors(mut self, colors: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.colors = colors.into_iter().map(|c| c.into()).collect();
        if self.colors.is_empty() {
            self.colors = ColorScheme::Ocean
                .colors()
                .into_iter()
                .map(String::from)
                .collect();
        }
        self
    }

    pub fn font(mut self, font_data: Vec<u8>) -> Self {
        self.font_data = Some(font_data);
        self
    }

    pub fn mask(mut self, image_data: Vec<u8>) -> Self {
        self.mask_data = Some(image_data);
        self
    }

    pub fn mask_preset(mut self, shape: MaskShape) -> Self {
        self.mask_data = Some(shape.bytes().to_vec());
        self
    }

    pub fn padding(mut self, padding: u32) -> Self {
        self.padding = padding;
        self
    }

    pub fn font_size_range(mut self, min: f32, max: f32) -> Self {
        self.min_font_size = min.max(8.0);
        self.max_font_size = max.max(self.min_font_size);
        self
    }

    pub fn angles(mut self, angles: Vec<f32>) -> Self {
        self.angles = if angles.is_empty() { vec![0.0] } else { angles };
        self
    }

    pub fn word_spacing(mut self, spacing: f32) -> Self {
        self.word_spacing = spacing.max(0.0);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn build(self, words: &[WordInput]) -> Result<WordCloud, Error> {
        if words.is_empty() {
            return Err(Error::Input("Word list cannot be empty".into()));
        }

        // 验证并过滤单词
        let valid_words: Vec<_> = words
            .iter()
            .filter(|w| !w.text.trim().is_empty() && w.weight > 0.0)
            .cloned()
            .collect();

        if valid_words.is_empty() {
            return Err(Error::Input("No valid words provided".into()));
        }

        // 加载字体
        let font_info = self.load_font()?;
        let font = Font::from_bytes(font_info.data.as_slice(), FontSettings::default())
            .map_err(|e| Error::Font(e.to_string()))?;

        // 初始化碰撞图
        let mut collision_map = CollisionMap::new(self.width, self.height);

        // 应用蒙版
        if let Some(mask_bytes) = &self.mask_data {
            self.apply_mask(&mut collision_map, mask_bytes)?;
        }

        // 初始化随机数
        let mut rng = match self.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_os_rng(),
        };

        // 排序单词（权重从大到小）
        let mut sorted_words = valid_words;
        sorted_words.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());

        // 计算权重归一化因子
        let max_weight = sorted_words.first().map(|w| w.weight).unwrap_or(1.0);
        let min_weight = sorted_words.last().map(|w| w.weight).unwrap_or(1.0);
        let weight_range = max_weight - min_weight;

        let mut placed_words = Vec::with_capacity(sorted_words.len());
        let effective_padding = self.padding + (self.word_spacing / 2.0) as u32;

        for word in &sorted_words {
            let normalized = if weight_range > 0.0 {
                (word.weight - min_weight) / weight_range
            } else {
                1.0
            };

            // 简单的线性插值计算字体大小
            let font_size =
                self.min_font_size + normalized * (self.max_font_size - self.min_font_size);

            let angle = self.angles[rng.random_range(0..self.angles.len())];

            // 尝试放置
            if let Some(pos) = self.try_place_word(
                &word.text,
                font_size,
                angle,
                &font,
                &mut collision_map,
                effective_padding,
                &mut rng,
            ) {
                let color = self.colors[rng.random_range(0..self.colors.len())].clone();
                placed_words.push(PlacedWord {
                    text: word.text.clone(),
                    font_size,
                    x: pos.0,
                    y: pos.1,
                    rotation: angle,
                    color,
                });
            }
        }

        if placed_words.is_empty() {
            return Err(Error::Render("Could not place any words".into()));
        }

        Ok(WordCloud {
            width: self.width,
            height: self.height,
            background: self.background,
            words: placed_words,
            font_data: font_info.data,
            font_family: font_info.family_name,
        })
    }

    fn load_font(&self) -> Result<FontInfo, Error> {
        let data = match &self.font_data {
            Some(d) => d.clone(),
            None => include_bytes!("../assets/HarmonyOS_Sans_SC_Bold.ttf").to_vec(),
        };

        let family_name =
            extract_font_family_name(&data).unwrap_or_else(|| "HarmonyOS Sans SC".to_string());

        Font::from_bytes(data.as_slice(), FontSettings::default())
            .map_err(|e| Error::Font(e.to_string()))?;

        Ok(FontInfo { data, family_name })
    }

    /// 应用蒙版：支持 SVG 和光栅图片
    fn apply_mask(&self, collision_map: &mut CollisionMap, mask_bytes: &[u8]) -> Result<(), Error> {
        // 内部闭包：处理像素数据并写入 Grid
        let mut apply_pixels =
            |width: u32, height: u32, get_pixel: &dyn Fn(u32, u32) -> Option<(u8, u8, u8, u8)>| {
                for y in 0..height {
                    for x in 0..width {
                        if let Some((r, g, b, a)) = get_pixel(x, y) {
                            // 1. alpha < 128 (透明) -> 视为占用 (Grid 设为 1)
                            // 2. r + g + b >= 750 (接近白色) -> 视为占用 (Grid 设为 1)
                            // 注意：CollisionMap 中 1 表示"不可放置"，0 表示"可放置"
                            // 通常蒙版图中，白色背景是不可放置区域，黑色形状是可放置区域

                            let sum = r as u16 + g as u16 + b as u16;
                            let is_blocked = a < 128 || sum >= 750;

                            if is_blocked {
                                collision_map.set(x as i32, y as i32);
                            }
                        }
                    }
                }
            };

        // 尝试 1: 解析 SVG
        let opt = usvg::Options::default();
        if let Ok(tree) = usvg::Tree::from_data(mask_bytes, &opt) {
            let size = tree.size().to_int_size();
            let scale_x = self.width as f32 / size.width() as f32;
            let scale_y = self.height as f32 / size.height() as f32;

            let mut pixmap = Pixmap::new(self.width, self.height)
                .ok_or(Error::Render("Failed to create mask buffer".into()))?;

            // 填充白色背景（防止 SVG 透明部分被误判）
            pixmap.fill(tiny_skia::Color::WHITE);

            let transform = Transform::from_scale(scale_x, scale_y);
            resvg::render(&tree, transform, &mut pixmap.as_mut());

            apply_pixels(self.width, self.height, &|x, y| {
                pixmap
                    .pixel(x, y)
                    .map(|p| (p.red(), p.green(), p.blue(), p.alpha()))
            });
            return Ok(());
        }

        // 尝试 2: 解析光栅图片 (PNG, JPG)
        if let Ok(img) = image::load_from_memory(mask_bytes) {
            let resized = img.resize_exact(
                self.width,
                self.height,
                image::imageops::FilterType::Nearest,
            );

            apply_pixels(self.width, self.height, &|x, y| {
                if x < resized.width() && y < resized.height() {
                    let p = resized.get_pixel(x, y);
                    Some((p[0], p[1], p[2], p[3]))
                } else {
                    None
                }
            });
            return Ok(());
        }

        Err(Error::Image(
            "The mask format could not be determined".into(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_place_word(
        &self,
        text: &str,
        font_size: f32,
        angle: f32,
        font: &Font,
        map: &mut CollisionMap,
        padding: u32,
        rng: &mut ChaCha8Rng,
    ) -> Option<(f32, f32)> {
        // 生成单词的位图 Sprite
        let sprite = rasterize_text(text, font_size, angle, font, padding);

        // 中心点
        let start_x = map.width as i32 / 2;
        let start_y = map.height as i32 / 2;

        // 随机初始方向
        let dt = if rng.random_bool(0.5) { 1 } else { -1 };

        // 初始化螺旋迭代器
        let spiral = ArchimedeanSpiral::new(map.width as i32, map.height as i32, dt);
        let max_iter = 10000;

        for (dx, dy) in spiral.take(max_iter) {
            // 计算左上角坐标 (TS逻辑: current_x 是 sprite 左上角)
            let current_x = start_x + dx - (sprite.bbox_width as i32 / 2);
            let current_y = start_y + dy - (sprite.bbox_height as i32 / 2);

            // 检查是否碰撞
            if !map.check_collision(&sprite, current_x, current_y) {
                // 写入 Grid
                map.write_sprite(&sprite, current_x, current_y);

                // 返回中心点坐标 (用于 SVG 渲染)
                return Some((
                    current_x as f32 + sprite.anchor_x,
                    current_y as f32 + sprite.anchor_y,
                ));
            }
        }

        None
    }
}

// =============================================================================
// Collision Detection (Optimized)
// =============================================================================

struct CollisionMap {
    width: u32,
    height: u32,
    stride: usize, // 每行有多少个 u32
    data: Vec<u32>,
}

impl CollisionMap {
    fn new(width: u32, height: u32) -> Self {
        // TS逻辑: width >> 5
        let stride = ((width + 31) >> 5) as usize;
        Self {
            width,
            height,
            stride,
            data: vec![0; stride * height as usize],
        }
    }

    /// 设置某个点为占用 (用于 Mask 初始化)
    fn set(&mut self, x: i32, y: i32) {
        if x >= 0 && y >= 0 && x < self.width as i32 && y < self.height as i32 {
            let row_idx = y as usize * self.stride;
            let col_idx = (x as usize) >> 5;
            let bit_idx = 31 - (x & 31);
            self.data[row_idx + col_idx] |= 1 << bit_idx;
        }
    }

    /// 高效碰撞检测 (位运算优化，对齐 TS checkCollision)
    fn check_collision(&self, sprite: &TextSprite, start_x: i32, start_y: i32) -> bool {
        let sprite_w32 = sprite.width_u32;
        let sprite_h = sprite.bbox_height;

        // 计算 X 轴上的位移
        // TS logic: shift = startX & 31
        let shift = (start_x & 31).unsigned_abs(); // abs purely for safety, logical & handles neg
        let r_shift = 32 - shift;

        // 计算 grid 中的起始索引
        // 注意：这里需要处理 start_x 为负数的情况，以及裁剪

        for sy in 0..sprite_h {
            let gy = start_y + sy as i32;

            // Y 轴越界检查
            if gy < 0 || gy >= self.height as i32 {
                return true; // 越界视为碰撞
            }

            let grid_row_idx = gy as usize * self.stride;

            // X 轴起始 Block 索引
            let grid_col_start = (start_x >> 5) as isize;

            let mut carry = 0u32;

            for sx in 0..=sprite_w32 {
                // 获取 Sprite 当前块的数据
                let s_val = if sx < sprite_w32 {
                    sprite.data[sy as usize * sprite_w32 + sx]
                } else {
                    0
                };

                // 构造 Mask: 上一块的剩余部分 | 当前块的移位部分
                // TS: (carry << rShift) | (sVal >>> shift)
                // Rust u32 >> is logical shift (zero-fill), matching JS >>>
                // 需要注意 shift == 0 的情况，Rust shift overflow 会 panic
                let mask = if shift == 0 {
                    s_val
                } else {
                    (carry << r_shift) | (s_val >> shift)
                };

                // 计算当前 grid 的实际列索引
                let gx = grid_col_start + sx as isize;

                // 只有 mask 不为 0 时才检查，节省性能
                if mask != 0 {
                    // X 轴越界检查
                    if gx < 0 || gx >= self.stride as isize {
                        return true;
                    }

                    if (self.data[grid_row_idx + gx as usize] & mask) != 0 {
                        return true;
                    }
                }

                carry = s_val;
            }
        }
        false
    }

    /// 将 Sprite 写入 Grid (位运算优化)
    fn write_sprite(&mut self, sprite: &TextSprite, start_x: i32, start_y: i32) {
        let sprite_w32 = sprite.width_u32;
        let sprite_h = sprite.bbox_height;
        let shift = (start_x & 31).unsigned_abs();
        let r_shift = 32 - shift;

        for sy in 0..sprite_h {
            let gy = start_y + sy as i32;
            if gy < 0 || gy >= self.height as i32 {
                continue;
            }

            let grid_row_idx = gy as usize * self.stride;
            let grid_col_start = (start_x >> 5) as isize;
            let mut carry = 0u32;

            for sx in 0..=sprite_w32 {
                let s_val = if sx < sprite_w32 {
                    sprite.data[sy as usize * sprite_w32 + sx]
                } else {
                    0
                };

                let mask = if shift == 0 {
                    s_val
                } else {
                    (carry << r_shift) | (s_val >> shift)
                };

                let gx = grid_col_start + sx as isize;
                if mask != 0 && gx >= 0 && gx < self.stride as isize {
                    self.data[grid_row_idx + gx as usize] |= mask;
                }

                carry = s_val;
            }
        }
    }
}

struct TextSprite {
    data: Vec<u32>,   // 扁平化的位图数据
    width_u32: usize, // 每行有多少个 u32
    bbox_width: u32,
    bbox_height: u32,
    anchor_x: f32, // 用于定位回 SVG
    anchor_y: f32,
}

fn rasterize_text(text: &str, size: f32, angle_deg: f32, font: &Font, padding: u32) -> TextSprite {
    // 1. 光栅化字符
    let metrics = font
        .horizontal_line_metrics(size)
        .unwrap_or(fontdue::LineMetrics {
            ascent: size * 0.8,
            descent: size * -0.2,
            line_gap: 0.0,
            new_line_size: size,
        });

    let mut glyphs = Vec::new();
    let mut total_width = 0.0f32;

    for ch in text.chars() {
        let (glyph_metrics, bitmap) = font.rasterize(ch, size);
        glyphs.push((total_width, glyph_metrics, bitmap));
        total_width += glyph_metrics.advance_width;
    }

    // 2. 计算变换参数
    let padding_f = padding as f32;
    // 原始包围盒大小
    let unrotated_w = total_width.ceil() + padding_f * 2.0;
    let unrotated_h = metrics.new_line_size.ceil() + padding_f * 2.0;

    // 旋转中心
    let cx = unrotated_w / 2.0;
    let cy = unrotated_h / 2.0;

    let rad = angle_deg.to_radians();
    let (sin, cos) = rad.sin_cos();

    // 变换函数
    let transform = |x: f32, y: f32| -> (f32, f32) {
        let dx = x - cx;
        let dy = y - cy;
        (dx * cos - dy * sin + cx, dx * sin + dy * cos + cy)
    };

    // 3. 计算旋转后的新包围盒
    let corners = [
        transform(0.0, 0.0),
        transform(unrotated_w, 0.0),
        transform(0.0, unrotated_h),
        transform(unrotated_w, unrotated_h),
    ];

    let min_x = corners.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let max_x = corners
        .iter()
        .map(|p| p.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = corners.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let max_y = corners
        .iter()
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max);

    let bbox_width = (max_x - min_x).ceil() as u32;
    let bbox_height = (max_y - min_y).ceil() as u32;

    // Sprite 在 Grid 中的 u32 宽度
    let width_u32 = ((bbox_width + 31) >> 5) as usize;

    // 4. 生成位图数据 (Vec<u32>)
    let mut data = vec![0u32; width_u32 * bbox_height as usize];

    let base_x = padding_f;
    let base_y = padding_f + metrics.ascent;

    // 锚点偏移量: 旋转后左上角 相对于 原始左上角 的偏移
    // 我们将把 sprite 放在 (current_x, current_y)，那么文字的实际位置就是:
    // x = current_x - min_x
    // y = current_y - min_y
    // 但还要加上 base_x/base_y 的基线调整...
    // 简化处理：anchor 保存旋转中心相对于旋转后bbox左上角的偏移
    let (rot_base_x, rot_base_y) = transform(base_x, base_y);
    // 这里 anchor 计算需要非常小心，用于 SVG 最后的定位
    // 平移到中心旋转。
    // 我们记录旋转后的基准点相对于 min_x/min_y 的位置
    let anchor_x = rot_base_x - min_x;
    let anchor_y = rot_base_y - min_y;

    for (offset_x, glyph_metrics, bitmap) in &glyphs {
        let char_left = base_x + offset_x + glyph_metrics.xmin as f32;
        let char_top = base_y - glyph_metrics.height as f32 - glyph_metrics.ymin as f32;

        for y in 0..glyph_metrics.height {
            for x in 0..glyph_metrics.width {
                if bitmap[y * glyph_metrics.width + x] > 10 {
                    // Threshold
                    let ox = char_left + x as f32;
                    let oy = char_top + y as f32;
                    let (rx, ry) = transform(ox, oy);

                    let fx = (rx - min_x).round() as i32;
                    let fy = (ry - min_y).round() as i32;

                    // 膨胀 padding
                    let pad = padding as i32;
                    for py in -pad..=pad {
                        for px in -pad..=pad {
                            let dx = fx + px;
                            let dy = fy + py;

                            if dx >= 0
                                && dy >= 0
                                && dx < bbox_width as i32
                                && dy < bbox_height as i32
                            {
                                let row_idx = dy as usize * width_u32;
                                let col_idx = (dx as usize) >> 5;
                                let bit_idx = 31 - (dx & 31);
                                data[row_idx + col_idx] |= 1 << bit_idx;
                            }
                        }
                    }
                }
            }
        }
    }

    TextSprite {
        data,
        width_u32,
        bbox_width,
        bbox_height,
        anchor_x,
        anchor_y,
    }
}

// =============================================================================
// Helper: Archimedean Spiral
// =============================================================================

struct ArchimedeanSpiral {
    t: i32,
    dt: i32,
    dx: f64,
    dy: f64,
    ratio: f64,
    e: f64,
}

impl ArchimedeanSpiral {
    fn new(width: i32, height: i32, dt: i32) -> Self {
        // TS parameters: e = 4
        let e = 4.0;
        let ratio = e * width as f64 / height as f64;
        Self {
            t: 0,
            dt,
            dx: 0.0,
            dy: 0.0,
            ratio,
            e,
        }
    }
}

impl Iterator for ArchimedeanSpiral {
    type Item = (i32, i32);

    fn next(&mut self) -> Option<Self::Item> {
        self.t += self.dt;
        let sign = if self.t < 0 { -1.0 } else { 1.0 };
        // TS Spiral logic approximation
        let idx = ((1.0 + 4.0 * sign * self.t as f64).sqrt() - sign) as i32 & 3;
        match idx {
            0 => self.dx += self.ratio,
            1 => self.dy += self.e,
            2 => self.dx -= self.ratio,
            _ => self.dy -= self.e,
        }
        Some((self.dx as i32, self.dy as i32))
    }
}

// =============================================================================
// Output Generation
// =============================================================================

pub struct WordCloud {
    pub width: u32,
    pub height: u32,
    pub background: String,
    pub words: Vec<PlacedWord>,
    font_data: Vec<u8>,
    font_family: String,
}

impl WordCloud {
    pub fn to_svg(&self) -> String {
        let mut svg = String::with_capacity(8192);

        svg.push_str(&format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.width, self.height, self.width, self.height
        ));

        svg.push_str(&format!(
            r#"<rect width="100%" height="100%" fill="{}"/>"#,
            self.background
        ));

        svg.push_str(&format!(
            r#"<style>text{{font-family:'{}',Arial,sans-serif}}</style>"#,
            escape_xml(&self.font_family)
        ));

        for word in &self.words {
            // 计算旋转中心
            svg.push_str(&format!(
                r#"<text x="{:.1}" y="{:.1}" fill="{}" font-size="{:.1}" transform="rotate({:.1} {:.1} {:.1})">{}</text>"#,
                word.x,
                word.y,
                word.color,
                word.font_size,
                word.rotation,
                word.x,
                word.y,
                escape_xml(&word.text)
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    pub fn to_png(&self, scale: f32) -> Result<Vec<u8>, Error> {
        let svg_content = self.to_svg();
        let mut fontdb = usvg::fontdb::Database::new();
        // fontdb.load_system_fonts();
        fontdb.load_font_source(usvg::fontdb::Source::Binary(Arc::new(
            self.font_data.clone(),
        )));

        println!("=== 正在渲染 SVG，请求的字体名: '{}' ===", self.font_family);
        println!("=== fontdb 中已加载的字体列表: ===");
        for face in fontdb.faces() {
            println!(
                "  Family: {:?}, Weight: {:?}, Style: {:?}, Stretch: {:?}",
                face.families, face.weight, face.style, face.stretch
            );
        }

        let options = usvg::Options {
            font_family: self.font_family.clone(),
            fontdb: Arc::new(fontdb),
            ..Default::default()
        };

        let tree =
            usvg::Tree::from_str(&svg_content, &options).map_err(|e| Error::Svg(e.to_string()))?;
        let size = tree.size().to_int_size();
        let out_width = (size.width() as f32 * scale).max(1.0) as u32;
        let out_height = (size.height() as f32 * scale).max(1.0) as u32;

        let mut pixmap = Pixmap::new(out_width, out_height)
            .ok_or_else(|| Error::Render("Failed to create pixel buffer".into()))?;

        if let Some(color) = parse_hex_color(&self.background) {
            pixmap.fill(color);
        }

        let transform = Transform::from_scale(scale, scale);
        resvg::render(&tree, transform, &mut pixmap.as_mut());

        pixmap
            .encode_png()
            .map_err(|e| Error::Render(e.to_string()))
    }
}

fn parse_hex_color(hex: &str) -> Option<tiny_skia::Color> {
    let hex = hex.trim_start_matches('#');
    if hex.len() == 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        Some(tiny_skia::Color::from_rgba8(r, g, b, 255))
    } else {
        None
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

pub fn generate(words: &[(&str, f32)]) -> Result<WordCloud, Error> {
    let inputs: Vec<WordInput> = words
        .iter()
        .map(|(text, weight)| WordInput::new(*text, *weight))
        .collect();

    WordCloudBuilder::new().build(&inputs)
}
