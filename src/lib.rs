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
    /// 是否为竖排正写
    pub is_vertical: bool,
}

/// 预设配色方案
#[derive(Debug, Clone, Copy, Default)]
pub enum ColorScheme {
    #[default]
    Default,
    Contrasting1,
    Blue,
    Green,
    Cold1,
    Black,
    White,
}

impl ColorScheme {
    pub fn colors(&self) -> Vec<&'static str> {
        match self {
            ColorScheme::Default => vec!["#0b100c", "#bb0119", "#c7804b", "#bca692", "#1c4e17"],
            ColorScheme::Contrasting1 => {
                vec!["#e76f3d", "#feab6b", "#f3e9e7", "#9bcfe0", "#00a7c7"]
            }
            ColorScheme::Blue => vec!["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
            ColorScheme::Green => vec!["#386641", "#6a994e", "#a7c957", "#f2e8cf", "#bc4749"],
            ColorScheme::Cold1 => vec!["#252b31", "#5e6668", "#c1c8c7", "#f6fafb", "#d49c6b"],
            ColorScheme::Black => vec!["#000000"],
            ColorScheme::White => vec!["#ffffff"],
        }
    }

    pub fn background_color(&self) -> &'static str {
        match self {
            ColorScheme::Default => "#ffffff",
            ColorScheme::Contrasting1 => "#000000",
            ColorScheme::Blue => "#ffffff",
            ColorScheme::Green => "#ffffff",
            ColorScheme::Cold1 => "#000000",
            ColorScheme::Black => "#ffffff",
            ColorScheme::White => "#000000",
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
    vertical_writing: bool,
}

impl Default for WordCloudBuilder {
    fn default() -> Self {
        // 配置与 JS 默认值一致
        let scheme = ColorScheme::Default;
        Self {
            width: 800,
            height: 600,
            background: scheme.background_color().into(),
            colors: scheme.colors().into_iter().map(String::from).collect(),
            font_data: None,
            mask_data: None,
            padding: 5,
            min_font_size: 10.0,
            max_font_size: 100.0,
            angles: vec![0.0],
            seed: None,
            vertical_writing: false,
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
        self.background = scheme.background_color().into();
        self
    }

    pub fn colors(mut self, colors: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.colors = colors.into_iter().map(|c| c.into()).collect();
        if self.colors.is_empty() {
            self.colors = ColorScheme::Default
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
        self.min_font_size = min.max(4.0);
        self.max_font_size = max.max(self.min_font_size);
        self
    }

    pub fn angles(mut self, angles: Vec<f32>) -> Self {
        self.angles = if angles.is_empty() { vec![0.0] } else { angles };
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// 是否开启竖排正写（将 90 度旋转的词改为文字直立但竖向排列）
    pub fn vertical_writing(mut self, enable: bool) -> Self {
        self.vertical_writing = enable;
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

        for word in &sorted_words {
            let normalized = if weight_range > 0.0 {
                (word.weight - min_weight) / weight_range
            } else {
                1.0
            };

            // 线性插值计算字体大小
            let font_size =
                self.min_font_size + normalized * (self.max_font_size - self.min_font_size);

            let angle = self.angles[rng.random_range(0..self.angles.len())];

            // 尝试放置
            if let Some((pos, placed_angle, is_vertical)) = self.try_place_word(
                &word.text,
                font_size,
                angle,
                &font,
                &mut collision_map,
                self.padding,
                &mut rng,
            ) {
                let color = self.colors[rng.random_range(0..self.colors.len())].clone();
                placed_words.push(PlacedWord {
                    text: word.text.clone(),
                    font_size,
                    x: pos.0,
                    y: pos.1,
                    rotation: placed_angle,
                    color,
                    is_vertical,
                });
            }
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
        let mut apply_pixels =
            |width: u32, height: u32, get_pixel: &dyn Fn(u32, u32) -> Option<(u8, u8, u8, u8)>| {
                for y in 0..height {
                    for x in 0..width {
                        if let Some((r, g, b, a)) = get_pixel(x, y) {
                            let sum = r as u16 + g as u16 + b as u16;
                            // Alpha < 128 is effectively transparent -> blocked
                            // RGB sum >= 750 (near white) -> blocked
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
    ) -> Option<((f32, f32), f32, bool)> {
        // 判断是否触发竖排正写逻辑：开启了选项，且角度接近 90 或 -90 度
        let is_vertical = self.vertical_writing && (angle.abs() - 90.0).abs() < 1.0;

        let sprite = rasterize_text(text, font_size, angle, font, padding, is_vertical);

        if sprite.bbox_width == 0 || sprite.bbox_height == 0 {
            return None;
        }

        let start_x = map.width as i32 / 2;
        let start_y = map.height as i32 / 2;

        let dt = if rng.random_bool(0.5) { 1 } else { -1 };

        // 螺旋迭代
        let spiral = ArchimedeanSpiral::new(map.width as i32, map.height as i32, dt);
        let max_iter = (map.width * map.height) as usize / 2;

        for (dx, dy) in spiral.take(max_iter) {
            // 计算左上角坐标
            let current_x = start_x + dx - (sprite.bbox_width as i32 / 2);
            let current_y = start_y + dy - (sprite.bbox_height as i32 / 2);

            // 检查碰撞
            if !map.check_collision(&sprite, current_x, current_y) {
                // 写入 Grid
                map.write_sprite(&sprite, current_x, current_y);

                // 返回中心点坐标 (用于 SVG text-anchor="middle" 的渲染)
                // 如果是竖排，angle 改为 0，因为字体不再旋转，而是排版旋转
                let final_angle = if is_vertical { 0.0 } else { angle };

                return Some((
                    (
                        current_x as f32 + sprite.text_center_x,
                        current_y as f32 + sprite.text_center_y,
                    ),
                    final_angle,
                    is_vertical,
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

    /// 高效碰撞检测
    fn check_collision(&self, sprite: &TextSprite, start_x: i32, start_y: i32) -> bool {
        let sprite_w32 = sprite.width_u32;
        let sprite_h = sprite.bbox_height;
        let shift = (start_x & 31).unsigned_abs();
        let r_shift = 32 - shift;

        // 边界预检查
        if start_x + (sprite.bbox_width as i32) < 0
            || start_x >= self.width as i32
            || start_y + (sprite.bbox_height as i32) < 0
            || start_y >= self.height as i32
        {
            return true;
        }

        for sy in 0..sprite_h {
            let gy = start_y + sy as i32;

            if gy < 0 || gy >= self.height as i32 {
                return true;
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

                // 构造 Mask: 结合移位
                let mask = if shift == 0 {
                    s_val
                } else {
                    (carry << r_shift) | (s_val >> shift)
                };

                let gx = grid_col_start + sx as isize;

                if mask != 0 {
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
    text_center_x: f32, // TopLeft 到 Text Center 的偏移
    text_center_y: f32,
}

fn rasterize_text(
    text: &str,
    size: f32,
    angle_deg: f32,
    font: &Font,
    padding: u32,
    vertical_layout: bool,
) -> TextSprite {
    // 1. 获取字体度量
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
    let mut max_glyph_width = 0.0f32;

    for ch in text.chars() {
        let (glyph_metrics, bitmap) = font.rasterize(ch, size);
        glyphs.push((total_width, glyph_metrics, bitmap));
        total_width += glyph_metrics.advance_width;

        // 使用 advance_width 而不是 width 计算垂直列宽
        // width 只是墨迹宽度，而 vertical-rl 布局时浏览器按 advance_width 对齐
        // 这解决了包含窄字母（如英文）时，蒙版过窄导致重叠的问题
        if glyph_metrics.advance_width > max_glyph_width {
            max_glyph_width = glyph_metrics.advance_width;
        }
    }

    // 2. 变换参数与布局计算
    let padding_f = padding as f32;
    let unrotated_w;
    let unrotated_h;

    if vertical_layout {
        // 竖排模式：宽度由最宽的字排版宽度决定
        unrotated_w = max_glyph_width.ceil() + padding_f * 2.0;
        // 简单估算：字符数 * line_size
        unrotated_h =
            (text.chars().count() as f32 * metrics.new_line_size).ceil() + padding_f * 2.0;
    } else {
        // 横排模式（默认）
        unrotated_w = total_width.ceil() + padding_f * 2.0;
        unrotated_h = metrics.new_line_size.ceil() + padding_f * 2.0;
    }

    // 旋转中心 (Geometric Center)
    let cx = unrotated_w / 2.0;
    let cy = unrotated_h / 2.0;

    let (sin, cos) = if vertical_layout {
        // 竖排正写：不旋转字符（相当于0度）
        (0.0, 1.0)
    } else {
        // 正常旋转
        let rad = angle_deg.to_radians();
        rad.sin_cos()
    };

    // 变换函数
    let transform = |x: f32, y: f32| -> (f32, f32) {
        let dx = x - cx;
        let dy = y - cy;
        (dx * cos - dy * sin + cx, dx * sin + dy * cos + cy)
    };

    // 3. 计算旋转后的边界
    // 注意：如果是 vertical_layout，sin=0, cos=1，边界即为 unrotated 宽高
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

    let buf_width = (max_x - min_x).ceil() as i32;
    let buf_height = (max_y - min_y).ceil() as i32;

    // 4. 像素收集与 Tight Bounding Box 计算
    let mut pixels = Vec::new();

    let mut tight_min_x = i32::MAX;
    let mut tight_max_x = i32::MIN;
    let mut tight_min_y = i32::MAX;
    let mut tight_max_y = i32::MIN;

    let base_x = padding_f;
    let base_y = padding_f + metrics.ascent;

    for (i, (offset_x, glyph_metrics, bitmap)) in glyphs.iter().enumerate() {
        // 计算字符基准位置
        let (char_left, char_top) = if vertical_layout {
            // 竖排：居中对齐 X（基于列宽），Y 逐行递增
            // 使用 max_glyph_width (advance) 来计算居中，确保与浏览器行为一致
            let center_offset = (max_glyph_width - glyph_metrics.width as f32) / 2.0;
            let x = base_x + center_offset;
            let y = base_y + (i as f32 * metrics.new_line_size)
                - glyph_metrics.height as f32
                - glyph_metrics.ymin as f32;
            (x, y)
        } else {
            // 横排
            let x = base_x + offset_x + glyph_metrics.xmin as f32;
            let y = base_y - glyph_metrics.height as f32 - glyph_metrics.ymin as f32;
            (x, y)
        };

        for y in 0..glyph_metrics.height {
            for x in 0..glyph_metrics.width {
                // Alpha threshold > 10
                if bitmap[y * glyph_metrics.width + x] > 10 {
                    let ox = char_left + x as f32;
                    let oy = char_top + y as f32;
                    let (rx, ry) = transform(ox, oy);

                    let fx = (rx - min_x).round() as i32;
                    let fy = (ry - min_y).round() as i32;

                    // 应用 padding (膨胀)
                    let pad = padding as i32;
                    for py in -pad..=pad {
                        for px in -pad..=pad {
                            let px_x = fx + px;
                            let px_y = fy + py;

                            if px_x >= 0 && px_y >= 0 && px_x < buf_width && px_y < buf_height {
                                pixels.push((px_x, px_y));
                                tight_min_x = tight_min_x.min(px_x);
                                tight_max_x = tight_max_x.max(px_x);
                                tight_min_y = tight_min_y.min(px_y);
                                tight_max_y = tight_max_y.max(px_y);
                            }
                        }
                    }
                }
            }
        }
    }

    if pixels.is_empty() {
        return TextSprite {
            data: vec![],
            width_u32: 0,
            bbox_width: 0,
            bbox_height: 0,
            text_center_x: 0.0,
            text_center_y: 0.0,
        };
    }

    // 5. 生成位图数据 (压缩到 Tight Box)
    let tight_w = (tight_max_x - tight_min_x + 1) as u32;
    let tight_h = (tight_max_y - tight_min_y + 1) as u32;
    let width_u32 = ((tight_w + 31) >> 5) as usize;
    let mut data = vec![0u32; width_u32 * tight_h as usize];

    for (px, py) in pixels {
        let rel_x = (px - tight_min_x) as usize;
        let rel_y = (py - tight_min_y) as usize;

        let row_idx = rel_y * width_u32;
        let col_idx = rel_x >> 5;
        let bit_idx = 31 - (rel_x & 31);
        data[row_idx + col_idx] |= 1 << bit_idx;
    }

    // 6. 计算中心偏移
    // cx, cy 是旋转中心相对于旋转前 TopLeft 的坐标
    // 在 buffer 坐标系中，旋转中心位置是:
    let center_x_in_buffer = cx - min_x;
    let center_y_in_buffer = cy - min_y;

    // 相对于 Tight Box 左上角的偏移
    let text_center_x = center_x_in_buffer - tight_min_x as f32;
    let text_center_y = center_y_in_buffer - tight_min_y as f32;

    TextSprite {
        data,
        width_u32,
        bbox_width: tight_w,
        bbox_height: tight_h,
        text_center_x,
        text_center_y,
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
        let e = 4.0; // Aspect Ratio correction
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
            r#"<rect x="0" y="0" width="100%" height="100%" fill="{}"/>"#,
            self.background
        ));

        // SVG Styling: 使用 central 基线对齐，对于垂直/旋转混合排版通常比 middle 更稳健
        svg.push_str(&format!(
            r#"<style>text{{font-family:'{}',Arial,sans-serif;text-anchor:middle;dominant-baseline:central}}</style>"#,
            escape_xml(&self.font_family)
        ));

        for word in &self.words {
            // 如果是竖排正写模式，需要添加 vertical-rl 样式
            // writing-mode: vertical-rl 让文字从上到下排列
            // text-orientation: upright 让字符保持直立（不旋转90度）
            // glyph-orientation-vertical: 0deg 兼容性补充
            let style = if word.is_vertical {
                r#" style="writing-mode: vertical-rl; text-orientation: upright; glyph-orientation-vertical: 0deg;""#
            } else {
                ""
            };

            svg.push_str(&format!(
                r#"<text transform="translate({:.1},{:.1}) rotate({:.1})" fill="{}" font-size="{:.1}"{}>{}</text>"#,
                word.x,
                word.y,
                word.rotation,
                word.color,
                word.font_size,
                style,
                escape_xml(&word.text)
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    pub fn to_png(&self, scale: f32) -> Result<Vec<u8>, Error> {
        let svg_content = self.to_svg();
        let mut fontdb = usvg::fontdb::Database::new();

        fontdb.load_font_source(usvg::fontdb::Source::Binary(Arc::new(
            self.font_data.clone(),
        )));

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
