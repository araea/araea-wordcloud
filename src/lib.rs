/*!
 * Araea WordCloud Library
 *
 * A pure Rust implementation of the Word Cloud algorithm, aligned with the logic
 * found in wordcloud2.js / B8yHTEJ1.js.
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

#[derive(Debug, Clone)]
pub struct PlacedWord {
    pub text: String,
    pub font_size: f32,
    pub x: f32,
    pub y: f32,
    pub rotation: f32,
    pub color: String,
}

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
        let scheme = ColorScheme::Default;
        Self {
            width: 800,
            height: 600,
            background: scheme.background_color().into(),
            colors: scheme.colors().into_iter().map(String::from).collect(),
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

        let valid_words: Vec<_> = words
            .iter()
            .filter(|w| !w.text.trim().is_empty() && w.weight > 0.0)
            .cloned()
            .collect();

        if valid_words.is_empty() {
            return Err(Error::Input("No valid words provided".into()));
        }

        let font_info = self.load_font()?;
        let font = Font::from_bytes(font_info.data.as_slice(), FontSettings::default())
            .map_err(|e| Error::Font(e.to_string()))?;

        // 1. Init Grid
        let mut collision_map = CollisionMap::new(self.width, self.height);

        // 2. Apply Mask
        if let Some(mask_bytes) = &self.mask_data {
            self.apply_mask(&mut collision_map, mask_bytes)?;
        }

        let mut rng = match self.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_os_rng(),
        };

        // Sort by weight desc
        let mut sorted_words = valid_words;
        sorted_words.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());

        let max_weight = sorted_words.first().map(|w| w.weight).unwrap_or(1.0);
        let min_weight = sorted_words.last().map(|w| w.weight).unwrap_or(1.0);
        let weight_range = max_weight - min_weight;

        let mut placed_words = Vec::with_capacity(sorted_words.len());
        let effective_padding = self.padding + (self.word_spacing / 2.0) as u32;

        // 3. Layout Loop
        for word in &sorted_words {
            let normalized = if weight_range > 0.0 {
                (word.weight - min_weight) / weight_range
            } else {
                1.0
            };

            // Linear sizing logic (matches JS simple scaling)
            let font_size =
                self.min_font_size + normalized * (self.max_font_size - self.min_font_size);

            let angle = self.angles[rng.random_range(0..self.angles.len())];

            // 4. Try Place
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

    fn apply_mask(&self, collision_map: &mut CollisionMap, mask_bytes: &[u8]) -> Result<(), Error> {
        let mut apply_pixels =
            |width: u32, height: u32, get_pixel: &dyn Fn(u32, u32) -> Option<(u8, u8, u8, u8)>| {
                for y in 0..height {
                    for x in 0..width {
                        if let Some((r, g, b, a)) = get_pixel(x, y) {
                            // Logic matches JS: white (sum >= 750) or transparent (a < 128) is blocked
                            let sum = r as u16 + g as u16 + b as u16;
                            let is_blocked = a < 128 || sum >= 750;

                            if is_blocked {
                                collision_map.set(x as i32, y as i32);
                            }
                        }
                    }
                }
            };

        let opt = usvg::Options::default();
        if let Ok(tree) = usvg::Tree::from_data(mask_bytes, &opt) {
            let size = tree.size().to_int_size();
            let scale_x = self.width as f32 / size.width() as f32;
            let scale_y = self.height as f32 / size.height() as f32;

            let mut pixmap = Pixmap::new(self.width, self.height)
                .ok_or(Error::Render("Failed to create mask buffer".into()))?;

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
        // Rasterize text to tight bounding box bitmask
        let sprite = rasterize_text(text, font_size, angle, font, padding);

        if sprite.bbox_width == 0 || sprite.bbox_height == 0 {
            return None;
        }

        let start_x = map.width as i32 / 2;
        let start_y = map.height as i32 / 2;

        let dt = if rng.random_bool(0.5) { 1 } else { -1 };

        // 5. Spiral Search (Archimedean)
        let spiral = ArchimedeanSpiral::new(map.width as i32, map.height as i32, dt);
        let max_iter = (map.width * map.height) as usize / 2; // Reasonable limit

        for (dx, dy) in spiral.take(max_iter) {
            // Attempt placement at (current_x, current_y) which represents Top-Left of Sprite
            let current_x = start_x + dx - (sprite.bbox_width as i32 / 2);
            let current_y = start_y + dy - (sprite.bbox_height as i32 / 2);

            // 6. Collision Check
            if !map.check_collision(&sprite, current_x, current_y) {
                // 7. Update Grid
                map.write_sprite(&sprite, current_x, current_y);

                // Return CENTER coordinates for SVG transformation
                // The sprite was placed at top-left `current_x`, `current_y`.
                // The center is simply half dimensions away.
                // NOTE: We don't use anchor_x/y here anymore because rasterize_text
                // returns a tight box, and we position that tight box centered on the spiral point.
                // For SVG `text-anchor="middle"`, we need the coordinates of the text origin/center.
                // Since `rasterize_text` now returns the offset from the TightBox-TopLeft
                // to the Text-Center (text_center_x, text_center_y), we add that.

                return Some((
                    current_x as f32 + sprite.text_center_x,
                    current_y as f32 + sprite.text_center_y,
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
    stride: usize,
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

    fn set(&mut self, x: i32, y: i32) {
        if x >= 0 && y >= 0 && x < self.width as i32 && y < self.height as i32 {
            let row_idx = y as usize * self.stride;
            let col_idx = (x as usize) >> 5;
            let bit_idx = 31 - (x & 31);
            self.data[row_idx + col_idx] |= 1 << bit_idx;
        }
    }

    fn check_collision(&self, sprite: &TextSprite, start_x: i32, start_y: i32) -> bool {
        let sprite_w32 = sprite.width_u32;
        let sprite_h = sprite.bbox_height;
        let shift = (start_x & 31).unsigned_abs();
        let r_shift = 32 - shift;

        // Bounding box pre-check
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
                return true; // Out of bounds usually means collision in this context
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
    data: Vec<u32>,
    width_u32: usize,
    bbox_width: u32,
    bbox_height: u32,
    text_center_x: f32, // Offset from TopLeft to Text Center
    text_center_y: f32,
}

fn rasterize_text(text: &str, size: f32, angle_deg: f32, font: &Font, padding: u32) -> TextSprite {
    // 1. Basic Rasterization
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

    // 2. Transformations
    let padding_f = padding as f32;
    // Initial geometric box (untight)
    let unrotated_w = total_width.ceil() + padding_f * 2.0;
    let unrotated_h = metrics.new_line_size.ceil() + padding_f * 2.0;

    // Center of text
    let cx = unrotated_w / 2.0;
    let cy = unrotated_h / 2.0;

    let rad = angle_deg.to_radians();
    let (sin, cos) = rad.sin_cos();

    let transform = |x: f32, y: f32| -> (f32, f32) {
        let dx = x - cx;
        let dy = y - cy;
        (dx * cos - dy * sin + cx, dx * sin + dy * cos + cy)
    };

    // Calculate geometric bounds for buffer allocation
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

    // 3. Pixel Collection (finding tight bounds)
    // We map pixels to a set of (x,y) points
    let mut pixels = Vec::new();
    let base_x = padding_f;
    let base_y = padding_f + metrics.ascent;

    // To align with JS behavior, we collect actual pixels to find the "Tight" bounding box.
    // The "geometric" box is often too large for diagonal text.
    let mut tight_min_x = i32::MAX;
    let mut tight_max_x = i32::MIN;
    let mut tight_min_y = i32::MAX;
    let mut tight_max_y = i32::MIN;

    for (offset_x, glyph_metrics, bitmap) in &glyphs {
        let char_left = base_x + offset_x + glyph_metrics.xmin as f32;
        let char_top = base_y - glyph_metrics.height as f32 - glyph_metrics.ymin as f32;

        for y in 0..glyph_metrics.height {
            for x in 0..glyph_metrics.width {
                // Alpha threshold
                if bitmap[y * glyph_metrics.width + x] > 10 {
                    let ox = char_left + x as f32;
                    let oy = char_top + y as f32;
                    let (rx, ry) = transform(ox, oy);

                    // Map to buffer coordinates
                    let fx = (rx - min_x).round() as i32;
                    let fy = (ry - min_y).round() as i32;

                    // Apply padding
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

    // 4. Create Tight Sprite
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

    // 5. Calculate Center Offset
    // We need the position of the text's rotation center (cx, cy)
    // relative to the Top-Left of the Tight Bounding Box.
    // (cx, cy) after transform is simply (cx, cy) relative to origin if purely rotated?
    // transform() rotates around (cx, cy).
    // The rotated point corresponding to (cx, cy) is ... (cx, cy).
    // In buffer coords (relative to min_x, min_y):
    // center_in_buffer = transform(cx, cy) - (min_x, min_y)
    //                  = (cx, cy) - (min_x, min_y) ? NO.
    // transform(cx, cy) = (cx, cy) by definition of rotation center.
    // So buffer_cx = cx - min_x; buffer_cy = cy - min_y;
    //
    // The Tight Box Top Left in buffer coords is (tight_min_x, tight_min_y).
    //
    // So offset = buffer_center - tight_top_left
    //           = (cx - min_x - tight_min_x, cy - min_y - tight_min_y)

    let center_x_in_buffer = cx - min_x;
    let center_y_in_buffer = cy - min_y;

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

        svg.push_str(&format!(
            r#"<style>text{{font-family:'{}',Arial,sans-serif;text-anchor:middle;dominant-baseline:middle}}</style>"#,
            escape_xml(&self.font_family)
        ));

        for word in &self.words {
            // JS Output uses: transform="translate(x,y) rotate(deg)" with text-anchor="middle"
            svg.push_str(&format!(
                r#"<text transform="translate({:.1},{:.1}) rotate({:.1})" fill="{}" font-size="{:.1}">{}</text>"#,
                word.x,
                word.y,
                word.rotation,
                word.color,
                word.font_size,
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
