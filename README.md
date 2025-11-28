araea-wordcloud
===============

[<img alt="github" src="https://img.shields.io/badge/github-araea/araea__wordcloud-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/araea/araea-wordcloud)
[<img alt="crates.io" src="https://img.shields.io/crates/v/araea-wordcloud.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/araea-wordcloud)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-araea__wordcloud-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/araea-wordcloud)

一个纯 Rust 实现的高性能词云可视化库。
支持蒙版遮罩、SVG/PNG 双输出、自定义字体与配色。

## 特性

- ⚡ **纯 Rust 实现** - 基于位操作的高效碰撞检测算法
- 🖼️ **多格式输出** - 支持导出为矢量图 (SVG) 或位图 (PNG)  
- 🎭 **蒙版支持** - 内置多种形状，支持自定义图片遮罩
- 🎨 **高度定制** - 自定义字体、配色、旋转角度、间距
- 📦 **开箱即用** - 内置中文字体支持

## 安装

```toml
[dependencies]
araea-wordcloud = "0.1"
```

## 快速开始

```rust
use araea_wordcloud::generate;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = vec![
        ("Rust", 100.0),
        ("Fast", 80.0), 
        ("Safe", 60.0),
        ("WordCloud", 40.0),
    ];

    let wordcloud = generate(&words)?;
    fs::write("output.png", wordcloud.to_png(2.0)?)?;

    Ok(())
}
```

## 高级用法

```rust
use araea_wordcloud::{WordCloudBuilder, WordInput, ColorScheme, MaskShape};

let words = vec![
    WordInput::new("Love", 100.0),
    WordInput::new("Rust", 80.0),
];

let wordcloud = WordCloudBuilder::new()
    .size(800, 800)
    .background("#FFFFFF")
    .color_scheme(ColorScheme::Berry)
    .mask_preset(MaskShape::Heart)
    .font_size_range(20.0, 100.0)
    .angles(vec![0.0, 90.0])
    .build(&words)?;
```

## 配置速查

### 预设配色

| 方案       | 风格               |
|------------|--------------------|
| `Ocean`    | 海洋蓝绿色调 (默认) |
| `Sunset`   | 暖色调，红橙黄     |
| `Forest`   | 森林绿，自然风格   |
| `Berry`    | 紫色与亮橙色       |
| `Monochrome` | 黑白灰单色调     |
| `Rainbow`  | 彩虹色             |

### 预设蒙版

| 形状      | 描述         |
|-----------|--------------|
| `Circle`  | 圆形 (默认)  |
| `Heart`   | 心形         |
| `Cloud`   | 云朵形状     |
| `Star`    | 星形         |
| `Triangle`| 三角形       |
| `Skull`   | 骷髅头       |

### 构建器选项

| 方法               | 说明                     | 默认值        |
|--------------------|--------------------------|---------------|
| `.size(w, h)`      | 画布尺寸                 | 800x600       |
| `.background(hex)` | 背景颜色                 | #FFFFFF       |
| `.colors(vec![...])` | 自定义颜色列表         | Ocean Scheme  |
| `.font(bytes)`     | 自定义字体文件数据       | HarmonyOS Sans SC |
| `.mask(bytes)`     | 自定义蒙版图片           | None          |
| `.padding(px)`     | 单词碰撞内边距           | 2             |
| `.word_spacing(px)`| 单词间距                 | 4.0           |
| `.seed(u64)`       | 随机数种子 (固定布局)    | Random        |

## 示例

- `cargo run --example simple` - 基础用法
- `cargo run --example mask_shape` - 心形蒙版词云  
- `cargo run --example chinese_dense` - 高密度中文词云
- `cargo run --example advanced` - 自定义配色与布局

## 致谢

感谢 [wordcloud.online](https://wordcloud.online/zh) 提供的灵感与参考，
词云图渲染方案借鉴自该网站，实现了高效且美观的词云效果。

<br>

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
</sub>
