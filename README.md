# araea-wordcloud

Rust 词云库：把带权重的词语排布为 SVG 或 PNG 词云图片

[![GitHub](https://img.shields.io/badge/GitHub-araea%2Faraea--wordcloud-181717?logo=github&logoColor=white)](https://github.com/araea/araea-wordcloud)
[![crates.io](https://img.shields.io/crates/v/araea-wordcloud?logo=rust&logoColor=white&color=CC342D)](https://crates.io/crates/araea-wordcloud)
[![docs.rs](https://img.shields.io/docsrs/araea-wordcloud?logo=docs.rs&logoColor=white)](https://docs.rs/araea-wordcloud)

## 安装

```toml
[dependencies]
araea-wordcloud = "0.1.13"
```

## 快速使用

```rust
use araea_wordcloud::generate;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = [("Rust", 100.0), ("Code", 60.0), ("Safe", 30.0)];
    let cloud = generate(&words)?;
    std::fs::write("output.svg", cloud.to_svg())?;
    std::fs::write("output.png", cloud.to_png(2.0)?)?;
    Ok(())
}
```

## API

`generate(&[(&str, f32)])` 是最简入口，接收词语与权重，返回 `WordCloud`。需要更多控制时用 `WordCloudBuilder`；`WordInput::new(text, weight)` 构造单个词，权重小于 0 会被夹为 0。

| `WordCloudBuilder` 方法 | 作用 |
| --- | --- |
| `size(width, height)` | 画布尺寸，各不小于 100，默认 800×600 |
| `background(color)` | 背景色，默认随配色方案 |
| `color_scheme(ColorScheme)` | 套用预设配色与背景 |
| `colors(iter)` | 自定义颜色列表 |
| `font(Vec<u8>)` | 传入字体字节替换内置字体 |
| `mask(Vec<u8>)` | 传入 SVG / PNG / JPEG 蒙版字节 |
| `mask_preset(MaskShape)` | 使用内置形状蒙版 |
| `padding(u32)` | 词间距，默认 5 |
| `font_size_range(min, max)` | 字号范围，min 不小于 4，默认 10–100 |
| `angles(Vec<f32>)` | 可选旋转角度，默认 `[0.0]` |
| `seed(u64)` | 固定随机种子以复现布局 |
| `vertical_writing(bool)` | 竖排正写（CJK），默认关 |
| `trim(bool)` | 裁到内容边界，默认关 |
| `trim_margin(u32)` | 裁切后的留白像素，默认 0 |
| `build(&[WordInput])` | 生成 `WordCloud` |

- `ColorScheme`：`Default`、`Contrasting1`、`Blue`、`Green`、`Cold1`、`Black`、`White`
- `MaskShape`：`Circle`、`Cloud`、`Heart`、`Skull`、`Star`、`Triangle`
- `WordCloud`：字段 `width`、`height`、`background`、`words`（`Vec<PlacedWord>`）、`viewport`（`Bounds`）；方法 `to_svg() -> String`、`to_png(scale: f32) -> Result<Vec<u8>, Error>`、`content_bounds() -> Option<Bounds>`

布局基于像素掩码与阿基米德螺线，词按权重从大到小排列。

## 限制 / 风险

空词与权重不大于 0 的词会被忽略；过滤后没有有效词时 `build` / `generate` 返回 `Error::Input`。

放不进画布的词会被跳过，不报错。

内置字体为 HarmonyOS Sans SC Bold，主要覆盖中文与拉丁字符；其他文字用 `font` 替换。

蒙版中接近白色（RGB 之和不小于 750）或 alpha 小于 128 的区域视为不可放置。

未设 `seed` 时布局随机。

## 链接

- [crates.io](https://crates.io/crates/araea-wordcloud)
- [docs.rs 文档](https://docs.rs/araea-wordcloud)
- [MIT](LICENSE-MIT) / [Apache-2.0](LICENSE-APACHE)
