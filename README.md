# araea-wordcloud

Rust 库：把带权词语排布成 SVG 或 PNG。文字按像素掩码沿阿基米德螺线放置。放不进画布的词语会被跳过。

## 安装

```toml
[dependencies]
araea-wordcloud = "0.1"
```

## 快速开始

```rust
use araea_wordcloud::generate;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = [("Rust", 100.0), ("Code", 60.0)];
    let cloud = generate(&words)?;
    std::fs::write("output.svg", cloud.to_svg())?;
    std::fs::write("output.png", cloud.to_png(2.0)?)?;
    Ok(())
}
```

空字符串与非正权重会被忽略。过滤后没有词语时返回错误。

## 配置

需要自定义布局时使用 `WordCloudBuilder`：

- `size(w, h)`：画布尺寸，默认 800×600
- `font(bytes)`：TTF 或 OTF 字体，默认附带 HarmonyOS Sans SC Bold
- `font_size_range(min, max)`：字号范围，默认 10–100
- `mask_preset` / `mask`：内置形状，或 SVG、PNG、JPEG 掩码
- `color_scheme` / `colors` / `background`：颜色
- `angles`：旋转角度。配合 `vertical_writing(true)` 支持竖排 CJK
- `padding(px)`：碰撞间距，默认 5
- `trim(bool)`：成图裁到内容边界，默认 `false`
- `trim_margin(px)`：`trim` 后内容外留的余量，默认 `0`
- `seed(u64)`：固定布局，便于复现

内置掩码有 Circle、Cloud、Heart、Skull、Star 和 Triangle。不调用 `mask_preset` 时使用完整矩形画布。

词是绕着画布中心摆的，词少时四周会空出一大片背景。开启 `trim` 后，成图大小跟着内容走，画布只决定词排得开不开——`size(800, 600)` 排出几个词，出来的可能是一张 260×200 的图。裁切按布局算出的包围盒在矢量层面做，SVG 与 PNG 一致，不靠扫描像素、也不依赖背景色。

## 输出与测试

`to_svg()` 返回 SVG 字符串，`to_png(scale)` 返回 PNG 字节。`WordCloud` 同时提供画布（`width`、`height`）、成图区域（`viewport`）、背景，以及已放置词语的位置、字号、颜色、旋转与包围盒（`bounds`）。`content_bounds()` 返回所有词覆盖到的最小矩形。

```sh
cargo run --example simple
cargo run --example trim
cargo run --example chinese_vertical
cargo test
```

## 许可证

可按 [Apache-2.0](LICENSE-APACHE) 或 [MIT](LICENSE-MIT) 使用。
